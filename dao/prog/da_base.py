import datetime
import re
import sys
import os
import time
import threading
import warnings
from dataclasses import dataclass
from homeassistant_api import Client as HAClient
from homeassistant_api.models import State as HAState
from homeassistant_api.errors import InternalServerError, RequestTimeoutError
from niquests.exceptions import RequestException as HARequestException
import pandas as pd
from subprocess import PIPE, run
import logging
from logging import Handler
from sqlalchemy import Table, select, func, and_
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# from dao.prog.solar_predictor import SolarPredictor
from dao.prog.utils import get_tibber_data, error_handling
from dao.prog.version import __version__
from pathlib import Path
from dao.prog.config.loader import ConfigurationLoader
from dao.prog.config.models.base import UNUSABLE_STATES
from dao.prog import tasks as task_registry
from dao.lib.db_connections import make_db_da, make_db_ha
from dao.lib.da_meteo import Meteo
from dao.lib.da_prices import DaPrices

# from db_manager import DBmanagerObj
from typing import Optional, Union


@dataclass
class HAContext:
    """Runtime values fetched from Home Assistant on start-up.

    These are not part of the static configuration in options.json and must
    never be written back to disk.  Pass instances of this dataclass to any
    collaborator that needs location or timezone information.
    """

    latitude: float
    longitude: float
    time_zone: str
    country: str


#: A network hiccup or a momentarily overloaded Home Assistant must not
#: itself take down calc_optimum (~40 HA reads per run); three attempts with
#: a short backoff absorb a single blip. homeassistant_api only wraps a
#: timeout (RequestTimeoutError) and a >=500 response (InternalServerError)
#: in its own exception types; anything below that in the transport itself
#: (connection refused, DNS failure) surfaces as a raw niquests
#: RequestException instead. A 401/403/404/429 is not retried: those come
#: back as their own homeassistant_api.errors classes (UnauthorizedError,
#: EndpointNotFoundError, ...), none of which are in this tuple, so
#: retry_if_exception_type leaves them alone — a config error retrying
#: cannot fix would otherwise just delay the FlexValue default-fallback
#: (see models/base.py) by several seconds for nothing.
_retry_ha_call = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(
        (
            HARequestException,
            RequestTimeoutError,
            InternalServerError,
        )
    ),
)


def _parse_calendar_datetime(value, tz) -> datetime.datetime:
    """A calendar event edge as a tz-aware datetime.

    Home Assistant returns a timed event's edge as an ISO string directly
    and an all-day event's as ``{"date": "YYYY-MM-DD"}``; both are handled
    here rather than assuming the shape of the entity that happens to be
    configured.
    """
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date")
    moment = datetime.datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=tz)
    return moment


class NotificationHandler(Handler):
    def __init__(self, _hass: "DaBase", _entity=None):
        """
        Initialize the handler.
        """
        Handler.__init__(self)
        self.hass = _hass
        self.entity = _entity
        self.count = 0

    def emit(self, record):
        if self.entity and record.levelno >= logging.WARNING and self.count == 0:
            if record.levelno >= logging.ERROR:
                self.count += 1
            msg = self.format(record)
            msg = msg.partition("\n")[0]
            # Bypasses DaBase.set_value's own retry/read-back/logging: a
            # warning raised while reporting a warning through this same
            # handler would recurse.
            self.hass._raw_set_value(self.entity, msg)


class DaBase:
    _config = None
    _loader = None
    _init_lock = threading.Lock()

    def __init__(self, file_name: str = None):
        self.file_name = file_name
        path = os.getcwd()
        new_path = "/".join(list(path.split("/")[0:-2]))
        if new_path not in sys.path:
            sys.path.append(new_path)
        self.make_data_path()
        self.debug = False
        self.tasks = self.generate_tasks()
        self.log_level = logging.INFO
        self.notification_entity = None
        self.ha_context: HAContext | None = None
        # Subclasses and main() test `self.config is None` to detect a failed
        # configuration load. Set the attribute before anything can return,
        # otherwise that check raises AttributeError instead of telling the
        # user what was wrong with the configuration.
        self.config = None
        self.loader = None
        self._owns_root_logger = self._configure_root_logging()
        # Load config exactly once, even when multiple threads construct a
        # DaBase subclass concurrently (e.g. gunicorn workers sharing a process).
        # DB singletons are managed separately in db_connections.py.
        with DaBase._init_lock:
            if DaBase._config is None:
                try:
                    DaBase._loader = ConfigurationLoader(
                        Path(self.file_name)
                        if self.file_name
                        else Path("../data/options.json")
                    )
                    DaBase._config = DaBase._loader.load_and_validate()
                except FileNotFoundError as e:
                    logging.error(f"Configuratiebestand niet gevonden: {e}")
                    return
                except (ValueError, TypeError, RuntimeError) as e:
                    logging.error(f"Configuratie kon niet worden geladen: {e}")
                    return

        self.config = DaBase._config
        self.loader = DaBase._loader

        self.db_da = make_db_da(self.config, self.loader.secrets)
        if self.db_da is None:
            raise RuntimeError('No database connection for Day Ahead')
        self.db_ha = make_db_ha(self.config, self.loader.secrets)
        if self.db_ha is None:
            raise RuntimeError('No database connection for Home Assistant')

        log_level_str = self.config.logging_level or "info"
        _log_level = getattr(logging, log_level_str.upper(), None)
        if not isinstance(_log_level, int):
            raise ValueError("Invalid log level: %s" % _log_level)
        self.log_level = _log_level
        logging.addLevelName(logging.DEBUG, "debug")
        logging.addLevelName(logging.INFO, "info")
        logging.addLevelName(logging.WARNING, "waarschuwing")
        logging.addLevelName(logging.ERROR, "fout")
        logging.addLevelName(logging.CRITICAL, "kritiek")
        if self._owns_root_logger:
            logging.getLogger().setLevel(self.log_level)
        ha = self.config.homeassistant
        self.protocol_api = ha.protocol_api
        self.ip_address = ha.ip_address
        self.ip_port = ha.ip_port
        if self.ip_port is None:
            self.hassurl = self.protocol_api + "://" + self.ip_address + "/core/"
        else:
            self.hassurl = (
                self.protocol_api
                + "://"
                + self.ip_address
                + ":"
                + str(self.ip_port)
                + "/"
            )
        _tok = ha.hasstoken
        if _tok is None:
            self.hasstoken = os.environ.get("SUPERVISOR_TOKEN")
        else:
            self.hasstoken = _tok.resolve(self.loader.secrets)

        # A single persistent Session (niquests, via homeassistant_api) is
        # reused for every HA call this instance makes, instead of hassapi's
        # bare requests.get()/post() opening a new TCP+TLS connection per
        # call — real overhead at ~40 HA reads per calc_optimum run.
        self._ha_client = HAClient(
            api_url=self.hassurl + "api",
            token=self.hasstoken,
            global_request_kwargs={"timeout": 10},
        )
        try:
            resp_dict = self._ha_client.get_config()
        except Exception as ex:
            # Deliberately broad: a construction-time reachability failure
            # can surface as a homeassistant_api.errors.* exception (a non-2xx
            # response) or as a raw niquests exception (connection refused,
            # DNS failure) — homeassistant_api only wraps a timeout, nothing
            # below that in the transport. Either way the instance is
            # unusable and the caller needs the same clear message.
            raise RuntimeError(
                f"Home Assistant API niet bereikbaar op {self.hassurl}api: {ex}"
            ) from ex
        logging.debug(f"hass/api/config: {resp_dict}")
        try:
            self.ha_context = HAContext(
                latitude=resp_dict["latitude"],
                longitude=resp_dict["longitude"],
                time_zone=resp_dict["time_zone"],
                country=resp_dict.get("country") or "NL",
            )
        except (KeyError, TypeError) as ex:
            raise RuntimeError(
                f"Onverwacht antwoord van Home Assistant api/config: {resp_dict!r}"
            ) from ex
        self.time_zone = self.ha_context.time_zone
        # One clock. The epoch columns are read and written against this
        # zone, so the database layer has to agree with what Home Assistant
        # reports rather than fall back to the container's own setting. An
        # explicit time_zone in options.json still wins: it is there for the
        # case where the database genuinely disagrees.
        if not (self.config.time_zone or None):
            for manager in (self.db_da, self.db_ha):
                if manager is not None:
                    manager.TARGET_TIMEZONE = self.time_zone
        self.meteo = Meteo(
            self.config,
            self.db_da,
            latitude=self.ha_context.latitude,
            longitude=self.ha_context.longitude,
            secrets=self.loader.secrets,
            country=self.ha_context.country,
        )
        if (self.ha_context.country == "NL") or (self.ha_context.country == "BE"):
            from dao.forecast.weather.observations import nearest_knmi_station

            self.knmi_station = str(
                nearest_knmi_station(
                    self.ha_context.latitude, self.ha_context.longitude
                )
            )
        self.solar = self.config.solar
        self.interval = self.config.interval
        self.interval_s = 3600 if self.interval == "1hour" else 900

        self.prices = DaPrices(
            self.config,
            self.db_da,
            country=self.ha_context.country,
            secrets=self.loader.secrets,
        )
        self.prices_options = self.config.prices
        # eb + ode levering
        self.taxes_l_def = (
            self.prices_options.energy_taxes_consumption
            if self.prices_options
            else None
        )
        # opslag kosten leverancier
        self.ol_l_def = (
            self.prices_options.cost_supplier_consumption
            if self.prices_options
            else None
        )
        # eb+ode teruglevering
        self.taxes_t_def = (
            self.prices_options.energy_taxes_production if self.prices_options else None
        )
        self.ol_t_def = (
            self.prices_options.cost_supplier_production
            if self.prices_options
            else None
        )
        self.btw_l_def = (
            self.prices_options.vat_consumption if self.prices_options else None
        )
        self.btw_t_def = (
            self.prices_options.vat_production
            if self.prices_options
            else self.btw_l_def
        )
        self.multiplier_l_def = (
            self.prices_options.multiplier_consumption if self.prices_options else None
        )
        self.multiplier_t_def = (
            self.prices_options.multiplier_production if self.prices_options else None
        )
        self.salderen = self.prices_options.tax_refund if self.prices_options else True

        self.history_options = self.config.history
        self.strategy = self.config.strategy.resolve(
            self.ha_getter, default="minimize cost"
        )
        self.tibber_options = self.config.tibber
        notif = self.config.notifications
        self.notification_entity = notif.notification_entity
        self.notification_opstarten = notif.opstarten
        self.notification_berekening = notif.berekening
        self.last_activity_entity = notif.last_activity_entity
        self.set_last_activity()
        self.graphics_options = self.config.graphics
        self.db_da.log_pool_status()
        warnings.simplefilter("ignore", ResourceWarning)

    @_retry_ha_call
    def get_state(self, entity_id: str) -> HAState:
        return self._ha_client.get_state(entity_id=entity_id)

    @_retry_ha_call
    def get_calendar_events(
        self, entity_id: str, start: datetime.datetime, end: datetime.datetime
    ) -> list:
        """Events on ``entity_id`` in ``[start, end]``, via the raw calendar API.

        ``homeassistant_api`` has no calendar support of its own; the
        underlying client's generic ``request()`` reaches the same
        ``GET /api/calendars/<entity_id>`` endpoint the frontend uses.
        """
        from dao.forecast.baseload.absence import CalendarEvent

        payload = self._ha_client.request(
            f"calendars/{entity_id}",
            params={"start": start.isoformat(), "end": end.isoformat()},
        )
        events = []
        for item in payload or []:
            try:
                events.append(
                    CalendarEvent(
                        start=_parse_calendar_datetime(item.get("start"), start.tzinfo),
                        end=_parse_calendar_datetime(item.get("end"), start.tzinfo),
                        summary=item.get("summary") or "",
                    )
                )
            except (TypeError, ValueError) as ex:
                logging.warning(
                    f"Kalenderevent van {entity_id} overgeslagen: {ex}"
                )
        return events

    @_retry_ha_call
    def call_service(self, service: str, entity_id: str, **kwargs) -> tuple:
        # turn_on/turn_off/select_option/set_value below are all thin
        # wrappers around call_service, so this one override also covers
        # every one of them. homeassistant_api's trigger_service() wants the
        # domain as its own argument rather than deriving it from entity_id
        # itself (hassapi's behaviour, kept here so every call site that
        # passes entity_id, positionally or as a keyword, is unaffected).
        domain = entity_id.split(".")[0]
        return self._ha_client.trigger_service(
            domain, service, entity_id=entity_id, **kwargs
        )

    @_retry_ha_call
    def set_state(self, entity_id: str, state, attributes: Optional[dict] = None) -> HAState:
        return self._ha_client.set_state(
            HAState(entity_id=entity_id, state=str(state), attributes=attributes or {})
        )

    def turn_on(self, entity_id: str) -> tuple:
        return self.call_service("turn_on", entity_id)

    def turn_off(self, entity_id: str) -> tuple:
        return self.call_service("turn_off", entity_id)

    def select_option(self, entity_id: str, option: str) -> tuple:
        return self.call_service("select_option", entity_id, option=option)

    def _raw_set_value(self, entity_id: str, value) -> tuple:
        """Call the set_value service directly against the HA client,
        bypassing set_value()'s own retry/read-back/logging wrapper. Used
        only by NotificationHandler.emit (see its comment for why)."""
        domain = entity_id.split(".")[0]
        return self._ha_client.trigger_service(
            domain, "set_value", entity_id=entity_id, value=value
        )

    # Callable passed to FlexValue.resolve() — returns HA state as a plain string.
    def ha_getter(self, eid):
        return self.get_state(eid).state

    # -- guarded reads ----------------------------------------------------
    #
    # Home Assistant reports "unavailable" or "unknown" while an integration
    # starts, reconnects or is broken, and a misspelled entity id gives a 404.
    # An optimisation run must survive that: it logs what it assumed and
    # carries on with a default instead of crashing before the battery gets
    # its setpoint. These helpers are the only sanctioned way to read a state
    # inside calc_optimum.

    def read_state(self, entity_id: str, what: str = "") -> Optional[str]:
        """The raw state string of *entity_id*, or None when it has no usable value."""
        label = what or entity_id
        try:
            raw = self.get_state(entity_id).state
        except Exception as ex:  # noqa: BLE001 - REST/HTTP failures of any kind
            logging.warning(f"{label}: entity {entity_id} niet leesbaar ({ex})")
            return None
        if raw is None or str(raw).strip().lower() in UNUSABLE_STATES:
            logging.warning(f"{label}: entity {entity_id} heeft state {raw!r}")
            return None
        return str(raw)

    def get_str(self, entity_id: str, default: str, what: str = "") -> str:
        raw = self.read_state(entity_id, what)
        if raw is None:
            logging.warning(f"{what or entity_id}: standaardwaarde {default!r} gebruikt")
            return default
        return raw

    def get_float(self, entity_id: str, default: float, what: str = "") -> float:
        raw = self.read_state(entity_id, what)
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                logging.warning(
                    f"{what or entity_id}: state {raw!r} van {entity_id} is geen getal"
                )
        logging.warning(f"{what or entity_id}: standaardwaarde {default} gebruikt")
        return default

    def get_bool(self, entity_id: str, default: bool, what: str = "") -> bool:
        """True for "on"/"true"/"1"/"yes", False for their opposites, default otherwise."""
        raw = self.read_state(entity_id, what)
        if raw is not None:
            lowered = raw.strip().lower()
            if lowered in ("on", "true", "1", "yes"):
                return True
            if lowered in ("off", "false", "0", "no"):
                return False
            logging.warning(
                f"{what or entity_id}: state {raw!r} van {entity_id} is geen aan/uit"
            )
        logging.warning(f"{what or entity_id}: standaardwaarde {default} gebruikt")
        return default

    def get_datetime(
        self,
        entity_id: str,
        default: Optional[datetime.datetime],
        what: str = "",
        formats: tuple = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"),
    ) -> Optional[datetime.datetime]:
        """Parse an input_datetime style state; default when missing or malformed."""
        raw = self.read_state(entity_id, what)
        if raw is not None:
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(raw.strip(), fmt)
                except ValueError:
                    continue
            logging.warning(
                f"{what or entity_id}: state {raw!r} van {entity_id} is geen datum/tijd"
            )
        logging.warning(f"{what or entity_id}: standaardwaarde {default} gebruikt")
        return default

    def set_value(self, entity_id: str, value: Union[int, float, str]) -> tuple:
        """Write *value* through the set_value service and check the result.

        A failing service call is an error and is raised. A read-back that does
        not match is only reported: entities backed by a device (a `number.`
        of an inverter integration, for instance) update their state after the
        device confirms, which can be later than this read. Raising there
        aborted the rest of the device block, leaving for example the battery
        power written but its operating mode not.
        """
        try:
            result = self.call_service("set_value", entity_id, value=value)
        except Exception:
            logging.error(f"Fout bij schrijven naar {entity_id}, waarde {value}")
            raise
        try:
            state = self.get_state(entity_id).state
            if isinstance(value, (int, float)):
                mismatch = round(float(state), 5) != round(float(value), 5)
            else:
                mismatch = state != value
        except Exception as ex:  # noqa: BLE001 - read-back is best effort
            logging.warning(
                f"Waarde {value} naar {entity_id} geschreven, controle lezen mislukt: {ex}"
            )
            return result
        if mismatch:
            logging.warning(
                f"Waarde {value} naar {entity_id} geschreven, maar de entity meldt "
                f"nog {state!r}"
            )
        return result

    def _configure_root_logging(self, logger: logging.Logger = None) -> bool:
        """Configure *logger* (the root logger by default), unless a host
        application already owns it.

        Returns whether this instance is the one that configured it, which
        gates the second ``setLevel`` later in ``__init__`` (the level is
        only known once the configuration has been read).

        In the command line processes (day_ahead.py, da_scheduler.py,
        da_fast.py) nothing has configured logging when a DaBase is built,
        so it configures the root logger as before. Inside the web server
        app/__init__.py owns it, and a Report() constructed to render a
        report page used to reset the root level to whatever
        ``logging_level`` the DAO configuration carries. Opening one report
        page could therefore switch the entire dashboard to debug and flood
        the add-on log with output from every library, as a side effect of
        rendering a page.

        Spelled out rather than left to ``logging.basicConfig``, whose
        "do nothing when handlers exist" rule is the same decision made
        invisibly: here the condition is the thing being returned, and a
        logger can be passed in so the behaviour is testable without
        fighting whatever else has configured the real root logger.
        """
        logger = logging.getLogger() if logger is None else logger
        if logger.handlers:
            return False
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(self.log_level)
        return True

    @staticmethod
    def generate_tasks():
        """The task registry, kept in dao.prog.tasks.

        This used to be the definition itself, one of four separate task
        lists that had drifted apart (see the module docstring of
        dao.prog.tasks). It stays as a method because the scheduler and the
        v2 API already call DaBase.generate_tasks().
        """
        return dict(task_registry.TASKS)

    def start_logging(self):
        logging.debug(f"python pad:{sys.path}")
        logging.info(f"Day Ahead Optimalisering versie: {__version__}")
        logging.info(
            f"Day Ahead Optimalisering gestart op: "
            f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )
        if self.config is not None and self.ha_context is not None:
            logging.debug(
                f"Locatie: latitude {str(self.ha_context.latitude)} "
                f"longitude: {str(self.ha_context.longitude)}"
            )

    @staticmethod
    def make_data_path():
        if os.path.lexists("../data"):
            return
        else:
            os.symlink("/config/dao_data", "../data")

    def set_last_activity(self):
        if self.last_activity_entity is not None:
            self.call_service(
                "set_datetime",
                entity_id=self.last_activity_entity,
                datetime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

    def get_meteo_data(self, show_graph: bool = False):
        self.meteo.get_meteo_data(show_graph)

    @staticmethod
    def get_tibber_data():
        get_tibber_data()

    @staticmethod
    def consolidate_data():
        from da_report import Report

        report = Report()
        start_dt = None
        if len(sys.argv) > 2:
            # datetime start is given
            start_str = sys.argv[2]
            try:
                start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
            except Exception as ex:
                error_handling(ex)
                return
        report.consolidate_data(start_dt)

    def get_day_ahead_prices(self):
        """Fetch day-ahead prices; ``day_ahead.py prices [start [end]]`` backfills.

        The optional dates (YYYY-MM-DD) come from the command line. They are
        interpreted here, at the entry point, so the fetch itself never looks
        at sys.argv (it also runs inside the web server, where argv is
        gunicorn's).
        """
        source = (
            self.prices_options.source_day_ahead if self.prices_options else "nordpool"
        )
        start = end = None
        # Other tokens on the command line are task keywords ("debug", "calc");
        # only date-shaped tokens are taken as the range.
        dates = [a for a in sys.argv[1:] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a)][:2]
        try:
            if len(dates) >= 1:
                start = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
            if len(dates) >= 2:
                end = datetime.datetime.strptime(dates[1], "%Y-%m-%d")
        except ValueError:
            logging.error(
                f"Ongeldige datum in argumenten {dates}; gebruik YYYY-MM-DD "
                f"(bijvoorbeeld: day_ahead.py prices 2026-01-01 2026-01-03)"
            )
            return
        if start is not None and end is None:
            end = start + datetime.timedelta(days=1)
        self.prices.get_prices(source, _start=start, _end=end)

    def save_df(
        self, tablename: str, tijd: list, df: pd.DataFrame, vintage: bool = False
    ):
        """
        Slaat de data in het dataframe op in de tabel "table"
        :param tablename: de naam van de tabel waarin de data worden opgeslagen
        :param tijd: de datum tijd van de rijen in het dataframe
        :param df: het dataframe met de code van de variabelen in de kolomheader
        :param vintage: ook archiveren met de vooruitblik waarmee ze zijn gemaakt,
            zodat de prognosefout achteraf te meten is
        :return: None
        """
        df = df.reset_index(drop=True)
        columns = df.columns.values.tolist()[1:]
        # Melt (time, col1, col2, ...) into long-format (time, code, value)
        # rows via a plain list instead of df_db.loc[df_db.shape[0]] = row
        # per (index, column) pair: that copies the whole frame on every one
        # of the rows*columns appends.
        rows = []
        for index in range(min(len(tijd), len(df))):
            # db_da.epoch rather than a local pytz.localize: one conversion
            # against the configured zone for everything that reads or writes
            # these epoch columns.
            utc = self.db_da.epoch(pd.to_datetime(tijd[index]).to_pydatetime())
            for c in columns:
                rows.append((str(utc), c, float(df.loc[index, c])))
        df_db = pd.DataFrame(rows, columns=["time", "code", "value"])
        logging.debug("Save calculated data:\n{}".format(df_db.to_string()))
        self.db_da.savedata(df_db, tablename=tablename)
        if vintage:
            # "prognoses" is upserted, so it only ever holds the most recent
            # forecast for a moment. The archive additionally keeps what was
            # predicted at longer lead times, which is what the plan was
            # actually built on.
            try:
                self.db_da.save_forecasts(
                    (
                        (int(row.time), row.code, row.value)
                        for row in df_db.itertuples()
                    ),
                    issued_ts=int(time.time()),
                )
            except Exception as ex:
                error_handling(ex)
                logging.warning(f"Prognose-archief niet bijgewerkt: {ex}")
        return

    def calc_da_avg(self) -> float:
        """
        calculates the average of the last '24' hour values of the day ahead prices
        :return: the calculated average
        """
        # old sql query
        """
        sql_avg = (
        "SELECT AVG(t1.`value`) avg_da FROM "
        "(SELECT `time`, `value`,  from_unixtime(`time`) 'begin' "
        "FROM `values` , `variabel` "
        "WHERE `variabel`.`code` = 'da' AND `values`.`variabel` = `variabel`.`id` "
        "ORDER BY `time` desc LIMIT 24) t1 "
        )
        """
        # Reflect existing tables from the database
        values_table = Table(
            "values", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        variabel_table = Table(
            "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
        )

        # Construct the inner query
        inner_query = (
            select(
                values_table.c.time,
                values_table.c.value,
                self.db_da.from_unixtime(values_table.c.time).label("begin"),
            )
            .where(
                and_(
                    variabel_table.c.code == "da",
                    values_table.c.variabel == variabel_table.c.id,
                )
            )
            .order_by(values_table.c.time.desc())
            .limit(24)
            .alias("t1")
        )

        # Construct the outer query
        outer_query = select(func.avg(inner_query.c.value).label("avg_da"))

        # Execute the query and fetch the result
        with self.db_da.engine.connect() as connection:
            query_str = str(inner_query.compile(connection))
            logging.debug(f"inner query p_avg: {query_str}")
            query_str = str(outer_query.compile(connection))
            logging.debug(f"outer query p_avg: {query_str}")
            result = connection.execute(outer_query)
            return result.scalar()

    # TODO: _get_option and the set_entity_*/get_entity_state helpers below are
    #   generic HA interaction utilities that don't belong on DaBase. Consider
    #   extracting them into a dedicated HAEntityHelper class (or mixin) that
    #   wraps the HA client, so DaBase stays focused on config/orchestration.

    @staticmethod
    def _get_option(key: str, options, default=None):
        """Get a value from a dict or Pydantic model by key (snake_case or original)."""
        if options is None:
            return default
        if isinstance(options, dict):
            return options.get(key, default)
        snake_key = key.replace(" ", "_").replace("-", "_")
        val = getattr(options, snake_key, None)
        if val is None:
            val = getattr(options, key, None)
        return val if val is not None else default

    def set_entity_value(self, entity_key: str, options, value: int | float | str):
        entity_id = self._get_option(entity_key, options)
        if entity_id is not None:
            self.set_value(entity_id, value)

    def set_entity_option(self, entity_key: str, options, value: int | float | str):
        entity_id = self._get_option(entity_key, options)
        if entity_id is not None:
            self.select_option(entity_id, value)

    def set_entity_state(self, entity_key: str, options, value: int | float | str):
        entity_id = self._get_option(entity_key, options)
        if entity_id is not None:
            self.set_state(entity_id, value)

    def get_entity_state(self, entity_key: str, options) -> int | float | str | None:
        entity_id = self._get_option(entity_key, options)
        if entity_id is not None:
            result = self.get_state(entity_id).state
        else:
            result = None
        return result

    def clean_data(self):
        """
        takes care for cleaning folders data/log and data/images
        """

        def clean_folder(folder: str, pattern: str):
            # Path.glob() instead of os.chdir(): chdir changes the working
            # directory of the whole process, and every other CWD-relative
            # path in this codebase (../data, ../prog, ...) would resolve
            # wrongly for the rest of the run if this method raised before
            # its own os.chdir(current_dir) ran.
            current_time = time.time()
            day = 24 * 60 * 60
            logging.info(f"Start removing files in {folder} with pattern {pattern}")
            save_days = self.history_options.save_days
            for path in Path(folder).glob(pattern):
                if (current_time - path.stat().st_ctime) >= save_days * day:
                    path.unlink()
                    logging.info(f"{path.name} removed")

        clean_folder("../data/log", "*.log")
        clean_folder("../data/log", "dashboard.log.*")
        clean_folder("../data/images", "*.png")

    def calc_optimum_met_debug(self):
        from day_ahead import DaCalc

        dacalc = DaCalc(self.file_name)
        # dacalc = DaCalc("../data/tst_options/options_mirabis.json")
        dacalc.debug = True
        dacalc.calc_optimum()
        # dacalc.calc_optimum(_start_dt=datetime.datetime(2025, 9, 28, 21, minute=0), _start_soc=50)

    def calc_optimum(self):
        from day_ahead import DaCalc

        dacalc = DaCalc(self.file_name)
        dacalc.debug = False
        dacalc.calc_optimum()

    def baseload_service(self):
        from dao.forecast.baseload.service import BaseloadService

        return BaseloadService(
            self.config,
            self.db_da,
            self.db_ha,
            Path("../data/forecast/baseload"),
            self.time_zone,
            ha=self,
            latitude=self.ha_context.latitude,
            longitude=self.ha_context.longitude,
        )

    def pv_service(self):
        from dao.forecast.pv.service import PVService

        return PVService(
            self.config,
            self.db_da,
            self.db_ha,
            self.ha_context.latitude,
            self.ha_context.longitude,
            Path("../data/forecast/pv"),
            self.time_zone,
            self.interval,
        )

    def calc_baseloads(self):
        from da_report import Report

        Report(self.file_name).check_baseload_sensors()
        self.baseload_service().fit()

    def forecast_accuracy(self, days: int = 30):
        """Report how far the forecasts were off, and prune the archive.

        This is the loop that was missing: DAO wrote forecasts and it wrote
        measurements, but never subtracted the two. Without it there is no
        way to tell whether the consumption forecast is 5 percent or 40
        percent off, and therefore no way to tell whether any change to it
        helped.

        The report covers every archived component over a short and a long
        window, is logged as tables, and is written to
        ``../data/forecast/accuracy.json`` for the dashboard.
        """
        from pathlib import Path as _Path

        from dao.forecast.baseload.store import write_json
        from dao.forecast.evaluate import archive_accuracy

        now = datetime.datetime.now(datetime.timezone.utc)
        windows = tuple(sorted({7, days}))
        logging.info(
            f"Prognosefout over de laatste {', '.join(str(w) for w in windows)} dagen"
        )

        report = archive_accuracy(
            self.config, self.db_da, self.db_ha, self.time_zone, days=windows, now=now
        )

        any_data = False
        for component, accuracy in report.components.items():
            for window_days, window in accuracy.windows.items():
                if not window["pairs"]:
                    continue
                any_data = True
                logging.info(
                    f"  {component} ({accuracy.unit}), {window_days} dagen: "
                    f"{window['pairs']} paren, {window['missing']} zonder meting"
                )
                logging.info(
                    f"    {'vooruitblik':<14}{'n':>6}{'bias':>10}{'MAE':>10}{'RMSE':>10}"
                )
                for bucket in sorted(window["by_lead"]):
                    score = window["by_lead"][bucket]
                    logging.info(
                        f"    {self._lead_label(bucket):<14}{score.n:>6}"
                        f"{score.bias:>10.3f}{score.mae:>10.3f}{score.rmse:>10.3f}"
                    )
                self._log_hour_bias(component, window)

        if not any_data:
            logging.info(
                "Nog geen gepaarde prognoses en metingen. Het archief vult zich bij "
                "elke optimalisatie; zet de snelle regellaag minimaal in 'shadow' "
                "zodat de gemeten huisvraag wordt vastgelegd."
            )

        try:
            write_json(_Path("../data/forecast/accuracy.json"), report.to_dict())
        except Exception as ex:  # noqa: BLE001 - the log already has the tables
            logging.warning(f"accuracy.json niet geschreven: {ex}")

        keep_days = max(days, self.history_options.forecast_days)
        try:
            removed = self.db_da.prune_forecasts(
                int((now - datetime.timedelta(days=keep_days)).timestamp())
            )
            if removed:
                logging.info(f"Prognose-archief opgeschoond: {removed} rijen verwijderd")
        except Exception as ex:
            logging.debug(f"Prognose-archief niet opgeschoond: {ex}")

    @staticmethod
    def _lead_label(bucket: int) -> str:
        labels = {0: "< 1 uur", 1: "1-4 uur", 4: "4-12 uur", 12: "12-24 uur"}
        return labels.get(bucket, f">= {bucket} uur")

    @staticmethod
    def _log_hour_bias(component: str, window: dict) -> None:
        """The actionable table: is this component structurally off at some hour?"""
        by_hour = window.get("by_hour") or {}
        if not by_hour:
            return
        logging.info(
            f"    {component}: afwijking per uur van de dag (prognose - gemeten)"
        )
        worst_hour = max(by_hour, key=lambda hour: abs(by_hour[hour].bias))
        for hour in sorted(by_hour):
            score = by_hour[hour]
            bar = ("+" if score.bias > 0 else "-") * min(20, int(abs(score.bias) * 20))
            logging.info(
                f"      {hour:02d}:00{score.n:>6}{score.bias:>10.3f}"
                f"{score.mae:>10.3f}  {bar}"
            )
        worst = by_hour[worst_hour]
        if abs(worst.bias) > 0.1:
            direction = "te hoog" if worst.bias > 0 else "te laag"
            logging.warning(
                f"{component} wordt rond {worst_hour:02d}:00 structureel "
                f"{direction} ingeschat ({worst.bias:+.3f} per interval). Daardoor "
                f"reserveert de optimalisatie de verkeerde hoeveelheid energie; "
                f"de snelle regellaag kan dat achteraf niet repareren."
            )


    def fast_control_simulate(self):
        """Backtest the fast control layer on the recorded history.

        Exposed as a task so it can be started from the dashboard and from
        ``GET /api/run/fast_control_simulate``; the result lands in the task
        log. Use ``python3 da_fast.py simulate`` for other periods.
        """
        from da_fast import main as fast_main

        days = 7
        if len(sys.argv) > 2:
            try:
                days = int(sys.argv[2])
            except ValueError:
                logging.warning(f"Ongeldig aantal dagen: {sys.argv[2]}, 7 gebruikt")
        fast_main(
            [
                "--options",
                self.file_name or "../data/options.json",
                "simulate",
                "--days",
                str(days),
            ]
        )

    def calc_solar_predictions(
        self,
        solar_option: dict,
        vanaf: datetime.datetime,
        tot: datetime.datetime,
        interval: str = None,
    ) -> pd.DataFrame:
        """
        berekent de solar production
        :param solar_option: dict van de solar-device
        :param vanaf: datetime start
        :param tot: datetime tot
        :param interval: 15"min of 1 hour of None, als None wordt self.interval genomen
        :return: dataframe met kolommen tijd en prediction

        Welk model dat doet (fysisch of ML) bepaalt de PV-service zelf uit
        de opgeslagen keuze; de terugval naar het fysische model zit daar
        ook, zodat elke aanroeper dezelfde kolommen terugkrijgt.
        """
        return self.pv_service().forecast(
            solar_option, vanaf, tot, interval or self.interval
        )

    def train_ml_predictions(self):
        # Calibrates the physical model for every installation and, for
        # those configured for ml/auto, trains the ML model too -- in that
        # order, since the ML model's own features include the physical
        # model's (now current) output.
        self.pv_service().run_training()

    def run_task_function(self, task, logfile: bool = True):
        """Run *task* in this process, logging it to its own file.

        Every handler this adds to the root logger is removed again in the
        finally, which is what the old version got wrong in three ways:

        * the cleanup sat *after* a ``try`` that re-raised, so a task that
          failed -- exactly when you want the log -- never reached it;
        * ``removeHandler`` was never called for any of the three, only
          ``close()``, and only for two of them. A closed FileHandler that
          is still attached is worse than one that is left open: its
          ``emit`` reopens the file on the next record, so the task's log
          file was quietly reopened and appended to after the task had
          finished (``main()`` logs the pool status after this returns);
        * the NotificationHandler was added outside the ``if logfile:``
          block and removed nowhere at all, so a second call in one process
          left two attached and pushed every warning to Home Assistant
          twice.
        """
        if task not in self.tasks:
            logging.error(f"Onbekende taak: {task}")
            return
        run_task = self.tasks[task]
        function_name = run_task.get("function")
        if not function_name:
            # fast_once and friends exist only as a subprocess; there is no
            # method to call here.
            logging.error(
                f"Taak {task} kan niet in dit proces draaien, "
                f"gebruik het commando: {' '.join(run_task['cmd'])}"
            )
            return

        logger = logging.getLogger()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        added: list[Handler] = []
        replaced: list[Handler] = []
        file_handler = None
        # The handlers below are given self.log_level, which only has any
        # effect if the logger itself passes those records on. __init__ sets
        # the root level, but relying on that made this method silently
        # depend on how it was reached; set it here too and restore it after.
        previous_level = logger.level
        try:
            logger.setLevel(self.log_level)
            if logfile:
                # The task's output belongs in its own file, so whatever was
                # configured before (the basicConfig handler from __init__)
                # is set aside for the duration and restored in the finally.
                replaced = logger.handlers[:]
                for handler in replaced:
                    logger.removeHandler(handler)
                file_name = (
                    "../data/log/"
                    + run_task["file_name"]
                    + "_"
                    + datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
                    + ".log"
                )
                file_handler = logging.FileHandler(file_name)
                file_handler.setLevel(self.log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                added.append(file_handler)
                # Also to stdout, which is what Home Assistant's supervisor
                # captures as the add-on log.
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                stream_handler.setLevel(self.log_level)
                logger.addHandler(stream_handler)
                added.append(stream_handler)
            if self.notification_entity is not None:
                notification_handler = NotificationHandler(
                    _hass=self, _entity=self.notification_entity
                )
                notification_handler.setFormatter(formatter)
                logger.addHandler(notification_handler)
                added.append(notification_handler)

            self.start_logging()
            logging.info(
                f"Day Ahead Optimalisatie gestart: "
                f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} "
                f"taak: {function_name}"
            )
            self.db_da.log_pool_status()
            getattr(self, function_name)()
            self.set_last_activity()
            self.db_da.log_pool_status()
        except Exception:
            logging.exception("Er is een fout opgetreden, zie de fout-tracering")
            raise
        finally:
            # Detach first, then close: a handler that is closed while still
            # attached gets used again by the next log record.
            for handler in added:
                logger.removeHandler(handler)
            if file_handler is not None:
                file_handler.flush()
            for handler in added:
                try:
                    handler.close()
                except Exception:  # noqa: BLE001 - closing must not mask the task's own error
                    pass
            for handler in replaced:
                logger.addHandler(handler)
            logger.setLevel(previous_level)

    def run_task_cmd(self, task):
        if task not in self.tasks:
            logging.error(f"Onbekende taak: {task}")
            return
        run_task = self.tasks[task]
        cmd = run_task["cmd"]
        proc = run(cmd, stdout=PIPE, stderr=PIPE)
        data = proc.stdout.decode()
        err = proc.stderr.decode()
        log_content = data + err
        filename = (
            "../data/log/"
            + run_task["file_name"]
            + "_"
            + datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")
            + ".log"
        )
        with open(filename, "w") as f:
            f.write(log_content)

        """
        # klass = globals()["class_name"]
        # instance = klass()

        # oude task
        if task not in self.tasks:
            return
        run_task = self.tasks[task]

        # old_stdout = sys.stdout
        # log_file = open("../data/log/" + run_task["file_name"] + "_" +
        #                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") + ".log", "w")
        # sys.stdout = log_file
        try:
            logging.info(f"Day Ahead Optimalisatie gestart: "
                         f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} "
                         f" taak: {run_task['task']}")
            getattr(self, run_task["task"])()
            self.set_last_activity()
        except Exception as ex:
            logging.error(ex)
            logging.error(error_handling())
        # log_file.flush()
        # sys.stdout = old_stdout
        # log_file.close()
        """
