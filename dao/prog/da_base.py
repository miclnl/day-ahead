import datetime
import sys
import os
import errno
import fnmatch
import time
from requests import get
import json
import hassapi as hass
import pandas as pd
from subprocess import PIPE, run
import logging
from logging import Handler
from sqlalchemy import Table, select, func, and_
from dao.prog.utils import get_tibber_data, error_handling
from dao.prog.version import __version__
from dao.prog.da_config import Config
from dao.prog.da_meteo import Meteo
from dao.prog.da_prices import DaPrices

# from db_manager import DBmanagerObj
from typing import Union
from hassapi.models import StateList


class NotificationHandler(Handler):
    def __init__(self, _hass: hass.Hass, _entity=None):
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
            self.hass.set_value(self.entity, msg)


class DaBase(hass.Hass):
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
        try:
            self.config = Config(self.file_name)
        except ValueError:
            self.config = None
            return
        log_level_str = self.config.get(["logging level"], None, "info")
        _log_level = getattr(logging, log_level_str.upper(), None)
        if not isinstance(_log_level, int):
            raise ValueError("Invalid log level: %s" % _log_level)
        self.log_level = _log_level
        logging.addLevelName(logging.DEBUG, "debug")
        logging.addLevelName(logging.INFO, "info")
        logging.addLevelName(logging.WARNING, "waarschuwing")
        logging.addLevelName(logging.ERROR, "fout")
        logging.addLevelName(logging.CRITICAL, "kritiek")
        logging.getLogger().setLevel(self.log_level)
        self.protocol_api = self.config.get(
            ["homeassistant", "protocol api"], default="http"
        )

        ha_options = self.config.get(["homeassistant"])
        if "ip adress" in ha_options:
            self.ip_address = self.config.get(
                ["homeassistant", "ip adress"], default="supervisor"
            )
            logging.warning(
                f"the use of 'ip adress' is deprecated, change it to 'host' "
                f"in the near future 'ip adress' cannot be used any more."
            )
        else:
            self.ip_address = self.config.get(
                ["homeassistant", "host"], default="supervisor"
            )

        self.ip_port = self.config.get(["homeassistant", "ip port"], default=None)
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
        self.hasstoken = self.config.get(
            ["homeassistant", "token"], default=os.environ.get("SUPERVISOR_TOKEN")
        )
        super().__init__(hassurl=self.hassurl, token=self.hasstoken)
        headers = {
            "Authorization": "Bearer " + self.hasstoken,
            "content-type": "application/json",
        }
        resp = get(self.hassurl + "api/config", headers=headers)
        resp_dict = json.loads(resp.text)
        # logging.debug(f"hass/api/config: {resp.text}")
        self.config.set("latitude", resp_dict["latitude"])
        self.config.set("longitude", resp_dict["longitude"])
        self.config.set("time_zone", resp_dict["time_zone"])
        self.db_da = self.config.get_db_da()
        self.db_ha = self.config.get_db_ha()
        self.meteo = Meteo(self.config, self.db_da)
        self.solar = self.config.get(["solar"])

        self.prices = DaPrices(self.config, self.db_da)
        self.prices_options = self.config.get(["prices"])
        # eb + ode levering
        self.taxes_l_def = self.config.get(
            ["energy taxes consumption"], self.prices_options, None
        )
        if self.taxes_l_def is None:
            logging.warning(f"Vervang 'delivery' in je settings door 'consumption'")
            self.taxes_l_def = self.config.get(
                ["energy taxes delivery"], self.prices_options, None
            )
        # opslag kosten leverancier
        self.ol_l_def = self.config.get(
            ["cost supplier consumption"], self.prices_options, None
        )
        if self.ol_l_def is None:
            logging.warning(f"Vervang 'delivery' in je settings door 'consumption'")
            self.ol_l_def = self.config.get(
                ["cost supplier delivery"], self.prices_options, None
            )
        # eb+ode teruglevering
        self.taxes_t_def = self.config.get(
            ["energy taxes production"], self.prices_options, None
        )
        if self.taxes_t_def is None:
            logging.warning(f"Vervang 'redelivery' in je settings door 'production'")
            self.taxes_t_def = self.config.get(
                ["energy taxes redelivery"], self.prices_options, None
            )
        self.ol_t_def = self.config.get(
            ["cost supplier production"], self.prices_options, None
        )
        if self.ol_t_def is None:
            logging.warning(f"Vervang 'redelivery' in je settings door 'production'")
            self.ol_t_def = self.config.get(
                ["cost supplier redelivery"], self.prices_options, None
            )
        if "vat consumption" in self.prices_options:
            self.btw_l_def = self.prices_options["vat consumption"]
        else:
            self.btw_l_def = self.prices_options["vat"]
        if "vat production" in self.prices_options:
            self.btw_t_def = self.prices_options["vat production"]
        else:
            self.btw_t_def = self.btw_l_def.copy()
        self.salderen = (
            self.config.get(["tax refund"], self.prices_options, "true").lower()
            == "true"
        )

        self.history_options = self.config.get(["history"])
        self.strategy = self.config.get(["strategy"], None, "minimize cost").lower()
        self.tibber_options = self.config.get(["tibber"], None, None)
        self.notification_entity = self.config.get(
            ["notifications", "notification entity"], None, None
        )
        self.notification_opstarten = self.config.get(
            ["notifications", "opstarten"], None, False
        )
        if (
            type(self.notification_opstarten) is str
            and self.notification_opstarten.lower() == "true"
        ):
            self.notification_opstarten = True
        else:
            self.notification_opstarten = False
        self.notification_berekening = self.config.get(
            ["notifications", "berekening"], None, False
        )
        if (
            type(self.notification_berekening) is str
            and self.notification_berekening.lower() == "true"
        ):
            self.notification_berekening = True
        else:
            self.notification_berekening = False
        self.last_activity_entity = self.config.get(
            ["notifications", "last activity entity"], None, None
        )
        self.set_last_activity()
        self.graphics_options = self.config.get(["graphics"])
        self.db_da.log_pool_status()

    def set_value(self, entity_id: str, value: Union[int, float, str]) -> StateList:
        try:
            result = super().set_value(entity_id, value)
            state = self.get_state(entity_id).state
            if isinstance(value, (int, float)):
                if round(float(state), 5) != round(float(value), 5):
                    raise ValueError
            else:
                if state != value:
                    raise ValueError
        except Exception:
            logging.error(f"Fout bij schrijven naar {entity_id}, waarde {value}")
            # error_handling(ex)
            raise
        return result

    @staticmethod
    def generate_tasks():
        tasks = {
            "calc_optimum_met_debug": {
                "name": "Optimaliseringsberekening met debug",
                "cmd": ["python3", "../prog/day_ahead.py", "debug", "calc"],
                "object": "DaCalc",
                "function": "calc_optimum_met_debug",
                "file_name": "calc_debug",
            },
            "calc_optimum": {
                "name": "Optimaliseringsberekening zonder debug",
                "cmd": ["python3", "../prog/day_ahead.py", "calc"],
                "function": "calc_optimum",
                "file_name": "calc",
            },
            "tibber": {
                "name": "Verbruiksgegevens bij Tibber ophalen",
                "cmd": ["python3", "../prog/day_ahead.py", "tibber"],
                "function": "get_tibber_data",
                "file_name": "tibber",
            },
            "meteo": {
                "name": "Meteoprognoses ophalen",
                "cmd": ["python3", "day_ahead.py", "meteo"],
                "function": "get_meteo_data",
                "file_name": "meteo",
            },
            "prices": {
                "name": "Day ahead prijzen ophalen",
                "cmd": ["python3", "../prog/day_ahead.py", "prices"],
                "function": "get_day_ahead_prices",
                "file_name": "prices",
            },
            "calc_baseloads": {
                "name": "Bereken de baseloads",
                "cmd": ["python3", "../prog/day_ahead.py", "calc_baseloads"],
                "function": "calc_baseloads",
                "file_name": "baseloads",
            },
            "calc_pr": {
                "name": "Kalibreer PV PR-correcties",
                "cmd": ["python3", "../prog/day_ahead.py", "calc_pr"],
                "function": "calc_pr",
                "file_name": "calc_pr",
            },
            "clean": {
                "name": "Bestanden opschonen",
                "cmd": ["python3", "../prog/day_ahead.py", "clean_data"],
                "function": "clean_data",
                "file_name": "clean",
            },
            "consolidate": {
                "name": "Verbruik/productie consolideren",
                "cmd": ["python3", "../prog/day_ahead.py", "consolidate"],
                "function": "consolidate_data",
                "file_name": "consolidate",
            },
        }
        return tasks

    def start_logging(self):
        logging.debug(f"python pad:{sys.path}")
        logging.info(f"Day Ahead Optimalisering versie: {__version__}")
        logging.info(
            f"Day Ahead Optimalisering gestart op: "
            f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )
        if self.config is not None:
            logging.debug(
                f"Locatie: latitude {str(self.config.get(['latitude']))} "
                f"longitude: {str(self.config.get(['longitude']))}"
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
        self.prices.get_prices(
            self.config.get(["source day ahead"], self.prices_options, "nordpool")
        )

    def save_df(self, tablename: str, tijd: list, df: pd.DataFrame):
        """
        Slaat de data in het dataframe op in de tabel "table"
        :param tablename: de naam van de tabel waarin de data worden opgeslagen
        :param tijd: de datum tijd van de rijen in het dataframe
        :param df: het dataframe met de code van de variabelen in de kolomheader
        :return: None
        """
        df_db = pd.DataFrame(columns=["time", "code", "value"])
        df = df.reset_index(drop=True)
        columns = df.columns.values.tolist()[1:]
        for index in range(len(tijd)):
            utc = int(tijd[index].timestamp())
            for c in columns:
                db_row = [str(utc), c, float(df.loc[index, c])]
                df_db.loc[df_db.shape[0]] = db_row
        logging.debug("Save calculated data:\n{}".format(df_db.to_string()))
        self.db_da.savedata(df_db, tablename=tablename)
        return

    @staticmethod
    def get_calculated_baseload(weekday: int) -> list:
        """
        Haalt de berekende baseload op voor de weekdag.
        :param weekday: : 0 = maandag, 6 zondag
        :return: een lijst van eerder berekende baseload van 24uurvoor de betreffende dag
        """
        in_file = "../data/baseload/baseload_" + str(weekday) + ".json"
        with open(in_file, "r") as f:
            result = json.load(f)
        return result

    def calc_prod_solar(
        self,
        solar_opt: dict,
        act_time: int,
        act_gr: float,
        hour_fraction: float,
        apply_pr: bool = True,
    ):
        """
        Berekent de productie van een zonnepanelen string/array met PR- en temperatuurcorrecties.
        - Ondersteunt per-string/array velden: "power temp coeff %/C", "voltage temp coeff %/C",
          "current temp coeff %/C", "NOCT (°C)" (NOCT informatief, niet gebruikt in deze formule).
        - Globale defaults uit pv.*: pr_factor, pr_hourly_cal, temp_coeff_pct_per_C, temp_ref_C, panel_temp_offset_C.
        - Oriëntatie: gebruikt "orientation" (oost=-90, zuid=0, west=+90). Indien alleen "azimuth" (0–360) aanwezig,
          wordt omgezet naar orientation via: ((azimuth - 180 + 540) % 360) - 180.
        """
        # Bepaal uur en datum
        try:
            moment = datetime.datetime.fromtimestamp(int(act_time))
            hour_idx = moment.hour
            day_date = datetime.datetime(moment.year, moment.month, moment.day)
        except Exception:
            moment = datetime.datetime.now()
            hour_idx = moment.hour
            day_date = datetime.datetime(moment.year, moment.month, moment.day)

        # PR-factoren (globaal) — optioneel toe te passen
        if apply_pr:
            pr_factor = float(self.config.get(["pv", "pr_factor"], None, 1.0) or 1.0)
            pr_hourly = self.config.get(["pv", "pr_hourly_cal"], None, None)
            if isinstance(pr_hourly, list) and len(pr_hourly) == 24:
                try:
                    pr_hour = float(pr_hourly[hour_idx])
                except Exception:
                    pr_hour = 1.0
            else:
                pr_hour = 1.0
        else:
            pr_factor = 1.0
            pr_hour = 1.0

        # Temperatuurparameters (globaal)
        # Globale power temp coeff in factor/°C; default -0.004
        global_temp_coeff = float(self.config.get(["pv", "temp_coeff_pct_per_C"], None, -0.004) or -0.004)
        temp_ref = float(self.config.get(["pv", "temp_ref_C"], None, 25.0) or 25.0)
        panel_offset = float(self.config.get(["pv", "panel_temp_offset_C"], None, 20.0) or 20.0)

        # Voorspelde omgevingstemperatuur (uurwaarde) en paneeltemperatuur benadering
        try:
            amb_temp = float(self.meteo.get_hour_temperature(moment))
        except Exception:
            try:
                amb_temp = float(self.meteo.get_avg_temperature(day_date))
            except Exception:
                amb_temp = 15.0
        panel_temp = amb_temp + panel_offset

        def calc_temp_coeff(local: dict) -> float:
            # Voorkeur: expliciete power temp coeff (in %/°C)
            pct = local.get("power temp coeff %/C")
            if pct is not None:
                try:
                    return float(pct) / 100.0
                except Exception:
                    pass
            # Anders: som van voltage + current temp coeffs (in %/°C)
            v_pct = local.get("voltage temp coeff %/C")
            i_pct = local.get("current temp coeff %/C")
            try:
                if v_pct is not None or i_pct is not None:
                    v = float(v_pct) if v_pct is not None else 0.0
                    i = float(i_pct) if i_pct is not None else 0.0
                    return (v + i) / 100.0
            except Exception:
                pass
            # Fallback: globale coeff
            return global_temp_coeff

        def ensure_orientation(local: dict) -> dict:
            if "orientation" not in local and "azimuth" in local:
                try:
                    az = float(local["azimuth"])  # 0..360 (180=zuid)
                    # convert to orientation: oost=-90, zuid=0, west=+90
                    ori = ((az - 180.0 + 540.0) % 360.0) - 180.0
                    local = dict(local)
                    local["orientation"] = ori
                except Exception:
                    pass
            return local

        # Basisproductie zonder correcties
        if "strings" in solar_opt:
            prod = 0.0
            for s_def in solar_opt["strings"]:
                s_def = ensure_orientation(s_def)
                poa_j_cm2 = self.meteo.calc_solar_rad(s_def, act_time, act_gr)
                base = poa_j_cm2 * float(s_def.get("yield", 0.0)) * hour_fraction
                coeff = calc_temp_coeff(s_def)
                # NOCT-gereguleerde paneeltemperatuur indien beschikbaar
                noct = s_def.get("NOCT (°C)")
                if noct is not None:
                    try:
                        noct_val = float(noct)
                        # omzetting naar W/m²: J/cm² per uur -> W/m²
                        poa_w_m2 = (poa_j_cm2 / 3600.0) * 10000.0
                        panel_temp_loc = amb_temp + ((noct_val - 20.0) / 800.0) * poa_w_m2
                    except Exception:
                        panel_temp_loc = panel_temp
                else:
                    panel_temp_loc = panel_temp
                temp_factor = 1.0 + coeff * (panel_temp_loc - temp_ref)
                temp_factor = max(0.7, min(1.05, temp_factor))
                prod += base * temp_factor
        else:
            local = ensure_orientation(solar_opt)
            poa_j_cm2 = self.meteo.calc_solar_rad(local, act_time, act_gr)
            base = poa_j_cm2 * float(local.get("yield", 0.0)) * hour_fraction
            coeff = calc_temp_coeff(local)
            noct = local.get("NOCT (°C)")
            if noct is not None:
                try:
                    noct_val = float(noct)
                    poa_w_m2 = (poa_j_cm2 / 3600.0) * 10000.0
                    panel_temp_loc = amb_temp + ((noct_val - 20.0) / 800.0) * poa_w_m2
                except Exception:
                    panel_temp_loc = panel_temp
            else:
                panel_temp_loc = panel_temp
            temp_factor = 1.0 + coeff * (panel_temp_loc - temp_ref)
            temp_factor = max(0.7, min(1.05, temp_factor))
            prod = base * temp_factor

        # PR-correcties (ondersteun maand-specifieke overrides)
        if apply_pr:
            month_idx = moment.month
            pr_factor_by_month = self.config.get(["pv", "pr_factor_by_month"], None, None) or {}
            try:
                pr_factor_m = float(pr_factor_by_month.get(str(month_idx))) if isinstance(pr_factor_by_month, dict) else None
            except Exception:
                pr_factor_m = None
            pr_factor_eff = pr_factor_m if pr_factor_m is not None else pr_factor

            pr_hour_m = pr_hour
            pr_hourly_by_month = self.config.get(["pv", "pr_hourly_cal_by_month"], None, None)
            if isinstance(pr_hourly_by_month, dict):
                arr = pr_hourly_by_month.get(str(month_idx))
                if isinstance(arr, list) and len(arr) == 24:
                    try:
                        pr_hour_m = float(arr[hour_idx])
                    except Exception:
                        pr_hour_m = pr_hour

            prod *= pr_factor_eff * pr_hour_m

        # Begrenzen op max vermogen (indien opgegeven)
        max_power = self.config.get(["max power"], solar_opt, None)
        if max_power is not None:
            try:
                prod = min(prod, float(max_power))
            except Exception:
                pass
        return prod

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

    def set_entity_value(
        self, entity_key: str, options: dict, value: int | float | str
    ):
        entity_id = self.config.get([entity_key], options, None)
        if entity_id is not None:
            self.set_value(entity_id, value)

    def set_entity_option(
        self, entity_key: str, options: dict, value: int | float | str
    ):
        entity_id = self.config.get([entity_key], options, None)
        if entity_id is not None:
            self.select_option(entity_id, value)

    def set_entity_state(
        self, entity_key: str, options: dict, value: int | float | str
    ):
        entity_id = self.config.get([entity_key], options, None)
        if entity_id is not None:
            self.set_state(entity_id, value)

    def clean_data(self):
        """
        takes care for cleaning folders data/log and data/images
        """

        def clean_folder(folder: str, pattern: str):
            current_time = time.time()
            day = 24 * 60 * 60
            logging.info(f"Start removing files in {folder} with pattern {pattern}")
            current_dir = os.getcwd()
            os.chdir(os.path.join(os.getcwd(), folder))
            list_files = os.listdir()
            for f in list_files:
                if fnmatch.fnmatch(f, pattern):
                    creation_time = os.path.getctime(f)
                    if (current_time - creation_time) >= self.config.get(
                        ["save days"], self.history_options, 7
                    ) * day:
                        os.remove(f)
                        logging.info(f"{f} removed")
            os.chdir(current_dir)

        clean_folder("../data/log", "*.log")
        clean_folder("../data/log", "dashboard.log.*")
        clean_folder("../data/images", "*.png")

    def calc_optimum_met_debug(self):
        from day_ahead import DaCalc

        dacalc = DaCalc(self.file_name)
        # dacalc = DaCalc("../data/test.json")
        dacalc.debug = True
        dacalc.calc_optimum()
        # dacalc.calc_optimum(_start_dt=datetime.datetime(2025, 5, 17, 7))

    def calc_optimum(self):
        from day_ahead import DaCalc

        dacalc = DaCalc(self.file_name)
        dacalc.debug = False
        dacalc.calc_optimum()

    @staticmethod
    def calc_baseloads():
        from da_report import Report

        report = Report()
        report.calc_save_baseloads()

    def calc_pr(self):
        """Calibreer PR-correcties (globaal, per uur, per maand) op basis van historische data.
        Schrijft resultaten terug in options.json (pv.pr_factor, pv.pr_hourly_cal, pv.pr_factor_by_month).
        """
        import numpy as np
        import pandas as pd

        if self.config is None:
            logging.error("Geen config geladen; PR-calibratie afgebroken")
            return
        # Vereiste sensoren voor actuele PV (AC)
        report_opts = self.config.get(["report"])
        sensors_ac = self.config.get(["entities solar production ac"], report_opts, []) or []
        if not isinstance(sensors_ac, list) or len(sensors_ac) == 0:
            logging.error("Geen 'report.entities solar production ac' sensoren gedefinieerd")
            return

        # Bereik bepalen
        days = int(self.config.get(["pv", "pr_calibration_days"], None, 60))
        now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        start = now - datetime.timedelta(days=days)
        end = now

        # Historische globale straling (gr) uit DA-db
        df_gr = self.db_da.get_column_data("values", "gr", start, end)
        if df_gr is None or df_gr.empty:
            logging.error("Geen globale stralingsdata (gr) gevonden in 'values'")
            return
        df_gr = df_gr.rename(columns={"value": "gr"})
        df_gr["time"] = pd.to_datetime(df_gr["time"])
        df_gr.set_index("time", inplace=True)

        # Actuele PV uit HA-db (som van AC-productie sensoren), per uur
        from dao.prog.da_report import Report
        report = Report(self.file_name)
        df_actual = None
        for sensor in sensors_ac:
            try:
                df_s = report.get_sensor_data(sensor, start, end, col_name="pv", agg="uur")
                df_s = df_s.rename(columns={"tijd": "time", "pv": "value"})
                df_s["time"] = pd.to_datetime(df_s["time"])  # ensure datetime
                df_s = df_s[["time", "value"]]
                df_s.set_index("time", inplace=True)
                df_actual = df_s if df_actual is None else df_actual.add(df_s, fill_value=0.0)
            except Exception as ex:
                logging.warning(f"Sensor {sensor} niet opgehaald: {ex}")
        if df_actual is None or df_actual.empty:
            logging.error("Geen PV-actuals gevonden uit HA-db")
            return

        # Verwachte PV met fysisch model zonder PR (apply_pr=False)
        # Gebruik 'solar' definitie voor arrays
        if not isinstance(self.solar, list) or len(self.solar) == 0:
            logging.error("Geen 'solar' configuratie aanwezig")
            return
        # Combineer gr met expected
        df = pd.DataFrame(index=df_gr.index)
        df["gr"] = df_gr["gr"]
        def _expect_at(ts: pd.Timestamp, gr_val: float) -> float:
            try:
                if pd.isna(gr_val):
                    return 0.0
                tot = 0.0
                for s in range(len(self.solar)):
                    tot += self.calc_prod_solar(self.solar[s], int(ts.timestamp()), float(gr_val), 1.0, apply_pr=False)
                return float(tot)
            except Exception:
                return 0.0
        df["exp"] = [
            _expect_at(ts, df.loc[ts, "gr"]) for ts in df.index
        ]
        # Align met actuals (inner join op uren)
        df = df.join(df_actual.rename(columns={"value": "act"}), how="inner")
        df = df[(df["exp"] > 1e-6) & (df["act"] >= 0.0)]
        if df.empty:
            logging.error("Onvoldoende overlappende data voor PR-calibratie")
            return

        # Globale factor
        pr_global = float(df["act"].sum() / max(df["exp"].sum(), 1e-6))

        # Uurprofiel
        df["hour"] = df.index.hour
        hourly = df.groupby("hour").apply(lambda g: float((g["act"].sum()) / max(g["exp"].sum(), 1e-6)))
        pr_hourly = [float(hourly.get(h, pr_global)) for h in range(24)]

        # Maandfactoren
        df["month"] = df.index.month
        by_month = df.groupby("month").apply(lambda g: float((g["act"].sum()) / max(g["exp"].sum(), 1e-6)))
        pr_by_month = {str(int(m)): float(by_month[m]) for m in sorted(by_month.index)}

        # Schrijf naar options.json (pv.*)
        try:
            with open(self.file_name, "r") as f:
                opts = json.load(f)
            pv = opts.get("pv", {})
            pv["pr_factor"] = round(pr_global, 4)
            pv["pr_hourly_cal"] = [round(x, 4) for x in pr_hourly]
            pv["pr_factor_by_month"] = {k: round(v, 4) for k, v in pr_by_month.items()}
            opts["pv"] = pv
            with open(self.file_name, "w") as f:
                json.dump(opts, f, indent=2)
            logging.info(
                f"PR-calibratie opgeslagen: pr_factor={pv['pr_factor']}, pr_factor_by_month={pv['pr_factor_by_month']}"
            )
        except Exception as ex:
            logging.error(f"Fout bij opslaan van PR-calibratie in options.json: {ex}")

    def run_task_function(self, task, logfile: bool = True):
        # klass = globals()["class_name"]
        # instance = klass()

        # oude task
        if task not in self.tasks:
            return
        run_task = self.tasks[task]
        file_handler = None
        stream_handler = None
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        if logfile:
            # old_stdout = sys.stdout
            for handler in logger.handlers[:]:  # make a copy of the list
                logger.removeHandler(handler)
            file_name = (
                "../data/log/"
                + run_task["file_name"]
                + "_"
                + datetime.datetime.now().strftime("%Y-%m-%d__%H:%M")
                + ".log"
            )

            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(self.log_level)
            logger.addHandler(stream_handler)
        if self.notification_entity is not None:
            notification_handler = NotificationHandler(
                _hass=super(), _entity=self.notification_entity
            )
            notification_handler.setFormatter(formatter)
            logger.addHandler(notification_handler)
        # ---- Simple cross-process lock using lockfiles in /data/run ----
        def _run_dir() -> str:
            path = os.path.abspath(os.path.join(os.getcwd(), "../data", "run"))
            os.makedirs(path, exist_ok=True)
            return path

        def _lock_path(task_key: str) -> str:
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_key)
            return os.path.join(_run_dir(), f"{safe}.lock")

        def _acquire_task_lock(task_key: str) -> bool:
            path = _lock_path(task_key)
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(datetime.datetime.now()))
                return True
            except OSError as ex:
                if ex.errno == errno.EEXIST:
                    return False
                raise

        def _release_task_lock(task_key: str) -> None:
            path = _lock_path(task_key)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        def _wait_for_locks(task_keys: list[str], timeout_seconds: int) -> bool:
            """Wait until none of the given locks exist. Returns True if all cleared within timeout, else False."""
            import time as _time
            end = _time.time() + timeout_seconds if timeout_seconds > 0 else None
            while True:
                busy = [k for k in task_keys if os.path.exists(_lock_path(k))]
                if not busy:
                    return True
                if end is not None and _time.time() > end:
                    logging.warning(f"Wachttijd verstreken; ga verder ondanks actieve locks: {busy}")
                    return False
                logging.info(f"Wachten op afronden van taken: {', '.join(busy)}")
                _time.sleep(2.0)

        self.start_logging()
        try:
            # Voor calc-optimum: wacht op lopende meteo/prices/baseloads
            if run_task["function"] in ("calc_optimum", "calc_optimum_met_debug"):
                wait_seconds = int(self.config.get(["scheduler", "wait_seconds_for_calc"], None, 600))
                _wait_for_locks(["meteo", "prices", "calc_baseloads"], wait_seconds)

            # Acquire lock for this task (cross-process de-dup)
            acquired = _acquire_task_lock(task)
            if not acquired:
                logging.info(f"Taak '{task}' draait al; overslaan.")
                return
            logging.info(
                f"Day Ahead Optimalisatie gestart: "
                f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')} "
                f"taak: {run_task['function']}"
            )
            self.db_da.log_pool_status()
            getattr(self, run_task["function"])()
            self.set_last_activity()
            self.db_da.log_pool_status()
        except Exception:
            logging.exception("Er is een fout opgetreden, zie de fout-tracering")
            raise
        finally:
            # release lock and handlers
            try:
                _release_task_lock(task)
            except Exception:
                pass
            if logfile:
                try:
                    file_handler.flush()
                    file_handler.close()
                    stream_handler.close()
                except Exception:
                    pass

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
