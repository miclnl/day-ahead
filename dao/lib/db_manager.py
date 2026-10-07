import math
import numbers
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import (
    create_engine,
    Table,
    MetaData,
    select,
    insert,
    update,
    func,
    and_,
    text,
    TIMESTAMP,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import URL
import sqlalchemy_utils
import os
import logging
from zoneinfo import ZoneInfo

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    bindparam,
    delete,
    inspect,
)
from sqlalchemy.exc import NoSuchTableError

from dao.prog.utils import interpolate


# import utils as utils

#: Lead time buckets in hours used by the forecast archive.
#:
#: Storing every forecast of every run would grow without bound and is not what
#: you want to analyse anyway. Keeping exactly one row per (variable, target,
#: bucket) caps the table at ``codes x targets x 5`` rows no matter how often
#: the optimizer runs, which keeps it small enough for the eMMC of a Home
#: Assistant Yellow while still answering "how good is my forecast a day out?".
LEAD_BUCKETS = (0, 1, 4, 12, 24)

#: Which forecasts are worth archiving.
#:
#: Deliberately short. These are the series whose error actually moves the
#: plan: the net house demand, the PV production (AC and DC), the baseload,
#: and the weather inputs they are derived from. Archiving every optimizer
#: output would multiply the table for no analytical gain.
#:
#: ``winds`` is the one entry nothing reports on. It is here because the
#: physical PV model's cell temperature needs it, and an installation
#: without local weather observations has only this archive to calibrate
#: and backtest against -- without it every such run came out NaN.
ARCHIVED_FORECAST_CODES = frozenset(
    {"hload", "pv_ac", "pv_dc", "base", "gr", "dni", "dhi", "temp", "winds"}
)


def lead_bucket(lead_hours: float) -> int:
    """Largest bucket that is still below or equal to *lead_hours*."""
    chosen = LEAD_BUCKETS[0]
    for bucket in LEAD_BUCKETS:
        if lead_hours >= bucket:
            chosen = bucket
        else:
            break
    return chosen


def forecasts_table(metadata: MetaData) -> Table:
    """Define the forecast archive on *metadata*, which must hold "variabel".

    Unlike "values" and "prognoses" this table keeps the lead time at which
    a forecast was made, so forecast quality can be measured afterwards. The
    unique key caps it at one row per (variable, target, lead bucket), which
    bounds its size regardless of how often the optimizer runs.

    The "variabel" column deliberately carries no type of its own.
    MySQL and MariaDB accept a foreign key only when both columns have
    exactly the same type, signedness included, and databases that predate
    the schema moving into Python have "variabel.id" as INT(10) UNSIGNED.
    Writing Integer here renders a signed INTEGER and the server refuses the
    table with errno 150, "Foreign key constraint is incorrectly formed".
    Leaving the type out makes SQLAlchemy copy it from the column the key
    points at, so this works on both the old and the new schema. That only
    holds when "variabel" in *metadata* was reflected from the database
    rather than declared here; see :meth:`DBmanagerObj.ensure_forecasts_table`.
    """
    return Table(
        "forecasts",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column(
            "variabel",
            ForeignKey("variabel.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("target_time", BigInteger, nullable=False),
        Column("lead_bucket", Integer, nullable=False),
        Column("issued_time", BigInteger, nullable=False),
        Column("value", Float),
        Column("source", String(16), nullable=True),
        UniqueConstraint("variabel", "target_time", "lead_bucket"),
        sqlite_autoincrement=True,
        extend_existing=True,
    )


def _container_zone_name() -> str:
    """The container's own timezone name.

    Used only as a last resort. The Home Assistant supervisor sets TZ for
    add-ons, which is why SQLite's "localtime" and MySQL's session zone have
    been giving the right answers all along; this makes the same assumption
    explicit instead of leaving the zone unset.
    """
    name = os.environ.get("TZ")
    if name:
        return name
    local = datetime.datetime.now().astimezone().tzinfo
    return getattr(local, "key", None) or str(local) or "UTC"


class DBmanagerObj(object):
    """
    Database manager class.
    """

    def __init__(
        self,
        db_dialect: str,
        db_name: str,
        db_server=None,
        db_user=None,
        db_password=None,
        db_port=None,
        db_path=None,
        db_time_zone: str = "Europe/Amsterdam",
    ):
        """
        Initializes a DBManager object
        Args:
            db_dialect   :Dialect: mysql(=mariadb), sqlite, postgresql
            db_name      :Name of the DB
            db_server    :Server/host (mysql, postgresql)
            db_user      :User (mysql, postgresql)
            db_password  :Password (mysql, postgresql)
            db_port      :port(mysql and postgresql via TCP only) Necessary if not default
            db_path      :path if sqlite db (sqlite only)
            db_time_zone :time_zone (postgresql only)
        """

        self.db_dialect = db_dialect
        self.db_name = db_name
        self.server = db_server
        self.user = db_user
        self.password = db_password
        self.port = db_port
        self.db_path = db_path
        # The optional database override from options.json, which is None
        # unless the operator set it. DaBase overwrites this with the zone
        # Home Assistant reports as soon as it knows it (see
        # DaBase.__init__), so the order of authority is: explicit override,
        # then Home Assistant, then the container's own zone.
        self.TARGET_TIMEZONE = db_time_zone or _container_zone_name()

        self.engine = create_engine(
            self.db_url(
                db_dialect=self.db_dialect,
                db_name=self.db_name,
                db_server=self.server,
                db_user=self.user,
                db_password=self.password,
                db_port=self.port,
                db_path=self.db_path,
            ),
            pool_recycle=3600,
            pool_pre_ping=True,
        )

        # Postgres: set timezone
        # with self.engine.connect() as connection:
        # connection.execute(text(f"SET timezone = '{self.TARGET_TIMEZONE}';"))

        # Probe the connection once at construction to fail fast with a clear
        # error message.  Using a context manager returns the connection to the
        # pool on exit regardless of success or failure — the engine stays valid.
        with self.engine.connect():
            pass
        self.metadata = MetaData()

    @staticmethod
    def db_url(
        db_dialect: str,
        db_name: str,
        db_server=None,
        db_user=None,
        db_password=None,
        db_port=0,
        db_path=None,
    ) -> URL:
        """Build the SQLAlchemy URL.

        URL.create() escapes the credentials, so a password with '@', '/' or
        '%' works, and the URL renders with the password hidden in logs.
        """
        if db_dialect in ("mysql", "postgresql"):
            driver = "mysql+pymysql" if db_dialect == "mysql" else "postgresql+psycopg2"
            result = URL.create(
                driver,
                username=db_user,
                password=db_password,
                host=db_server,
                port=int(db_port) if db_port else None,
                database=db_name,
            )
        else:  # sqlite3
            if db_path is None:
                db_path = "../data"
            result = URL.create(
                "sqlite", database=os.path.join(os.path.abspath(db_path), db_name)
            )
        logging.debug(f"db_url: {result.render_as_string(hide_password=True)}")
        return result

    def log_pool_status(self):
        from inspect import currentframe, getframeinfo

        cf = currentframe()
        cf = cf.f_back
        filename = getframeinfo(cf).filename
        lineno = getframeinfo(cf).lineno
        logging.debug(
            f"Connection status {self.engine.pool.status()} at line "
            f"{lineno} in {filename}"
        )

    # Custom function to handle from_unixtime
    @property
    def tzinfo(self) -> ZoneInfo:
        """The configured Home Assistant timezone.

        TARGET_TIMEZONE is config.time_zone (see db_connections.py). It used
        to be stored and then never used: the one line that applied it,
        ``SET timezone``, is commented out, so PostgreSQL rendered and parsed
        everything in its own session zone -- usually UTC -- which is why
        those installations saw every timestamp shifted by one or two hours.
        """
        try:
            return ZoneInfo(self.TARGET_TIMEZONE)
        except Exception:  # noqa: BLE001 - an unknown zone name from HA
            logging.warning(
                f"Onbekende tijdzone {self.TARGET_TIMEZONE!r}, UTC gebruikt"
            )
            return ZoneInfo("UTC")

    def epoch(self, moment: datetime.datetime) -> int:
        """Epoch seconds for *moment*, reading a naive value as local time.

        Every caller used to hand SQL a formatted string
        (``unix_timestamp(moment.strftime(...))``) and let the database parse
        it back into an epoch. Which zone that string was taken to be in
        depended on the dialect and on server settings, so the same query
        selected a different range on SQLite, MySQL and PostgreSQL. The
        column holds epoch integers, so the conversion belongs here, once,
        against the configured zone.
        """
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=self.tzinfo)
        return int(moment.timestamp())

    def _pg_local(self, column):
        """PostgreSQL: the epoch column as a naive timestamp in the configured
        zone.

        ``to_timestamp`` yields a timestamptz, which ``to_char`` then renders
        in the session's TimeZone -- UTC on a stock server, since nothing
        sets it. ``timezone(zone, ...)`` pins it to the zone the operator
        configured in Home Assistant, which is the whole point.
        """
        return func.timezone(self.TARGET_TIMEZONE, func.to_timestamp(column))

    def from_unixtime(self, column):
        if self.db_dialect == "sqlite":
            return func.datetime(column, "unixepoch", "localtime")
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM-DD HH24:MI:SS")
        else:  # mysql/mariadb
            return func.from_unixtime(column)

    # Custom function to handle UNIX_TIMESTAMP
    def unix_timestamp(self, date_str):
        if self.db_dialect == "sqlite":
            return func.strftime("%s", date_str, "utc")
        elif self.db_dialect == "postgresql":
            return func.extract(
                "epoch",
                func.to_timestamp(date_str, "YYYY-MM-DD hh24:mi:ss"),
                # func.timezone(self.TARGET_TIMEZONE, func.cast(date_str, TIMESTAMP)),
                # EXTRACT(epoch FROM to_timestamp('2025-01-07 00:07:00', 'YYYY-MM-DD hh24:mi:ss'))
            )
        else:  # mysql/mariadb
            return func.unix_timestamp(date_str)

    def month(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM")
        else:  # mysql/mariadb
            return func.concat(
                func.year(func.from_unixtime(column)),
                "-",
                func.lpad(func.month(func.from_unixtime(column)), 2, "0"),
            )

    def month_start(self, column):
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-01", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM-01")
        else:  # mysql/mariadb
            return func.concat(
                func.year(func.from_unixtime(column)),
                "-",
                func.lpad(func.month(func.from_unixtime(column)), 2, "0"),
                "-01",
            )

    def day(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-%d", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM-DD")
        else:  # mysql/mariadb
            return func.date(func.from_unixtime(column))

    def day_start(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-%d", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM-DD")
        else:  # mysql/mariadb
            return func.date(func.from_unixtime(column))

    def hour(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%H:00", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "HH24:00")
        else:  # mysql/mariadb
            return func.time_format(func.time(func.from_unixtime(column)), "%H:00")

    def hour_start(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-%d %H:00", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(self._pg_local(column), "YYYY-MM-DD HH24:00")
        else:  # mysql/mariadb
            return func.date_format(func.from_unixtime(column), "%Y-%m-%d %H:00")

    def savedata(self, df: pd.DataFrame, tablename: str = "values"):
        """
        save data in dateframe,
        if id exist then update else insert
        Args:
            df: Dataframe that we wish to save in table tablename
               columns
               code	string
               calculated datetime, 0 if realised
               time	timestamp in sec
               value	float
            tablename: values or prognoses
        """
        if df is None or len(df) == 0:
            return
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Opslaan dataframe:\n{df.to_string()}")

        # One statement per batch instead of three round trips per row, and
        # atomic: the table has UNIQUE(variabel, time), so two writers (the
        # fast-control thread and a scheduler subprocess, for instance) used
        # to race between the SELECT and the INSERT, and the IntegrityError
        # rolled back the whole batch.
        table = Table(tablename, self.metadata, autoload_with=self.engine)
        ids = self.variabel_ids(list(df["code"].unique()))
        records = []
        skipped_codes = set()
        for row in df.itertuples(index=False):
            code = row.code
            if code not in ids:
                skipped_codes.add(code)
                continue
            value = row.value
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            try:
                stamp = int(float(row.time))
            except (TypeError, ValueError):
                logging.warning(f"Ongeldig tijdstip {row.time!r} voor {code}, overgeslagen")
                continue
            records.append({"variabel": ids[code], "time": stamp, "value": value})
        for code in sorted(skipped_codes):
            logging.error(f"Onbekende code opslaan data: {code}")
        if not records:
            return
        # The last value for a (variabel, time) pair wins, as with the old
        # row-by-row update; duplicates within one statement would otherwise
        # make the upsert ambiguous on PostgreSQL.
        deduplicated = {(r["variabel"], r["time"]): r for r in records}
        records = list(deduplicated.values())

        self.log_pool_status()
        with self.engine.begin() as connection:
            connection.execute(self._upsert_statement(table), records)
        self.log_pool_status()

    def _upsert_statement(self, table: Table):
        """INSERT ... ON CONFLICT/DUPLICATE KEY UPDATE for the current dialect."""
        if self.db_dialect == "mysql":
            statement = mysql_insert(table)
            return statement.on_duplicate_key_update(value=statement.inserted.value)
        if self.db_dialect == "postgresql":
            statement = postgresql_insert(table)
        else:
            statement = sqlite_insert(table)
        return statement.on_conflict_do_update(
            index_elements=["variabel", "time"], set_={"value": statement.excluded.value}
        )

    def save_daily_value(self, code: str, day: datetime.date, value: float) -> None:
        """Upsert one value at the local midnight epoch of ``day``.

        For once-a-day labels (``away``) that belong to a calendar date
        rather than a measured hour.
        """
        midnight = datetime.datetime.combine(day, datetime.time.min, tzinfo=self.tzinfo)
        self.savedata(
            pd.DataFrame([{"code": code, "time": self.epoch(midnight), "value": value}])
        )

    def save_hourly_value(self, code: str, moment: datetime.datetime, value: float) -> None:
        """Upsert one value at the epoch of ``moment``."""
        self.savedata(
            pd.DataFrame([{"code": code, "time": self.epoch(moment), "value": value}])
        )

    def get_time_border_record(
        self, code: str, latest: bool = True, table_name: str = "values"
    ) -> datetime.datetime:
        """
        Zoekt de tijd op van het laatst aanwezige record van "code"
        :param code: de code van het record
        :param latest: boolean, if true latest record else first record
        :param table_name: table name van het record
        :return: datum en tijd van het laatst aanwezige record
        """
        """
        query = ("SELECT from_unixtime(`time`) tijd, `value` "
                 "FROM `values`, `variabel` "
                 "WHERE `variabel`.`code` = '" + code +
                 "'  and `values`.`variabel` = `variabel`.`id` "
                 "ORDER BY `time` desc LIMIT 1")
        """
        # Reflect existing tables from the database
        with self.engine.connect() as connection:
            values_table = Table(table_name, self.metadata, autoload_with=connection)
            variabel_table = Table("variabel", self.metadata, autoload_with=connection)

        # Construct the query
        query = select(
            self.from_unixtime(values_table.c.time).label("tijd"),
            values_table.c.value,
        ).where(
            and_(
                variabel_table.c.code == code,
                values_table.c.variabel == variabel_table.c.id,
            )
        )

        if latest:
            query = query.order_by(values_table.c.time.desc()).limit(1)
        else:
            query = query.order_by(values_table.c.time.asc()).limit(1)

        # Execute the query and fetch the result
        with self.engine.connect() as connection:
            result = connection.execute(query)
            result = result.scalar()
            if type(result) is str:
                result = datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        return result

    def get_prognose_field(self, field: str, start, end=None, interval="1hour"):
        values_table = Table("prognoses", self.metadata, autoload_with=self.engine)
        t1 = values_table.alias("t1")
        variabel_table = Table("variabel", self.metadata, autoload_with=self.engine)
        v1 = variabel_table.alias("v1")
        # Build the SQLAlchemy query
        query = select(
            t1.c.time.label("time"),
            self.from_unixtime(t1.c.time).label("tijd"),
            t1.c.value.label(field),
        ).where(
            and_(
                t1.c.variabel == v1.c.id,
                v1.c.code == field,
                t1.c.time
                >= start,
            )
        )
        if end is not None:
            query = query.where(
                t1.c.time
                < end  # self.unix_timestamp(end.strftime("%Y-%m-%d %H:%M:%S"))
            )
        else:
            start_dt = datetime.datetime.fromtimestamp(start)
            if start_dt.hour < 13:
                num_days = 1
            else:
                num_days = 2
            end_dt = start_dt + datetime.timedelta(days=num_days)
            end_dt = datetime.datetime(end_dt.year, end_dt.month, end_dt.day)
            query = query.where(t1.c.time < self.epoch(end_dt))

        query = query.order_by(t1.c.time)

        # Execute the query and fetch the result into a pandas DataFrame
        with self.engine.connect() as connection:
            result = connection.execute(query)

        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        df["tijd"] = pd.to_datetime(df["tijd"])
        return df

    def get_prognose_data(self, start, end=None, interval="1hour"):
        values_table = Table("prognoses", self.metadata, autoload_with=self.engine)
        variabel_table = Table("variabel", self.metadata, autoload_with=self.engine)
        if interval == "1hour":
            # Aliases for the values table
            t1 = values_table.alias("t1")
            t0 = values_table.alias("t0")

            # Aliases for the variabel table
            v1 = variabel_table.alias("v1")
            v0 = variabel_table.alias("v0")

            # Build the SQLAlchemy query
            query = select(
                t1.c.time.label("time"),
                self.from_unixtime(t1.c.time).label("tijd"),
                t0.c.value.label("temp"),
                t1.c.value.label("glob_rad"),
            ).where(
                and_(
                    t1.c.time == t0.c.time,
                    t1.c.variabel == v1.c.id,
                    v1.c.code == "gr",
                    t0.c.variabel == v0.c.id,
                    v0.c.code == "temp",
                    t1.c.time
                    >= start,
                )
            )
            if end is not None:
                query = query.where(
                    t1.c.time
                    < end  # self.unix_timestamp(end.strftime("%Y-%m-%d %H:%M:%S"))
                )
            else:
                start_dt = datetime.datetime.fromtimestamp(start)
                if start_dt.hour < 13:
                    num_days = 1
                else:
                    num_days = 2
                end_dt = start_dt + datetime.timedelta(days=num_days)
                end_dt = datetime.datetime(end_dt.year, end_dt.month, end_dt.day)
                query = query.where(t1.c.time < self.epoch(end_dt))

            query = query.order_by(t1.c.time)

            # Execute the query and fetch the result into a pandas DataFrame
            with self.engine.connect() as connection:
                result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df["tijd"] = pd.to_datetime(df["tijd"])
            return df
        else:  # interval == "15min"
            # Every hourly field is interpolated to quarters separately and the
            # results are joined on the epoch. interpolate() derives the quarter
            # epochs from the hourly "time" column, so no local-time to epoch
            # conversion is needed here (that conversion was wrong by the UTC
            # offset and broke outright under pandas 3 unit inference).
            fields = [("temp", "temp"), ("gr", "glob_rad")]
            columns = ["time", "tijd", "temp", "glob_rad"]
            result_df = None
            for field, new_field in fields:
                fld_df = self.get_prognose_field(field, start, end, interval)
                if fld_df is None or len(fld_df) < 2:
                    logging.warning(
                        f"Te weinig uurwaarden voor '{field}' om kwartierwaarden "
                        f"te maken ({0 if fld_df is None else len(fld_df)})"
                    )
                    return pd.DataFrame(columns=columns)
                fld_df = interpolate(fld_df, field, False).reset_index(drop=True)
                fld_df = fld_df.rename(columns={field: new_field})
                if result_df is None:
                    result_df = fld_df[["time", "tijd", new_field]]
                else:
                    result_df = result_df.merge(
                        fld_df[["time", new_field]], on="time", how="inner"
                    )
            result_df["time"] = result_df["time"].astype("int64")
            return result_df[columns]

    def forecast_rows(
        self, codes, lead_buckets, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """Archived forecasts for ``codes``/``lead_buckets`` in ``[start_ts, end_ts)``.

        Columns: ``target_time``, ``code``, ``lead_bucket``, ``value``,
        ``source``. Used to train the PV ML model on what a forecast at a
        given lead time actually looked like, rather than on measurements
        it will never be fed again at prediction time.
        """
        from sqlalchemy import Table, and_, select
        from sqlalchemy.exc import NoSuchTableError

        columns = ["target_time", "code", "lead_bucket", "value", "source"]
        try:
            forecasts = Table("forecasts", self.metadata, autoload_with=self.engine)
            variabel = Table("variabel", self.metadata, autoload_with=self.engine)
        except NoSuchTableError:
            return pd.DataFrame(columns=columns)

        query = select(
            forecasts.c.target_time,
            variabel.c.code,
            forecasts.c.lead_bucket,
            forecasts.c.value,
            forecasts.c.source,
        ).where(
            and_(
                forecasts.c.variabel == variabel.c.id,
                variabel.c.code.in_(list(codes)),
                forecasts.c.lead_bucket.in_(list(lead_buckets)),
                forecasts.c.target_time >= int(start_ts),
                forecasts.c.target_time < int(end_ts),
            )
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).fetchall()
        return pd.DataFrame(rows, columns=columns)

    def get_prognose_fields(
        self, codes, start, end=None, interval: str = "1hour"
    ) -> pd.DataFrame:
        """One column per code, outer-merged on "time".

        Unlike :meth:`get_prognose_data`'s fixed temp/gr pair (an implicit
        inner join: an hour missing either field drops out entirely), this
        takes an arbitrary set of codes and keeps every hour any of them
        has, leaving ``NaN`` for a code that has nothing to say about it --
        dni/dhi from Meteoserver, say, which never reports them.
        """
        codes = list(codes)
        merged = None
        for code in codes:
            field_df = self.get_prognose_field(code, start, end, interval)
            if interval != "1hour":
                if field_df is None or len(field_df) < 2:
                    continue
                field_df = interpolate(field_df, code, False).reset_index(drop=True)
            if field_df is None or len(field_df) == 0:
                continue
            piece = field_df[["time", code]]
            merged = piece if merged is None else merged.merge(
                piece, on="time", how="outer"
            )

        if merged is None:
            return pd.DataFrame(columns=["time", "tijd", *codes])

        merged["time"] = merged["time"].astype("int64")
        merged = merged.sort_values("time").reset_index(drop=True)
        merged["tijd"] = (
            pd.to_datetime(merged["time"], unit="s", utc=True)
            .dt.tz_convert(self.tzinfo)
            .dt.tz_localize(None)
        )
        for code in codes:
            if code not in merged.columns:
                merged[code] = float("nan")
        return merged[["time", "tijd", *codes]]

    def get_column_data(
        self,
        tablename: str,
        column_name: str,
        start: datetime.datetime = None,
        end: datetime.datetime = None,
        agg_func: str | None = None,
    ):
        """
        Retourneert een dataframe
        :param tablename: de naam van de tabel "prognoses" of "values"
        :param column_name: de code van het veld
        :param start: eerste uur, als deze "None" dan vanaf vandaag
        :param end: tot het laatste uur, als deze "None: dan tot alle aanwezige data
        :return:
        """
        if start is None:
            start = datetime.datetime.now()
        # Converted here rather than handed to SQL as a formatted string:
        # the column holds epoch integers and which zone the database read
        # that string in depended on the dialect and on server settings.
        start_ts = self.epoch(start)
        end_ts = None if end is None else self.epoch(end)
        """
        #  old style sql query
        sqlQuery = (
            "SELECT `time`, `value` " \
            "FROM `variabel`, `" + tablename + "` " \
            "WHERE `variabel`.`code` = '" + column_name + "' " \
            "AND `variabel`.`id` = `" + table + "`.`variabel` " \
            "AND `time` >= UNIX_TIMESTAMP('" + start + "') "
            )
        if end:
            sqlQuery += "AND `time` < UNIX_TIMESTAMP('" + end + "') "
        sqlQuery += "ORDER BY `time`;"
        # print (sqlQuery)
        """
        variabel_table = Table("variabel", self.metadata, autoload_with=self.engine)
        values_table = Table(tablename, self.metadata, autoload_with=self.engine)
        hour_column = self.hour_start(values_table.c.time).label("uur")
        if agg_func is None:
            time_column = values_table.c.time.label("time")
            agg_column = values_table.c.value.label("value")
        elif agg_func == "avg":
            time_column = func.min(values_table.c.time).label("time")
            agg_column = func.avg(values_table.c.value).label("value")
        else:
            time_column = func.min(values_table.c.time).label("time")
            agg_column = func.sum(values_table.c.value).label("value")
        # test zonder agg
        time_column = values_table.c.time.label("time")
        agg_column = values_table.c.value.label("value")

        query = select(
            hour_column,
            time_column,
            values_table.c.time.label("utc"),
            agg_column,
        ).where(
            and_(
                variabel_table.c.code == column_name,
                values_table.c.variabel == variabel_table.c.id,
                values_table.c.time >= start_ts,
            )
        )
        """
        if agg_func is not None:
            query = query.group_by("uur", "time")
        """
        if end is not None:
            query = query.where(values_table.c.time < end_ts)
        query = query.order_by("time")

        with self.engine.connect() as connection:
            query_str = str(query.compile(connection))
            logging.debug(f"query get column data da:\n {query_str}")
            result = connection.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if agg_func is not None:
            df = df.groupby("uur").agg(
                {
                    "uur": "min",
                    "time": "min",
                    "utc": "min",
                    "value": "mean" if agg_func == "avg" else "sum",
                }
            )
        now_ts = datetime.datetime.now().timestamp()
        df["datasoort"] = np.where(df["time"] <= now_ts, "recorded", "expected")
        df["time"] = df["time"].apply(
            lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M")
        )
        return df

    def get_consumption(self, start: datetime.datetime, end=datetime.datetime.now()):
        """
        retourneert een dataframe met consumption en production in periode vanaf start tot until
        :param start: start moment
        :param end: eindmoment , default nu
        :return: dataframe
        """
        values_table = Table("values", self.metadata, autoload_with=self.engine)
        # Aliases for the values table
        t1 = values_table.alias("t1")
        t2 = values_table.alias("t2")

        variabel_table = Table("variabel", self.metadata, autoload_with=self.engine)
        # Aliases for the variabel table
        v1 = variabel_table.alias("v1")
        v2 = variabel_table.alias("v2")

        # Build the SQLAlchemy query
        query = select(
            func.sum(t1.c.value).label("consumed"),
            func.sum(t2.c.value).label("produced"),
        ).where(
            and_(
                t1.c.time == t2.c.time,
                t1.c.variabel == v1.c.id,
                v1.c.code == "cons",
                t2.c.variabel == v2.c.id,
                v2.c.code == "prod",
                t1.c.time >= self.epoch(start),
                t1.c.time < self.epoch(end),
            )
        )

        with self.engine.connect() as connection:
            result = connection.execute(query)

        data = pd.DataFrame(result.fetchall(), columns=result.keys())
        if len(data.index) == 1:
            consumption = data["consumed"][0]
            production = data["produced"][0]
        else:
            consumption = 0
            production = 0

        result = {"consumption": consumption, "production": production}
        return result

    # ------------------------------------------------------------------
    # forecast archive
    #
    # "values" holds what happened, "prognoses" holds the latest forecast for
    # each moment. Neither remembers what was predicted *when*, because
    # savedata() upserts on (variabel, time). Without that the question "how
    # far off was the forecast that actually drove this morning's plan?" cannot
    # be answered afterwards, so forecast quality cannot be improved in a
    # measured way. This table keeps one row per lead time bucket, which is
    # enough to answer it and small enough to keep.
    # ------------------------------------------------------------------

    def variabel_ids(self, codes) -> dict:
        """Map variable codes to ids in one query, cached for the session."""
        if not hasattr(self, "_variabel_cache"):
            self._variabel_cache = {}
        missing = [c for c in set(codes) if c not in self._variabel_cache]
        if missing:
            variabel_table = Table(
                "variabel", self.metadata, autoload_with=self.engine
            )
            query = select(variabel_table.c.code, variabel_table.c.id).where(
                variabel_table.c.code.in_(missing)
            )
            with self.engine.connect() as connection:
                for code, ident in connection.execute(query):
                    self._variabel_cache[code] = ident
        return {c: self._variabel_cache[c] for c in codes if c in self._variabel_cache}

    def ensure_forecasts_source_column(self) -> bool:
        """Add the "source" column to an existing "forecasts" table.

        A database whose table predates this column keeps working without
        it (``save_forecasts``'s own default is ``None``), but reporting
        accuracy per source needs it there. Plain ``ALTER TABLE ADD COLUMN``
        for a nullable column with no default works unchanged on SQLite,
        MySQL and PostgreSQL, so no dialect branch is needed here.
        """
        if not inspect(self.engine).has_table("forecasts"):
            return False
        columns = {c["name"] for c in inspect(self.engine).get_columns("forecasts")}
        if "source" in columns:
            return True
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE forecasts ADD COLUMN source VARCHAR(16)")
                )
        except Exception as exception:  # noqa: BLE001 - best-effort migration
            logging.warning(
                f'Kolom "source" kon niet worden toegevoegd aan "forecasts" '
                f"({exception})"
            )
            return False
        logging.info('Kolom "source" toegevoegd aan tabel "forecasts".')
        return True

    def ensure_forecasts_table(self) -> bool:
        """Create the forecast archive table if it is not there yet.

        Lives here rather than only in check_db.py because this is where the
        table is used. check_db.py runs once at start-up and run.sh swallows
        its failure with a single log line, so anything going wrong earlier
        in that script left the table uncreated and every optimiser run
        warning "Prognose-archief niet bijgewerkt: forecasts" -- a
        NoSuchTableError whose str() is just the table name, which says
        nothing about what to do. :meth:`save_forecasts` now calls this and
        recovers on its own.

        Returns whether the table exists afterwards.
        """
        if inspect(self.engine).has_table("forecasts"):
            return True
        try:
            # Reflect the real "variabel" first, and drop any local
            # declaration of it: :func:`forecasts_table` takes the type of
            # its foreign key straight from "variabel.id", so that column
            # has to be the one the database actually has.
            if "variabel" in self.metadata.tables:
                self.metadata.remove(self.metadata.tables["variabel"])
            Table("variabel", self.metadata, autoload_with=self.engine)
            forecasts = forecasts_table(self.metadata)
            forecasts.create(self.engine, checkfirst=True)
            Index("ix_forecasts_target", forecasts.c.target_time).create(
                bind=self.engine, checkfirst=True
            )
        except Exception as exception:  # noqa: BLE001 - the archive is optional
            logging.warning(
                f"Tabel \"forecasts\" kon niet worden aangemaakt ({exception}); "
                f"het prognose-archief wordt overgeslagen. De rest van de "
                f"berekening is hierdoor niet be\u00efnvloed."
            )
            return False
        logging.info('Tabel "forecasts" aangemaakt voor het prognose-archief.')
        return True

    def save_forecasts(
        self,
        rows,
        issued_ts: int,
        tablename: str = "forecasts",
        codes_filter=ARCHIVED_FORECAST_CODES,
        source: str | None = None,
    ):
        """Archive forecast values with the lead time at which they were made.

        ``rows`` is an iterable of ``(target_time, code, value)``. Rows whose
        target already lies in the past are dropped: a "forecast" for a moment
        that has been and gone carries no information about forecast skill.
        Codes outside ``codes_filter`` are ignored, which is what keeps the
        table small; pass ``None`` to archive everything. ``source`` names
        which provider the values came from (``"meteoserver"``,
        ``"openmeteo"``, ...), written the same for every row of this call.

        Written as two executemany statements inside one transaction, rather
        than the row-at-a-time select-then-update that :meth:`savedata` uses,
        because this runs on every optimizer pass.
        """
        prepared = []
        codes = set()
        for target_time, code, value in rows:
            if codes_filter is not None and code not in codes_filter:
                continue
            try:
                target_time = int(target_time)
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value != value:  # NaN
                continue
            lead_h = (target_time - issued_ts) / 3600.0
            if lead_h < 0:
                continue
            prepared.append((target_time, code, value, lead_bucket(lead_h)))
            codes.add(code)
        if not prepared:
            return 0

        ids = self.variabel_ids(codes)
        unknown = codes - set(ids)
        if unknown:
            logging.debug(f"Prognose-archief: onbekende codes overgeslagen: {unknown}")

        records = [
            {
                "variabel": ids[code],
                "target_time": target_time,
                "lead_bucket": bucket,
                "issued_time": int(issued_ts),
                "value": value,
                "source": source,
            }
            for target_time, code, value, bucket in prepared
            if code in ids
        ]
        if not records:
            return 0

        try:
            table = Table(tablename, self.metadata, autoload_with=self.engine)
        except NoSuchTableError:
            # check_db.py creates this at start-up, but run.sh swallows its
            # failure, so a database that never got the table would warn on
            # every single optimiser run with nothing but the table name to
            # go on. Create it here and carry on; give up quietly (one clear
            # line from ensure_forecasts_table) if that is not possible
            # either, because the archive is a diagnostic, not part of the
            # plan.
            if not (
                tablename == "forecasts" and self.ensure_forecasts_table()
            ):
                return 0
            table = Table(tablename, self.metadata, autoload_with=self.engine)
        remove = delete(table).where(
            and_(
                table.c.variabel == bindparam("b_variabel"),
                table.c.target_time == bindparam("b_target_time"),
                table.c.lead_bucket == bindparam("b_lead_bucket"),
            )
        )
        keys = [
            {
                "b_variabel": r["variabel"],
                "b_target_time": r["target_time"],
                "b_lead_bucket": r["lead_bucket"],
            }
            for r in records
        ]
        with self.engine.begin() as connection:
            connection.execute(remove, keys)
            connection.execute(insert(table), records)
        logging.debug(f"Prognose-archief: {len(records)} rijen weggeschreven")
        return len(records)

    def prune_forecasts(self, before_ts: int, tablename: str = "forecasts") -> int:
        """Drop archived forecasts whose target lies before *before_ts*."""
        table = Table(tablename, self.metadata, autoload_with=self.engine)
        statement = delete(table).where(table.c.target_time < int(before_ts))
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        return result.rowcount or 0
