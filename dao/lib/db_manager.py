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
import sqlalchemy_utils
import os
import logging

from sqlalchemy import bindparam, delete

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
#: plan: the net house demand, the PV production, and the two weather inputs
#: they are derived from. Archiving every optimizer output would multiply the
#: table for no analytical gain.
ARCHIVED_FORECAST_CODES = frozenset({"hload", "pv_ac", "gr", "temp"})


def lead_bucket(lead_hours: float) -> int:
    """Largest bucket that is still below or equal to *lead_hours*."""
    chosen = LEAD_BUCKETS[0]
    for bucket in LEAD_BUCKETS:
        if lead_hours >= bucket:
            chosen = bucket
        else:
            break
    return chosen


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
        self.TARGET_TIMEZONE = db_time_zone

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
    ) -> str:
        if db_dialect == "mysql":
            if db_port == 0:
                result = (
                    f"mysql+pymysql://{db_user}:{db_password}@{db_server}/{db_name}"
                )
            else:
                result = f"mysql+pymysql://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}"
        elif db_dialect == "postgresql":
            if db_port == 0:
                result = f"postgresql+psycopg2://{db_user}:{db_password}@{db_server}/{db_name}"
            else:
                result = (
                    f"postgresql+psycopg2://{db_user}:{db_password}@{db_server}:"
                    f"{db_port}/{db_name}"
                )
        else:  # sqlite3
            if db_path is None:
                db_path = "../data"
            abs_db_path = os.path.abspath(db_path)
            result = f"sqlite:////{abs_db_path}/{db_name}"
        logging.debug(f"db_url: {result}")
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
    def from_unixtime(self, column):
        if self.db_dialect == "sqlite":
            return func.datetime(column, "unixepoch", "localtime")
        elif self.db_dialect == "postgresql":
            return func.to_char(func.to_timestamp(column), "YYYY-MM-DD HH24:MI:SS")
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
            return func.to_char(func.to_timestamp(column), "YYYY-MM")
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
            return func.to_char(func.to_timestamp(column), "YYYY-MM-01")
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
            return func.to_char(func.to_timestamp(column), "YYYY-MM-DD")
        else:  # mysql/mariadb
            return func.date(func.from_unixtime(column))

    def day_start(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-%d", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(func.to_timestamp(column), "YYYY-MM-DD")
        else:  # mysql/mariadb
            return func.date(func.from_unixtime(column))

    def hour(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%H:00", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(func.to_timestamp(column), "HH24:00")
        else:  # mysql/mariadb
            return func.time_format(func.time(func.from_unixtime(column)), "%H:00")

    def hour_start(self, column) -> func:
        if self.db_dialect == "sqlite":
            return func.strftime(
                "%Y-%m-%d %H:00", func.datetime(column, "unixepoch", "localtime")
            )
        elif self.db_dialect == "postgresql":
            return func.to_char(func.to_timestamp(column), "YYYY-MM-DD HH24:00")
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
        logging.debug(f"Opslaan dataframe:\n{df.to_string()}")

        # with self.engine.connect() as connection:
        connection = self.engine.connect()
        try:
            self.log_pool_status()
            # Reflect existing tables from the database
            values_table = Table(tablename, self.metadata, autoload_with=self.engine)
            variabel_table = Table("variabel", self.metadata, autoload_with=self.engine)
            df = df.reset_index()  # make sure indexes pair with number of rows
            df["tijd"] = df["time"].apply(
                lambda x: datetime.datetime.fromtimestamp(int(float(x))).strftime(
                    "%Y-%m-%d %H:%M"
                )
            )
            for index, dfrow in df.iterrows():
                logging.debug(
                    f"Save record: {dfrow['tijd']} {dfrow['code']} "
                    f"{dfrow['time']} {dfrow['value']}"
                )
                code = dfrow["code"]
                time = dfrow["time"]
                value = dfrow["value"]
                if not isinstance(value, (int, float)):
                    continue
                if pd.isna(value):
                    continue
                if value == float("inf"):
                    continue

                # Get the variabel_id
                select_variabel = select(variabel_table.c.id).where(
                    variabel_table.c.code == code
                )
                variabel_result = connection.execute(select_variabel).first()
                if variabel_result:
                    variabel_id = variabel_result[0]
                else:
                    logging.error(f"Onbekende code opslaan data: {code}")
                    continue

                # Query to check if the record exists
                select_value = select(values_table.c.id).where(
                    (values_table.c.variabel == variabel_id)
                    & (values_table.c.time == time)
                )
                value_result = connection.execute(select_value).first()
                if value_result:
                    # Update existing record
                    value_id = value_result[0]
                    update_value = (
                        update(values_table)
                        .values(value=value)
                        .where(values_table.c.id == value_id)
                    )
                    connection.execute(update_value)
                else:
                    # Record does not exist, perform insert
                    insert_value = insert(values_table).values(
                        variabel=variabel_id, time=time, value=value
                    )
                    connection.execute(insert_value)
            connection.commit()
        finally:
            connection.close()
        self.log_pool_status()

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
                >= start,  # self.unix_timestamp(start.strftime('%Y-%m-%d %H:%M:%S'))
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
            end_ts = end_dt.timestamp()
            query = query.where(
                t1.c.time < self.unix_timestamp(end_dt.strftime("%Y-%m-%d %H:%M:%S"))
            )

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
                    >= start,  # self.unix_timestamp(start.strftime('%Y-%m-%d %H:%M:%S'))
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
                end_ts = end_dt.timestamp()
                query = query.where(
                    t1.c.time
                    < self.unix_timestamp(end_dt.strftime("%Y-%m-%d %H:%M:%S"))
                )

            query = query.order_by(t1.c.time)

            # Execute the query and fetch the result into a pandas DataFrame
            with self.engine.connect() as connection:
                result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df["tijd"] = pd.to_datetime(df["tijd"])
            return df
        else:  # interval == "15min"
            fields = [("temp", "temp"), ("gr", "glob_rad")]
            result_df = None
            for field, new_field in fields:
                fld_df = self.get_prognose_field(field, start, end, interval)
                # fld_df.index = pd.to_datetime(fld_df["tijd"])
                # fld_df = interpolate(fld_df, field, 15, (field == "gr"))
                if fld_df is not None and len(fld_df) > 0:
                    fld_df = interpolate(fld_df, field, False)
                if result_df is None:
                    result_df = fld_df
                else:
                    result_df[new_field] = fld_df[field]
            if result_df is not None:
                result_df["time"] = result_df["tijd"].astype(int) // 1e9
            return result_df

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
        start = start.strftime("%Y-%m-%d %H:%M")
        if end is not None:
            end = end.strftime("%Y-%m-%d %H:%M")
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
                values_table.c.time >= self.unix_timestamp(start),
            )
        )
        """
        if agg_func is not None:
            query = query.group_by("uur", "time")
        """
        if end is not None:
            query = query.where(values_table.c.time < self.unix_timestamp(end))
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
                t1.c.time >= self.unix_timestamp(start.strftime("%Y-%m-%d %H:%M:%S")),
                t1.c.time < self.unix_timestamp(end.strftime("%Y-%m-%d %H:%M:%S")),
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

    def save_forecasts(
        self,
        rows,
        issued_ts: int,
        tablename: str = "forecasts",
        codes_filter=ARCHIVED_FORECAST_CODES,
    ):
        """Archive forecast values with the lead time at which they were made.

        ``rows`` is an iterable of ``(target_time, code, value)``. Rows whose
        target already lies in the past are dropped: a "forecast" for a moment
        that has been and gone carries no information about forecast skill.
        Codes outside ``codes_filter`` are ignored, which is what keeps the
        table small; pass ``None`` to archive everything.

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
            }
            for target_time, code, value, bucket in prepared
            if code in ids
        ]
        if not records:
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

    def _accuracy_query(
        self, code: str, realised_table: str, realised_code: str, start_ts, end_ts
    ):
        """Join the archive to the realised series. Shared by the two reports."""
        forecasts = Table("forecasts", self.metadata, autoload_with=self.engine)
        realised = Table(realised_table, self.metadata, autoload_with=self.engine)
        var_f = Table("variabel", self.metadata, autoload_with=self.engine).alias("vf")
        var_r = Table("variabel", self.metadata, autoload_with=self.engine).alias("vr")
        joined = (
            forecasts.join(var_f, var_f.c.id == forecasts.c.variabel)
            .join(realised, realised.c.time == forecasts.c.target_time)
            .join(var_r, var_r.c.id == realised.c.variabel)
        )
        condition = and_(
            var_f.c.code == code,
            var_r.c.code == realised_code,
            forecasts.c.target_time >= int(start_ts),
            forecasts.c.target_time < int(end_ts),
        )
        error = forecasts.c.value - realised.c.value
        return joined, condition, error, forecasts, realised

    def forecast_accuracy(
        self, code: str, realised_table: str, realised_code: str, start_ts, end_ts
    ) -> list:
        """Error statistics per lead time bucket, aggregated by the database.

        Returns at most five rows, so nothing large ever reaches Python. That
        matters on low powered hardware where the alternative -- pulling the
        whole join into pandas -- would be the heaviest thing DAO does all day.
        """
        joined, condition, error, forecasts, realised = self._accuracy_query(
            code, realised_table, realised_code, start_ts, end_ts
        )
        query = (
            select(
                forecasts.c.lead_bucket.label("lead_bucket"),
                func.count().label("n"),
                func.avg(func.abs(error)).label("mae"),
                func.avg(error).label("bias"),
                func.avg(error * error).label("mse"),
                func.avg(func.abs(realised.c.value)).label("scale"),
            )
            .select_from(joined)
            .where(condition)
            .group_by(forecasts.c.lead_bucket)
            .order_by(forecasts.c.lead_bucket)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return [dict(row) for row in rows]

    def forecast_bias_by_hour(
        self,
        code: str,
        realised_table: str,
        realised_code: str,
        start_ts,
        end_ts,
        bucket: int | None = None,
    ) -> list:
        """Mean error per hour of the local day. This is the actionable one.

        A forecast that is consistently too low between 17:00 and 20:00 makes
        the optimizer reserve too little energy for the evening peak, and no
        amount of realtime correction can repair that afterwards.
        """
        joined, condition, error, forecasts, realised = self._accuracy_query(
            code, realised_table, realised_code, start_ts, end_ts
        )
        if bucket is not None:
            condition = and_(condition, forecasts.c.lead_bucket == bucket)
        hour = self.hour(forecasts.c.target_time)
        query = (
            select(
                hour.label("uur"),
                func.count().label("n"),
                func.avg(error).label("bias"),
                func.avg(func.abs(error)).label("mae"),
                func.avg(realised.c.value).label("realised"),
            )
            .select_from(joined)
            .where(condition)
            .group_by(hour)
            .order_by(hour)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return [dict(row) for row in rows]
