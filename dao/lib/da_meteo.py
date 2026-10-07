import datetime
import logging
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from dao.lib.da_graph import GraphBuilder
from dao.lib.db_manager import DBmanagerObj
from sqlalchemy import Table, select, func, and_


# noinspection PyUnresolvedReferences
class Meteo:
    def __init__(
        self,
        config,
        db_da: DBmanagerObj,
        latitude: float,
        longitude: float,
        secrets: dict = None,
        country: str = "NL",
        time_zone: str = "Europe/Amsterdam",
    ):
        self.config = config
        self.db_da = db_da
        self.time_zone = time_zone
        self.secrets = secrets or {}
        mk = config.meteoserver_key
        self.meteoserver_key = mk.resolve(self.secrets) if mk is not None else None
        self.meteoserver_model = config.meteoserver_model
        self.meteoserver_attempts = config.meteoserver_attempts
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.solar = config.solar
        self.bat = config.battery
        self.graphics_style = config.graphics.style
        # Used to evaluate the sun's position at the middle of the interval
        # a radiation value stands for, whatever that interval's length is.
        self.interval_s = 3600 if config.interval == "1hour" else 900

    def make_graph_meteo(self, df, file=None, show=False):
        # The weather frame carries an epoch and no local-time column; both
        # the hour axis and the title have to come from it in the
        # configured zone. Reading the title from column 2 by position used
        # to land on "dni", which Meteoserver never supplies, so the title
        # said "vanaf nan"; reading the hour in UTC put the whole axis one
        # or two hours out.
        if "tijd_nl" in df.columns:
            df["uur"] = df.tijd_nl.apply(lambda x: x[11:13])
            first_moment = df["tijd_nl"].iloc[0]
        else:
            local = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(
                self.time_zone
            )
            df["uur"] = local.dt.strftime("%H")
            first_moment = local.iloc[0].strftime("%Y-%m-%d %H:%M")
        meteo_options = {
            "title": f"Opgehaalde meteodata vanaf {first_moment}",
            "style": self.graphics_style,
            "graphs": [
                {
                    "vaxis": [{"title": "J/cm2"}, {"title": "°C"}],
                    "align_zeros": "True",
                    "series": [
                        {
                            "column": "gr",
                            "name": "Globale straling",
                            "type": "stacked",
                            "color": "blue",
                            "width": 0.8,
                        },
                        {
                            "column": "temp",
                            "name": "Temperatuur",
                            "type": "line",
                            "color": "green",
                            "vaxis": "right",
                        },
                    ],
                }
            ],
            "haxis": {"values": "uur", "title": "uur"},
        }

        gb = GraphBuilder()
        plot = gb.build(df, meteo_options, show=show)
        if file is not None:
            plot.savefig(file)
        """
        plt.figure(figsize=(15, 10))
        df["gr"] = pd.to_numeric(df["gr"])
        x_axis = np.arange(len(df["tijd_nl"].values))
        plt.bar(x_axis - 0.1, df["gr"].values, width=0.7, label="global rad")
        # plt.bar(x_axis + 0.1, df["solar_rad"].values, width=0.2, label="netto rad")
        plt.xticks(x_axis + 0.1, df["tijd_nl"].values, rotation=45)
        if file is not None:
            plt.savefig(file)
        if show:
            plt.show()
        plt.close("all")
        return
        """

    def get_meteo_data(self, show_graph=False):
        from pathlib import Path

        from dao.forecast.weather.service import WeatherService

        service = WeatherService(
            self.config,
            self.db_da,
            self.latitude,
            self.longitude,
            self.secrets,
            Path("../data/forecast/weather"),
            self.country,
        )
        status = service.update()
        df = status.frame
        if len(df) > 0:
            style = self.graphics_style
            plt.style.use(style)
            self.make_graph_meteo(
                df,
                file="../data/images/meteo_"
                + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
                + ".png",
                show=show_graph,
            )
        return status

    def get_avg_temperature(self, date: datetime.datetime = None) -> Optional[float]:
        """
        Berekent gewogen met temperatuur grens van 16 oC
        :param date: de datum waarvoor de berekening wordt gevraagd
        als None: vandaag
        :return: berekende gewogen graaddagen, of None als er geen
            temperatuurprognoses voor die dag beschikbaar zijn
        """
        if date is None:
            date = datetime.datetime.combine(
                datetime.datetime.today(), datetime.datetime.min.time()
            )
        date_utc = int(date.timestamp())

        # Reflect existing tables from the database
        values_table = Table(
            "prognoses", self.db_da.metadata, autoload_with=self.db_da.engine
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
                    variabel_table.c.code == "temp",
                    values_table.c.variabel == variabel_table.c.id,
                    values_table.c.time >= date_utc,
                    # Without this, a day with no data at all would silently
                    # average whatever forecast exists for a *later* day.
                    values_table.c.time < date_utc + 86400,
                )
            )
            .order_by(values_table.c.time.asc())
            .limit(24)
            .alias("t1")
        )

        # Construct the outer query
        outer_query = select(func.avg(inner_query.c.value).label("avg_temp"))

        # Execute the query and fetch the result
        with self.db_da.engine.connect() as connection:
            result = connection.execute(outer_query)
            avg_temp = result.scalar()
        if avg_temp is None:
            logging.warning(
                f"Geen temperatuurprognose beschikbaar voor {date:%Y-%m-%d}"
            )
            return None
        """
        sql_avg_temp = (
            "SELECT AVG(t1.`value`) avg_temp FROM "
            "(SELECT `time`, `value`,  from_unixtime(`time`) 'begin' "
            "FROM `values` , `variabel` "
            "WHERE `variabel`.`code` = 'temp' 
                AND `values`.`variabel` = `variabel`.`id` 
                AND time >= " + str(date_utc) + " "
            "ORDER BY `time` ASC LIMIT 24) t1 "
        )
        data = self.db_da.run_select_query(sql_avg_temp)
        avg_temp = float(data['avg_temp'].values[0])
        """
        return avg_temp

    def calc_graaddagen(
        self,
        date: datetime.datetime = None,
        avg_temp: float | None = None,
        weighted: bool = False,
    ) -> float:
        """
        Berekent graaddagen met temperatuur grens van 16 oC
        :param date: de datum waarvoor de berekening wordt gevraagd
                    als None: vandaag
        :param avg_temp: de gemiddelde temperatuur, default None
        :param weighted: boolean, gewogen als true, default false
        :return: berekende eventueel gewogen graaddagen
        """
        if date is None:
            date = datetime.datetime.combine(
                datetime.datetime.today(), datetime.datetime.min.time()
            )
        if avg_temp is None:
            avg_temp = self.get_avg_temperature(date)
        if avg_temp is None:
            # No temperature forecast for this day at all: 0 degree days is
            # the safe fallback (day_ahead.py already treats heat_needed<=0
            # as "skip the heat pump this run" rather than crashing on it).
            logging.warning(
                f"Graaddagen voor {date:%Y-%m-%d} niet te berekenen (geen "
                f"temperatuurprognose); 0 graaddagen aangenomen"
            )
            return 0.0
        weight_factor = 1
        if weighted:
            mon = date.month
            if mon <= 2 or mon >= 11:
                weight_factor = 1.1
            elif 4 <= mon <= 9:
                weight_factor = 0.8
        if avg_temp >= 16:
            result = 0
        else:
            result = weight_factor * (16 - avg_temp)
        return result
