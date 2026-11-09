import base64
import calendar
import datetime
import json
import logging
import math
from decimal import Decimal, localcontext
from io import BytesIO
from itertools import accumulate
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

# from unittest.mock import inplace
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.core.dtypes.inference import is_number
from sklearn.metrics import r2_score
from sqlalchemy import (
    BigInteger,
    String,
    Table,
    and_,
    case,
    column,
    func,
    literal,
    select,
    union,
    values,
)

from dao.lib.da_graph import GraphBuilder
from dao.prog import production_bonus
from dao.prog.da_base import DaBase
from dao.prog.utils import get_value_from_dict

AGGREGATE_DELTAS = {
    "15min": datetime.timedelta(minutes=15),
    "hour": datetime.timedelta(hours=1),
    "day": datetime.timedelta(days=1),
    "week": datetime.timedelta(days=7),
    "month": datetime.timedelta(days=31),
}


def legacy_prognosis_revenue(objective_profit):
    """prognoses.profit bewaart de negatieve objectivebijdrage, niet ontvangst.

    Alleen deze benoemde adapter keert het teken om; negatieve exportprijzen
    blijven negatieve opbrengsten. Historische values.profit is al ontvangst.
    """
    return -objective_profit


def _amount_sum(values):
    values = tuple(values)
    if not values or any(value is None or pd.isna(value) for value in values):
        return None
    with localcontext() as context:
        context.prec = 36
        # Alleen aan deze rapportgrens komen historische legacyfloats binnen.
        return sum((value if isinstance(value, Decimal) else Decimal(str(value)) for value in values), Decimal("0"))


def calc_r2(serie_x: pd.Series, serie_y: pd.Series) -> float:
    if serie_x.count() < 24 or serie_y.count() < 24:
        return pd.NA
    return r2_score(serie_x, serie_y)


def localize_hour_buckets(timestamps, time_zone):
    """Attach ``time_zone`` to naive hour buckets without raising on a DST day.

    Accepts a ``DatetimeIndex`` or a datetime ``Series`` and returns the
    same shape, localized.

    Every report in this module builds one row per hour *label* -- the API
    data from an SQL GROUP BY on the hour string, the solar report from a
    loop adding an hour at a time to local midnight. Neither produces the
    repeated 02:00 that ``ambiguous="infer"`` needs to work out which
    occurrence is meant, so a fixed policy is the only one that is
    deterministic here: the bucket is standard (winter) time.
    ``nonexistent="shift_forward"`` covers the spring day, whose
    02:00-03:00 bucket does not exist at all.

    Without both, any report covering one of those two days a year raised
    outright.
    """
    localize = getattr(timestamps, "dt", timestamps).tz_localize
    return localize(time_zone, ambiguous=False, nonexistent="shift_forward")


class Report(DaBase):
    @classmethod
    def settlement_reader(cls, file_name):
        report = cls.local_settlement_app(file_name)
        if report.time_zone is None:
            raise ValueError("Settlementrapport vereist een expliciete contracttijdzone")
        report.prices_options = report.config.prices
        report.periodes = {}
        report.make_periodes(_now=datetime.datetime.now(ZoneInfo(report.time_zone)).replace(tzinfo=None))
        return report

    def __init__(
            self, file_name: str = "../data/options.json", _now: datetime.datetime = None
    ):
        super().__init__(file_name=file_name)
        if self.config is None:
            return

        self.report_options = self.config.report
        if self.report_options is None:
            logging.error("Er zijn geen report-instellingen gevonden")
        # Per-instance, not a class attribute: it used to be declared on the
        # class body, so every Report instance (a new one per web request)
        # mutated the same shared dict via self.periodes.update(...) below.
        self.periodes = {}
        self.make_periodes(_now=_now)
        _r = self.report_options
        self.grid_consumption_sensors = _r.entities_grid_consumption if _r else []
        self.grid_production_sensors = _r.entities_grid_production if _r else []
        self.battery_production_sensors = _r.entities_battery_production if _r else []
        self.battery_consumption_sensors = _r.entities_battery_consumption if _r else []
        self.solar_production_ac_sensors = _r.entities_solar_production_ac if _r else []
        # co2_intensity_sensor is a single entity id (Optional[EntityId] on the
        # model), but every "sensors" dict elsewhere in this class is iterated
        # as a list of entity ids. A bare string was iterated character by
        # character, so every CO2 lookup silently found nothing.
        self.co2_intensity_sensor = (
            [_r.co2_intensity_sensor] if _r and _r.co2_intensity_sensor else []
        )
        self.ev_consumption_sensors = _r.entities_ev_consumption if _r else []
        self.wp_consumption_sensors = _r.entities_wp_consumption if _r else []
        self.boiler_consumption_sensors = _r.entities_boiler_consumption if _r else []
        self.machine_consumption_sensors = _r.entities_machine_consumption if _r else []

        self.saving_consumption_dict = {
            "calc_interval": "uur",
            "series": {
                "cons_zonder": {
                    "dim": "kWh",
                    "sign": "pos",
                    "header": "Zonder batterij",
                    "name": "Verbruik",
                    "source": "db",
                    "sensors": self.grid_consumption_sensors,
                    "tabel": True,
                    "agg": "sum",
                    "color": "#00bfff",
                },
                "prod_zonder": {
                    "dim": "kWh",
                    "sign": "neg",
                    "header": "Zonder batterij",
                    "name": "Productie",
                    "source": "calc",
                    "sensors": self.grid_production_sensors,
                    "tabel": True,
                    "agg": "sum",
                    "color": "#0080ff",
                },
                "netto_cons_zonder": {
                    "dim": "kWh",
                    "header": "Zonder batterij",
                    "name": "Netto verbruik",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "header": "Met batterij",
                    "name": "Verbruik",
                    "source": "db",
                    "sensors": self.grid_consumption_sensors,
                    "tabel": True,
                    "agg": "sum",
                    "color": "#00bfff",
                },
                "prod": {
                    "dim": "kWh",
                    "sign": "neg",
                    "header": "Met batterij",
                    "name": "Productie",
                    "source": "db",
                    "sensors": self.grid_production_sensors,
                    "tabel": True,
                    "agg": "sum",
                    "color": "#0080ff",
                },
                "netto_cons_met": {
                    "dim": "kWh",
                    "header": "Met batterij",
                    "name": "Netto verbruik",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "saving": {
                    "dim": "kWh",
                    "header": "Besparing(+)",
                    "name": "Extra(-)",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "bat_out": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Accu_uit",
                    "source": "db",
                    "sensors": self.battery_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "red",
                },
                "bat_in": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Accu in",
                    "source": "db",
                    "sensors": self.battery_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                },
            },
        }

        self.saving_cost_dict = {
            "calc_interval": "uur",
            "series": {
                "cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Verbruik",
                    "source": "db",
                    "sensors": self.grid_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#00bfff",
                },
                "prod": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Productie",
                    "source": "db",
                    "sensors": self.grid_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#0080ff",
                },
                "bat_out": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Accu_uit",
                    "source": "db",
                    "sensors": self.battery_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "red",
                },
                "bat_in": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Accu in",
                    "source": "db",
                    "sensors": self.battery_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                },
                "da_cons": {
                    "dim": "eur/kWh",
                    "sign": "pos",
                    "name": "Tarief verbruik",
                    "source": "prices",
                    "sensors": [],
                    "tabel": False,
                    "agg": "sum",
                },
                "da_prod": {
                    "dim": "eur/kWh",
                    "sign": "pos",
                    "name": "Tarief productie",
                    "source": "prices",
                    "sensors": [],
                    "tabel": False,
                    "agg": "sum",
                },
                "cost_zonder": {
                    "dim": "eur",
                    "header": "Zonder batterij",
                    "name": "Kosten",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "profit_zonder": {
                    "dim": "eur",
                    "header": "Zonder batterij",
                    "name": "Opbrengst",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "netto_cost_zonder": {
                    "dim": "eur",
                    "header": "Zonder batterij",
                    "name": "Netto kosten",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "cost": {
                    "dim": "eur",
                    "header": "Met batterij",
                    "name": "Kosten",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "profit": {
                    "dim": "eur",
                    "header": "Met batterij",
                    "name": "Opbrengst",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "netto_cost": {
                    "dim": "eur",
                    "header": "Met batterij",
                    "name": "Netto kosten",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
                "saving": {
                    "dim": "eur",
                    "header": "Besparing(+)",
                    "name": "Extra(-)",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
            },
        }

        self.saving_co2_dict = {
            "calc_interval": "uur",
            "series": {
                "netto_cons_zonder": {
                    "dim": "kWh",
                    "sign": "pos",
                    "header": "Zonder batterij",
                    "name": "Netto verbr.",
                    "source": "calc",
                    "sensors": "calc",
                    "agg": "sum",
                },
                "emissie_zonder": {
                    "dim": "kg CO2",
                    "header": "Zonder batterij",
                    "sign": "pos",
                    "type": "bar",
                    "width": 0.7,
                    "name": "CO2 emissie",
                    "source": "calc",
                    "agg": "sum",
                },
                "cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Verbruik",
                    "source": "db",
                    "sensors": self.grid_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#00bfff",
                },
                "prod": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Productie",
                    "source": "db",
                    "sensors": self.grid_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#0080ff",
                },
                "netto_cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "header": "Met batterij",
                    "name": "Netto verbr.",
                    "source": "calc",
                    "sensors": "calc",
                    "agg": "sum",
                },
                "bat_out": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Accu_uit",
                    "source": "db",
                    "sensors": self.battery_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "red",
                },
                "bat_in": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Accu in",
                    "source": "db",
                    "sensors": self.battery_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                },
                "co2_intensity": {
                    "dim": "g CO2 eq/kWh",
                    "header": "",
                    "name": "CO2 Intensity",
                    "source": "db",
                    "type": "step",
                    "sensor_type": "factor",
                    "sensors": self.co2_intensity_sensor,
                    "agg": "mean",
                },
                "emissie_met": {
                    "dim": "kg CO2",
                    "header": "Met batterij",
                    "sign": "pos",
                    "type": "bar",
                    "width": 0.7,
                    "name": "CO2 emissie",
                    "source": "calc",
                    "agg": "sum",
                },
                "saving": {
                    "dim": "kg CO2",
                    "header": "Besparing(+)",
                    "name": "Extra(-)",
                    "source": "calc",
                    "color": "#ff8000",
                    "tabel": True,
                    "agg": "sum",
                },
            },
        }

        self.calc_co2_emission_dict = {
            "calc_interval": "uur",
            "with header": False,
            "series": {
                "cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "name": "Verbruik",
                    "source": "db",
                    "sensors": self.grid_consumption_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#00bfff",
                },
                "prod": {
                    "dim": "kWh",
                    "sign": "neg",
                    "name": "Productie",
                    "source": "db",
                    "sensors": self.grid_production_sensors,
                    "tabel": False,
                    "agg": "sum",
                    "color": "#0080ff",
                },
                "netto_cons": {
                    "dim": "kWh",
                    "sign": "pos",
                    "header": "Met batterij",
                    "name": "Netto verbr.",
                    "source": "calc",
                    "sensors": "calc",
                    "agg": "sum",
                },
                "co2_intensity": {
                    "dim": "g CO2 eq/kWh",
                    "header": "",
                    "name": "CO2 Intensity",
                    "source": "db",
                    "type": "step",
                    "sensor_type": "factor",
                    "sensors": self.co2_intensity_sensor,
                    "agg": "mean",
                },
                "emissie": {
                    "dim": "kg CO2",
                    "sign": "pos",
                    "type": "bar",
                    "name": "CO2 emissie",
                    "source": "calc",
                    "agg": "sum",
                },
            },
        }

        self.energy_balance_dict = {
            "cons": {
                "dim": "kWh",
                "sign": "pos",
                "name": "Verbruik",
                "sensors": self.grid_consumption_sensors,
                "color": "#00bfff",
            },
            "prod": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Productie",
                "sensors": self.grid_production_sensors,
                "color": "#0080ff",
            },
            "bat_out": {
                "dim": "kWh",
                "sign": "pos",
                "name": "Accu_uit",
                "sensors": self.battery_production_sensors,
                "color": "red",
            },
            "bat_in": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Accu in",
                "sensors": self.battery_consumption_sensors,
                "color": "#ff8000",
            },
            "pv_ac": {
                "dim": "kWh",
                "sign": "pos",
                "name": "PV ac",
                "sensors": self.solar_production_ac_sensors,
                "color": "green",
            },
            "ev": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Elec. vehicle",
                "sensors": self.ev_consumption_sensors,
                "color": "yellow",
            },
            "wp": {
                "dim": "kWh",
                "sign": "neg",
                "name": "WP",
                "sensors": self.wp_consumption_sensors,
                "color": "#a32cc4",
            },
            "boil": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Boiler",
                "sensors": self.boiler_consumption_sensors,
                "color": "#e39ff6",
            },
            "mach": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Machines",
                "sensors": self.machine_consumption_sensors,
                "color": "brown",
            },
            "base": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Baseload",
                "sensors": "calc",
                "function": "calc_base",
                "color": "#f1a603",
            },
        }

        self.grid_dict = {
            "cons": {
                "dim": "kWh",
                "sign": "pos",
                "name": "Verbruik",
                "sensors": self.grid_consumption_sensors,
            },
            "prod": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Productie",
                "sensors": self.grid_production_sensors,
            },
            "cost": {
                "dim": "eur",
                "sign": "neg",
                "name": "Kosten",
                "sensors": "calc",
                "function": "calc_cost",
            },
            "profit": {
                "dim": "eur",
                "sign": "pos",
                "name": "Opbrengst",
                "sensors": "calc",
                "function": "calc_cost",
            },
        }

        self.several_dict = {
            "pv_dc": {
                "dim": "kWh",
                "sign": "pos",
                "name": "PV DC",
                "sensors": (_r.entities_solar_production_dc if _r else []),
                "color": "#e39ff6",
            },
        }
        self.co2_dict = {
            "cons": {
                "dim": "kWh",
                "sign": "pos",
                "name": "Verbruik",
                "sensors": self.grid_consumption_sensors,
                "color": "red",
            },
            "prod": {
                "dim": "kWh",
                "sign": "neg",
                "name": "Productie",
                "sensors": self.grid_production_sensors,
                "color": "green",
            },
            "netto_cons": {
                "dim": "kWh",
                "sign": "pos",
                "name": "Netto verbr.",
                "sensors": "calc",
                "function": "calc_netto_cons",
            },
            "co2_intensity": {
                "dim": "g CO2 eq/kWh",
                "name": "CO2 Intensity",
                "type": "step",
                "sensor_type": "factor",
                "sensors": self.co2_intensity_sensor,
                "color": "olive",
            },
            "emissie": {
                "dim": "kg CO2",
                "sign": "pos",
                "type": "bar",
                "width": 0.7,
                "name": "CO2 emissie",
                "sensors": "calc",
                "function": "calc_emissie",
                "vaxis": "right",
                "color": "blue",
            },
        }

        _graphics_style = self.config.graphics.style
        self.balance_graph_options = {
            "title": "Energiebalans",
            "style": _graphics_style,
            "haxis": {"values": "#interval"},
            "graphs": [
                {
                    "vaxis": [{"title": "kWh"}],
                    "series_keys": [
                        "base",
                        "wp",
                        "boil",
                        "ev",
                        "bat_in",
                        "prod",
                        "pv_ac",
                        "bat_out",
                        "cons",
                    ],
                    "series": [],
                }
            ],
        }
        for key in self.balance_graph_options["graphs"][0]["series_keys"]:
            # key, serie in self.energy_balance_dict.items():
            serie = self.energy_balance_dict[key]
            serie["column"] = serie["name"]
            serie["type"] = "stacked"
            serie["title"] = serie["name"]
            self.balance_graph_options["graphs"][0]["series"].append(serie)

        self.co2_graph_options = {
            "title": "CO2 Emissie",
            "style": _graphics_style,
            "haxis": {"values": "#interval"},
            "graphs": [
                {
                    "vaxis": [{"title": "kWh"}],
                    "series_keys": [
                        "cons",
                        "prod",
                    ],
                    "series": [],
                },
                {
                    "vaxis": [{"title": "g CO2/kWh"}, {"title": "kg CO2"}],
                    "series_keys": [
                        "co2_intensity",
                        "emissie",
                    ],
                    "series": [],
                },
            ],
        }

        for graph_num in range(len(self.co2_graph_options["graphs"])):
            for key in self.co2_graph_options["graphs"][graph_num]["series_keys"]:
                serie = self.co2_dict[key]
                serie["column"] = key
                if "type" not in serie:
                    serie["type"] = "stacked"
                serie["title"] = serie["name"]
                self.co2_graph_options["graphs"][graph_num]["series"].append(serie)

        self.saving_cons_graph_options = {
            "title": "Besparing verbruik door batterij",
            "style": _graphics_style,
            "graphs": [
                {
                    "graph_type": "waterfall",
                    "vaxis": [{"title": "kWh"}],
                    "align_zeros": "True",
                    "series": [
                        {"column": "saving", "title": "Besparing", "name": "Besparing"}
                    ],
                }
            ],
        }

        self.saving_cost_graph_options = {
            "title": "Besparing kosten door batterij",
            "style": _graphics_style,
            "graphs": [
                {
                    "graph_type": "waterfall",
                    "vaxis": [{"title": "eur"}],
                    "align_zeros": "True",
                    "series": [
                        {"column": "saving", "title": "Besparing", "name": "Besparing"}
                    ],
                }
            ],
        }

        self.saving_co2_graph_options = {
            "title": "Besparing CO2-emissie door batterij",
            "style": _graphics_style,
            "graphs": [
                {
                    "graph_type": "waterfall",
                    "vaxis": [{"title": "kg CO2"}],
                    "align_zeros": "True",
                    "series": [
                        {"column": "saving", "title": "Besparing", "name": "Besparing"}
                    ],
                }
            ],
        }

        style = _graphics_style
        self.solar_graph_options = {
            "title": "Solar production",
            "style": style,
            "haxis": {"values": "uur", "title": "uren van de dag"},
            "graphs": [
                {
                    "vaxis": [{"title": "J/cm2"}, {"title": "kWh"}],
                    "series": [
                        {
                            "column": "prognose_straling",
                            "title": "Straling prognose",
                            "type": "line",
                            "color": "purple",
                        },
                        {
                            "column": "gemeten_straling",
                            "title": "Straling gemeten",
                            "source": "da",
                            "type": "line",
                            "color": "red",
                        },
                        {
                            "column": "gemeten_prod",
                            "title": "Productie gemeten",
                            "vaxis": "right",
                            "type": "bar",
                            "color": "blue",
                        },
                        {
                            "column": "prognose_dao",
                            "vaxis": "right",
                            "type": "bar",
                            "title": "Prod. progn. DAO",
                            "color": "yellow",
                        },
                        {
                            "column": "prognose_ml",
                            "vaxis": "right",
                            "type": "bar",
                            "title": "Prod. progn. ml",
                            "color": "green",
                        },
                    ],
                }
            ],
        }

        """
        self.saving_cost_graph_options = self.saving_cons_graph_options.copy()
        self.saving_cost_graph_options['title'] = "Besparing kosten door batterij"
        self.saving_cost_graph_options["graphs"][0]["vaxis"] = [{"title": "eur"}]
        self.saving_co2_graph_options = self.saving_cons_graph_options.copy()
        self.saving_co2_graph_options['title'] = "Besparing CO2-emissie door batterij"
        self.saving_co2_graph_options["graphs"][0]["vaxis"] = [{"title": "kg CO2"}]
        """
        return

    def make_periodes(self, _now: datetime.datetime = None):
        def create_dict(name, _vanaf, _tot, interval):
            return {name: {"vanaf": _vanaf, "tot": _tot, "interval": interval}}

        # vandaag
        if _now is None:
            now = datetime.datetime.now()
        else:
            now = _now
        vanaf = datetime.datetime(now.year, now.month, now.day)
        tot = vanaf + datetime.timedelta(days=1)
        if now.hour < 13:
            max_tot = tot
        else:
            max_tot = tot + datetime.timedelta(days=1)
        self.periodes.update(create_dict("vandaag", vanaf, tot, interval="uur"))

        # morgen
        vanaf_m = tot
        tot_m = vanaf_m + datetime.timedelta(days=1)
        self.periodes.update(create_dict("morgen", vanaf_m, tot_m, interval="uur"))

        # vandaag en morgen
        self.periodes.update(
            create_dict("vandaag en morgen", vanaf, tot_m, interval="uur")
        )

        # gisteren
        tot_g = vanaf
        vanaf_g = vanaf + datetime.timedelta(days=-1)
        self.periodes.update(create_dict("gisteren", vanaf_g, tot_g, interval="uur"))

        # deze week
        delta = vanaf.weekday()
        vanaf = vanaf + datetime.timedelta(days=-delta)
        tot = min(vanaf + datetime.timedelta(days=7), max_tot)
        self.periodes.update(create_dict("deze week", vanaf, tot, "dag"))

        # vorige week
        vanaf += datetime.timedelta(days=-7)
        tot = vanaf + datetime.timedelta(days=7)
        self.periodes.update(create_dict("vorige week", vanaf, tot, "dag"))

        # deze maand
        vanaf = datetime.datetime(now.year, now.month, 1)
        tot = min(vanaf + relativedelta(months=1), max_tot)
        self.periodes.update(create_dict("deze maand", vanaf, tot, "dag"))

        # vorige maand
        tot = vanaf
        vanaf += relativedelta(months=-1)
        self.periodes.update(create_dict("vorige maand", vanaf, tot, "dag"))

        # dit jaar
        vanaf = datetime.datetime(now.year, 1, 1)
        tot = max_tot
        self.periodes.update(create_dict("dit jaar", vanaf, tot, "maand"))

        # vorig jaar
        tot = vanaf
        vanaf = datetime.datetime(vanaf.year - 1, 1, 1)
        self.periodes.update(create_dict("vorig jaar", vanaf, tot, "maand"))

        # dit contractjaar
        vanaf = datetime.datetime.combine(
            self.prices_options.last_invoice
            if self.prices_options and not isinstance(self.prices_options, dict)
            else datetime.date.fromisoformat(self.prices_options["last invoice"]),
            datetime.time(),
        )
        if vanaf + datetime.timedelta(days=366) < vanaf_m:
            logging.warning(
                f'"last invoice" ({self.prices_options.last_invoice if hasattr(self.prices_options, "last_invoice") else self.prices_options["last invoice"]}) '
                f"is verouderd en moet worden bijgewerkt"
            )
            vanaf = vanaf_m - datetime.timedelta(days=366)
        tot = max_tot
        self.periodes.update(create_dict("dit contractjaar", vanaf, tot, "maand"))

        # 365 dagen
        tot = tot_g
        vanaf = tot + datetime.timedelta(days=-365)
        self.periodes.update(create_dict("365 dagen", vanaf, tot, "maand"))
        return

    def get_time_border_ha_record(
            self, sensor: str, latest: bool = True
    ) -> datetime.datetime:
        """
        Zoekt de tijd op van het laatst aanwezige record van "code"
        :param sensor: de code van het record
        :param latest: boolean, if true latest record else first record
        :return: datum en tijd van het laatst aanwezige record
        """

        statistics = Table(
            "statistics", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )
        statistics_meta = Table(
            "statistics_meta", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )
        # Define aliases for the tables
        t1 = statistics.alias("t1")
        v1 = statistics_meta.alias("v1")

        # Construct the query
        query = select(
            self.db_ha.from_unixtime(t1.c.start_ts).label("tijd"),
            t1.c.state,
        ).where(and_(v1.c.statistic_id == sensor, t1.c.metadata_id == v1.c.id))

        if latest:
            query = query.order_by(t1.c.start_ts.desc()).limit(1)
        else:
            query = query.order_by(t1.c.start_ts.asc()).limit(1)

        # Execute the query and fetch the result
        with self.db_ha.engine.connect() as connection:
            result = connection.execute(query)
            result = result.scalar()
            if type(result) is str:
                result = datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        return result

    def get_sensor_data(
            self,
            sensor: str,
            vanaf: datetime.datetime,
            tot: datetime.datetime,
            col_name: str,
            agg: str = "uur",
            sensor_type: str = "quantity",
    ) -> pd.DataFrame:
        """
        Retrieves and aggregates sensordata from ha database
        :param sensor: name off the sensor in ha
        :param vanaf: begin date/time
        :param tot: end date/time
        :param col_name: name off the column in the df
        :param agg: "maand", "dag" or "uur"
        :param sensor_type: "quantity" of "factor"
        :return: dataframe with the data
        """
        """
        if agg == "uur":
            sql = "SELECT FROM_UNIXTIME(t2.`start_ts`) 'tijd', " \
                  "FROM_UNIXTIME(t2.`start_ts`) 'tot', " \
              "round(greatest(t2.`state` - t1.`state`, 0),3) '" + col_name + "' " \
              "FROM `statistics` t1,`statistics` t2, `statistics_meta` " \
              "WHERE statistics_meta.`id` = t1.`metadata_id` 
                AND statistics_meta.`id` = t2.`metadata_id` " \
              "AND statistics_meta.`statistic_id` = '" + sensor + "' " \
              "AND (t2.`start_ts` = t1.`start_ts` + 3600) " \
              "AND t1.`state` IS NOT null AND t2.`state` IS NOT null " \
              "AND t1.`start_ts` >= UNIX_TIMESTAMP('" + str(vanaf) + "') - 3600 " \
              "AND t1.`start_ts` < UNIX_TIMESTAMP('" + str(tot) + "') - 3600 " \
              "ORDER BY t1.`start_ts`;"
        elif agg == "maand":
            sql = "SELECT concat(year(from_unixtime(t2.`start_ts`)), "\
                "LPAD(MONTH(from_unixtime(t2.`start_ts`)),3, ' ')) AS 'maand', " \
                "date_format(from_unixtime(t2.`start_ts`),'%Y-%m-01 00:00:00') AS 'tijd', " \
                "MAX(FROM_UNIXTIME(t2.`start_ts`)) 'tot', " \
                "ROUND(sum(greatest(t2.`state` - t1.`state`, 0)),3) '" + col_name + "' " \
                "FROM `statistics` t1,`statistics` t2, `statistics_meta` " \
                "WHERE statistics_meta.`id` = t1.`metadata_id` 
                AND statistics_meta.`id` = t2.`metadata_id` " \
                "AND statistics_meta.`statistic_id` = '" + sensor + "' " \
                "AND (t2.`start_ts` = t1.`start_ts` + 3600) " \
                "AND t1.`state` IS NOT null AND t2.`state` IS NOT null " \
                "AND t1.`start_ts` >= UNIX_TIMESTAMP('" + str(vanaf) + "') - 3600 " \
                "AND t1.`start_ts` < UNIX_TIMESTAMP('" + str(tot) + "') - 3600 " \
                "GROUP BY maand;"
        else:  # agg == "dag":
            sql = "SELECT date(from_unixtime(t2.`start_ts`)) AS 'dag', " \
                "date_format(from_unixtime(t2.`start_ts`),'%Y-%m-%d 00:00:00') AS 'tijd', " \
                "MAX(FROM_UNIXTIME(t2.`start_ts`)) AS 'tot', " \
                "ROUND(sum(greatest(t2.`state` - t1.`state`, 0)),3) '" + col_name + "' " \
                "FROM `statistics` t1,`statistics` t2, `statistics_meta` " \
                "WHERE statistics_meta.`id` = t1.`metadata_id` 
                AND statistics_meta.`id` = t2.`metadata_id` " \
                "AND statistics_meta.`statistic_id` = '" + sensor + "' " \
                "AND (t2.`start_ts` = t1.`start_ts` + 3600) " \
                "AND t1.`state` IS NOT null AND t2.`state` IS NOT null " \
                "AND t1.`start_ts` >= UNIX_TIMESTAMP('" + str(vanaf) + "') - 3600 " \
                "AND t1.`start_ts` < UNIX_TIMESTAMP('" + str(tot) + "') - 3600  "\
                "GROUP BY dag;"
        # print(sql)
        df = self.db_ha.run_select_query(sql)
        # print(df_sensor)
        return df
        """
        statistics = Table(
            "statistics", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )
        statistics_meta = Table(
            "statistics_meta", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )

        # Define aliases for the tables
        t1 = statistics.alias("t1")
        v1 = statistics_meta.alias("v1")

        # Define parameters
        # Epoch integers rather than formatted strings for SQL to parse
        # back: the column holds epochs, and which zone the database read
        # such a string in depended on the dialect and on server settings.
        start_ts_param1 = self.db_ha.epoch(vanaf)
        start_ts_param2 = self.db_ha.epoch(tot)
        if sensor_type == "quantity":
            t2 = statistics.alias("t2")
            if agg == "maand":
                column = self.db_ha.month(t2.c.start_ts).label("maand")
                column2 = func.min(self.db_ha.month_start(t2.c.start_ts)).label("tijd")
            elif agg == "dag":
                column = self.db_ha.day(t2.c.start_ts).label("dag")
                column2 = func.min(self.db_ha.day_start(t2.c.start_ts)).label("tijd")
            else:  # interval == "uur
                column = self.db_ha.hour(t2.c.start_ts).label("uur")
                column2 = self.db_ha.from_unixtime(t2.c.start_ts).label("tijd")

            if agg == "uur":
                columns = [
                    column,
                    column2,
                    self.db_ha.from_unixtime(t2.c.start_ts).label("tot"),
                    t2.c.start_ts.label("utc"),
                    case(
                        (t2.c.state > t1.c.state, t2.c.state - t1.c.state), else_=0
                    ).label(col_name),
                    v1.c.unit_of_measurement.label("dim"),
                ]
            else:
                # func.min/func.max, not the bare columns: PostgreSQL and
                # MySQL 8 (ONLY_FULL_GROUP_BY, the default there) reject a
                # non-aggregated column in a SELECT with GROUP BY. SQLite
                # and MariaDB are lenient about it, which is the only reason
                # this ever worked. unit_of_measurement is constant for one
                # sensor, so max() does not change its value.
                columns = [
                    column,
                    column2,
                    func.max(self.db_ha.from_unixtime(t2.c.start_ts)).label("tot"),
                    func.min(t2.c.start_ts).label("utc"),
                    func.sum(
                        case(
                            (t2.c.state > t1.c.state, t2.c.state - t1.c.state), else_=0
                        )
                    ).label(col_name),
                    func.max(v1.c.unit_of_measurement).label("dim"),
                ]

            # Build the query to retrieve raw data
            query = (
                select(*columns)
                .select_from(
                    t1.join(t2, t2.c.start_ts == t1.c.start_ts + 3600).join(
                        v1,
                        (v1.c.id == t1.c.metadata_id) & (v1.c.id == t2.c.metadata_id),
                    )
                )
                .where(
                    (v1.c.statistic_id == sensor)
                    & (t1.c.state.isnot(None))
                    & (t2.c.state.isnot(None))
                    & (
                            t1.c.start_ts
                            >= start_ts_param1 - 3600
                    )
                    & (
                            t1.c.start_ts
                            < start_ts_param2 - 3600
                    )
                )
            )
            if agg != "uur":
                query = query.group_by(agg)
        else:
            columns = [
                self.db_ha.hour(t1.c.start_ts).label("uur"),
                self.db_ha.from_unixtime(t1.c.start_ts).label("tijd"),
                self.db_ha.from_unixtime(t1.c.start_ts).label("tot"),
                t1.c.mean.label(col_name),
                v1.c.unit_of_measurement.label("dim"),
            ]
            query = (
                select(*columns)
                .select_from(
                    t1.join(
                        v1,
                        (v1.c.id == t1.c.metadata_id),
                    )
                )
                .where(
                    (v1.c.statistic_id == sensor)
                    & (t1.c.mean.isnot(None))
                    & (t1.c.start_ts >= start_ts_param1)
                    & (t1.c.start_ts < start_ts_param2)
                )
            )

        # Execute the query and load results into a DataFrame
        with self.db_ha.engine.connect() as connection:
            query_str = str(query.compile(connection))
            logging.debug(f"query get sensor data:\n {query_str}")
            df_raw = pd.read_sql(query, connection)

        if len(df_raw) == 0:
            df_raw = pd.DataFrame(columns=[agg, "tijd", "tot", col_name])
        df_raw["tijd"] = pd.to_datetime(df_raw["tijd"])
        df_raw.index = df_raw["tijd"]
        df_raw["tot"] = pd.to_datetime(df_raw["tot"])

        # when NaN in result replace with zero (0)
        df_raw[col_name] = df_raw[col_name].fillna(0)
        if len(df_raw) > 0:
            dim = df_raw.iloc[0]["dim"]
            if dim == "Wh":
                df_raw[col_name] = df_raw[col_name] / 1000
            elif dim == "MWh":
                df_raw[col_name] = df_raw[col_name] * 1000
        # Print the raw DataFrame

        # fill 0 for missing records
        if (agg == "uur") and (sensor_type == "quantity"):
            columns = list(df_raw.columns.values)
            df_insert = pd.DataFrame(columns=columns)
            """
            # vanaf hier voor de test:
            df_raw.reset_index(drop=True, inplace=True)
            df_raw.drop([0, 1, 4, 5, 6, 13], inplace=True)
            df_raw.index = df_raw["tijd"]
            # tot hier voor de test
            """
            row = None

            border = self.get_time_border_ha_record(sensor, latest=False)
            border = vanaf if border is None else max(border, vanaf)
            prev_time = pd.to_datetime(border - datetime.timedelta(hours=1))
            for row in df_raw.itertuples():
                new_tijd = prev_time + datetime.timedelta(hours=1)
                while new_tijd < row.tijd:
                    new_row_values = [
                        new_tijd.strftime("%Y-%m-%d %H:%M")[11:14] + "00",
                        pd.to_datetime(new_tijd),
                        pd.to_datetime(new_tijd),
                        # to_pydatetime(): a naive pandas Timestamp reads its
                        # wall clock as UTC in .timestamp(), datetime as local
                        pd.Timestamp(new_tijd).to_pydatetime().timestamp(),
                        0,
                        row.dim,
                    ]
                    df_insert.loc[len(df_insert.index)] = new_row_values
                    new_tijd += datetime.timedelta(hours=1)
                prev_time = row.tijd
            now = datetime.datetime.now()
            tot = min(datetime.datetime(now.year, now.month, now.day, now.hour), tot)
            if row is not None and row.tijd < tot - datetime.timedelta(hours=1):
                new_tijd = row.tijd + datetime.timedelta(hours=1)
                while new_tijd < tot:
                    new_row_values = [
                        new_tijd.strftime("%Y-%m-%d %H:%M")[11:14] + "00",
                        pd.to_datetime(new_tijd),
                        pd.to_datetime(new_tijd),
                        # to_pydatetime(): a naive pandas Timestamp reads its
                        # wall clock as UTC in .timestamp(), datetime as local
                        pd.Timestamp(new_tijd).to_pydatetime().timestamp(),
                        0,
                        row.dim,
                    ]
                    df_insert.loc[len(df_insert.index)] = new_row_values
                    new_tijd += datetime.timedelta(hours=1)

            if len(df_insert) > 0:
                df_insert.index = df_insert["tijd"]
                df_raw = pd.concat([df_raw, df_insert])
                df_raw.sort_index(inplace=True)
        logging.debug(f"sensordata raw, sensor {sensor},\n {df_raw.to_string()}\n")
        return df_raw

    @staticmethod
    def aggregate_data(
            df_raw: pd.DataFrame, col_name: str, agg: str = "uur"
    ) -> pd.DataFrame:
        df_raw["tot"] = df_raw.apply(
            lambda x: datetime.datetime.fromtimestamp(x["tijd"]), axis=1
        )

        if len(df_raw) > 0:
            # Extract year and month or day
            if agg == "maand":
                df_raw["maand"] = df_raw["start_ts_t2"].dt.to_period("M")
                df_aggregated = (
                    df_raw.groupby("maand")
                    .agg(
                        maand=(
                            "start_ts_t2",
                            lambda x: f"{x.dt.year.iloc[0]}-{x.dt.month.iloc[0]:2}",
                        ),
                        tijd=(
                            "start_ts_t2",
                            lambda x: x.dt.strftime("%Y-%m-01 00:00:00").iloc[0],
                        ),
                        tot=("start_ts_t2", "max"),
                        col_name=(col_name, "sum"),
                    )
                    .reset_index(drop=True)
                )
            elif agg == "dag":
                df_raw["dag"] = df_raw["start_ts_t2"].dt.to_period("D")
                df_aggregated = (
                    df_raw.groupby("dag")
                    .agg(
                        dag=(
                            "start_ts_t2",
                            lambda x: f"{x.dt.year.iloc[0]}-{x.dt.month.iloc[0]:2}-"
                                      f"{x.dt.day.iloc[0]:2}",
                        ),
                        tijd=(
                            "start_ts_t2",
                            lambda x: x.dt.strftime("%Y-%m-01 00:00:00").iloc[0],
                        ),
                        tot=("start_ts_t2", "max"),
                        col_name=(col_name, "sum"),
                    )
                    .reset_index(drop=True)
                )
            else:  # agg == "uur"
                df_raw["uur"] = df_raw["start_ts_t2"].dt.to_period("h")
                df_aggregated = df_raw
        else:
            df_aggregated = pd.DataFrame(columns=[agg, "tijd", "tot", col_name])
            df_aggregated.index = pd.to_datetime(df_aggregated["tijd"])
        # Round the  values

        # Print the aggregated DataFrame
        logging.debug(f"sensordata aggregated:\n {df_aggregated.to_string()}\n")
        return df_aggregated

    @staticmethod
    def copy_col_df(
            copy_from: pd.DataFrame, copy_to: pd.DataFrame, col_name: str
    ) -> pd.DataFrame:
        """
        kopieert kolom "col_name" van copy_from naar copy_to,
        :param copy_from:
        :param copy_to:
        :param col_name:
        :return: de ingevuld copy_to
        """
        dt = copy_from[col_name].dtype
        # Use is_numeric_dtype to catch all numeric types (float64, int64, etc.)
        if pd.api.types.is_numeric_dtype(dt):
            copy_to[col_name] = 0.0
        else:
            copy_to[col_name] = ""
        # copy_from = copy_from.reset_index()
        for row in copy_from.itertuples():
            copy_to.at[row.tijd, col_name] = copy_from.at[row.tijd, col_name]
        return copy_to

    @staticmethod
    def add_col_df(
            add_from: pd.DataFrame,
            add_to: pd.DataFrame,
            col_name_from: str,
            col_name_to: str = None,
            negation: bool = False,
    ) -> pd.DataFrame:
        # add_from = add_from.reset_index()
        if add_from is None:
            return add_to
        factor = -1 if negation else +1
        if col_name_to is None:
            col_name_to = col_name_from
        col_index = add_from.columns.get_loc(col_name_from) + 1
        if "tijd" in add_from.columns:
            for row in add_from.itertuples():
                # add_from.at[row.tijd, col_name_from])
                if row.tijd in add_to.index:
                    org_value = add_to.at[row.tijd, col_name_to]
                    try:
                        if pd.isna(org_value):
                            org_value = 0
                    except ValueError:
                        org_value = 0
                    if pd.notna(row[col_index]):
                        add_to.at[row.tijd, col_name_to] = (
                                org_value + factor * row[col_index]
                        )
        else:
            for row in add_from.itertuples():
                # add_from.at[row.tijd, col_name_from])
                if row.time in add_to.index:
                    org_value = add_to.at[row.tijd, col_name_to]
                    if pd.isna(org_value):
                        org_value = 0
                    add_to.at[row.time, col_name_to] = (
                            org_value + factor * row[col_index]
                    )
        return add_to

    def get_latest_present(self, code: str) -> datetime.datetime:
        """
        :param code: de code van de variabele
        :return: datetime van het laatste record
        """
        """
        sql = "SELECT `time`, `variabel`.`id`, `value` \
                FROM `values` , `variabel`  \
                WHERE `variabel`.`code` = '" + code + "'  \
                AND `values`.`variabel` = `variabel`.`id`  \
                ORDER BY `time` DESC  \
                LIMIT 1;"
        data = self.db_da.run_select_query(sql)
        """
        values_table = Table(
            "values", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the values table
        t1 = values_table.alias("t1")
        variabel_table = Table(
            "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the variabel table
        v1 = variabel_table.alias("v1")

        query = (
            select(t1.c.time)
            .where(
                and_(
                    v1.c.code == code,
                    t1.c.variabel == v1.c.id,
                )
            )
            .order_by(t1.c.time.desc())
            .limit(1)
        )
        with self.db_da.engine.connect() as connection:
            latest = connection.execute(query).scalar()
        if latest is not None:
            result = datetime.datetime.fromtimestamp(int(latest))
        else:
            result = datetime.datetime(year=2020, month=1, day=1)
        return result

    def get_sensor_sum(
            self,
            sensor_list: list,
            vanaf: datetime.datetime,
            tot: datetime.datetime,
            col_name: str,
    ) -> pd.DataFrame:
        """
        Berekent een dataframe met sum van de waarden van de sensoren in de list
        :param sensor_list: een list of strings met de entiteiten van de sensoren
        :param vanaf: berekenen vanaf
        :param tot: berekenen tot
        :param col_name: string naam van de kolom
        :return: het dataframe met de sum
        """
        counter = 0
        result = None
        for sensor in sensor_list:
            df = self.get_sensor_data(sensor, vanaf, tot, col_name)
            df.index = pd.to_datetime(df["tijd"])
            if counter == 0:
                result = df
            else:
                result = self.add_col_df(df, result, col_name)
            counter = +1
        return result

    CONSOLIDATE_COLUMNS = ["time", "code", "value", "tijd"]

    def _settlement_epoch(self, value):
        if type(value) is int:
            return value
        moment = pd.Timestamp(value).to_pydatetime()
        return self.db_da.epoch(moment) if hasattr(self, "db_da") else int(moment.timestamp())

    def _financial_ranges(self, start, end):
        """Kies één bedragenbron per halfopen deelbereik; onbekende historie legacy."""
        pricing = getattr(self, "prices_options", None)
        cfg = getattr(pricing, "settlement", None)
        lower, upper = self._settlement_epoch(start), self._settlement_epoch(end)
        if cfg is None or cfg.mode != "active":
            return [(False, lower, upper)]
        contracts = getattr(cfg, "contracts", None)
        if contracts is None:
            raise ValueError("Actieve afrekening vereist expliciete contractdekking")
        windows, points = [], {lower, upper}
        for contract in contracts:
            zone = ZoneInfo(contract.timezone)
            left = int(datetime.datetime.combine(contract.valid_from, datetime.time.min, zone).timestamp())
            right = upper if contract.valid_to is None else int(datetime.datetime.combine(
                contract.valid_to, datetime.time.min, zone).timestamp())
            if left < upper and lower < right:
                windows.append((left, right))
                points.update((max(lower, left), min(upper, right)))
        ordered = sorted(points)
        return [(any(left <= at < right for left, right in windows), at, stop)
                for at, stop in zip(ordered, ordered[1:], strict=False) if at < stop]

    def _local_datetime(self, at):
        zone = ZoneInfo(self.time_zone) if hasattr(self, "time_zone") else getattr(self.db_da, "tzinfo", None)
        return datetime.datetime.fromtimestamp(at, zone).replace(tzinfo=None)

    def _pin_settlement_report(self, start, end):
        service = self.settlement_service()
        return service.repository.read(service.meter_id, self._settlement_epoch(start), self._settlement_epoch(end))

    @staticmethod
    def _check_report_basis(view, basis):
        if (view.generation, view.input_revision, view.current_input_revision) != (
                basis.generation, basis.input_revision, basis.current_input_revision):
            from dao.prog.settlement.repository import ConcurrentSettlementError

            raise ConcurrentSettlementError("Afrekenbasis wijzigde tijdens het samengestelde rapport")

    @staticmethod
    def _check_report_sources(view, basis):
        """De aanvullende bronread moet hetzelfde boek beschrijven als de rapportbasis.

        De drie revisievelden, de versheid en de exacte regel- en creditinhoud
        mogen niet verschillen; anders zou het rapport stilletjes een ander
        boek tonen dan het bedrag dat het zelf gelezen heeft.
        """
        if ((view.generation, view.input_revision, view.current_input_revision, view.freshness)
                != (basis.generation, basis.input_revision, basis.current_input_revision, basis.freshness)
                or view.lines != basis.lines or view.adjustments != basis.adjustments):
            from dao.prog.settlement.repository import ConcurrentSettlementError

            raise ConcurrentSettlementError("Afrekenbron wijzigde tijdens het samengestelde rapport")

    def _book_energy(self, service, view, start, end):
        """Exact bewezen historische energie voor de gepinde boekgeneratie.

        De gedeelde historische lezer bindt elke boekregel aan zijn eigen
        geregistreerde invoer; de actuele invoerlookup is geen geschiedenis.
        Alleen een regel zonder ontbrekende of strijdige bron levert kWh: een
        onvolledig of tegenstrijdig bewezen boekregel blijft onbekend, net als
        een ontbrekende invoer. De financiële onvolledigheid van de regel zelf
        (bonus-btw, onbekende tariefregel) blijft buiten deze kWh-toets.
        """
        basis = service.repository.read_reconciliation(service.meter_id, self._settlement_epoch(start),
                                                       self._settlement_epoch(end), generation=view.generation)
        self._check_report_sources(view, basis.view)
        proven = {}
        for source in basis.sources.lines:
            if source.issues or source.item is None:
                continue  # Zonder bewezen identiteit blijft de energie onbekend.
            proven[(source.start_ts, source.input_revision, source.input_hash)] = source.item
        return proven

    def get_settlement_data(self, start, end, *, generation=None) -> pd.DataFrame:
        service = self.settlement_service()
        view = service.repository.read(service.meter_id, self._settlement_epoch(start), self._settlement_epoch(end),
                                       generation=generation)
        metadata = {"generation": view.generation, "input_revision": view.input_revision,
                    "current_input_revision": view.current_input_revision, "freshness": view.freshness,
                    "bonus_kind": "calculated"}
        rows = []
        for line in view.lines:
            amounts = (line.import_cost, line.base_export_revenue, line.bonus_revenue)
            with localcontext() as context:
                context.prec = 36
                net = None if any(value is None for value in amounts) else amounts[0] - amounts[1] - amounts[2]
            rows.append({**metadata, "kind": "interval", "time": line.start_ts, "end_ts": line.end_ts,
                         "import_cost": line.import_cost, "base_export_revenue": line.base_export_revenue,
                         "bonus_revenue": line.bonus_revenue, "bonus_kwh": line.bonus_kwh,
                         "credit_total": None, "net_cost": net, "snapshot_hash": line.snapshot_hash,
                         "input_hash": line.input_hash, "line_input_revision": line.input_revision,
                         "quality": line.quality, "issues": line.issues})
        for adjustment in view.adjustments:
            rows.append({**metadata, "kind": adjustment.kind, "time": adjustment.period_end_ts - 1,
                         "end_ts": adjustment.period_end_ts, "period_start_ts": adjustment.period_start_ts,
                         "period_end_ts": adjustment.period_end_ts, "import_cost": None,
                         "base_export_revenue": None, "bonus_revenue": None, "bonus_kwh": None,
                         "credit_total": adjustment.credit_total, "credit_ex_vat": adjustment.credit_ex_vat,
                         "credit_vat": adjustment.credit_vat, "basis_ex_vat": adjustment.basis_ex_vat,
                         "net_cost": adjustment.credit_total.copy_negate(), "quality": adjustment.quality, "issues": ()})
        columns = ["generation", "input_revision", "current_input_revision", "freshness", "bonus_kind", "kind",
                   "time", "end_ts", "import_cost", "base_export_revenue", "bonus_revenue", "bonus_kwh",
                   "credit_total", "net_cost", "quality", "issues"]
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
        frame["quality_label"] = frame.quality.map({"measured": "berekend", "estimated": "geschat",
                                                   "incomplete": "onvolledig"})
        frame.attrs["view"] = view
        return frame

    def _book_grid(self, start, end, interval, *, generation=None, basis=None):
        frame = self.get_settlement_data(start, end, generation=basis.generation if basis is not None else generation)
        service = self.settlement_service()
        view = frame.attrs["view"]
        if basis is not None:
            self._check_report_basis(view, basis)
        energy = self._book_energy(service, view, start, end)
        rows = []
        for row in frame.to_dict("records"):
            if row["kind"] == "interval" and row["end_ts"] > end:
                continue  # Geen heel intervalbedrag aan een gedeeltelijk bereik toerekenen.
            at = row["time"]
            moment = datetime.datetime.fromtimestamp(at, ZoneInfo(self.time_zone))
            label = moment.strftime("%Y-%m-%d %H:00%z" if interval == "uur" else
                                    "%Y-%m-%d" if interval == "dag" else "%Y-%m")
            is_line = row["kind"] == "interval"
            item = energy.get((at, row.get("line_input_revision"), row.get("input_hash"))) if is_line else None
            profit = _amount_sum((row["base_export_revenue"], row["bonus_revenue"])) if is_line else row["credit_total"]
            rows.append({interval: label, "vanaf": moment,
                         "tot": datetime.datetime.fromtimestamp(row["end_ts"], ZoneInfo(self.time_zone)),
                         "consumption": item.import_kwh if item is not None else None if is_line else Decimal("0"),
                         "production": item.export_kwh if item is not None else None if is_line else Decimal("0"),
                         "cost": row["import_cost"] if is_line else Decimal("0"), "profit": profit,
                         "datasoort": "calculated", "quality": row["quality"], "freshness": view.freshness,
                         "generation": view.generation})
        cursor = start
        covered = [(line.start_ts, line.end_ts) for line in view.lines if line.end_ts <= end]
        for left, right in [*covered, (end, end)]:
            while cursor < left:
                stop = min(left, (cursor // 3600 + 1) * 3600)
                moment = datetime.datetime.fromtimestamp(cursor, ZoneInfo(self.time_zone))
                label = moment.strftime("%Y-%m-%d %H:00%z" if interval == "uur" else
                                        "%Y-%m-%d" if interval == "dag" else "%Y-%m")
                rows.append({interval: label, "vanaf": moment,
                             "tot": datetime.datetime.fromtimestamp(stop, ZoneInfo(self.time_zone)),
                             "consumption": None, "production": None,
                             "cost": None, "profit": None, "datasoort": "calculated", "quality": "incomplete",
                             "freshness": view.freshness, "generation": view.generation})
                cursor = stop
            cursor = max(cursor, right)
        columns = [interval, "vanaf", "tot", "consumption", "production", "cost", "profit", "datasoort",
                   "quality", "freshness", "generation"]
        result = pd.DataFrame(rows, columns=columns)
        if not result.empty:
            result = result.groupby(interval, as_index=False).agg({
                "vanaf": "min", "tot": "max", "consumption": _amount_sum, "production": _amount_sum,
                "cost": _amount_sum, "profit": _amount_sum, "datasoort": "first", "freshness": "first",
                "generation": "first", "quality": lambda values: "incomplete" if "incomplete" in set(values)
                else "estimated" if "estimated" in set(values) else "measured",
            })
            # Dubbele wintertijduren op werkelijk tijdstip ordenen, niet op offsetlabel.
            result = result.sort_values("vanaf").reset_index(drop=True)
        with localcontext() as context:
            context.prec = 36
            result["netto_consumption"] = result["consumption"] - result["production"]
            result["netto_cost"] = result["cost"] - result["profit"]
        result["bonus_kind"] = "calculated"
        result["quality_label"] = result.quality.map({"measured": "berekend", "estimated": "geschat",
                                                      "incomplete": "onvolledig"})
        result.attrs["view"] = view
        return result

    def calc_cost(
            self, vanaf: datetime.datetime, tot: datetime.datetime, code: str
    ) -> pd.DataFrame | None:
        frames = []
        ranges = self._financial_ranges(vanaf, tot)
        basis = self._pin_settlement_report(vanaf, tot) if any(use for use, _, _ in ranges) else None
        for active, start, end in ranges:
            if active:
                data = self.get_settlement_data(start, end, generation=basis.generation)
                self._check_report_basis(data.attrs["view"], basis)
                rows = []
                for row in data.to_dict("records"):
                    value = row["import_cost"] if code == "cost" else (
                        _amount_sum((row["base_export_revenue"], row["bonus_revenue"]))
                        if row["kind"] == "interval" else row["credit_total"])
                    if value is not None:
                        rows.append({"time": str(row["time"]), "code": code, "value": value,
                                     "tijd": self._local_datetime(row["time"])})
                frames.append(pd.DataFrame(rows, columns=self.CONSOLIDATE_COLUMNS))
            else:
                # De ongewijzigde legacygrens blijft float-/rapportcompatibel.
                frame = self._calc_legacy_cost(vanaf if start == self._settlement_epoch(vanaf) else
                                               self._local_datetime(start), tot if end == self._settlement_epoch(tot)
                                               else self._local_datetime(end), code)
                if frame is not None:
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else None

    def _calc_legacy_cost(self, vanaf, tot, code):
        """Hourly cost ("cost") or revenue ("profit") from the grid sensors and
        the tariff of that hour, as rows ready for savedata()."""
        if code == "cost":
            sensors, price_column = self.grid_dict["cons"]["sensors"], "da_cons"
        else:
            sensors, price_column = self.grid_dict["prod"]["sensors"], "da_prod"
        energy = self.get_sensor_sum(sensors, vanaf, tot, "energy")
        if energy is None or len(energy) == 0:
            return None
        prices = self.get_price_data(vanaf, tot)
        if len(prices) == 0:
            logging.warning(f"Geen tarieven tussen {vanaf} en {tot}, {code} niet berekend")
            return None
        if "da_bonus" in prices.columns:
            # Legacyraming: potentiële bonus is geen bewijs van leveranciersbetaling.
            prices["da_prod"] = prices["da_prod"] + prices["da_bonus"]
        energy = energy.reset_index(drop=True)
        energy["tijd"] = pd.to_datetime(energy["tijd"])
        prices = prices.rename(columns={"time": "tijd"})
        prices["tijd"] = pd.to_datetime(prices["tijd"])
        merged = energy.merge(prices[["tijd", price_column]], on="tijd", how="inner")
        rows = [
            {
                # utc is the recorder's start_ts, the true epoch of the hour.
                "time": str(int(row.utc)),
                "code": code,
                "value": float(row.energy) * float(getattr(row, price_column)),
                "tijd": row.tijd,
            }
            for row in merged.itertuples()
            if pd.notna(row.energy) and pd.notna(getattr(row, price_column))
        ]
        return pd.DataFrame(rows, columns=self.CONSOLIDATE_COLUMNS)

    def consolidate_data(self, _start=None, _end=None) -> None:
        """Copy the hourly grid consumption, production, cost and revenue from
        the Home Assistant recorder into the day-ahead database.

        Each code continues where the database stops. The epoch stored is the
        recorder's own start_ts; deriving it from the naive local "tijd" column
        would shift every value by the UTC offset.
        """
        if _end is None:
            now = datetime.datetime.now()
            tot = datetime.datetime(now.year, now.month, now.day)
        else:
            tot = _end
        for code, categorie in self.grid_dict.items():
            if _start is None:
                start = self.get_latest_present(code) + datetime.timedelta(hours=1)
            else:
                start = _start
            if start >= tot:
                logging.info(f"Consolideren {code}: al bijgewerkt tot {start}")
                continue
            if categorie["sensors"] == "calc":
                # Grootboekbedragen worden nooit naar historische values.cost/profit gekopieerd.
                ranges = self._financial_ranges(start, tot)
                frames = ([self._calc_legacy_cost(start, tot, code)] if not any(use for use, _, _ in ranges) else
                          [self._calc_legacy_cost(self._local_datetime(left), self._local_datetime(right), code)
                           for active, left, right in ranges if not active])
                frames = [frame for frame in frames if frame is not None]
                df_db = pd.concat(frames, ignore_index=True) if frames else None
            else:
                data = self.get_sensor_sum(categorie["sensors"], start, tot, code)
                if data is None or len(data) == 0:
                    df_db = None
                else:
                    data = data.reset_index(drop=True)
                    data["tijd"] = pd.to_datetime(data["tijd"])
                    rows = [
                        {
                            "time": str(int(row.utc)),
                            "code": code,
                            "value": float(getattr(row, code)),
                            "tijd": row.tijd,
                        }
                        for row in data.itertuples()
                        if pd.notna(getattr(row, code)) and pd.notna(row.utc)
                    ]
                    df_db = pd.DataFrame(rows, columns=self.CONSOLIDATE_COLUMNS)
            if df_db is None or df_db.empty:
                logging.info(f"Consolideren {code}: geen nieuwe gegevens vanaf {start}")
                continue
            logging.info(
                f"Consolideren {code}: {len(df_db)} uren vanaf "
                f"{df_db['tijd'].iloc[0]} tot en met {df_db['tijd'].iloc[-1]}"
            )
            logging.debug(f"\n{df_db.to_string(index=False)}")
            self.db_da.savedata(df_db, tablename="values")

    def recalc_df_ha(self, org_data_df: pd.DataFrame, interval: str) -> pd.DataFrame:
        def get_datasoort(ds):
            for s in ds:
                if s == "expected":
                    return "expected"
            return "recorded"

        columns = [
            interval,
            "vanaf",
            "tot",
            "consumption",
            "production",
            "cost",
            "profit",
            "datasoort",
        ]
        if len(org_data_df.index) == 0:
            return pd.DataFrame(columns=columns)
        # One list of tuples built up front, then a single DataFrame(), not
        # fi_df.loc[fi_df.shape[0]] = row per source row: that copies the
        # whole (growing) frame on every append.
        rows = []
        basis = None
        for row in org_data_df.itertuples():
            if pd.isnull(row.tijd):
                continue
            if not isinstance(row.tijd, datetime.datetime):
                print(row)
            if interval == "uur":
                tijd_str = str(row.tijd)[0:14] + "00"
            elif interval == "dag":
                tijd_str = str(row.tijd)[0:10]
            else:
                tijd_str = str(row.tijd)[0:7]  # jaar maand
            col_1 = row.consumption
            col_2 = row.production
            end = getattr(row, "tot", row.tijd + datetime.timedelta(hours=1))
            active = any(use for use, _, _ in self._financial_ranges(row.tijd, end))
            if active:
                if basis is None:
                    basis = self._pin_settlement_report(row.tijd, end)
                data = self._book_grid(self._settlement_epoch(row.tijd), self._settlement_epoch(end), "uur", basis=basis)
                col_3, col_4 = (_amount_sum(data.cost), _amount_sum(data.profit))
            elif is_number(row.consumption) and is_number(row.da_cons):
                col_3 = row.consumption * row.da_cons
            else:
                col_3 = 0
            if not active and is_number(row.production) and is_number(row.da_prod):
                bonus = getattr(row, "da_bonus", 0.0)
                if not is_number(bonus) or pd.isna(bonus):
                    bonus = 0.0
                col_4 = row.production * (row.da_prod + bonus)
            elif not active:
                col_4 = 0
            col_5 = row.datasoort
            rows.append(
                (
                    tijd_str,
                    row.tijd,
                    row.tijd + datetime.timedelta(hours=1),
                    col_1,
                    col_2,
                    col_3,
                    col_4,
                    col_5,
                )
            )

        fi_df = pd.DataFrame(rows, columns=columns)
        fi_df = fi_df.groupby([interval], as_index=False).agg(
            {
                "vanaf": "min",
                "tot": "max",
                "consumption": "sum",
                "production": "sum",
                "cost": lambda values: None if values.isna().any() else values.sum(),
                "profit": lambda values: None if values.isna().any() else values.sum(),
                "datasoort": get_datasoort,
            }
        )
        if basis is not None:
            fi_df.attrs["view"] = basis
        return fi_df

    def aggregate_balance_df(self, df: pd.DataFrame, interval: str):
        df = df.rename(columns={"tijd": "vanaf"})
        df["tot"] = df["vanaf"]
        tot_values = df.pop("tot")
        df.insert(1, "tot", tot_values)
        datasoort_values = df.pop("datasoort")
        df.insert(2, "datasoort", datasoort_values)
        columns = [interval] + df.columns.tolist()
        rows = []
        for row in df.itertuples():
            if interval == "uur":
                tijd_str = str(row.vanaf)[10:16]
            elif interval == "dag":
                tijd_str = str(row.vanaf)[0:10]
            else:
                tijd_str = str(row.vanaf)[0:7]  # jaar maand
            rows.append(
                (
                    tijd_str,
                    row.vanaf,
                    row.vanaf + datetime.timedelta(hours=1),
                    row.datasoort,
                    row.cons,
                    row.prod,
                    row.bat_out,
                    row.bat_in,
                    row.pv_ac,
                    row.ev,
                    row.wp,
                    row.boil,
                    row.base,
                )
            )
        result = pd.DataFrame(rows, columns=columns)

        if interval != "uur":
            agg_dict = {"vanaf": "min"}
            for key, categorie in self.energy_balance_dict.items():
                agg_dict[key] = "sum"
            result = result.groupby([interval], as_index=False).agg(agg_dict)

        return result

    @staticmethod
    def calc_base(df: pd.DataFrame) -> pd.DataFrame:
        base_load = []
        for row in df.itertuples():
            base_load.append(
                row.cons
                - row.prod
                + row.bat_out
                - row.bat_in
                + row.pv_ac
                - row.ev
                - row.wp
                - row.boil
                - row.mach
            )
        result = df.assign(base=base_load)
        return result

    @staticmethod
    def calc_netto_cons(df: pd.DataFrame) -> pd.DataFrame:
        df["netto_cons"] = df["cons"] - df["prod"]
        return df

    @staticmethod
    def calc_emissie(df: pd.DataFrame) -> pd.DataFrame:
        df["emissie"] = df["netto_cons"] * df["co2_intensity"] / 1000
        return df

    @staticmethod
    def tijd_at_interval(
            interval: str, moment: datetime.datetime, as_index: bool = False
    ) -> str | int:
        if interval == "maand":
            result = datetime.datetime(moment.year, moment.month, day=1)
            if as_index:
                return result.strftime("%Y-%m")
        elif interval == "dag":
            result = datetime.datetime(moment.year, moment.month, moment.day)
            if as_index:
                return result.strftime("%Y-%m-%d")
        elif interval == "weekdag":
            return moment.weekday()
        elif interval == "heel_uur":
            return moment.hour
        else:  # uur
            result = moment
        return result.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def generate_df(
            vanaf: datetime.datetime,
            tot: datetime.datetime,
            rep_interval: str,
            get_interval: str | None = None,
            column: str | None = None,
    ) -> pd.DataFrame:
        # Building the frame with one .loc[shape[0]] append per row is O(n^2)
        # (a full copy on every append) and materialises to 8760 rows for an
        # hourly "dit jaar" report. Rows are collected as plain tuples and
        # handed to pd.DataFrame(...) once instead.
        moment = vanaf
        now = datetime.datetime.now()
        now = datetime.datetime(now.year, now.month, now.day, now.hour)
        if get_interval is None:
            step_interval = rep_interval
        else:
            step_interval = get_interval
        rows = []
        while moment < tot:
            if get_interval == "maand":
                old_moment = datetime.datetime(moment.year, moment.month, day=1)
            else:
                old_moment = moment
            moment_str = str(moment)
            if rep_interval == "uur":
                tijd_str = moment_str[10:16]
            elif rep_interval == "dag":
                tijd_str = moment_str[0:10]
            else:  # maand
                tijd_str = moment_str[0:7]  # jaar maand
            if step_interval == "uur":
                moment = moment + datetime.timedelta(hours=1)
            elif step_interval == "dag":
                moment = moment + datetime.timedelta(days=1)
            else:  # "maand":
                moment = old_moment + relativedelta(months=1)
            datasoort = "recorded" if moment <= now else "expected"
            rows.append((tijd_str, old_moment, moment, datasoort))
        result = pd.DataFrame(rows, columns=[rep_interval, "tijd", "tot", "datasoort"])
        result.index = pd.to_datetime(result["tijd"])
        if column is not None:
            result[column] = 0.0
        return result

    def get_energy_balance_data(
            self,
            periode: str,
            col_dict: dict = None,
            field: str = None,
            _vanaf: datetime.datetime = None,
            _tot: datetime.datetime = None,
            _interval: str = None,
    ):
        """
        berekent een report conform de col_dict configuratie
        :param periode: key van een van de self.periodes
        :param col_dict: of self.energy_balance_dict of self.co2_dict
        :param field: one particular field
        :param _vanaf: als afwijkt van periode.vanaf
        :param _tot: als afwijkt van periode.tot
        :param _interval: als afwijkt van periode.interval
        :return: dataframe met data, last_moment van de data
        """
        periode_d = self.periodes[periode]
        vanaf = _vanaf if _vanaf else periode_d["vanaf"]
        tot = _tot if _tot else periode_d["tot"]
        interval = periode_d["interval"] if _interval is None else _interval
        if col_dict is None:
            col_dict = self.energy_balance_dict

        last_realised_moment = datetime.datetime.fromtimestamp(
            math.floor(datetime.datetime.now().timestamp() / 3600) * 3600
        )
        result = self.generate_df(vanaf, tot, periode_d["interval"], interval)

        values_table = Table(
            "values", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the values table
        t1 = values_table.alias("t1")
        variabel_table = Table(
            "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the variabel table
        v1 = variabel_table.alias("v1")
        groupby_str = interval
        # column2, when set, is the bucket's own start moment (the 1st of
        # the month, midnight, or the hour mark) rather than the earliest
        # timestamp that happens to have data in it. Using min(time) as the
        # label made the first, partial bucket of a period that does not
        # start on a bucket boundary (a "dit contractjaar" or "365 dagen"
        # view starting mid-month) fail to match generate_df's bucket-start
        # label further down, silently dropping that bucket from the report.
        column2 = None
        if interval == "maand":
            column = self.db_da.month(t1.c.time).label("maand")
            column2 = func.min(self.db_da.month_start(t1.c.time)).label("tijd")
        elif interval == "dag":
            column = self.db_da.day(t1.c.time).label("dag")
            column2 = func.min(self.db_da.day_start(t1.c.time)).label("tijd")
        else:  # interval == "uur"
            if interval == periode_d["interval"]:
                column = self.db_da.hour(t1.c.time).label("uur")
                column2 = func.min(self.db_da.hour_start(t1.c.time)).label("tijd")
            else:
                # get_interval is finer than rep_interval: each group is a
                # single exact timestamp already, so it needs no separate
                # bucket-start column.
                column = self.db_da.from_unixtime(t1.c.time).label("tijd")
                groupby_str = "tijd"
        if field is not None and col_dict[field]["sensors"] == "calc":
            field = None
        last_moment = vanaf
        for key, categorie in col_dict.items():
            if field is not None and key != field:
                continue
            result[key] = 0.0
            """
            if interval == "maand":
                sql = "SELECT concat(year(from_unixtime(t1.`time`)),
                LPAD(MONTH(from_unixtime(t1.`time`)),3, ' ')) "\
                      "AS 'maand', " \
                    "date_format(from_unixtime(t1.`time`),'%Y-%m-01 00:00:00') AS 'tijd', " \
                    "MAX(from_unixtime(t1.`time`)) AS 'tot', " \
                    "sum(t1.`value`) " + key + " " \
                    "FROM `values` AS t1, `variabel`AS v1  " \
                    "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                    "t1.`time` >= UNIX_TIMESTAMP('"+str(vanaf)+"') 
                    AND t1.`time` < UNIX_TIMESTAMP('"+str(tot)+"') " \
                    "GROUP BY maand;"
            elif interval == "dag":
                sql = "SELECT date(from_unixtime(t1.`time`)) AS 'dag', " \
                    "date_format(from_unixtime(t1.`time`),'%Y-%m-%d 00:00:00') AS 'tijd', " \
                    "MAX(from_unixtime(t1.`time`)) AS 'tot', " \
                    "sum(t1.`value`) " + key + " " \
                    "FROM `values` AS t1, `variabel`AS v1  " \
                    "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                    "t1.`time` >= UNIX_TIMESTAMP('"+str(vanaf)+"') 
                    AND t1.`time` < UNIX_TIMESTAMP('"+str(tot)+"') " \
                    "GROUP BY dag;"
            else:  # interval == "uur"
                sql = "SELECT from_unixtime(t1.`time`) AS 'uur', " \
                  "from_unixtime(t1.`time`) AS 'tijd', from_unixtime(t1.`time`) AS 'tot', " \
                  "t1.`value` '" + key + "' " \
                  "FROM `values` AS t1, `variabel`AS v1  " \
                  "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                  "t1.`time`>= UNIX_TIMESTAMP('" + str(vanaf) + "') 
                  AND t1.`time` < UNIX_TIMESTAMP('" + str(tot) + "');"
            # print(sql)
            code_result = self.db_da.run_select_query(sql)
            """

            select_columns = [column]
            if column2 is not None:
                select_columns.append(column2)
            select_columns.extend(
                [
                    func.min(self.db_da.from_unixtime(t1.c.time)).label("vanaf"),
                    func.max(self.db_da.from_unixtime(t1.c.time)).label("tot"),
                    func.sum(t1.c.value).label(key),
                ]
            )
            query = (
                select(*select_columns)
                .where(
                    and_(
                        v1.c.code == key,
                        t1.c.variabel == v1.c.id,
                        t1.c.time
                        >= self.db_da.epoch(vanaf),
                        t1.c.time
                        < self.db_da.epoch(tot),
                    )
                )
                .group_by(groupby_str)
            )

            with self.db_da.engine.connect() as connection:
                code_result = pd.read_sql(query, connection)
            code_result["vanaf"] = pd.to_datetime(code_result["vanaf"])
            # "tijd" already holds the bucket start (column2) or the exact
            # timestamp (column, in the finer-than-bucket case); do not
            # overwrite it with "vanaf" (the earliest timestamp that
            # happened to have data), which broke the first partial bucket.
            code_result["tijd"] = pd.to_datetime(code_result["tijd"])
            code_result["tot"] = pd.to_datetime(code_result["tot"])

            # if len(code_result) > 0:
            #     code_result['tijd'] = code_result.apply(lambda x: self.tijd_at_interval(interval,
            #     x['tijd']), axis=1)

            code_result.index = code_result["tijd"]

            if code_result.shape[0] == 0:
                # datetime.datetime.combine(vanaf, datetime.time(0,0)) - datetime.timedelta(hours=1)
                last_moment = vanaf
            else:
                self.add_col_df(code_result, result, key)
                last_moment = code_result["tot"].iloc[-1] + datetime.timedelta(hours=1)
            if last_moment < tot:
                ha_result = None
                if categorie["sensors"] == "calc":
                    function = categorie["function"]
                    ha_result = getattr(self, function)(result)
                else:
                    for sensor in categorie["sensors"]:
                        if "sensor_type" in categorie:
                            sensor_type = categorie["sensor_type"]
                        else:
                            sensor_type = "quantity"
                        ha_result = self.get_sensor_data(
                            sensor,
                            last_moment,
                            tot,
                            key,
                            agg=interval,
                            sensor_type=sensor_type,
                        )
                        ha_result["tot"] = pd.to_datetime(ha_result["tijd"])
                        if interval == "maand":
                            ha_result["tijd"] = pd.to_datetime(ha_result[interval])
                        ha_result.index = pd.to_datetime(ha_result["tijd"])
                        result = self.add_col_df(ha_result, result, key)
                if ha_result is not None and len(ha_result) > 0:
                    if categorie["sensors"] == "calc":
                        now = datetime.datetime.now()
                        last_moment = max(
                            datetime.datetime(now.year, now.month, now.day, now.hour),
                            vanaf,
                        )
                    else:
                        last_moment = ha_result["tot"].iloc[-1] + datetime.timedelta(
                            hours=1
                        )
                else:
                    last_moment = vanaf

            if last_moment < last_realised_moment:
                last_moment = last_realised_moment
            if last_moment < tot:
                """
                if interval == "maand":
                    sql = "SELECT concat(year(from_unixtime(t1.`time`)), 
                                  LPAD(MONTH(from_unixtime(t1.`time`)),3, ' ')) AS 'maand', " \
                          "date_format(from_unixtime(t1.`time`),'%Y-%m-01 00:00:00') AS 'tijd', " \
                          "MAX(from_unixtime(t1.`time`)) AS 'tot', " \
                          "sum(t1.`value`) " + key + " " \
                          "FROM `prognoses` AS t1, `variabel`AS v1  " \
                          "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                          "t1.`time` >= UNIX_TIMESTAMP('" + str(last_moment) + ("') AND " 
                          "t1.`time` < UNIX_TIMESTAMP('") + str(tot) + "') " \
                          "GROUP BY maand;"
                elif interval == "dag":
                    sql = "SELECT date(from_unixtime(t1.`time`)) AS 'dag', " \
                          "date_format(from_unixtime(t1.`time`),'%Y-%m-%d 00:00:00') AS 'tijd', " \
                          "MAX(from_unixtime(t1.`time`)) AS 'tot', " \
                          "sum(t1.`value`) " + key + " " \
                          "FROM `prognoses` AS t1, `variabel`AS v1  " \
                          "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                          "t1.`time` >= UNIX_TIMESTAMP('" + str(last_moment) + ("') AND " 
                          "t1.`time` < UNIX_TIMESTAMP('") + str(tot) + "') " \
                          "GROUP BY dag;"
                else:  # interval == "uur"
                    sql = "SELECT from_unixtime(t1.`time`) AS 'uur', " \
                          "from_unixtime(t1.`time`) AS 'tijd', from_unixtime(t1.`time`) AS 'tot'," \
                          "t1.`value` '" + key + "' " \
                          "FROM `prognoses` AS t1, `variabel`AS v1  " \
                          "WHERE (v1.`code` = '" + key + "') AND (v1.id = t1.variabel) AND  " \
                          "t1.`time`>= UNIX_TIMESTAMP('" + str(last_moment) + ("') AND " 
                          "t1.`time` < UNIX_TIMESTAMP('") + str(tot) + "');"
                    sql = "SELECT from_unixtime(t1.`time`) tijd,  " \
                          " t1.`value` '" + key + "' " \
                          "FROM `prognoses` AS t1,  `variabel` AS v1  " \
                          "WHERE v1.`code` ='" + key + "' " \
                          "AND v1.id = t1.variabel " \
                          "AND t1.`time` >= UNIX_TIMESTAMP('" + str(last_moment) + "') " \
                          "AND t1.`time` < UNIX_TIMESTAMP('" + str(tot) + "');"
                prog_result = self.db_da.run_select_query(sql)
                """
                prog_table = Table(
                    "prognoses", self.db_da.metadata, autoload_with=self.db_da.engine
                )
                p1 = prog_table.alias("p1")
                # Build the SQLAlchemy query
                """
            query = (
                select(
                    column,
                    func.min(self.db_da.from_unixtime(t1.c.time)).label("vanaf"),
                    func.max(self.db_da.from_unixtime(t1.c.time)).label("tot"),
                    func.sum(t1.c.value).label(key),
                )
                .where(
                    and_(
                        v1.c.code == key,
                        t1.c.variabel == v1.c.id,
                        t1.c.time
                        >= self.db_da.epoch(vanaf),
                        t1.c.time
                        < self.db_da.epoch(tot),
                    )
                )
                .group_by(groupby_str)
            )
                """
                query = (
                    select(
                        self.db_da.hour_start(p1.c.time).label("uur"),
                        func.min(self.db_da.from_unixtime(p1.c.time)).label("tijd"),
                        func.sum(p1.c.value).label(key),
                    )
                    .where(
                        and_(
                            v1.c.code == key,
                            p1.c.variabel == v1.c.id,
                            p1.c.time
                            >= self.db_da.epoch(last_moment),
                            p1.c.time
                            < self.db_da.epoch(tot),
                        )
                    )
                    .group_by("uur")
                )

                with self.db_da.engine.connect() as connection:
                    prog_result = pd.read_sql_query(query, connection)

                prog_result["tijd"] = pd.to_datetime(prog_result["tijd"])
                prog_result["tot"] = prog_result["tijd"]
                if len(prog_result) > 0:
                    prog_result["tijd"] = prog_result.apply(
                        lambda x: self.tijd_at_interval(interval, x["tijd"]), axis=1
                    )
                prog_result.index = pd.to_datetime(prog_result["tijd"])
                if len(prog_result) > 0:
                    self.add_col_df(prog_result, result, key)
                    last_moment = prog_result["tot"].iloc[-1] + datetime.timedelta(
                        hours=1
                    )
            if categorie["sensors"] == "calc":
                function = categorie["function"]
                result = getattr(self, function)(result)
        return result, last_moment

    def get_da_data(
            self,
            key: str,
            vanaf: datetime.datetime,
            tot: datetime.datetime,
            get_interval: str,
            rep_interval: str,
            table: str = "values",
    ) -> pd.DataFrame:
        """
        genereert een dataframe van de data in de da-database
        :param key: code van de data
        :param vanaf:
        :param tot:
        :param get_interval: interval resulting dataframe
        :param rep_interval: aggregatie interval
        :param table: str name of database table: values (default) or prognoses
        :return:  resulting dataframe
        """
        values_table = Table(
            table, self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the values table
        t1 = values_table.alias("t1")
        variabel_table = Table(
            "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the variabel table
        v1 = variabel_table.alias("v1")
        column = self.db_da.hour(t1.c.time).label("uur")
        column2 = func.min(self.db_da.from_unixtime(t1.c.time)).label("tijd")
        if rep_interval == "maand":
            column = self.db_da.month(t1.c.time).label("maand")
            column2 = func.min(self.db_da.month_start(t1.c.time)).label("tijd")
        elif rep_interval == "dag":
            column = self.db_da.day(t1.c.time).label("dag")
            column2 = func.min(self.db_da.day_start(t1.c.time)).label("tijd")
        else:  # rep_interval == "uur"
            column = self.db_da.hour(t1.c.time).label("uur")
            column2 = func.min(self.db_da.hour_start(t1.c.time)).label("tijd")

        if get_interval == "uur":
            column = self.db_da.hour_start(t1.c.time).label("uur")
            column2 = func.min(self.db_da.hour_start(t1.c.time)).label("tijd")
            groupby_str = "uur"
        else:
            groupby_str = rep_interval

        query = (
            select(
                column,
                column2,
                func.min(self.db_da.from_unixtime(t1.c.time)).label("vanaf"),
                func.max(self.db_da.from_unixtime(t1.c.time)).label("tot"),
                func.sum(t1.c.value).label(key),
            )
            .where(
                and_(
                    v1.c.code == key,
                    t1.c.variabel == v1.c.id,
                    t1.c.time
                    >= self.db_da.epoch(vanaf),
                    t1.c.time
                    < self.db_da.epoch(tot),
                )
            )
            .group_by(groupby_str)
        )

        with self.db_da.engine.connect() as connection:
            code_result = pd.read_sql(query, connection)
        # "tijd" already holds the bucket start (column2, month_start/
        # day_start/hour_start) from the query; it used to be overwritten
        # with "vanaf" (the earliest timestamp that happened to have data),
        # which broke the first, partial bucket of any period that does not
        # start exactly on a bucket boundary.
        code_result["tijd"] = pd.to_datetime(code_result["tijd"])
        code_result["tot"] = pd.to_datetime(code_result["tot"])
        code_result.index = code_result["tijd"]
        if table == "prognoses" and key == "profit":
            code_result[key] = legacy_prognosis_revenue(code_result[key])
        return code_result

    def get_columns(
            self, calc_dict, active_period: str, _tot: datetime.datetime | None = None
    ) -> pd.DataFrame:
        if "calc_interval" in calc_dict:
            get_interval = calc_dict["calc_interval"]
        else:
            get_interval = None
        columns = list(calc_dict["series"].keys())
        periode_d = self.periodes[active_period]
        vanaf = periode_d["vanaf"]
        if _tot is None:
            tot = periode_d["tot"]
        else:
            tot = _tot
        rep_interval = periode_d["interval"]
        if get_interval is None:
            interval = rep_interval
        else:
            interval = get_interval
        result = self.generate_df(vanaf, tot, rep_interval, get_interval)
        for key in columns:
            if calc_dict["series"][key]["source"] == "db":
                result[key] = 0.0
                df_column = self.get_da_data(
                    key,
                    vanaf,
                    tot,
                    get_interval=get_interval,
                    rep_interval=rep_interval,
                )
                if len(df_column) == 0:
                    last_moment = vanaf
                else:
                    last_moment = df_column["tot"].iloc[-1] + datetime.timedelta(
                        hours=1
                    )
                self.add_col_df(df_column, result, key)
                if last_moment < tot:
                    if "sensor_type" in calc_dict["series"][key]:
                        sensor_type = calc_dict["series"][key]["sensor_type"]
                    else:
                        sensor_type = "quantity"
                    vanaf_ha = last_moment
                    for sensor in calc_dict["series"][key]["sensors"]:
                        ha_data = self.get_sensor_data(
                            sensor,
                            vanaf_ha,
                            tot,
                            key,
                            interval,
                            sensor_type=sensor_type,
                        )
                        ha_data[key].fillna(0.0)
                        self.add_col_df(ha_data, result, key)
                        if len(ha_data) > 0:
                            last_moment = max(
                                last_moment,
                                ha_data["tot"].iloc[-1] + datetime.timedelta(hours=1),
                            )
                if last_moment < tot:
                    df_column = self.get_da_data(
                        key,
                        last_moment,
                        tot,
                        get_interval=get_interval,
                        rep_interval=rep_interval,
                        table="prognoses",
                    )
                    self.add_col_df(df_column, result, key)
            elif calc_dict["series"][key]["source"] == "prices":
                if key not in result.columns:
                    df_prices = self.get_price_data(vanaf, tot)
                    # Merge on the actual timestamp, not row position: a gap
                    # or a different length between result and df_prices
                    # (missing prices, DST, ...) used to shift every price
                    # onto the wrong row instead of leaving it blank.
                    price_columns = ["tijd", "da_cons", "da_prod"]
                    if "da_bonus" in df_prices.columns:
                        price_columns.append("da_bonus")
                    prices = df_prices.rename(columns={"time": "tijd"})[price_columns]
                    result = result.reset_index(drop=True).merge(
                        prices, on="tijd", how="left"
                    )
                    result.index = result["tijd"]
        return result

    def get_grid_data(self, periode, _vanaf=None, _tot=None, _interval=None, _source="all"):
        period = self.periodes[periode] if periode else {}
        start, end = _vanaf if _vanaf is not None else period["vanaf"], _tot if _tot is not None else period["tot"]
        interval = _interval or period.get("interval", "uur")
        ranges = self._financial_ranges(start, end)
        if not any(use for use, _, _ in ranges):
            return self._get_legacy_grid_data(periode, _vanaf, _tot, _interval, _source)
        basis = self._pin_settlement_report(start, end)
        frames = [self._book_grid(left, right, interval, basis=basis) if active else self._get_legacy_grid_data(
            "", self._local_datetime(left), self._local_datetime(right), interval, _source)
                  for active, left, right in ranges]
        result = pd.concat(frames, ignore_index=True)
        result.attrs["view"] = basis
        return result

    def _get_legacy_grid_data(
            self,
            periode: str,
            _vanaf=None,
            _tot=None,
            _interval: str | None = None,
            _source: str = "all",
    ) -> pd.DataFrame:
        """
        Haalt de grid data: consumptie, productie, cost, profit op de drie tabellen:
        db_da: values tibber data
        aangevuld met
        db_ha: sensoren Home Assistant tot het laatste uur
        voor prognoses (expected):
        db_da: progoses
        :param periode: dus een van alle gedefinieerde perioden: vandaag, gisteren enz.
        :param _vanaf: als != None dan geldt dit als begintijdstip en overrullt
            begintijdstip van periode
        :param _tot: als  != None dan hier het eindtijdstip
        :param _interval: als != None dan hier het gewenste interval
        :param _source: als != None dan hier de source all, da of ha
        :return: een dataframe met de gevraagde griddata
        """

        values_table = Table(
            "values", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the values table
        t1 = values_table.alias("t1")
        variabel_table = Table(
            "variabel", self.db_da.metadata, autoload_with=self.db_da.engine
        )
        # Aliases for the variabel table
        v1 = variabel_table.alias("v1")
        v2 = variabel_table.alias("v2")

        if periode == "":
            vanaf = _vanaf
            tot = _tot
            interval = _interval if _interval else "uur"
        else:
            periode_d = self.periodes[periode]
            vanaf = _vanaf if _vanaf else periode_d["vanaf"]
            tot = _tot if _tot else periode_d["tot"]
            interval = _interval if _interval else periode_d["interval"]

        source = _source
        if interval == "maand":
            column = self.db_da.month(t1.c.time).label("maand")
        elif interval == "dag":
            column = self.db_da.day(t1.c.time).label("dag")
        else:  # interval == "uur"
            column = self.db_da.hour(t1.c.time).label("uur")
        result = None
        if source == "all" or source == "da":
            for cat, label in [
                ("cons", "consumption"),
                ("prod", "production"),
                ("cost", "cost"),
                ("profit", "profit"),
            ]:
                query = (
                    select(
                        column,
                        func.min(self.db_da.from_unixtime(t1.c.time)).label("vanaf"),
                        func.max(self.db_da.from_unixtime(t1.c.time)).label("tot"),
                        func.sum(t1.c.value).label(label),
                    )
                    .where(
                        and_(
                            t1.c.variabel == v1.c.id,
                            v1.c.code == cat,
                            t1.c.time
                            >= self.db_da.epoch(vanaf),
                            t1.c.time
                            < self.db_da.epoch(tot),
                        )
                    )
                    .group_by(interval)
                )

                with self.db_da.engine.connect() as connection:
                    query_str = str(query.compile(connection))
                    logging.debug(f"Query: \n {query_str}")
                    result_cat = pd.read_sql_query(query, connection)
                result_cat.index = result_cat[
                    interval
                ]  # pd.to_datetime(result_cat["vanaf"])
                if result is None:
                    result = result_cat
                else:
                    result[label] = result_cat[label]
        else:
            result = pd.DataFrame(
                columns=[
                    "uur",
                    "vanaf",
                    "tot",
                    "consumption",
                    "production",
                    "cost",
                    "profit",
                ]
            )
            result.index = result["uur"]  # vanaf

        result["datasoort"] = "recorded"
        # aanvullende prijzen ophalen
        if result.shape[0] == 0:
            # datetime.datetime.combine(vanaf, datetime.time(0,0)) - datetime.timedelta(hours=1)
            last_moment = vanaf
        else:
            result["vanaf"] = pd.to_datetime(result["vanaf"])
            result["tot"] = pd.to_datetime(result["tot"])
            last_moment = result["tot"].iloc[-1] + datetime.timedelta(hours=1)
        if last_moment < tot:
            """
            # get the prices:
            query = (
                select(
                    self.db_da.from_unixtime(t1.c.time).label("tijd"),
                    t1.c.value.label("price"),
                )
                .where(
                    and_(
                        v1.c.code == "da",
                        t1.c.variabel == v1.c.id,
                        t1.c.time
                        >= self.db_da.epoch(last_moment),
                        t1.c.time
                        < self.db_da.epoch(tot),
                    )
                )
                .order_by(t1.c.time)
            )

            with self.db_da.engine.connect() as connection:
                query_str = str(query.compile(connection))
                logging.debug(query_str)
                df_prices = pd.read_sql_query(query, connection)
                
            logging.debug(f"Prijzen \n{df_prices.to_string()}\n")
            """

            df_ha = pd.DataFrame()
            if source == "all" or source == "ha":
                # data uit ha ophalen
                count = 0
                for sensor in self.grid_consumption_sensors:
                    if count == 0:
                        df_ha = self.get_sensor_data(
                            sensor, last_moment, tot, "consumption", "uur"
                        )
                        df_ha.index = pd.to_datetime(df_ha["tijd"])
                        df_ha["tijd"] = pd.to_datetime(df_ha["tijd"])
                    else:
                        df_2 = self.get_sensor_data(
                            sensor, last_moment, tot, "consumption", "uur"
                        )
                        df_2.index = pd.to_datetime(df_2["tijd"])
                        df_ha = self.add_col_df(df_2, df_ha, "consumption")
                        # df_cons = df_cons.merge(df_2, on=['tijd']).set_index(['tijd']).sum(axis=1)
                    if len(df_ha) > 0:
                        df_ha["datasoort"] = "recorded"
                    count = +1
                count = 0
                for sensor in self.grid_production_sensors:
                    df_p = self.get_sensor_data(
                        sensor, last_moment, tot, "production", "uur"
                    )
                    df_p.index = pd.to_datetime(df_p["tijd"])
                    if count == 0:
                        df_ha = self.copy_col_df(df_p, df_ha, "production")
                    else:
                        df_ha = self.add_col_df(df_p, df_ha, "production")
                    count = +1
                if len(df_ha) > 0:
                    last_moment = df_ha["tijd"].iloc[-1] + datetime.timedelta(hours=1)
                    df_ha["datasoort"] = "recorded"
                # An empty HA query keeps last_moment as it already was (set
                # above from the "values"/da table): HA's own statistics
                # compile a few minutes after the hour, so an empty result
                # here is routine and must not throw away already-known
                # progress by resetting to the start of the whole period,
                # which duplicated every already-fetched hour below.
                df_prices = self.get_price_data(vanaf, last_moment, interval="1hour")

            if source == "all" or source == "da":
                if last_moment < tot:
                    # get prognose consumption and production:
                    prog_table = Table(
                        "prognoses",
                        self.db_da.metadata,
                        autoload_with=self.db_da.engine,
                    )
                    p1 = prog_table.alias("p1")
                    p2 = prog_table.alias("p2")
                    # Build the SQLAlchemy query
                    query = select(
                        self.db_da.from_unixtime(p1.c.time).label("tijd"),
                        p1.c.value.label("consumption"),
                        p2.c.value.label("production"),
                        literal("expected").label("datasoort"),
                    ).where(
                        and_(
                            p1.c.time == p2.c.time,
                            p1.c.variabel == v1.c.id,
                            v1.c.code == "cons",
                            p2.c.variabel == v2.c.id,
                            v2.c.code == "prod",
                            p1.c.time
                            >= self.db_da.epoch(last_moment),
                            p1.c.time
                            < self.db_da.epoch(tot),
                        )
                    )

                    with self.db_da.engine.connect() as connection:
                        query_str = str(query.compile(connection))
                        logging.debug(f"query get prognose data:\n {query_str}")
                        df_prog = pd.read_sql_query(query, connection)

                    df_prog.index = pd.to_datetime(df_prog["tijd"])
                    df_prog["datasoort"] = "expected"
                    if len(df_prog) > 0:
                        df_prog_prices = self.get_price_data(
                            last_moment, tot, interval=self.interval
                        )
                        if len(df_ha) == 0:
                            df_ha = df_prog
                            df_prices = df_prog_prices
                        else:
                            df_ha = pd.concat([df_ha, df_prog])
                            df_prices.index = pd.to_datetime(df_prices["time"])
                            df_prog_prices.index = pd.to_datetime(
                                df_prog_prices["time"]
                            )
                            df_prices = pd.concat([df_prices, df_prog_prices])

            df_prices.rename(columns={"time": "tijd"}, inplace=True)
            df_prices.index = pd.to_datetime(df_prices["tijd"])
            df_ha = self.copy_col_df(df_prices, df_ha, "da_cons")
            df_ha = self.copy_col_df(df_prices, df_ha, "da_prod")
            if "da_bonus" in df_prices.columns:
                df_ha = self.copy_col_df(df_prices, df_ha, "da_bonus")
            df_ha["tijd"] = pd.to_datetime(df_ha["tijd"])
            df_ha = self.recalc_df_ha(df_ha, interval)

            if len(result) == 0:
                result = df_ha
            else:
                if len(df_ha) > 0:
                    result = pd.concat([result, df_ha])

        result["netto_consumption"] = result["consumption"] - result["production"]
        result["netto_cost"] = result["cost"] - result["profit"]

        return result

    def agg_interval(self, calc_dict: dict, input_df: pd.DataFrame, rep_interval: str):
        if rep_interval != "uur":
            agg_dict = {}
            columns = input_df.columns
            for key in list(calc_dict["series"].keys()):
                if key in columns:
                    agg_dict[key] = calc_dict["series"][key]["agg"]
            result_df = input_df.groupby([rep_interval], as_index=False).agg(agg_dict)
            return result_df
        return input_df

    def clean_df(
            self,
            calc_dict: dict,
            df_input: pd.DataFrame,
            rep_columns: list,
            active_view: str,
            rep_interval: str,
    ):
        """

        :param calc_dict:
        :param df_input:
        :param rep_columns: report columns
        :param active_view: tabel of grafiek
        :param rep_interval:
        :return:
        """
        df_result = pd.DataFrame(columns=[rep_interval] + rep_columns)
        df_result[rep_interval] = df_input[rep_interval]
        for column in rep_columns:
            df_result[column] = df_input[column]
            # df_result[rep_columns] = df_input[rep_columns]
        financial = df_input.attrs.get("settlement_amounts", False)
        df_result = (df_result.groupby(rep_interval, as_index=False).agg(dict.fromkeys(rep_columns, _amount_sum))
                     if financial and rep_interval != "uur" else
                     df_result if financial else self.agg_interval(calc_dict, df_result, rep_interval))
        if active_view == "tabel":
            df_result.loc["Total"] = ([None, *(_amount_sum(df_result[key]) for key in rep_columns)]
                                      if financial else df_result.sum(axis=0, numeric_only=True))
            df_result[rep_interval] = df_result[rep_interval].astype(object)
            df_result.at[df_result.index[-1], rep_interval] = "Totaal"
            for key in calc_dict["series"]:
                agg = calc_dict["series"][key]["agg"]
                if agg == "mean" and key in df_result.columns:
                    mean = df_result[key][:-1].mean()
                    df_result.at[df_result.index[-1], key] = mean
            header_col = []
            if "with header" in calc_dict:
                with_header = calc_dict["with header"]
            else:
                with_header = True
            if with_header:
                header_col = [rep_interval.capitalize()]
                name_col = [""]
            else:
                name_col = [rep_interval.capitalize()]
            dim_col = [""]
            for key in rep_columns:
                if with_header:
                    header_col += [calc_dict["series"][key]["header"]]
                name_col += [calc_dict["series"][key]["name"]]
                dim_col += [calc_dict["series"][key]["dim"]]
            labels = df_result[rep_columns].isna().any(axis=1).map({True: "onvolledig", False: "geschat"})
            if with_header:
                df_result.columns = [header_col, name_col, dim_col]
            else:
                df_result.columns = [name_col, dim_col]
            if financial:
                df_result[("Kwaliteit", *("",) * (df_result.columns.nlevels - 1))] = labels
        return df_result

    def calc_report(
            self,
            active_period: str,
            active_interval: str | None = None,
            active_view: str = "table",
            _tot: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        return

    def calc_saving_consumption(
            self,
            active_period: str,
            active_interval: str | None = None,
            active_view: str = "table",
            _tot: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """
        Berekent besparing op verbruik
        :param active_period: vandaag .... vorig jaar
        :param active_view:
        :param _tot:
        :return: dataframe met
        - kolom 1 netto consumption zonder batterij
        - kolom 2 idem met batterij
        - kolom 3 = kolom 2 - kolom 1
        als active_view = table: met kolomtotalen
        """
        calc_dict = self.saving_consumption_dict
        df = self.get_columns(calc_dict, active_period, _tot)
        df["bruto_cons_zonder"] = df["cons"] - df["prod"] - df["bat_in"] + df["bat_out"]
        df = self.split_column(df, "bruto_cons_zonder", "cons_zonder", "prod_zonder")
        df["netto_cons_zonder"] = df["cons_zonder"] - df["prod_zonder"]
        df["netto_cons_met"] = df["cons"] - df["prod"]
        df["saving"] = df["netto_cons_zonder"] - df["netto_cons_met"]
        df_result = self.clean_df(
            calc_dict,
            df,
            [
                "cons_zonder",
                "prod_zonder",
                "netto_cons_zonder",
                "cons",
                "prod",
                "netto_cons_met",
                "saving",
            ],
            active_view,
            self.periodes[active_period]["interval"],
        )
        return df_result

    @staticmethod
    def split_column(df: pd.DataFrame, col0: str, col1: str, col2: str) -> pd.DataFrame:
        mask1 = df[col0] < 0
        mask2 = df[col0] >= 0
        df[col1] = df[col0].mask(mask1)
        df[col2] = -df[col0].mask(mask2)
        df[col1] = df[col1].fillna(0.0)
        df[col2] = df[col2].fillna(0.0)
        # df.fillna(0.0, inplace=True)
        return df

    def calc_saving_cost(
            self,
            active_period: str,
            active_interval: str | None = None,
            active_view: str = "table",
            _tot: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """
        Berekent besparing op kosten
        :param active_period: vandaag .... vorig jaar
        :param active_interval:
        :param active_view:
        :param _tot:
        :return: dataframe met
        - kolom 1 netto consumption zonder batterij
        - kolom 2 idem met batterij
        - kolom 3 = kolom 2 - kolom 1
        als active_view = table: met kolomtotalen
        """
        calc_dict = self.saving_cost_dict
        df = self.get_columns(calc_dict, active_period, _tot)
        period = self.periodes[active_period]
        if "vanaf" in period and any(active for active, _, _ in self._financial_ranges(
                period["vanaf"], _tot or period["tot"])):
            return self._settlement_saving_cost(df, active_period, active_view, _tot)
        paid_prod_price = (
            df["da_prod"] + df["da_bonus"] if "da_bonus" in df.columns else df["da_prod"]
        )
        df["cost"] = df["cons"] * df["da_cons"]
        df["profit"] = df["prod"] * paid_prod_price
        df["netto_cost"] = df["cost"] - df["profit"]
        df["bruto_cons_zonder"] = df["cons"] - df["prod"] - df["bat_in"] + df["bat_out"]
        df = self.split_column(df, "bruto_cons_zonder", "cons_zonder", "prod_zonder")
        df["cost_zonder"] = df["cons_zonder"] * df["da_cons"]
        df["profit_zonder"] = df["prod_zonder"] * paid_prod_price
        df["netto_cost_zonder"] = df["cost_zonder"] - df["profit_zonder"]
        df["saving"] = df["netto_cost_zonder"] - df["netto_cost"]
        df_result = self.clean_df(
            calc_dict,
            df,
            [
                "cost_zonder",
                "profit_zonder",
                "netto_cost_zonder",
                "cost",
                "profit",
                "netto_cost",
                "saving",
            ],
            active_view,
            self.periodes[active_period]["interval"],
        )
        return df_result

    @staticmethod
    def _legacy_saving_amounts(row):
        """Benoemde legacygrens: energie en tarieven eenmaal via decimale tekst."""
        raw = [row[key] for key in ("cons", "prod", "bat_in", "bat_out", "da_cons", "da_prod")]
        raw.append(row.get("da_bonus", 0))
        if any(pd.isna(value) for value in raw):
            return (None,) * 7
        with localcontext() as context:
            context.prec = 36
            cons, prod, charge, discharge, import_rate, export_rate, bonus = (Decimal(str(value)) for value in raw)
            export_rate += bonus
            cost, profit = cons * import_rate, prod * export_rate
            net = cons - prod - charge + discharge
            ref_cost, ref_profit = max(Decimal("0"), net) * import_rate, max(Decimal("0"), -net) * export_rate
            actual, reference = cost - profit, ref_cost - ref_profit
            return cost, profit, actual, ref_cost, ref_profit, reference, reference - actual

    @staticmethod
    def _reference_meter(at, stop, row):
        from dao.prog.settlement.types import MeterInterval

        with localcontext() as context:
            context.prec = 36
            quantities = [None if pd.isna(row.get(key)) else Decimal(str(row[key]))
                          for key in ("cons", "prod", "bat_in", "bat_out")]
            net = None if any(value is None for value in quantities) else (
                quantities[0] - quantities[1] - quantities[2] + quantities[3])
            return MeterInterval(at, stop, None if net is None else max(Decimal("0"), net).normalize(),
                                 None if net is None else max(Decimal("0"), -net).normalize(), Decimal("0"),
                                 "unknown", "incomplete" if net is None else "estimated",
                                 ("counterfactual:without-battery",))

    def _reference_energy(self, start, end):
        """Dezelfde opgeslagen componenten voor prefix en rapport, zonder nulvulling."""
        if not hasattr(self.db_da, "engine"):
            return {}
        stored = Table("values", self.db_da.metadata, autoload_with=self.db_da.engine)
        variables = Table("variabel", self.db_da.metadata, autoload_with=self.db_da.engine)
        with self.db_da.engine.connect() as connection:
            rows = connection.execute(select(stored.c.time, variables.c.code, stored.c.value).where(
                stored.c.variabel == variables.c.id, variables.c.code.in_(("cons", "prod", "bat_in", "bat_out")),
                stored.c.time >= start, stored.c.time < end,
            )).all()
        energy = {}
        for moment, key, value in rows:
            energy.setdefault(int(moment), {})[key] = value
        return energy

    def _reference_opening(self, service, at, prices):
        """Eigen periodeprefix uit opgeslagen uurenergie, nooit werkelijke geldstaat.

        Alle vier energiecomponenten zijn vereist. Geen zero-filled rapportframe,
        prognose of verdwenen recorderdata als bewijs voor de referentie gebruiken.
        """
        from dao.prog.settlement.engine import settle_series
        from dao.prog.settlement.metering import merge_inputs
        from dao.prog.settlement.types import SettlementState

        start = service._prefix_start(at, prices.snapshots)
        if start == at or not hasattr(self.db_da, "engine"):
            return SettlementState(at, None, None)
        energy = self._reference_energy(start, at)
        meters = []
        for moment in range(start, at, 3600):
            stop = min(at, moment + 3600)
            # Een gedeeltelijk uur is geen proportioneel deel van een uurmeting.
            row = energy.get(moment, {}) if moment % 3600 == 0 and stop - moment == 3600 else {}
            meters.append(self._reference_meter(moment, stop, row))
        history = service.repository.read_prices(service.meter_id, start, at,
                                                  expected_input_revision=prices.input_revision)
        inputs = merge_inputs(meters, tuple(row.spot for row in history.items), history.snapshots,
                              meter_id=service.meter_id)
        return settle_series(inputs, history.snapshots, SettlementState(start, None, None)).end_state

    def _settlement_saving_cost(self, df, period, view, end):
        """Eigen chronologie voor de referentie; uurenergie wordt geen PV-kwartierbewijs."""
        from dao.prog.settlement.engine import settle_series
        from dao.prog.settlement.metering import merge_inputs

        service = self.settlement_service()
        stop_ts = self._settlement_epoch(end or self.periodes[period]["tot"])
        start_ts = self._settlement_epoch(self.periodes[period]["vanaf"])
        pinned = service.repository.read(service.meter_id, start_ts, stop_ts)
        energy = self._reference_energy(start_ts, stop_ts)
        result = df.copy()
        fields = ("cost", "profit", "netto_cost", "cost_zonder", "profit_zonder", "netto_cost_zonder", "saving")
        for field in fields:
            result[field] = pd.Series([None] * len(result), index=result.index, dtype=object)
        reference_state = None
        for position, (index, row) in enumerate(df.iterrows()):
            at = self._settlement_epoch(row["tijd"])
            stop = min(stop_ts, self._settlement_epoch(df.iloc[position + 1]["tijd"]) if position + 1 < len(df)
                       else at + 3600)
            if stop <= at:
                continue
            active = any(use for use, _, _ in self._financial_ranges(at, stop))
            if not active:
                values = self._legacy_saving_amounts(row)
                reference_state = None
            else:
                actual = self._book_grid(at, stop, "uur", basis=pinned)
                cost, profit = _amount_sum(actual.cost), _amount_sum(actual.profit)
                actual_view = actual.attrs["view"]
                revision = actual_view.current_input_revision
                prices = service.repository.read_prices(service.meter_id, at, stop,
                                                         expected_input_revision=revision)
                if reference_state is None or reference_state.asof_ts != at:
                    reference_state = self._reference_opening(service, at, prices)
                with localcontext() as context:
                    context.prec = 36
                    meter = self._reference_meter(at, stop, energy.get(at, {}))
                    inputs = merge_inputs((meter,), tuple(row.spot for row in prices.items), prices.snapshots,
                                          meter_id=service.meter_id)
                    reference = settle_series(inputs, prices.snapshots, reference_state)
                    reference_state = reference.end_state
                    ref_cost = _amount_sum(line.import_cost for line in reference.lines)
                    ref_profit = _amount_sum([_amount_sum((line.base_export_revenue, line.bonus_revenue))
                                              for line in reference.lines])
                    if actual_view.freshness != "current" or not reference.complete:
                        ref_cost = ref_profit = None
                    if ref_profit is not None:
                        ref_profit += sum((credit.credit_total for credit in reference.adjustments), Decimal("0"))
                    net_actual = None if cost is None or profit is None else cost - profit
                    net_reference = None if ref_cost is None or ref_profit is None else ref_cost - ref_profit
                    saving = None if net_actual is None or net_reference is None else net_reference - net_actual
                    values = (cost, profit, net_actual, ref_cost, ref_profit, net_reference, saving)
            for field, value in zip(fields, values, strict=True):
                result.at[index, field] = value
        result.attrs["settlement_amounts"] = True
        return self.clean_df(self.saving_cost_dict, result, list(fields[3:6]) + list(fields[:3]) + ["saving"],
                             view, self.periodes[period]["interval"])

    def calc_saving_co2(
            self,
            active_period: str,
            active_interval: str | None = None,
            active_view: str = "table",
            _tot: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """
        Berekent besparing op kosten
        :param active_period: vandaag .... vorig jaar
        :param active_interval:
        :param active_view:
        :param _tot:
        :return: dataframe met
        - kolom 1 netto consumption zonder batterij
        - kolom 2 idem met batterij
        - kolom 3 = kolom 2 - kolom 1
        als active_view = table: met kolomtotalen
        """
        calc_dict = self.saving_co2_dict
        df = self.get_columns(calc_dict, active_period, _tot)
        df["netto_cons_zonder"] = df["cons"] - df["prod"] - df["bat_in"] + df["bat_out"]
        df["emissie_zonder"] = df["netto_cons_zonder"] * df["co2_intensity"] / 1000
        df["netto_cons"] = df["cons"] - df["prod"]
        df["emissie_met"] = df["netto_cons"] * df["co2_intensity"] / 1000
        df["saving"] = df["emissie_zonder"] - df["emissie_met"]
        df_result = self.clean_df(
            calc_dict,
            df,
            [
                "netto_cons_zonder",
                "emissie_zonder",
                "netto_cons",
                "emissie_met",
                "saving",
            ],
            active_view,
            self.periodes[active_period]["interval"],
        )
        return df_result

    def calc_co2_emission(
            self,
            active_period: str,
            active_interval: str | None = None,
            active_view: str = "table",
            _tot: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """
        Berekent besparing op kosten
        :param active_period: vandaag .... vorig jaar
        :param active_interval:
        :param active_view:
        :param _tot:
        :return: dataframe met
        """
        calc_dict = self.calc_co2_emission_dict
        df = self.get_columns(calc_dict, active_period, _tot)
        df["netto_cons"] = df["cons"] - df["prod"]
        df["emissie"] = df["netto_cons"] * df["co2_intensity"] / 1000
        df_result = self.clean_df(
            calc_dict,
            df,
            ["cons", "prod", "netto_cons", "co2_intensity", "emissie"],
            active_view,
            self.periodes[active_period]["interval"],
        )
        return df_result

    @staticmethod
    def get_last_day_month(input_dt: datetime):
        # Get the last day of the month from a given datetime
        res = calendar.monthrange(input_dt.year, input_dt.month)
        return res[1]

    def calc_grid_columns(self, report_df, active_interval, active_view):
        if "bonus_kind" in report_df and (report_df.bonus_kind == "calculated").any():
            return self._settlement_grid_columns(report_df, active_interval, active_view)
        first_col = active_interval.capitalize()
        # if active_subject == "verbruik":
        #    columns.extend(["Verbruik", "Productie", "Netto"])
        #    columns = [columns]
        #    columns.append(["", "kWh", "kWh", "kWh"])
        # else:  #kosten
        columns = [
            first_col,
            "Verbruik",
            "Productie",
            "Netto verbr.",
            "Kosten",
            "Opbrengst",
            "Netto kosten",
        ]
        # columns.extend(ext_columns)
        if len(report_df.index) == 0:
            return pd.DataFrame(columns=columns)
        rows = []
        for row in report_df.itertuples():
            if pd.isnull(row.vanaf):
                continue
            if active_interval == "uur":
                tijd_str = str(row.vanaf)[10:16]
            elif active_interval == "dag":
                tijd_str = str(row.vanaf)[0:10]
            else:
                tijd_str = str(row.vanaf)[0:7]  # jaar maand
            col_1 = row.consumption
            col_2 = row.production
            col_3 = col_1 - col_2
            col_4 = row.cost
            col_5 = row.profit
            col_6 = col_4 - col_5
            rows.append((tijd_str, col_1, col_2, col_3, col_4, col_5, col_6))
        fi_df = pd.DataFrame(rows, columns=columns)

        # , "Tarief verbr.", "Tarief prod."
        # , "Tarief verbr.":'mean', "Tarief prod.":"mean"
        # fi_df.set_index([columns[0][0]])
        if active_interval != "uur":
            fi_df = fi_df.groupby([first_col], as_index=False).agg(
                {
                    "Verbruik": "sum",
                    "Productie": "sum",
                    "Netto verbr.": "sum",
                    "Kosten": "sum",
                    "Opbrengst": "sum",
                    "Netto kosten": "sum",
                }
            )
        fi_df["Tarief verbr."] = fi_df.apply(
            lambda rw: rw.Kosten / rw.Verbruik if rw.Verbruik != 0.0 else rw.Verbruik,
            axis=1,
        )
        fi_df["Tarief prod."] = fi_df.apply(
            lambda rw: (
                rw.Opbrengst / rw.Productie if rw.Productie != 0.0 else rw.Productie
            ),
            axis=1,
        )
        if active_view == "tabel":
            fi_df.loc["Total"] = fi_df.sum(axis=0, numeric_only=True)
            fi_df.at[fi_df.index[-1], first_col] = "Totaal"
            row = fi_df.iloc[-1]
            if row.Verbruik == 0:
                tarief = 0
            else:
                tarief = row.Kosten / row.Verbruik
            fi_df.at[fi_df.index[-1], "Tarief verbr."] = tarief
            if row.Productie == 0:
                tarief = 0
            else:
                tarief = row.Opbrengst / row.Productie
            fi_df.at[fi_df.index[-1], "Tarief prod."] = tarief
            # value = fi_df.iloc[-1][7]
            # fi_df.at[fi_df.index[-1], "Tarief"] = value / (len(fi_df.index)-1)

            # fi_df.loc[fi_df.shape[0]] = ["Totaal", col_1_tot, col_2_tot, col_3_tot, col_4_tot,
            #                         col_5_tot, col_6_tot,
            #                         col_7_tot / count_tot]
            columns = fi_df.columns.values.tolist()
            # columns.append(["", "kWh", "kWh", "kWh", "eur", "eur", "eur", "eur/kWh",  "eur/kWh"])
            # columns = [columns,
            fi_df.columns = [
                columns,
                ["", "kWh", "kWh", "kWh", "eur", "eur", "eur", "eur/kWh", "eur/kWh"],
            ]
        fi_df = fi_df.round(3)
        return fi_df

    def _settlement_grid_columns(self, frame, interval, view):
        """Bedragcompatibiliteit zonder Decimal→float of onbekend→nul in totalen."""
        first = interval.capitalize()
        fields = {"consumption": "Verbruik", "production": "Productie", "netto_consumption": "Netto verbr.",
                  "cost": "Kosten", "profit": "Opbrengst", "netto_cost": "Netto kosten"}
        result = frame[list(fields)].rename(columns=fields).copy()
        result.insert(0, first, frame[interval].astype(str))
        result["Kwaliteit"] = frame.quality_label.fillna("geschat")

        def quality(labels):
            return "onvolledig" if "onvolledig" in set(labels) else "geschat" if "geschat" in set(labels) else "berekend"

        result = result.groupby(first, as_index=False).agg({**dict.fromkeys(fields.values(), _amount_sum),
                                                          "Kwaliteit": quality})
        if view == "tabel":
            result.loc["Total"] = {first: "Totaal", **{key: _amount_sum(result[key]) for key in fields.values()},
                                    "Kwaliteit": quality(result.Kwaliteit)}
        with localcontext() as context:
            context.prec = 36
            for amount, energy, name in (("Kosten", "Verbruik", "Tarief verbr."),
                                          ("Opbrengst", "Productie", "Tarief prod.")):
                result[name] = [None if pd.isna(value) or pd.isna(kwh) else value / kwh if kwh else Decimal("0")
                                for value, kwh in zip(result[amount], result[energy], strict=True)]
        columns = [first, *fields.values(), "Tarief verbr.", "Tarief prod.", "Kwaliteit"]
        result = result[columns]
        if view == "tabel":
            result.columns = [columns, ["", "kWh", "kWh", "kWh", "eur", "eur", "eur", "eur/kWh", "eur/kWh", ""]]
        return result

    def calc_balance_columns(self, report_df, active_interval, active_view):
        first_col = active_interval.capitalize()
        # report_df = report_df.drop('vanaf', axis=1)
        report_df.style.format("{:.3f}")
        report_df = report_df.drop("tijd", axis=1)
        report_df = report_df.drop("tot", axis=1)
        report_df = report_df.drop("datasoort", axis=1)
        if "datasoort" in report_df.columns:
            report_df = report_df.drop("datasoort", axis=1)
        key_columns = report_df.columns.values.tolist()[1:]
        columns_1 = [first_col]
        columns_2 = [""]
        for key in key_columns:
            columns_1 = columns_1 + [self.energy_balance_dict[key]["name"]]
            columns_2 = columns_2 + [self.energy_balance_dict[key]["dim"]]
        if active_view == "tabel":
            report_df.loc["Total"] = report_df.sum(axis=0, numeric_only=True)
            report_df[active_interval] = report_df[active_interval].astype(object)
            report_df.at["Total", active_interval] = "Totaal"
            columns = [columns_1, columns_2]
            report_df.columns = columns
        else:
            report_df.columns = columns_1

        return report_df

    """
    def calc_co2_columns(self, report_df, active_interval, active_view):
        first_col = active_interval.capitalize()
        # report_df = report_df.drop('vanaf', axis=1)
        report_df.style.format("{:.3f}")
        report_df = report_df.drop("tijd", axis=1)
        # report_df =  report_df.drop('datasoort', axis=1)
        key_columns = report_df.columns.values.tolist()[1:]
        columns_1 = [first_col]
        columns_2 = [""]
        for key in key_columns:
            columns_1 = columns_1 + [self.co2_dict[key]["name"]]
            columns_2 = columns_2 + [self.co2_dict[key]["dim"]]
        if active_interval != "uur":
            report_df["uur"] = pd.to_datetime(report_df["uur"])
            report_df["uur"] = report_df.apply(
                lambda x: self.tijd_at_interval(
                    active_interval, x["uur"], as_index=True
                ),
                axis=1,
            )
            report_df = report_df.groupby(["uur"], as_index=False).agg(
                {
                    "cons": "sum",
                    "prod": "sum",
                    "netto_cons": "sum",
                    "co2_intensity": "mean",
                    "emissie": "sum",
                }
            )

        if active_view == "tabel":
            report_df.loc["Total"] = report_df.sum(axis=0, numeric_only=True)
            report_df.at["Total", "uur"] = "Totaal"
            row = report_df.iloc[-1]
            if row.netto_cons == 0:
                co2_intensity = 0
            else:
                co2_intensity = row.emissie * 1000 / row.netto_cons
            report_df.at[report_df.index[-1], "co2_intensity"] = co2_intensity

            columns = [columns_1, columns_2]
            report_df.columns = columns
        else:
            report_df.columns = columns_1

        return report_df
    """

    #  ------------------------------------------------
    def check_baseload_sensors(self) -> list:
        """Warn when a modelled device has no meter to subtract it with.

        The baseload is the measured total minus everything the optimizer
        schedules itself. A device that is modelled but not metered therefore
        stays inside the baseload *and* is added again by the optimizer, which
        silently double counts it, every hour of every day. The reverse, a
        meter without a modelled device, subtracts consumption that is never
        added back and quietly underforecasts.

        Both are easy to end up with and impossible to see in the output, so
        they are reported explicitly.
        """
        checks = [
            (
                "boiler",
                bool(getattr(self.config.boiler, "boiler_present", False)),
                self.boiler_consumption_sensors,
                "entities boiler consumption",
            ),
            (
                "warmtepomp",
                bool(getattr(self.config.heating, "heater_present", False)),
                self.wp_consumption_sensors,
                "entities wp consumption",
            ),
            (
                "elektrische auto",
                bool(self.config.electric_vehicle),
                self.ev_consumption_sensors,
                "entities ev consumption",
            ),
            (
                "apparatuur",
                bool(self.config.machines),
                self.machine_consumption_sensors,
                "entities machine consumption",
            ),
            (
                "batterij",
                bool(self.config.battery),
                self.battery_consumption_sensors,
                "entities battery consumption",
            ),
            (
                "zonnepanelen",
                bool(self.config.solar),
                self.solar_production_ac_sensors,
                "entities solar production ac",
            ),
        ]
        problems = []
        for name, modelled, sensors, setting in checks:
            if modelled and not sensors:
                message = (
                    f"{name} wordt wel meegerekend door de optimalisatie maar er is "
                    f"geen '{setting}' ingesteld; het verbruik zit daardoor zowel in "
                    f"de baseload als in de planning en wordt dubbel geteld"
                )
                problems.append(message)
                logging.warning(f"Baseload: {message}")
            elif sensors and not modelled:
                message = (
                    f"er is een '{setting}' ingesteld maar {name} wordt niet door de "
                    f"optimalisatie ingepland; dat verbruik wordt van de baseload "
                    f"afgetrokken en nergens weer opgeteld"
                )
                problems.append(message)
                logging.warning(f"Baseload: {message}")
        problems.extend(self._check_sensor_kinds())
        if not problems:
            logging.info("Baseload: de meetpunten voor de apparaten zijn consistent")
        return problems

    def _baseload_sensor_groups(self) -> dict:
        """The configured sensors per component group of the history reader."""
        return {
            "grid_in": list(self.grid_consumption_sensors or []),
            "grid_out": list(self.grid_production_sensors or []),
            "pv_ac": list(self.solar_production_ac_sensors or []),
            "ev": list(self.ev_consumption_sensors or []),
            "wp": list(self.wp_consumption_sensors or []),
            "boiler": list(self.boiler_consumption_sensors or []),
            "machines": list(self.machine_consumption_sensors or []),
            "bat_in": list(self.battery_consumption_sensors or []),
            "bat_out": list(self.battery_production_sensors or []),
        }

    def _check_sensor_kinds(self) -> list:
        """Every configured meter must exist and be an energy or power sensor.

        A state-of-charge or temperature sensor in one of the lists used to
        contribute nothing, silently. An entity that no longer exists is
        not fatal to the reader any more -- it degrades its whole group to
        NaN -- which is exactly why it has to be named here. Reported per
        sensor so the operator can see which entry to fix.
        """
        db_ha = getattr(self, "db_ha", None)
        if db_ha is None:
            return []
        from dao.forecast.history import HistoryReader, UnsupportedSensorError

        reader = HistoryReader(db_ha, getattr(self, "time_zone", None) or "UTC")
        problems = []
        for sensors in self._baseload_sensor_groups().values():
            for sensor in sensors:
                try:
                    if reader.sensor_meta([sensor]):
                        continue
                    reason = (
                        f"{sensor}: geen statistieken gevonden in de Home "
                        f"Assistant database"
                    )
                except UnsupportedSensorError as ex:
                    reason = str(ex)
                message = f"{reason}; deze meter telt niet mee in de baseload"
                problems.append(message)
                logging.warning(f"Baseload: {message}")
        return problems

    def calc_baseload_frame(self) -> pd.DataFrame:
        """Measured baseload per hour over the whole calculation period.

        baseload = grid in - grid out + pv ac - ev - heat pump - boiler
                   - machines - battery in + battery out

        Read through the history reader, which takes the right statistics
        column for each sensor kind and leaves hours without a measurement
        as NaN instead of zero.
        """
        from zoneinfo import ZoneInfo

        from dao.forecast.history import (
            HistoryReader,
            baseload_from_components,
            component_caps,
        )

        zone = ZoneInfo(self.time_zone)
        calc_periode = self.config.baseload_calc_periode
        now = datetime.datetime.now(tz=zone)
        tot = now.replace(hour=0, minute=0, second=0, microsecond=0)
        vanaf = tot - datetime.timedelta(days=calc_periode)

        reader = HistoryReader(self.db_ha, self.time_zone)
        frame = reader.read_components(
            self._baseload_sensor_groups(), vanaf, tot, component_caps(self.config)
        )
        base = baseload_from_components(frame)
        result = pd.DataFrame(
            {
                "tijd": base.index,
                "weekdag": base.index.weekday,
                "uur": base.index.hour,
                "baseload": base.values,
            }
        )
        logging.debug(f"Baseload berekening per uur:\n {result.to_string()}\n")
        return result

    # ------------------------------------------------
    def get_field_data(self, field: str, periode: str, tot=None, dict=None):
        period = self.periodes[periode]
        if dict is None:
            dict = self.energy_balance_dict
        if field not in dict:
            result = None
            return result
        categorie = dict[field]
        df = self.db_da.get_column_data(
            "values",
            field,
            start=period["vanaf"],
            end=period["tot"] if tot is None else tot,
        )
        df.index = pd.to_datetime(df["time"])
        df = df.rename(columns={"value": field})
        df["datasoort"] = "recorded"

        df_ha_result = pd.DataFrame()
        if len(df) > 0:
            last_moment = df["time"].iloc[-1] + datetime.timedelta(hours=1)
        else:
            last_moment = self.periodes[periode]["vanaf"]
        if last_moment < self.periodes[periode]["tot"]:
            count = 0
            for sensor in categorie["sensors"]:
                df_ha = self.get_sensor_data(sensor, last_moment, period["tot"], field)
                df_ha.index = pd.to_datetime(df_ha["tijd"])
                if count == 0:
                    df_ha_result = df_ha
                else:
                    df_ha_result = self.add_col_df(df_ha, df_ha_result, field)
                count += 1
            df_ha_result["datasoort"] = "recorded"
            df_ha_result = df_ha_result.rename(columns={"tijd": "time"})
            if len(df_ha_result) > 0:
                last_moment = df_ha_result["time"].iloc[-1] + datetime.timedelta(
                    hours=1
                )
                df_ha_result["time"] = df_ha_result["time"].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M")
                )

        if last_moment < self.periodes[periode]["tot"]:
            df_prog = self.db_da.get_column_data(
                "prognoses", field, start=last_moment, end=period["tot"]
            )
            if len(df_prog) > 0:
                last_moment = df_prog["time"].iloc[-1]
            df_prog.index = pd.to_datetime(df_prog["time"])
            df_prog = df_prog.rename(columns={"value": field})
            df_prog["datasoort"] = "expected"
            if len(df_ha_result) == 0:
                df_uur = df_prog
            elif len(df_prog) == 0:
                df_uur = df_ha_result
            else:
                df_uur = pd.concat([df_ha_result, df_prog])
        else:
            df_uur = df_ha_result
        if len(df_uur) > 0:
            if len(df) == 0:
                df = df_uur
            else:
                df = pd.concat([df, df_uur])
        return df, last_moment

    def get_price_data(self, start, end, interval: str = "1hour"):
        pricing = getattr(self, "prices_options", None)
        cfg = getattr(pricing, "settlement", None)
        if cfg is None and not getattr(self, "_legacy_settlement_warned", False):
            logging.warning("Geen settlementconfiguratie: legacyraming, geen bewijs van 2027-gereedheid")
            self._legacy_settlement_warned = True
        if cfg is None or cfg.mode != "active":
            frame = self._legacy_price_data(start, end, interval)
            frame["bonus_kind"] = "potential"
            return frame
        if not getattr(cfg, "contracts", None):
            raise ValueError("Actieve afrekening vereist expliciete contractdekking")
        if end is None:
            last = self.db_da.get_time_border_record("da")
            if last is None:
                raise ValueError("Geen basistariefdekking")
            end = last + datetime.timedelta(hours=1)
        rows = []
        for active, lower, upper in self._financial_ranges(start, end):
            if not active:
                # Historische legacydekking gebruikt zijn eigen bestaande gedateerde adapter.
                frame = self._legacy_price_data(self._local_datetime(lower), self._local_datetime(upper), interval,
                                                 historical=True)
                rows.extend(frame.to_dict("records"))
                continue
            service = self.settlement_service()
            raw = service.price_adapter(lower // 3600 * 3600, (upper + 3599) // 3600 * 3600)
            for at in range(lower, upper, 900):
                stop = min(upper, (at // 900 + 1) * 900)
                try:
                    price = service._quotes(at, stop, service.snapshots, raw)[0]
                    amounts = (price.spot_ex_vat, price.import_rate, price.export_rate, price.bonus_rate)
                except ValueError:
                    amounts = (None,) * 4
                rows.append(dict(zip(("da_ex", "da_cons", "da_prod", "da_bonus"), amounts, strict=True),
                                 time=self._local_datetime(at), datasoort="potential"))
        frame = pd.DataFrame(rows, columns=["time", "da_ex", "da_cons", "da_prod", "da_bonus", "datasoort"])
        if interval == "1hour" and not frame.empty:
            frame["time"] = pd.to_datetime(frame.time).dt.floor("h")
            def mean_rate(values):
                with localcontext() as context:
                    context.prec = 36
                    total = _amount_sum(values)
                    return None if total is None else total / len(values)
            frame = frame.groupby("time", as_index=False).agg({**{name: mean_rate for name in
                ("da_ex", "da_cons", "da_prod", "da_bonus")}, "datasoort": "first"})
        frame["bonus_kind"] = "potential"
        return frame

    def _legacy_price_data(self, start, end, interval="1hour", *, historical=False):
        settlement = getattr(self.prices_options, "settlement", None)
        if settlement is not None and settlement.mode == "active" and not historical:
            raise ValueError("Actieve afrekening mag de legacy-prijsfunctie niet gebruiken")
        if interval == "1hour":
            agg_func = "avg"
        else:
            agg_func = None
        df_da = self.db_da.get_column_data(
            "values", "da", start=start, end=end, agg_func=agg_func
        )
        old_dagstr = ""
        taxes_l = 0
        taxes_t = 0
        ol_l = 0
        ol_t = 0
        btw_l = 0
        btw_t = 0
        multiplier_l = 1
        multiplier_t = 1
        columns = ["time", "da_ex", "da_cons", "da_prod", "da_bonus", "datasoort"]
        salderen = self.prices_options.tax_refund if self.prices_options else True
        bonus_cfg = getattr(self, "production_bonus", None)
        bonus_on = bonus_cfg is not None and bonus_cfg.enabled
        cap_exhausted = False
        if bonus_on:
            # De limiet wordt eenmaal per aanroep beoordeeld, op het beginmoment.
            remaining = self.production_bonus_remaining_kwh(
                start if start is not None else datetime.datetime.now()
            )
            cap_exhausted = remaining is not None and remaining <= 0
        rows = []
        for row in df_da.itertuples():
            if pd.isnull(row.time):
                continue
            dag_str = row.time[:10]
            if dag_str != old_dagstr:
                ol_l = get_value_from_dict(dag_str, self.ol_l_def)
                ol_t = get_value_from_dict(dag_str, self.ol_t_def)
                taxes_l = get_value_from_dict(dag_str, self.taxes_l_def)
                taxes_t = get_value_from_dict(dag_str, self.taxes_t_def)
                btw_l = get_value_from_dict(dag_str, self.btw_l_def)
                btw_t = get_value_from_dict(dag_str, self.btw_t_def)
                multiplier_l = get_value_from_dict(dag_str, self.multiplier_l_def)
                multiplier_t = get_value_from_dict(dag_str, self.multiplier_t_def)
                old_dagstr = dag_str
            da_cons = (row.value * multiplier_l + taxes_l + ol_l) * (1 + btw_l / 100)
            if salderen:
                da_prod = (row.value * multiplier_t + taxes_t + ol_t) * (
                        1 + btw_t / 100
                )
            else:
                # De exportspotfactor geldt ook zonder belastingteruggave.
                da_prod = (row.value * multiplier_t + ol_t) * (1 + btw_t / 100)
            moment = datetime.datetime.strptime(row.time, "%Y-%m-%d %H:%M")
            if bonus_on and not cap_exhausted:
                da_bonus = production_bonus.bonus_per_kwh(bonus_cfg, moment, row.value, btw_t)
            else:
                da_bonus = 0.0
            rows.append(
                (
                    moment,
                    row.value,
                    da_cons,
                    da_prod,
                    da_bonus,
                    "expected",
                )
            )
        return pd.DataFrame(rows, columns=columns)

    def calc_solar_data(self, device, day: datetime.date, active_view: str):
        start = datetime.datetime(day.year, day.month, day.day)
        end = start + datetime.timedelta(days=1)
        begin = dati = datetime.datetime(start.year, start.month, start.day, hour=0)
        hour_int = 0
        rows = []
        while dati < end:
            hour_int += 1
            hour_str = dati.strftime("%H:%M")
            rows.append((hour_str, dati))
            dati = begin + datetime.timedelta(hours=hour_int)
        result = pd.DataFrame(rows, columns=["uur", "tijd"])
        result.index = pd.to_datetime(result["tijd"])

        # prognose straling
        rad_prog = self.get_da_data("gr", start, end, "uur", "uur", "prognoses")
        result["prognose_straling"] = rad_prog["gr"]

        # gemeten straling
        rad_real = self.get_da_data("gr", start, end, "uur", "uur", "values")
        result["gemeten_straling"] = rad_real["gr"]

        # gemeten productie
        sensors = device.entities_sensors
        count = 0
        for sensor in sensors:
            df_sensor = self.get_sensor_data(sensor, start, end, "gemeten")
            if count == 0:
                df_solar = df_sensor
            else:
                self.add_col_df(df_sensor, df_solar, "gemeten")
            count += 1
        result["gemeten_prod"] = pd.NA
        self.add_col_df(df_solar, result, "gemeten", "gemeten_prod")

        # voorspelling DAO
        from dao.forecast.weather.schema import jcm2h_to_wm2

        straling = result["gemeten_straling"].fillna(result["prognose_straling"])
        temp_real = self.get_da_data("temp", start, end, "uur", "uur", "values")
        winds_real = self.get_da_data("winds", start, end, "uur", "uur", "values")
        temp = (
            pd.to_numeric(temp_real["temp"], errors="coerce").reindex(result.index)
            if "temp" in temp_real.columns
            else pd.Series(float("nan"), index=result.index)
        ).fillna(15.0)
        wind = (
            pd.to_numeric(winds_real["winds"], errors="coerce").reindex(result.index)
            if "winds" in winds_real.columns
            else pd.Series(float("nan"), index=result.index)
        ).fillna(3.0)

        weather = pd.DataFrame(
            index=localize_hour_buckets(result.index, self.time_zone)
        )
        weather["ghi"] = jcm2h_to_wm2(pd.to_numeric(straling, errors="coerce").to_numpy())
        weather["dni"] = float("nan")
        weather["dhi"] = float("nan")
        weather["temp"] = temp.to_numpy()
        weather["wind"] = wind.to_numpy()

        pred_dao = pd.Series(pd.NA, index=result.index, dtype="object")
        if len(weather) > 0:
            predictions = self.pv_service().forecast_from_weather(device, weather)
            valid = straling.notna()
            pred_dao.loc[valid] = predictions.to_numpy()[valid.to_numpy()]
        result["prognose_dao"] = pred_dao

        # voorspelling ML

        # solar_predictor = SolarPredictor()
        # solar_prog = solar_predictor.predict_solar_device(device, start, end)
        solar_prog = self.pv_service().forecast(
            device, start, end, "1hour", model="ml"
        )
        if "date_time" in solar_prog.columns:
            solar_prog["tijd"] = solar_prog["date_time"].dt.tz_localize(None)
        solar_prog.index = solar_prog["tijd"]
        result["prognose_ml"] = pd.NA
        self.add_col_df(solar_prog, result, "prediction", "prognose_ml")
        # result["prognose_ml"] = solar_prog["prediction"]

        result.drop(columns=["tijd"], inplace=True)
        result = result[
            [
                "uur",
                "gemeten_straling",
                "prognose_straling",
                "gemeten_prod",
                "prognose_dao",
                "prognose_ml",
            ]
        ]
        if active_view == "tabel":
            r2_radiation = calc_r2(
                result["gemeten_straling"], result["prognose_straling"]
            )
            r2_dao = calc_r2(result["gemeten_prod"], result["prognose_dao"])
            r2_ml = calc_r2(result["gemeten_prod"], result["prognose_ml"])
            result.loc["Total"] = result.sum(axis=0)
            result.at[result.index[-1], "uur"] = "Totaal"
            result.columns = [
                ["", "Straling", "Straling", "Productie", "Productie", "Productie"],
                [
                    "uur",
                    "gemeten",
                    "prognose",
                    "gemeten",
                    "prognose dao",
                    "prognose ml",
                ],
                ["", "J/cm2", "J/cm2", "kWh", "kWh", "kWh"],
            ]
            row = ["R²", "", r2_radiation, "", r2_dao, r2_ml]
            result.loc[result.shape[0]] = row
            result.fillna("", inplace=True)

        return result

    def get_heatpump_run_hours(self, entity):
        if entity is None or not self.wp_consumption_sensors:
            return -1
        now = datetime.datetime.now()
        vanaf = datetime.datetime(
            now.year, now.month, now.day, now.hour
        ) - datetime.timedelta(hours=5)
        sensor_data = None
        if entity is not None:
            sensor_data = self.get_sensor_sum(
                sensor_list=[entity], vanaf=vanaf, tot=now, col_name="hp"
            )
        if sensor_data is None or len(sensor_data) == 0:
            sensor_data = self.get_sensor_sum(
                sensor_list=self.wp_consumption_sensors,
                vanaf=vanaf,
                tot=now,
                col_name="hp",
            )
        if len(sensor_data) == 0:
            return -2
        count = 0
        df = sensor_data.iloc[::-1]
        for row in df.itertuples():
            if row.hp > 0:
                count += 1
            else:
                break
        return count

    def get_soc_data(
            self, field: str, start: datetime.datetime, end: datetime.datetime
    ) -> pd.DataFrame:
        df = self.db_da.get_column_data("prognoses", field, start=start, end=end)
        return df

    def get_pv_prognose(
            self, field: str, vanaf: datetime.datetime, tot: datetime.datetime
    ) -> pd.DataFrame:
        df_result = pd.DataFrame(columns=["tijd", "prediction"])
        if field == "pv_ac":
            solar_num = len(self.solar)
            for s in range(solar_num):
                solar_option = self.solar[s]
                df_data = self.calc_solar_predictions(
                    solar_option, vanaf, tot, interval="1hour"
                )
                if s == 0:
                    df_result = df_data
                else:
                    df_result["prediction"] = (
                            df_result["prediction"] + df_data["prediction"]
                    )
            df_result = df_result.rename(
                columns={"prediction": field, "date_time": "time"}
            )
        else:  # pv_dc
            battery_options = self.config.battery
            B = len(battery_options)
            count = 0
            for b in range(B):
                solar_options = battery_options[b].solar
                solar_num = len(solar_options)
                for s in range(solar_num):
                    solar_option = solar_options[s]
                    df_data = self.calc_solar_predictions(
                        solar_option, vanaf, tot, interval="1hour"
                    )
                    if count == 0:
                        df_result = df_data
                    else:
                        df_result["prediction"] = (
                                df_result["prediction"] + df_data["prediction"]
                        )
                    count += 1
            df_result = df_result.rename(
                columns={"prediction": field, "date_time": "time"}
            )
        # Behoud de bekende tijdidentiteit van PVService tot aan de API-weergave.
        df_result["time"] = df_result["tijd"]
        df_result["datasoort"] = "expected"
        df_result.drop("tijd", axis=1, inplace=True)
        df_result.index = df_result["time"]
        return df_result

    def get_api_data(self, field: str, _periode: str, cumulate: bool = False, *, generation=None):
        periode = _periode.replace("_", " ")
        if periode not in self.periodes.keys():
            result = f'{{"message": "Failed", "reason": "Periode: \'{_periode}\' is not allowed"}}'
            return result

        if generation is not None and (type(generation) is not int or not 0 <= generation < 2**63 - 1):
            raise ValueError("Generatie moet een niet-negatief integer zijn")
        if field == "settlement":
            from dao.prog.settlement.tariffs import _canonical_default
            frame = self.get_settlement_data(self.periodes[periode]["vanaf"], self.periodes[periode]["tot"],
                                             generation=generation)
            view = frame.attrs["view"]
            rows = [{key: None if isinstance(value, float) and math.isnan(value) else value
                     for key, value in row.items()} for row in frame.to_dict("records")]
            return json.dumps({"message": "Success", "generation": view.generation,
                               "input_revision": view.input_revision, "freshness": view.freshness,
                               "complete": view.complete, "captured_ts": view.captured_ts,
                               "source_watermark_ts": view.source_watermark_ts,
                               "known_subtotal": view.known_subtotal, "net_cost": view.net_cost, "data": rows},
                              default=_canonical_default, allow_nan=False)

        grid_fields = [
            "consumption",
            "production",
            "netto_consumption",
            "cost",
            "profit",
            "netto_cost",
        ]
        tot = self.periodes[periode]["tot"]
        df = pd.DataFrame()
        if field in ["grid"] + grid_fields:  # grid data
            df_grid = self.get_grid_data(periode, _tot=tot)
            df_grid["time"] = df_grid["vanaf"]

            def cumulative_grid(values):
                if "view" in df_grid.attrs:
                    # Boek/legacybedragen exact combineren; een onbekende prefix blijft onbekend.
                    return list(accumulate(values, lambda total, value: _amount_sum((total, value))))
                return values.cumsum()

            if field in grid_fields:
                df = df_grid[["time", field, "datasoort"]].copy()
                if cumulate:
                    df[field] = cumulative_grid(df_grid[field])
                df.rename({field: "value"}, axis=1, inplace=True)
            if field == "grid":
                df = df_grid[["time", "datasoort"] + grid_fields].copy()
                if cumulate:
                    for field in grid_fields:
                        df[field] = cumulative_grid(df[field])
        elif field == "da":
            df = self.get_price_data(
                self.periodes[periode]["vanaf"],
                self.periodes[periode]["tot"],
                self.interval,
            )
        elif field[0:3] == "soc":
            df = self.get_soc_data(field, self.periodes[periode]["vanaf"], tot)
            df["time"] = pd.to_datetime(df["time"])
        elif field in ["pv_ac", "pv_dc"] or field in self.energy_balance_dict:
            # df, last_moment = self.get_field_data(field, periode, dict=self.several_dict)
            if field == "pv_dc":
                dict = self.several_dict
            else:
                dict = self.energy_balance_dict
            df_balance, last_moment = self.get_energy_balance_data(
                periode, field=field, _tot=tot, col_dict=dict
            )
            df_balance["time"] = df_balance["tijd"]
            df = df_balance[["time", field, "datasoort"]].copy()
            if periode == "vandaag en morgen" and field in ["pv_ac", "pv_dc"]:
                vanaf = last_moment
                tot = last_moment + datetime.timedelta(days=2)
                df_pv = self.get_pv_prognose(field, vanaf, tot)
                if len(df_pv) > 0:
                    df = pd.concat([df, df_pv])
            if cumulate:
                df[field] = df[field].cumsum()
            df.rename({field: "value"}, axis=1, inplace=True)
        else:
            result = (
                f'{{"message":"Failed", "reason": "field: \'{field}\' is not allowed"}}'
            )
            return result

        moments = df["time"].map(pd.Timestamp)
        # Alleen naïeve legacylabels lokaliseren; de boekidentiteit nooit opnieuw raden.
        df["time_ts"] = moments.map(lambda moment: moment.tz_convert(self.time_zone) if moment.tzinfo is not None
                                  else moment.tz_localize(self.time_zone, ambiguous=False, nonexistent="shift_forward"))
        df["time"] = df["time_ts"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
        df.rename(columns={"datasoort": "datatype"}, inplace=True)
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        data_json = df.to_json(orient="records")
        result = '{ "message":"Success", "data": ' + data_json + " }"
        return result

    def make_graph(self, df, period, _options=None, _title: str | None = None):
        if _options:
            options = _options
        else:
            _gs = self.config.graphics.style
            options = {
                "title": "Verbruik en kosten",
                "style": _gs,
                "graphs": [
                    {
                        "vaxis": [{"title": "kWh"}, {"title": "euro"}],
                        "align_zeros": "True",
                        "series": [
                            {
                                "column": "Verbruik",
                                "title": "Verbruik",
                                "type": "stacked",
                                "color": "#00bfff",
                            },
                            {
                                "column": "Productie",
                                "title": "Productie",
                                "negativ": "true",
                                "type": "stacked",
                                "color": "green",
                            },
                            {
                                "column": "Kosten",
                                "title": "Kosten",
                                "type": "stacked",
                                "color": "red",
                                "vaxis": "right",
                            },
                            {
                                "column": "Opbrengst",
                                "title": "Opbrengst",
                                "negativ": "true",
                                "type": "stacked",
                                "color": "#ff8000",
                                "vaxis": "right",
                            },
                        ],
                    }
                ],
            }
        if _title is None:
            options["title"] = options["title"] + " " + period
        else:
            options["title"] = _title
        options["haxis"] = {
            "values": self.periodes[period]["interval"]
            if self.periodes[period]["interval"] in df
            else self.periodes[period]["interval"].capitalize(),
            "title": self.periodes[period]["interval"],
        }

        gb = GraphBuilder()
        fig = gb.build(df, options, False)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        report_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close(fig)
        return report_data

    def get_vars(self):
        metadata = self.db_da.metadata
        engine = self.db_da.engine
        variabel = Table("variabel", metadata, autoload_with=engine)
        prognoses = Table("prognoses", metadata, autoload_with=engine)
        values = Table("values", metadata, autoload_with=engine)

        gebruikte_variabelen = union(
            select(prognoses.c.variabel),
            select(values.c.variabel),
        )

        query = (
            select(
                variabel.c.code,
                variabel.c.name,
            )
            .where(variabel.c.id.in_(gebruikte_variabelen))
            .order_by(variabel.c.name)
        )

        with self.db_da.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        return [dict(row) for row in rows]

    @staticmethod
    def create_interval_cte(
            start: datetime.datetime,
            end: datetime.datetime,
            step: datetime.timedelta,
            max_datapoints: int = 500,
    ):
        if end <= start:
            raise ValueError("end moet na start liggen")

        if step <= datetime.timedelta(0):
            raise ValueError("step moet groter zijn dan 0")

        intervals: list[tuple[int, int]] = []
        current = start

        while current < end:
            if len(intervals) >= max_datapoints:
                raise ValueError(
                    f"Meer dan {max_datapoints} datapoints"
                )

            interval_end = min(current + step, end)

            intervals.append((
                int(current.timestamp()),
                int(interval_end.timestamp()),
            ))

            current = interval_end

        return (
            values(
                column("ts_start", BigInteger()),
                column("ts_end", BigInteger()),
                name="interval_values",
            )
            .data(intervals)
            .cte("intervals")
        )

    def create_ha_sensors_cte(self, var_codes: list = None):
        sensor_rows = []

        for code, config in self.energy_balance_dict.items():
            if var_codes and code not in var_codes:
                continue

            sensors = config.get("sensors", [])

            # Speciale waarden zoals "calc" niet als sensor opnemen.
            if not isinstance(sensors, (list, tuple, set)) or not sensors:
                continue

            if sensors:
                sensor_rows.extend(
                    (code, sensor)
                    for sensor in sensors
                )
            else:
                # Code behouden wanneer de sensorlijst leeg is.
                sensor_rows.append((code, None))

        if len(sensor_rows) == 0:
            return None

        return (
            values(
                column("code", String),
                column("sensor", String),
                name="sensor_values",
            )
            .data(sensor_rows)
            .cte("sensors")
        )

    def get_ha_data_query(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            step: datetime.timedelta,
            var_codes: list = None
    ):
        statistics = Table(
            "statistics", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )
        statistics_meta = Table(
            "statistics_meta", self.db_ha.metadata, autoload_with=self.db_ha.engine
        )

        intervals_cte = self.create_interval_cte(
            start=start,
            end=end,
            step=step,
        )
        sensors_cte = self.create_ha_sensors_cte(var_codes)

        if sensors_cte is None:
            return None

        sensor_values_cte = (
            select(
                intervals_cte.c.ts_start.label("ts"),
                sensors_cte.c.code.label("code"),
                sensors_cte.c.sensor.label("sensor"),
                (

                                func.max(statistics.c.sum)
                                - func.min(statistics.c.sum)

                ).label("sensor_value"),
            )
            .select_from(
                intervals_cte
                .join(
                    statistics,
                    and_(
                        statistics.c.start_ts >= intervals_cte.c.ts_start,
                        statistics.c.start_ts <= intervals_cte.c.ts_end,
                    ),
                )
                .join(
                    statistics_meta,
                    statistics_meta.c.id == statistics.c.metadata_id,
                )
                .join(
                    sensors_cte,
                    sensors_cte.c.sensor == statistics_meta.c.statistic_id,
                )
            )
            .group_by(
                intervals_cte.c.ts_start,
                sensors_cte.c.code,
                sensors_cte.c.sensor,
            )
            .cte("sensor_values")
        )

        query = (
            select(
                sensor_values_cte.c.ts,
                sensor_values_cte.c.code,
                func.sum(sensor_values_cte.c.sensor_value).label("v"),
            )
            .group_by(
                sensor_values_cte.c.ts,
                sensor_values_cte.c.code,
            )
            .order_by(
                sensor_values_cte.c.ts,
                sensor_values_cte.c.code,
            )
        )

        return query

    def get_da_data_query(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            step: datetime.timedelta,
            var_codes: list = None
    ):
        metadata = self.db_da.metadata
        engine = self.db_da.engine

        variabel = Table("variabel", metadata, autoload_with=engine)
        values_table = Table("values", metadata, autoload_with=engine)
        prognoses = Table("prognoses", metadata, autoload_with=engine)

        intervals_cte = self.create_interval_cte(
            start=start,
            end=end,
            step=step,
        )

        if var_codes:
            selected_vars = (
                values(
                    column("code", String),
                    name="selected_var_values",
                )
                .data([(code,) for code in var_codes])
                .cte("selected_vars")
            )
        else:
            selected_vars = (
                select(variabel.c.code)
                .cte("selected_vars")
            )

        # vals CTE
        vals = (
            select(
                intervals_cte.c.ts_start.label("ts"),
                values_table.c.variabel,
                func.sum(values_table.c.value).label("csum"),
                func.avg(values_table.c.value).label("cavg"),
            )
            .select_from(
                intervals_cte
                .join(
                    values_table,
                    and_(
                        values_table.c.time >= intervals_cte.c.ts_start,
                        values_table.c.time < intervals_cte.c.ts_end,
                    ),
                )
            )
            .group_by(
                intervals_cte.c.ts_start,
                values_table.c.variabel,
            )
            .cte("vals")
        )

        # forecasts CTE
        forecasts = (
            select(
                intervals_cte.c.ts_start.label("ts"),
                prognoses.c.variabel,
                func.sum(prognoses.c.value).label("csum"),
                func.avg(prognoses.c.value).label("cavg"),
            )
            .select_from(
                intervals_cte
                .join(
                    prognoses,
                    and_(
                        prognoses.c.time >= intervals_cte.c.ts_start,
                        prognoses.c.time < intervals_cte.c.ts_end,
                    ),
                )
            )
            .group_by(
                intervals_cte.c.ts_start,
                prognoses.c.variabel,
            )
            .cte("forecasts")
        )

        # final SELECT
        query = (
            select(
                intervals_cte.c.ts_start.label("ts"),
                variabel.c.code,
                variabel.c.dim,
                case(
                    (
                        variabel.c.aggregate == "sum",
                        vals.c.csum,
                    ),
                    else_=vals.c.cavg,
                ).label("v"),
                case(
                    (
                        variabel.c.aggregate == "sum",
                        forecasts.c.csum,
                    ),
                    else_=forecasts.c.cavg,
                ).label("f"),
            )
            .select_from(
                intervals_cte
                .join(
                    variabel,
                    variabel.c.code.in_(
                        select(selected_vars.c.code)
                    ),
                )
                .join(
                    vals,
                    and_(
                        vals.c.ts == intervals_cte.c.ts_start,
                        vals.c.variabel == variabel.c.id,
                    ),
                    isouter=True,
                )
                .join(
                    forecasts,
                    and_(
                        forecasts.c.ts == intervals_cte.c.ts_start,
                        forecasts.c.variabel == variabel.c.id,
                    ),
                    isouter=True,
                )
            )
        )

        return query

    def get_data(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            aggregate: str = "hour",
            var_codes: list[str] = None
    ):
        if aggregate not in AGGREGATE_DELTAS:
            raise ValueError(f"Invalid aggregate interval: {aggregate}")

        query_da = self.get_da_data_query(
            start=start,
            end=end,
            step=AGGREGATE_DELTAS[aggregate],
            var_codes=var_codes,
        )

        with self.db_da.engine.connect() as connection:
            # return str(query_da.compile(connection, compile_kwargs={"literal_binds": True}))
            df = pd.read_sql_query(query_da, connection)

        query_ha = self.get_ha_data_query(
            start=start,
            end=end,
            step=AGGREGATE_DELTAS[aggregate],
            var_codes=var_codes,
        )

        if query_ha is not None:
            with self.db_ha.engine.connect() as connection:
                # return str(query_ha.compile(connection, compile_kwargs={"literal_binds": True}))
                df_ha = pd.read_sql_query(query_ha, connection)

            # Alleen mergen wanneer de query daadwerkelijk data oplevert.
            if not df_ha.empty:
                df = df.merge(
                    df_ha[["ts", "code", "v"]].rename(columns={"v": "v_ha"}),
                    on=["ts", "code"],
                    how="left",
                )

                mask = df["v"].isna() | df["v"].eq(0)
                df.loc[mask, "v"] = df.loc[mask, "v_ha"]
                df = df.drop(columns="v_ha")

        target_tz = start.tzinfo
        df["ts"] = (
            pd.to_datetime(df["ts"], unit="s", utc=True)
            .dt.tz_convert(target_tz)
        )

        df[["v", "f"]] = df[["v", "f"]].round(3)

        return [
            {"ts": ts, **row.unstack(0).to_dict()}
            for ts, row in (
                df.pivot(index="ts", columns="code", values=["v", "f"])
                .swaplevel(axis=1)
                .sort_index(axis=1)
                .astype(object)
                .where(pd.notnull, None)
            ).iterrows()
        ]
