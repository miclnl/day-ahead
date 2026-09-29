import pandas as pd
from dao.lib.db_manager import DBmanagerObj
from entsoe import EntsoePandasClient
import datetime
import requests
from requests import get, post
from nordpool.elspot import Prices
import pytz
import json
import math
import pprint as pp
import logging


class DaPrices:
    def __init__(
        self, config, db_da: DBmanagerObj, country: str = None, secrets: dict = None
    ):
        self.config = config
        self.db_da = db_da
        self._secrets = secrets or {}
        self.interval = str(config.interval or "1hour").lower()
        self.country = country if country is not None else "NL"

    def get_prices(
        self, source, _start: datetime.datetime = None, _end: datetime.datetime = None
    ):
        """Fetch day-ahead prices from *source* and store them as code "da".

        Without an explicit range the prices for today (and, after noon, for
        tomorrow) are fetched, unless the database already holds them. An
        explicit ``_start``/``_end`` is a backfill request: the range is
        fetched as given and the "already present" check is skipped. The CLI
        entry point (DaBase.get_day_ahead_prices) is the only place that
        turns command line arguments into that range; this method must not
        look at sys.argv, it also runs inside the web server.
        """
        if self.interval == "1hour":
            resolution = 60
        else:
            resolution = 15
        now = datetime.datetime.now()
        explicit_range = _start is not None or _end is not None
        # start
        if _start is None:
            start = pd.Timestamp(year=now.year, month=now.month, day=now.day, tz="CET")
        else:
            start = _start
        # end
        if _end is None:
            if now.hour < 12:
                end = start + datetime.timedelta(days=1)
            else:
                end = start + datetime.timedelta(days=2)
        else:
            end = _end

        if not explicit_range:
            present = self.db_da.get_time_border_record("da")
            if not (present is None):
                tz = pytz.timezone("CET")
                present = tz.normalize(tz.localize(present))
                if end.tzinfo is None:
                    end = tz.normalize(tz.localize(end))
                if present >= (end - datetime.timedelta(hours=1)):
                    logging.info(f"Day ahead data already present")
                    return

        # day-ahead market prices (€/MWh)
        if source.lower() == "entsoe":
            start = pd.Timestamp(
                year=start.year, month=start.month, day=start.day, tz="CET"
            )
            end = pd.Timestamp(year=end.year, month=end.month, day=end.day, tz="CET")
            _ak = self.config.prices.entsoe_api_key
            api_key = _ak.resolve(self._secrets) if _ak is not None else None
            client = EntsoePandasClient(api_key=api_key)
            da_prices = pd.DataFrame()
            try:
                da_prices = client.query_day_ahead_prices(
                    self.country, start=start, end=end
                )
            except Exception as ex:
                logging.error(ex)
                logging.error(f"Geen data van Entsoe: tussen {start} en {end}")
            if len(da_prices.index) > 0:
                da_prices = (
                    da_prices.reset_index()
                )  # make sure indexes pair with number of rows
                logging.info(
                    f"Day ahead prijzen van Entsoe: \n{da_prices.to_string(index=False)}"
                )
                last_time = start
                rows = []
                for row in da_prices.itertuples():
                    last_time = int(datetime.datetime.timestamp(row[1]))
                    rows.append((str(last_time), "da", row[2] / 1000))
                df_db = pd.DataFrame(rows, columns=["time", "code", "value"])
                logging.debug(
                    f"Day ahead prijzen (source: entsoe, db-records): \n"
                    f"{df_db.to_string(index=False)}"
                )
                self.db_da.savedata(df_db)
                end_dt = datetime.datetime(end.year, end.month, end.day, 23)
                last_time_dt = datetime.datetime.fromtimestamp(last_time)
                if last_time_dt < end_dt:
                    if len(df_db) == 0:
                        logging.error(f"Geen data van Entsoe tot en met {end_dt}")
                    else:
                        logging.warning(
                            f"Geen data van Entsoe tussen {last_time_dt} en {end_dt}"
                        )

        if source.lower() == "nordpool":
            # ophalen bij Nordpool. Without an explicit range the library
            # fetches the prices for tomorrow (end_date=None).
            prices_spot = Prices()
            end_date = start if explicit_range else None
            day_label = end_date.strftime("%Y-%m-%d") if end_date else "tomorrow"
            try:
                act_spot_prices = prices_spot.fetch(
                    areas=[self.country], end_date=end_date, resolution=resolution
                )
            except Exception as ex:
                logging.error(f"Geen data van Nordpool voor {day_label}: {ex}")
                return
            if not act_spot_prices:
                logging.error(f"Geen data van Nordpool voor {day_label}")
                return
            try:
                act_values = act_spot_prices["areas"][self.country]["values"]
            except (KeyError, TypeError):
                logging.error(
                    f"Onverwacht antwoord van Nordpool voor {day_label}: "
                    f"{str(act_spot_prices)[:200]}"
                )
                return
            s = pp.pformat(act_values, indent=2)
            logging.info(f"Day ahead prijzen van Nordpool:\n {s}")
            rows = []
            for act_value in act_values:
                value = act_value.get("value")
                if value is None or not math.isfinite(value):
                    continue
                rows.append([str(int(act_value["start"].timestamp())), "da", value / 1000])
            df_db = pd.DataFrame(rows, columns=["time", "code", "value"])
            logging.debug(
                f"Day ahead prices for {day_label}"
                f" (source: nordpool, db-records): \n {df_db.to_string(index=False)}"
            )
            # A full day has 24 hours or 96 quarters; the autumn DST day has one
            # hour more, the spring day one less, so allow one hour of slack.
            expected = 24 * 60 // resolution
            if len(df_db) < expected - 60 // resolution:
                logging.warning(
                    f"Retrieve of day ahead prices for {day_label} incomplete: "
                    f"{len(df_db)} of {expected} values"
                )
            if len(df_db) > 0:
                self.db_da.savedata(df_db)

        if source.lower() == "easyenergy":
            # ophalen bij EasyEnergy
            # 2022-06-25T00:00:00
            startstr = start.strftime("%Y-%m-%dT%H:%M:%S")
            endstr = end.strftime("%Y-%m-%dT%H:%M:%S")
            url = "https://mijn.easyenergy.com/nl/api/tariff/getapxtariffs"
            try:
                resp = get(
                    url,
                    params={"startTimestamp": startstr, "endTimestamp": endstr},
                    timeout=(5, 30),
                )
                resp.raise_for_status()
                json_object = resp.json()
            except (requests.RequestException, ValueError) as ex:
                logging.error(f"Ophalen day-ahead prijzen bij EasyEnergy mislukt: {ex}")
                return
            logging.debug(json_object)
            df = pd.DataFrame.from_records(json_object)
            if df.empty or not {"Timestamp", "TariffReturn"} <= set(df.columns):
                logging.error(
                    f"Onverwacht antwoord van EasyEnergy, geen prijzen opgeslagen: "
                    f"{str(json_object)[:200]}"
                )
                return
            logging.info(
                f"Day ahead prijzen van Easyenergy:\n {df.to_string(index=False)}"
            )
            # datetime.datetime.strptime('Tue Jun 22 12:10:20 2010 EST', '%a %b %d %H:%M:%S %Y %Z')
            df = df.reset_index()  # make sure indexes pair with number of rows
            rows = []
            for row in df.itertuples():
                dtime = str(
                    int(datetime.datetime.fromisoformat(row.Timestamp).timestamp())
                )
                rows.append((dtime, "da", row.TariffReturn))
            df_db = pd.DataFrame(rows, columns=["time", "code", "value"])

            logging.debug(
                f"Day ahead prijzen (source: easy energy, db-records): \n "
                f"{df_db.to_string(index=False)}"
            )
            self.db_da.savedata(df_db)

        if source.lower() == "tibber":
            now_ts = datetime.datetime.now().timestamp()
            get_ts = start.timestamp()
            count = 1 + math.ceil((now_ts - get_ts) / 3600)
            if self.interval == "1hour":
                resolution = "HOURLY"
            else:
                resolution = "QUARTER_HOURLY"
                count = count * 4
                if count > 674:
                    count = 674
                    logging.warning(
                        "Je kunt met Tibber maximaal 7 dagen terug opvragen"
                    )
            count = max(1, min(674, count))
            query = (
                "{ "
                '"query": '
                ' "{ '
                "  viewer { "
                "    homes { "
                "      currentSubscription { "
                "        priceInfo(resolution: " + resolution + "){ "
                "          today { "
                "            energy "
                "            startsAt "
                "          } "
                "          tomorrow { "
                "            energy "
                "            startsAt "
                "          } "
                "        } "
                "        priceInfoRange(resolution: "
                + resolution
                + ", last: "
                + str(count)
                + ") { "
                "          nodes { "
                "            energy "
                "            startsAt "
                "          } "
                "        } "
                "      } "
                "    } "
                "  } "
                '}" '
                "}"
            )

            logging.debug(query)
            _tibber = self.config.tibber
            _tok = _tibber.api_token
            api_token = _tok.resolve(self._secrets)
            url = _tibber.api_url or "https://api.tibber.com/v1-beta/gql"
            headers = {
                "Authorization": "Bearer " + api_token,
                "content-type": "application/json",
            }
            try:
                resp = post(url, headers=headers, data=query, timeout=(5, 30))
                resp.raise_for_status()
                tibber_dict = resp.json()
            except (requests.RequestException, ValueError) as ex:
                logging.error(f"Ophalen day-ahead prijzen bij Tibber mislukt: {ex}")
                return
            if tibber_dict.get("errors"):
                logging.error(f"Tibber API gaf fouten terug: {tibber_dict['errors']}")
                return
            try:
                subscription = tibber_dict["data"]["viewer"]["homes"][0][
                    "currentSubscription"
                ]
                today_nodes = subscription["priceInfo"]["today"]
                tomorrow_nodes = subscription["priceInfo"]["tomorrow"]
                range_nodes = subscription["priceInfoRange"]["nodes"]
            except (KeyError, IndexError, TypeError) as ex:
                logging.error(
                    f"Onverwacht antwoord van Tibber ({ex}), geen prijzen opgeslagen: "
                    f"{str(tibber_dict)[:200]}"
                )
                return
            rows = []
            for lst in [today_nodes, tomorrow_nodes, range_nodes]:
                for node in lst:
                    dt = datetime.datetime.strptime(
                        node["startsAt"], "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
                    time_stamp = int(dt.timestamp())
                    value = float(node["energy"])
                    logging.info(f"{node} {dt} {time_stamp} {value}")
                    rows.append((time_stamp, "da", value))
            df_db = pd.DataFrame(rows, columns=["time", "code", "value"])
            logging.debug(
                f"Day ahead prijzen (source: tibber, db-records): \n "
                f"{df_db.to_string(index=False)}"
            )
            self.db_da.savedata(df_db)
