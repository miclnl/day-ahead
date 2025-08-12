import json
import logging
import os

# from logging import raiseExceptions
# Support both execution contexts (module run from /app/dao/prog and from webserver cwd)
try:
    from dao.prog.db_manager import DBmanagerObj  # type: ignore
except Exception:  # pragma: no cover - fallback for webserver context
    from db_manager import DBmanagerObj  # type: ignore
import sqlalchemy_utils


class Config:
    db_da = None
    db_ha = None

    @staticmethod
    def parse(file_name: str):
        with open(file_name, "r") as file_json:
            try:
                return json.load(file_json)
            except ValueError as e:
                logging.error(f"Invalid json in {file_name}: {e}")
                raise e

    def __init__(self, file_name: str):
        self.options = self.parse(file_name)
        datapath = os.path.dirname(file_name)
        file_secrets = os.path.join(datapath, "secrets.json")
        # Secrets zijn optioneel; als het bestand ontbreekt val terug op leeg dict
        if os.path.exists(file_secrets):
            try:
                self.secrets = self.parse(file_secrets)
            except Exception:
                logging.warning(
                    f"Kon secrets.json niet parsen ({file_secrets}), gebruik lege secrets"
                )
                self.secrets = {}
        else:
            logging.info(
                f"secrets.json niet gevonden op {file_secrets} – ga verder met lege secrets"
            )
            self.secrets = {}

    def get(
        self, keys: list, options: dict = None, default=None
    ) -> str | dict | list | None:
        if options is None:
            options = self.options
        if options and keys[0] in options:
            result = options[keys[0]]
            if str(result).lower().find("!secret", 0) == 0:
                result = self.secrets[result[8:]]
            if type(result) is dict:
                if len(keys) > 1:
                    result = self.get(keys[1:], result, default)
                else:
                    for key in result:
                        result[key] = self.get([key], result, default)
        else:
            result = default
        return result

    def set(self, key, value):
        self.options[key] = value

    def get_db_da(self, check_create: bool = False):
        if Config.db_da is None:
            # Bepaal engine: respecteer nested config, val terug op top-level 'database_engine', default sqlite
            top_engine = self.get(["database_engine"], None, "sqlite")
            db_da_engine = self.get(["database da", "engine"], None, top_engine)

            # Fallbacks naar eenvoudige, platte add-on opties
            db_da_server = self.get(["database da", "server"], None,
                                     self.get(["database_server"], None, "core-mariadb"))
            db_da_port = int(self.get(["database da", "port"], None,
                                      int(self.get(["database_port"], None, 0))))
            if db_da_engine == "sqlite":
                db_da_name = self.get(["database da", "database"], None,
                                      self.get(["database_database"], None, "day_ahead.db"))
            else:
                db_da_name = self.get(["database da", "database"], None,
                                      self.get(["database_database"], None, "day_ahead"))
            db_da_user = self.get(["database da", "username"], None,
                                   self.get(["database_username"], None, "day_ahead"))
            db_da_password = self.get(["database da", "password"], None)
            if db_da_password is None:
                db_da_password = self.get(["database_password"], None)
            # Voor HA add-on omgeving: standaard /data als sqlite pad
            default_db_path = "/data" if db_da_engine == "sqlite" else "../data"
            db_da_path = self.get(["database da", "db_path"], None,
                                  self.get(["database_path"], None, default_db_path))
            db_time_zone = self.get(["time_zone"])
            if check_create:
                db_url = DBmanagerObj.db_url(
                    db_dialect=db_da_engine,
                    db_name=db_da_name,
                    db_server=db_da_server,
                    db_user=db_da_user,
                    db_password=db_da_password,
                    db_port=db_da_port,
                    db_path=db_da_path,
                )
                if not sqlalchemy_utils.database_exists(db_url):
                    sqlalchemy_utils.create_database(db_url)
            try:
                _db_da = DBmanagerObj(
                    db_dialect=db_da_engine,
                    db_name=db_da_name,
                    db_server=db_da_server,
                    db_user=db_da_user,
                    db_password=db_da_password,
                    db_port=db_da_port,
                    db_path=db_da_path,
                    db_time_zone=db_time_zone,
                )
            except Exception as ex:
                logging.error("Check your settings for day_ahead database")
                return None
            Config.db_da = _db_da
        return Config.db_da

    def get_db_ha(self):
        if Config.db_ha is None:
            db_ha_engine = self.get(["database ha", "engine"], None, "mysql")
            db_ha_server = self.get(["database ha", "server"], None, "core-mariadb")
            db_ha_port = int(self.get(["database ha", "port"], None, 0))
            if db_ha_engine == "sqlite":
                db_ha_name = self.get(
                    ["database ha", "database"], None, "home-assistant_v2.db"
                )
            else:
                db_ha_name = self.get(
                    ["database ha", "database"], None, "homeassistant"
                )
            db_ha_user = self.get(["database ha", "username"], None, "homeassistant")
            db_ha_password = self.get(["database ha", "password"])
            db_ha_path = self.get(["database ha", "db_path"], None, "/homeassistant")
            db_time_zone = self.get(["time_zone"])
            try:
                db_ha = DBmanagerObj(
                    db_dialect=db_ha_engine,
                    db_name=db_ha_name,
                    db_server=db_ha_server,
                    db_user=db_ha_user,
                    db_password=db_ha_password,
                    db_port=db_ha_port,
                    db_path=db_ha_path,
                    db_time_zone=db_time_zone,
                )
            except Exception as ex:
                logging.error("Check your settings for Home Assistant database")
                return None
            Config.db_ha = db_ha
        return Config.db_ha


def get_config(file_name: str, keys: list, default=None):
    config = Config(file_name=file_name)
    return config.get(keys, None, default)
