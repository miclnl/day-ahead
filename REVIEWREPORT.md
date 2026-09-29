# Review DAO+ (day-ahead) — bugs, verbeterkansen, library-hergebruik

Datum: 2026-09-28
Basis: commit `2971722` (Format fast-control event timestamps as human-readable)

**Scope**: alle Python in `dao/` (~30.700 regels) volledig gelezen; `day_ahead.py`, `da_base.py`, `utils.py`, `da_scheduler.py`, `da_fast.py`, `fastctrl/*` direct; `lib/`, `webserver/`, `config/`, `da_report.py`, `solar_predictor.py`, `baseload.py`, `check_db.py` via parallelle deelreviews die zijn gecontroleerd. Claims met **[geverifieerd]** zijn gereproduceerd in `.venv` (pandas 3.0.6, pydantic 2.13.4, SQLAlchemy 2.0.54, Python 3.14).

## Status van de reparaties (bijgewerkt 2026-09-29, Fase 3 afgerond)

Alle 18 kritieke bugs en alle 8 verbeterkansen zijn behandeld; zie de tabellen hieronder voor wat volledig is opgelost versus welke verbeterkansen bewust een beperkte, praktische invulling hebben gekregen in plaats van de volledige (1-3 dagen geschatte) herarchitectuur. Van de bijlage (medium/low) is alles opgelost op drie stukjes bewust ongemoeide dode code na (zie de bijlage zelf). Testsuite na afloop: `779 passed, 5 skipped` (de 5 zijn integratietests die live databases en HA nodig hebben; zet `DAO_INTEGRATION_TESTS=1` om ze lokaal te draaien). CI draait pytest als eerste job in `test_build.yaml`.

### Kritieke bugs

| Bevinding | Status | Commit |
|---|---|---|
| Bug #1 datamap via `/static/data` | opgelost: symlink verwijderd (stond in git), `/images/<name>`-route | `14337f8` |
| Bug #2 geen auth / traversal / XSS / CSRF | opgelost: ingress-guard, poort standaard uit + optie `allow_direct_access`, whitelist, `abort(404)`, Flask-WTF | `14337f8` |
| Bug #3 crash op `unavailable` | opgelost: `read_state/get_float/get_bool/get_str/get_datetime`, `FlexValue.resolve(default=)`, alle 39 callsites | `6112eeb` |
| Bug #4 15-min epoch | opgelost: epoch uit DB door `interpolate()` heen, join op `time` | `98daac1` |
| Bug #5 scheduler-migratie | opgelost: alleen legacy-vorm converteren, loader-test op `options_example.json` | `ce14162` |
| Bug #6 v1→v2 alias-keys (+ v0→v1 meteoserver) | opgelost | `db2a5cb` |
| Bug #7 loader/model dump, enum-injectie | opgelost: migratie schrijft het document zelf, atomair; `inject_flex_enum_values` kopieert | `ce14162`, `db2a5cb` |
| Bug #8 `savedata` race | opgelost: dialect-upsert per batch, `URL.create` | `5a5903b` |
| Bug #9 machines IndexError/UnboundLocal | opgelost | `6112eeb` |
| Bug #10 EV zonder `charge_scheduler` | opgelost in de consumer (niet ingepland buiten instant-modus) | `6112eeb` |
| Bug #11 één try/except om publicatie | opgelost: try/except per device, `grid_balance` vooraf berekend; `set_value` waarschuwt i.p.v. raise bij read-back | `6112eeb` |
| Bug #12 scheduler mist ticks | opgelost: APScheduler, per-taak lock, misfire grace | `417d09a` |
| Bug #13 consolidatie | opgelost (drie oorzaken), incl. gap-fill epoch | `7f5985e` |
| Bug #14 PostgreSQL DDL | opgelost, creates idempotent | `7f5985e` |
| Bug #15 `self.config` ontbreekt | opgelost | `5eef5da` |
| Bug #16 HTTP zonder timeout | opgelost, incl. retry met backoff voor meteoserver | `d163398` |
| Bug #17 `da_prices` argv/Nordpool | opgelost; ENTSO-E 0.8.1 levert zelf de juiste resolutie | `622ac52` |
| Bug #18 fast-control regressies | opgelost: override vrijgeven bij `off`, eventlog-baseline | `d166e04` |

### Verbeterkansen (should-fix)

| Verbetering | Status | Commit |
|---|---|---|
| #1 APScheduler | volledig uitgevoerd | `417d09a` |
| #2 Eén robuuste HA-client | grotendeels uitgevoerd: guarded reads (`get_float`/`get_bool`/`get_str`/`get_datetime`), `FlexValue.resolve(default=)`, en een tenacity retry-wrapper (`_retry_ha_call`, 3 pogingen, exponentiële backoff) om `get_state`/`call_service`/`set_state` die alleen transiënte fouten (`ConnectionError`, `Timeout`, hassapi 429/500/502/503) opnieuw probeert, niet 401/403/404. `hassapi` zelf is niet vervangen door `requests.Session`/`homeassistant-api`; dat is een aparte, grotere migratie gebleven | `6112eeb`, `7915fb9` |
| #3 Tijdzonebeleid "epoch in, epoch uit" | gedeeltelijk: de concrete symptomen zijn gefixt (DST-crash in `get_api_data`, bucket-labels, ENTSO-E-resolutie), maar de architecturale opschoning (overal epoch opslaan, `pytz`/`unix_timestamp()` uit de SQL-laag, `TARGET_TIMEZONE` daadwerkelijk gebruiken) is niet gedaan | `5cce551`, `db08de4`, `622ac52` |
| #4 Eén config-schrijfpad | volledig uitgevoerd: `validate_config_data`, `atomic_write_text/json`, `set_fast_control_mode` | `ce14162` |
| #5 Taken uit de gunicorn-worker | gedeeltelijk: de webserver-taken draaien nog in de worker, maar wel geïsoleerd (`start_new_session=True`) zodat cancel de hele procesgroep opruimt in plaats van alleen het directe kind, en zodat een signaal aan de webserver zelf de taak niet halverwege meesleurt. De grotere herindeling (request-bestand + uitvoering door `da_scheduler.py`) is niet gedaan | `28b3da5` |
| #6 Requirements/Dockerfile opschonen | volledig uitgevoerd: 5 packages weg, mariadb-toolchain weg, `requirements-dev.txt` | `5bd7a07` |
| #7 pandas-antipatronen (`.loc[shape[0]]` in een lus) | volledig uitgevoerd: alle bereikbare O(n²)-lussen in `da_report.py` (7), `solar_predictor.py` (2), `da_base.py` (2), `da_meteo.py` (1), `da_prices.py` (3), `utils.py` (1) en `day_ahead.py` (4) vervangen door een lijst met tuples + één `pd.DataFrame(...)`. Drie overgebleven treffers zijn bewust niet aangeraakt: `da_meteo.py`'s tweede blok en `day_ahead.py`'s `df_pv_prog`-blok zijn dode code (staan in een niet-uitgevoerde `"""`-string), `utils.py`'s `interpol_rows` wordt alleen aangeroepen door het ongebruikte `interpolate_old` | `b2302d2`, `101e5c6`, `354a675`, `44d8e6e`, `d5a4f69`, `f6b3e2f` |
| #8 Solar-ML methodologisch repareren | volledig uitgevoerd: `resample("h").sum(min_count=1)` (een leeg uur wordt NaN, niet een verzonnen 0), `GridSearchCV(cv=TimeSeriesSplit(n_splits=3))` i.p.v. gewone KFold op een tijdreeks, `warnings.filterwarnings("ignore")` niet meer op module-niveau maar gescoped rond de `GridSearchCV.fit()`-aanroep zelf, `from pip._internal.utils import datetime` verwijderd, en `save_model()`/`load_model()` (xgboost's eigen formaat) i.p.v. `joblib.dump`/`load` met een `*.meta.json`-sidecar die de featurelijst vastlegt zodat een mismatch een duidelijke `ValueError` geeft in plaats van een stille verkeerde voorspelling | `81d9154`, `d0df427`, `101e5c6` |

Twee bugs uit de bijlage die niet in de eerste ronde waren meegenomen, zijn alsnog gefixt: de solar-key-normalisatie miste `.replace("-", "_")` op één van de vijf plekken (`day_ahead.py:702`, batterij-gekoppelde zonnepanelen), en de blok-optimalisatie van de warmtepomp deelde door `hours_avail` zonder te controleren op 0 (`boiler_int >= U`) — commit `8a01960`.

**Feiten vooraf (bij aanvang van de review)**
- Testsuite: `591 passed, 7 failed`. 2 failures zijn echte regressies uit commit `c30494f` (`test_runner_events.py`), 5 komen door bug #15. **CI draait pytest niet** (alleen build + docs-check).
- Dependencies: `mysql`, `mysql-connector-python`, `mariadb`, `cffi` en `freezegun` staan in `requirements.txt` maar worden in productiecode niet gebruikt (alleen `pymysql` en `psycopg2` via SQLAlchemy). `mariadb` is de enige reden voor `gcc/g++/libmariadb-dev` en de aarch64-hack in de Dockerfile.
- `calc_optimum` is één methode van ~5000 regels (`day_ahead.py:103-5144`).

---

## 1. Kritieke bugs (moeten direct gefixt worden)

### Bug #1: secrets.json en de hele datamap zijn publiek via `/static/data/`
- **Locatie**: `dao/run/run.sh` regel 33-40 (symlink `app/static/data -> /config/dao_data`), `dao/webserver/da_server.py` regel 4-5, gebruikt door `app/routes.py:25-27` en `app/v2/routes.py:278`
- **Type**: security / information disclosure
- **Root cause**: de datamap (met `secrets.json`, `options.json`, `day_ahead.db`, logs) is een symlink *binnen* Flask's static-map. `send_from_directory` volgt symlinks. **[geverifieerd door deelreview: `GET /static/data/secrets.json` → 200]**
- **Impact**: iedereen op het LAN (poort 5000 staat open, zie #2) of elke ingelogde HA-gebruiker leest DB-wachtwoorden, HA-token, Tibber/ENTSO-E/meteo-keys.
- **Fix**:
```python
# app/__init__.py — verwijder de symlink in run.sh/da_server.py en serveer alleen images
IMAGES_DIR = Path("../data/images").resolve()

@app.route("/images/<name>")
def image(name):
    if not re.fullmatch(r"[\w.\-]+\.png", name):
        abort(404)
    return send_from_directory(IMAGES_DIR, name)
```
Zet `app_datapath = "../data/"` en vervang alle `url_for('static', filename="data/images/...")` door `url_for('image', name=...)`.

### Bug #2: Geen authenticatie op poort 5000 + path traversal in settings-editor + reflected XSS
- **Locatie**: `dao/config.yaml` (`ports: 5000/tcp: 5000`); `app/routes.py:1034-1056` (`active_setting` uit formulier, `open(app_datapath + active_setting + ".json", "w")`); `app/routes.py:1169` (`return "Onbekende bewerking: " + bewerking`)
- **Type**: security (missing auth, arbitrary file write, XSS)
- **Root cause**: de app vertrouwt uitsluitend op HA-ingress, maar `ports:` omzeilt ingress. `settngs = ["options","secrets"]` (regel 1029) wordt alleen voor rendering gebruikt, nooit gevalideerd. **[geverifieerd door deelreview: `cur_setting=../../etc/pwned` schrijft `/etc/pwned.json` als root]**
- **Impact**: config/secrets lezen en schrijven, optimizer en fast-control op afstand aansturen, bestanden verwijderen; XSS levert JS in de origin die #1 kan exfiltreren.
- **Fix**:
```python
# config.yaml: verwijder de 'ports:' sectie (ingress-only).
# app/__init__.py:
SUPERVISOR = "172.30.32.2"
@app.before_request
def ingress_only():
    if request.remote_addr != SUPERVISOR or "X-Ingress-Path" not in request.headers:
        abort(401)

# routes.py:1034
if active_setting not in ("options", "secrets"):
    abort(400)
# routes.py:1169
abort(404)
```
Zet in `gunicorn_config.py` `forwarded_allow_ips = "172.30.32.2"`. Voeg `flask_wtf.csrf.CSRFProtect(app)` toe (alle POST-formulieren zijn zonder token; `GET /v2/task-cancel` muteert state).

### Bug #3: Optimizer crasht op elke HA-entity die `unavailable`/`unknown` is
- **Locatie**: `dao/prog/day_ahead.py` — 39 `self.get_state(...)` aanroepen, 5 in een `try` (o.a. regels 641, 1066, 1072, 1174-1189, 1615-1620, 2225, 2320, 2327, 4423); `dao/prog/config/models/base.py:172-183` (`FlexValue.resolve` doet `float(state)` zonder fallback)
- **Type**: edge case / ontbrekende foutafhandeling (systemisch)
- **Root cause**: `float("unavailable")` → `ValueError`; er is geen try/except rond het model-opbouwdeel (de `try` begint pas op regel 3837). HA levert deze states routinematig bij herstart, Wi-Fi-uitval, integratie-reload.
- **Impact**: geen plan, geen batterij-setpoint, geen `fast_plan.json` voor die run. Bij een structureel defecte sensor staat de sturing volledig stil.
- **Fix** (één helper, overal gebruiken):
```python
# da_base.py
INVALID = {"unavailable", "unknown", "", "none", None}

def get_float(self, entity_id: str, default: float, what: str = "") -> float:
    try:
        raw = self.get_state(entity_id).state
    except Exception as ex:
        logging.warning(f"{what or entity_id}: niet leesbaar ({ex}), default {default}")
        return default
    if str(raw).strip().lower() in INVALID:
        logging.warning(f"{what or entity_id} is '{raw}', default {default} gebruikt")
        return default
    try:
        return float(raw)
    except ValueError:
        logging.warning(f"{what or entity_id}='{raw}' geen getal, default {default}")
        return default
```
In `FlexValue.resolve()` dezelfde check toevoegen met een optionele `default=`-parameter en een eigen `FlexResolveError`. Voor SoC-achtige waarden: default = laatste bekende waarde of 50.

### Bug #4: 15-minuten prognoses krijgen een foute epoch → PV-voorspelling onzin
- **Locatie**: `dao/lib/db_manager.py:498` (`result_df["time"] = result_df["tijd"].astype(int) // 1e9`), consumer `dao/prog/da_base.py:855` (`calc_prod_solar(solar_option, row.time, ...)`)
- **Type**: bug (pandas 3 unit-inferentie + tijdzone)
- **Root cause**: `astype(int)` geeft in pandas 3 het aantal eenheden van de kolom (`datetime64[s]` of `[us]`), niet nanoseconden. **[geverifieerd: `time` = 1781524 i.p.v. 1781517600]**. Daarnaast is `tijd` naïeve lokale tijd, dus ook na correctie van de eenheid zit er een UTC-offset fout in.
- **Impact**: in `interval: 15min` (de standaard in `options_example.json`) berekent de DAO-predictor de zonnestand voor januari 1970 → PV-prognose fout voor elke solar zonder ML-model.
- **Fix**:
```python
tz = ZoneInfo(self.TARGET_TIMEZONE)
t = result_df["tijd"].dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
result_df["time"] = ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
```

### Bug #5: Migratie unversioned→v0 vernielt een al-nieuwe scheduler; `options_example.json` laadt niet via de loader
- **Locatie**: `dao/prog/config/migrations/unversioned_to_v0.py:48-74`; trigger `loader.py:107-109`
- **Type**: migratie-datacorruptie
- **Root cause**: `isinstance(migrated["scheduler"], dict)` behandelt élke dict als legacy `{"HHMM": action}`; de key `"schedule"` wordt een entry `{"time": "schedule", "action": [...]}`. Het voorbeeldbestand heeft geen `config_version`. **[geverifieerd door deelreview]** `test_load_example.py` maskeert dit door `config_version=0` te injecteren.
- **Impact**: nieuwe gebruikers die vanaf het voorbeeld starten krijgen een validatiefout; elke poging schrijft bovendien `options_unversioned.json`.
- **Fix**:
```python
_TIME_RE = re.compile(r"^(\d{2}|xx)(\d{2}|xx)$")
old = migrated["scheduler"]
if isinstance(old, dict) and not isinstance(old.get("schedule"), list):
    schedule = [{"time": k, "action": v} for k, v in old.items()
                if k != "active" and _TIME_RE.match(k)]
    migrated["scheduler"] = {"active": old.get("active", True), "schedule": schedule}
```
Voeg `"config_version": 2` toe aan `options_example.json` en `options_start.json`; voeg een test toe die de loader op een kopie van het voorbeeld draait.

### Bug #6: Migratie v1→v2 mist alias-keys → grid balance switch en grid setpoint verdwijnen stil
- **Locatie**: `dao/prog/config/migrations/v1_to_v2.py:45, 63`
- **Type**: migratie-dataverlies
- **Root cause**: checkt alleen `"entity_balance_switch"` / `"entity_grid_setpoint"`, terwijl echte configs de alias `"entity balance switch"` gebruiken (zoals het oude model definieerde). Door `extra="allow"` blijft de key onopgemerkt in `battery[0]` hangen. **[geverifieerd door deelreview: `grid.entity_balance_switch is None` na migratie]**
- **Impact**: grid-balanceren en ESS-setpoint (`day_ahead.py:3892-3902`) werken na de upgrade niet meer, zonder waarschuwing.
- **Fix**:
```python
migrated["grid"] = migrated.get("grid") or {}
for snake, spaced in (("entity_balance_switch", "entity balance switch"),
                      ("entity_grid_setpoint", "entity grid setpoint")):
    for battery in migrated["battery"]:
        for k in (snake, spaced):
            if k in battery:
                migrated["grid"].setdefault(spaced, battery[k])
                del battery[k]
```
Zelfde patroon in `v0_to_v1.py:41-49`: zoekt `meteo_attemps`, het echte veld heet `meteoserver-attemps` → retry-instelling wordt stil 2.

### Bug #7: Loader herschrijft `options.json` in ander formaat, bevriest alle defaults, dropt `null` en schrijft `enum_values` naar disk
- **Locatie**: `dao/prog/config/loader.py:131-140` (`model_dump(mode="json", exclude_none=True)` zonder `by_alias`); `dao/prog/config/models/base.py:345-348` (`inject_flex_enum_values` muteert de input-dict); webserver `app/routes.py:394-402`, `app/v2/routes.py:782-787` (dumpt hele model met `by_alias=True`)
- **Type**: round-trip corruptie
- **Root cause**: de migratiepad-dump gebruikt snake_case, de web-UI aliassen met spaties → formaat wisselt per schrijver. `exclude_none` verwijdert bewuste `null`s (bv. `"half life days": null`). De before-validator schrijft `{"value":..,"enum_values":[..]}` in de gedeelde `_raw_options`. **[geverifieerd door deelreview]**
- **Impact**: gebruikersconfig wordt onherkenbaar (580 B → 3.5 kB), toekomstige default-wijzigingen bereiken bestaande installaties niet, één klik op de fast-control modusknop herschrijft de hele config.
- **Fix**:
```python
# loader.py:131 en beide webserver-writers
save_data = model.model_dump(mode="json", by_alias=True, exclude_unset=True)

# base.py:345 — nooit input muteren
data = dict(data)
data[key] = {**value, "enum_values": enum_values} if isinstance(value, dict) \
            else {"value": value, "enum_values": enum_values}

# fast-control modusknop: chirurgisch, niet het hele model
raw = json.load(open(path)); raw.setdefault("fast control", {})["mode"] = new_mode
ConfigurationV2.model_validate(raw); atomic_write_json(path, raw)
```

### Bug #8: `savedata` is een niet-atomaire per-rij select-then-insert → race tussen fast-control-thread en scheduler-subprocess
- **Locatie**: `dao/lib/db_manager.py:272-333`; schrijvers `fastctrl/runner.py:643` (thread) en `da_meteo.py`/`da_prices.py`/`da_base.py:413` (subprocessen)
- **Type**: race condition / transactiesemantiek / performance
- **Root cause**: per rij SELECT variabel-id, SELECT values-id, dan UPDATE of INSERT; één commit op regel 331. Tabel heeft `UNIQUE(variabel, time)`. Twee schrijvers missen beide de SELECT → `IntegrityError` → hele batch rolt terug. ~1150 queries per meteo-run van 384 rijen.
- **Impact**: verloren prijs-/meteobatches; op SQLite "database is locked".
- **Fix**:
```python
from sqlalchemy.dialects import mysql, postgresql, sqlite
def _upsert(self, table):
    if self.db_dialect == "mysql":
        stmt = mysql.insert(table)
        return stmt.on_duplicate_key_update(value=stmt.inserted.value)
    mod = postgresql if self.db_dialect == "postgresql" else sqlite
    stmt = mod.insert(table)
    return stmt.on_conflict_do_update(index_elements=["variabel", "time"],
                                      set_={"value": stmt.excluded.value})
records = [{"variabel": ids[c], "time": int(t), "value": float(v)} for t, c, v in rows]
with self.engine.begin() as conn:
    conn.execute(self._upsert(values_table), records)
```

### Bug #9: Machines: `program_selected[m]` IndexError en `start_window_dt` UnboundLocalError
- **Locatie**: `dao/prog/day_ahead.py:2796-2811` en `2863-2888`
- **Type**: bug (index-misalignment / ongedefinieerde variabele)
- **Root cause**: als `self.get_state(entity_machine_program)` faalt wordt niets aan `program_selected` toegevoegd, maar regel 2808 indexeert `program_selected[m]`. Als `start_window_entity`/`end_window_entity` `None` is wordt `error=True` gezet, maar de code loopt door naar regel 2888 `if end_window_dt < start_window_dt` zonder dat die variabelen bestaan (of met stale waarden van de vorige machine).
- **Impact**: hele optimalisatie crasht bij één onbereikbare machine-entity.
- **Fix**:
```python
# 2797
try:
    program_selected.append(self.get_state(entity_machine_program).state)
except Exception as ex:
    logging.error(f"Machines: entity_machine_program: {ex}")
    program_selected.append(self.machines[m].programs[0].name)  # zero-programma
# 2863: initialiseer per machine
start_window_dt = end_window_dt = None
...
if error or start_window_dt is None or end_window_dt is None:
    KW.append(0); R.append(0); ma_uur_kw.append([[] for _ in range(U)]); ma_kw_dt.append([])
    ma_planned_start_dt.append(planned_start_dt); ma_planned_end_dt.append(planned_end_dt)
    continue
```

### Bug #10: EV met alleen `entity instant start` (geen `charge_scheduler`) crasht elke run
- **Locatie**: `dao/prog/config/models/devices/ev.py:202-256` (validator staat `charge_scheduler=None` toe); `dao/prog/day_ahead.py:1617-1621` en `1630-1632` (ongeguarded `.entity_set_level`, `.entity_ready_datetime`)
- **Type**: contract-mismatch model vs consumer
- **Root cause**: regel 1626 guardt `level_margin` wel, regels 1619 en 1630 niet.
- **Impact**: `AttributeError: 'NoneType' has no attribute 'entity_ready_datetime'` zodra instant-charge uit staat.
- **Fix**: maak `charge_scheduler: EVChargeScheduler` verplicht in het model (de consumer heeft altijd een ready-tijd en doelniveau nodig) en verwijder de "either/or"-validator.

### Bug #11: Eén `try/except` om het hele HA-publicatieblok: een boilerfout blokkeert de batterij-aansturing
- **Locatie**: `dao/prog/day_ahead.py:3837-4507`
- **Type**: foutafhandeling / ontwerpfout
- **Root cause**: boiler, grid, EV, solar, batterij, warmtepomp en machines zitten in één `try`. Een `ValueError` bij bv. `self.get_state(entity_charge_switch)` (regel 3985) slaat de batterij-, WP- en machine-publicatie over.
- **Impact**: batterij krijgt geen nieuw setpoint terwijl het plan wel is berekend; `published_battery` blijft leeg → `fast_plan.json` krijgt solver-waarden i.p.v. gepubliceerde.
- **Fix**: één `try/except` per device-blok, in de volgorde batterij → grid → boiler → EV → WP → machines:
```python
for name, publish in (("batterij", self._publish_battery), ("grid", self._publish_grid), ...):
    try:
        publish(...)
    except Exception as ex:
        error_handling(ex)
        logging.error(f"Publiceren {name} mislukt: {ex}")
```

### Bug #12: Hand-rolled scheduler mist ticks tijdens lange taken; dubbele tijden overschrijven elkaar
- **Locatie**: `dao/prog/da_scheduler.py:19-21` (`{entry.time: entry.action}`), `62-97` (sleep tot hele minuut, dan blokkerend `proc.wait()`)
- **Type**: logische fout
- **Root cause**: taken draaien sequentieel en blokkerend; na een taak van 3 minuten worden de tussenliggende minuten nooit geëvalueerd. In het voorbeeldschema: `train_ml_predictions` om 23:50 (GridSearchCV, minuten tot uren) → `clean_data` 23:59 en `calc_optimum` 00:00 vervallen; `calc_baseloads` 09:35 → `forecast_accuracy` 09:40 vervalt. Twee acties op dezelfde tijd: alleen de laatste blijft over (validator `scheduler.py:40-54` wijst daarnaast `HHxx` af, wat de runtime wel ondersteunt).
- **Impact**: gemiste optimalisaties, stille gaten in data.
- **Fix**: zie Verbetering #1 (APScheduler).

### Bug #13: `consolidate_data` is drie keer stuk
- **Locatie**: `dao/prog/da_report.py:1292-1308` (`dict(result_row)` op SQLAlchemy 2.0 `Row` → `TypeError`; bovendien `order_by(time)` oplopend + `.first()` = *oudste* i.p.v. nieuwste), `1373-1376` (`calc_cost(start, tot, code)` met 3 args op een 2-arg functie), `1383-1386` (`row.tijd.timestamp()` op naïeve `pd.Timestamp` = UTC-interpretatie, **[geverifieerd: 2 uur verschil]**)
- **Type**: bug / logische fout / tijdzone
- **Impact**: consolidatie-taak crasht bij de eerste code; na alleen de crash fixen zouden alle geconsolideerde `cons`/`prod` 1-2 uur verschoven worden opgeslagen.
- **Fix**:
```python
# 1292
q = select(t1.c.time).where(...).order_by(t1.c.time.desc()).limit(1)
ts = conn.execute(q).scalar()
return datetime.datetime.fromtimestamp(ts) if ts is not None else datetime.datetime(2020, 1, 1)
# 1373: haal "cost"/"profit" uit de consolidatie-loop (afgeleide grootheden)
# 1385
db_row = [str(int(row.utc)), code, float(row.value), row.tijd]
```

### Bug #14: `ALTER TABLE ... DEFAULT "avg"` blokkeert opstarten op PostgreSQL
- **Locatie**: `dao/prog/check_db.py:366-371`
- **Type**: SQL-dialect
- **Root cause**: dubbele quotes = identifier in PostgreSQL → `column "avg" does not exist`; binnen `engine.begin()` → rollback → herhaalt bij elke start.
- **Impact**: PostgreSQL-installaties komen niet meer door `update_db_da`.
- **Fix**: `... VARCHAR(3) NOT NULL DEFAULT 'avg'` (enkele quotes), plus `Table.create(engine, checkfirst=True)` op regels 209, 252, 270.

### Bug #15: `DaBase.__init__` returnt zonder `self.config` te zetten → `AttributeError` in alle subclasses
- **Locatie**: `dao/prog/da_base.py:104-109`; consumers `day_ahead.py:81`, `da_report.py:60`, `solar_predictor.py:62` (`if self.config is None: return`)
- **Type**: bug (broken guard) **[geverifieerd: 5 testfailures `'Report' object has no attribute 'config'`]**
- **Impact**: bij een configfout ziet de gebruiker een misleidende `AttributeError` in plaats van de echte fout.
- **Fix**: zet `self.config = None` vóór de `with DaBase._init_lock:` en laat de constructor bij een laadfout een `SystemExit(str(e))` gooien in plaats van stil te returnen.

### Bug #16: HTTP-calls zonder timeout kunnen scheduler en optimizer eeuwig blokkeren
- **Locatie**: `dao/prog/da_base.py:158` (`get(self.hassurl + "api/config")`), `dao/lib/da_meteo.py:413`, `dao/lib/da_prices.py:175, 255`, `dao/prog/utils.py:210`
- **Type**: missing timeout / geen statuscheck
- **Root cause**: `requests.get/post` zonder `timeout=`, geen `raise_for_status()`, `json.loads(resp.text)` ongeguarded. `da_base.py:159` doet `resp_dict["latitude"]` → `KeyError` bij een 401.
- **Impact**: hangende socket = scheduler-tick blijft hangen; meteo-retry (`da_meteo.py:412-423`) hamert zonder backoff.
- **Fix**: overal `timeout=(5, 30)` + `resp.raise_for_status()` + `resp.json()`; retries via `tenacity` (zie sectie 3).

### Bug #17: `da_prices` leest `sys.argv`, Nordpool-failcheck crasht, ENTSO-E ignoreert 15 min
- **Locatie**: `dao/lib/da_prices.py:35-57, 115` (`sys.argv[2]`), `152-161` (`end_date.year` terwijl `end_date=None`; `time_ts` unbound), `69-81` (geen `resolution` bij `query_day_ahead_prices`)
- **Type**: verborgen globale koppeling / bug / 15-min handling
- **Impact**: onder gunicorn (`/api/run/prices`) crasht `strptime(sys.argv[2])`; Nordpool-storing geeft `AttributeError` i.p.v. waarschuwing; ENTSO-E + 15min → `day_ahead.py:160-167` breekt af met "Er ontbreken kwartierwaarden".
- **Fix**: `get_prices(source, start=None, end=None, force=False)` zonder argv; `client.query_day_ahead_prices(..., resolution="15min" if self.interval != "1hour" else "60min")`; failcheck met `expected = 24*60//resolution`.

### Bug #18: Fast-control regressies op HEAD
- **Locatie**: `dao/prog/fastctrl/runner.py:285-288, 758-759` (`_last_setpoints_recorded`, commit `c30494f`) vs `dao/tests/prog/test_runner_events.py:180-202`; `runner.py:486-489` (`mode == OFF` → `return None` zonder override vrij te geven)
- **Type**: regressie / logische fout **[geverifieerd: 2 tests falen]**
- **Root cause**: de eerste tick logt nu altijd een `setpoint_change`-event; tests documenteren het tegendeel. Bij omschakelen naar `off` tijdens een actieve override blijft het overridden setpoint + `NO_STOP_SENTINEL` op de omvormer staan tot de volgende optimizer-run.
- **Fix**: kies één gedrag en pas test óf code aan; in `tick()` bij `MODE_OFF`: als `any(b.override_active for b in self.state.batteries)` eerst `_all_plan(..., "mode_off")` actueren en state persisteren, daarna returnen. Zet `python -m pytest dao/tests` in `test_build.yaml`.

---

## 2. Belangrijke verbeterkansen (should-fix)

### Verbetering #1: Vervang de eigen cron-lus door APScheduler
- **Huidige code**: `dao/prog/da_scheduler.py:56-97`
- **Probleem**: blokkerend, mist ticks (bug #12), geen catch-up, dubbele tijden verdwijnen, `print()` i.p.v. logging, `HHxx` ondersteund in runtime maar afgekeurd door validator.
- **Betere aanpak**: één `BackgroundScheduler` met `CronTrigger`, `max_instances=1` per taak, `misfire_grace_time`, taken in threads zodat de fast-control-lus en de minuut-tick elkaar niet raken.
- **Aanbevolen library**: `APScheduler` 3.11.x (stabiel; 4.x is nog alpha)
- **Voorbeeld-implementatie**:
```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def _trigger(pattern: str) -> CronTrigger:  # "0544", "xx15", "02xx"
    hour = "*" if pattern[:2] == "xx" else int(pattern[:2])
    minute = "*" if pattern[2:] == "xx" else int(pattern[2:])
    return CronTrigger(hour=hour, minute=minute, timezone=self.time_zone)

sched = BackgroundScheduler(job_defaults={"max_instances": 1, "coalesce": True,
                                          "misfire_grace_time": 120})
for entry in self.config.scheduler.schedule:
    sched.add_job(self.run_task_process, _trigger(entry.time),
                  args=[self._task_key(entry.action)], id=f"{entry.time}-{entry.action}")
sched.start()
```
- **Trade-offs**: +1 dependency; gedrag bij overlappende taken moet expliciet gekozen worden (`max_instances=1` = overslaan met logregel). Inspanning: 0.5 dag.

### Verbetering #2: Eén robuuste HA-client met sessie, timeouts, retries en veilige parsing
- **Huidige code**: `dao/prog/da_base.py:153-166, 239-256` (hassapi + read-back verificatie), `fastctrl/runner.py:91-215`
- **Probleem**: `hassapi` 0.3.0 is een dunne wrapper zonder sessiehergebruik (nieuwe TCP-verbinding per call), zonder retry; `DaBase.set_value` verifieert direct na schrijven — bij device-backed `number.`-entities loopt de state achter → valse `ValueError` → bug #11 wordt getriggerd. Het gedrag "unavailable → default" is nergens gecentraliseerd (bug #3).
- **Betere aanpak**: eigen `HAClient` op `requests.Session` met `urllib3.Retry`, `get_float(entity, default)`, `get_bool`, `get_datetime`, en `set_value` zonder read-back (of read-back met korte poll en alleen een *warning*).
- **Aanbevolen library**: `requests` (al aanwezig) + `tenacity` 9.x voor retries; alternatief `homeassistant-api` 5.x (onderhouden, sessie, typed models). `hassapi` verwijderen.
- **Trade-offs**: alle ~60 callsites migreren (mechanisch); winst: één plek voor foutafhandeling. Inspanning: 1-1.5 dag.

### Verbetering #3: Tijdzonebeleid "epoch in, epoch uit"
- **Huidige code**: `db_manager.py:165-254` (conversie in SQL in de DB-sessie-TZ; `TARGET_TIMEZONE` ongebruikt), `da_report.py:1115, 1131, 1383, 3217, 3430` (naïeve `pd.Timestamp.timestamp()` / `tz_localize` zonder `ambiguous`), `da_prices.py:40-73` (hard-coded `"CET"`), `utils.py:332, 339` (`.seconds`)
- **Probleem**: drie klokken (container-TZ, DB-sessie-TZ, `config.time_zone`) die alleen toevallig gelijk lopen; `/api/report/...` geeft 500 op beide DST-dagen; PostgreSQL-installaties zien alles 1-2 uur verschoven.
- **Betere aanpak**: vergelijk en sla altijd epoch-integers op; converteer alleen in Python: `pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(config.time_zone)` en `datetime.fromtimestamp(ts, tz=ZoneInfo(...))`. Verwijder `unix_timestamp()`/`from_unixtime()` uit de WHERE-clausules.
- **Aanbevolen library**: stdlib `zoneinfo` (vervang `pytz`), pandas tz-API.
- **Trade-offs**: raakt veel queries; per query goed testbaar met een vaste epoch. Inspanning: 2-3 dagen.

### Verbetering #4: Eén `config_io`-module voor lezen, valideren en atomair schrijven
- **Huidige code**: `loader.py:97-141, 186-213`, `app/routes.py:395-401, 667-670, 1053-1056`, `app/v2/routes.py:105-110, 648-650, 676-678, 783-787` — vijf verschillende schrijfpaden, waarvan drie truncerend en drie zonder pydantic-validatie.
- **Probleem**: een typo in de UI slaat een niet-laadbare config op → scheduler in herstartlus; crash tijdens schrijven = 0-byte `options.json`; `flock` in de loader beschermt niets omdat de andere schrijvers hem niet nemen.
- **Betere aanpak**:
```python
def save_config(path: Path, data: dict) -> None:
    ConfigurationV2.model_validate(data)          # gooit ConfigValidationError → HTTP 400
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
```
- **Aanbevolen library**: pydantic (al aanwezig): `exclude_unset=True`, `AliasChoices` i.p.v. `populate_by_name` op 25 modellen, `Discriminator(callable)` voor de `heater present`/`boiler present`-unions (die nu string-booleans afwijzen).
- **Trade-offs**: geen. Inspanning: 0.5 dag + tests.

### Verbetering #5: Taken uit de gunicorn-worker halen
- **Huidige code**: `app/routes.py:750-752, 1136-1143` (daemon-thread + `subprocess.run(timeout=300)` in request), `app/v2/routes.py:140-186, 374-378` (raadt eigen logfile, `proc.kill()`), `app/v2/api/routes.py:58` (geen timeout)
- **Probleem**: elke config-write → watchdog HUP → worker sterft → thread weg, kind blijft hangen op volle pipe; 2 sync-workers met `timeout=120` → UI bevriest bij twee gelijktijdige runs; check-then-act race op `task_state.json` tussen v1 en v2.
- **Betere aanpak**: de web-UI schrijft een request-bestand (`../data/task_request.json`); de al draaiende `da_scheduler.py` pakt het op (APScheduler-job elke 5 s) en voert het uit, met PID + logfile in `task_state.json`. Status-poll leest alleen dat bestand.
- **Aanbevolen library**: geen extra nodig; `filelock` 3.x voor de state-file.
- **Trade-offs**: één indirectie extra; winst: geen orphan-processen, één uitvoerpad. Inspanning: 1 dag.

### Verbetering #6: Requirements en Dockerfile opschonen
- **Huidige code**: `dao/requirements.txt` regel 8-10, 20, 35; `dao/Dockerfile:22, 46-50`
- **Probleem**: `mysql`, `mysql-connector-python`, `mariadb`, `cffi` ongebruikt (alleen `pymysql`/`psycopg2` via URL-schema's in `db_manager.py:132-148`); `freezegun` is een testdependency; `mariadb` dwingt `gcc/g++/libmariadb-dev` in het runtime-image en de aarch64 `mariadb_config`-symlink.
- **Betere aanpak**: verwijder de vijf packages; verplaats `freezegun`/`pytest` naar `requirements-dev.txt`; laat de compiler-toolchain alleen in een aparte stage voor `use_self_compiled_miplib`.
- **Trade-offs**: kleiner image, snellere build; controleer of `psycopg2-binary` gepind wordt. Inspanning: 2 uur.

### Verbetering #7: pandas-antipatronen vervangen
- **Huidige code**: `df.loc[df.shape[0]] = row` in `da_base.py:405-411`, `da_meteo.py:447-469`, `da_prices.py:96, 146, 189, 275`, `da_report.py:1440, 1480, 1587, 2582, 3165, 3183`, `solar_predictor.py:781-783, 826-830`, `utils.py:225-245`, `day_ahead.py:3585, 3634, 3659, 3719`; chained assignment `solar_predictor.py:1084-1085` (no-op in pandas 3 **[geverifieerd door deelreview]**)
- **Probleem**: O(n²), object-dtypes, 8760-rij rapporten duren seconden tot minuten op een HA Yellow.
- **Betere aanpak**: lijst van dicts → `pd.DataFrame(rows)`; `pd.melt` voor long-format; `pd.date_range(..., freq="15min")` i.p.v. handmatige tijdlijsten; `Series.clip(lower=0).round(3)`.
- **Inspanning**: 1 dag verspreid.

### Verbetering #8: Solar-ML methodologisch repareren
- **Huidige code**: `solar_predictor.py:347` (`resample("h").sum()` maakt 0 van ontbrekende uren), `569-583` (`GridSearchCV(cv=3)` = KFold op tijdreeks, getuned op de oudste 5000 rijen), `641, 729` (pickle van `XGBRegressor`, featurelijst niet opgeslagen), `21` (`from pip._internal.utils import datetime` — privé-import), `32` (`warnings.filterwarnings("ignore")` globaal)
- **Probleem**: systematische onderschatting na omvormer-uitval; optimistische CV; na een xgboost-bump crasht `calc_solar_predictions` (`da_base.py:819` vangt alleen `FileNotFoundError`).
- **Betere aanpak**: `sum(min_count=1)`, `TimeSeriesSplit`, `model.save_model("*.ubj")` + sidecar JSON met features, `except Exception` → fallback naar DAO-predictor.
- **Aanbevolen library**: `sklearn.model_selection.TimeSeriesSplit`, `sklearn.pipeline.Pipeline`, `pvlib` voor clear-sky/AOI-features.
- **Inspanning**: 1-2 dagen.

---

## 3. Library-hergebruik kansen (quick wins)

| Functionaliteit | Huidige implementatie | Aanbevolen library | Besparing | Risico's |
|---|---|---|---|---|
| Cron-scheduler | zelfgebouwd, `da_scheduler.py:56-97` | `APScheduler` 3.11 | 0.5 dag; lost bug #12 op | trigger-semantiek `HHxx` zelf mappen |
| HTTP retry met backoff | while-lus zonder delay, `da_meteo.py:412-423` | `tenacity` 9.x of `urllib3.Retry` op `requests.Session` | 2 uur | geen |
| HA REST-client | `hassapi` 0.3.0 + read-back in `da_base.py:242-256` | `requests.Session` wrapper of `homeassistant-api` 5.x | 1 dag, lost bug #3/#11 deels op | callsite-migratie |
| Upsert per rij | `db_manager.py:284-330` | SQLAlchemy `insert().on_conflict_do_update` / `on_duplicate_key_update` | 3 uur; 300x minder queries | dialect-test op 3 DB's |
| DB-URL opbouwen | f-strings `db_manager.py:129-148` (wachtwoord met `@`/`/` breekt; wachtwoord in DEBUG-log) | `sqlalchemy.engine.URL.create` + `render_as_string(hide_password=True)` | 1 uur | geen |
| DB-schema-migraties | if-ladder op versienummer, `check_db.py` | `Alembic` | 1 dag; lost bug #14 en niet-idempotente `create` op | leercurve |
| Feestdagen NL | hard-coded, `utils.py:24-45`, `baseload.py:86-119` | `holidays` (`holidays.NL()`) | 1 uur; Koningsdag-op-zondag regel gratis | geen |
| Zonnestand/irradiantie | raw `ephem` + eigen splitsing, `da_meteo.py:50-71, 171-191, 267-303`; solstice-only bounds `solar_predictor.py:221-271` | `pvlib` (`solarposition`, `irradiance.erbs`, `get_total_irradiance`, `clearsky.ineichen`) | 1-2 dagen; corrigeert ook de 1800 s-offsetfout `da_meteo.py:182 vs 282` | resultaten wijken bewust af van huidige curve |
| Uur→kwartier-interpolatie | `utils.py:429-516` | `Series.resample("15min").interpolate("pchip")` + gemiddelde-correctie | 2 uur | gedragsverandering op randen |
| Tijdzone | `pytz` + `"CET"` literals, `da_prices.py:40-73` | stdlib `zoneinfo` | 2 uur | geen |
| Atomaire JSON-writes | 5 varianten (zie Verbetering #4) | één helper (`tempfile` + `os.replace`), hergebruik `fastctrl/runner.py:232-251` | 2 uur | geen |
| CSRF / auth | afwezig | `Flask-WTF` `CSRFProtect`; `before_request` ingress-guard | 2 uur | ingress-IP hardcoden |
| CLI-argumenten | positioneel keyword-scannen `day_ahead.py:5259-5294` (elk argv-token is een taak → argument-injectie vanuit `routes.py:733-747`) | `argparse` | 1 uur | geen |
| Statistische helpers | `baseload.py:127-209` (quantile, weighted median, trimmed mean) | `numpy.percentile`, `scipy.stats.trim_mean`, `numpy.average(weights=)` | 2 uur | geen |
| Tibber GraphQL | string-concat + handmatig parsen, `da_prices.py:212-275`, `utils.py:179-250` | `pyTibber` of `gql` met variabelen | 3 uur | API-token scope |
| Log-tail voor UI | `readlines()[-20:]` per seconde, `routes.py:814-818`; hele log per seconde `v2/routes.py:415` | `collections.deque(f, maxlen=20)`; offset-based tail | 1 uur | geen |
| Grafieken thread-safe | pyplot global state, `da_graph.py:19, 96, 99` (+ figure-leak `da_meteo.py:378`) | matplotlib OO-API (`Figure` + `FigureCanvasAgg`) | 3 uur | geen |
| Secrets-type | eigen `SecretStr(str)` die pydantic's schaduwt, `base.py:382-454`; valt terug op de *key-naam* als secret (`base.py:453`) | `pydantic.SecretStr` + kleine `SecretRef` | 2 uur | testupdate |

---

## 4. Architecturale adviezen (lange termijn)

- **Splits `calc_optimum` (5000 regels) in device-modules** met een vaste interface: `build(model, ctx) -> DeviceVars`, `publish(solution, ha)`, `export_plan(...)`. Per device (battery, boiler, ev, heatpump, machines, solar, grid) een eigen bestand met eigen tests. De EV- en boiler-blokken hebben al een natuurlijke grens; begin daar. Verwijder de ~600 regels uitgecommentarieerde alternatieve formuleringen (git heeft de historie).
- **Eén procesmodel voor taken**: `da_scheduler.py` is het enige langlevende proces; web-UI, API en cron sturen alleen verzoeken. Dat elimineert de drie taakregisters (`da_base.py:259-330`, `routes.py:243-301`, `v2/routes.py:339-357`), de `task_state.json`-races en de orphan-processen.
- **Verwijder de v1 web-UI** zodra v2 CO2-rapport en prijsdatum-parameters heeft. Halveert het aanvalsoppervlak (bug #2 zit alleen in v1) en de onderhoudslast.
- **Config als enige bron van waarheid**: laat modellen de sentinel-stages (`battery.py:456-457`, `heating.py:276-279`, `day_ahead.py:1660`) als `@computed_field` leveren i.p.v. ze in de opgeslagen lijst te injecteren; verplaats validatie die nu in `day_ahead.py` staat (stage-volgorde, `reduced_hours`-keys, EV-ampere-volgorde, `reduce_power_*_soc` >= 2 entries) naar pydantic-validators zodat een fout bij het opslaan zichtbaar is, niet om 00:00 in de log.
- **Observability**: log naar stdout (Supervisor verzamelt), geen `TimedRotatingFileHandler` vanuit twee gunicorn-processen (`routes.py:210-221`); `NotificationHandler` en file-handlers netjes verwijderen in een `finally` (`da_base.py:869-927` lekt handlers bij exceptions).

---

## 5. Prioriteitenlijst voor implementatie

**Fase 1 (deze week)** — security en "de sturing valt stil"
- Fix bug #1: symlink uit `static/`, dedicated `/images/<name>`-route
- Fix bug #2: `ports:` verwijderen uit `config.yaml`, ingress-guard, whitelist in `settings()`, `abort(404)` op `/api/run/<x>`, CSRF
- Fix bug #3: `get_float()`-helper in `DaBase` + `FlexValue.resolve(default=...)`; alle 39 callsites in `day_ahead.py`
- Fix bug #11: try/except per device-blok, batterij eerst
- Fix bug #9 en #10: machines-indexering, EV `charge_scheduler` verplicht
- Fix bug #4: 15-min epoch in `db_manager.py:498`
- Fix bug #15: `self.config = None` vóór de lock
- Fix bug #16: `timeout=(5,30)` + `raise_for_status()` op de 5 HTTP-calls
- CI: `python -m pytest dao/tests` toevoegen aan `test_build.yaml`; bug #18 (2 falende tests) oplossen

**Fase 2 (volgende 2 weken)** — data-integriteit
- Fix bug #5, #6 (+ `v0_to_v1` meteoserver-key) met loader-niveau regressietests op `options_example.json`
- Fix bug #7 + implementeer Verbetering #4 (`config_io`, `exclude_unset`, `by_alias`, geen input-mutatie)
- Fix bug #8: dialect-upsert in `savedata`; `URL.create`
- Fix bug #13, #14: consolidatie en PostgreSQL-DDL; `checkfirst=True`
- Fix bug #17: `da_prices` zonder `sys.argv`, ENTSO-E `resolution`, Nordpool-check
- Implementeer Verbetering #1: APScheduler (lost #12 op)
- Implementeer Verbetering #6: requirements/Dockerfile opschonen
- Vervang feestdagen door `holidays`, retries door `tenacity`, `pytz` door `zoneinfo`

**Fase 3 (lange termijn)**
- Verbetering #3: epoch-in/epoch-uit door `db_manager.py` en `da_report.py` (lost de DST-500's, PostgreSQL-verschuiving en de rapportfouten uit de bijlage op)
- Verbetering #2 + #5: eigen `HAClient`, taken via het scheduler-proces, v1-UI uitfaseren
- Verbetering #7 + #8: pandas-vectorisatie, ML-pipeline met `TimeSeriesSplit` en `save_model`
- Architectuur: `calc_optimum` opsplitsen per device; `pvlib` voor zonnestand; Alembic voor schema

---

## Bijlage: overige bevindingen (medium/low, niet blokkerend)

Alles hieronder is opgelost, op de drie expliciet gemarkeerde dode-code-gevallen na (die kosten niets om te laten staan: ze worden nooit uitgevoerd).

| Locatie | Probleem | Fix | Status | Commit |
|---|---|---|---|---|
| `day_ahead.py:688` | `solar_name` zonder `.replace("-", "_")` terwijl regels 349/357/393 dat wel doen → `KeyError` bij batterij-solar met `-` in de naam | zelfde normalisatie | opgelost | `8a01960` |
| `day_ahead.py:2330-2335` | Warning zegt "uitgegaan van 1,5 kW-e" maar `hp_power` wordt niet aangepast | `hp_power = 1.5` in de if-tak | opgelost | `6112eeb` |
| `day_ahead.py:624-626` | `sum_eff / (DS[b] - 1)` → `ZeroDivisionError` als alleen de 0-stage bestaat (model staat dat toe) | validator: minstens één stage met `power > 0` | opgelost | `15f6b12` |
| `day_ahead.py:2645` | `hp_hours / hours_avail` → `ZeroDivisionError` als `boiler_int >= U` | `if hours_avail <= 0: blocks_num = 0` | opgelost | `8a01960` |
| `day_ahead.py:3013-3015` | `delta.seconds / 900` i.p.v. `total_seconds()` **[geverifieerd]** (alleen fout bij window >= 24 h) | `math.ceil(delta.total_seconds() / 900)` | opgelost | `15f6b12` |
| `day_ahead.py:1645-1648` | EV ready-tijd exact gelijk aan nu wordt niet naar morgen geschoven → "verouderd" | `<=` i.p.v. `<` | opgelost | `6112eeb` |
| `day_ahead.py:3199` | `cost` gebonden op ±1000 euro; `delivery/production` op 1000 kWh — hard-coded | dynamische grenzen uit netaansluiting/horizon | opgelost | `15f6b12` |
| `day_ahead.py:4929, 5054, 5075, 5087, 5108` | Muteren `uur`, `pl`, `pt`, `p_spot`, `pl_avg` in de grafiekcode | kopieën gebruiken (rebind i.p.v. append) | opgelost | `15f6b12` |
| `day_ahead.py: df_accu/df_soc/df_pv_dc/d_f` | `.loc[df.shape[0]] = row` per interval in `calc_optimum` (O(n²), tot 96 rijen) | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `f6b3e2f` |
| `da_base.py:242-256` | Read-back na `set_value` faalt bij device-backed entities → valse errors | `set_value` waarschuwt i.p.v. raise bij read-back-mismatch | opgelost | `6112eeb` |
| `da_base.py:594-608` | `os.chdir` zonder `try/finally` → cwd blijft fout na exception | `Path(folder).glob(pattern)` | opgelost | `d7f8750` |
| `da_base.py:283` | `"cmd": ["python3", "day_ahead.py", "meteo"]` mist `../prog/` → `/v2/api/run/meteo` 500 | pad corrigeren | opgelost | `6112eeb` |
| `da_base.py: save_df/calc_solar_predictions` | `.loc[shape[0]] = row` per (interval, kolom)-paar resp. per interval (O(n²)) | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `101e5c6` |
| `utils.py:74-83` | `get_value_from_dict`: datum vóór eerste key → wrapt naar laatste entry; keys niet gesorteerd/gevalideerd | validator in `pricing.py` (ISO-datum, gesorteerd), clamp naar eerste i.p.v. wrap naar laatste | opgelost | `d7f8750` |
| `utils.py: get_tibber_data` | `.loc[shape[0]] = row` per node/veld-paar (O(n²)) | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `d5a4f69` |
| `fastctrl/runner.py:533, 695` | Plan verouderd wordt gerapporteerd als `sensor_stale` | eigen reden `plan_stale` | opgelost | `547e5b1` |
| `fastctrl/runner.py:557-562` | `saved_today_eur` telt geschatte baten ook in shadow-modus | `saved_today_is_estimate`-vlag | opgelost | `547e5b1` |
| `da_fast.py:103-104, 133, 151` | `.value` als literal (`float("input_number.x")`, `bool("False") is True`) | `.resolve(report.ha_getter)` | opgelost | `547e5b1` |
| `da_meteo.py:182 vs 282` | Direct en diffuus op verschillend tijdstip geëvalueerd (start vs midden uur) → PV te laag bij zonsop-/ondergang | beide op hetzelfde moment (interval-midden via `self.interval_s/2`) | opgelost | `c7f8222` |
| `da_meteo.py:653, 696` | `get_avg_temperature` returnt `None` bij lege data → `TypeError` in `calc_graaddagen` → optimizer-run dood | `None` afhandelen, bovengrens op query | opgelost | `c7f8222` |
| `da_meteo.py: get_meteo_data` | `.loc[shape[0]] = row` × 4 per rij (O(n²)); een tweede vergelijkbare lus bleek dode code (gfs-fallback in een niet-uitgevoerde `"""`-string) | lijst van tuples + één `pd.DataFrame(...)`; dode code ongemoeid gelaten | opgelost | `354a675` |
| `da_prices.py: entsoe/easyenergy/tibber` | `.loc[shape[0]] = row` per uurprijs (O(n²)), drie plekken | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `354a675` |
| `da_report.py:54` | `periodes = {}` als class-attribuut, per instance gemuteerd | `self.periodes = {}` in `__init__` | opgelost | `db08de4` |
| `da_report.py:73, 1996` | `co2_intensity_sensor` (str) wordt als lijst geïtereerd → CO2 altijd 0 | `[sensor] if sensor else []` | opgelost | `db08de4` |
| `da_report.py:1008-1019, 1044` | Aggregatie-query selecteert niet-gegroepeerde kolommen → faalt op PostgreSQL/MySQL 8 | `func.min(...)`, `func.max(...)` | opgelost | `5cce551` |
| `da_report.py:2024-2032` | Prijzen positioneel gejoind i.p.v. op tijd | `merge(on="tijd")` | opgelost | `db08de4` |
| `da_report.py:2217-2221` | `last_moment = vanaf` bij lege HA-rijen → dubbele uren in grid-rapport (eerste 12 min van elk uur) | `else`-tak verwijderd | opgelost | `db08de4` |
| `da_report.py:1711-1719, 1948` | Bucket-label = `min(time)` i.p.v. bucketstart → eerste deelmaand van "contractjaar"/"365 dagen" valt weg | `month_start`/`day_start`/`hour_start` in SQL, niet overschrijven | opgelost | `db08de4` |
| `da_report.py:3430-3431` | `tz_localize` zonder `ambiguous`/`nonexistent` → 500 op beide DST-dagen | `ambiguous=False, nonexistent="shift_forward"` | opgelost | `5cce551` |
| `da_report.py: recalc_df_ha/aggregate_balance_df/calc_grid_columns/get_price_data/calc_solar_data` | `.loc[shape[0]] = row` per rij (O(n²), tot 8760 rijen voor een jaarrapport) | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `44d8e6e` |
| `solar_predictor.py:21` | `from pip._internal.utils import datetime` | verwijderd | opgelost | `81d9154` |
| `solar_predictor.py: import_weatherdata/get_and_save_knmi_data` | `.loc[shape[0]] = row` × 3 per rij (O(n²), ~26.000 brondata voor 3 jaar KNMI-import) | lijst van tuples + één `pd.DataFrame(...)` | opgelost | `b2302d2` |
| `solar_predictor.py: resample/cv/warnings/model-opslag` | zie Verbetering #8 | zie Verbetering #8 | opgelost | `81d9154`, `d0df427`, `101e5c6` |
| `check_db.py:150-167, 296-301` | KNMI-observaties overschrijven forecasts in `prognoses` en worden uit `values` verwijderd, terwijl `solar_predictor.py:929` daar nog leest | filtert al bestaande timestamps eruit vóór de upsert | opgelost | `81d9154` |
| `db_connections.py:125-147` | Singleton pint de eerste config; `_build_db_da` gooit `OperationalError` i.p.v. `None` zoals docstring belooft | cache op resolved `db_url`; `try/except` retourneert `None` | opgelost | `03ac313` |
| `models/grid.py:13-22` | `max_power` default 17 kW ook zonder `grid`-sectie (1-fase 25 A = 5.75 kW) | warning bij gebruik van de default (`model_fields_set`-check) | opgelost | `2af258b` |
| `models/scheduler.py:57-78`, `battery.py:78` | `extra="ignore"`/`"forbid"` terwijl de rest `allow` is → `//comment`-keys verdwijnen of breken de config | `extra="allow"` | opgelost | `2af258b` |
| `models/base.py:446-454` | `SecretStr.resolve()` valt terug op de key-naam als secret → misleidende "access denied" | `KeyError` | opgelost | `2af258b` |
| `app/__init__.py:21` | `app.secret_key = "secret_cookie_key"` | `secrets.token_hex(32)`, persistent | opgelost | `14337f8` |
| `templates/fast_control.html:14, 16, 45` | Absolute URL's (`/run`, `/fast_control/state.json`) werken niet onder ingress | `url_for(...)` | opgelost | `14337f8` |
| `v2/api/routes.py:26, 83` | `request.args.get('timezone') if None else "Europe/Amsterdam"` → altijd Amsterdam | `_requested_timezone()`-helper: `or` i.p.v. de omgekeerde ternary, valideert tegen `zoneinfo.available_timezones()` | opgelost | `2af258b` |
| `routes.py:482 → 385-391` | Fast-control statuspoll (elke 5 s) draait `ConfigurationLoader.load_and_validate()` met `flock` en kan vanuit een GET `options.json` herschrijven (migratiepad) | `_cached_config_v1()`: GET's hergebruiken de gecachte config, POST herlaadt altijd | opgelost | `2af258b` |
| `webserver: run/task-runner` | Cancel/kill raakte alleen het directe kind; een signaal aan de webserver zelf kon de v1-taak halverwege raken | `start_new_session=True` + `os.killpg()` in v2's cancel-pad | opgelost | `28b3da5` |
| `da_meteo.py: gfs-fallback` (dode code) | tweede `.loc[shape[0]]`-lus | staat in een niet-uitgevoerde `"""`-string; bewust ongemoeid | n.v.t. | — |
| `day_ahead.py: df_pv_prog` (dode code) | derde `.loc[shape[0]]`-lus | staat in een niet-uitgevoerde `"""`-string; bewust ongemoeid | n.v.t. | — |
| `utils.py: interpol_rows` (dode code) | vierde `.loc[shape[0]]`-lus | wordt alleen aangeroepen door het nergens meer gebruikte `interpolate_old`; bewust ongemoeid | n.v.t. | — |
