# Settlement-publicatiegate — gerichte A7/A10-amendment

**Status:** architectcorrectie, 31 oktober 2026; uitsluitend planwerk op HEAD `94dc319ba7cbb389f16c94604d1312be7e34a8ab`. Geen implementatie, testresultaat of activeringsgoedkeuring.
**Doel:** de ontbrekende producer-/consumergrens vóór A7 leveren zonder G1–G3 naar de afrekenmodule te verplaatsen.
**Voorrang:** leidende aanvulling op uitsluitend A7/A10 van `docs/superpowers/plans/2026-10-13-production-bonus.md` en ontwerp §3/8/9 in `docs/superpowers/specs/2026-10-13-production-bonus-design.md`. A1–A6 blijven gerealiseerde basis; historische Tasks 1–10 niet uitvoeren.
**Uitvoering:** controller synchroniseert afgeleide SDD-briefs; verse implementatietaak per onderstaande commitgrens, TDD en review volgens bestaande uitvoering. Deze architect maakt geen commits.

**Gerichte vervolgcorrectie:** A10a.1/.2 zijn gerealiseerd op codeHEAD `9819202`. Voor de nog ontbrekende scheiding tussen geregistreerde horizonprijzen en gesloten meet-/afrekeninvoer is uitsluitend `docs/plans/2026-10-31-settlement-price-input-amendment.md` leidend: A7p vóór A7, één additieve prijstabel en gedeelde globale revisie. Dit vervangt op dat punt de eerdere “geen schemawijziging”-afbakening; slot/guard/fallback/harddeny blijven behouden. Onderstaande gerealiseerde A10a-taken niet opnieuw uitvoeren.

## 1. Probleem, alternatieven en besluit

A7 vraagt een lokale runnerinvalidatie en revisiehercontrole onder hetzelfde slot als inputregistratie. De huidige `fastctrl/plan.py` kent alleen v1; `runner.tick()` keert bij `plan is None` terug zonder actuatoractie. `day_ahead.calc_optimum()` schrijft bovendien boiler/grid/EV/PV/batterij/warmtepomp/apparaatcommando's vóór `export_fast_plan()`. Een exception in `Report.get_price_data()` of unlink van het plan voorkomt daarom geen voortgezette oude aansturing.

1. **Alleen A7 weigeren/v1-marker toevoegen:** kleine diff, maar geen consumptie bij de runner, geen betrouwbare revisiegrens en een tweede tijdelijk protocol. Valt af.
2. **Heel A10 vóór A7:** één protocol, maar trekt terminalwaarde, replay en A8/A9-afhankelijkheden naar voren en introduceert opnieuw een cyclus. Valt af.
3. **A10 splitsen — gekozen:** A10a levert eerst het v2-veiligheidsomhulsel, slot, registratiehook, deny-gate en fallbackverzoek; A7 consumeert die interfaces. A10b vult op zijn bestaande plaats de economische payload/policy/replay in. Kleine additieve interfaces rond A6, geen ander boek/model of nieuwe servicearchitectuur.

**Volgorde:** A1–A6 (niet herhalen) → **A10a.1 → A10a.2 → A7p → A7 → A8/A9 → A10b → A11**. A10a.1/.2 zijn gerealiseerd; A7p levert nu het aanvullende prijs-/meetgrensvlak. G1–G3 blijven aparte activeringsgates uit `2026-09-24-fast-control-ui.md`.

**Bewijsgrens:** softwarematig blokkeren en een fallbackverzoek consumeren is niet fysiek stoppen. A10a/A7 leveren géén bewezen hardwarefallback, eigenaarsoverdracht of recht op economische actuatie. Zolang G1–G3 ontbreken blijft nieuwe financiële actuatie hard uit, óók bij active/confirmed. A10b mag dit niet inschakelen. Een reeds aangestuurde installatie mag niet naar deze tussenstand worden uitgerold zonder afzonderlijk bewezen stop-/eigenaarschapsroute. Geen nieuwe runtime-instelling waarmee deze blokkade te omzeilen is.

## 2. Constraints en gerichte bestanden

- Python 3.12 runtime; `dao/prog/config/` blijft importeerbaar onder Python 3.11 met alleen Pydantic 2. Geen nieuwe dependencies; Linux `fcntl.flock`, stdlib-dataclasses en bestaande SQLAlchemy Core volstaan.
- Bestaande Decimal-stringinvoer, precisie36, 9/12 invoerdecimalen en uitsluitend presentatieafronding blijven gelden. Geen floatprijs of geschatte bonus als grondslag gebruiken.
- Geen HA-writes in tests; fake gateways/sinks, tijdelijke bestanden en tijdelijke test-DB's. Geen live/SSH/productie-DB, echte hardware, dependency-installatie, commit/push/merge tijdens dit planwerk.
- Geen wijziging van MIP-, service-/API-/CLI-functionele architectuur, schema of persoonlijke instellingen. Nieuwe configvelden zijn niet nodig; bij onverhoopte configwijziging eerst scopecheck en daarna docs/schema plus echte lichte 3.11-importcheck.

| Eigenaar | Bestanden / verantwoordelijkheid |
|---|---|
| A10a.1 | `dao/prog/fastctrl/plan.py`: strikt v2-omhulsel; nieuw `dao/prog/fastctrl/publication.py`: pad, slot, atomische publicatie, lokale verdicts en fallbackprotocol (geen DB/pandas/mip-imports) |
| A10a.1 | `dao/prog/settlement/repository.py`: kleine optionele registratieguard + actuele boekreferentie; bestaande CAS/immutable bedragen ongewijzigd; nieuw `dao/prog/settlement/publication.py`: DB-/configadapter op het lokale protocol |
| A10a.2 | `dao/prog/fastctrl/runner.py`: invalidatie/lease vóór vroege returns consumeren; `dao/prog/day_ahead.py`: preflight en uiteindelijke exportgate, alle directe economische uitvoer geblokkeerd bij nieuw regime |
| A7 | Bestaande A7-bestanden; `service.py` levert gepinde planningcontext en gebruikt registratieguard. `da_base.py` bouwt één gedeelde adapter voor scheduler/CLI/planner. `da_report.py` blijft prijs-/rapportadapter, geen verborgen actuator |
| A10b | Oorspronkelijke `fastctrl/{policy,simulate}.py`, economische v2-payload in `plan.py`, `day_ahead.py`, replaytests. Geen tweede slot/capabilityprotocol |

## 3. A10a.1 — protocol en gedeelde publicatiegrens

### 3.1 Eén bestand, expliciete v2-toestanden

Behoud `FAST_PLAN_FILE = "../data/fast_plan.json"`. Geen apart capabilitybestand, v1-marker of leeg geldig plan. `plan.py` introduceert deze immutable DTO's:

```python
FinancialBasis(meter_id: str, input_revision: int,
               snapshot_hashes: tuple[str, ...], prefix_start_ts: int,
               planning_start_ts: int, horizon_end_ts: int,
               captured_ts: int, source_watermark_ts: int | None)
FinancialContext(basis: FinancialBasis, not_after_ts: int,
                 price_interval_s: int, terminal_value: TerminalValue,
                 economics_supported: bool, issues: tuple[str, ...])
BlockedPlan(format_version: int, created_ts: int, reason: str)
```

`TerminalValue` is de bestaande DTO uit `settlement/types.py`, geen duplicaat. `FastPlan` krijgt `financial_context: FinancialContext | None`. V2-JSON heeft een verplichte discriminator `status: "ready"|"blocked"`:

- `ready`: volledige FastPlan-payload plus volledige `financial_context`; geen deels gevulde ready-v2. `format_version=2`, `status="ready"` worden door serialisatie geschreven. Bestaande v1-builders blijven expliciet v1 totdat A10b een volledige financiële payload produceert; een bump van alleen de default is fout.
- `blocked`: uitsluitend `format_version=2,status="blocked",created_ts,reason`. Geen economische intervallen, financiële defaults of actuator-entiteiten. Ook zonder bruikbaar oud plan te schrijven.
- `FinancialBasis.input_revision > 0`; expliciete meteridentiteit, unieke canonieke SHA-256-hashes en `prefix_start_ts <= planning_start_ts < horizon_end_ts`. Prefix dekt alle daadwerkelijk gebruikte cap-/maandhistoriek; bij onbekende geschiedenis conservatief vanaf contract-/relevante periodebegin, niet vanaf laatste recorderdag.
- `not_after_ts = min(horizon_end_ts, eerstvolgende contractgrens, created_ts + max_plan_age)`. Niet vernieuwen door lezen/ticken. `now_ts >= not_after_ts` is ongeldig. Future-created/teruglopende klok is geen leaseverlenging maar blokkade.
- V2 valideert types zonder coercion (`bool` is geen int), enumwaarden, arrays/objects, tijden, finite numerieke waarden, verplichte rates, batterijaantallen en intervaldekking. Decimal-terminalwaarde canoniek als tekst. Geen `_f(..., 0)` voor financiële v2-velden. Onbekend formaat/top-level lijst/null/NaN is onbruikbaar, niet legacy.

Nieuwe interfaces in `fastctrl/publication.py` (DTO's hieronder eveneens daar):

```python
GatePolicy(required_from_ts: int | None)  # geconfigureerde eerste Budget-ingang, niet mode
PlanRead(status: Literal["ready", "blocked", "missing", "invalid"],
         plan: FastPlan | None, reason: str)
GateVerdict(allowed: bool, reason: str)
PublicationLockBusy(Exception)
PublicationError(Exception)

canonical_plan_path(path: str | Path) -> Path
gate_policy(pricing: PricingConfig | None) -> GatePolicy
publication_lock(path: str | Path) -> ContextManager[PublicationLease]
read_publication(path: str | Path) -> PlanRead
invalidate_plan(*, lease: PublicationLease, now_ts: int, reason: str) -> None
write_publication(plan: FastPlan, *, lease: PublicationLease) -> None
following_verdict(read: PlanRead, *, policy: GatePolicy, now_ts: int) -> GateVerdict
```

`PublicationLease` is een niet-zelf te construeren, process/thread/path-gebonden bewijs van het gehouden slot. Schrijven na context-exit/met ander pad is een fout. `gate_policy` is puur: neem de vroegste expliciete Budget-valid_from als lokale Amsterdam-middernacht omgerekend naar UTC, onafhankelijk van mode/status; geen contract betekent None. `read_publication` valideert alles, geen mtime-cache als autorisatie. Bestaande `load_plan()` mag als compatibility wrapper ready→plan, anders None blijven, maar de runner gebruikt voor veiligheid uitsluitend de nieuwe read/verdict-route. Bestaande `write_plan()` wordt een slotnemende compatibility wrapper voor v1; hij mag een blocked/v2-artefact niet overschrijven. Ready-v2 kan uitsluitend via de settlement-publicatieadapter. De compatibility wrappers importeren hun I/O-helper lokaal om de `plan`/`publication`-importcyclus te vermijden.

**Slotbesluit:** canoniek absoluut planpad, vaste sibling `Path(str(plan_path) + ".lock")`, dus standaard `fast_plan.json.lock`. Pad éénmaal vanuit dezelfde DAO-datadirectory door bootstrap bepalen/doorgeven; geen onafhankelijk cwd-afgeleid slot in CLI/runner. `flock(LOCK_EX | LOCK_NB)` plus een process-lokale per-pad threadlock; contention betekent afbreken/volgende job of tick, niet onbegrensd wachten. Geen reentrancy/nestende lockacquisitie. Lockfile nooit unlinken/vervangen; geen PID/stale-timeout-locks. Eén host/lokaal filesystem en meewerkende schrijvers zijn expliciete voorwaarden; NFS/multi-host/directe SQL-writers zijn niet ondersteund door deze garantie.

Canonicaliseer de parentdirectory, behoud de vaste bestandsnaam; plan/lock mogen geen symlink zijn. Gebruik een uitsluitend door vertrouwde DAO-processen schrijfbare datadirectory, geen pad uit API-invoer. Het slot is geen beveiliging tegen een lokale gebruiker die deze directory kan vervangen.

Atomisch schrijven met unieke tempfile in dezelfde directory, `allow_nan=False`, file flush/fsync, `os.replace`, directory-fsync; fouten niet slikken zoals de huidige `write_plan` bij directory-fsync doet. Invalidatie is duurzaam afgerond vóór DB-publicatie. Fout/lockbusy ⇒ geen DB-revisie of nieuw plan publiceren. Geen bronnen ophalen, solver, historische replay of HA-I/O onder het korte publicatieslot.

### 3.2 Additieve A6-aansluiting, geen nieuw boek

In `repository.py` komen uitsluitend deze DTO/Protocol-interface en boeklezing erbij:

```python
InputChange(meter_id: str, previous_revision: int, next_revision: int,
            changed_intervals: tuple[tuple[int, int], ...],
            changed_prices: tuple[tuple[int, int], ...],
            changed_contracts: tuple[tuple[int, int | None], ...])
BookReference(input_revision: int, snapshot_hashes: tuple[str, ...],
              dirty_from_ts: int | None)
class InputPublicationGuard(Protocol):
    def hold(self) -> ContextManager[None]: ...
    def before_revision(self, change: InputChange) -> None: ...

SettlementRepository(engine: Engine, *, publication: InputPublicationGuard | None = None)
SettlementRepository.read_book_reference(meter_id: str) -> BookReference
```

`read_book_reference` leest één boekrij/consistent referentiebeeld; ontbrekend boek is revision0/lege hashes. Bestaande public signatures/dual-CAS blijven verder gelijk. `register_inputs` neemt `publication.hold()` vóór `_ensure_book`/DB-lock en houdt het tot DB-commit/rollback. Na bestaande inhoudsvergelijking, vóór input-/tariefinserts en revisie-update, roept het alleen bij werkelijk gewijzigde inhoud `before_revision(change)` aan. De callback mag geen repository terug aanroepen. Bij no-op geen invalidatie. Wijzigingsberekening gebruikt dezelfde canonieke inhoud als A6; `changed_prices` alleen bij prijs/prijsbron/prijsqualiteitswijziging, inclusief nieuwe prijsdekking; contractranges uit oude/nieuwe selectie. Een geldige hook is geen vervanging van DB-CAS.

`publication=None` blijft uitsluitend voor geïsoleerde repositorytests/offline boeken zonder runner. Alle applicatiewriters via A7-bootstrap (CLI `--publish`, scheduler, actieve bronrefresh) krijgen de guard, óók in shadow. Geen productiecodepad mag ongemerkt een onbewaakte tweede repository maken. `preview` gebruikt geen registratie en heeft geen file-/DB-writes. `commit` verandert geen inputreferentie en houdt zijn bestaande dual-CAS; hiervoor is geen extra filesystemtransactie nodig.

`settlement/publication.py` produceert:

```python
FinancialPlanBlocked(Exception)
SettlementPublication(repository: SettlementRepository, *, plan_path: Path,
                      pricing: PricingConfig, max_plan_age: int)
SettlementPublication.registration_guard() -> InputPublicationGuard
SettlementPublication.policy() -> GatePolicy
SettlementPublication.check_basis(basis: FinancialBasis, *, now_ts: int) -> None
SettlementPublication.publish(plan: FastPlan, *, now_ts: int) -> None
SettlementPublication.block(*, now_ts: int, reason: str) -> None
make_publication(engine: Engine, *, plan_path: Path, pricing: PricingConfig,
                 max_plan_age: int) -> tuple[SettlementRepository, SettlementPublication]
```

`make_publication` lost de constructiecyclus op met twee repositoryfacades over dezelfde engine: één uitsluitend voor referentielezingen door `SettlementPublication`, één schrijvende facade met `publication=adapter.registration_guard()` voor service. Alleen die laatste en de adapter worden teruggegeven; geen mutabele late-binding callback of globale singleton. Dit is een A10a.1-factory: A10a.2 kan hem reeds vanuit DaCalc met `self.db_da.engine` gebruiken, zonder de nog ontbrekende A7-service/DaBase-bootstrap. A7 gebruikt dezelfde factory vanuit zijn applicatiebootstrap. Aanmaken doet geen schema-/DB-/bestand-I/O. De guard houdt zijn lease process/thread-gebonden alleen binnen `hold()`; `before_revision()` zonder geldige lease faalt.

Relevantie: invalideer ready-plan wanneer gewijzigde intervallen de gebruikte `[prefix_start_ts,planning_start_ts)` raken, of gewijzigde prijs-/contractranges die prefix of de horizon raken. Een gewone nieuwe meting vanaf planningstart zonder prijs-/contractwijziging invalideert een reeds uitgegeven prognose niet. Bij onbekend/onleesbaar artefact of onbeslisbare verandering conservatief blocked schrijven. Een v1-plan bij een geconfigureerde reeds ingegane gate wordt eveneens blocked. Bij uitsluitend toekomstig contract mag bestaande legacy vóór de grens blijven werken; geen v1-rechten ná de grens.

`check_basis` neemt het slot en leest een verse boekreferentie; `publish` doet **zelf opnieuw** diezelfde check onder zijn eigen slot en houdt dat tot de volledige atomic replace. Interne `_check_basis_locked(basis, *, lease, now_ts)` voorkomt geneste locks. Exacte positieve inputrevisie én canonieke snapshotselectie moeten gelijk zijn. Iedere revisiewijziging tijdens rekenen weigert een kandidaat, ook wanneer de wijziging voor een reeds uitgegeven plan niet relevant was. Geen token achteraf herlabelen met een nieuw revisienummer. Een bruikbare opnieuw berekende/geverifieerde basis is nodig. A6 contract-ABA/no-op-hervalidatie blijft geldig: gebruik boek/view-validatierevisie, niet de oude immutable run-/lijnrevisie.

`check_basis` controleert daarnaast toepasselijk confirmed-contract, `mode=active` en contractdekking van de opgegeven horizon; het kent nog geen terminalwaarde/planlease. `publish` controleert bovendien de volledige v2-payload, basisprijzen en onverlopen/binnen de bovengrens gelegen lease. A7 controleert quote-dekking vóór het maken van de context. Dirty oude regels niet als actuele prefix consumeren; onbekende bonusprefix mag expliciet onbekend blijven, zonder positieve bonusrechten. Publicatiehashes moeten exact de gepinde selectie uit de berekening zijn. Een configwijziging vergt gecontroleerde procesherstart vóór actieve publicatie; geen claim van automatische synchronisatie met een willekeurig gewijzigde options.json. Herstart leest nieuwe gatepolicy; geen runtime downgrade van required naar legacy zonder operationele stop.

Bij weigering onder het slot: blocked-artefact schrijven, daarna `FinancialPlanBlocked`; invalidatieschrijffout als `PublicationError` doorgeven. Nooit een bestaand plan opnieuw geldig verklaren bij exception. Bij lockbusy geen mutatie: kandidaat wordt geweigerd, bestaande lease is niet verlengd; runner kan geen nieuwe financiële writes doen.

## 4. A10a.2 — actuele producer/consumer verbinden, zonder G1 te claimen

### 4.1 Runner en fallbackverzoek

In `fastctrl/publication.py`:

```python
FallbackRequest(reason: str, requested_ts: int,
                power_w: int = 0, mode: Literal["configured_off"] = "configured_off",
                inhibit_restart: bool = True)
FallbackReceipt(status: Literal["unconfirmed", "requested", "acknowledged"], reason: str)
class FallbackSink(Protocol):
    def request(self, request: FallbackRequest) -> FallbackReceipt: ...
```

`FastControlRunner.__init__` behoudt bestaande parameters en voegt keyword-only `gate_policy: GatePolicy | None=None, fallback: FallbackSink | None=None` toe. Bootstrap leidt ontbrekende policy af uit gevalideerde `hass.config.prices.settlement`, inclusief off/shadow/draft en toekomstige ingangsdatum. Ontbrekende configuratie is iets anders dan onleesbare configuratie; laatstgenoemde blokkeert initialisatie, geen permissieve default. Tests kunnen policy expliciet injecteren.

Nieuwe `FastControlRunner._consume_block(verdict: GateVerdict, *, now_ts: int) -> FallbackReceipt`:

1. Zet lokale financiële bevoegdheid uit en verwijder gecachte economische beslissing. Geen policy-release die het oude planvermogen terugzet, geen benefit-accounting als uitgevoerd.
2. Geef `FallbackRequest(reason=verdict.reason, requested_ts=now_ts, power_w=0, mode="configured_off", inhibit_restart=True)` via het benoemde sink door; ontleen entiteiten/stopgedrag **niet** aan corrupt/blocked plan. De bestaande `_actuate` is niet deze fallbackadapter.
3. Zonder sink: expliciete `unconfirmed`-receipt en gelogd alarm. Een sinkexception/ongeldige receipt is ook unconfirmed, niet tickcrash gevolgd door permissieve herstart. Geen HA-aanroep via een geïmproviseerde nulsetpoint-route.
4. Hoogstens één request per tick; bij herhaling niet vaker dan `max(1, config.interval)` seconden op monotone tijd. Geen snelle retryloop. Status/laatste receipt lokaal beschikbaar als `runner.last_fallback`; verzoek blijft pending zolang geen afzonderlijk responsbewijs bestaat. `acknowledged` betekent uitsluitend gateway-acceptatie, nooit gemeten stilstand.

`tick()` doet de lokale gatecontrole vóór de huidige `plan is None`, mode-off, sensor- en verlopenplanreturns: neem hetzelfde `publication_lock(self.plan_path)`, lees met `read_publication`, bepaal `following_verdict` en geef het slot vrij vóór sink/sensor-I/O. Lockbusy/readfout wordt een blocked verdict, geen gebruik van het gecachte plan. Missing/corrupt/blocked/verlopen v2, of v1 vanaf required_from, gaat naar `_consume_block`; geen oude setpointrestore. Off/shadow kan een verzoek registreren, maar de standaard sink doet geen HA-writes. Ook bij restart zonder oud planspec bestaat dus een expliciete blocked/unconfirmed-toestand. Een werkelijke G1-sink gebruikt later eigen gevalideerde apparaatconfig en eigenaarstoestand; shadowconsolidatie krijgt zo niet heimelijk actuatietaken. Dit vroege readslot is nadrukkelijk niet de nog ontbrekende G1-serialisatie tot en met commandosend.

**Tijdgate onafhankelijk van optimizer:** v1 vóór toekomstige Budget-ingang blijft legacy; exact op ingang wordt v1 geweigerd, ook zonder nieuwe calc/schedulerjob. Na datumgate geen fallback naar legacy doordat planbestand verdwijnt of v1 wordt teruggezet. Een blocked-artefact blokkeert ook vóór die datum. Voor bestaande configuratie zonder settlement blijft de gewone legacyroute compatibel; dat is geen financiële/hardwareveiligheidsgarantie voor legacy.

**Harde tussenstand:** `following_verdict` weigert nieuw-regime-actuatie met `actuation_gates_unproven`, ook bij synthetisch geldige ready-v2. A10a en A10b bevatten geen productiepad om dit open te zetten. Alleen de afzonderlijke G1–G3-uitvoering mag die codeblokkade na haar bewijzen vervangen. Daardoor is de huidige check→HA-write-race niet zogenaamd opgelost: voor nieuw regime bestaat nog geen toegestane niet-nulroute.

### 4.2 Optimizer en directe writes

Voeg in `DaCalc` toe:

```python
DaCalc._financial_preflight(*, planning_start_ts: int, horizon_end_ts: int) -> None
DaCalc._check_financial_output(*, basis: FinancialBasis | None, now_ts: int) -> None
```

Preflight gebruikt de expliciete datumgate (niet alleen settlement.mode). Vanaf ingang weigert A10a de bestaande legacy-objective met `FinancialPlanBlocked`, schrijft lokaal blocked via de adapter en start geen uitvoerpad. Geen nieuwe quotes/MIP nu invoeren. A7 levert de voorbereiding hieronder; A8/A9/A10b vervangen later uitsluitend de rekengrondslag, niet de gate. Bij A10a ontbrekende A7-planningservice is de nieuwe route dus dicht, niet legacy.

Een legacyhorizon die de toekomstige contractingang overschrijdt mag geen opdrachten/start-stophelpers voorbij die grens publiceren. In deze prerequisite wordt zo'n gemengde legacyhorizon conservatief geweigerd (niet stil op nieuwe tarieven doorgerekend); `horizon_end_ts == required_from_ts` mag nog wel. Dit kan planning vlak vóór de ingang eerder blokkeren en wordt expliciet gelogd. Zonder deze begrenzing zouden vooraf geschreven apparaathelpers de datumgate kunnen omzeilen. De uitvoercheck herbeoordeelt ook de actuele klok: een vóór de grens begonnen berekening krijgt na de grens geen legacy-uitvoerrecht.

`_check_financial_output` komt vóór de eerste financieel bepaalde apparaatactie, buiten de losse `try/except`-blokken. Het toetst de vastgelegde basis via `check_basis` en weigert vervolgens nieuwe hardware-uitvoer zolang G1–G3 ontbreken. Alle financieel bepaalde writes vallen eronder, niet alleen batterijwrites: boiler, grid, EV, PV/curtailment, batterij, warmtepomp, apparaten, geplande start/stophelpers. Inventariseer bovendien de eerdere `self.set_value(entity_avg_temp, ...)` in de solve-opbouw; bij de nieuwe geblokkeerde route ook geen dergelijke HA-writes. Test via fake HA-callregistratie, niet alleen brontekstzoeken.

Opnieuw controleren vlak vóór `export_fast_plan()` en daarin de uiteindelijke `write_plan` vervangen door `SettlementPublication.publish` voor v2. Voeg aan `export_fast_plan` keyword-only `financial_context: FinancialContext | None=None` toe; None is uitsluitend legacy vóór/zonder gate, nooit een v2-default. De financiële context komt uit de berekening, niet uit een tweede losse options-lezing. V1-export onder slot mag blocked/v2 niet overschrijven. Een gate-exception mag niet door een devicecatch worden ingeslikt waarna volgende apparaten/export doorgaan. Debug/replay heeft geen schrijfrechten op het operationele planpad.

**Geen schijnbare transacties:** een preflight-revisiecheck is geen toestemming voor later HA-I/O. Voor de tussenstand is iedere nieuwe financiële HA-write onvoorwaardelijk geweigerd. De latere G1-eigenaar moet hetzelfde slot gebruiken voor finale artefact-/leasecontrole en eigenaarsoverdracht, en de volledige begrensde commando-uitgifte tegen invalidatie serialiseren; een lock vrijgeven tussen check en send is onvoldoende. Dat kan het korte publicatieslot beïnvloeden en behoort tot G1 met bounded gatewaytimeouts, ack en gemeten respons. Dit amendment voert die actuatieroute niet vooruit in.

## 5. A7-consumptie en A10b-restwerk

A7 behoudt `preview`, `consolidate`, `state_at`, CLI/API/rapport/scheduler en hun bestaande betekenis. Aanvullende expliciete service-interface:

```python
PlanningContext(basis: FinancialBasis, state: SettlementState,
                quotes: tuple[PriceQuote, ...], price_batch: PriceBatch)  # A7p-prijsinputamendment
SettlementService(repository, metering_adapter, price_adapter, snapshots,
                  *, publication: SettlementPublication)
SettlementService.prepare_planning(start_ts: int, end_ts: int) -> PlanningContext
```

`prepare_planning`: bronnen ophalen buiten slot; bevestigde selectie registreren via guarded repository; gepinde batch lezen; actuele prefix/ontbrekende gemeten staart eenmaal verwerken met bestaande A7-state_at-regels; quotes en context uit dezelfde selectie/revisie opbouwen. Reads van boek/prefix/batch die verschillende revisies tonen ⇒ conflict, opnieuw lezen/rekenen buiten slot of expliciet afbreken. Geen ongeregistreerde staart in een gepubliceerde planningcontext; revisie0 blijft preview-only. `check_basis` vóór teruggeven en **opnieuw** bij iedere daadwerkelijke publicatie. Bestaande state_at-signature blijft bruikbaar voor rapportage; geen mutable `service.last_revision` als impliciete overdracht tussen gelijktijdige jobs.

Precisering door A7p: toekomstige SpotIntervals via `register_prices`/`read_prices`, niet als InputInterval. Dezelfde globale revisie pinnt prijzen, gesloten inputs en snapshotselectie; alleen gesloten metingen verhogen source_watermark_ts. Bij historische prijs-/prefixcorrectie eerst gesloten repricing/herberekening, geen toekomstige settlementregels om dirty te wissen. Exacte signatures en consumerregels staan in het prijsinputamendment §4–6.

Een complete bonusgeschiedenis is geen vereiste voor geldige basisplanning: onbekende bonusstaat blijft onbekend, A8 schakelt positieve bonusprikkel uit. Ontbrekende basistarieven blokkeren wel. `prepare_planning` is expliciet een publicerende bronrefresh voor planner, niet het pad voor CLI-preview. `preview` wijzigt ook geen capability/lockbestand. Geplande consolidatie en expliciet CLI `--publish` mogen door relevante registratie het lokale plan invalideren, maar roepen geen HA/gateway/sink aan. `Report.get_price_data` mag financiële legacymisbruik blijven weigeren, maar is nooit eigenaar van invalidatie.

A10b levert pas na A8/A9 de volledige ready-v2 met echte terminalwaarde, basisprijzen, `economics_supported=False` voor nieuwe contractfeatures, policy-/replaypariteit. Een ready-v2 is een reproduceerbaar planningsartefact, **geen actuatietoestemming**. A10b hergebruikt slot, DTO's, registratieguard en runner-verdict; herbouwt ze niet. `FinancialContext` zonder terminalwaarde kan in A10a alleen als ongeldige testinput bestaan; geen 0-terminalwaarde bedenken om A9-afhankelijkheid te verhullen.

## 6. Crash/concurrentie/fouten — acceptatiecontract

| Gebeurtenis | Vereist gedrag / grens |
|---|---|
| Correctie na berekening, vóór publish | Zelfde slot: oude revisie kan niet publiceren; blocked blijft, herberekenen |
| Twee registraties | Filelock vóór DB-lock, daarna bestaande CAS; loser leest opnieuw, geen verloren revisie |
| Invalidatie fsync faalt | Geen DB-revisie publiceren; alarm/exception; geen fysieke stopclaim |
| Crash ná blocked-fsync, vóór DB-commit | DB oud of rollback, bestand blocked; geen automatisch herstel van ready |
| Crash ná DB-commit | Bestand was reeds blocked; nieuw plan vereist verse basis/check |
| Crash vóór ready-rename | Oude geldige lease of blocked blijft; geen half JSON; geen extra actuatierechten |
| Crash ná ready-rename | Volledig gecheckte grondslag; volgende relevante registratie moet eerst invalideren |
| Ongewijzigde inhoud / contract-ABA | No-op invalideert niet; A→B→A heeft nieuwe revisie, oud token geweigerd |
| Gewone nieuwe meting na planningstart | Reeds uitgegeven prognose blijft herkenbaar geldig tot lease; kandidaat op oude revisie moet opnieuw rekenen |
| Zelfde mtime, ander inode/inhoud | Finale read valideert opnieuw, cached ready verleent geen bevoegdheid |
| Lockbusy / fout type / onbekend formaat | Geen nieuwe publicatie/financiële write; runner pending/unconfirmed, geen nul-default/release-naar-oud-plan |
| Proces/host uit, HA niet bereikbaar | Software kan niet fysiek stoppen; lease begrenst alleen een levende conforme consumer. G1 watchdog/operator-/hardwarestop blijft vereist |

De garantie is consistentie met **geregistreerde** input, niet met nog niet opgehaalde externe correcties. Filesystem en DB zijn niet atomair; de gekozen volgorde maakt de relevante crashvensters conservatief. Niet-meewerkende oude images/directe schrijvers vallen buiten het slotprotocol: deployment/downgrade blijft afzonderlijk geblokkeerd tot operationele stop bewezen is.

## 7. TDD, verificatie en commitgrenzen

Per hieronder genoemde implementatiecommit: eerst tests schrijven → bedoelde RED waarnemen → kleinste implementatie → GREEN/regressies → één volledige foreground-suite met `timeout=1200000` → gerichte Ruff → review/commit volgens controller. Deze opdrachten zijn **niet tijdens dit architectwerk uitgevoerd**. Geen nieuwe dependencies/installaties om ontbrekende runtimes te maskeren.

### A10a.1 — commit `Leg de lokale financiële publicatiegrens vast`

- [ ] Maak `dao/tests/settlement/test_publication.py` en `dao/tests/settlement/test_fast_contract.py`; voeg gerichte tests toe aan `dao/tests/settlement/test_repository.py` en `dao/tests/prog/test_fast_plan.py`.
- [ ] V2 ready/blocked roundtrip; v1 leesbaar maar geen nieuwe rechten; unknown/missing/null/NaN/Inf/bool-revision/lijstpayload weigeren; terminalwaarde exact bewaren; geen stiekeme v2-default voor legacybuild.
- [ ] Twee echte processen + tempfile/test-SQLite met barrières (geen sleeps als synchronisatie): correctie vóór publish, inverse volgorde, lockbusy, exception/crash vóór/na invalidatie en vóór/na DB-commit. Assert inhoud/revisies en blokkade, niet alleen mock-callvolgorde. Fsyncfouten injecteren; DB-hash/revisie ongewijzigd bij mislukte invalidatie. Twee threads testen naast processen.
- [ ] Input-no-op/ABA en prefixcorrectie onderscheiden van nieuwe meterstaart; gewijzigde spot/contract binnen horizon invalideert; oud kandidaat-token na gewone append wordt geweigerd. Raw v1-writer kan blocked/v2 niet herstellen.
- [ ] RED/GREEN: `.venv/bin/python -m pytest -q dao/tests/settlement/test_publication.py dao/tests/settlement/test_fast_contract.py dao/tests/settlement/test_repository.py dao/tests/prog/test_fast_plan.py`.
- [ ] Implementeer uitsluitend §3; geen service/MIP/replay/hardwarecode. Bestaande A6-tests blijven groen, inclusief inputrevision0 en contract-ABA-hervalidatie.

### A10a.2 — commit `Laat planner en runner financiële blokkades consumeren`

- [ ] Maak `dao/tests/settlement/test_publication_consumers.py`; breid `dao/tests/prog/test_fast_runner.py` en `dao/tests/prog/test_fast_export.py` uit.
- [ ] Runner start met oud niet-nulplan → blocked/None/corrupt/verlopen/v1-op-ingang → fake sink ontvangt `(power_w=0, mode="configured_off", inhibit_restart=True)`; **geen** old-plan-release en **nul** normale actuatorwrites. Start zonder planspec/configsensor werkt eveneens fail-closed. Geen sink/exception/False-verkeerd-type ⇒ unconfirmed, geen geclaimde ack/stilstand.
- [ ] Gates testen off/shadow/active × draft/confirmed, seconde vóór/op ingang, legacyhorizon eindigend op/over ingang en berekening die de grens passeert, restart/zelfde mtime, ontbrekende config versus configuratiefout. Geldig nieuw ready-v2 blijft actuation_gates_unproven; fake clock bewijst requestretrybegrenzing.
- [ ] Echte planner-/export-adapter met fake HA (geen live DaBase-initialisatie): alle economische devicewrites en vroegere avg-temp-write nul bij gateweigering; devicecatch kan gate niet inslikken; debug wijzigt plan niet. Basis veranderd vóór export ⇒ blocked en geen write; v1 mag niet ontsnappen via standaardargument.
- [ ] RED/GREEN: `.venv/bin/python -m pytest -q dao/tests/settlement/test_publication_consumers.py dao/tests/prog/test_fast_runner.py dao/tests/prog/test_fast_export.py dao/tests/prog/test_fast_plan.py`.
- [ ] Implementeer §4; default fallback is een expliciet unconfirmed no-I/O-verzoek, geen hardwareadapter. Review bewaakt dat dit niet als bewezen fallback wordt gerapporteerd.

### A7 — bestaande commit, aanvullende acceptatie

- [ ] Bestaande A7-tests ongewijzigd behouden; voeg `test_prepare_planning_pins_registered_revision`, `test_correction_between_prepare_and_publish_blocks`, `test_all_application_registrations_share_guard`, `test_preview_never_invalidates_or_creates_lock`, `test_shadow_registration_invalidates_without_ha_writes` toe aan service-/publicatietests.
- [ ] Assert context.basis.input_revision > 0, quotes/state uit dezelfde snapshotselectie; onafhankelijke contexten kruisen niet via mutable servicevelden. Late correctie/ABA verwerpt oude context; gewone append behoudt oud uitgegeven plan maar weigert oude kandidaat.
- [ ] RED/GREEN: `.venv/bin/python -m pytest -q dao/tests/settlement/test_service.py dao/tests/settlement/test_reports.py dao/tests/settlement/test_cli.py dao/tests/settlement/test_publication.py dao/tests/settlement/test_publication_consumers.py dao/tests/webserver/test_settlement_api.py` plus de bestaande A7-adapterregressies.
- [ ] A7 mag softwarematig gereed worden gemeld met expliciete grens **actuatie/fysieke fallback niet bewezen; G1–G3 open**. Geen onbereikbare helper als bewijs: tests gebruiken werkelijke runner-/planner-/registratie-entrypoints.

### A10b — oorspronkelijke A10-commit op bestaande positie

- [ ] Bestaande A10 financiële-/replaytests uitvoeren, aangevuld met alle A10a-regressies. Geen protocolherontwerp en geen vrijgave van hardware.
- [ ] `.venv/bin/python -m pytest -q dao/tests/settlement/test_fast_contract.py dao/tests/settlement/test_publication.py dao/tests/settlement/test_publication_consumers.py dao/tests/prog/test_fast_*.py`.

Voor iedere implementatiecommit: `.venv/bin/python -m pytest -q` éénmaal foreground (`timeout=1200000`), daarna `.venv/bin/ruff check <uitsluitend-gewijzigde-Python-bestanden>` met concrete lijst in taakrapport. Leg aantallen/exitcodes/RED vast; geen eerdere suite als nieuwe verificatie. Bij configwijziging aanvullend `.venv/bin/python scripts/generate_docs.py` en een daadwerkelijk bestaande minimale Python3.11/Pydantic-omgeving; ontbrekende omgeving als bewijsgrens rapporteren, niet dependencies installeren.

## 8. Zelfreview en resterende beslissingen

- Producer/consumer sluit: repositoryguard → hetzelfde planbestand → `read_publication`/runner → expliciete fallbackreceipt; servicecontext → optimizer → `publish` met verse referentiecheck. Geen impliciete revisie uit singleton, run-id, mtime of v1.
- Slotvolgorde is steeds filelock → DB-lock; callback heeft reeds lease, doet geen DB-read en geen HA. Finale publishcheck en replace delen lease. Compatibiliteitswriter kan invalidatie niet terugdraaien.
- Geen positieve financiële rechten uit onbekende types, revisie0, missing/null-prijzen of verlopen leases. Basistariefplanning versus onbekende bonus blijft onderscheiden.
- Bewuste open grens: fysieke fallback, stophelpersemantiek, gatewaytimeouts, exclusieve actuator, gemeten respons en watchdog/operatorstop blijven G1; energie/exclusiviteit G2, benefit/replay-uitvoering G3. Deze scopes worden niet hier gerealiseerd of afgevinkt.
- Geen aanvullende menselijke keuze nodig voor lokale implementatie van dit amendment binnen de gegeven autorisatie. Wel onverminderd expliciet akkoord/bewijs vóór deployment en latere hardwareactivering. Als A7-acceptatie toch **daadwerkelijk bewezen stoppen op hardware** moet omvatten in plaats van bovenstaande softwaregrens: **STOP**; dan is een aparte opdracht voor G1 nodig, niet een fallback improviseren.
