# Afzonderlijke prijsinvoer — gerichte prerequisite vóór A7

**Status:** architectcorrectie, 31 oktober 2026, op codeHEAD `9819202437df0f041d9272414ca9c7735c24a16a`; alleen documentatie. A10a.1/.2 zijn uitgevoerd/gereviewd en worden niet opnieuw ontworpen.
**Doel:** horizonprijzen duurzaam pinnen onder de bestaande inputrevisie zonder het gesloten meetwatermerk te verhogen of toekomstige afrekenregels te maken.
**Voorrang:** uitsluitend voor deze prijs-/meetgrens leidend boven `docs/plans/2026-10-31-settlement-publication-gate-amendment.md` §3/5, actieve A7 in `docs/superpowers/plans/2026-10-13-production-bonus.md` en actieve spec §4/5/8/9 in `docs/superpowers/specs/2026-10-13-production-bonus-design.md`. Overige architectuur ongewijzigd.
**Volgorde:** gerealiseerd A1–A6/A10a.1/.2 → **A7p (dit prerequisite)** → A7 → A8/A9 → A10b → A11. Controller synchroniseert briefs/ledger; architect wijzigt die niet.

## 1. Probleem en keuze

De lokale karakterisering `task-A7-price-watermark-probe.py` toont: een toekomstige prijs als `InputInterval` verhoogt `source_watermark_ts`. Een daarna gecorrigeerde gesloten prefix kan alleen schoon worden als ook toekomstige onbekende financiële regels worden gepubliceerd. Dat is geen juiste betekenis van prijsregistratie of gesloten consolidatie. De bestaande `check_basis` weigert die dirty prefix terecht.

Opties:

1. **Prijs-only InputInterval plus uitzonderingen op watermerk/commit:** klein ogende wijziging, maar hetzelfde type blijft zowel gesloten meting als toekomstige prijs voorstellen. Iedere reader/commit moet uitzonderingen kennen; foutgevoelig en strijdig met de bestaande input-FK en één-regel-per-inputcontrole.
2. **Afzonderlijk immutable prijsregister, dezelfde boekrevisie — gekozen:** één additieve tabel en expliciete DTO's/read/register; prijswijzigingen gebruiken bestaande CAS/guard. Gesloten samengestelde inputs blijven de engine-ingang en financiële FK. Een aanvullende commitcontrole voorkomt schoonverklaren met achterhaalde samengestelde prijzen.
3. **Eigen prijsrevisie/boek of volledige splitsing van alle meetinvoer:** meer onafhankelijke tokens, migraties en joinlogica; onnodig voor deze grens en zou de gerealiseerde A6/A10a-contracten herontwerpen.

Keuze 2 scheidt bronsoorten zonder twee publicatieprotocollen. Een prijsregistratie is nog geen afrekening. Een historische prijscorrectie maakt bestaande relevante afrekening dirty, maar hoeft niet meteen financiële regels te schrijven. Bestaande immutable geld-/inputrijen, keys en runmetadata worden niet herschreven.

## 2. Constraints en bestandseigenaarschap

- Python3.12 runtime, lichte configuratie-import Python3.11 met uitsluitend Pydantic2; geen dependencies of nieuwe configvelden. SQLAlchemy Core en bestaande pure Decimal-kern behouden.
- Decimal uit strings, 9 decimalen energie/12 tarieven, precisie36, geen tussenafronding; UTC halfopen intervallen, contractkalender Europe/Amsterdam. Uurprijs, kwartiermeter en optimizerstap blijven onafhankelijk.
- G1–G3, harddeny voor nieuwe actuatie, fallback-unconfirmed en het bestaande lokale slot/invalidatieprotocol blijven ongewijzigd. Geen DB-call in de runner erbij.
- Geen live/SSH/HA-/DB-operaties tijdens dit planwerk; ook de probe niet opnieuw uitvoeren. Implementatietests gebruiken uitsluitend tijdelijke testdatabases/bestanden en fake gateways, nooit HA-writes. Geen bron-/testcode, dependencies, afgeleide briefs of commits in deze ronde.

| Eigenaar | Concrete bestanden en beperkte wijziging |
|---|---|
| A7p | `dao/prog/settlement/types.py`: RegisteredSpot/PriceBatch/ResolvedSpot, bestaande DTO's ongewijzigd |
| A7p | `dao/prog/settlement/repository.py`: zevende tabel, prijsregister/read, prijsconsistentie bij inputregistratie/commit; bestaande CAS/guard hergebruiken |
| A7p | `dao/prog/settlement/metering.py`: bestaande prijsresolutie als pure herbruikbare interface; repricing van gearchiveerde gesloten input, geen nieuwe prijsformule |
| A7p | `dao/tests/settlement/test_price_repository.py`, `test_price_input_bridge.py` nieuw; gerichte aanvullingen `test_schema.py`, `test_repository.py`, `test_publication.py`, `test_metering.py` |
| A7 | `dao/prog/settlement/service.py`: nieuwe interfaces consumeren in consolidatie/state/planning; bestaande A7-routebestanden en hun tests |

`CheckDB.ensure_settlement_tables()` roept reeds `SettlementRepository.ensure_schema()` aan; die route maakt ook de nieuwe tabel. Geen algemene migratieherbouw of verplichte wijziging van `check_db.py`. `settlement/publication.py`, `fastctrl/*`, engine, MIP, CLI/API-architectuur en schedulerconfig hebben voor A7p geen functionele wijziging nodig. Tests bewijzen hun bestaande consumptie.

## 3. Additieve opslag en types

### Eén nieuwe tabel `settlement_spot`

PK **`(meter_id, interval_s, start_ts, input_revision)`**; daarnaast `end_ts`, `price_ex_vat` nullable canonieke Decimal-tekst, `source` niet-lege tekst, `quality`, `content_hash` SHA-256. Types voor meter/revisie/tijd gelijk aan A6 (`String(128)`, `BigInteger`); prijs/source als bestaande dialectportable tekst. Checks: revisie >0, interval_s in {900,3600}, end_ts=start_ts+interval_s, geldige quality. Applicatie valideert alignering en Decimal-precisie/finite waarden. Hash over expliciete canonieke SpotInterval-inhoud, niet over revisie of ophaaltijd.

`interval_s` is de **bronprijsresolutie**, afgeleid van de expliciete grenzen, geen optimizerinterval. Een native uur en vier marktkwartieren kunnen naast elkaar geregistreerd zijn; de bevestigde snapshot bepaalt welke reeks wordt gebruikt (`native` of `mean_quarters`). Binnen dezelfde meter/resolutie/start bestaat één geselecteerde waarde; wisseling van bron/kwaliteit is een nieuwe inhoud/revisie. Geen impliciete concurrente bronprioriteiten. Onondersteunde of niet-uitgelijnde intervallen worden afgewezen, niet opgesplitst met verzonnen prijzen.

Latest-as-of selectie per `(meter_id,interval_s,start_ts)` is hoogste rijrevisie ≤ gepinde **globale** boekinputrevisie. Geen eigen prijscounter. Identieke inhoud is no-op; A→B→A schrijft een nieuwe rij bij de derde registratie, nooit de eerste rij reactiveren. Weglaten uit een volgende bronbatch is geen delete. Expliciet ingetrokken/onbekende prijs wordt een `SpotInterval(price_ex_vat=None, quality="incomplete", source=...)`; die overschrijft bij selectie de oude bekende prijs, zonder daarop terug te vallen. Bronmetadata is geen verborgen prijsopslag in `sources`, `spot_source` of tariefhash.

Alle zes bestaande tabellen/kolommen/keys blijven intact. `ensure_schema()` doet idempotent create-if-absent voor de zevende tabel; geen automatische backfill/reconstructie van oude raw prijzen uit samengestelde kwartierregels. Een gemiddelde uurprijs bewijst immers geen vier oorspronkelijke marktkwartieren.

In `types.py`, frozen/slotted en dezelfde validatiestijl:

```python
RegisteredSpot(spot: SpotInterval, input_revision: int, content_hash: str)
PriceBatch(meter_id: str, input_revision: int, start_ts: int, end_ts: int,
           items: tuple[RegisteredSpot, ...], snapshots: tuple[TariffSnapshot, ...],
           captured_ts: int, source_watermark_ts: int | None)
ResolvedSpot(price_ex_vat: Decimal | None, source: str | None, quality: Quality)
```

`RegisteredSpot.input_revision` is positief, ≤ `PriceBatch.input_revision`; het is de rijrevisie, niet noodzakelijk de actuele boekrevisie. Een lege ongeregistreerde batch mag revisie0 hebben, maar geeft geen planningsrechten. `PriceBatch.start_ts/end_ts` zijn het gevraagde bereik, niet een energiedekking of publicatiewatermerk. De rows omvatten de benodigde **volledige prijsuren**: floor(start/3600) tot ceil(end/3600), beide resoluties. Halfopen UTC-selectie en dubbele DST-uren behouden afzonderlijke epochs. Deterministische rowvolgorde `(interval_s,start_ts)`; geen prijzen uit de andere resolutie stil invoegen bij een gat.

`PriceBatch.source_watermark_ts` blijft uitsluitend het bestaande **gesloten invoerwatermerk uit het boek**. Geen nieuwe betekenis geven aan `FinancialBasis.source_watermark_ts` of bestaande run/view-velden. Prijsbereik blijkt uit de geregistreerde rows; maximale prijseindtijd is geen bewijs van aaneengesloten dekking of actualiteit van externe bronnen.

## 4. Register/read/CAS en prijsconsistentie

### 4.1 Publieke repository-interfaces (A7p)

```python
SettlementRepository.register_prices(
    meter_id: str, prices: Sequence[SpotInterval], expected_input_revision: int,
) -> int
SettlementRepository.read_prices(
    meter_id: str, start_ts: int, end_ts: int, *, expected_input_revision: int | None = None,
) -> PriceBatch
SettlementRepository.read_inputs(
    meter_id: str, start_ts: int, end_ts: int, *, expected_input_revision: int | None = None,
) -> InputBatch
SettlementRepository.register_inputs(
    meter_id: str, inputs: Sequence[InputInterval], snapshots: Sequence[TariffSnapshot],
    expected_input_revision: int, *, price_batch: PriceBatch | None = None,
    closed_before_ts: int | None = None,
) -> int
class PriceInputMismatch(ValueError): ...
```

`read_inputs` behoudt zonder nieuw keyword zijn huidige contract. Beide reads lezen eerst **één boekreferentie**, pinnen revisie/snapshotselectie/watermerk en lezen vervolgens immutable rows met revisie ≤ die pin. Opgegeven expected die niet overeenkomt met deze referentie ⇒ `ConcurrentSettlementError`; geen ongeautoriseerde historische contractselectie reconstrueren. Een wijziging ná de eerste boeklezing verandert niet de gepinde inhoud; de gezamenlijke A7-context en uiteindelijke publicatie toetsen dezelfde revisie opnieuw. Een herlezing van de actuele bookpointer mag een oude batch niet stil naar een nieuw nummer optillen.

`register_prices`:

1. Valideer/canonicaliseer de aangeleverde prices; dubbele keys/conflicten afwijzen. Alleen feitelijk aangeleverde bronprijzen opslaan, geen fictieve intervalrijen. Gebruik de bestaande `_registration()` (filelock → DB-lock) en `_lock_book(...expected_input_revision)`.
2. Vergelijk met nieuwste opgeslagen inhoud. Bij no-op geen nieuwe revisie, geen invalidatie, geen dirty-/watermerkwijziging. Bij wijziging kies r+1 uit dezelfde bookpointer als meter-/contractregistratie.
3. Bepaal prijsafhankelijkheid onder de **huidige geregistreerde snapshots**, niet huidige losse options. `InputChange.changed_intervals=()`, `changed_contracts=()`, `changed_prices` bevat gewijzigde bronranges plus de effectieve contractprijsvensters: één gewijzigd marktkwartier bij `mean_quarters` raakt het hele uur. Dit voorkomt dat een correctie in het eerste kwartier een plan vanaf het tweede kwartier ongemerkt mist. Conservatieve extra invalidatie door ongebruikte bronresolutie is toegestaan; geen gemiste gebruikte range.
4. Voor `dirty_from_ts` afzonderlijk: vind bestaande geselecteerde `settlement_input`-rijen waarvan het gebruikte **prijsvenster en geselecteerde bronresolutie** de wijziging raakt. Markeer de vroegste start van deze gesloten inputrijen dirty, gecombineerd met reeds bestaande dirty. Ook bron-/kwaliteitscorrecties en het voor het eerst registreren van hun prijsvenster vragen hervalidatie. Nieuwe zuiver toekomstige prijzen zonder geraakte gesloten inputs veranderen dirty niet. Een gat zonder bestaande input blijft onbekend, geen fictieve afrekenrij erbij.
5. Onder het reeds gehouden slot, vóór price-inserts/revisiepublicatie, `before_revision(InputChange(...r,r+1...))`. Daarna immutable price-rows en bookinputrevisie/dirty in één DB-transactie. **Niet** `source_watermark_ts`, generatie, input-/run-/interval-/adjustmentrijen wijzigen. Er is geen nieuwe guard of capability.

Berekening van gewijzigde ranges mag bulkgewijs; geen N+1 queries per kwartier, externe bron-/solvercalls of financiële replay onder het slot. Voorbereiding buiten het slot mag, maar pas gebruiken na expected-revisiecheck; niet half met nieuwe snapshots hergebruiken. De huidige DB-CAS verhindert ook een concurrente financiële commit op een oude prijsrevisie.

### 4.2 Gesloten inputregistratie en commit mogen de prijswijziging niet omzeilen

Een prijscorrectie schrijft bewust nog geen nieuwe samengestelde `InputInterval`. `read_inputs` retourneert dus de echte immutable geregistreerde inputs, **geen overlay met gewijzigde prijzen en oud input_hash/FK**. Daarom zijn twee aanvullende controles verplicht:

- **`register_inputs`:** A7 geeft altijd `price_batch` én `closed_before_ts` mee bij niet-lege meetinvoer. Beide of geen van beide; `closed_before_ts <= int(time())`, en iedere input eindigt ≤ deze grens. Tests bevriezen de klok. Batchmeter/revisie/snapshotselectie moeten gelijk zijn aan de registratiegrondslag; benodigde volledige prijsvensters moeten in het batchvraagbereik/envelope vallen. Onder dezelfde CAS controleert de repository dat de prijsrows/rijrevisies/hashes exact overeenkomen met de geregistreerde selectie voor dat bereik. Foute batch ⇒ `PriceInputMismatch`, gewijzigde boekrevisie ⇒ `ConcurrentSettlementError`. Geen caller-zelfverklaring dat losse quotes geregistreerd zijn.
- **Prijscontrole:** waar het geselecteerde raw prijsvenster in `settlement_spot` bestaat, moeten `spot_ex_vat` en `spot_source` van iedere gesloten input overeenkomen met de pure resolver (§5), met verenigbare conservatieve kwaliteit. Kwaliteit mag niet beter zijn dan de slechtste vereiste klasse volgens bestaande merge-regels (measured < estimated < incomplete; ontbrekende energie/prijs en duur >900s behouden hun bestaande effect). Een gedeeltelijk gevuld vereiste prijsvenster is onvolledig, geen terugval op oude samengestelde prijs. `spot_source` blijft de bestaande herkomstbeschrijving; raw prijswaarden staan alleen in hun expliciete velden/tabel.
- **`commit`**, bestaande signature/dual-CAS behouden: valideert binnen zijn transactie **opnieuw** de prijsconsistentie van alle batchinputs tegen geregistreerde prijzen op `batch.input_revision`, vóór financiële inserts/dirty-wissen. Dit geldt óók wanneer `batch` via gewone `read_inputs` kwam en de boekrevisie wel actueel is. Een oude samengestelde prijs bij een nieuwe raw prijs mag dus niet door een ongewijzigde resultaat-hash/no-op-branch dirty wissen. Gebruik een gedeelde interne resolver/validator, niet een tweede prijsformule. Mismatch ⇒ rollback en `PriceInputMismatch`.

Compatibiliteit: bestaande A6-offline callers zonder price-register mogen hun bestaande positional `register_inputs` blijven gebruiken. Zodra een meter geregistreerde raw prijzen heeft, vereist niet-lege inputregistratie de twee nieuwe keywords; snapshot-only registratie met `inputs=()` blijft mogelijk. A7 gebruikt de strikte route altijd, ook bij een lege PriceBatch. Geen toekomstige prijs-only InputInterval via deze route. Onbekende energie in een **werkelijk gesloten** meetinterval blijft wel toegestaan als incomplete; niet verwarren met toekomstige pseudo-invoer.

Oude gesloten inputs zonder enig geregistreerd raw prijsvenster houden hun opgeslagen samengestelde spot/herkomst als expliciete bestaande bewijsbasis. Geen achteraf verzonnen raw uur-/kwartierprijzen en geen claim dat zulke prijzen nieuw extern geverifieerd zijn. Zodra relevante raw prijzen worden geregistreerd, gelden bovenstaande dirty-/consistentiecontroles. `prepare_planning` krijgt **nooit** horizonquotes uit deze legacy-uitzondering: daarvoor is de volledige geregistreerde PriceBatch verplicht.

Deze uitzondering geldt alleen voor reeds geregistreerde prijsinhoud op **dezelfde inputstart/grenzen**: bij een energiecorrectie mogen precies die bestaande samengestelde spot/herkomst behouden blijven. Nieuwe gesloten inputs zonder raw prijsdekking krijgen onbekende spot, geen zelf aangeleverde bekende prijs. Nieuwe of gewijzigde bekende spotvelden vereisen eerst `register_prices`; een lege PriceBatch mag geen vrijbrief zijn. Afwezige raw dekking bewijst ook geen betere prijs-/meetkwaliteit. Test expliciet dat ongemerkte invoer van een bekende prijs via `register_inputs` wordt afgewezen.

### 4.3 Watermerk, dirty en ABA

`source_watermark_ts` wordt alleen door registratie van gesloten samengestelde meetinvoer verhoogd; het blijft het hoogste geregistreerde gesloten intervaleinde, **geen garantie van gatloze/complete meting**. `register_prices` verandert dat veld nooit. Bestaande `commit`-regel voor dirty-wissen blijft: geverifieerde herberekening vanaf dirty/benodigde eerdere afhankelijkheidsgrens door de volledige gesloten staart tot dit meetwatermerk. Geen nieuwe “clear dirty”-API en geen bypass in `check_basis`.

Nieuwe toekomstige prijzen kunnen een schoon financieel boek schoon laten, terwijl inputrevision stijgt. Een actuele view mag dan de nieuwe gevalideerde boekrevisie tonen zonder nieuwe euro's/run/generatie. Bij historische prijs-/prefixcorrectie blijft een view outdated tot de gesloten afhankelijkheidsstaart hervalideerd is. Onder dezelfde dual-CAS kan ongewijzigde volledige hervalidatie dirty wissen **zonder nieuwe financiële generatie**, zoals de bestaande A6-contract-ABA-test vereist. Run-/lijnrevisies blijven immutable; gebruik voor planning de actuele view-/boekvalidatierevisie, niet de oorspronkelijke runrevisie.

Prijs-ABA verhoogt altijd de globale revisie en bewaart alle prijsrows. Als B en daarna A nog vóór materialisatie/financiële publicatie worden geregistreerd, kan de reeds geregistreerde A-meetinvoer na volledige CAS-herberekening opnieuw geldig zijn zonder nieuwe geldversies. Als tussendoor daadwerkelijk nieuwe input-/geldversies zijn gemaakt, blijven die geschiedenis; geen garantie van no-version bij iedere ABA. Stale tokens worden in beide gevallen geweigerd.

## 5. Pure prijsresolutie en gesloten repricing (A7p)

In `metering.py`:

```python
resolve_spot(snapshot: TariffSnapshot, start_ts: int, end_ts: int,
             spots: Sequence[SpotInterval]) -> ResolvedSpot
reprice_inputs(batch: InputBatch, prices: PriceBatch) -> tuple[InputInterval, ...]
```

`resolve_spot` is de extractie van de bestaande `_price_for_interval`-logica: geen MeterInterval of gefingeerde energie nodig om een prijs te bepalen. Kies uitsluitend de bronresolutie die snapshot.price_interval_seconds/hour_price_rule voorschrijft; volledig native uur of expliciet bevestigde duurgewogen vier kwartieren. Behoud bestaande kwaliteits-/bronbeschrijving, quotegrenzen, precisie- en DST-tests. `merge_inputs` consumeert dezelfde resolver en behoudt signature/prijsregels. Zijn huidige globale `_check_order(spots)`/`dict[start_ts]` wordt voor prijzen vervangen door validatie **per bronresolutie** en selectie van de vereiste resolutie vóór indexing; uur en kwartier mogen elkaar daardoor kruisen zonder elkaar te overschrijven. Dubbele/conflicterende keys binnen een resolutie weigeren, niet laatste-wint kiezen. Meteroverlapvalidatie blijft ongewijzigd.

`reprice_inputs` vereist gelijke globale revisie en snapshotselectie; iedere `batch.items[i].meter_id` moet gelijk zijn aan `prices.meter_id` (InputBatch zelf heeft geen meter_id). Een lege inputbatch levert `()`. Het produceert alleen kandidaten voor **de bestaande batchitems**, met `input_revision=0`, voor latere registratie; geen financiële writes, geen nieuwe meetintervallen buiten de batch. Voor geraakt geregistreerd prijsvenster: neem energie, bruto-ontlading, origin en meetsources uit de gearchiveerde input; pas resolver toe en bouw de bestaande canonieke samengestelde input/hash. Voor niet-geraakte prijsvensters zonder raw archief: bewaar bestaande samengestelde prijs/herkomst. Bij volledig ongewijzigde inhoud behoud input_hash zodat no-op/contract-ABA niet onnodig nieuwe inputrijen krijgt. Contractselectie alleen wijzigen is geen meetcorrectie; engine gebruikt nog steeds de batchsnapshot.

Let op bronretentie: een InputInterval bewaart samengevoegde quality, niet altijd de oorspronkelijke zelfstandige meetquality. Bij repricing vanuit dit archief mag quality **niet worden opgewaardeerd**. Een eerdere incomplete/estimated kwaliteit blijft conservatief behouden; nieuw incomplete prijsbewijs kan wel afwaarderen. Herstel naar betere meetkwaliteit vereist opnieuw expliciet meetbewijs via de bestaande meteringadapter. Energie/bronbewijs niet uit prijzen afleiden. Dit kan current-maar-incomplete opleveren; nooit fictief exact. Geen extra meterarchief of aanpassing van financiële kern nodig voor deze correctie.

De repository-validatie hergebruikt `resolve_spot`, maar neemt geen complete service/replay/importpad als dependency. `commit` controleert alleen geregistreerde prijs-/kwaliteitscompatibiliteit en bestaande financiële consistentie, niet de solver. Imports blijven acyclisch: types → tariffs/metering → repository → publication/service; geen import van repository in metering.

## 6. A7-producer/consumer en operationele betekenis

De bestaande serviceconstructor, `preview`, `consolidate`, `state_at` en `prepare_planning` behouden hun signatures. De nog niet geïmplementeerde A7-context wordt expliciet:

```python
PlanningContext(basis: FinancialBasis, state: SettlementState,
                quotes: tuple[PriceQuote, ...], price_batch: PriceBatch)
```

`FinancialBasis`/FastPlan-format veranderen niet. Hun globale input_revision pinnt nu ook `settlement_spot`; `price_batch` houdt de concrete prijsrows/referenties bij de berekening. Geen extra prijstoken of prijsgegevens in snapshot-hashes.

Aan de pure merge-/resolvergrens wordt `tuple(row.spot for row in price_batch.items)` doorgegeven; RegisteredSpot is niet stil een SpotInterval. De bijbehorende rijrevisies/hashes blijven in PriceBatch voor registratiecontrole en herleidbaarheid. Alleen bij bekende opgeloste spot wordt `quote(snapshot,start_ts,end_ts,resolved.price_ex_vat)` aangeroepen.

**`prepare_planning(start_ts,end_ts)`:**

1. Buiten lock bronnen verversen, inclusief benodigde volledige prijsvensters en gesloten meetstaart. Snapshotselectie expliciet bevestigen/registreren met bestaande snapshot-only route. Geen draft stil bevestigen.
2. Raw verleden-/horizonprijzen via guarded `register_prices` registreren; daarna verse prijsbatch + inputbatch op één expected boekrevisie lezen. Bronnen kunnen afzonderlijk geregistreerd worden; een afgewerkte context mag alleen één uiteindelijk revisiebeeld gebruiken.
3. Nieuwe/gewijzigde gesloten meetdata met uitsluitend **geregistreerde** prijzen mergen. Geraakte oude samengestelde inputs via `reprice_inputs` uit het archief herstellen; geen afhankelijkheid van nog aanwezige HA-recorderhistoriek voor al gearchiveerde kWh. Registreer kandidaten via strikte `register_inputs`, met gesloten grens ≤ nu. Na iedere echte revisiestap opnieuw de benodigde batches op de nieuwe revisie lezen; een oude PriceBatch niet alleen herlabelen.
4. Dirty bestaande financiële prefix via de bestaande engine/CAS herberekenen vanaf geverifieerde maand-/jaaropening door de **gesloten** afhankelijke staart tot het meetwatermerk. Revalue/scheduler/planner delen deze A7-route. Een crash of race laat dirty staan; geen context opleveren door `check_basis` te omzeilen. `PriceInputMismatch` vraagt repricing; revisieconflict vraagt opnieuw lezen/rekenen buiten slot.
5. `state_at(start_ts)` gebruikt de gepinde actuele view plus alleen ontbrekende **geregistreerde gesloten gemeten** staart, geen dubbel prefix. Exacte niet-gemeten rest tot planningsbegin blijft onbekend volgens bestaande A7/A8-regels. Geen toekomstige energie of afrekenregels uit PriceBatch maken.
6. Horizonquotes uit `resolve_spot` + `quote`, uitsluitend uit dezelfde finale PriceBatch/snapshots. Volledige basistariefdekking vereist; ontbrekende prijs weigert de planning, onbekende bonusgeschiedenis blijft zonder positieve bonusprikkel. Context.basis.input_revision == price_batch.input_revision == gepinde input-/viewvalidatierevisie. Bestaande `check_basis` vóór teruggeven en opnieuw bij publicatie; iedere latere relevante price-/prefixcorrectie gebruikt dezelfde guard.

**`state_at` zelfstandig:** blijft lees-/in-memoryfunctie. Een dirty gepubliceerde prefix is niet actueel; rapporteer conflict/onbruikbare basis, laat de expliciete publicerende consolidatie/planningsroute eerst herstellen. Ontbrekende nog niet gepubliceerde, maar geregistreerde gemeten staart mag in memory worden aangevuld; een op die staart gebaseerde planpublicatie blijft onder de bestaande dirtygate vallen en wordt zo nodig eerst geconsolideerd. Geen reader die zelf dirty wist.

**Scheduler/consolidate:** blijft uitsluitend gesloten meetintervallen consolideren. Registratie van toekomstige prijzen verandert haar eindtijd of afrekenstaart niet. Bij bekende dirty eerder dan de normale 48h-lookback herberekenen met gearchiveerde inputs vanaf de benodigde afhankelijkheidsgrens; 48h is slechts externe bronrefresh, niet de grens voor replay van bekende correcties. Reeds verdwenen niet-gearchiveerde brondata blijven onbekend. Oudere onbekende externe wijzigingen vereisen expliciete backfill/leveranciersimport; geen garantie van detectie buiten retentie.

**Rapporten/CLI:** rapportbedragen blijven uit gepinde financiële generaties, nooit uit horizonprijzen opnieuw uitrekenen. Future-only prijsregistratie maakt geen toekomstige settlementregels of bedrag0; toekomst blijft prognose, geen meting. Historische prijswijziging geeft outdated totdat herberekend; old generation blijft identiek opvraagbaar. `preview`/what-if en CLI zonder `--publish` schrijven niets (ook geen price-rows, snapshots, revisies of capability); expliciet publiceren gebruikt dezelfde guard. Geen nieuw API-schrijfendpoint of automatische schedulewijziging.

## 7. Fout-/concurrentiecontract

| Situatie | Vereist resultaat |
|---|---|
| Gesloten kwartier → toekomstprijzen → late prefixcorrectie | Watermerk blijft gesloten kwartiereinde; gesloten replay kan dirty wissen; geen future settlement_input/interval/adjustment |
| Historische raw prijscorrectie, nog oude samengestelde input | Guard invalideert, dirty gezet; commit op nieuwe revisie met oude prijs faalt; pas repricing + volledige gesloten replay herstelt |
| Toekomstprijs wijzigt na contextberekening | Inputrevisie stijgt; relevante horizoninvalidatie; oude kandidaat faalt bestaande publicatiecheck |
| Marktkwartiercorrectie bij mean_quarters, plan begint later in hetzelfde uur | Hele effectieve uur in changed_prices; geen relevantiegat |
| Andere meter of buiten gebruikte prefix/horizon | Andere meter geen invloed; eigen globale revisie blijft exact voor kandidaten, bestaand plan volgt bestaande relevantiesemantiek |
| Nieuwe meting na planbegin met al geregistreerde gelijke uurprijs | Geen nieuwe prijsdekking/-correctie uit de materialisatie afleiden; bestaand prognoseplan niet hierdoor invalideren |
| Prijs- en meterwriter tegelijk / commit oud r | Eén bestaande CAS-winnaar; loser herleest. Geen overschreven registratie of geld op oude prijsbasis |
| Invalidatie/fsync faalt | Prijsrijen/revisie/dirty niet publiceren; bestaand errorprotocol |
| Crash na invalidatie, vóór DB-commit | Blocked bestand, oude DB; conservatief. Geen fictieve DB+filesystematomiciteit |
| Crash na prijscommit, vóór repricing/financiële commit | Prijzen duurzaam, relevante historische prefix dirty; restart kan uit archief herstellen |
| Onvolledige prijsbatch/withdrawal/onbekende kwaliteit | Geen silent fallback naar oude prijs/0 of andere resolutie; planning weigeren, afrekening incomplete |

Voor nieuwe meetmaterialisatie moet `register_inputs` zijn `InputChange.changed_prices` niet langer uitsluitend uit oude samengestelde inputs afleiden: gebruik waar aanwezig de al geregistreerde prijsreeks als vergelijking. Identieke materialisatie is een meetwijziging (`changed_intervals`), geen nieuwe bronprijsdekking. Een werkelijk afwijkende spot kan alleen via `register_prices` binnenkomen. De oude `_changed_prices`-route blijft alleen voor bestaande offline/legacy-input zonder raw register. Dit is een wijziging in **change-productie**, geen aanpassing van de werkende A10a-guard.

## 8. TDD en commitgrens

### A7p — één prerequisite-commit: `Scheid geregistreerde prijzen van gesloten afrekeninvoer`

- [ ] Schrijf eerst `test_price_repository.py`/`test_price_input_bridge.py` en gerichte regressies. RED uitvoeren en de bedoelde ontbrekende interfaces/semantiek vastleggen; de oude probe-pass is geen nieuwe RED/GREEN-test.
- [ ] **Kernregressie met echte SQLite/guard/engine:** registreer/publiceer één gesloten kwartier; registreer toekomstige SpotIntervals; corrigeer de historische kWh; publiceer alleen gesloten herberekening. Assert watermark onveranderd, dirty=None, basischeck accepteert actuele revisie, geen toekomstige input-/financiële rows. Dit is de gewenste tegenhanger van de karakterisering.
- [ ] **Historische prijsbrug:** na register_prices op gesloten uur is view outdated; `commit` met oude samengestelde input maar verse globale batchrevisie geeft `PriceInputMismatch` en verandert geen boek/geldrows. Na reprice/register/read/settle/commit correcte bedragen en schone gesloten staart. Herhaal met recorderbron verdwenen; gearchiveerde kWh blijven identiek en kwaliteit wordt niet opgewaardeerd.
- [ ] **Future-only/no-op/ABA:** identieke prijs geen revisie of invalidatie; prijs A→B→A eigen immutable revisions; oud planningtoken geweigerd. Future-only verhoogt geen financial generation en wijzigt geen runmetadata/watermerk/dirty. Historische A→B→A vóór materialisatie kan na volledige hervalidatie no-version blijven; gedeeltelijke replay wist dirty niet.
- [ ] **Bestaande A6-contract-ABA:** houd `test_contract_roundtrip_revalidation_preserves_financial_generation` groen; voeg variant met geregistreerde raw prijzen toe. Assert huidige view.input_revision == bookrevision, oorspronkelijke run-/lijnrevision en bytes onveranderd. Oude generatie blijft gepind na een latere echte geldgeneratie.
- [ ] **Types/schema:** zevende tabel idempotent toegevoegd aan bestaande zes met gevulde financiële geschiedenis; bestaande PK/FK/rijen ongewijzigd. Canonieke Decimals, negatieve prijzen, None, boolrevision, fout interval/overlap/dubbele keys; 900/3600 naast elkaar, native versus mean_quarters, 23/25h en dubbele UTC-uren. DDL compileert SQLite/PostgreSQL/MariaDB; echte serverruntime alleen in reeds beschikbare expliciete testdatabases, anders bewijsgrens noteren.
- [ ] **Pinning/guard:** twee echte connecties/processen met barriers voor pricewriter/closedwriter/commit; geen sleep-gebaseerd bewijs. Correctie tussen beide reads of tussen context en publish weigert; gewijzigde eerste marktkwartier invalideert later plan in hetzelfde uur. Nieuwe meetkwartieren bij gelijke geregistreerde uurprijs geven geen changed_prices. Fsync-/crashinjectie vóór/na prijscommit volgt bestaande conservatieve volgorde; geen HA-calls.
- [ ] RED, daarna GREEN: `.venv/bin/python -m pytest -q dao/tests/settlement/test_price_repository.py dao/tests/settlement/test_price_input_bridge.py dao/tests/settlement/test_schema.py dao/tests/settlement/test_repository.py dao/tests/settlement/test_metering.py dao/tests/settlement/test_publication.py dao/tests/settlement/test_publication_consumers.py`.
- [ ] Implementeer uitsluitend §3–5 en de change-productie uit §7, met bestaande guard/commit als consumenten; nog geen A7-service bouwen. Bestaande schema-testnaam/verwachting “six” aanpassen naar zeven, niet de bestaande FK-acceptatie verwijderen.
- [ ] Zelfde gerichte tests groen; `.venv/bin/ruff check dao/prog/settlement/types.py dao/prog/settlement/repository.py dao/prog/settlement/metering.py dao/tests/settlement/test_price_repository.py dao/tests/settlement/test_price_input_bridge.py dao/tests/settlement/test_schema.py dao/tests/settlement/test_repository.py dao/tests/settlement/test_metering.py dao/tests/settlement/test_publication.py`.
- [ ] Eén finale `.venv/bin/python -m pytest -q` foreground, tooltimeout `1200000`, vóór implementatiecommit. Exacte aantallen/exitcodes rapporteren. Geen dependencies installeren; config niet gewijzigd, dus geen docs/schema-configregeneratie nodig. Lichte configimport mag geen nieuwe runtime-import krijgen.
- [ ] Review en gerichte commit volgens controller; geen deployment, live schema-upgrade of hardwareactie.

### A7 — bestaande taak hervatten na A7p-review

- [ ] Consumeer §6; voeg service-/CLI-/rapporttests toe voor duurzame horizonprijzen, late price/prefixcorrectie met gesloten replay, ontbrekende prijsdekking, geregistreerde tail zonder dubbele prefix, pricebatch-revisieconflict en preview zonder enige write.
- [ ] `.venv/bin/python -m pytest -q dao/tests/settlement/test_service.py dao/tests/settlement/test_reports.py dao/tests/settlement/test_cli.py dao/tests/settlement/test_price_repository.py dao/tests/settlement/test_price_input_bridge.py dao/tests/settlement/test_publication.py dao/tests/webserver/test_settlement_api.py`, plus de oorspronkelijke A7-adapterregressies, gerichte Ruff en één finale volledige foreground-suite vóór de bestaande A7-commit.
- [ ] Geen verandering aan A8/A9-formules, contractkeuzes, A10a-gates of G1–G3. Softwarematig accepteren van een actuele basis blijft iets anders dan toestemming tot actuatie.

## 9. Zelfreview en bewijsgrenzen

- Eén revisiedomein: raw prijsrows → PriceBatch → gesloten merge/prijsconsistentie → InputBatch/engine → bestaande dual-CAS; daarnaast PriceBatch → PlanningContext → FinancialBasis → bestaande publicatiecheck. Rowrevision en boekrevision niet verwarren.
- Gescheiden voortgang: toekomstige prijzen raken uitsluitend prijsregister/globale revisie/zo nodig planinvalidatie; historische prijsimpact raakt dirty maar niet meetwatermerk; alleen gesloten meetinvoer verhoogt dat watermerk. Geen extra bookkolom, aparte lock of capability nodig.
- Een actuele batchrevisie alleen bewijst niet dat oude samengestelde spotvelden geactualiseerd zijn: de nieuwe commitcontrole sluit precies die omweg. No-version-hervalidatie blijft toegestaan na volledige correcte gesloten replay; immutable geldgeschiedenis blijft onaangeroerd.
- Repricing gebruikt gearchiveerde energie en conservatieve kwaliteit; geen reconstructie van onbekende raw marktkwartieren, bronretentie of meetbewijs. Preview blijft write-free, ook lokaal.
- Geen verdere menselijke keuze nodig voor dit additieve grensvlak binnen de gegeven autorisatie. Als een echte bestaande DB reeds prijs-only toekomstige InputIntervals/financiële rows bevat: niet automatisch watermerk terugzetten of rijen wissen; STOP voor een afzonderlijke herstelopdracht. De aangeleverde probe gebruikte alleen een tijdelijke DB en is geen migratieopdracht.
- Deze ronde heeft alleen documenten gelezen/gewijzigd en structuur/werkboom gecontroleerd; geen probe/suite/DB-operatie uitgevoerd. De eerder gerapporteerde 1 karakteriseringstest en 84 prerequisite-tests zijn geen bewijs voor deze nog niet geïmplementeerde prijsgrens. G1–G3 en liveactivatie blijven open.
