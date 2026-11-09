# A8 — voorstel voor een beperkt bruto-celontlaaddomein

## STATUS: DOMEINVOORSTEL GOEDGEKEURD — UITVOERINGSPLAN TER REVIEW

**Basis:** HEAD `c0db962d99a407883c718664b81bf3145d370373`; `2026-10-31-settlement-source-domain-analysis.md` en de huidige A8-brief. De gebruiker heeft het volledige domeinvoorstel expliciet goedgekeurd, inclusief mogelijke planuitval en strikte numerieke afwijzing; uitsluitend de uitvoeringsplanning mocht daarop volgen.
**Leidende uitvoeringsaanvulling:** `docs/plans/2026-10-31-settlement-source-domain-amendment.md`. Geïsoleerde uitvoering is daarna geautoriseerd; Tasks1–4 zijn gerealiseerd, Task5 is numeriek gestopt. **Besluit 1 november 2026 op “Maak het besluit”: §9 van dat amendment vervangt uitsluitend het eerdere retryverbod en de bit-exacte hulpvariabeleacceptatie door één gecontroleerde vaste-keuzeherberekening en onafhankelijke hoeveelheidscontrole.** Nul/L/celbroncontrole blijft strikt; engine, G1–G3 en harddeny veranderen niet. Dit besluit is nog niet geïmplementeerd en geeft geen livevrijgave. De overige voorsteltekst hieronder is de historische goedgekeurde domeinbasis, niet een opnieuw open domeinkeuze.

## 1. Aanbevolen besluit in één tabel

| Onderwerp | Expliciet goedgekeurd domeinvoorstel |
|---|---|
| Beleidsnaam | `aggregate_cell_min_1wh_v1` |
| Domein | Groepsontlading **exact 0 óf minstens 0,001 kWh = 1 Wh** bruto celenergie per toepasselijk volledig leverancierkwartier |
| Groep | Alle gemodelleerde batterijen achter dezelfde afrekenmeter; som van niet-negatieve bruto celontlaadenergie, nooit netto of gemiddeld |
| Boven het minimum | Continu tot een aantoonbaar eindige bovengrens U; **geen energielattice**, geen stappen van 1 Wh |
| Toepassing | Alleen volledige uitgelijnde 900s-intervallen waarin de bonusmodule op grond van de vooraf bekende voorwaarden een positieve bonus kan toekennen; exacte afbakening §3 |
| Restinterval / uurplanning | Geen minimum uit dit beleid; bestaande uitschakeling van de bonusprikkel blijft gelden. Restexport verbruikt wel cap als de prefix bekend is |
| U < 0,001 kWh | Alleen groepsontlading0 toegestaan; L niet verkleinen. Indien dat strijdig is met andere constraints: geen geldige oplossing |
| Standaard | Geen geselecteerd beleid. Geen stil inschakelen door code-deployment, `mode=active` of een bevestigd contract |
| Onbruikbare oplossing | Afwijzen; geen epsilon, clamp, automatisch kleinere L of heroptimalisatie op een stil versoepeld domein |
| Metingen/engine | Ongewijzigd: iedere echte positieve bruto ontlading sluit bonus uit, ook onder 1 Wh |

**Waarom 1 Wh?** Dit is mijn engineeringkeuze voor een eerste expliciet modelbeleid: een eenvoudig uitlegbare, vaste en kleine uitgesloten energierange, met op een heel kwartier een equivalent van **4 W gemiddeld bruto celvermogen voor de gehele groep**. Geen capaciteitsschaling, geen apparaatinstelling raden en geen gebruikersspecifieke financiële parameter. Het doel is zo weinig mogelijk aan de continue beslisruimte weg te nemen met één eenvoudig controleerbare domeingrens. Het getal komt **niet** uit bewezen hardwaregedrag, de 9-decimalenregel, een leverancierseis of een bekende CBC-tolerantie.

1 Wh is geen bewezen optimale keuze en geen gegarandeerde grens voor economisch verlies. Het is verantwoord als **bewust te accepteren planningsbeperking met fail-closed numerieke verificatie**, niet als bewezen geschikte instelling voor iedere installatie. Een andere waarde betekent een ander beleid en vraagt een expliciet besluit; geen automatische tuning in deze versie. Rekentijd, afwijzingsfrequentie en economische impact zijn nog niet gemeten.

## 2. Alternatieven en keuze

1. **Semi-continu groepsminimum — aanbevolen.** Eén binary per toepasselijk interval; alle energie boven L blijft continu. Alleen de totale bruto ontlading bepaalt de enginebronregel, dus een minimum per afzonderlijke batterij zou méér plannen uitsluiten dan nodig. Een groep mag bijvoorbeeld 0,4 Wh uit batterij A en 0,6 Wh uit B leveren; de som1Wh voldoet.
2. **Lattice, bijvoorbeeld gehele veelvouden van een expliciet q.** Levert ook een exacte nulindicator, maar sluit óók allerlei hoeveelheden boven het minimum uit. Extra integeraantallen, mogelijk grote U/q en meer invloed op energiebalansen/SoC. Voor uitsluitend exact onderscheid tussen nul en positief is dit onnodig sterk; daarom geen q voorstellen.
3. **Per-batterij minimum.** Eenvoudig per apparaat te verklaren, maar een minimum op elk apparaat heeft hier geen hardwarebasis en is restrictiever dan de aggregate enginebronregel. Geen reden om nu B×T bronbinaries en meerdere parameters te introduceren.

De aanbeveling kiest dus de **kleinste soort beperking voor deze bronregel**, niet een wiskundig kleinste positieve L: daarvoor bestaat geen kleinste positief reëel getal. `0<D<1Wh` wordt verboden in de toepasselijke planningsstappen; het wordt nooit als nul of bonusgerechtigde kleine ontlading behandeld.

## 3. Precieze scope en gedragsimpact

### Welke intervallen krijgen de kloof?

De bonusbuilder bepaalt vóór oplossen een vaste `required[i]` uit dezelfde voorwaarden die de bestaande A8-bonusprikkel toelaten:

- volledig en uitgelijnd `[start_ts,end_ts)` van900s, niet het gedeeltelijke eerste restinterval;
- bonus ingegaan/in venster, positieve potentiële bonusrate en bevestigde bruikbare contractbasis;
- voldoende bekende prefix/cap-/broninformatie om volgens de bestaande regels een positieve bonusprikkel toe te laten, rekening houdend met een expliciete leveringsjaarreset;
- een niet-nulle mogelijke first-hoeveelheid volgens de vooraf beschikbare bounds.

De te beslissen **nul/positiefcelontlading** hoort niet bij deze voorafcheck: juist dat bepaalt de nieuwe binary. `required` mag evenmin van de gevonden objective of van de achteraf gekozen bronbinary afhangen. Een caller krijgt geen vrijblijvende permission-boolean waarmee de koppeling kan worden overgeslagen.

Bij `required=False` blijft het ontlaaddomein zoals het was; de bonusprikkel is volgens de bestaande A8-regel nul/uitgeschakeld, met behoud van onbekend-status waar de engine geen bedrag kent. Hiermee verandert het voorstel niet stiekem basisplanning zonder mogelijke bonus. Bij `required=True` blijft de kloof gelden, ook als de optimizer uiteindelijk first0 kiest of de cap eerder in de horizon opmaakt. De kloof dynamisch uitschakelen op “first>0” zou een nieuw nultestprobleem introduceren; dat wordt niet gedaan.

### Grenzen en gevolgen

- L is **per volledig kwartier**, niet per apparaat, seconde, tariefuur of dag. Voor restintervallen wordt L niet naar beneden geschaald: daar is geen positieve bonusprikkel en wordt deze aanvullende domeinconstraint niet opgelegd. Een modelbegin precies op een kwartiergrens kan wel onder de regel vallen.
- Een uurprijs blijft een uurprijs; vier kwartierquotes mogen dezelfde prijs hebben. Een uur-optimizerrooster wordt hierdoor geen kwartierrooster en krijgt geen bonusrechten.
- U=0 of 0<U<L betekent in een toepasselijk interval groepsontlading0. Ook als een kleinere ontlading voorheen een harde eind-SoC-/balansvoorwaarde vervulde: nu kan het probleem infeasible worden. Geen stille versoepeling of ongemarkeerde terugval naar het oude bonusmodel.
- Een groep kan niet aan de regel ontsnappen door vermogen over batterijen te verdelen. Alle bruto bijdragen worden opgeteld; laden trekt niets af. Gelijktijdige laad-/ontlaadstromen kunnen nog steeds een onrealistische oplossing opleveren: **G2 is hierdoor niet opgelost**.
- DC-PV-pass-through met nul celontlading blijft toegestaan. AC-inverterstand, AC-gridvermogen, netto batterijvermogen of netto SoC-delta vervangt de bruto som nooit.
- De beperking kan kosten, plankeuzes en haalbaarheid veranderen. Er is geen gegarandeerde winst of maximumbedrag aan verlies: reoptimalisatie kan een heel ander traject kiezen. Hoogstens de uitgesloten range van de beslisvariabele is expliciet klein; dat is geen bovengrens aan het totale effect.

## 4. Wiskundig contract en echte producerbinding

Voor batterij b en interval i, met DC-busuitvoer `p_bi = dc_from_bat[b][i]` in kW, duur `h_i` in uren en cel→bus-rendement `η_b`:

```text
d_bi = p_bi * h_i / η_b                 # bruto CEL-kWh, niet bus-kWh
D_i  = Σ_b d_bi                        # alle niet-negatieve celbijdragen
U_i  >= Σ_b (ub(p_bi) * h_i / η_b)      # geldig eindig groepsmaximum
L    = Decimal("0.001") kWh
```

De genoemde bound mag strakker zijn als die aantoonbaar voor alle oplossingen geldt. Niet delen door AC-inverterrendement of nog een round-tripfactor; de SoC-producent gebruikt al precies deze cel→busconversie. `h_i>0`, `0<η_b<=1`, finite niet-negatieve variabele-bounds zijn verplicht. Ongeldige/ontbrekende bounds of rendementen blokkeren modelbouw, niet vervangen door een geraden M. De bovengrens wordt met conservatieve opwaartse afronding afgeleid, niet met neerwaartse financiële afronding. Werkelijke producentcoëfficiënten en bounds moeten overeenkomen met het model waarop de constraints worden gezet.

Voor `required[i]=True` en U_i≥L, introduceer z_i binary:

```text
L*z_i <= D_i <= U_i*z_i
0 <= eligible_i <= first_i
eligible_i <= F_i*(1-z_i)
eligible_i >= first_i - F_i*z_i         # 0 <= first_i <= F_i, finite geldige bound
```

Voor U_i<L leg D_i=0 op en gebruik bronstatus0. F_i komt uit export-/capbounds, niet uit een willekeurige grote constante. De caprecursie en jaarreset blijven onafhankelijk exact en tellen **alle** export, ook buiten bonusvensters en bij positieve celontlading. Bij een onbekende cap/prefix worden bestaande uitschakel-/resetregels gebruikt, niet een onbeperkte cap ingevoerd.

**Bewijs:** bij D=0 kan z niet1 zijn door L>0, dus eligible=first. Bij D>0 kan z niet0 zijn, dus eligible=0 en D≥L. Dit hangt niet af van de objective. Het is exact op het nieuwe toegestane domein; kleine positieve hoeveelheden zijn niet langer haalbaar. Voor niet-toepasselijke intervallen wordt geen bronexactheid geclaimd waar de engine zelf onbekende bonusgegevens heeft.

**Eigenaarschap:** `day_ahead.py` levert de volledige echte `dc_from_bat`-variabelenmatrix, bestaande tijdfracties/rendementen en hun geldige bounds. De bonusmodule zet bovenstaande constraints op expressies die **diezelfde variabelen** gebruiken. Geen losse vrije `discharge`-proxy, voorspelde HA-mode of boolean zonder flowbinding. De nulclassificatie in de nacontrole leest opnieuw de originele variabelen; een ongebonden proxy kan dus niet als bewijs dienen.

## 5. Voorgestelde minimale configuratie en interfaces

Dit is een **ontwerprichting ter goedkeuring**, geen reeds geldende A8-signature of implementatieopdracht.

### Eén optionele beleidskeuze, geen vrij instelbare L

In `dao/prog/config/models/settlement.py`, op SettlementConfig (planningsbeleid, geen contracttarief):

```text
planning_source_policy: Literal["aggregate_cell_min_1wh_v1"] | None = None
alias: "planning source policy"
```

- None selecteert geen nieuw domein. Zodra een run `required=True` heeft en de exacte bonusplanning vraagt, ontbreekt zonder expliciete policy de vereiste grondslag: duidelijke modelbouwweigering `source_domain_policy_required`, niet stil eenzijdige eligibility of “bonus onbekend dus maar uit” als vervanging. Reeds expliciet uitgesloten bonusroosters/unknowngevallen blijven hun bestaande basisplanning behouden.
- Alleen de benoemde policy legt L=Decimal("0.001") vast. Geen stille defaultwaarde bij laden van oude config, geen vrije `minimum power` hergebruiken. Offline fixtures mogen de policy expliciet kiezen; persoonlijke config wordt niet aangepast.
- Dit plannerbeleid is geen TariffSnapshot-/leveranciersregel. De bestaande financiële allowlist/hashes veranderen er niet door; beleidsversie wel expliciet in optimizerdiagnostiek/modeldump en later in het bestaande financiële `issues`-veld van het plan opnemen (`planning_source_policy=aggregate_cell_min_1wh_v1`). Geen nieuw capability-/opslagprotocol.
- Config blijft Python3.11/Pydantic-only; MIP-types leven niet in config. Een latere wijziging vereist uiteraard gegenereerde docs/schema en importtests, nu niet uitvoeren.

### Producer/consumer-richting in `settlement/optimization.py`

```python
make_cell_discharge_input(
    dc_from_bat: Sequence[Sequence[mip.Var]], *,
    hour_fractions: Sequence[Decimal],
    discharge_efficiencies: Sequence[Decimal],
) -> CellDischargeInput

build_bonus_terms(
    model: mip.Model, quotes: Sequence[PriceQuote],
    grid_export: Sequence[mip.Var], cell_discharge: CellDischargeInput,
    export_bounds_kwh: Sequence[float], opening: BonusState, *,
    source_policy: Literal["aggregate_cell_min_1wh_v1"] | None = None,
) -> BonusModelTerms
```

Voorgestelde immutable runtime-DTO's, in dezelfde optimizer-module:

- `CellDischargeInput`: originele `bus_power_vars` per batterij/interval, `hour_fractions`, `discharge_efficiencies`, de afgeleide `per_battery_kwh`, `total_kwh` en `upper_bounds_kwh`. Alle matrixdimensies gecontroleerd; geen dubbele variabele-identiteiten of vermenging van modellen, geen negatieve lb/ongeldige ub. Een lege batterijmatrix mag alleen bij een expliciet zonder batterij gemodelleerde installatie D=U=0 opleveren; ontbrekende batterij-/brondata is geen lege bewezen installatie.
- `SourceDomainTerms`: `policy_version: str|None`, `required: tuple[bool,...]`, `discharge_on: tuple[mip.Var|None,...]` en de gebruikte `CellDischargeInput`. None betekent geen binary (niet-toepasselijk of zero-only); de beide gevallen worden door required/U onderscheiden.
- `BonusModelTerms` behoudt eligible_kwh/revenue/issues en krijgt expliciet `source: SourceDomainTerms` voor nacontrole. Het is nog geen geïmplementeerde interface, dus geen verborgen compatibiliteitsdefault of reeds opgeleverde consumer wijzigen.

De factory bewijst geen hardwaredekking op zichzelf: de day-ahead-adapter moet alle B-batterijen doorgeven en integratietests moeten die volledigheid bewijzen. De builder controleert dat de ontvangen afleidingen met de originele variabelen/conversies overeenkomen en bouwt/bindt ze zo nodig zelf opnieuw; alleen een door de caller ingevulde bound/boolean vertrouwen is onvoldoende. De exacte dataclass-veldtypen en postsolve-interface worden na domeingoedkeuring in de gerichte uitvoeringsaanvulling vastgelegd, vóór implementatie.

## 6. Numerieke criteria — geen verstopte nultolerantie

**Leeshulp na numerieke STOP:** onderstaande criteria behouden hun bron-/domeinbetekenis. Voor herberekening en controle van uitsluitend afgeleide C/Q/first/eligible is voortaan amendment §9 leidend. Een toegestane nieuwe solve is geen clamp van de oude oplossing; de nieuw opgeloste oorspronkelijke celvariabelen moeten opnieuw strikt slagen. De volgende zin over ontbrekende CBC-metingen beschrijft uitsluitend de oorspronkelijke voorstelronde; latere metingen en hun beperkingen staan in het amendment.

**Geen CBC-metingen uitgevoerd en geen defaults gegokt.** De keuze1Wh is onafhankelijk van solverinstellingen. Exacte constraints in reële rekenkunde betekenen niet dat iedere CBC-uitvoer automatisch voldoet.

Voorgesteld acceptatiebeleid:

1. **Schaal expliciet.** De bronconstraints kunnen als `z <= D/L <= (U/L)*z` worden geschreven; de onderste sprong is dan1 in plaats van0,001. Registreer U/L, coëfficiëntenbereik en effectieve solverversie/integrality-/primal-feasibility-instellingen. `max_mip_gap_abs`/nodes zijn optimaliteits-/zoekgrenzen, geen nulenergiecriteria. Geen solverinstellingen stil wijzigen om een probleemgeval passend te maken.
2. **Binary interpretatie.** Een binary wordt slechts als0/1 geïnterpreteerd als hij finite is en binnen de daadwerkelijk bekende integralitytolerantie van precies één integer ligt; die tolerantie moet <1/2 zijn. Anders afwijzen. Dit is interpretatie van een gedeclareerde integer, niet een energiedrempel.
3. **Nul is strikt.** Bij bronstatus0, inclusief U<L, moeten de originele niet-negatieve `dc_from_bat`-resultaten van **iedere** batterij werkelijk0 zijn (`-0.0` is dezelfde nul). Een kleine positieve of negatieve solverrest wordt niet tot nul geclamped. De kandidaat faalt dan de bronacceptatie. Solverfeasibility alleen is niet voldoende.
4. **Positief is strikt binnen het domein.** Bij bronstatus1 moet de onafhankelijk uit originele waarden/tijd/rendementen gereconstrueerde bruto som voldoen aan `L <= D <= U`, met niet-negatieve bijdragen. De grensvergelijking gebruikt exacte rationale getallen uit de eindige decimale tekst van die waarden (stdlib `Fraction`), zodat afronding tijdens delen door η niet een waarde nét onder L kan laten slagen. Dit certificeert het gereconstrueerde kandidaatvolume, niet onbekende fysieke nauwkeurigheid of een exact solveroptimum. Ook `D=0,000999999999` is geen geaccepteerde L. Niet ophogen naar L of het minimum verlagen. Dezelfde rationele afleiding kan een veilig opwaarts afgeronde U onderbouwen.
5. **Decimal-/DTO-grens expliciet.** Financiële berekening blijft decimale tekst/precisie36; energienormalisatie tot maximaal9 decimalen volgens het bestaande kwaliteitscontract. Eerst de niet-genormaliseerde rationele bruto som en bronstatus toetsen, daarna nogmaals het daadwerkelijk aan de engine geleverde groepsvolume. Niet per batterij kleine positieve bijdragen naar nul afronden vóór sommeren. Normalisatie mag nul/positiefclassificatie of L-toelaatbaarheid niet wijzigen; anders afwijzen. De bronvalidatie met Fraction is geen wijziging van de financiële engine of haar Decimal-contract.
6. **Pariteit als afzonderlijke backstop.** De echte engine rekent dezelfde geëxtraheerde prognose, quotes en opening na. Bij positieve ontlading blijft eligible0; bij exact nul en alle bekende geldige voorwaarden precies first. Controleer deze logische richting expliciet, niet alleen het euroverschil. De bestaande maximaal1e-6 EUR MIP-/enginevergelijking blijft een financiële rekentolerantie, **nooit** een tolerantie voor nulontlading. Onbekende engine-uitkomst wordt niet als exacte0-pariteit gepresenteerd.
7. **Geen certificering bij onduidelijkheid.** Onbekende effectieve numerieke instellingen, onbetrouwbare bound/conversie, failed source-audit, ongeldige modeloplossing of mislukte enginepariteit ⇒ geen geaccepteerd nieuw financieel plan. Geen claim dat de huidige physical modelconstraints hiermee volledig gecertificeerd zijn; gewone model-/balansvalidatie blijft nodig en G2 blijft open.

De strikte nul-/L-nacontrole voorkomt dat big-M-“trickle flow” ten onrechte bonus geeft, maar kan een mathematisch haalbare oplossing wegens floating-pointresten **afwijzen**. Dat beschikbaarheidsrisico hoort expliciet bij dit voorstel. Er wordt geen automatische retry met grotere L, verborgen tolerantie of gewijzigde objective goedgekeurd. Als de latere CBC-tests aantonen dat bruikbare resultaten structureel niet geaccepteerd kunnen worden, stoppen en een nieuw numeriek ontwerp voorleggen; niet dit beleid stil aanpassen.

Er is geen op voorhand bewezen universele veilige U/L-limiet. Een empirisch onbewezen maximum daarvoor verzinnen zou extra installaties uitsluiten zonder grond. Strakke bounds, onafhankelijke resultaatcontrole en latere schaalbaarheidstests zijn hier de backstop, geen theoretische runtimegarantie. Het aantal bronbinaries groeit lineair met de toepasselijke intervallen, niet met het aantal batterijen; andere binaries/constraints van de bestaande optimizer blijven bestaan.

## 7. Engine, historie en baseline blijven betekenisvast

- **Gemeten 0,2 Wh bruto ontlading:** de engine sluit bonus uit, ook al zou zo'n interval als geoptimaliseerd kandidaatplan bij required=True verboden zijn. Metingen worden niet aan dit planningbeleid aangepast.
- **DC-PV-pass-through:** alle cellen0 betekent bronstatus0; AC-uitvoer mag positief zijn. Ander bronbewijs blijft noodzakelijk, nul ontlading alléén is geen leveranciersbewijs.
- **Cap:** alle export chronologisch tellen; prijsvenster/bronstatus bepaalt alleen de bonus op first. Geen herverdeling van cap naar latere dure kwartieren. Onbekende historie blijft onbekend; resets volgen bestaande A8-regels.
- **Referentie zonder batterij:** D=0, zelfstandige export/cap-/maandontwikkeling vanaf dezelfde opening. Geen bonus uitsluitend van de slimme variant aftrekken.
- **Vaste referentie zonder optimalisatie:** een opgegeven referentietraject kan buiten het nieuwe planningsdomein liggen; het wordt niet kunstmatig naar nul/L aangepast. De engine rekent zijn werkelijke expliciete prognosehoeveelheden af, met eigen staat. Benoem dan dat dit een vaste referentie is, geen oplossing onder hetzelfde beperkte optimalisatiedomein. Een eventueel óók geoptimaliseerde comparator krijgt dezelfde policyvoorwaarden; geen asymmetrische solvervrijheid verbergen.
- **G2:** laad-/ontlaadexclusiviteit, uitvoerbaar duty-cycle-/stopprofiel en eerlijke fysieke replay moeten afzonderlijk bewezen worden. Dit voorstel bewijst alleen de modelmatige bronkoppeling op zijn beperkte domein. G1/G3 en harddeny eveneens ongewijzigd.

## 8. Acceptatierichting na eventuele goedkeuring

Geen tests/solveropdrachten in deze ronde. Een latere uitvoeringsaanvulling moet ten minste onderstaande tests aan de echte producer/helper/engine koppelen:

| Testklasse | Concrete verwachting |
|---|---|
| Objective onafhankelijk | Bij first1/D0 geeft min-, max- én constante objective eligible1; bij D=0,001 of0,0015 geeft ieder eligible0. €0,10 spot/30% geeft bij D0 €0,03 |
| Verboden gat | Fixed D=0,0005 in een required-kwartier is infeasible, niet “eligible0 maar toegelaten” |
| Aggregate, geen lattice | 0,0004+0,0006 kWh toegestaan; 0,0002+0,0003 verboden; 0,0015 toegestaan. Geen per-batterijminimum of multiples van0,001 |
| U-grenzen | U0 of0,0008 dwingt D0; een andere constraint die positieve ontlading vereist maakt de run infeasible. Geen kleinere L |
| Scope | Restinterval, uurrooster, venster/bonus uit en unknown-prefix behouden bestaande bonusuitschakeling; geen geschaalde L. Een later volledig geldig kwartier/reset krijgt de echte policy |
| Bronbinding | De echte matrix `dc_from_bat` voert de constraints aan; bij één positieve celbijdrage mag geen losse nulproxy bonus verlenen. DC-PV zonder cellen mag dat wel |
| Eenheden | Bus-kW×werkelijke uren/η precies eenmaal naar cel-kWh; meerdere verschillende η, zero-/nonfinite efficiency/bounds afgewezen; partial durations niet als kwartier behandelen |
| Numerieke fouten | Tiny positieve/negatieve celrest bij z0, waarde nét onder L bij z1, ambiguous/nonfinite binary en normalisatie die tekenklasse verandert worden afgewezen. Geen energie-epsilon in engine |
| Financiële regressies | Capchronologie, capoverschrijding/jaarreset/unknownhistorie en onafhankelijke baseline blijven met echte engine getoetst; nieuwe policy wijzigt TariffSnapshot niet |
| Config/deployment | DefaultNone activeert niets; benodigde maar ontbrekende policy weigert duidelijk; typo/onjuiste beleidsversie ongeldig; lichte configimport; geen HA-writes/harddeny-omzeiling |
| Schaalbaarheid | Representatieve batterijgroepen/horizonlengtes en kleine/grote U/L testen met echte CBC; timings, statuses, residuals en afwijzingen rapporteren. Geen succes of optimale oplossing aannemen bij node-/gaplimiet |

**Bestandsconsequenties na akkoord:** gerichte A8-interfaceaanvulling voor `dao/prog/settlement/optimization.py`, `dao/prog/day_ahead.py`, modeldumpadapter `dao/prog/da_debug.py`, optionele policy in `dao/prog/config/models/settlement.py`, bijbehorende gegenereerde configdocs en `dao/tests/settlement/test_optimization_bonus.py`/config-/producerregressies. Geen engine-, meet-/prijsrepository-, service-, capability-, hardware- of G2-herontwerp. Exacte velden, postsolve-signatures, TDD-opdrachten en commitgrenzen horen eerst in die nog te autoriseren uitvoeringsaanvulling.

## 9. Zelfreview en gevraagde goedkeuring

- De nieuwe verzameling `{0}∪[0,001,U]` heeft de benodigde gesloten positieve tak; boven L blijft energie continu. Constraintbewijs gebruikt geen gunstige objective.
- Groep/bronstroom/eenheid zijn expliciet; geen AC-binary, SoC-netto, verborgen prijs-/broncodering, DTO-quantum of hardwareminimum afgeleid.
- De producent bindt dezelfde variabelen als de energie-/SoC-balans; de nacontrole kijkt naar die oorspronkelijke cellen, niet alleen naar een toestemming.
- Mathematische exactheid, numerieke acceptatie, enginepariteit en fysieke uitvoering blijven onderscheiden. Een numeriek twijfelachtige uitkomst wordt geweigerd, niet als0 “veilig” verklaard.
- In de oorspronkelijke voorstelronde zijn alleen relevante documenten gelezen en dit voorstel geschreven; geen bestaande plannen/briefs/code/config gewijzigd, geen solver/tests/DB/HA/deployment uitgevoerd. Het daaropvolgende domeinakkoord en de reviewstatus staan bovenaan.

**Oorspronkelijke goedkeuringsvraag — inmiddels expliciet goedgekeurd:** *Keur je `aggregate_cell_min_1wh_v1` als planningsbeleid goed: in toepasselijke bonuskwartieren groepsgewijs óf exact nul óf minstens 1 Wh bruto celontlading, met mogelijke planuitval en strikte afwijzing van numeriek ongeldige uitkomsten, zonder wijziging van metingen/engine of hardwarevrijgave?*

Een ja keurt dit **domeinvoorstel** goed; het is op zichzelf geen toestemming om nu code, persoonlijke instellingen of live aansturing te wijzigen.
