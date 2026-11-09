# A8 — analyse van het bruto-celontlaaddomein

**Status: BLOCKED voor hervatting van A8. Onderzoek afgerond, geen domeinbeslissing goedgekeurd.**
**Basis:** codeHEAD `c0db962d99a407883c718664b81bf3145d370373`, 31 oktober 2026. Alleen gerichte lokale bestandslezing en dit analysedocument; geen code, tests, solver-/DB-/HA-operaties, dependencies of commits.
**Autorisatie:** onderzoek objective-onafhankelijke exacte bronkoppeling. Niet toegestaan: eenzijdige eligibiliteit als vervanging, verborgen epsilon/minimumenergie, stilzwijgende discretisatie of wijziging van de enginebronregel.
**Afbakening:** A1–A7, A10a, A7p en hun fixes niet heropenen. Bestaand plan/spec en beide amendments blijven ongewijzigd. G1–G3 en harddeny blijven open/ongewijzigd; dit document geeft geen hardwarevrijgave of leveranciersgarantie.

## 1. Besluit in het kort

De onderzochte code/config levert **geen positieve ondergrens op iedere niet-nulle bruto celontlaadenergie**, en geen onafhankelijke discrete bronindicator die exact gelijkwaardig is aan werkelijk nul versus positief celverbruik. De bestaande AC-binary, minimumvermogen, SOS-vermogensstappen en commandoronding zijn daarvoor geen bewijs.

Daarom kan de gevraagde grafiek op het huidige algemene continue domein niet exact als eindig begrensd MILP worden opgelegd. A8 blijft geblokkeerd. De capminimumrecursie heeft dit probleem niet, maar kan niet als complete A8 worden opgeleverd terwijl de bronkoppeling ontbreekt.

Voor hervatting is één nieuwe, expliciet goedgekeurde grondslag nodig:

1. aantoonbaar fysiek/modelmatig bewijs dat het bruto-celdomein een **positieve kloof boven nul** heeft, met correcte interval-, eenheids- en installatiebetekenis; of
2. toestemming om het **planningsdomein bewust te beperken**, bijvoorbeeld tot nul of minstens een benoemde celenergie, dan wel gehele veelvouden van een benoemd energiequantum. Dat geeft exacte bronlogica op een **ander domein**, niet een exacte representatie van de huidige continue beslisruimte.

Er is nu geen onderbouwde numerieke waarde voor zo'n minimum/quantum. Die wordt hier niet gekozen.

## 2. Vereiste en wiskundig obstakel

Volgens actieve spec §6/9 en `task-A8-brief.md`:

- Alle export verbruikt cap; `first_i = min(export_i, max(0, K-C_i))` is chronologisch exact.
- Bij bekende bron, geldig venster en positieve bonusprijs krijgt een interval met **nul bruto celontlading** precies `eligible_i = first_i`.
- Bij **enige positieve bruto celontlading**, uit welke batterij dan ook, krijgt het interval `eligible_i = 0`. Bruto is niet netto batterijvermogen, netto AC-export of SoC-delta.
- Beide richtingen moeten door constraints gelden, ook als de objective eligibiliteit minimaliseert of negeert.

Fixeer alle overige voorwaarden geldig, `first=1` en een eindige bovengrens `U>0`. De gevraagde verzameling is:

```text
S = {(0,1)} ∪ {(d,0) : 0 < d <= U}.
```

Voor `d_n=U/n` behoort `(d_n,0)` tot S, maar de limiet `(0,0)` niet. S is dus niet gesloten. Fixeren van een eindig aantal binaries/begrensde integers in een MILP levert een polyhedron; projectie daarvan is opnieuw een polyhedron en dus gesloten. De eindige unie over de discrete toekenningen is gesloten. Zij kan S niet zijn. Extra continue hulpvariabelen, big-M, indicatorconstraints of een eindige SOS-indeling lossen dit niet op; een strikte `d>0`-tak is precies het ontbrekende stuk.

Dit is een uitspraak over het **algemene ondersteunde domein** met willekeurig kleine positieve d. Een afzonderlijke vastgelegde instantie kan toevallig alleen d=0 toelaten, d vooraf vastleggen of een positieve minimumoplossing hebben. Dat is geen bewijs van een uniforme domeinregel voor de huidige A8-interface `cell_discharge: Sequence[mip.LinExpr]` of voor iedere horizon/configuratie.

De door de implementer gerapporteerde CBC-karakterisering maakt de ontbrekende richting zichtbaar: export1kWh, bruto celontlading0, first1, spot€0,10 en bonus30%; bij minimaliseren van eligible geeft de eenzijdige formulering 0/€0 terwijl de echte engine 1/€0,03 geeft. Dit rapport is gelezen, **niet opnieuw uitgevoerd**. De 19 eerdere groene drafttests bewijzen alleen niet deze richting; de aanvullende falende test is terecht blokkerend.

**Cap is anders:** `min`/`max` van begrensde affine expressies heeft een gesloten, stukgewijs-lineaire grafiek en kan met finite bounds/binaries in beide richtingen worden gemodelleerd. Dat maakt de nultest voor d nog niet representeerbaar.

## 3. Feitelijke code-/configbevindingen

Regelnummers verwijzen naar bovengenoemde HEAD; geen persoonlijke opties of live hardware onderzocht.

| Gecontroleerd pad | Feit | Waarom geen vereist domeinbewijs |
|---|---|---|
| `dao/prog/day_ahead.py:956–968` | `dc_from_bat` en `dc_to_bat` zijn CONTINUOUS, lb0, met maxima | Geen integerenergie, positieve minimumcelstroom of ontlaadbronbinary |
| `day_ahead.py:628–645`, `config/models/devices/battery.py:257–289` | `bat_to_dc_max_power`/`dc_to_bat_max_power` geven bovengrenzen; defaults zijn afgeleid van invertermaxima | Een maximum geeft geen positieve ondergrens |
| `day_ahead.py:1082–1104`, `battery.py:451–496` | SOS2-interpolatie van continue gewichten; nulstage wordt expliciet toegevoegd | Een positieve eerste stage is een knikpunt, geen minimum toegelaten vermogen. Gewichten kunnen naar nul; de nulstage is opzettelijk onderdeel van het domein |
| `day_ahead.py:1114–1137` | SoC-afhankelijke vermogensderating bestaat uit bovengrenzen | Geen sprong van nul naar een strikt positieve minimumenergie; grenzen kunnen juist naar nul dalen |
| `day_ahead.py:1170–1193` | Continue SoC-balans, DC-busbalans met PV, AC-aan/uit-binaries en exclusiviteit van AC-conversierichting | Binary zegt niet dat cellen ontladen. `ac_from_dc_on=1` kan bij nul AC-vermogen; AC-uitvoer kan volledig DC-PV-pass-through zijn |
| `battery.py:231–241`, `day_ahead.py:4364–4439` | `minimum_power` default0; na solve int/round op **netto AC-vermogen**, vaste ≤20W-uitbranch en soms minimumvermogen met berekend stopmoment | Geen constraint op bruto celenergie. Documentatie noemt afronden naar nul, uitvoer bevat ook conditionele duty-cycle. Geen van beide bewijst een uniforme celondergrens |
| `day_ahead.py:4397–4406,4437–4439` | Dutyduur uit verhouding gepland AC-vermogen/minimumvermogen; tijdconversie naar int en uiteindelijk minutenstring | Geen gedocumenteerde, in MIP afgedwongen minimale positieve **cel**duur. Commandotijdresolutie achteraf is geen bruto-energielattice |
| `day_ahead.py:195–219,442–446` | Eerste restinterval heeft eigen tijdfactor; overige stappen vaste intervalduur | Vermenigvuldigen van een continue stroom met positieve tijd verandert nabij-nulcontinuïteit niet. Een kort restinterval vereist eigen bounds |
| `battery.py:305–365`, `day_ahead.py:4448–4497` | Operating-modevelden zijn configureerbare strings; `entity_from_battery` ontvangt berekende uitvoer | Geen geverifieerde discrete bronstaat of onafhankelijk gemeten bewijs van bruto celontlading |
| `dao/prog/fastctrl/policy.py:635–656` | Minimumvermogen verdeelt AC-commandobereik; planvermogen binnen dead zone blijft expliciet toegelaten voor duty-cycle | Evenmin een ondergrens voor het optimizer-celdomein; niet terugprojecteren als bestaande MIP-regel |
| `dao/prog/fastctrl/simulate.py:80–108` | AC-commando/tijdsduur en rendementen bepalen continue energie, met verdere SoC-begrenzing | Simulator is geen bewijs van een celquantum of minimum gebeurtenisenergie |
| `dao/prog/settlement/engine.py:109–157` | Engine sluit `battery_discharge_kwh > 0` uit; bij nul en geldige overige voorwaarden wordt first uitgekeerd | De engine bevat geen drempel en mag die hier niet krijgen |

**Tegenvoorbeelden tegen afleiding uit AC:**

- Stel DC-naar-AC-vraag P en PV-aanvoer P: celontlading0 terwijl inverteruitvoer positief is. Dit moet zonder celontlading bonusgerechtigd kunnen blijven.
- Met PV-aanvoer `P-δ` en overige DC-stromen passend is celbijdrage δ. Zelfs een discreet AC-setpoint kan dus een willekeurig kleine positieve celbijdrage hebben. Een algemene geldige ondergrens vereist ook beheersing van mengstromen/duur, niet alleen AC-stepmetadata.
- Een celontlaad-**toestemming** of “ontlaadmodus” met `0 <= d <= U*z` laat bij z=1 nog d=0 toe. Eligibility aan z hangen verandert dan de enginebronregel. Ook een nieuwe celrichtingbinary zonder positieve onderste energiebound zou de ontbrekende richting niet leveren.
- Net SoC-/celenergie0 kan tegelijk positieve bruto laad- en ontlaadstromen verbergen. De bestaande AC-richtingsexclusiviteit bewijst geen celstroomexclusiviteit. Dit is een bekende G2-grens, hier niet repareren of als opgelost aannemen.

**Eenheid:** de SoC-vergelijking trekt `dc_from_bat * hour_fraction / eff_bat_to_dc` van celenergie af. De variabele zelf staat dus op de DC-busuitvoerzijde in kW. Bij eindige strikt positieve rendementen hebben bus- en celenergie dezelfde nul/positiefclassificatie; hun **numerieke minimum/quantum verschilt**. Een eventuele nieuwe grens moet expliciet op dezelfde bruto-cel-kWh-basis staan als de engine-invoer. De config accepteert voor het rendement zelfs 0 (`ge=0`); dat is geen bewijs van een geldige conversie. Voor een later domeinvoorstel zijn finite positieve conversies een validatievoorwaarde, niet een nu uitgevoerde config-/G2-fix.

## 4. Opties en trade-offs

### A. Een bewezen fysieke domeinkloof gebruiken — exact, maar bewijs ontbreekt

Veronderstel **voor ieder betrokken interval** aantoonbaar:

```text
d ∈ {0} ∪ [L,U],   0 < L <= U < ∞.
```

Dan kan met binaire z, waar z=1 betekent positieve bruto ontlading:

```text
L*z <= d <= U*z
0 <= eligible <= first
eligible <= F*(1-z)
eligible >= first - F*z       # 0 <= first <= F, finite fysieke/capbound
```

Bij d=0 dwingt L>0 z=0, dus eligible=first; bij d>0 dwingt de bovengrens z=1, dus eligible=0. Dit is objective-onafhankelijk exact **op dit domein**. Buiten venster/zonder bekende bron blijven de bestaande uitsluitingsregels gelden. Bij U=0 is d vast0; dat degenerate geval behoeft geen fictief L.

Een bewijs kan bijvoorbeeld een echte minimale bruto **cel**vermogensafname én minimale positieve duur en gegarandeerde ondergrens tijdens ramps/overgangen leveren. Het product daarvan vergt correcte eenheden en intervaltoerekening. Een vaste positieve aan-stroom alleen is niet voldoende wanneer de positieve duur naar nul kan. Een ontlaadevent dat een kwartiergrens kruist kan in één kwartier een willekeurig kleine staart hebben; een minimum per heel event bewijst dus nog geen minimum per leverancierkwartier.

Bij meerdere batterijen moeten alle bruto bijdragen niet-negatief en volledig inbegrepen zijn. Per-batterij garanties `d_b ∈ {0}∪[L_b,U_b]` kunnen met hun eigen binaries worden opgelegd; totale ontlading is positief zodra minstens één positief is. Geen saldering tussen batterijen. Alleen een aggregate L toevoegen is geen vervanging van ontbrekende per-apparaatfysica; het is slechts voldoende voor de nultest indien het aggregate domein zelf gerechtvaardigd is.

**Plus:** eenvoudige, sterk begrensde constraints; continue vermogens blijven mogelijk boven L; circa één extra binary per interval of per batterij/interval. **Min:** vereist echt domeinbewijs en verwerking daarvan in de producent, niet alleen een los getal in de bonushelper. Hardwaremapping/exclusiviteit/uitvoering blijven afzonderlijk G1/G2. Geen bruikbaar bewijs aangetroffen in de onderzochte code/config.

### B. Het planningsdomein expliciet beperken — mathematisch uitvoerbaar, nieuwe modelkeuze

Twee varianten, **geen van beide goedgekeurd**:

- **Semi-continu minimum als planningbeleid:** declareer voortaan zelf d=0 of d≥L, ook als de werkelijke installatie kleinere ontlading kan uitvoeren. Gebruik de constraints van A. Waarden `0<d<L` zijn dan **verboden**, niet gratis bonusgerechtigd en niet afgerond naar nul. Dit bewaart de enginebronregel voor toegelaten plannen, maar verwijdert voorheen haalbare plannen en kan andere kosten/haalbaarheid geven.
- **Energielattice:** kies expliciet bruto-celenergiequantum q>0 en modelleer `d=q*n`, `n` integer, `0<=n<=N=floor(U/q)`. Met binary z en `z<=n<=N*z` is z exact het positief-zijn van n. Dezelfde eligibleconstraints geven exacte enginepariteit op dit discrete domein. Bij N=0 is alleen d=0 mogelijk. Per-batterij quantums/multiples en intervalgrenzen moeten expliciet worden gekozen; een alleen aggregate quantum legitimeert geen per-batterij quantum.

Dit is geen tolerantieregel: echte of gemeten energie onder L/q wordt in de ongewijzigde engine nog steeds als positieve ontlading behandeld. Het is ook geen conclusie uit maximaal negen invoerdecimalen. Die regel begrenst representatie van DTO-input, niet het continue optimalisatieprobleem. `q=1e-9 kWh` wordt dus niet afgeleid of voorgesteld als automatische oplossing.

**Plus:** exact toetsbare bronlogica zonder de engine te wijzigen; geen hardwareclaim nodig indien eerlijk als **modelbeperking** goedgekeurd. Semi-continu L behoudt meer continuïteit dan een heel energielattice. **Min:** nieuwe inhoudelijke keuze over toegelaten planning, verliezen/SoC/intervalgrenzen, infeasibility en baselinevergelijking. Een lattice introduceert daarnaast begrensde integeraantallen; kleine q geeft grote N en potentieel slechte numeriek/rekentijd. Geen bewijs dat meer decimalen een betere uitvoerbare regeling opleveren.

Voor een modelbeleid is minimaal akkoord nodig op **betekenis, scope en concrete L/q-waarden of een verifieerbare parameterregel**, plus gedrag wanneer actuele bounds kleiner zijn, bij eerste restinterval, meerdere batterijen en baseline zonder optimalisatie. Niet alleen “voeg een binary toe”. Eerst dat besluit, daarna gerichte plan-/interfacecorrectie; hier geen nieuwe product-API vastleggen alsof het domein reeds bewezen is.

### C. Huidig continue domein én beide exacte richtingen behouden — nu stoppen

Zonder extra domeinbewijs/beperking bestaat geen geschikte finite-MILP-formulering voor de algemene gevraagde grafiek. Exact berekenen met de engine **na** oplossen is mogelijk voor een vast plan, maar maakt de MIP-eligiblevariabele of de eerdere planselectie niet exact. Alleen rapportage achteraf zou een andere A8-opdracht zijn.

**Plus:** geen onjuiste bonus-/hardwareclaim, geen wijziging van reeds goedgekeurde financiële regels of haalbare beslisruimte. **Min:** A8 kan niet worden hervat/afgevinkt op zijn huidige brief.

**Keuze van deze analyse: C, BLOCKED.** A heeft de voorkeur als werkelijk bewijs beschikbaar komt; B is een beslisbaar modelvoorstel, geen geautoriseerde oplossing. De huidige feiten rechtvaardigen geen L/q of discrete celmodus. Een simpele eenzijdige indicator zou onderhoudbaar lijken maar de expliciet gevraagde exactheid schenden; die optie is door de gebruiker niet gekozen.

## 5. Wat geen oplossing is

- `d <= M*(1-source)` zonder omgekeerde richting: conservatieve toestemming, niet exacte eligibiliteit bij d=0.
- Hoge bonusweging, slijtagekosten, tie-break of een tweede objective: selecteert hoogstens een gewenste oplossing, verandert de feasible set niet. Een financieel optimum is geen bewijs onder adversariële objective.
- Willekeurige epsilon of “kleiner dan solvertolerantie is nul”: verandert de bronregel of het domein zonder autorisatie. Bij first1 kan de bonus €0,03 springen terwijl d willekeurig klein is; een kleine energiefout impliceert geen kleine eurofout bij deze discontinuïteit.
- AC-standby, inverterbinary, positieve vermogensstage, helperstap van 1W, minuutstopmoment of SoC-meetresolutie als celquantum behandelen: geen sluitende keten naar bruto celenergie per interval.
- Binaire modus “mag ontladen” als “heeft daadwerkelijk positief ontladen”: de nul-in-actieve-modus-tak blijft mogelijk. Een bewezen eindige bronmodus met positieve energie per actieve mode zou pas optie A/B zijn.
- Alleen sensorbewijs van het vorige interval: bewijst niet de nog te kiezen toekomstige celstroom. Geen nieuwe leverancierinterpretatie of claim dat nulcommandos fysiek elke celontlading uitsluiten.

## 6. CBC-toleranties, schaalbaarheid en enginepariteit

**Drie verschillende exactheden:** (1) mathematisch exacte constraints op het gekozen domein; (2) een numeriek geaccepteerde CBC-oplossing; (3) exact/traceerbaar narekenen van een expliciete Decimal-prognose met de engine. Geen van die drie bewijst gemeten fysieke uitvoering of leveranciersuitbetaling.

De huidige optimizer zet `max_mip_gap_abs` via config (`day_ahead.py:3420–3424`, geklemd tussen 0,00001 en1 euro) en `max_nodes=1500`. Dat zijn optimaliteits-/zoekgrenzen, geen positieve celenergiedrempel. In het bekeken modelbouwpad is geen onderbouwde instelling van integrality-/primal-feasibility-toleranties voor deze nieuwe nulclassificatie vastgelegd. Hier worden geen CBC-defaults gegokt of aangepast.

Als A/B later wordt goedgekeurd:

- Bounds moeten per werkelijk interval/batterij finite en fysiek afgeleid zijn; expliciete eenheid en positieve rendementen. Geen generieke huge-M uit een onbeperkte LinExpr. Gebruik een strakke F voor first, niet een willekeurige eurocoëfficiënt.
- De domeinwaarde L/q komt uit het goedgekeurde model/bewijs, **niet uit een wens om de solver tevreden te stellen**. Daarna controleren of gebruikte schaling, integer- en primal-toleranties voldoende onderscheid/validatie mogelijk maken. Bij onvoldoende numerieke scheiding geen groen exactheidslabel; configuratie/solve weigeren of opnieuw laten beoordelen, niet stil L verhogen.
- Een fijn lattice kan U/q zeer groot maken en kleine kWh-coëfficiënten aan grote integerbounds koppelen. Herschalen lost dat niet automatisch op. Benchmark representatieve aantallen batterijen en 900s-intervallen, inclusief beperkte nodes/gap; geen claim van gegarandeerde optimale uitkomst of rekentijd. Semi-continu domeinen voegen O(intervals) of O(batteries×intervals) binaries toe; een lattice bovendien evenveel bounded integers. Dat is geen bewijs van praktische tractabiliteit.
- Valideer de **uiteindelijke** bronlogica/constraints en engine-uitkomst, niet alleen status optimal/feasible. Gebruik bronbewezen, gekende prognosefixtures voor parity; onbekende bron/cap blijft onbekend/zonder positieve bonus volgens bestaande regels.
- Een lattice kan zijn gedeclareerde hoeveelheid reconstrueren als Decimal(q)×gevalideerde integer n. Dat is expliciete modelreconstructie, geen algemene afronding van willekeurige positieve floats naar nul. Ook de overige balansen/rendementen/SoC moeten bij die reconstructie consistent zijn. Zonder lattice is eveneens een expliciete adapter-/residuvalidatie nodig; geen verborgen zero-clamp. Kandidaat weigeren als veilige classificatie/pariteit niet kan worden vastgesteld.
- Engine behoudt iedere positieve geregistreerde bruto ontlading als uitsluiting. Pariteit betreft dezelfde toegelaten/prognose-invoer met dezelfde eenheden, opening en onafhankelijke capontwikkeling van baseline en geoptimaliseerd scenario. Er komt geen fysieke quantisatie van metingen uit dit planningsvoorstel voort.

## 7. Expliciet benodigde beslissing/bewijs vóór hervatting

**Vraag aan gebruiker/controller:** wilt u eerst aantoonbaar hardware-/modelbewijs aanleveren voor een positieve minimale **bruto celontlaadenergie per relevant interval**, of wilt u een nieuwe, expliciete beperking van het planningsdomein laten ontwerpen en goedkeuren? Zolang beide ontbreken blijft A8 BLOCKED; “exactheid behouden” alleen levert de ontbrekende domeinsemantiek niet.

Voor de bewijsroute zijn nodig: betrokken installatie/topologie en conversiezijde; betekenis van bestaande modes; ondergrens op werkelijke bruto celbijdrage inclusief PV-menging, ramps, afschakeling, SoC-begrenzing en intervaloverschrijdende events; geldige tijdsduur/eenheden en uitzonderingen. Alleen een HA-helperbereik of fabrikantminimum-AC-vermogen is onvoldoende. Deze analyse vraagt geen live meting/activering en voert G2 niet uit.

Voor de modelroute zijn nodig: expliciet akkoord dat plannen met bepaalde kleine positieve celontlading **uitgesloten** mogen worden; keuze semi-continu minimum versus energielattice; parameters/parameterregel, interval-/batterijscope en eenheden; omgang met infeasibility, restkwartieren en baseline. Parameters zijn modelbeleid, niet tarief-/leveranciersvoorwaarden. Zonder die keuzes is er geen volledig onderbouwde nieuwe signature om aan een implementer over te dragen.

## 8. Concrete planconsequenties, alleen na zo'n besluit

- **Nu:** geen bestaande spec/plan/brief/API wijzigen; A8 niet hervatten met de draft; geen partial capimplementatie als gehele A8 claimen. Dit analysedocument is het enige nieuwe artefact.
- **Bij akkoord op A/B:** eerst een gerichte domein-/interfaceamendment maken. Producent `dao/prog/day_ahead.py` moet het domein daadwerkelijk aan alle relevante bruto celvariabelen opleggen en finite bounds/eenheden leveren. `dao/prog/settlement/optimization.py::build_bonus_terms` kan niet langer op alleen een ongespecificeerde `Sequence[mip.LinExpr]` vertrouwen: de goedgekeurde domeinparameters of bewezen indicator en hun binding moeten expliciet in het contract komen. Een los doorgegeven boolean zonder producerconstraints is onvoldoende.
- `dao/prog/settlement/engine.py`, meet-/prijsregister, service/publicatie/fallback en G1–G3 blijven buiten deze domeinwijziging. Eventuele nieuwe configuratie/DTO vereist dan afzonderlijk gedefinieerde validatie en de bestaande lichte Python3.11-importgrens; nu geen velden/defaults invoeren.
- **Benodigde tests na besluit:** adversariële min/max/constante objective bij d=0; d=L of q en grotere waarden; `0<d<L` respectievelijk off-lattice waarden expliciet infeasible (niet eligibility0 als alternatieve toegelaten oplossing); zero-only U=0; PV-pass-through; meerdere batterijen en geen netting; unit-/rendements-/restintervalconversie; ongeldige of numeriek ongeschikte domeinparameters; bron/cap onbekend; bestaande capchronologie/jaarreset. Via de echte `settle_series` narekenen, geen kopieformule.
- **Integratietests:** echte day-ahead producerbinding, zodat een vrijblijvende binary niet aan de helper kan ontsnappen; baseline eigen capstaat; representatieve CBC-grootte/stopstatus en reconstructieresiduen. G2-celstroomexclusiviteit wordt hierdoor niet ineens bewezen.
- Pas na goedgekeurde signatures komt een uitvoerbare TDD-/commitopdracht. Verwachte testpaden blijven `dao/tests/settlement/test_optimization_bonus.py` en bestaande MIP-/adaptertests; gerichte pytest, werkende Ruff op PATH en één volledige foreground-suite volgens constraints horen dan bij implementatie. In deze ronde niets daarvan uitgevoerd; geen nieuwe GREEN-, hardware- of runtimeprestatieclaim.

**Zelfreview:** non-closed bewijs geldt voor het bedoelde continue domein; positieve-gap/latticevoorstellen zijn expliciet conditioneel en wijzigen de haalbare set tenzij extern bewezen. Bruto-cel versus AC/bus/SoC en datum-/intervalsemantiek zijn onderscheiden. Geen aangenomen minimum, q, solvertolerantie of leveranciersregel als goedgekeurd gepresenteerd. Resterende blokkade is een echte domeinbeslissing, niet een ontbrekende big-M-truc.
