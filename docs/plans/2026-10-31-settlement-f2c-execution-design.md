# F2c — uitvoeringsontwerp: eigendom, ontvangende fencing en begrensde uitgifte

**Status:** ontwerp, 4 november 2026; gelezen codeHEAD `71feb38`, branch `fix/settlement-finalization-2026-11-02`. Geen implementatie, testuitslag, adapterkwalificatie of activeringsgoedkeuring. Uitvoeringsbijlage bij het bestaande source-domain-amendment, geen vervangend financieel plan.

**Leidend:** source-domain-amendment F2 en §15.7; publication-gate-amendment §3.1/4.1/4.2/5/6/8. De F2-bullets staan onder het kopje F2 vóór het kopje “15.6 Upstreamtriage”; hieronder verwijst “F2” naar die bullets. F1/F2a/F2b/B10d niet herbouwen.

## 1. Gelezen uitgangssituatie — verwijzingen op bovenstaande HEAD

| Feit | Codebewijs |
|---|---|
| Eén niet-reëntrant publicatieslot: threadlock plus nonblocking flock; lease is proces/thread/padgebonden. Atomische replace met beide fsyncs. | `dao/prog/fastctrl/publication.py:95-114,151-179,226-252` |
| Iedere v2 blijft hard geweigerd; v1 alleen vóór/zonder datumgate. | `dao/prog/fastctrl/publication.py:289-308` |
| Registratie houdt filelock buiten de DB-transactie. Callback invalideert vóór revisiewijziging, onderscheidt relevante correctie en gewone meterstaart. Kandidaatcheck vergelijkt exacte revisie/hash. | `dao/prog/settlement/repository.py:352-366`; `dao/prog/settlement/publication.py:50-69,92-163` |
| Publish checkt onder slot, bewaart/verifieert ReplayBasis en schrijft ready. | `dao/prog/settlement/publication.py:180-217` |
| Tick geeft vroege readslot vrij vóór sensoren en `_actuate`; nog geen finale sendgrens. | `dao/prog/fastctrl/runner.py:551-578,675-679` |
| `_actuate` retourneert momenteel **None**, bewaart receipts in `last_receipts`, werkt accepted state alleen bij confirmed bij en laat mislukte restore open. | `dao/prog/fastctrl/runner.py:861-906`; off-release: `712-756` |
| F2b-receipt mist epoch/sequence; plan identity is versie/created/replay-hash, geen volledige plandigest. Twee helperpogingen plus één waardereadback; geen absolute gatewaydeadline. | `dao/prog/fastctrl/actuator.py:28-86,89-131` |
| Generieke HA-gateway geeft bool op het slagen van een HA-aanroep, zonder ontvangend fencecontract. | `dao/prog/fastctrl/runner.py:187-224` |
| Controllerstate scheidt held/confirmed/restore. Policy kan held op planvermogen zetten zonder write; acceptatiehelper vertrouwt caller. Simulator past voorstellen rechtstreeks toe. | `dao/prog/fastctrl/policy.py:162-206,302-351,940-996`; `dao/prog/fastctrl/simulate.py:455` |
| `fast_state.json` is best-effort, reset bij ontbrekende/corrupte data, throttled; ongeschikt als fencingbewijs. | `dao/prog/fastctrl/runner.py:60-71,243-276,839-859` |
| Financiële optimizer verzamelt intents; legacy delegeert direct, ook batterijsetpoint/mode/stop. Vroege temperatuurwrite is financieel uitgesteld. | `dao/prog/day_ahead.py:5912-5917,4947-4979,2698-2708,4442-4452` |
| Financiële export gaat vóór de aparte hard geweigerde uitvoerpoging; legacy heeft eigen checks. | `dao/prog/day_ahead.py:5230-5284,6015-6044` |
| B10d heeft proceslevensduurlease met stabiel inode en close-zonder-LOCK_UN, inclusief descendantoverdracht; huidige resourcekeuze is taakgebonden. | `dao/prog/task_execution.py:1-11,41-87,90-153` |
| Profielhelpers en strict-v2-serialisatie bestaan: hergebruiken. | `dao/prog/fastctrl/plan.py:302-372,502-560` |
| Benefit wordt momenteel vóór `_actuate` geboekt; geen bewezen uitgevoerde euro-uitkomst. | `dao/prog/fastctrl/runner.py:663-675` |

## 2. Beslissingen C1–C5

### C1 — hetzelfde slot, expliciete G1-uitzondering, geen netwerktransactie

**Keuze:** exact `fast_plan.json.lock` behouden. Onder één ononderbroken lease: finale verse artefact-/tijd-/eigenaarcontrole → commandosequence duurzaam reserveren → volledige begrensde uitgifte inclusief eventuele retry/ACK-query → resultaat vastleggen. Geen publieke slotnemende functie vanuit een `_locked`-functie. Policy, sensoren, DNS/connect/auth-handshake, solver en replay blijven buiten deze sectie.

Het absolute HA-I/O-verbod uit publication-gate-amendment regel 87 kan niet tegelijk letterlijk gelden met F2-bullet 2. **Gerichte normprecisering:** het verbod blijft voor publicatievoorbereiding, vroege tick-read, repositorycallbacks en willekeurige HA-helpers; uitsluitend de nieuwe G1-sectie mag begrensde I/O met de gekwalificeerde ontvanger doen. Dit concretiseert de aangekondigde uitzondering uit dat amendment regel 184, niet bestaande HA-calls onder het slot schuiven. `publication.py` zelf importeert geen actuator; de eigenaarcoördinator houdt zijn lease tijdens de ontvangeroproep.

**Revisie zonder DB in de tick:** finale controle leest complete huidige ready-bytes opnieuw, vergelijkt volledige digest met het voorstel en valideert basis/lease. Geregistreerde relevante correcties schrijven eerst onder hetzelfde slot blocked; dit is het actuele revisiebewijs voor een **uitgegeven** plan. Geen per-tick DB-read of extra revisiebestand. Exacte globale revisiegelijkheid blijft kandidaat-/publishcheck. Gewone meterstaart mag een uitgegeven prognose niet alsnog ongeldig maken (publication-gate-amendment §3.2/6). Deze twee revisiebetekenissen niet vermengen.

**Intrekking na registratie:** registratie/consolidatie doet **helemaal geen gateway-/sink-I/O**, ook niet na commit (publication-gate-amendment §5). `before_revision` schrijft uitsluitend blocked vóór DB-wijziging. De owner consumeert dit artefact vervolgens en vraagt in zijn eigen nieuwe publicatielease een revoke/stopbarrière; daarvoor is geen bruikbaar planspec nodig. Ook bij DB-rollback na invalidatie blijft dit herstel nodig. Geen callback, post-commit-send of tweede bestandsnotificatie in de registrator. De vroege readlease is dan al vrij; de stopsectie acquiret eenmaal opnieuw, zonder nesting, en controleert de actuele owner/receiveridentiteit. Dit is veiligheidsintrekking, geen financieel commando uit een ongeldig plan.

- Correctie eerst: blocked duurzaam vóór DB-commit; latere send ziet blocked en doet nul commandosends.
- Send eerst: complete begrensde send eindigt vóór correctie het slot krijgt; daarna invalidatie en owner-intrekkingspoging. DB-commit is **geen fysieke stopgrens**. Alleen exacte barrier-ACK bewijst dat oude queued writes niet meer kunnen volgen. Tussen invalidatie en die ACK blijft uitvoering expliciet onbevestigd; geen onmiddellijke fysieke stopclaim.
- Owner/ontvanger onbereikbaar na correctie: registratie is uitgevoerd, plan blijft blocked, stopstatus unconfirmed/alarm. Ontvanger-expiry/watchdog begrenst de laatste werking; geen extra financiële rechten of DB-rollback. Een correctiejob hoeft dus nooit op HA te wachten.

**Duur/voortgang — vaste ontwerpbudgetten, geen vrijgavevlag:**

1. Financiële ownersectie: **maximaal 1.000 ms logisch budget voor de hele installatie**, niet per batterij/helper. Maximaal 750 ms voor transport/retry/query; overige tijd voor controle/persistence. Elke operatie gebruikt dezelfde resterende absolute deadline. Geen nieuw budget na retry of thread die na timeout onbeheerst HA schrijft.
2. Op dezelfde host: gedeelde `CLOCK_BOOTTIME`-cadans van 5 s. Finale ownersectie alleen starten in fase `[0,1)`; uiterlijk vóór fase 2 gereed. Fase `[2,5)` is voor publicatie/registratie. Ook vroege financiële readlocks uitsluitend binnen admissionvenster; maximaal één finale send- of stopsectie per cyclus. Veel batterijen krijgen geen extra secties. Financiële loop plant de eerstvolgende toegestane fase expliciet; alleen skippen op de bestaande periodieke tick kan anders permanent de verkeerde fase raken. Sensor-I/O buiten het slot kan het venster missen: dan voorstel weggooien/opnieuw controleren, niet alsnog sturen. Vanaf detectie kost stop maximaal circa 5 s wachttijd plus 1 s sectie; vanaf correctie komt de detectie-/ticktijd erbij. Geen kwalificatie voor installaties die een snellere veiligheidsreactie eisen; receiverwatchdog is onafhankelijk hiervan.
3. `publication_lock()` blijft nonblocking. Producer/registrator mogen **buiten** iedere lease/DB-transactie acquisitie herhalen, maximaal 6 s, maximaal iedere 50 ms. Niet hele solve of reeds gecommitte registratie herhalen. Elke succesvolle publicatiepoging doet opnieuw finale check. Timeout: geen mutatie/leaseverlenging. Andere publishers kunnen nog contention geven; de snelle laag kan niet permanent alle kansen bezetten.
4. Registratie houdt uitsluitend haar bestaande DB-/fsyncwerk onder slot. De 1 s begrenst de **nieuwe owner-commandosectie**, niet solver/server-DB/filesystem of de hele dagelijkse job. Ontvangerstop zit nooit in een DB-transactie of registratiejob.
5. Linux scheduling/fsync hebben geen harde real-time bovengrens. Deze bounds gelden bij responsieve host en deadlineconform transport. Bij host/I/O-stall blijft OS-slot exclusief; nooit timer-“stale”-ontgrendeling. Receiverdeadline/watchdog moet veilig aflopen. Ontbrekend bewijs van transport-/hostaannamen blokkeert kwalificatie; geen softwaretimeout als hardwaregarantie verkopen.

**Lockvolgorde:** levensduurlease → publicatieslot → korte lokale persistence/DB-transactie. Geen DB-transactie tijdens ontvanger-I/O; receiver roept nooit DAO/publicatie terug. Levensduurlease vóór loop/start verwerven, niet vanuit publicatielease. Geen lock in ACK-callbacks of recursive fallback: sendfout markeert pending, volgende ownersectie herstelt. Fallbacksink buiten vroege readlease mag een nieuwe niet-geneste ownersectie vragen. Receiver-noodstop mag zelfstandig alleen intrekken/stoppen; hij wacht niet op de cadence en kan geen niet-nulrecht uitgeven.

**Afgevallen:** check/unlock/send (klein maar TOCTOU); tweede sendlock/capabilityfile (twee waarheden/lockordering); alleen langere nonblocking lock (dagpublicatie kan steeds botsen). **Kosten als verkeerd:** stale financieel commando, deadlock of langdurig ontbrekende dagplanning. Cadans/budget wijzigen vereist opnieuw contentionbewijs, niet een bypassinstelling.

### C2 — financiële eigenaar is de snelle laag; legacycompatibiliteit is geen eigendomsbewijs

**Keuze:** runner is enige positieve commando-eigenaar voor **alle batterijen van één financiële installatie**. Optimizer levert plan/intenties. Receiver is de enige laatste fysieke writer; operator heeft stop-only bevoegdheid, registrator uitsluitend invalidatiebevoegdheid. Tweede runner krijgt geen levensduurlease. Hergebruik B10d met kleine generieke resource-acquire in `task_execution.py`, geen PID-/heartbeatmechanisme.

**Onvermijdelijke scopegrens:** universeel één batterijschrijversproces én ongewijzigde v1 zijn niet samen haalbaar; de twee huidige paden zijn hierboven geciteerd. V1 blijft daarom exact legacy en **ongekwalificeerd voor F2-eigendom**. Geen stille verplaatsing van v1-writes naar runner of uitschakeling van legacy-override. F2c bewijst exclusiviteit voor de nieuwe route; universeel installatiebewijs blijft open tot gecontroleerde migratie. Als “één schrijver” ook vóór migratie voor alle v1-installaties geldt, vergt dat expliciete wijziging van R11, niet een implementerdetail.

**Geen migratie-overlap:** eerst oude optimizer/runner/automations quiescent, oude endpointcredentials/routes afsluiten, dan geauthenticeerde revoke/veilige barrière, dan nieuwe owner. Geen automatische legacy→fenced-overgang door datum/v2-file/config. Datumgate blokkeert legacyrechten maar bewijst geen lege oude queues. Downgrade vereist gecontroleerde stop en blijft buiten automatische recovery.

| Financiële lifecycle | Expliciete uitkomst |
|---|---|
| Normale shutdown/off/handoff | Nieuwe niet-nulopdrachten verbieden; revoke/barrière onder eigen ownersectie. Alleen exacte veilige ACK ⇒ released. Anders unconfirmed/recovery_required, nooit “released omdat thread eindigde”. |
| Procescrash/OS-lease vrij | Nieuwe owner start inhibited, reconcilet receiver/anker en verkrijgt nieuwe epoch/barrière; fast_state is hoogstens historie. |
| Operatorstop/handmatig verzoek | Receiver trekt huidige epoch in; DAO verliest niet-nulrecht en blijft inhibited. Geen handmatige tweede HA-writer introduceren. Handmatige niet-nulbediening valt buiten F2c en moet later via dezelfde gekwalificeerde laatste writer/ownerhandoff. |
| Herstart na stop of terugkeer active | Nieuwe session, veilige barrière en receiverbinding plus verse plancontrole; modewissel alleen herstelt geen bevoegdheid. |

Bij stopfout houdt een nog draaiende owner zijn OS-lease zolang hij nog kan schrijven; shutdown mag uiteindelijk zijn descriptor sluiten, maar kan daarmee nooit pending herstel als voltooid markeren. Descendants behouden de lease totdat ook hun schrijfvermogen weg is.

**Identiteit:** exact gevalideerde settlement-`meter_id` als `installation_id`, beperkt tot één uitvoerinstallatie per meter. Dit is al een stabiele metering-installation identity (`dao/prog/config/models/settlement.py:141-142`). Receiverprovisioning bevestigt dezelfde binding plus unieke batterijnamen. Geen hostname/cwd/PID/index als identiteit. Meer installaties achter één meter of meter-idwijziging: apart provisioningbesluit.

**Epochautoriteit:** uitsluitend receiver kent nieuwe epoch toe, strikt groter dan duurzame high-water mark. Sender vraagt met unieke owner-session-id en verwachte huidige epoch; concurrente wijziging faalt CAS. Geen tijdstempel. Startup/handoff/operatorstop/ambigue send vereisen intrekking; vrij OS-slot betekent niet veilige oude epoch.

**Operationele spiegel:** `<plan-parent>/actuation_owner/<sha256(installation_id)>.json`; B10d-resource `actuator-<hash>` in canonieke leasedirectory. Alleen owner schrijft spiegel, onder levensduurlease en publicatielease. Uitsluitend operationele toestand (§3), geen tarieven/revisietoestemming/gateboolean. Temp+flush+fsync+replace+directory-fsync, restrictive rechten, geen symlinks/best-effort fallback. Sequence vóór send duurzaam reserveren; gaten toegestaan, hergebruik niet.

**Rollback:** lokale files kunnen eigen rollback niet detecteren. Receiver-high-water mark vereist een **onafhankelijk, niet met die files teruggezet monotoniciteitsanker**. Concrete hardware/extern vertrouwd opslagbewijs is kwalificatiewerk. Iedere restart begint inhibited en vergelijkt geauthenticeerd met anker; ontbrekende/corrupte/teruggezette staat of onbekende queue ⇒ recovery_required, nooit epoch=1/defaults. Geen “max(lokaal,remote)+1 en doorgaan” bij onbekende herkomst. Herstel vereist aantoonbare queuequiescentie/veilige barrière en nieuwe epoch. Verlies anker vereist reprovisioning met uitsluiting van alle oude credentials/opdrachten. Fake bewijst weigeringen, geen echte rollbackbestendigheid.

Eerste provisioning is eveneens expliciet: alleen de synthetische fixture maakt een vertrouwde initiële safe/inhibited staat en anker. Productiebootstrap kan ontbrekende staat niet onderscheiden van verlies en mag die dus nooit automatisch initialiseren.

**Afgevallen:** optimizer als owner (kortlevende solvejob beheert geen doorlopend profiel/watchdog); twee writers met korte mutex (twee beleidsautoriteiten zonder handoff); lokale JSON als enige epochbron (restore heractiveert oude opdracht). **Kosten als verkeerd:** split-brain, oude niet-nulopdracht na restart of ongemerkte legacybreuk.

### C3 — ontvangende adapter, niet een wrapper om HA-helpers

**Kandidaat:** native fenced device-gateway (`FencedDeviceGateway`-protocol), die exclusieve actuatorcredentials en de laatste schrijfqueue bezit. Kandidaattransport: vooraf verbonden lokale Unix `SOCK_SEQPACKET`, afgeschermd pad en geverifieerde peer-UID/service-identiteit aan beide kanten; eigenaarrol/installatie/session aan sessie gebonden. Payloadidentiteit alleen is geen authenticatie. Remote variant vereist gelijkwaardig auth-/clockbewijs.

Dit is een concrete integratiegrens, **geen claim dat een apparaatdriver al bestaat**. F2c levert contract/coördinator/fake, geen nieuwe daemon of generieke HA-integratie. Werkelijke apparaat-/HA-integratie is pas kandidaat als zij tot laatste writer fences controleert, of aantoonbare downstream purge/drain plus veilige stopbarrière heeft. Ondoorzichtige HA-automation/apparaatqueue diskwalificeert. `HomeAssistantGateway` blijft expliciet `legacy_only`, nooit via duck typing/bool/configflag qualified.

Receiver reserveert `(epoch,sequence,command_id,payload_digest)` duurzaam **vóór** effect en controleert opnieuw bij **ieder uiteindelijk helper-/device-effect**. Duplicaat send wordt afgewezen, nooit opnieuw uitgevoerd; aparte query kan exacte eerdere ACK teruggeven. Oude sequence/epoch, verkeerde installatie/session, andere payload onder dezelfde id en verlopen opdracht afwijzen. Hogere sequence invalideert ook wachtende lagere sequence-opdrachten. Crash tussen effect en ACK ⇒ unconfirmed, niet opnieuw schrijven.

Binnen de receiver zijn effectcommit en revoke lineair geordend: er mag geen onbeheerde check→asynchrone write-kloof achter de laatste fence zitten. Een al begonnen effect vóór revoke moet eindigen of aantoonbaar geneutraliseerd zijn vóór barrier-ACK. Die interne queueordening is geen tweede DAO-publicatieslot/capabilityprotocol; zij is juist het vereiste ontvangende uitvoercontract.

Revoke verhoogt eerst duurzaam epoch en inhibiteert restart; maakt vervolgens oude queues ongeldig/drained; bevestigt dan veilige toestand. Alleen daarna **barrier ACK**; niets van oude epoch kan daarna actueren. Gedeeltelijk slagen houdt nieuwe epoch revoked/inhibited en safe_state onbekend. Nieuwe owner krijgt pas niet-nul na barrière, ownerbinding en verse plancontrole. Noodstop mag zonder financiële lease, maar uitsluitend veilige toestand/nieuwe epoch; nooit verlopen financieel commando als herstel.

**Clock:** wire-UTC in integer ms; lokale budgets monotone. Clock health geeft bewezen `epsilon_ms`; onbekend ⇒ geen niet-nul. Deadlines conservatief vóór planlease/profielgrens; receiver vereist `receiver_now_upper < execute_before_ms`, ook bij daadwerkelijke uitvoering. Watchdog beëindigt werking uiterlijk `hold_until_ms`; “mag starten vóór deadline” is onvoldoende. Clockjump/terugloop/suspend/ongeldige bootidentiteit ⇒ inhibited/herstel, nooit leaseverlenging. Gedeelde hostklok sluit hostklokfouten niet uit.

**Afgevallen:** HA-helperretry/readback (annuleert geen oude write); proxy vóór onbekende queue (fence niet op effectgrens); timeout plus wachten (geen ordeningsbewijs). **Kosten als verkeerd:** oude write start batterij na de zogenaamd bevestigde stop.

### C4 — contract → receiver → owner → serialisatie → consumers → bewijs

Zes sequentiële sneden (§4), ieder RED/GREEN en onafhankelijke review. Tekstuele F2-volgorde is geen uitvoeringsvolgorde: bounded send/stop kan pas geloofwaardig getest na receiver/barrière. Eerst contractbewijs, dan consumerbinding; **niet harddeny openen voor bereikbaarheid**. Geen parallelle wijzigingen aan gedeelde actuator/runner/day_ahead-files.

Alles-in-één beperkt tijdelijke interfaces maar maakt foutoorzaak/review onhanteerbaar. Eerst runner aan HA koppelen levert onveilige tussenstand. Verkeerde volgorde kost herwerk en kan vals groen hardwarebewijs produceren.

### C5 — softwarebewijs, geen activering

Geen F3-twee-scenario-settlement/replay/eurovergelijking; geen E1-factuur-/backend-/restorebewijs; geen E2-hardware/watchdogkwalificatie/activering. Geen nieuw boek/model, tweede financieel capabilityartefact, bypassflag, YAML-stopgarantie of onbewaakte HA-route. Profiel/ReplayBasis/publicatieformats behouden. Financiële niet-batterijintenties blijven **ongeconsumeerd/hard denied**: geen gedeeltelijke boiler/EV/PV/HP/start-stopactivering. Latere uitvoering moet hetzelfde finale slot-/basis-/receivercontract gebruiken; F2c bouwt geen onveilige intent-dispatcher.

“Nu alleen v2-planvolgen activeren” valt af wegens ontbrekende F3/E1/E2/receiverkwalificatie. Tweede uitvoerrechtenfile valt af: operationele epochstaat mag alleen beperken, nooit financiële toestemming creëren. Verkeerde afbakening produceert veiligheids-/financiële claims zonder bewijs.

## 3. DTO's, interfaces en invarianten

Hieronder **ontwerp**, geen aanwezige code. DTO's frozen; strict wirevalidatie, geen bool→int/coercion/NaN/duplicaatkeys/ontbrekende identiteit. Epoch/sequence positieve integers zonder wrap; uitputting blokkeert. Sequence installatiebreed, niet per batterij. Canonieke JSON `sort_keys=True,separators=(",",":"),allow_nan=False`, UTF-8; SHA-256 over volledige gevalideerde `plan.to_dict()`-snapshot inclusief profiel/context/replayhash, niet korte `plan_identity`.

```python
# fastctrl/command.py (nieuw; puur, geen policy/HA/DB-imports)
BatteryCommand(installation_id: str, battery_name: str,
    owner_session_id: str, owner_epoch: int, sequence: int,
    command_id: str, plan_digest: str,
    issued_at_ms: int, execute_before_ms: int, hold_until_ms: int,
    setpoint_w: int, mode: str, stop_at_ms: int | None)

ActuationReceipt(installation_id: str, battery_name: str,
    owner_epoch: int, sequence: int, command_id: str, plan_digest: str,
    accepted_ts: int | None, status: Literal['accepted','rejected','unconfirmed'],
    helper_results: tuple[HelperResult, ...],
    # Aanvullend bewijs, geen extra §15.7-statussen:
    owner_session_id: str, payload_digest: str, reason: str,
    observed_at_ms: int, effect: Literal['none','complete','partial','unknown'])

RevokeRequest(installation_id: str, expected_epoch: int,
    barrier_id: str, reason: str, requested_at_ms: int)
BarrierReceipt(installation_id: str, barrier_id: str,
    revoked_through_epoch: int, current_epoch: int,
    status: Literal['accepted','rejected','unconfirmed'], accepted_ts: int | None,
    queues_fenced: bool, safe_state: Literal['confirmed','unknown'],
    receiver_generation: str, reason: str)

Deadline(monotonic_end_ns: int, execute_before_ms: int)
ClockSample(utc_ms: int, boottime_ns: int, boot_id: str,
    uncertainty_ms: int, healthy: bool)
OwnerBinding(installation_id: str, receiver_id: str, receiver_generation: str,
    owner_session_id: str, owner_epoch: int, barrier_id: str,
    owner_not_after_ms: int)
```

`command_id` uniek, bijvoorbeeld `installation-hash/epoch/sequence`; retry exact dezelfde bytes/id/deadline, nooit nieuwe sequence voor onzekere opdracht. Geen timestamp/indexidentiteit. Payloaddigest bindt vermogen/mode/stop/tijden. Entity-id's komen uit gepinde receiverprovisioning, niet vrije HA-servicecalls. Alle vereiste helperuitkomsten gedekt; alleen setpointsucces accepteert geen complete opdracht.

`accepted_ts`: UTC-ms van **volledige acceptatie op laatste writergrens**, niet gemeten vermogen/transportontvangst/proposal timestamp; alleen gevuld bij accepted. Rejected vereist bewezen géén effect door deze opdracht; gedeeltelijk/ambigu altijd unconfirmed. Rejected duplicate wist eerdere werking niet. ACK moet exact pending command/barrier plus payload dekken, anders unconfirmed. Late ACK na lokale deadline mag uitsluitend historie aanvullen, nooit herstelplicht opheffen.

`issued_at_ms < execute_before_ms <= hold_until_ms <= min(plan.not_after_ts*1000, toepasselijke profiel-/intervalgrens, owner_not_after_ms)`. Lokale deadline met clockmarge nooit voorbij execute_before. Kortere resterende lease verkort 1s-budget; onvoldoende restbudget ⇒ niet verzenden. Werking na profielgrens vereist nieuw commando, niet wachten op toekomstige tick. Deadlineklok is overal dezelfde `CLOCK_BOOTTIME`, niet mengen met suspend-uitsluitende monotonic. Een definitieve planidentiteitswissel of verlopen ownerbinding vereist eerst stopbarrière/nieuwe epoch, geen stil herlabelen van oude queued opdrachten.

```python
# fastctrl/receiver.py (nieuw; contract, geen productiedaemon)
class FencedDeviceGateway(Protocol):
    def inspect(self, *, deadline: Deadline) -> ReceiverSnapshot: ...
    def revoke_and_stop(self, request: RevokeRequest,
                        *, deadline: Deadline) -> BarrierReceipt: ...
    def bind_owner(self, *, installation_id: str, barrier_id: str,
                   expected_epoch: int, owner_session_id: str,
                   requested_not_after_ms: int,
                   deadline: Deadline) -> OwnerBinding: ...
    def send(self, command: BatteryCommand, *, deadline: Deadline) -> ActuationReceipt: ...
    def query(self, *, installation_id: str, owner_epoch: int,
              sequence: int, command_id: str, payload_digest: str,
              deadline: Deadline) -> ActuationReceipt: ...
```

`inspect` verleent geen recht. Snapshot bevat installatie/receiver-generation/anchor-checkpoint, epoch/high-water sequence, bound session, inhibited, queue-known, laatste barrière. `bind_owner` bindt de door exacte barrière veilig gemaakte epoch eenmalig; vergelijking faalt na concurrente operatorstop. De receiver begrenst de gevraagde ownerdeadline met zijn gekwalificeerde maximale lease, bewaakt die zelfstandig en verlengt niet door query/retry. Auth-rol bij iedere call controleren. Registrator heeft geen receiverpoort. Fallback-`acknowledged` blijft gatewayacceptatie; alleen uitgebreide `BarrierReceipt` kan stopbarrière bewijzen.

```python
# fastctrl/ownership.py (nieuw)
OwnerState(schema_version: int, installation_id: str, receiver_id: str,
    receiver_generation: str, anchor_checkpoint: str,
    owner_session_id: str, owner_epoch: int, last_reserved_sequence: int,
    phase: Literal['inhibited','binding','owned','unconfirmed','revoking','released'],
    pending_command: BatteryCommand | None,
    last_receipt: ActuationReceipt | None,
    pending_barrier_id: str | None, last_barrier: BarrierReceipt | None)

# task_execution.py: bestaande taakfuncties blijven gelijkgedrag-wrappers.
acquire_resource(resource: str, *, lease_dir: Path) -> Lease

OwnerStore.load_strict() -> OwnerState       # absent/corrupt: RecoveryRequired
OwnerStore.write_locked(state: OwnerState, *, lease: PublicationLease,
                        writer: Lease) -> None
OwnerCoordinator.start(*, writer: Lease, plan_digest: str,
                       requested_not_after_ms: int) -> OwnerBinding | RecoveryRequired
OwnerCoordinator.stop(*, reason: str) -> BarrierReceipt

# Adapter op bestaand publication.FallbackSink; geen losse HA-stoproute.
OwnerFallbackSink.request(request: FallbackRequest) -> FallbackReceipt
OwnerFallbackSink.last_barrier: BarrierReceipt | None
```

Receiver heeft eigen durable `ReceiverState`: schema/installatie/generation/anchor-checkpoint, epoch-high-water, bound session/deadline, sequence-high-water, id+digest+fase+resultaat van relevante opdrachten, inhibited, pending/laatste barrière en queuegeneratie. Reserveer/persist vóór effect; pending effect na restart niet opnieuw uitvoeren. Bewaar replayafwijzing tot alle opdrachten van epoch verlopen én onuitvoerbaar zijn; high-water/epoch nooit resetten. Anker bindt zowel epoch als state-generation/high-water: een oude checkpoint met dezelfde epoch mag niet onopgemerkt sequence hergebruiken. Crash tussen anker- en state-update ⇒ inhibited/reconcile, niet default-herstel. Onbekende queue blijft inhibited. Fake gebruikt aparte sender/receiverfiles plus onafhankelijk fake-anchor; restoretests zetten ook receiverfiles terug maar niet anker. Fake-anchor nooit als productieoplossing exporteren. Meerdere batterijen worden sequentieel binnen één installatiebudget behandeld: maximaal één pending command; bij ambigue uitkomst geen volgende niet-nulwrite, maar installatiebreed herstel. Een hoger sequence maakt alleen nog wachtende oude effecten ongeldig, niet de afzonderlijke bevestigde hold van een andere batterij.

`OwnerFallbackSink` gebruikt alleen gepinde receiverprovisioning/ownerstaat, nooit entiteiten uit blocked/corrupt plan. Buiten admissionvenster geeft hij requested met pending herstel, bij onbekende ontvanger unconfirmed. Alleen volledige exacte barrier-ACK geeft acknowledged en ruimt herstelplicht op; fysieke meting blijft afzonderlijk. Deze adapter wordt in F2c uitsluitend met fake receiver gebruikt; de bestaande default no-I/O-fallback blijft behouden.

```python
# fastctrl/execution.py (nieuw; geen DB/HA/solver-import)
plan_digest(plan: FastPlan) -> str
validate_current_locked(*, lease: PublicationLease, expected_digest: str,
                        clock: ClockSample, policy: GatePolicy) -> FastPlan
send_locked(command: BatteryCommand, *, lease: PublicationLease,
            owner: OwnerCoordinator, receiver: FencedDeviceGateway,
            deadline: Deadline) -> ActuationReceipt

# fastctrl/publication.py: acquisitionbeleid, geen tweede slot.
publication_attempts(path: Path, *, deadline_ns: int) -> ContextManager[PublicationLease]

# runner.py: retourneren én last_receipts vullen; no-op: lege tuple.
FastControlRunner._actuate(decision, plan, mode) -> tuple[CommandReceipt, ...]

# policy.py: receipt aan pending command gebonden; geen I/O.
apply_actuation_receipt(state: BatteryControllerState,
                        receipt: ActuationReceipt) -> None
```

`validate_current_locked` controleert structuur/identiteit/basis/tijd, **vervangt niet** `following_verdict`. Publieke runnerentrypoint blijft harddeny toepassen. `send_locked` roept die finale validatie zelf aan met de commandodigest en actuele klok, controleert live writer/publicatielease, pending immutable command en receiverbinding; een eerdere callercheck volstaat niet. Het is een interne mechanische grens, geen alternatieve publieke execute-API. Geen `allow_financial`, `skip_gate` of runtime-testmodus. Tests roepen mechanische grens met fake receiver aan; aparte entrypointtests bewijzen dat productie-v2 haar nog niet bereikt. `publication_attempts` herhaalt alleen acquisition en yieldt precies één lease; nooit callerbody/DB-commit opnieuw uitvoeren. `OwnerCoordinator.start/stop` verwerven zelf één lease; interne `_..._locked`-varianten nemen de al gehouden lease, nooit nesting.

### F2b → §15.7: minimale, expliciete compatibiliteitsgrens

- Hernoem bestaande receipt tot `LegacyActuationReceipt`; behoud velden/statuses/deterministische id/retry/readback en `execute_battery_command(...)` voor **v1**. Nieuwe canonieke `ActuationReceipt` in `command.py`, re-export vanuit `actuator.py`. Verzamelingen: `CommandReceipt = ActuationReceipt | LegacyActuationReceipt`; geen verzonnen epoch/installation-defaults.
- Verplaats `HelperResult` ongewijzigd naar de pure `command.py` en re-export vanuit `actuator.py`; zo kan policy het canonical receipt importeren zonder de bestaande actuator→policy-afhankelijkheid cyclisch te maken. Legacy receiptserialisatie blijft exact gelijk; geen oud persisted receipt als financieel bewijs migreren.
- **Geen** alias `confirmed == accepted`: helperacceptatie mist fence/laatste-writerbewijs. Geen `unknown -> rejected`: gedeeltelijke write blijft ambigue. Legacylogs/tests houden oude statuses; financiële protocol alleen accepted/rejected/unconfirmed. Eventuele telemetrynormalisatie heet `legacy_helper_confirmed`, nooit canonieke accepted-receipt.
- `_actuate` retourneert receipts/lege tuple; `last_receipts` blijft compatibel. Zonder geldige owner/adapter is er nog geen canonical commandidentiteit: lege receipttuple plus expliciet deny-verdict (`adapter_unqualified`/`owner_unavailable`), nooit verzonnen epoch. Een al geïdentificeerde opdracht die aantoonbaar vóór send wordt geweigerd krijgt rejected. Shadow/deadband ⇒ geen receipt/accepted-update; onderscheid blijft zichtbaar in verdict/decision.
- Canonieke `apply_actuation_receipt` matcht installatie/batterij/epoch/sequence/id/digest met pending state. Alleen accepted neemt **opgeslagen pending-commandwaarden** plus accepted_ts over, niet mutable latere decision; ms expliciet naar bestaande controllerseconden converteren. Oude/wrong ACK muteert niets. `apply_accepted_command` blijft expliciet legacy/synthetische simulatie; geen F3-herbouw.
- Financiële controller krijgt `execution_status=unknown|accepted|unconfirmed|stopped`, pending-commandidentiteit, last-accepted-identiteit en `recovery_required`. Held/accepted blijven gescheiden. Partial/timeout: vorige ACK uitsluitend historie, held niet als feitelijk lopend vermogen gebruiken; geen held-sensorfallback/euroclaim over onzeker segment. Waardereadback alleen herstelt dit niet.
- `restore_pending` niet wissen bij timeout/off/`_consume_block`. Financieel herstel is stopbarrière, niet oud planvermogen terugschrijven. Alleen exacte veilige barrière eindigt herstelplicht; daarna verse ownerbinding/plancheck. Legacyrestore blijft gelijk.
- Financiële init/profielovergang moet werkelijk commando voorstellen: legacy-aanname “optimizer heeft planvermogen al gezet” (`policy.py:958-962`) geldt niet. V2-held niet bijwerken op alleen planreconciliatie; ontbrekende acceptatie/profielgrens niet door deadband verbergen. `economics_supported=False` verhindert overrides, niet uiteindelijk gecontroleerd planvolgen. Alleen synthetisch testen, harddeny blijft.

## 4. Sneden, concrete files, regressies en bewijsgrenzen

Iedere snede: bedoelde RED → minimale implementatie → GREEN/aangrenzende regressies → onafhankelijke spec-/veiligheidsreview → fix/herreview vóór volgende snede. Uitvoerder legt testnamen/exitcodes/bewijsgrenzen/files vast in bestaande ledger. Geen implementatiecommits vanuit deze architectopdracht.

### F2c1 — commandocontract en receiptcompatibiliteit

- **Doel/eisen:** immutable command/ACK/barrier/deadline, volledige plandigest, legacy/canoniek zonder HA-ACK opwaarderen.
- **Files:** nieuw `dao/prog/fastctrl/command.py`; `actuator.py` (legacytype/re-export/v1-only); `runner.py` (returntuple/imports); `policy.py` (receipttoepassing/pendingvelden). Nieuw `dao/tests/prog/test_fast_command.py`; bestaande `test_fast_actuator.py`, `test_fast_runner.py`, `test_fast_policy.py`; zo nodig uitsluitend simulatorimports/tests.
- **Regressies:** F2b helperfouten/150W-deadband/shadow/restore behouden; legacyreceiptvelden/waarden/status exact gelijk; canonical bool/NaN/tijdvolgorde/identiteit afwijzen; andere payload bij zelfde created/replayhash geeft andere digest; wrong ACK verandert niets; `_actuate` retourneert resultaten; oude controllerfile leesbaar maar nul financiële rechten.
- **Grens/dependency:** baseline F2b; geen receiver/actieve route. Review op semantische typebreuk, geen “alleen rename”-goedkeuring.

### F2c2 — deterministische ontvangende toestandsmachine

- **Doel/eisen:** fence bij ontvangst én effect, durable reserve vóór effect, exacte ACK/query/revoke/stop/bind, rollbackafwijzing.
- **Files:** nieuw `dao/prog/fastctrl/receiver.py` (contract/validatie, geen daemon); nieuw `dao/tests/prog/fakes/fenced_receiver.py`, `test_fast_receiver.py`. Fixturepad mag bestaande conventie volgen, protocol niet veranderen.
- **Fake:** expliciete `enqueue`, `deliver`, `step_helper`, `lose_ack`, `advance_clock`, `crash`, `restart`, `restore_files`; handmatig aangedreven eventqueue, fake UTC/monotone klok, tmpstorage, onafhankelijk fake-anchor, effectlog. Geen sleeps als racevolgorde, geen HA/netwerk.
- **Regressies:** oud niet-nul hangt → timeout → nieuwe epoch/stopbarrière ACK → oude delivery → **geen nieuw effect**. Herhalen met retry, tussen helpers, crash vóór/na reserve/effect/ACK, receiverrestart en teruggezette/ontbrekende staat. Received-event nooit accepted. Lage sequence/changed payload/verkeerde principal/installatie/verlopen queueopdracht afwijzen; query ≠ resend; clockjump/leasegrens; safe-state onbekend geeft geen barrier ACK/bind. Barrière omvat alle batterijqueues.
- **Grens/dependency:** c1; bewijst fake-ankermodel, geen echte downstream/opslagveiligheid. Review vóór ownerbouw.

### F2c3 — exclusieve owner, durable mirror en lifecycle

- **Doel/eisen:** B10d-resourcelease, sequence reserve, inhibited startup, restart/handoff/operatorstop/timeout via epochbarrière, nooit legacyfallback.
- **Files:** `dao/prog/task_execution.py` (generieke primitive, taskwrappers intact); nieuw `fastctrl/ownership.py`; lifecyclehooks `runner.py`; nieuw `dao/tests/prog/test_fast_ownership.py`, bestaande `test_task_execution.py`.
- **Regressies:** twee echte sandboxprocessen niet beiden owner; parentclose/exec/descendant behoudt lease; heartbeat/PID geeft geen recht. Crash na reserve vóór send slaat sequence over; fsync/replacefout vóór send. Missing/restore mirror of receiver, onbekende queue/verloren anker ⇒ inhibited. Stop/handoff-failure blijft pending; oude barrier ACK releaset nieuwe owner niet; operatorstop tegen bind/send. Vrij OS-slot zonder barrière geeft nul niet-nul.
- **Grens/dependency:** c1–2; lifecycle uitsluitend synthetisch bereikbaar. Geen receiverprovisioning, automatische v1-migratie of kwalificatie fast_state. Review crashvensters/descriptorownership.

### F2c4 — finale slotgrens en voortgang van publicatie

- **Doel/eisen:** C1 exact; lease tot einde bounded send; acquisitionretry buiten slot/DB; owner consumeert invalidatie en revoket onder eigen nieuwe lease; geen nested lock/DB-in-tick.
- **Files:** nieuw `fastctrl/execution.py`; `fastctrl/publication.py` (acquisition/cadence); `settlement/publication.py` (uitsluitend bounded acquisition, geen receiverhook). `day_ahead.py` uitsluitend noodzakelijke acquisitioncallers, geen solveherhaling. `settlement/repository.py` behoudt zijn transactie/guardcontract. Nieuw `dao/tests/prog/test_fast_execution.py`; bestaande `dao/tests/settlement/test_publication.py`.
- **Regressies:** twee **echte processen** (owner en registrator), IPC-barrières, tijdelijke SQLite: correctie eerst versus send eerst, vervolgens ownerintrekking; assert planbytes/revisie/receiver-effectlog/exacte epoch/barrier, niet alleen mockvolgorde. Receiver-I/O alleen owner, nul in volledige registratiejob, nooit open DB-transactie bij send/revoke. No-op/gewone append versus relevante correctie; rollback na invalidatie; onbereikbare owner na DB-commit laat plan blocked en uitvoer unconfirmed, nooit een fysieke stopclaim.
- **Voortgang:** fake gedeelde boottime voor admission + echte lockcontention; owner biedt voortdurend werk, publisher slaagt in gereserveerd venster zonder nieuwe solve. Installatiebreed budget, maximaal één retry/één exacte query **binnen** restbudget; geen late state-update/lockrelease tijdens send/LOCK_UN/unlink op busy. Kortere lease verkort deadline. Geen sleeps als causale synchronisatie.
- **Grens/dependency:** c1–3. Mechanische fake-race bewijst niet open publiek v2-entrypoint; diens blokkade apart testen. Host/fsync real-time/server-DB blijven E2/E1. Review lockgraph/fairnessaannamen.

### F2c5 — consumers, herstelplicht en legacygrens

- **Doel/eisen:** enige financiële owner, echte init/profilevoorstellen, ongeschikte/ambigue uitvoering sluit niet-nul/euroclaims, geen intent bypass.
- **Files:** `fastctrl/{runner,policy,actuator}.py`; `day_ahead.py` alleen intent-/legacyafbakening, harddeny behouden. Adapterclassificatie/diagnostics-overlapvalidatie bij ownerbootstrap. Tests: `dao/tests/prog/test_fast_runner.py`, `test_fast_policy.py`, `test_fast_export.py`, `dao/tests/settlement/test_publication_consumers.py`, `test_optimization_producer.py` (bestaande producer-/intentharness: regels 319,424-441,566-575; geen nieuwe F1-harness herbouwen).
- **Entryregressies:** runner met ready-v2 + fake/HA blijft `actuation_gates_unproven`, nul helpers, ook off/shadow/restart. Financiële producer nul HA-writes inclusief vroege temperatuur/boiler/grid/EV/PV/batterij/HP/apparaten/start-stop. Legacy calltrace/waarden/volgorde gelijk aan F2b/R11, geen nieuwe v1-leasevoorwaarde. Generieke HA kan ook via canonical-actuatoraanroep niet financieel sturen.
- **Synthetische ownerconsumer:** eerste v2-plan, planvolgen zonder override, profielstop/nieuwe planidentiteit/meerdere batterijen. Geen fictieve accepted/held-reconciliatie; shadow/deadband geen accepted-update. Partial/timeout → unconfirmed/herstelplicht/nul nieuwe niet-nul tot barrière; geen alsof-uitgevoerd benefit/held-sensorfallback. Geen F3-euroberekening: onbekend tonen, niet nul als gemeten besparing.
- **Extra rand:** diagnostics mogen geen actuator/mode/stopentity aliasen. Ze gebruiken nu zelf HA-writes (`runner.py:931-947`); naam “diagnostiek” geeft geen vrijstelling. Financiële bootstrap weigert overlap, v1 niet stil aanpassen. Receiverprovisioning sluit andere writers uit.
- **Grens/dependency:** c1–4; geen replayreparatie/andere-apparatenadapter. Review bypasses/legacybehoud.

### F2c6 — samengestelde failurematrix en overdracht

- **Doel/eisen:** geordende receiverproef door owner/sendgrens herhalen; gezamenlijke invarianten, geen overstated vrijgave.
- **Files:** tests c1–c5, integratie in `test_fast_actuator.py`/`test_fast_execution.py`; bestaande ledger/deze bijlage met werkelijk bewijs actualiseren, normen niet stil veranderen.
- **Matrix:** delay vóór acceptatie / na reserve / na eerste helper / na effect vóór ACK; elk met retry/callercrash/receivercrash/restore mirror en receiver/concurrente correctie of operatorstop/lease vóór-op-na grens. Altijd oud nonzero → timeout → nieuwe-epochbarrière → oude delivery, geen oud effect na barrier. Late ACK nooit nieuwe accepted state; nieuwe niet-nul vereist exacte ownerbinding én actuele basis, uitsluitend synthetische mechanische harness.
- **Gates:** F2b/F2a/F1/B10d regressies, volledige forksuite sequentieel, gerichte Ruff op gewijzigde Pythonfiles. Geen live HA/installaties/netwerk/externe DB/nieuwe dependencies. Uitvoerder levert testuitslagen; deze ontwerpronde heeft geen tests uitgevoerd.
- **Grens/dependency:** vijf sneden gereviewd; onafhankelijke integratiereview hele F2c-delta. Softwarebewijs kan gereed zijn terwijl hardware/activering/universele legacy-exclusiviteit extern_geblokkeerd blijven. Daarna F3a→F3b→E1/E2.

**Uitvoeringsbewijs na herstelopdracht “repareer het!”:** F2c6-code gecommit als `ab584be`, LOG1-testisolatiereparatie als `ce18795`. De aanvankelijke matrix had bewijsleemten en een bestaande reducerfout; die zijn niet administratief gesloten maar in drie scoped fixronden hersteld en onafhankelijk goedgekeurd. De84synthetische matrixgevallen bewijzen daadwerkelijke retry, callerprocesdood met behouden echte effectentries, receivercrash/restore, helpergrenzen −1/0/+1ms, exacte receipt/barrièrebinding, late ACK en positieve nieuwe sectie. De reducer houdt levende pending_effect onzeker tegenover een duplicaatweigering; transportfouten zijn machineleesbaar. Niets hiervan kwalificeert hardware of opent v2.

Verse controllergates op exact de gecommitte kandidaatbytes:399gerichte regressies;2366pass/3skip §8;4301pass/8skip aangrenzend;**5024pass/8skip/80warnings volledige forksuite, geen failures**; zes legacy-BASEprobes veld-/byte-identiek; elf frozen bronnen identiek; gerichte Ruff/diffcheck schoon. De eerdere stdoutfailure is causaal hersteld in uitsluitend testfixtures; productielogging ongewijzigd. Acht skips blijven5liveHA/DB+1serverrowlock+2serverruntimes. Warnings47Pandas+33fork blijven zichtbaar. Bewijs `exec-17-controller-report.md` en logs in de bestaande planworkspace. **De onafhankelijke integratiereview71feb38→ce18795 volgt nog; geen F3/E/R-/hardwarevrijgave.**

## 5. Externe blokkades en stopcriteria

**Aanvullend intern checkpoint na uitvoering:** de onafhankelijke integratiereview `71feb38..ce18795` is uitgevoerd en geeft **wijzigingen nodig** (7Important). Receiptidentiteit/pending, meter-ownerbinding, eenmalige barrièrebinding, accepted-baseline na restart, monotone planleasegrens, directe adapterclassificatie en admission/voortgang moeten worden uitgezocht/hersteld. Verkeerde ACK en hergebruikte barrière zijn door twee nieuwe synthetische controllerproeven causaal RED bevestigd. Dit zijn interne open punten naast de onderstaande externe blokkades; de groene reguliere suite maakt geheel F2c nog niet gereed. Zie masterplan§17.4 en `exec-17-F2c-integration-review.md` in de bestaande planworkspace. Geen productwijziging buiten de geautoriseerde reparatiescope uitgevoerd.

1. **Geen concrete receiver/driver gekwalificeerd:** actuator/queues/stop/auth/credentialexclusiviteit/watchdog/rollbackanker moeten integratie-eigenaar/operator bewijzen. Generieke HA voldoet niet; geen workaround binnen implementersnede.
2. **R11 versus universele exclusiviteit:** legacy bewust intact. Quiescentie van oude processes/automations is operationeel bewijs, niet afleidbaar uit flock. Onmiddellijke universele exclusiviteit vereist nieuw scopebesluit.
3. **Deadline/klok/voortgang:** niet-deadlineconform transport/onbekende clockmarge/apparaat blijft werken na lease ⇒ ongeschikt. Geen harde Linux-fsynclatentie claimen. Publisherretry beschermt tegen snelle-laagcontention, niet uitval of eindeloze externe writers.
4. **Gate behouden:** `following_verdict(...)=actuation_gates_unproven` en directe financiële uitvoerweigering blijven. Alleen latere geautoriseerde activatie na F3/E1/E2 mag vervangen. Fakekwalificatie telt niet.
5. **Stop voor uitvoerder:** lijkt een bypassflag/tweede financieel artefact/epochreset/ongemarkeerde HA-write/legacygedragswijziging nodig: herbeoordeel ontwerp, versoepel geen veiligheidsassertie.

**Volgorde:** F2c1 → review → F2c2 → review → F2c3 → review → F2c4 → review → F2c5 → review → F2c6 + integratiereview. Geen snede verleent uitvoerrechten. Alleen synthetisch ontwerpwerk; geen productcode, commit, branch, index of HEAD gewijzigd.
