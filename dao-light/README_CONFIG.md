# DAO Light - Configuratie Gids

## Overzicht

DAO Light gebruikt een eenvoudige gelaagde configuratie-aanpak:

### 1. Home Assistant Addon Configuratie
**Alleen basis systeeminstellingen** voor het opstarten van de addon:

- **Database Engine**: SQLite (aanbevolen), MySQL
- **Log Level**: Debug, Info, Warning, Error
- **Home Assistant Token**: Vereist voor API toegang

### 2. DAO Application Configuratie
**Alle energiespecifieke instellingen** worden binnen de webapp geconfigureerd via:
- `/config/dao_light_data/options.json` - Hoofdconfiguratie
- `/config/dao_light_data/secrets.json` - API keys en gevoelige gegevens

## Waarom Deze Scheiding?

✅ **Minimal addon config**: Alleen wat nodig is om op te starten  
✅ **Complete app config**: Alle energieparameters in de webapp  
✅ **No restart needed**: Energieparameters aanpassen zonder herstart  
✅ **Simple setup**: Minder verwarring over waar wat in te stellen  

## Configuratie Voorbeeld

### Addon Config (via HA interface):
```yaml
database_engine: sqlite
log_level: info  
homeassistant_token: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

### App Config (via webapp):
```json
{
  "energy_tax_percentage": 21.0,
  "energy_surcharge_per_kwh": 0.03,
  "nordpool_region": "DE", 
  "battery_capacity": 10.5,
  "optimization_interval": 60
}
```

## Eerste Setup

1. **Installeer addon** - vul alleen HA token in
2. **Start addon** - toegang via ingress (poort 8099)
3. **Open webapp** - configureer energie instellingen  
4. **Test configuratie** - controleer verbindingen
5. **Start optimalisatie** - klaar voor gebruik!

## Light vs Modern

DAO Light heeft minder configuratie-opties dan Modern:
- Geen ML features
- Geen real-time optimalisatie
- Eenvoudigere energie parameters
- Focus op stabiliteit en lage resource usage