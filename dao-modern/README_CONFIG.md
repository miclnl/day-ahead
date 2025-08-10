# DAO Modern - Configuratie Gids

## Overzicht

DAO Modern gebruikt een gelaagde configuratie-aanpak:

### 1. Home Assistant Addon Configuratie
**Alleen basis systeeminstellingen** voor het opstarten van de addon:

- **Database Engine**: SQLite (standaard), MySQL of PostgreSQL
- **Log Level**: Debug, Info, Warning, Error
- **Home Assistant Token**: Vereist voor API toegang
- **Home Assistant URL**: Optioneel (wordt automatisch gedetecteerd)

### 2. DAO Application Configuratie
**Alle energiespecifieke instellingen** worden binnen de webapp geconfigureerd via:
- `/config/dao_modern_data/options.json` - Hoofdconfiguratie
- `/config/dao_modern_data/secrets.json` - API keys en gevoelige gegevens

## Waarom Deze Scheiding?

✅ **Addon config**: Minimaal - alleen wat nodig is om op te starten  
✅ **App config**: Compleet - alle energiespecifieke parameters  
✅ **Flexibiliteit**: Geen addon herstart nodig voor energieparameters  
✅ **Veiligheid**: Gevoelige API keys apart opgeslagen  

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
  "solar_panel_capacity": 8.5,
  "optimization_interval": 60,
  "enable_ml_predictions": true
}
```

## Eerste Setup

1. **Installeer addon** - vul alleen HA token in
2. **Start addon** - toegang via ingress (poort 8099)  
3. **Open webapp** - configureer energie instellingen
4. **Test configuratie** - controleer API verbindingen
5. **Start optimalisatie** - alles klaar!

## Migratie

Bestaande configuratie wordt automatisch gemigreerd van addon config naar app config bij eerste start.