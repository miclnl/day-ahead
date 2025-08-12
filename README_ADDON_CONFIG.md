# Home Assistant Add-on Configuratie

Deze DAO add-on leest nu automatisch de configuratie uit `/data/options.json` en past deze toe op de applicatie.

## Functionaliteit

### 1. Automatische Configuratie Lezen

- De add-on leest bij het opstarten automatisch de configuratie uit `/data/options.json`
- Deze configuratie wordt gebruikt voor alle instellingen die in de Home Assistant interface zijn ingesteld

### 2. Logging Configuratie

- **`log_level`**: Bepaalt het logging niveau (debug, info, warning, error)
- Dit niveau wordt toegepast op:
  - De Flask applicatie
  - Gunicorn webserver
  - SQLAlchemy database logging
  - Alle logging gaat naar STDOUT voor Home Assistant integratie

### 3. Database Configuratie

- **`database_engine`**: Database type (sqlite, mysql, postgresql)
- **`homeassistant_token`**: API token voor Home Assistant integratie
- **`homeassistant_url`**: URL naar Home Assistant (optioneel)

### 4. Optimalisatie Configuratie

- **`optimization_mode`**: Optimalisatie modus (momenteel alleen "statistical")
- **`cloud_ai_enabled`**: Schakelt cloud AI functionaliteit in/uit

## API Endpoints

### Configuratie Status

```http
GET /api/config
```

Toont de huidige add-on configuratie.

### Configuratie Herladen

```http
POST /api/reload-config
```

Herlaadt de configuratie uit `/data/options.json` en past deze toe.

### Health Check

```http
GET /api/health-check
```

Toont de systeem status inclusief add-on configuratie status.

## Configuratie Bestand Structuur

Het `/data/options.json` bestand wordt automatisch aangemaakt door Home Assistant wanneer de gebruiker de add-on configureert:

```json
{
  "log_level": "info",
  "database_engine": "sqlite",
  "homeassistant_token": "your_token_here",
  "homeassistant_url": "http://localhost:8123",
  "optimization_mode": "statistical",
  "cloud_ai_enabled": false
}
```

## Implementatie Details

### AddonConfig Klasse

- **Locatie**: `webserver/app/addon_config.py`
- **Functie**: Beheert alle add-on configuratie
- **Automatisch laden**: Wordt geladen bij applicatie start

### Gunicorn Integratie

- **Locatie**: `webserver/gunicorn_config.py`
- **Functie**: Past loglevel toe op basis van add-on configuratie
- **Fallback**: Gebruikt environment variable als add-on configuratie niet beschikbaar is

### Flask Integratie

- **Locatie**: `webserver/app/__init__.py`
- **Functie**: Laadt add-on configuratie bij applicatie start
- **Logging**: Past logging niveau toe op basis van configuratie

## Voordelen

1. **Centrale Configuratie**: Alle instellingen op één plek
2. **Home Assistant Integratie**: Configuratie via de officiële interface
3. **Automatisch Herladen**: Configuratie kan worden herladen zonder restart
4. **Consistente Logging**: Alle componenten gebruiken hetzelfde loglevel
5. **STDOUT Logging**: Logs zijn zichtbaar in Home Assistant logs

## Troubleshooting

### Configuratie wordt niet geladen

- Controleer of `/data/options.json` bestaat
- Controleer of het bestand geldige JSON bevat
- Kijk naar de applicatie logs voor foutmeldingen

### Logging werkt niet

- Controleer of `log_level` correct is ingesteld
- Gebruik `/api/reload-config` om configuratie te herladen
- Controleer of de applicatie is herstart

### API Endpoints niet beschikbaar

- Controleer of de routes correct zijn geladen
- Gebruik `/debug/routes` om alle beschikbare routes te zien
- Controleer de applicatie logs voor import fouten

## Ontwikkeling

Voor het ontwikkelen van nieuwe configuratie opties:

1. Voeg de optie toe aan `config.yaml` in de `options` en `schema` secties
2. Voeg een getter methode toe aan de `AddonConfig` klasse
3. Test de functionaliteit via de API endpoints
4. Update de documentatie

## Migratie van Oude Configuratie

De add-on ondersteunt zowel de nieuwe add-on configuratie als de oude configuratie structuur:

- **Nieuwe configuratie**: Wordt gelezen uit `/data/options.json`
- **Oude configuratie**: Wordt gelezen uit de bestaande configuratie bestanden
- **Fallback**: Als nieuwe configuratie niet beschikbaar is, worden standaard waarden gebruikt
