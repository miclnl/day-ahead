# Lokaal draaien tegen echte statistics

`docker-compose.dev.yml` bouwt het add-on-image en zet er een eigen MariaDB
naast voor de DAO-database. De recorder-database van Home Assistant wordt over
het netwerk gelezen met de `database ha`-instellingen in `dev-data/options.json`
(gebruik een alleen-lezen account).

`dev-data/` staat in `.gitignore` en wordt nooit gecommit: daar staan
`options.json` en `secrets.json`. Poorten: dashboard `DAO_PORT` (default 5150),
MariaDB `DB_PORT` (default 5151); zet ze in een `.env` naast dit bestand als de
defaults bezet zijn.

```bash
docker compose -f release-testing/docker-compose.dev.yml up -d --build
docker compose -f release-testing/docker-compose.dev.yml exec dao bash -lc "cd /root/dao/prog && python3 day_ahead.py calc_baseloads"
docker compose -f release-testing/docker-compose.dev.yml exec dao bash -lc "cd /root/dao/prog && python3 day_ahead.py accuracy 28"
docker compose -f release-testing/docker-compose.dev.yml down
```

Het dashboard start niet vanzelf; wie het wil zien start het in de container:

```bash
docker compose -f release-testing/docker-compose.dev.yml exec -d dao bash -lc "cd /root/dao/webserver && gunicorn --config gunicorn_config.py app:app"
```
