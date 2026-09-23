#!/usr/bin/env bash

WATCH_FILES=(
    "/root/dao/data/options.json"
    "/root/dao/data/secrets.json"
)

SCHEDULER_PID=""
GUNICORN_PID=""
INOTIFY_PID=""

start_scheduler() {
    echo "Starting scheduler..."

    (
        cd /root/dao/prog || exit 1
        exec python3 da_scheduler.py
    ) &

    SCHEDULER_PID=$!

    echo "Scheduler PID: $SCHEDULER_PID"
}

start_gunicorn() {
    echo "Starting gunicorn..."

    (
        cd /root/dao/webserver || exit 1
        exec gunicorn --config gunicorn_config.py app:app
    ) &

    GUNICORN_PID=$!

    echo "Gunicorn PID: $GUNICORN_PID"
}

start_inotify() {
    inotifywait -q -e modify "${WATCH_FILES[@]}" &
    INOTIFY_PID=$!
}

stop_scheduler() {
    if [ -n "$SCHEDULER_PID" ] &&
       kill -0 "$SCHEDULER_PID" 2>/dev/null; then

        echo "Stopping scheduler..."

        kill -TERM "$SCHEDULER_PID" 2>/dev/null
        wait "$SCHEDULER_PID" 2>/dev/null
    fi
}

stop_gunicorn() {
    if [ -n "$GUNICORN_PID" ] &&
       kill -0 "$GUNICORN_PID" 2>/dev/null; then

        echo "Stopping gunicorn..."

        kill -TERM "$GUNICORN_PID" 2>/dev/null
        wait "$GUNICORN_PID" 2>/dev/null
    fi
}

cleanup() {
    echo "Watchdog stopping..."

    if [ -n "$INOTIFY_PID" ] &&
       kill -0 "$INOTIFY_PID" 2>/dev/null; then

        kill "$INOTIFY_PID" 2>/dev/null
    fi

    stop_scheduler
    stop_gunicorn

    exit 0
}

trap cleanup SIGTERM SIGINT


# ============================================================
# Start beide processen één keer
# ============================================================

start_scheduler
start_gunicorn


# ============================================================
# Watchdog loop
# ============================================================

while true; do

    start_inotify

    while true; do

        # ----------------------------------------------------
        # Scheduler gestopt/gecrasht
        # ----------------------------------------------------

        if ! kill -0 "$SCHEDULER_PID" 2>/dev/null; then

            wait "$SCHEDULER_PID" 2>/dev/null
            EXIT=$?

            kill "$INOTIFY_PID" 2>/dev/null

            if [ "$EXIT" -eq 0 ]; then
                echo "Scheduler stopped normally"
            else
                echo "Scheduler crashed with exit code $EXIT"
                echo "Keeping the web UI up; restarting the scheduler with backoff."
                echo "Edit the config from the settings page if it needs fixing."
                # Backoff so we do not spam logs while the user fixes
                # the config via the web UI. The inotify watcher picks
                # up config edits on its own, so the user does not have
                # to wait for this sleep to end.
                sleep 30
            fi

            # Alleen de scheduler herstarten. Gunicorn en daarmee de
            # web UI blijven draaien zodat de gebruiker de config
            # kan repareren. Inotify ook herstarten zodat
            # config-wijzigingen nog steeds worden opgepikt.
            start_scheduler
            start_inotify
        fi


        # ----------------------------------------------------
        # Gunicorn gestopt/gecrasht
        # ----------------------------------------------------

        if ! kill -0 "$GUNICORN_PID" 2>/dev/null; then

            wait "$GUNICORN_PID" 2>/dev/null
            EXIT=$?

            kill "$INOTIFY_PID" 2>/dev/null

            if [ "$EXIT" -eq 0 ]; then
                echo "Gunicorn stopped normally"
            else
                echo "Gunicorn crashed with exit code $EXIT"
                echo "Keeping the scheduler up; restarting gunicorn with backoff."
                sleep 30
            fi

            # Alleen gunicorn herstarten; de scheduler doorlatend.
            start_gunicorn
            start_inotify
        fi


        # ----------------------------------------------------
        # Configuratie gewijzigd
        # ----------------------------------------------------

        if ! kill -0 "$INOTIFY_PID" 2>/dev/null; then

            echo "Configuration changed..."

            kill "$INOTIFY_PID" 2>/dev/null


            # Scheduler volledig opnieuw starten
            stop_scheduler
            start_scheduler


            # Bestaande Gunicorn master reloaden.
            # NIET opnieuw starten!
            if kill -0 "$GUNICORN_PID" 2>/dev/null; then
                echo "Reloading gunicorn..."
                kill -HUP "$GUNICORN_PID" 2>/dev/null
            fi

            # Verlaat alleen deze inner loop zodat een nieuwe
            # inotifywait wordt gestart.
            break
        fi

        sleep 1
    done

done