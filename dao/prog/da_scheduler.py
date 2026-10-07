"""The long-running scheduler process of the add-on.

Runs the configured tasks (price and weather fetches, the optimisation, the
maintenance jobs) at their cron-like times and hosts the fast control thread.
Every task runs in its own subprocess so a crash in one task cannot take the
scheduler down.

Scheduling is done by APScheduler. The previous hand-written minute loop ran
tasks one after another and slept until the next whole minute afterwards, so
every minute that passed while a long task ran (ML training, a slow
optimisation) was never evaluated and the tasks planned in it were silently
skipped. Now each schedule entry is a job of its own; jobs run in a small
thread pool, a job that is still running is not started a second time, and a
job that was delayed by a few minutes still runs.
"""

import datetime
import logging
import os
import signal
import sys
import time
from subprocess import Popen
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from da_base import DaBase
from dao.prog import task_state
from dao.prog import tasks as task_registry
from dao.prog.fastctrl.runner import start_if_enabled

#: How late a job may still be started when the scheduler was busy or the
#: process was blocked, in seconds. Later than this it is dropped and logged.
MISFIRE_GRACE_S = 300

#: Tasks that may run at the same time. Different tasks can overlap (the
#: optimisation does not have to wait for ML training); the same task never
#: overlaps itself, see _run_exclusive.
MAX_PARALLEL_TASKS = 4

#: How often a running task refreshes its claim. Well under
#: task_state.STALE_AFTER_S so a task is never mistaken for dead, and long
#: enough that a multi-hour ML training does not rewrite the state file
#: thousands of times.
HEARTBEAT_S = 30

#: How often the dashboards' requests are picked up. Short enough that
#: pressing a button feels immediate, and well under
#: task_state.PENDING_TIMEOUT_S.
REQUEST_POLL_S = 5

#: How often liveness is recorded while polling. Deliberately much longer
#: than REQUEST_POLL_S: it only has to stay fresher than the pending
#: timeout, and writing on every poll would be thousands of small writes a
#: day to an SD card.
ALIVE_NOTE_S = 20


def _quiet_apscheduler() -> None:
    """Keep APScheduler's per-job bookkeeping out of the add-on log.

    APScheduler logs 'Running job "..."' and 'Job "..." executed
    successfully' at INFO for every execution of every job. The request
    poll alone runs every REQUEST_POLL_S seconds, so with the root logger
    at INFO -- the add-on's default -- that is two lines every five
    seconds, over thirty thousand a day, and the lines that actually say
    something scroll past between them.

    WARNING and above still come through, which is where APScheduler says
    anything worth reading: a missed run, a job that raised, a job skipped
    because the previous one was still going. This mirrors what the code
    already does for PIL and matplotlib, which were quieted for exactly
    the same reason.
    """
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def cron_trigger(pattern: str, timezone) -> CronTrigger:
    """Translate a schedule time pattern into a cron trigger.

    "0544" runs at 05:44, "xx15" every hour at :15, "02xx" every minute
    between 02:00 and 02:59. This is the same set of shapes the configuration
    validator accepts.
    """
    hours, minutes = pattern[0:2], pattern[2:4]
    return CronTrigger(
        hour="*" if hours == "xx" else int(hours),
        minute="*" if minutes == "xx" else int(minutes),
        timezone=timezone,
    )


class DaScheduler(DaBase):
    # Resolve once at import time so all subprocesses share the same
    # working directory regardless of where the watchdog started us.
    PROG_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, file_name: str = None):
        super().__init__(file_name)
        self.active = self.config.scheduler.active
        self.schedule = list(self.config.scheduler.schedule)
        self.fast_control = None
        self.scheduler: BlockingScheduler | None = None
        self._last_alive_note = 0.0


    # -- running one task ---------------------------------------------------

    def task_key_for(self, action: str) -> str | None:
        """The key in self.tasks whose function is *action*."""
        for key, task in self.tasks.items():
            if task["function"] == action:
                return key
        return None

    @property
    def log_dir(self) -> str:
        return os.path.join(self.PROG_DIR, "..", "data", "log")

    def _newest_log_for(self, key_task: str) -> str | None:
        """The newest log file belonging to *key_task*, or None.

        Each task writes its own log through DaBase.run_task_function, named
        "<file_name>_<timestamp>.log". The dashboard needs that path to show
        the output, and only the registry knows the prefix -- the v2
        dashboard used to watch for the newest *.log of any kind, which
        picked up another task's output when two ran at once.
        """
        prefix = self.tasks[key_task]["file_name"]
        try:
            names = [
                name
                for name in os.listdir(self.log_dir)
                if name.startswith(prefix + "_") and name.endswith(".log")
            ]
        except OSError:
            return None
        if not names:
            return None
        return os.path.join("../data/log", max(names))

    def run_task_process(
        self, key_task: str, parameters: dict | None = None
    ) -> tuple[str, int | None, str | None]:
        """Run the task and report (status, returncode, logfile).

        Deliberately does not touch the claim: _run_exclusive owns that
        from start to finish, so there is one place responsible for
        releasing it however the run ends.
        """
        run_task = self.tasks[key_task]
        cmd = task_registry.build_cmd(key_task, parameters) or list(run_task["cmd"])
        # Pin CWD to the prog directory: the calc and forecast tasks use
        # CWD-relative paths (../data, ../prog) and silently misbehave
        # if the scheduler is started from a different working directory
        # (e.g. by a manual /api/run trigger or a future debug entrypoint).
        logging.info(f"Taak {key_task} gestart")
        started = datetime.datetime.now()
        before = self._newest_log_for(key_task)
        # start_new_session=True: the task gets its own process group, so a
        # cancel reaches anything it spawned rather than just the direct
        # child.
        proc = Popen(cmd, cwd=self.PROG_DIR, start_new_session=True)
        logfile = None
        cancelled = False
        # Keep the claim fresh while the task runs, so a long one (ML
        # training) is not mistaken for a dead claim and started a second
        # time by the dashboard. Polling instead of proc.wait() is what
        # makes that possible, and it is also how a cancel request gets
        # noticed.
        while proc.poll() is None:
            if logfile is None:
                found = self._newest_log_for(key_task)
                if found is not None and found != before:
                    logfile = found
            entry = task_state.heartbeat(key_task, logfile=logfile)
            if entry.get("cancel"):
                logging.warning(f"Taak {key_task} wordt afgebroken op verzoek")
                task_registry.kill_process_group(proc)
                cancelled = True
                break
            time.sleep(HEARTBEAT_S)
        proc.wait()
        duration = (datetime.datetime.now() - started).total_seconds()
        if cancelled:
            return "cancelled", proc.returncode, logfile
        if proc.returncode != 0:
            logging.error(
                f"Taak {key_task} eindigde met exit code {proc.returncode} "
                f"na {duration:.0f} s"
            )
            return "error", proc.returncode, logfile
        logging.info(f"Taak {key_task} klaar na {duration:.0f} s")
        return "done", proc.returncode, logfile

    def _run_exclusive(self, key_task: str) -> None:
        """Run a task unless the same task is already running.

        The claim is taken through dao.prog.task_state, the same file-locked
        registry the dashboards use, rather than an in-process
        threading.Lock. A lock held only in this process was invisible to
        the web server, so cron starting calc_optimum at 05:44 and a user
        pressing the button at 05:44 produced two optimisation runs writing
        the same tables and pushing conflicting setpoints to Home Assistant.
        """
        granted = task_state.claim(key_task, source="scheduler")
        if not granted:
            holder = task_state.running_tasks().get(key_task, {})
            logging.warning(
                f"Taak {key_task} overgeslagen: draait al "
                f"(gestart door {holder.get('source', 'onbekend')})"
            )
            return
        self._run_claimed(key_task)

    def _run_claimed(self, key_task: str, parameters: dict | None = None) -> None:
        """Run a task whose claim this process already holds, and release it.

        Shared by the cron path (_run_exclusive, which just claimed) and the
        request path (_pick_up_requests, which took over a claim the
        dashboard made), so both release on exactly the same paths.
        """
        try:
            status, returncode, logfile = self.run_task_process(
                key_task, parameters
            )
        except Exception:  # noqa: BLE001 - the scheduler must keep running
            logging.exception(f"Taak {key_task} is mislukt")
            task_state.release(key_task, "error")
        else:
            task_state.release(
                key_task, status, returncode=returncode, logfile=logfile
            )

    # -- requests from the dashboards ---------------------------------------

    def _pick_up_requests(self) -> None:
        """Run whatever the dashboards have asked for.

        This is what takes task execution out of the gunicorn worker. The
        dashboards used to run tasks themselves in a daemon thread; that
        worker is recycled on every configuration change (the watchdog sends
        gunicorn a HUP), which killed the thread while its subprocess kept
        running, leaving nothing to record the result and the task registered
        as running until its claim went stale. With two workers a request
        could also land in either one.

        Requests are claims in the "pending" state, so they already exclude
        everything else; take_pending flips one to "running" atomically,
        which is what stops two poll cycles from starting the same request.
        """
        task_state.expire_overdue_pending()
        self._note_alive()
        for key_task in task_state.pending_requests():
            if key_task not in self.tasks:
                logging.error(
                    f"Aanvraag voor onbekende taak {key_task!r}, overgeslagen"
                )
                task_state.release(key_task, "error")
                continue
            entry = task_state.take_pending(key_task)
            if entry is None:
                continue  # another poll got there first
            logging.info(
                f"Taak {key_task} opgepakt "
                f"(aangevraagd door {entry.get('source', 'onbekend')})"
            )
            self.scheduler.add_job(
                self._run_claimed,
                args=[key_task, entry.get("parameters") or {}],
                id=f"request-{key_task}-{int(time.time() * 1000)}",
                misfire_grace_time=None,
            )

    def _note_alive(self) -> None:
        """Record liveness now and then, so the dashboard can warn when this
        process is not running and a request would sit there unanswered.

        Rate-limited rather than written on every poll: at a five second
        interval that would be some seventeen thousand small writes a day to
        what is often an SD card, for a signal that only needs to be fresher
        than the pending timeout.
        """
        now = time.time()
        if now - self._last_alive_note < ALIVE_NOTE_S:
            return
        self._last_alive_note = now
        task_state.note_scheduler_alive()

    # -- fast control -------------------------------------------------------

    def start_fast_control(self):
        """Start the realtime feedback layer next to the cron loop.

        It lives in this process on purpose. The watchdog restarts the
        scheduler whenever options.json changes, so the fast layer picks up
        configuration changes for free, and a task subprocess blocking the
        minute tick cannot stall it.
        """
        try:
            self.fast_control = start_if_enabled(self)
        except Exception as exception:  # noqa: BLE001 - never block the scheduler
            logging.exception(f"Fast control kon niet worden gestart: {exception}")
            self.fast_control = None

    def stop_fast_control(self):
        if self.fast_control is not None:
            self.fast_control.stop()
            self.fast_control = None

    # -- the schedule -------------------------------------------------------

    def build_scheduler(self) -> BlockingScheduler:
        _quiet_apscheduler()
        try:
            timezone = ZoneInfo(self.time_zone)
        except Exception:  # noqa: BLE001 - unknown zone name from HA
            logging.warning(f"Onbekende tijdzone {self.time_zone!r}, systeemtijd gebruikt")
            timezone = None
        scheduler = BlockingScheduler(
            executors={"default": ThreadPoolExecutor(MAX_PARALLEL_TASKS)},
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": MISFIRE_GRACE_S,
            },
            timezone=timezone,
        )
        # Added before the active check on purpose: scheduler.active = false
        # means "run no cron schedule", not "ignore the dashboard". Without
        # this, every button in the web UI would silently do nothing on an
        # installation with the schedule switched off.
        scheduler.add_job(
            self._pick_up_requests,
            "interval",
            seconds=REQUEST_POLL_S,
            id="pick-up-task-requests",
            name="taakaanvragen oppakken",
            # The poll is idempotent and cheap; a missed one just means the
            # next runs a few seconds later, so there is nothing to catch up.
            misfire_grace_time=None,
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.datetime.now(timezone),
        )
        if not self.active:
            logging.warning(
                "Scheduler staat uit (scheduler.active = false); geen taken "
                "gepland. Taken uit het dashboard worden nog wel uitgevoerd."
            )
            return scheduler
        for index, entry in enumerate(self.schedule):
            key_task = self.task_key_for(entry.action)
            if key_task is None:
                logging.error(f"Onbekende taak {entry.action!r} in het schema, overgeslagen")
                continue
            scheduler.add_job(
                self._run_exclusive,
                cron_trigger(entry.time, timezone),
                args=[key_task],
                id=f"{index}-{entry.time}-{entry.action}",
                name=f"{entry.time} {entry.action}",
            )
            logging.info(f"Gepland: {entry.time} {entry.action}")
        return scheduler

    def scheduler_loop(self):
        self.start_fast_control()
        self.scheduler = self.build_scheduler()

        def _terminate(signum, frame):
            logging.info("Scheduler stopt (signaal ontvangen)")
            # wait=False: start() returns immediately, running tasks are
            # subprocesses and finish on their own.
            self.scheduler.shutdown(wait=False)

        signal.signal(signal.SIGTERM, _terminate)
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pass





def main():
    da_sched = DaScheduler("../data/options.json")
    if da_sched.config is None:
        sys.exit(1)
    try:
        da_sched.scheduler_loop()
    finally:
        da_sched.stop_fast_control()


if __name__ == "__main__":
    main()
