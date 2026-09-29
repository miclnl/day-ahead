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
import threading
from subprocess import Popen
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from da_base import DaBase
from dao.prog.fastctrl.runner import start_if_enabled

#: How late a job may still be started when the scheduler was busy or the
#: process was blocked, in seconds. Later than this it is dropped and logged.
MISFIRE_GRACE_S = 300

#: Tasks that may run at the same time. Different tasks can overlap (the
#: optimisation does not have to wait for ML training); the same task never
#: overlaps itself, see _run_exclusive.
MAX_PARALLEL_TASKS = 4


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
        self._task_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- running one task ---------------------------------------------------

    def task_key_for(self, action: str) -> str | None:
        """The key in self.tasks whose function is *action*."""
        for key, task in self.tasks.items():
            if task["function"] == action:
                return key
        return None

    def run_task_process(self, key_task: str) -> bool:
        run_task = self.tasks[key_task]
        # Pin CWD to the prog directory: the calc and forecast tasks use
        # CWD-relative paths (../data, ../prog) and silently misbehave
        # if the scheduler is started from a different working directory
        # (e.g. by a manual /api/run trigger or a future debug entrypoint).
        logging.info(f"Taak {key_task} gestart")
        started = datetime.datetime.now()
        proc = Popen(run_task["cmd"], cwd=self.PROG_DIR)
        proc.wait()
        duration = (datetime.datetime.now() - started).total_seconds()
        if proc.returncode != 0:
            logging.error(
                f"Taak {key_task} eindigde met exit code {proc.returncode} "
                f"na {duration:.0f} s"
            )
            return False
        logging.info(f"Taak {key_task} klaar na {duration:.0f} s")
        return True

    def _run_exclusive(self, key_task: str) -> None:
        """Run a task unless the previous run of the same task is still busy."""
        with self._locks_guard:
            lock = self._task_locks.setdefault(key_task, threading.Lock())
        if not lock.acquire(blocking=False):
            logging.warning(
                f"Taak {key_task} overgeslagen: de vorige run loopt nog"
            )
            return
        try:
            self.run_task_process(key_task)
        except Exception:  # noqa: BLE001 - the scheduler must keep running
            logging.exception(f"Taak {key_task} is mislukt")
        finally:
            lock.release()

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
        if not self.active:
            logging.warning("Scheduler staat uit (scheduler.active = false); geen taken gepland")
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
