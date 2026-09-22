import datetime
import logging
import sys
import time
from da_base import DaBase
from dao.prog.fastctrl.runner import start_if_enabled
from subprocess import Popen


class DaScheduler(DaBase):
    def __init__(self, file_name: str = None):
        super().__init__(file_name)
        self.active = self.config.scheduler.active
        self.scheduler_tasks = {
            entry.time: entry.action for entry in self.config.scheduler.schedule
        }
        self.fast_control = None

    def run_task_process(self, key_task):
        run_task = self.tasks[key_task]
        proc = Popen(run_task["cmd"])
        proc.wait()
        if proc.returncode != 0 and proc.returncode is not None:
            print(f"Task {key_task} crashed with exit code {proc.returncode}")
            return False
        return True

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

    def scheduler(self):
        # if not (self.notification_entity is None) and self.notification_opstarten:
        #     self.set_value(self.notification_entity, "DAO scheduler gestart " +
        #                    datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
        self.start_fast_control()

        while True:
            t = datetime.datetime.now()
            next_min = t - datetime.timedelta(
                minutes=-1, seconds=t.second, microseconds=t.microsecond
            )
            # wacht tot hele minuut 0% cpu
            time.sleep((next_min - t).total_seconds())
            if not self.active:
                continue
            hour = next_min.hour
            minute = next_min.minute
            key0 = str(hour).zfill(2) + str(minute).zfill(2)
            # ieder uur in dezelfde minuut voorbeeld xx15
            key1 = "xx" + str(minute).zfill(2)
            # iedere minuut in een uur voorbeeld 02xx
            key2 = str(hour).zfill(2) + "xx"
            tasks = []
            for key in self.scheduler_tasks:
                if key == key0:
                    tasks.append(self.scheduler_tasks[key])
                elif key == key1:
                    tasks.append(self.scheduler_tasks[key])
                elif key == key2:
                    tasks.append(self.scheduler_tasks[key])
            for task in tasks:
                for key_task in self.tasks:
                    if self.tasks[key_task]["function"] == task:
                        try:
                            self.run_task_process(key_task)
                        except KeyboardInterrupt:
                            self.stop_fast_control()
                            sys.exit()
                        except Exception as e:
                            print(e)
                            continue
                        break


def main():
    da_sched = DaScheduler("../data/options.json")
    try:
        da_sched.scheduler()
    finally:
        da_sched.stop_fast_control()


if __name__ == "__main__":
    main()
