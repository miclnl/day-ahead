import datetime
import sys
import time
import threading
import queue
from da_base import DaBase


class DaScheduler(DaBase):
    def __init__(self, file_name: str = None):
        super().__init__(file_name)
        self.scheduler_tasks = self.config.get(["scheduler"])
        self.active = True
        if "active" in self.scheduler_tasks:
            self.active = not (self.scheduler_tasks["active"].lower() == "false")
        # Minimal refactor: background worker to run tasks so the tick loop never blocks
        self._task_queue: "queue.Queue[str]" = queue.Queue()
        self._running_tasks = set()  # keys of self.tasks currently running
        self._running_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._worker, name="scheduler-worker", daemon=True)
        self._worker_thread.start()

    def _enqueue(self, key_task: str) -> None:
        with self._running_lock:
            if key_task in self._running_tasks:
                return
            # mark enqueued to prevent rapid duplicates within same minute
            self._running_tasks.add(key_task)
        self._task_queue.put(key_task)

    def _worker(self) -> None:
        while True:
            key_task = self._task_queue.get()
            try:
                try:
                    self.run_task_function(key_task, True)
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    print(e)
            finally:
                with self._running_lock:
                    self._running_tasks.discard(key_task)
                self._task_queue.task_done()

    def scheduler(self):
        # if not (self.notification_entity is None) and self.notification_opstarten:
        #     self.set_value(self.notification_entity, "DAO scheduler gestart " +
        #                    datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))

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
            key1 = str(hour).zfill(2) + str(minute).zfill(2)
            # ieder uur in dezelfde minuut voorbeeld xx15
            key2 = "xx" + str(minute).zfill(2)
            # iedere minuut in een uur voorbeeld 02xx
            key3 = str(hour).zfill(2) + "xx"
            task = None
            if key1 in self.scheduler_tasks:
                task = self.scheduler_tasks[key1]
            elif key2 in self.scheduler_tasks:
                task = self.scheduler_tasks[key2]
            elif key3 in self.scheduler_tasks:
                task = self.scheduler_tasks[key3]
            if task is not None:
                for key_task in self.tasks:
                    if self.tasks[key_task]["function"] == task:
                        # Enqueue task to background worker if not already running
                        self._enqueue(key_task)
                        break


def main():
    da_sched = DaScheduler("../data/options.json")
    da_sched.scheduler()


if __name__ == "__main__":
    main()
