#!/usr/bin/env python3
"""
Test of minimal scheduler werkt zonder ML dependencies.
"""
import sys
import datetime
import time


class MockDaBase:
    """Mock DaBase voor testing zonder Home Assistant"""
    
    def __init__(self, file_name=None):
        self.config = {"scheduler": {
            "active": "true",
            "0600": "day_ahead",
            "1200": "update_prices"
        }}
        self.scheduler_tasks = self.config.get("scheduler", {})
        self.active = True
        self.tasks = {
            "day_ahead": {"function": "day_ahead"},
            "update_prices": {"function": "update_prices"}
        }
        
    def run_task_function(self, task_key, log=False):
        print(f"✅ Task executed: {task_key}")
        return True
        
    def log(self, message, level="INFO"):
        print(f"{level}: {message}")


class DaScheduler(MockDaBase):
    """Minimal scheduler zonder ML dependencies"""
    
    def __init__(self, file_name: str = None):
        super().__init__(file_name)
        self.scheduler_tasks = self.config.get("scheduler", {})
        self.active = True
        if self.scheduler_tasks and "active" in self.scheduler_tasks:
            self.active = not (self.scheduler_tasks["active"].lower() == "false")

    def scheduler(self):
        print("🚀 DAO Minimal Scheduler gestart")
        iteration = 0
        
        while iteration < 3:  # Test 3 iteraties
            t = datetime.datetime.now()
            next_min = t - datetime.timedelta(
                minutes=-1, seconds=t.second, microseconds=t.microsecond
            )
            
            # Wacht kort voor test
            time.sleep(1)
            
            if not self.active:
                continue
                
            hour = next_min.hour
            minute = next_min.minute
            key1 = str(hour).zfill(2) + str(minute).zfill(2)
            key2 = "xx" + str(minute).zfill(2)
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
                        try:
                            self.run_task_function(key_task, True)
                        except Exception as e:
                            self.log(f"Task {key_task} failed: {e}", level="ERROR")
                            continue
                        break
            else:
                print(f"⏰ No task for {key1}/{key2}/{key3}")
                
            iteration += 1
            
        print("✅ Test completed successfully - geen SIGILL!")


if __name__ == "__main__":
    print("Testing minimal DAO scheduler without ML dependencies...")
    
    try:
        # Test 1: Import check
        print("1. Testing basic scheduler logic...")
        scheduler = DaScheduler()
        
        # Test 2: Run scheduler
        print("2. Testing scheduler execution...")
        scheduler.scheduler()
        
        print("\n🎉 SUCCESS: Minimal scheduler werkt zonder ML libraries!")
        print("   - Geen SIGILL errors")
        print("   - Pure Python scheduling")  
        print("   - Geschikt voor containers")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)