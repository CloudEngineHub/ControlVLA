import time

class Rate:
    def __init__(self, hz):
        self.interval = 1.0 / hz
        self.last_time = time.perf_counter()

    def sleep(self):
        elapsed = time.perf_counter() - self.last_time
        sleep_time = max(0.0, self.interval - elapsed)
        time.sleep(sleep_time)
        self.last_time = time.perf_counter()