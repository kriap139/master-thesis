from typing import Callable, Sequence
import time
import numpy as np

class Benchmark:

    def __init__(self, name: str, func: Callable, *args, **kwargs):
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.times = []
        self.results= []
    
    def __run(self):
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        end = start - time.perf_counter()

        self.times.append(end)
        self.results.append(result)
    
    def run(self, n_repeats: int = 5):
        for i in range(n_repeats):
            self.__run()
    
    def elapsed_time(self) -> float:
        return sum(self.times) / max(1, len(self.times))