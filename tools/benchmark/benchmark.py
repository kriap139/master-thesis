from typing import Callable, Sequence
import time
import numpy as np
import json
import os
from datetime import datetime
from ..Util.io_util import load_json, save_json

class Benchmark:

    def __init__(self, name: str, func: Callable, save: bool = true, save_path: str = None):
        self.name = name
        self.func = func
        self.times = []
        self.results= []
        self._save_timestamp = datetime.now().strftime("%d-%m-%Y@%H:%M:%S")
        self._save_path = save_path
        self._save = save
        self.n_repeats = 0

        if save and save_path is None:
            raise RuntimeError(f"'save_path' argument is required for saveing benchmark")
    
    def set_name(name: str):
        self.name = name
    
    def __run(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        end = start - time.perf_counter()

        self.times.append(end)
        self.results.append(result)
    
    def run(self, n_repeats: int = 1, *args, **kwargs):
        if self._save:
            self.init_save()
            for i in range(n_repeats):
                self.__run(*args, **kwargs)
                self.update_save()
                self.n_repeats += 1
            self.update_save( 
                skip_benchmark_update=True,
                exstra_keys=dict(n_repeats=self.n_repeats,average_time=self.average_time())
            )
        else:
            for i in range(n_repeats):
                self.__run()
    
    def average_time(self) -> float:
        return sum(self.times) / max(1, len(self.times))
    
    def init_save(self):
        folder = os.path.dirname(self._save_path)
        fn, ext = os.path.splitext(self._save_path)
        self._save_path = os.path.join(folder, f"{fn}@{self.save_timestamp}{ext}")
        save_json(self._save_path, {})
    
    def update_save(self, skip_benchmark_update=False, exstra_keys: dict = None):
        data = load_json(self._save_path, default={})

        if self.__save_identifier not in data:
            data = dict(
                name = name if name is not None else self.name,
                results = self.results,
                times = self.times
            )
        elif not skip_benchmark_update:
            data["results"].append(self.results[-1])
            data["times"].append(self.times[-1])
        
        if exstra_keys is not None:
            data.update(exstra_keys)

        save_json(fp, data, overwrite=True)