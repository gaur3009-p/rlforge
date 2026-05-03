"""
Logger
======
Lightweight CSV + console logger. Saves all metrics
to runs/<timestamp>/metrics.csv for later plotting.
"""
from __future__ import annotations
import csv
import os
import time
from pathlib import Path
from typing import Dict, Any


class Logger:
    def __init__(self, log_dir: str = "runs"):
        run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.run_dir / "metrics.csv"
        self._writer = None
        self._file = None
        self._keys: list = []

    def _init_csv(self, keys: list):
        self._keys = keys
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=keys)
        self._writer.writeheader()

    def log(self, metrics: Dict[str, Any]):
        if self._writer is None:
            self._init_csv(list(metrics.keys()))
        self._writer.writerow({k: metrics.get(k, "") for k in self._keys})
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()

    def __del__(self):
        self.close()
