"""Rolling latency statistics for Infer calls."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict

import numpy as np


class LatencyStats:
    def __init__(self, window: int = 1000) -> None:
        self._window = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_errors = 0

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._window.append(float(latency_ms))
            self._total_calls += 1

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            values = np.asarray(self._window, dtype=np.float64)
            total = self._total_calls
            errors = self._total_errors
        out: Dict[str, Any] = {
            "total_calls": total,
            "total_errors": errors,
            "window_size": int(values.size),
        }
        if values.size:
            out.update(summarize_ms(values))
        else:
            out.update({"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0})
        return out


def summarize_ms(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }
