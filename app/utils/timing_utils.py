from __future__ import annotations

import time


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> int:
    return int(max(0.0, monotonic_ms() - start_ms))
