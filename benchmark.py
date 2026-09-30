"""Benchmark inference latency and peak memory of a GA-selected model.

Run: python benchmark.py
The default individual is the best one found by the baseline GA run.
"""
import time
import tracemalloc
from typing import Dict, List

import ga_optimizer as ga

BATCH_SIZES: List[int] = [1, 10, 100, 300]
REPEATS = 20
DEFAULT_INDIVIDUAL: List[int] = [36, 13, 9]


def benchmark(ind: List[int]) -> List[Dict[str, float]]:
    """Return mean latency (ms) and peak memory (KB) per batch size."""
    X_train, X_val, _, y_train, _, _ = ga.load_data()
    model = ga.build_model(ind).fit(X_train, y_train)
    rows: List[Dict[str, float]] = []
    for size in BATCH_SIZES:
        batch = X_val[:size]
        model.predict(batch)  # warm-up call
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            model.predict(batch)
            times.append((time.perf_counter() - start) * 1000)
        tracemalloc.start()
        model.predict(batch)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "batch": size,
                "mean_ms": sum(times) / len(times),
                "ms_per_sample": sum(times) / len(times) / size,
                "peak_kb": peak / 1024,
            }
        )
    return rows


if __name__ == "__main__":
    print(f"Individual: {DEFAULT_INDIVIDUAL}")
    print(f"{'batch':>6} {'mean_ms':>9} {'ms/sample':>10} {'peak_KB':>9}")
    for r in benchmark(DEFAULT_INDIVIDUAL):
        print(f"{r['batch']:>6} {r['mean_ms']:>9.2f} {r['ms_per_sample']:>10.3f} {r['peak_kb']:>9.1f}")
