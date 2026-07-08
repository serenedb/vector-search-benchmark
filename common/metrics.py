"""Recall, latency, and result-record helpers."""

import json
import os
import time


def recall_at_k(returned_ids, gt_ids, k):
    """Mean recall@k: |returned[:k] ∩ groundtruth[:k]| / k, averaged over queries.

    `returned_ids` and `gt_ids` are lists (per query) of neighbor ids.
    """
    assert len(returned_ids) == len(gt_ids)
    total = 0.0
    for got, gt in zip(returned_ids, gt_ids):
        truth = set(gt[:k])
        if not truth:
            continue
        hit = sum(1 for x in got[:k] if x in truth)
        total += hit / min(k, len(truth))
    return total / len(returned_ids) if returned_ids else 0.0


def percentiles(latencies_ms, ps=(50, 95, 99)):
    if not latencies_ms:
        return {f"p{p}": 0.0 for p in ps}
    s = sorted(latencies_ms)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        out[f"p{p}"] = s[idx]
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0

    @property
    def ms(self):
        return self.dt * 1000.0


def latency_summary(latencies_ms):
    pct = percentiles(latencies_ms)
    return {
        "lat_ms_mean": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        "lat_ms_p50": pct["p50"],
        "lat_ms_p95": pct["p95"],
        "lat_ms_p99": pct["p99"],
        "lat_ms_min": min(latencies_ms) if latencies_ms else 0.0,
        "lat_ms_max": max(latencies_ms) if latencies_ms else 0.0,
    }


def write_records(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    return path


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
