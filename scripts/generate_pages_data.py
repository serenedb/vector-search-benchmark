#!/usr/bin/env python3
"""Distill results/serenedb/summary.csv and results/qdrant/qdrant_tuning_all.csv
into docs/data/comparison.json for the GitHub Pages comparison site.

For each recall@10 band, picks the highest-QPS config per engine (ties broken
by lower p50 latency) -- the same methodology results/qdrant/qdrant_tuning.md
already uses for Qdrant. Bands are 0.05-wide from 0.55 to 0.95, then 0.01-wide
from 0.95 to 1.00 where the interesting differentiation happens.

    python3 scripts/generate_pages_data.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERENEDB_CSV = os.path.join(ROOT, "results", "serenedb", "summary.csv")
QDRANT_CSV = os.path.join(ROOT, "results", "qdrant", "qdrant_tuning_all.csv")
OUT_JSON = os.path.join(ROOT, "docs", "data", "comparison.json")

COARSE_LO, FINE_LO, FINE_HI = 0.55, 0.95, 1.00
COARSE_STEP, FINE_STEP = 0.05, 0.01


def band_for(recall):
    """Recall -> band label, or None if below the chart floor (0.55)."""
    if recall < COARSE_LO:
        return None
    if recall < FINE_LO:
        n = int((recall - COARSE_LO) / COARSE_STEP)
        lo = round(COARSE_LO + n * COARSE_STEP, 2)
        hi = min(round(lo + COARSE_STEP, 2), FINE_LO)
        return f"{lo:.2f}-{hi:.2f}"
    n = int((recall - FINE_LO) / FINE_STEP + 1e-9)  # +eps guards float noise, e.g. 0.96-0.95 != 0.01 exactly
    n = min(n, int(round((FINE_HI - FINE_LO) / FINE_STEP)) - 1)  # recall==1.0 -> last bin
    lo = round(FINE_LO + n * FINE_STEP, 2)
    hi = min(round(lo + FINE_STEP, 2), FINE_HI)
    return f"{lo:.2f}-{hi:.2f}"


def all_band_labels():
    labels = []
    lo = COARSE_LO
    while lo < FINE_LO - 1e-9:
        hi = min(round(lo + COARSE_STEP, 2), FINE_LO)
        labels.append(f"{lo:.2f}-{hi:.2f}")
        lo = round(lo + COARSE_STEP, 2)
    lo = FINE_LO
    while lo < FINE_HI - 1e-9:
        hi = min(round(lo + FINE_STEP, 2), FINE_HI)
        labels.append(f"{lo:.2f}-{hi:.2f}")
        lo = round(lo + FINE_STEP, 2)
    return labels


def band_order(label):
    return float(label.split("-")[0])


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def serenedb_config(r):
    return (f"{r['quant']}/nlist={r['nlist']}/settle={r['settle']}/"
            f"np={r['nprobe']}/rr={r['rerank_factor']}")


def qdrant_config(r):
    return f"m={r['m']}/efc={r['ef_construct']}/quant={r['quant']}/os={r['oversampling']}/ef={r['hnsw_ef']}"


def serenedb_build_cost(r):
    total = float(r["build_total_s"])
    if r["settle"] == "compact":
        total += float(r["compact_s"])
    return total


def pick_winners(rows, config_fn, build_cost_fn):
    """One row per recall band: max qps, ties broken by lower p50."""
    by_band = {}
    for r in rows:
        band = band_for(float(r["recall_at_k"]))
        if band is None:
            continue
        qps = float(r["qps"])
        p50 = float(r["lat_ms_p50"])
        cur = by_band.get(band)
        if cur is None or (qps, -p50) > (cur["qps"], -cur["p50_ms"]):
            by_band[band] = {
                "band": band,
                "recall": round(float(r["recall_at_k"]), 4),
                "qps": round(qps, 1),
                "p50_ms": round(p50, 3),
                "p95_ms": round(float(r["lat_ms_p95"]), 3),
                "build_s": round(build_cost_fn(r), 1),
                "index_mb": round(float(r["index_disk_bytes"]) / 1e6, 1),
                "config": config_fn(r),
            }
    return [by_band[label] for label in sorted(by_band, key=band_order)]


def project(winners, *fields):
    return [{f: w[f] for f in fields} for w in winners]


# quant order follows the scenarios table in README.md
SERENEDB_QUANT_ORDER = ["sq8", "sq4", "pq", "rabitq"]
SERENEDB_BUILD_NLIST_FACTOR = "2"


def serenedb_build_size_by_quant(rows, nlist_factor):
    """One row per (quant, settle) at a fixed nlist_factor -- fields are
    constant across nprobe/rerank_factor within a (quant, settle) build, so
    any matching row carries the right numbers."""
    seen = {}
    for r in rows:
        if r["nlist_factor"] != nlist_factor:
            continue
        key = (r["quant"], r["settle"])
        if key in seen:
            continue
        seen[key] = {
            "quant": r["quant"],
            "settle": r["settle"],
            "nlist": int(r["nlist"]),
            "index_build_s": round(float(r["index_build_s"]), 1),
            "compact_s": round(float(r["compact_s"]), 1),
            "build_s": round(serenedb_build_cost(r), 1),
            "index_mb": round(float(r["index_disk_bytes"]) / 1e6, 1),
        }
    order = {q: i for i, q in enumerate(SERENEDB_QUANT_ORDER)}
    return sorted(seen.values(), key=lambda w: (order.get(w["quant"], 99), w["settle"]))


def qdrant_build_size_by_config(rows):
    """One row per (m, ef_construct, quant) -- build_s/index_mb are constant
    across oversampling/hnsw_ef within a build, so any matching row works."""
    seen = {}
    for r in rows:
        key = (int(r["m"]), int(r["ef_construct"]), r["quant"])
        if key in seen:
            continue
        seen[key] = {
            "m": key[0],
            "ef_construct": key[1],
            "quant": key[2],
            "build_s": round(float(r["build_s"]), 1),
            "index_mb": round(float(r["index_disk_bytes"]) / 1e6, 1),
        }
    quant_order = {"none": 0, "scalar": 1}
    return sorted(seen.values(), key=lambda w: (w["m"], w["ef_construct"], quant_order.get(w["quant"], 9)))


def main():
    sdb_rows = load_csv(SERENEDB_CSV)
    qdr_rows = load_csv(QDRANT_CSV)

    sdb_all = pick_winners(sdb_rows, serenedb_config, serenedb_build_cost)
    qdr_all = pick_winners(qdr_rows, qdrant_config, lambda r: float(r["build_s"]))

    meta = {
        "dataset": sdb_rows[0]["dataset"],
        "nb": int(sdb_rows[0]["nb"]),
        "dim": int(sdb_rows[0]["dim"]),
        "k": int(sdb_rows[0]["k"]),
        "clients": int(sdb_rows[0]["clients"]),
    }
    assert meta["dataset"] == qdr_rows[0]["dataset"], "serenedb/qdrant dataset mismatch"

    sdb_build_size = serenedb_build_size_by_quant(sdb_rows, SERENEDB_BUILD_NLIST_FACTOR)
    qdr_build_size = qdrant_build_size_by_config(qdr_rows)
    # Build-time panel shows one stacked bar per quantizer (index build +
    # compact merge) -- both numbers come from the same compact-settle
    # measurement, so it's a single coherent decomposition, not a
    # settle-policy comparison (that's what the index-size panel is for).
    sdb_build_time = [w for w in sdb_build_size if w["settle"] == "compact"]

    out = {
        "meta": meta,
        "recall_qps": {
            "serenedb": project(sdb_all, "band", "recall", "qps", "p50_ms", "p95_ms", "config"),
            "qdrant": project(qdr_all, "band", "recall", "qps", "p50_ms", "p95_ms", "config"),
        },
        "build_time": {
            "serenedb_by_quant": project(sdb_build_time, "quant", "nlist", "index_build_s", "compact_s"),
            "qdrant_by_config": project(qdr_build_size, "m", "ef_construct", "quant", "build_s"),
        },
        "index_size": {
            "serenedb_by_quant": project(sdb_build_size, "quant", "settle", "nlist", "index_mb"),
            "qdrant_by_config": project(qdr_build_size, "m", "ef_construct", "quant", "index_mb"),
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    all_bands = all_band_labels()
    print(f"wrote {OUT_JSON}")
    print(f"bands: {len(all_bands)} total ({all_bands[0]} .. {all_bands[-1]})")
    for name, winners in (("serenedb (recall/qps)", sdb_all), ("qdrant (recall/qps)", qdr_all)):
        covered = {w["band"] for w in winners}
        missing = [b for b in all_bands if b not in covered]
        print(f"  {name}: {len(covered)}/{len(all_bands)} bands covered"
              + (f", missing: {missing}" if missing else ""))
    print(f"serenedb build/size @ nlist_factor={SERENEDB_BUILD_NLIST_FACTOR}: "
          f"{len(sdb_build_size)} (quant, settle) combos "
          f"({sorted({w['quant'] for w in sdb_build_size})})")
    print(f"qdrant build/size: {len(qdr_build_size)} (m, ef_construct, quant) combos")


if __name__ == "__main__":
    main()
