#!/usr/bin/env python3
"""Tune Qdrant HNSW + scalar-quantization parameters on Text-to-Image and pick the
best config per recall@10 segment.

For every (m, ef_construct, quant) build config the index is built ONCE, then the
query grid (hnsw_ef x oversampling) is swept at a fixed concurrency, measuring
recall@10 and QPS. "Best per segment" = the config with the highest QPS whose
achieved recall@10 lands in that 0.05-wide recall band.

    python tune_qdrant.py --data-dir <t2i> --nb 1000000 --nq 10000 --clients 32 \
        --m-list 16,32,48,64 --ef-construct-list 100,200 --quant-list none,scalar \
        --oversampling-list 1.0,2.0,4.0 --search-params 16,32,64,128,256,512

Results accumulate in <out>_all.json / <out>_all.csv (across dataset sizes, keyed
by nb + build/query params). The best-per-segment table is written to <out>.md.
Use --report-only to regenerate the tables from existing raw results without running.
"""

import argparse
import csv
import json
import os

from common import cli, metrics
from common.server import RssSampler
from engines import harness
from engines.qdrant_engine import QdrantEngine

METRIC = "ip"  # Text2Image is max inner product
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# recall band edges: 0.55-0.60, 0.60-0.65, ... , 0.90-0.95, then finer near the
# ceiling (0.95-0.97, 0.97-0.99, 0.99-0.995, 0.995-1.0) so a cheap high-QPS config
# can't bury a much-higher-recall config by winning a too-wide top band.
SEGMENTS = [(round(0.55 + 0.05 * i, 2), round(0.60 + 0.05 * i, 2)) for i in range(8)] + \
           [(0.95, 0.97), (0.97, 0.99), (0.99, 0.995), (0.995, 1.0)]

# every field written to the raw CSV/JSON, in order
FIELDS = ["nb", "dataset", "dim", "clients", "k", "m", "ef_construct", "quant",
          "oversampling", "hnsw_ef", "recall_at_k", "qps",
          "lat_ms_mean", "lat_ms_p50", "lat_ms_p95", "lat_ms_p99",
          "build_s", "index_disk_bytes", "ram_peak_build_mb", "ram_peak_query_mb",
          "n_queries"]

# fields that identify a unique measurement (for de-dup / merge across runs)
KEY_FIELDS = ["nb", "clients", "k", "m", "ef_construct", "quant", "oversampling", "hnsw_ef"]


def rec_key(r):
    return tuple(r.get(f) for f in KEY_FIELDS)


def load_raw(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def merge_raw(existing, new):
    """New records overwrite existing ones with the same key; order preserved."""
    by_key = {rec_key(r): r for r in existing}
    for r in new:
        by_key[rec_key(r)] = r
    return list(by_key.values())


def write_raw(out, records):
    with open(out + "_all.json", "w") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    with open(out + "_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: (r.get("nb", 0), r.get("recall_at_k", 0))))


def best_per_segment(records):
    """For each recall band, the config with the max QPS whose recall lands in it.

    QPS saturates at the Python client's throughput ceiling for cheap configs, so
    ties near that ceiling are broken by lower p50 latency (the practical winner)."""
    out = []
    for lo, hi in SEGMENTS:
        in_band = [r for r in records
                   if lo <= r["recall_at_k"] < hi or (hi >= 1.0 and r["recall_at_k"] >= lo)]
        best = max(in_band, key=lambda r: (r["qps"], -r["lat_ms_p50"])) if in_band else None
        out.append(((lo, hi), best))
    return out


def write_report(out, records, nb_filter, dataset, k):
    if nb_filter is not None:
        records = [r for r in records if r["nb"] == nb_filter]
    nbs = sorted({r["nb"] for r in records})
    nb_label = str(nb_filter) if nb_filter is not None else ",".join(str(n) for n in nbs)
    cl = sorted({r["clients"] for r in records})
    cl_label = ",".join(str(c) for c in cl)
    lines = [
        "# Qdrant parameter tuning — Text-to-Image (best config per recall segment)",
        "",
        f"dataset={dataset} nb={nb_label} dim=200 metric=ip k={k} — "
        f"search concurrency = {cl_label} clients (QPS is closed-loop throughput).",
        "",
        "Winner per recall@10 band = **highest QPS** among all "
        f"(m, ef_construct, quant, oversampling, hnsw_ef) configs at {cl_label} clients "
        f"whose achieved recall@10 falls in the band (ties broken by lower p50 latency). "
        f"Total configs measured: {len(records)}.",
        "",
        "> Note: at this concurrency the Python gRPC client saturates around a fixed "
        "throughput ceiling, so QPS is flat across the cheaper configs and the **p50/p95 "
        "latency** columns are the practical discriminator there; QPS only separates configs "
        "at the expensive high-recall end.",
        "",
        "| recall band | m | ef_construct | quant | oversampling | hnsw_ef | recall@10 | QPS | p50 ms | p95 ms | build s | index MB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (lo, hi), r in best_per_segment(records):
        # 3 decimals: the finer top-end bands (0.99-0.995, 0.995-1.0) round
        # identically at 2 decimals due to float representation of 0.995.
        band = f"{lo:.3f}–{hi:.3f}"
        if r is None:
            lines.append(f"| {band} | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        ov = "—" if r["oversampling"] is None else f"{r['oversampling']:g}"
        lines.append(
            f"| {band} | {r['m']} | {r['ef_construct']} | {r['quant']} | {ov} | "
            f"{r['hnsw_ef']} | {r['recall_at_k']:.4f} | {r['qps']:.1f} | "
            f"{r['lat_ms_p50']:.2f} | {r['lat_ms_p95']:.2f} | {r['build_s']:.1f} | "
            f"{r['index_disk_bytes'] / 1e6:.0f} |")
    lines += ["", f"Raw per-config results: `{os.path.basename(out)}_all.csv` "
              f"({len(records)} rows).", ""]
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")


def run_sweep(args):
    ds = cli.load_dataset(args)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} metric={METRIC}")
    queries = [list(map(float, q)) for q in ds.queries]
    gt = ds.gt_list()

    ms = cli.parse_int_list(args.m_list)
    efcs = cli.parse_int_list(args.ef_construct_list)
    quants = [q.strip() for q in args.quant_list.split(",") if q.strip()]
    oversamplings = [float(x) for x in str(args.oversampling_list).split(",") if x.strip()]
    hnsw_efs = cli.parse_int_list(args.search_params)

    new_records = []
    for quant in quants:
        for m in ms:
            for efc in efcs:
                print(f"\n===== build: m={m} ef_construct={efc} quant={quant} =====")
                ddir = os.path.join(args.datadir, f"m{m}_efc{efc}_{quant}")
                engine = QdrantEngine(ddir, binary=args.qdrant_binary, m=m, ef_construct=efc,
                                      quant=quant, upload_parallel=args.upload_parallel)
                engine.start()
                sampler = RssSampler(engine.pid)
                sampler.start()
                try:
                    sampler.start_phase("build")
                    build_s = engine.build(ds.ids(), ds.base, ds.dim, METRIC)
                    sampler.end_phase()
                    ram_build = sampler.phase_peak_mb("build")
                    disk = engine.disk_bytes()
                    print(f"  build={build_s:.1f}s disk={disk / 1e6:.0f}MB ram={ram_build:.0f}MB")

                    ov_list = oversamplings if quant != "none" else [None]
                    for ov in ov_list:
                        engine.oversampling = ov  # read by new_session()
                        for ef in hnsw_efs:
                            rec = harness._combo(engine, sampler, ef, args.clients,
                                                 queries, gt, args.k, args.warmup, print)
                            row = {
                                "nb": ds.nb, "dataset": ds.name, "dim": ds.dim,
                                "clients": args.clients, "k": args.k,
                                "m": m, "ef_construct": efc, "quant": quant,
                                "oversampling": ov, "hnsw_ef": ef,
                                "recall_at_k": rec["recall_at_k"], "qps": rec["qps"],
                                "lat_ms_mean": rec["lat_ms_mean"], "lat_ms_p50": rec["lat_ms_p50"],
                                "lat_ms_p95": rec["lat_ms_p95"], "lat_ms_p99": rec["lat_ms_p99"],
                                "build_s": build_s, "index_disk_bytes": disk,
                                "ram_peak_build_mb": ram_build,
                                "ram_peak_query_mb": rec["ram_peak_query_mb"],
                                "n_queries": rec["n_queries"],
                            }
                            new_records.append(row)
                            print(f"  m={m} efc={efc} quant={quant} ov={ov} hnsw_ef={ef:<5d} "
                                  f"recall@{args.k}={rec['recall_at_k']:.4f} qps={rec['qps']:8.1f}")
                finally:
                    sampler.stop()
                    engine.stop()
    return ds, new_records


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    p.add_argument("--clients", type=int, default=32,
                   help="concurrent client connections (QPS = closed-loop throughput)")
    p.add_argument("--m-list", default="16,32,48,64", help="HNSW m sweep")
    p.add_argument("--ef-construct-list", default="100,200", help="HNSW ef_construct sweep")
    p.add_argument("--quant-list", default="none,scalar", help="none,scalar")
    p.add_argument("--oversampling-list", default="1.0,2.0,4.0",
                   help="quantized-search oversampling sweep (rescore on); ignored for quant=none")
    p.add_argument("--search-params", default="16,32,64,128,256,512", help="hnsw_ef sweep")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--upload-parallel", type=int, default=16,
                   help="qdrant upload parallelism (uniform across combos)")
    p.add_argument("--qdrant-binary", default=None)
    p.add_argument("--datadir", default="/tmp/vecbench_qdrant_tune")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "qdrant_tuning"),
                   help="output prefix; writes <out>.md, <out>_all.json, <out>_all.csv")
    p.add_argument("--report-only", action="store_true",
                   help="skip the sweep; just regenerate report from <out>_all.json")
    p.add_argument("--only-nb", type=int, default=None,
                   help="restrict the best-per-segment report to this dataset size")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dataset_name = "t2i"

    if args.report_only:
        records = load_raw(args.out + "_all.json")
        if not records:
            raise SystemExit(f"no records in {args.out}_all.json to report on")
        write_report(args.out, records, args.only_nb, dataset_name, args.k)
        print(f"wrote {args.out}.md (report-only, {len(records)} raw records)")
        return

    ds, new_records = run_sweep(args)
    all_records = merge_raw(load_raw(args.out + "_all.json"), new_records)
    write_raw(args.out, all_records)
    write_report(args.out, all_records, args.only_nb, ds.name, args.k)
    print(f"\nwrote {args.out}.md, {args.out}_all.json, {args.out}_all.csv "
          f"({len(new_records)} new, {len(all_records)} total records)")


if __name__ == "__main__":
    main()
