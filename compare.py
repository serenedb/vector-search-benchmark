#!/usr/bin/env python3
"""Head-to-head ANN comparison: SereneDB vs Qdrant vs Elasticsearch on the SAME
Text-to-Image data, ids, queries, and ground truth -- no Docker (each engine runs
a standalone server). Reports recall@k, QPS, latency, build time, index disk, and
peak RAM per engine across a search-effort x concurrency sweep.

    python compare.py --dataset t2i --data-dir <t2i> --gt-file <gt> --nb 1000000 \
        --engines serenedb,qdrant --search-params 16,32,64,128 --clients 1,8

The --search-params list maps to each engine's search-effort knob
(SereneDB nprobe, Qdrant hnsw_ef, Elasticsearch num_candidates).
"""

import argparse
import csv
import json
import os

from common import cli
from engines import harness

METRIC = "ip"  # Text2Image is max inner product


def make_engine(name, args):
    ddir = os.path.join(args.datadir, name)
    wdir = os.path.join(args.workdir, name)
    if name == "serenedb":
        from engines.serenedb_engine import SereneDBEngine
        return SereneDBEngine(ddir, binary=args.serened_binary, quant_kind=args.sdb_quant,
                              nlist=args.sdb_nlist, settle=args.sdb_settle,
                              load_via=args.load_via, workdir=wdir,
                              rabitq_bits=args.sdb_rabitq_bits,
                              rerank_factor=args.sdb_rerank_factor)
    if name == "qdrant":
        from engines.qdrant_engine import QdrantEngine
        return QdrantEngine(ddir, binary=args.qdrant_binary, m=args.hnsw_m,
                            ef_construct=args.hnsw_ef_construct)
    if name == "elasticsearch":
        from engines.es_engine import ElasticsearchEngine
        return ElasticsearchEngine(ddir, home=args.es_home, m=args.hnsw_m,
                                   ef_construct=args.hnsw_ef_construct)
    raise SystemExit(f"unknown engine {name!r}")


def write_summary(out_dir, records, dim, nb, dataset, k):
    if not records:
        return
    cols = list(records[0].keys())
    with open(os.path.join(out_dir, "compare_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    seen, builds = set(), []
    for r in records:
        if r["engine"] not in seen:
            seen.add(r["engine"])
            builds.append(r)
    lines = ["# SereneDB vs Qdrant vs Elasticsearch — Text-to-Image", "",
             f"dataset={dataset} nb={nb} dim={dim} metric=ip k={k}", "",
             "## Build cost", "",
             "| engine | build_s | index_MB | ram_build_MB |",
             "|---|---:|---:|---:|"]
    for b in builds:
        lines.append(f"| {b['engine']} | {b['build_s']:.2f} | "
                     f"{b['index_disk_bytes'] / 1e6:.1f} | {b['ram_peak_build_mb']:.0f} |")
    lines += ["", "## Recall / throughput", "",
              "| engine | search_effort | clients | recall@k | qps | p50_ms | p95_ms | ram_query_MB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['engine']} | {r['search_effort']} | {r['clients']} | "
                     f"{r['recall_at_k']:.4f} | {r['qps']:.1f} | {r['lat_ms_p50']:.3f} | "
                     f"{r['lat_ms_p95']:.3f} | {r['ram_peak_query_mb']:.0f} |")
    with open(os.path.join(out_dir, "compare_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    cli.add_concurrency_arg(p)
    p.add_argument("--engines", default="serenedb,qdrant",
                   help="comma list of {serenedb,qdrant,elasticsearch}")
    p.add_argument("--search-params", default="16,32,64,128",
                   help="search-effort sweep -> nprobe / hnsw_ef / num_candidates")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--datadir", default="/tmp/vecbench_cmp/data")
    p.add_argument("--workdir", default="/tmp/vecbench_cmp/work")
    p.add_argument("--out-dir", default="results")
    # per-engine knobs
    p.add_argument("--serened-binary", default=None)
    p.add_argument("--sdb-quant", default="none",
                   help="SereneDB IVF quant (none/sq8/sq4/pq/rabitq); 'none' = full precision like Qdrant")
    p.add_argument("--sdb-rabitq-bits", type=int, default=None,
                   help="RaBitQ bits per dimension, 1-9 (--sdb-quant rabitq only, default 1)")
    p.add_argument("--sdb-rerank-factor", type=int, default=None,
                   help="sdb_rerank_factor for quantized SereneDB indexes -- exact-rerank pool "
                        "= factor * k; 0 disables reranking. Default: unset (server default 4). "
                        "No effect when --sdb-quant none")
    p.add_argument("--sdb-nlist", type=int, default=None)
    p.add_argument("--sdb-settle", default="compact", choices=["compact", "no-compact", "wait", "none"])
    p.add_argument("--load-via", default="parquet", choices=["parquet", "copy"])
    p.add_argument("--hnsw-m", type=int, default=16, help="HNSW M (Qdrant/ES)")
    p.add_argument("--hnsw-ef-construct", type=int, default=100, help="HNSW ef_construction")
    p.add_argument("--qdrant-binary", default=None)
    p.add_argument("--es-home", default=None, help="path to an unpacked Elasticsearch dir")
    args = p.parse_args()

    ds = cli.load_dataset(args)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} metric={METRIC}")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    sps = cli.parse_int_list(args.search_params)
    clients = cli.parse_int_list(args.clients)

    records = []
    for name in engines:
        print(f"\n===== engine: {name} =====")
        engine = make_engine(name, args)
        engine.start()
        try:
            out = harness.run_engine(engine, ds, k=args.k, search_params=sps,
                                     clients_list=clients, warmup=args.warmup, metric=METRIC)
        finally:
            engine.stop()
        for q in out["query"]:
            rec = {"engine": name, "dataset": ds.name, "nb": ds.nb, "dim": ds.dim,
                   "build_s": out["build_s"], "index_disk_bytes": out["index_disk_bytes"],
                   "ram_peak_build_mb": out["ram_peak_build_mb"],
                   "search_effort": q[engine.search_param_name], "clients": q["clients"],
                   "k": q["k"], "recall_at_k": q["recall_at_k"], "qps": q["qps"],
                   "ram_peak_query_mb": q["ram_peak_query_mb"],
                   "lat_ms_p50": q["lat_ms_p50"], "lat_ms_p95": q["lat_ms_p95"],
                   "lat_ms_p99": q["lat_ms_p99"], "lat_ms_mean": q["lat_ms_mean"]}
            records.append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "compare_results.json"), "w") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    write_summary(args.out_dir, records, ds.dim, ds.nb, ds.name, args.k)
    print(f"\nwrote {args.out_dir}/compare_results.json, compare_summary.csv, compare_summary.md")


if __name__ == "__main__":
    main()
