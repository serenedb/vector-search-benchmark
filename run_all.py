#!/usr/bin/env python3
"""Run the full vector-ANN benchmark matrix and write a summary.

Sweeps {sources} x {quantization} x {sdb_nprobe}, one fresh serened per
(source, quant), and emits results/all_results.json, results/summary.csv, and a
human-readable results/summary.md (build-cost table + recall/QPS table).

    python run_all.py --dataset synthetic --nb 200000 \
        --sources local,iceberg --quant none,sq8,sq4,pq --nprobe 8,32,128
"""

import argparse
import csv
import json
import os

from common import cli, quant, remote, runner
from common.remote import ensure_vectors_parquet
from common.server import Server

BUILD_FIELDS = ["rows", "load_s", "index_build_s", "compact_s", "build_total_s",
                "index_disk_bytes", "datadir_bytes", "ram_peak_build_mb", "remote_prep_s",
                "build_threads"]


def make_remote(source, parquet_path, workdir, args):
    if source == "file":
        return remote.FileRemote(parquet_path)
    if source == "http":
        return remote.HttpRemote(parquet_path)
    if source == "iceberg":
        return remote.IcebergRemote(parquet_path, os.path.join(workdir, "iceberg_table"))
    if source == "hf":
        return remote.HfRemote(parquet_path=parquet_path, repo=args.hf_repo,
                               path_in_repo=args.hf_path_in_repo, hf_uri=args.hf_uri)
    raise SystemExit(f"unknown source {source}")


def run_matrix(args, ds):
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    quants = [q.strip() for q in args.quant.split(",") if q.strip()]
    nprobe_list = cli.parse_int_list(args.nprobe)
    rerank_factor_list = cli.parse_int_list(args.rerank_factor)
    clients_list = cli.parse_int_list(args.clients)
    records = []

    for source in sources:
        rem = None
        if source != "local":
            os.makedirs(args.workdir, exist_ok=True)
            parquet_path = None
            if not (source == "hf" and args.hf_uri):
                parquet_path = os.path.join(args.workdir, f"{source}_vectors.parquet")
                print(f"[{source}] ensuring parquet ({ds.nb} vectors) -> {parquet_path}")
                ensure_vectors_parquet(parquet_path, ds.ids(), ds.base, fixed=True)
            rem = (remote.HfRemote(hf_uri=args.hf_uri) if (source == "hf" and args.hf_uri)
                   else make_remote(source, parquet_path, args.workdir, args))

        for q in quants:
            datadir = os.path.join(args.datadir, f"{source}_{q}")
            print(f"\n=== source={source} quant={q} ===")
            server = Server(datadir, binary=args.binary, keep_datadir=False)
            server.start()
            try:
                if source == "local":
                    build = runner.build_local(server, ds, q, nlist=args.nlist,
                                               nlist_factor=args.nlist_factor,
                                               train_sample=args.train_sample, pq_m=args.pq_m,
                                               rabitq_bits=args.rabitq_bits,
                                               load_via=args.load_via, settle=args.settle,
                                               workdir=os.path.join(args.workdir, "local"))
                else:
                    build = runner.build_remote(server, ds, q, rem, nlist=args.nlist,
                                                nlist_factor=args.nlist_factor,
                                                train_sample=args.train_sample, pq_m=args.pq_m,
                                                rabitq_bits=args.rabitq_bits, settle=args.settle,
                                                build_threads=args.build_threads)
                print(f"  build: total={build['build_total_s']:.2f}s "
                      f"index_disk={build['index_disk_bytes'] / 1e6:.1f}MB "
                      f"ram={build['ram_peak_build_mb']:.0f}MB")
                qres = runner.run_queries(server, ds, build["target"], k=args.k,
                                          nprobe_list=nprobe_list,
                                          rerank_factor_list=rerank_factor_list,
                                          clients_list=clients_list,
                                          warmup=args.warmup, repeats=args.repeats)
                for r in qres:
                    rec = {"source": source, "quant": q, "settle": args.settle,
                           "dataset": ds.name, "dim": ds.dim, "nb": ds.nb}
                    rec.update({f: build.get(f) for f in BUILD_FIELDS})
                    rec.update(r)
                    records.append(rec)
            finally:
                server.stop()
        if rem is not None:
            rem.teardown()
    return records


def write_summary(out_dir, records):
    if not records:
        return
    cols = list(records[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    seen, builds = set(), []
    for r in records:
        key = (r["source"], r["quant"])
        if key not in seen:
            seen.add(key)
            builds.append(r)

    lines = ["# Vector-ANN benchmark summary", ""]
    d0 = records[0]
    lines.append(f"dataset={d0['dataset']} nb={d0['nb']} dim={d0['dim']} "
                 f"metric=ip k={d0['k']}")
    lines += ["", "## Build cost", "",
              "| source | quant | rows | build_s | index_MB | ram_build_MB | build_threads |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for b in builds:
        lines.append(f"| {b['source']} | {b['quant']} | {b['rows']} | "
                     f"{b['build_total_s']:.2f} | {b['index_disk_bytes'] / 1e6:.1f} | "
                     f"{b['ram_peak_build_mb']:.0f} | {b.get('build_threads', '')} |")
    lines += ["", "## Recall / throughput", "",
              "| source | quant | nprobe | rerank_factor | clients | recall@k | qps | "
              "p50_ms | p95_ms | ram_query_MB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['source']} | {r['quant']} | {r['nprobe']} | "
                     f"{r['rerank_factor']} | {r['clients']} | "
                     f"{r['recall_at_k']:.4f} | {r['qps']:.1f} | "
                     f"{r['lat_ms_p50']:.3f} | {r['lat_ms_p95']:.3f} | "
                     f"{r['ram_peak_query_mb']:.0f} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    cli.add_index_args(p)
    cli.add_nprobe_arg(p)
    cli.add_rerank_factor_arg(p)
    cli.add_concurrency_arg(p)
    p.add_argument("--sources", default="local",
                   help="comma list of {local,iceberg,hf,http,file} (default: local)")
    p.add_argument("--quant", default=",".join(quant.SCENARIOS),
                   help="comma list of quant scenarios (default: none,sq8,sq4,pq,rabitq)")
    p.add_argument("--datadir", default="/tmp/sdb_vecbench/data")
    p.add_argument("--workdir", default="/tmp/sdb_vecbench/work")
    p.add_argument("--load-via", choices=["copy", "parquet"], default="copy",
                   help="local ingest: 'copy' (client) or 'parquet' (server-side CTAS, "
                        "much faster for large nb; needs workdir space)")
    p.add_argument("--binary", default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-path-in-repo", default="vectors.parquet")
    p.add_argument("--hf-uri", default=None)
    args = p.parse_args()

    ds = cli.load_dataset(args)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} metric={ds.metric}")

    records = run_matrix(args, ds)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    write_summary(args.out_dir, records)
    print(f"\nwrote {args.out_dir}/all_results.json, summary.csv, summary.md")


if __name__ == "__main__":
    main()
