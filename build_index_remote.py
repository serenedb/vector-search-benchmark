#!/usr/bin/env python3
"""Build a SereneDB IVF vector index over REMOTE data, then query it.

SereneDB indexes remote data by scanning the source at CREATE INDEX time; only
the index (not the row data) is stored locally. This publishes the benchmark
vectors to the chosen source and builds a zero-copy index over a view of it:

    --source iceberg   local Iceberg table (embedding stored as FLOAT[] LIST)
    --source hf         hf://datasets/... parquet (upload with HF_TOKEN, or --hf-uri)
    --source http       parquet served over local HTTP + httpfs (offline stand-in)
    --source file       read_parquet('<local path>') (debugging)

Build and query run in one process so the remote source stays alive throughout.
Example:

    python build_index_remote.py --source iceberg --dataset synthetic \
        --nb 100000 --quant sq8 --nprobe 8,32,128
"""

import argparse
import os

from common import cli, metrics, quant, remote, runner
from common.remote import ensure_vectors_parquet
from common.server import Server


def make_remote(args, parquet_path, workdir):
    if args.source == "file":
        return remote.FileRemote(parquet_path)
    if args.source == "http":
        return remote.HttpRemote(parquet_path)
    if args.source == "iceberg":
        return remote.IcebergRemote(parquet_path, os.path.join(workdir, "iceberg_table"))
    if args.source == "hf":
        return remote.HfRemote(parquet_path=parquet_path, repo=args.hf_repo,
                               path_in_repo=args.hf_path_in_repo, hf_uri=args.hf_uri)
    raise SystemExit(f"unknown source {args.source}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    cli.add_index_args(p)
    cli.add_nprobe_arg(p)
    cli.add_rerank_factor_arg(p)
    cli.add_concurrency_arg(p)
    p.add_argument("--source", choices=["iceberg", "hf", "http", "file"], default="iceberg")
    p.add_argument("--quant", choices=quant.SCENARIOS, default="sq8")
    p.add_argument("--datadir", default="/tmp/sdb_vecbench_remote_data")
    p.add_argument("--workdir", default="/tmp/sdb_vecbench_remote_work",
                   help="scratch dir for published parquet / iceberg table")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--binary", default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", default="results/remote_results.json")
    p.add_argument("--hf-repo", default=None, help="HF dataset repo to upload to (needs HF_TOKEN)")
    p.add_argument("--hf-path-in-repo", default="vectors.parquet")
    p.add_argument("--hf-uri", default=None,
                   help="existing hf:// parquet to index instead of uploading")
    args = p.parse_args()

    ds = cli.load_dataset(args)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} source={args.source}")

    os.makedirs(args.workdir, exist_ok=True)
    src = None
    if args.source == "hf" and args.hf_uri:
        rem = remote.HfRemote(hf_uri=args.hf_uri)
    else:
        parquet_path = os.path.join(args.workdir, "vectors.parquet")
        print(f"ensuring parquet ({ds.nb} vectors) -> {parquet_path}")
        ensure_vectors_parquet(parquet_path, ds.ids(), ds.base, fixed=True)
        rem = make_remote(args, parquet_path, args.workdir)

    server = Server(args.datadir, port=args.port, binary=args.binary, keep_datadir=False)
    server.start()
    print(f"serened pid={server.proc.pid} port={server.port}")

    try:
        build = runner.build_remote(server, ds, args.quant, rem, nlist=args.nlist,
                                    train_sample=args.train_sample, pq_m=args.pq_m,
                                    rabitq_bits=args.rabitq_bits, settle=args.settle,
                                    build_threads=args.build_threads)
        print(f"built {build['rows']} rows over {build['remote_uri']}: "
              f"prep={build['remote_prep_s']:.2f}s index={build['index_build_s']:.2f}s "
              f"index_disk={build['index_disk_bytes'] / 1e6:.1f}MB "
              f"ram_peak={build['ram_peak_build_mb']:.0f}MB")

        results = runner.run_queries(server, ds, build["target"], k=args.k,
                                     nprobe_list=cli.parse_int_list(args.nprobe),
                                     rerank_factor_list=cli.parse_int_list(args.rerank_factor),
                                     clients_list=cli.parse_int_list(args.clients),
                                     warmup=args.warmup, repeats=args.repeats)
        out = {
            "mode": "remote",
            "source": args.source,
            "quant": args.quant,
            "dataset": ds.name,
            "dim": ds.dim,
            "build": build,
            "query": results,
        }
        metrics.write_records(args.out, out)
        print(f"\nresults -> {args.out}")
    finally:
        rem.teardown()
        server.stop()


if __name__ == "__main__":
    main()
