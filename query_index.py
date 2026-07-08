#!/usr/bin/env python3
"""Query a prepared SereneDB vector index and measure recall + latency.

Attaches to the serened left running by build_index.py (via its manifest),
sweeps sdb_nprobe, and reports recall@k, QPS, and latency percentiles.
Queries use server-side prepared statements with a binary-bound query vector
(inlining the vector as text costs ~5ms/query at dim 200). Example:

    python query_index.py --manifest results/manifest.json \
        --dataset synthetic --nb 100000 --nprobe 8,32,128
"""

import argparse
import json

from common import cli, metrics, runner
from common.server import Server


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    cli.add_nprobe_arg(p)
    cli.add_rerank_factor_arg(p)
    cli.add_concurrency_arg(p)
    p.add_argument("--manifest", default="results/manifest.json",
                   help="build manifest from build_index.py / build_index_remote.py")
    p.add_argument("--warmup", type=int, default=50, help="warmup queries per nprobe")
    p.add_argument("--repeats", type=int, default=1, help="timed passes over the query set")
    p.add_argument("--out", default="results/query_results.json", help="results JSON output")
    p.add_argument("--stop", action="store_true", help="kill serened after querying")
    args = p.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    ds = cli.load_dataset(args)
    if ds.dim != manifest["dim"]:
        raise SystemExit(f"dataset dim {ds.dim} != manifest dim {manifest['dim']}")

    server = Server.attach(manifest["pid"], manifest["port"], manifest["datadir"],
                           keep_datadir=not args.stop)
    print(f"attached pid={manifest['pid']} port={manifest['port']} "
          f"target={manifest['target']} quant={manifest['quant']}")

    results = runner.run_queries(server, ds, manifest["target"], k=args.k,
                                 nprobe_list=cli.parse_int_list(args.nprobe),
                                 rerank_factor_list=cli.parse_int_list(args.rerank_factor),
                                 clients_list=cli.parse_int_list(args.clients),
                                 warmup=args.warmup, repeats=args.repeats)

    out = {
        "manifest": manifest,
        "dataset": ds.name,
        "query": results,
        "ram_peak_query_mb": max((r["ram_peak_query_mb"] for r in results), default=0.0),
    }
    metrics.write_records(args.out, out)
    print(f"\nresults -> {args.out}")

    if args.stop:
        server.stop()
        print(f"stopped serened pid={manifest['pid']}")
    else:
        server.detach()


if __name__ == "__main__":
    main()
