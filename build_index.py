#!/usr/bin/env python3
"""Build a SereneDB IVF vector index over LOCAL data and leave serened running.

Prepares the index (table + COPY load + IVF build) for one quantization
scenario, records build time / disk / peak-RAM, writes a manifest, and leaves
the server up so query_index.py can query it. Example:

    python build_index.py --dataset synthetic --nb 100000 --quant sq8 \
        --datadir /tmp/sdb_bench_data --manifest results/manifest.json
"""

import argparse
import json
import os

from common import cli, quant, runner
from common.server import Server


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dataset_args(p)
    cli.add_index_args(p)
    p.add_argument("--quant", choices=quant.SCENARIOS, default="sq8",
                   help="quantization scenario (default: sq8)")
    p.add_argument("--datadir", default="/tmp/sdb_vecbench_data",
                   help="serened data directory")
    p.add_argument("--load-via", choices=["copy", "parquet"], default="copy",
                   help="ingest via client COPY or server-side read_parquet CTAS")
    p.add_argument("--workdir", default="/tmp/sdb_vecbench_work",
                   help="scratch dir for the parquet when --load-via parquet")
    p.add_argument("--port", type=int, default=None, help="listen port (default: auto)")
    p.add_argument("--binary", default=None, help="path to serened binary")
    p.add_argument("--manifest", default="results/manifest.json",
                   help="where to write the build manifest")
    args = p.parse_args()

    ds = cli.load_dataset(args)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} metric={ds.metric}")

    server = Server(args.datadir, port=args.port, binary=args.binary, keep_datadir=True)
    server.start()
    print(f"serened pid={server.proc.pid} port={server.port} datadir={server.datadir}")

    build = runner.build_local(server, ds, args.quant, nlist=args.nlist,
                               nlist_factor=args.nlist_factor,
                               train_sample=args.train_sample, pq_m=args.pq_m,
                               rabitq_bits=args.rabitq_bits, load_via=args.load_via,
                               workdir=args.workdir, settle=args.settle)
    print(f"built {build['rows']} rows: load={build['load_s']:.2f}s "
          f"index={build['index_build_s']:.2f}s "
          f"disk={build['index_disk_bytes'] / 1e6:.1f}MB "
          f"ram_peak={build['ram_peak_build_mb']:.0f}MB")

    manifest = {
        "mode": "local",
        "pid": server.proc.pid,
        "port": server.port,
        "datadir": server.datadir,
        "quant": args.quant,
        "rabitq_bits": args.rabitq_bits,
        "dim": ds.dim,
        "dataset": ds.name,
        "target": build["target"],
        "build": build,
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    server.detach()
    print(f"\nmanifest -> {args.manifest}")
    print(f"query it:  python query_index.py --manifest {args.manifest} "
          f"--dataset {args.dataset}" + (f" --data-dir {args.data_dir}" if args.data_dir else ""))
    print(f"stop it:   kill {server.proc.pid}   (or query_index.py --stop)")


if __name__ == "__main__":
    main()
