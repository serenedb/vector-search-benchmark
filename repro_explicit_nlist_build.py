#!/usr/bin/env python3
"""Reproducer for the explicit-`nlist` IVF build-perf bug.

Builds the SAME sq8 IVF index (auto vs explicit nlist that resolve to the same
value) and sweeps CREATE INDEX parallelism. Demonstrates the backwards signature:
for an explicit `nlist`, build time RISES with thread count, because the parallel
sink trains the full coarse quantizer per build segment (later discarded by
VACUUM REFRESH). The auto path stays fast because it scales each segment's
centroid count to that segment's row count.

Run from this repo's root with the vecbench venv:
    ~/.venvs/vecbench/bin/python repro_explicit_nlist_build.py            # 10M (paper numbers)
    ~/.venvs/vecbench/bin/python repro_explicit_nlist_build.py --nb 2000000   # faster

Expected (10M, dim 200, ip, build_bench Release, 88-core host):
    explicit nlist=6325  threads=1    index_build_s ~ 232
    explicit nlist=6325  threads=8    index_build_s ~ 339
    explicit nlist=6325  threads=88   index_build_s ~ 620   <- more threads, SLOWER
    auto     nlist=None  threads=88   index_build_s ~ 120   <- same index, 5x faster
"""

import argparse
import os
import shutil
import time

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "T2I_DATA_DIR", os.path.join(REPO, "..", "big-ann-benchmarks", "data", "text2image1B"))
BINARY = os.environ.get("SERENED_BINARY", shutil.which("serened") or "serened")

from common import dataset, sdb
from common.remote import ensure_vectors_parquet
from common.server import Server

DIM = 200


def build_once(ds, parquet, nlist_opt, threads, datadir):
    server = Server(datadir, binary=BINARY, keep_datadir=False)
    server.start()
    cur = server.connect().cursor()
    sdb.execute(cur, "DROP TABLE IF EXISTS vec CASCADE")
    sdb.execute(cur, f"CREATE TABLE vec AS SELECT id, emb::FLOAT[{DIM}] AS emb "
                     f"FROM read_parquet('{parquet}')")
    sdb.execute(cur, f"SET threads = {threads}")
    nlist_sql = "" if nlist_opt is None else f"nlist = {nlist_opt}, "
    ddl = (f"CREATE INDEX vec_idx ON vec USING inverted("
           f"id, emb ivf (metric = 'ip', {nlist_sql}quant = 'sq8')) "
           f"WITH (compaction_interval = 0, refresh_interval = 0)")
    t = time.perf_counter()
    sdb.execute(cur, ddl)
    sdb.execute(cur, "VACUUM (REFRESH_TABLE) vec")
    dt = time.perf_counter() - t
    cur.close()
    server.stop()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=10_000_000)
    ap.add_argument("--nlist", type=int, default=6325)
    ap.add_argument("--threads", default="1,8,88")
    ap.add_argument("--workdir", default=os.path.join(REPO, "repro_nlist_work"))
    ap.add_argument("--datadir-root", default=os.path.join(REPO, "repro_nlist_data"))
    args = ap.parse_args()
    threads = [int(x) for x in args.threads.split(",") if x.strip()]

    print(f"loading t2i nb={args.nb} ...", flush=True)
    ds = dataset.load_t2i(DATA_DIR, nb=args.nb, nq=100, k=10,
                          gt_file=os.path.join(DATA_DIR, "text2image-10M")
                          if args.nb == 10_000_000 else None)
    os.makedirs(args.workdir, exist_ok=True)
    parquet = os.path.join(args.workdir, "vectors.parquet")
    ensure_vectors_parquet(parquet, ds.ids(), ds.base, fixed=True)

    print(f"\n{'case':22s} {'nlist':>6s} {'threads':>8s} {'index_build_s':>14s}", flush=True)
    for th in threads:
        dt = build_once(ds, parquet, args.nlist, th,
                        os.path.join(args.datadir_root, f"explicit_{args.nlist}_t{th}"))
        print(f"{'explicit nlist':22s} {args.nlist:>6d} {th:>8d} {dt:>14.1f}", flush=True)
    dt = build_once(ds, parquet, None, max(threads),
                    os.path.join(args.datadir_root, "auto"))
    print(f"{'auto nlist':22s} {'-':>6s} {max(threads):>8d} {dt:>14.1f}", flush=True)


if __name__ == "__main__":
    main()
