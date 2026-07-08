#!/usr/bin/env python3
"""T2I-10M sweep over quant {sq4,sq8,pq} x nlist_factor {4,8} x settle {compact,no-compact}.

Loads the dataset once, then for each (quant, factor, settle) builds a fresh
IVF index via the auto-path nlist_factor knob and sweeps sdb_nprobe x
sdb_rerank_factor at clients=32. nprobe scales with the resolved nlist
(= round(factor*sqrt(rows))) so recall bands are comparable across factors.
settle='compact' runs VACUUM (COMPACT_TABLE) after the build (merged, single-
segment index); settle='no-compact' leaves the build's segments un-merged
(faster build, typically slower search) -- a clean A/B on the merge cost.

Requires a serened built with the Flat-shape fix (kFlatMaxRows = row group),
otherwise every row-group segment takes the O(N*nlist) flat build path.

    ~/.venvs/vecbench/bin/python t2i_nlist_factor_quant_sweep.py
"""

import argparse
import csv
import json
import math
import os
import shutil

VEC_ANN = os.path.dirname(os.path.abspath(__file__))

from common import dataset, runner            # noqa: E402
from common.server import Server              # noqa: E402

DATA_DIR = os.environ.get(
    "T2I_DATA_DIR", os.path.join(VEC_ANN, "..", "big-ann-benchmarks", "data", "text2image1B"))
BINARY = os.environ.get("SERENED_BINARY", shutil.which("serened") or "serened")   # Release, has the fix
GT_FILE = os.path.join(DATA_DIR, "text2image-10M")

FRACS = (0.005, 0.01, 0.02, 0.04, 0.08)   # 5 nprobe points, iso-fraction of nlist

BUILD_FIELDS = ["rows", "load_s", "index_build_s", "compact_s", "build_total_s",
                "index_disk_bytes", "datadir_bytes", "ram_peak_build_mb", "build_threads"]


def resolved_nlist(factor, nb):
    return max(1, round(factor * math.sqrt(nb)))


def nprobe_grid(nlist):
    return sorted({max(8, round(f * nlist)) for f in FRACS})


def iso_recall_best(records, thresholds=(0.90, 0.95, 0.99)):
    by = {}
    for r in records:
        by.setdefault((r["nlist_factor"], r["settle"]), []).append(r)
    lines = []
    for f in sorted(by, key=lambda k: (k[0], k[1])):
        row = [f"{f[0]} ({by[f][0]['nlist']}) / {f[1]}"]
        for t in thresholds:
            ok = [r for r in by[f] if r["recall_at_k"] >= t]
            if ok:
                b = max(ok, key=lambda r: r["qps"])
                row.append(f"{b['qps']:.0f} @np{b['nprobe']}/rr{b['rerank_factor']} "
                           f"(r={b['recall_at_k']:.3f})")
            else:
                row.append("-")
        lines.append(row)
    return thresholds, lines


def write_summary(out_dir, records):
    if not records:
        return
    allkeys = []
    for r in records:
        for k in r:
            if k not in allkeys:
                allkeys.append(k)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allkeys)
        w.writeheader()
        w.writerows(records)

    quants = []
    for r in records:
        if r["quant"] not in quants:
            quants.append(r["quant"])

    d0 = records[0]
    lines = ["# T2I nlist_factor x quant sweep", "",
             f"dataset={d0['dataset']} nb={d0['nb']} dim={d0['dim']} metric=ip "
             f"k={d0['k']} clients=32", ""]

    for quant in quants:
        qrecs = [r for r in records if r["quant"] == quant]
        ordered = sorted(qrecs, key=lambda r: (r["nlist_factor"], r["settle"], r["nprobe"],
                                               r["rerank_factor"]))
        seen, builds = set(), []
        for r in ordered:
            key = (r["nlist_factor"], r["settle"])
            if key not in seen:
                seen.add(key)
                builds.append(r)

        lines += [f"## quant = {quant}", "", "### Build cost", "",
                  "| nlist_factor | settle | nlist | rows | build_s | index_s | compact_s | index_MB | ram_build_MB |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for b in builds:
            lines.append(f"| {b['nlist_factor']} | {b['settle']} | {b['nlist']} | {b.get('rows')} | "
                         f"{(b.get('build_total_s') or 0):.1f} | "
                         f"{(b.get('index_build_s') or 0):.1f} | "
                         f"{(b.get('compact_s') or 0):.1f} | "
                         f"{(b.get('index_disk_bytes') or 0) / 1e6:.0f} | "
                         f"{(b.get('ram_peak_build_mb') or 0):.0f} |")

        thresholds, iso = iso_recall_best(ordered)
        lines += ["", "### Best QPS at iso-recall (32 clients)", "",
                  "| factor (nlist) / settle | " + " | ".join(f"recall>={t:.2f}" for t in thresholds) + " |",
                  "|---|" + "|".join(["---"] * len(thresholds)) + "|"]
        for row in iso:
            lines.append("| " + " | ".join(row) + " |")

        lines += ["", "### Recall / throughput (full grid)", "",
                  "| factor | settle | nlist | nprobe | rerank | recall@k | qps | p50_ms | p95_ms |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in ordered:
            lines.append(f"| {r['nlist_factor']} | {r['settle']} | {r['nlist']} | {r['nprobe']} | "
                         f"{r['rerank_factor']} | {r['recall_at_k']:.4f} | {r['qps']:.1f} | "
                         f"{r['lat_ms_p50']:.3f} | {r['lat_ms_p95']:.3f} |")
        lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def dump(out_dir, records):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_results.json"), "w") as f:
        json.dump(records, f, indent=2, sort_keys=True)
    write_summary(out_dir, records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=10_000_000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--quants", default="sq4,sq8,")
    ap.add_argument("--factors", default="1,2,4,8")
    ap.add_argument("--settle", default="compact,no-compact",
                    help="comma list of settle policies to sweep: compact (VACUUM "
                         "COMPACT_TABLE after build, merged single-segment index) or "
                         "no-compact (leave build segments un-merged)")
    ap.add_argument("--rerank", default="0,2,4,8,16")
    ap.add_argument("--clients", default="32")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--no-gt-file", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(VEC_ANN, "results/t2i_nlist_factor_quant"))
    ap.add_argument("--datadir-root", default=os.path.join(VEC_ANN, "build_nlist_quant_data"))
    ap.add_argument("--workdir", default=os.path.join(VEC_ANN, "build_nlist_sweep_work"))
    args = ap.parse_args()

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    factors = [int(f) if f.is_integer() else f for f in factors]
    settles = [s.strip() for s in args.settle.split(",") if s.strip()]
    rerank = [int(x) for x in args.rerank.split(",") if x.strip()]
    clients = [int(x) for x in args.clients.split(",") if x.strip()]

    gt_file = None if args.no_gt_file else GT_FILE
    print(f"loading t2i nb={args.nb} nq={args.nq} k={args.k} gt_file={gt_file}", flush=True)
    ds = dataset.load_t2i(DATA_DIR, nb=args.nb, nq=args.nq, k=args.k, gt_file=gt_file)
    print(f"dataset={ds.name} nb={ds.nb} nq={ds.nq} dim={ds.dim} metric={ds.metric}", flush=True)

    records = []
    for quant in quants:
        for factor in factors:
            nlist = resolved_nlist(factor, ds.nb)
            nprobe = nprobe_grid(nlist)
            for settle in settles:
                print(f"\n=== quant={quant} nlist_factor={factor} (nlist~{nlist}) "
                      f"settle={settle} nprobe={nprobe} rerank={rerank} "
                      f"clients={clients} ===", flush=True)
                datadir = os.path.join(args.datadir_root, f"{quant}_factor_{factor}_{settle}")
                got = False
                for attempt in (1, 2):
                    shutil.rmtree(datadir, ignore_errors=True)
                    server = Server(datadir, binary=BINARY, keep_datadir=False)
                    try:
                        server.start()
                        print(f"  serened pid={server.proc.pid} port={server.port} "
                              f"(attempt {attempt})", flush=True)
                        build = runner.build_local(server, ds, quant, nlist_factor=factor,
                                                   load_via="parquet", settle=settle,
                                                   workdir=args.workdir)
                        print(f"  build: total={build['build_total_s']:.1f}s "
                              f"index={build['index_build_s']:.1f}s "
                              f"compact={build.get('compact_s') or 0:.1f}s "
                              f"index_disk={build['index_disk_bytes'] / 1e6:.0f}MB "
                              f"ram={build['ram_peak_build_mb']:.0f}MB rows={build['rows']}", flush=True)
                        qres = runner.run_queries(server, ds, build["target"], k=args.k,
                                                  nprobe_list=nprobe, rerank_factor_list=rerank,
                                                  clients_list=clients, warmup=args.warmup)
                        for r in qres:
                            rec = {"quant": quant, "nlist_factor": factor, "nlist": nlist,
                                   "source": "local", "settle": settle, "dataset": ds.name,
                                   "dim": ds.dim, "nb": ds.nb}
                            for f in BUILD_FIELDS:
                                rec[f] = build.get(f)
                            rec.update(r)
                            records.append(rec)
                        dump(args.out_dir, records)
                        print(f"  [checkpoint] wrote {len(records)} rows so far", flush=True)
                        got = True
                    except Exception as e:
                        print(f"  [WARN] {quant} factor={factor} settle={settle} "
                              f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                    finally:
                        try:
                            server.stop()
                        except Exception as e:
                            print(f"  [WARN] server.stop() raised: {type(e).__name__}: {e}",
                                  flush=True)
                        shutil.rmtree(datadir, ignore_errors=True)
                    if got:
                        break
                if not got:
                    print(f"  [SKIP] {quant} factor={factor} settle={settle}: "
                          f"no result after retries", flush=True)

    dump(args.out_dir, records)
    print(f"\nDONE. wrote {args.out_dir}/all_results.json, summary.csv, summary.md", flush=True)


if __name__ == "__main__":
    main()
