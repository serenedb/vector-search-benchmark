# Vector-search benchmark — Yandex Text-to-Image

End-to-end ANN benchmark for SereneDB's IVF vector index, built on the Yandex
**Text-to-Image** dataset (the max-inner-product dataset from
[big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks), so
Qdrant and other systems already have published numbers on it).

For each **quantization** scenario and each `sdb_nprobe` setting it reports:

- **index build time** (load + IVF construction),
- **on-disk index size**,
- **peak RAM** during build and during querying (serened process RSS),
- **query latency** (mean / p50 / p95 / p99) and **QPS**,
- **recall@k** vs. the dataset ground truth.

It builds the index over three kinds of source — **local** files, a remote
**Iceberg** table, and a remote **HuggingFace `hf://`** parquet — plus an
`http` stand-in and a plain local-`file` source.

## How it maps to SereneDB

| benchmark concept | SereneDB |
|---|---|
| vector column (dim 200) | `emb FLOAT[200]` |
| index + quantization | `CREATE INDEX i ON t USING inverted(id, emb ivf (metric='ip', quant='sq8', nlist=…))` |
| quant scenarios | `quant = 'none' | 'sq8' | 'sq4' | 'pq' | 'rabitq'` (T2I uses `metric='ip'`) |
| k-NN (max inner product) | `SELECT id FROM i ORDER BY emb <#> $q LIMIT k` |
| recall knob | `SET sdb_nprobe = N` |
| index over remote data | `CREATE VIEW v AS SELECT id, emb::FLOAT[200] FROM read_parquet('hf://…')` / `iceberg_scan(…)`, then index `v` |

Query vectors are sent as **binary-bound prepared-statement parameters**, not
inlined as text — at dim 200 an inlined literal adds ~5 ms/query of parse time.

## Prerequisites

1. **A `serened` binary.** This repo doesn't vendor the SereneDB source —
   build one from a SereneDB checkout with the bench preset:
   ```bash
   cmake --preset bench && ninja -C build_bench serened   # produces build_bench/bin/serened
   ```
   Point the harness at it either by putting it on `PATH` as `serened`, or
   with `--binary /path/to/build_bench/bin/serened` (most scripts take this
   flag directly; the `profiling/*.sh` wrappers use the `PERF_SERENED_BIN`
   env var — see [Profiling scripts](#profiling-scripts) below).
2. **Python 3.10+** with a venv (on Debian/Ubuntu you may need
   `sudo apt-get install -y python3-pip python3-venv` first):
   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   ```

Run the scripts from this repo's root.

## Quick start (offline smoke, synthetic vectors)

```bash
python run_all.py --dataset synthetic --nb 200000 --nq 1000 --dim 200 \
    --sources local,iceberg,http --quant none,sq8,sq4,pq,rabitq --nprobe 8,32,128
# -> results/summary.md, results/summary.csv, results/all_results.json
```

Sweep concurrency levels (sequential vs. parallel clients) at a fixed nprobe:

```bash
python run_all.py --dataset synthetic --nb 200000 --nq 1000 --dim 200 \
    --sources local --quant none --nprobe 32 --clients 1,8,32
```

## Full run (Text-to-Image, default 10M)

### 1. Get the dataset

Fetch the T2I `.fbin`/`.ibin` files with big-ann-benchmarks (recommended, also
used by the Qdrant comparison) into a directory:

```bash
git clone https://github.com/harsha-simhadri/big-ann-benchmarks && cd big-ann-benchmarks
# NOTE: its pinned requirements.txt is bit-rotted and fails to build on Python >=3.11.
# Install modern wheels instead (download needs no Docker):
pip install numpy h5py pyyaml ansicolors docker matplotlib scikit-learn pandas psutil
python create_dataset.py --dataset text2image-10M    # downloads base/queries/GT
```

If you only want the data (not the Qdrant comparison), any tool that writes the
T2I `.fbin`/`.ibin` files into `--data-dir` works — the harness doesn't depend on
big-ann-benchmarks at runtime.

Point `--data-dir` at the folder holding the files. The loader auto-detects
`base*.fbin`, `query*.fbin`, and `groundtruth*.ibin` (override with
`--base-file/--query-file/--gt-file`). If no ground-truth file matches the
chosen `--nb`, exact max-IP ground truth is computed once and cached next to the
data as `gt_cache_*.npy`.

### 2. Run the matrix

```bash
python run_all.py --dataset t2i --data-dir /path/to/t2i --nb 10000000 \
    --sources local,iceberg --quant none,sq8,sq4,pq,rabitq \
    --nprobe 8,16,32,64,128,256

# RaBitQ defaults to 1 bit/dim; sweep more bits for a recall/size trade-off
# by rerunning quant=rabitq with a different --rabitq-bits (e.g. 3, 5).
```

The `run_all.py` matrix uses one `--rabitq-bits` for the whole run; to compare
several bit widths, run `quant=rabitq` once per width. `build_index.py` /
`build_index_remote.py` take the same `--rabitq-bits` flag for one-off builds.

## Scripts

- **`run_all.py`** — the whole matrix `{sources} × {quant} × {nprobe}`, one fresh
  serened per (source, quant); writes `results/{all_results.json,summary.csv,summary.md}`.
- **`build_index.py`** / **`query_index.py`** — the explicit two-phase local flow:
  `build_index.py` prepares the index and leaves serened running with a manifest;
  `query_index.py` attaches to it and measures recall/latency (`--stop` to shut down).
- **`build_index_remote.py`** — build over a remote source and query it in one
  process (so the source stays alive): `--source {iceberg,hf,http,file}`.

## Profiling scripts

`profiling/` wraps the harness with `perf record` / a real-embedding workload
instead of running the plain matrix:

- **`profile_ivf_build.sh`** — profiles `build_index.py`'s load + `CREATE
  INDEX` path on T2I, leaving `serened` running afterward.
- **`profile_ivf_query.sh`** — attaches to the `serened` left running by
  `profile_ivf_build.sh` (via its `manifest.json`) and profiles a
  `query_index.py` nprobe × clients sweep.

Both scripts read config from env vars (see each script's header comment) and
neither builds `serened` — set `PERF_SERENED_BIN` (or put `serened` on `PATH`)
per the [Prerequisites](#prerequisites) above. They additionally need:

- `perf` (`sudo apt install linux-tools-common linux-tools-generic`) with
  `kernel.perf_event_paranoid <= 1`,
- a T2I data directory (`PERF_T2I_DIR`, default `../big-ann-benchmarks/data/text2image1B`
  next to this repo — see [Get the dataset](#1-get-the-dataset)),
- optionally, [FlameGraph](https://github.com/brendangregg/FlameGraph) on
  `PATH` (or cloned to `.flamegraph-tools/FlameGraph` next to this repo) for
  SVG output — it's a third-party tool, not vendored here.

```bash
profiling/profile_ivf_build.sh                                          # nb=1e6, quant=sq4
PERF_MANIFEST=results/ivf-build-*/manifest.json profiling/profile_ivf_query.sh
```

### Remote sources

The index is built by **scanning the source at `CREATE INDEX` time**; only the
index lives locally (row data stays remote).

```bash
# Iceberg (fully local, recall-measured): embeddings stored as FLOAT[] LIST
python build_index_remote.py --source iceberg --dataset t2i --data-dir /path/to/t2i \
    --nb 1000000 --quant sq8 --nprobe 8,32,128

# HuggingFace: upload the prepared parquet to your repo (needs HF_TOKEN + huggingface_hub)
HF_TOKEN=... python build_index_remote.py --source hf --hf-repo you/t2i-1m \
    --dataset t2i --data-dir /path/to/t2i --nb 1000000 --quant sq8
# ...or index an existing public hf:// parquet directly:
python build_index_remote.py --source hf \
    --hf-uri 'hf://datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K/**/*.parquet' \
    --dataset parquet --parquet-path /local/copy.parquet   # local copy is only for ground truth
```

## Quantization scenarios

| `quant` | what | needs |
|---|---|---|
| `none` | full float32 vectors | — |
| `sq8` | 8-bit scalar quantization | metric l2/ip |
| `sq4` | 4-bit scalar quantization | metric l2/ip |
| `pq`  | product quantization (`pq_m` subquantizers, must divide dim) | metric l2/ip |
| `rabitq` | RaBitQ binary quantization (`rabitq_bits` bits/dim, 1-9, default 1) | metric l2/ip |

## Results schema

`summary.csv` / `all_results.json` — one record per `(source, quant, nprobe, rerank_factor, clients)`:

```
source, quant, dataset, dim, nb, rows,
load_s, index_build_s, build_total_s, index_disk_bytes, datadir_bytes,
ram_peak_build_mb, remote_prep_s,
nprobe, rerank_factor, clients, k, recall_at_k, qps,
lat_ms_mean, lat_ms_p50, lat_ms_p95, lat_ms_p99, lat_ms_min, lat_ms_max,
ram_peak_query_mb, n_queries
```

`summary.md` renders a build-cost table and a recall/throughput table.

## Build time & the `--settle` policy

`index_build_s` is the **time to a queryable index** (`CREATE INDEX` + `VACUUM
REFRESH`); `load_s` is the ingest; `build_total_s = load_s + index_build_s`.
`CREATE INDEX` on a populated table is where the heavy, currently single-threaded
build happens — it *is* included.

After the index is queryable, serened keeps compacting index segments in the
background. `--settle` controls how the harness handles that before timing queries
(reported separately as `compact_s`, not folded into `build_total_s`):

- `compact` (default) — create the index `WITH (compaction_interval = 0,
  refresh_interval = 0)` to **disable the background refresh/compaction loops**,
  then run `VACUUM (COMPACT_TABLE)` to merge segments. Slower build (pays the
  merge), faster search, deterministic settled size. Merge cost is reported
  separately as `compact_s` (not folded into `build_total_s`).
- `no-compact` — same background loops off, but **skip the final compaction** so
  the segments produced during loading are left un-merged. **Faster build, more
  segments, typically slower search.** `compact` vs `no-compact` is a clean A/B on
  the merge cost (identical except the final `VACUUM COMPACT`). Note the effect
  only shows when loading actually produced multiple segments (large `nb`); a
  small single-shot load is one segment, so the two modes coincide.
- `wait` — leave the background loops on and poll until the datadir stops changing
  (the engine's own steady state).
- `none` — leave the background loops on and query immediately; fastest, but
  query latency/QPS/RAM overlap background compaction and are less reproducible.

(View-backed remote indexes have no base table, so `compact` degrades to `wait`.)

## Notes & caveats

- **QPS reflects `--clients`** — `--clients 1` (the default) is a single
  connection issuing prepared statements sequentially, a clean per-query
  latency measurement. `--clients N` runs N persistent connections
  concurrently, each continuously firing its next query as soon as the
  previous one returns (closed-loop, like `pgbench -c`); QPS is then
  `total_queries / wall_clock_time` of the timed section, and the latency
  percentiles reflect latency under that concurrent load. Pass a
  comma-separated list (e.g. `--clients 1,8,32`) to sweep concurrency levels
  in one run, same as `--nprobe` and `--rerank-factor`.
- **`--rerank-factor` (`sdb_rerank_factor`) only matters for quantized indexes**
  (`quant` != `none`). It sizes the exact-distance candidate pool used to correct
  the top-k selected by approximate quantized distance: pool = `rerank_factor * k`.
  Higher values trade query latency for recall; `0` disables reranking entirely
  (top-k is picked by the raw quantized distance, cheapest but least accurate).
  Default `4` matches the server default, so a run that doesn't pass
  `--rerank-factor` behaves exactly as before this option existed.
- **Peak RAM** is the serened process RSS high-water mark sampled during each
  phase; because RSS grows with warm caches, the query-phase peak is typically ≥
  the build-phase peak.
- **`sdb_nprobe` is applied per index segment, so recall depends on segment
  count.** IVF creates one segment per parallel scan unit at `CREATE INDEX` time; a
  query probes `nprobe` cells in *each* segment (and reranks a pool per segment),
  then merges. The `local` path consolidates to a single segment via `VACUUM
  (REFRESH_TABLE)`, but a view-backed remote build (`iceberg`/`hf`/`http`) has no
  base table to consolidate and fragments into one segment per parallel scan unit —
  so at the same `nprobe` it searches more cells and reports **higher recall and
  slower queries**, purely from fragmentation, not a real quality difference. To
  keep sources comparable the harness pins **`--build-threads` (default 1)** so
  remote indexes build as a single segment (verified: makes `iceberg` recall match
  `local` exactly). Set `--build-threads N` to study fragmentation deliberately; the
  value is recorded per run (`build_threads` column). This is why comparing engines
  at a fixed `nprobe` is only meaningful at equal segment counts — prefer the
  recall-vs-QPS curve.
- **`index_disk_bytes` is index-only**, read directly from the storage engine's
  own accounting (`sdb_metrics`'s `index_size`, keyed by the index's relation
  id) rather than a datadir-size delta — it does *not* include the base
  table's columnstore copy, WAL, or catalog bytes, so it's comparable across
  `local` and view-backed sources alike. `datadir_bytes` (the whole datadir,
  still a directory-size read) is reported alongside it for anyone who wants
  the full-footprint number instead.
- **Cleanup** is automatic: each run kills its serened and removes scratch
  datadirs. `build_index.py` intentionally leaves serened running — stop it with
  the printed `kill <pid>` or `query_index.py --stop`.

## Comparing against Qdrant and Elasticsearch (in-harness, no Docker)

`compare.py` runs a head-to-head across engines on the **same** T2I base/queries/
ground truth. Each engine runs a standalone server (no Docker); recall, latency,
QPS, build time, on-disk index size, and peak RAM are all measured identically
(process-tree RSS + data-dir size). Qdrant was never part of big-ann-benchmarks,
so this is the path to compare against it; Elasticsearch uses **native
`dense_vector` kNN with `similarity: max_inner_product`** (the elastiknn plugin
has no max-IP model), and SereneDB uses its IVF index.

### One-time setup (standalone servers + clients)

```bash
pip install "qdrant-client" "elasticsearch>=8.15,<9"   # NOTE: ES client must match ES 8.x server major

# Qdrant binary
mkdir -p ~/.cache/vecbench/qdrant && cd ~/.cache/vecbench/qdrant
curl -sSL https://github.com/qdrant/qdrant/releases/download/v1.18.2/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz

# Elasticsearch tarball (bundles its own JDK — no system Java needed)
mkdir -p ~/.cache/vecbench/es && cd ~/.cache/vecbench/es
curl -sSL https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.3-linux-x86_64.tar.gz | tar xz
```
(Override the locations with `--qdrant-binary` / `--es-home` if you put them elsewhere.)

### Run

```bash
python compare.py --dataset t2i --data-dir <t2i> --gt-file <t2i>/text2image-10M \
    --nb 1000000 --nq 10000 --k 10 \
    --engines serenedb,qdrant,elasticsearch \
    --sdb-quant none --hnsw-m 16 --hnsw-ef-construct 100 \
    --search-params 16,32,64,128,256 --clients 1,8 \
    --datadir ~/workspace/vecbench_cmp/data --workdir ~/workspace/vecbench_cmp/work
# -> results/compare_summary.md, compare_summary.csv, compare_results.json
```

### Reading the results — important caveats

- **`--search-params` maps to a *different* knob per engine**: SereneDB `nprobe`,
  Qdrant `hnsw_ef`, Elasticsearch `num_candidates`. They aren't equivalent at the
  same number — compare the **recall-vs-QPS curves**, not points at equal effort.
- **Different algorithms**: SereneDB=IVF(+optional quant), Qdrant/ES=HNSW. Use
  `--sdb-quant none` for the fairest recall comparison against full-precision HNSW.
- **ES RAM includes the reserved JVM heap** (default 8 GB); it's the honest
  resident footprint but dominated by heap sizing, not index size.
- **QPS is concurrent** across `--clients` threads (each a separate connection).
- Synthetic data is unrepresentative (IVF needs cluster structure) — use real T2I.

## License

This repository's own code is licensed under the Apache License, Version 2.0
(see `LICENSE`). The benchmark results in `results/` and `docs/` are derived
from the Yandex Text-to-Image-1B dataset (CC BY 4.0) via big-ann-benchmarks
(MIT) — see `NOTICE` for full attribution.
