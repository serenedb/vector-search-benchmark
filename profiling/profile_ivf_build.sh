#!/usr/bin/env bash
# Profile the IVF vector-index load + CREATE INDEX path with perf, driving the
# two-phase harness (build_index.py / query_index.py) one level up instead of
# reimplementing T2I loading and IVF DDL construction here.
#
# - Launches build_index.py in the background; it starts serened and prints
#   its pid before doing any load/build work.
# - Once the pid is known, attaches `perf record --pid` to it for the rest of
#   build_index.py's run (load + CREATE INDEX + VACUUM REFRESH, gated by
#   --settle -- 'no-compact' by default here, matching a compare.py A/B run).
# - build_index.py leaves serened running afterward (its own detach()
#   behavior) so profile_ivf_query.sh can attach to the same server.
#
# Pre-reqs:
#   sudo apt install linux-tools-common linux-tools-generic
#   sudo sysctl kernel.perf_event_paranoid=1   (or run this script under sudo)
#   FlameGraph (stackcollapse-perf.pl + flamegraph.pl) on PATH for SVG output;
#   .flamegraph-tools/FlameGraph next to this repo is picked up automatically
#   if present -- otherwise clone https://github.com/brendangregg/FlameGraph
#   and put it on PATH yourself.
#
# Tunables:
#   PERF_T2I_DIR      T2I .fbin/.ibin directory. Deliberately NOT under
#                     $HOME/data like the other profile_*.sh scripts: T2I is
#                     multi-GB and $HOME may be a small root partition here.
#                     Defaults to a big-ann-benchmarks checkout next to the repo.
#   PERF_GT_FILE      explicit ground-truth file (passed through as --gt-file)
#   PERF_NB           base vectors to index (default 1,000,000 -- pass
#                     10000000 to match a full T2I-10M compare.py run, but
#                     expect a much longer single-threaded CREATE INDEX)
#   PERF_NQ           query vectors loaded alongside (default 10,000; not
#                     exercised by the build phase but required by the CLI)
#   PERF_K            neighbors per query (default 10)
#   PERF_QUANT        IVF quant scenario: none|sq8|sq4|pq|rabitq (default sq4)
#   PERF_SETTLE       compact|no-compact|wait|none (default no-compact)
#   PERF_LOAD_VIA     copy|parquet (default parquet)
#   PERF_BUILD_DIR    serened build dir -- if set, uses ${PERF_BUILD_DIR}/bin/serened
#   PERF_SERENED_BIN  explicit serened binary path/name (default: `serened` on
#                     PATH, unless PERF_BUILD_DIR is set); build one from a
#                     serenedb checkout: cmake --preset perf && ninja -C
#                     <build-dir> serened
#   PERF_FREQ         perf sample rate (default 199)
#   PERF_CALL_GRAPH   fp (default; perf preset preserves frame pointers) | dwarf
#   PERF_PYTHON       python interpreter with psycopg/numpy/pyarrow
#                     (defaults to ~/.venvs/vecbench/bin/python if present)
#   PERF_DATA_DIR     serened scratch datadir (default under the repo root,
#                     never /tmp -- keep large scratch off a small root fs)
#   PERF_WORK_DIR     parquet staging dir for --load-via parquet

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

T2I_DIR="${PERF_T2I_DIR:-${ROOT}/big-ann-benchmarks/data/text2image1B}"
NB="${PERF_NB:-1000000}"
# The published text2image-10M ground truth only matches the full 10M-vector
# slice; for a smaller --nb the harness computes (and caches) exact GT itself
# unless a matching --gt-file is given, so only default it at nb=10M.
GT_FILE="${PERF_GT_FILE:-}"
if [[ -z "${GT_FILE}" && "${NB}" == "10000000" ]]; then
	GT_FILE="${T2I_DIR}/text2image-10M"
fi
NQ="${PERF_NQ:-10000}"
K="${PERF_K:-10}"
QUANT="${PERF_QUANT:-sq4}"
SETTLE="${PERF_SETTLE:-no-compact}"
LOAD_VIA="${PERF_LOAD_VIA:-parquet}"
BUILD_DIR="${PERF_BUILD_DIR:-}"
if [[ -n "${BUILD_DIR}" ]]; then
	SERENED_BIN="${PERF_SERENED_BIN:-${BUILD_DIR}/bin/serened}"
else
	SERENED_BIN="${PERF_SERENED_BIN:-serened}"
fi
FREQ="${PERF_FREQ:-199}"
DATA_DIR="${PERF_DATA_DIR:-${ROOT}/build_perf_ivf_data}"
WORK_DIR="${PERF_WORK_DIR:-${ROOT}/build_perf_ivf_work}"
RESULTS_DIR="${ROOT}/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RESULTS_DIR}/ivf-build-${STAMP}"
PERF_DATA="${OUT_DIR}/perf.data"
VECTOR_ANN_DIR="${ROOT}"

PYTHON="${PERF_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
	if [[ -x "${HOME}/.venvs/vecbench/bin/python" ]]; then
		PYTHON="${HOME}/.venvs/vecbench/bin/python"
	else
		PYTHON="python3"
	fi
fi

if [[ ! -d "${T2I_DIR}" ]]; then
	echo "missing T2I data dir ${T2I_DIR} -- see README.md to fetch it" >&2
	exit 1
fi
if ! SERENED_BIN="$(command -v "${SERENED_BIN}")"; then
	echo "missing serened binary '${SERENED_BIN}'" >&2
	echo "  build one from a serenedb checkout: cmake --preset perf && ninja -C <build-dir> serened" >&2
	echo "  then set PERF_SERENED_BIN=/path/to/<build-dir>/bin/serened (or PERF_BUILD_DIR=<build-dir>)" >&2
	exit 1
fi
if ! command -v perf >/dev/null 2>&1; then
	echo "perf not found -- install linux-tools" >&2
	exit 1
fi
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)"
if [[ "${PARANOID}" -gt 1 ]]; then
	echo "kernel.perf_event_paranoid=${PARANOID} -- unprivileged perf record will fail." >&2
	echo "  sudo sysctl kernel.perf_event_paranoid=1   (or run this script under sudo)" >&2
	exit 1
fi

mkdir -p "${OUT_DIR}"
rm -rf "${DATA_DIR}" "${WORK_DIR}"

GT_ARGS=()
[[ -n "${GT_FILE}" ]] && GT_ARGS=(--gt-file "${GT_FILE}")

echo "starting build_index.py (nb=${NB} quant=${QUANT} settle=${SETTLE} load-via=${LOAD_VIA})"
"${PYTHON}" -u "${VECTOR_ANN_DIR}/build_index.py" \
	--binary "${SERENED_BIN}" \
	--dataset t2i --data-dir "${T2I_DIR}" "${GT_ARGS[@]}" \
	--nb "${NB}" --nq "${NQ}" --k "${K}" \
	--quant "${QUANT}" --settle "${SETTLE}" --load-via "${LOAD_VIA}" \
	--datadir "${DATA_DIR}" --workdir "${WORK_DIR}" \
	--manifest "${OUT_DIR}/manifest.json" \
	>"${OUT_DIR}/build_index.log" 2>&1 &
DRIVER_PID=$!

echo "waiting for serened to start (build_index.py pid=${DRIVER_PID}) ..."
# cli.load_dataset() runs before Server.start() and, for a large --nq without
# a matching gt_cache_*.npy on disk, computes exact ground truth by brute
# force client-side -- this can take several minutes for nb in the millions,
# well before serened is even spawned. Give it generous headroom (~20min)
# rather than timing out mid-computation.
SERVER_PID=""
for _ in $(seq 1 6000); do
	if line=$(grep -m1 'serened pid=' "${OUT_DIR}/build_index.log" 2>/dev/null); then
		SERVER_PID="$(sed -n 's/.*serened pid=\([0-9]\+\).*/\1/p' <<<"${line}")"
		[[ -n "${SERVER_PID}" ]] && break
	fi
	if ! kill -0 "${DRIVER_PID}" 2>/dev/null; then
		echo "build_index.py exited before starting serened; see ${OUT_DIR}/build_index.log" >&2
		cat "${OUT_DIR}/build_index.log" >&2
		exit 1
	fi
	sleep 0.2
done
if [[ -z "${SERVER_PID}" ]]; then
	echo "timed out waiting for serened pid; see ${OUT_DIR}/build_index.log" >&2
	exit 1
fi
echo "serened pid=${SERVER_PID} -- attaching perf record (freq=${FREQ})"

perf record -F "${FREQ}" -g --call-graph "${PERF_CALL_GRAPH:-fp}" \
	--pid "${SERVER_PID}" --output "${PERF_DATA}" \
	>"${OUT_DIR}/perf_record.log" 2>&1 &
PERF_PID=$!
sleep 0.2

echo "profiling load + CREATE INDEX (this can take a while for large --nb) ..."
set +e
wait "${DRIVER_PID}"
DRIVER_STATUS=$?
set -e

kill -INT "${PERF_PID}" 2>/dev/null || true
wait "${PERF_PID}" 2>/dev/null || true

if [[ "${DRIVER_STATUS}" -ne 0 ]]; then
	echo "build_index.py exited with status ${DRIVER_STATUS}; see ${OUT_DIR}/build_index.log" >&2
	tail -40 "${OUT_DIR}/build_index.log" >&2
	exit "${DRIVER_STATUS}"
fi

cat "${OUT_DIR}/build_index.log"

echo
echo "=== top symbols (self time) ==="
# `head` closing the pipe early SIGPIPEs perf report; under pipefail that
# reads as a pipeline failure, so trailing `|| true` keeps set -e from
# aborting on a truncation that's expected, not an error.
(perf report --no-children --stdio -g none --input "${PERF_DATA}" 2>/dev/null |
	head -40 | tee "${OUT_DIR}/top_symbols.txt") || true

echo
echo "=== top stacks (callee, depth-pruned) ==="
(perf report --stdio -g graph,1.5,callee --input "${PERF_DATA}" 2>/dev/null |
	head -120 | tee "${OUT_DIR}/top_stacks.txt") || true

FLAMEGRAPH_DIR="${ROOT}/.flamegraph-tools/FlameGraph"
if [[ -x "${FLAMEGRAPH_DIR}/flamegraph.pl" ]]; then
	export PATH="${FLAMEGRAPH_DIR}:${PATH}"
fi
if command -v stackcollapse-perf.pl >/dev/null 2>&1 &&
	command -v flamegraph.pl >/dev/null 2>&1; then
	echo
	echo "=== flame graph ==="
	perf script --input "${PERF_DATA}" 2>/dev/null |
		stackcollapse-perf.pl |
		flamegraph.pl --title "IVF build (nb=${NB} quant=${QUANT} settle=${SETTLE})" \
			>"${OUT_DIR}/flamegraph.svg"
	echo "flamegraph: ${OUT_DIR}/flamegraph.svg"
fi

echo
echo "manifest: ${OUT_DIR}/manifest.json (serened pid=${SERVER_PID} left running)"
echo "next:     profile_ivf_query.sh with PERF_MANIFEST=${OUT_DIR}/manifest.json"
echo "results:  ${OUT_DIR}"
