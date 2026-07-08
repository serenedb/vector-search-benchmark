#!/usr/bin/env bash
# Profile ANN query serving (IVF search) with perf, attaching to the serened
# left running by profile_ivf_build.sh (via its manifest). Mirrors
# profile_query.sh's attach-by-pid pattern, but drives query_index.py (one
# level up) for the nprobe x clients sweep instead of a single hand-written
# SQL query.
#
# Pre-reqs: same as profile_ivf_build.sh (perf, perf_event_paranoid<=1,
# FlameGraph on PATH for SVG output).
#
# Tunables:
#   PERF_MANIFEST     manifest.json written by profile_ivf_build.sh (required)
#   PERF_T2I_DIR      T2I data dir (must match what built the index)
#   PERF_GT_FILE      ground-truth file (for recall reporting only)
#   PERF_NB / PERF_NQ / PERF_K   must match the build invocation (dataset dim
#                     is checked against the manifest; nb affects the ground
#                     truth cache key)
#   PERF_NPROBE       comma list, sdb_nprobe sweep (default 32,64,128)
#   PERF_CLIENTS      comma list, concurrent clients sweep (default 8,32)
#   PERF_WARMUP       warmup queries per (nprobe,clients) combo (default 50)
#   PERF_STOP         1 (default) to SIGTERM serened when done, 0 to leave it
#                     running (e.g. to attach something else afterward)
#   PERF_FREQ / PERF_CALL_GRAPH / PERF_PYTHON   same meaning as in
#                     profile_ivf_build.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

MANIFEST="${PERF_MANIFEST:-}"
if [[ -z "${MANIFEST}" ]]; then
	echo "PERF_MANIFEST is required -- point it at the manifest.json from profile_ivf_build.sh" >&2
	exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
	echo "missing manifest ${MANIFEST}" >&2
	exit 1
fi

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
NPROBE="${PERF_NPROBE:-32,64,128}"
CLIENTS="${PERF_CLIENTS:-8,32}"
WARMUP="${PERF_WARMUP:-50}"
STOP="${PERF_STOP:-1}"
FREQ="${PERF_FREQ:-199}"
RESULTS_DIR="${ROOT}/results"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RESULTS_DIR}/ivf-query-${STAMP}"
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
	echo "missing T2I data dir ${T2I_DIR}" >&2
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

SERVER_PID="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['pid'])" "${MANIFEST}")"
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
	echo "manifest pid ${SERVER_PID} is not running -- rerun profile_ivf_build.sh" >&2
	exit 1
fi

mkdir -p "${OUT_DIR}"

echo "attaching perf record to serened pid=${SERVER_PID} (freq=${FREQ})"
perf record -F "${FREQ}" -g --call-graph "${PERF_CALL_GRAPH:-fp}" \
	--pid "${SERVER_PID}" --output "${PERF_DATA}" \
	>"${OUT_DIR}/perf_record.log" 2>&1 &
PERF_PID=$!
sleep 0.2

STOP_FLAG=()
[[ "${STOP}" == "1" ]] && STOP_FLAG=(--stop)

GT_ARGS=()
[[ -n "${GT_FILE}" ]] && GT_ARGS=(--gt-file "${GT_FILE}")

echo "running query sweep (nprobe=${NPROBE} clients=${CLIENTS}) ..."
"${PYTHON}" -u "${VECTOR_ANN_DIR}/query_index.py" \
	--manifest "${MANIFEST}" \
	--dataset t2i --data-dir "${T2I_DIR}" "${GT_ARGS[@]}" \
	--nb "${NB}" --nq "${NQ}" --k "${K}" \
	--nprobe "${NPROBE}" --clients "${CLIENTS}" --warmup "${WARMUP}" \
	--out "${OUT_DIR}/query_results.json" \
	"${STOP_FLAG[@]}" \
	2>&1 | tee "${OUT_DIR}/query_index.log"

kill -INT "${PERF_PID}" 2>/dev/null || true
wait "${PERF_PID}" 2>/dev/null || true

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
		flamegraph.pl --title "IVF query (nprobe=${NPROBE} clients=${CLIENTS})" \
			>"${OUT_DIR}/flamegraph.svg"
	echo "flamegraph: ${OUT_DIR}/flamegraph.svg"
fi

echo
echo "results: ${OUT_DIR}"
