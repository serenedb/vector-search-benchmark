"""Engine-agnostic build + concurrent-query measurement.

Drives any Engine through a search-effort x concurrency sweep, measuring recall
(vs shared ground truth), QPS, latency percentiles, and per-phase peak RSS -- the
same methodology used for SereneDB, so cross-engine numbers are comparable.
"""

import threading
import time

from common import metrics
from common.server import RssSampler


def _client_run(session, queries, k, nq, warmup, indices, latencies, returned,
                barrier, progress, progress_lock, step, total, t_loop, log):
    for i in range(warmup):
        session.query_one(queries[i % nq], k)
    barrier.wait()
    for idx in indices:
        qi = idx % nq
        t0 = time.perf_counter()
        ids = session.query_one(queries[qi], k)
        latencies[idx] = (time.perf_counter() - t0) * 1000.0
        if idx < nq:
            returned[qi] = ids
        with progress_lock:
            progress[0] += 1
            done = progress[0]
        if done % step == 0 and done < total:
            elapsed = time.perf_counter() - t_loop[0]
            log(f"    {done}/{total} queries ({done / elapsed:,.0f} q/s, {elapsed:.1f}s)")
    session.close()


def _combo(engine, sampler, sp, clients, queries, gt, k, warmup, log):
    nq = len(queries)
    total = nq
    step = max(200, total // 5)
    latencies = [0.0] * total
    returned = [None] * nq
    progress = [0]
    lock = threading.Lock()
    t_loop = [None]
    phase = f"query_ef{sp}_c{clients}"
    t_warm = time.perf_counter()

    def _on_start():
        t_loop[0] = time.perf_counter()
        sampler.start_phase(phase)
        log(f"  [{engine.search_param_name}={sp} clients={clients}] warmup done in "
            f"{t_loop[0] - t_warm:.1f}s; timing {total} queries ...")

    sessions = [engine.new_session(sp) for _ in range(clients)]
    log(f"  [{engine.search_param_name}={sp} clients={clients}] warming up "
        f"({warmup} queries/client) ...")
    barrier = threading.Barrier(clients, action=_on_start)
    threads = [threading.Thread(target=_client_run, args=(
        sessions[c], queries, k, nq, warmup, range(c, total, clients), latencies,
        returned, barrier, progress, lock, step, total, t_loop, log)) for c in range(clients)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    total_s = time.perf_counter() - t_loop[0]
    sampler.end_phase()

    rec = {
        engine.search_param_name: sp,
        "clients": clients,
        "k": k,
        "recall_at_k": metrics.recall_at_k(returned, gt, k),
        "qps": total / total_s if total_s > 0 else 0.0,
        "ram_peak_query_mb": sampler.phase_peak_mb(phase),
        "n_queries": nq,
    }
    rec.update(metrics.latency_summary(latencies))
    return rec


def run_engine(engine, ds, *, k=10, search_params=(16, 32, 64, 128), clients_list=(1,),
               warmup=50, metric="ip", log=print):
    """Start (assumed already started) engine's build+query sweep; return a dict of
    build metrics + per-(search_param, clients) query records. Caller starts/stops
    the engine and owns the RssSampler lifetime."""
    sampler = RssSampler(engine.pid)
    sampler.start()
    try:
        queries = [list(map(float, q)) for q in ds.queries]
        gt = ds.gt_list()

        sampler.start_phase("build")
        t0 = time.perf_counter()
        build_s = engine.build(ds.ids(), ds.base, ds.dim, metric, log=log)
        sampler.end_phase()
        ram_build = sampler.phase_peak_mb("build")
        disk = engine.disk_bytes()
        log(f"  [{engine.name}] build={build_s:.2f}s disk={disk / 1e6:.1f}MB ram={ram_build:.0f}MB")

        records = []
        for sp in search_params:
            for clients in clients_list:
                rec = _combo(engine, sampler, sp, clients, queries, gt, k, warmup, log)
                records.append(rec)
                log(f"  {engine.search_param_name}={sp:<5d} clients={clients:<3d} "
                    f"recall@{k}={rec['recall_at_k']:.4f} qps={rec['qps']:8.1f} "
                    f"p50={rec['lat_ms_p50']:.3f}ms p95={rec['lat_ms_p95']:.3f}ms")
        return {
            "engine": engine.name,
            "build_s": build_s,
            "index_disk_bytes": disk,
            "ram_peak_build_mb": ram_build,
            "query": records,
        }
    finally:
        sampler.stop()
