"""Shared build + query routines used by the CLI scripts and orchestrator."""

import os
import threading
import time

from . import metrics, quant, sdb
from .remote import ensure_vectors_parquet


def _phase(server, name):
    if server.sampler:
        server.sampler.start_phase(name)


def _end_phase(server, name):
    if server.sampler:
        server.sampler.end_phase()
        return server.sampler.phase_peak_mb(name)
    return 0.0


def _index_with_opts(settle):
    # Disable the background refresh/compaction loops so all the work happens
    # synchronously inside our timed calls. Used by both 'compact' and
    # 'no-compact' so the ONLY difference between them is the final VACUUM
    # COMPACT (clean A/B on the merge cost). 'wait'/'none' leave the loops on.
    if settle in ("compact", "no-compact"):
        return {"compaction_interval": 0, "refresh_interval": 0}
    return None


def _index_size_bytes(cur, index):
    """Index-only on-disk size in bytes, straight from the storage engine's
    own accounting (sdb_metrics 'index_size'), not a datadir-size delta --
    so it never includes the base-table columnstore copy, WAL, or catalog
    bytes that a whole-datadir diff would pick up."""
    return float(sdb.scalar(
        cur, f"SELECT value FROM sdb_metrics "
             f"WHERE metric = 'index_size' AND relation_id = '{index}'::regclass::BIGINT"))


def _wait_quiescent(server, poll=2.0, stable_needed=3, timeout=3600, log=print):
    """Poll the datadir size until it stops changing (background work settled)."""
    last, stable, start = -1, 0, time.perf_counter()
    while time.perf_counter() - start < timeout:
        size = server.datadir_bytes()
        if size == last:
            stable += 1
            if stable >= stable_needed:
                return
        else:
            stable, last = 0, size
        time.sleep(poll)
    log("    (quiescence wait timed out)")


def _settle(server, cur, table, policy, log):
    """Bring the index to a defined steady state before querying.

    'compact' forces VACUUM COMPACT (needs a base table); 'wait' polls until
    background compaction stops changing the datadir; 'none' does neither.
    Returns (seconds spent settling, cursor to use afterward).
    """
    if policy in ("none", "no-compact"):
        # Leave the segments produced during loading un-merged (fast build,
        # more segments -> typically slower search).
        return 0.0, cur
    t = time.perf_counter()
    if policy == "compact" and table is not None:
        log("    settling: VACUUM (COMPACT_TABLE) ...")
        sdb.execute(cur, f"VACUUM (COMPACT_TABLE) {table}")
        settle_s = time.perf_counter() - t
        # VACUUM (COMPACT_TABLE) merges segments but never unlinks the ones it
        # supersedes -- only the background refresh loop's cleanup pass does
        # that, and it's disabled here (compaction_interval=0/refresh_interval=0)
        # for clean timing. Restart so the index-open path sweeps them instead,
        # otherwise datadir_bytes() below double-counts old + merged segments.
        cur.close()
        log("    settling: restarting server to reclaim superseded segments ...")
        server.restart()
        return settle_s, server.connect().cursor()
    log("    settling: waiting for background compaction to quiesce ...")
    _wait_quiescent(server, log=log)
    return time.perf_counter() - t, cur


def build_local(server, ds, scenario, *, nlist=None, nlist_factor=None,
                train_sample=None, pq_m=None,
                rabitq_bits=None, table="vec", index="vec_idx", load_via="copy",
                workdir=None, settle="compact", log=print):
    dim = ds.dim
    cur = server.connect().cursor()
    sdb.execute(cur, f"DROP TABLE IF EXISTS {table} CASCADE")

    _phase(server, "build")
    t0 = time.perf_counter()
    if load_via == "parquet":
        if not workdir:
            raise ValueError("load_via='parquet' needs a workdir")
        os.makedirs(workdir, exist_ok=True)
        pq_path = os.path.join(workdir, "local_vectors.parquet")
        log(f"  [{scenario}] ensuring parquet ({ds.nb} vectors) -> {pq_path} ...")
        ensure_vectors_parquet(pq_path, ds.ids(), ds.base, fixed=True)
        log(f"  [{scenario}] loading via CREATE TABLE AS SELECT read_parquet ...")
        sdb.execute(cur, f"CREATE TABLE {table} AS "
                         f"SELECT id, emb::FLOAT[{dim}] AS emb FROM read_parquet('{pq_path}')")
    else:
        sdb.execute(cur, f"CREATE TABLE {table} (id BIGINT, emb FLOAT[{dim}])")
        log(f"  [{scenario}] loading {ds.nb} vectors (dim {dim}) via COPY ...")
        sdb.copy_vectors(cur, table, ds.ids(), ds.base, log=log)
    load_s = time.perf_counter() - t0

    ddl = quant.index_ddl(index, table, "id", "emb", scenario, with_opts=_index_with_opts(settle),
                          nlist=nlist, nlist_factor=nlist_factor,
                          train_sample=train_sample, pq_m=pq_m,
                          rabitq_bits=rabitq_bits, dim=dim)
    log(f"  [{scenario}] building index (CREATE INDEX + REFRESH, single-threaded) ...")
    t1 = time.perf_counter()
    sdb.execute(cur, ddl)                                # heavy build happens here on a populated table
    sdb.execute(cur, f"VACUUM (REFRESH_TABLE) {table}")
    index_s = time.perf_counter() - t1                   # time to a queryable index
    compact_s, cur = _settle(server, cur, table, settle, log)
    ram_build = _end_phase(server, "build")

    disk_after = server.datadir_bytes()                  # measured after settle -> stable size
    index_disk_bytes = _index_size_bytes(cur, index)
    rows = sdb.scalar(cur, f"SELECT count(*) FROM {index}")
    cur.close()
    return {
        "target": index,
        "rows": rows,
        "ddl": ddl,
        "load_s": load_s,
        "index_build_s": index_s,
        "compact_s": compact_s,
        "build_total_s": load_s + index_s,
        "datadir_bytes": disk_after,
        "index_disk_bytes": index_disk_bytes,
        "ram_peak_build_mb": ram_build,
        # VACUUM (REFRESH_TABLE) consolidates the local index to a single segment,
        # so it behaves like a single-threaded (single-segment) build for nprobe.
        "build_threads": 1,
    }


def build_remote(server, ds, scenario, remote, *, nlist=None, nlist_factor=None,
                 train_sample=None,
                 pq_m=None, rabitq_bits=None, view="vec_v", index="vec_v_idx",
                 settle="compact", build_threads=1, log=print):
    dim = ds.dim
    cur = server.connect().cursor()

    log(f"  [{scenario}] preparing remote source ({remote.kind}) ...")
    t_prep = time.perf_counter()
    info = remote.setup(cur, dim)
    prep_s = time.perf_counter() - t_prep

    sdb.execute(cur, f"DROP VIEW IF EXISTS {view} CASCADE")
    sdb.execute(cur, remote.view_ddl(view, dim))
    view_rows = sdb.scalar(cur, f"SELECT count(*) FROM {view}")
    log(f"  [{scenario}] view exposes {view_rows} rows from {info.get('uri')}")

    _phase(server, "build")
    ddl = quant.index_ddl(index, view, "id", "emb", scenario, with_opts=_index_with_opts(settle),
                          nlist=nlist, nlist_factor=nlist_factor,
                          train_sample=train_sample, pq_m=pq_m,
                          rabitq_bits=rabitq_bits, dim=dim)
    # IVF creates one segment per parallel scan unit and sdb_nprobe is applied per
    # segment, so a view-backed build (no base table to VACUUM REFRESH-consolidate)
    # fragments into N segments and searches N*nprobe cells -> inflated recall vs the
    # single-segment local path. Pin build parallelism so segment count is comparable.
    # SET threads is global in DuckDB, so restore it before queries run.
    if build_threads:
        sdb.execute(cur, f"SET threads = {int(build_threads)}")
    log(f"  [{scenario}] building index over remote source "
        f"(CREATE INDEX, build_threads={build_threads}) ...")
    t1 = time.perf_counter()
    sdb.execute(cur, ddl)
    index_s = time.perf_counter() - t1
    if build_threads:
        sdb.execute(cur, "RESET threads")
    # No base table behind a view-backed index, so 'compact' has nothing to
    # VACUUM -> fall back to waiting for quiescence.
    compact_s, cur = _settle(server, cur, None, "wait" if settle == "compact" else settle, log)
    ram_build = _end_phase(server, "build")

    disk_after = server.datadir_bytes()
    index_disk_bytes = _index_size_bytes(cur, index)
    rows = sdb.scalar(cur, f"SELECT count(*) FROM {index}")
    cur.close()
    return {
        "target": index,
        "rows": rows,
        "ddl": ddl,
        "remote_uri": info.get("uri"),
        "remote_prep_s": prep_s,
        "load_s": 0.0,
        "index_build_s": index_s,
        "compact_s": compact_s,
        "build_total_s": index_s,
        "datadir_bytes": disk_after,
        "index_disk_bytes": index_disk_bytes,
        "ram_peak_build_mb": ram_build,
        "build_threads": build_threads,
    }


def _client_run(server, sql, nprobe, rerank_factor, q_lists, nq, warmup, indices, latencies,
                returned, barrier, progress, progress_lock, step, total, t_loop, log, errors):
    try:
        conn = server.connect()
        cur = conn.cursor()
        sdb.set_knob(cur, "sdb_nprobe", nprobe)
        sdb.set_knob(cur, "sdb_rerank_factor", rerank_factor)
        for i in range(warmup):
            ql = q_lists[i % nq]
            cur.execute(sql, (ql,), prepare=True)
            cur.fetchall()

        barrier.wait()

        for idx in indices:
            query_idx = idx % nq
            ql = q_lists[query_idx]
            t0 = time.perf_counter()
            cur.execute(sql, (ql,), prepare=True)
            rows = cur.fetchall()
            latencies[idx] = (time.perf_counter() - t0) * 1000.0
            if idx < nq:
                returned[query_idx] = [r[0] for r in rows]
            with progress_lock:
                progress[0] += 1
                done = progress[0]
            if done % step == 0 and done < total:
                elapsed = time.perf_counter() - t_loop[0]
                rate = done / elapsed if elapsed else 0.0
                log(f"    {done}/{total} queries ({rate:,.0f} q/s, {elapsed:.1f}s)")
        cur.close()
        conn.close()
    except threading.BrokenBarrierError:
        # another client failed and aborted the barrier -- exit quietly, the
        # failure itself is recorded in `errors` by whichever thread hit it.
        pass
    except Exception as e:  # noqa: BLE001 - surface to the main thread and
        # unblock every other client instead of leaving them stuck in
        # barrier.wait() forever (a thread that errors before reaching the
        # barrier never arrives, so without this the rest hang indefinitely).
        errors.append(e)
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001 - barrier already broken
            pass


def _run_query_combo(server, sql, nprobe, rerank_factor, clients, q_lists, gt, k, warmup,
                     repeats, nq, progress_every, log):
    total = repeats * nq
    step = progress_every or max(200, total // 5)
    latencies = [0.0] * total
    returned = [None] * nq
    progress = [0]
    progress_lock = threading.Lock()
    t_loop = [None]
    t_warm = time.perf_counter()

    def _on_timed_start():
        t_loop[0] = time.perf_counter()
        _phase(server, "query")
        log(f"  [nprobe={nprobe} rerank_factor={rerank_factor} clients={clients}] warmup done "
            f"in {t_loop[0] - t_warm:.1f}s; timing {total} queries ...")

    log(f"  [nprobe={nprobe} rerank_factor={rerank_factor} clients={clients}] warming up "
        f"({warmup} queries/client) ...")
    barrier = threading.Barrier(clients, action=_on_timed_start)
    errors = []

    threads = [
        threading.Thread(target=_client_run, args=(
            server, sql, nprobe, rerank_factor, q_lists, nq, warmup, range(c, total, clients),
            latencies, returned, barrier, progress, progress_lock, step, total, t_loop, log, errors))
        for c in range(clients)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    total_s = time.perf_counter() - t_loop[0]
    ram_query = _end_phase(server, "query")

    recall = metrics.recall_at_k(returned, gt, k)
    qps = total / total_s if total_s > 0 else 0.0
    rec = {
        "nprobe": nprobe,
        "rerank_factor": rerank_factor,
        "clients": clients,
        "k": k,
        "recall_at_k": recall,
        "qps": qps,
        "ram_peak_query_mb": ram_query,
        "n_queries": nq,
    }
    rec.update(metrics.latency_summary(latencies))
    return rec


def run_queries(server, ds, target, *, k=10, nprobe_list=(8, 16, 32, 64, 128),
                rerank_factor_list=(4,), clients_list=(1,), warmup=50, repeats=1,
                progress_every=None, log=print):
    dim = ds.dim
    sql = sdb.knn_sql(target, dim, k=k, binary=True)
    q_lists = [sdb.float_list(q) for q in ds.queries]
    gt = ds.gt_list()
    nq = len(q_lists)
    results = []

    for nprobe in nprobe_list:
        for rerank_factor in rerank_factor_list:
            for clients in clients_list:
                rec = _run_query_combo(server, sql, nprobe, rerank_factor, clients, q_lists,
                                       gt, k, warmup, repeats, nq, progress_every, log)
                results.append(rec)
                log(f"  nprobe={nprobe:<5d} rerank_factor={rerank_factor:<4d} "
                    f"clients={clients:<4d} recall@{k}={rec['recall_at_k']:.4f} "
                    f"qps={rec['qps']:8.1f} p50={rec['lat_ms_p50']:.3f}ms p95={rec['lat_ms_p95']:.3f}ms")
    return results
