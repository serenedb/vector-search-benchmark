"""SereneDB engine: wraps the serened Server + IVF build + a psycopg query
session, exposing the same Engine interface as Qdrant/Elasticsearch so all three
are measured identically. Search knob is `nprobe`.
"""

import os
import time

from common import quant, sdb
from common.remote import ensure_vectors_parquet
from common.server import Server

from .base import Engine, QuerySession


class _SereneSession(QuerySession):
    def __init__(self, port, target, dim, nprobe, rerank_factor=None):
        self.conn = sdb.connect(port)
        self.cur = self.conn.cursor()
        sdb.set_knob(self.cur, "sdb_nprobe", nprobe)
        if rerank_factor is not None:
            sdb.set_knob(self.cur, "sdb_rerank_factor", rerank_factor)
        self.target, self.dim, self._sql = target, dim, None

    def query_one(self, vec, k):
        if self._sql is None:
            self._sql = sdb.knn_sql(self.target, self.dim, k=k, binary=True)
        self.cur.execute(self._sql, ([float(x) for x in vec],), prepare=True)
        return [r[0] for r in self.cur.fetchall()]

    def close(self):
        try:
            self.cur.close()
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


class SereneDBEngine(Engine):
    name = "serenedb"
    search_param_name = "nprobe"

    def __init__(self, data_dir, binary=None, quant_kind="sq8", nlist=None,
                 settle="compact", load_via="parquet", workdir=None, rabitq_bits=None,
                 rerank_factor=None):
        super().__init__(data_dir)
        self.server = Server(data_dir, binary=binary, keep_datadir=False)
        self.quant_kind = quant_kind
        self.rabitq_bits = rabitq_bits
        self.nlist = nlist
        self.settle = settle
        self.load_via = load_via
        self.workdir = workdir or (data_dir + "_work")
        self.rerank_factor = rerank_factor
        self.table, self.index, self._dim = "vec", "vec_idx", None

    def start(self):
        self.server.start()
        self.proc = self.server.proc
        return self

    def build(self, ids, vectors, dim, metric, log=print):
        self._dim = dim
        cur = self.server.connect().cursor()
        t0 = time.perf_counter()
        sdb.execute(cur, f"DROP TABLE IF EXISTS {self.table} CASCADE")
        if self.load_via == "parquet":
            os.makedirs(self.workdir, exist_ok=True)
            pq = os.path.join(self.workdir, "vec.parquet")
            log(f"  [serenedb] ensuring parquet ({len(vectors)} vectors) ...")
            ensure_vectors_parquet(pq, ids, vectors, fixed=True)
            sdb.execute(cur, f"CREATE TABLE {self.table} AS "
                             f"SELECT id, emb::FLOAT[{dim}] AS emb FROM read_parquet('{pq}')")
        else:
            sdb.execute(cur, f"CREATE TABLE {self.table} (id BIGINT, emb FLOAT[{dim}])")
            log(f"  [serenedb] COPY {len(vectors)} vectors ...")
            sdb.copy_vectors(cur, self.table, ids, vectors, log=log)
        if self.settle == "compact":
            wo = {"compaction_interval": 0, "refresh_interval": 200, "cleanup_interval_step": 1}
        elif self.settle == "no-compact":
            wo = {"compaction_interval": 0, "refresh_interval": 0}
        else:
            wo = None
        ddl = quant.index_ddl(self.index, self.table, "id", "emb", self.quant_kind,
                              with_opts=wo, metric=metric, nlist=self.nlist,
                              rabitq_bits=self.rabitq_bits, dim=dim)
        log(f"  [serenedb] CREATE INDEX (ivf quant={self.quant_kind}) + REFRESH ...")
        sdb.execute(cur, ddl)
        sdb.execute(cur, f"VACUUM (REFRESH_TABLE) {self.table}")
        if self.settle == "compact":
            sdb.execute(cur, f"VACUUM (COMPACT_TABLE) {self.table}")
        elapsed = time.perf_counter() - t0
        if self.settle == "compact":
            log("  [serenedb] waiting for background cleanup of superseded segments ...")
            time.sleep(0.5)
        cur.close()
        return elapsed

    def new_session(self, search_param):
        return _SereneSession(self.server.port, self.index, self._dim, search_param,
                              rerank_factor=self.rerank_factor)

    def disk_bytes(self):
        # Index-only size from the storage engine's own accounting, not a
        # datadir directory-size read (Engine.disk_bytes' default) -- the
        # datadir here also holds the base table's columnstore copy.
        cur = self.server.connect().cursor()
        try:
            return sdb.scalar(
                cur, f"SELECT value FROM sdb_metrics "
                     f"WHERE metric = 'index_size' AND relation_id = '{self.index}'::regclass::BIGINT")
        finally:
            cur.close()

    def stop(self):
        self.server.stop()
