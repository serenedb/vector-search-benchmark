"""Qdrant engine: standalone `qdrant` binary (no Docker) + qdrant-client.

HNSW index; Text2Image is max-inner-product so we use Distance.DOT. Search knob
is `hnsw_ef`. The binary is configured entirely via QDRANT__* env vars.
"""

import os
import shutil
import signal
import subprocess
import time

from .base import Engine, QuerySession, free_port

DEFAULT_BINARY = os.path.expanduser("~/.cache/vecbench/qdrant/qdrant")
COLL = "bench"


class _QdrantSession(QuerySession):
    def __init__(self, host, grpc_port, hnsw_ef, k, oversampling=None, rescore=True):
        from qdrant_client import QdrantClient, models
        self.client = QdrantClient(host=host, grpc_port=grpc_port, prefer_grpc=True, timeout=120, check_compatibility=False)
        quant = None
        if oversampling is not None:
            quant = models.QuantizationSearchParams(
                ignore=False, rescore=rescore, oversampling=float(oversampling))
        self.sp = models.SearchParams(hnsw_ef=int(hnsw_ef), exact=False, quantization=quant)

    def query_one(self, vec, k):
        res = self.client.query_points(COLL, query=[float(x) for x in vec], limit=k,
                                       search_params=self.sp, with_payload=False)
        return [p.id for p in res.points]

    def close(self):
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass


class QdrantEngine(Engine):
    name = "qdrant"
    search_param_name = "hnsw_ef"

    def __init__(self, data_dir, binary=None, m=16, ef_construct=100, upload_batch=1024,
                 quant="none", quant_quantile=0.99, quant_always_ram=True,
                 upload_parallel=1):
        super().__init__(data_dir)
        self.binary = binary or DEFAULT_BINARY
        self.m = m
        self.ef_construct = ef_construct
        self.upload_batch = upload_batch
        self.quant = quant
        self.quant_quantile = quant_quantile
        self.quant_always_ram = quant_always_ram
        self.upload_parallel = upload_parallel
        # Search-time quantization knobs; the driver sets these between query
        # sub-sweeps so oversampling/rescore can vary without a rebuild.
        self.oversampling = None
        self.rescore = True
        self.client = None

    def start(self):
        from qdrant_client import QdrantClient
        if not os.path.exists(self.binary):
            raise FileNotFoundError(
                f"qdrant binary not found at {self.binary}; download it from "
                f"https://github.com/qdrant/qdrant/releases (see README).")
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.http_port = free_port()
        self.grpc_port = free_port()
        env = dict(os.environ,
                   QDRANT__SERVICE__HTTP_PORT=str(self.http_port),
                   QDRANT__SERVICE__GRPC_PORT=str(self.grpc_port),
                   QDRANT__STORAGE__STORAGE_PATH=os.path.join(self.data_dir, "storage"),
                   QDRANT__STORAGE__SNAPSHOTS_PATH=os.path.join(self.data_dir, "snapshots"),
                   QDRANT__TELEMETRY_DISABLED="true")
        self._logf = open(self.data_dir + ".log", "w")
        self.proc = subprocess.Popen([self.binary], env=env, cwd=os.path.dirname(self.binary),
                                     stdout=self._logf, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid)
        self._wait_ready()
        self.client = QdrantClient(host="127.0.0.1", grpc_port=self.grpc_port,
                                   prefer_grpc=True, timeout=120, check_compatibility=False)
        return self

    def _wait_ready(self, timeout=60):
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("qdrant exited during startup; see log")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.http_port}/healthz", timeout=2).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.3)
        raise TimeoutError("qdrant not ready")

    def build(self, ids, vectors, dim, metric, log=print):
        from qdrant_client import models
        dist = {"ip": models.Distance.DOT, "l2": models.Distance.EUCLID,
                "cosine": models.Distance.COSINE}[metric]
        quant_cfg = None
        if self.quant == "scalar":
            quant_cfg = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=self.quant_quantile,
                    always_ram=self.quant_always_ram))
        elif self.quant not in ("none", None):
            raise ValueError(f"unsupported qdrant quant {self.quant!r} (none|scalar)")
        try:
            self.client.delete_collection(COLL)
        except Exception:  # noqa: BLE001
            pass
        self.client.create_collection(
            COLL,
            vectors_config=models.VectorParams(
                size=dim, distance=dist,
                hnsw_config=models.HnswConfigDiff(m=self.m, ef_construct=self.ef_construct)),
            quantization_config=quant_cfg,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
        n = len(vectors)
        t0 = time.perf_counter()
        log(f"  [qdrant] uploading {n} vectors (HNSW m={self.m} ef_construct={self.ef_construct} "
            f"quant={self.quant} parallel={self.upload_parallel}) ...")
        self.client.upload_collection(COLL, vectors=vectors, ids=list(range(n)),
                                      batch_size=self.upload_batch,
                                      parallel=self.upload_parallel, wait=True)
        log("  [qdrant] waiting for HNSW indexing to finish ...")
        self._wait_indexed(n, log)
        return time.perf_counter() - t0

    def _wait_indexed(self, n, log, timeout=7200):
        deadline = time.time() + timeout
        last, stable = -1, 0
        while time.time() < deadline:
            info = self.client.get_collection(COLL)
            status = str(info.status).lower()
            indexed = info.indexed_vectors_count or 0
            if "green" in status and indexed >= n:
                return
            if "green" in status:  # small sets can stay indexed==0 (exact); break when stable
                stable = stable + 1 if indexed == last else 0
                if stable >= 3:
                    return
            last = indexed
            time.sleep(1.0)
        log("    (qdrant indexing wait timed out)")

    def new_session(self, search_param):
        return _QdrantSession("127.0.0.1", self.grpc_port, search_param, None,
                              oversampling=self.oversampling, rescore=self.rescore)

    def stop(self):
        if self.client:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if getattr(self, "_logf", None):
            self._logf.close()
