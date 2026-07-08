"""Elasticsearch engine: standalone ES (bundled JDK, no Docker) using native
dense_vector kNN.

Text2Image is max-inner-product, so the vector field uses
`similarity: max_inner_product` with an HNSW index. (The elastiknn plugin has no
max-IP model, so native kNN is the correct apples-to-apples choice.) Search knob
is `num_candidates`.
"""

import os
import shutil
import signal
import subprocess
import time

from .base import Engine, QuerySession, free_port

DEFAULT_HOME = os.path.expanduser("~/.cache/vecbench/es/elasticsearch-8.15.3")
INDEX = "bench"
_SIM = {"ip": "max_inner_product", "l2": "l2_norm", "cosine": "cosine"}


class _ESSession(QuerySession):
    def __init__(self, url, num_candidates):
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch(url, request_timeout=120, retry_on_timeout=True)
        self.nc = int(num_candidates)

    def query_one(self, vec, k):
        knn = {"field": "vec", "query_vector": [float(x) for x in vec],
               "k": k, "num_candidates": max(self.nc, k)}
        resp = self.client.search(index=INDEX, size=k, source=False, knn=knn)
        return [int(h["_id"]) for h in resp["hits"]["hits"]]

    def close(self):
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass


class ElasticsearchEngine(Engine):
    name = "elasticsearch"
    search_param_name = "num_candidates"

    def __init__(self, data_dir, home=None, m=16, ef_construct=100, quant="none",
                 heap="8g", bulk_chunk=2000):
        super().__init__(data_dir)
        self.home = home or DEFAULT_HOME
        self.m = m
        self.ef_construct = ef_construct
        self.quant = quant
        self.heap = heap
        self.bulk_chunk = bulk_chunk
        self.url = None
        self.client = None

    def start(self):
        from elasticsearch import Elasticsearch
        exe = os.path.join(self.home, "bin", "elasticsearch")
        if not os.path.exists(exe):
            raise FileNotFoundError(
                f"elasticsearch not found at {exe}; download the tarball from "
                f"https://www.elastic.co/downloads/elasticsearch (see README).")
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.port = free_port()
        env = dict(os.environ, ES_JAVA_OPTS=f"-Xms{self.heap} -Xmx{self.heap}")
        self._logf = open(self.data_dir + ".log", "w")
        self.proc = subprocess.Popen(
            [exe,
             "-E", "cluster.name=vecbench", "-E", "node.name=n1",
             "-E", f"path.data={os.path.join(self.data_dir, 'data')}",
             "-E", f"path.logs={os.path.join(self.data_dir, 'logs')}",
             "-E", "network.host=127.0.0.1", "-E", f"http.port={self.port}",
             "-E", "discovery.type=single-node",
             "-E", "xpack.security.enabled=false",
             "-E", "xpack.security.http.ssl.enabled=false",
             "-E", "xpack.ml.enabled=false",
             "-E", "ingest.geoip.downloader.enabled=false",
             "-E", "bootstrap.memory_lock=false"],
            env=env, stdout=self._logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        self.url = f"http://127.0.0.1:{self.port}"
        self._wait_ready()
        self.client = Elasticsearch(self.url, request_timeout=300, retry_on_timeout=True)
        return self

    def _wait_ready(self, timeout=180):
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"elasticsearch exited during startup; see {self.data_dir}.log")
            try:
                urllib.request.urlopen(f"{self.url}/_cluster/health", timeout=3).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
        raise TimeoutError("elasticsearch not ready")

    def build(self, ids, vectors, dim, metric, log=print):
        from elasticsearch import helpers
        n = len(vectors)
        index_type = {"none": "hnsw", "int8": "int8_hnsw", "int4": "int4_hnsw"}.get(self.quant)
        if index_type is None:
            raise ValueError(f"unsupported elasticsearch quant {self.quant!r} (none|int8|int4)")
        if self.client.indices.exists(index=INDEX):
            self.client.indices.delete(index=INDEX)
        self.client.indices.create(
            index=INDEX,
            settings={"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
            mappings={"properties": {"vec": {
                "type": "dense_vector", "dims": dim, "index": True,
                "similarity": _SIM[metric],
                "index_options": {"type": index_type, "m": self.m,
                                  "ef_construction": self.ef_construct}}}})
        t0 = time.perf_counter()
        log(f"  [elasticsearch] bulk-indexing {n} vectors (HNSW m={self.m} "
            f"ef_construction={self.ef_construct} quant={self.quant}) ...")

        def actions():
            for i in range(n):
                yield {"_index": INDEX, "_id": str(i), "vec": [float(x) for x in vectors[i]]}
        helpers.bulk(self.client, actions(), chunk_size=self.bulk_chunk, request_timeout=300)
        log("  [elasticsearch] refresh + force-merge to 1 segment ...")
        self.client.indices.refresh(index=INDEX)
        self.client.indices.forcemerge(index=INDEX, max_num_segments=1, request_timeout=7200)
        self.client.cluster.health(index=INDEX, wait_for_status="green", timeout="7200s")
        return time.perf_counter() - t0

    def new_session(self, search_param):
        return _ESSession(self.url, search_param)

    def stop(self):
        if self.client:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if getattr(self, "_logf", None):
            self._logf.close()
