"""Common Engine interface for a cross-database ANN comparison.

Every engine (SereneDB, Qdrant, Elasticsearch) runs a standalone server (no
Docker), indexes the SAME base vectors under the SAME 0-based ids, and answers
k-NN over the SAME queries, so recall (vs a shared ground truth), latency, and
QPS are directly comparable. RAM/disk are measured identically for all: process
RSS via common.server.RssSampler + on-disk data-directory size.

Concurrency mirrors the SereneDB harness: a query "session" is one client
connection; the runner spawns `clients` sessions and splits the query load
across them, so QPS reflects concurrent serving.
"""

import socket

from common.server import dir_size_bytes


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QuerySession:
    """One client connection, configured for a given search-effort value."""

    def query_one(self, vec, k):
        """Return the list of neighbor ids for a single query vector."""
        raise NotImplementedError

    def close(self):
        pass


class Engine:
    name = "base"
    # human label for this engine's search-effort knob (nprobe / hnsw_ef / num_candidates)
    search_param_name = "search_param"

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.proc = None

    def start(self):
        """Launch the server, block until ready. Must set self.proc."""
        raise NotImplementedError

    def build(self, ids, vectors, dim, metric, log=print):
        """Index all vectors; return build time (s), blocking until fully queryable."""
        raise NotImplementedError

    def new_session(self, search_param):
        """Return a QuerySession bound to this search-effort value."""
        raise NotImplementedError

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def disk_bytes(self):
        return dir_size_bytes(self.data_dir)

    def stop(self):
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
