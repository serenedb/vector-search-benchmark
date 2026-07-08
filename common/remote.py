"""Publish benchmark vectors to a 'remote' source and expose them as a view.

SereneDB builds a vector index over remote data by scanning the source at
CREATE INDEX time. The supported sources here all reduce to a view of the form
`SELECT id, emb::FLOAT[dim] FROM <reader>('<uri>')`:

  * iceberg - COPY the vectors to a local Iceberg table (embedding written as an
              unbounded FLOAT[] LIST, which Iceberg accepts), read via iceberg_scan.
  * hf      - read_parquet('hf://datasets/<repo>/...'); either upload the prepared
              parquet to your HF repo (needs HF_TOKEN + huggingface_hub) or point
              at an existing public hf:// parquet.
  * http    - serve the parquet from a local HTTP server and read it back over the
              wire via httpfs (offline-friendly stand-in for real object storage).
  * file    - read_parquet('<local path>') (not remote, handy for debugging).
"""

import os
import subprocess
import sys
import time


def ensure_vectors_parquet(path, ids, vectors, fixed=True):
    """Write the parquet only if a valid cache isn't already there.

    A cache is valid only if it matches the current row count AND embedding
    dimension -- otherwise a leftover parquet from a different --nb/--dim run
    would be silently reused (e.g. building over 10M when you asked for 100).
    """
    import pyarrow.parquet as pq
    import numpy as np
    n, dim = np.asarray(vectors).shape
    if os.path.exists(path):
        try:
            pf = pq.ParquetFile(path)
            ok_rows = pf.metadata.num_rows == n
            ftype = pf.schema_arrow.field("emb").type
            ok_dim = getattr(ftype, "list_size", None) in (dim, None)
            if ok_rows and ok_dim:
                return path
        except Exception:  # noqa: BLE001 - corrupt/unn readable cache -> rewrite
            pass
    return write_vectors_parquet(path, ids, vectors, fixed=fixed)


def write_vectors_parquet(path, ids, vectors, fixed=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    if fixed:
        emb = pa.FixedSizeListArray.from_arrays(flat, dim)
    else:
        offsets = pa.array(np.arange(0, (n + 1) * dim, dim, dtype=np.int32))
        emb = pa.ListArray.from_arrays(offsets, flat)
    table = pa.table({"id": pa.array(np.asarray(ids, dtype=np.int64)), "emb": emb})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pq.write_table(table, path)
    return path


class Remote:
    kind = "base"

    def setup(self, cur, dim):
        raise NotImplementedError

    def view_ddl(self, view, dim):
        raise NotImplementedError

    def teardown(self):
        pass


class FileRemote(Remote):
    kind = "file"

    def __init__(self, parquet_path):
        self.parquet_path = os.path.abspath(parquet_path)

    def setup(self, cur, dim):
        return {"uri": self.parquet_path}

    def view_ddl(self, view, dim):
        return (f"CREATE VIEW {view} AS SELECT id, emb::FLOAT[{dim}] AS emb "
                f"FROM read_parquet('{self.parquet_path}')")


class HttpRemote(Remote):
    kind = "http"

    def __init__(self, parquet_path, port=None, host="127.0.0.1"):
        self.parquet_path = os.path.abspath(parquet_path)
        self.dir = os.path.dirname(self.parquet_path)
        self.fname = os.path.basename(self.parquet_path)
        self.host = host
        self.port = port
        self.proc = None

    def setup(self, cur, dim):
        if self.proc is not None:
            return {"uri": self.url}
        from .server import find_free_port
        self.port = self.port or find_free_port()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(self.port), "--bind", self.host],
            cwd=self.dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        self.url = f"http://{self.host}:{self.port}/{self.fname}"
        return {"uri": self.url}

    def view_ddl(self, view, dim):
        return (f"CREATE VIEW {view} AS SELECT id, emb::FLOAT[{dim}] AS emb "
                f"FROM read_parquet('{self.url}')")

    def teardown(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class IcebergRemote(Remote):
    kind = "iceberg"

    def __init__(self, parquet_path, table_dir):
        self.parquet_path = os.path.abspath(parquet_path)
        self.table_dir = os.path.abspath(table_dir)
        self._built = False

    def setup(self, cur, dim):
        if self._built:
            return {"uri": self.table_dir}
        if os.path.isdir(self.table_dir):
            import shutil
            shutil.rmtree(self.table_dir)
        cur.execute(
            f"COPY (SELECT id, CAST(emb AS FLOAT[]) AS emb "
            f"FROM read_parquet('{self.parquet_path}')) "
            f"TO '{self.table_dir}' (FORMAT ICEBERG)")
        self._built = True
        return {"uri": self.table_dir}

    def view_ddl(self, view, dim):
        return (f"CREATE VIEW {view} AS SELECT id, emb::FLOAT[{dim}] AS emb "
                f"FROM iceberg_scan('{self.table_dir}')")


class HfRemote(Remote):
    kind = "hf"

    def __init__(self, parquet_path=None, repo=None, path_in_repo="vectors.parquet",
                 hf_uri=None, id_col="id", emb_col="emb"):
        self.parquet_path = os.path.abspath(parquet_path) if parquet_path else None
        self.repo = repo
        self.path_in_repo = path_in_repo
        self.hf_uri = hf_uri
        self.id_col = id_col
        self.emb_col = emb_col

    def setup(self, cur, dim):
        if self.hf_uri:
            return {"uri": self.hf_uri}
        if not (self.repo and self.parquet_path):
            raise ValueError("hf source needs either --hf-uri (existing) or "
                             "--hf-repo + a prepared parquet to upload")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF upload needs HF_TOKEN in the environment")
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(self.repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=self.parquet_path,
                        path_in_repo=self.path_in_repo,
                        repo_id=self.repo, repo_type="dataset")
        self.hf_uri = f"hf://datasets/{self.repo}/{self.path_in_repo}"
        return {"uri": self.hf_uri}

    def view_ddl(self, view, dim):
        return (f"CREATE VIEW {view} AS SELECT {self.id_col} AS id, "
                f"{self.emb_col}::FLOAT[{dim}] AS emb "
                f"FROM read_parquet('{self.hf_uri}')")
