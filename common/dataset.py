"""Dataset loading for the vector-ANN benchmark.

Supports:
  * `synthetic` - random gaussian vectors (fast, offline; for smoke tests).
  * `t2i`       - Yandex Text-to-Image in the big-ann-benchmarks binary format
                  (.fbin base/queries, .ibin ground truth). Point --data-dir at
                  a directory that already holds the files (see README for how
                  to fetch them with big-ann-benchmarks or directly).
  * `parquet`   - an existing local parquet of (id, embedding); used to compute
                  ground truth for a "bring your own remote parquet" run.

Vectors are float32; the metric is inner product (max-IP), matching T2I.
Ids are 0-based to line up with big-ann ground-truth ids.
"""

import glob
import os
import numpy as np


def read_fbin(path, count=None, offset=0):
    """Read a big-ann .fbin: int32 nvecs, int32 dim, then float32 data."""
    with open(path, "rb") as f:
        n, dim = np.fromfile(f, dtype=np.int32, count=2)
    n, dim = int(n), int(dim)
    if count is None or count > n - offset:
        count = n - offset
    arr = np.memmap(path, dtype=np.float32, mode="r", offset=8 + offset * dim * 4,
                    shape=(count, dim))
    return arr, dim


def read_ibin(path, count=None):
    """Read a big-ann .ibin (ground-truth ids): int32 n, int32 k, then int32 ids."""
    with open(path, "rb") as f:
        n, k = np.fromfile(f, dtype=np.int32, count=2)
    n, k = int(n), int(k)
    if count is None or count > n:
        count = n
    ids = np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(n, k))[:count]
    return np.asarray(ids, dtype=np.int64)


def compute_gt_ip(base, queries, k, batch=64):
    """Exact max-inner-product top-k ids for each query (0-based)."""
    nb = base.shape[0]
    nq = queries.shape[0]
    k = min(k, nb)
    out = np.empty((nq, k), dtype=np.int64)
    for start in range(0, nq, batch):
        qb = queries[start:start + batch]
        scores = np.asarray(base, dtype=np.float32) @ qb.T  # (nb, bq)
        part = np.argpartition(-scores, kth=k - 1, axis=0)[:k]  # (k, bq)
        for j in range(qb.shape[0]):
            idx = part[:, j]
            order = idx[np.argsort(-scores[idx, j])]
            out[start + j] = order
    return out


class Dataset:
    def __init__(self, name, base, queries, gt, metric="ip"):
        self.name = name
        self.base = base
        self.queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.gt = gt
        self.metric = metric

    @property
    def nb(self):
        return self.base.shape[0]

    @property
    def nq(self):
        return self.queries.shape[0]

    @property
    def dim(self):
        return self.base.shape[1]

    def ids(self):
        return np.arange(self.nb, dtype=np.int64)

    def gt_list(self):
        return [row.tolist() for row in self.gt]


def load_synthetic(nb=100000, nq=1000, dim=200, k=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((nb, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    gt = compute_gt_ip(base, queries, k)
    return Dataset(f"synthetic-{nb}", base, queries, gt)


def _find_one(data_dir, patterns, what):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"could not find {what} in {data_dir} (looked for {patterns}); see README")


def load_t2i(data_dir, nb=None, nq=None, k=10, base_file=None, query_file=None,
             gt_file=None, gt_batch=64):
    base_path = base_file or _find_one(
        data_dir, ["base*.fbin", "base*fbin*", "*base*fbin*"], "T2I base .fbin")
    query_path = query_file or _find_one(
        data_dir, ["query.public*fbin*", "query*.fbin", "query*fbin*", "*query*fbin*"],
        "T2I query .fbin")
    base, dim = read_fbin(base_path, count=nb)
    queries, _ = read_fbin(query_path, count=nq)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    nb_eff = base.shape[0]

    gt = None
    gt_path = gt_file
    if gt_path is None:
        try:
            gt_path = _find_one(data_dir, ["groundtruth*.ibin", "gt*.ibin", "*gt*.ibin"], "gt")
        except FileNotFoundError:
            gt_path = None
    if gt_path is not None:
        gt_full = read_ibin(gt_path, count=queries.shape[0])
        if k > gt_full.shape[1]:
            raise SystemExit(
                f"--gt-file {os.path.basename(gt_path)} has only {gt_full.shape[1]} "
                f"neighbors/query but --k={k}. Omit --gt-file to compute exact "
                f"ground truth at k={k} (brute force, cached).")
        gt = gt_full[:, :k]
        # The official GT's neighbor ids index the full published slice. If we
        # loaded fewer base vectors, those neighbors aren't present and recall is
        # meaningless (it collapses to ~nb/base_size). Refuse instead of lying.
        max_id = int(gt.max()) if gt.size else 0
        if max_id >= nb_eff:
            raise SystemExit(
                f"--gt-file {os.path.basename(gt_path)} references neighbor ids up to "
                f"{max_id} but only --nb={nb_eff} base vectors were loaded. This ground "
                f"truth was computed over a larger base, so recall would be meaningless. "
                f"Either load the full published slice (e.g. --nb matching the GT, ~10M for "
                f"text2image-10M) or omit --gt-file to compute exact GT over your "
                f"{nb_eff}-vector subset.")

    # Official GT only matches the exact published slice size; if we sliced to a
    # custom nb, recompute (and cache) exact GT to stay correct.
    cache = os.path.join(data_dir, f"gt_cache_nb{nb_eff}_nq{queries.shape[0]}_k{k}.npy")
    if gt is None:
        if os.path.exists(cache):
            gt = np.load(cache)
        else:
            gt = compute_gt_ip(base, queries, k, batch=gt_batch)
            np.save(cache, gt)
    name = f"t2i-{nb_eff}"
    return Dataset(name, base, queries, gt)


def load_parquet(path, dim=None, nq=1000, k=10, id_col="id", emb_col="emb",
                 seed=0, gt_batch=64):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=[id_col, emb_col])
    embs = table.column(emb_col).to_pylist()
    base = np.asarray(embs, dtype=np.float32)
    if dim is not None and base.shape[1] != dim:
        raise ValueError(f"parquet emb dim {base.shape[1]} != expected {dim}")
    rng = np.random.default_rng(seed)
    qidx = rng.choice(base.shape[0], size=min(nq, base.shape[0]), replace=False)
    queries = base[qidx]
    gt = compute_gt_ip(base, queries, k, batch=gt_batch)
    return Dataset(f"parquet-{base.shape[0]}", base, queries, gt)
