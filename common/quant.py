"""Quantization scenarios for the SereneDB IVF vector index.

Each scenario maps to the `ivf (...)` opclass options accepted by
`CREATE INDEX ... USING inverted(id, emb ivf (...))`. See
tests/sqllogic/sdb/pg/index/vector_search.test for the option grammar:

    metric (l2|l1|cosine|ip, REQUIRED)
    nlist (int >= 1, default auto ~sqrt(rows))
    train_sample (int >= 1, default auto)
    quant (sq8|sq4|pq|rabitq|none, default none; sq4/pq/rabitq need l2|ip)
    pq_m (int >= 1, divides dimension, quant='pq' only, default auto)
    rabitq_bits (int 1-9, quant='rabitq' only, default 1)
"""

SCENARIOS = ["none", "sq8", "sq4", "pq", "rabitq"]

# Text2Image is a max-inner-product dataset.
DEFAULT_METRIC = "ip"


def largest_divisor_leq(dim, cap):
    for m in range(min(cap, dim), 0, -1):
        if dim % m == 0:
            return m
    return 1


def default_pq_m(dim):
    # PQ codes are 4 bits/subquantizer (fast-scan), which needs a narrower
    # subspace than 8-bit PQ to hold recall. Aim for ~2 dims per
    # subquantizer, but must divide the dimension.
    return largest_divisor_leq(dim, max(1, dim // 2))


def ivf_options(quant, *, metric=DEFAULT_METRIC, nlist=None, nlist_factor=None,
                train_sample=None, pq_m=None, rabitq_bits=None, dim=None):
    if quant not in SCENARIOS:
        raise ValueError(f"unknown quant scenario {quant!r}; expected one of {SCENARIOS}")
    opts = [f"metric = '{metric}'"]
    if nlist is not None:
        opts.append(f"nlist = {int(nlist)}")
    if nlist_factor is not None:
        opts.append(f"nlist_factor = {float(nlist_factor)}")
    if train_sample is not None:
        opts.append(f"train_sample = {int(train_sample)}")
    opts.append(f"quant = '{quant}'")
    if quant == "pq":
        if pq_m is None:
            if dim is None:
                raise ValueError("pq quant needs pq_m or dim to derive pq_m")
            pq_m = default_pq_m(dim)
        opts.append(f"pq_m = {int(pq_m)}")
    # rabitq_bits is only accepted alongside quant='rabitq'; leave it off
    # otherwise (the engine rejects it) and let the server default it to 1.
    if quant == "rabitq" and rabitq_bits is not None:
        opts.append(f"rabitq_bits = {int(rabitq_bits)}")
    return ", ".join(opts)


def index_ddl(index_name, target, id_col, emb_col, quant, *, with_opts=None, **kw):
    opts = ivf_options(quant, **kw)
    ddl = (f"CREATE INDEX {index_name} ON {target} "
           f"USING inverted({id_col}, {emb_col} ivf ({opts}))")
    if with_opts:
        ddl += " WITH (" + ", ".join(f"{k} = {v}" for k, v in with_opts.items()) + ")"
    return ddl
