"""psycopg helpers for driving a serened instance over the pgwire protocol.

Connection defaults mirror tests/drivers/harness/spec_loader.py::conn_kwargs().
"""

import os
import psycopg


def conn_kwargs(port=None, **overrides):
    kw = {
        "host": os.environ.get("SDB_DRV_HOST", "127.0.0.1"),
        "port": int(port if port is not None else os.environ.get("SDB_DRV_PORT", "5432")),
        "dbname": os.environ.get("SDB_DRV_DATABASE", "postgres"),
        "user": os.environ.get("SDB_DRV_USER", "postgres"),
    }
    kw.update(overrides)
    return kw


def connect(port=None, autocommit=True, **overrides):
    conn = psycopg.connect(**conn_kwargs(port, **overrides))
    conn.autocommit = autocommit
    return conn


def scalar(cur, sql, params=None):
    cur.execute(sql, params)
    row = cur.fetchone()
    return None if row is None else row[0]


def execute(cur, sql, params=None):
    cur.execute(sql, params)
    try:
        return cur.fetchall()
    except psycopg.ProgrammingError:
        return None


def set_knob(cur, name, value):
    cur.execute(f"SET {name} = {int(value)}")


def emb_to_pgarray(vec):
    """Row value for a COPY into a FLOAT[N] column: a bracketed float list."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def copy_vectors(cur, table, ids, vectors, id_col="id", emb_col="emb", log=None,
                 log_every=1_000_000):
    """Bulk-load (id, embedding) rows via COPY. `vectors` is an (N, dim) array.

    Formatting the FLOAT[dim] text literal is client-side work; at 10M rows this
    dominates load time, so we emit progress.
    """
    import time
    n = len(vectors)
    t0 = time.perf_counter()
    with cur.copy(f"COPY {table} ({id_col}, {emb_col}) FROM STDIN") as cp:
        for i in range(n):
            cp.write_row((int(ids[i]), emb_to_pgarray(vectors[i])))
            if log is not None and (i + 1) % log_every == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                log(f"    loaded {i + 1:,}/{n:,} ({rate:,.0f} rows/s)")
    return n


def knn_sql(index_or_table, dim, id_col="id", emb_col="emb", k=10, binary=True):
    """Prepared-statement KNN over the max-inner-product operator `<#>`.

    The query vector is a bound parameter (psycopg `%b` = binary, `%s` = text)
    so the statement text is constant and re-planning / literal parsing is
    avoided. Verified to still fire the IVF ANN scan (VECTOR_KNN).
    """
    ph = "%b" if binary else "%s"
    return (f"SELECT {id_col} FROM {index_or_table} "
            f"ORDER BY {emb_col} <#> {ph}::FLOAT[{dim}] LIMIT {k}")


def float_list(vec):
    return [float(x) for x in vec]
