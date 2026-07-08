"""Shared argparse wiring for dataset selection and common knobs."""

from . import dataset


def add_dataset_args(p):
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", choices=["t2i", "synthetic", "parquet"], default="t2i",
                   help="which dataset to benchmark (default: t2i)")
    g.add_argument("--data-dir", default=None,
                   help="directory holding big-ann T2I .fbin/.ibin files (dataset=t2i)")
    g.add_argument("--nb", type=int, default=10_000_000,
                   help="number of base vectors to use (default: 10M)")
    g.add_argument("--nq", type=int, default=10_000,
                   help="number of query vectors to use (default: 10000)")
    g.add_argument("--dim", type=int, default=200,
                   help="vector dimension (synthetic only; T2I is 200)")
    g.add_argument("--k", type=int, default=10, help="neighbors per query (recall@k)")
    g.add_argument("--seed", type=int, default=0, help="synthetic RNG seed")
    g.add_argument("--parquet-path", default=None, help="parquet file (dataset=parquet)")
    g.add_argument("--base-file", default=None, help="explicit T2I base .fbin")
    g.add_argument("--query-file", default=None, help="explicit T2I query .fbin")
    g.add_argument("--gt-file", default=None, help="explicit ground-truth .ibin")


def load_dataset(args):
    if args.dataset == "synthetic":
        return dataset.load_synthetic(nb=args.nb, nq=args.nq, dim=args.dim,
                                      k=args.k, seed=args.seed)
    if args.dataset == "parquet":
        if not args.parquet_path:
            raise SystemExit("--parquet-path is required for dataset=parquet")
        return dataset.load_parquet(args.parquet_path, dim=None, nq=args.nq, k=args.k)
    if not args.data_dir:
        raise SystemExit("--data-dir is required for dataset=t2i (see README to fetch T2I)")
    return dataset.load_t2i(args.data_dir, nb=args.nb, nq=args.nq, k=args.k,
                            base_file=args.base_file, query_file=args.query_file,
                            gt_file=args.gt_file)


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def add_nprobe_arg(p):
    p.add_argument("--nprobe", default="8,16,32,64,128",
                   help="comma-separated sdb_nprobe sweep (default: 8,16,32,64,128)")


def add_rerank_factor_arg(p):
    p.add_argument("--rerank-factor", default="4",
                   help="comma-separated sdb_rerank_factor sweep -- sizes the exact-rerank "
                        "candidate pool for quantized IVF indexes as rerank_factor * k; "
                        "0 disables reranking (top-k picked by the approximate quantized "
                        "distance). No effect on quant='none' indexes. Default: 4 (server "
                        "default, i.e. unswept)")


def add_concurrency_arg(p):
    p.add_argument("--clients", default="1",
                   help="comma-separated sweep of concurrent client connections issuing "
                        "queries at once (default: 1, i.e. sequential single-client)")


def add_index_args(p):
    g = p.add_argument_group("index")
    g.add_argument("--nlist", type=int, default=None,
                   help="IVF cluster count (default: auto ~sqrt(rows))")
    g.add_argument("--nlist-factor", type=float, default=None,
                   help="IVF auto-nlist multiplier: nlist = round(nlist_factor * "
                        "sqrt(rows)); default server factor is 2.0. Mutually "
                        "exclusive with --nlist.")
    g.add_argument("--train-sample", type=int, default=None,
                   help="IVF training sample size (default: auto)")
    g.add_argument("--pq-m", type=int, default=None,
                   help="PQ subquantizers (default: auto, must divide dim)")
    g.add_argument("--rabitq-bits", type=int, default=None,
                   help="RaBitQ bits per dimension, 1-9 (quant='rabitq' only, default 1)")
    g.add_argument("--settle", choices=["compact", "no-compact", "wait", "none"],
                   default="compact",
                   help="post-build handling of index segments: 'compact' (bg loops off + "
                        "VACUUM COMPACT -> merged, slower build, faster search), 'no-compact' "
                        "(bg loops off, skip compaction -> segments left un-merged, faster "
                        "build, slower search), 'wait' (bg loops on, poll until they quiesce), "
                        "'none' (bg loops on, query immediately). compact vs no-compact is a "
                        "clean A/B on the merge cost.")
    g.add_argument("--build-threads", type=int, default=1,
                   help="threads for CREATE INDEX on view-backed (remote) sources. IVF "
                        "creates one segment per parallel scan unit and sdb_nprobe is applied "
                        "PER SEGMENT, so >1 fragments the index into N segments and searches "
                        "N*nprobe cells -> inflated recall (and slower queries) vs a single-"
                        "segment index. Default 1 keeps remote indexes single-segment, matching "
                        "the local path (consolidated by VACUUM REFRESH), so nprobe is "
                        "comparable across sources. Set >1 to deliberately study fragmentation.")
