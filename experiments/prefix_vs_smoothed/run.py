"""Entry point for the prefix-vs-smoothed experiment.

Run with the synthetic corpus and TF-IDF embedder (no extra deps):

    python -m experiments.prefix_vs_smoothed.run

Edit ``CONFIG`` below (or pass --embedder / --corpus) to switch
embedders or use a real corpus. See README.md for details.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.prefix_vs_smoothed.corpus import (
    load_jsonl,
    synthetic_corpus,
)
from experiments.prefix_vs_smoothed.embedders import make_embedder
from experiments.prefix_vs_smoothed.metrics import (
    changepoint_recall,
    operator_agreement,
    pointwise_cosine,
    segmentation_agreement,
    trajectory_dtw,
    trajectory_shape,
)
from experiments.prefix_vs_smoothed.paths import (
    normalize_rows,
    per_item_embeddings,
    prefix_path,
)
from experiments.prefix_vs_smoothed.weightings import all_candidates


CONFIG = {
    "embedder": "tfidf",       # "tfidf" | "sentence-transformers" | "openai"
    "corpus": "synthetic",     # "synthetic" or path to a JSONL file
    "n_conversations": 20,
    "seed": 0,
    "alpha_grid": (0.5, 0.7, 0.85, 0.95, 0.99),
    "include_lstsq_ceiling": True,
    "auto_segment_alpha": 0.85,
    # Default 5 (embflow's own default) is conservative for short
    # synthetic conversations; lower it to see segmentation fire.
    "auto_segment_min_size": 5,
    "output_dir": "experiments/prefix_vs_smoothed/results",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default=CONFIG["embedder"])
    p.add_argument("--corpus", default=CONFIG["corpus"])
    p.add_argument("--n", type=int, default=CONFIG["n_conversations"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--no-lstsq", action="store_true",
                   help="skip the per-conversation least-squares ceiling")
    p.add_argument("--min-segment-size", type=int,
                   default=CONFIG["auto_segment_min_size"],
                   help="passed to auto_segment; lower values let "
                        "segmentation fire on short conversations")
    p.add_argument("--output-dir", default=CONFIG["output_dir"])
    return p.parse_args()


def load_corpus(corpus_arg, n_conversations, seed):
    """Returns (conversations, ground_truth_changepoints_or_None)."""
    if corpus_arg == "synthetic":
        return synthetic_corpus(n_conversations, seed=seed)
    convs = load_jsonl(corpus_arg)[:n_conversations]
    return convs, None


def build_embedder(name, corpus):
    if name == "tfidf":
        emb = make_embedder("tfidf")
        emb.fit([msg for conv in corpus for msg in conv])
        return emb
    return make_embedder(name)


def run(cfg):
    corpus, ground_truth = load_corpus(
        cfg["corpus"], cfg["n_conversations"], cfg["seed"]
    )
    embedder = build_embedder(cfg["embedder"], corpus)

    accumulator = defaultdict(list)
    gt_recall_prefix = []
    gt_recall_per_candidate = defaultdict(list)

    for ci, conv in enumerate(corpus):
        if len(conv) < 3:
            continue
        gt = ground_truth[ci] if ground_truth is not None else None

        prefix = normalize_rows(prefix_path(conv, embedder))
        per_item = normalize_rows(per_item_embeddings(conv, embedder))
        lengths = [max(len(m), 1) for m in conv]

        if gt is not None:
            import embflow as ef
            segs_prefix = ef.auto_segment(
                prefix,
                alpha=cfg["auto_segment_alpha"],
                min_segment_size=cfg["auto_segment_min_size"],
            )
            gt_recall_prefix.append(changepoint_recall(segs_prefix, gt))

        candidates = all_candidates(
            per_item, lengths,
            prefix=prefix if cfg["include_lstsq_ceiling"] else None,
            alpha_grid=cfg["alpha_grid"],
            include_lstsq=cfg["include_lstsq_ceiling"],
        )

        for cname, cand in candidates.items():
            cosines = pointwise_cosine(prefix, cand)
            shape = trajectory_shape(prefix, cand)
            ops = operator_agreement(prefix, cand)
            seg_jaccard, n_bnd_prefix, n_bnd_cand = segmentation_agreement(
                prefix, cand,
                alpha=cfg["auto_segment_alpha"],
                min_segment_size=cfg["auto_segment_min_size"],
            )

            run_record = {
                "conversation": ci,
                "n": len(conv),
                "mean_cosine": float(np.mean(cosines)),
                "min_cosine": float(np.min(cosines)),
                "dtw": trajectory_dtw(prefix, cand),
                "shape": shape,
                "operator_corr": ops,
                "segmentation_jaccard": seg_jaccard,
                "n_changepoints_prefix": n_bnd_prefix,
                "n_changepoints_candidate": n_bnd_cand,
            }
            accumulator[cname].append(run_record)

            if gt is not None:
                import embflow as ef
                segs_cand = ef.auto_segment(
                    cand,
                    alpha=cfg["auto_segment_alpha"],
                    min_segment_size=cfg["auto_segment_min_size"],
                )
                gt_recall_per_candidate[cname].append(
                    changepoint_recall(segs_cand, gt)
                )

    summary = summarize(accumulator)
    if ground_truth is not None:
        summary["_ground_truth"] = {
            "prefix_recall_mean": _mean_or_none(gt_recall_prefix),
            "candidate_recall_mean": {
                cname: _mean_or_none(rs)
                for cname, rs in gt_recall_per_candidate.items()
            },
        }

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "raw.json", "w") as f:
        json.dump({k: v for k, v in accumulator.items()}, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, embedder.name, len([c for c in corpus if len(c) >= 3]))
    return summary


def summarize(accumulator):
    summary = {}
    for cname, runs in accumulator.items():
        summary[cname] = {
            "n_conversations": len(runs),
            "mean_pointwise_cosine": _mean_or_none([r["mean_cosine"] for r in runs]),
            "mean_min_pointwise_cosine": _mean_or_none([r["min_cosine"] for r in runs]),
            "mean_dtw": _mean_or_none([r["dtw"] for r in runs]),
            "mean_shape_distance": _mean_or_none(
                [r["shape"] for r in runs if r["shape"] is not None]
            ),
            "mean_segmentation_jaccard": _mean_or_none(
                [r["segmentation_jaccard"] for r in runs]
            ),
            "mean_changepoints_prefix": _mean_or_none(
                [r["n_changepoints_prefix"] for r in runs]
            ),
            "mean_changepoints_candidate": _mean_or_none(
                [r["n_changepoints_candidate"] for r in runs]
            ),
            "mean_op_corr_angular_velocity": _mean_or_none(
                [r["operator_corr"]["angular_velocity"] for r in runs
                 if r["operator_corr"]["angular_velocity"] is not None]
            ),
            "mean_op_corr_arc_length": _mean_or_none(
                [r["operator_corr"]["arc_length"] for r in runs
                 if r["operator_corr"]["arc_length"] is not None]
            ),
            "mean_op_corr_local_curvature_radius": _mean_or_none(
                [r["operator_corr"]["local_curvature_radius"] for r in runs
                 if r["operator_corr"]["local_curvature_radius"] is not None]
            ),
        }
    return summary


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(mean(values)) if values else None


def _fmt(v, precision=3):
    return "n/a" if v is None else f"{v:.{precision}f}"


def print_summary(summary, embedder_name, n_conv):
    print(f"\n=== prefix-vs-smoothed experiment ===")
    print(f"embedder      : {embedder_name}")
    print(f"conversations : {n_conv}")
    print()

    headers = [
        "candidate", "mean_cos", "min_cos", "dtw",
        "shape", "seg_jacc", "cps_pre", "cps_cand",
        "op(angvel)", "op(arclen)", "op(curvR)",
    ]
    widths = [30, 10, 10, 8, 8, 9, 8, 9, 12, 12, 12]
    rows = [headers, ["-" * w for w in widths]]
    for cname, s in summary.items():
        if cname.startswith("_"):
            continue
        rows.append([
            cname,
            _fmt(s["mean_pointwise_cosine"]),
            _fmt(s["mean_min_pointwise_cosine"]),
            _fmt(s["mean_dtw"]),
            _fmt(s["mean_shape_distance"]),
            _fmt(s["mean_segmentation_jaccard"]),
            _fmt(s["mean_changepoints_prefix"], precision=1),
            _fmt(s["mean_changepoints_candidate"], precision=1),
            _fmt(s["mean_op_corr_angular_velocity"]),
            _fmt(s["mean_op_corr_arc_length"]),
            _fmt(s["mean_op_corr_local_curvature_radius"]),
        ])
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if "_ground_truth" in summary:
        gt = summary["_ground_truth"]
        print()
        print("ground-truth changepoint recall (synthetic corpus only)")
        print(f"  prefix path                : {_fmt(gt['prefix_recall_mean'])}")
        for cname, recall in gt["candidate_recall_mean"].items():
            print(f"  {cname:30s}: {_fmt(recall)}")

    real_summary = {k: v for k, v in summary.items() if not k.startswith("_")}
    if real_summary:
        print()
        best_cos = max(
            real_summary.items(),
            key=lambda kv: kv[1]["mean_pointwise_cosine"] or -1,
        )
        best_seg = max(
            real_summary.items(),
            key=lambda kv: kv[1]["mean_segmentation_jaccard"] or -1,
        )
        print(f"best by pointwise cosine: {best_cos[0]}")
        print(f"best by segmentation    : {best_seg[0]}")


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg["embedder"] = args.embedder
    cfg["corpus"] = args.corpus
    cfg["n_conversations"] = args.n
    cfg["seed"] = args.seed
    cfg["include_lstsq_ceiling"] = not args.no_lstsq
    cfg["auto_segment_min_size"] = args.min_segment_size
    cfg["output_dir"] = args.output_dir
    run(cfg)


if __name__ == "__main__":
    main()
