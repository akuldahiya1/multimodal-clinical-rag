"""
src/evaluation.py
=================
P@10, NDCG@10, and MRR evaluation metrics.
Handles multiple retrieval systems for comparison.
"""

import math
import pandas as pd

from src.utils import get_logger

logger = get_logger("evaluation")


def precision_at_k(relevance: list, k: int = 10) -> float:
    return sum(relevance[:k]) / k


def ndcg_at_k(relevance: list, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg  = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(relevance), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevance: list) -> float:
    """Mean Reciprocal Rank."""
    for i, rel in enumerate(relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(results_df: pd.DataFrame, judgments_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Compute P@K, NDCG@K, and MRR for each (system, query) combination.

    Args:
        results_df:   Columns: system, query, rank, docid, contents
        judgments_df: Columns: query, docid, relevant (0 or 1)
        k:            Cutoff rank.

    Returns:
        DataFrame with one row per (system, query) and metric columns.
    """
    merged = results_df.merge(
        judgments_df[["query", "docid", "relevant"]],
        on=["query", "docid"],
        how="left"
    )
    merged["relevant"] = merged["relevant"].fillna(0).astype(int)

    rows = []
    for (system, query), grp in merged.groupby(["system", "query"]):
        grp_sorted = grp.sort_values("rank").head(k)
        rel_list   = grp_sorted["relevant"].tolist()
        rows.append({
            "system":  system,
            "query":   query,
            f"P@{k}":  round(precision_at_k(rel_list, k), 4),
            f"NDCG@{k}": round(ndcg_at_k(rel_list, k), 4),
            "MRR":     round(mrr(rel_list), 4),
        })

    return pd.DataFrame(rows)


def print_report(metrics_df: pd.DataFrame, k: int = 10) -> None:
    """Pretty-print evaluation results."""
    summary = metrics_df.groupby("system")[[f"P@{k}", f"NDCG@{k}", "MRR"]].mean().round(4)
    summary = summary.sort_values(f"P@{k}", ascending=False)

    print(f"\n{'='*55}")
    print(f"  Retrieval Evaluation -- P@{k} / NDCG@{k} / MRR")
    print(f"{'='*55}")
    print(summary.to_string())
    print(f"{'='*55}\n")

    print("Per-query breakdown:")
    for system, grp in metrics_df.groupby("system"):
        print(f"\n  [{system}]")
        for _, row in grp.iterrows():
            q = row["query"][:50]
            p = row[f"P@{k}"]
            bar = "#" * int(p * 20)
            print(f"    {q:<50}  {p:.2f}  |{bar}")


def create_judgment_template(results_df: pd.DataFrame, output_path) -> pd.DataFrame:
    """Create a blank CSV for manual relevance annotation."""
    template = (
        results_df[["query", "docid", "contents", "modality"]]
        .drop_duplicates(subset=["query", "docid"])
        .reset_index(drop=True)
    )
    template["relevant"] = ""
    template.to_csv(output_path, index=False)
    logger.info(f"Judgment template saved: {output_path} ({len(template)} pairs)")
    return template


def load_judgments(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["relevant"] = (
        df["relevant"].astype(str).str.strip()
        .replace({"": "0", "nan": "0"}).astype(int)
    )
    n_rel = df["relevant"].sum()
    logger.info(f"Loaded {len(df)} judgments: {n_rel} relevant, {len(df)-n_rel} not relevant")
    return df
