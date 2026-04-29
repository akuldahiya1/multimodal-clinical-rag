"""
src/ingest/text.py
==================
Load existing PMC passages from your previous project,
or re-parse from scratch if needed.
Outputs: data/processed/text_passages.jsonl
"""

import json
import pandas as pd
from pathlib import Path

from src.utils import get_logger, chunk_words, save_jsonl, passage_record

logger = get_logger("ingest.text")


def load_from_existing_parquet(parquet_path: Path, output_path: Path) -> int:
    """
    Re-use the passages you already built in the previous project.
    This is the fast path -- no re-downloading needed.

    Args:
        parquet_path:  Path to pmc_30k_passages.parquet from old project.
        output_path:   Where to write the unified JSONL.

    Returns:
        Number of passages written.
    """
    logger.info(f"Loading existing passages from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    records = []
    for _, row in df.iterrows():
        records.append(passage_record(
            doc_id   = row["id"],
            text     = row["contents"],
            modality = "text",
            source   = "pmc_open_access",
            title    = row.get("title", ""),
            doc_id_  = row.get("doc_id", ""),
        ))

    save_jsonl(records, output_path)
    logger.info(f"Written {len(records):,} text passages -> {output_path}")
    return len(records)


def ingest_text(config) -> int:
    """
    Main entry point for text ingestion.
    Uses existing passages if available, otherwise raises a clear error.
    """
    from configs.config import LEGACY_PASSAGES, TEXT_JSONL

    output_path = Path(TEXT_JSONL)

    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        logger.info(f"Text passages already exist ({existing:,} records). Skipping.")
        return existing

    if Path(LEGACY_PASSAGES).exists():
        logger.info("Using passages from existing rag_project (no re-download needed).")
        return load_from_existing_parquet(Path(LEGACY_PASSAGES), output_path)

    raise FileNotFoundError(
        f"No existing passages found at {LEGACY_PASSAGES}.\n"
        "Run notebook 01 to re-download and process PMC articles."
    )
