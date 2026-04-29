"""
src/ingest/images.py
====================
Download PMC-VQA and ROCO v2 datasets from HuggingFace.
Extracts image paths + captions -> data/processed/image_passages.jsonl

We treat each image+caption as a "passage":
  - contents = caption text (indexed for BM25 + text embedding)
  - image_path = local path (used by BiomedCLIP for image embedding)
"""

import os
import json
from pathlib import Path

from src.utils import get_logger, save_jsonl, passage_record

logger = get_logger("ingest.images")


def download_pmcvqa(output_dir: Path, sample: int = 5000) -> list:
    """
    Download PMC-VQA dataset (image + question-answer pairs with captions).
    Uses HuggingFace datasets library.
    """
    from datasets import load_dataset

    logger.info(f"Downloading PMC-VQA (sample={sample})...")
    ds = load_dataset("xmcmic/PMC-VQA", split="train", streaming=True)

    records = []
    img_dir = output_dir / "pmcvqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(ds):
        if i >= sample:
            break

        doc_id = f"pmcvqa_{i:06d}"
        caption = item.get("Caption", "") or item.get("Question", "")
        answer  = item.get("Answer", "")

        # Save image to disk
        if "Image" in item and item["Image"] is not None:
            img_path = img_dir / f"{doc_id}.jpg"
            if not img_path.exists():
                try:
                    item["Image"].save(img_path)
                except Exception as e:
                    logger.warning(f"Could not save image {doc_id}: {e}")
                    img_path = None
        else:
            img_path = None

        # Caption is the searchable text; image_path enables visual embedding
        combined_text = f"{caption} {answer}".strip()

        records.append(passage_record(
            doc_id     = doc_id,
            text       = combined_text,
            modality   = "image",
            source     = "pmc_vqa",
            image_path = str(img_path) if img_path else "",
            caption    = caption,
            answer     = answer,
        ))

        if (i + 1) % 500 == 0:
            logger.info(f"  PMC-VQA: {i+1}/{sample}")

    logger.info(f"PMC-VQA done: {len(records)} records")
    return records


def download_roco(output_dir: Path, sample: int = 5000) -> list:
    """
    Download ROCO v2 radiology dataset (image + caption pairs).
    """
    from datasets import load_dataset

    logger.info(f"Downloading ROCO v2 radiology (sample={sample})...")
    ds = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)

    records = []
    img_dir = output_dir / "roco"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(ds):
        if i >= sample:
            break

        doc_id  = f"roco_{i:06d}"
        caption = item.get("caption", "") or item.get("text", "")

        if "image" in item and item["image"] is not None:
            img_path = img_dir / f"{doc_id}.jpg"
            if not img_path.exists():
                try:
                    item["image"].save(img_path)
                except Exception as e:
                    logger.warning(f"Could not save {doc_id}: {e}")
                    img_path = None
        else:
            img_path = None

        records.append(passage_record(
            doc_id     = doc_id,
            text       = caption,
            modality   = "image",
            source     = "roco_v2_radiology",
            image_path = str(img_path) if img_path else "",
            caption    = caption,
        ))

        if (i + 1) % 500 == 0:
            logger.info(f"  ROCO: {i+1}/{sample}")

    logger.info(f"ROCO done: {len(records)} records")
    return records


def ingest_images(config) -> int:
    """Main entry point for image ingestion."""
    from configs.config import IMAGE_JSONL, IMAGE_DIR, PMCVQA_SAMPLE, ROCO_SAMPLE

    output_path = Path(IMAGE_JSONL)
    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        logger.info(f"Image passages already exist ({existing:,}). Skipping.")
        return existing

    image_dir = Path(IMAGE_DIR)
    all_records = []

    try:
        all_records += download_pmcvqa(image_dir, sample=PMCVQA_SAMPLE)
    except Exception as e:
        logger.warning(f"PMC-VQA download failed: {e}")

    try:
        all_records += download_roco(image_dir, sample=ROCO_SAMPLE)
    except Exception as e:
        logger.warning(f"ROCO download failed: {e}")

    save_jsonl(all_records, output_path)
    logger.info(f"Total image passages: {len(all_records):,} -> {output_path}")
    return len(all_records)
