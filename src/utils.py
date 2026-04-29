"""
src/utils.py  --  shared helpers for the Multimodal Clinical RAG project
"""

import os
import re
import json
import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s -- %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def setup_java():
    """Set JAVA_HOME for Pyserini on HPC. Call once per notebook."""
    from configs.config import JAVA_HOME
    java_home = str(JAVA_HOME)
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = os.path.join(java_home, "bin") + ":" + os.environ.get("PATH", "")
    get_logger("utils").info(f"JAVA_HOME={java_home}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def chunk_words(text: str, size: int = 250, overlap: int = 50, min_chars: int = 100):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size]).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def load_jsonl(path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_paths(*paths) -> bool:
    """Print status of each path. Return True if all exist."""
    ok = True
    for p in paths:
        p = Path(p)
        ex = p.exists()
        size = f"({p.stat().st_size / 1e6:.1f} MB)" if ex and p.is_file() else ""
        print(f"  {'[OK]' if ex else '[MISSING]'}  {p.name}  {size}")
        if not ex:
            ok = False
    return ok


def passage_record(doc_id: str, text: str, modality: str, source: str, **extra) -> dict:
    """Create a standard passage record that works across all modalities."""
    rec = {
        "id":       doc_id,
        "contents": text,
        "modality": modality,   # "text" | "image" | "audio" | "pdf"
        "source":   source,
    }
    rec.update(extra)
    return rec
