"""
src/retrieval.py
================
Full hybrid retrieval pipeline:
  1. BM25 (Pyserini) -- keyword matching
  2. Dense (FAISS + BioLORD) -- semantic text search
  3. RRF fusion -- merges BM25 + Dense ranked lists
  4. Cross-encoder reranker -- precision pass on top-50

For image queries:
  - BiomedCLIP encodes the image
  - Searches FAISS image index
  - Results merged with text results via RRF

All retrieval functions return a consistent pandas DataFrame.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import get_logger

logger = get_logger("retrieval")

# Module-level singletons
_bm25_searcher  = None
_text_model     = None
_image_model    = None
_image_processor= None
_reranker       = None
_faiss_text     = None
_faiss_text_ids = None
_faiss_image    = None
_faiss_image_ids= None
_metadata_df    = None


#  Loaders 

def load_bm25():
    global _bm25_searcher
    if _bm25_searcher is not None:
        return _bm25_searcher
    from configs.config import BM25_INDEX_DIR
    from pyserini.search.lucene import LuceneSearcher
    path = str(BM25_INDEX_DIR)
    logger.info(f"Loading BM25 index: {path}")
    _bm25_searcher = LuceneSearcher(path)
    logger.info("BM25 ready")
    return _bm25_searcher


def load_text_model():
    global _text_model
    if _text_model is not None:
        return _text_model
    from configs.config import TEXT_EMBED_MODEL
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading text embedding model: {TEXT_EMBED_MODEL}")
    _text_model = SentenceTransformer(TEXT_EMBED_MODEL)
    logger.info("Text model ready")
    return _text_model


def load_image_model():
    global _image_model, _image_processor
    if _image_model is not None:
        return _image_model, _image_processor
    import open_clip
    import torch
    logger.info("Loading BiomedCLIP via open_clip...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    _image_model     = model
    _image_processor = preprocess
    logger.info(f"BiomedCLIP ready on {device}")
    return _image_model, _image_processor


def load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    from configs.config import RERANKER_MODEL
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading reranker: {RERANKER_MODEL}")
    _reranker = CrossEncoder(RERANKER_MODEL)
    logger.info("Reranker ready")
    return _reranker


def load_faiss_text():
    global _faiss_text, _faiss_text_ids
    if _faiss_text is not None:
        return _faiss_text, _faiss_text_ids
    import faiss
    from configs.config import FAISS_TEXT_PATH, FAISS_IDS_TEXT
    logger.info("Loading text FAISS index...")
    _faiss_text     = faiss.read_index(str(FAISS_TEXT_PATH))
    _faiss_text_ids = json.load(open(FAISS_IDS_TEXT))
    logger.info(f"Text FAISS ready: {_faiss_text.ntotal:,} vectors")
    return _faiss_text, _faiss_text_ids


def load_faiss_image():
    global _faiss_image, _faiss_image_ids
    if _faiss_image is not None:
        return _faiss_image, _faiss_image_ids
    import faiss
    from configs.config import FAISS_IMAGE_PATH, FAISS_IDS_IMAGE
    if not Path(FAISS_IMAGE_PATH).exists():
        logger.warning("Image FAISS index not found. Image search disabled.")
        return None, None
    logger.info("Loading image FAISS index...")
    _faiss_image     = faiss.read_index(str(FAISS_IMAGE_PATH))
    _faiss_image_ids = json.load(open(FAISS_IDS_IMAGE))
    logger.info(f"Image FAISS ready: {_faiss_image.ntotal:,} vectors")
    return _faiss_image, _faiss_image_ids


def load_metadata():
    global _metadata_df
    if _metadata_df is not None:
        return _metadata_df
    from configs.config import METADATA_PARQUET
    logger.info("Loading metadata...")
    _metadata_df = pd.read_parquet(METADATA_PARQUET).set_index("id")
    logger.info(f"Metadata loaded: {len(_metadata_df):,} records")
    return _metadata_df


#  BM25 

def search_bm25(query: str, top_k: int = 50) -> pd.DataFrame:
    """Keyword search over all text-modality passages."""
    searcher = load_bm25()
    hits     = searcher.search(query, k=top_k)
    rows = []
    for rank, hit in enumerate(hits, 1):
        raw = json.loads(searcher.doc(hit.docid).raw())
        rows.append({
            "rank":       rank,
            "docid":      hit.docid,
            "bm25_score": float(hit.score),
            "contents":   raw.get("contents", ""),
            "modality":   raw.get("modality", "text"),
        })
    return pd.DataFrame(rows)


#  Dense text 

def search_dense_text(query: str, top_k: int = 50) -> pd.DataFrame:
    """Semantic search using BioLORD embeddings over FAISS text index."""
    model            = load_text_model()
    index, ids       = load_faiss_text()
    meta             = load_metadata()

    query_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(query_vec, top_k)

    rows = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        docid    = ids[idx]
        contents = meta.loc[docid, "contents"] if docid in meta.index else ""
        modality = meta.loc[docid, "modality"] if docid in meta.index else "text"
        rows.append({
            "rank":        rank,
            "docid":       docid,
            "dense_score": float(score),
            "contents":    contents,
            "modality":    modality,
        })
    return pd.DataFrame(rows)


#  Dense image 

def search_dense_image_from_query(query: str, top_k: int = 20) -> pd.DataFrame:
    """
    Search image index using a TEXT query encoded by BiomedCLIP.
    This lets text queries also retrieve relevant images.
    """
    index, ids = load_faiss_image()
    if index is None:
        return pd.DataFrame()

    model, processor = load_image_model()
    import torch

    import open_clip
    tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    with torch.no_grad():
        text_tokens = tokenizer([query]).to(next(model.parameters()).device)
        text_feat   = model.encode_text(text_tokens)
        text_feat   = text_feat / text_feat.norm(dim=-1, keepdim=True)

    query_vec = text_feat.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vec, top_k)

    meta = load_metadata()
    rows = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        docid    = ids[idx]
        contents = meta.loc[docid, "contents"] if docid in meta.index else ""
        image_path = meta.loc[docid, "image_path"] if docid in meta.index else ""
        rows.append({
            "rank":        rank,
            "docid":       docid,
            "dense_score": float(score),
            "contents":    contents,
            "modality":    "image",
            "image_path":  image_path,
        })
    return pd.DataFrame(rows)


#  RRF fusion 

def rrf_merge(*ranked_lists, k: int = 60) -> dict:
    """
    Reciprocal Rank Fusion across any number of ranked lists.
    Each list is a DataFrame with columns: docid, rank.
    Returns {docid: rrf_score} sorted descending.
    """
    scores = {}
    for df in ranked_lists:
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            docid = row["docid"]
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + row["rank"])
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


#  Cross-encoder reranker 

def rerank(query: str, candidates: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Cross-encoder reranker: scores each (query, passage) pair precisely.
    This is the single biggest accuracy boost in the pipeline.
    """
    if candidates.empty:
        return candidates

    reranker = load_reranker()
    pairs    = [(query, row["contents"]) for _, row in candidates.iterrows()]

    scores = reranker.predict(pairs)

    result          = candidates.copy().reset_index(drop=True)
    result["rerank_score"] = scores
    result          = result.sort_values("rerank_score", ascending=False).head(top_n)
    result["rank"]  = range(1, len(result) + 1)
    return result.reset_index(drop=True)


#  Main retrieval function 

def retrieve(
    query:        str,
    top_k:        int   = 10,
    bm25_k:       int   = 50,
    dense_k:      int   = 50,
    rrf_k:        int   = 60,
    include_images: bool = True,
    use_reranker: bool  = True,
) -> pd.DataFrame:
    """
    Full hybrid retrieval pipeline:
      BM25 + Dense text + (optional) Dense image -> RRF -> reranker

    Returns top_k passages as a DataFrame with all score columns.
    """
    meta = load_metadata()

    # Run all retrievers
    bm25_df    = search_bm25(query, top_k=bm25_k)
    dense_text = search_dense_text(query, top_k=dense_k)
    dense_img  = search_dense_image_from_query(query, top_k=20) if include_images else pd.DataFrame()

    # Collect content for all candidates
    text_lookup = {}
    for df in [bm25_df, dense_text, dense_img]:
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                if row["docid"] not in text_lookup and row.get("contents"):
                    text_lookup[row["docid"]] = row

    # RRF merge
    rrf_scores = rrf_merge(bm25_df, dense_text, dense_img, k=rrf_k)

    # Build candidate DataFrame
    candidate_pool = min(50, len(rrf_scores))
    top_docids     = list(rrf_scores.keys())[:candidate_pool]

    rows = []
    for docid in top_docids:
        info = text_lookup.get(docid, {})
        rows.append({
            "docid":      docid,
            "rrf_score":  round(rrf_scores[docid], 6),
            "contents":   info.get("contents", ""),
            "modality":   info.get("modality", "text"),
            "image_path": info.get("image_path", ""),
        })
    candidates = pd.DataFrame(rows)

    # Rerank for precision
    if use_reranker and not candidates.empty:
        result = rerank(query, candidates, top_n=top_k)
    else:
        result = candidates.head(top_k).copy()
        result["rank"] = range(1, len(result) + 1)

    return result


def retrieve_all_systems(query: str, top_k: int = 10) -> dict:
    """
    Run all three systems independently for comparison/evaluation.
    Returns {"BM25": df, "Dense": df, "Hybrid+Rerank": df}
    """
    bm25_raw   = search_bm25(query, top_k=top_k)
    dense_raw  = search_dense_text(query, top_k=top_k)
    hybrid     = retrieve(query, top_k=top_k)

    return {
        "BM25":           bm25_raw.head(top_k),
        "Dense":          dense_raw.head(top_k),
        "Hybrid+Rerank":  hybrid,
    }
