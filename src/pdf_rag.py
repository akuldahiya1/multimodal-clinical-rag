"""
src/pdf_rag.py
==============
Smart PDF RAG with query routing.

When user uploads a PDF:
1. Extract text chunks with page numbers
2. Extract page images
3. Embed everything on the fly (in memory)
4. Route queries: PDF-specific vs global knowledge
5. Weighted fusion of both indexes
6. Return answers with page citations
"""

import numpy as np
import faiss
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from src.utils import get_logger, chunk_words

logger = get_logger("pdf_rag")

# In-memory PDF index (reset on each new upload)
_pdf_index      = None
_pdf_chunks     = []   # list of {text, page, source}
_pdf_embeddings = None
_pdf_name       = ""


def reset_pdf_index():
    """Clear the in-memory PDF index."""
    global _pdf_index, _pdf_chunks, _pdf_embeddings, _pdf_name
    _pdf_index      = None
    _pdf_chunks     = []
    _pdf_embeddings = None
    _pdf_name       = ""


def extract_pdf_chunks(pdf_path: str) -> list:
    """
    Extract text chunks from PDF with page numbers and metadata.
    Uses section-aware chunking -- splits at headings and paragraphs.
    """
    doc    = fitz.open(pdf_path)
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = " ".join(text.split())

        if len(text) < 50:
            continue

        # Split into chunks
        page_chunks = chunk_words(text, size=120, overlap=40, min_chars=60)
        for j, chunk in enumerate(page_chunks):
            chunks.append({
                "text":    chunk,
                "page":    page_num + 1,
                "chunk_id": f"pdf_p{page_num+1}_c{j}",
                "source":  Path(pdf_path).name,
            })

    doc.close()
    logger.info(f"Extracted {len(chunks)} chunks from {Path(pdf_path).name}")
    return chunks


def extract_pdf_page_images(pdf_path: str, max_pages: int = 8) -> list:
    """
    Render each PDF page as a JPEG image.
    Returns list of image file paths in system temp directory.
    """
    doc     = fitz.open(pdf_path)
    tmp_dir = Path(tempfile.gettempdir()) / "pdf_pages"
    tmp_dir.mkdir(exist_ok=True)

    img_paths = []
    for page_num in range(min(len(doc), max_pages)):
        page     = doc[page_num]
        mat      = fitz.Matrix(1.8, 1.8)  # 1.8x zoom = good quality
        pix      = page.get_pixmap(matrix=mat)
        img_path = tmp_dir / f"pdf_{Path(pdf_path).stem}_p{page_num+1}.jpg"
        pix.save(str(img_path))
        img_paths.append(str(img_path))

    doc.close()
    logger.info(f"Rendered {len(img_paths)} page images")
    return img_paths


def build_pdf_index(pdf_path: str) -> dict:
    """
    Build an in-memory FAISS index from a PDF.
    Returns info dict with chunk count and page count.
    """
    global _pdf_index, _pdf_chunks, _pdf_embeddings, _pdf_name

    from src.retrieval import load_text_model

    reset_pdf_index()
    _pdf_name = Path(pdf_path).name

    # Extract chunks
    chunks = extract_pdf_chunks(pdf_path)
    if not chunks:
        return {"chunks": 0, "pages": 0, "name": _pdf_name}

    # Embed chunks
    model  = load_text_model()
    texts  = [c["text"] for c in chunks]
    embeds = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    # Build FAISS index
    dim         = embeds.shape[1]
    index       = faiss.IndexFlatIP(dim)
    index.add(embeds)

    _pdf_index      = index
    _pdf_chunks     = chunks
    _pdf_embeddings = embeds

    pages = max(c["page"] for c in chunks)
    logger.info(f"PDF index built: {len(chunks)} chunks, {pages} pages")

    return {
        "chunks": len(chunks),
        "pages":  pages,
        "name":   _pdf_name,
    }


def search_pdf(query: str, top_k: int = 5) -> list:
    """
    Search the in-memory PDF index.
    Returns list of chunk dicts with similarity scores.
    """
    global _pdf_index, _pdf_chunks

    if _pdf_index is None or not _pdf_chunks:
        return []

    from src.retrieval import load_text_model
    model     = load_text_model()
    query_emb = model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")

    scores, indices = _pdf_index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _pdf_chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def compute_pdf_relevance(query: str) -> float:
    """
    Compute how relevant the query is to the uploaded PDF.
    Returns score 0-1. Above 0.5 = PDF-focused query.
    """
    if _pdf_index is None or not _pdf_chunks:
        return 0.0

    results = search_pdf(query, top_k=3)
    if not results:
        return 0.0

    return max(r["score"] for r in results)


def route_query(query: str, threshold: float = 0.32) -> dict:
    """
    Decide how to weight PDF vs global retrieval.

    Returns:
        {
            "pdf_weight":    float,
            "global_weight": float,
            "pdf_score":     float,
            "strategy":      str,
        }
    """
    pdf_score = compute_pdf_relevance(query)

    if _pdf_index is None:
        return {
            "pdf_weight":    0.0,
            "global_weight": 1.0,
            "pdf_score":     0.0,
            "strategy":      "global_only",
        }

    if pdf_score >= threshold:
        return {
            "pdf_weight":    0.8,
            "global_weight": 0.2,
            "pdf_score":     round(pdf_score, 4),
            "strategy":      "pdf_focused",
        }
    elif pdf_score >= 0.20:
        return {
            "pdf_weight":    0.5,
            "global_weight": 0.5,
            "pdf_score":     round(pdf_score, 4),
            "strategy":      "balanced",
        }
    else:
        return {
            "pdf_weight":    0.2,
            "global_weight": 0.8,
            "pdf_score":     round(pdf_score, 4),
            "strategy":      "global_focused",
        }


def hybrid_search_with_pdf(query: str, top_k: int = 10) -> tuple:
    """
    Full hybrid search combining PDF and global indexes.

    Returns:
        (results_list, routing_info_dict)

    Each result has: text, page, source, score, modality
    """
    import pandas as pd
    from src.retrieval import search_bm25, search_dense_text, rrf_merge, load_metadata

    # Get routing decision
    routing = route_query(query)

    logger.info(
        f"Routing: strategy={routing['strategy']} | "
        f"pdf_score={routing['pdf_score']} | "
        f"pdf_weight={routing['pdf_weight']}"
    )

    all_results = []

    # PDF search
    if routing["pdf_weight"] > 0 and _pdf_index is not None:
        pdf_results = search_pdf(query, top_k=top_k)
        for r in pdf_results:
            all_results.append({
                "docid":    r["chunk_id"],
                "contents": r["text"],
                "page":     r["page"],
                "source":   r["source"],
                "score":    r["score"] * routing["pdf_weight"],
                "modality": "pdf",
                "from_pdf": True,
            })

    # Global search
    if routing["global_weight"] > 0:
        bm25_df  = search_bm25(query, top_k=30)
        dense_df = search_dense_text(query, top_k=30)
        rrf      = rrf_merge(bm25_df, dense_df, k=60)
        meta     = load_metadata()

        for rank, (docid, rrf_score) in enumerate(list(rrf.items())[:top_k]):
            contents = ""
            modality = "text"
            if docid in meta.index:
                contents = str(meta.loc[docid, "contents"])
                modality = str(meta.loc[docid, "modality"])

            all_results.append({
                "docid":    docid,
                "contents": contents,
                "page":     None,
                "source":   "Global Knowledge Base",
                "score":    rrf_score * routing["global_weight"],
                "modality": modality,
                "from_pdf": False,
            })

    # Sort by score and deduplicate
    seen = set()
    final = []
    for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
        key = r["docid"]
        if key not in seen:
            seen.add(key)
            final.append(r)
        if len(final) >= top_k:
            break

    return final, routing


def format_pdf_context(results: list, n: int = 5) -> str:
    """
    Format results into a prompt context with page citations.
    PDF chunks are labeled with page numbers.
    """
    parts = []
    for i, r in enumerate(results[:n], 1):
        if r.get("from_pdf") and r.get("page"):
            label = f"[Source {i} | PDF Page {r['page']}]"
        else:
            modality = r.get("modality", "text").upper()
            label    = f"[Source {i} | {modality}]"

        parts.append(f"{label}\n{r['contents'][:400]}")

    return "\n\n".join(parts)
