# Multimodal Clinical RAG System
CS734 -- Information Retrieval Course Project
Student: Akul Dahiya

## Overview

A state-of-the-art Retrieval-Augmented Generation (RAG) system for clinical
research, supporting four modalities: text, images, audio, and PDFs.

## Architecture

```
4 Modalities --> Ingestion --> Unified FAISS Index
                                     |
Query (text/image/voice) --> BM25 + Dense (BioLORD) + RRF Fusion
                                     |
                             Cross-Encoder Reranker
                                     |
                        Llama-3.2-3B-Instruct (LLM)
                                     |
                         Cited, grounded answer
```

## Technology Choices (all research-backed)

| Component | Choice | Why |
|-----------|--------|-----|
| Text embedding | BioLORD-2023-C | Top biomedical BEIR benchmark |
| Image embedding | BiomedCLIP | Trained on 15M PubMed image-text pairs |
| Audio | OpenAI Whisper | Best open speech-to-text |
| PDF parsing | PyMuPDF (fitz) | Fastest, most accurate |
| Retrieval | BM25 + Dense + RRF | Hybrid consistently beats either alone |
| Reranker | ms-marco MiniLM | Single biggest accuracy boost per research |
| LLM | Llama-3.2-3B-Instruct | Best open model at this size |

## Datasets

| Modality | Dataset | Size |
|----------|---------|------|
| Text | PMC Open Access (existing) | 30k articles |
| Images | PMC-VQA + ROCO v2 | 10k images |
| Audio | MedQA synthesized | 500 clips |
| PDFs | PMC Open Access PDFs | 200 documents |

## Quick Start (Wahab HPC)

```bash
# 1. Setup (once)
bash scripts/setup_env.sh

# 2. In JupyterHub, select "Python (rag310)" kernel
# 3. Run notebooks in order:
#    01 -> text ingestion (uses existing PMC passages)
#    02 -> image download
#    03 -> audio synthesis + transcription
#    04 -> PDF download + extraction
#    05 -> build all indexes (takes ~1-2 hrs with GPU)
#    06 -> retrieval evaluation + P@10
#    07 -> RAG answer generation
```

## Switching the LLM

Edit one line in `configs/config.py`:

```python
# Options:
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # default
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # better quality
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"          # fastest
```

## Project Structure

```
multimodal_rag/
+-- data/
|   +-- text/         PMC passage parquet (from existing files)
|   +-- images/       PMC-VQA and ROCO v2 images
|   +-- audio/        Synthesized MedQA audio files
|   +-- pdfs/         Downloaded PMC PDFs
|   `-- processed/    Unified JSONL + metadata parquet
+-- indexes/
|   +-- bm25_index/   Pyserini Lucene index
|   +-- faiss_text.bin  BioLORD text vectors
|   `-- faiss_image.bin BiomedCLIP image vectors
+-- src/
|   +-- ingest/       One module per modality
|   +-- retrieval.py  Full hybrid pipeline
|   +-- generation.py LLM answer generation
|   +-- evaluation.py P@10 / NDCG / MRR metrics
|   `-- utils.py      Shared helpers
+-- configs/config.py All settings in one file
+-- notebooks/        01-07 run in order
+-- evaluation/       Queries + relevance judgments
`-- scripts/          HPC setup
```

## Evaluation Metrics

- Precision@10 (P@10) -- required by assignment
- NDCG@10 -- accounts for ranking quality
- MRR (Mean Reciprocal Rank) -- measures first relevant result

Systems compared: BM25 | Dense | Hybrid+Rerank
