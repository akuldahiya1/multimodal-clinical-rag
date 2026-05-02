# Multimodal Clinical RAG System

**CS734 — Introduction to Information Retrieval | Old Dominion University | Spring 2026**
**Student: Akul Dahiya | aakul001@odu.edu**

---

## Results

| System | P@10 | NDCG@10 | MRR |
|--------|------|---------|-----|
| BM25 (baseline) | 0.74 | 0.9022 | 0.9000 |
| Dense (BioLORD) | 0.89 | 0.9415 | 0.8833 |
| **Hybrid + Rerank** | **0.93** | **0.9918** | **1.0000** |

- **Faithfulness**: 99.1% (NLI-verified, 9/10 queries at 100%)
- **Dataset**: 679,137 passages across 4 modalities
- **GitHub**: https://github.com/akuldahiya1/multimodal-clinical-rag

---

## Overview

A hybrid multimodal RAG system for intelligent biomedical search combining
BM25 + Dense retrieval via RRF fusion, cross-encoder reranking, adaptive
query understanding, smart PDF routing, and NLI faithfulness verification.

**Key features:**
- Hybrid BM25 + Dense retrieval fused via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for precision
- Adaptive query understanding (visual / clinical / factual / general)
- Smart PDF routing using embedding-based similarity
- NLI-based faithfulness verification (99.1% mean faithfulness)
- Multilingual output (10 languages via deep-translator)
- Voice input via Whisper ASR
- Medical image retrieval via BiomedCLIP
- Gradio web interface with live public demo

---

## Technology Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Text embedding | BioLORD-2023-C | Top biomedical BEIR benchmark |
| Image embedding | BiomedCLIP | 15M PubMed image-text pairs |
| Audio | OpenAI Whisper | Best open speech-to-text |
| PDF parsing | PyMuPDF (fitz) | Fast and accurate |
| BM25 index | Pyserini / Lucene | Standard IR baseline |
| Dense index | FAISS IndexFlatIP | Efficient cosine similarity |
| Fusion | Reciprocal Rank Fusion | Beats weighted score combination |
| Reranker | MiniLM cross-encoder | Highest precision boost |
| LLM | Llama-3.2-3B-Instruct | Best open model at this size |
| Verification | DeBERTa NLI | Faithfulness checking |
| UI | Gradio | Fast web interface |
| Translation | deep-translator | 10 language output |

---

## Dataset

| Modality | Source | Passages |
|----------|--------|----------|
| Text | PMC Open Access (30k articles) | 662,362 |
| Images | PMC-VQA + ROCO v2 | 10,000 |
| Audio | MedMCQA (gTTS + Whisper) | 500 |
| PDF | PMC E-utilities API | 6,275 |
| **Total** | | **679,137** |

---

## Quick Start (ODU Wahab HPC)

Run notebooks 01 through 08 in order then launch the UI:

```python
exec(open('/home/aakul001/multimodal_rag/app.py').read())
```

---

## Smart PDF Routing

```
similarity >= 0.42  ->  PDF-focused  (PDF=92%, Global=8%)
0.35 - 0.42         ->  Balanced     (PDF=50%, Global=50%)
< 0.35              ->  Global       (PDF=20%, Global=80%)
```

---

## Evaluation Queries

| QID | Query | Type |
|-----|-------|------|
| q01 | COVID-19 impact on healthcare systems | general |
| q02 | Digital health transformation in primary care | general |
| q03 | Gut microbiota changes during viral infection | general |
| q04 | Chest X-ray findings in pneumonia | visual |
| q05 | Hypertension treatment in primary care | clinical |
| q06 | Benefits of telemedicine for rural patients | factual |
| q07 | MRI brain scan interpretation | visual |
| q08 | How does COVID-19 affect lung tissue | factual |
| q09 | Antibiotic resistance mechanisms in bacteria | factual |
| q10 | Nurse leadership in digital healthcare | general |