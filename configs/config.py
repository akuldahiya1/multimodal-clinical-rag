"""
configs/config.py
=================
Single source of truth for the entire Multimodal Clinical RAG project.
Change here, nothing else needs updating.
"""

from pathlib import Path

#  Root paths 
PROJECT_ROOT = Path.home() / "multimodal_rag"

# Where your existing PMC work lives (from previous project)
LEGACY_PMC_ROOT = Path.home() / "rag_project"

#  Data paths 
DATA_DIR           = PROJECT_ROOT / "data"
TEXT_DIR           = DATA_DIR / "text"
IMAGE_DIR          = DATA_DIR / "images"
AUDIO_DIR          = DATA_DIR / "audio"
PDF_DIR            = DATA_DIR / "pdfs"
PROCESSED_DIR      = DATA_DIR / "processed"

# Processed unified index files (one JSONL per modality)
TEXT_JSONL         = PROCESSED_DIR / "text_passages.jsonl"
IMAGE_JSONL        = PROCESSED_DIR / "image_passages.jsonl"
AUDIO_JSONL        = PROCESSED_DIR / "audio_passages.jsonl"
PDF_JSONL          = PROCESSED_DIR / "pdf_passages.jsonl"
UNIFIED_JSONL      = PROCESSED_DIR / "unified_passages.jsonl"
METADATA_PARQUET   = PROCESSED_DIR / "metadata.parquet"

#  Index paths 
INDEXES_DIR        = PROJECT_ROOT / "indexes"
BM25_INDEX_DIR     = INDEXES_DIR / "bm25_index"
BM25_INPUT_DIR     = INDEXES_DIR / "bm25_input"

# Separate FAISS indexes per modality (unified search merges them)
FAISS_TEXT_PATH    = INDEXES_DIR / "faiss_text.bin"
FAISS_IMAGE_PATH   = INDEXES_DIR / "faiss_image.bin"
FAISS_IDS_TEXT     = INDEXES_DIR / "faiss_ids_text.json"
FAISS_IDS_IMAGE    = INDEXES_DIR / "faiss_ids_image.json"

#  HPC / Java 
CONDA_ENV          = "rag310"
JAVA_HOME = Path.home() / ".conda/envs/rag310/lib/jvm"
JVM_PATH  = Path.home() / ".conda/envs/rag310/lib/jvm/lib/server/libjvm.so"

#  Text data (existing PMC) 
# Point to your already-built passages parquet
LEGACY_PASSAGES    = LEGACY_PMC_ROOT / "data_final" / "pmc_30k_passages.parquet"
PMC_FILE_LIST_URL  = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
PMC_BASE_URL       = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"

CHUNK_SIZE         = 250    # words per text chunk
CHUNK_OVERLAP      = 50     # word overlap
MIN_CHUNK_CHARS    = 100    # discard shorter chunks

#  Image data 
# PMC-VQA from HuggingFace
PMCVQA_HF_DATASET  = "xmcmic/PMC-VQA"
PMCVQA_SAMPLE      = 5000    # how many image-QA pairs to use

# ROCO v2 (radiology images + captions)
ROCO_HF_DATASET    = "eltorio/ROCOv2-radiology"
ROCO_SAMPLE        = 5000

#  Audio data 
# We synthesize audio from MedQA text questions using gTTS (free)
MEDQA_HF_DATASET   = "bigbio/med_qa"
AUDIO_SAMPLE       = 500     # number of questions to synthesize as audio
WHISPER_MODEL      = "openai/whisper-base"   # runs on CPU or GPU fine

#  PDF data 
# PubMed Central open access PDFs (same source as text, different format)
PDF_SAMPLE         = 200     # number of PDFs to download and parse

#  Embedding models 
# Best biomedical text model per current benchmarks
TEXT_EMBED_MODEL   = "FremyCompany/BioLORD-2023-C"
TEXT_EMBED_DIM     = 768
EMBED_BATCH_SIZE   = 64

# Best biomedical image+text model (trained on 15M PubMed image-text pairs)
IMAGE_EMBED_MODEL  = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
IMAGE_EMBED_DIM    = 512

#  Retrieval settings 
BM25_TOP_K         = 50     # BM25 candidate pool
DENSE_TOP_K        = 50     # FAISS candidate pool
RRF_K              = 60     # RRF constant (standard value from original paper)
FINAL_TOP_K        = 10     # results returned after reranking

# Cross-encoder reranker (single most impactful upgrade per research)
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N       = 50     # how many candidates to rerank

#  LLM 
# Swap to change model -- nothing else needs changing
#
# Options (in order of quality/size):
#   "meta-llama/Llama-3.2-3B-Instruct"         recommended -- good balance
#   "mistralai/Mistral-7B-Instruct-v0.2"        best quality, needs ~14GB GPU
#   "Qwen/Qwen2.5-1.5B-Instruct"               fastest, smallest
#
LLM_MODEL          = "meta-llama/Llama-3.2-3B-Instruct"
LLM_MAX_INPUT      = 3072   # tokens
LLM_MAX_OUTPUT     = 300    # tokens
LLM_CONTEXT_DOCS   = 5      # number of retrieved passages sent to LLM

#  Evaluation 
EVAL_DIR           = PROJECT_ROOT / "evaluation"
QUERIES_FILE       = EVAL_DIR / "queries.jsonl"
JUDGMENTS_FILE     = EVAL_DIR / "relevance_judgments.csv"
RESULTS_DIR        = PROJECT_ROOT / "results"
EVAL_TOP_K         = 10     # P@10
