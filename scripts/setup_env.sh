#!/usr/bin/env bash
# scripts/setup_env.sh
# Run once on Wahab HPC: bash scripts/setup_env.sh

set -e

ENV_NAME="rag310"

echo "=================================================="
echo "  Multimodal RAG -- HPC Setup"
echo "=================================================="

# 1. Create/update conda env
echo "[1/5] Creating conda environment..."
conda create -y -n "$ENV_NAME" python=3.10 || true
conda install -y -n "$ENV_NAME" -c conda-forge openjdk=21 || true

# 2. Register Jupyter kernel
echo "[2/5] Registering kernel..."
~/.conda/envs/"$ENV_NAME"/bin/python -m pip install ipykernel -q
~/.conda/envs/"$ENV_NAME"/bin/python -m ipykernel install \
    --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 3. Install all Python packages
echo "[3/5] Installing Python packages..."
~/.conda/envs/"$ENV_NAME"/bin/pip install --upgrade pip -q
~/.conda/envs/"$ENV_NAME"/bin/pip install \
    pyserini \
    sentence-transformers \
    faiss-cpu \
    transformers \
    accelerate \
    torch \
    torchvision \
    pandas pyarrow numpy tqdm \
    datasets \
    Pillow \
    PyMuPDF \
    gTTS \
    openai-whisper \
    -q

# 4. Create project directories
echo "[4/5] Creating directories..."
mkdir -p ~/multimodal_rag/{data/{text,images,audio,pdfs,processed},indexes,evaluation,results}

# 5. Copy project code
echo "[5/5] Copying project..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp -r "$SCRIPT_DIR/.." ~/multimodal_rag 2>/dev/null || true

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "  Kernel: Python ($ENV_NAME)"
echo "  Run notebooks 01 -> 07 in order"
echo "=================================================="
