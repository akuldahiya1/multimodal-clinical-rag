"""
src/ingest/audio.py
===================
Two modes:
  1. SYNTHESIZE: Take MedQA text questions and convert to speech with gTTS.
     Realistic for a course project -- voice queries are a legitimate use case.
  2. TRANSCRIBE: Use OpenAI Whisper to convert any audio file to text.

Pipeline:
  MedQA questions -> gTTS audio files -> Whisper transcription -> passages
"""

import os
from pathlib import Path

from src.utils import get_logger, save_jsonl, passage_record, clean_text

logger = get_logger("ingest.audio")


def synthesize_audio_from_medqa(audio_dir: Path, sample: int = 500) -> list:
    """
    Download MedQA questions from HuggingFace and synthesize audio with gTTS.
    This creates realistic voice query data.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("Install gTTS: pip install gTTS")

    from datasets import load_dataset

    logger.info(f"Downloading MedQA questions (sample={sample})...")
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)

    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i, item in enumerate(ds):
        if i >= sample:
            break

        doc_id   = f"medqa_audio_{i:04d}"
        question = item.get("question", "") or ""
        if not question.strip():
            continue

        audio_path = audio_dir / f"{doc_id}.mp3"

        # Synthesize speech if not already done
        if not audio_path.exists():
            try:
                tts = gTTS(text=question, lang="en", slow=False)
                tts.save(str(audio_path))
            except Exception as e:
                logger.warning(f"gTTS failed for {doc_id}: {e}")
                continue

        # Immediately transcribe with Whisper
        transcript = transcribe_audio(audio_path)
        if not transcript:
            transcript = question  # fallback to original text

        records.append(passage_record(
            doc_id     = doc_id,
            text       = transcript,
            modality   = "audio",
            source     = "medqa_synthesized",
            audio_path = str(audio_path),
            original_question = question,
        ))

        if (i + 1) % 100 == 0:
            logger.info(f"  Audio: {i+1}/{sample}")

    logger.info(f"Audio ingestion done: {len(records)} records")
    return records


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe a single audio file using OpenAI Whisper.
    Returns the transcription string.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Install whisper: pip install openai-whisper")

    try:
        model = _get_whisper_model()
        result = model.transcribe(str(audio_path), fp16=False)
        return clean_text(result.get("text", ""))
    except Exception as e:
        logger.warning(f"Transcription failed for {audio_path}: {e}")
        return ""


_whisper_model = None

def _get_whisper_model():
    """Load Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    import whisper
    from configs.config import WHISPER_MODEL

    # Extract size from model name
    model_size = WHISPER_MODEL.split("/")[-1].replace("whisper-", "")
    logger.info(f"Loading Whisper model: {model_size}")
    _whisper_model = whisper.load_model(model_size)
    logger.info("Whisper loaded")
    return _whisper_model


def transcribe_directory(audio_dir: Path) -> list:
    """Transcribe all audio files in a directory."""
    records = []
    for i, audio_file in enumerate(audio_dir.glob("*.mp3")):
        doc_id     = audio_file.stem
        transcript = transcribe_audio(audio_file)
        if transcript:
            records.append(passage_record(
                doc_id     = doc_id,
                text       = transcript,
                modality   = "audio",
                source     = "audio_directory",
                audio_path = str(audio_file),
            ))
        if (i + 1) % 50 == 0:
            logger.info(f"  Transcribed {i+1} files")
    return records


def ingest_audio(config) -> int:
    """Main entry point for audio ingestion."""
    from configs.config import AUDIO_JSONL, AUDIO_DIR, AUDIO_SAMPLE

    output_path = Path(AUDIO_JSONL)
    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        logger.info(f"Audio passages already exist ({existing:,}). Skipping.")
        return existing

    audio_dir = Path(AUDIO_DIR)
    records   = synthesize_audio_from_medqa(audio_dir, sample=AUDIO_SAMPLE)

    from configs.config import AUDIO_JSONL
    save_jsonl(records, output_path)
    logger.info(f"Total audio passages: {len(records):,} -> {output_path}")
    return len(records)
