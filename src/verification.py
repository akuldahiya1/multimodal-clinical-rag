"""
src/verification.py
===================
Upgrade 2: Answer verification and faithfulness scoring.

After generating an answer, this module:
1. Breaks the answer into individual claims
2. Checks each claim against retrieved passages (NLI-based)
3. Returns a confidence score and flags unsupported claims

This is critical for biomedical reliability.
Uses a lightweight NLI model -- no extra GPU memory needed.
"""

import re
from src.utils import get_logger

logger = get_logger("verification")

_nli_model  = None
_nli_tokenizer = None


def load_nli_model():
    """
    Load a lightweight NLI model for claim verification.
    Uses DeBERTa-v3-small-mnli -- fast and accurate.
    """
    global _nli_model, _nli_tokenizer
    if _nli_model is not None:
        return _nli_tokenizer, _nli_model

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "cross-encoder/nli-deberta-v3-small"
    logger.info(f"Loading NLI model: {model_name}")

    _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _nli_model     = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nli_model = _nli_model.to(device)
    _nli_model.eval()

    logger.info(f"NLI model ready on {device}")
    return _nli_tokenizer, _nli_model


def extract_claims(answer: str) -> list:
    """
    Split an answer into individual verifiable claims.
    Uses sentence splitting -- each sentence is one claim.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    # Filter out very short fragments
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    return claims


def check_claim_against_passage(claim: str, passage: str) -> dict:
    """
    Check if a claim is supported by a passage using NLI.

    Returns:
        {
            "label":      "entailment" | "neutral" | "contradiction",
            "confidence": float,
            "supported":  bool,
        }
    """
    import torch

    tokenizer, model = load_nli_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        passage, claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    # DeBERTa NLI labels: 0=contradiction, 1=neutral, 2=entailment
    labels = ["contradiction", "neutral", "entailment"]
    best_idx   = probs.index(max(probs))
    best_label = labels[best_idx]

    return {
        "label":      best_label,
        "confidence": round(max(probs), 4),
        "supported":  best_label == "entailment",
    }


def verify_answer(answer: str, passages: list, top_n_passages: int = 3) -> dict:
    """
    Verify an answer against its source passages.

    Args:
        answer:         Generated answer string.
        passages:       List of passage dicts (with "contents" key).
        top_n_passages: How many passages to check each claim against.

    Returns:
        {
            "overall_score":     float (0-1, higher = more faithful),
            "supported_claims":  int,
            "total_claims":      int,
            "claim_results":     list of per-claim dicts,
            "verdict":           "faithful" | "partially_faithful" | "unfaithful",
            "unsupported":       list of unsupported claim strings,
        }
    """
    claims = extract_claims(answer)
    if not claims:
        return {
            "overall_score": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_results": [],
            "verdict": "unfaithful",
            "unsupported": [],
        }

    passage_texts = [p.get("contents", "") for p in passages[:top_n_passages]]
    claim_results = []
    supported_count = 0

    for claim in claims:
        # Check claim against all passages, take best result
        best_result = {"label": "neutral", "confidence": 0.0, "supported": False}

        for passage in passage_texts:
            if not passage:
                continue
            result = check_claim_against_passage(claim, passage[:1000])
            if result["confidence"] > best_result["confidence"]:
                best_result = result

        claim_results.append({
            "claim":     claim,
            "supported": best_result["supported"],
            "label":     best_result["label"],
            "confidence": best_result["confidence"],
        })

        if best_result["supported"]:
            supported_count += 1

    overall_score = supported_count / len(claims) if claims else 0.0

    if overall_score >= 0.8:
        verdict = "faithful"
    elif overall_score >= 0.5:
        verdict = "partially_faithful"
    else:
        verdict = "unfaithful"

    unsupported = [
        r["claim"] for r in claim_results if not r["supported"]
    ]

    logger.info(
        f"Verification: {supported_count}/{len(claims)} claims supported | "
        f"score={overall_score:.2f} | verdict={verdict}"
    )

    return {
        "overall_score":    round(overall_score, 4),
        "supported_claims": supported_count,
        "total_claims":     len(claims),
        "claim_results":    claim_results,
        "verdict":          verdict,
        "unsupported":      unsupported,
    }


def format_verification_report(verification: dict) -> str:
    """Format verification results as a readable string."""
    lines = [
        f"Faithfulness Score: {verification['overall_score']:.0%}",
        f"Verdict: {verification['verdict'].upper()}",
        f"Claims: {verification['supported_claims']}/{verification['total_claims']} supported",
        "",
    ]

    for r in verification["claim_results"]:
        icon = "[OK]" if r["supported"] else "[?]"
        lines.append(f"  {icon} {r['claim'][:100]}")

    if verification["unsupported"]:
        lines.append("\nUnsupported claims (verify manually):")
        for claim in verification["unsupported"]:
            lines.append(f"  - {claim[:100]}")

    return "\n".join(lines)
