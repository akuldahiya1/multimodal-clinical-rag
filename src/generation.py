"""
src/generation.py
=================
RAG answer generation -- upgraded with:
- Query-type-aware prompting
- Structured chain-of-thought reasoning
- Answer verification (faithfulness scoring)
- LLM-as-judge evaluation
"""

import os
import torch
import pandas as pd
from src.utils import get_logger

logger = get_logger("generation")

_tokenizer = None
_model     = None


def load_llm():
    global _tokenizer, _model
    if _tokenizer is not None:
        return _tokenizer, _model

    from configs.config import LLM_MODEL
    from transformers import AutoTokenizer, AutoModelForCausalLM

    token  = os.environ.get("HF_TOKEN", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading LLM: {LLM_MODEL} | device={device}")
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=token)
    _model     = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=token,
    )
    if device == "cpu":
        _model = _model.to(device)
    _model.eval()
    logger.info("LLM ready")
    return _tokenizer, _model


def generate_answer(
    query:          str,
    retrieval_df:   pd.DataFrame,
    n_passages:     int  = None,
    query_type:     str  = "general",
    verify:         bool = False,
) -> dict:
    """
    Generate a cited, grounded answer using advanced prompting.

    Args:
        query:        The biomedical question.
        retrieval_df: DataFrame from retrieve().
        n_passages:   Number of passages to use as context.
        query_type:   From classify_query()["type"] -- drives prompt selection.
        verify:       Whether to run faithfulness verification.

    Returns:
        dict: {query, answer, passages_used, modalities_used,
               query_type, verification (if verify=True)}
    """
    from configs.config import LLM_CONTEXT_DOCS, LLM_MAX_INPUT, LLM_MAX_OUTPUT
    from src.prompts import build_prompt, clean_cot_answer

    n        = n_passages or LLM_CONTEXT_DOCS
    top_rows = retrieval_df.head(n).to_dict("records")

    # Build query-type-aware prompt
    prompt = build_prompt(query, top_rows, query_type=query_type, use_cot=True)

    tok, model = load_llm()
    device     = next(model.parameters()).device

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LLM_MAX_INPUT,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_OUTPUT,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    full_text = tok.decode(out[0], skip_special_tokens=True)
    answer    = clean_cot_answer(full_text, query_type)
    modalities= list(set(r.get("modality", "text") for r in top_rows))

    result = {
        "query":           query,
        "answer":          answer,
        "passages_used":   top_rows,
        "modalities_used": modalities,
        "query_type":      query_type,
    }

    # Optional faithfulness verification
    if verify:
        from src.verification import verify_answer
        result["verification"] = verify_answer(answer, top_rows)
        logger.info(
            f"Faithfulness: {result['verification']['verdict']} "
            f"({result['verification']['overall_score']:.0%})"
        )

    return result


def generate_batch(
    queries:      list,
    retrieval_fn,
    n_passages:   int  = None,
    top_k:        int  = 10,
    verify:       bool = False,
    use_query_understanding: bool = True,
) -> list:
    """
    Generate answers for a list of queries with full pipeline.

    Args:
        queries:     List of {qid, query} dicts.
        retrieval_fn: Callable(query, top_k) -> DataFrame.
        n_passages:  Context window.
        top_k:       Retrieval depth.
        verify:      Run faithfulness verification.
        use_query_understanding: Auto-detect query type and adapt retrieval.
    """
    results = []

    for q in queries:
        qid   = q["qid"]
        query = q["query"]
        logger.info(f"Processing {qid}: {query[:60]}...")

        # Query understanding
        query_type = "general"
        if use_query_understanding:
            from src.query_understanding import classify_query
            analysis   = classify_query(query)
            query_type = analysis["type"]
            logger.info(f"  Query type: {query_type} | {analysis['explanation']}")

        # Retrieval
        df = retrieval_fn(query, top_k=top_k)

        # Generation with advanced prompting
        result = generate_answer(
            query      = query,
            retrieval_df = df,
            n_passages = n_passages,
            query_type = query_type,
            verify     = verify,
        )
        result["qid"] = qid
        results.append(result)
        logger.info(f"  Done: {qid}")

    return results
