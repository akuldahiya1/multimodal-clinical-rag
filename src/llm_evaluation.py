"""
src/llm_evaluation.py
=====================
Upgrade 4: LLM-as-judge evaluation.

Evaluates generated answers on:
1. Faithfulness  -- is the answer grounded in the retrieved passages?
2. Relevance     -- does the answer actually address the question?
3. Completeness  -- does it cover the key aspects of the question?
4. Citation quality -- does it correctly use the sources?

Uses the same LLM as generation (no extra model needed).
This is a research-standard evaluation approach.
"""

import json
from src.utils import get_logger

logger = get_logger("llm_evaluation")


JUDGE_PROMPT = """You are an expert biomedical research evaluator.
Evaluate the following answer to a biomedical question.

Question: {query}

Retrieved Context:
{context}

Generated Answer: {answer}

Rate the answer on these 4 dimensions (score 1-5 each):

1. FAITHFULNESS: Is every claim in the answer supported by the context? 
   (5=fully grounded, 1=mostly hallucinated)

2. RELEVANCE: Does the answer directly address the question?
   (5=perfectly on-topic, 1=completely off-topic)

3. COMPLETENESS: Does the answer cover the key aspects of the question?
   (5=comprehensive, 1=barely addresses the question)

4. CITATION_QUALITY: Does the answer correctly reference the sources?
   (5=excellent citations, 1=no citations or wrong citations)

Respond ONLY with a JSON object in exactly this format:
{{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "citation_quality": <1-5>,
  "overall": <average of above>,
  "reasoning": "<one sentence explaining the scores>"
}}"""


def evaluate_answer_llm(
    query:        str,
    answer:       str,
    passages:     list,
    n_context:    int = 3,
) -> dict:
    """
    Use the LLM to judge the quality of a generated answer.

    Args:
        query:      The original question.
        answer:     The generated answer to evaluate.
        passages:   Retrieved passages used for generation.
        n_context:  How many passages to include in judge context.

    Returns:
        dict with scores for each dimension + overall + reasoning
    """
    from src.generation import load_llm
    import torch

    context = "\n\n".join(
        f"[Source {i+1}] {p.get('contents', '')[:400]}"
        for i, p in enumerate(passages[:n_context])
    )

    prompt = JUDGE_PROMPT.format(
        query   = query,
        context = context,
        answer  = answer,
    )

    tokenizer, model = load_llm()
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the JSON part
    scores = _parse_judge_output(full_text)
    logger.info(
        f"LLM judge scores: faithfulness={scores.get('faithfulness')} "
        f"relevance={scores.get('relevance')} "
        f"overall={scores.get('overall')}"
    )
    return scores


def _parse_judge_output(text: str) -> dict:
    """Parse JSON scores from LLM judge output."""
    default = {
        "faithfulness":    0,
        "relevance":       0,
        "completeness":    0,
        "citation_quality":0,
        "overall":         0,
        "reasoning":       "Could not parse judge output.",
    }

    try:
        # Find JSON block in output
        if "{" in text and "}" in text:
            json_str = text[text.rfind("{") : text.rfind("}") + 1]
            scores   = json.loads(json_str)

            # Compute overall if not provided
            dims = ["faithfulness", "relevance", "completeness", "citation_quality"]
            if "overall" not in scores:
                vals = [scores.get(d, 0) for d in dims if scores.get(d, 0) > 0]
                scores["overall"] = round(sum(vals) / len(vals), 2) if vals else 0

            return scores
    except Exception as e:
        logger.warning(f"Could not parse judge output: {e}")

    return default


def evaluate_batch(results: list, n_context: int = 3) -> list:
    """
    Run LLM-as-judge evaluation over a batch of RAG results.

    Args:
        results: List of dicts from generate_batch(), each with
                 keys: qid, query, answer, passages_used
        n_context: Number of passages to show the judge.

    Returns:
        List of result dicts with added "evaluation" key.
    """
    evaluated = []

    for r in results:
        logger.info(f"Evaluating {r['qid']}...")
        scores = evaluate_answer_llm(
            query    = r["query"],
            answer   = r["answer"],
            passages = r.get("passages_used", []),
            n_context= n_context,
        )
        r_copy = dict(r)
        r_copy["evaluation"] = scores
        evaluated.append(r_copy)

    return evaluated


def print_evaluation_summary(evaluated: list) -> None:
    """Print a summary table of LLM judge scores."""
    dims = ["faithfulness", "relevance", "completeness", "citation_quality", "overall"]

    print("\n" + "=" * 65)
    print("  LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  {'Query':<35} {'Faith':>5} {'Rel':>5} {'Comp':>5} {'Cite':>5} {'Avg':>5}")
    print("-" * 65)

    totals = {d: 0.0 for d in dims}

    for r in evaluated:
        ev  = r.get("evaluation", {})
        q   = r["query"][:35]
        fa  = ev.get("faithfulness",    0)
        re  = ev.get("relevance",       0)
        co  = ev.get("completeness",    0)
        ci  = ev.get("citation_quality",0)
        ov  = ev.get("overall",         0)
        print(f"  {q:<35} {fa:>5} {re:>5} {co:>5} {ci:>5} {ov:>5.1f}")

        for d in dims:
            totals[d] += ev.get(d, 0)

    n = len(evaluated)
    if n > 0:
        print("-" * 65)
        avgs = {d: round(totals[d] / n, 2) for d in dims}
        print(
            f"  {'AVERAGE':<35} "
            f"{avgs['faithfulness']:>5} "
            f"{avgs['relevance']:>5} "
            f"{avgs['completeness']:>5} "
            f"{avgs['citation_quality']:>5} "
            f"{avgs['overall']:>5.1f}"
        )
    print("=" * 65)
