"""
src/prompts.py
==============
Upgrade 5: Advanced RAG prompting strategies.

Replaces the basic single-prompt approach with:
1. Structured chain-of-thought reasoning prompt
2. Query-type-specific prompt templates
3. Citation-aware prompting
4. Evidence quality assessment in prompt

Research shows structured prompts improve answer quality
significantly for biomedical QA tasks.
"""


#  Prompt templates per query type 

STRUCTURED_REASONING_PROMPT = """You are an expert biomedical research assistant.
Your task is to answer a medical question using ONLY the provided evidence.

Follow these steps in your response:

Step 1 - Evidence Summary: Briefly note what each source says about the topic.
Step 2 - Synthesis: Identify agreements and contradictions across sources.
Step 3 - Answer: Write a final concise answer (2-3 sentences) with citations like [Source 1].

Evidence:
{context}

Question: {query}

Response:
Step 1 - Evidence Summary:"""


VISUAL_QUERY_PROMPT = """You are an expert radiologist and biomedical imaging specialist.
Answer the question using the provided text descriptions and image captions.
When referencing image findings, cite the source like [Source 1 | IMAGE].
Be specific about visual findings -- mention location, size, appearance when available.

Evidence (includes image captions):
{context}

Question: {query}

Answer (focus on visual findings):"""


CLINICAL_QUERY_PROMPT = """You are a clinical medicine expert following evidence-based guidelines.
Answer the clinical question using ONLY the provided research evidence.
Structure your answer as: [Finding] -> [Evidence] -> [Clinical implication].
Always cite sources like [Source 1]. If evidence is limited, say so explicitly.

Clinical Evidence:
{context}

Clinical Question: {query}

Evidence-based Answer:"""


COMPARATIVE_QUERY_PROMPT = """You are a biomedical research analyst.
Compare the options mentioned in the question using the provided evidence.
Structure: 1) Option A findings, 2) Option B findings, 3) Comparison verdict.
Cite sources throughout using [Source N].

Evidence:
{context}

Comparison Question: {query}

Structured Comparison:"""


FACTUAL_QUERY_PROMPT = """You are a biomedical knowledge expert.
Answer the factual question directly and precisely using the provided evidence.
Start with a one-sentence direct answer, then provide supporting detail.
Cite sources using [Source N]. Do not speculate beyond the evidence.

Evidence:
{context}

Question: {query}

Direct Answer:"""


#  Prompt builder 

def build_context(passages: list, max_chars_per_passage: int = 500) -> str:
    """Build a formatted context string from retrieved passages."""
    parts = []
    for i, p in enumerate(passages, 1):
        modality = p.get("modality", "text").upper()
        text     = p.get("contents", "")[:max_chars_per_passage]
        title    = p.get("title", "")

        header = f"[Source {i} | {modality}]"
        if title:
            header += f" {title[:60]}"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def build_prompt(
    query:       str,
    passages:    list,
    query_type:  str = "general",
    use_cot:     bool = True,
) -> str:
    """
    Build the best prompt for a given query type.

    Args:
        query:      The user question.
        passages:   Retrieved passages (list of dicts).
        query_type: Output of classify_query()["type"].
        use_cot:    Whether to use chain-of-thought reasoning.

    Returns:
        Formatted prompt string ready to send to LLM.
    """
    context = build_context(passages)

    # Select template based on query type
    if query_type == "visual":
        template = VISUAL_QUERY_PROMPT
    elif query_type == "clinical":
        template = CLINICAL_QUERY_PROMPT
    elif query_type == "comparative":
        template = COMPARATIVE_QUERY_PROMPT
    elif query_type == "factual":
        template = FACTUAL_QUERY_PROMPT
    elif use_cot:
        template = STRUCTURED_REASONING_PROMPT
    else:
        # Simple fallback
        template = (
            "You are a biomedical research assistant.\n"
            "Answer the question using ONLY the provided evidence.\n"
            "Cite sources using [Source N].\n\n"
            "Evidence:\n{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )

    return template.format(context=context, query=query)


def clean_cot_answer(full_output: str, query_type: str = "general") -> str:
    """
    Extract the final answer from a chain-of-thought response.
    Removes the reasoning steps, keeps only the final answer.
    """
    text = full_output

    # Remove everything before "Answer:" in structured responses
    for marker in ["Step 3 - Answer:", "Answer:", "Direct Answer:",
                   "Evidence-based Answer:", "Structured Comparison:",
                   "[/INST]", "<|assistant|>"]:
        if marker in text:
            text = text.split(marker)[-1]

    text = text.replace("\n", " ").strip()

    # Trim to last complete sentence
    # Remove LLM artifacts
    for artifact in ["Best regards", "Let me know", "Please let me know",
                     "Radiologist &", "[Your Name]", "---\n", "Note:"]:
        if artifact in text:
            text = text[:text.find(artifact)].strip()

    # Remove common LLM artifacts that leak into answers
    artifacts = [
        "Best regards", "Let me know", "Please let me know",
        "Radiologist &", "[Your Name]", "Note:", "Also, I can",
        "I can provide", "If you need", "---"
    ]
    for artifact in artifacts:
        if artifact in text:
            text = text[:text.find(artifact)].strip()

    if "." in text:
        text = text[: text.rfind(".") + 1]

    return text.strip() or "Insufficient evidence in retrieved passages."
