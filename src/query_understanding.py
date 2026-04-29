"""
src/query_understanding.py
==========================
Upgrade 1: Adaptive query understanding and retrieval routing.

Instead of fixed weights, this module:
1. Classifies the query type (factual, visual, comparative, clinical)
2. Extracts medical entities and intent
3. Returns dynamic retrieval weights per query

This makes the system feel intelligent -- different queries
trigger different retrieval strategies automatically.
"""

import re
from src.utils import get_logger

logger = get_logger("query_understanding")


#  Query type definitions 

QUERY_TYPES = {
    "visual": {
        "description": "Query asking about images, scans, X-rays, figures",
        "keywords": [
            "image", "scan", "x-ray", "xray", "mri", "ct scan", "ultrasound",
            "figure", "diagram", "photograph", "picture", "show", "look like",
            "appearance", "finding", "visible", "radiograph", "histology",
            "microscopy", "biopsy", "lesion", "tumor", "mass"
        ],
        "weights": {"bm25": 0.2, "dense_text": 0.3, "dense_image": 0.5},
        "top_k_image": 10,
    },
    "clinical": {
        "description": "Query about treatment, diagnosis, patient care",
        "keywords": [
            "treatment", "therapy", "diagnosis", "symptom", "patient",
            "clinical", "dose", "drug", "medication", "side effect",
            "prognosis", "outcome", "guideline", "protocol", "management",
            "surgery", "procedure", "intervention", "complication", "risk"
        ],
        "weights": {"bm25": 0.4, "dense_text": 0.5, "dense_image": 0.1},
        "top_k_image": 3,
    },
    "comparative": {
        "description": "Query comparing two or more things",
        "keywords": [
            "compare", "versus", "vs", "difference between", "better than",
            "worse than", "contrast", "similar", "advantage", "disadvantage",
            "which is", "what is the difference", "more effective", "less effective"
        ],
        "weights": {"bm25": 0.35, "dense_text": 0.55, "dense_image": 0.1},
        "top_k_image": 3,
    },
    "factual": {
        "description": "Direct factual or definition question",
        "keywords": [
            "what is", "what are", "define", "definition", "explain",
            "how does", "mechanism", "pathway", "cause", "effect",
            "function", "role", "why", "when was", "who discovered"
        ],
        "weights": {"bm25": 0.45, "dense_text": 0.50, "dense_image": 0.05},
        "top_k_image": 2,
    },
}

DEFAULT_WEIGHTS = {"bm25": 0.35, "dense_text": 0.50, "dense_image": 0.15}


#  Medical entity extraction 

MEDICAL_TERMS = {
    "diseases": [
        "cancer", "diabetes", "hypertension", "covid", "alzheimer", "parkinson",
        "asthma", "pneumonia", "sepsis", "stroke", "myocardial", "infarction",
        "tuberculosis", "hiv", "aids", "hepatitis", "arthritis", "lupus"
    ],
    "anatomy": [
        "lung", "heart", "liver", "kidney", "brain", "pancreas", "thyroid",
        "colon", "breast", "prostate", "blood", "bone", "muscle", "nerve"
    ],
    "procedures": [
        "surgery", "biopsy", "transplant", "chemotherapy", "radiation",
        "immunotherapy", "dialysis", "catheter", "endoscopy", "mri", "ct"
    ],
    "drugs": [
        "aspirin", "metformin", "insulin", "antibiotic", "vaccine", "steroid",
        "antiviral", "antifungal", "analgesic", "statin", "beta blocker"
    ],
}


def extract_medical_entities(query: str) -> dict:
    """
    Extract medical entities from a query using keyword matching.
    Returns dict: {entity_type: [found_terms]}
    """
    query_lower = query.lower()
    found = {}

    for entity_type, terms in MEDICAL_TERMS.items():
        matches = [t for t in terms if t in query_lower]
        if matches:
            found[entity_type] = matches

    return found


def classify_query(query: str) -> dict:
    """
    Classify a query and return its type with confidence scores.

    Returns:
        {
            "query":        original query,
            "type":         "visual" | "clinical" | "comparative" | "factual" | "general",
            "confidence":   float,
            "all_scores":   {type: score},
            "entities":     {entity_type: [terms]},
            "weights":      {bm25, dense_text, dense_image},
            "top_k_image":  int,
            "explanation":  str,
        }
    """
    query_lower = query.lower()

    # Score each query type by keyword matches
    scores = {}
    for qtype, info in QUERY_TYPES.items():
        matches = sum(1 for kw in info["keywords"] if kw in query_lower)
        # Normalize by number of keywords
        scores[qtype] = matches / len(info["keywords"])

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Only assign a type if score is above threshold
    if best_score < 0.01:
        best_type = "general"
        weights   = DEFAULT_WEIGHTS
        top_k_img = 5
        explanation = "General biomedical query. Using balanced retrieval weights."
    else:
        weights   = QUERY_TYPES[best_type]["weights"]
        top_k_img = QUERY_TYPES[best_type]["top_k_image"]
        explanation = (
            f"Detected as '{best_type}' query. "
            f"Adjusting weights: BM25={weights['bm25']}, "
            f"Dense={weights['dense_text']}, "
            f"Image={weights['dense_image']}."
        )

    entities = extract_medical_entities(query)

    result = {
        "query":       query,
        "type":        best_type,
        "confidence":  round(best_score, 4),
        "all_scores":  {k: round(v, 4) for k, v in scores.items()},
        "entities":    entities,
        "weights":     weights,
        "top_k_image": top_k_img,
        "explanation": explanation,
    }

    logger.info(
        f"Query type: [{best_type}] confidence={best_score:.3f} | "
        f"weights={weights} | entities={list(entities.keys())}"
    )

    return result


def expand_query(query: str, entities: dict) -> str:
    """
    Expand a query with related medical terms for better BM25 recall.
    Simple but effective -- adds synonyms for detected entities.
    """
    expansions = {
        "myocardial infarction": "myocardial infarction heart attack MI",
        "hypertension":          "hypertension high blood pressure HTN",
        "diabetes":              "diabetes mellitus DM blood glucose insulin",
        "covid":                 "COVID-19 SARS-CoV-2 coronavirus",
        "cancer":                "cancer tumor malignancy neoplasm carcinoma",
        "stroke":                "stroke cerebrovascular accident CVA brain",
        "alzheimer":             "alzheimer dementia cognitive decline",
        "pneumonia":             "pneumonia lung infection pulmonary",
    }

    query_lower = query.lower()
    expanded    = query

    for term, expansion in expansions.items():
        if term in query_lower and expansion not in expanded:
            expanded = expanded + " " + expansion

    if expanded != query:
        logger.info(f"Query expanded: '{query}' -> '{expanded[:100]}...'")

    return expanded
