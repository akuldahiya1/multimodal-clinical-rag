# RAG Generation Results

## System Configuration
- **LLM:** Llama-3.2-3B-Instruct (GPU)
- **Retrieval:** Hybrid BM25 + BioLORD Dense + RRF Fusion
- **Embeddings:** BioLORD-2023-C (text), BiomedCLIP (images)
- **Query Understanding:** Adaptive routing per query type
- **Dataset:** 679,137 passages across 4 modalities

## Generated Answers

| QID | Query | Type | Answer Preview |
|-----|-------|------|----------------|
| q01 | COVID-19 impact on healthcare systems | general | The COVID-19 pandemic had a profound impact on healthcare systems, with a shift towards online and t... |
| q02 | digital health transformation in primary care | general | Digital health transformation in primary care requires a multi-level, integrated learning capability... |
| q03 | gut microbiota changes during viral infection | general | Changes in the gut microbiota can induce and develop the host’s immune system, allowing it to combat... |
| q04 | chest X-ray findings in pneumonia | visual | Pneumonia is characterized by several visual findings on chest X-ray, including: - Alveolar consolid... |
| q05 | hypertension treatment in primary care settin | clinical | [Finding] -> [Evidence] -> [Clinical implication] Hypertension treatment in primary care settings is... |
| q06 | what are the benefits of telemedicine for rur | factual | Telemedicine can improve access to cost-effective quality care and reduce the risks of diabetes comp... |
| q07 | MRI brain scan interpretation in neurological | visual | MRI brain scans are a crucial diagnostic tool in neurological disorders. The interpretation of MRI b... |
| q08 | how does COVID-19 affect lung tissue | factual | COVID-19 can cause significant damage to lung tissue, leading to impaired respiratory function and p... |
| q09 | antibiotic resistance mechanisms in bacteria | factual | The emergence of antibiotic resistance in bacteria is supported by various factors, including the su... |
| q10 | nurse leadership in digital healthcare innova | general | Nurse leadership plays a fundamental role in establishing a digitally enabled care environment that ... |

## Retrieval Evaluation (P@10)

| System | P@10 | NDCG@10 | MRR |
|--------|------|---------|-----|
| **Hybrid+Rerank** | **0.93** | **0.99** | **1.00** |
| Dense (BioLORD) | 0.89 | 0.94 | 0.88 |
| BM25 (Pyserini) | 0.74 | 0.90 | 0.90 |

## Query Type Distribution

| Query Type | Queries | Retrieval Strategy |
|------------|---------|-------------------|
| general | q01, q02, q03, q10 | BM25=0.35, Dense=0.50, Image=0.15 |
| visual | q04, q07 | BM25=0.20, Dense=0.30, Image=0.50 |
| clinical | q05 | BM25=0.40, Dense=0.50, Image=0.10 |
| factual | q06, q08, q09 | BM25=0.45, Dense=0.50, Image=0.05 |

## Dataset Statistics

| Modality | Source | Passages |
|----------|--------|----------|
| Text | PMC Open Access | 662,362 |
| Images | PMC-VQA + ROCO v2 | 10,000 |
| Audio | MedQA synthesized | 500 |
| PDFs | PMC Open Access | 6,275 |
| **Total** | **All sources** | **679,137** |