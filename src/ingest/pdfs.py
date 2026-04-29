
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from src.utils import get_logger, save_jsonl, passage_record, chunk_words, clean_text

logger = get_logger("ingest.pdfs")


def fetch_pmc_article_ids(n=300):
    logger.info(f"Fetching {n} PMC article IDs from NCBI...")
    params = urllib.parse.urlencode({
        "db": "pmc",
        "term": "open access[filter]",
        "retmax": n,
        "retmode": "json",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
        ids = data["esearchresult"]["idlist"]
        logger.info(f"Got {len(ids)} PMC IDs")
        return ids
    except Exception as e:
        logger.warning(f"Fetch failed: {e}")
        return []


def fetch_article_xml(pmc_id):
    import xml.etree.ElementTree as ET
    params = urllib.parse.urlencode({
        "db": "pmc",
        "id": pmc_id,
        "retmode": "xml",
        "rettype": "full",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            xml_data = r.read()
        root = ET.fromstring(xml_data)
        title = ""
        t = root.find(".//article-title")
        if t is not None:
            title = " ".join(t.itertext()).strip()
        parts = []
        for tag in [".//abstract", ".//body"]:
            for el in root.findall(tag):
                parts.append(" ".join(el.itertext()))
        full_text = " ".join(parts).strip()
        full_text = " ".join(full_text.split())
        return title, full_text
    except Exception as e:
        logger.debug(f"Failed PMC{pmc_id}: {e}")
        return "", ""


def ingest_pdfs(config):
    from configs.config import PDF_JSONL, PDF_DIR, PDF_SAMPLE
    from configs.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS

    output_path = Path(PDF_JSONL)
    if output_path.exists():
        output_path.unlink()

    pdf_dir = Path(PDF_DIR)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pmc_ids = fetch_pmc_article_ids(n=PDF_SAMPLE * 3)
    if not pmc_ids:
        logger.warning("No IDs found.")
        return 0

    records = []
    fetched = 0

    for pmc_id in pmc_ids:
        if fetched >= PDF_SAMPLE:
            break
        cache = pdf_dir / f"PMC{pmc_id}.txt"
        if cache.exists():
            full_text = cache.read_text(encoding="utf-8")
            title = ""
        else:
            title, full_text = fetch_article_xml(pmc_id)
            if not full_text or len(full_text) < 300:
                continue
            cache.write_text(full_text, encoding="utf-8")
            time.sleep(0.4)

        chunks = chunk_words(full_text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS)
        if not chunks:
            continue

        for j, chunk in enumerate(chunks):
            records.append(passage_record(
                doc_id=f"PMC{pmc_id}_chunk{j}",
                text=chunk,
                modality="pdf",
                source="pmc_open_access",
                pmcid=f"PMC{pmc_id}",
                title=title,
                chunk_idx=j,
            ))

        fetched += 1
        if fetched % 10 == 0:
            logger.info(f"  {fetched}/{PDF_SAMPLE} articles | {len(records)} chunks")

    if not records:
        logger.warning("No records. Check internet connection.")
        return 0

    save_jsonl(records, output_path)
    logger.info(f"Done: {fetched} articles, {len(records):,} chunks")
    return len(records)
