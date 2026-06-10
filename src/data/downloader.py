"""
downloader.py — Downloads public legal PDFs and saves a JSON manifest.

WHY THIS EXISTS:
  We need a reproducible corpus. The manifest lets us re-run any phase
  without re-downloading, and gives every downstream step rich metadata
  (source, date, num_pages) without parsing the PDF again.

INTERVIEW NOTE:
  We use Indian Kanoon judgments (open, machine-readable) and SEC EDGAR
  filings (real-world legal documents with complex cross-page arguments).
  Two different corpora stress-test the chunker more than one homogeneous set.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model — what we store per document in the manifest
# ---------------------------------------------------------------------------
@dataclass
class DocMetadata:
    doc_id: str          # Stable unique ID (slugified title or hash)
    title: str
    source: str          # URL the PDF came from
    domain: str          # "indian_kanoon" | "sec_edgar" | "other"
    date: str            # Publication / judgment date (YYYY-MM-DD or best guess)
    num_pages: int       # Filled in by processor.py after parsing
    local_path: str      # Relative path under data/raw/


# ---------------------------------------------------------------------------
# Public legal PDF URLs
# INTERVIEW NOTE: We use direct PDF links so there's no scraping logic.
# In production you'd hit the Indian Kanoon API or SEC EDGAR full-text search.
# ---------------------------------------------------------------------------
LEGAL_PDF_URLS: list[dict] = [
    # --- Indian Supreme Court judgments (public domain) ---
    {
        "url": "https://main.sci.gov.in/supremecourt/2023/20686/20686_2023_1_1501_56099_Judgement_10-Nov-2023.pdf",
        "title": "SCI_Judgment_Nov2023_20686",
        "domain": "indian_kanoon",
        "date": "2023-11-10",
    },
    {
        "url": "https://main.sci.gov.in/supremecourt/2022/16706/16706_2022_1_1501_47599_Judgement_24-Mar-2023.pdf",
        "title": "SCI_Judgment_Mar2023_16706",
        "domain": "indian_kanoon",
        "date": "2023-03-24",
    },
    # --- SEC EDGAR 10-K filings (publicly accessible) ---
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1318605/000095017022000796/tsla-20211231.htm",
        "title": "Tesla_10K_2021",
        "domain": "sec_edgar",
        "date": "2022-02-07",
    },
    # --- World Bank open legal documents ---
    {
        "url": "https://openknowledge.worldbank.org/bitstream/handle/10986/2124/9780821373446.pdf",
        "title": "WorldBank_Legal_Framework",
        "domain": "world_bank",
        "date": "2012-01-01",
    },
    # --- US court opinions via CourtListener (public domain) ---
    {
        "url": "https://storage.courtlistener.com/pdf/2013/06/26/roe_v._wade.pdf",
        "title": "Roe_v_Wade_Opinion",
        "domain": "us_court",
        "date": "1973-01-22",
    },
]

# Fallback: well-known open legal PDFs from gov sites that rarely change
FALLBACK_PDF_URLS: list[dict] = [
    {
        "url": "https://www.justice.gov/d9/2023-03/2023.03.01_-_nsd_disclosing_cyber_foia_guide.pdf",
        "title": "DOJ_Cyber_FOIA_Guide_2023",
        "domain": "us_doj",
        "date": "2023-03-01",
    },
    {
        "url": "https://www.ftc.gov/system/files/ftc_gov/pdf/p085406morrisbig.pdf",
        "title": "FTC_BigData_Report",
        "domain": "us_ftc",
        "date": "2016-01-01",
    },
    {
        "url": "https://www.irs.gov/pub/irs-pdf/p15.pdf",
        "title": "IRS_Circular_E_2024",
        "domain": "us_irs",
        "date": "2024-01-01",
    },
    {
        "url": "https://www.fdic.gov/regulations/applications/pdf/de_novo_manual.pdf",
        "title": "FDIC_DeNovo_Manual",
        "domain": "us_fdic",
        "date": "2020-07-01",
    },
    {
        "url": "https://www.federalreserve.gov/pubs/feds/2021/202139/202139pap.pdf",
        "title": "FedReserve_Research_2021",
        "domain": "us_fed",
        "date": "2021-06-01",
    },
    {
        "url": "https://www.sec.gov/files/33-11312.pdf",
        "title": "SEC_Rule_33_11312",
        "domain": "sec_edgar",
        "date": "2023-07-26",
    },
    {
        "url": "https://www.sec.gov/files/rules/final/2023/33-11216.pdf",
        "title": "SEC_Rule_33_11216_Cybersecurity",
        "domain": "sec_edgar",
        "date": "2023-07-26",
    },
    {
        "url": "https://www.occ.gov/publications-and-resources/publications/comptrollers-handbook/files/bank-supervision-process/pub-ch-bank-supervision-process.pdf",
        "title": "OCC_Bank_Supervision_Process",
        "domain": "us_occ",
        "date": "2022-01-01",
    },
    {
        "url": "https://www.fdic.gov/bank/individual/failed/sifi-guidance.pdf",
        "title": "FDIC_SIFI_Guidance",
        "domain": "us_fdic",
        "date": "2021-01-01",
    },
    {
        "url": "https://www.dol.gov/sites/dolgov/files/EBSA/about-ebsa/our-activities/resource-center/fact-sheets/erisa.pdf",
        "title": "DOL_ERISA_FactSheet",
        "domain": "us_dol",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.ftc.gov/system/files/documents/reports/privacy-online-fair-information-practices-electronic-marketplace/privacy2000text.pdf",
        "title": "FTC_Privacy_Online_2000",
        "domain": "us_ftc",
        "date": "2000-05-01",
    },
    {
        "url": "https://www.justice.gov/d9/2022-11/ccips_guidance_on_electronic_evidence.pdf",
        "title": "DOJ_Electronic_Evidence_Guide",
        "domain": "us_doj",
        "date": "2022-11-01",
    },
    {
        "url": "https://www.copyright.gov/circs/circ01.pdf",
        "title": "Copyright_Office_Circular_01",
        "domain": "us_copyright",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.copyright.gov/circs/circ09.pdf",
        "title": "Copyright_Office_Works_for_Hire",
        "domain": "us_copyright",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.irs.gov/pub/irs-pdf/p583.pdf",
        "title": "IRS_Starting_Business_2023",
        "domain": "us_irs",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.consumer.ftc.gov/sites/default/files/games/off-site/youarehere/pages/pdf/FTC-Competition-Consumer-Guide.pdf",
        "title": "FTC_Competition_Consumer_Guide",
        "domain": "us_ftc",
        "date": "2019-01-01",
    },
    {
        "url": "https://www.hud.gov/sites/dfiles/OCHCO/documents/FEOA.PDF",
        "title": "HUD_Fair_Housing_Act",
        "domain": "us_hud",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.eeoc.gov/sites/default/files/2023-06/22-088_EEOC_GUIDE.pdf",
        "title": "EEOC_Employer_Guide_2023",
        "domain": "us_eeoc",
        "date": "2023-06-01",
    },
    {
        "url": "https://www.law.cornell.edu/uscode/text/17/107",
        "title": "Cornell_Fair_Use_Doctrine",
        "domain": "cornell_law",
        "date": "2023-01-01",
    },
    {
        "url": "https://www.sec.gov/files/litigation/admin/2023/34-98543.pdf",
        "title": "SEC_Admin_Proceeding_98543",
        "domain": "sec_edgar",
        "date": "2023-09-26",
    },
    {
        "url": "https://www.sec.gov/files/litigation/admin/2023/34-98093.pdf",
        "title": "SEC_Admin_Proceeding_98093",
        "domain": "sec_edgar",
        "date": "2023-08-21",
    },
    {
        "url": "https://www.sec.gov/files/litigation/admin/2023/34-97946.pdf",
        "title": "SEC_Admin_Proceeding_97946",
        "domain": "sec_edgar",
        "date": "2023-08-08",
    },
    {
        "url": "https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-234.pdf",
        "title": "SEC_Complaint_2023_234",
        "domain": "sec_edgar",
        "date": "2023-11-01",
    },
    {
        "url": "https://www.sec.gov/files/litigation/complaints/2023/comp-pr2023-222.pdf",
        "title": "SEC_Complaint_2023_222",
        "domain": "sec_edgar",
        "date": "2023-10-17",
    },
    {
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL&type=10-K&dateb=&owner=include&count=1&search_text=",
        "title": "AAPL_10K_Latest",
        "domain": "sec_edgar",
        "date": "2023-10-27",
    },
]


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------
def download_pdf(
    url: str,
    dest_path: Path,
    timeout: int = 30,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> bool:
    """
    Download a single PDF from `url` to `dest_path` with exponential retry.

    WHY RETRY WITH BACKOFF:
      Legal databases (Indian Kanoon, SEC) enforce rate limits. A 429 or 503
      is transient — sleeping before retry is enough in most cases.

    Returns True on success, False if all retries exhausted.
    """
    headers = {
        "User-Agent": (
            "LexRAG-Research-Bot/1.0 "
            "(Educational legal NLP project; contact: research@example.com)"
        )
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading [{attempt}/{max_retries}]: {url}")
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()

            # Validate it's actually a PDF (check magic bytes)
            content_type = resp.headers.get("Content-Type", "")
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Sanity-check: PDF magic bytes are %PDF
            with open(dest_path, "rb") as f:
                magic = f.read(4)
            if magic != b"%PDF":
                logger.warning(f"Not a valid PDF (magic={magic}): {url}")
                dest_path.unlink(missing_ok=True)
                return False

            logger.info(f"Saved to {dest_path} ({dest_path.stat().st_size // 1024} KB)")
            return True

        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            logger.warning(f"HTTP {status} for {url} — attempt {attempt}")
            if status == 404:
                return False  # No point retrying a missing resource
        except requests.RequestException as e:
            logger.warning(f"Request error for {url} — {e} — attempt {attempt}")

        if attempt < max_retries:
            sleep_time = backoff ** attempt
            logger.info(f"Retrying in {sleep_time:.1f}s …")
            time.sleep(sleep_time)

    logger.error(f"Failed after {max_retries} attempts: {url}")
    return False


# ---------------------------------------------------------------------------
# Batch downloader
# ---------------------------------------------------------------------------
def download_corpus(config: dict, extra_urls: Optional[list[dict]] = None) -> list[DocMetadata]:
    """
    Download up to `config.data.download.max_docs` PDFs.

    Strategy:
      1. Try PRIMARY_URLS first (curated, high quality)
      2. Fill remaining slots from FALLBACK_PDF_URLS
      3. Append any caller-supplied extra_urls last

    Returns the list of successfully downloaded DocMetadata objects.
    The manifest is also written to config.data.manifest_file.

    INTERVIEW NOTE:
      We don't re-download files that already exist on disk (idempotent).
      This means running the script twice is safe — useful during development
      when you're iterating on processor.py and don't want to hammer servers.
    """
    dl_cfg = config["data"]["download"]
    raw_dir = Path(config["data"]["raw_dir"])
    manifest_path = Path(config["data"]["manifest_file"])
    max_docs = dl_cfg["max_docs"]

    all_sources = LEGAL_PDF_URLS + FALLBACK_PDF_URLS + (extra_urls or [])
    all_sources = all_sources[:max_docs]  # Respect the cap

    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[DocMetadata] = []

    with tqdm(total=len(all_sources), desc="Downloading PDFs", unit="doc") as pbar:
        for source in all_sources:
            url = source["url"]
            title = source["title"]
            slug = title.replace(" ", "_").replace("/", "-")[:60]
            dest = raw_dir / f"{slug}.pdf"

            # Skip if already on disk (idempotent re-runs)
            if dest.exists() and dest.stat().st_size > 1024:
                logger.info(f"Already exists, skipping: {dest.name}")
                meta = DocMetadata(
                    doc_id=slug,
                    title=title,
                    source=url,
                    domain=source.get("domain", "unknown"),
                    date=source.get("date", "unknown"),
                    num_pages=0,   # Filled in by processor.py
                    local_path=str(dest),
                )
                downloaded.append(meta)
                pbar.update(1)
                continue

            success = download_pdf(
                url=url,
                dest_path=dest,
                timeout=dl_cfg["timeout_seconds"],
                max_retries=dl_cfg["max_retries"],
                backoff=dl_cfg["retry_backoff"],
            )

            if success:
                meta = DocMetadata(
                    doc_id=slug,
                    title=title,
                    source=url,
                    domain=source.get("domain", "unknown"),
                    date=source.get("date", "unknown"),
                    num_pages=0,
                    local_path=str(dest),
                )
                downloaded.append(meta)

            pbar.update(1)

    # Write manifest
    manifest_data = [asdict(m) for m in downloaded]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest written: {manifest_path} ({len(downloaded)} docs)")
    return downloaded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the downloader from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Download legal PDF corpus")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Override config max_docs")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.max_docs is not None:
        config["data"]["download"]["max_docs"] = args.max_docs

    docs = download_corpus(config)
    print(f"\nDownloaded {len(docs)} documents.")
    for d in docs:
        print(f"  {d.doc_id:50s}  {d.domain}")


if __name__ == "__main__":
    main()
