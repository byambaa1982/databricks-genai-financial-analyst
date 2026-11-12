"""Download the project's source data from SEC EDGAR. Run this on your own computer, not in Databricks.

Databricks Free Edition cannot reach sec.gov, so the data is fetched locally and the result is
committed to this repo. You only need to run this script if you want fresher filings or other
companies.

    python data/download_sec_data.py            # numbers (JSON) and filings (HTML)
    python data/download_sec_data.py --pdf      # also print each filing to PDF (needs: pip install playwright
                                                #   and: python -m playwright install chromium)

What it writes, next to this file:

    companyfacts/<TICKER>.json   every number the company has reported to the SEC, as XBRL facts
    filings/<TICKER>_10-K_<fiscal year end>.pdf   the latest annual report
    filings/manifest.csv         one row per filing: ticker, company, form, dates, SEC link

SEC asks automated clients to identify themselves. The numbers download with the default client
name, but the filing archive answers 403 until you set SEC_USER_AGENT to "Your Name your@email":

    SEC_USER_AGENT="Jane Doe jane@example.com" python data/download_sec_data.py --pdf
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ticker -> SEC company id (CIK). Add a line here to add a company.
COMPANIES = {
    "SNOW": 1640147,  # Snowflake Inc.
    "DDOG": 1561550,  # Datadog, Inc.
    "MDB": 1441816,   # MongoDB, Inc.
}
FORM = "10-K"
HERE = Path(__file__).resolve().parent
USER_AGENT = os.environ.get("SEC_USER_AGENT", "databricks-genai-financial-analyst tutorial")


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    time.sleep(0.3)  # SEC allows 10 requests a second; stay far below it
    return data


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="load", timeout=300_000)
        page.pdf(path=str(pdf_path), format="Letter", margin={"top": "0.6in", "bottom": "0.6in", "left": "0.6in", "right": "0.6in"})
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pdf", action="store_true", help="also print each filing to PDF")
    parser.add_argument("--years", type=int, default=1, help="how many annual reports per company (default 1, the latest)")
    args = parser.parse_args()

    facts_dir, filings_dir, html_dir = HERE / "companyfacts", HERE / "filings", HERE / "_html"
    for d in (facts_dir, filings_dir, html_dir):
        d.mkdir(exist_ok=True)

    manifest = []
    for ticker, cik in COMPANIES.items():
        # 1. The numbers: one JSON file with every XBRL fact the company has filed.
        facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        (facts_dir / f"{ticker}.json").write_bytes(facts)
        company = json.loads(facts)["entityName"]
        print(f"{ticker}: companyfacts {len(facts) / 1e6:.1f} MB")

        # 2. The words: the latest annual report(s).
        recent = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]["recent"]
        rows = [i for i, form in enumerate(recent["form"]) if form == FORM][: args.years]
        for i in rows:
            accession, document = recent["accessionNumber"][i], recent["primaryDocument"][i]
            period_end = recent["reportDate"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}"
            name = f"{ticker}_{FORM}_{period_end}"
            html_path, pdf_path = html_dir / f"{name}.htm", filings_dir / f"{name}.pdf"
            manifest.append({"ticker": ticker, "company": company, "form": FORM, "period_end": period_end,
                             "filed": recent["filingDate"][i], "accession_number": accession,
                             "file_name": pdf_path.name, "sec_url": url})
            if not html_path.exists():
                try:
                    html_path.write_bytes(get(url))
                except urllib.error.HTTPError as e:
                    if e.code != 403:
                        raise
                    print(f"{ticker}: SEC refused the filing download (403). Set SEC_USER_AGENT as described at the top of this file.")
                    continue
            print(f"{ticker}: {FORM} for period ending {period_end}, {html_path.stat().st_size / 1e6:.1f} MB of HTML")
            if args.pdf and not pdf_path.exists():
                html_to_pdf(html_path, pdf_path)
                print(f"{ticker}: wrote {pdf_path.name}, {pdf_path.stat().st_size / 1e6:.1f} MB")

    with open(filings_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"manifest: {len(manifest)} filings")


if __name__ == "__main__":
    main()
