# Databricks GenAI end-to-end project: an AI financial analyst

Build an AI analyst that answers questions about real companies and shows its sources.
It reads two things and needs both:

| | Source | Where it lives |
|---|---|---|
| **The number** | What the company reported to the SEC, as data | A Unity Catalog **table** |
| **The reason** | What management wrote in the annual report (10-K) | A PDF in a Unity Catalog **volume** |

One question runs through the whole project: *"Revenue grew. Was it price or volume?"*
The growth rate is in a table. The reason is in a PDF. Each chapter adds one piece until an agent
can answer with both, and cite them.

Companies: Snowflake (SNOW), Datadog (DDOG), MongoDB (MDB). All data is public, from
[SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

> This project reads filings and cites them. It does not give investment advice.

## Status

Work in progress. Chapters are added in order.

| # | Chapter | Notebook | Status |
|---|---|---|---|
| 1 | Set up the project in Unity Catalog: schema and two volumes | `notebooks/01_setup.py` | Runs on Databricks |
| 2 | Load the raw data: files into volumes, numbers into a raw table | `notebooks/02_load_raw_data.py` | Runs on Databricks |
| 3 | Clean the numbers into one row per company and year | `notebooks/03_clean_financials.py` | Runs on Databricks |
| 4 | Turn the PDFs into text, labelled by 10-K Item, section and page | `notebooks/04_parse_filings.py` | Runs on Databricks |
| 5 | Chunk the filings: about 1,600 chunks with id, Item, section and page | `notebooks/05_chunk_filings.py` | Runs on Databricks |
| 6 | Build a Vector Search index and search the filings by meaning | `notebooks/06_vector_search_index.py` | Runs on Databricks |
| 7 | A retrieval chain over the filings | | Planned |
| 8 | Unity Catalog functions as tools on the numbers | | Planned |
| 9 | The agent that uses both tools and cites them | | Planned |
| 10 | Evaluate the answers | | Planned |
| 11 | Permissions and lineage | | Planned |

Chapters 1 to 6 have been run end to end on a serverless Databricks workspace (2026-11-12):
34,657 raw facts, 27 company-years in `financials_annual`, and 526 pages of filings parsed into
about 1,250 labelled rows in `filing_text`.

## Run it

1. In Databricks, go to **Workspace > Create > Git folder** and paste this repo's URL.
2. Open `notebooks/01_setup` and run it, then each chapter in order.

The notebooks use the catalog `workspace` (the default on Databricks Free Edition) and create a
schema called `fin_analyst`. To use another catalog, change the `catalog` widget at the top of
any notebook.

After chapter 6 you have:

```
workspace.fin_analyst
├── raw_facts            volume   SEC companyfacts JSON, one file per company
├── raw_filings          volume   annual reports as PDF
├── sec_facts_raw        table    every reported fact, with repeats
├── filings              table    one row per annual report
├── metric_map           table    accounting tag -> business name
├── financial_facts      table    one value per company, metric and year
├── financials_annual    table    one row per company and year
├── filings_parsed_raw   table    ai_parse_document output, one row per PDF
├── filing_elements      table    every parsed element with its Item and section
├── filing_text          table    paragraphs, tables and footnotes
├── filing_chunks        table    search-sized chunks
└── filing_chunks_index  index    the chunks, searchable by meaning

fin_analyst_search       Vector Search endpoint serving the index
```

The first build of the index embeds every chunk and can take 30 to 45 minutes on a small workspace.
Later runs only embed chunks that changed.

## The data

`data/` holds everything the notebooks need, so nothing is downloaded inside Databricks.

| Path | What it is |
|---|---|
| `data/companyfacts/<TICKER>.json` | Every number the company has reported, from the SEC XBRL "company facts" API |
| `data/filings/manifest.csv` | One row per annual report: ticker, fiscal year end, filing date, SEC link |
| `data/filings/<TICKER>_10-K_<date>.pdf` | The annual report itself, printed to PDF from the SEC filing |
| `data/download_sec_data.py` | The script that fetched all of the above. Run it on your own computer to refresh the data or add a company |

## Repo layout

```
data/         source files and the download script
notebooks/    one Databricks notebook per chapter, plus 00_config (shared settings)
```
