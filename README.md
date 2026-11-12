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
| 1 | Set up the project in Unity Catalog: schema and two volumes | `notebooks/01_setup.py` | Written |
| 2 | Load the raw data: files into volumes, numbers into a raw table | `notebooks/02_load_raw_data.py` | Written |
| 3 | Clean the numbers into one row per company and year | `notebooks/03_clean_financials.py` | Written |
| 4 | Turn the PDFs into text | | Planned |
| 5 | Chunk the filings | | Planned |
| 6 | Build a Vector Search index | | Planned |
| 7 | A retrieval chain over the filings | | Planned |
| 8 | Unity Catalog functions as tools on the numbers | | Planned |
| 9 | The agent that uses both tools and cites them | | Planned |
| 10 | Evaluate the answers | | Planned |
| 11 | Permissions and lineage | | Planned |

Chapters 1 to 3 have been run end to end on local Spark with Delta. Their run on a Databricks
workspace is still to be recorded here.

## Run it

1. In Databricks, go to **Workspace > Create > Git folder** and paste this repo's URL.
2. Open `notebooks/01_setup` and run it, then `02_load_raw_data`, then `03_clean_financials`.

The notebooks use the catalog `workspace` (the default on Databricks Free Edition) and create a
schema called `fin_analyst`. To use another catalog, change the `catalog` widget at the top of
any notebook.

After chapter 3 you have:

```
workspace.fin_analyst
├── raw_facts            volume   SEC companyfacts JSON, one file per company
├── raw_filings          volume   annual reports as PDF
├── sec_facts_raw        table    every reported fact, with repeats
├── filings              table    one row per annual report
├── metric_map           table    accounting tag -> business name
├── financial_facts      table    one value per company, metric and year
└── financials_annual    table    one row per company and year
```

## The data

`data/` holds everything the notebooks need, so nothing is downloaded inside Databricks.

| Path | What it is |
|---|---|
| `data/companyfacts/<TICKER>.json` | Every number the company has reported, from the SEC XBRL "company facts" API |
| `data/filings/manifest.csv` | One row per annual report: ticker, fiscal year end, filing date, SEC link |
| `data/filings/<TICKER>_10-K_<date>.pdf` | The annual report itself |
| `data/download_sec_data.py` | The script that fetched all of the above. Run it on your own computer to refresh the data or add a company |

## Repo layout

```
data/         source files and the download script
notebooks/    one Databricks notebook per chapter, plus 00_config (shared settings)
```
