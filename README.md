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

Complete: eleven chapters, each one a notebook that runs on its own after the ones before it.

| # | Chapter | Notebook | Status |
|---|---|---|---|
| 1 | Set up the project in Unity Catalog: schema and two volumes | `notebooks/01_setup.py` | Runs on Databricks |
| 2 | Load the raw data: files into volumes, numbers into a raw table | `notebooks/02_load_raw_data.py` | Runs on Databricks |
| 3 | Clean the numbers into one row per company and year | `notebooks/03_clean_financials.py` | Runs on Databricks |
| 4 | Turn the PDFs into text, labelled by 10-K Item, section and page | `notebooks/04_parse_filings.py` | Runs on Databricks |
| 5 | Chunk the filings: about 1,600 chunks with id, Item, section and page | `notebooks/05_chunk_filings.py` | Runs on Databricks |
| 6 | Build a Vector Search index and search the filings by meaning | `notebooks/06_vector_search_index.py` | Runs on Databricks |
| 7 | A retrieval chain: answers from the filings with chunk ids as sources | `notebooks/07_rag_chain.py` | Runs on Databricks |
| 8 | Tools: two Unity Catalog functions the model can call, one for the numbers and one for the words | `notebooks/08_tools.py` | Runs on Databricks |
| 9 | The agent: both tools in a loop, answers that cite the table row and the page | `notebooks/09_agent.py` | Runs on Databricks |
| 10 | Evaluate the answers: 14 questions, five scorers, two versions compared in MLflow | `notebooks/10_evaluate.py` | Runs on Databricks |
| 11 | Permissions and lineage: who may use the analyst, and where an answer came from | `notebooks/11_permissions_and_lineage.py` | Runs on Databricks |

All eleven chapters have been run end to end on a serverless Databricks workspace (chapters 8 to 11
on 2026-11-14): 34,657 raw facts, 27 company-years in `financials_annual`, and 526 pages of filings
parsed into about 1,250 labelled rows in `filing_text`.

What the finished analyst does with the project's question:

```
Q: How much did Snowflake's revenue grow in fiscal 2026, and was it higher prices or more usage?

   step 1: company_financials({"company_ticker": "SNOW"})
   step 2: search_filings({"question": "What drove revenue growth in fiscal 2026", "company_ticker": "SNOW"})

A: Snowflake's revenue grew 29.2% in fiscal 2026 [financials_annual: SNOW FY2026]. The growth was
   primarily due to increased consumption of the platform by existing customers, as evidenced by the
   net revenue retention rate of 125% as of January 31, 2026 [SNOW-2026-0353]. ...
```

In chapter 10 the first version of the analyst passes 12 of 14 evaluation questions. One failure is the
judge being wrong and one is the analyst being wrong. After one fix to each, the second version
passes all 14.

## Run it

1. In Databricks, go to **Workspace > Create > Git folder** and paste this repo's URL.
2. Open `notebooks/01_setup` and run it, then each chapter in order.

The notebooks use the catalog `workspace` (the default on Databricks Free Edition) and create a
schema called `fin_analyst`. To use another catalog, change the `catalog` widget at the top of
any notebook.

After chapter 11 you have:

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
├── filing_chunks_index  index    the chunks, searchable by meaning
├── company_financials   function tool: one company's numbers, last five fiscal years
├── search_filings       function tool: the 8 most relevant excerpts for a question
└── eval_questions       table    14 questions with the facts a correct answer must contain

fin_analyst_search       Vector Search endpoint serving the index
fin_analyst_evaluation   MLflow experiment: one run per version of the analyst, one trace per question
```

The two functions are the only things the model can call. It picks one from their `COMMENT` text:
`company_financials` for a number, `search_filings` for what management wrote.

Chapter 11 grants `USE SCHEMA` and `EXECUTE` on the two functions to the group named in
`ANALYST_USERS` (`00_config`, default `account users`). Change it there before running if that
group is wider than you want.

Chapters 9 to 11 call the model many times. On Free Edition the endpoint limits requests per minute,
so the agent waits and retries, and chapter 10 asks one question at a time and takes about six
minutes.

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
              and analyst.py (the agent, written by chapter 9 and imported by chapters 10 and 11)
```
