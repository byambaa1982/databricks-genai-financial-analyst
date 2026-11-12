# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 2: Load the raw data
# MAGIC
# MAGIC Free Edition cannot download from sec.gov, so the source files ship with this repo in `data/`.
# MAGIC This chapter:
# MAGIC
# MAGIC 1. copies them from the repo into the two volumes,
# MAGIC 2. turns the numbers into one raw table, `sec_facts_raw`, one row per reported fact,
# MAGIC 3. registers the filings in a small table, `filings`.
# MAGIC
# MAGIC Nothing is cleaned here. A raw table keeps everything the source said, so later steps can be re-run
# MAGIC without going back to the files.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Copy the files from the repo into the volumes

# COMMAND ----------

import os
import shutil

REPO_DATA = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))


def copy_folder(source_folder, volume_path, extensions):
    copied = []
    for name in sorted(os.listdir(source_folder)):
        if name.lower().endswith(extensions):
            shutil.copy(os.path.join(source_folder, name), os.path.join(volume_path, name))
            copied.append(name)
    return copied


facts_files = copy_folder(os.path.join(REPO_DATA, "companyfacts"), FACTS_VOLUME, (".json",))
filing_files = copy_folder(os.path.join(REPO_DATA, "filings"), FILINGS_VOLUME, (".pdf", ".csv"))

print("raw_facts  :", facts_files)
print("raw_filings:", filing_files)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The numbers: from nested JSON to one row per fact
# MAGIC
# MAGIC Each JSON file holds every number one company has ever reported, nested four levels deep:
# MAGIC
# MAGIC ```
# MAGIC facts
# MAGIC └── us-gaap                                  taxonomy (the accounting standard)
# MAGIC     └── GrossProfit                          tag (what is measured)
# MAGIC         └── units
# MAGIC             └── USD                          unit
# MAGIC                 └── [ {start, end, val, fy, fp, form, filed, accn}, ... ]   one entry per time it was reported
# MAGIC ```
# MAGIC
# MAGIC We flatten that to a plain table. The files are small (about 2 MB each), so plain Python is enough.

# COMMAND ----------

import json

from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType


def flatten_companyfacts(path):
    ticker = os.path.basename(path).replace(".json", "")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    for taxonomy, tags in doc["facts"].items():
        for tag, body in tags.items():
            for unit, entries in body["units"].items():
                for e in entries:
                    yield (ticker, int(doc["cik"]), doc["entityName"], taxonomy, tag, body.get("label"), unit,
                           e.get("start"), e["end"], float(e["val"]), e.get("fy"), e.get("fp"), e.get("form"),
                           e.get("filed"), e.get("accn"), e.get("frame"))


schema = StructType([
    StructField("ticker", StringType()), StructField("cik", LongType()), StructField("company", StringType()),
    StructField("taxonomy", StringType()), StructField("tag", StringType()), StructField("label", StringType()),
    StructField("unit", StringType()), StructField("period_start", StringType()), StructField("period_end", StringType()),
    StructField("value", DoubleType()), StructField("fy", IntegerType()), StructField("fp", StringType()),
    StructField("form", StringType()), StructField("filed", StringType()), StructField("accession_number", StringType()),
    StructField("frame", StringType()),
])

rows = [row for name in facts_files for row in flatten_companyfacts(os.path.join(FACTS_VOLUME, name))]

(spark.createDataFrame(rows, schema)
    .selectExpr("ticker", "cik", "company", "taxonomy", "tag", "label", "unit",
                "to_date(period_start) AS period_start", "to_date(period_end) AS period_end", "value",
                "fy", "fp", "form", "to_date(filed) AS filed", "accession_number", "frame")
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("sec_facts_raw"))

print(f"{len(rows):,} facts written to sec_facts_raw")

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE sec_facts_raw IS
# MAGIC   'Raw: every XBRL fact each company reported to the SEC, one row per time it was reported. Not cleaned. The same period appears several times.';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, company, count(*) AS facts, count(DISTINCT tag) AS tags, min(period_end) AS first_period, max(period_end) AS last_period
# MAGIC FROM sec_facts_raw
# MAGIC GROUP BY ticker, company
# MAGIC ORDER BY ticker;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look at one number before trusting it
# MAGIC
# MAGIC Snowflake's revenue for the year that ended 31 January 2024. One period, one true value.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT period_start, period_end, value, fy, form, filed
# MAGIC FROM sec_facts_raw
# MAGIC WHERE ticker = 'SNOW'
# MAGIC   AND tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
# MAGIC   AND period_end = '2024-01-31' AND period_start = '2023-02-01'
# MAGIC ORDER BY filed;

# COMMAND ----------

# MAGIC %md
# MAGIC The same number comes back three times, with three different `fy` values. Each annual report repeats the
# MAGIC two prior years for comparison, and `fy` is the year of the **report**, not of the **period**.
# MAGIC
# MAGIC Chapter 3 deals with this. For now the raw table keeps all of it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The filings: a small table that says which PDF is which

# COMMAND ----------

(spark.read.option("header", True).csv(os.path.join(FILINGS_VOLUME, "manifest.csv"))
    .selectExpr("ticker", "company", "form", "to_date(period_end) AS period_end", "year(to_date(period_end)) AS fiscal_year",
                "to_date(filed) AS filed", "accession_number", "file_name", "sec_url")
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("filings"))

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE filings IS
# MAGIC   'One row per annual report (10-K). file_name is the PDF in the raw_filings volume; sec_url is the original on sec.gov.';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, company, fiscal_year, period_end, filed, file_name FROM filings ORDER BY ticker;

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── raw_facts          volume   3 JSON files
# MAGIC ├── raw_filings        volume   3 PDFs + manifest.csv
# MAGIC ├── sec_facts_raw      table    every reported fact, with repeats
# MAGIC └── filings            table    one row per annual report
# MAGIC ```
# MAGIC
# MAGIC Next: `03_clean_financials` turns the raw facts into one clean row per company and year.
