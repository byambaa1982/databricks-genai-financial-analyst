# Databricks notebook source
# MAGIC %md
# MAGIC # Project settings
# MAGIC
# MAGIC Every chapter starts with `%run ./00_config`. It picks the catalog and schema the project lives in
# MAGIC and gives the two volume paths a name.
# MAGIC
# MAGIC On Databricks Free Edition the default catalog is called `workspace`. On another workspace, type your
# MAGIC own catalog name in the widget at the top of the notebook.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "fin_analyst")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA} COMMENT 'AI financial analyst project: SEC filings and reported numbers'")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Files live in volumes. Tables live next to them in the same schema.
FACTS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw_facts"      # the numbers, as JSON from SEC
FILINGS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw_filings"  # the words, as annual report PDFs

# Search by meaning (chapter 6 onwards).
VS_ENDPOINT = "fin_analyst_search"                          # Free Edition allows one Vector Search endpoint
VS_INDEX = f"{CATALOG}.{SCHEMA}.filing_chunks_index"
EMBEDDING_MODEL = "databricks-gte-large-en"

print(f"Project schema: {CATALOG}.{SCHEMA}")
print(f"Numbers volume: {FACTS_VOLUME}")
print(f"Filings volume: {FILINGS_VOLUME}")
