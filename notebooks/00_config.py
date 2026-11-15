# Databricks notebook source
# MAGIC %md
# MAGIC # Project settings
# MAGIC
# MAGIC Every chapter starts with `%run ./00_config`. It picks the catalog and schema the project lives in
# MAGIC and gives the two volume paths a name.
# MAGIC
# MAGIC The default catalog is `workspace`, which every new Databricks workspace has. To use another one, type
# MAGIC its name in the widget at the top of the notebook.

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
VS_ENDPOINT = "fin_analyst_search"                          # one endpoint can serve many indexes
VS_INDEX = f"{CATALOG}.{SCHEMA}.filing_chunks_index"
EMBEDDING_MODEL = "databricks-gte-large-en"

# The model that reads and answers (chapter 7 onwards).
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# The model that marks the answers (chapter 10). Any chat endpoint in the workspace can be the judge.
JUDGE_MODEL = LLM_MODEL

# The functions the model can call as tools (chapter 8 onwards).
TOOL_FUNCTIONS = [f"{CATALOG}.{SCHEMA}.company_financials", f"{CATALOG}.{SCHEMA}.search_filings"]

# The group allowed to use the analyst (chapter 11). "account users" is everyone in the Databricks account:
# fine for a workspace of one, too wide for a company. Put a narrower group here, for example "analysts".
ANALYST_USERS = "account users"

# The analyst as a registered model and a serving endpoint (chapter 12).
UC_MODEL = f"{CATALOG}.{SCHEMA}.analyst"


def report_checks(checks, model_checks=None, what="this chapter"):
    """Print each check. `checks` must hold on every run. `model_checks` depend on what the model wrote this time."""
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", "-", name)
    for name, ok in (model_checks or {}).items():
        print("PASS" if ok else "WARN", "-", name)
    assert all(checks.values()), f"{what} failed a check"
    if model_checks and not all(model_checks.values()):
        print()
        print("WARN: the model answered differently on this run. Read that answer above before changing anything.")


print(f"Project schema: {CATALOG}.{SCHEMA}")
print(f"Numbers volume: {FACTS_VOLUME}")
print(f"Filings volume: {FILINGS_VOLUME}")
