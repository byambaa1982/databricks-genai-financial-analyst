# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 11: Permissions and lineage
# MAGIC
# MAGIC The analyst answers, cites its sources and passes its evaluation. Two questions are left before
# MAGIC anyone else is allowed to use it:
# MAGIC
# MAGIC | Question | Answered by |
# MAGIC |---|---|
# MAGIC | What can the analyst reach, and who may use it? | **Permissions**: grants in Unity Catalog |
# MAGIC | Where did this answer come from? | **Lineage**: the path from an answer back to the SEC filing |
# MAGIC
# MAGIC Neither needs new machinery. Every piece of this project, the volumes, the tables, the index and the
# MAGIC two functions, was created in Unity Catalog, so the catalog already knows who can use each one and what
# MAGIC each one was built from.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain==0.22.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What the analyst can reach
# MAGIC
# MAGIC The model has no connection to the catalog. It can ask for two functions and nothing else. So the
# MAGIC whole surface of the analyst is this list:

# COMMAND ----------

display(spark.sql(f"""
  SELECT routine_name AS function, security_type, left(comment, 110) AS description
  FROM {CATALOG}.information_schema.routines
  WHERE routine_schema = '{SCHEMA}' AND routine_name IN ('company_financials', 'search_filings')
  ORDER BY routine_name
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC `security_type = DEFINER` is the important column. A function runs with the rights of the person who
# MAGIC **defined** it, not of the person who calls it. Someone allowed to run `company_financials` gets the
# MAGIC rows the function returns and does not need, or get, permission to read `financials_annual` itself.
# MAGIC
# MAGIC That is why the tools are functions. A model given free SQL can reach whatever its user can reach.
# MAGIC A model given two functions can reach two result sets.
# MAGIC
# MAGIC ## 2. Grant the least that works
# MAGIC
# MAGIC To use the analyst, a person needs four things, from the outside in:
# MAGIC
# MAGIC | Privilege | On | Why |
# MAGIC |---|---|---|
# MAGIC | `USE CATALOG` | the catalog | to get in at all. Without it nothing below counts |
# MAGIC | `USE SCHEMA` | the schema | to see what is in it |
# MAGIC | `EXECUTE` | `company_financials` | to run the tool on the numbers |
# MAGIC | `EXECUTE` | `search_filings` | to run the tool on the words |
# MAGIC
# MAGIC Nothing on the tables, the volumes or the index.
# MAGIC
# MAGIC The grants go to a **group**, never to a person. `ANALYST_USERS` is set in `00_config`. Here it is
# MAGIC `account users`, which is everyone in the Databricks account. In a workspace of one that is only you.
# MAGIC In a company, put a narrower group there, such as `analysts`, before running the next cell.

# COMMAND ----------

for statement in (
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{ANALYST_USERS}`",
    f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.{SCHEMA} TO `{ANALYST_USERS}`",
    f"GRANT EXECUTE ON FUNCTION {CATALOG}.{SCHEMA}.company_financials TO `{ANALYST_USERS}`",
    f"GRANT EXECUTE ON FUNCTION {CATALOG}.{SCHEMA}.search_filings TO `{ANALYST_USERS}`",
):
    spark.sql(statement)
    print(statement)

# COMMAND ----------

# MAGIC %md
# MAGIC What the group holds in this schema, read back from the catalog:

# COMMAND ----------

grants = spark.sql(f"""
  SELECT 'catalog' AS object_type, catalog_name AS object, replace(privilege_type, '_', ' ') AS privilege_type
  FROM {CATALOG}.information_schema.catalog_privileges
  WHERE catalog_name = '{CATALOG}' AND grantee = '{ANALYST_USERS}'
  UNION ALL
  SELECT 'schema', schema_name, replace(privilege_type, '_', ' ') AS privilege_type
  FROM {CATALOG}.information_schema.schema_privileges
  WHERE schema_name = '{SCHEMA}' AND grantee = '{ANALYST_USERS}'
  UNION ALL
  SELECT 'function', routine_name, replace(privilege_type, '_', ' ') AS privilege_type
  FROM {CATALOG}.information_schema.routine_privileges
  WHERE routine_schema = '{SCHEMA}' AND grantee = '{ANALYST_USERS}'
  UNION ALL
  SELECT 'table', table_name, replace(privilege_type, '_', ' ') AS privilege_type
  FROM {CATALOG}.information_schema.table_privileges
  WHERE table_schema = '{SCHEMA}' AND grantee = '{ANALYST_USERS}'
  UNION ALL
  SELECT 'volume', volume_name, replace(privilege_type, '_', ' ') AS privilege_type
  FROM {CATALOG}.information_schema.volume_privileges
  WHERE volume_schema = '{SCHEMA}' AND grantee = '{ANALYST_USERS}'
  ORDER BY object_type, object
""")
display(grants)

# COMMAND ----------

# MAGIC %md
# MAGIC `USE CATALOG`, `USE SCHEMA` and two `EXECUTE`. No table and no volume appears. With these grants alone,
# MAGIC a member of the group who types `SELECT * FROM sec_facts_raw` is refused, while
# MAGIC `SELECT * FROM company_financials('MDB')` works.
# MAGIC
# MAGIC Two limits on what this cell proves. It lists the grants made to this one group: a person who is also
# MAGIC in another group, or who was given more on the catalog, keeps those rights. And this notebook cannot
# MAGIC show the refusal, because you own everything in the schema and an owner is never refused. To see it,
# MAGIC sign in as a second user who is in the group and nothing else.
# MAGIC
# MAGIC ## 3. Lineage: what the tools are built from
# MAGIC
# MAGIC Unity Catalog recorded every read and write the earlier chapters made. `system.access.table_lineage`
# MAGIC holds one row per "this was read to write that". Walking it backwards from the two things the tools
# MAGIC read gives the whole project as a tree.
# MAGIC
# MAGIC Lineage is collected in the background. On a first run it can take up to a few hours before every
# MAGIC step appears.

# COMMAND ----------

import re

prefix = f"{CATALOG}.{SCHEMA}."
try:
    edges = spark.sql(f"""
      SELECT DISTINCT coalesce(source_table_full_name, source_path) AS source, target_table_full_name AS target
      FROM system.access.table_lineage
      WHERE target_table_catalog = '{CATALOG}' AND target_table_schema = '{SCHEMA}'
        AND coalesce(source_table_full_name, source_path) IS NOT NULL
    """).collect()
except Exception as error:   # the lineage system table has to be enabled and readable by you
    edges = []
    print("Could not read system.access.table_lineage:", str(error)[:200])
    print("Ask an account admin for access to the system.access schema. The Lineage tab in Catalog shows the same thing.")
    print()


def short_name(name):
    volume = re.match(r"/Volumes/[^/]+/[^/]+/([^/]+)", name)
    return f"{volume.group(1)} (volume)" if volume else name.replace(prefix, "")


parents = {}
for e in edges:
    source, target = short_name(e["source"]), short_name(e["target"])
    if source != target and not target.startswith("_") and not source.startswith("_"):
        parents.setdefault(target, set()).add(source)


def print_upstream(name, depth=0, seen=()):
    print("    " * depth + ("└── built from " if depth else "") + name)
    for parent in sorted(parents.get(name, ())):
        if parent not in seen:
            print_upstream(parent, depth + 1, seen + (name,))


for tool, reads in (("company_financials", "financials_annual"), ("search_filings", "filing_chunks_index")):
    print(f"{tool} reads:")
    print_upstream(reads, depth=0)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC The same picture is in the workspace under **Catalog > fin_analyst > financials_annual > Lineage**,
# MAGIC drawn as a graph. It answers two practical questions. Before changing `filing_chunks`: what depends on
# MAGIC it? After a wrong answer: which step could have caused it?
# MAGIC
# MAGIC Both trees stop early, and the reason is worth knowing. The catalog can follow data only while it stays
# MAGIC inside Spark or SQL:
# MAGIC
# MAGIC - `financial_facts` and `financials_annual` were built with `CREATE TABLE ... AS SELECT`, so each step
# MAGIC   is recorded,
# MAGIC - `sec_facts_raw` (chapter 2) and `filing_chunks` (chapter 5) were built by reading rows into Python,
# MAGIC   working on them there and writing the result back. The catalog saw a table being written and could
# MAGIC   not see what it was made from.
# MAGIC
# MAGIC So the trees end at `sec_facts_raw` and `filing_chunks`, not at the volumes. Where a step has to leave
# MAGIC SQL, carry the origin in the data itself. This project did: the next section uses it.
# MAGIC
# MAGIC ## 4. One answer, traced to the SEC
# MAGIC
# MAGIC Table lineage says which table fed which. The project also kept lineage at the level of a single row:
# MAGIC every number carries the filing it was reported in, and every chunk carries its file and page. So one
# MAGIC answer can be followed all the way back.

# COMMAND ----------

from analyst import SYSTEM_V2, build_analyst, chunks_cited, rows_cited, show

ask = build_analyst(TOOL_FUNCTIONS, LLM_MODEL, system=SYSTEM_V2)   # the prompt that passed chapter 10
answer = ask("How much did Snowflake's revenue grow in fiscal 2026, and was it higher prices or more usage?")
show(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC **The number.** The row the answer cites, the fact behind it, and the filing that fact was reported in:

# COMMAND ----------

row_filter = " OR ".join(f"(a.ticker = '{t}' AND a.fiscal_year = {y})" for t, y in sorted(rows_cited(answer))) or "false"
number_trace = spark.sql(f"""
  SELECT a.ticker, a.fiscal_year, a.revenue_growth_pct, f.value_usd AS revenue_usd,
         f.accession_number AS reported_in_filing, f.filed, s.sec_url
  FROM financials_annual a
  JOIN financial_facts f ON f.ticker = a.ticker AND f.fiscal_year = a.fiscal_year AND f.metric = 'revenue'
  LEFT JOIN filings s ON s.accession_number = f.accession_number
  WHERE {row_filter}
""")
display(number_trace)

# COMMAND ----------

# MAGIC %md
# MAGIC **The reason.** The excerpt the answer cites, the PDF it was cut from, and the page:

# COMMAND ----------

ids = ", ".join(f"'{i}'" for i in sorted(chunks_cited(answer))) or "''"
words_trace = spark.sql(f"""
  SELECT c.chunk_id, c.item, c.section, c.page, '{FILINGS_VOLUME}/' || c.file_name AS pdf_in_volume, s.sec_url,
         left(c.content, 200) AS content_start
  FROM filing_chunks c
  LEFT JOIN filings s ON s.file_name = c.file_name
  WHERE c.chunk_id IN ({ids})
""")
display(words_trace)

# COMMAND ----------

# MAGIC %md
# MAGIC From a sentence written by a model to a page of a document filed with the SEC, in two queries. Nothing
# MAGIC in the answer has to be taken on trust.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check it

# COMMAND ----------

import os

held = {(r["object_type"], r["object"], r["privilege_type"]) for r in grants.collect()}
definer = spark.sql(f"""
  SELECT count(*) FROM {CATALOG}.information_schema.routines
  WHERE routine_schema = '{SCHEMA}' AND routine_name IN ('company_financials', 'search_filings') AND security_type = 'DEFINER'
""").first()[0]
numbers, words = number_trace.collect(), words_trace.collect()

# The grants: the same on every run.
checks = {
    "the group may enter the catalog and the schema": {("catalog", CATALOG, "USE CATALOG"), ("schema", SCHEMA, "USE SCHEMA")} <= held,
    "the group may run both functions": {("function", "company_financials", "EXECUTE"), ("function", "search_filings", "EXECUTE")} <= held,
    "the group holds nothing on any table or volume in the schema": not any(kind in ("table", "volume") for kind, _, _ in held),
    "both functions run with their owner's rights": definer == 2,
    "every number row that was traced leads to a filing on sec.gov": all((r["sec_url"] or "").startswith("https://www.sec.gov/") for r in numbers),
    "every excerpt that was traced leads to a PDF that is in the volume": all(os.path.exists(r["pdf_in_volume"]) for r in words),
}
# These depend on what the model cited this time, and on lineage that is collected in the background.
model_checks = {
    "the answer cited a table row and an excerpt to trace": len(numbers) >= 1 and len(words) >= 1,
}
if parents:
    model_checks["lineage links financials_annual back to the raw facts"] = "financial_facts" in parents.get("financials_annual", ()) \
        and "sec_facts_raw" in parents.get("financial_facts", ())
else:
    print("NOTE - no lineage to show yet. It is collected in the background: run this cell again later.")
report_checks(checks, model_checks, "permissions or lineage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC SEC EDGAR
# MAGIC   ├── numbers (JSON) ─► raw_facts ─► sec_facts_raw ─► financial_facts ─► financials_annual ─► company_financials ─┐
# MAGIC   │                                                                                                              ├─► analyst ─► answer with sources
# MAGIC   └── annual reports (PDF) ─► raw_filings ─► filing_text ─► filing_chunks ─► filing_chunks_index ─► search_filings ─┘
# MAGIC ```
# MAGIC
# MAGIC The question from chapter 1 was *"Revenue grew. Was it price or volume?"* The analyst now answers it
# MAGIC with a growth rate from a governed table, a reason from page 72 of the annual report, and a citation
# MAGIC for each.
# MAGIC
# MAGIC It still lives in a notebook, and only someone who opens the notebook can ask it anything.
# MAGIC
# MAGIC Next: `12_deploy_and_monitor` packages the analyst as a model, registers it in Unity Catalog and puts
# MAGIC it behind an endpoint.
