# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 1: Set up the project in Unity Catalog
# MAGIC
# MAGIC An analyst's question needs two things: **a number** and **a reason**.
# MAGIC
# MAGIC | | Where it comes from | Where it lives in Unity Catalog |
# MAGIC |---|---|---|
# MAGIC | The number | What the company reported to the SEC, as data | A **table** |
# MAGIC | The reason | What management wrote in the annual report | A file in a **volume** |
# MAGIC
# MAGIC This chapter creates one schema with two volumes. Tables are added in the next chapters.
# MAGIC Everything sits under one schema, so one set of permissions covers the files and the tables.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Two volumes: one for numbers, one for filings
# MAGIC
# MAGIC A volume is a governed folder. It holds files of any type, and Unity Catalog controls who can read
# MAGIC them, the same way it does for tables.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS raw_facts
# MAGIC   COMMENT 'SEC companyfacts JSON, one file per company. Source of every reported number.';
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS raw_filings
# MAGIC   COMMENT 'Annual reports (10-K) as PDF, one file per company and fiscal year.';

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW VOLUMES;

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace                      catalog
# MAGIC └── fin_analyst                schema
# MAGIC     ├── raw_facts              volume  (empty)
# MAGIC     └── raw_filings            volume  (empty)
# MAGIC ```
# MAGIC
# MAGIC Next: `02_load_raw_data` fills both volumes and builds the first table.
