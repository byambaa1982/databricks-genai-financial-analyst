# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 3: Clean the numbers
# MAGIC
# MAGIC The raw table has about 34,000 facts. An analyst asks about perhaps a dozen of them per year.
# MAGIC This chapter builds the table the AI analyst will query:
# MAGIC
# MAGIC **`financials_annual`: one row per company and fiscal year, one column per metric.**
# MAGIC
# MAGIC Three problems in the raw data have to be solved on the way:
# MAGIC
# MAGIC | Problem | Rule |
# MAGIC |---|---|
# MAGIC | Tags are accounting names, not business names | A small mapping table, `metric_map` |
# MAGIC | Annual and quarterly values share the same tag | Keep periods of about one year |
# MAGIC | Each period is reported up to three times | Keep the most recently filed value |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Name the metrics
# MAGIC
# MAGIC `kind` says how the number is measured. A **duration** covers a period (revenue for the year).
# MAGIC An **instant** is a balance on one day (cash at year end).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE metric_map AS
# MAGIC SELECT * FROM VALUES
# MAGIC   ('RevenueFromContractWithCustomerExcludingAssessedTax', 'revenue',             'duration'),
# MAGIC   ('CostOfGoodsAndServicesSold',                          'cost_of_revenue',     'duration'),
# MAGIC   ('GrossProfit',                                         'gross_profit',        'duration'),
# MAGIC   ('ResearchAndDevelopmentExpense',                       'rd_expense',          'duration'),
# MAGIC   ('SellingAndMarketingExpense',                          'sm_expense',          'duration'),
# MAGIC   ('GeneralAndAdministrativeExpense',                     'ga_expense',          'duration'),
# MAGIC   ('OperatingIncomeLoss',                                 'operating_income',    'duration'),
# MAGIC   ('NetIncomeLoss',                                       'net_income',          'duration'),
# MAGIC   ('NetCashProvidedByUsedInOperatingActivities',          'operating_cash_flow', 'duration'),
# MAGIC   ('PaymentsToAcquirePropertyPlantAndEquipment',          'capex',               'duration'),
# MAGIC   ('ShareBasedCompensation',                              'stock_comp',          'duration'),
# MAGIC   ('CashAndCashEquivalentsAtCarryingValue',               'cash',                'instant'),
# MAGIC   ('Assets',                                              'total_assets',        'instant')
# MAGIC AS t(tag, metric, kind);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. One value per company, metric and fiscal year
# MAGIC
# MAGIC Durations first. Three filters and one tie-break:
# MAGIC
# MAGIC - only facts from annual reports (`form = '10-K'`),
# MAGIC - only periods of about one year (350 to 380 days),
# MAGIC - when the same period was reported more than once, keep the latest filing. If a company later
# MAGIC   corrects a number, the corrected one wins.
# MAGIC
# MAGIC The fiscal year is taken from the **end of the period**, never from the `fy` column.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE financial_facts AS
# MAGIC WITH ranked AS (
# MAGIC   SELECT f.ticker, f.company, m.metric, f.period_start, f.period_end, f.value, f.filed, f.accession_number,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY f.ticker, m.metric, f.period_end ORDER BY f.filed DESC) AS rn
# MAGIC   FROM sec_facts_raw f
# MAGIC   JOIN metric_map m ON f.tag = m.tag
# MAGIC   WHERE m.kind = 'duration'
# MAGIC     AND f.taxonomy = 'us-gaap' AND f.unit = 'USD' AND f.form = '10-K'
# MAGIC     AND datediff(f.period_end, f.period_start) BETWEEN 350 AND 380
# MAGIC )
# MAGIC SELECT ticker, company, year(period_end) AS fiscal_year, period_start, period_end,
# MAGIC        metric, CAST(value AS BIGINT) AS value_usd, filed, accession_number
# MAGIC FROM ranked
# MAGIC WHERE rn = 1;

# COMMAND ----------

# MAGIC %md
# MAGIC Instants next. A balance has no start date, so the one-year filter does not apply. We keep the balances
# MAGIC that fall on a fiscal year end, which we now know from the revenue rows.

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO financial_facts
# MAGIC WITH year_ends AS (
# MAGIC   SELECT DISTINCT ticker, period_end FROM financial_facts WHERE metric = 'revenue'
# MAGIC ),
# MAGIC ranked AS (
# MAGIC   SELECT f.ticker, f.company, m.metric, f.period_end, f.value, f.filed, f.accession_number,
# MAGIC          ROW_NUMBER() OVER (PARTITION BY f.ticker, m.metric, f.period_end ORDER BY f.filed DESC) AS rn
# MAGIC   FROM sec_facts_raw f
# MAGIC   JOIN metric_map m ON f.tag = m.tag
# MAGIC   JOIN year_ends y ON f.ticker = y.ticker AND f.period_end = y.period_end
# MAGIC   WHERE m.kind = 'instant'
# MAGIC     AND f.taxonomy = 'us-gaap' AND f.unit = 'USD' AND f.form = '10-K'
# MAGIC     AND f.period_start IS NULL
# MAGIC )
# MAGIC SELECT ticker, company, year(period_end) AS fiscal_year, CAST(NULL AS DATE) AS period_start, period_end,
# MAGIC        metric, CAST(value AS BIGINT) AS value_usd, filed, accession_number
# MAGIC FROM ranked
# MAGIC WHERE rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, count(*) AS clean_facts, count(DISTINCT metric) AS metrics, min(fiscal_year) AS first_year, max(fiscal_year) AS last_year
# MAGIC FROM financial_facts
# MAGIC GROUP BY ticker
# MAGIC ORDER BY ticker;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The table the analyst will query
# MAGIC
# MAGIC One row per company and year. The ratios an analyst asks for are worked out once, here, so every
# MAGIC answer later uses the same definition.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE financials_annual AS
# MAGIC WITH wide AS (
# MAGIC   SELECT ticker, company, fiscal_year, max(period_end) AS period_end,
# MAGIC          max(CASE WHEN metric = 'revenue'             THEN value_usd END) AS revenue,
# MAGIC          max(CASE WHEN metric = 'cost_of_revenue'     THEN value_usd END) AS cost_of_revenue,
# MAGIC          max(CASE WHEN metric = 'gross_profit'        THEN value_usd END) AS gross_profit,
# MAGIC          max(CASE WHEN metric = 'rd_expense'          THEN value_usd END) AS rd_expense,
# MAGIC          max(CASE WHEN metric = 'sm_expense'          THEN value_usd END) AS sm_expense,
# MAGIC          max(CASE WHEN metric = 'ga_expense'          THEN value_usd END) AS ga_expense,
# MAGIC          max(CASE WHEN metric = 'operating_income'    THEN value_usd END) AS operating_income,
# MAGIC          max(CASE WHEN metric = 'net_income'          THEN value_usd END) AS net_income,
# MAGIC          max(CASE WHEN metric = 'operating_cash_flow' THEN value_usd END) AS operating_cash_flow,
# MAGIC          max(CASE WHEN metric = 'capex'               THEN value_usd END) AS capex,
# MAGIC          max(CASE WHEN metric = 'stock_comp'          THEN value_usd END) AS stock_comp,
# MAGIC          max(CASE WHEN metric = 'cash'                THEN value_usd END) AS cash,
# MAGIC          max(CASE WHEN metric = 'total_assets'        THEN value_usd END) AS total_assets
# MAGIC   FROM financial_facts
# MAGIC   GROUP BY ticker, company, fiscal_year
# MAGIC )
# MAGIC SELECT *,
# MAGIC        round(100 * (revenue / lag(revenue) OVER (PARTITION BY ticker ORDER BY fiscal_year) - 1), 1) AS revenue_growth_pct,
# MAGIC        round(100 * gross_profit / revenue, 1)                    AS gross_margin_pct,
# MAGIC        round(100 * operating_income / revenue, 1)                AS operating_margin_pct,
# MAGIC        operating_cash_flow - capex                               AS free_cash_flow,
# MAGIC        round(100 * (operating_cash_flow - capex) / revenue, 1)   AS free_cash_flow_margin_pct,
# MAGIC        round(100 * stock_comp / revenue, 1)                      AS stock_comp_pct_of_revenue
# MAGIC FROM wide
# MAGIC WHERE revenue IS NOT NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check it before anything is built on it

# COMMAND ----------

checks = {
    "one row per company and year": spark.sql(
        "SELECT count(*) FROM (SELECT ticker, fiscal_year FROM financials_annual GROUP BY ticker, fiscal_year HAVING count(*) > 1)"
    ).first()[0] == 0,
    "three companies": spark.sql("SELECT count(DISTINCT ticker) FROM financials_annual").first()[0] == 3,
    "gross profit never above revenue": spark.sql(
        "SELECT count(*) FROM financials_annual WHERE gross_profit > revenue").first()[0] == 0,
    "latest year has every metric": spark.sql("""
        SELECT count(*) FROM (
          SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY fiscal_year DESC) AS rn FROM financials_annual
        ) WHERE rn = 1 AND (gross_profit IS NULL OR operating_income IS NULL OR net_income IS NULL
                            OR operating_cash_flow IS NULL OR capex IS NULL OR stock_comp IS NULL
                            OR cash IS NULL OR total_assets IS NULL)""").first()[0] == 0,
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "financials_annual failed a check"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Describe the table for whoever reads it next
# MAGIC
# MAGIC In chapter 8 the agent reads these comments to decide which column answers a question. A column with
# MAGIC no description is a column the agent has to guess about.

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE financials_annual IS
# MAGIC   'Annual financials as reported to the SEC in 10-K filings. One row per company (ticker) and fiscal year. All amounts in US dollars. fiscal_year is the calendar year in which the fiscal year ended.';
# MAGIC
# MAGIC ALTER TABLE financials_annual ALTER COLUMN fiscal_year               COMMENT 'Calendar year in which the fiscal year ended. Snowflake and MongoDB end in January, Datadog in December.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN revenue                   COMMENT 'Total revenue for the fiscal year, USD.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN revenue_growth_pct        COMMENT 'Revenue growth versus the prior fiscal year, percent.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN gross_margin_pct          COMMENT 'Gross profit divided by revenue, percent.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN operating_margin_pct      COMMENT 'Operating income (GAAP) divided by revenue, percent. Negative means an operating loss.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN free_cash_flow            COMMENT 'Operating cash flow minus capital expenditure (purchases of property and equipment), USD.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN free_cash_flow_margin_pct COMMENT 'Free cash flow divided by revenue, percent.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN stock_comp                COMMENT 'Stock-based compensation expense for the fiscal year, USD.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN stock_comp_pct_of_revenue COMMENT 'Stock-based compensation divided by revenue, percent.';
# MAGIC ALTER TABLE financials_annual ALTER COLUMN cash                      COMMENT 'Cash and cash equivalents at fiscal year end, USD. Excludes investments.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. The first question the table can answer, and the one it cannot

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, fiscal_year,
# MAGIC        round(revenue / 1e6)        AS revenue_musd,
# MAGIC        revenue_growth_pct,
# MAGIC        gross_margin_pct,
# MAGIC        operating_margin_pct,
# MAGIC        free_cash_flow_margin_pct
# MAGIC FROM financials_annual
# MAGIC WHERE fiscal_year >= 2023
# MAGIC ORDER BY ticker, fiscal_year;

# COMMAND ----------

# MAGIC %md
# MAGIC The table says **how much** revenue grew. It cannot say **why**: more customers, higher prices, or more
# MAGIC usage by the same customers. That answer is written in the annual report, in the PDFs sitting in
# MAGIC `raw_filings`.
# MAGIC
# MAGIC ## Why the rules in step 2 matter
# MAGIC
# MAGIC Skip them and the numbers are wrong in a way that still looks reasonable. This query trusts the `fy`
# MAGIC column and sums whatever it finds.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT r.fy AS fiscal_year_from_fy_column,
# MAGIC        round(sum(r.value) / 1e6)   AS revenue_musd_wrong,
# MAGIC        round(max(c.revenue) / 1e6) AS revenue_musd_correct
# MAGIC FROM sec_facts_raw r
# MAGIC LEFT JOIN financials_annual c ON c.ticker = r.ticker AND c.fiscal_year = r.fy
# MAGIC WHERE r.ticker = 'SNOW'
# MAGIC   AND r.tag = 'RevenueFromContractWithCustomerExcludingAssessedTax'
# MAGIC   AND r.form = '10-K'
# MAGIC GROUP BY r.fy
# MAGIC ORDER BY r.fy;

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── raw_facts            volume
# MAGIC ├── raw_filings          volume
# MAGIC ├── sec_facts_raw        table   raw, with repeats
# MAGIC ├── filings              table   one row per annual report
# MAGIC ├── metric_map           table   accounting tag -> business name
# MAGIC ├── financial_facts      table   one value per company, metric and year
# MAGIC └── financials_annual    table   one row per company and year  <- the analyst's table
# MAGIC ```
# MAGIC
# MAGIC Next: `04_parse_filings` turns the PDFs into text the analyst can search.
