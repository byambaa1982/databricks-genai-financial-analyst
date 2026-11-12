# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 4: Turn the PDFs into text
# MAGIC
# MAGIC The reason behind a number is written in the annual report. Right now the reports are three PDFs in a
# MAGIC volume, about 150 to 200 pages each. A model cannot search a PDF, so this chapter turns them into rows:
# MAGIC
# MAGIC **`filing_text`: one row per paragraph or table, labelled with the company, the page, and the part of
# MAGIC the report it came from.**
# MAGIC
# MAGIC The label matters as much as the text. An annual report always has the same numbered parts
# MAGIC (Item 1 Business, Item 1A Risk Factors, Item 7 Management's Discussion, Item 8 Financial Statements).
# MAGIC "Why did revenue grow" is answered in Item 7. If every paragraph knows its Item, the analyst can look
# MAGIC in the right place and can say where an answer came from.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parse each PDF once
# MAGIC
# MAGIC `ai_parse_document` reads a file and returns its layout: every heading, paragraph and table, in reading
# MAGIC order, with the page it sits on.
# MAGIC
# MAGIC Parsing is the slow and costly step (about a minute per report), so the result is stored as it comes
# MAGIC back, and only files that are not in the table yet are parsed. Run this cell twice and the second run
# MAGIC does nothing.

# COMMAND ----------

spark.sql("""
  CREATE TABLE IF NOT EXISTS filings_parsed_raw (file_name STRING, parsed VARIANT, parsed_at TIMESTAMP)
  COMMENT 'Raw output of ai_parse_document, one row per PDF in the raw_filings volume. Not cleaned.'
""")

new_files = spark.sql(f"""
  INSERT INTO filings_parsed_raw
  SELECT _metadata.file_name, ai_parse_document(content), current_timestamp()
  FROM read_files('{FILINGS_VOLUME}/', format => 'binaryFile', fileNamePattern => '*.pdf')
  WHERE _metadata.file_name NOT IN (SELECT file_name FROM filings_parsed_raw)
""").first()[0]

print(f"{new_files} new file(s) parsed")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT file_name,
# MAGIC        parsed:error_status::string                                   AS error,
# MAGIC        size(try_cast(parsed:document:pages    AS ARRAY<VARIANT>))    AS pages,
# MAGIC        size(try_cast(parsed:document:elements AS ARRAY<VARIANT>))    AS elements
# MAGIC FROM filings_parsed_raw
# MAGIC ORDER BY file_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. What the parser found
# MAGIC
# MAGIC Each element has a type. Text, tables and footnotes carry content. Section headers tell us where we
# MAGIC are. Page numbers and page headers are noise.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT e.value:type::string AS element_type, count(*) AS elements, sum(length(e.value:content::string)) AS characters
# MAGIC FROM filings_parsed_raw p, LATERAL variant_explode(p.parsed:document:elements) e
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2 DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. One row per element, each one knowing its Item and its section
# MAGIC
# MAGIC Two labels are carried down the document in reading order:
# MAGIC
# MAGIC - **item**: set when a line starts with "Item 7." and kept until the next Item starts,
# MAGIC - **section**: the nearest heading above the element.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE filing_elements AS
# MAGIC WITH elements AS (
# MAGIC   SELECT p.file_name,
# MAGIC          e.pos                                 AS element_id,
# MAGIC          e.value:type::string                  AS element_type,
# MAGIC          e.value:bbox[0]:page_id::int + 1      AS page,
# MAGIC          e.value:content::string               AS content
# MAGIC   FROM filings_parsed_raw p, LATERAL variant_explode(p.parsed:document:elements) e
# MAGIC ),
# MAGIC marked AS (
# MAGIC   SELECT *,
# MAGIC          CASE WHEN element_type <> 'table' AND content RLIKE '(?i)^ *item +[0-9]+[A-C]?[. :]'
# MAGIC               THEN concat('Item ', upper(regexp_extract(content, '(?i)^ *item +([0-9]+[A-C]?)', 1))) END AS item_starts_here,
# MAGIC          CASE WHEN element_type IN ('section_header', 'title') THEN content END                          AS heading_here
# MAGIC   FROM elements
# MAGIC )
# MAGIC SELECT file_name, element_id, page, element_type,
# MAGIC        last_value(item_starts_here, true) OVER reading_order AS item,
# MAGIC        last_value(heading_here, true)     OVER reading_order AS section,
# MAGIC        content
# MAGIC FROM marked
# MAGIC WINDOW reading_order AS (PARTITION BY file_name ORDER BY element_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);

# COMMAND ----------

# MAGIC %md
# MAGIC Did every report give us the four Items the analyst will use most?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT file_name,
# MAGIC        min(CASE WHEN item = 'Item 1'  THEN page END) AS item_1_business,
# MAGIC        min(CASE WHEN item = 'Item 1A' THEN page END) AS item_1a_risk_factors,
# MAGIC        min(CASE WHEN item = 'Item 7'  THEN page END) AS item_7_managements_discussion,
# MAGIC        min(CASE WHEN item = 'Item 8'  THEN page END) AS item_8_financial_statements,
# MAGIC        count(DISTINCT item)                          AS items_found
# MAGIC FROM filing_elements
# MAGIC GROUP BY file_name
# MAGIC ORDER BY file_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The table the analyst will search
# MAGIC
# MAGIC Keep what carries meaning (text, tables, footnotes), drop the cover pages before Item 1, and add the
# MAGIC company and fiscal year from the `filings` table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE filing_text AS
# MAGIC SELECT f.ticker, f.company, f.fiscal_year, e.file_name,
# MAGIC        e.element_id, e.page, e.element_type, e.item, n.item_name, e.section, e.content
# MAGIC FROM filing_elements e
# MAGIC JOIN filings f ON f.file_name = e.file_name
# MAGIC LEFT JOIN (
# MAGIC   SELECT * FROM VALUES
# MAGIC     ('Item 1',  'Business'),
# MAGIC     ('Item 1A', 'Risk Factors'),
# MAGIC     ('Item 1C', 'Cybersecurity'),
# MAGIC     ('Item 2',  'Properties'),
# MAGIC     ('Item 3',  'Legal Proceedings'),
# MAGIC     ('Item 5',  'Market for Common Equity'),
# MAGIC     ('Item 7',  'Management Discussion and Analysis (MD&A)'),
# MAGIC     ('Item 7A', 'Market Risk'),
# MAGIC     ('Item 8',  'Financial Statements and Notes'),
# MAGIC     ('Item 9A', 'Controls and Procedures')
# MAGIC   AS t(item, item_name)
# MAGIC ) n ON n.item = e.item
# MAGIC WHERE e.element_type IN ('text', 'table', 'footnote')
# MAGIC   AND e.item IS NOT NULL
# MAGIC   AND length(trim(e.content)) > 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE filing_text IS
# MAGIC   'Text of each annual report (10-K), one row per paragraph, table or footnote, in reading order. item is the numbered part of the report (Item 7 = Management Discussion and Analysis); section is the nearest heading above the row; page is the page in the PDF.';
# MAGIC
# MAGIC ALTER TABLE filing_text ALTER COLUMN item    COMMENT 'Numbered part of the 10-K the row belongs to, for example Item 1A or Item 7.';
# MAGIC ALTER TABLE filing_text ALTER COLUMN section COMMENT 'Nearest heading above this row in the report.';
# MAGIC ALTER TABLE filing_text ALTER COLUMN page    COMMENT 'Page number in the PDF in the raw_filings volume, starting at 1.';
# MAGIC ALTER TABLE filing_text ALTER COLUMN content COMMENT 'The text of the paragraph or footnote. Tables are stored as HTML.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Check it

# COMMAND ----------

checks = {
    "three reports": spark.sql("SELECT count(DISTINCT file_name) FROM filing_text").first()[0] == 3,
    "no report failed to parse": spark.sql(
        "SELECT count(*) FROM filings_parsed_raw WHERE parsed:error_status::string IS NOT NULL").first()[0] == 0,
    "every report has Items 1, 1A, 7 and 8": spark.sql("""
        SELECT count(*) FROM (
          SELECT file_name FROM filing_text WHERE item IN ('Item 1', 'Item 1A', 'Item 7', 'Item 8')
          GROUP BY file_name HAVING count(DISTINCT item) = 4)""").first()[0] == 3,
    "every row has a page and a section": spark.sql(
        "SELECT count(*) FROM filing_text WHERE page IS NULL OR section IS NULL").first()[0] == 0,
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "filing_text failed a check"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, item, item_name, count(*) AS elements, sum(length(content)) AS characters, min(page) AS first_page, max(page) AS last_page
# MAGIC FROM filing_text
# MAGIC WHERE item IN ('Item 1', 'Item 1A', 'Item 7', 'Item 8')
# MAGIC GROUP BY ticker, item, item_name
# MAGIC ORDER BY ticker, first_page;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. The question from chapter 3, asked of the text
# MAGIC
# MAGIC The table said Snowflake's revenue grew 29.2% in fiscal 2026. Here is what management wrote about it.
# MAGIC This is a plain keyword filter, only to show that the answer is in there.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT page, section, content
# MAGIC FROM filing_text
# MAGIC WHERE ticker = 'SNOW' AND item = 'Item 7' AND element_type = 'text'
# MAGIC   AND lower(content) LIKE '%revenue increased%'
# MAGIC ORDER BY element_id
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC The reason is there. But finding it took the exact words "revenue increased". Ask the same thing in
# MAGIC other words ("what drove growth", "was it price or volume") and a keyword filter finds nothing.
# MAGIC
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── raw_filings            volume   3 PDFs
# MAGIC ├── filings_parsed_raw     table    parser output, one row per PDF
# MAGIC ├── filing_elements        table    every element with its Item and section
# MAGIC ├── filing_text            table    paragraphs, tables and footnotes  <- what the analyst will search
# MAGIC └── financials_annual      table    the numbers, from chapter 3
# MAGIC ```
# MAGIC
# MAGIC Next: `05_chunk_filings` cuts the text into pieces sized for search by meaning.
