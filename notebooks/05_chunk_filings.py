# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 5: Chunk the filings
# MAGIC
# MAGIC Search by meaning works on short pieces of text. A piece that is too long mixes several topics and
# MAGIC matches nothing well. A piece that is too short loses the sentence that explains it.
# MAGIC
# MAGIC This chapter cuts `filing_text` into chunks and builds:
# MAGIC
# MAGIC **`filing_chunks`: one row per chunk, with an id, the company, the Item, the section and the page.**
# MAGIC
# MAGIC Three decisions, each explained where it is made:
# MAGIC
# MAGIC | Decision | Choice here |
# MAGIC |---|---|
# MAGIC | Where a chunk may never cross | A section heading. A chunk about revenue never continues into cost of revenue. |
# MAGIC | How long a chunk is | About 1,200 characters, with 150 characters of overlap. |
# MAGIC | What goes in | Paragraphs and footnotes. Tables stay out: the numbers already live in `financials_annual`. |

# COMMAND ----------

# MAGIC %pip install --quiet langchain-text-splitters
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the text, one section at a time
# MAGIC
# MAGIC The rows come back in reading order. The whole table is about 1.2 million characters, so it fits in
# MAGIC memory and plain Python is enough.

# COMMAND ----------

from itertools import groupby

rows = spark.sql("""
  SELECT ticker, company, fiscal_year, file_name, item, item_name, section, element_id, page, content
  FROM filing_text
  WHERE element_type IN ('text', 'footnote')
  ORDER BY file_name, element_id
""").collect()

# A section = consecutive rows that share the same report, Item and heading.
sections = [list(group) for _, group in groupby(rows, key=lambda r: (r.file_name, r.item, r.section))]

print(f"{len(rows):,} paragraphs in {len(sections):,} sections")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split each section
# MAGIC
# MAGIC The splitter tries to cut at a paragraph break first, then at the end of a sentence, and only then
# MAGIC between words. Overlap repeats the end of one chunk at the start of the next, so a sentence cut in
# MAGIC two is still whole in one of them.
# MAGIC
# MAGIC Inside a PDF paragraph a line break is only where the line ran out of room, so single line breaks are
# MAGIC turned back into spaces before splitting.

# COMMAND ----------

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def clean(text):
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def chunk_sections(sections, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", ". ", " "], add_start_index=True)
    chunks = []
    for section_rows in sections:
        # Join the section's paragraphs, remembering where each one starts so a chunk can be traced to its page.
        text, starts = "", []
        for r in section_rows:
            starts.append((len(text), r.page))
            text += clean(r.content) + "\n\n"
        first = section_rows[0]
        for doc in splitter.create_documents([text.strip()]):
            page = [p for start, p in starts if start <= doc.metadata["start_index"]][-1]
            chunks.append((first.ticker, first.company, first.fiscal_year, first.file_name, first.item,
                           first.item_name, first.section, page, doc.page_content.strip()))
    return chunks


chunks = chunk_sections(sections, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"{len(chunks):,} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why 1,200? Try other sizes on the same text
# MAGIC
# MAGIC Chunk size changes how many rows the index will hold and how focused each one is. Nothing is saved
# MAGIC here; this only counts.

# COMMAND ----------

for size, overlap in [(400, 50), (1200, 150), (4000, 500)]:
    trial = chunk_sections(sections, size, overlap)
    lengths = sorted(len(c[-1]) for c in trial)
    print(f"chunk_size {size:>5}, overlap {overlap:>3}: {len(trial):>5,} chunks, median {lengths[len(lengths) // 2]:>5,} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC - **400**: nearly three times as many chunks, most of them one or two sentences. A sentence like "The
# MAGIC   increase was primarily due to..." ends up without the sentence that says what increased.
# MAGIC - **4,000**: half as many chunks. Short sections are unchanged, but every long section becomes one
# MAGIC   passage covering several ideas, and the model has to find the relevant line inside it.
# MAGIC - **1,200**: about one to three paragraphs, which is how a 10-K states a fact and its reason.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save the chunks
# MAGIC
# MAGIC Two details matter for the next chapter:
# MAGIC
# MAGIC - **`chunk_id`** is a stable key, like `SNOW-2026-0412`. A search index needs one row = one id.
# MAGIC - **`chunk_text`** starts with one line saying where the chunk is from. A chunk that only says "revenue
# MAGIC   increased 29%" does not say whose revenue. The first line fixes that, for the search and for the model.
# MAGIC
# MAGIC Change Data Feed is switched on so the index in chapter 6 can follow changes to this table.

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

records, counter = [], {}
for ticker, company, fiscal_year, file_name, item, item_name, section, page, content in chunks:
    counter[file_name] = counter.get(file_name, 0) + 1
    chunk_id = f"{ticker}-{fiscal_year}-{counter[file_name]:04d}"
    where = f"{company} annual report (10-K), fiscal year {fiscal_year}. {item}{' ' + item_name if item_name else ''} > {section}"
    records.append((chunk_id, ticker, company, fiscal_year, file_name, item, item_name, section, page,
                    content, f"{where}\n\n{content}"))

schema = StructType([
    StructField("chunk_id", StringType(), False), StructField("ticker", StringType()), StructField("company", StringType()),
    StructField("fiscal_year", IntegerType()), StructField("file_name", StringType()), StructField("item", StringType()),
    StructField("item_name", StringType()), StructField("section", StringType()), StructField("page", IntegerType()),
    StructField("content", StringType()), StructField("chunk_text", StringType()),
])

(spark.createDataFrame(records, schema)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("filing_chunks"))

print(f"{len(records):,} chunks written to filing_chunks")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE filing_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC COMMENT ON TABLE filing_chunks IS
# MAGIC   'Annual report (10-K) text cut into chunks of about 1,200 characters for search. One row per chunk. chunk_text is what gets embedded: a first line naming the company, fiscal year, Item and section, then the text. Tables are not included.';
# MAGIC
# MAGIC ALTER TABLE filing_chunks ALTER COLUMN chunk_id   COMMENT 'Stable key: ticker, fiscal year and position in the report, for example SNOW-2026-0412.';
# MAGIC ALTER TABLE filing_chunks ALTER COLUMN page       COMMENT 'Page in the PDF where the chunk starts.';
# MAGIC ALTER TABLE filing_chunks ALTER COLUMN content    COMMENT 'The chunk text alone.';
# MAGIC ALTER TABLE filing_chunks ALTER COLUMN chunk_text COMMENT 'Source line plus the chunk text. This column is embedded and searched.';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check it

# COMMAND ----------

checks = {
    "chunk_id is unique": spark.sql(
        "SELECT count(*) - count(DISTINCT chunk_id) FROM filing_chunks").first()[0] == 0,
    "three companies": spark.sql("SELECT count(DISTINCT ticker) FROM filing_chunks").first()[0] == 3,
    "no chunk longer than the limit": spark.sql(
        f"SELECT count(*) FROM filing_chunks WHERE length(content) > {CHUNK_SIZE}").first()[0] == 0,
    "no empty chunk, every chunk has a page": spark.sql(
        "SELECT count(*) FROM filing_chunks WHERE length(trim(content)) = 0 OR page IS NULL").first()[0] == 0,
    "every chunk_text starts with its source line": spark.sql(
        "SELECT count(*) FROM filing_chunks WHERE chunk_text NOT LIKE concat(company, ' annual report (10-K)%')").first()[0] == 0,
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "filing_chunks failed a check"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, count(*) AS chunks, cast(avg(length(content)) AS INT) AS avg_characters,
# MAGIC        sum(CASE WHEN item = 'Item 1A' THEN 1 ELSE 0 END) AS risk_factor_chunks,
# MAGIC        sum(CASE WHEN item = 'Item 7'  THEN 1 ELSE 0 END) AS managements_discussion_chunks
# MAGIC FROM filing_chunks
# MAGIC GROUP BY ticker
# MAGIC ORDER BY ticker;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The answer from chapter 4, now as one chunk

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT chunk_id, page, chunk_text
# MAGIC FROM filing_chunks
# MAGIC WHERE ticker = 'SNOW' AND item = 'Item 7' AND content LIKE '%increased consumption of our platform by existing customers%'
# MAGIC LIMIT 1;

# COMMAND ----------

# MAGIC %md
# MAGIC One row, one id, one page, and a first line that says which company and which part of the report.
# MAGIC That row is what the analyst will be handed when someone asks why Snowflake's revenue grew.
# MAGIC
# MAGIC It is still found by exact words. Chapter 6 fixes that.
# MAGIC
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── filing_text          table   paragraphs, tables and footnotes
# MAGIC ├── filing_chunks        table   search-sized chunks with id, Item, section, page  <- what gets indexed
# MAGIC └── financials_annual    table   the numbers
# MAGIC ```
# MAGIC
# MAGIC Next: `06_vector_search_index` builds a Vector Search index on `filing_chunks.chunk_text`.
