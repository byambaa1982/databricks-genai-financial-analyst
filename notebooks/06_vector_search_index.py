# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 6: Build a Vector Search index
# MAGIC
# MAGIC Chapter 5 ended with the right chunk, found by typing the exact words it contains. Nobody asks
# MAGIC questions that way. This chapter makes the chunks searchable **by meaning**:
# MAGIC
# MAGIC | Piece | What it is | Here |
# MAGIC |---|---|---|
# MAGIC | Embedding model | Turns a text into a list of numbers. Texts that mean similar things get similar numbers. | `databricks-gte-large-en` |
# MAGIC | Index | Stores those numbers for every chunk and finds the nearest ones to a question. | `filing_chunks_index` |
# MAGIC | Endpoint | The compute that serves the index. | `fin_analyst_search` |
# MAGIC
# MAGIC The index is a **Delta Sync** index: it is built from the `filing_chunks` table and Databricks computes
# MAGIC the embeddings for us. We never handle a vector ourselves.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The endpoint
# MAGIC
# MAGIC Created once. The first time takes several minutes; after that this cell finds it and moves on.
# MAGIC Free Edition allows one Vector Search endpoint, and this project uses it.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT in existing:
    print(f"Endpoint {VS_ENDPOINT} already exists")
else:
    vsc.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Endpoint {VS_ENDPOINT} created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The index
# MAGIC
# MAGIC Four choices define it:
# MAGIC
# MAGIC - **source table**: `filing_chunks` (Change Data Feed was switched on in chapter 5 for this),
# MAGIC - **primary key**: `chunk_id`, so every search result can be traced to one row,
# MAGIC - **column to embed**: `chunk_text`, the text with its source line on top,
# MAGIC - **sync mode**: `TRIGGERED`. The index updates when we ask it to, not continuously. The filings change
# MAGIC   once a year, so there is nothing to keep running.

# COMMAND ----------

import time

source_table = f"{CATALOG}.{SCHEMA}.filing_chunks"

if VS_INDEX in [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)
    index.sync()
    print(f"Index {VS_INDEX} already exists, sync started")
else:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=source_table,
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
        pipeline_type="TRIGGERED",
    )
    print(f"Index {VS_INDEX} created, first sync started")


def wait_for_sync(index):
    """Wait until the index is online and has nothing left to process."""
    time.sleep(30)  # give the sync a moment to register
    while True:
        status = index.describe()["status"]
        if status.get("ready") and status.get("detailed_state", "").endswith("NO_PENDING_UPDATE"):
            return status
        time.sleep(30)


status = wait_for_sync(index)
print(f"Ready: {status['ready']}, rows indexed: {status['indexed_row_count']:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ask by meaning
# MAGIC
# MAGIC The question from chapter 3, in plain words. None of the key words in the answer ("consumption",
# MAGIC "existing customers", "net revenue retention") appear in the question.
# MAGIC
# MAGIC The search is limited to Snowflake with a filter. The index keeps the other columns of the table, so
# MAGIC any of them can narrow a search.

# COMMAND ----------

QUESTION = "Why did Snowflake's revenue grow? Was it higher prices or more usage?"
COLUMNS = ["chunk_id", "ticker", "item", "section", "page", "content"]


def search(question, num_results=5, filters=None, query_type="ANN"):
    result = index.similarity_search(query_text=question, columns=COLUMNS, num_results=num_results,
                                     filters=filters, query_type=query_type, disable_notice=True)
    names = [c["name"] for c in result["manifest"]["columns"]]
    return [dict(zip(names, row)) for row in result["result"]["data_array"]]


def show(hits):
    for rank, h in enumerate(hits, 1):
        print(f"{rank:>2}  {h['score']:.3f}  {h['chunk_id']}  p.{int(h['page']):<4} {h['item']:<8} {h['section'][:60]}")


hits = search(QUESTION, filters={"ticker": "SNOW"})
show(hits)
print()
print(hits[0]["content"][:700])

# COMMAND ----------

# MAGIC %md
# MAGIC The top results come from "Expanding Within our Existing Customer Base" and "Revenue", the parts of the
# MAGIC report that discuss growth. Not every one states the reason, and one is a risk factor that talks about
# MAGIC revenue without explaining this year's growth. Search finds candidates. Reading them and answering is
# MAGIC the model's job, in chapter 7.
# MAGIC
# MAGIC ## 4. A tighter filter: search only where the answer can be
# MAGIC
# MAGIC "Why did revenue grow" is answered in Item 7, Management's Discussion. Adding the Item to the filter
# MAGIC removes the risk factor and makes room for one more useful chunk.

# COMMAND ----------

show(search(QUESTION, filters={"ticker": "SNOW", "item": "Item 7"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The same search from SQL
# MAGIC
# MAGIC `vector_search()` is a table function, so the index can be used in any query, joined to any table.
# MAGIC This is the form the agent's tool will use in chapter 8.

# COMMAND ----------

display(spark.sql(f"""
  SELECT chunk_id, ticker, item, section, page, left(content, 200) AS content_start
  FROM vector_search(
    index => '{VS_INDEX}',
    query_text => 'How many large customers does Datadog have?',
    num_results => 5
  )
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Check it

# COMMAND ----------

chunk_count = spark.sql("SELECT count(*) FROM filing_chunks").first()[0]
answer_chunk = spark.sql("""
  SELECT chunk_id FROM filing_chunks
  WHERE ticker = 'SNOW' AND content LIKE '%increased consumption of our platform by existing customers%' LIMIT 1
""").first()[0]
top_ids = [h["chunk_id"] for h in search(QUESTION, num_results=10, filters={"ticker": "SNOW"})]
print(f"The chunk found by exact words in chapter 5, {answer_chunk}, is result number {top_ids.index(answer_chunk) + 1 if answer_chunk in top_ids else 'beyond 10'} for the plain-words question")

checks = {
    "every chunk is in the index": index.describe()["status"]["indexed_row_count"] == chunk_count,
    "a filtered search returns only that company": all(
        h["ticker"] == "DDOG" for h in search("What are the main risks?", filters={"ticker": "DDOG"})),
    f"the chunk with the answer ({answer_chunk}) is in the top 10 for the plain-words question": answer_chunk in top_ids,
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "the index failed a check"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── filing_chunks          table   about 1,600 chunks
# MAGIC ├── filing_chunks_index    index   the same chunks, searchable by meaning
# MAGIC └── financials_annual      table   the numbers
# MAGIC
# MAGIC fin_analyst_search         Vector Search endpoint serving the index
# MAGIC ```
# MAGIC
# MAGIC The analyst now has both halves: a table that knows **how much**, and an index that can find **why**.
# MAGIC
# MAGIC Next: `07_rag_chain` puts a model in front of the index, so the answer comes back as a sentence with
# MAGIC its sources.
