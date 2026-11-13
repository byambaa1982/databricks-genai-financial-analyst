# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 7: A retrieval chain over the filings
# MAGIC
# MAGIC Chapter 6 returns the five most relevant chunks. Someone still has to read them and write the answer.
# MAGIC This chapter gives that job to a model:
# MAGIC
# MAGIC ```
# MAGIC question ─► search the index ─► put the chunks in the prompt ─► model ─► answer with sources
# MAGIC ```
# MAGIC
# MAGIC This pattern is called retrieval-augmented generation (RAG). The model does not answer from memory.
# MAGIC It answers from the text we hand it, and it says which chunk each statement came from.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The two parts: a retriever and a model

# COMMAND ----------

import contextlib
import io

from databricks_langchain import ChatDatabricks, DatabricksVectorSearch

TOP_K = 8   # how many chunks the model gets to read. Section 6 shows why not fewer.

llm = ChatDatabricks(endpoint=LLM_MODEL, temperature=0, max_tokens=600)

vector_store = DatabricksVectorSearch(
    index_name=VS_INDEX,
    columns=["chunk_id", "ticker", "company", "fiscal_year", "item", "section", "page", "content"],
)


def retrieve(inputs):
    """Find the chunks closest to the question. If a ticker is given, search that company only."""
    filters = {"ticker": inputs["ticker"]} if inputs.get("ticker") else None
    with contextlib.redirect_stdout(io.StringIO()):   # hides the client's notice about notebook tokens
        return vector_store.similarity_search(inputs["question"], k=inputs.get("k") or TOP_K, filter=filters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The prompt: what the model may and may not do
# MAGIC
# MAGIC Three rules carry the whole chapter:
# MAGIC
# MAGIC - answer **only** from the excerpts,
# MAGIC - put the **chunk id** after each statement, so every claim can be checked,
# MAGIC - if the excerpts do not contain the answer, **say so** instead of guessing.
# MAGIC
# MAGIC Each excerpt goes into the prompt with its id, company, Item, section and page on the first line.

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND = "The filings I was given do not say."

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst's assistant. Answer the question using only the excerpts from annual reports "
     "(10-K filings) below. After each statement, cite the excerpt it came from by its id in square brackets, "
     "for example [SNOW-2026-0123]. Use two to four sentences. Do not give investment advice. "
     f"If the excerpts do not contain the answer, reply exactly: {NOT_FOUND}"),
    ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['chunk_id']}] {d.metadata['company']}, fiscal year {int(d.metadata['fiscal_year'])}, "
        f"{d.metadata['item']} > {d.metadata['section']}, page {int(d.metadata['page'])}\n{d.metadata['content']}"
        for d in docs
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The chain
# MAGIC
# MAGIC Each step adds one key to a dictionary and passes it on: first the chunks, then the chunks as text,
# MAGIC then the answer. Keeping the chunks in the result is what lets us show the sources afterwards.

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

rag_chain = (
    RunnablePassthrough.assign(docs=RunnableLambda(retrieve))
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
    | RunnablePassthrough.assign(answer=prompt | llm | StrOutputParser())
)


def ask(question, ticker=None, k=None):
    result = rag_chain.invoke({"question": question, "ticker": ticker, "k": k})
    print("Q:", question)
    print()
    print("A:", result["answer"])
    print()
    print("Chunks the model was given:")
    for d in result["docs"]:
        m = d.metadata
        print(f"   {m['chunk_id']}  p.{int(m['page']):<4} {m['item']:<8} {m['section'][:60]}")
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. The question this project started with

# COMMAND ----------

snow = ask("Why did Snowflake's revenue grow? Was it higher prices or more usage?", ticker="SNOW")

# COMMAND ----------

# MAGIC %md
# MAGIC **More usage, not higher prices.** Revenue grew because existing customers consumed more, and the model
# MAGIC points to the chunk that says so. Open the PDF at page 72 and the sentence is there.
# MAGIC
# MAGIC That is the answer chapter 3 could not give. The table knew revenue grew 29.2%. The filing knew why.

# COMMAND ----------

# MAGIC %md
# MAGIC The same chain, another company. Nothing changes but the ticker.

# COMMAND ----------

ddog = ask("Why did Datadog's revenue grow? Did it come from new customers or existing ones?", ticker="DDOG")

# COMMAND ----------

# MAGIC %md
# MAGIC About 75% from existing customers and 25% from new ones, with the chunk and page to check it against.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. What happens without retrieval
# MAGIC
# MAGIC The same model, the same question, no excerpts.

# COMMAND ----------

print(llm.invoke("Why did Snowflake's revenue grow in fiscal year 2026? Was it higher prices or more usage? "
                 "Answer in two to four sentences.").content)

# COMMAND ----------

# MAGIC %md
# MAGIC The model has never seen this report: it was trained before the report was written. So it says what is
# MAGIC usually true of the company and offers no source. It sounds reasonable and cannot be checked.
# MAGIC Retrieval is what turns "usually" into "page 72 says".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Why eight chunks and not five
# MAGIC
# MAGIC The same Snowflake question, with the model given only the top five chunks.

# COMMAND ----------

snow_top5 = ask("Why did Snowflake's revenue grow? Was it higher prices or more usage?", ticker="SNOW", k=5)

# COMMAND ----------

# MAGIC %md
# MAGIC With five chunks the model refuses. The sentence that states the reason is in the chunk ranked sixth, so
# MAGIC the model never saw it, and it followed the rule: no answer in the excerpts, no answer.
# MAGIC
# MAGIC Two things to take from this:
# MAGIC
# MAGIC - The refusal is the chain working. A wrong answer here would have been worse.
# MAGIC - How many chunks to retrieve is a setting to test, not to guess. Too few and the answer is missed. Too
# MAGIC   many and the prompt fills with text that does not matter, which costs more and can distract the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. A question the filings cannot answer

# COMMAND ----------

unknown = ask("What was Snowflake's share price on the day this annual report was filed?", ticker="SNOW")

# COMMAND ----------

# MAGIC %md
# MAGIC The search always returns something: here, the nearest chunks are about stock ownership and the market
# MAGIC the shares trade on. None of them holds a share price for that day, and the model says so.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. A question about numbers

# COMMAND ----------

numbers = ask("What was MongoDB's free cash flow margin in fiscal 2026, and how does it compare with Datadog's latest year?")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, fiscal_year, free_cash_flow_margin_pct
# MAGIC FROM financials_annual
# MAGIC WHERE (ticker = 'MDB' AND fiscal_year = 2026) OR (ticker = 'DDOG' AND fiscal_year = 2025);

# COMMAND ----------

# MAGIC %md
# MAGIC The chain refuses, and it is right to: no paragraph in either report states a free cash flow margin. But
# MAGIC the answer exists. It is two rows in `financials_annual`, the table from chapter 3.
# MAGIC
# MAGIC Searching text is the wrong tool for a question about numbers. The analyst needs a second tool that
# MAGIC queries the table, and something that decides which tool a question needs. That is the next two chapters.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Check it

# COMMAND ----------

import re


def cited_ids(result):
    return set(re.findall(r"\[([A-Z]+-\d{4}-\d{4})\]", result["answer"]))


def given_ids(result):
    return {d.metadata["chunk_id"] for d in result["docs"]}


checks = {
    "the Snowflake answer cites at least one chunk": len(cited_ids(snow)) >= 1,
    "every chunk cited was one the model was given": all(cited_ids(r) <= given_ids(r) for r in (snow, ddog, numbers)),
    "a ticker filter keeps the search inside that company": all(d.metadata["ticker"] == "DDOG" for d in ddog["docs"]),
    "an unanswerable question is refused, not guessed": NOT_FOUND.rstrip(".") in unknown["answer"],
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "the chain failed a check"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC question ─► filing_chunks_index ─► prompt with 8 chunks ─► model ─► answer with chunk ids
# MAGIC ```
# MAGIC
# MAGIC | Question | Chain |
# MAGIC |---|---|
# MAGIC | Why did revenue grow? | Answers, with the page to check |
# MAGIC | Something the filings do not contain | Says so |
# MAGIC | A number, or a comparison of numbers | Cannot answer: the numbers are in a table, not in the text |
# MAGIC
# MAGIC Next: `08_tools` turns a query on `financials_annual` into a tool the analyst can call.
