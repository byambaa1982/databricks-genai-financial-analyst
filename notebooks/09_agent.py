# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 9: The agent
# MAGIC
# MAGIC Chapter 8 stopped in the middle of a question. The model had MongoDB's numbers, asked for Datadog's,
# MAGIC and our code had nothing more to give: it ran the tools once and expected an answer.
# MAGIC
# MAGIC An **agent** is the same model and the same tools with one thing added, a loop:
# MAGIC
# MAGIC ```
# MAGIC            ┌──────────────────────────────────────────────┐
# MAGIC            ▼                                              │
# MAGIC question ─► model ─► tool call? ── yes ─► run the tool ─► rows
# MAGIC                          │
# MAGIC                          no
# MAGIC                          ▼
# MAGIC                 answer with sources
# MAGIC ```
# MAGIC
# MAGIC The model decides what to look up and when it has enough. Our code only keeps handing back rows until
# MAGIC the model replies with text. This chapter writes that loop, gives it rules for citing sources, and asks
# MAGIC the question this project started with.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The analyst, in one file
# MAGIC
# MAGIC The agent is written to a Python file, `analyst.py`, so the next chapters can import the same code
# MAGIC instead of copying it. It has three parts:
# MAGIC
# MAGIC | Part | What it does |
# MAGIC |---|---|
# MAGIC | `SYSTEM` | The rules: every fact from a tool, every statement cited, say so when the tools have no answer |
# MAGIC | the `for` loop in `ask` | Calls the model, runs the tools it asks for, hands back the rows, repeats |
# MAGIC | `steps` | A record of every tool call and what it returned, so an answer can be checked afterwards |
# MAGIC
# MAGIC Three details make the loop safe to leave running:
# MAGIC
# MAGIC - **a limit**, `max_steps`. A model that keeps asking for tools would otherwise run, and cost, forever,
# MAGIC - **a failed tool call goes back to the model as text.** If it forgets an argument, it reads the error and
# MAGIC   tries again instead of stopping the whole answer,
# MAGIC - **a wait and a retry** when the model endpoint says it is busy.

# COMMAND ----------

# MAGIC %%writefile analyst.py
# MAGIC """The AI financial analyst: two Unity Catalog functions in a loop. Written by notebooks/09_agent."""
# MAGIC import json
# MAGIC import re
# MAGIC import time
# MAGIC import warnings
# MAGIC
# MAGIC from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# MAGIC
# MAGIC NOT_FOUND = "The filings and the financials I can search do not say."
# MAGIC NO_ADVICE = "I cannot give investment advice."
# MAGIC
# MAGIC SYSTEM = f"""You are a financial analyst's assistant for three companies: Snowflake (SNOW), Datadog (DDOG) and MongoDB (MDB).
# MAGIC
# MAGIC Rules:
# MAGIC 1. Get every fact from a tool. Never answer from memory.
# MAGIC 2. Numbers, ratios and trends come from company_financials. Reasons, strategy, customers and risks come from search_filings. A question that asks how much and why needs both tools.
# MAGIC 3. Cite the source after every statement. After a number, cite the table row like this: [financials_annual: SNOW FY2026]. After a statement taken from a filing, cite the excerpt id like this: [SNOW-2026-0123]. Only cite rows and excerpts a tool returned.
# MAGIC 4. If the tools do not return the answer, reply exactly: {NOT_FOUND}
# MAGIC 5. If asked whether to buy, sell or hold a stock, or for a price forecast, reply exactly: {NO_ADVICE}
# MAGIC
# MAGIC Answer in two to five sentences."""
# MAGIC
# MAGIC CHUNK_ID = r"[A-Z]+-\d{4}-\d{4}"
# MAGIC
# MAGIC
# MAGIC def build_analyst(tool_functions, llm_endpoint, system=SYSTEM, max_steps=6):
# MAGIC     """Returns ask(question). tool_functions are full Unity Catalog function names."""
# MAGIC     from databricks_langchain import ChatDatabricks, UCFunctionToolkit
# MAGIC
# MAGIC     warnings.filterwarnings("ignore", message="Ignoring the default notebook Spark session")
# MAGIC     tools = {t.name: t for t in UCFunctionToolkit(function_names=tool_functions).tools}
# MAGIC     llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0, max_tokens=800).bind_tools(list(tools.values()))
# MAGIC
# MAGIC     def call_model(messages):
# MAGIC         for attempt in range(5):                          # a shared endpoint limits requests per minute
# MAGIC             try:
# MAGIC                 return llm.invoke(messages)
# MAGIC             except Exception as error:
# MAGIC                 if attempt == 4 or not any(w in str(error) for w in ("429", "REQUEST_LIMIT_EXCEEDED", "rate limit")):
# MAGIC                     raise
# MAGIC                 time.sleep(20 * (attempt + 1))
# MAGIC
# MAGIC     def ask(question):
# MAGIC         messages = [SystemMessage(system), HumanMessage(question)]
# MAGIC         steps = []
# MAGIC         for _ in range(max_steps):
# MAGIC             reply = call_model(messages)
# MAGIC             messages.append(reply)
# MAGIC             if not reply.tool_calls:                      # text instead of a tool call: the answer
# MAGIC                 return {"question": question, "answer": reply.content, "steps": steps}
# MAGIC             for call in reply.tool_calls:
# MAGIC                 try:
# MAGIC                     rows = tools[call["name"]].invoke(call["args"])
# MAGIC                 except Exception as error:                # a failed call goes back as text, so the model can correct it
# MAGIC                     rows = f"ERROR: {type(error).__name__}: {str(error)[:300]}"
# MAGIC                 messages.append(ToolMessage(content=rows, tool_call_id=call["id"]))
# MAGIC                 steps.append({"tool": call["name"].split("__")[-1], "args": call["args"], "rows": rows})
# MAGIC         return {"question": question, "answer": f"Stopped after {max_steps} steps without an answer.", "steps": steps}
# MAGIC
# MAGIC     return ask
# MAGIC
# MAGIC
# MAGIC def tools_used(result):
# MAGIC     return [s["tool"] for s in result["steps"]]
# MAGIC
# MAGIC
# MAGIC def chunks_returned(result):
# MAGIC     """Ids of every excerpt search_filings gave the model."""
# MAGIC     return {i for s in result["steps"] if s["tool"] == "search_filings" for i in re.findall(CHUNK_ID, s["rows"])}
# MAGIC
# MAGIC
# MAGIC def chunks_cited(result):
# MAGIC     return set(re.findall(rf"\[({CHUNK_ID})\]", result["answer"]))
# MAGIC
# MAGIC
# MAGIC def rows_cited(result):
# MAGIC     """(ticker, fiscal year) of every table row the answer cites."""
# MAGIC     return {(t, int(y)) for t, y in re.findall(r"\[financials_annual: ([A-Z]+) FY(\d{4})\]", result["answer"])}
# MAGIC
# MAGIC
# MAGIC def show(result):
# MAGIC     print("Q:", result["question"])
# MAGIC     print()
# MAGIC     for n, s in enumerate(result["steps"], 1):
# MAGIC         print(f"   step {n}: {s['tool']}({json.dumps(s['args'])})")
# MAGIC     print()
# MAGIC     print("A:", result["answer"])

# COMMAND ----------

from analyst import NO_ADVICE, NOT_FOUND, build_analyst, chunks_cited, chunks_returned, rows_cited, show, tools_used

ask = build_analyst(TOOL_FUNCTIONS, LLM_MODEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The question chapter 8 could not finish

# COMMAND ----------

compare = ask("What was MongoDB's free cash flow margin in fiscal 2026, and how does it compare with Datadog's latest year?")
show(compare)

# COMMAND ----------

# MAGIC %md
# MAGIC Two steps, one per company, then an answer. The loop did what our hand-written round trip could not:
# MAGIC it gave the model a second turn.
# MAGIC
# MAGIC Look at the citations. Each number points to a row, `[financials_annual: MDB FY2026]`, and the model
# MAGIC compared fiscal 2026 for MongoDB with fiscal 2025 for Datadog. It took "latest year" from the rows, not
# MAGIC from the calendar. The function's description said the fiscal years differ, and the model used it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The question this project started with
# MAGIC
# MAGIC *Revenue grew. Was it price or volume?* The growth rate is in a table. The reason is in a PDF.

# COMMAND ----------

why = ask("How much did Snowflake's revenue grow in fiscal 2026, and was it higher prices or more usage?")
show(why)

# COMMAND ----------

# MAGIC %md
# MAGIC One question, two sources, one answer:
# MAGIC
# MAGIC - **29.2%** came from the table, cited as a row,
# MAGIC - **more usage by existing customers, not higher prices** came from the filing, cited as an excerpt.
# MAGIC
# MAGIC Nobody told the agent to call the table first and the search second. Rule 2 in the prompt says a
# MAGIC question about how much and why needs both, and the loop let it take the two steps in turn.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checking an answer against its sources
# MAGIC
# MAGIC Because the agent keeps its steps, every citation can be traced. The excerpt ids in the answer:

# COMMAND ----------

ids = sorted(chunks_cited(why))
display(spark.sql(f"""
  SELECT chunk_id, item, section, page, content
  FROM filing_chunks
  WHERE chunk_id IN ({", ".join(f"'{i}'" for i in ids) or "''"})
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC And the table rows it cites:

# COMMAND ----------

row_filter = " OR ".join(f"(ticker = '{t}' AND fiscal_year = {y})" for t, y in sorted(rows_cited(why))) or "false"
display(spark.sql(f"""
  SELECT ticker, fiscal_year, round(revenue / 1e6, 1) AS revenue_musd, revenue_growth_pct
  FROM financials_annual WHERE {row_filter}
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Two companies, in their own words

# COMMAND ----------

words = ask("How do Datadog and MongoDB each describe where their revenue growth comes from: new customers or existing ones?")
show(words)

# COMMAND ----------

# MAGIC %md
# MAGIC Here the agent ran the same search twice, once per company, and kept the two sets of excerpts apart.
# MAGIC Datadog states the split outright: about 75% of the increase from existing customers. MongoDB gives no
# MAGIC such split, and the answer does not invent one. It reports what MongoDB does say, an expansion rate of
# MAGIC about 121%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. What the analyst will not do

# COMMAND ----------

unknown = ask("What was Snowflake's share price on the day its latest annual report was filed?")
show(unknown)

# COMMAND ----------

advice = ask("Should I buy Snowflake stock?")
show(advice)

# COMMAND ----------

# MAGIC %md
# MAGIC Two different refusals:
# MAGIC
# MAGIC - **The share price.** The agent searched, read eight excerpts, found no price and said so. It looked
# MAGIC   before refusing.
# MAGIC - **Buy or sell.** No tool call at all. Rule 5 answers this one before any lookup, so it costs one model
# MAGIC   call and cannot be talked round by whatever the search might have returned.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Check it

# COMMAND ----------

fcf = {(r["ticker"], r["fiscal_year"]): r["free_cash_flow_margin_pct"] for r in spark.sql("""
    SELECT ticker, fiscal_year, free_cash_flow_margin_pct FROM financials_annual
    WHERE (ticker = 'MDB' AND fiscal_year = 2026) OR (ticker = 'DDOG' AND fiscal_year = 2025)""").collect()}
snow_growth = spark.sql(
    "SELECT revenue_growth_pct FROM financials_annual WHERE ticker = 'SNOW' AND fiscal_year = 2026").first()[0]

checks = {
    "the comparison looked up both companies": sorted(s["args"]["company_ticker"].upper() for s in compare["steps"]) == ["DDOG", "MDB"],
    "the comparison states both margins exactly as the table has them": all(str(v) in compare["answer"] for v in fcf.values()),
    "the comparison cites the two table rows": rows_cited(compare) == set(fcf),
    "'how much and why' used both tools": set(tools_used(why)) == {"company_financials", "search_filings"},
    "'how much and why' states the growth rate from the table": str(snow_growth) in why["answer"],
    "'how much and why' cites a table row and at least one excerpt": bool(rows_cited(why)) and bool(chunks_cited(why)),
    "every excerpt cited was one a tool returned": all(chunks_cited(r) <= chunks_returned(r) for r in (compare, why, words)),
    "a question the tools cannot answer is refused": NOT_FOUND.rstrip(".") in unknown["answer"],
    "a request for investment advice is declined": NO_ADVICE.rstrip(".") in advice["answer"],
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "the agent failed a check"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC question ─► model ─► company_financials ─► model ─► search_filings ─► model ─► answer with sources
# MAGIC                      (the number)                   (the reason)
# MAGIC ```
# MAGIC
# MAGIC | Question | Chapter 7 chain | Chapter 9 agent |
# MAGIC |---|---|---|
# MAGIC | Why did revenue grow? | Answers | Answers |
# MAGIC | A number | Cannot | Answers, cites the table row |
# MAGIC | A comparison of two companies | Cannot | Answers, one call per company |
# MAGIC | How much and why | Cannot | Answers, cites the row and the page |
# MAGIC | Something neither source contains | Says so | Says so |
# MAGIC
# MAGIC Five questions passed here because we read five answers. That does not scale, and it says nothing
# MAGIC about question six.
# MAGIC
# MAGIC Next: `10_evaluate` builds a set of questions with known answers and scores the analyst on all of them.
