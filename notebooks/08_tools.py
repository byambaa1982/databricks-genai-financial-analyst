# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 8: Tools the model can call
# MAGIC
# MAGIC Chapter 7 ended on a question the chain could not answer: a free cash flow margin. The number was two
# MAGIC rows away in `financials_annual`, and the chain had no way to reach it. It can only search text.
# MAGIC
# MAGIC A **tool** fixes that. A tool is a function with a name, a description and typed parameters. The model
# MAGIC never writes SQL and never touches a table. It reads the descriptions, picks a function and fills in
# MAGIC the arguments. Our code runs the function and hands back the rows.
# MAGIC
# MAGIC ```
# MAGIC question ─► model picks a function and its arguments ─► we run it ─► rows ─► model writes the answer
# MAGIC ```
# MAGIC
# MAGIC This chapter writes two functions in Unity Catalog, one for the numbers and one for the words, and
# MAGIC shows the model choosing between them.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. A function on the numbers
# MAGIC
# MAGIC A SQL function that returns a table. It takes a ticker and returns that company's last five fiscal
# MAGIC years from `financials_annual`, with amounts in millions so the rows are easy to read.
# MAGIC
# MAGIC Every `COMMENT` below is read by the model: the one on the function says when to use it, the one on
# MAGIC the parameter says what to pass, the ones on the columns say what came back. They are the tool's
# MAGIC instructions, so they are written for a reader who has never seen the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION company_financials(
# MAGIC   company_ticker STRING COMMENT 'Stock ticker of the company: SNOW (Snowflake), DDOG (Datadog) or MDB (MongoDB).'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   ticker                    STRING COMMENT 'Stock ticker.',
# MAGIC   company                   STRING COMMENT 'Company name.',
# MAGIC   fiscal_year               INT    COMMENT 'Calendar year in which the fiscal year ended.',
# MAGIC   fiscal_year_end           DATE   COMMENT 'Last day of the fiscal year.',
# MAGIC   revenue_musd              DOUBLE COMMENT 'Total revenue, millions of US dollars.',
# MAGIC   revenue_growth_pct        DOUBLE COMMENT 'Revenue growth versus the prior fiscal year, percent.',
# MAGIC   gross_margin_pct          DOUBLE COMMENT 'Gross profit divided by revenue, percent.',
# MAGIC   operating_margin_pct      DOUBLE COMMENT 'Operating income (GAAP) divided by revenue, percent. Negative means an operating loss.',
# MAGIC   net_income_musd           DOUBLE COMMENT 'Net income, millions of US dollars. Negative means a net loss.',
# MAGIC   free_cash_flow_musd       DOUBLE COMMENT 'Operating cash flow minus capital expenditure, millions of US dollars.',
# MAGIC   free_cash_flow_margin_pct DOUBLE COMMENT 'Free cash flow divided by revenue, percent.',
# MAGIC   stock_comp_pct_of_revenue DOUBLE COMMENT 'Stock-based compensation divided by revenue, percent.',
# MAGIC   cash_musd                 DOUBLE COMMENT 'Cash and cash equivalents at fiscal year end, millions of US dollars.'
# MAGIC )
# MAGIC COMMENT 'Reported annual financials for one company, as filed with the SEC: revenue, growth, margins, net income, free cash flow, stock-based compensation and cash. Returns one row per fiscal year for the last five fiscal years, newest first. Use it for any question about a number, a ratio or a trend. Fiscal years end in January for Snowflake and MongoDB and in December for Datadog, so the latest fiscal year is not the same for every company. Call it once per company to compare companies.'
# MAGIC RETURN
# MAGIC   SELECT ticker, company, fiscal_year, period_end,
# MAGIC          round(revenue / 1e6, 1), revenue_growth_pct, gross_margin_pct, operating_margin_pct,
# MAGIC          round(net_income / 1e6, 1), round(free_cash_flow / 1e6, 1), free_cash_flow_margin_pct,
# MAGIC          stock_comp_pct_of_revenue, round(cash / 1e6, 1)
# MAGIC   FROM financials_annual
# MAGIC   WHERE ticker = upper(trim(company_ticker))
# MAGIC   ORDER BY fiscal_year DESC
# MAGIC   LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC It is an ordinary object in the schema, next to the tables. Anyone with permission can call it from SQL:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM company_financials('MDB');

# COMMAND ----------

# MAGIC %md
# MAGIC The first row is the answer chapter 7 refused to give: MongoDB's free cash flow margin in fiscal 2026.
# MAGIC
# MAGIC Why a function and not "let the model write SQL"? The function can only do one thing. It cannot read
# MAGIC another table, it cannot get the definition of a margin wrong, and it gives the same rows for the
# MAGIC same ticker every time.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. A function on the words
# MAGIC
# MAGIC The search from chapter 6, in the SQL form shown there, wrapped in a function so it becomes a tool too.
# MAGIC
# MAGIC One detail: in SQL, a `WHERE` on `vector_search()` runs **after** the search. Asking for the top 8 and
# MAGIC then keeping one company can leave only two or three rows. So the function asks for the top 50, keeps
# MAGIC the company if one was given, and returns the best 8 of what is left.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION search_filings(
  question STRING COMMENT 'What to look for, as a full question or sentence in plain English.',
  company_ticker STRING DEFAULT NULL COMMENT 'Optional. Search only this company: SNOW, DDOG or MDB. Leave empty to search all three.'
)
RETURNS TABLE (
  chunk_id    STRING COMMENT 'Id of the excerpt. Cite it in square brackets after each statement taken from it.',
  company     STRING COMMENT 'Company that filed the report.',
  fiscal_year INT    COMMENT 'Fiscal year the annual report covers.',
  item        STRING COMMENT 'Part of the 10-K, for example Item 1A (Risk Factors) or Item 7 (Management Discussion).',
  section     STRING COMMENT 'Heading the excerpt sits under.',
  page        INT    COMMENT 'Page of the PDF the excerpt starts on.',
  content     STRING COMMENT 'The text of the excerpt.'
)
COMMENT 'Searches the text of the latest annual reports (10-K filings) of Snowflake, Datadog and MongoDB by meaning and returns the 8 most relevant excerpts. Use it for questions about reasons, strategy, customers, products, competition and risks: what management wrote, not a number. For revenue, margins, cash flow and other reported numbers use company_financials instead.'
RETURN
  SELECT chunk_id, company, CAST(fiscal_year AS INT), item, section, CAST(page AS INT), content
  FROM vector_search(index => '{VS_INDEX}', query_text => question, num_results => 50)
  WHERE company_ticker IS NULL OR ticker = upper(trim(company_ticker))
  ORDER BY search_score DESC
  LIMIT 8
""")

display(spark.sql("""
  SELECT chunk_id, item, section, page, left(content, 160) AS content_start
  FROM search_filings('Why did revenue grow? Was it higher prices or more usage?', 'SNOW')
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Functions become tools
# MAGIC
# MAGIC `UCFunctionToolkit` reads each function's definition from Unity Catalog and builds a tool from it.
# MAGIC Nothing is described twice: the comments written in SQL are what the model receives.

# COMMAND ----------

import json
import warnings

warnings.filterwarnings("ignore", message="Ignoring the default notebook Spark session")   # a notice from the client that runs the functions

from databricks_langchain import ChatDatabricks, UCFunctionToolkit

toolkit = UCFunctionToolkit(function_names=TOOL_FUNCTIONS)
tools = toolkit.tools
tools_by_name = {t.name: t for t in tools}

for t in tools:
    print(t.name)
    print("  ", t.description[:150] + "...")
    for name, spec in t.args.items():
        print(f"   parameter {name}: {spec.get('description', '')}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC That is everything the model knows about the project: two names, two descriptions, three parameters.
# MAGIC It does not see the tables or the index.
# MAGIC
# MAGIC ## 4. The model picks a tool
# MAGIC
# MAGIC `bind_tools` sends the tool list along with every question. The model can now reply in one of two ways:
# MAGIC with text, or with a **tool call**, a request to run a named function with given arguments.
# MAGIC
# MAGIC Three questions. Nothing is run yet. We only look at what the model asks for.

# COMMAND ----------

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

SYSTEM = ("You are a financial analyst's assistant for three companies: Snowflake (SNOW), Datadog (DDOG) and "
          "MongoDB (MDB). Look facts up with the tools. Do not answer from memory. Do not give investment advice.")

llm = ChatDatabricks(endpoint=LLM_MODEL, temperature=0, max_tokens=600)
llm_with_tools = llm.bind_tools(tools)


def short(name):
    return name.split("__")[-1]


def tool_calls_for(question):
    reply = llm_with_tools.invoke([SystemMessage(SYSTEM), HumanMessage(question)])
    print("Q:", question)
    for call in reply.tool_calls:
        print(f"   -> {short(call['name'])}({json.dumps(call['args'])})")
    if not reply.tool_calls:
        print("   -> no tool call:", reply.content[:200])
    print()
    return reply


pick_numbers = tool_calls_for("What was Datadog's operating margin in its latest fiscal year?")
pick_words = tool_calls_for("What does MongoDB say are the main risks from competition?")
pick_both = tool_calls_for("How fast did Snowflake's revenue grow last year, and what does the company say drove it?")

# COMMAND ----------

# MAGIC %md
# MAGIC The model read the descriptions and sorted the questions itself: the number went to `company_financials`,
# MAGIC the risks went to `search_filings`, and it turned "Datadog" and "MongoDB" into the tickers the functions
# MAGIC expect. Nobody wrote a rule for that. The `COMMENT` text is the rule.
# MAGIC
# MAGIC The third question needs both tools, and the model asked for one: the number first. It works one step
# MAGIC at a time. Keep that in mind for section 6.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. One full round trip
# MAGIC
# MAGIC Half of the question from the end of chapter 7. Four steps, written out by hand so each one is visible:
# MAGIC
# MAGIC 1. the model replies with a tool call,
# MAGIC 2. we run it,
# MAGIC 3. the rows go back to the model as a tool message,
# MAGIC 4. the model writes the answer from the rows.

# COMMAND ----------

def one_round_trip(question):
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    reply = llm_with_tools.invoke(messages)                      # 1. the model asks for a tool
    messages.append(reply)
    print("Q:", question)
    print()
    for call in reply.tool_calls:
        rows = tools_by_name[call["name"]].invoke(call["args"])  # 2. we run the function
        messages.append(ToolMessage(content=rows, tool_call_id=call["id"]))   # 3. the rows go back
        print(f"called {short(call['name'])}({json.dumps(call['args'])})")
        print("   returned", rows[:230], "...")
    final = llm_with_tools.invoke(messages)                      # 4. the model reads the rows
    print()
    if final.tool_calls:
        print("No answer yet. The model asked for another tool:")
        for call in final.tool_calls:
            print(f"   -> {short(call['name'])}({json.dumps(call['args'])})")
    else:
        print("A:", final.content)
    return final


one_company = one_round_trip("What was MongoDB's free cash flow margin in fiscal 2026?")

# COMMAND ----------

# MAGIC %md
# MAGIC 20.3%, the same value as the first row in section 1. The model did not calculate it and did not
# MAGIC remember it. It read it from rows that came out of a governed table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Where one round trip stops
# MAGIC
# MAGIC Now the whole question from chapter 7, with the comparison.

# COMMAND ----------

two_companies = one_round_trip(
    "What was MongoDB's free cash flow margin in fiscal 2026, and how does it compare with Datadog's latest year?")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, fiscal_year, free_cash_flow_margin_pct
# MAGIC FROM financials_annual
# MAGIC WHERE (ticker = 'MDB' AND fiscal_year = 2026) OR (ticker = 'DDOG' AND fiscal_year = 2025);

# COMMAND ----------

# MAGIC %md
# MAGIC The model got MongoDB's rows and, instead of answering, asked for Datadog's. That is the right next
# MAGIC step. Our code is what stopped: it runs the tools once and expects an answer.
# MAGIC
# MAGIC Both numbers are in the table above, one call away. What is missing is a **loop**: run the tools, hand
# MAGIC back the rows, and repeat until the model replies with text instead of another tool call. That loop,
# MAGIC with rules for citing sources, is an agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Check it

# COMMAND ----------

def called(reply):
    return [short(c["name"]) for c in reply.tool_calls]


mdb_margin = spark.sql(
    "SELECT free_cash_flow_margin_pct FROM financials_annual WHERE ticker = 'MDB' AND fiscal_year = 2026").first()[0]

checks = {
    "company_financials returns five years for each company": spark.sql("""
        SELECT count(*) FROM (SELECT * FROM company_financials('SNOW') UNION ALL SELECT * FROM company_financials('ddog')
                              UNION ALL SELECT * FROM company_financials('MDB'))""").first()[0] == 15,
    "search_filings with a ticker returns 8 excerpts from that company only": [r["company"] for r in spark.sql(
        "SELECT company FROM search_filings('What are the main risks?', 'DDOG')").collect()] == ["Datadog, Inc."] * 8,
    "a numbers question goes to company_financials": called(pick_numbers) == ["company_financials"],
    "a question about what management wrote goes to search_filings": called(pick_words) == ["search_filings"],
    "one round trip answers with the margin exactly as the table has it": str(mdb_margin) in one_company.content,
}
for name, passed in checks.items():
    print("PASS" if passed else "FAIL", "-", name)
assert all(checks.values()), "the tools failed a check"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── financials_annual     table      the numbers
# MAGIC ├── filing_chunks_index   index      the words
# MAGIC ├── company_financials    function   tool: the numbers of one company, last five years
# MAGIC └── search_filings        function   tool: the 8 most relevant excerpts for a question
# MAGIC ```
# MAGIC
# MAGIC | Question | Tool the model picks |
# MAGIC |---|---|
# MAGIC | A number, a ratio, a trend | `company_financials` |
# MAGIC | A reason, a risk, a strategy | `search_filings` |
# MAGIC
# MAGIC One round trip answers a question that needs one call. A comparison needs two, and "how much and why"
# MAGIC needs both tools, one after the other.
# MAGIC
# MAGIC Next: `09_agent` puts the two tools in a loop that runs until the answer is complete, and cites the
# MAGIC chunk ids and the table.
