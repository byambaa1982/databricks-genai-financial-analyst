"""The AI financial analyst: two Unity Catalog functions in a loop. Written by notebooks/09_agent."""
import json
import re
import time
import warnings

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

NOT_FOUND = "The filings and the financials I can search do not say."
NO_ADVICE = "I cannot give investment advice."

SYSTEM = f"""You are a financial analyst's assistant for three companies: Snowflake (SNOW), Datadog (DDOG) and MongoDB (MDB).

Rules:
1. Get every fact from a tool. Never answer from memory.
2. Numbers, ratios and trends come from company_financials. Reasons, strategy, customers and risks come from search_filings. A question that asks how much and why needs both tools.
3. Cite the source after every statement. After a number, cite the table row like this: [financials_annual: SNOW FY2026]. After a statement taken from a filing, cite the excerpt id like this: [SNOW-2026-0123]. Only cite rows and excerpts a tool returned.
4. If the tools do not return the answer, reply exactly: {NOT_FOUND}
5. If asked whether to buy, sell or hold a stock, or for a price forecast, reply exactly: {NO_ADVICE}

Answer in two to five sentences."""

CHUNK_ID = r"[A-Z]+-\d{4}-\d{4}"


def build_analyst(tool_functions, llm_endpoint, system=SYSTEM, max_steps=6):
    """Returns ask(question). tool_functions are full Unity Catalog function names."""
    from databricks_langchain import ChatDatabricks, UCFunctionToolkit

    warnings.filterwarnings("ignore", message="Ignoring the default notebook Spark session")
    tools = {t.name: t for t in UCFunctionToolkit(function_names=tool_functions).tools}
    llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0, max_tokens=800).bind_tools(list(tools.values()))

    def call_model(messages):
        for attempt in range(5):                          # a shared endpoint limits requests per minute
            try:
                return llm.invoke(messages)
            except Exception as error:
                if attempt == 4 or not any(w in str(error) for w in ("429", "REQUEST_LIMIT_EXCEEDED", "rate limit")):
                    raise
                time.sleep(20 * (attempt + 1))

    def ask(question):
        messages = [SystemMessage(system), HumanMessage(question)]
        steps = []
        for _ in range(max_steps):
            reply = call_model(messages)
            messages.append(reply)
            if not reply.tool_calls:                      # text instead of a tool call: the answer
                return {"question": question, "answer": reply.content, "steps": steps}
            for call in reply.tool_calls:
                try:
                    rows = tools[call["name"]].invoke(call["args"])
                except Exception as error:                # a failed call goes back as text, so the model can correct it
                    rows = f"ERROR: {type(error).__name__}: {str(error)[:300]}"
                messages.append(ToolMessage(content=rows, tool_call_id=call["id"]))
                steps.append({"tool": call["name"].split("__")[-1], "args": call["args"], "rows": rows})
        return {"question": question, "answer": f"Stopped after {max_steps} steps without an answer.", "steps": steps}

    return ask


def tools_used(result):
    return [s["tool"] for s in result["steps"]]


def chunks_returned(result):
    """Ids of every excerpt search_filings gave the model."""
    return {i for s in result["steps"] if s["tool"] == "search_filings" for i in re.findall(CHUNK_ID, s["rows"])}


def chunks_cited(result):
    return set(re.findall(rf"\[({CHUNK_ID})\]", result["answer"]))


def rows_cited(result):
    """(ticker, fiscal year) of every table row the answer cites."""
    return {(t, int(y)) for t, y in re.findall(r"\[financials_annual: ([A-Z]+) FY(\d{4})\]", result["answer"])}


def show(result):
    print("Q:", result["question"])
    print()
    for n, s in enumerate(result["steps"], 1):
        print(f"   step {n}: {s['tool']}({json.dumps(s['args'])})")
    print()
    print("A:", result["answer"])
