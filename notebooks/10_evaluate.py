# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 10: Evaluate the answers
# MAGIC
# MAGIC Chapter 9 ended with five good answers. We know they were good because we read them. That works for
# MAGIC five questions on one afternoon. It does not work the next time the prompt, the chunks or the model
# MAGIC changes, because every change needs all the answers read again.
# MAGIC
# MAGIC Evaluation replaces the reading with three things:
# MAGIC
# MAGIC | Piece | What it is | Here |
# MAGIC |---|---|---|
# MAGIC | Evaluation set | Questions with the answer we expect, written down once | the `eval_questions` table |
# MAGIC | Scorers | Functions that mark one answer pass or fail | four in code, one judged by a model |
# MAGIC | A run | Every question asked, every answer scored, the results kept | an MLflow run with one trace per question |

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The evaluation set
# MAGIC
# MAGIC Fourteen questions of four kinds. For each one we write down what a correct answer must contain and
# MAGIC which tools it should take to get there.
# MAGIC
# MAGIC The expected numbers were read from `financials_annual`, the expected statements from the filings, with
# MAGIC the page open. The answers are written by a person **before** the analyst is asked. An evaluation set
# MAGIC written from the model's own answers only measures whether the model agrees with itself.

# COMMAND ----------

from analyst import NO_ADVICE, NOT_FOUND

N, W = "company_financials", "search_filings"

eval_rows = [
    # (id, kind, question, expected facts, expected tools)
    ("n1", "number", "What was Snowflake's revenue in fiscal 2026?",
     ["Snowflake's revenue in fiscal 2026 was about $4,684 million ($4.68 billion)"], [N]),
    ("n2", "number", "What was Datadog's operating margin in its latest fiscal year?",
     ["Datadog's operating margin in fiscal 2025 was -1.3%, an operating loss"], [N]),
    ("n3", "number", "What was MongoDB's free cash flow margin in fiscal 2026, and how does it compare with Datadog's latest year?",
     ["MongoDB's free cash flow margin in fiscal 2026 was 20.3%", "Datadog's free cash flow margin in fiscal 2025 was 29.2%"], [N]),
    ("n4", "number", "Which of the three companies had the highest gross margin in its latest fiscal year?",
     ["Datadog had the highest gross margin, 80.0%", "MongoDB's was 71.7% and Snowflake's was 67.2%"], [N]),
    ("n5", "number", "How has Snowflake's stock-based compensation as a share of revenue changed over the last three fiscal years?",
     ["Stock-based compensation was 41.6% of revenue in fiscal 2024, 40.8% in fiscal 2025 and 34.1% in fiscal 2026", "It fell"], [N]),
    ("w1", "words", "Why did Snowflake's revenue grow in fiscal 2026?",
     ["Revenue grew mainly because existing customers consumed more of the platform"], [W]),
    ("w2", "words", "Did Datadog's revenue growth in 2025 come from new customers or existing ones?",
     ["About 75% of the increase came from existing customers and about 25% from new customers"], [W]),
    ("w3", "words", "What is MongoDB's net ARR expansion rate?",
     ["MongoDB's net ARR expansion rate was approximately 121% as of January 31, 2026"], [W]),
    ("w4", "words", "How many customers with ARR of $100,000 or more does Datadog have?",
     ["Datadog had approximately 4,310 customers with ARR of $100,000 or more as of December 31, 2025"], [W]),
    ("b1", "both", "How much did Snowflake's revenue grow in fiscal 2026, and was it higher prices or more usage?",
     ["Snowflake's revenue grew 29.2% in fiscal 2026", "The growth came from more usage (consumption) by existing customers, not higher prices"], [N, W]),
    ("b2", "both", "How fast did Datadog's revenue grow in fiscal 2025, and where did the growth come from?",
     ["Datadog's revenue grew 27.7% (about 28%) in fiscal 2025", "About 75% of the increase came from existing customers"], [N, W]),
    ("r1", "refuse", "What was Snowflake's share price on the day its latest annual report was filed?", [NOT_FOUND], []),
    ("r2", "refuse", "Should I buy Snowflake stock?", [NO_ADVICE], []),
    ("r3", "refuse", "What was Microsoft's revenue in fiscal 2025?", [NOT_FOUND], []),
]

(spark.createDataFrame(eval_rows, "question_id STRING, kind STRING, question STRING, expected_facts ARRAY<STRING>, expected_tools ARRAY<STRING>")
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("eval_questions"))
spark.sql("COMMENT ON TABLE eval_questions IS 'Evaluation set for the AI financial analyst: questions, the facts a correct answer must contain, and the tools it should use. Expected facts were written by a person from financials_annual and the 10-K filings.'")

display(spark.sql("SELECT kind, count(*) AS questions FROM eval_questions GROUP BY kind ORDER BY kind"))

# COMMAND ----------

# MAGIC %md
# MAGIC The set lives in a table for the same reason the numbers do: one copy, with a history. When a question
# MAGIC is added next month, the table's history shows which results were measured on which version.
# MAGIC
# MAGIC ## 2. The scorers
# MAGIC
# MAGIC Four checks need no judgement, so they are plain code. They are exact, free and instant.
# MAGIC
# MAGIC | Scorer | Passes when |
# MAGIC |---|---|
# MAGIC | `right_tools` | every tool the question needs was called (and none, for a request for advice) |
# MAGIC | `cites_a_source` | an answer carries at least one citation; a refusal carries none |
# MAGIC | `citations_exist` | every excerpt or table row cited is one a tool returned |
# MAGIC | `refuses_when_it_should` | a "refuse" question gets the exact refusal sentence, and other questions do not |
# MAGIC
# MAGIC The fifth question, *is the answer right?*, cannot be a string comparison: "$4.68 billion" and
# MAGIC "$4,684 million" are the same answer. For that we use a **judge**, a model that reads the answer and the
# MAGIC expected facts and replies yes or no with a reason.

# COMMAND ----------

import os
import re
import warnings

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"   # one question at a time: a shared endpoint limits requests per minute

import mlflow
from mlflow.genai.scorers import Correctness, scorer

from analyst import build_analyst, chunks_cited, chunks_returned, rows_cited, tools_used

warnings.filterwarnings("ignore")

REFUSALS = (NOT_FOUND.rstrip("."), NO_ADVICE.rstrip("."))


def make_predict_fn(ask):
    """Returns what MLflow calls for each row: ask the analyst, return the answer and the evidence for the scorers."""

    @mlflow.trace(name="analyst")
    def predict_fn(question):
        result = ask(question)
        rows_returned = sorted({f"{s['args'].get('company_ticker', '').upper()} FY{year}"
                                for s in result["steps"] if s["tool"] == "company_financials"
                                for year in re.findall(r",(\d{4}),\d{4}-\d{2}-\d{2},", s["rows"])})
        return {
            "answer": result["answer"],
            "tools_used": tools_used(result),
            "chunks_returned": sorted(chunks_returned(result)),
            "chunks_cited": sorted(chunks_cited(result)),
            "rows_returned": rows_returned,
            "rows_cited": sorted(f"{t} FY{y}" for t, y in rows_cited(result)),
        }

    return predict_fn


def refused(outputs):
    return any(r in outputs["answer"] for r in REFUSALS)


@scorer
def right_tools(outputs, expectations):
    if expectations["kind"] == "refuse":
        return True          # looking before refusing is allowed; refuses_when_it_should marks these
    return set(expectations["expected_tools"]) <= set(outputs["tools_used"])


@scorer
def cites_a_source(outputs):
    has_citation = bool(outputs["chunks_cited"] or outputs["rows_cited"])
    return not has_citation if refused(outputs) else has_citation


@scorer
def citations_exist(outputs):
    return set(outputs["chunks_cited"]) <= set(outputs["chunks_returned"]) and set(outputs["rows_cited"]) <= set(outputs["rows_returned"])


@scorer
def refuses_when_it_should(outputs, expectations):
    return refused(outputs) == (expectations["kind"] == "refuse")


# The judge. Free Edition has no built-in judge model, so we name one: the same endpoint the analyst uses.
correct = Correctness(model=f"databricks:/{JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC A note on the judge. Here the judge is the same model that wrote the answers, because it is the one
# MAGIC every workspace has. That is acceptable for this task, comparing a short answer with a written fact,
# MAGIC but a judge is a model and can be wrong. Section 4 reads its reasons instead of trusting the score.
# MAGIC With a stronger model available, change `JUDGE_MODEL` in `00_config`.
# MAGIC
# MAGIC ## 3. Run it
# MAGIC
# MAGIC `mlflow.genai.evaluate` asks every question, records a trace of each (every model call and tool call
# MAGIC inside the loop), and runs every scorer on every answer.

# COMMAND ----------

import pandas as pd

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fin_analyst_evaluation")

SCORERS = ["right_tools", "cites_a_source", "citations_exist", "refuses_when_it_should", "correctness"]


def passed(value):
    return value in (True, "yes")


def evaluate(run_name, ask):
    """Ask every question in eval_questions, score every answer, keep the results in an MLflow run."""
    eval_data = [
        {"inputs": {"question": r["question"]},
         "expectations": {"expected_facts": list(r["expected_facts"]), "expected_tools": list(r["expected_tools"]),
                          "kind": r["kind"], "question_id": r["question_id"]}}
        for r in spark.table("eval_questions").orderBy("question_id").collect()
    ]
    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=make_predict_fn(ask),
            scorers=[right_tools, cites_a_source, citations_exist, refuses_when_it_should, correct],
        )
    return results


def scores(results):
    return {name: results.metrics[name + "/mean"] for name in SCORERS}


def per_question(results):
    df = results.result_df
    return pd.DataFrame({
        "id": df["question_id/value"],
        "kind": df["kind/value"],
        "question": df["request"].apply(lambda r: r["question"][:70]),
        **{name: df[f"{name}/value"].apply(lambda v: "pass" if passed(v) else "FAIL") for name in SCORERS},
    }).sort_values("id").reset_index(drop=True)


def show_failures(results):
    df = results.result_df.sort_values("question_id/value")
    failed = df[~df[[f"{n}/value" for n in SCORERS]].apply(lambda col: col.map(passed)).all(axis=1)]
    for _, row in failed.iterrows():
        print(row["question_id/value"], "-", row["request"]["question"])
        print("   failed:", ", ".join(n for n in SCORERS if not passed(row[f"{n}/value"])))
        print("   tools :", row["response"]["tools_used"])
        print("   answer:", row["response"]["answer"])
        if not passed(row["correctness/value"]):
            print("   judge :", row["correctness/rationale"])
        print()
    if failed.empty:
        print("No question failed a scorer.")
    return list(failed["question_id/value"])


v1 = evaluate("analyst v1", build_analyst(TOOL_FUNCTIONS, LLM_MODEL))
for name, value in scores(v1).items():
    print(f"{name:<24} {value:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read the failures, not the averages
# MAGIC
# MAGIC An average says how many. It does not say which, or why. One row per question:

# COMMAND ----------

display(spark.createDataFrame(per_question(v1)))

# COMMAND ----------

v1_failed = show_failures(v1)

# COMMAND ----------

# MAGIC %md
# MAGIC Two questions failed, for two different reasons.
# MAGIC
# MAGIC **`n1`: the analyst is right and the judge is wrong.** The answer says $4683.9 million. The expected
# MAGIC fact says "about $4,684 million". They are the same number, and the judge marked it wrong over the
# MAGIC rounding. Nothing is wrong with the analyst here. The expected fact was written loosely, and that
# MAGIC is fixed in the evaluation set, not in the agent.
# MAGIC
# MAGIC **`w3`: the analyst is wrong.** MongoDB's net ARR expansion rate is on page 109 of its report: about
# MAGIC 121%. The analyst looked for it in the financials table, did not find it, and refused without ever
# MAGIC searching the filings. Three scorers caught the same mistake from three sides: wrong tool, a refusal
# MAGIC that should not have happened, wrong answer.
# MAGIC
# MAGIC A failing score is a place to look, not a verdict. Had we tuned the prompt until `n1` passed, we would
# MAGIC have been fixing something that was not broken.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Fix one thing at a time
# MAGIC
# MAGIC **The evaluation set.** The expected fact for `n1` now states the number the way the table has it. The
# MAGIC change is an `UPDATE`, so the table's history records what the expectation was for each run.

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE eval_questions
# MAGIC SET expected_facts = array("Snowflake's revenue in fiscal 2026 was $4,683.9 million")
# MAGIC WHERE question_id = 'n1';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT version, operation, operationMetrics.numUpdatedRows AS rows_updated
# MAGIC FROM (DESCRIBE HISTORY eval_questions)
# MAGIC ORDER BY version DESC
# MAGIC LIMIT 2;

# COMMAND ----------

# MAGIC %md
# MAGIC **The analyst.** One more rule in the prompt, telling the model where the business metrics live.
# MAGIC `build_analyst` takes the prompt as an argument, so version 2 is the same code with a longer prompt.

# COMMAND ----------

from analyst import SYSTEM

SYSTEM_V2 = SYSTEM + ("\n\nThe financials table holds only the metrics listed in the description of company_financials. "
                      "Customer counts, ARR, retention rates and expansion rates are not in it: "
                      "they are in the filings, so use search_filings for them.")

v2 = evaluate("analyst v2", build_analyst(TOOL_FUNCTIONS, LLM_MODEL, system=SYSTEM_V2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare the two runs

# COMMAND ----------

comparison = pd.DataFrame({"scorer": SCORERS,
                           "v1": [f"{scores(v1)[n]:.0%}" for n in SCORERS],
                           "v2": [f"{scores(v2)[n]:.0%}" for n in SCORERS]})
display(spark.createDataFrame(comparison))

# COMMAND ----------

v2_failed = show_failures(v2)

# COMMAND ----------

# MAGIC %md
# MAGIC Both failures are gone, and none of the other twelve questions got worse. That second half is why the
# MAGIC whole set is run again and not only the question that failed: a new rule in a prompt can fix one answer
# MAGIC and break another, and only the full run shows it.
# MAGIC
# MAGIC 100% here means fourteen questions out of fourteen. It does not mean the analyst is always right. The
# MAGIC set is small and we wrote it. It grows the honest way: every wrong answer found in real use becomes a
# MAGIC new row in `eval_questions`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Check it
# MAGIC
# MAGIC These are the bars the analyst has to clear. Change the prompt, the chunk size or the model, run this
# MAGIC notebook again, and the same bars say whether the change made things better or worse.

# COMMAND ----------

m1, m2 = scores(v1), scores(v2)
checks = {
    "all 14 questions were asked and scored in both runs": len(v1.result_df) == 14 and len(v2.result_df) == 14,
    "no answer cites a source the tools did not return": m2["citations_exist"] == 1.0,
    "every refusal is a question it should refuse, and the reverse": m2["refuses_when_it_should"] == 1.0,
    "the right tools were used on at least 90% of questions": m2["right_tools"] >= 0.9,
    "at least 90% of answers are correct according to the judge": m2["correctness"] >= 0.9,
    "version 2 is no worse than version 1 on any scorer": all(m2[n] >= m1[n] for n in SCORERS),
}
for name, ok in checks.items():
    print("PASS" if ok else "FAIL", "-", name)
assert all(checks.values()), "the analyst is below the bar"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What we have now
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── eval_questions        table      14 questions with expected facts and tools
# MAGIC ├── company_financials    function   tool
# MAGIC └── search_filings        function   tool
# MAGIC
# MAGIC MLflow experiment fin_analyst_evaluation
# MAGIC ├── run "analyst v1"      5 scores per question, one trace per question
# MAGIC └── run "analyst v2"      the same questions, after the fix
# MAGIC ```
# MAGIC
# MAGIC Open **Experiments > fin_analyst_evaluation** in the workspace to see each trace: the question, every
# MAGIC model call and tool call with its rows, the answer, and the five scores next to it.
# MAGIC
# MAGIC The new rule has earned its place. To keep it, move the sentence into `SYSTEM` in `09_agent`. It is left
# MAGIC out there on purpose, so that this chapter has something to find.
# MAGIC
# MAGIC The analyst now answers, cites and is measured. One question is left: who is allowed to use it, and
# MAGIC what can it reach?
# MAGIC
# MAGIC Next: `11_permissions_and_lineage` shows what the analyst can touch and where every answer came from.
