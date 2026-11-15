# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 12: Deploy and monitor
# MAGIC
# MAGIC The analyst lives in a notebook. Only someone who opens the notebook can ask it anything. This chapter
# MAGIC turns it into something an application can call:
# MAGIC
# MAGIC ```
# MAGIC analyst.py ─► an MLflow model ─► a version in Unity Catalog ─► a serving endpoint ─► traces of every request
# MAGIC   (code)        (packaged)          (governed)                   (callable)            (watched)
# MAGIC ```
# MAGIC
# MAGIC | Step | What it adds |
# MAGIC |---|---|
# MAGIC | Log | The code, its settings and its package versions, frozen together as one model |
# MAGIC | Register | A numbered version in Unity Catalog, next to the tables, under the same permissions |
# MAGIC | Deploy | A REST endpoint. **This step costs money while the endpoint is up**, so it is switched off by default |
# MAGIC | Monitor | Every request kept as a trace, the same kind chapter 10 produced |

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain==0.22.0 mlflow==3.18.0 databricks-agents==1.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The analyst behind a standard interface
# MAGIC
# MAGIC `ask(question)` returns a dictionary of our own design. A serving endpoint needs a shape every client
# MAGIC already knows. MLflow's `ResponsesAgent` is that shape: a list of messages in, a list of output items
# MAGIC out. The class below is a thin wrapper. It takes the last user message, calls the same `ask` from
# MAGIC chapter 9 with the prompt that passed chapter 10, and returns the answer. The tools it used and the
# MAGIC sources it cited go in `custom_outputs`.
# MAGIC
# MAGIC The file ends with `mlflow.models.set_model(...)`. That line is what lets MLflow log the **file** as the
# MAGIC model ("models from code"), with nothing pickled.

# COMMAND ----------

# MAGIC %%writefile analyst_agent.py
# MAGIC """The analyst as an MLflow model. Written by notebooks/12_deploy_and_monitor."""
# MAGIC import importlib.util
# MAGIC import os
# MAGIC import sys
# MAGIC import uuid
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC try:
# MAGIC     import analyst                                        # where the model is served, analyst.py is on the path
# MAGIC except ModuleNotFoundError:                               # while MLflow logs this file from a notebook it is not:
# MAGIC     spec = importlib.util.spec_from_file_location(        # load the analyst.py that sits next to this file
# MAGIC         "analyst", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyst.py"))
# MAGIC     analyst = importlib.util.module_from_spec(spec)
# MAGIC     sys.modules["analyst"] = analyst
# MAGIC     spec.loader.exec_module(analyst)
# MAGIC
# MAGIC from analyst import SYSTEM_V2, build_analyst, chunks_cited, rows_cited, tools_used
# MAGIC
# MAGIC # Settings are stored with the model when it is logged. The values here are only used when this file runs on its own.
# MAGIC config = mlflow.models.ModelConfig(development_config={
# MAGIC     "tool_functions": ["workspace.fin_analyst.company_financials", "workspace.fin_analyst.search_filings"],
# MAGIC     "llm_endpoint": "databricks-meta-llama-3-3-70b-instruct",
# MAGIC })
# MAGIC
# MAGIC
# MAGIC def last_user_text(request):
# MAGIC     """The newest user message. Its content is a string, or a list of parts with a text field."""
# MAGIC     for message in reversed([m.model_dump() for m in request.input]):
# MAGIC         if message.get("role") == "user":
# MAGIC             content = message.get("content")
# MAGIC             if isinstance(content, str):
# MAGIC                 return content
# MAGIC             return " ".join(part.get("text", "") for part in content or [])
# MAGIC     raise ValueError("The request holds no user message.")
# MAGIC
# MAGIC
# MAGIC class AnalystAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self._ask = None
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         if self._ask is None:                             # built on the first request, where the model is served
# MAGIC             self._ask = build_analyst(config.get("tool_functions"), config.get("llm_endpoint"), system=SYSTEM_V2)
# MAGIC         result = self._ask(last_user_text(request))
# MAGIC         return ResponsesAgentResponse(
# MAGIC             output=[self.create_text_output_item(text=result["answer"], id=str(uuid.uuid4()))],
# MAGIC             custom_outputs={
# MAGIC                 "tools_used": tools_used(result),
# MAGIC                 "chunks_cited": sorted(chunks_cited(result)),
# MAGIC                 "rows_cited": sorted(f"{t} FY{y}" for t, y in rows_cited(result)),
# MAGIC             },
# MAGIC         )
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()   # every model call and tool call becomes a step in the request's trace
# MAGIC mlflow.models.set_model(AnalystAgent())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Log the model
# MAGIC
# MAGIC One call stores everything needed to run the analyst somewhere else:
# MAGIC
# MAGIC - **the code**: `analyst_agent.py`, and `analyst.py` which it imports,
# MAGIC - **the settings**: which functions and which model endpoint, as `model_config`,
# MAGIC - **the packages**, pinned to the versions this notebook runs,
# MAGIC - **the resources**: every Databricks object the analyst touches. When the model is deployed, the endpoint
# MAGIC   is given access to exactly these and nothing else. Leave one out and the deployed analyst fails on
# MAGIC   its first question with a permission error.

# COMMAND ----------

import warnings

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex

warnings.filterwarnings("ignore")

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/fin_analyst_evaluation")
mlflow.set_registry_uri("databricks-uc")

QUESTION = "How much did Snowflake's revenue grow in fiscal 2026, and was it higher prices or more usage?"
request = {"input": [{"role": "user", "content": QUESTION}]}

with mlflow.start_run(run_name="analyst v2 model"):
    logged = mlflow.pyfunc.log_model(
        name="analyst",
        python_model="analyst_agent.py",
        code_paths=["analyst.py"],
        model_config={"tool_functions": TOOL_FUNCTIONS, "llm_endpoint": LLM_MODEL},
        resources=[DatabricksServingEndpoint(endpoint_name=LLM_MODEL),
                   DatabricksVectorSearchIndex(index_name=VS_INDEX),
                   *[DatabricksFunction(function_name=f) for f in TOOL_FUNCTIONS]],
        pip_requirements=["mlflow==3.18.0", "databricks-langchain==0.22.0"],
        input_example=request,
    )

print("Logged:", logged.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load it back and ask
# MAGIC
# MAGIC Before a model is deployed, load it the way the endpoint will and ask it something. This runs the
# MAGIC stored copy of the code, not the files in this folder, so it catches a missing file or setting now
# MAGIC and not twenty minutes into a deployment.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(logged.model_uri)
response = loaded.predict(request)

answer = response["output"][-1]["content"][0]["text"]
print("Q:", QUESTION)
print()
print("A:", answer)
print()
print("custom_outputs:", response["custom_outputs"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register it in Unity Catalog
# MAGIC
# MAGIC Registering gives the model a three-part name, like a table, and a version number. Each time this cell
# MAGIC runs, a new version is added and the old ones stay. The model is now governed like everything else in
# MAGIC the schema: `EXECUTE` on it decides who may load or deploy it, and its lineage links it to the run
# MAGIC that logged it.

# COMMAND ----------

version = mlflow.register_model(logged.model_uri, UC_MODEL)
print(f"{UC_MODEL} version {version.version}")

# COMMAND ----------

from mlflow import MlflowClient

registry = MlflowClient(registry_uri="databricks-uc")
for v in sorted(registry.search_model_versions(f"name='{UC_MODEL}'"), key=lambda v: -int(v.version))[:5]:
    print(f"{UC_MODEL}  version {v.version}  {v.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy it
# MAGIC
# MAGIC `agents.deploy` creates a Model Serving endpoint for one version of the model. It takes about fifteen
# MAGIC minutes the first time. With `scale_to_zero` the endpoint stops when nobody calls it and the first call
# MAGIC after a pause is slow.
# MAGIC
# MAGIC The endpoint is billed while it is running, so this cell does nothing until you set the `deploy`
# MAGIC widget at the top of the notebook to `yes`.

# COMMAND ----------

dbutils.widgets.dropdown("deploy", "no", ["no", "yes"])
DEPLOY = dbutils.widgets.get("deploy") == "yes"

endpoint_name = None
if DEPLOY:
    from databricks import agents

    deployment = agents.deploy(UC_MODEL, version.version, scale_to_zero=True)
    endpoint_name = deployment.endpoint_name
    print("Endpoint:", endpoint_name)
else:
    print("Not deployed. Set the deploy widget to yes and run this cell again to create the endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC Once the endpoint is ready, any application with a token can ask the analyst. The request has the same
# MAGIC shape as in section 3.

# COMMAND ----------

if DEPLOY:
    import time

    from databricks.sdk import WorkspaceClient
    from mlflow.deployments import get_deploy_client

    workspace_client = WorkspaceClient()
    deadline = time.time() + 40 * 60
    while time.time() < deadline:
        state = workspace_client.serving_endpoints.get(endpoint_name).state
        if state.ready and state.ready.value == "READY" and state.config_update.value == "NOT_UPDATING":
            break
        if state.config_update and state.config_update.value == "UPDATE_FAILED":
            raise RuntimeError("The deployment failed. Open Serving in the workspace and read the endpoint's build log.")
        time.sleep(30)
    else:
        raise TimeoutError("The endpoint was not ready after 40 minutes. It keeps starting: run this cell again.")

    served = get_deploy_client("databricks").predict(endpoint=endpoint_name, inputs=request)
    print("A:", served["output"][-1]["content"][0]["text"])
else:
    print("Skipped: no endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monitor it
# MAGIC
# MAGIC Deploying does not end the work. A deployed analyst meets questions nobody wrote down, and the model
# MAGIC behind it can change. What to watch is already in place: every request, in the notebook or on the
# MAGIC endpoint, is kept as a **trace** in the MLflow experiment, with each model call and tool call inside it.
# MAGIC
# MAGIC Three numbers cover most of monitoring. They are read here from the traces chapter 10 produced, one
# MAGIC per evaluation question. On a deployed endpoint the same query runs over real requests.

# COMMAND ----------

# One trace per whole question. Chapter 10 named them "analyst". Without the filter, every single model
# call made outside a question, retries included, would be counted as a request of its own.
try:
    traces = mlflow.search_traces(filter_string="trace.name = 'analyst'", max_results=500)
except Exception:
    traces = mlflow.search_traces(filter_string="tags.`mlflow.traceName` = 'analyst'", max_results=500)

print(f"{len(traces)} questions traced in the experiment")
if len(traces):
    seconds = traces["execution_duration"].dropna() / 1000
    state = traces["state"].astype(str)
    print(f"failed requests : {(state.str.contains('ERROR')).mean():.0%}")
    print(f"median time     : {seconds.median():.1f} s")
    print(f"slowest 5%      : {seconds.quantile(0.95):.1f} s and up")

# COMMAND ----------

# MAGIC %md
# MAGIC | Watch | Why | When it moves |
# MAGIC |---|---|---|
# MAGIC | Failed requests | The endpoint, a tool or a permission broke | Read the trace: the failing step is marked |
# MAGIC | Time per request | Each extra tool call adds a model call | Look for answers that took many steps |
# MAGIC | Quality | The answers can get worse with no error at all | Run chapter 10's scorers on a sample of real traces |
# MAGIC
# MAGIC The third row is the one that matters most and the one an error rate never shows. The scorers from
# MAGIC chapter 10 that need no expected answer, `cites_a_source` and `citations_exist`, can run on every real
# MAGIC request. A real question that got a wrong answer becomes a new row in `eval_questions`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Check it

# COMMAND ----------

registered = registry.get_model_version(UC_MODEL, version.version)

# Packaging and registration: the same on every run.
checks = {
    "the logged model loads and answers": isinstance(answer, str) and len(answer) > 0,
    "the answer comes back with the tools it used": "tools_used" in response["custom_outputs"],
    "the model version is registered in Unity Catalog": str(registered.version) == str(version.version),
    "the registered version points at the model logged here": logged.model_id in (registered.source or "") or registered.run_id == logged.run_id,
    "the experiment holds traces to monitor": len(traces) > 0,
}
# What the model wrote this time.
model_checks = {
    "the loaded model used both tools for 'how much and why'": set(response["custom_outputs"]["tools_used"]) == {"company_financials", "search_filings"},
    "the loaded model cited a table row and an excerpt": bool(response["custom_outputs"]["rows_cited"]) and bool(response["custom_outputs"]["chunks_cited"]),
}
if DEPLOY:
    checks["the endpoint answers"] = len(served["output"][-1]["content"][0]["text"]) > 0
report_checks(checks, model_checks, "the deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC
# MAGIC Two things in this project are billed while they exist, whether or not anyone uses them:
# MAGIC
# MAGIC - the **serving endpoint** from section 5, if you created it: delete it under **Serving**, or run
# MAGIC   `agents.delete_deployment(UC_MODEL, version.version)`,
# MAGIC - the **Vector Search endpoint** from chapter 6: delete it under **Compute > Vector Search**.
# MAGIC
# MAGIC The tables, volumes, functions and model versions cost only their storage.
# MAGIC
# MAGIC ## The finished project
# MAGIC
# MAGIC ```
# MAGIC workspace.fin_analyst
# MAGIC ├── financials_annual     table      the numbers
# MAGIC ├── filing_chunks_index   index      the words
# MAGIC ├── company_financials    function   tool
# MAGIC ├── search_filings        function   tool
# MAGIC ├── eval_questions        table      how the analyst is measured
# MAGIC └── analyst               model      the analyst, versioned and ready to serve
# MAGIC ```
# MAGIC
# MAGIC | Chapters | What they built |
# MAGIC |---|---|
# MAGIC | 1 to 3 | The numbers: from raw facts to one clean row per company and year |
# MAGIC | 4 to 6 | The words: PDFs parsed, chunked and made searchable by meaning |
# MAGIC | 7 | A chain that answers from the filings, with sources |
# MAGIC | 8 to 9 | Two tools and a loop: an agent that uses the numbers and the words together |
# MAGIC | 10 | An evaluation set and scorers, so a change can be measured |
# MAGIC | 11 | Who may use it, and where every answer came from |
# MAGIC | 12 | A registered model, an endpoint, and what to watch once it is live |
# MAGIC
# MAGIC To add a company, add its ticker to `data/download_sec_data.py`, run the script, and run the chapters
# MAGIC again in order. Then add questions about it to `eval_questions` before trusting the answers.
