"""The analyst as an MLflow model. Written by notebooks/12_deploy_and_monitor."""
import importlib.util
import os
import sys
import uuid

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

try:
    import analyst                                        # where the model is served, analyst.py is on the path
except ModuleNotFoundError:                               # while MLflow logs this file from a notebook it is not:
    spec = importlib.util.spec_from_file_location(        # load the analyst.py that sits next to this file
        "analyst", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyst.py"))
    analyst = importlib.util.module_from_spec(spec)
    sys.modules["analyst"] = analyst
    spec.loader.exec_module(analyst)

from analyst import SYSTEM_V2, build_analyst, chunks_cited, rows_cited, tools_used

# Settings are stored with the model when it is logged. The values here are only used when this file runs on its own.
config = mlflow.models.ModelConfig(development_config={
    "tool_functions": ["workspace.fin_analyst.company_financials", "workspace.fin_analyst.search_filings"],
    "llm_endpoint": "databricks-meta-llama-3-3-70b-instruct",
})


def last_user_text(request):
    """The newest user message. Its content is a string, or a list of parts with a text field."""
    for message in reversed([m.model_dump() for m in request.input]):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return " ".join(part.get("text", "") for part in content or [])
    raise ValueError("The request holds no user message.")


class AnalystAgent(ResponsesAgent):
    def __init__(self):
        self._ask = None

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        if self._ask is None:                             # built on the first request, where the model is served
            self._ask = build_analyst(config.get("tool_functions"), config.get("llm_endpoint"), system=SYSTEM_V2)
        result = self._ask(last_user_text(request))
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=result["answer"], id=str(uuid.uuid4()))],
            custom_outputs={
                "tools_used": tools_used(result),
                "chunks_cited": sorted(chunks_cited(result)),
                "rows_cited": sorted(f"{t} FY{y}" for t, y in rows_cited(result)),
            },
        )


mlflow.langchain.autolog()   # every model call and tool call becomes a step in the request's trace
mlflow.models.set_model(AnalystAgent())
