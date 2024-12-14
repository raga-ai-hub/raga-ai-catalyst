from typing import Dict
import contextvars

from ...ragaai_catalyst import RagaAICatalyst
from ..upload_traces import UploadTraces

from .base import BaseTracer
from .llm_tracer import LLMTracerMixin
from .tool_tracer import ToolTracerMixin
from .agent_tracer import AgentTracerMixin
from .tool import Tool
from .network_tracer import NetworkTracer
from .agentneo import AgentNeo

class AgenticTracing(LLMTracerMixin, ToolTracerMixin, AgentTracerMixin, BaseTracer):
    def __init__(self, user_detail, auto_instrument_llm: bool = True):
        self.project_name = user_detail["project_name"]
        self.project_id = user_detail["project_id"]
        self.dataset_name = user_detail["dataset_name"]
        self.user_detail = user_detail["trace_user_detail"]
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 10
        
        # Initialize AgentNeo
        session = AgentNeo(session_name=self.project_name)
        session.project_name = self.project_name

        super().__init__(session)
        self.auto_instrument_llm = auto_instrument_llm
        self.tools: Dict[str, Tool] = {}
        self.call_depth = contextvars.ContextVar("call_depth", default=0)
        self.network_tracer = NetworkTracer()

    def start(self):
        # Start base tracer
        super().start()
        # Instrument calls from mixins
        if self.auto_instrument_llm:
            self.instrument_llm_calls()

    def stop(self):
        # Unpatch methods from mixins
        self.unpatch_llm_calls()

        # Stop base tracer
        super().stop()

        # Upload traces
        upload_traces = UploadTraces(
            json_file_path="random.json",
            project_name=self.project_name,
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            user_detail=self.user_detail,
            base_url=self.base_url
        )
        upload_traces.upload_traces()

    # If you need an unpatch_methods method
    def unpatch_methods(self):
        # Unpatch methods from all mixins
        self.unpatch_llm_calls()


    