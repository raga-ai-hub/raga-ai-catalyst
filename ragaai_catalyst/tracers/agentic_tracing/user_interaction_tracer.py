import builtins
from datetime import datetime
import contextvars
import inspect

class UserInteractionModel:
    def __init__(self, project_id, trace_id, agent_id, interaction_type, content, timestamp):
        self.project_id = project_id
        self.trace_id = trace_id
        self.agent_id = agent_id
        self.interaction_type = interaction_type
        self.content = content
        self.timestamp = timestamp


class UserInteractionTracer:
    def __init__(self, *args, **kwargs):
        self.project_id = contextvars.ContextVar("project_id", default=None)
        self.trace_id = contextvars.ContextVar("trace_id", default=None)
        self.tracer = contextvars.ContextVar("tracer", default=None)
        self.original_input = builtins.input
        self.original_print = builtins.print

    def traced_input(self, prompt=""):
        # Get caller information
        caller_frame = inspect.currentframe().f_back
        caller_info = {
            'function': caller_frame.f_code.co_name,
        }
        
        if prompt:
            self.traced_print(prompt, end="")
        try:
            content = self.original_input()
        except EOFError:
            content = ""  # Return empty string on EOF
        if hasattr(self.tracer, "trace") and self.tracer.trace is not None:
            self.tracer.trace.add_interaction("user_input", {
                'content': content,
                'caller': caller_info
            })
            
        return content

    def traced_print(self, *args, **kwargs):
        # Get caller information
        caller_frame = inspect.currentframe().f_back
        caller_info = {
            'function': caller_frame.f_code.co_name,
        }
        
        content = " ".join(str(arg) for arg in args)
        if hasattr(self.tracer, "trace") and self.tracer.trace is not None:
            self.tracer.trace.add_interaction("print", {
                'content': content,
                'caller': caller_info
            })
        return self.original_print(*args, **kwargs)

    def _log_interaction(self, interaction_type, content):
        agent_id = self.tracer.current_agent_id.get()
        
        # Extract content and caller info if it's a dict
        if isinstance(content, dict) and 'content' in content and 'caller' in content:
            interaction_content = content['content']
            caller_info = content['caller']
        else:
            interaction_content = content
            caller_info = None
            
        user_interaction = UserInteractionModel(
            project_id=self.project_id,
            trace_id=self.trace_id,
            agent_id=agent_id,
            interaction_type=interaction_type,
            content={
                'content': interaction_content,
                'caller': caller_info
            },
            timestamp=datetime.now(),
        )

        # Also add to trace data
        self.tracer.trace.add_interaction("interactions", []).append(
            user_interaction
        )
