from typing import Optional, Any, Dict, List
import asyncio
import psutil
import json
import wrapt
import functools
from datetime import datetime
import uuid
import os
import contextvars

from .unique_decorator import mydecorator
from .utils.trace_utils import calculate_cost, load_model_costs
from .utils.llm_utils import extract_llm_output

class LLMTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = []
        self.model_costs = load_model_costs()
        self.current_llm_call_name = contextvars.ContextVar("llm_call_name", default=None)
        # Apply decorator to trace_llm_call method
        self.trace_llm_call = mydecorator(self.trace_llm_call)

    def instrument_llm_calls(self):
        # Use wrapt to register post-import hooks
        wrapt.register_post_import_hook(self.patch_openai_methods, "openai")
        wrapt.register_post_import_hook(self.patch_litellm_methods, "litellm")
        wrapt.register_post_import_hook(self.patch_anthropic_methods, "anthropic")
        wrapt.register_post_import_hook(self.patch_google_genai_methods, "google.generativeai")
        wrapt.register_post_import_hook(self.patch_vertex_ai_methods, "vertexai")

    def patch_openai_methods(self, module):
        if hasattr(module, "OpenAI"):
            client_class = getattr(module, "OpenAI")
            self.wrap_openai_client_methods(client_class)
        if hasattr(module, "AsyncOpenAI"):
            async_client_class = getattr(module, "AsyncOpenAI")
            self.wrap_openai_client_methods(async_client_class)

    def patch_anthropic_methods(self, module):
        if hasattr(module, "Anthropic"):
            client_class = getattr(module, "Anthropic")
            self.wrap_anthropic_client_methods(client_class)

    def patch_google_genai_methods(self, module):
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_genai_model_methods(model_class)

    def patch_vertex_ai_methods(self, module):
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_vertex_model_methods(model_class)

    def wrap_openai_client_methods(self, client_class):
        original_init = client_class.__init__

        @functools.wraps(original_init)
        def patched_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            self.wrap_method(client_self.chat.completions, "create")
            if hasattr(client_self.chat.completions, "acreate"):
                self.wrap_method(client_self.chat.completions, "acreate")

        setattr(client_class, "__init__", patched_init)

    def wrap_anthropic_client_methods(self, client_class):
        original_init = client_class.__init__

        @functools.wraps(original_init)
        def patched_init(client_self, *args, **kwargs):
            original_init(client_self, *args, **kwargs)
            self.wrap_method(client_self.messages, "create")
            if hasattr(client_self.messages, "acreate"):
                self.wrap_method(client_self.messages, "acreate")

        setattr(client_class, "__init__", patched_init)

    def wrap_genai_model_methods(self, model_class):
        original_init = model_class.__init__

        @functools.wraps(original_init)
        def patched_init(model_self, *args, **kwargs):
            original_init(model_self, *args, **kwargs)
            self.wrap_method(model_self, "generate_content")
            if hasattr(model_self, "generate_content_async"):
                self.wrap_method(model_self, "generate_content_async")

        setattr(model_class, "__init__", patched_init)

    def wrap_vertex_model_methods(self, model_class):
        original_init = model_class.__init__

        @functools.wraps(original_init)
        def patched_init(model_self, *args, **kwargs):
            original_init(model_self, *args, **kwargs)
            self.wrap_method(model_self, "generate_content")
            if hasattr(model_self, "generate_content_async"):
                self.wrap_method(model_self, "generate_content_async")

        setattr(model_class, "__init__", patched_init)

    def patch_litellm_methods(self, module):
        self.wrap_method(module, "completion")
        self.wrap_method(module, "acompletion")

    def wrap_method(self, obj, method_name):
        original_method = getattr(obj, method_name)

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            return self.trace_llm_call(wrapped, *args, **kwargs)

        wrapped_method = wrapper(original_method)
        setattr(obj, method_name, wrapped_method)
        self.patches.append((obj, method_name, original_method))

    def _extract_model_name(self, kwargs):
        """Extract model name from kwargs or result"""
        model = kwargs.get("model", "")
        if not model:
            # Try to extract from messages
            if "messages" in kwargs:
                messages = kwargs["messages"]
                if messages and isinstance(messages, list) and messages[0].get("role") == "system":
                    model = messages[0].get("model", "")
            # Try to extract from GenerativeModel instance
            elif hasattr(kwargs.get("self", None), "model_name"):
                model = kwargs["self"].model_name
        return model or "unknown"

    def _extract_parameters(self, kwargs, result=None):
        """Extract parameters from kwargs or result"""
        params = {
            "temperature": kwargs.get("temperature", getattr(result, "temperature", 0.7)),
            "top_p": kwargs.get("top_p", getattr(result, "top_p", 1.0)),
            "max_tokens": kwargs.get("max_tokens", getattr(result, "max_tokens", 512))
        }
        
        # Add Google AI specific parameters if available
        if hasattr(kwargs.get("self", None), "generation_config"):
            gen_config = kwargs["self"].generation_config
            params.update({
                "candidate_count": getattr(gen_config, "candidate_count", 1),
                "stop_sequences": getattr(gen_config, "stop_sequences", []),
                "top_k": getattr(gen_config, "top_k", 40)
            })
        
        return params

    def _extract_token_usage(self, result):
        """Extract token usage from result"""
        # Handle standard OpenAI/Anthropic format
        if hasattr(result, "usage"):
            usage = result.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0)
            }
        
        # Handle Google GenerativeAI format
        if hasattr(result, "parts"):
            total_tokens = sum(len(part.text.split()) for part in result.parts)
            return {
                "prompt_tokens": 0,  # Google AI doesn't provide this breakdown
                "completion_tokens": total_tokens,
                "total_tokens": total_tokens
            }
            
        # Handle Vertex AI format
        if hasattr(result, "text"):
            total_tokens = len(result.text.split())
            return {
                "prompt_tokens": 0,  # Vertex AI doesn't provide this breakdown
                "completion_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def _calculate_cost(self, token_usage, model_name):
        """Calculate cost based on token usage and model"""
        model_cost = self.model_costs.get(model_name, self.model_costs.get("default", {
            "input_cost_per_token": 0.00002,
            "output_cost_per_token": 0.00002
        }))

        return {
            "prompt_cost": round(token_usage["prompt_tokens"] * model_cost["input_cost_per_token"], 5),
            "completion_cost": round(token_usage["completion_tokens"] * model_cost["output_cost_per_token"], 5),
            "total_cost": round(
                (token_usage["prompt_tokens"] * model_cost["input_cost_per_token"] +
                 token_usage["completion_tokens"] * model_cost["output_cost_per_token"]), 5
            )
        }

    def create_llm_component(self, **kwargs):
        """Create an LLM component according to the data structure"""
        start_time = kwargs["start_time"]
        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "source_hash_id": None,
            "type": "llm",
            "name": kwargs["name"],
            "start_time": start_time.isoformat(),
            "end_time": kwargs["end_time"].isoformat(),
            "error": kwargs.get("error"),
            "parent_id": self.current_agent_id.get(),
            "info": {
                "llm_type": kwargs["llm_type"],
                "version": kwargs["version"],
                "memory_used": kwargs["memory_used"],
                "cost": kwargs.get("cost", 0),
                "tokens": kwargs.get("tokens", 0)
            },
            "data": {
                "input": kwargs["input_data"],
                "output": kwargs["output_data"],
                "memory_used": kwargs["memory_used"]
            },
            "network_calls": self.component_network_calls.get(kwargs["component_id"], []),
            "interactions": [{
                "id": f"int_{uuid.uuid4()}",
                "interaction_type": "input",
                "timestamp": start_time.isoformat(),
                "content": kwargs["input_data"]
            }, {
                "id": f"int_{uuid.uuid4()}",
                "interaction_type": "output",
                "timestamp": kwargs["end_time"].isoformat(),
                "content": kwargs["output_data"]
            }]
        }

        return component
    
    def trace_llm_call(self, original_func, *args, **kwargs):
        """Trace an LLM API call"""
        if not self.is_active:
            return original_func(*args, **kwargs)
        
        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self.trace_llm_call.hash_id  # Get hash_id from decorator

        try:
            # Execute the LLM call
            result = original_func(*args, **kwargs)
            
            # Calculate memory usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract model and parameters
            model_name = self._extract_model_name(kwargs)
            parameters = self._extract_parameters(kwargs, result)
            
            # Extract token usage and calculate cost
            token_usage = self._extract_token_usage(result)
            cost = self._calculate_cost(token_usage, model_name)

            # Create component
            component = {
                "id": component_id,
                "hash_id": hash_id,
                "source_hash_id": None,
                "type": "llm",
                "name": "llm_call",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "error": None,
                "parent_id": None,
                "info": {
                    "model": model_name,
                    "parameters": parameters,
                    "token_usage": token_usage,
                    "cost": cost
                },
                "data": {
                    "input": {
                        "args": list(args),
                        "kwargs": kwargs
                    },
                    "output": extract_llm_output(result),
                    "memory_used": memory_used
                },
                "network_calls": [],
                "interactions": []
            }

            self.add_component(component)
            return result

        except Exception as e:
            end_time = datetime.now()
            
            # Create error component
            error_component = {
                "code": getattr(e, 'code', 500),
                "type": type(e).__name__,
                "message": str(e),
                "details": getattr(e, '__dict__', {})
            }

            # Create error trace component
            component = {
                "id": component_id,
                "hash_id": hash_id,
                "source_hash_id": None,
                "type": "llm",
                "name": "llm_call",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": error_component,
                "parent_id": None,
                "info": {
                    "model": self._extract_model_name(kwargs),
                    "parameters": self._extract_parameters(kwargs),
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    },
                    "cost": {
                        "prompt_cost": 0.0,
                        "completion_cost": 0.0,
                        "total_cost": 0.0
                    }
                },
                "data": {
                    "input": {
                        "args": list(args),
                        "kwargs": kwargs
                    },
                    "output": None,
                    "memory_used": 0
                },
                "network_calls": [],
                "interactions": []
            }
            
            self.add_component(component)
            raise

    def unpatch_llm_calls(self):
        """Remove all patches"""
        for obj, method_name, original_method in self.patches:
            if hasattr(obj, method_name):
                setattr(obj, method_name, original_method)
        self.patches.clear()

    def _sanitize_api_keys(self, data):
        """Remove sensitive information from data"""
        if isinstance(data, dict):
            return {k: self._sanitize_api_keys(v) for k, v in data.items() 
                    if not any(sensitive in k.lower() for sensitive in ['key', 'token', 'secret', 'password'])}
        elif isinstance(data, list):
            return [self._sanitize_api_keys(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_api_keys(item) for item in data)
        return data
