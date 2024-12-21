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
        self.component_network_calls = {}  
        self.current_component_id = None  
        self.total_tokens = 0
        self.total_cost = 0.0
        # Apply decorator to trace_llm_call method
        self.trace_llm_call = mydecorator(self.trace_llm_call)

    def instrument_llm_calls(self):
        # Use wrapt to register post-import hooks
        wrapt.register_post_import_hook(self.patch_openai_methods, "openai")
        wrapt.register_post_import_hook(self.patch_litellm_methods, "litellm")
        wrapt.register_post_import_hook(self.patch_anthropic_methods, "anthropic")
        wrapt.register_post_import_hook(self.patch_google_genai_methods, "google.generativeai")
        wrapt.register_post_import_hook(self.patch_vertex_ai_methods, "vertexai")
        
        # Add hooks for LangChain integrations
        wrapt.register_post_import_hook(self.patch_langchain_google_methods, "langchain_google_vertexai")
        wrapt.register_post_import_hook(self.patch_langchain_google_methods, "langchain_google_genai")

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
        # Patch direct Google GenerativeAI usage
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_genai_model_methods(model_class)
        
        # Patch LangChain integration
        if hasattr(module, "ChatGoogleGenerativeAI"):
            chat_class = getattr(module, "ChatGoogleGenerativeAI")
            # Wrap invoke method to capture messages
            original_invoke = chat_class.invoke
            
            def patched_invoke(self, messages, *args, **kwargs):
                # Store messages in the instance for later use
                self._last_messages = messages
                return original_invoke(self, messages, *args, **kwargs)
            
            chat_class.invoke = patched_invoke
            
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "_generate")
            if hasattr(chat_class, "_agenerate"):
                self.wrap_method(chat_class, "_agenerate")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

    def patch_vertex_ai_methods(self, module):
        # Patch direct VertexAI usage
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_vertex_model_methods(model_class)
            
        # Patch LangChain integration
        if hasattr(module, "ChatVertexAI"):
            chat_class = getattr(module, "ChatVertexAI")
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "_generate")
            if hasattr(chat_class, "_agenerate"):
                self.wrap_method(chat_class, "_agenerate")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

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
            self.wrap_method(model_self, "predict")
            self.wrap_method(model_self, "predict_streaming")
            if hasattr(model_self, "predict_async"):
                self.wrap_method(model_self, "predict_async")

        setattr(model_class, "__init__", patched_init)

    def patch_litellm_methods(self, module):
        self.wrap_method(module, "completion")
        self.wrap_method(module, "acompletion")

    def patch_langchain_google_methods(self, module):
        """Patch LangChain's Google integration methods"""
        if hasattr(module, "ChatVertexAI"):
            chat_class = getattr(module, "ChatVertexAI")
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "invoke")
            if hasattr(chat_class, "ainvoke"):
                self.wrap_method(chat_class, "ainvoke")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

        if hasattr(module, "ChatGoogleGenerativeAI"):
            chat_class = getattr(module, "ChatGoogleGenerativeAI")
            # LangChain v0.2+ uses invoke/ainvoke
            self.wrap_method(chat_class, "invoke")
            if hasattr(chat_class, "ainvoke"):
                self.wrap_method(chat_class, "ainvoke")
            # Fallback for completion methods
            if hasattr(chat_class, "complete"):
                self.wrap_method(chat_class, "complete")
            if hasattr(chat_class, "acomplete"):
                self.wrap_method(chat_class, "acomplete")

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
        # First try direct model parameter
        model = kwargs.get("model", "")
        
        if not model:
            # Try to get from instance
            instance = kwargs.get("self", None)
            if instance:
                # Try model_name first (Google format)
                if hasattr(instance, "model_name"):
                    model = instance.model_name
                # Try model attribute
                elif hasattr(instance, "model"):
                    model = instance.model
        
        # Normalize Google model names
        if model and isinstance(model, str):
            model = model.lower()
            if "gemini-1.5-flash" in model:
                return "gemini-1.5-flash"
            if "gemini-1.5-pro" in model:
                return "gemini-1.5-pro"
            if "gemini-pro" in model:
                return "gemini-pro"
        
        return model or "default"

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
        
        # Handle Google GenerativeAI format with usage_metadata
        if hasattr(result, "usage_metadata"):
            metadata = result.usage_metadata
            return {
                "prompt_tokens": metadata.get("input_tokens", 0),
                "completion_tokens": metadata.get("output_tokens", 0),
                "total_tokens": metadata.get("total_tokens", 0)
            }
        
        # Handle Vertex AI format
        if hasattr(result, "text"):
            # For LangChain ChatVertexAI
            total_tokens = getattr(result, "token_count", 0)
            if not total_tokens and hasattr(result, "_raw_response"):
                # Try to get from raw response
                total_tokens = getattr(result._raw_response, "token_count", 0)
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
        end_time = kwargs["end_time"]
        
        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "source_hash_id": None,
            "type": "llm",
            "name": kwargs["name"],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
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
                "output": kwargs["output_data"].output_response if kwargs["output_data"] else None,
                "memory_used": kwargs["memory_used"]
            },
            "network_calls": self.component_network_calls.get(kwargs["component_id"], []),
            "interactions": [
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "input",
                    "timestamp": start_time.isoformat(),
                    "content": kwargs["input_data"]
                },
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "output",
                    "timestamp": end_time.isoformat(),
                    "content": kwargs["output_data"].output_response if kwargs["output_data"] else None
                }
            ]
        }

        return component
    
    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        self.component_network_calls[component_id] = []
        self.current_component_id = component_id

    def end_component(self, component_id):
        """Stop tracking network calls for a component"""
        self.current_component_id = None

    # def track_network_call(self, request, response, error=None):
    #     """Track a network call for the current component"""
    #     if self.current_component_id is None:
    #         return

    #     start_time = getattr(request, '_start_time', datetime.now())
    #     end_time = datetime.now()
    #     response_time = (end_time - start_time).total_seconds()

    #     # Calculate bytes sent/received
    #     request_body = getattr(request, 'body', None)
    #     bytes_sent = len(str(request_body).encode('utf-8')) if request_body else 0
    #     response_body = getattr(response, 'text', None)
    #     bytes_received = len(str(response_body).encode('utf-8')) if response_body else 0

    #     # Extract URL components
    #     url = str(request.url) if hasattr(request, 'url') else None
    #     protocol = url.split('://')[0] if url else None

    #     network_call = {
    #         "url": url,
    #         "method": request.method.lower() if hasattr(request, 'method') else None,
    #         "status_code": response.status_code if hasattr(response, 'status_code') else None,
    #         "response_time": response_time,
    #         "bytes_sent": bytes_sent,
    #         "bytes_received": bytes_received,
    #         "protocol": protocol,
    #         "connection_id": str(uuid.uuid4()),
    #         "parent_id": None,
    #         "request": {
    #             "headers": dict(request.headers) if hasattr(request, 'headers') else {},
    #             "body": request_body
    #         },
    #         "response": {
    #             "headers": dict(response.headers) if hasattr(response, 'headers') else {},
    #             "body": response_body
    #         } if response else None,
    #         "error": str(error) if error else None
    #     }

    #     self.component_network_calls[self.current_component_id].append(network_call)

    def trace_llm_call(self, original_func, *args, **kwargs):
        """Trace an LLM API call"""
        if not self.is_active:
            return original_func(*args, **kwargs)
        
        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self.trace_llm_call.hash_id

        self.start_component(component_id)

        try:
            # Execute the LLM call
            result = original_func(*args, **kwargs)
            
            # Calculate memory usage
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract model name before processing result
            model_name = self._extract_model_name(kwargs)
            
            # For Google API, get the actual result object
            if hasattr(result, "result"):
                processed_result = result.result
            else:
                processed_result = result
            
            # Extract parameters and usage
            parameters = self._extract_parameters(kwargs, processed_result)
            token_usage = self._extract_token_usage(result)  # Pass original result for token extraction
            cost = self._calculate_cost(token_usage, model_name)

            # Get input/output data
            input_data = self._extract_input_data(kwargs, result)
            output_data = extract_llm_output(result)

            end_time = datetime.now().astimezone()
            self.end_component(component_id)

            # Create component
            component = {
                "id": component_id,
                "hash_id": hash_id,
                "source_hash_id": None,
                "type": "llm",
                "name": self.current_llm_call_name.get(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": None,
                "parent_id": self.current_agent_id.get(),
                "info": {
                    "llm_type": "llm",
                    "version": "1.0.0",
                    "memory_used": memory_used,
                    "cost": cost,
                    "tokens": token_usage,
                    "model": model_name,  # Add model name to info
                    "parameters": parameters  # Add parameters to info
                },
                "data": {
                    "input": input_data,
                    "output": output_data.output_response if output_data else None,
                    "memory_used": memory_used
                },
                "network_calls": self.component_network_calls.get(component_id, []),
                "interactions": [
                    {
                        "id": f"int_{uuid.uuid4()}",
                        "interaction_type": "input",
                        "timestamp": start_time.isoformat(),
                        "content": input_data
                    },
                    {
                        "id": f"int_{uuid.uuid4()}",
                        "interaction_type": "output",
                        "timestamp": end_time.isoformat(),
                        "content": output_data.output_response if output_data else None
                    }
                ]
            }

            self.add_component(component)
            return result

        except Exception as e:
            self.end_component(component_id)
            raise e

    def _extract_input_data(self, kwargs, result):
        """Extract input data from kwargs and result"""
        # Try to get messages from instance
        instance = kwargs.get("self")
        if instance and hasattr(instance, "_last_messages"):
            messages = instance._last_messages
            if isinstance(messages, list):
                return [{"role": msg[0], "content": msg[1]} if isinstance(msg, tuple) else msg 
                       for msg in messages]
            return messages
        
        # Try standard messages format (OpenAI/Anthropic)
        messages = kwargs.get("messages", [])
        if messages:
            return messages

        # Try Gemini format
        if hasattr(result, "prompt"):
            return [{"role": "user", "content": str(result.prompt)}]
        
        # Try content/prompt in kwargs
        content = kwargs.get("content", kwargs.get("prompt", ""))
        if content:
            return [{"role": "user", "content": str(content)}]
        
        return []

    def trace_llm(self, name: str, tool_type: str = "llm", version: str = "1.0.0"):
        def decorator(func_or_class):
            if isinstance(func_or_class, type):
                for attr_name, attr_value in func_or_class.__dict__.items():
                    if callable(attr_value) and not attr_name.startswith("__"):
                        setattr(
                            func_or_class,
                            attr_name,
                            self.trace_llm(f"{name}.{attr_name}", tool_type, version)(attr_value),
                        )
                return func_or_class
            else:
                @functools.wraps(func_or_class)
                async def async_wrapper(*args, **kwargs):
                    token = self.current_llm_call_name.set(name)
                    try:
                        return await func_or_class(*args, **kwargs)
                    finally:
                        self.current_llm_call_name.reset(token)

                @functools.wraps(func_or_class)
                def sync_wrapper(*args, **kwargs):
                    token = self.current_llm_call_name.set(name)
                    try:
                        return func_or_class(*args, **kwargs)
                    finally:
                        self.current_llm_call_name.reset(token)

                return async_wrapper if asyncio.iscoroutinefunction(func_or_class) else sync_wrapper

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

    def _create_llm_component(self, component_id, hash_id, name, llm_type, version, memory_used, start_time, end_time, input_data, output_data, usage=None, error=None):
        cost = None
        tokens = None
        
        if usage:
            tokens = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            }
            cost = calculate_cost(usage)
            
            # Update total metrics
            self.total_tokens += tokens["total_tokens"]
            self.total_cost += cost["total"]

        component = {
            "id": component_id,
            "hash_id": hash_id,
            "source_hash_id": None,
            "type": "llm",
            "name": name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "error": error,
            "parent_id": self.current_agent_id.get(),
            "info": {
                "llm_type": llm_type,
                "version": version,
                "memory_used": memory_used,
                "cost": cost,
                "tokens": tokens
            },
            "data": {
                "input": input_data,
                "output": output_data.output_response if output_data else None,
                "memory_used": memory_used
            },
            "network_calls": self.component_network_calls.get(component_id, []),
            "interactions": [
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "input",
                    "timestamp": start_time.isoformat(),
                    "content": input_data
                },
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "output",
                    "timestamp": end_time.isoformat(),
                    "content": output_data.output_response if output_data else None
                }
            ]
        }

        return component

def extract_llm_output(result):
    """Extract output from LLM response"""
    class OutputResponse:
        def __init__(self, output_response):
            self.output_response = output_response

    # Handle Google GenerativeAI format
    if hasattr(result, "result"):
        candidates = getattr(result.result, "candidates", [])
        output = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    if hasattr(part, "text"):
                        output.append({
                            "content": part.text,
                            "role": getattr(content, "role", "assistant"),
                            "finish_reason": getattr(candidate, "finish_reason", None)
                        })
        return OutputResponse(output)
    
    # Handle Vertex AI format
    if hasattr(result, "text"):
        return OutputResponse([{
            "content": result.text,
            "role": "assistant"
        }])
    
    # Handle OpenAI format
    if hasattr(result, "choices"):
        return OutputResponse([{
            "content": choice.message.content,
            "role": choice.message.role
        } for choice in result.choices])
    
    # Handle Anthropic format
    if hasattr(result, "completion"):
        return OutputResponse([{
            "content": result.completion,
            "role": "assistant"
        }])
    
    # Default case
    return OutputResponse(str(result))
