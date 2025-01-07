from typing import Optional, Any, Dict, List
import asyncio
import psutil
import wrapt
import functools
import json
import os
import time
from datetime import datetime
import uuid
import contextvars
import traceback

from ..data.data_classes import LLMCall
from ..utils.llm_utils import (
    extract_model_name,
    extract_parameters,
    extract_token_usage,
    extract_input_data,
    calculate_llm_cost,
    sanitize_api_keys,
    sanitize_input,
    extract_llm_output,
)
from ..utils.trace_utils import load_model_costs
from ..utils.unique_decorator import generate_unique_hash_simple   
from ..utils.file_name_tracker import TrackName


class LLMTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.patches = []
        try:
            self.model_costs = load_model_costs()
        except Exception as e:
            self.model_costs = {
                # TODO: Default cost handling needs to be improved
                "default": {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0
                }
            }
        self.current_llm_call_name = contextvars.ContextVar("llm_call_name", default=None)
        self.component_network_calls = {}  
        self.component_user_interaction = {}
        self.current_component_id = None  
        self.total_tokens = 0
        self.total_cost = 0.0
        self.llm_data = {}

    def instrument_llm_calls(self):
        # Handle modules that are already imported
        import sys
        
        if "vertexai" in sys.modules:
            self.patch_vertex_ai_methods(sys.modules["vertexai"])
        if "vertexai.generative_models" in sys.modules:
            self.patch_vertex_ai_methods(sys.modules["vertexai.generative_models"])
            
        if "openai" in sys.modules:
            self.patch_openai_methods(sys.modules["openai"])
        if "litellm" in sys.modules:
            self.patch_litellm_methods(sys.modules["litellm"])
        if "anthropic" in sys.modules:
            self.patch_anthropic_methods(sys.modules["anthropic"])
        if "google.generativeai" in sys.modules:
            self.patch_google_genai_methods(sys.modules["google.generativeai"])
        if "langchain_google_vertexai" in sys.modules:
            self.patch_langchain_google_methods(sys.modules["langchain_google_vertexai"])
        if "langchain_google_genai" in sys.modules:
            self.patch_langchain_google_methods(sys.modules["langchain_google_genai"])

        # Register hooks for future imports
        wrapt.register_post_import_hook(self.patch_vertex_ai_methods, "vertexai")
        wrapt.register_post_import_hook(self.patch_vertex_ai_methods, "vertexai.generative_models")
        wrapt.register_post_import_hook(self.patch_openai_methods, "openai")
        wrapt.register_post_import_hook(self.patch_litellm_methods, "litellm")
        wrapt.register_post_import_hook(self.patch_anthropic_methods, "anthropic")
        wrapt.register_post_import_hook(self.patch_google_genai_methods, "google.generativeai")
        
        # Add hooks for LangChain integrations
        wrapt.register_post_import_hook(self.patch_langchain_google_methods, "langchain_google_vertexai")
        wrapt.register_post_import_hook(self.patch_langchain_google_methods, "langchain_google_genai")

    def patch_openai_methods(self, module):
        try:
            if hasattr(module, "OpenAI"):
                client_class = getattr(module, "OpenAI")
                self.wrap_openai_client_methods(client_class)
            if hasattr(module, "AsyncOpenAI"):
                async_client_class = getattr(module, "AsyncOpenAI")
                self.wrap_openai_client_methods(async_client_class)
        except Exception as e:
            # Log the error but continue execution
            print(f"Warning: Failed to patch OpenAI methods: {str(e)}")

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
        # Patch the GenerativeModel class
        if hasattr(module, "generative_models"):
            gen_models = getattr(module, "generative_models")
            if hasattr(gen_models, "GenerativeModel"):
                model_class = getattr(gen_models, "GenerativeModel")
                self.wrap_vertex_model_methods(model_class)
        
        # Also patch the class directly if available
        if hasattr(module, "GenerativeModel"):
            model_class = getattr(module, "GenerativeModel")
            self.wrap_vertex_model_methods(model_class)

    def wrap_vertex_model_methods(self, model_class):
        # Patch both sync and async methods
        self.wrap_method(model_class, "generate_content")
        if hasattr(model_class, "generate_content_async"):
            self.wrap_method(model_class, "generate_content_async")

    def patch_litellm_methods(self, module):
        self.wrap_method(module, "completion")
        self.wrap_method(module, "acompletion")

    def patch_langchain_google_methods(self, module):
        """Patch LangChain's Google integration methods"""
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

        if hasattr(module, "ChatGoogleGenerativeAI"):
            chat_class = getattr(module, "ChatGoogleGenerativeAI")
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

    def wrap_method(self, obj, method_name):
        """
        Wrap a method with tracing functionality.
        Works for both class methods and instance methods.
        """
        # If obj is a class, we need to patch both the class and any existing instances
        if isinstance(obj, type):
            # Store the original class method
            original_method = getattr(obj, method_name)
            
            @wrapt.decorator
            def wrapper(wrapped, instance, args, kwargs):
                if asyncio.iscoroutinefunction(wrapped):
                    return self.trace_llm_call(wrapped, *args, **kwargs)
                return self.trace_llm_call_sync(wrapped, *args, **kwargs)
            
            # Wrap the class method
            wrapped_method = wrapper(original_method)
            setattr(obj, method_name, wrapped_method)
            self.patches.append((obj, method_name, original_method))
            
        else:
            # For instance methods
            original_method = getattr(obj, method_name)
            
            @wrapt.decorator
            def wrapper(wrapped, instance, args, kwargs):
                if asyncio.iscoroutinefunction(wrapped):
                    return self.trace_llm_call(wrapped, *args, **kwargs)
                return self.trace_llm_call_sync(wrapped, *args, **kwargs)
            
            wrapped_method = wrapper(original_method)
            setattr(obj, method_name, wrapped_method)
            self.patches.append((obj, method_name, original_method))

    def create_llm_component(self, component_id, hash_id, name, llm_type, version, memory_used, start_time, end_time, input_data, output_data, cost={}, usage={}, error=None, parameters={}):
        # Update total metrics
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_cost += cost.get("total_cost", 0)

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
                "model": llm_type,
                "version": version,
                "memory_used": memory_used,
                "cost": cost,
                "tokens": usage,
                **parameters
            },
            "data": {
                "input": input_data['args'] if hasattr(input_data, 'args') else input_data,
                "output": output_data.output_response if output_data else None,
                "memory_used": memory_used
            },
            "network_calls": self.component_network_calls.get(component_id, []),
            "interactions": self.component_user_interaction.get(component_id, [])
        }

        if self.gt: 
            component["data"]["gt"] = self.gt

        return component
    
    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        self.component_network_calls[component_id] = []
        self.current_component_id = component_id

    def end_component(self, component_id):
        """Stop tracking network calls for a component"""
        self.current_component_id = None


    async def trace_llm_call(self, original_func, *args, **kwargs):
        """Trace an LLM API call"""
        if not self.is_active:
            return await original_func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash_simple(original_func) 

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the LLM call
            result = await original_func(*args, **kwargs)

            # Calculate resource usage
            end_time = datetime.now().astimezone()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract token usage and calculate cost
            model_name = extract_model_name(args, kwargs, result)
            token_usage = extract_token_usage(result)
            cost = calculate_llm_cost(token_usage, model_name, self.model_costs)
            parameters = extract_parameters(kwargs)
            input_data = extract_input_data(args, kwargs, result)

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__
            
            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type=model_name,
                version="1.0.0",
                memory_used=memory_used,
                start_time=start_time,
                end_time=end_time,
                input_data=input_data,
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage,
                parameters=parameters
            )
                
            # self.add_component(llm_component)
            self.llm_data = llm_component
            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {}
            }
            
            # End tracking network calls for this component
            self.end_component(component_id)
            
            end_time = datetime.now().astimezone()

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__
            
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type="unknown",
                version="1.0.0",
                memory_used=0,
                start_time=start_time,
                end_time=end_time,
                input_data=extract_input_data(args, kwargs, None),
                output_data=None,
                error=error_component
            )
    
            self.add_component(llm_component)
            raise

    def trace_llm_call_sync(self, original_func, *args, **kwargs):
        """Sync version of trace_llm_call"""
        if not self.is_active:
            if asyncio.iscoroutinefunction(original_func):
                return asyncio.run(original_func(*args, **kwargs))
            return original_func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        component_id = str(uuid.uuid4())
        hash_id = generate_unique_hash_simple(original_func)

        # Start tracking network calls for this component
        self.start_component(component_id)

        # Calculate resource usage
        end_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(original_func):
                result = asyncio.run(original_func(*args, **kwargs))
            else:
                result = original_func(*args, **kwargs)

            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract token usage and calculate cost
            model_name = extract_model_name(args, kwargs, result)
            token_usage = extract_token_usage(result)
            cost = calculate_llm_cost(token_usage, model_name, self.model_costs)
            parameters = extract_parameters(kwargs)
            input_data = extract_input_data(args, kwargs, result)

            # End tracking network calls for this component
            self.end_component(component_id)

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__
            
            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type=model_name,
                version="1.0.0",
                memory_used=memory_used,
                start_time=start_time,
                end_time=end_time,
                input_data=input_data,
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage,
                parameters=parameters
            )
            
            self.add_component(llm_component)
            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {}
            }
            
            # End tracking network calls for this component
            self.end_component(component_id)
            
            end_time = datetime.now().astimezone()

            name = self.current_llm_call_name.get()
            if name is None:
                name = original_func.__name__

            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)
            
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                llm_type="unknown",
                version="1.0.0",
                memory_used=memory_used,
                start_time=start_time,
                end_time=end_time,
                input_data=extract_input_data(args, kwargs, None),
                output_data=None,
                error=error_component
            )
    
            self.add_component(llm_component, is_error=True)
            raise

    def trace_llm(self, name: str = None):
        self.current_llm_call_name.set(name)
        def decorator(func):
            @self.file_tracker.trace_decorator
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                self.gt = kwargs.get('gt', None) if kwargs else None
                if not self.is_active:
                    return await func(*args, **kwargs)
                
                hash_id = generate_unique_hash_simple(func)                
                component_id = str(uuid.uuid4())
                parent_agent_id = self.current_agent_id.get()
                self.start_component(component_id)
                
                error_info = None
                result = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_info = {
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    raise
                finally:

                    llm_component = self.llm_data
                    llm_component['name'] = name

                    if self.gt:
                        llm_component["data"]["gt"] = self.gt

                    if error_info:
                        llm_component["error"] = error_info["error"]
                    
                    if parent_agent_id:
                        children = self.agent_children.get()
                        children.append(llm_component)
                        self.agent_children.set(children)
                    else:
                        self.add_component(llm_component)
                    
                    self.end_component(component_id)

            @self.file_tracker.trace_decorator
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                self.gt = kwargs.get('gt', None) if kwargs else None
                if not self.is_active:
                    return func(*args, **kwargs)
                
                hash_id = generate_unique_hash_simple(func)

                component_id = str(uuid.uuid4())
                parent_agent_id = self.current_agent_id.get()
                self.start_component(component_id)
                
                start_time = datetime.now()
                error_info = None
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_info = {
                        "error": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    raise
                finally:

                    llm_component = self.llm_data

                    if error_info:
                        llm_component["error"] = error_info["error"]
                    
                    if parent_agent_id:
                        children = self.agent_children.get()
                        children.append(llm_component)
                        self.agent_children.set(children)
                    else:
                        self.add_component(llm_component)
                    
                    self.end_component(component_id)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def unpatch_llm_calls(self):
        # Remove all patches
        for obj, method_name, original_method in self.patches:
            try:
                setattr(obj, method_name, original_method)
            except Exception as e:
                print(f"Error unpatching {method_name}: {str(e)}")
        self.patches = []