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
import sys
import gc

from .unique_decorator import mydecorator
from .utils.trace_utils import calculate_cost, load_model_costs
from .utils.llm_utils import extract_llm_output


class LLMTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = []
        try:
            self.model_costs = load_model_costs()
        except Exception as e:
            # If model costs can't be loaded, use default costs
            self.model_costs = {
                "default": {
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00002
                }
            }
        self.current_llm_call_name = contextvars.ContextVar("llm_call_name", default=None)
        self.component_network_calls = {}  
        self.current_component_id = None  
        self.total_tokens = 0
        self.total_cost = 0.0
        # Track seen hash IDs to prevent duplicate traces
        self.seen_hash_ids = set()
        # Apply decorator to trace_llm_call method
        self.trace_llm_call = mydecorator(self.trace_llm_call)

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
        # Handle coroutines
        if asyncio.iscoroutine(result):
            # Get the current event loop
            loop = asyncio.get_event_loop()
            # Run the coroutine in the current event loop
            result = loop.run_until_complete(result)

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
                "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0)
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

    def _extract_input_data(self, kwargs, result):
        """Extract input data from kwargs and result"""

        # For Vertex AI GenerationResponse
        if hasattr(result, 'candidates') and hasattr(result, 'usage_metadata'):
            # Extract generation config
            generation_config = kwargs.get('generation_config', {})
            config_dict = {}
            if hasattr(generation_config, 'temperature'):
                config_dict['temperature'] = generation_config.temperature
            if hasattr(generation_config, 'top_p'):
                config_dict['top_p'] = generation_config.top_p
            if hasattr(generation_config, 'max_output_tokens'):
                config_dict['max_tokens'] = generation_config.max_output_tokens
            if hasattr(generation_config, 'candidate_count'):
                config_dict['n'] = generation_config.candidate_count

            return {
                "prompt": kwargs.get('contents', ''),
                "model": "gemini-1.5-flash-002",  
                **config_dict
            }

        # For standard OpenAI format
        messages = kwargs.get("messages", [])
        if messages:
            return {
                "messages": messages,
                "model": kwargs.get("model", "unknown"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", None),
                "top_p": kwargs.get("top_p", None),
                "frequency_penalty": kwargs.get("frequency_penalty", None),
                "presence_penalty": kwargs.get("presence_penalty", None)
            }

        # For text completion format
        if "prompt" in kwargs:
            return {
                "prompt": kwargs["prompt"],
                "model": kwargs.get("model", "unknown"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", None),
                "top_p": kwargs.get("top_p", None),
                "frequency_penalty": kwargs.get("frequency_penalty", None),
                "presence_penalty": kwargs.get("presence_penalty", None)
            }

        # For any other case, try to extract from kwargs
        if "contents" in kwargs:
            return {
                "prompt": kwargs["contents"],
                "model": kwargs.get("model", "unknown"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", None),
                "top_p": kwargs.get("top_p", None)
            }

        print("No input data found")
        return {}

    def _calculate_cost(self, token_usage, model_name):
        """Calculate cost based on token usage and model"""
        if not isinstance(token_usage, dict):
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": token_usage if isinstance(token_usage, (int, float)) else 0
            }

        # Get model costs, defaulting to Vertex AI PaLM2 costs if unknown
        model_cost = self.model_costs.get(model_name, {
            "input_cost_per_token": 0.0005,  # $0.0005 per 1K input tokens
            "output_cost_per_token": 0.0005  # $0.0005 per 1K output tokens
        })

        # Calculate costs per 1K tokens
        input_cost = (token_usage.get("prompt_tokens", 0) / 1000.0) * model_cost.get("input_cost_per_token", 0.0005)
        output_cost = (token_usage.get("completion_tokens", 0) / 1000.0) * model_cost.get("output_cost_per_token", 0.0005)
        total_cost = input_cost + output_cost

        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }

    def create_llm_component(self, **kwargs):
        """Create an LLM component according to the data structure"""
        start_time = kwargs["start_time"]
        
        # Ensure cost and usage are dictionaries
        cost = kwargs.get("cost", {})
        if not isinstance(cost, dict):
            cost = {"total_cost": cost}
            
        usage = kwargs.get("usage", {})
        if not isinstance(usage, dict):
            usage = {"total_tokens": usage}
            
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
                "llm_type": kwargs.get("llm_type", "unknown"),
                "version": kwargs.get("version", "1.0.0"),
                "memory_used": kwargs.get("memory_used", 0),
                "cost": cost,
                "tokens": usage
            },
            "data": {
                "input": kwargs.get("input_data"),
                "output": kwargs.get("output_data"),
                "memory_used": kwargs.get("memory_used", 0)
            },
            "network_calls": self.component_network_calls.get(kwargs["component_id"], []),
            "interactions": [
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "input",
                    "timestamp": start_time.isoformat(),
                    "content": kwargs.get("input_data")
                },
                {
                    "id": f"int_{uuid.uuid4()}",
                    "interaction_type": "output",
                    "timestamp": kwargs["end_time"].isoformat(),
                    "content": kwargs.get("output_data")
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


    async def trace_llm_call(self, original_func, *args, **kwargs):
        """Trace an LLM API call"""
        if not self.is_active:
            return await original_func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self.trace_llm_call.hash_id

        # Skip if we've already seen this hash_id
        if hash_id in self.seen_hash_ids:
            return await original_func(*args, **kwargs)

        # Add hash_id to seen set
        self.seen_hash_ids.add(hash_id)

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
            token_usage = await self._extract_token_usage(result)
            model_name = self._extract_model_name(kwargs)
            cost = self._calculate_cost(token_usage, model_name)

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=self.current_llm_call_name.get(),
                llm_type=model_name,
                version="1.0.0",
                memory_used=memory_used,
                start_time=start_time,
                end_time=end_time,
                input_data=self._extract_input_data(kwargs, result),
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage
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
            
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=self.current_llm_call_name.get(),
                llm_type="unknown",
                version="1.0.0",
                memory_used=0,
                start_time=start_time,
                end_time=end_time,
                input_data=self._extract_input_data(kwargs, None),
                output_data=None,
                error=error_component
            )

            self.add_component(llm_component)
            raise

    def _extract_token_usage_sync(self, result):
        """Sync version of extract token usage"""
        # Handle coroutines
        if asyncio.iscoroutine(result):
            # Get the current event loop
            loop = asyncio.get_event_loop()
            # Run the coroutine in the current event loop
            result = loop.run_until_complete(result)

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
                "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0)
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

    def trace_llm_call_sync(self, original_func, *args, **kwargs):
        """Sync version of trace_llm_call"""
        if not self.is_active:
            if asyncio.iscoroutinefunction(original_func):
                return asyncio.run(original_func(*args, **kwargs))
            return original_func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self.trace_llm_call.hash_id

        # Skip if we've already seen this hash_id
        if hash_id in self.seen_hash_ids:
            if asyncio.iscoroutinefunction(original_func):
                return asyncio.run(original_func(*args, **kwargs))
            return original_func(*args, **kwargs)

        # Add hash_id to seen set
        self.seen_hash_ids.add(hash_id)

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the LLM call
            result = None
            if asyncio.iscoroutinefunction(original_func):
                result = asyncio.run(original_func(*args, **kwargs))
            else:
                result = original_func(*args, **kwargs)

            # If result is a coroutine, run it
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

            # Calculate resource usage
            end_time = datetime.now().astimezone()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Extract token usage and calculate cost
            token_usage = self._extract_token_usage_sync(result)
            model_name = self._extract_model_name(kwargs)
            cost = self._calculate_cost(token_usage, model_name)

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create LLM component
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=self.current_llm_call_name.get(),
                llm_type=model_name,
                version="1.0.0",
                memory_used=memory_used,
                start_time=start_time,
                end_time=end_time,
                input_data=self._extract_input_data(kwargs, result),
                output_data=extract_llm_output(result),
                cost=cost,
                usage=token_usage
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
            
            llm_component = self.create_llm_component(
                component_id=component_id,
                hash_id=hash_id,
                name=self.current_llm_call_name.get(),
                llm_type="unknown",
                version="1.0.0",
                memory_used=0,
                start_time=start_time,
                end_time=end_time,
                input_data=self._extract_input_data(kwargs, None),
                output_data=None,
                error=error_component
            )

            self.add_component(llm_component)
            raise

    async def _extract_token_usage(self, result):
        """Extract token usage from result"""
        # Handle coroutines
        if asyncio.iscoroutine(result):
            # Get the current event loop
            loop = asyncio.get_event_loop()
            # Run the coroutine in the current event loop
            result = loop.run_until_complete(result)

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
                "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0)
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
                if asyncio.iscoroutinefunction(func_or_class):
                    @functools.wraps(func_or_class)
                    async def async_wrapper(*args, **kwargs):
                        token = self.current_llm_call_name.set(name)
                        try:
                            return await self.trace_llm_call(func_or_class, *args, **kwargs)
                        finally:
                            self.current_llm_call_name.reset(token)
                    return async_wrapper
                else:
                    @functools.wraps(func_or_class)
                    def sync_wrapper(*args, **kwargs):
                        token = self.current_llm_call_name.set(name)
                        try:
                            return self.trace_llm_call_sync(func_or_class, *args, **kwargs)
                        finally:
                            self.current_llm_call_name.reset(token)
                    return sync_wrapper
        return decorator

    def unpatch_llm_calls(self):
        """Remove all patches"""
        # Clear seen hash IDs
        self.seen_hash_ids.clear()
        
        # Remove all patches
        for obj, method_name, original_method in self.patches:
            try:
                setattr(obj, method_name, original_method)
            except Exception as e:
                print(f"Error unpatching {method_name}: {str(e)}")
        self.patches = []

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

    # Handle coroutines
    if asyncio.iscoroutine(result):
        # For sync context, run the coroutine
        if not asyncio.get_event_loop().is_running():
            result = asyncio.run(result)
        else:
            # We're in an async context, but this function is called synchronously
            # Return a placeholder and let the caller handle the coroutine
            return OutputResponse("Coroutine result pending")

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
