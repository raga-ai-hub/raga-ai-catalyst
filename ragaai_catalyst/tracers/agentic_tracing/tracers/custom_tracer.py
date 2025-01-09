import sys
import uuid
import psutil
import threading
from datetime import datetime
import functools
from typing import Optional, Any, Dict, List
from ..utils.unique_decorator import generate_unique_hash_simple, mydecorator
import contextvars
import asyncio
from ..utils.file_name_tracker import TrackName


class CustomTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.current_custom_name = contextvars.ContextVar("custom_name", default=None)
        self.current_custom_id = contextvars.ContextVar("custom_id", default=None)
        self.component_network_calls = {}
        self.component_user_interaction = {}
        self.gt = None

    @mydecorator
    def trace_custom(self, name: str = None, custom_type: str = None, trace_lines: bool = True):
        """
        Decorator for tracing custom functions.
        Usage:
            @tracer.trace_custom(name="my_function", custom_type="data_processor", trace_lines=True)
            def my_function():
                pass
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._trace_custom_execution(func, args, kwargs, name, custom_type, trace_lines)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if self.is_active:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self._trace_custom_execution(func, args, kwargs, name, custom_type, trace_lines))
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(loop)
                    else:
                        return loop.run_until_complete(self._trace_custom_execution(func, args, kwargs, name, custom_type, trace_lines))
                else:
                    return func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    async def _trace_custom_execution(self, func, args, kwargs, name=None, custom_type=None, trace_lines=True):
        """Execute a function with tracing"""
        if not self.is_active:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        start_time = datetime.now().astimezone()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        line_traces = []

        # Set up line tracing if enabled
        def trace_lines_func(frame, event, arg):
            if event == 'line' and frame.f_code == func.__code__:
                try:
                    # Get local variables, excluding special names
                    locals_dict = {}
                    for key, value in frame.f_locals.items():
                        if not key.startswith('__'):
                            try:
                                # Include all data types
                                if isinstance(value, (int, float, bool, str)):
                                    locals_dict[key] = value
                                elif isinstance(value, (list, dict, tuple, set)):
                                    locals_dict[key] = value
                                else:
                                    locals_dict[key] = str(value)
                            except:
                                pass

                    # Only record if we have variables to track
                    if locals_dict:
                        line_traces.append({
                            'variables': locals_dict,
                            'timestamp': datetime.now().astimezone().isoformat()
                        })
                except Exception:
                    pass  # Skip any errors in tracing
            return trace_lines_func

        # Start tracking network calls
        self.start_component(component_id)

        # Enable line tracing if requested
        if trace_lines:
            sys.settrace(trace_lines_func)

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Calculate resource usage
            end_time = datetime.now().astimezone()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Create component
            component = self.create_custom_component(
                component_id=component_id,
                hash_id=generate_unique_hash_simple(func),
                name=name or func.__name__,
                custom_type=custom_type or "function",
                version="1.0.0",
                start_time=start_time,
                end_time=end_time,
                memory_used=memory_used,
                line_traces=line_traces if trace_lines else [],
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
                error=None
            )

            self.add_component(component)
            return result

        except Exception as e:
            # Calculate resource usage even if there's an error
            end_time = datetime.now().astimezone()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Create error component
            error = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {}
            }

            component = self.create_custom_component(
                component_id=component_id,
                hash_id=generate_unique_hash_simple(func),
                name=name or func.__name__,
                custom_type=custom_type or "function",
                version="1.0.0",
                start_time=start_time,
                end_time=end_time,
                memory_used=memory_used,
                line_traces=line_traces if trace_lines else [],
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error
            )

            self.add_component(component)
            raise

        finally:
            # Disable line tracing if it was enabled
            if trace_lines:
                sys.settrace(None)
            # End tracking network calls
            self.end_component(component_id)

    def create_custom_component(self, **kwargs):
        """Create a custom component according to the data structure"""
        start_time = kwargs["start_time"]
        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "type": "custom",
            "name": kwargs["name"],
            "start_time": start_time.isoformat(),
            "end_time": kwargs["end_time"].isoformat(),
            "error": kwargs.get("error"),
            "parent_id": self.current_agent_id.get() if hasattr(self, 'current_agent_id') else None,
            "info": {
                "custom_type": kwargs.get("custom_type", "generic"),
                "version": kwargs.get("version", "1.0.0"),
                "memory_used": kwargs.get("memory_used", 0)
            },
            "data": {
                "input": kwargs.get("input_data"),
                "output": kwargs.get("output_data"),
                "memory_used": kwargs.get("memory_used", 0),
                "line_traces": kwargs.get("line_traces", [])
            },
            "network_calls": self.component_network_calls.get(kwargs["component_id"], []),
            "interactions": self.component_user_interaction.get(kwargs["component_id"], [])
        }

        if self.gt:
            component["data"]["gt"] = self.gt

        return component

    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        self.component_network_calls[component_id] = []

    def end_component(self, component_id):
        """End tracking network calls for a component"""
        pass

    def _sanitize_input(self, args, kwargs):
        """Sanitize input data for storage"""
        try:
            sanitized_args = [arg if isinstance(arg, (int, float, bool, str, list, dict, tuple, set)) else str(arg) for arg in args]
            sanitized_kwargs = {k: v if isinstance(v, (int, float, bool, str, list, dict, tuple, set)) else str(v) for k, v in kwargs.items()}
            return {"args": sanitized_args, "kwargs": sanitized_kwargs}
        except:
            return None

    def _sanitize_output(self, result):
        """Sanitize output data for storage"""
        try:
            if isinstance(result, (int, float, bool, str, list, dict, tuple, set)):
                return result
            return str(result)
        except:
            return None
