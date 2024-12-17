import functools
import uuid
from datetime import datetime
import psutil
from typing import Optional, Any, Dict, List
from .unique_decorator import mydecorator

import contextvars
import asyncio

class AgentTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_agent_id = contextvars.ContextVar("agent_id", default=None)
        self.current_agent_name = contextvars.ContextVar("agent_name", default=None)
        self.agent_children = contextvars.ContextVar("agent_children", default=[])
        self.component_network_calls = contextvars.ContextVar("component_network_calls", default={})
        self._trace_sync_agent_execution = mydecorator(self._trace_sync_agent_execution)
        self._trace_agent_execution = mydecorator(self._trace_agent_execution)


    def trace_agent(self, name: str, agent_type: str = "generic", version: str = "1.0.0", capabilities: List[str] = None):
        def decorator(func):
            # Check if the function is async
            is_async = asyncio.iscoroutinefunction(func)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._trace_agent_execution(
                    func, name, agent_type, version, capabilities, *args, **kwargs
                )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._trace_sync_agent_execution(
                    func, name, agent_type, version, capabilities, *args, **kwargs
                )

            return async_wrapper if is_async else sync_wrapper

        return decorator

    def _trace_sync_agent_execution(self, func, name, agent_type, version, capabilities, *args, **kwargs):
        """Synchronous version of agent tracing"""
        if not self.is_active:
            return func(*args, **kwargs)

        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self._trace_sync_agent_execution.hash_id

        # Initialize empty children list for this agent
        children_token = self.agent_children.set([])
        
        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the agent
            result = func(*args, **kwargs)

            # Calculate resource usage
            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Get children components collected during execution
            children = self.agent_children.get()
            
            # Set parent_id for all children
            for child in children:
                child["parent_id"] = component_id

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create agent component with children
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                end_time=end_time,
                memory_used=memory_used,
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
                children=children
            )

            self.add_component(agent_component)
            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {}
            }
            
            # Get children even in case of error
            children = self.agent_children.get()
            
            # Set parent_id for all children
            for child in children:
                child["parent_id"] = component_id
            
            # End tracking network calls for this component
            self.end_component(component_id)
            
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                end_time=datetime.now(),
                memory_used=0,
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error_component,
                children=children
            )

            self.add_component(agent_component)
            raise
        finally:
            # Reset context variables
            self.current_agent_id.reset(agent_token)
            self.current_agent_name.reset(agent_name_token)
            self.agent_children.reset(children_token)

    async def _trace_agent_execution(self, func, name, agent_type, version, capabilities, *args, **kwargs):
        """Asynchronous version of agent tracing"""
        if not self.is_active:
            return await func(*args, **kwargs)

        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self._trace_agent_execution.hash_id

        # Initialize empty children list for this agent
        children_token = self.agent_children.set([])
        
        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Start tracking network calls for this component
        self.start_component(component_id)

        try:
            # Execute the agent
            result = await func(*args, **kwargs)

            # Calculate resource usage
            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Get children components collected during execution
            children = self.agent_children.get()
            
            # Set parent_id for all children
            for child in children:
                child["parent_id"] = component_id

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create agent component with children
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                end_time=end_time,
                memory_used=memory_used,
                input_data=self._sanitize_input(args, kwargs),
                output_data=self._sanitize_output(result),
                children=children
            )

            self.add_component(agent_component)
            return result

        except Exception as e:
            error_component = {
                "code": 500,
                "type": type(e).__name__,
                "message": str(e),
                "details": {}
            }
            
            # Get children even in case of error
            children = self.agent_children.get()
            
            # Set parent_id for all children
            for child in children:
                child["parent_id"] = component_id
            
            # End tracking network calls for this component
            self.end_component(component_id)
            
            agent_component = self.create_agent_component(
                component_id=component_id,
                hash_id=hash_id,
                name=name,
                agent_type=agent_type,
                version=version,
                capabilities=capabilities or [],
                start_time=start_time,
                end_time=datetime.now(),
                memory_used=0,
                input_data=self._sanitize_input(args, kwargs),
                output_data=None,
                error=error_component,
                children=children
            )

            self.add_component(agent_component)
            raise
        finally:
            # Reset context variables
            self.current_agent_id.reset(agent_token)
            self.current_agent_name.reset(agent_name_token)
            self.agent_children.reset(children_token)

    def create_agent_component(self, **kwargs):
        """Create an agent component according to the data structure"""
        start_time = kwargs["start_time"]
        component = {
            "id": kwargs["component_id"],
            "hash_id": kwargs["hash_id"],
            "source_hash_id": None,
            "type": "agent",
            "name": kwargs["name"],
            "start_time": start_time.isoformat(),
            "end_time": kwargs["end_time"].isoformat(),
            "error": kwargs.get("error"),
            "parent_id": None,
            "info": {
                "agent_type": kwargs["agent_type"],
                "version": kwargs["version"],
                "capabilities": kwargs["capabilities"],
                "memory_used": kwargs["memory_used"]
            },
            "data": {
                "input": kwargs["input_data"],
                "output": kwargs["output_data"],
                "children": kwargs.get("children", [])
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

    def start_component(self, component_id):
        """Start tracking network calls for a component"""
        component_network_calls = self.component_network_calls.get()
        if component_id not in component_network_calls:
            component_network_calls[component_id] = []
        self.component_network_calls.set(component_network_calls)

    def end_component(self, component_id):
        """End tracking network calls for a component"""
        component_network_calls = self.component_network_calls.get()
        if component_id in component_network_calls:
            component_network_calls[component_id] = []
        self.component_network_calls.set(component_network_calls)

    def _sanitize_input(self, args: tuple, kwargs: dict) -> Dict:
        """Sanitize and format input data"""
        return {
            "args": [str(arg) if not isinstance(arg, (int, float, bool, str, list, dict)) else arg for arg in args],
            "kwargs": {
                k: str(v) if not isinstance(v, (int, float, bool, str, list, dict)) else v 
                for k, v in kwargs.items()
            }
        }

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize and format output data"""
        if isinstance(output, (int, float, bool, str, list, dict)):
            return output
        return str(output)