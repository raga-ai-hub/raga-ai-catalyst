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
        def decorator(target):
            # Check if target is a class
            is_class = isinstance(target, type)
            tracer = self  # Store reference to tracer instance
            
            if is_class:
                # Store original __init__
                original_init = target.__init__
                
                def wrapped_init(self, *args, **kwargs):
                    # Set agent context before initializing
                    component_id = str(uuid.uuid4())
                    hash_id = tracer._trace_sync_agent_execution.hash_id if hasattr(tracer, '_trace_sync_agent_execution') else str(uuid.uuid4())
                    
                    # Store the component ID in the instance
                    self._agent_component_id = component_id
                    
                    # Get parent agent ID if exists
                    parent_agent_id = tracer.current_agent_id.get()
                    
                    # Create agent component
                    agent_component = tracer.create_agent_component(
                        component_id=component_id,
                        hash_id=hash_id,
                        name=name,
                        agent_type=agent_type,
                        version=version,
                        capabilities=capabilities or [],
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        memory_used=0,
                        input_data=tracer._sanitize_input(args, kwargs),
                        output_data=None,
                        children=[],
                        parent_id=parent_agent_id
                    )
                    
                    # Store component for later updates
                    if not hasattr(tracer, '_agent_components'):
                        tracer._agent_components = {}
                    tracer._agent_components[component_id] = agent_component
                    
                    # If this is a nested agent, add it to parent's children
                    if parent_agent_id:
                        parent_children = tracer.agent_children.get()
                        parent_children.append(agent_component)
                        tracer.agent_children.set(parent_children)
                    else:
                        # Only add to root components if no parent
                        tracer.add_component(agent_component)
                    
                    # Call original __init__ with this agent as current
                    token = tracer.current_agent_id.set(component_id)
                    try:
                        original_init(self, *args, **kwargs)
                    finally:
                        tracer.current_agent_id.reset(token)
                
                # Wrap all public methods to track execution
                for attr_name in dir(target):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(target, attr_name)
                        if callable(attr_value):
                            def wrap_method(method):
                                @functools.wraps(method)
                                def wrapped_method(self, *args, **kwargs):
                                    # Set this agent as current during method execution
                                    token = tracer.current_agent_id.set(self._agent_component_id)
                                    
                                    # Store parent's children before setting new empty list
                                    parent_children = tracer.agent_children.get()
                                    children_token = tracer.agent_children.set([])
                                    
                                    try:
                                        start_time = datetime.now()
                                        result = method(self, *args, **kwargs)
                                        end_time = datetime.now()
                                        
                                        # Update agent component with method result
                                        if hasattr(tracer, '_agent_components'):
                                            component = tracer._agent_components.get(self._agent_component_id)
                                            if component:
                                                component['data']['output'] = tracer._sanitize_output(result)
                                                component['data']['input'] = {
                                                    'args': tracer._sanitize_input(args, kwargs),
                                                    'kwargs': {}
                                                }
                                                component['start_time'] = start_time.isoformat()
                                                component['end_time'] = end_time.isoformat()
                                                
                                                # Get children accumulated during method execution
                                                children = tracer.agent_children.get()
                                                if children:
                                                    if 'children' not in component['data']:
                                                        component['data']['children'] = []
                                                    component['data']['children'].extend(children)
                                                    
                                                    # Add this component as a child to parent's children list
                                                    parent_children.append(component)
                                                    tracer.agent_children.set(parent_children)
                                        return result
                                    finally:
                                        tracer.current_agent_id.reset(token)
                                        tracer.agent_children.reset(children_token)
                                return wrapped_method
                            
                            setattr(target, attr_name, wrap_method(attr_value))
                
                # Replace __init__ with wrapped version
                target.__init__ = wrapped_init
                return target
            else:
                # For function decorators, use existing sync/async tracing
                is_async = asyncio.iscoroutinefunction(target)
                if is_async:
                    async def wrapper(*args, **kwargs):
                        return await self._trace_agent_execution(target, name, agent_type, version, capabilities, *args, **kwargs)
                    return wrapper
                else:
                    def wrapper(*args, **kwargs):
                        return self._trace_sync_agent_execution(target, name, agent_type, version, capabilities, *args, **kwargs)
                    return wrapper

        return decorator

    def _trace_sync_agent_execution(self, func, name, agent_type, version, capabilities, *args, **kwargs):
        """Synchronous version of agent tracing"""
        if not self.is_active:
            return func(*args, **kwargs)

        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())
        hash_id = self._trace_sync_agent_execution.hash_id

        # Get parent agent ID if exists
        parent_agent_id = self.current_agent_id.get()
        
        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Initialize empty children list for this agent
        parent_children = self.agent_children.get()
        children_token = self.agent_children.set([])
        
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

            # End tracking network calls for this component
            self.end_component(component_id)

            # Create agent component with children and parent if exists
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
                children=children,
                parent_id=parent_agent_id
            )

            # Store component for later reference
            if not hasattr(self, '_agent_components'):
                self._agent_components = {}
            self._agent_components[component_id] = agent_component

            # If this is a nested agent, add it to parent's children list
            if parent_agent_id:
                parent_children.append(agent_component)
                self.agent_children.set(parent_children)
            else:
                # Only add to root components if no parent
                self.add_component(agent_component)

            return result
        finally:
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
        parent_children = self.agent_children.get()
        children_token = self.agent_children.set([])
        
        # Get parent agent ID if exists
        parent_agent_id = self.current_agent_id.get()
        
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

            # Create agent component with children and parent if exists
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
                children=children,
                parent_id=parent_agent_id  # Add parent ID if exists
            )

            # If this is a nested agent, add it to parent's children
            if parent_agent_id:
                parent_component = self._agent_components.get(parent_agent_id)
                if parent_component:
                    if 'children' not in parent_component['data']:
                        parent_component['data']['children'] = []
                    parent_component['data']['children'].append(agent_component)
            else:
                # Only add to root components if no parent
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
                children=children,
                parent_id=parent_agent_id  # Add parent ID if exists
            )

            # If this is a nested agent, add it to parent's children
            if parent_agent_id:
                parent_component = self._agent_components.get(parent_agent_id)
                if parent_component:
                    if 'children' not in parent_component['data']:
                        parent_component['data']['children'] = []
                    parent_component['data']['children'].append(agent_component)
            else:
                # Only add to root components if no parent
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
            "parent_id": kwargs.get("parent_id"),
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