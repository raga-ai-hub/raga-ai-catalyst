import functools
import uuid
from datetime import datetime
import psutil
from typing import Optional, Any, Dict, List
from ..utils.unique_decorator import mydecorator, generate_unique_hash_simple
import contextvars
import asyncio
from ..utils.file_name_tracker import TrackName


class AgentTracerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_tracker = TrackName()
        self.current_agent_id = contextvars.ContextVar("agent_id", default=None)
        self.current_agent_name = contextvars.ContextVar("agent_name", default=None)
        self.agent_children = contextvars.ContextVar("agent_children", default=[])
        self.component_network_calls = contextvars.ContextVar("component_network_calls", default={})
        self.component_user_interaction = contextvars.ContextVar("component_user_interaction", default={})
        self.version = contextvars.ContextVar("version", default="1.0.0")
        self.agent_type = contextvars.ContextVar("agent_type", default="generic")
        self.capabilities = contextvars.ContextVar("capabilities", default=[])
        self.start_time = contextvars.ContextVar("start_time", default=None)
        self.input_data = contextvars.ContextVar("input_data", default=None)
        self.gt = None


    def trace_agent(self, name: str, agent_type: str = None, version: str = None, capabilities: List[str] = None):
        def decorator(target):
            # Check if target is a class
            is_class = isinstance(target, type)
            tracer = self  # Store reference to tracer instance
            top_level_hash_id = generate_unique_hash_simple(target)   # Generate hash based on the decorated target code
            self.version.set(version)
            self.agent_type.set(agent_type)
            self.capabilities.set(capabilities)
            
            if is_class:
                # Store original __init__
                original_init = target.__init__
                
                def wrapped_init(self, *args, **kwargs):
                    self.gt = kwargs.get('gt', None) if kwargs else None
                    # Set agent context before initializing
                    component_id = str(uuid.uuid4())
                    hash_id = top_level_hash_id
                    
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
                                @self.file_tracker.trace_decorator
                                @functools.wraps(method)
                                def wrapped_method(self, *args, **kwargs):
                                    self.gt = kwargs.get('gt', None) if kwargs else None
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
                                                component['data']['input'] = tracer._sanitize_input(args, kwargs)
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
                        return await self._trace_agent_execution(target, name, agent_type, version, capabilities, top_level_hash_id, *args, **kwargs)
                    return wrapper
                else:
                    def wrapper(*args, **kwargs):
                        return self._trace_sync_agent_execution(target, name, agent_type, version, capabilities, *args, **kwargs)
                    return wrapper

        return decorator

    def _trace_sync_agent_execution(self, func, name, agent_type, version, capabilities, *args, **kwargs):
        # Generate a unique hash_id for this execution context
        hash_id = str(uuid.uuid4())

        """Synchronous version of agent tracing"""
        if not self.is_active:
            return func(*args, **kwargs)

        start_time = datetime.now()
        self.start_time = start_time
        self.input_data = self._sanitize_input(args, kwargs)
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())

        # Extract ground truth if present
        ground_truth = kwargs.pop('gt', None) if kwargs else None

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
                input_data=self.input_data,
                output_data=self._sanitize_output(result),
                children=children,
                parent_id=parent_agent_id
            )
            # Add ground truth to component data if present
            if ground_truth is not None:
                agent_component["data"]["gt"] = ground_truth
                        
            # Add this component as a child to parent's children list
            parent_children.append(agent_component)
            self.agent_children.set(parent_children)

            # Only add to root components if no parent
            if not parent_agent_id:
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
                input_data=self.input_data,
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
            self.current_agent_id.reset(agent_token)
            self.current_agent_name.reset(agent_name_token)
            self.agent_children.reset(children_token)

    async def _trace_agent_execution(self, func, name, agent_type, version, capabilities, hash_id, *args, **kwargs):
        """Asynchronous version of agent tracing"""
        if not self.is_active:
            return await func(*args, **kwargs)

        start_time = datetime.now()
        start_memory = psutil.Process().memory_info().rss
        component_id = str(uuid.uuid4())

        # Extract ground truth if present
        ground_truth = kwargs.pop('gt', None) if kwargs else None

        # Get parent agent ID if exists
        parent_agent_id = self.current_agent_id.get()
        
        # Set the current agent context
        agent_token = self.current_agent_id.set(component_id)
        agent_name_token = self.current_agent_name.set(name)

        # Initialize empty children list for this agent
        parent_children = self.agent_children.get()
        children_token = self.agent_children.set([])

        try:
            # Execute the agent
            result = await func(*args, **kwargs)

            # Calculate resource usage
            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss
            memory_used = max(0, end_memory - start_memory)

            # Get children components collected during execution
            children = self.agent_children.get()

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

            # Add ground truth to component data if present
            if ground_truth is not None:
                agent_component["data"]["gt"] = ground_truth

            # Add this component as a child to parent's children list
            parent_children.append(agent_component)
            self.agent_children.set(parent_children)

            # Only add to root components if no parent
            if not parent_agent_id:
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
            "interactions": self.component_user_interaction.get(kwargs["component_id"], [])
        }

        if self.gt: 
            component["data"]["gt"] = self.gt

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

    def _sanitize_input(self, args: tuple, kwargs: dict) -> str:
        """Convert input arguments to text format.
        
        Args:
            args: Input arguments tuple
            kwargs: Input keyword arguments dict
            
        Returns:
            str: Text representation of the input arguments
        """
        def _sanitize_value(value):
            if isinstance(value, dict):
                return str({k: _sanitize_value(v) for k, v in value.items()})
            elif isinstance(value, (list, tuple)):
                return str([_sanitize_value(item) for item in value])
            return str(value)
        
        sanitized_args = [_sanitize_value(arg) for arg in args]
        sanitized_kwargs = {k: _sanitize_value(v) for k, v in kwargs.items()}
        return str({"args": sanitized_args, "kwargs": sanitized_kwargs})

    def _sanitize_output(self, output: Any) -> Any:
        """Sanitize and format output data"""
        if isinstance(output, (int, float, bool, str, list, dict)):
            return output
        return str(output)