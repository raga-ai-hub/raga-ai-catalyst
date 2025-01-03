import json
import os
import platform
import re
import psutil
import pkg_resources
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import sys

from .data_structure import (
    Trace, Metadata, SystemInfo, OSInfo, EnvironmentInfo,
    Resources, CPUResource, MemoryResource, DiskResource, NetworkResource,
    ResourceInfo, MemoryInfo, DiskInfo, NetworkInfo,
    Component, LLMComponent, AgentComponent, ToolComponent,
    NetworkCall, Interaction, Error
)

from ..upload_traces import UploadTraces
from .upload_agentic_traces import UploadAgenticTraces
from .upload_code import upload_code
from ...ragaai_catalyst import RagaAICatalyst

from .file_name_tracker import TrackName
from .zip_list_of_unique_files import zip_list_of_unique_files

class TracerJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return str(obj)  # Fallback to string representation
        if hasattr(obj, 'to_dict'):  # Handle objects with to_dict method
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            # Filter out None values and handle nested serialization
            return {k: v for k, v in obj.__dict__.items() 
                   if v is not None and not k.startswith('_')}
        try:
            # Try to convert to a basic type
            return str(obj)
        except:
            return None  # Last resort: return None instead of failing

class BaseTracer:
    def __init__(self, user_details):
        self.user_details = user_details
        self.project_name = self.user_details['project_name']  # Access the project_name
        self.dataset_name = self.user_details['dataset_name']  # Access the dataset_name
        self.project_id = self.user_details['project_id']  # Access the project_id
        
        # Initialize trace data
        self.trace_id = None
        self.start_time = None 
        self.components: List[Component] = []
        self.file_tracker = TrackName()
        
    def _get_system_info(self) -> SystemInfo:
        # Get OS info
        os_info = OSInfo(
            name=platform.system(),
            version=platform.version(),
            platform=platform.machine(),
            kernel_version=platform.release()
        )
        
        # Get Python environment info
        installed_packages = [f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]
        env_info = EnvironmentInfo(
            name="Python",
            version=platform.python_version(),
            packages=installed_packages,
            env_path=sys.prefix,
            command_to_run=f"python {sys.argv[0]}"
        )
        
        return SystemInfo(
            id=f"sys_{self.trace_id}",
            os=os_info,
            environment=env_info,
            source_code="Path to the source code .zip file in format hashid.zip"  # TODO: Implement source code archiving
        )
        
    def _get_resources(self) -> Resources:
        # CPU info
        cpu_info = ResourceInfo(
            name=platform.processor(),
            cores=psutil.cpu_count(logical=False),
            threads=psutil.cpu_count(logical=True)
        )
        cpu = CPUResource(
            info=cpu_info,
            interval="5s",
            values=[psutil.cpu_percent()]
        )
        
        # Memory info
        memory = psutil.virtual_memory()
        mem_info = MemoryInfo(
            total=memory.total / (1024**3),  # Convert to GB
            free=memory.available / (1024**3)
        )
        mem = MemoryResource(
            info=mem_info,
            interval="5s",
            values=[memory.percent]
        )
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_info = DiskInfo(
            total=disk.total / (1024**3),
            free=disk.free / (1024**3)
        )
        disk_io = psutil.disk_io_counters()
        disk_resource = DiskResource(
            info=disk_info,
            interval="5s",
            read=[disk_io.read_bytes / (1024**2)],  # MB
            write=[disk_io.write_bytes / (1024**2)]
        )
        
        # Network info
        net_io = psutil.net_io_counters()
        net_info = NetworkInfo(
            upload_speed=net_io.bytes_sent / (1024**2),  # MB
            download_speed=net_io.bytes_recv / (1024**2)
        )
        net = NetworkResource(
            info=net_info,
            interval="5s",
            uploads=[net_io.bytes_sent / (1024**2)],
            downloads=[net_io.bytes_recv / (1024**2)]
        )
        
        return Resources(cpu=cpu, memory=mem, disk=disk_resource, network=net)
        
    def start(self):
        """Initialize a new trace"""
        metadata = Metadata(
            cost={},
            tokens={},
            system_info=self._get_system_info(),
            resources=self._get_resources()
        )

        # Generate a unique trace ID, when trace starts
        self.trace_id = str(uuid.uuid4()) 
        
        # Get the start time
        self.start_time = datetime.now().isoformat()
        
        self.data_key = [{"start_time": datetime.now().isoformat(), 
                        "end_time": "",
                        "spans": self.components
                        }]
        
        self.trace = Trace(
            id=self.trace_id,
            project_name=self.project_name,
            start_time=datetime.now().isoformat(),
            end_time="",  # Will be set when trace is stopped
            metadata=metadata,
            data=self.data_key,
            replays={"source": None}
        )
        
    def stop(self):
        """Stop the trace and save to JSON file"""
        if hasattr(self, 'trace'):
            self.trace.data[0]["end_time"] = datetime.now().isoformat()
            self.trace.end_time = datetime.now().isoformat()

            # Change span ids to int
            self.trace = self._change_span_ids_to_int(self.trace)
            self.trace = self._change_agent_intput_output(self.trace)
            self.trace = self._extract_cost_tokens(self.trace)
            
            # Create traces directory if it doesn't exist
            self.traces_dir = Path("traces")
            self.traces_dir.mkdir(exist_ok=True)
            filename = self.trace.id + ".json"
            filepath = self.traces_dir / filename

            #get unique files and zip it. Generate a unique hash ID for the contents of the files
            list_of_unique_files = self.file_tracker.get_unique_files()
            hash_id, zip_path = zip_list_of_unique_files(list_of_unique_files)

            #replace source code with zip_path
            self.trace.metadata.system_info.source_code = hash_id

            # # Save to JSON file using custom encoder
            # import pdb; pdb.set_trace()

            # Clean up trace_data before saving
            trace_data = self.trace.__dict__
            cleaned_trace_data = self._clean_trace(trace_data)

            with open(filepath, 'w') as f:
                json.dump(cleaned_trace_data, f, cls=TracerJSONEncoder, indent=2)
                
            print(f"Trace saved to {filepath}")
            # Upload traces
            json_file_path = str(filepath)
            project_name = self.project_name
            project_id = self.project_id 
            dataset_name = self.dataset_name
            user_detail = self.user_details
            base_url = os.getenv('RAGAAI_CATALYST_BASE_URL')
            upload_traces = UploadAgenticTraces(
                json_file_path=json_file_path,
                project_name=project_name,
                project_id=project_id,
                dataset_name=dataset_name,
                user_detail=user_detail,
                base_url=base_url
            )
            upload_traces.upload_agentic_traces()

            #Upload Codehash
            response = upload_code(
                hash_id=hash_id,
                zip_path=zip_path,
                project_name=project_name,
                dataset_name=dataset_name
            )
            print(response)
            
        # Cleanup
        self.components = []
        self.file_tracker = TrackName()
                
    def add_component(self, component: Component):
        """Add a component to the trace"""
        self.components.append(component)
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _change_span_ids_to_int(self, trace):
        id, parent_id = 1, 0
        for span in trace.data[0]["spans"]:
            span.id = id
            span.parent_id = parent_id
            id += 1
            if span.type=="agent":
                for children in span.data["children"]:
                    children["id"] = id
                    children["parent_id"] = span.id
                    id += 1
        return trace

    def _change_agent_intput_output(self, trace):
        for span in trace.data[0]["spans"]:
            if span.type == "agent":
                childrens = span.data["children"]
                span.data["input"] = None
                span.data["output"] = None
                if childrens:
                    # Find first non-null input going forward
                    for child in childrens:
                        input_data = child["data"].get("input")
                        if input_data:
                            span.data["input"] = input_data['args'] if hasattr(input_data, 'args') else input_data
                            break
                    
                    # Find first non-null output going backward
                    for child in reversed(childrens):
                        output_data = child["data"].get("output")
                        if output_data:
                            span.data["output"] = output_data
                            break
        return trace
    
    def _extract_cost_tokens(self, trace):
        cost = {}
        tokens = {}
        for span in trace.data[0]["spans"]:
            if span.type == "llm":
                info = span.info
                if isinstance(info, dict):
                    cost_info = info.get('cost', {})
                    for key, value in cost_info.items():
                        if key not in cost:
                            cost[key] = 0 
                        cost[key] += value
                    token_info = info.get('tokens', {})
                    for key, value in token_info.items():
                        if key not in tokens:
                            tokens[key] = 0
                        tokens[key] += value
            if span.type == "agent":
                for children in span.data["children"]:
                    if 'type' not in children:
                        continue
                    if children["type"] != "llm":
                        continue
                    info = children["info"]
                    if isinstance(info, dict):
                        cost_info = info.get('cost', {})
                        for key, value in cost_info.items():
                            if key not in cost:
                                cost[key] = 0 
                            cost[key] += value
                        token_info = info.get('tokens', {})
                        for key, value in token_info.items():
                            if key not in tokens:
                                tokens[key] = 0
                            tokens[key] += value
        trace.metadata.cost = cost
        trace.metadata.tokens = tokens
        return trace

    def _clean_trace(self, trace):
        # Convert span to dict if it has to_dict method
        def _to_dict_if_needed(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj

        # Remove any spans without hash ids
        trace['data'][0]['spans'] = [
            _to_dict_if_needed(span) for span in trace['data'][0]['spans'] 
            if 'hash_id' in _to_dict_if_needed(span)
        ]

        # For spans of type agent, remove any children without hash ids upto any depth
        for span in trace['data'][0]['spans']:
            if span['type'] == 'agent':
                span['data']['children'] = [
                    _to_dict_if_needed(child) for child in span['data']['children'] 
                    if 'hash_id' in _to_dict_if_needed(child)
                ]

                # Recursively remove any children without hash ids upto any depth
                for child in span['data']['children']:
                    if 'children' in child:
                        child['children'] = self._clean_trace(child['children'])

        return trace