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
from ...ragaai_catalyst import RagaAICatalyst

class BaseTracer:
    def __init__(self, user_details):
        self.user_details = user_details
        self.project_name = self.user_details['project_name']  # Access the project_name
        self.dataset_name = self.user_details['dataset_name']  # Access the dataset_name
        self.project_id = self.user_details['project_id']  # Access the project_id
        
        # Initialize trace data
        self.trace_id = str(uuid.uuid4())
        self.start_time = datetime.now().isoformat()
        self.components: List[Component] = []
        self.data_key = [{"start_time": self.start_time, 
                        "end_time": "",
                        "spans": self.components
                        }]
        
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
            cost=0.0,
            tokens=0,
            system_info=self._get_system_info(),
            resources=self._get_resources()
        )
        
        self.trace = Trace(
            id=self.trace_id,
            project_name=self.project_name,
            start_time=self.start_time,
            end_time="",  # Will be set when trace is stopped
            metadata=metadata,
            data=self.data_key,
            replays={"source": None}
        )
        
    def stop(self):
        """Stop the trace and save to JSON file"""
        if self.trace:
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
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(self.trace.__dict__, f, default=lambda o: o.__dict__, indent=2)
                
            print(f"Trace saved to {filepath}")
            # import pdb; pdb.set_trace()
            # Upload traces
            json_file_path = str(filepath)
            project_name = self.project_name
            project_id = self.project_id  # TODO: Replace with actual project ID
            dataset_name = self.dataset_name
            user_detail = self.user_details
            base_url = os.getenv('RAGAAI_CATALYST_BASE_URL')
            upload_traces = UploadTraces(
                json_file_path=json_file_path,
                project_name=project_name,
                project_id=project_id,
                dataset_name=dataset_name,
                user_detail=user_detail,
                base_url=base_url
            )
            upload_traces.upload_traces()
                
    def add_component(self, component: Component):
        """Add a component to the trace"""
        self.components.append(component)
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _change_span_ids_to_int(self, trace):
        # import pdb; pdb.set_trace()
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
                # import pdb; pdb.set_trace()
                childrens = span.data["children"]
                if childrens != []:
                    span.data["input"] = childrens[0]["data"]["input"]
                    span.data["output"] = childrens[-1]["data"]["output"]
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