import json
import os
import platform
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
        
        # Create traces directory if it doesn't exist
        self.traces_dir = Path("traces")
        self.traces_dir.mkdir(exist_ok=True)
        
        # Initialize trace data
        self.trace_id = str(uuid.uuid4())
        self.start_time = datetime.now().isoformat()
        self.components: List[Component] = []
        
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
            source_code="Path to source code .zip file"  # TODO: Implement source code archiving
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
            total_cost=0.0,
            total_tokens=0,
            system_info=self._get_system_info(),
            resources=self._get_resources()
        )
        
        self.trace = Trace(
            id=self.trace_id,
            project_name=self.project_name,
            start_time=self.start_time,
            end_time="",  # Will be set when trace is stopped
            metadata=metadata,
            data=self.components,
            replays={"source": None}
        )
        
    def stop(self):
        """Stop the trace and save to JSON file"""
        if self.trace:
            self.trace.end_time = datetime.now().isoformat()
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.project_name}_{timestamp}.json"
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