from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class OSInfo:
    name: str
    version: str
    platform: str
    kernel_version: str

@dataclass
class EnvironmentInfo:
    name: str
    version: str
    packages: List[str]
    env_path: str
    command_to_run: str

@dataclass
class SystemInfo:
    id: str
    os: OSInfo
    environment: EnvironmentInfo
    source_code: str

@dataclass
class ResourceInfo:
    name: str
    cores: int
    threads: int

@dataclass
class CPUResource:
    info: ResourceInfo
    interval: str
    values: List[float]

@dataclass
class MemoryInfo:
    total: float
    free: float

@dataclass
class MemoryResource:
    info: MemoryInfo
    interval: str
    values: List[float]

@dataclass
class DiskInfo:
    total: float
    free: float

@dataclass
class DiskResource:
    info: DiskInfo
    interval: str
    read: List[float]
    write: List[float]

@dataclass
class NetworkInfo:
    upload_speed: float
    download_speed: float

@dataclass
class NetworkResource:
    info: NetworkInfo
    interval: str
    uploads: List[float]
    downloads: List[float]

@dataclass
class Resources:
    cpu: CPUResource
    memory: MemoryResource
    disk: DiskResource
    network: NetworkResource

@dataclass
class Metadata:
    total_cost: float
    total_tokens: int
    system_info: SystemInfo
    resources: Resources

@dataclass
class NetworkCall:
    url: str
    method: str
    status_code: int
    response_time: float
    bytes_sent: int
    bytes_received: int
    protocol: str
    connection_id: str
    parent_id: str
    request: Dict[str, Any]
    response: Dict[str, Any]

@dataclass
class Interaction:
    id: str
    interaction_type: str
    content: Optional[str]
    timestamp: str

@dataclass
class Error:
    code: int
    type: str
    message: str
    details: Dict[str, Any]

@dataclass
class LLMParameters:
    temperature: float
    top_p: float
    max_tokens: int

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Cost:
    prompt_cost: float
    completion_cost: float
    total_cost: float

@dataclass
class LLMInfo:
    model: str
    parameters: LLMParameters
    token_usage: TokenUsage
    cost: Cost

@dataclass
class AgentInfo:
    agent_type: str
    version: str
    capabilities: List[str]

@dataclass
class ToolInfo:
    tool_type: str
    version: str
    memory_used: int

@dataclass
class LLMComponent:
    id: str
    hash_id: str
    source_hash_id: Optional[str]
    type: str = "llm"
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    error: Optional[Error] = None
    parent_id: Optional[str] = None
    info: LLMInfo = None
    data: Dict[str, Any] = None
    network_calls: List[NetworkCall] = None
    interactions: List[Interaction] = None

@dataclass
class AgentComponent:
    id: str
    hash_id: str
    source_hash_id: Optional[str]
    type: str = "agent"
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    error: Optional[Error] = None
    parent_id: Optional[str] = None
    info: AgentInfo = None
    data: Dict[str, Any] = None
    network_calls: List[NetworkCall] = None
    interactions: List[Interaction] = None
    # children: List['Component'] = None

@dataclass
class ToolComponent:
    id: str
    hash_id: str
    source_hash_id: Optional[str]
    type: str = "tool"
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    error: Optional[Error] = None
    parent_id: Optional[str] = None
    info: ToolInfo = None
    data: Dict[str, Any] = None
    network_calls: List[NetworkCall] = None
    interactions: List[Interaction] = None

@dataclass
class ComponentInfo:
    tool_type: Optional[str] = None
    agent_type: Optional[str] = None
    version: str = ""
    capabilities: Optional[List[str]] = None
    memory_used: Optional[int] = None
    model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, int]] = None
    cost: Optional[Dict[str, float]] = None

@dataclass
class Component:
    id: str
    hash_id: str
    source_hash_id: Optional[str]
    type: str
    name: str
    start_time: str
    end_time: str
    error: Optional[Error]
    parent_id: Optional[str]
    info: ComponentInfo
    data: Dict[str, Any]
    network_calls: List[NetworkCall]
    interactions: List[Interaction]
    children: Optional[List['Component']] = None

@dataclass
class Trace:
    id: str
    project_name: str
    start_time: str
    end_time: str
    metadata: Metadata
    data: List[Dict[str, Any]]
    replays: Optional[Dict[str, Any]]