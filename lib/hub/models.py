from pydantic import BaseModel
from typing import Dict, Any, Optional

class ResourceCaps(BaseModel):
    cpu_cores: int
    ram_gb: float
    os_name: str
    ollama_available: bool
    ollama_models: list[str] = []
    vllm_available: bool = False
    vllm_models: list[str] = []
    mlx_available: bool = False
    mlx_models: list[str] = []
    parallel_slots: int = 1
    streaming_capable: bool = False
    context_size: int = 8192

class RegistrationRequest(BaseModel):
    api_key: str
    resources: ResourceCaps
    node_fingerprint: str | None = None

class InferenceRequest(BaseModel):
    owner_id: str
    prompt: Optional[str] = None
    messages: list[dict] = []
    model: str = "llama3"

class AnthropicMessage(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1024

class HeartbeatRequest(BaseModel):
    ollama_available: bool
    vllm_available: bool = False
    mlx_available: bool = False
    cpu_load: float = 0.0
    latency_ms: float = 0.0

class Node(BaseModel):
    node_id: str
    owner_id: str
    resources: ResourceCaps
    last_seen: float
    cpu_load: float = 0.0
    latency_ms: float = 0.0
    node_token: str = ""

class RegistrationResponse(BaseModel):
    node_id: str
    node_token: str
