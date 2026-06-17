from enum import Enum
from typing import Literal, Union, List
from pydantic import BaseModel


class TaskKind(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"


class ResourceCaps(BaseModel):
    cpu_cores: int
    ram_gb: float
    os_name: str
    ollama_available: bool
    ollama_models: list[str] = []
    embedding_models: list[str] = []
    vllm_available: bool = False
    vllm_models: list[str] = []
    mlx_available: bool = False
    mlx_models: list[str] = []
    # llama.cpp `llama-server` backend (D104). OpenAI-compatible, wire-identical
    # to MLX. Additive: pre-D104 agents omit these fields → defaults apply.
    llamacpp_available: bool = False
    llamacpp_models: list[str] = []
    parallel_slots: int = 1
    streaming_capable: bool = False
    context_size: int = 8192
    # Per-model context windows. Falls back to context_size when a model is not
    # present in this map. Populated by the agent at registration / heartbeat.
    model_context: dict[str, int] = {}
    # Image-gen capability (D064). Agent advertises `image_available=True` when
    # at least one verified model is present in `LLMESH_IMAGE_MODELS_DIR` AND
    # mflux import succeeds. `image_models` lists model_ids advertised
    # to the hub. `vram_gb` is UMA-available on Apple Silicon; on other
    # platforms it is best-effort (may be 0 → routing filter skips).
    image_available: bool = False
    image_models: list[str] = []
    vram_gb: float = 0.0
    # D076: agent code version read from /VERSION at agent process import.
    # Pre-D076 agents send no field → default "0.1x" (D077 tune from "unknown"):
    # the absence of the field implies the 0.1x series, which is the version
    # range that lacks D076's agent_version reporting. The dashboard adds the
    # "v" prefix; storage stays bare. Operators see "v0.1x" warning badge.
    agent_version: str = "0.1x"

class RegistrationRequest(BaseModel):
    api_key: str
    resources: ResourceCaps
    node_fingerprint: str | None = None

class InferenceRequest(BaseModel):
    owner_id: str
    prompt: str | None = None
    messages: list[dict] = []
    model: str = "llama3"
    num_ctx: int | None = None


class EmbeddingsRequest(BaseModel):
    model: str = "nomic-embed-text"
    input: Union[str, List[str]]
    encoding_format: Literal["float"] = "float"


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
    llamacpp_available: bool = False
    image_available: bool = False
    cpu_load: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Image generation (D064) — OpenAI Images API compatible
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible `POST /v1/images/generations` request.

    Subset of the OpenAI surface. Not supported in v1: `style`, `quality:"hd"`
    on dall-e-3-style models (mapped to our `quality:"quality"`), `user`,
    `moderation`, `output_format` (png-only via b64). See D064.
    """
    model: str
    prompt: str
    negative_prompt: str | None = None
    # Operator-facing shape: square / portrait / landscape (mapped to model-
    # native pixels at the agent driver). Raw "1024x1024" string passthrough
    # accepted for OpenAI-spec compat; non-matching strings 400 hub-side.
    size: Literal["square", "portrait", "landscape",
                  "256x256", "512x512", "1024x1024",
                  "1024x1792", "1792x1024",
                  "512x768", "768x512"] = "square"
    n: int = 1
    seed: int | None = None
    # Maps to step count at the agent driver:
    #   test    → 1 step (D081 — smoke-test tier, ~5s on M1 Ultra)
    #   draft   → 4 steps (default — fast iteration on schnell)
    #   quality → 20 steps (operator opt-in for final renders)
    quality: Literal["test", "draft", "quality"] = "draft"
    # b64_json only in v1; URL response mode deferred to v2.
    response_format: Literal["b64_json"] = "b64_json"


class ImageGenerationResponse(BaseModel):
    """OpenAI-shaped response: `{"created":N,"data":[{"b64_json":"..."}]}`."""
    created: int
    data: list[dict]

class Node(BaseModel):
    node_id: str
    owner_id: str
    resources: ResourceCaps
    last_seen: float
    cpu_load: float = 0.0
    latency_ms: float = 0.0
    node_token: str = ""
    # sha256(node_token) hex digest. Persisted to the nodes table; restored on
    # hub restart so already-connected agents can verify against their existing
    # plaintext without re-registration. Empty on freshly-loaded restored rows
    # only when the prior schema lacked the column. See D058.
    node_token_hash: str = ""
    # Fingerprint claimed at registration. Stored for the index in node_store
    # and so restored nodes can still be looked up by hardware identity.
    fingerprint: str = ""

class RegistrationResponse(BaseModel):
    node_id: str
    node_token: str
