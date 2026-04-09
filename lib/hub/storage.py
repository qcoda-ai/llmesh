import time
import secrets
from typing import Dict, Optional
from .models import Node
from .config import api_keys

# In-memory storage for POC
_nodes: Dict[str, Node] = {}


def authenticate_owner(api_key: str) -> Optional[str]:
    return api_keys.get(api_key)

def store_node(node: Node) -> None:
    _nodes[node.node_id] = node

def get_node(node_id: str) -> Optional[Node]:
    return _nodes.get(node_id)

def get_all_nodes() -> list[Node]:
    return list(_nodes.values())

def verify_node_token(node_id: str, token: str) -> bool:
    node = get_node(node_id)
    if node is None or not node.node_token:
        return False
    return secrets.compare_digest(node.node_token, token)

def prune_inactive_nodes(max_age_sec: float = 90.0) -> list[str]:
    cutoff = time.time() - max_age_sec
    stale = [nid for nid, n in _nodes.items() if n.last_seen < cutoff]
    for nid in stale:
        del _nodes[nid]
    return stale
