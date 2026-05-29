import hashlib
import logging
import time
import secrets
from typing import Dict, Optional
from .models import Node, ResourceCaps
from .config import api_keys
from . import node_store as _node_store_mod

logger = logging.getLogger("llmesh.hub.storage")

# In-memory authoritative state. Persistence via node_store is additive and
# best-effort; in-memory mutations never roll back on store failure (D003 policy).
_nodes: Dict[str, Node] = {}


def authenticate_owner(api_key: str) -> Optional[str]:
    return api_keys.get(api_key)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def store_node(node: Node) -> None:
    """Insert/replace node in the in-memory registry and write through to the
    durability layer. The write-through is fire-and-forget at the asyncio
    layer (best-effort, matches D003); save_node itself swallows DB errors."""
    _nodes[node.node_id] = node
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(_node_store_mod.get_node_store().save_node(
                node, node.fingerprint or node.node_id,
            ))
    except RuntimeError:
        # No running loop — sync test context. Persistence is skipped; tests
        # that exercise the persistence path drive the store directly.
        pass


def get_node(node_id: str) -> Optional[Node]:
    return _nodes.get(node_id)


def get_all_nodes() -> list[Node]:
    return list(_nodes.values())


def verify_node_token(node_id: str, token: str) -> bool:
    """Validate a presented node_token against the stored credential.

    Two acceptance paths:
      1. Plaintext compare — used during normal operation while the hub still
         holds the plaintext token issued at /register.
      2. Hash compare — used after a hub restart that restored the node from
         the persistence layer. Plaintext is empty; node_token_hash holds
         sha256(token) for the original credential. The running agent presents
         the same plaintext it received at register time; we hash it and
         compare against the stored digest.

    Both paths use `secrets.compare_digest` for timing-attack safety.
    """
    node = get_node(node_id)
    if node is None:
        return False
    if node.node_token:
        return secrets.compare_digest(node.node_token, token)
    if node.node_token_hash:
        return secrets.compare_digest(
            node.node_token_hash, _hash_token(token),
        )
    return False


def prune_inactive_nodes(max_age_sec: float = 90.0) -> list[str]:
    """Remove nodes whose last_seen is older than max_age_sec. Caller is
    expected to call tasks.drop_node_queue(node_id) for each returned id
    AFTER recovering any pending/claimed tasks (see D035 + server.cleanup_loop).

    Also deletes the persisted row so the durability layer stays in sync
    with the in-memory state (D058)."""
    cutoff = time.time() - max_age_sec
    stale = [nid for nid, n in _nodes.items() if n.last_seen < cutoff]
    if not stale:
        return stale
    for nid in stale:
        del _nodes[nid]
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        if loop.is_running():
            store = _node_store_mod.get_node_store()
            for nid in stale:
                loop.create_task(store.delete_node(nid))
    except RuntimeError:
        pass
    return stale


async def load_persisted_nodes(max_age_sec: float = 90.0) -> int:
    """Restore nodes from the persistence layer into `_nodes` at startup.

    Rows with `last_seen` older than max_age_sec are pruned from the store
    before loading — matches steady-state behaviour so a hub restart after a
    long downtime does not surface dead nodes (feature_hub_state_durability.md
    §4.4 step 1). Restored Node objects carry `node_token_hash` populated and
    an empty `node_token` plaintext; verify_node_token falls back to the hash
    path until those nodes re-register (which overwrites with fresh plaintext).

    Returns the count of nodes restored.
    """
    store = _node_store_mod.get_node_store()
    rows = await store.load_persisted(max_age_sec=max_age_sec)
    restored = 0
    for row in rows:
        try:
            resources = ResourceCaps(**row["resources"])
        except Exception as exc:
            logger.warning(
                "Skipping persisted node %s: resources deserialization failed: %s",
                row["node_id"], exc,
            )
            continue
        node = Node(
            node_id=row["node_id"],
            owner_id=row["owner_id"],
            resources=resources,
            last_seen=row["last_seen"],
            cpu_load=row.get("cpu_load", 0.0),
            latency_ms=row.get("latency_ms", 0.0),
            node_token="",
            node_token_hash=row["node_token_hash"],
            fingerprint=row.get("fingerprint", ""),
        )
        _nodes[node.node_id] = node
        restored += 1
    return restored
