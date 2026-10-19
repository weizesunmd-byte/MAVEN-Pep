"""Route only public output contracts before generation; no scoring input.

The compatibility registry contains two reusable system-level API documents,
never question identifiers, question text or answers. Modern contracts must
match a complete registered output schema in the sole trusted system message.
User messages/evidence cannot select an adapter. Unknown/conflicting contracts
fail closed. This is a two-adapter system, not recovered single-adapter memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import TASKS, task_schema

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTES = ROOT / "configs/adapter_routes_v3.json"
ROUTE_KEYS = frozenset(("active_adapter_role", "public_contract_sha256", "router_fingerprint"))


class ContractRoutingError(ValueError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _public_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractRoutingError("Duplicate key in public output schema")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ContractRoutingError("Nonfinite constant in public output schema")


def canonical_schema(value, parent_key=None):
    """Normalize object order and only schema collections with set semantics."""
    if isinstance(value, dict):
        return {key: canonical_schema(child, key) for key, child in sorted(value.items())}
    if isinstance(value, list):
        result = [canonical_schema(child) for child in value]
        if parent_key in {"required", "enum", "anyOf", "allOf", "oneOf", "type"}:
            encoded = [_json(child) for child in result]
            if len(encoded) != len(set(encoded)):
                raise ContractRoutingError("Duplicate entries in public schema contract")
            result = sorted(result, key=_json)
        return result
    return value


def _configuration(config_path=None):
    path = Path(config_path) if config_path is not None else DEFAULT_ROUTES
    config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if (config.get("schema") != "maven_final.adapter_routes.v3" or config.get("model_kind") != "dual_adapter_system"
            or config.get("decision_role") != "decision" or config.get("compatibility_role") != "compatibility"
            or config.get("new_schema_provider") != "maven_final.contracts.task_schema"
            or config.get("supported_new_tasks") != list(TASKS)):
        raise ContractRoutingError("Unsupported adapter routing configuration")
    documents = config.get("legacy_system_contracts")
    if not isinstance(documents, list) or len(documents) != 2:
        raise ContractRoutingError("Exactly two public legacy system contracts are registered")
    legacy = {}
    for document in documents:
        if not isinstance(document, dict) or set(document) != {"text", "sha256"} or not isinstance(document["text"], str):
            raise ContractRoutingError("Legacy registry may contain only public document text and its SHA256")
        text = _public_text(document["text"])
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != document["sha256"] or actual in legacy:
            raise ContractRoutingError("Public legacy contract hash is invalid or duplicated")
        legacy[actual] = text
    schemas = {task: canonical_schema(task_schema(task)) for task in TASKS}
    registry = {_digest(schema): task for task, schema in schemas.items()}
    descriptor = {"router_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "routes_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "schema_registry_sha256": _digest(schemas)}
    descriptor["router_fingerprint"] = _digest(descriptor)
    return config, legacy, registry, descriptor


def router_descriptor(*, config_path=None):
    """Content fingerprints of router code, rules and complete schema registry."""
    return _configuration(config_path)[3]


def route_public_contract(messages, *, config_path=None):
    """Return exactly three public-derived route fields; never inspect answers."""
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ContractRoutingError("Public routing input must be a message list")
    systems = [message.get("content") for message in messages if message.get("role") == "system"]
    if len(systems) != 1 or not isinstance(systems[0], str) or not systems[0].strip():
        raise ContractRoutingError("Exactly one trusted system contract is required")
    _, legacy, registry, descriptor = _configuration(config_path)
    system = _public_text(systems[0])
    public_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()
    marker = "Output schema:"
    if public_hash in legacy:
        role = "compatibility"
    else:
        if system.count(marker) != 1 or any(document in system for document in legacy.values()):
            raise ContractRoutingError("Unknown or conflicting public output contract")
        raw_schema = system.split(marker, 1)[1].strip()
        try:
            schema = json.loads(raw_schema, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            public_hash = _digest(canonical_schema(schema))
        except (ValueError, TypeError) as error:
            raise ContractRoutingError("Public output schema is not one complete valid JSON contract") from error
        if public_hash not in registry:
            raise ContractRoutingError("Complete public output schema is not in the frozen registry")
        role = "decision"
    return {"active_adapter_role": role, "public_contract_sha256": public_hash,
            "router_fingerprint": descriptor["router_fingerprint"]}
