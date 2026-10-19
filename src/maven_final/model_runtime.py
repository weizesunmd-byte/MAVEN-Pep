"""Bounded local-only model runtime shared by training, inference and promotion.

Validation has no GPU/model-generation claim. Importing this module requires
only the standard library; tokenizer/tensor/GPU dependencies are loaded lazily.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import sys
from contextlib import contextmanager

BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
OLD_ADAPTER_SHA256 = "c347c41a29570367f4443eb8de4a4a2c6265c1d2d1290722b334a51042e48f9a"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
ACTION_TASKS = frozenset({"plan_discovery", "resolve_measurement", "identity_gate", "claim_scope", "interpret_ranking", "feedback_gate"})
ACTION_VALUES = frozenset({"run", "stop", "request_information"})


class RuntimeContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeContractError(f"Blank JSONL line {number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeContractError("JSONL rows must be objects")
            rows.append(value)
    if not rows:
        raise RuntimeContractError(f"Empty dataset: {path}")
    return rows


def write_json(path: Path, value, *, exclusive=False) -> None:
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def runtime_versions(*, require_gpu_stack=False) -> dict:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "safetensors", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if require_gpu_stack and any(value is None for value in versions.values()):
        raise RuntimeContractError("Actual-model inference requires the complete installed GPU runtime version fingerprint")
    return versions


def select_inference_models(names, split: str, candidate_adapter: Path | None, *, validate_only=False) -> tuple[str, ...]:
    if not names or len(names) != len(set(names)) or any(name not in {"base", "old", "candidate"} for name in names):
        raise RuntimeContractError("Select each supported model at most once")
    selected = tuple(name for name in ("base", "old", "candidate") if name in names)
    if split == "sealed" and selected != ("base", "old", "candidate"):
        raise RuntimeContractError("Sealed inference requires all three models; use dev for preliminary baselines")
    if "candidate" in selected and candidate_adapter is None and not validate_only:
        raise RuntimeContractError("Actual candidate inference requires --candidate-adapter")
    return selected


def verify_source_pins(manifest: dict, project_root: Path) -> dict:
    """Verify contracts/builder bytes at the actual entry point; never update pins."""
    hashes = manifest.get("code_hashes")
    required = {"src/maven_final/contracts.py", "scripts/build_dataset.py"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise RuntimeContractError("Dataset manifest lacks the contracts/builder source pins")
    root = Path(project_root).resolve(strict=True)
    checked = {}
    for relative, expected in sorted(hashes.items()):
        if not isinstance(relative, str) or PureWindowsPath(relative).drive or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative.replace("\\", "/")).parts:
            raise RuntimeContractError("Source pins must use project-relative paths without traversal")
        if not isinstance(expected, str) or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeContractError("Source pin is not a canonical SHA-256 digest")
        try:
            path = (root / relative).resolve(strict=True)
        except OSError as error:
            raise RuntimeContractError(f"Pinned source is missing: {relative}") from error
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeContractError(f"Pinned source resolves outside the project or is not a file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeContractError(f"Contracts/builder source pin mismatch: {relative}; the lock is not refreshed")
        checked[relative] = actual
    return {"verified": True, "code_hashes": checked}


def verify_training_sources(adapter_dir: Path, training_manifest: dict) -> dict:
    """Check captured training bytes against their receipt, not today's source tree."""
    expected_fields = {
        "training_sources/train_candidate.py": "training_script_sha256",
        "training_sources/model_runtime.py": "runtime_script_sha256",
        "training_sources/train_config.json": "config_sha256",
    }
    captured = training_manifest.get("training_sources")
    if not isinstance(captured, dict) or set(captured) != set(expected_fields):
        raise RuntimeContractError("Completed training must include exactly the three captured source/config files")
    root = Path(adapter_dir).resolve(strict=True)
    checked = {}
    for relative, receipt_field in expected_fields.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeContractError("Captured training source is missing or resolves outside the adapter")
        actual = sha256_file(path)
        if captured[relative] != actual or training_manifest.get(receipt_field) != actual:
            raise RuntimeContractError(f"Captured training source hash mismatch: {relative}")
        checked[relative] = actual
    if read_json(root / "training_sources/train_config.json") != training_manifest.get("config"):
        raise RuntimeContractError("Captured train config is not the effective training configuration")
    return {"verified": True, "files_sha256": checked, "fingerprint": fingerprint(checked)}


def check_legacy39_retention(candidate_metrics: dict, old_metrics: dict, policy: dict) -> dict:
    gate = policy["legacy39_retention"]
    if gate.get("metric") != "field_delivery_score" or gate.get("metric_scale") != [0, 100] or gate.get("maximum_drop_points") != 1.0:
        raise RuntimeContractError("Legacy39 retention is fixed at a maximum 1.0-point field-delivery drop")
    values = [candidate_metrics.get(gate["metric"]), old_metrics.get(gate["metric"])]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise RuntimeContractError("Legacy39 field-delivery retention needs actual scores on the 0..100 scale")
    drop = values[1] - values[0]
    if drop > gate["maximum_drop_points"] + 1e-12:
        raise RuntimeContractError("Candidate exceeds the predeclared Legacy39 field-delivery regression limit")
    return {"metric": gate["metric"], "candidate": values[0], "old": values[1], "drop_points": drop,
            "maximum_drop_points": gate["maximum_drop_points"], "passed": True}


def verify_sealed_input_binding(dataset_manifest: dict, input_sha256: str, case_count: int) -> None:
    expected = dataset_manifest.get("artifact_hashes", {}).get("benchmarks/test_inputs.jsonl")
    if not isinstance(expected, str) or input_sha256 != expected or case_count != 360:
        raise RuntimeContractError("Sealed inference requires the frozen 360-case test_inputs SHA, not dev or another public file")


def case_owner(index: int, world_size: int) -> int:
    if type(index) is not int or index < 0 or type(world_size) is not int or world_size < 1:
        raise RuntimeContractError("Invalid case index or inference world size")
    return index % world_size


def inference_batches(owned: list[tuple], completed: dict, model: str, batch_size: int):
    if type(batch_size) is not int or batch_size < 1:
        raise RuntimeContractError("Inference batch size must be positive")
    pending = [(case_id, case) for case_id, case in owned if (model, case_id) not in completed]
    return [pending[start:start + batch_size] for start in range(0, len(pending), batch_size)]


def pad_inference_batch(tokenizer, sequences: list[list[int]]):
    if not sequences or any(not sequence for sequence in sequences):
        raise RuntimeContractError("Cannot pad an empty inference prompt/batch")
    tokenizer.padding_side = "left"
    lengths = [len(sequence) for sequence in sequences]
    padded = tokenizer.pad({"input_ids": sequences, "attention_mask": [[1] * size for size in lengths]},
                           padding=True, return_tensors="pt")
    width = padded["input_ids"].shape[1]
    for index, sequence in enumerate(sequences):
        if padded["input_ids"][index, width - len(sequence):].tolist() != sequence or int(padded["attention_mask"][index].sum()) != len(sequence):
            raise RuntimeContractError("Left padding changed prompt tokens or their attention mask")
        if bool(padded["attention_mask"][index, :width - len(sequence)].any()):
            raise RuntimeContractError("Padding tokens must not receive attention")
    return padded, lengths


def trim_generated_sequence(output_ids: list[int], padded_input_width: int, eos_ids, max_new_tokens: int):
    generated = list(output_ids[padded_input_width:])
    if not generated:
        raise RuntimeContractError("Generation returned no new token")
    eos = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])
    for position, token in enumerate(generated):
        if token in eos:
            generated = generated[:position + 1]
            break
    if len(generated) > max_new_tokens:
        raise RuntimeContractError("A generated sequence exceeded its fixed token budget")
    finish = "eos" if generated[-1] in eos else "length" if len(generated) >= max_new_tokens else "other"
    return generated, finish


def recover_batch_checkpoints(batch_dir: Path, records_dir: Path, immutable: dict, run_fingerprint: str, rank: int):
    """Recover first answers from atomic whole-batch records; never regenerate them."""
    known = set()
    for path in sorted(Path(batch_dir).glob("*.json")):
        batch = read_json(path)
        if batch.get("generation_rank") != rank:
            continue
        records = batch.get("records", [])
        identifier = batch.get("generation_batch_id")
        if batch.get("status") != "completed_real_generation_batch" or not records or path.stem != identifier:
            raise RuntimeContractError("Incomplete or malformed batch checkpoint")
        audit_inference_records(records, immutable, run_fingerprint)
        if len(records) > immutable["execution"]["per_rank_batch_size"]:
            raise RuntimeContractError("Batch checkpoint exceeds the frozen configured batch size")
        for record in records:
            if record.get("generation_batch_id") != identifier or record.get("generation_batch_size") != len(records) or record.get("generation_rank") != rank:
                raise RuntimeContractError("Batch checkpoint membership/owner mismatch")
            destination = Path(records_dir) / f"{record['model']}-{record['case_sha256']}.json"
            if destination.exists():
                if read_json(destination) != record:
                    raise RuntimeContractError("Existing first answer differs from its batch checkpoint")
            else:
                temporary = destination.with_suffix(f".rank{rank}.incomplete")
                write_json(temporary, record)
                temporary.replace(destination)
        known.add(identifier)
    return known


def audit_inference_records(records: list[dict], immutable: dict, run_fingerprint: str, *, complete=False) -> dict:
    cases = immutable["case_ids"]
    models = immutable["models"]
    world = immutable["execution"]["world_size"]
    owners = {case_id: case_owner(index, world) for index, case_id in enumerate(cases)}
    observed = {}
    routes = immutable.get("case_routes")
    if any(spec.get("model_kind") == "dual_adapter_system" for spec in models.values()):
        if not isinstance(routes, dict) or set(routes) != set(cases):
            raise RuntimeContractError("Routed inference requires a public contract route for every case")
    batch_adapters = {}
    for record in records:
        key = (record.get("model"), record.get("case_id"))
        if key[0] not in models or key[1] not in owners or key in observed:
            raise RuntimeContractError("Duplicate or unknown inference record; one response per model/case is required")
        accept_completed_record(record, run_fingerprint, immutable["case_hashes"][key[1]], models[key[0]]["model_fingerprint"])
        if record.get("generation_rank") != owners[key[1]] or record.get("world_size") != world:
            raise RuntimeContractError("Inference record violates deterministic case/rank ownership")
        for name in ("input_sha256", "settings_fingerprint", "runtime_versions"):
            if record.get(name) != immutable[name]:
                raise RuntimeContractError(f"Inference record {name} binding mismatch")
        routed = models[key[0]].get("model_kind") == "dual_adapter_system"
        verify_record_adapter_binding(record, models[key[0]], routes[key[1]] if routed else None)
        if routed:
            batch_id = record.get("generation_batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise RuntimeContractError("Routed generation requires a recorded single-adapter batch")
            batch_key = (key[0], batch_id)
            active = record["active_adapter_sha256"]
            if batch_key in batch_adapters and batch_adapters[batch_key] != active:
                raise RuntimeContractError("A generation batch cannot contain different active adapters")
            batch_adapters[batch_key] = active
        observed[key] = record
    if complete and set(observed) != {(name, case_id) for name in models for case_id in cases}:
        raise RuntimeContractError("Distributed inference is incomplete; completed manifest cannot be published")
    return observed


def _adapter_digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeContractError(f"{label} requires a lowercase SHA256 digest")
    return value


def model_candidate_component(model_spec: dict) -> dict:
    """Expose the actual decision component without mislabelling R1 responses."""
    if not isinstance(model_spec, dict):
        raise RuntimeContractError("Model descriptor must be an object")
    if model_spec.get("model_kind") != "dual_adapter_system":
        return dict(model_spec)
    allowed = {"model_kind", "base_model", "adapter_sha256", "components", "training_sources_fingerprint", "router_descriptor", "model_fingerprint"}
    if set(model_spec) - allowed or not isinstance(model_spec.get("base_model"), dict):
        raise RuntimeContractError("System identity must contain base/components/router only, not case routes or benchmark metadata")
    if "adapter_sha256" not in model_spec or model_spec["adapter_sha256"] is not None:
        raise RuntimeContractError("A dual-adapter system has no single top-level adapter SHA")
    components = model_spec.get("components")
    if not isinstance(components, dict) or set(components) != {"decision", "compatibility"}:
        raise RuntimeContractError("A dual-adapter system requires both named components")
    for role in ("decision", "compatibility"):
        component = components[role]
        fields = {"adapter_sha256", "adapter_config_sha256", "training_sources_fingerprint"} if role == "decision" else {"adapter_sha256", "adapter_config_sha256"}
        if not isinstance(component, dict) or set(component) != fields:
            raise RuntimeContractError("Adapter components must be objects")
        _adapter_digest(component.get("adapter_sha256"), role + " weights")
        _adapter_digest(component.get("adapter_config_sha256"), role + " configuration")
    if components["compatibility"]["adapter_sha256"] != OLD_ADAPTER_SHA256:
        raise RuntimeContractError("Compatibility component is not the frozen original R1")
    sources = _adapter_digest(model_spec.get("training_sources_fingerprint"), "Decision training sources")
    if components["decision"].get("training_sources_fingerprint") != sources:
        raise RuntimeContractError("Decision component and top-level training source fingerprints disagree")
    return {**components["decision"], "base_model": model_spec.get("base_model"), "training_sources_fingerprint": sources}


def dual_adapter_model_spec(base_model: dict, decision: dict, compatibility: dict, router: dict) -> dict:
    """Bind both weight components and router content, independently of cases."""
    spec = {"model_kind": "dual_adapter_system", "base_model": base_model, "adapter_sha256": None,
            "components": {"decision": dict(decision), "compatibility": dict(compatibility)},
            "training_sources_fingerprint": decision.get("training_sources_fingerprint"), "router_descriptor": dict(router)}
    model_candidate_component(spec)
    _verify_router_descriptor(spec["router_descriptor"])
    spec["model_fingerprint"] = fingerprint(spec)
    return spec


def _verify_router_descriptor(descriptor):
    keys = {"router_code_sha256", "routes_config_sha256", "schema_registry_sha256", "router_fingerprint"}
    if not isinstance(descriptor, dict) or set(descriptor) != keys:
        raise RuntimeContractError("Routed system requires code, rule and schema-registry fingerprints")
    for key in keys:
        _adapter_digest(descriptor[key], key)
    if fingerprint({key: value for key, value in descriptor.items() if key != "router_fingerprint"}) != descriptor["router_fingerprint"]:
        raise RuntimeContractError("Router descriptor fingerprint is invalid")


def verify_record_adapter_binding(record: dict, model_spec: dict, expected_route=None) -> None:
    """Verify actual adapter provenance against an independently derived route.

Single-adapter artifacts retain their existing flat SHA interpretation. A
routed record must identify its real active component and cannot use the
decision SHA to label a compatibility-generated response.
    """
    if model_spec.get("model_kind") != "dual_adapter_system":
        if record.get("adapter_sha256") != model_spec.get("adapter_sha256"):
            raise RuntimeContractError("Inference record adapter identity mismatch")
        if "active_adapter_sha256" in record and record["active_adapter_sha256"] != model_spec.get("adapter_sha256"):
            raise RuntimeContractError("Single-adapter active identity mismatch")
        return
    model_candidate_component(model_spec)
    descriptor = model_spec.get("router_descriptor")
    _verify_router_descriptor(descriptor)
    identity = fingerprint({key: value for key, value in model_spec.items() if key != "model_fingerprint"})
    if model_spec.get("model_fingerprint") != identity or record.get("model_fingerprint") != identity:
        raise RuntimeContractError("Routed record/model system fingerprint mismatch")
    keys = {"active_adapter_role", "public_contract_sha256", "router_fingerprint"}
    if not isinstance(expected_route, dict) or set(expected_route) != keys:
        raise RuntimeContractError("Routed response requires an independently verified public contract route")
    role = expected_route.get("active_adapter_role")
    if role not in {"decision", "compatibility"}:
        raise RuntimeContractError("Unknown routed adapter role")
    for key in ("public_contract_sha256", "router_fingerprint"):
        _adapter_digest(expected_route[key], key)
    if expected_route["router_fingerprint"] != descriptor["router_fingerprint"]:
        raise RuntimeContractError("Public contract route uses a different router definition")
    if any(record.get(key) != expected_route[key] for key in keys):
        raise RuntimeContractError("Record route differs from the verified public output contract")
    active_sha = model_spec["components"][role]["adapter_sha256"]
    if record.get("active_adapter_sha256") != active_sha or record.get("adapter_sha256") != active_sha:
        raise RuntimeContractError("Record does not identify its actual active adapter component")


@contextmanager
def exclusive_output_lock(path: Path):
    """OS-released per-rank lock: stale files after crashes do not block resume."""
    handle = Path(path).open("a+b")
    locked = False
    try:
        if sys.platform == "win32":
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeContractError("Another inference process owns this rank/output directory") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeContractError("Another inference process owns this rank/output directory") from error
        locked = True
        yield
    finally:
        if locked:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def tokenizer_fingerprint(path: Path) -> dict:
    result = {name: sha256_file(Path(path) / name) for name in TOKENIZER_FILES if (Path(path) / name).is_file()}
    if "tokenizer.json" not in result or not ("chat_template.jinja" in result or "tokenizer_config.json" in result):
        raise RuntimeContractError("Local tokenizer or chat-template metadata is missing")
    return result


def load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("CPU validation requires an existing transformers/tokenizers runtime; no installation is attempted") from error
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        template = Path(path) / "chat_template.jinja"
        if not template.is_file():
            raise RuntimeContractError("No existing chat template is available")
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def validate_messages(messages, *, training: bool) -> None:
    if not isinstance(messages, list) or not messages:
        raise RuntimeContractError("messages must be a nonempty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"} or not isinstance(message.get("content"), str):
            raise RuntimeContractError("Unsupported role/content in messages")
        if message["role"] == "assistant" and (not training or index != len(messages) - 1):
            raise RuntimeContractError("Only the final training message may contain an assistant target")
    if training:
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise RuntimeContractError("A nonempty final assistant target is required")
    elif messages[-1]["role"] not in {"user", "tool"}:
        raise RuntimeContractError("A public input must end in a user/tool request")


def action_value_character_span(text: str) -> tuple[int, int]:
    """Locate one top-level JSON action string using JSON parsing, not substring search."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeContractError("Duplicate JSON key in a supervised action target")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_pairs)
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError) as error:
        raise RuntimeContractError("Weighted action target must be one valid JSON object") from error
    if not isinstance(value, dict) or not isinstance(value.get("action"), str) or value["action"] not in ACTION_VALUES:
        raise RuntimeContractError("Weighted target needs one top-level action in the registered enum")
    cursor = 0
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        raise RuntimeContractError("Weighted target must start with a JSON object")
    cursor += 1
    found = []
    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if text[cursor] == "}":
            break
        key, cursor = decoder.raw_decode(text, cursor)
        while text[cursor].isspace():
            cursor += 1
        if text[cursor] != ":":
            raise RuntimeContractError("Malformed action JSON key separator")
        cursor += 1
        while text[cursor].isspace():
            cursor += 1
        start = cursor
        item, cursor = decoder.raw_decode(text, cursor)
        if key == "action":
            if not isinstance(item, str) or item not in ACTION_VALUES:
                raise RuntimeContractError("Top-level action is not a registered string")
            found.append((start + 1, cursor - 1))  # Content inside JSON quotes.
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if text[cursor] == ",":
            cursor += 1
        elif text[cursor] == "}":
            break
        else:
            raise RuntimeContractError("Malformed JSON target boundary")
    if len(found) != 1:
        raise RuntimeContractError("Exactly one top-level action is required")
    return found[0]


def encode_training_example(row: dict, tokenizer, max_length: int, action_loss_total_weight=None) -> dict:
    messages = row["messages"]
    validate_messages(messages, training=True)
    rendered = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    target = messages[-1]["content"]
    completion_text = target + "<|im_end|>\n"
    weighted = action_loss_total_weight is not None
    if weighted and (isinstance(action_loss_total_weight, bool) or not isinstance(action_loss_total_weight, (int, float)) or not math.isfinite(action_loss_total_weight) or action_loss_total_weight <= 0):
        raise RuntimeContractError("Action span total weight must be finite and positive")
    eligible = weighted and row.get("task") in ACTION_TASKS and row.get("origin") != "private_frozen_sft_replay"
    if eligible and not getattr(tokenizer, "is_fast", False):
        raise RuntimeContractError("Action span weighting requires real fast-tokenizer offsets")
    encoded_target = tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True) if eligible else tokenizer(completion_text, add_special_tokens=False)
    completion = encoded_target["input_ids"]
    if not completion or tokenizer.eos_token_id not in completion:
        raise RuntimeContractError("Completion lacks the chat end token")
    ids = list(prompt) + list(completion)
    if len(ids) > max_length:
        raise RuntimeContractError(f"{row.get('case_id', row.get('id'))}: {len(ids)} tokens exceeds {max_length}; truncation is forbidden")
    labels = [-100] * len(prompt) + list(completion)
    if not prompt or not any(value != -100 for value in labels):
        raise RuntimeContractError("Invalid assistant-only label mask")
    encoded = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    if weighted:
        weights = [0.0] * len(prompt) + [1.0] * len(completion)
        audit = {"weighted_action": False, "action_tokens": 0, "action_weight_sum": 0.0}
        if eligible:
            start, end = action_value_character_span(target)
            offsets = encoded_target["offset_mapping"]
            positions = [index for index, (left, right) in enumerate(offsets) if right > left and right > start and left < end]
            if not positions:
                raise RuntimeContractError("Fast-tokenizer offsets do not cover the top-level action value")
            covered = start
            for index in positions:
                left, right = offsets[index]
                if left > covered:
                    raise RuntimeContractError("Gap in action token-offset coverage")
                covered = max(covered, right)
            if covered < end:
                raise RuntimeContractError("Action token offsets do not cover the complete value")
            for index in positions:
                weights[len(prompt) + index] = float(action_loss_total_weight) / len(positions)
            audit = {"weighted_action": True, "action_tokens": len(positions),
                     "action_weight_sum": sum(weights[len(prompt) + index] for index in positions),
                     "action_character_span": [start, end], "completion_token_indices": positions}
        encoded["loss_weights"] = weights
        encoded["_action_loss_audit"] = audit
    return encoded


class WeightedAssistantCollator:
    """Preserve the existing HF padding, adding only zero-padded float32 loss weights."""
    def __init__(self, base_collator, tokenizer):
        self.base_collator, self.tokenizer = base_collator, tokenizer

    def __call__(self, features):
        import torch
        ordinary = [{key: value for key, value in feature.items() if key in {"input_ids", "attention_mask", "labels"}} for feature in features]
        batch = self.base_collator(ordinary)
        width = batch["input_ids"].shape[1]
        padded = []
        for feature in features:
            weights = list(feature["loss_weights"])
            if len(weights) != len(feature["input_ids"]):
                raise RuntimeContractError("Loss weights must align with every input token")
            zeros = [0.0] * (width - len(weights))
            padded.append(zeros + weights if self.tokenizer.padding_side == "left" else weights + zeros)
        batch["loss_weights"] = torch.tensor(padded, dtype=torch.float32)
        if bool((batch["loss_weights"][batch["labels"] == -100] != 0).any()):
            raise RuntimeContractError("Prompt and padding labels must have zero loss weight")
        return batch


def weighted_causal_cross_entropy(logits, labels, loss_weights):
    import torch
    from torch.nn import functional as F
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or loss_weights.shape != labels.shape:
        raise RuntimeContractError("Weighted causal loss tensor shapes do not align")
    if not bool(torch.isfinite(loss_weights).all()) or bool((loss_weights < 0).any()):
        raise RuntimeContractError("Loss weights must be finite and nonnegative")
    targets = labels[:, 1:].contiguous()
    weights = loss_weights[:, 1:].float() * (targets != -100)
    denominator = weights.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator.detach()) <= 0:
        raise RuntimeContractError("Weighted causal loss has no valid supervised target")
    losses = F.cross_entropy(logits[:, :-1, :].contiguous().float().view(-1, logits.shape[-1]),
                            targets.view(-1), ignore_index=-100, reduction="none").view_as(targets)
    return (losses * weights).sum() / denominator


def model_training_loss(model, inputs):
    if "loss_weights" not in inputs:
        outputs = model(**inputs)
        return outputs.loss, outputs
    model_inputs = {key: value for key, value in inputs.items() if key not in {"labels", "loss_weights", "_action_loss_audit"}}
    outputs = model(**model_inputs)
    return weighted_causal_cross_entropy(outputs.logits, inputs["labels"], inputs["loss_weights"]), outputs


def validate_public_cases(rows: list[dict]) -> None:
    ids = []
    for row in rows:
        identifier = row.get("case_id", row.get("id"))
        if not isinstance(identifier, str) or not identifier:
            raise RuntimeContractError("Each public case needs a case_id")
        ids.append(identifier)
        if {"expected", "gold", "answer", "reference_output", "raw_output", "output"} & row.keys():
            raise RuntimeContractError("Public inference input contains an answer/gold/output field")
        validate_messages(row["messages"], training=False)
    if len(ids) != len(set(ids)):
        raise RuntimeContractError("Duplicate public case IDs")


def validate_init_adapter(path: Path, config: dict) -> dict:
    metadata = read_json(Path(path) / "adapter_config.json")
    weights = Path(path) / "adapter_model.safetensors"
    actual = sha256_file(weights)
    expected = config.get("init_adapter_sha256", OLD_ADAPTER_SHA256)
    if actual != expected or expected != OLD_ADAPTER_SHA256:
        raise RuntimeContractError("Initialization is not the frozen old R1 adapter")
    if metadata.get("revision") != config["base_revision"] or config["base_revision"] != BASE_REVISION:
        raise RuntimeContractError("Base revision and old adapter revision do not match")
    for field, key in (("r", "lora_r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        if metadata[field] != config[key]:
            raise RuntimeContractError(f"Continued adapter configuration mismatch: {field}")
    if set(metadata["target_modules"]) != set(config["target_modules"]):
        raise RuntimeContractError("Continued adapter target modules differ")
    return {"adapter_model_sha256": actual, "adapter_config_sha256": sha256_file(Path(path) / "adapter_config.json"), "revision": metadata["revision"]}


def base_model_fingerprint(path: Path, expected_weights: dict) -> dict:
    path = Path(path)
    metadata = read_json(path / "config.json")
    if metadata.get("model_type") != "qwen3":
        raise RuntimeContractError("Expected a local Qwen3 base model")
    hashes = {}
    for name, expected in sorted(expected_weights.items()):
        actual = sha256_file(path / name)
        if actual != expected:
            raise RuntimeContractError(f"Frozen base-model weight mismatch: {name}")
        hashes[name] = actual
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        if (path / name).is_file():
            hashes[name] = sha256_file(path / name)
    return {"model_id": "Qwen/Qwen3-8B", "revision": BASE_REVISION, "files_sha256": hashes, "fingerprint": fingerprint(hashes)}


def validate_data_manifest(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = read_json(manifest_path)
    artifacts = manifest.get("artifact_hashes")
    if not isinstance(artifacts, dict):
        raise RuntimeContractError("Data manifest must contain artifact_hashes")
    checked = {}
    try:
        relative_directory = data_dir.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        relative_directory = None
    for name in ("train.jsonl", "dev.jsonl"):
        allowed = {name}
        if relative_directory is not None:
            allowed.add(relative_directory + "/" + name)
        choices = [key for key in artifacts if key in allowed]
        if len(choices) != 1:
            raise RuntimeContractError(f"Training split lacks an unambiguous frozen hash: {name}")
        expected = artifacts[choices[0]]
        if isinstance(expected, dict):
            expected = expected["sha256"]
        actual = sha256_file(data_dir / name)
        if actual != expected:
            raise RuntimeContractError(f"Frozen training split changed: {name}")
        checked[name] = actual
    return {"manifest_sha256": sha256_file(manifest_path), "files_sha256": checked, "manifest": manifest}


def prepare_datasets(data_dir: Path, tokenizer, max_length: int, action_loss_total_weight=None) -> tuple[dict, dict]:
    binding = validate_data_manifest(data_dir)
    datasets, statistics = {}, {}
    group_sets = {}
    id_sets = {}
    for split, filename in (("train", "train.jsonl"), ("dev", "dev.jsonl")):
        raw = read_jsonl(Path(data_dir) / filename)
        ids = [row.get("case_id", row.get("id")) for row in raw]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise RuntimeContractError(f"Invalid/duplicate training IDs in {split}")
        if any(not row.get("group_id") for row in raw):
            raise RuntimeContractError(f"Every {split} row requires an independent grouping key")
        group_sets[split] = {row["group_id"] for row in raw}
        id_sets[split] = set(ids)
        encoded = [encode_training_example(row, tokenizer, max_length, action_loss_total_weight) for row in raw]
        lengths = [len(row["input_ids"]) for row in encoded]
        supervised = [sum(value != -100 for value in row["labels"]) for row in encoded]
        datasets[split] = encoded
        statistics[split] = {
            "rows": len(raw), "min_tokens": min(lengths), "max_tokens": max(lengths),
            "total_tokens": sum(lengths), "assistant_supervised_tokens": sum(supervised),
            "prompt_masked_tokens": sum(lengths) - sum(supervised),
            "truncated_rows": 0, "zero_supervision_rows": 0, "groups": len(group_sets[split]),
        }
        if action_loss_total_weight is not None:
            audits = [row["_action_loss_audit"] for row in encoded if row["_action_loss_audit"]["weighted_action"]]
            statistics[split]["action_weighting"] = {
                "rows_with_weighted_action": len(audits), "ordinary_rows": len(encoded) - len(audits),
                "fixed_total_weight": action_loss_total_weight,
                "action_tokens_histogram": {str(count): sum(audit["action_tokens"] == count for audit in audits) for count in sorted({audit["action_tokens"] for audit in audits})},
                "action_weight_sum_min": min((audit["action_weight_sum"] for audit in audits), default=None),
                "action_weight_sum_max": max((audit["action_weight_sum"] for audit in audits), default=None),
                "ordinary_assistant_weights_all_one": all(
                    all(weight == 1 for weight, label in zip(row["loss_weights"], row["labels"]) if label != -100)
                    for row in encoded if not row["_action_loss_audit"]["weighted_action"]),
            }
    if group_sets["train"] & group_sets["dev"] or id_sets["train"] & id_sets["dev"]:
        raise RuntimeContractError("Training and dev IDs/groups overlap")
    return datasets, {"binding": {key: value for key, value in binding.items() if key != "manifest"}, "splits": statistics}


def adapter_tensor_integrity(path: Path, *, old_adapter: Path | None = None, require_changed=True) -> dict:
    weight = Path(path) / "adapter_model.safetensors"
    if not weight.is_file():
        raise RuntimeContractError("No completed candidate adapter weights exist; GPU training is pending or failed")
    from safetensors import safe_open
    import torch
    actual_hash = sha256_file(weight)
    old_weight = Path(old_adapter) / "adapter_model.safetensors" if old_adapter is not None else None
    if old_weight is not None and require_changed and actual_hash == sha256_file(old_weight):
        raise RuntimeContractError("Candidate is the unchanged old adapter")
    changed, count, elements = 0, 0, 0
    with safe_open(str(weight), framework="pt", device="cpu") as candidate:
        keys = list(candidate.keys())
        if not keys or not all("lora_" in key for key in keys):
            raise RuntimeContractError("Candidate must contain LoRA adapter tensors")
        old = safe_open(str(old_weight), framework="pt", device="cpu") if old_weight is not None else None
        if old is not None:
            old.__enter__()
        try:
            if old is not None and set(old.keys()) != set(keys):
                raise RuntimeContractError("Candidate adapter tensor keys differ from initialization")
            for key in keys:
                tensor = candidate.get_tensor(key)
                if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                    raise RuntimeContractError(f"Non-finite/non-floating candidate tensor: {key}")
                count += 1
                elements += tensor.numel()
                if old is not None:
                    reference = old.get_tensor(key)
                    if reference.shape != tensor.shape:
                        raise RuntimeContractError("Candidate adapter tensor shapes changed")
                    changed += not torch.equal(reference, tensor)
        finally:
            if old is not None:
                old.__exit__(None, None, None)
    if require_changed and old_weight is not None and changed == 0:
        raise RuntimeContractError("Candidate serialization changed but no adapter tensor changed")
    return {"sha256": actual_hash, "tensor_count": count, "elements": elements, "all_finite": True, "changed_tensors": changed if old_weight is not None else None}


def accept_completed_record(record: dict, expected_run: str, case_hash: str, model_fingerprint: str) -> None:
    if record.get("status") != "completed" or record.get("inference_backend") != "transformers_real_model":
        raise RuntimeContractError("Resume requires a completed actual-model generation")
    if record.get("run_fingerprint") != expected_run or record.get("case_sha256") != case_hash or record.get("model_fingerprint") != model_fingerprint:
        raise RuntimeContractError("Resume record fingerprint mismatch")
    if not isinstance(record.get("raw_output"), str) or not isinstance(record.get("generated_tokens"), int) or record["generated_tokens"] < 1:
        raise RuntimeContractError("Resume record lacks a generated answer/token count")


def check_promotion_metrics(candidate: dict, old: dict, policy: dict, *, base: dict | None = None) -> dict:
    """Apply the explicitly selected policy; existing v1 behavior stays intact.

V2 compares strict success with max(base, old), retains zero unsafe acceptance,
and conditions utility/error checks on the same gold-run cases. Counts, not
rounded display rates, determine the two gates. Zero attempts are unassessable
and cannot be promoted. This function neither trains nor changes encoding.
    """
    if policy.get("schema") == "maven_final.promotion.v2":
        if (policy.get("primary_comparator") != "max_base_old"
                or policy.get("require_gold_run_coverage_no_worse_than_base") is not True
                or policy.get("require_conditional_run_error_no_worse_than_base") is not True
                or policy.get("require_positive_primary_group_ci") is not True):
            raise RuntimeContractError("V2 requires max(base,old), base execution coverage, conditional-error gates and positive group CI")
        minimum = policy.get("minimum_absolute_improvement")
        if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0.03:
            raise RuntimeContractError("V2 minimum strict-success improvement cannot be below +3 percentage points")
        if base is None:
            raise RuntimeContractError("V2 promotion requires the actual base-model evaluation")
        metric = policy["primary_metric"]
        try:
            rates = {name: view["metrics"][metric] for name, view in (("candidate", candidate), ("old", old), ("base", base))}
        except (KeyError, TypeError) as error:
            raise RuntimeContractError("V2 requires primary metrics for candidate, old and base") from error
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in rates.values()):
            raise RuntimeContractError("Invalid real benchmark primary metric")
        reference = max(rates["base"], rates["old"])
        improvement = rates["candidate"] - reference
        if improvement + 1e-12 < minimum:
            raise RuntimeContractError("Candidate does not exceed max(base,old) by the declared strict-success margin")
        dangerous = candidate.get("dangerous_acceptance_count")
        if type(dangerous) is not int or dangerous != 0:
            raise RuntimeContractError("Dangerous error acceptance must be explicitly counted as zero")
        recorded_dangerous = candidate["metrics"].get("dangerous_acceptance_count", dangerous)
        if type(recorded_dangerous) is not int or recorded_dangerous != dangerous:
            raise RuntimeContractError("Candidate dangerous-acceptance counts disagree")

        def count_view(view, label):
            names = ("gold_run_count", "gold_run_strict_success_count", "gold_run_attempt_count", "gold_run_contract_error_count")
            values = {name: view["metrics"].get(name) for name in names}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise RuntimeContractError(f"V2 {label} requires nonnegative integer gold-run counts")
            total, success, attempts, errors = (values[name] for name in names)
            if total == 0 or attempts == 0:
                raise RuntimeContractError(f"V2 {label} has a zero gold-run/attempt denominator; no-execution is not zero risk")
            if not 0 <= success <= attempts <= total or errors != attempts - success:
                raise RuntimeContractError(f"V2 {label} gold-run counts are inconsistent")
            return {**values, "gold_run_strict_execution_coverage": success / total,
                    "conditional_run_contract_error_rate": errors / attempts}

        current, baseline = count_view(candidate, "candidate"), count_view(base, "base")
        if current["gold_run_count"] != baseline["gold_run_count"]:
            raise RuntimeContractError("V2 candidate/base must describe the same gold-run case denominator")
        if (current["gold_run_strict_success_count"] * baseline["gold_run_count"]
                < baseline["gold_run_strict_success_count"] * current["gold_run_count"]):
            raise RuntimeContractError("Candidate gold-run strict execution coverage is below base")
        if (current["gold_run_contract_error_count"] * baseline["gold_run_attempt_count"]
                > baseline["gold_run_contract_error_count"] * current["gold_run_attempt_count"]):
            raise RuntimeContractError("Candidate conditional gold-run contract error rate exceeds base")
        return {"policy_schema": "maven_final.promotion.v2", "primary_metric": metric,
                "primary_comparator": "max_base_old", "candidate": rates["candidate"], "old": rates["old"],
                "base": rates["base"], "reference": reference,
                "reference_models": [name for name in ("base", "old") if rates[name] == reference],
                "absolute_improvement": improvement, "dangerous_acceptance_count": dangerous,
                "gold_run_candidate": current, "gold_run_base": baseline,
                "raw_critical_error_counts": {name: view.get("critical_error_count") for name, view in (("candidate", candidate), ("base", base), ("old", old))},
                "raw_critical_count_used_as_old_zero_attempt_threshold": False}

    metric = policy["primary_metric"]
    values = [candidate["metrics"][metric], old["metrics"][metric]]
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise RuntimeContractError("Invalid real benchmark primary metric")
    improvement = values[0] - values[1]
    if improvement + 1e-12 < policy["minimum_absolute_improvement"]:
        raise RuntimeContractError("Candidate does not meet the predeclared semantic-success improvement")
    dangerous = candidate.get("dangerous_acceptance_count")
    if not isinstance(dangerous, int) or isinstance(dangerous, bool) or dangerous != 0:
        raise RuntimeContractError("Dangerous error acceptance must be explicitly counted as zero")
    if policy["require_no_increase_in_critical_errors"]:
        if any(not isinstance(value.get("critical_error_count"), int) or isinstance(value.get("critical_error_count"), bool) or value["critical_error_count"] < 0 for value in (candidate, old)):
            raise RuntimeContractError("Critical error counts are required for promotion")
        if candidate["critical_error_count"] > old["critical_error_count"]:
            raise RuntimeContractError("Candidate increases critical errors")
    return {"primary_metric": metric, "candidate": values[0], "old": values[1], "absolute_improvement": improvement, "dangerous_acceptance_count": dangerous}


def check_primary_group_ci(comparison: dict, policy: dict) -> dict:
    """Check already computed, paired group CIs without recomputing statistics.

The scorer's algorithm and data stay unchanged. Normalize left/right signs to
candidate-minus-comparator, require both base and old comparisons exactly once,
and demand a strictly positive 95% lower bound. V1 remains report-only.
    """
    if policy.get("require_positive_primary_group_ci") is not True:
        return {"status": "not_required", "required": False}
    metric = "semantic_execution_success_rate"
    if policy.get("primary_metric") != metric or not isinstance(comparison, dict) or comparison.get("suite") != "final6":
        raise RuntimeContractError("Positive primary group CI requires suite final6 and strict semantic-execution success")
    pairs = comparison.get("paired_comparisons")
    if not isinstance(pairs, list) or any(not isinstance(row, dict) for row in pairs):
        raise RuntimeContractError("Primary group comparisons must be recorded as paired comparison objects")
    canonical = {}
    for reference in ("base", "old"):
        matches = [row for row in pairs if isinstance(row.get("left"), str) and isinstance(row.get("right"), str)
                   and {row["left"], row["right"]} == {"candidate", reference}]
        if len(matches) != 1:
            raise RuntimeContractError(f"Primary group CI needs exactly one candidate/{reference} record")
        row = matches[0]
        if row.get("metric") != metric:
            raise RuntimeContractError(f"Candidate/{reference} CI uses a different metric")
        interval, point = row.get("conditional_group_bootstrap_ci95_pp"), row.get("difference_pp")
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(value) not in (int, float) or not -100 <= value <= 100 or not math.isfinite(value)
                       for value in [point, *interval]) or interval[0] > interval[1]):
            raise RuntimeContractError("Group CI must be a finite ordered percentage-point interval")
        if row["left"] == "candidate":
            lower, upper, difference = interval[0], interval[1], point
        else:
            lower, upper, difference = -interval[1], -interval[0], -point
        if lower <= 0:
            raise RuntimeContractError(f"Candidate-minus-{reference} strict-success group CI lower bound must be strictly positive")
        canonical[reference] = {"difference_pp": difference, "conditional_group_bootstrap_ci95_pp": [lower, upper],
                                "source_left": row["left"], "source_right": row["right"]}
    return {"status": "passed_positive_primary_group_ci", "required": True, "suite": "final6",
            "metric": metric, "candidate_minus_reference": canonical,
            "scope": "Existing paired fact-group bootstrap intervals; not biological or population-wide efficacy."}
