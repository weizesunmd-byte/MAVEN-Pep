"""Evidence-grounded decisions followed by an immutable, executable mechanism.

The model chooses a validated action and plan. It cannot edit the scoring
engine, numerical outputs, source registry or historical references. Sessions
are published atomically only after the complete candidate artifacts verify.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from .contracts import reference_answer, score_response, task_instruction
from .mechanism import ROOT, apply_constraints, load_pool, rank_pool, sha256, validate_snapshot, verify_assets
from .identity_policy import assert_runtime_blinded

NUMERIC_FIELDS = ("mu", "sigma", "LCB")
IDENTITY_FIELDS = ("rank", "candidate_id", "sequence", "n_chemistry", "c_chemistry", "orientation")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish an empty candidate table")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_request(disease="C5a-mediated immunosuppression", target="C5a", constraints=None):
    return {"input_origin": "existing_tool_result", "disease": disease, "target": target,
            "objective": "prioritize modular candidates under the preserved C5a mechanism",
            "available_database_tools": ["protein_sequence", "target_information"],
            "available_expert_tools": ["esm", "properties", "interaction", "structure", "md"],
            "constraints": constraints or {}}


def public_workflow_case(payload, case_id="workflow-request-001"):
    assert_runtime_blinded(payload)
    return {"case_id": case_id, "task": "plan_discovery", "group_id": "user_workflow_request",
            "origin": "existing_tool_result", "input": payload,
            "messages": [{"role": "system", "content": task_instruction("plan_discovery")},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}


def verified_model_record(case, record, manifest, *, prediction_file):
    """Bind a completed model response to the exact public request and run."""
    from .model_runtime import fingerprint, accept_completed_record, read_jsonl, verify_record_adapter_binding
    if manifest.get("status") != "completed" or manifest.get("source_type") != "real_model":
        raise ValueError("Workflow requires a completed actual-model run")
    if sha256(prediction_file) != manifest.get("predictions_sha256"):
        raise ValueError("Model prediction file changed after inference")
    if record not in read_jsonl(Path(prediction_file)):
        raise ValueError("Response record is not in the completed prediction file")
    name, case_id = record.get("model"), case.get("case_id", case.get("id"))
    if name not in manifest["models"] or case_id not in manifest["case_ids"] or record.get("case_id") != case_id:
        raise ValueError("Model response is not for this workflow request")
    if fingerprint(case) != manifest["case_hashes"][case_id]:
        raise ValueError("Public workflow request changed after model inference")
    accept_completed_record(record, manifest["run_fingerprint"], fingerprint(case), manifest["models"][name]["model_fingerprint"])
    if record.get("settings_fingerprint") != manifest["settings_fingerprint"] or record.get("input_sha256") != manifest["input_sha256"]:
        raise ValueError("Model response settings/input binding differs")
    model_spec = manifest["models"][name]
    expected_route = None
    if model_spec.get("model_kind") == "dual_adapter_system":
        from .adapter_router import route_public_contract
        expected_route = route_public_contract(case["messages"])
        if expected_route != manifest.get("case_routes", {}).get(case_id):
            raise ValueError("Workflow public contract differs from the recorded adapter route")
        if expected_route["active_adapter_role"] != "decision":
            raise ValueError("Modern discovery must execute the trained decision adapter")
    verify_record_adapter_binding(record, model_spec, expected_route)
    return record["raw_output"], {"mode": "saved_real_model_response", "model": name,
                                   "model_fingerprint": record["model_fingerprint"],
                                   "model_kind": model_spec.get("model_kind", "single_adapter"),
                                   "adapter_sha256": record.get("adapter_sha256"),
                                   "active_adapter_role": record.get("active_adapter_role"),
                                   "active_adapter_sha256": record.get("active_adapter_sha256"),
                                   "public_contract_sha256": record.get("public_contract_sha256"),
                                   "router_fingerprint": record.get("router_fingerprint"),
                                   "run_fingerprint": manifest["run_fingerprint"],
                                   "settings_fingerprint": record["settings_fingerprint"]}


class HTTPModelClient:
    def __init__(self, endpoint, model, max_tokens=512):
        self.endpoint, self.model, self.max_tokens = endpoint.rstrip("/"), model, max_tokens

    def respond(self, task, payload):
        with urlopen(self.endpoint + "/health", timeout=30) as response:
            health = json.load(response)
        if health.get("ready") is not True or health.get("adapter_loaded") is not True:
            raise ValueError("Requested model service has no ready domain adapter")
        with urlopen(self.endpoint + "/v1/models", timeout=30) as response:
            models = json.load(response)
        if self.model not in {row.get("id") for row in models.get("data", [])}:
            raise ValueError("Requested model is absent from the model service")
        request = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": task_instruction(task)},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        with urlopen(Request(self.endpoint + "/v1/chat/completions", data=json.dumps(request).encode("utf-8"),
                             headers={"Content-Type": "application/json"}), timeout=180) as response:
            result = json.load(response)
        if result.get("model") != self.model:
            raise ValueError("Model service returned a different model")
        return result["choices"][0]["message"]["content"], {
            "mode": "live_model_response", "model": result.get("model", self.model),
            "usage": result.get("usage", {}), "finish_reason": result["choices"][0].get("finish_reason")}


def match_references(top10, registry):
    """Called after freezing; unmatched candidates remain explicitly unknown."""
    matches = []
    for row in top10:
        aliases = []
        for record in registry:
            orientation = {"helper-linker-b_cell": "helper_to_b", "b_cell-linker-helper": "b_to_helper"}.get(record["orientation"])
            if (record["exact_full_sequence"], record["n_terminal_state"], record["c_terminal_state"], orientation) == (
                row["sequence"], row["n_chemistry"], row["c_chemistry"], row["orientation"]
            ):
                aliases.append(record["historical_alias"] or record["construct_id"])
        matches.append({**row, "reference_identity": ";".join(aliases),
                        "reference_match_status": "reference_match" if aliases else "unknown",
                        "wetlab_identity_status": "not_established_by_reference_match"})
    return matches


def compare_ranking(before, after):
    differences = []
    if len(before) != len(after):
        differences.append({"field": "row_count", "before": len(before), "after": len(after)})
    for position, (left, right) in enumerate(zip(before, after), 1):
        for field in IDENTITY_FIELDS:
            if str(left[field]) != str(right[field]):
                differences.append({"row": position, "field": field})
        for field in NUMERIC_FIELDS:
            value = float(right[field])
            if not math.isfinite(value) or value != float(left[field]):
                differences.append({"row": position, "field": field})
    return {"ranking_changed": bool(differences), "comparisons": min(len(before), len(after)) * (len(IDENTITY_FIELDS) + len(NUMERIC_FIELDS)),
            "differences": differences, "scope": "actual file comparison; not historical blinding or signed timestamp"}


def verify_session(directory):
    directory = Path(directory)
    verify_assets()
    state = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if state["status"] != "completed_computational_selection":
        if state["status"] not in {"stopped", "needs_information"}:
            raise ValueError("Unknown session state")
        return state
    receipt = json.loads((directory / "freeze_receipt.json").read_text(encoding="utf-8"))
    if set(receipt.get("artifact_sha256", {})) != {"top10_blinded.json", "top10_blinded.csv", "all_scores.csv"}:
        raise ValueError("Incomplete frozen artifact hashes")
    for relative, expected in receipt["artifact_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError("Frozen candidate artifact changed")
    top10 = json.loads((directory / "top10_blinded.json").read_text(encoding="utf-8"))
    after = read_csv(directory / "top10_reference_matches.csv")
    comparison = compare_ranking(top10, after)
    if comparison["ranking_changed"]:
        raise ValueError("Post-freeze candidate ranking changed")
    if [int(r["rank"]) for r in top10] != list(range(1, 11)) or len({r["candidate_id"] for r in top10}) != 10:
        raise ValueError("Invalid frozen Top-10")
    expected_matches = match_references(top10, read_csv(ROOT / "assets/reference_registry.csv"))
    for expected, actual in zip(expected_matches, after):
        for field in ("reference_identity", "reference_match_status", "wetlab_identity_status"):
            if expected[field] != actual.get(field):
                raise ValueError("Post-freeze reference annotation is inconsistent")
    return state


def run_discovery(payload, output, *, model_response=None, client=None, reference_policy=False, model_metadata=None):
    """Use a real model, explicit saved response, or an explicitly labelled policy.

reference_policy is an executor smoke mode, not a model capability result.
There is no implicit model fallback and no reference answer in live requests.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Session output already exists")
    if sum((model_response is not None, client is not None, reference_policy)) != 1:
        raise ValueError("Choose exactly one explicit planner source")
    assert_runtime_blinded(payload)
    manifest = verify_assets()
    if client is not None:
        response, metadata = client.respond("plan_discovery", payload)
    elif reference_policy:
        response = json.dumps(reference_answer("plan_discovery", payload), ensure_ascii=False)
        metadata = {"mode": "deterministic_reference_policy", "model_inference": False}
    else:
        response = model_response if isinstance(model_response, str) else json.dumps(model_response, ensure_ascii=False)
        metadata = {"mode": "provided_response", **(model_metadata or {})}
    evaluation = score_response("plan_discovery", payload, response)
    if not evaluation.get("semantic_success"):
        raise ValueError("Model plan violates executable task contracts: " + ",".join(evaluation.get("error_codes", [])))
    decision = evaluation.get("parsed_response")
    if decision is None:
        from .contracts import validate_response
        decision = validate_response("plan_discovery", response)["parsed_response"]
    assert_runtime_blinded(decision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".maven-final-", dir=output.parent) as temporary:
        staging = Path(temporary)
        write_json(staging / "request.json", payload)
        write_json(staging / "decision.json", {"response": decision, "model": metadata, "validation": evaluation})
        session = {"schema": "maven.final.session.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "model": metadata, "experiments_performed": False, "foundation_model_retrained": False,
                   "primary_scoring_changed": False, "source_manifest_sha256": sha256(ROOT / "assets/asset_manifest.json")}
        if decision["action"] != "run":
            session["status"] = "needs_information" if decision["action"] == "request_information" else "stopped"
            session["reason_codes"] = decision["error_codes"]
        else:
            pool = load_pool()
            coverage = validate_snapshot(pool)
            target = json.loads((ROOT / "assets/target_snapshot.json").read_text(encoding="utf-8"))
            if not any(r.get("evidence_type") == "protein_sequence" and r.get("status") == "available"
                       and "KDMQLGR" in str(r.get("value", "")) for r in target):
                raise ValueError("No verified target sequence in the bound snapshot")
            ranked = apply_constraints(rank_pool(), payload.get("constraints", {}))
            write_json(staging / "target_evidence.json", target)
            write_json(staging / "expert_coverage.json", coverage)
            write_csv(staging / "all_scores.csv", ranked)
            top10 = ranked[:10]
            write_json(staging / "top10_blinded.json", top10)
            write_csv(staging / "top10_blinded.csv", top10)
            write_json(staging / "freeze_receipt.json", {
                "schema": "maven.final.freeze.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_sha256": {name: sha256(staging / name) for name in ("top10_blinded.json", "top10_blinded.csv", "all_scores.csv")},
                "source_binding": manifest["schema"], "scope": "local completeness and integrity; no third-party timestamp"})
            registry = read_csv(ROOT / "assets/reference_registry.csv")
            matched = match_references(top10, registry)
            write_csv(staging / "top10_reference_matches.csv", matched)
            comparison = compare_ranking(top10, read_csv(staging / "top10_reference_matches.csv"))
            if comparison["ranking_changed"]:
                raise ValueError("Reference annotation changed the frozen ranking")
            write_json(staging / "ranking_comparison.json", comparison)
            session.update(status="completed_computational_selection", constructs=len(pool), eligible_constructs=len(ranked),
                           unique_sequences=1200, optional_evidence_missing=coverage["optional_missing"],
                           expert_inference_mode=coverage["mode"], ranking_changed=comparison["ranking_changed"],
                           reference_matches=sum(r["reference_match_status"] == "reference_match" for r in matched),
                           unknown_references=sum(r["reference_match_status"] == "unknown" for r in matched))
        write_json(staging / "session.json", session)
        verify_session(staging)
        verify_assets()
        if output.exists():
            raise FileExistsError("Session output appeared while executing")
        os.rename(staging, output)
    return session
