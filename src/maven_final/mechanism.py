"""Preserve the original scientific ranking while binding its actual source bytes."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "maven.final.assets.v1" or not manifest.get("files"):
        raise ValueError("Invalid frozen asset manifest")
    for relative, record in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Missing or escaped frozen asset: {relative}")
        if sha256(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise ValueError(f"Frozen asset changed: {relative}")
    return manifest


def load_pool():
    verify_assets()
    recovery = importlib.import_module("legacy_maven.recovery")
    blind = importlib.import_module("legacy_maven.blind")
    for module in (recovery, blind):
        if not Path(module.__file__).resolve().is_relative_to(ROOT / "src/legacy_maven"):
            raise ValueError("Wrong mechanism source root loaded")
    catalog = json.loads((ROOT / "assets/library_catalog.json").read_text(encoding="utf-8"))
    pool = blind.blind_library(recovery.generate_search_universe(catalog))
    if len(pool) != 4800 or len({r["sequence"] for r in pool}) != 1200:
        raise ValueError("Preserved candidate universe changed")
    return pool


def rank_pool(seed=2101):
    pool = load_pool()
    core = importlib.import_module("legacy_maven.mavenpep21")
    if not Path(core.__file__).resolve().is_relative_to(ROOT / "src/legacy_maven"):
        raise ValueError("Wrong scoring source loaded")
    ranked = core.rank_mavenpep21(pool, "mavenpep21_strict", seed)
    result = []
    for rank, row in enumerate(ranked, 1):
        if not all(math.isfinite(row[k]) for k in ("score_mean", "score_std", "score_lcb")):
            raise ValueError("Nonfinite mechanism score")
        result.append({"rank": rank, "candidate_id": row["candidate_id"], "sequence": row["sequence"],
                       "n_chemistry": row["n_chemistry"], "c_chemistry": row["c_chemistry"],
                       "orientation": row["orientation"], "mu": row["score_mean"], "sigma": row["score_std"],
                       "LCB": row["score_lcb"], "score_scope": "researcher_prior_priority_not_efficacy_probability"})
    return result


def validate_snapshot(pool):
    verify_assets()
    snapshot = json.loads((ROOT / "assets/expert_snapshot.json").read_text(encoding="utf-8"))
    sequences = {r["sequence"] for r in pool}
    if set(snapshot) != sequences:
        raise ValueError("Expert snapshot does not exactly cover the preserved sequence pool")
    types = {"esm": "protein_language_model_representation", "properties": "peptide_physicochemical_properties",
             "interaction": "protein_peptide_interaction", "structure": "structure_evidence", "md": "md_evidence"}
    counts = {k: 0 for k in types}
    for sequence in sequences:
        for tool, evidence_type in types.items():
            records = [r for r in snapshot[sequence] if r.get("evidence_type") == evidence_type]
            available = [r for r in records if r.get("status") == "available"]
            if tool in {"esm", "properties"} and len(available) != 1:
                raise ValueError(f"Required {tool} evidence is not uniquely available")
            if tool == "esm":
                record = available[0]["value"]
                embedding = record.get("embedding")
                if record.get("embedding_dim") != 640 or not isinstance(embedding, list) or len(embedding) != 640:
                    raise ValueError("Wrong ESM snapshot dimension")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in embedding):
                    raise ValueError("Invalid ESM vector")
            counts[tool] += bool(available)
    return {"mode": "frozen_real_expert_snapshot_reuse", "unique_sequences": len(sequences), "available_sequences": counts,
            "expert_snapshot_sha256": sha256(ROOT / "assets/expert_snapshot.json"),
            "fresh_expert_inference": False, "expert_values_used_in_primary_ranking": False,
            "optional_missing": [k for k in ("interaction", "structure", "md") if counts[k] == 0]}


def apply_constraints(ranked, constraints):
    allowed = {"min_length", "max_length", "allowed_n_terminal", "allowed_c_terminal"}
    if set(constraints) - allowed:
        raise ValueError("Unknown candidate constraint")
    low, high = constraints.get("min_length", 1), constraints.get("max_length", 1000)
    if type(low) is not int or type(high) is not int or not 1 <= low <= high:
        raise ValueError("Invalid length constraints")
    for key, valid in [("allowed_n_terminal", {"free", "acetyl"}), ("allowed_c_terminal", {"free", "amide"})]:
        values = constraints.get(key)
        if values is not None and (not isinstance(values, list) or not values or any(v not in valid for v in values)):
            raise ValueError("Invalid terminal constraints")
    selected = [r.copy() for r in ranked if low <= len(r["sequence"]) <= high
                and r["n_chemistry"] in constraints.get("allowed_n_terminal", ["free", "acetyl"])
                and r["c_chemistry"] in constraints.get("allowed_c_terminal", ["free", "amide"])]
    if len(selected) < 10:
        raise ValueError("Fewer than ten eligible constructs; cannot fabricate Top-10")
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected
