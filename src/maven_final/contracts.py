"""Six deterministic software tasks shared by SFT and executable benchmarks.

All numerical payloads must declare ``synthetic_contract`` or
``existing_tool_result``.  Reference answers execute transport, provenance,
identity, arithmetic and scope rules only; they are not biological efficacy
labels.  Explicit violations stop before missing facts request information.
No model, network, GPU or original project module is invoked here.
"""

from __future__ import annotations

import copy
from decimal import Decimal
import json
import math
import re
import unicodedata

TASKS = ("plan_discovery", "resolve_measurement", "identity_gate", "claim_scope", "interpret_ranking", "feedback_gate")
ACTIONS = ("run", "request_information", "stop")
INPUT_ORIGINS = ("synthetic_contract", "existing_tool_result")
DATABASE_TOOLS = ("disease_targets", "target_information", "protein_sequence", "known_peptides", "target_structure")
EXPERT_TOOLS = ("esm", "properties", "interaction", "structure", "md")
REQUIRED_DATABASE_TOOLS = frozenset(("target_information", "protein_sequence"))
REQUIRED_EXPERT_TOOLS = frozenset(("esm", "properties"))
CLAIM_LEVELS = ("mhc_prediction", "binding_prediction", "native_recognition", "neutralization", "mouse_endpoint", "human_clinical_endpoint")
ABS_TOLERANCE = 1e-8
REL_TOLERANCE = 1e-6
N_STATES = ("free", "acetyl")
C_STATES = ("free", "amide")
ORIENTATIONS = ("helper_to_b", "b_to_helper")
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Factors convert to g/L, mol/L, or fraction. They are physical unit rules,
# not fitted scientific parameters. Public symbols are canonical output units.
UNIT_TABLE = {
    "g/L": ("mass_concentration", "1"), "g/mL": ("mass_concentration", "1000"),
    "mg/mL": ("mass_concentration", "1"), "ug/mL": ("mass_concentration", "0.001"),
    "ng/mL": ("mass_concentration", "0.000001"), "mg/L": ("mass_concentration", "0.001"),
    "ug/L": ("mass_concentration", "0.000001"), "ng/L": ("mass_concentration", "0.000000001"),
    "M": ("molar_concentration", "1"), "mM": ("molar_concentration", "0.001"),
    "uM": ("molar_concentration", "0.000001"), "nM": ("molar_concentration", "0.000000001"),
    "pM": ("molar_concentration", "0.000000000001"),
    "%": ("dimensionless", "0.01"), "fraction": ("dimensionless", "1"),
}


def _object(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required, "additionalProperties": False}


def _array(items, unique=False):
    schema = {"type": "array", "items": items}
    if unique:
        schema["uniqueItems"] = True
    return schema


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
STRING = {"type": "string"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
INTEGER = {"type": "integer"}
STRINGS = _array(STRING, unique=True)
CONSTRUCT = _object({key: _nullable(STRING) for key in ("sequence", "n_chemistry", "c_chemistry", "orientation")}, required=[])
MEASUREMENT_INPUT = _object({"value": _nullable(NUMBER), "unit": _nullable(STRING), "assay_id": _nullable(STRING),
                             "construct_id": _nullable(STRING), "measurement_kind": _nullable(STRING)}, required=[])
OBSERVATION_INPUT = _object({
    **{key: _nullable(STRING) for key in ("candidate_id", "biological_unit_id", "assay_id", "readout", "unit", "source_reference")},
    "value": _nullable(NUMBER), "identity_verified": _nullable(BOOLEAN),
    "technical_repeat_count": _nullable(INTEGER), "technical_repeats_aggregated": _nullable(BOOLEAN),
}, required=[])
RANK_ROW = _object({"candidate_id": TEXT, "rank": {"type": "integer", "minimum": 1}, "mu": NUMBER,
                    "sigma": {"type": "number", "minimum": 0}, "LCB": NUMBER})


def _input_properties(task):
    if task == "plan_discovery":
        return {**{key: _nullable(STRING) for key in ("disease", "target", "objective")},
                **{key: _nullable(_array(STRING)) for key in ("available_database_tools", "available_expert_tools", "requested_database_tools", "requested_expert_tools")},
                "constraints": _nullable({"type": "object"})}
    if task == "resolve_measurement":
        return {"measurements": _nullable(_array(MEASUREMENT_INPUT)), "target_unit": _nullable(STRING),
                "molecular_weight_g_mol": _nullable(NUMBER), "combine": _nullable(BOOLEAN)}
    if task == "identity_gate":
        evidence = copy.deepcopy(CONSTRUCT)
        evidence["properties"].update({"source_reference": _nullable(STRING), "evidence_kind": _nullable(STRING), "identity_verified": _nullable(BOOLEAN)})
        return {"candidate": _nullable(CONSTRUCT), "evidence": _nullable(evidence)}
    if task == "claim_scope":
        return {"evidence": _nullable(_array(_object({"level": _nullable(STRING), "source_reference": _nullable(STRING)}, required=[]))),
                "requested_claim": _nullable(STRING)}
    if task == "interpret_ranking":
        row = _object({key: _nullable(value) for key, value in RANK_ROW["properties"].items()}, required=[])
        # Negative sigma/rank and noninteger ranks must produce explicit stops.
        row["properties"].update({"rank": _nullable(NUMBER), "sigma": _nullable(NUMBER)})
        row["properties"]["candidate_id"] = _nullable(STRING)
        return {"rows": _nullable(_array(row)), "penalty": _nullable(NUMBER), "requested_inference": _nullable(STRING), "score_basis": _nullable(STRING)}
    return {"frozen_candidate_ids": _nullable(_array(STRING)), "observations": _nullable(_array(OBSERVATION_INPUT)),
            "request_foundation_training": _nullable(BOOLEAN)}


def input_schema(task):
    """Permits missing facts so the reference can request them; rejects unknown keys."""
    _check_task(task)
    return _object({"input_origin": _nullable(STRING), **copy.deepcopy(_input_properties(task))}, required=[])


payload_schema = input_schema


def task_schema(task):
    """Return a closed JSON output schema; stop/request carry null plan/result."""
    _check_task(task)
    common = {"task": {"type": "string", "const": task}, "action": {"type": "string", "enum": list(ACTIONS)},
              "rationale": TEXT, "missing_fields": STRINGS, "error_codes": STRINGS}
    if task == "plan_discovery":
        content = _object({**{key: TEXT for key in ("target_query", "target_accession", "library_profile", "rationale")},
                           "database_tools": _array({"type": "string", "enum": list(DATABASE_TOOLS)}, True),
                           "expert_tools": _array({"type": "string", "enum": list(EXPERT_TOOLS)}, True)})
    elif task == "resolve_measurement":
        converted = _object({"value": NUMBER, "unit": {"type": "string", "enum": list(UNIT_TABLE)},
                             "assay_id": TEXT, "construct_id": TEXT, "measurement_kind": {"type": "string", "enum": ["experimental", "predicted"]}})
        content = _object({"converted_measurements": _array(converted), "aggregate_value": _nullable(NUMBER),
                           "aggregation": {"type": "string", "enum": ["none", "mean_same_assay_construct_kind"]},
                           "scope": {"const": "unit_conversion_only"}})
    elif task == "identity_gate":
        content = _object({"construct_match": {"const": True}, "evidence_kind": {"enum": ["computed", "experimental", "registry"]},
                           "eligible_for_construct_linkage": {"const": True}, "establishes_biological_efficacy": {"const": False},
                           "scope": {"const": "sequence_chemistry_orientation_only"}})
    elif task == "claim_scope":
        content = _object({"requested_claim": {"enum": list(CLAIM_LEVELS)}, "supported_claims": _array({"enum": list(CLAIM_LEVELS)}, True),
                           "claim_supported": {"const": True}, "scope": {"const": "supplied_evidence_type_not_source_quality"}})
    elif task == "interpret_ranking":
        content = _object({"ranked_candidates": _array(RANK_ROW), "score_scope": {"const": "prior_priority_not_efficacy"},
                           "uncertainty_scope": {"const": "researcher_prior_weight_sensitivity"}, "efficacy_established": {"const": False}})
    else:
        content = _object({"accepted_observations": {"type": "integer", "minimum": 1},
                           "independent_biological_units": {"type": "integer", "minimum": 1},
                           "next_action": {"const": "design_review_only"}, "foundation_model_training": {"const": False},
                           "frozen_ranking_modified": {"const": False}})
    common["plan" if task == "plan_discovery" else "result"] = _nullable(content)
    return copy.deepcopy(_object(common))


def _check_task(task):
    if task not in TASKS:
        raise ValueError(f"Unknown contract task: {task}")


def _schema_errors(value, schema, path="$"):
    """Small explicit JSON-schema subset used by the exported closed schemas."""
    if "anyOf" in schema:
        return [] if any(not _schema_errors(value, branch, path) for branch in schema["anyOf"]) else [path + ":anyOf"]
    kind = schema.get("type")
    try:
        number = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        number = False
    valid = {"object": type(value) is dict, "array": type(value) is list, "string": type(value) is str,
             "null": value is None, "boolean": type(value) is bool, "number": number,
             "integer": type(value) is int}.get(kind, True)
    if not valid:
        return [path + ":type"]
    issues = []
    if "const" in schema and (value != schema["const"] or (type(schema["const"]) is bool and type(value) is not bool)):
        issues.append(path + ":const")
    if "enum" in schema and value not in schema["enum"]:
        issues.append(path + ":enum")
    if kind == "object":
        props = schema.get("properties", {})
        issues += [path + "." + key + ":required" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            issues += [path + "." + str(key) + ":unknown" for key in value if key not in props]
        for key, child in value.items():
            if key in props:
                issues += _schema_errors(child, props[key], path + "." + key)
    elif kind == "array":
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value}) != len(value):
            issues.append(path + ":duplicate_items")
        for index, child in enumerate(value):
            issues += _schema_errors(child, schema.get("items", {}), f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or (schema.get("pattern") and not re.search(schema["pattern"], value)):
            issues.append(path + ":empty")
    elif kind in ("number", "integer") and "minimum" in schema and value < schema["minimum"]:
        issues.append(path + ":minimum")
    return issues


class _DuplicateKey(ValueError):
    pass


class _NonfiniteConstant(ValueError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value):
    raise _NonfiniteConstant(value)


def validate_response(task, response):
    """Parse one JSON object or one fence, rejecting duplicate keys/NaN/unknowns."""
    _check_task(task)
    errors, details = [], []
    try:
        if isinstance(response, str):
            text = response.strip()
            if text.startswith("```"):
                fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.I | re.S)
                if not fence:
                    raise ValueError("Incomplete or multiple JSON fences")
                text = fence.group(1)
            parsed = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
        elif type(response) is dict:
            json.dumps(response, ensure_ascii=False, allow_nan=False)
            parsed = copy.deepcopy(response)
        else:
            raise ValueError("Response must be JSON text or dict")
    except _DuplicateKey as exc:
        errors, details = ["DUPLICATE_JSON_KEY"], [str(exc)]
    except _NonfiniteConstant as exc:
        errors, details = ["NONFINITE_JSON_NUMBER"], [str(exc)]
    except (ValueError, TypeError, OverflowError) as exc:
        errors, details = ["INVALID_JSON"], [str(exc)]
    if errors:
        return {"json_valid": False, "schema_valid": False, "parsed_response": None, "error_codes": errors, "validation_errors": details}
    details = _schema_errors(parsed, task_schema(task))
    if not details:
        content_key = "plan" if task == "plan_discovery" else "result"
        if (parsed["action"] == "run") != (parsed[content_key] is not None):
            details.append("$:action_content_mismatch")
        if parsed["action"] == "run" and (parsed["error_codes"] or parsed["missing_fields"]):
            details.append("$:run_cannot_have_errors_or_missing_fields")
        if parsed["action"] == "request_information" and not parsed["missing_fields"]:
            details.append("$:request_requires_missing_fields")
        if parsed["action"] == "stop" and not parsed["error_codes"]:
            details.append("$:stop_requires_error_codes")
    return {"json_valid": True, "schema_valid": not details, "parsed_response": parsed,
            "error_codes": ["SCHEMA_VALIDATION_FAILED"] if details else [], "validation_errors": details}


def _answer(task, action, rationale, *, content=None, missing=(), error=None):
    return {"task": task, "action": action, "rationale": rationale, "missing_fields": sorted(set(missing)),
            "error_codes": [error] if error else [], "plan" if task == "plan_discovery" else "result": content}


def _stop(task, code):
    return _answer(task, "stop", f"The software contract rejects this request: {code}.", error=code)


def _request(task, missing):
    return _answer(task, "request_information", "Required facts are missing; obtain them before continuing.", missing=missing, error="MISSING_INFORMATION")


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _field_missing(obj, fields, prefix=""):
    return [prefix + key for key in fields if _blank(obj.get(key))]


def _plan(payload):
    task = "plan_discovery"
    constraints = payload.get("constraints") or {}
    if set(constraints) - {"min_length", "max_length", "allowed_n_terminal", "allowed_c_terminal"}:
        return _stop(task, "UNKNOWN_CONSTRAINT")
    for key in ("min_length", "max_length"):
        if key in constraints and (type(constraints[key]) is not int or constraints[key] < 1):
            return _stop(task, "INVALID_CONSTRAINTS")
    if constraints.get("min_length", 1) > constraints.get("max_length", 1000):
        return _stop(task, "CONTRADICTORY_CONSTRAINTS")
    for key, states in (("allowed_n_terminal", N_STATES), ("allowed_c_terminal", C_STATES)):
        if key in constraints and (type(constraints[key]) is not list or not constraints[key] or any(value not in states for value in constraints[key])):
            return _stop(task, "INVALID_CONSTRAINTS")
    target = payload.get("target")
    if not _blank(target) and target.strip().casefold() not in {"c5a", "c5", "p01031", "complement c5", "complement c5a"}:
        return _stop(task, "UNSUPPORTED_TARGET")
    for key, registered in (("available_database_tools", DATABASE_TOOLS), ("available_expert_tools", EXPERT_TOOLS),
                            ("requested_database_tools", DATABASE_TOOLS), ("requested_expert_tools", EXPERT_TOOLS)):
        tools = payload.get(key)
        if tools is not None:
            if len(tools) != len(set(tools)):
                return _stop(task, "DUPLICATE_TOOLS")
            if set(tools) - set(registered):
                return _stop(task, "UNKNOWN_TOOL")
    available_db, available_expert = payload.get("available_database_tools"), payload.get("available_expert_tools")
    if available_db is not None and not REQUIRED_DATABASE_TOOLS <= set(available_db):
        return _stop(task, "REQUIRED_TOOL_UNAVAILABLE")
    if available_expert is not None and not REQUIRED_EXPERT_TOOLS <= set(available_expert):
        return _stop(task, "REQUIRED_TOOL_UNAVAILABLE")
    db = REQUIRED_DATABASE_TOOLS | set(payload.get("requested_database_tools") or [])
    experts = REQUIRED_EXPERT_TOOLS | set(payload.get("requested_expert_tools") or [])
    if (available_db is not None and not db <= set(available_db)) or (available_expert is not None and not experts <= set(available_expert)):
        return _stop(task, "REQUESTED_TOOL_UNAVAILABLE")
    missing = _field_missing(payload, ("disease", "target", "available_database_tools", "available_expert_tools"))
    if missing:
        return _request(task, missing)
    plan = {"target_query": target.strip(), "target_accession": "P01031", "library_profile": "c5a_modular_v21",
            "database_tools": sorted(db), "expert_tools": sorted(experts),
            "rationale": "Retrieve verified target information and sequence, then run the required frozen representation and properties tools."}
    return _answer(task, "run", "Use the registered C5a mechanism and available required tools.", content=plan)


def canonical_unit(unit):
    """Preserve SI case; allow micro-symbol and explicitly liter l/L aliases.

``m``/``mm``/``nm`` are length symbols, not molar concentrations.  Likewise
``Mg`` cannot be normalized to ``mg``.  Only the word ``fraction`` is folded.
    """
    if not isinstance(unit, str):
        return None
    key = re.sub(r"\s+", "", unicodedata.normalize("NFKC", unit).replace("μ", "u").replace("µ", "u"))
    if key.casefold() == "fraction":
        return "fraction"
    if key in UNIT_TABLE:
        return key
    liter_alias = re.fullmatch(r"(g|mg|ug|ng)/(m?)[lL]", key)
    if liter_alias:
        canonical = liter_alias.group(1) + "/" + liter_alias.group(2) + "L"
        return canonical if canonical in UNIT_TABLE else None
    return None


def _measurement(payload):
    task = "resolve_measurement"
    rows, target = payload.get("measurements") or [], canonical_unit(payload.get("target_unit"))
    weight = payload.get("molecular_weight_g_mol")
    if weight is not None and weight <= 0:
        return _stop(task, "INVALID_MOLECULAR_WEIGHT")
    if not _blank(payload.get("target_unit")) and target is None:
        return _stop(task, "UNSUPPORTED_UNIT")
    for row in rows:
        if not _blank(row.get("unit")) and canonical_unit(row["unit"]) is None:
            return _stop(task, "UNSUPPORTED_UNIT")
        if not _blank(row.get("measurement_kind")) and row["measurement_kind"] not in ("experimental", "predicted"):
            return _stop(task, "INVALID_MEASUREMENT_KIND")
    complete_context = [row for row in rows if not _field_missing(row, ("assay_id", "construct_id", "measurement_kind"))]
    contexts = {(row["assay_id"], row["construct_id"], row["measurement_kind"]) for row in complete_context}
    if payload.get("combine") and len(contexts) > 1:
        return _stop(task, "INCOMPATIBLE_AGGREGATION")
    needs_weight = False
    if target:
        target_dim = UNIT_TABLE[target][0]
        for row in rows:
            source = canonical_unit(row.get("unit"))
            if source:
                dims = {UNIT_TABLE[source][0], target_dim}
                if len(dims) > 1:
                    if dims != {"mass_concentration", "molar_concentration"}:
                        return _stop(task, "INCOMPATIBLE_UNITS")
                    needs_weight = True
    missing = _field_missing(payload, ("target_unit",))
    if not rows:
        missing.append("measurements")
    for index, row in enumerate(rows):
        missing += _field_missing(row, ("value", "unit", "assay_id", "construct_id", "measurement_kind"), f"measurements[{index}].")
    if needs_weight and weight is None:
        missing.append("molecular_weight_g_mol")
    if missing:
        return _request(task, missing)
    converted, values = [], []
    for row in rows:
        source = canonical_unit(row["unit"])
        source_dim, source_factor = UNIT_TABLE[source]
        target_dim, target_factor = UNIT_TABLE[target]
        value = Decimal(str(row["value"])) * Decimal(source_factor)
        if source_dim == "mass_concentration" and target_dim == "molar_concentration":
            value /= Decimal(str(weight))
        elif source_dim == "molar_concentration" and target_dim == "mass_concentration":
            value *= Decimal(str(weight))
        value /= Decimal(target_factor)
        as_float = float(value)
        if not math.isfinite(as_float):
            return _stop(task, "CONVERSION_OVERFLOW")
        values.append(value)
        converted.append({**row, "value": as_float, "unit": target})
    combine = bool(payload.get("combine"))
    content = {"converted_measurements": converted, "aggregate_value": float(sum(values) / len(values)) if combine else None,
               "aggregation": "mean_same_assay_construct_kind" if combine else "none", "scope": "unit_conversion_only"}
    return _answer(task, "run", "Convert units only; keep assay, construct and experimental/predicted contexts explicit.", content=content)


def _identity(payload):
    task = "identity_gate"
    candidate, evidence = payload.get("candidate") or {}, payload.get("evidence") or {}
    for obj in (candidate, evidence):
        for key, allowed in (("n_chemistry", N_STATES), ("c_chemistry", C_STATES), ("orientation", ORIENTATIONS)):
            if not _blank(obj.get(key)) and obj[key] not in allowed:
                return _stop(task, "UNREGISTERED_CONSTRUCT_STATE")
        if not _blank(obj.get("sequence")) and (not set(obj["sequence"]) <= AA20):
            return _stop(task, "INVALID_SEQUENCE")
    for key in ("sequence", "n_chemistry", "c_chemistry", "orientation"):
        if not _blank(candidate.get(key)) and not _blank(evidence.get(key)) and candidate[key] != evidence[key]:
            return _stop(task, "CONSTRUCT_MISMATCH")
    kind = evidence.get("evidence_kind")
    if not _blank(kind) and kind not in ("computed", "experimental", "registry"):
        return _stop(task, "INVALID_EVIDENCE_KIND")
    if kind == "experimental" and evidence.get("identity_verified") is False:
        return _stop(task, "UNVERIFIED_EXPERIMENTAL_IDENTITY")
    missing = _field_missing(candidate, ("sequence", "n_chemistry", "c_chemistry", "orientation"), "candidate.")
    missing += _field_missing(evidence, ("sequence", "n_chemistry", "c_chemistry", "orientation", "source_reference", "evidence_kind"), "evidence.")
    if kind == "experimental" and evidence.get("identity_verified") is None:
        missing.append("evidence.identity_verified")
    if missing:
        return _request(task, missing)
    return _answer(task, "run", "Link evidence by complete construct identity; this gate does not establish biological efficacy.",
                   content={"construct_match": True, "evidence_kind": kind, "eligible_for_construct_linkage": True,
                            "establishes_biological_efficacy": False, "scope": "sequence_chemistry_orientation_only"})


def _claims(payload):
    task = "claim_scope"
    claim, rows = payload.get("requested_claim"), payload.get("evidence") or []
    if not _blank(claim) and claim not in CLAIM_LEVELS:
        return _stop(task, "UNKNOWN_CLAIM_SCOPE")
    if any(not _blank(row.get("level")) and row["level"] not in CLAIM_LEVELS for row in rows):
        return _stop(task, "UNKNOWN_EVIDENCE_LEVEL")
    levels = {row["level"] for row in rows if not _field_missing(row, ("level", "source_reference"))}
    # No automatic hierarchy: prediction, recognition, function, mouse and human
    # endpoints are distinct. A missing reference cannot support its level.
    if not _blank(claim) and rows and all(not _blank(row.get("level")) for row in rows) and claim not in {row["level"] for row in rows}:
        return _stop(task, "CLAIM_EXCEEDS_EVIDENCE")
    missing = _field_missing(payload, ("requested_claim",))
    if not rows:
        missing.append("evidence")
    for index, row in enumerate(rows):
        missing += _field_missing(row, ("level", "source_reference"), f"evidence[{index}].")
    if missing:
        return _request(task, missing)
    if claim not in levels:
        return _stop(task, "CLAIM_EXCEEDS_EVIDENCE")
    return _answer(task, "run", "Report only supplied evidence types; source presence does not certify evidence quality or clinical efficacy.",
                   content={"requested_claim": claim, "supported_claims": sorted(levels), "claim_supported": True,
                            "scope": "supplied_evidence_type_not_source_quality"})


def _ranking(payload):
    task = "interpret_ranking"
    rows = payload.get("rows") or []
    inference = payload.get("requested_inference") or "priority"
    basis = payload.get("score_basis") or "prior_weight_sensitivity"
    penalty = 0.25 if payload.get("penalty") is None else payload["penalty"]
    if inference != "priority":
        return _stop(task, "EFFICACY_NOT_ESTABLISHED" if inference == "efficacy" else "UNSUPPORTED_INFERENCE")
    if basis != "prior_weight_sensitivity":
        return _stop(task, "UNSUPPORTED_SCORE_BASIS")
    if penalty < 0:
        return _stop(task, "INVALID_PENALTY")
    identifiers = [row["candidate_id"] for row in rows if not _blank(row.get("candidate_id"))]
    ranks = [row["rank"] for row in rows if row.get("rank") is not None]
    if len(identifiers) != len(set(identifiers)) or len(ranks) != len(set(ranks)):
        return _stop(task, "DUPLICATE_RANK_OR_ID")
    if any(type(rank) is not int or rank < 1 for rank in ranks):
        return _stop(task, "INVALID_RANK")
    for row in rows:
        if row.get("sigma") is not None and row["sigma"] < 0:
            return _stop(task, "INVALID_SIGMA")
        if all(row.get(key) is not None for key in ("mu", "sigma", "LCB")) and not math.isclose(row["LCB"], row["mu"] - penalty * row["sigma"], abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE):
            return _stop(task, "INCONSISTENT_TOOL_SCORES")
    complete = [row for row in rows if not _field_missing(row, ("candidate_id", "rank", "mu", "sigma", "LCB"))]
    ordered = sorted(complete, key=lambda row: row["rank"])
    if any(left["LCB"] + ABS_TOLERANCE < right["LCB"] for left, right in zip(ordered, ordered[1:])):
        return _stop(task, "INCONSISTENT_RANK_ORDER")
    missing = [] if rows else ["rows"]
    for index, row in enumerate(rows):
        missing += _field_missing(row, ("candidate_id", "rank", "mu", "sigma", "LCB"), f"rows[{index}].")
    if not missing and sorted(ranks) != list(range(1, len(rows) + 1)):
        return _stop(task, "NONCONSECUTIVE_RANKS")
    if missing:
        return _request(task, missing)
    return _answer(task, "run", "Read the supplied tool ranking as prior-based computational priority; sigma measures prior-weight sensitivity, not predictive efficacy uncertainty.",
                   content={"ranked_candidates": ordered, "score_scope": "prior_priority_not_efficacy",
                            "uncertainty_scope": "researcher_prior_weight_sensitivity", "efficacy_established": False})


def _feedback(payload):
    task = "feedback_gate"
    rows, frozen = payload.get("observations") or [], payload.get("frozen_candidate_ids")
    if payload.get("request_foundation_training"):
        return _stop(task, "FOUNDATION_TRAINING_NOT_AUTHORIZED")
    if frozen is not None and len(frozen) != len(set(frozen)):
        return _stop(task, "DUPLICATE_FROZEN_IDS")
    if any(row.get("identity_verified") is False for row in rows):
        return _stop(task, "UNVERIFIED_EXPERIMENTAL_IDENTITY")
    if frozen and any(not _blank(row.get("candidate_id")) and row["candidate_id"] not in frozen for row in rows):
        return _stop(task, "UNKNOWN_FROZEN_CANDIDATE")
    keys = []
    units = {}
    for row in rows:
        count = row.get("technical_repeat_count")
        if count is not None and count < 1:
            return _stop(task, "INVALID_TECHNICAL_REPEAT_COUNT")
        if count is not None and count > 1 and row.get("technical_repeats_aggregated") is not True:
            return _stop(task, "UNAGGREGATED_TECHNICAL_REPEATS")
        fields = ("candidate_id", "biological_unit_id", "assay_id", "readout")
        if not _field_missing(row, fields):
            key = tuple(row[field] for field in fields)
            if key in keys:
                return _stop(task, "UNAGGREGATED_TECHNICAL_REPEATS")
            keys.append(key)
        if not _field_missing(row, ("assay_id", "readout", "unit")):
            assay = (row["assay_id"], row["readout"])
            unit = canonical_unit(row["unit"]) or row["unit"].strip()
            if assay in units and units[assay] != unit:
                return _stop(task, "INCONSISTENT_ASSAY_UNITS")
            units[assay] = unit
    missing = []
    if not rows:
        missing.append("observations")
    if not frozen:
        missing.append("frozen_candidate_ids")
    for index, row in enumerate(rows):
        missing += _field_missing(row, ("candidate_id", "biological_unit_id", "assay_id", "readout", "value", "unit", "source_reference", "identity_verified"), f"observations[{index}].")
    if missing:
        return _request(task, missing)
    return _answer(task, "run", "Accept qualified observations for next-cycle design review only; do not retrain the foundation model or modify the frozen ranking.",
                   content={"accepted_observations": len(rows), "independent_biological_units": len({row["biological_unit_id"] for row in rows}),
                            "next_action": "design_review_only", "foundation_model_training": False, "frozen_ranking_modified": False})


_REFERENCES = dict(zip(TASKS, (_plan, _measurement, _identity, _claims, _ranking, _feedback)))


def reference_answer(task, payload):
    """Execute only the documented software rules; explicit violations win."""
    _check_task(task)
    try:
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        return _stop(task, "INVALID_PAYLOAD")
    if type(payload) is not dict or _schema_errors(payload, input_schema(task)):
        return _stop(task, "INVALID_PAYLOAD")
    origin = payload.get("input_origin")
    if not _blank(origin) and origin not in INPUT_ORIGINS:
        return _stop(task, "INVALID_INPUT_ORIGIN")
    result = _REFERENCES[task](payload)
    if result["action"] != "stop" and _blank(origin):
        return _request(task, result["missing_fields"] + ["input_origin"])
    return result


def _semantic_mismatches(expected, observed, path="$"):
    if type(expected) is dict and type(observed) is dict:
        result = []
        if set(expected) != set(observed):
            return [path + ":keys"]
        for key, value in expected.items():
            if key == "rationale":
                continue
            if key in {"database_tools", "expert_tools", "missing_fields", "error_codes", "supported_claims"}:
                if set(value) != set(observed[key]):
                    result.append(path + "." + key)
            else:
                result += _semantic_mismatches(value, observed[key], path + "." + key)
        return result
    if type(expected) is list and type(observed) is list:
        if len(expected) != len(observed):
            return [path + ":length"]
        return [mismatch for index, (a, b) in enumerate(zip(expected, observed)) for mismatch in _semantic_mismatches(a, b, f"{path}[{index}]")]
    if type(expected) in (int, float) and type(observed) in (int, float):
        if isinstance(expected, int) and isinstance(observed, int):
            return [] if expected == observed else [path]
        return [] if math.isclose(expected, observed, abs_tol=ABS_TOLERANCE, rel_tol=REL_TOLERANCE) else [path]
    return [] if type(expected) is type(observed) and expected == observed else [path]


def score_response(task, payload, response):
    """Schema plus deterministic semantic scoring; no reference-text matching."""
    validation = validate_response(task, response)
    expected = reference_answer(task, payload)
    parsed = validation["parsed_response"]
    observed_action = parsed.get("action") if type(parsed) is dict else None
    mismatches = _semantic_mismatches(expected, parsed) if validation["schema_valid"] else []
    codes = list(validation["error_codes"])
    if mismatches:
        codes.append("SEMANTIC_MISMATCH")
        if observed_action != expected["action"]:
            codes.append("ACTION_MISMATCH")
    return {"json_valid": validation["json_valid"], "schema_valid": validation["schema_valid"],
            "semantic_success": validation["schema_valid"] and not mismatches, "expected_action": expected["action"],
            "observed_action": observed_action, "dangerous_acceptance": expected["action"] != "run" and observed_action == "run",
            "error_codes": codes, "errors": list(codes), "mismatches": mismatches,
            "validation_errors": validation["validation_errors"]}


def example_payload(task):
    _check_task(task)
    origin = {"input_origin": "synthetic_contract"}
    if task == "plan_discovery":
        return {**origin, "disease": "synthetic C5a study", "target": "C5a", "objective": "prioritize candidates",
                "available_database_tools": sorted(REQUIRED_DATABASE_TOOLS), "available_expert_tools": sorted(REQUIRED_EXPERT_TOOLS), "constraints": {}}
    if task == "resolve_measurement":
        return {**origin, "measurements": [{"value": 1.0, "unit": "ug/mL", "assay_id": "synthetic_A", "construct_id": "C1", "measurement_kind": "experimental"}],
                "target_unit": "ng/mL", "combine": False}
    if task == "identity_gate":
        candidate = {"sequence": "ACDEFGH", "n_chemistry": "free", "c_chemistry": "amide", "orientation": "helper_to_b"}
        return {**origin, "candidate": candidate, "evidence": {**candidate, "source_reference": "synthetic:contract", "evidence_kind": "computed"}}
    if task == "claim_scope":
        return {**origin, "evidence": [{"level": "mhc_prediction", "source_reference": "synthetic:contract"}], "requested_claim": "mhc_prediction"}
    if task == "interpret_ranking":
        return {**origin, "rows": [{"candidate_id": "C1", "rank": 1, "mu": 0.8, "sigma": 0.04, "LCB": 0.79}], "penalty": 0.25,
                "requested_inference": "priority", "score_basis": "prior_weight_sensitivity"}
    return {**origin, "frozen_candidate_ids": ["C1"], "observations": [{"candidate_id": "C1", "biological_unit_id": "synthetic_B1", "assay_id": "synthetic_A",
            "readout": "synthetic_readout", "value": 1.0, "unit": "ng/mL", "source_reference": "synthetic:contract", "identity_verified": True}], "request_foundation_training": False}


_RULES = {
    "plan_discovery": "In order: unknown constraint key -> UNKNOWN_CONSTRAINT; invalid positive-integer length or registered terminal lists -> INVALID_CONSTRAINTS; min_length>max_length(default1000) -> CONTRADICTORY_CONSTRAINTS. Only C5a,C5,P01031,complement C5/C5a (case-insensitive) are supported -> otherwise UNSUPPORTED_TARGET. Duplicate tool items -> DUPLICATE_TOOLS; unregistered name -> UNKNOWN_TOOL. Required database tools protein_sequence,target_information and expert tools esm,properties must be available -> REQUIRED_TOOL_UNAVAILABLE; requested optional tools not available -> REQUESTED_TOOL_UNAVAILABLE. Missing disease,target or available-tool lists -> request. Run uses target_query=trimmed target, accession=P01031, library_profile=c5a_modular_v21, tool sets=required union requested. Original six plan fields stay inside plan.",
    "resolve_measurement": "In order: supplied nonpositive molecular_weight_g_mol -> INVALID_MOLECULAR_WEIGHT; unsupported unit -> UNSUPPORTED_UNIT; measurement_kind must be experimental/predicted -> INVALID_MEASUREMENT_KIND. combine=true with distinct (assay_id,construct_id,measurement_kind) -> INCOMPATIBLE_AGGREGATION. Dimensionless/concentration mixing -> INCOMPATIBLE_UNITS. Missing measurements,target_unit or row value/unit/assay_id/construct_id/measurement_kind -> request. Converting mass concentration to/from molar requires molecular_weight_g_mol; absent -> request. Convert independently; combine=false never averages. combine=true computes mean only in identical context. Mass base g/L factors: g/L=1,g/mL=1000,mg/mL=1,ug/mL=.001,ng/mL=.000001,mg/L=.001,ug/L=.000001,ng/L=1e-9. Molar base M: M=1,mM=.001,uM=1e-6,nM=1e-9,pM=1e-12. SI symbols/prefixes are case-sensitive: m/mm/nm, Mg/mL, ML are unsupported. NFKC, micro μ/µ->u, whitespace and liter l/L aliases are allowed; fraction word is case-insensitive. Percent values multiply .01 to become fractions. mass(g/L)/MW(g/mol)=M; inverse multiply MW. Output canonical units. Scope unit_conversion_only. Overflow -> CONVERSION_OVERFLOW.",
    "identity_gate": "In order: N chemistry free/acetyl, C free/amide, orientation helper_to_b/b_to_helper -> otherwise UNREGISTERED_CONSTRUCT_STATE; sequence uppercase AA20 -> otherwise INVALID_SEQUENCE. Any supplied mismatch of sequence,N/C,orientation -> CONSTRUCT_MISMATCH. evidence_kind computed/experimental/registry -> otherwise INVALID_EVIDENCE_KIND; experimental identity_verified=false -> UNVERIFIED_EXPERIMENTAL_IDENTITY. Missing candidate/evidence identity fields, evidence source_reference or evidence_kind -> request; experimental missing identity_verified also request. Run establishes complete-construct linkage only, never efficacy.",
    "claim_scope": "Public text view: evidence[] has source_excerpt/source_reference, and requested_claim may be a natural-language question. First infer each excerpt's scope and normalize the question to the requested_claim enum; do not treat source_excerpt or a natural question as an unknown field/value. Scope readouts: in-silico MHC binding/presentation, including NetMHCIIpan EL rank -> mhc_prediction; target-binding/docking prediction -> binding_prediction; measured antibody binding to native antigen -> native_recognition; measured functional inhibition/neutralization -> neutralization; in-vivo mouse endpoint -> mouse_endpoint; human participant clinical endpoint -> human_clinical_endpoint (human cells alone are not clinical endpoints). Unrecognized request -> UNKNOWN_CLAIM_SCOPE; explicitly unrecognized readout -> UNKNOWN_EVIDENCE_LEVEL. Blank excerpt maps to missing evidence[i].level; blank request maps to missing requested_claim; missing source_reference maps to evidence[i].source_reference. Then apply the original rules to normalized scopes: scopes are distinct, not a hierarchy. If all supplied levels are known/present but requested scope is absent -> CLAIM_EXCEEDS_EVIDENCE before missing references. Missing claim/evidence or row level/source_reference -> request. Run only if the exact requested level has a source; supported_claims lists sourced levels. Existing structured inputs use level directly. Source presence does not certify quality or clinical efficacy; no prediction-to-function or mouse-to-human inference.",
    "interpret_ranking": "requested_inference defaults priority: efficacy -> EFFICACY_NOT_ESTABLISHED; other inference -> UNSUPPORTED_INFERENCE. score_basis defaults prior_weight_sensitivity -> otherwise UNSUPPORTED_SCORE_BASIS; penalty defaults .25, negative -> INVALID_PENALTY. Duplicate ID/rank -> DUPLICATE_RANK_OR_ID; ranks must be positive integers -> INVALID_RANK; negative sigma -> INVALID_SIGMA. LCB must equal mu-penalty*sigma within abs1e-8/rel1e-6 -> INCONSISTENT_TOOL_SCORES; LCB must not increase with rank (abs1e-8) -> INCONSISTENT_RANK_ORDER. Missing rows/row candidate_id,rank,mu,sigma,LCB -> request; otherwise ranks must be 1..N -> NONCONSECUTIVE_RANKS. Copy tool values and sort by rank. Scores are prior priority, sigma prior-weight sensitivity, not efficacy or predictive interval.",
    "feedback_gate": "In order: request_foundation_training=true -> FOUNDATION_TRAINING_NOT_AUTHORIZED; duplicate frozen IDs -> DUPLICATE_FROZEN_IDS; identity_verified=false -> UNVERIFIED_EXPERIMENTAL_IDENTITY; observation ID not frozen -> UNKNOWN_FROZEN_CANDIDATE. technical_repeat_count<1 -> INVALID_TECHNICAL_REPEAT_COUNT; count>1 without technical_repeats_aggregated=true, or repeated (candidate_id,biological_unit_id,assay_id,readout) -> UNAGGREGATED_TECHNICAL_REPEATS. Different units for same assay_id/readout -> INCONSISTENT_ASSAY_UNITS (convert separately first; canonical aliases equivalent). Missing frozen IDs/observations or row candidate_id,biological_unit_id,assay_id,readout,value,unit,source_reference,identity_verified -> request. Run counts records and unique biological_unit_id; next_action design_review_only, foundation_model_training=false, frozen_ranking_modified=false.",
}


def task_instruction(task):
    """Complete deterministic rules and a compact schema, identical for all models."""
    _check_task(task)
    common = ("Execute a software contract, not a biological prediction. Return one JSON object and no prose. "
              "Payload input_origin must be synthetic_contract or existing_tool_result: other nonblank origin -> stop INVALID_INPUT_ORIGIN; missing origin -> request. "
              "Unknown keys/wrong JSON types/nonfinite numbers -> stop INVALID_PAYLOAD. Explicit stops precede missing facts. "
              "All missing facts: action=request_information, error_codes=[MISSING_INFORMATION], missing_fields=unique dotted/indexed field paths. "
              "Stops: action=stop, one first applicable error code, missing_fields=[]. Run: no errors/missing fields. "
              "For stop/request plan/result=null; run has the task object. All rationales must be nonempty; wording is free. "
              "Tools are unique sets. Numbers use abs1e-8 and rel1e-6 tolerance. Do not invent observations or clinical efficacy. ")
    if task == "claim_scope":
        common = common.replace(
            "Unknown keys/wrong JSON types/nonfinite numbers -> stop INVALID_PAYLOAD.",
            "Normalize the legitimate text view described below before checking unknown keys; source_excerpt is allowed public input. Unknown keys in the normalized contract/wrong JSON types/nonfinite numbers -> stop INVALID_PAYLOAD.",
        )
    return common + _RULES[task] + " Output schema: " + json.dumps(task_schema(task), separators=(",", ":"), ensure_ascii=False)
