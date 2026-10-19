"""Original stricter legacy39 document contract, separate from content delivery.

Ported without changing the calculation from the historical evaluate_benchmark.py.
The frozen content-delivery scorer remains byte-identical in _vendor/.
"""
import json
import math

MISSING = object()


def strict_document(text):
    def pairs(values):
        obj = {}
        for key, value in values:
            if key in obj: raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    def nonfinite(value): raise ValueError("Nonfinite JSON number")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number): raise ValueError("Nonfinite JSON number")
        return number
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
        return value if type(value) is dict else None
    except (TypeError, ValueError):
        return None


def _value(obj, path):
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return MISSING
        obj = obj[key]
    return obj


def _equal(actual, expected):
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_equal(actual[key], value) for key, value in expected.items())
    return actual == expected


def legacy_strict(case, text):
    obj = strict_document(text)
    if obj is None:
        return False
    keys = set()
    for rule in case["rules"]:
        kind = rule["type"]
        if kind in ("equals", "includes_all"):
            keys.add(rule["path"].split(".")[0])
            value = _value(obj, rule["path"])
            if kind == "equals" and (value is MISSING or not _equal(value, rule["expected"])):
                return False
            if kind == "includes_all" and not (isinstance(value, list) and all(isinstance(item, str) for item in value)
                    and len(value) == len(set(value)) and all(any(_equal(item, wanted) for item in value) for wanted in rule["expected"])):
                return False
        else:
            keys.add("steps")
            values = obj.get("steps")
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                return False
            if kind == "exact_step_set" and (len(values) != len(set(values)) or set(values) != set(rule["expected"])):
                return False
            if kind == "precedes" and not (rule["before"] in values and rule["after"] in values and values.index(rule["before"]) < values.index(rule["after"])):
                return False
    if "contract" in case:
        keys = {"target_query", "target_accession", "library_profile", "database_tools", "expert_tools", "rationale"}
        contract = case["contract"]
        if any(not isinstance(obj.get(key), str) or not obj[key].strip() for key in ("target_query", "rationale")):
            return False
        if obj.get("target_accession") != contract["expected_target_accession"] or obj.get("library_profile") != contract["expected_library_profile"]:
            return False
        for field, prefix in (("database_tools", "database"), ("expert_tools", "expert")):
            value = obj.get(field)
            if not (isinstance(value, list) and value and all(isinstance(item, str) for item in value) and len(value) == len(set(value))
                    and set(value) <= set(contract[f"allowed_{prefix}_tools"]) and set(contract[f"required_{prefix}_tools"]) <= set(value)):
                return False
    return set(obj) == keys
