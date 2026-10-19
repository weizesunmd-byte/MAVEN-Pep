"""Original semantic loss weighting helpers, extracted without calculation changes."""
from __future__ import annotations
import json, math
from maven_final.model_runtime import action_value_character_span, fingerprint, RuntimeContractError
def validate_semantic_loss_protocol(config):
    protocol = config.get("semantic_loss_protocol")
    keys = {"schema", "ordinary_token_weight", "plan_result_value_weight", "diagnostic_value_weight", "rationale_value_weight", "overlap_policy"}
    if (not isinstance(protocol, dict) or set(protocol) != keys or protocol.get("schema") != "maven_final.semantic_loss.v1"
            or protocol.get("overlap_policy") != "character_mean_rationale_over_semantic_then_action_total"):
        raise RuntimeContractError("V6 requires the complete explicit semantic loss protocol")
    for key in ("ordinary_token_weight", "plan_result_value_weight", "diagnostic_value_weight", "rationale_value_weight"):
        value = protocol[key]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value <= 0:
            raise RuntimeContractError("Semantic loss weights must be finite positive JSON numbers: "+key)
    if (protocol["ordinary_token_weight"] != 1 or protocol["plan_result_value_weight"] <= 1
            or protocol["diagnostic_value_weight"] <= 1 or protocol["rationale_value_weight"] > .1
            or config.get("action_loss_total_weight") != 12):
        raise RuntimeContractError("V6 preserves ordinary1/action-total12, emphasizes semantic/diagnostic values and limits rationale to0.1")
    return dict(protocol)


def json_value_spans(text):
    """Locate exact raw JSON value spans, including escaped strings and nesting."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeContractError("Duplicate JSON key in semantic target")
            result[key] = value
        return result
    def invalid(value):
        raise RuntimeContractError("Nonfinite JSON constant in semantic target: "+value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeContractError("Nonfinite JSON number in semantic target")
        return result
    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid, parse_float=finite_float)
    try:
        parsed = decoder.decode(text)
    except (ValueError, TypeError) as error:
        raise RuntimeContractError("Semantic target must be exactly one strict JSON object") from error
    if not isinstance(parsed, dict):
        raise RuntimeContractError("Semantic target must be a JSON object")
    spans = []
    def white(cursor):
        while cursor < len(text) and text[cursor].isspace(): cursor += 1
        return cursor
    def walk(cursor, path):
        start = cursor = white(cursor)
        value, decoded_end = decoder.raw_decode(text, cursor)
        if isinstance(value, dict):
            cursor = white(cursor+1)
            while text[cursor] != "}":
                key, cursor = decoder.raw_decode(text, cursor)
                cursor = white(cursor)
                if text[cursor] != ":": raise RuntimeContractError("Malformed semantic JSON separator")
                cursor = white(walk(cursor+1, path+(key,)))
                if text[cursor] == ",": cursor = white(cursor+1)
                elif text[cursor] != "}": raise RuntimeContractError("Malformed semantic JSON object boundary")
            cursor += 1
        elif isinstance(value, list):
            cursor, index = white(cursor+1), 0
            while text[cursor] != "]":
                cursor = white(walk(cursor, path+(index,))); index += 1
                if text[cursor] == ",": cursor = white(cursor+1)
                elif text[cursor] != "]": raise RuntimeContractError("Malformed semantic JSON array boundary")
            cursor += 1
        else:
            cursor = decoded_end
        if cursor != decoded_end:
            raise RuntimeContractError("Semantic JSON parser boundaries disagree")
        spans.append((path, start, cursor, value))
        return cursor
    if white(walk(0, ())) != len(text):
        raise RuntimeContractError("Trailing semantic JSON data")
    return spans


def semantic_reweight_example(raw, encoded, tokenizer, config):
    """Return reweighted copy plus separate audit; IDs/labels/masks never change.

    Value weights include raw JSON quotes/brackets. Child rationale characters
    override plan/result characters. A boundary token gets their character mean;
    action-content tokens then receive the original fixed total12 allocation.
    Special end tokens with zero-length offsets retain ordinary weight1.
    """
    protocol = validate_semantic_loss_protocol(config)
    result = dict(encoded)
    old_action = encoded.get("_action_loss_audit", {})
    if old_action.get("weighted_action") is not True:
        return result, {"weighted": False, "weights_fingerprint": fingerprint(encoded["loss_weights"])}
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeContractError("Semantic loss requires the real frozen fast tokenizer offsets")
    target = raw["messages"][-1]["content"]
    spans = json_value_spans(target)
    classes = ["ordinary"]*len(target)
    character_weights = [float(protocol["ordinary_token_weight"])]*len(target)
    categories = ((lambda path: path in (("plan",), ("result",)), "semantic", "plan_result_value_weight"),
                  (lambda path: path in (("missing_fields",), ("error_codes",)), "diagnostic", "diagnostic_value_weight"),
                  (lambda path: bool(path) and path[-1] == "rationale", "rationale", "rationale_value_weight"))
    for match, category, key in categories:
        for path, start, end, value in spans:
            if match(path):
                if category == "rationale" and not isinstance(value, str):
                    raise RuntimeContractError("A rationale target value must be a string")
                character_weights[start:end] = [float(protocol[key])]*(end-start)
                classes[start:end] = [category]*(end-start)
    completion_text = target+"<|im_end|>\n"
    tokenized = tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True)
    labels = encoded["labels"]
    prompt_size = next((index for index, value in enumerate(labels) if value != -100), None)
    if (prompt_size is None or labels[:prompt_size] != [-100]*prompt_size
            or labels[prompt_size:] != encoded["input_ids"][prompt_size:]
            or tokenized["input_ids"] != encoded["input_ids"][prompt_size:]):
        raise RuntimeContractError("Semantic offset tokenization changed original IDs or assistant-only labels")
    offsets = tokenized["offset_mapping"]
    if len(offsets) != len(tokenized["input_ids"]):
        raise RuntimeContractError("Semantic offsets are not token aligned")
    coverage, weights, mixed = [False]*len(target), [0.]*prompt_size, 0
    for pair in offsets:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(value) is not int for value in pair)
                or not 0 <= pair[0] <= pair[1] <= len(completion_text)):
            raise RuntimeContractError("Invalid semantic tokenizer character offset")
        left, right = pair
        inside_right = min(right, len(target))
        if right > left:
            used = character_weights[left:inside_right] if left < len(target) else []
            used += [float(protocol["ordinary_token_weight"])]*(right-max(left, len(target)))
            weights.append(sum(used)/(right-left))
            if left < inside_right:
                coverage[left:inside_right] = [True]*(inside_right-left)
                mixed += len(set(classes[left:inside_right])) > 1
        else:
            weights.append(float(protocol["ordinary_token_weight"]))
    if not all(coverage):
        raise RuntimeContractError("Semantic tokenizer offsets leave target characters uncovered")
    start, end = action_value_character_span(target)
    action_positions = [index for index, (left, right) in enumerate(offsets) if right > left and right > start and left < end]
    if action_positions != old_action["completion_token_indices"] or not action_positions:
        raise RuntimeContractError("Semantic weighting changed the original action token span")
    for index in action_positions:
        weights[prompt_size+index] = float(config["action_loss_total_weight"])/len(action_positions)
    if (len(weights) != len(encoded["input_ids"]) or any(not math.isfinite(value) or value < 0 for value in weights)
            or any(weight != 0 for weight, label in zip(weights, labels) if label == -100)):
        raise RuntimeContractError("Semantic loss weights are nonfinite or not aligned with the unchanged mask")
    result["loss_weights"] = weights
    audit = {"weighted": True, "weights_fingerprint": fingerprint(weights), "target_characters": len(target),
        "character_class_counts": {category: classes.count(category) for category in ("ordinary", "semantic", "diagnostic", "rationale")},
        "mixed_boundary_tokens": mixed, "action_tokens": len(action_positions),
        "action_weight_sum": sum(weights[prompt_size+index] for index in action_positions),
        "ids_labels_mask_unchanged": all(result[key] == encoded[key] for key in ("input_ids", "labels", "attention_mask"))}
    return result, audit
