"""Generate or independently score dev240/legacy39 first answers.

Generation reads public inputs only. Scoring is a separate CPU invocation.
Frozen checkpoints: base, old (R1), candidate (R6), A and B (continuations).
Historical batches use two index-modulo ranks; continuation batches use one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation"
MODEL_DIRS = {"old": "r1", "candidate": "r6", "A": "continuation_A", "B": "continuation_B"}
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    actual = digest(path)
    require(actual == expected, "Frozen asset changed: " + str(path))
    return actual


def suite_inputs(suite, manifest):
    name = "dev_inputs.jsonl" if suite == "dev" else "regression39_inputs.jsonl"
    path = DATA / name
    pinned(path, manifest["files"][path.relative_to(ROOT).as_posix()]["sha256"])
    result = rows(path)
    require(len(result) == manifest["suites"][suite]["cases"], "Unexpected case count")
    return result


def raw_paths(folder, suite, world_size):
    if (folder / (suite + "_raw.jsonl")).exists():
        return [folder / (suite + "_raw.jsonl")]
    return [folder / f"{suite}_raw.rank{rank}.jsonl" for rank in range(world_size)]


def generate(args, manifest):
    # No gold file, gold record or scorer is opened in this execution path.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    identity = manifest["models"][args.model]
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size or identity["world_size"]))
    rank = int(os.environ.get("RANK", args.rank))
    local_rank = int(os.environ.get("LOCAL_RANK", args.device))
    require(world_size == identity["world_size"], "Use the original rank count to preserve each generation batch")
    require(0 <= rank < world_size, "Invalid generation rank")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if world_size == 1 else f".rank{rank}"
    output_paths = [args.output_dir / f"{suite}_raw{suffix}.jsonl" for suite in args.suites]
    metadata_path = args.output_dir / f"generation_protocol{suffix}.json"
    require(not metadata_path.exists() and not any(p.exists() for p in output_paths), "First answers already exist; use a new output directory")
    inputs = {suite: suite_inputs(suite, manifest) for suite in args.suites}
    tokenizer_files = {name: pinned(args.tokenizer / name, value) for name, value in manifest["tokenizer_files_sha256"].items()}
    base_files = {name: pinned(args.base / name, value) for name, value in manifest["base_files_sha256"].items()}
    adapter = args.adapter or (ROOT / "models" / MODEL_DIRS[args.model] if args.model != "base" else None)
    adapter_hash = None
    if args.model == "base":
        require(args.adapter is None, "The base identity must not load an adapter")
    else:
        adapter_hash = pinned(adapter / "adapter_model.safetensors", identity["adapter_sha256"])
        if identity.get("adapter_config_sha256"):
            pinned(adapter / "adapter_config.json", identity["adapter_config_sha256"])
    set_seed(identity["seed"])
    torch.set_num_threads(identity["torch_threads"])
    device = f"cuda:{local_rank}"
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    native = load(args.base / "generation_config.json")
    eos = native.get("eos_token_id", tok.eos_token_id)
    eos = sorted(set((eos if isinstance(eos, list) else [eos]) + [tok.eos_token_id]))
    cfg = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0,
        encoder_repetition_penalty=1.0, length_penalty=1.0, min_length=0, min_new_tokens=0,
        no_repeat_ngram_size=0, early_stopping=False, temperature=1.0, top_p=1.0, top_k=50,
        max_new_tokens=512, eos_token_id=eos, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, use_cache=True, return_dict_in_generate=False)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
        attn_implementation="sdpa", local_files_only=True, device_map={"": device})
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    finite = [True]
    def check_finite(module, incoming, outgoing):
        if hasattr(outgoing, "logits"):
            finite[0] = finite[0] and bool(torch.isfinite(outgoing.logits[:, -1, :]).all().item())
    hook = model.register_forward_hook(check_finite)
    metadata = dict(model=args.model, adapter_sha256=adapter_hash, seed=identity["seed"],
        rank=rank, world_size=world_size, batch_size=4, partition=identity["partition"],
        base_revision=manifest["base_revision"], base_files_sha256=base_files,
        tokenizer_files_sha256=tokenizer_files, enable_thinking=False, gold_access=False,
        generation_config=cfg.to_dict(), use_model_defaults=False, precision="fp16", attention="sdpa",
        suite_budgets={s: manifest["suites"][s] for s in args.suites},
        input_sha256={s: manifest["files"]["data/evaluation/" + ("dev_inputs.jsonl" if s == "dev" else "regression39_inputs.jsonl")]["sha256"] for s in args.suites},
        versions={k: importlib.metadata.version(k) for k in ("torch", "transformers", "tokenizers", "peft")},
        worker_sha256=digest(__file__), device=torch.cuda.get_device_name(local_rank),
        scope="Development/regression diagnostic. New GPU results may vary with numerical kernels; compare regenerated scores explicitly.")
    save(metadata_path, metadata)
    for suite, output_path in zip(args.suites, output_paths):
        cases = [case for index, case in enumerate(inputs[suite]) if index % world_size == rank]
        budget = manifest["suites"][suite]
        cfg.max_new_tokens = budget["max_new_tokens"]
        with output_path.open("x", encoding="utf-8") as stream:
            for start in range(0, len(cases), 4):
                batch = cases[start:start + 4]
                prompts = [tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in batch]
                encoded = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                lengths = encoded["attention_mask"].sum(1).tolist()
                width = encoded["input_ids"].shape[1]
                require(width <= budget["max_input_tokens"], "Input exceeds the frozen token budget")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                torch.cuda.synchronize(local_rank)
                started = time.monotonic()
                with torch.inference_mode():
                    outputs = model.generate(**encoded, generation_config=cfg, use_model_defaults=False)
                torch.cuda.synchronize(local_rank)
                elapsed = time.monotonic() - started
                require(finite[0], "Nonfinite model logits; do not score these predictions")
                for index, (case, generated) in enumerate(zip(batch, outputs)):
                    ids = generated[width:].tolist()
                    stop = next((i + 1 for i, value in enumerate(ids) if value in eos), len(ids))
                    ids = ids[:stop]
                    record = dict(case_id=case["case_id"], group_id=case["group_id"], task=case["task"],
                        model=args.model, arm=args.model if args.model in ("A", "B") else None,
                        adapter_sha256=adapter_hash, raw_output=tok.decode(ids, skip_special_tokens=True),
                        generated_tokens=len(ids), prompt_tokens=lengths[index],
                        finish_reason="eos" if ids and ids[-1] in eos else "length",
                        generation_batch_size=len(batch), batch_elapsed_seconds=elapsed,
                        generation_rank=rank, world_size=world_size, status="completed")
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                print(json.dumps(dict(model=args.model, suite=suite, rank=rank, completed=start + len(batch), total=len(cases))), flush=True)
    hook.remove()
    save(args.output_dir / f"GENERATION_COMPLETE{suffix}.json", dict(model=args.model,
        generation_protocol_sha256=digest(metadata_path), predictions_sha256={p.name: digest(p) for p in output_paths}))


def score(args, manifest):
    from maven_final import contracts
    from maven_final.legacy_evaluation import legacy_strict
    from maven_final._vendor import field_delivery_scorer as legacy
    pinned(ROOT / "src/maven_final/_vendor/field_delivery_scorer.py", manifest["canonical_field_scorer_sha256"])
    identity = manifest["models"][args.model]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scored = []
    summaries = []
    for suite in args.suites:
        cases = {r["case_id"]: r for r in suite_inputs(suite, manifest)}
        paths = raw_paths(args.prediction_dir, suite, identity["world_size"])
        predictions = [r for path in paths for r in rows(path)]
        require(len(predictions) == len(cases) and {r["case_id"] for r in predictions} == set(cases), "Missing or duplicated first answers")
        for r in predictions:
            require(r.get("model", r.get("arm")) == args.model, "Prediction model identity mismatch")
            require(r.get("adapter_sha256") == identity["adapter_sha256"], "Prediction adapter identity mismatch")
            require(isinstance(r.get("raw_output"), str), "Prediction lacks an answer string")
        # The full answer set is loaded and verified before gold is opened.
        gold_path = DATA / ("dev_gold.jsonl" if suite == "dev" else "regression39_rules.jsonl")
        pinned(gold_path, manifest["files"][gold_path.relative_to(ROOT).as_posix()]["sha256"])
        gold = {r["case_id"]: r for r in rows(gold_path)}
        subset = []
        for pred in predictions:
            cid = pred["case_id"]
            case, truth = cases[cid], gold[cid]
            rec = dict(model=args.model, suite=suite, case_id=cid, group_id=case["group_id"], task=case["task"])
            if suite == "dev":
                payload = truth.get("scoring_input") if case.get("prompt_view") == "text_evidence" else case["input"]
                result = contracts.score_response(case["task"], payload, pred["raw_output"])
                rec.update({key: result[key] for key in ("semantic_success", "json_valid", "schema_valid", "dangerous_acceptance", "observed_action")})
                rec["expected_action"] = truth["expected"]["action"]
            else:
                result = legacy.score_case(truth["case"], dict(model=args.model, status="completed", answer=pred["raw_output"]))
                rec.update(semantic_success=legacy_strict(truth["case"], pred["raw_output"]), field_delivery_score=result["delivery_score"])
            subset.append(rec)
        scored.extend(subset)
        summary = dict(model=args.model, suite=suite, cases=len(subset), strict_success=sum(r["semantic_success"] for r in subset),
            generated_tokens=sum(r["generated_tokens"] for r in predictions), prediction_files_sha256={p.name: digest(p) for p in paths})
        if suite == "dev":
            summary.update(false_advance=sum(r["dangerous_acceptance"] for r in subset),
                nonrun_denominator=sum(r["expected_action"] != "run" for r in subset),
                correct_run=sum(r["semantic_success"] and r["observed_action"] == "run" for r in subset),
                gold_run_denominator=sum(r["expected_action"] == "run" for r in subset))
        else:
            summary["legacy_field_score"] = sum(r["field_delivery_score"] for r in subset) / len(subset)
        summaries.append(summary)
    if args.check_recorded:
        check_recorded(args.model, summaries)
    with (args.output_dir / "scored_cases.jsonl").open("x", encoding="utf-8") as stream:
        for r in scored:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    save(args.output_dir / "summary.json", dict(model=args.model, suites=summaries,
        matches_recorded=args.check_recorded, mode="rescored_first_answers", scope="development_and_regression_not_unseen_test"))
    print(json.dumps(summaries, indent=2))


def check_recorded(model, summaries):
    table = ROOT / ("results/retention_ablation/development_summary.csv" if model in ("A", "B") else "results/aggregate/development_summary.csv")
    with table.open(encoding="utf-8-sig", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    for s in summaries:
        if model in ("A", "B"):
            expected = next(r for r in recorded if r["arm"] == model and r["suite"] == s["suite"])
            pairs = {key: key for key in ("cases", "strict_success", "false_advance", "nonrun_denominator", "correct_run", "gold_run_denominator", "legacy_field_score") if key in s}
            pairs["generated_tokens"] = "evaluation_generated_tokens"
        else:
            expected = next(r for r in recorded if r["model"] == model)
            pairs = {"cases": "development_cases", "strict_success": "semantic_success", "false_advance": "false_advance", "nonrun_denominator": "nonrun_cases", "correct_run": "valid_run_success", "gold_run_denominator": "valid_run_cases"} if s["suite"] == "dev" else {"cases": "old_cases", "strict_success": "old_strict_success", "legacy_field_score": "old_field_score"}
        for key, column in pairs.items():
            require(math.isclose(s[key], float(expected[column]), rel_tol=1e-12, abs_tol=1e-12), f"Recorded result differs: {model}/{s['suite']}/{key}: {s[key]} vs {expected[column]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "score"), required=True)
    parser.add_argument("--model", choices=("base", "old", "candidate", "A", "B"), required=True)
    parser.add_argument("--suites", nargs="+", choices=("dev", "legacy39"), default=["dev", "legacy39"])
    parser.add_argument("--base", type=Path, default=ROOT / "models/Qwen3-8B")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/tokenizer")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-recorded", action="store_true", help="Require every score to match the saved manuscript table")
    args = parser.parse_args()
    if args.mode == "score" and args.prediction_dir is None:
        parser.error("--prediction-dir is required for scoring")
    if len(args.suites) != len(set(args.suites)):
        parser.error("A suite cannot be requested twice")
    manifest = load(DATA / "manifest.json")
    (generate if args.mode == "generate" else score)(args, manifest)


if __name__ == "__main__":
    main()
