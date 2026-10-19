from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache

from .recovery import CORE


FEATURE_VERSION = "orientation_corrected_20260908"


FEATURE_ORDER = (
    "core_neoepitope_integrity",
    "core_c_terminal_arg",
    "core_boundary_fidelity",
    "module_order",
    "linker_separation",
    "helper_platform_prior",
    "helper_balance",
    "junction_compatibility",
    "terminal_presentation",
    "developability",
)

# These are researcher-chosen mechanism priors, not fitted coefficients or
# experimentally calibrated uncertainties. This current-version verification
# cannot by itself establish the chronology or blinding of the original discovery.
_WEIGHT_PRIORS = {
    "core_neoepitope_integrity": (0.205, 0.025),
    "core_c_terminal_arg": (0.080, 0.018),
    "core_boundary_fidelity": (0.100, 0.020),
    "module_order": (0.095, 0.020),
    "linker_separation": (0.120, 0.025),
    "helper_platform_prior": (0.145, 0.030),
    "helper_balance": (0.055, 0.018),
    "junction_compatibility": (0.055, 0.018),
    "terminal_presentation": (0.080, 0.025),
    "developability": (0.065, 0.020),
}

_METHOD_ABLATIONS = {
    "mavenpep21_strict": frozenset(),
    "mavenpep21_no_uncertainty": frozenset(),
    "mavenpep21_ablate_modularity": frozenset(
        {"core_boundary_fidelity", "module_order", "junction_compatibility"}
    ),
    "mavenpep21_ablate_termini": frozenset(
        {"core_c_terminal_arg", "terminal_presentation"}
    ),
    "mavenpep21_ablate_core_mechanism": frozenset(
        {"core_neoepitope_integrity", "core_c_terminal_arg", "core_boundary_fidelity"}
    ),
    "mavenpep21_ablate_helper_linker": frozenset(
        {"helper_platform_prior", "helper_balance", "linker_separation"}
    ),
}


def _fraction(sequence: str, residues: set[str]) -> float:
    return 0.0 if not sequence else sum(aa in residues for aa in sequence) / len(sequence)


def _longest_hydrophobic_run(sequence: str) -> int:
    hydrophobic = set("AILMFWVY")
    longest = 0
    current = 0
    for residue in sequence:
        if residue in hydrophobic:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _core_integrity(core: str) -> float:
    if core == CORE:
        return 1.0
    if core.endswith(CORE):
        extra = len(core) - len(CORE)
        return max(0.72, 0.92 - 0.04 * extra)
    if len(core) == len(CORE):
        identity = sum(left == right for left, right in zip(core, CORE, strict=True)) / len(CORE)
        return 0.74 * identity
    if core in (CORE[1:], CORE[:-1]):
        return 0.45
    return 0.0


def _core_boundary(core: str) -> float:
    if core == CORE:
        return 1.0
    if core.endswith(CORE):
        return max(0.45, 0.86 - 0.08 * (len(core) - len(CORE)))
    if len(core) == len(CORE):
        return 0.78
    return 0.40


def _helper_balance(helper: str) -> float:
    hydrophobic = _fraction(helper, set("AILMFWVY"))
    aromatic = _fraction(helper, set("FWY"))
    positive = sum(aa in "KR" for aa in helper)
    negative = sum(aa in "DE" for aa in helper)
    charge_density = abs(positive - negative) / max(len(helper), 1)
    hydrophobic_penalty = max(0.0, abs(hydrophobic - 0.42) - 0.18) * 1.8
    aromatic_penalty = max(0.0, aromatic - 0.18) * 1.5
    charge_penalty = max(0.0, charge_density - 0.24) * 1.5
    run_penalty = max(0, _longest_hydrophobic_run(helper) - 4) * 0.04
    return max(
        0.0,
        min(1.0, 1.0 - hydrophobic_penalty - aromatic_penalty - charge_penalty - run_penalty),
    )


def _linker_separation(linker: str, prior: float) -> float:
    flexible = _fraction(linker, set("GPS"))
    charged = _fraction(linker, set("KRDE"))
    neutral_score = max(0.0, 1.0 - 1.5 * charged)
    # Some flexibility separates the epitopes; maximal Gly/Pro/Ser content is
    # not assumed to be monotonically better because it can also permit the
    # helper to fold back onto the short core.  GPSL-like intermediate
    # flexibility is therefore the broad optimum, with the empirical platform
    # prior carrying most of the evidence.
    intermediate_flexibility = max(0.0, 1.0 - 2.0 * abs(flexible - 0.5))
    return max(
        0.0,
        min(1.0, 0.75 * prior + 0.15 * intermediate_flexibility + 0.10 * neutral_score),
    )


def _junction_compatibility(helper: str, linker: str, core: str) -> float:
    # A sequence-only shielding proxy: long hydrophobic/charged runs crossing a
    # boundary are penalized, while a flexible, non-polybasic linker is favoured.
    left = helper[-3:] + linker[:3]
    right = linker[-3:] + core[:3]
    hydro_run = max(_longest_hydrophobic_run(left), _longest_hydrophobic_run(right))
    polybasic = max(left.count("K") + left.count("R"), right.count("K") + right.count("R"))
    flexible = _fraction(linker, set("GPS"))
    score = 0.72 + 0.22 * flexible - 0.06 * max(0, hydro_run - 2) - 0.05 * max(0, polybasic - 2)
    return max(0.0, min(1.0, score))


def _terminal_presentation(n_chemistry: str, c_chemistry: str) -> float:
    # The free N / amidated C state receives the highest manufacturability prior
    # but only a small total weight.  Free C remains close because it better
    # mimics the native C5a carboxylate and is an explicit experimental control.
    states = {
        ("free", "amide"): 1.00,
        ("free", "free"): 0.93,
        ("acetyl", "amide"): 0.80,
        ("acetyl", "free"): 0.75,
    }
    return states.get((n_chemistry, c_chemistry), 0.0)


def extract_modular_features(candidate: dict[str, object]) -> dict[str, float]:
    """Represent a vaccine as modules and boundaries, never as one anonymous string."""

    # Validate before min/max: Python clamps NaN to an apparently perfect score.
    # These checks change no formula or value in the frozen valid candidate pool.
    for name in ("helper_prior", "linker_prior", "developability_proxy"):
        value = candidate[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite numeric prior")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and within 0..1")
    if candidate["n_chemistry"] not in {"free", "acetyl"}:
        raise ValueError("Unknown N-terminal chemistry")
    if candidate["c_chemistry"] not in {"free", "amide"}:
        raise ValueError("Unknown C-terminal chemistry")
    helper = str(candidate["helper_sequence"])
    linker = str(candidate["linker_sequence"])
    core = str(candidate["core_sequence"])
    orientation = candidate["orientation"]
    if orientation not in {"helper_to_b", "b_to_helper"}:
        raise ValueError(f"Unknown module orientation: {orientation}")
    first, last = (helper, core) if orientation == "helper_to_b" else (core, helper)
    if str(candidate["sequence"]) != first + linker + last:
        raise ValueError("Sequence and declared modules/orientation disagree")
    features = {
        "core_neoepitope_integrity": _core_integrity(core),
        "core_c_terminal_arg": 1.0 if core.endswith("R") else 0.0,
        "core_boundary_fidelity": _core_boundary(core),
        "module_order": 1.0 if candidate["orientation"] == "helper_to_b" else 0.45,
        "linker_separation": _linker_separation(linker, float(candidate["linker_prior"])),
        "helper_platform_prior": max(0.0, min(1.0, float(candidate["helper_prior"]))),
        "helper_balance": _helper_balance(helper),
        "junction_compatibility": _junction_compatibility(first, linker, last),
        "terminal_presentation": _terminal_presentation(
            str(candidate["n_chemistry"]), str(candidate["c_chemistry"])
        ),
        "developability": max(0.0, min(1.0, float(candidate["developability_proxy"]))),
    }
    return {name: float(features[name]) for name in FEATURE_ORDER}


def _candidate_signature(candidate: dict[str, object]) -> str:
    fields = (
        str(candidate["sequence"]),
        str(candidate["helper_sequence"]),
        str(candidate["linker_sequence"]),
        str(candidate["core_sequence"]),
        str(candidate["orientation"]),
        str(candidate["n_chemistry"]),
        str(candidate["c_chemistry"]),
    )
    return "\x1f".join(fields)


@lru_cache(maxsize=None)
def _weight_draws(
    method: str, seed: int, count: int = 192
) -> tuple[dict[str, float], ...]:
    ablated = _METHOD_ABLATIONS[method]
    # Common random numbers isolate each ablation from Monte Carlo variation.
    # Keep the historical strict seed so the primary ranking is reproducible.
    digest = hashlib.sha256(f"mavenpep21\x1fmavenpep21_strict\x1f{seed}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(local_seed)
    draws: list[dict[str, float]] = []
    for _ in range(count):
        sampled = {
            name: max(0.002, rng.gauss(mean, std))
            for name, (mean, std) in _WEIGHT_PRIORS.items()
        }
        raw = {name: 0.0 if name in ablated else value for name, value in sampled.items()}
        total = sum(raw.values())
        draws.append({name: value / total for name, value in raw.items()})
    return tuple(draws)


def _score_candidate(
    candidate: dict[str, object], method: str, seed: int
) -> tuple[float, float, float, dict[str, float]]:
    features = extract_modular_features(candidate)
    draws = _weight_draws(method, seed)
    scores = [sum(weights[name] * features[name] for name in FEATURE_ORDER) for weights in draws]
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / max(len(scores) - 1, 1)
    std = math.sqrt(variance)
    if method == "mavenpep21_no_uncertainty":
        score = mean
    else:
        # A prior-spread penalty, not a statistical confidence bound on efficacy.
        score = mean - 0.25 * std
    return score, mean, std, features


def rank_mavenpep21(
    candidates: list[dict[str, object]], method: str, seed: int
) -> list[dict[str, object]]:
    """Rank one frozen pool without accepting target identities or effect labels."""

    if method not in _METHOD_ABLATIONS:
        raise ValueError(f"Unsupported MAVEN-Pep 2.1 method: {method}")
    scored: list[dict[str, object]] = []
    for candidate in candidates:
        score, mean, std, features = _score_candidate(candidate, method, seed)
        scored.append(
            {
                **candidate,
                "rank_method": method,
                "ranking_score": score,
                "score_mean": mean,
                "score_std": std,
                "score_lcb": score,
                "mavenpep21_features": features,
                "experimental_feedback_used": False,
                "strict_target_blind": False,
                "runtime_target_labels_unused": True,
                "score_uncertainty_scope": "researcher_prior_weight_sensitivity_not_predictive_interval",
            }
        )
    return sorted(
        scored,
        key=lambda row: (
            -float(row["ranking_score"]),
            str(row["sequence"]),
            str(row["n_chemistry"]),
            str(row["c_chemistry"]),
            str(row["candidate_id"]),
        ),
    )
