from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable


CORE = "KDMQLGR"


def _mutate(sequence: str, replacements: dict[int, str]) -> str:
    residues = list(sequence)
    for index, residue in replacements.items():
        residues[index] = residue
    return "".join(residues)


def _core_modules() -> list[dict[str, object]]:
    modules: list[dict[str, object]] = [
        {"module_id": "core_kdmqlgr", "sequence": CORE, "family": "core_retained"},
        {"module_id": "core_trunc_n1", "sequence": "DMQLGR", "family": "truncation_probe"},
        {"module_id": "core_trunc_c1", "sequence": "KDMQLG", "family": "truncation_probe"},
        {"module_id": "core_flank_c", "sequence": "CKDMQLGR", "family": "flank_optimized"},
        {"module_id": "core_flank_sh", "sequence": "SHKDMQLGR", "family": "flank_optimized"},
        {"module_id": "core_flank_nish", "sequence": "NISHKDMQLGR", "family": "flank_optimized"},
        {"module_id": "core_flank_canish", "sequence": "CANISHKDMQLGR", "family": "flank_optimized"},
    ]
    single_substitutions = {
        0: ("R",),
        1: ("E", "N"),
        2: ("L", "I"),
        3: ("N", "E"),
        4: ("I", "V"),
        5: ("A", "S"),
        6: ("K",),
    }
    for index, alternatives in single_substitutions.items():
        for residue in alternatives:
            modules.append(
                {
                    "module_id": f"core_single_{index + 1}_{residue.lower()}",
                    "sequence": _mutate(CORE, {index: residue}),
                    "family": "conservative_mimotope",
                }
            )
    double_substitutions = (
        {0: "R", 1: "E"},
        {1: "E", 2: "L"},
        {2: "L", 3: "N"},
        {3: "N", 4: "I"},
        {4: "I", 5: "A"},
        {5: "A", 6: "K"},
        {0: "R", 6: "K"},
        {1: "N", 3: "E"},
        {2: "I", 4: "V"},
        {3: "E", 5: "S"},
        {0: "R", 2: "L"},
    )
    for number, replacements in enumerate(double_substitutions, start=1):
        modules.append(
            {
                "module_id": f"core_double_{number:02d}",
                "sequence": _mutate(CORE, replacements),
                "family": "conservative_mimotope",
            }
        )
    if len(modules) != 30 or len({row["sequence"] for row in modules}) != 30:
        raise AssertionError("Default core library must contain 30 unique modules")
    return modules


def build_default_catalog() -> dict[str, object]:
    """Return the current frozen catalog for modular ranking verification.

    The original algorithm selected MAX449 before the wet experiments (project
    team-confirmed chronology). Re-running this catalog verifies the current
    implementation; it does not reconstruct that original discovery run.
    Historical identifiers are retained only for file/API compatibility.
    """

    return {
        "schema": "mavenpep2.public-component-catalog.v1",
        "catalog_scope": "construction_aware_retrospective",
        "helpers": [
            {
                "module_id": "helper_mvf_max449_external",
                "sequence": "KLLSLIKGVIVHRLEGVE",
                "prior": 0.85,
                "uncertainty": 0.35,
            },
            {
                "module_id": "helper_max450_external",
                "sequence": "FNNFTVSFWLRVPKVSASHL",
                "prior": 0.75,
                "uncertainty": 0.45,
            },
            {
                "module_id": "helper_padre_public",
                "sequence": "AKFVAAWTLKAAA",
                "prior": 0.80,
                "uncertainty": 0.35,
            },
            {
                "module_id": "helper_tt_830_844_public",
                "sequence": "QYIKANSKFIGITEL",
                "prior": 0.70,
                "uncertainty": 0.40,
            },
        ],
        "linkers": [
            {"module_id": "linker_gpsl", "sequence": "GPSL", "prior": 0.85, "uncertainty": 0.30},
            {"module_id": "linker_gpgpg", "sequence": "GPGPG", "prior": 0.80, "uncertainty": 0.35},
            {"module_id": "linker_ggggs", "sequence": "GGGGS", "prior": 0.75, "uncertainty": 0.40},
            {"module_id": "linker_eaaak", "sequence": "EAAAK", "prior": 0.70, "uncertainty": 0.45},
            {"module_id": "linker_kk", "sequence": "KK", "prior": 0.40, "uncertainty": 0.60},
        ],
        "cores": _core_modules(),
        "orientations": ["helper_to_b", "b_to_helper"],
        "terminal_chemistries": [
            {"n_chemistry": "free", "c_chemistry": "free"},
            {"n_chemistry": "free", "c_chemistry": "amide"},
            {"n_chemistry": "acetyl", "c_chemistry": "free"},
            {"n_chemistry": "acetyl", "c_chemistry": "amide"},
        ],
    }


def _candidate_id(parts: Iterable[str]) -> str:
    payload = "\x1f".join(parts)
    return f"mp2_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _identity_to_core(sequence: str) -> float:
    if CORE in sequence:
        return 1.0
    if len(sequence) == len(CORE):
        return sum(a == b for a, b in zip(sequence, CORE, strict=True)) / len(CORE)
    if sequence == CORE[1:] or sequence == CORE[:-1]:
        return 6 / 7
    return 0.0


def _developability(sequence: str) -> float:
    hydrophobic = set("AILMFWVY")
    charged_positive = set("KR")
    charged_negative = set("DE")
    hydrophobic_fraction = sum(aa in hydrophobic for aa in sequence) / len(sequence)
    net_charge_proxy = sum(aa in charged_positive for aa in sequence) - sum(
        aa in charged_negative for aa in sequence
    )
    length_penalty = min(abs(len(sequence) - 34) / 34, 1.0)
    hydrophobic_penalty = max(0.0, hydrophobic_fraction - 0.55) * 2
    charge_penalty = max(0.0, abs(net_charge_proxy) - 7) / 10
    return max(0.0, 1.0 - length_penalty * 0.35 - hydrophobic_penalty - charge_penalty)


def generate_search_universe(catalog: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for helper in catalog["helpers"]:  # type: ignore[index]
        for linker in catalog["linkers"]:  # type: ignore[index]
            for core in catalog["cores"]:  # type: ignore[index]
                for orientation in catalog["orientations"]:  # type: ignore[index]
                    for chemistry in catalog["terminal_chemistries"]:  # type: ignore[index]
                        if orientation == "helper_to_b":
                            sequence = helper["sequence"] + linker["sequence"] + core["sequence"]
                        else:
                            sequence = core["sequence"] + linker["sequence"] + helper["sequence"]
                        candidate_id = _candidate_id(
                            (
                                str(helper["module_id"]),
                                str(linker["module_id"]),
                                str(core["module_id"]),
                                str(orientation),
                                str(chemistry["n_chemistry"]),
                                str(chemistry["c_chemistry"]),
                            )
                        )
                        candidates.append(
                            {
                                "candidate_id": candidate_id,
                                "sequence": sequence,
                                "helper_id": helper["module_id"],
                                "helper_sequence": helper["sequence"],
                                "linker_id": linker["module_id"],
                                "linker_sequence": linker["sequence"],
                                "core_id": core["module_id"],
                                "core_sequence": core["sequence"],
                                "core_family": core["family"],
                                "orientation": orientation,
                                "n_chemistry": chemistry["n_chemistry"],
                                "c_chemistry": chemistry["c_chemistry"],
                                "core_similarity": _identity_to_core(str(core["sequence"])),
                                "helper_prior": helper["prior"],
                                "helper_uncertainty": helper["uncertainty"],
                                "linker_prior": linker["prior"],
                                "linker_uncertainty": linker["uncertainty"],
                                "developability_proxy": _developability(sequence),
                                "construction_aware": True,
                                "experimental_feedback_used": False,
                            }
                        )
    candidates.sort(key=lambda row: str(row["candidate_id"]))
    return candidates


def _kmer_set(sequence: str, k: int = 2) -> set[str]:
    return {sequence[index : index + k] for index in range(len(sequence) - k + 1)}


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else 1.0 - len(left & right) / len(union)


def _scored_copy(candidate: dict[str, object], method: str, score: float) -> dict[str, object]:
    return {**candidate, "rank_method": method, "ranking_score": float(score)}


def rank_by_method(
    candidates: list[dict[str, object]], method: str, seed: int
) -> list[dict[str, object]]:
    """Rank a frozen candidate pool without accepting target or wet-lab labels."""

    if method == "random":
        rng = random.Random(seed)
        scored = [_scored_copy(row, method, rng.random()) for row in candidates]
        return sorted(scored, key=lambda row: (-float(row["ranking_score"]), str(row["candidate_id"])))

    if method == "diversity":
        kmers = {str(row["candidate_id"]): _kmer_set(str(row["sequence"])) for row in candidates}
        remaining = {str(row["candidate_id"]): row for row in candidates}
        first = max(
            candidates,
            key=lambda row: (float(row["developability_proxy"]), str(row["candidate_id"])),
        )
        selected = [first]
        remaining.pop(str(first["candidate_id"]))
        while remaining and len(selected) < 32:
            selected_ids = [str(row["candidate_id"]) for row in selected]
            next_row = max(
                remaining.values(),
                key=lambda row: (
                    min(
                        _jaccard_distance(kmers[str(row["candidate_id"])], kmers[selected_id])
                        for selected_id in selected_ids
                    ),
                    float(row["developability_proxy"]),
                    str(row["candidate_id"]),
                ),
            )
            selected.append(next_row)
            remaining.pop(str(next_row["candidate_id"]))
        ranked = [
            _scored_copy(row, method, float(len(selected) - index))
            for index, row in enumerate(selected)
        ]
        ranked.extend(
            _scored_copy(row, method, 0.0)
            for row in sorted(remaining.values(), key=lambda item: str(item["candidate_id"]))
        )
        return ranked

    scored: list[dict[str, object]] = []
    for row in candidates:
        core = float(row["core_similarity"])
        helper = float(row["helper_prior"])
        linker = float(row["linker_prior"])
        developability = float(row["developability_proxy"])
        orientation = 1.0 if row["orientation"] == "helper_to_b" else 0.65
        chemistry = 0.70 if row["c_chemistry"] == "amide" else 0.50
        if method == "rule_based":
            score = 0.40 * core + 0.20 * helper + 0.20 * linker + 0.10 * orientation + 0.10 * developability
        elif method == "fixed_weight":
            score = 0.30 * core + 0.20 * helper + 0.15 * linker + 0.20 * developability + 0.10 * orientation + 0.05 * chemistry
        elif method == "mavenpep2_no_feedback":
            uncertainty = (
                (1.0 - core) + float(row["helper_uncertainty"]) + float(row["linker_uncertainty"])
            ) / 3
            prior_mean = 0.34 * core + 0.20 * helper + 0.18 * linker + 0.18 * developability + 0.10 * orientation
            score = prior_mean - 0.15 * uncertainty
        elif method == "ablation_no_core":
            uncertainty = (float(row["helper_uncertainty"]) + float(row["linker_uncertainty"])) / 2
            score = 0.30 * helper + 0.30 * linker + 0.25 * developability + 0.15 * orientation - 0.15 * uncertainty
        elif method == "ablation_no_helper":
            uncertainty = ((1.0 - core) + float(row["linker_uncertainty"])) / 2
            score = 0.45 * core + 0.25 * linker + 0.20 * developability + 0.10 * orientation - 0.15 * uncertainty
        elif method == "ablation_no_linker":
            uncertainty = ((1.0 - core) + float(row["helper_uncertainty"])) / 2
            score = 0.45 * core + 0.25 * helper + 0.20 * developability + 0.10 * orientation - 0.15 * uncertainty
        else:
            raise ValueError(f"Unsupported recovery method: {method}")
        scored.append(_scored_copy(row, method, score))
    return sorted(scored, key=lambda row: (-float(row["ranking_score"]), str(row["candidate_id"])))


def evaluate_target(
    ranked_candidates: list[dict[str, object]], target: dict[str, str]
) -> dict[str, object]:
    exact = [
        row
        for row in ranked_candidates
        if row["sequence"] == target["sequence"]
        and row["n_chemistry"] == target["n_chemistry"]
        and row["c_chemistry"] == target["c_chemistry"]
    ]
    result: dict[str, object] = {
        "target_id": target["target_id"],
        "target_in_universe": bool(exact),
        "exact_match_count": len(exact),
        "candidate_count": len(ranked_candidates),
    }
    if exact:
        matched_id = exact[0]["candidate_id"]
        rank = next(
            index
            for index, row in enumerate(ranked_candidates, start=1)
            if row["candidate_id"] == matched_id
        )
        result.update(
            {
                "rank": rank,
                "rank_percentile": 1.0 - (rank - 1) / max(len(ranked_candidates) - 1, 1),
                "top_10": rank <= 10,
                "top_32": rank <= 32,
                "top_1_percent": rank <= max(1, len(ranked_candidates) // 100),
            }
        )
        # Identifier-based tie breaking is convenient for reproducibility but
        # must not imply evidence for an exact position within a tied block.
        if "ranking_score" in exact[0]:
            score = float(exact[0]["ranking_score"])
            scores = [float(row["ranking_score"]) for row in ranked_candidates]
            tied = sum(math.isclose(value, score, rel_tol=1e-12, abs_tol=1e-12) for value in scores)
            better = sum(value > score and not math.isclose(value, score, rel_tol=1e-12, abs_tol=1e-12) for value in scores)
            result.update({
                "rank_best": better + 1,
                "rank_worst": better + tied,
                "rank_midrank": better + (tied + 1) / 2,
                "score_tie_count": tied,
                "top_10_tie_probability": min(max(10 - better, 0), tied) / tied,
                "top_32_tie_probability": min(max(32 - better, 0), tied) / tied,
                "rank_interpretation": "deterministic_tie_break_not_unique_evidence" if tied > 1 else "unique_score_position",
            })
    return result
