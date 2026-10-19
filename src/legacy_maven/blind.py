FIELDS = (
    "sequence",
    "helper_sequence",
    "linker_sequence",
    "core_sequence",
    "orientation",
    "n_chemistry",
    "c_chemistry",
    "helper_prior",
    "linker_prior",
    "helper_uncertainty",
    "linker_uncertainty",
    "developability_proxy",
    "core_similarity",
)


def signature(candidate):
    return tuple(str(candidate[k]) for k in ("sequence", "orientation", "n_chemistry", "c_chemistry"))


def blind_library(candidates):
    sanitized = [{k: row[k] for k in FIELDS} for row in candidates]
    sanitized.sort(key=signature)
    if len({signature(row) for row in sanitized}) != len(sanitized):
        raise ValueError("Duplicate complete construct identities in candidate pool")
    for index, row in enumerate(sanitized, 1):
        row["candidate_id"] = f"C{index:05d}"
    return sanitized
