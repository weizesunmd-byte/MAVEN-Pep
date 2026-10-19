"""Scoped identity checks for optional supplemental adapters only.

Recognizes the two known MAX identities, contextual candidate/peptide labels,
and exact numbers in explicitly historical identity fields.  NFKC handles
fullwidth forms.  It does not infer identity from sequence, erase measurements,
or claim general de-identification against arbitrary spelling/encoding attacks.
The original package's regex is never changed.
"""

from __future__ import annotations

import re
import unicodedata

HISTORICAL_FIELDS = {
    "identity", "historical_identity", "historical_alias", "legacy_identity",
    "legacy_id", "reference_identity", "candidate_identity", "peptide_identity",
    "historical_candidate_id", "historical_peptide_id", "candidate_id", "peptide_id",
}
_SEP = r"[\s_:#/\-–—]*"
_NUMBER = rf"(?:4{_SEP}4{_SEP}9|4{_SEP}5{_SEP}0)"
_KNOWN = re.compile(rf"(?<![A-Za-z0-9])m{_SEP}a{_SEP}x{_SEP}{_NUMBER}(?![A-Za-z0-9])", re.I)
_CONTEXT = re.compile(
    rf"(?<![A-Za-z0-9])(?:candidate|peptide|候选肽|候选|肽)(?:{_SEP}(?:id|no\.?|number|编号|序号))?{_SEP}"
    rf"(?:449|450)(?![A-Za-z0-9])", re.I,
)
_EXACT = {"449", "450"}


def normalized(text):
    return unicodedata.normalize("NFKC", text)


def identity_hits(value):
    """Return scoped detections with structured paths; numeric assays are retained."""
    hits = []

    def visit(item, path, historical=False):
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = normalized(str(key))
                visit(key_text, path + ".<key>")
                field = key_text.strip().casefold().replace("-", "_").replace(" ", "_")
                visit(child, path + "." + key_text, historical or field in HISTORICAL_FIELDS)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", historical)
        elif isinstance(item, str):
            text = normalized(item)
            reason = "known_MAX_identity" if _KNOWN.search(text) else (
                "contextual_candidate_or_peptide_identity" if _CONTEXT.search(text) else None
            )
            if reason is None and historical and text.strip() in _EXACT:
                reason = "exact_number_in_historical_identity_field"
            if reason:
                hits.append({"path": path, "reason": reason})
        elif historical and type(item) in (int, float) and item in (449, 450):
            hits.append({"path": path, "reason": "exact_number_in_historical_identity_field"})

    visit(value, "$")
    return hits


def assert_runtime_blinded(value):
    """Reject a detected known identity; return unchanged input otherwise."""
    hits = identity_hits(value)
    if hits:
        raise ValueError(f"Known historical identity detected at {hits[0]['path']} ({hits[0]['reason']})")
    return value


def scrub_runtime_evidence(records):
    """Replace identity-bearing evidence records, retaining clean measurements."""
    result = []
    for item in records:
        if identity_hits(item):
            result.append({"source": "supplemental_identity_policy", "entity": "redacted",
                           "evidence_type": "identity_redaction", "value": None, "confidence": None,
                           "reference": "", "status": "withheld",
                           "scope": "known historical identity withheld before supplemental planning"})
        else:
            result.append(item)
    return result
