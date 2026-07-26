from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable

from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.contracts import ParseFailureClass


@dataclass(frozen=True)
class RepairCandidate:
    candidate_reaction_smiles: str
    repair_type: str
    deterministic: bool
    parse_valid: bool
    canonical_reaction_smiles: str | None
    edit_similarity: float
    route_continuity_score: float
    rank_score: float
    accepted: bool
    rejection_reason: str | None = None


def _delimiter_candidates(text: str) -> Iterable[tuple[str, str]]:
    if text.count(">") == 2 and ">>" not in text:
        left, _, right = text.split(">", 2)
        yield f"{left}>>{right}", "drop_middle_condition_field"
    if text.count(">>") > 1:
        left, *rest = text.split(">>")
        yield f"{left}>>{rest[-1]}", "collapse_duplicate_reaction_delimiters"


def deterministic_repair_candidates(
    reaction_smiles: str,
    *,
    contextual_candidate: str | None = None,
    route_continuity_score: float = 0.0,
) -> list[RepairCandidate]:
    original = str(reaction_smiles or "").strip()
    parsed = parse_reaction(original)
    # A parse-valid reaction does not require repair. Previously, an alternative
    # canonical/contextual representation could be counted as an accepted repair,
    # which contaminated the repairability target with already-valid rows.
    if parsed.parse_ok:
        return []
    raw: list[tuple[str, str, bool]] = []
    if contextual_candidate and contextual_candidate != original:
        raw.append((contextual_candidate, "route_context_intermediate_resolution", True))
    if parsed.failure_class == ParseFailureClass.MALFORMED_DELIMITER:
        raw.extend((candidate, kind, True) for candidate, kind in _delimiter_candidates(original))
    # Common serialization damage: whitespace and escaped tab/newline characters.
    compact = original.replace("\\n", "").replace("\\t", "").replace(" ", "")
    if compact != original:
        raw.append((compact, "remove_serialization_whitespace", True))

    candidates: list[RepairCandidate] = []
    seen: set[str] = set()
    for candidate, kind, deterministic in raw:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        reparsed = parse_reaction(candidate)
        canonical = canonicalize_reaction(candidate) if reparsed.parse_ok else None
        edit_similarity = difflib.SequenceMatcher(a=original, b=candidate).ratio()
        score = 0.60 * float(reparsed.parse_ok) + 0.25 * route_continuity_score + 0.15 * edit_similarity
        accepted = reparsed.parse_ok and score >= 0.80
        candidates.append(
            RepairCandidate(
                candidate_reaction_smiles=candidate,
                repair_type=kind,
                deterministic=deterministic,
                parse_valid=reparsed.parse_ok,
                canonical_reaction_smiles=canonical,
                edit_similarity=edit_similarity,
                route_continuity_score=route_continuity_score,
                rank_score=score,
                accepted=accepted,
                rejection_reason=None if accepted else "Candidate failed strict parse/continuity acceptance.",
            )
        )
    return sorted(candidates, key=lambda item: item.rank_score, reverse=True)
