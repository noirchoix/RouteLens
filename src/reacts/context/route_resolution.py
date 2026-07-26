from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.contracts import ResolutionStatus
from reacts.data.parsing import INTERMEDIATE_RE, parse_list


@dataclass(frozen=True)
class RouteEdge:
    # route_id/step_id remain compatibility aliases for the instance identifiers.
    route_id: str
    source_step_id: str
    target_step_id: str
    source_step_index: int
    target_step_index: int
    intermediate_label: str | None
    connecting_molecules: tuple[str, ...]
    continuity_status: str
    continuity_confidence: float
    edge_evidence_type: str
    structural_identity_available: bool
    label_continuity_score: float
    structural_continuity_score: float
    metadata_consistency_score: float


@dataclass(frozen=True)
class IntermediateResolution:
    route_id: str
    step_id: str
    intermediate_label: str
    side: str
    status: ResolutionStatus
    resolved_molecules: tuple[str, ...]
    evidence_step_id: str | None
    confidence: float
    method: str
    reason: str | None = None
    evidence_status: str = "unresolved_label_only"
    structure_evidence_type: str = "symbolic_only"


@dataclass(frozen=True)
class ResolvedStep:
    source: dict[str, Any]
    original_reaction_smiles: str
    resolved_reaction_smiles: str | None
    canonical_resolved_reaction_smiles: str | None
    resolution_status: ResolutionStatus
    resolution_confidence: float
    resolutions: tuple[IntermediateResolution, ...]


def _concrete(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in values if not INTERMEDIATE_RE.fullmatch(item))


def _reaction_sides(reaction: str) -> tuple[list[str], list[str]]:
    parsed = parse_reaction(reaction)
    return list(parsed.reactants), list(parsed.products)


def _replace_placeholders(values: list[str], bindings: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    output: list[str] = []
    unresolved: list[str] = []
    for value in values:
        label = value.upper() if INTERMEDIATE_RE.fullmatch(value) else None
        if label and label in bindings:
            output.extend(bindings[label])
        else:
            output.append(value)
            if label:
                unresolved.append(label)
    return list(dict.fromkeys(output)), unresolved


class SymbolicIntermediateResolver:
    """Resolve only an explicitly labelled intermediate with observed structure.

    Symbolic labels establish route continuity, not molecular identity. A binding is
    accepted only when the immediately preceding step declares the same output label
    and has a concrete, parse-valid observed product. Product-side placeholders are
    never inferred from later products and no forward model is invoked here.
    """

    def resolve_route(self, rows: list[dict[str, Any]]) -> tuple[list[ResolvedStep], list[RouteEdge]]:
        ordered = sorted(rows, key=lambda row: (int(row.get("step_index", 0)), str(row.get("step_id", ""))))
        bindings: dict[str, tuple[str, ...]] = {}
        binding_evidence: dict[str, str] = {}
        resolved: list[ResolvedStep] = []
        edges: list[RouteEdge] = []
        previous_products: tuple[str, ...] = ()
        previous_step: dict[str, Any] | None = None

        for source in ordered:
            route_id = str(source["route_id"])
            step_id = str(source["step_id"])
            reaction = str(source.get("canonical_reaction_smiles") or source.get("reaction_smiles") or "")
            reactants, products = _reaction_sides(reaction)
            input_raw = source.get("input_intermediate")
            output_raw = source.get("output_intermediate")
            input_label = str(input_raw).upper() if input_raw and str(input_raw) != "nan" else None
            output_label = str(output_raw).upper() if output_raw and str(output_raw) != "nan" else None
            step_resolutions: list[IntermediateResolution] = []

            if previous_step is not None:
                previous_output_raw = previous_step.get("output_intermediate")
                previous_output = (
                    str(previous_output_raw).upper()
                    if previous_output_raw and str(previous_output_raw) != "nan"
                    else None
                )
                explicit_match = bool(input_label and previous_output and input_label == previous_output)
                if explicit_match and previous_products:
                    existing = bindings.get(input_label)
                    if existing is None or existing == previous_products:
                        bindings[input_label] = previous_products
                        binding_evidence[input_label] = str(previous_step["step_id"])
                    else:
                        bindings.pop(input_label, None)
                        binding_evidence.pop(input_label, None)
                        step_resolutions.append(
                            IntermediateResolution(
                                route_id=route_id,
                                step_id=step_id,
                                intermediate_label=input_label,
                                side="reactant",
                                status=ResolutionStatus.AMBIGUOUS,
                                resolved_molecules=(),
                                evidence_step_id=str(previous_step["step_id"]),
                                confidence=0.0,
                                method="conflicting_explicit_route_anchors",
                                reason="The same intermediate label maps to conflicting observed products.",
                                evidence_status="ambiguous_multiple_anchors",
                                structure_evidence_type="symbolic_only",
                            )
                        )

                edge_label = input_label or previous_output
                edge_molecules = bindings.get(edge_label, ()) if edge_label else ()
                if explicit_match and edge_molecules:
                    edge_status, confidence = "resolved", 1.0
                    evidence_type, structural = "explicit_structural_identity", True
                    label_score, structural_score, metadata_score = 1.0, 1.0, 1.0
                elif explicit_match:
                    edge_status, confidence = "label_only", 0.45
                    evidence_type, structural = "exact_symbolic_label", False
                    label_score, structural_score, metadata_score = 1.0, 0.0, 1.0
                elif edge_label:
                    edge_status, confidence = "metadata_mismatch", 0.0
                    evidence_type, structural = "ambiguous_metadata", False
                    label_score, structural_score, metadata_score = 0.0, 0.0, 0.0
                else:
                    edge_status, confidence = "adjacent_without_label", 0.25
                    evidence_type, structural = "adjacent_sequence", False
                    label_score, structural_score, metadata_score = 0.0, 0.0, 0.25
                edges.append(
                    RouteEdge(
                        route_id=route_id,
                        source_step_id=str(previous_step["step_id"]),
                        target_step_id=step_id,
                        source_step_index=int(previous_step.get("step_index", 0)),
                        target_step_index=int(source.get("step_index", 0)),
                        intermediate_label=edge_label,
                        connecting_molecules=edge_molecules,
                        continuity_status=edge_status,
                        continuity_confidence=confidence,
                        edge_evidence_type=evidence_type,
                        structural_identity_available=structural,
                        label_continuity_score=label_score,
                        structural_continuity_score=structural_score,
                        metadata_consistency_score=metadata_score,
                    )
                )

            if input_label and input_label in bindings:
                bound = bindings[input_label]
                reactants = list(dict.fromkeys([*bound, *reactants]))
                step_resolutions.append(
                    IntermediateResolution(
                        route_id=route_id,
                        step_id=step_id,
                        intermediate_label=input_label,
                        side="reactant",
                        status=ResolutionStatus.RESOLVED,
                        resolved_molecules=bound,
                        evidence_step_id=binding_evidence.get(input_label),
                        confidence=1.0,
                        method="explicit_adjacent_output_input_anchor",
                        evidence_status="resolved_observed_structure",
                        structure_evidence_type="observed_step_structure",
                    )
                )

            reactants, unresolved_left = _replace_placeholders(reactants, bindings)
            products, unresolved_right = _replace_placeholders(products, bindings)
            for label in unresolved_left:
                step_resolutions.append(
                    IntermediateResolution(
                        route_id=route_id,
                        step_id=step_id,
                        intermediate_label=label,
                        side="reactant",
                        status=ResolutionStatus.UNRESOLVED,
                        resolved_molecules=(),
                        evidence_step_id=None,
                        confidence=0.0,
                        method="no_unique_structural_anchor",
                        reason="The symbolic label has no observed route-local molecular structure.",
                        evidence_status="unresolved_label_only",
                        structure_evidence_type="symbolic_only",
                    )
                )
            for label in unresolved_right:
                step_resolutions.append(
                    IntermediateResolution(
                        route_id=route_id,
                        step_id=step_id,
                        intermediate_label=label,
                        side="product",
                        status=ResolutionStatus.UNRESOLVED,
                        resolved_molecules=(),
                        evidence_step_id=None,
                        confidence=0.0,
                        method="unsupported_product_placeholder",
                        reason="A product-side placeholder cannot be inferred from a later product without a reaction model.",
                        evidence_status="unsupported_product_placeholder",
                        structure_evidence_type="symbolic_only",
                    )
                )

            candidate = ".".join(reactants) + ">>" + ".".join(products) if reactants and products else None
            parsed_candidate = parse_reaction(candidate or "")
            canonical_candidate = canonicalize_reaction(candidate or "") if parsed_candidate.parse_ok else None
            has_resolved = any(item.status == ResolutionStatus.RESOLVED for item in step_resolutions)
            has_unresolved = any(item.status in {ResolutionStatus.UNRESOLVED, ResolutionStatus.AMBIGUOUS} for item in step_resolutions)
            if not step_resolutions:
                status = ResolutionStatus.NOT_REQUIRED
                confidence = 1.0 if parsed_candidate.parse_ok else 0.0
            elif parsed_candidate.parse_ok and has_resolved and not has_unresolved:
                status = ResolutionStatus.RESOLVED
                confidence = min(item.confidence for item in step_resolutions if item.status == ResolutionStatus.RESOLVED)
            elif parsed_candidate.parse_ok and has_resolved:
                status = ResolutionStatus.PARTIALLY_RESOLVED
                confidence = 0.6
            elif has_unresolved:
                status = ResolutionStatus.UNRESOLVED
                confidence = 0.0
            else:
                status = ResolutionStatus.INVALID_AFTER_RESOLUTION
                confidence = 0.0

            resolved.append(
                ResolvedStep(
                    source=source,
                    original_reaction_smiles=reaction,
                    resolved_reaction_smiles=candidate,
                    canonical_resolved_reaction_smiles=canonical_candidate,
                    resolution_status=status,
                    resolution_confidence=confidence,
                    resolutions=tuple(step_resolutions),
                )
            )

            concrete_products = _concrete(products) if parsed_candidate.products_valid else ()
            if output_label and concrete_products and parsed_candidate.parse_ok:
                existing = bindings.get(output_label)
                if existing is None or existing == concrete_products:
                    bindings[output_label] = concrete_products
                    binding_evidence[output_label] = step_id
            previous_products = concrete_products
            previous_step = source

        return resolved, edges
