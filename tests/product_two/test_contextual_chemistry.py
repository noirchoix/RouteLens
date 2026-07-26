from reacts.chemistry.mapping import AtomMappingEngine, derive_reaction_centre
from reacts.context.route_resolution import SymbolicIntermediateResolver
from reacts.contracts import MappingStatus, ResolutionStatus


def _row(step_id, index, reaction, *, input_label=None, output_label=None, parse_ok=True):
    return {
        "route_id": "route-1",
        "step_id": step_id,
        "step_index": index,
        "canonical_reaction_smiles": reaction,
        "reaction_smiles": reaction,
        "input_intermediate": input_label,
        "output_intermediate": output_label,
        "parse_ok": parse_ok,
    }


def test_symbolic_intermediate_resolves_only_from_explicit_adjacent_anchor():
    rows = [
        _row("route-1::000", 0, "CCO>>CC=O", output_label="M1"),
        _row("route-1::001", 1, "M1.N>>CCN", input_label="M1", parse_ok=False),
    ]
    resolved, edges = SymbolicIntermediateResolver().resolve_route(rows)
    assert resolved[1].resolution_status == ResolutionStatus.RESOLVED
    assert resolved[1].canonical_resolved_reaction_smiles == "CC=O.N>>CCN"
    assert edges[0].continuity_status == "resolved"
    assert edges[0].connecting_molecules == ("CC=O",)


def test_product_side_placeholder_without_anchor_is_quarantined():
    rows = [_row("route-1::000", 0, "CCO>>M1", output_label="M1", parse_ok=False)]
    resolved, _ = SymbolicIntermediateResolver().resolve_route(rows)
    assert resolved[0].resolution_status == ResolutionStatus.UNRESOLVED
    assert resolved[0].canonical_resolved_reaction_smiles is None


def test_mcs_mapper_and_reaction_centre_are_confidence_qualified():
    mapping = AtomMappingEngine("mcs_fallback", min_coverage=0.6).map_reaction("CCO>>CCO")
    assert mapping.status == MappingStatus.MAPPED
    assert mapping.mapped_reaction_smiles
    centre = derive_reaction_centre(mapping.mapped_reaction_smiles)
    assert centre.structural_family == "no_detected_core_change"
    assert centre.fingerprint
