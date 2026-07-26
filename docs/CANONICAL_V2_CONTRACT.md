# Canonical dataset v2 contract

## Identity contract

- `route_instance_id` is the unique Product Two route primary key. It is derived from Product One `route_uid`, with a deterministic occurrence suffix only when Product One repeats the same `route_uid`.
- `source_route_id` preserves the patent-source route identifier and may repeat across conflicting variants.
- compatibility `route_id` equals `route_instance_id`.
- `step_instance_id` is unique within the retained route instance; compatibility `step_id` equals `step_instance_id`.
- `source_step_id` preserves the Product One step identifier, including source duplicates.
- grouping, edges, mapping, derivation, retrieval, and validation operate on instance identity.

## Symbolic evidence contract

A label such as `M1` proves logical route continuity only. It is not a molecular structure. Automatic resolution is permitted only when a unique explicit route-local observed product supplies the structure. Product placeholders and label-only links remain unresolved. Model-generated intermediate hypotheses must be stored outside canonical observations.

## `steps`

Preserves Product One identity and observed fields while adding:

- `original_reaction_smiles`;
- `resolved_reaction_smiles`;
- `canonical_resolved_reaction_smiles`;
- contextual parse status and failure class;
- resolution status, confidence, and ambiguity reason;
- mapping status, backend, coverage, and confidence;
- reaction family and centre fingerprint;
- route continuity and contextual quality;
- task-specific eligibility flags.

## `routes`

One row per retained Product One route with contextual step counts, resolved/unresolved counts, route-continuity score, average route quality, split, and patent identity.

## `molecules` and `step_molecules`

Molecules are normalized by canonical isomeric SMILES and stable content ID. `step_molecules` records role (`reactant`, `product`, `solvent`, `agent`, `catalyst`, or symbolic intermediate), position, source role, and confidence.

## `route_edges`

Explicit step-to-step relationships. Each edge records the symbolic label or molecule anchor, source and target steps, continuity state, and confidence.

## `intermediate_resolution`

One row per symbolic reference. It records candidate anchors, selected structure, status, confidence, validation result, and quarantine reason. Original route text is never replaced.

## `condition_evidence`

Long-form evidence table with:

- condition type;
- original token;
- a homogeneous textual `normalized_value` for portable long-form interchange;
- `normalized_value_type` (`numeric` or `categorical`);
- typed `normalized_numeric_value` and `normalized_text_value` columns;
- normalized units;
- extraction rule;
- confidence;
- plausibility status;
- direct-observation versus inference flag;
- source field.

## `repair_candidates`

Ranked, deterministic candidates with edit similarity, route-continuity support, strict parse result, canonical candidate, acceptance status, and rejection reason.

## `reaction_mappings`

Contains mapped reaction, backend, status, confidence, atom coverage, timeout/failure information, and eligibility.

## `reaction_centres`

Contains formed, broken, and changed bonds; changed atom environments; reaction-centre fingerprint; template; and structural family.

## `reaction_families`

Versioned structural taxonomy assignment with family ID, derivation method, confidence, and mapping dependency.

## `quarantine`

Contains ambiguous symbolic resolution, invalid post-resolution reactions, mapping failures, and any other contextual process that cannot be accepted without guessing.

## v2.0.9 split-governance contract

The post-derivation scientific view persists `reaction_signature`, `split_component_id`, component size, split algorithm, and seed on split-bearing final tables. A route is atomic, and connected components are formed across both non-empty `patent_document_id` and exact `reaction_signature`. The assigned split is deterministic with seed 42. Strict release requires zero patent overlap, zero exact-reaction overlap, and zero route split conflicts. Product-scaffold overlap remains a reported challenge diagnostic.

This is a post-derivation metadata transformation. It does not mutate `data/canonical_v2_context`, `data/mapping_v2`, `data/derivation_v2`, or their queue databases. The split manifest records the parent final-manifest hash and the unchanged-upstream contract.
