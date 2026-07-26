# Route identity and symbolic evidence

Product One contains 235,265 retained route rows but 235,196 source route identifiers. Product Two v2.0.4 preserves every row as a distinct route instance. The existing Product One `route_uid` is the identity source; repeated `route_uid` values receive a deterministic occurrence suffix.

For the 64 duplicated source-route groups, step rows are assigned to route instances by exact `(step_index, raw_reaction_text)` correspondence with `multistep_reaction_text`. Assignment is one-to-one and fails closed if any source step is unassigned.

`M1`, `M2`, and similar labels are route topology evidence. They are never converted into molecular structures merely because adjacent steps use the same label. Accepted resolution requires a unique observed structural product from an explicitly matching predecessor.

The contextual manifest exposes route/step identity counts, cross-variant edge count, mapping eligibility, placeholder and symbolic-step counts, observed structure resolutions, inferred hypothesis count, unresolved symbolic steps, and invalid non-symbolic steps.
