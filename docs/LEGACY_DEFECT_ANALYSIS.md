# Legacy artifact defect analysis and repair hypotheses

This analysis was reconstructed from the uploaded EDA notebook, the `reaction_curation` producer module, and the generated USPTO multistep artifacts.

## Confirmed producer defects

### 1. Duplicate canonical columns

`USPTOllmIngestor._prepare_multistep_df` adds `route_id`, `route_source_id`, and `multistep_reaction_text`. `USPTOllmNormalizer._normalize_multistep_dataframe` then concatenates the original frame with records that contain the same names. CSV export therefore emits duplicate headers and pandas reloads them as `.1` variants.

**Product One repair:** canonical tables are constructed from explicit schemas. Duplicate source columns are coalesced once and never propagated.

### 2. Malformed `step_id`

`ConditionTableBuilder._multistep_step_table` calls `row.get("route_id")` on a frame with duplicate `route_id` columns. Pandas returns a Series rather than a scalar. Interpolating that value into an f-string serializes the complete Series, producing IDs such as:

```text
route_id    20150205-US20150038574A1-0427
route_id    20150205-US20150038574A1-0427
Name: 2, dtype: object::step_0
```

**Product One repair:** IDs are rebuilt as `{route_id}::{step_index:03d}` after scalar coalescence. Conflicting duplicate keys receive deterministic content-hash suffixes.

### 3. Temperature/time inversion in multistep middle fields

The multistep source uses three-field reactions of the practical form:

```text
reactants > solvents.agents.temperature.time > products
```

The legacy `_split_condition_tokens` checks every bare numeric with `_parse_time_token` before `_parse_temperature_token`. Since plain time numerics are interpreted as seconds, a sequence such as `100.64800` becomes two durations rather than `100 °C` and `18 h`. This explains the generated temperature index containing only `<0` and `0–25 °C` classes despite source examples with 40, 50, 60, 80, 85, 100, 160 °C and above.

**Product One repair:** the canonicalizer reparses the middle field using source order, aliases, sign, magnitude, and final-position evidence. The legacy values remain in `legacy_temperature_c` and `legacy_time_h`; repaired values carry method and confidence fields. Ambiguous cases are preserved and flagged instead of silently accepted.

### 4. False zero solvent/agent QC

Condition tables are written to CSV with Python-list literals. The QC stage reloads CSV values as strings and checks `isinstance(value, list)`, yielding zero coverage even when the lists are populated.

**Product One repair:** all list-like fields are decoded centrally. In Parquet they are native Arrow list columns. CSV is a compatibility fallback only.

### 5. Identifier-bearing condition signature

The condition signature includes `route_id` and `step_index`, making almost every signature unique. It therefore identifies a record, not a reusable condition profile.

**Product One repair:** record identity and condition-content identity are separated. The canonical contract does not treat source location as chemistry.

### 6. Patent leakage in benchmark splits

The legacy split groups by route ID. Multiple routes from one patent can enter different partitions, allowing patent-specific chemistry and prose conventions to leak across train, validation, and test.

**Product One repair:** the default split is a deterministic hash of the patent document ID, derived by removing the route suffix. Every route and step from one patent remains in one partition.

### 7. Over-destructive rare-molecule policy

The single-step EDA retained only 24 of 226,743 structurally eligible reactions when a minimum molecule frequency of 100 was applied to every component. A patent reaction usually contains at least one rare substrate or product, so row deletion is not an appropriate vocabulary policy.

**Product One repair:** rare molecules are retained. Individual tasks may use minimum label support, hierarchical labels, OOV handling, or retrieval fallbacks without deleting the underlying reaction.

### 8. Symbolic intermediates treated as invalid chemistry

`M1`, `M2`, and related placeholders are route-graph symbols rather than malformed SMILES. RDKit cannot parse them, but deleting them destroys partial-route evidence.

**Product One repair:** symbolic intermediates are assigned the distinct class `symbolic_intermediate`. They remain searchable and useful for parse-failure learning, while mapping-dependent condition models abstain.

### 9. Outliers used as ordinary labels

Generated values include temperatures such as `-2100 °C` and durations of tens of millions of hours.

**Product One repair:** observed values are immutable. Plausibility-clean values are separate nullable columns. Quality events record every exclusion from a training target.

## Important non-defects and limitations

- The EDA showed the original single-step transformer and atom-mapped lanes were structurally parseable at 247,317 rows. Parser failure in the multistep lane therefore reflects route encoding, symbolic intermediates, and malformed extracted chemistry rather than a universal source failure.
- Atom mapping is essentially absent from the multistep artifact. Reaction-centre and template models remain a later mapping pipeline, not an implied current capability.
- Yield labels are absent. Product One does not expose yield, reaction-success, or process-efficiency prediction.
