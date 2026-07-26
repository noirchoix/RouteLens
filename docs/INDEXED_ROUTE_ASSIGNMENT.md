# Indexed route-instance assignment

Product One contains 235,265 route instances and 326,787 step rows. Most source route identifiers are unique; only 64 source-route groups contain variants.

Product Two v2.0.5 uses two paths:

1. `unique_route_assignment` maps unique source route IDs to route instances through indexed SQL and inserts narrow assignment records.
2. Exact route-text matching runs only for duplicated source-route groups.

The staged `steps` table remains source-faithful. Instance identity is stored in `route_instance_assignment` and joined during contextual iteration. This avoids rewriting a wide SQLite table and prevents correlated full-table scans.

The checkpoint `.work/identity_assignment_manifest.json` records source and schema hashes, total assignments, duplicate-group scope, expensive match calls, assignment methods, failures, and elapsed time. A valid complete checkpoint is reused immediately on `--resume`. An interrupted v2.0.5 assignment reuses staged source tables and rebuilds only the assignment table.

A complete pre-v2.0.5 staging database can be adopted when its SQLite integrity, table schemas, and row counts match the Product One manifest. The patch then creates v2.0.5 metadata and runs only the new narrow assignment stage.
