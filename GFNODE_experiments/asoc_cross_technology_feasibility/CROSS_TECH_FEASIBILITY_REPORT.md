# Cross-technology PV forecasting feasibility

## Decision

`SYNCHRONIZED_MULTI_OUTPUT` is feasible as a same-site, three-array forecasting task. It is **not** evidence of geographical or cross-climate generalization.

- Revision-plan evidence states that all three technologies are at Alice Springs and share meteorology (`PV_improve_v1/GFNODE_Revision_Plan.md:74`).
- Every split has a complete common five-minute clock: Train 30,528; Validation 6,624; Test 6,912 timestamps. Pairwise intersections are identical.
- Seven weather columns have the same names; six are numerically identical in the raw data. `Performance_Ratio` differs by technology, which is scientifically appropriate and must not be treated as shared weather.
- `Active_Power` is the common target field. Capacity, power unit, coordinates and formal module-technology metadata are not evidenced by the supplied CSV or located project documentation and remain `UNKNOWN`.

## Risks

1. Identical meteorology can make a shared-backbone gain reflect common exogenous forcing rather than transferable PV physics.
2. Missing/invalid target windows differ greatly across technologies, so formal joint training must use per-head masks and the intersection of valid windows.
3. Missing capacity and unit metadata blocks capacity-normalized or operational comparisons.

## Sole next recommendation

Implement the masked synchronized three-head ModernTCN screening experiment only after recording authoritative capacity/unit/module metadata; do not make cross-site generalization claims.
