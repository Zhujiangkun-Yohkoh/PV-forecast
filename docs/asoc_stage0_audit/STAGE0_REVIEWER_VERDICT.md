# Applied Soft Computing pre-submission verdict

## NO-GO

The manuscript should not be submitted with current numerical tables and figures.

### Decisive blockers

- Global KNN/IF/MinMax fitting before split: PVforecast16/GFNODE_experiments/gfnode_solo_benchmark.py:44-75.
- Test set used for early stopping and baseline LR selection: fair_sota_comparison.py:1063-1075,1087-1122.
- Overlapping window split/no purge and filtered nonuniform timestamps treated as five-minute steps: gfnode_solo_benchmark.py:78-83,605-621.
- Tables 9–13 lack a matching single provenance chain; Table 12 no-retraining claim is directly contradicted by seasonal_robustness_experiment.py:157-216.
- Autonomous ODE does not consume time; H-specific models and synthetic Figure 12 cannot support arbitrary temporal resolution or latent convergence: gfnode_solo_benchmark.py:272-278,385-395; fig_neural_ode_analysis.py:119-160.

### Required next stage

Run a clean timestamp-level train/validation/test protocol with train-only preprocessing, purge gaps, real elapsed-time semantics, validation-only tuning, repeated seeds, immutable provenance and hardware manifests. Rebuild every table/figure and narrow claims to what the rerun supports.

Until then, do not describe a module combination as innovation by default, do not claim frozen seasonal transfer, arbitrary-resolution output, error-accumulation suppression, vector-field convergence or deployment readiness.

