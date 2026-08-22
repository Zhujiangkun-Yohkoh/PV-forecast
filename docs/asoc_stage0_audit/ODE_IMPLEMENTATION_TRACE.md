# Neural ODE implementation trace

Authoritative source: PVforecast16/GFNODE_experiments/gfnode_solo_benchmark.py.

1. z is [batch,128]: BiLSTM aggregate is 2×64; ODE output [H+1,B,128], then [B,H,128] (:356-395).
2. ODEFunc.forward(t,x) receives t but never uses it; f is autonomous (:272-278).
3. Grid is torch.linspace(0,1,H+1) (:385). H12=[0,1/12,…,1]; H144=[0,1/144,…,1].
4. RK4 requests step_size=0.1 (:287-295), but every studied output interval is <0.1. Effective update is 1/H: H12=0.083333; H144=0.006944.
5. 0.1 is solver option; 1/H is grid; physical five minutes is only horizon*5/60 label (:605-607), never an ODE input.
6. One RK4 update per output interval. Audit inference NFE=4H per batch forward: H12=48; H144=576. NFE is not logged.
7. tau_i=i/H is consistent; effective h_step=0.1 is contradicted. No unit-time→five-minute calibration exists.
8. Euler fallback catches AssertionError only; uses 0.2/loose tolerance and is not logged (:290-295). Occurrence CANNOT_VERIFY.
9. Models are independent per data set × H (:605-657), not one unified model.
10. Fixed registered grid/H-specific head means no released arbitrary-time query API (:368-395); arbitrary temporal resolution without retraining is UNSUPPORTED.
11. Figure 12 loads weights (:112-117), but PCA uses 300 random 128-D states (:119-129), f_dx omits residual (:72-84), and trajectories use random starts/separate Euler (:141-160). It is 2-D per-H SVD/PCA, not held-out latent-state evidence; convergence claim contradicted.

main.py:439-519,713-733 has the same pattern.

