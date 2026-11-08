# Reference-free self-convergence control

Protocol fixed on 4 November 2026 before running this additional, post-hoc
positioning experiment. This is not independent confirmation of the classifier.

Use shipped Gymnasium Acrobot, passive action 1, nominal energy and declared
order 4. Draw 40 starts with `shipped_acrobot.sample_states` and seed 20261104;
reuse them for every case and step. Cases: healthy RK4, RK4 with link-2 mass
+0.1%, RK4 with the nips Coriolis omission, midpoint RK2, Euler, and Euler
with link-2 mass +1%. Use h=0.2 (shipped) and h=0.05, 20 candidate macro
steps per start. Reduced smoke runs use 6 starts and 8 macro steps.

At each state compute endpoints X_m=Psi_(h/m)^m(x) for m=1,2,4,8, then
continue the candidate trajectory with X_1. Both diagnostics reuse exactly
these 15 step calls per macro step. Compute mean absolute energy residuals
and apply the frozen guarded classifier. Also compute d_m=mean ||X_m-X_2m||_2
for m=1,2,4, wrapping each angular difference into [-pi,pi]; state coordinates
are radians and radians/second with unit numerical weights. Report
p12=log2(d_1/d_2) and p24=log2(d_2/d_4), with no classifier fitted to them.
Mark self-convergence unresolved if any difference is below
50*eps*(1+maximum endpoint norm). Exclude a whole start/configuration from
both smooth diagnostics if any substep clips a velocity at the environment's
limit, or if an endpoint is nonfinite; retain the exclusion and its counts.

Archive per-step residuals/differences, initial states, per-start statistics,
exclusions, summaries and software/source/output hashes. Report all cells,
including mixed model/integrator faults and finite-step departures from the
formal order. Do not interpret self-convergence as a test of the physical
equations: its differences cancel the candidate's limiting flow. Do not claim
that invariant tests detect faults tangent to their invariant's level sets.
