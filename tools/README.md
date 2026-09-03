# Model inference and Boltz-2 representation analysis

These programs implement the model configurations and representation analyses described
in the manuscript appendix.

- `matched_inputs/`: construct matched HiQBind inputs, run Boltz-2 v2.2.1 for five
  seeds, and collect its regression output.
- `aqaffinity/`: run AQAffinity v0.0.1 with OpenFold3 v0.3.1 and the `of3_ft3_v1`
  trunk, cached ColabFold MSAs, three recycling steps, no templates, and five seeds.
- `aurobind/`: run the released AuroBind affinity-only path with cached ColabFold MSAs,
  ten recycling iterations, and five seeds.
- `trunk_probe/`: capture Boltz-2's two affinity-ensemble stacks and run the linear
  probe, conditional permutation, distogram-locality, and distance-shell analyses.

Each model has an isolated environment because the released packages require different
dependency versions. The checked-in prediction parquets are the inputs to
`scripts/main_results.py`.
