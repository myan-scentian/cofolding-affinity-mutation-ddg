# Boltz-2 inference

`run_boltz2.py` implements the Boltz-2 v2.2.1 configuration in Appendix C: ColabFold
MSAs, five structure recycling steps, one structure-diffusion sample, five
affinity-stage samples, and seeds 1--5. Inputs are generated from each HiQBind protein
sequence and ligand SMILES. Both sides of a mutation pair use the same protein construct.

The model configuration is `configuration/models/boltz2.yaml`; dataset paths are in
`configuration/datasets/hiqbind.yaml`. Raw inputs are not distributed with this package.
With those paths populated and Boltz-2 installed:

```bash
python tools/matched_inputs/run_boltz2.py --run hiqbind --seeds 5
python tools/matched_inputs/to_parquet.py --run hiqbind
```

Add `--instrument` to the inference command to capture the affinity representations
used by the probe. The final analysis table is
`artifacts/data/predictions/boltz2_v2_hiqbind.parquet`, with one row per complex and seed.
