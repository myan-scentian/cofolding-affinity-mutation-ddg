# AuroBind inference

This directory reproduces the AuroBind arm described in Appendix C. It uses the released
checkpoint, cached ColabFold MSAs, ten recycling iterations, seeds 1--5, reference
attention, and the affinity-only path. The scalar is read after the trunk and before
diffusion, matching the value returned by the full released path.

Set `AUROBIND_WORK` to a work directory, then run:

```bash
bash tools/aurobind/setup_env.sh
python tools/aurobind/build_inputs.py
python tools/aurobind/fetch_msa.py
python tools/aurobind/run.py
python tools/aurobind/to_parquet.py
```

The final table is `artifacts/data/predictions/aurobind_hiqbind.parquet`, with one row per
complex and seed. The analysis stores `6 - raw` so that larger values mean weaker
binding on the common nominal concentration scale.
