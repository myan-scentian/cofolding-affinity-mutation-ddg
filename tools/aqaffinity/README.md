# AQAffinity inference

This directory reproduces the AQAffinity arm described in Appendix C. It uses the
released AQAffinity v0.0.1 head, OpenFold3 v0.3.1 with the `of3_ft3_v1` trunk, three
recycling steps, cached ColabFold MSAs, no templates, and reference attention. It uses
the pair-matched construct when that construct is a single chain and otherwise uses the
ligand-bearing HiQBind chain. For each of seeds 1--5, the released pipeline selects the
highest-ipTM full-chain prediction, refolds the selected pocket, and draws five pocket
samples. The table uses the released writer's first pocket-sample value.

Set `AQAFFINITY_WORK` to a work directory. The setup script downloads the AQAffinity
release there. Then run:

```bash
bash tools/aqaffinity/setup_env_ft3.sh
artifacts/work/aqaffinity/.venv-ft3/bin/python tools/aqaffinity/build_queries.py
artifacts/work/aqaffinity/.venv-ft3/bin/python tools/aqaffinity/fetch_alignments.py
artifacts/work/aqaffinity/.venv-ft3/bin/python tools/aqaffinity/run_predictions.py
artifacts/work/aqaffinity/.venv-ft3/bin/python tools/aqaffinity/make_table_v2.py
```

The final table is `artifacts/data/predictions/aqaffinity_v2_hiqbind.parquet`, with one
row per complex and seed. The analysis changes the released raw sign so that larger
stored values mean weaker binding.
