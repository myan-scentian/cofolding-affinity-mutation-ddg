# Co-folding affinity heads underrespond to single-substitution binding-site mutations

Anonymous reproduction package for the manuscript. The analysis evaluates the released
Boltz-2, AQAffinity, and AuroBind affinity heads on 163 HiQBind mutation pairs and probes
Boltz-2's affinity-module representations.

## Reproduce the reported results

Install the locked analysis environment:

```bash
pixi install --locked
```

Run the numerical analyses:

```bash
pixi run results
pixi run decay
```

Regenerate the figures:

```bash
pixi run figure-benchmark
pixi run figure-scatter
pixi run figure-locality
```

The commands use `artifacts/` in this checkout. Set
`MUTATION_DDG_ARTIFACT_ROOT` to use an equivalent artifact tree elsewhere.

Fresh model inference also requires the wwPDB sequence archive at
`artifacts/data/pdb_seqres.txt.gz`:

```bash
curl -fL https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz \
  -o artifacts/data/pdb_seqres.txt.gz
```

## Included artifacts

- `artifacts/data/processed/ddg_bench_v2.parquet`: curated HiQBind pair table.
- `artifacts/data/processed/hiqbind_ligand_site_chains.csv`: chain-copy decisions used
  to match the protein constructs.
- `artifacts/data/predictions/`: five-seed per-complex outputs for the three heads.
- `artifacts/work/main_results.json`: cohort accounting and the reported performance,
  magnitude-bin, direction, and absolute-affinity analyses.
- `artifacts/work/trunk_probe/`: reported Boltz-2 probe, permutation, locality, and
  distance-decay results.
- `artifacts/results/boltz_hiqbind_unified/`: the five seeds for the illustrative
  distogram pair used in Figure 3.

The full captured Boltz-2 pair representations are about 7 GB and are not included.
Their derived statistics are included, and the capture and analysis programs are in
`tools/trunk_probe/`. Model checkpoints and MSA caches are obtained from the public
model releases cited in the manuscript.
