# Boltz-2 representation probing

The instrumented Boltz-2 run records the conditioned input and every block output from
both independently parameterised affinity-ensemble members: nine cut points for the
eight-block member and five for the four-block member.

For each cut point, `layer_analysis.py` applies the released head's protein-ligand and
ligand-ligand mean-pooling mask, subtracts the reference vector from the variant vector,
and fits ridge regression through the origin. Outer folds hold out complete proteins;
protein-grouped inner folds select the ridge penalty. `seed_spread.py` repeats this for
each inference seed and for the five-seed mean. The conditional permutation analysis
permutes measured labels within protein, repeats the complete nested fit, and takes the
maximum correlation over all 14 cut points for each of 200 relabellings.

`locality_decomposition.py` compares the substitution and re-fold contrasts in the
Boltz-2 distance bins. It reports the pooled entries involving the substituted residue
and the five distance shells used in Appendix F. Complete protein-ligand systems are the
bootstrap units.

The full captured representation set is not distributed because it is about 7 GB.
Derived results are in `artifacts/work/trunk_probe/`, and the ten included example NPZs
regenerate the signed distogram in Figure 3.
