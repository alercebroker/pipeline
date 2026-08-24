# This run against the WISE ablation

`WISE_NULL_CLASSIFICATION_IMPACT.md` showed that BHRF collapses toward Stochastic
when the AllWISE colors are missing, and predicted that restoring the crossmatch
would fix it. This run is the test at full scale. **It lands on the WISE-present
side, decisively.**

Reproduce with:

```bash
python features/offline/run_stats/plot_run_vs_ablation.py
```

Both ablation bars are recomputed from `wise_ablation/wise_ablation.csv` rather
than quoted from the note, so they cannot drift from the source data. This run's
bars come from `class_distribution.csv`, written by `offline_db_stats.py`.

## Read this first: the populations differ

Only the two ablation bars are a controlled comparison — the *same* 3,981
objects, predicted twice from the same feature vector, with only the eleven WISE
colors changed. This run's bar is a different population: 19.3M objects at
`n_det >= 2`, of which 80.5% matched AllWISE at all. The ablation sample was
selected for *having* WISE and a stored `27.5.6` vector, so it is far
better-observed than the catalogue at large.

So a bar-height difference between this run and the baseline is not by itself a
WISE effect. What survives that caveat is the Periodic/Stochastic axis, where the
ablation effect (87% → 37%) is far too large for any population difference to
imitate.

## Top head

![top head](run_vs_ablation_top.png)

| class | WISE present | WISE blanked | this run |
|---|---:|---:|---:|
| Periodic | 87.4% | 37.2% | **81.73%** |
| Stochastic | 12.4% | 57.2% | **6.70%** |
| Transient | 0.2% | 5.5% | **11.57%** |

**Periodic and Stochastic: the run is on the WISE-present side, unambiguously.**
81.7% Periodic against 87.4% with WISE and 37.2% without; 6.7% Stochastic against
12.4% and 57.2%. The WISE-null signature is a Stochastic share near 57% — this run
is nowhere near it. For comparison, the database's post-rollover state was 36.8%
Periodic. The crossmatch did what the note said it would.

**Transient does not fit either bar, and WISE cannot explain it.** At 11.57% the
run is *above* both the WISE-present baseline (0.2%) and the WISE-blanked one
(5.5%), and above the ~1.8% the note measured in the database. Blanking WISE
moves Transient by +5.3 points, so no amount of WISE effect produces 11.6% from a
0.2% baseline.

The likely cause is the population: `n_det >= 2` admits objects with almost no
light curve, and a sparse light curve is what a transient looks like. That is
consistent with the NaN rates in `DB_STATS.md` — `MHPS_*` is missing for 91% of
objects, `Rcs`/`Skew`/`Std` for 89% — all features that need a populated curve.
**This is a hypothesis, not a verified result.** The check that would settle it is
the Transient share as a function of `n_det`; it has not been run.

## Flat head

![flat head](run_vs_ablation_flat.png)

The WISE-null signature at leaf level is specific: CV/Nova 20.9% → 54.9% and YSO
8.1% → 25.6%, while Periodic-Other collapses 19.6% → 1.8%, LPV 17.9% → 1.4% and
RSCVn 13.2% → 0.9%.

None of that happened here. Periodic-Other is this run's largest class at 37.6%,
RSCVn holds at 11.2%, and CV/Nova (15.6%) and YSO (5.3%) sit at or below the
WISE-present baseline rather than three times above it. The leaf-level collapse
the note documented is absent.

Two differences from the baseline that are *not* the WISE signature:

- **LPV 3.6% against a 17.9% baseline**, with Periodic-Other correspondingly high
  (37.6% vs 19.6%). The pattern looks like Periodic-Other absorbing objects that
  a better-sampled light curve would resolve as LPV — LPVs are bright, slow and
  well-observed, exactly the objects the ablation sample was enriched in. Same
  population caveat, same status: plausible, unverified.
- **QSO 1.2% against 3.9%**, consistent with the same direction.

## Conclusion

The run reproduces the WISE-present condition on the axis the note was about.
The Periodic/Stochastic split is restored and the leaf-level collapse into
CV/Nova and YSO did not occur, so the predictions in the database are not
affected by the WISE-null bias that motivated the investigation.

The elevated Transient share is a separate, open question. It is not a WISE
effect, and it is the one number here that no prior measurement anticipates.
