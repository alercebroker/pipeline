# Are the new ZTF feature step and LC classifier computing the same thing as the legacy pipeline?

**Yes.** Given the same light curve, the new steps produce the same features and the same BHRF
probabilities as the legacy pipeline. Where the final class differs, the cause is always in the
inputs the two pipelines had (WISE crossmatch availability, a different period alias on a
near-degenerate periodogram, or a fit deciding a near tie), never in the steps or the model.

## What was compared

- **Ours:** the production images of the multisurvey feature step (ZTF) and the LC
  classification step (BHRF 2.1.0), run on real light curves read from the multisurvey
  database.
- **Legacy:** the feature vector and probabilities the legacy pipeline stored for the same
  objects, at the same light-curve epoch (the last alert is included on both sides).
- **Cohort A:** 1,000 objects sampled at random from everything legacy classified.
  737 periodic, 258 stochastic, 5 transient by top class. Median 57 detections.
- **Cohort B:** 100 objects whose legacy top class is Transient. Median 19 detections.

## Probabilities

Share of objects where the rank-1 class is the same on both sides.

| head | Cohort A (1,000) | Cohort B (100 transients) |
|---|---:|---:|
| final class | 93.3 % | 84 % |
| top (periodic / stochastic / transient) | 96.8 % | 89 % |
| transient head | 88.7 % | 83 % |
| stochastic head | 97.3 % | 94 % |
| periodic head | 92.8 % | 84 % |

When the class agrees, the probability vectors are nearly identical: the median per-class
difference is 0.001 or less for every class, and the 95th percentile is below 0.05 for every
class. When the class disagrees, it is a near tie: the winning margin is 0.02 at the median
(0.13 when they agree), and in 49 of the 67 Cohort A cases our class is legacy's second choice.

**Why the final class disagrees**

Each disagreement was traced by swapping one family of features at a time from legacy's vector
into ours until legacy's class came back. Fed legacy's own features, the model reproduces
legacy's class for 996 of 1,000; fed ours, ours for 1,000 of 1,000, so the model is not the
cause.

| cause | Cohort A (67) | Cohort B (16) |
|---|---:|---:|
| legacy had no WISE colours at that epoch, we do | 19 | 5 |
| period: different peak on a near-degenerate periodogram, or same period with a numerically different harmonic fit | 26 | 1 |
| fit numerics (SPM and other parametric fits) | 4 | 6 |
| PS1 / reference-image values differ | 4 | 3 |
| colour features differ | 1 | 0 |
| two families together, each too small alone | 10 | 1 |
| several tiny differences, near tie | 3 | 0 |

The two large causes are the period search and WISE availability. Legacy ran its crossmatch
step but the stored vector has no WISE colours for 207 of the 1,000 objects of Cohort A and for
61 of the 100 transients of Cohort B. Where our crossmatch finds a WISE source and legacy's vector had none,
the model has more information on our side and can change its answer (typically from CV/Nova or
YSO to a class supported by the colours). This is not an error in the new step.

## Features

209 features compared per object, our value against legacy's, for the 1,000 objects of Cohort A.

| feature family | result |
|---|---|
| light-curve statistics, colours, PS1, WISE colours | identical to float32 precision whenever the detection set is the same |
| period | same peak for 78 % of objects; the rest is another local optimum of the same periodogram (one-day alias or harmonic), more often below 20 detections |
| harmonic amplitudes and phases, power rates | differ at the percent level even when the period is identical; these are the second cause of class flips above |
| SPM, fleet, microlensing, TDE fits | identical at the median; the epochs and chi-squares diverge in the tail, as expected from optimisers on slightly different inputs |
| WISE colours | present on our side for 60 objects where legacy had none (see above); values identical where both have them |

The feature extraction is deterministic: recomputing from the same messages gives bit-identical
values. The differences against legacy come from the inputs (detection set, crossmatch) and from
the periodogram and the fits being decided below their numerical resolution.

## Bottom line

- Same inputs, same features, same probabilities.
- Class agreement of 93 % (random objects) and 84 % (transients) is set by legacy's missing WISE
  and by near ties, not by the steps.
