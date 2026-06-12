# P4J MHAOV Profiling Notes

## Baseline (float32, before refactor)

`cProfile` over the full production sequence on the `short_10yr_N800` scenario
(grid = 598,730 frequencies, single band):

```
         487 function calls in 9.892 seconds
   ncalls  tottime  cumtime  filename:lineno(function)
        1    0.000    9.892  go()
       11    0.005    9.886  periodograms.py:57(_compute_periodogram)
        1    0.000    9.885  base_periodogram.py:63(optimal_frequency_grid_evaluation)
       11    9.870    9.870  periodograms.py:66(<listcomp>)   <-- 99.8% of runtime
       11    0.011    0.011  {built-in method numpy.array}
        1    0.000    0.007  base_periodogram.py:136(optimal_finetune_best_frequencies)
```

**Conclusion:** 99.8% of wall time is the Python list comprehension at
`periodograms.py:66` —

```python
per = np.array([self.cython_per[fn].eval_frequency(freq) for freq in freqs], dtype=np.float32)
```

i.e. ~600k Python→Cython calls to `MHAOV.eval_frequency`, each boxing/unboxing a
float and dispatching a Python `max()` builtin several times internally. The
coarse-grid evaluation dominates; fine-tuning (the 10 small grids) costs 0.007s.

This is exactly the "function in Cython called from a Python loop" bottleneck.
Phase 4A moves the frequency loop into Cython (`eval_frequencies`, GIL released),
eliminating the per-call overhead and the Python-list/np.array materialisation.

## After refactor (optimized_f64)

```
         476 function calls in 9.305 seconds
   ncalls  tottime  cumtime  filename:lineno(function)
       11    9.297    9.298  periodograms.py:57(_compute_periodogram)   <-- now a single nogil Cython call
        1    0.000    0.007  base_periodogram.py:136(optimal_finetune_best_frequencies)
```

The Python list comprehension is gone: `_compute_periodogram` now makes one
`eval_frequencies` call per band, which loops over the whole grid inside Cython with
the GIL released. cProfile can no longer see inside the kernel (it's one C call), so
the remaining time is pure kernel compute. Deeper profiling would need `perf` or
Cython line-profiling.

## Findings & outcome (single-core constraint — no parallelism)

1. **The per-frequency Python loop was NOT the wall-clock bottleneck.** Moving it into
   Cython (`eval_frequencies`, GIL released) changed wall-time by <2%. The cost is
   genuine compute in the inner N×(2·Nharmonics+1) loops. The loop-move is kept as a
   clean entry point and because it removes ~100k Python-call/boxing crossings.

2. **float64 correctness gain (the headline).** Storing `mjd` and folding the phase in
   double precision fixes the float32 `mjd*freq` error (~0.04–0.09 cycle RMS at short
   periods). Effect on the `short_10yr_N800` case (P≈0.072 d, 10-yr baseline):
   period rel. error **6.4e-7 → 3.6e-8** (~17× better) and coarse peak height
   **1165 → 1248 (+7%)** — i.e. more detection power for short-period variables.

3. **Single-core kernel optimizations recovered the float64 cost and then some.**
   Naive float64 was ~18% slower than the float32 baseline; after (a) reusing the base
   complex exponential for the `Nharmonics==1` harmonic basis (halves the per-point
   `cosf`/`sinf`), (b) precomputing `1/err` and `(mag-wmean)/err` and multiplying
   instead of dividing in the hot loops, and (c) keeping the per-point recurrence in
   float32 (vectorizable) while only the cross-point reductions are double, the
   optimized float64 build is **~1.07–1.09× faster than the original float32 baseline**
   and ~1.25–1.29× faster than naive float64.

**Net: faster AND more numerically correct, single-core.** See `RESULTS.md` for the
full timing/accuracy tables (labels: baseline_f32 → loopmove_f32 → serial_f64 →
optimized_f64).
