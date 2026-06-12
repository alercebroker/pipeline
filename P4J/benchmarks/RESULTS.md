# P4J MHAOV Benchmark Results

Env: python 3.11.15, numpy 2.4.6. Median of 3 repeats (after warmup).

## Wall time (seconds)

| scenario | grid | baseline_f32 | loopmove_f32 | serial_f64 | optimized_f64 | speedup(first→last) |
|---|---|---|---|---|---|---|
| single_2yr_N500 | 197388 | 2.029 | 2.092 | 2.588 | 1.904 | 1.1x |
| single_2yr_N1000 | 197864 | 4.073 | 4.245 | 4.857 | 3.849 | 1.1x |
| multi_2yr_N500x2 | 197558 | 4.097 | 4.142 | 4.884 | 3.829 | 1.1x |
| short_10yr_N800 | 598730 | 10.230 | 10.357 | 12.133 | 9.375 | 1.1x |

## Accuracy (recovered period rel. error / coarse peak height)

| scenario | true P | baseline_f32 | loopmove_f32 | serial_f64 | optimized_f64 |
|---|---|---|---|---|---|
| single_2yr_N500 | 0.2734 | 7.9e-07 / 746 | 7.9e-07 / 746 | 1.6e-06 / 754 | 1.6e-06 / 754 |
| single_2yr_N1000 | 0.2734 | 1.3e-06 / 1498 | 1.3e-06 / 1498 | 1.3e-06 / 1505 | 1.3e-06 / 1505 |
| multi_2yr_N500x2 | 0.2734 | 7.9e-07 / 1569 | 7.9e-07 / 1569 | 8.7e-07 / 1566 | 8.7e-07 / 1566 |
| short_10yr_N800 | 0.0721 | 6.4e-07 / 1165 | 6.4e-07 / 1165 | 3.6e-08 / 1248 | 3.6e-08 / 1248 |
