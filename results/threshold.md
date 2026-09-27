## threshold

Provider: `sim` · seeds: [1, 2, 3, 4, 5] · 136 tasks × 3 epochs per arm · values are mean ± sd across seeds

| Arm | Success % | Precision % | Injected % | Misled % | Poisoned inj. % | Redundancy idx | Tokens/task | Ctx tokens | $/success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 65.0 ± 2.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.000 ± 0.000 | 928 ± 4 | 0 ± 0 | 0.0198 ± 0.0009 |
| threshold_0.00 | 84.3 ± 2.1 | 79.8 ± 2.0 | 97.7 ± 1.1 | 8.6 ± 2.2 | 6.2 ± 2.3 | 0.048 ± 0.020 | 621 ± 11 | 123 ± 1 | 0.0080 ± 0.0004 |
| threshold_0.30 | 86.8 ± 1.1 | 95.0 ± 1.1 | 84.4 ± 1.6 | 7.1 ± 1.2 | 4.4 ± 1.0 | 0.029 ± 0.006 | 591 ± 5 | 107 ± 2 | 0.0075 ± 0.0001 |

| vs baseline | Token savings % | Cost savings % | Latency reduction % | Success Δ (pp) |
|---|---:|---:|---:|---:|
| threshold_0.00 | 33.1 ± 1.2 | 47.5 ± 1.3 | 49.0 ± 1.2 | 19.4 ± 1.1 |
| threshold_0.30 | 36.4 ± 0.4 | 49.4 ± 0.5 | 50.6 ± 0.5 | 21.8 ± 2.3 |
