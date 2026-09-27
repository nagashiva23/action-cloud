## main

Provider: `sim` · seeds: [1, 2, 3, 4, 5] · 136 tasks × 3 epochs per arm · values are mean ± sd across seeds

| Arm | Success % | Precision % | Injected % | Misled % | Poisoned inj. % | Redundancy idx | Tokens/task | Ctx tokens | $/success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 65.0 ± 2.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.000 ± 0.000 | 928 ± 4 | 0 ± 0 | 0.0198 ± 0.0009 |
| flat_memory | 75.4 ± 4.9 | 95.2 ± 0.3 | 91.2 ± 0.6 | 22.5 ± 5.1 | 27.7 ± 5.3 | 0.017 ± 0.006 | 608 ± 14 | 112 ± 1 | 0.0089 ± 0.0007 |
| actioncloud | 86.8 ± 1.1 | 95.0 ± 1.1 | 84.4 ± 1.6 | 7.1 ± 1.2 | 4.4 ± 1.0 | 0.029 ± 0.006 | 591 ± 5 | 107 ± 2 | 0.0075 ± 0.0001 |

| vs baseline | Token savings % | Cost savings % | Latency reduction % | Success Δ (pp) |
|---|---:|---:|---:|---:|
| flat_memory | 34.5 ± 1.4 | 47.9 ± 1.6 | 49.1 ± 1.6 | 10.5 ± 4.3 |
| actioncloud | 36.4 ± 0.4 | 49.4 ± 0.5 | 50.6 ± 0.5 | 21.8 ± 2.3 |
