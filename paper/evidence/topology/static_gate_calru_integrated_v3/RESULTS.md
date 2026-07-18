# CA-LRU-inclusive approximate continuous-attractor comparison

All five models use three checkpoints per topology. CA-LRU uses the validation-selected 5,000-update topology-tuning-v2 cell and is reanalyzed here with exactly the same local Jacobian, finite-kick, blank-evolution, and persistent-homology implementation as split-field.

| Topology | Model | Task rad | Blank-2048 rad | Tangent error | Worst normal gain | Gap | Recovery | Shape distortion | Topology survival |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | RNN | 0.076 | 0.7676 | 0.04328 | 1.189 | -0.2321 | 0.0224 | 4.117 | 0 |
| S1 | GRU | 0.02592 | 0.6896 | 0.303 | 1.175 | -0.4781 | 0.1595 | 0.816 | 1 |
| S1 | LSTM | 0.01786 | 0.4598 | 0.3821 | 1.461 | -0.843 | 0.3443 | 0.913 | 1 |
| S1 | CA-LRU | 0.0135 | 0.02478 | 0.01305 | 1 | -0.01306 | 1 | 0.08699 | 0 |
| S1 | Split-field | 0.01667 | 0.27 | 0.1829 | 1.015 | -0.1944 | 0.1181 | 0.8647 | 0 |
| T2 | RNN | 0.2114 | 1.287 | 0.06239 | 1.212 | -0.2764 | 0.5213 | 0.7053 | 0 |
| T2 | GRU | 0.03784 | 0.7905 | 0.1199 | 1.176 | -0.2955 | 0.5663 | 0.7081 | 0 |
| T2 | LSTM | 0.03607 | 0.7996 | 0.1251 | 1.467 | -0.5925 | 0.3796 | 0.8848 | 0 |
| T2 | CA-LRU | 0.02754 | 0.04319 | 0.007722 | 1 | -0.007724 | 1 | 0.01045 | 0 |
| T2 | Split-field | 0.0318 | 0.556 | 0.189 | 1.022 | -0.2104 | 0.5416 | 0.5582 | 0 |
| S2 | RNN | 1.064 | 1.449 | 0.05149 | 1.089 | -0.1404 | 0 | 6.074 | 0 |
| S2 | GRU | 1.058 | 1.638 | 0.192 | 1.096 | -0.2861 | 0.04714 | 0.9616 | 0 |
| S2 | LSTM | 0.04735 | 1.09 | 0.2215 | 1.625 | -0.8428 | 0.3885 | 1.504 | 0 |
| S2 | CA-LRU | 0.06831 | 0.1091 | 0.009928 | 1 | -0.009929 | 1 | 0.01671 | 0 |
| S2 | Split-field | 0.08978 | 0.7704 | 0.3903 | 1.065 | -0.4562 | 0.08232 | 2.736 | 0 |

## Candidate verdicts

- S1 / CA-LRU: FAIL.
- S1 / Split-field: FAIL.
- T2 / CA-LRU: FAIL.
- T2 / Split-field: FAIL.
- S2 / CA-LRU: FAIL.
- S2 / Split-field: FAIL.

RNN, GRU, and LSTM are the conventional baselines. CA-LRU and split-field are reported as candidate architectures, so CA-LRU is not folded into the phrase “best baseline.”
