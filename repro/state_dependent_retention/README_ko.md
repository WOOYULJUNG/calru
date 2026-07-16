# State-dependent retention writer comparison

이 exploratory campaign은 동일한 state-dependent diagonal retention을 세 writer와
결합한다.

| ID | update |
|---|---|
| B | `Lambda(h) h + B u` |
| B+ | `Lambda(h) h + psi(u)`, `psi(0)=0` |
| C | `Lambda(h) h + Gamma[g(h,u)-g(h,0)]` |

공통 retention은

```text
lambda_j(h) = base_lambda_j * exp(0.05 * tanh(r_j(h)))
```

이다. `r`은 raw recurrent state를 입력받는 동일한 Linear-GELU-Linear gate이고
마지막 층은 0으로 초기화한다. 따라서 모든 모델은 기존 constant-retention
CA-LRU에서 시작한다. `lambda(h)>1`을 국소적으로 허용하는 이유는 inward radial
perturbation을 nonzero ring으로 되돌리려면 expansion이 필요하기 때문이다.

각 writer는 두 retention-learning 조건에서 실행한다.

* `gradient_only`: static base retention과 state-dependent gate를 모두 ordinary
  task-loss backpropagation으로 학습한다. RP는 없다.
* `hybrid_rp`: static base retention은 task gradient에서 분리하고 기존 RP
  `(eta=1000, epsilon=3e-5, interval=50)`로만 갱신한다. State-dependent gate는
  ordinary task-loss backpropagation으로 학습한다.

현재 screen은 width-matched이며 parameter-matched가 아니다. 공통 조건은 width 52,
T=256 angular integration, Adam LR 0.01, batch 64, 5,000 updates, state noise 0.01,
target noise/output dropout 0, seeds 0--2다.

실행 예:

```bash
PYTHONPATH=. /home/biadmin/ca_rnn/.conda-calru-p0/bin/python \
  -m repro.state_dependent_retention.run_writer_comparison \
  --stage main \
  --root /home/biadmin/ca_rnn/experiments/state_dependent_retention_writer_v1 \
  --bank /path/to/main_test.npz \
  --gpus 0,1,2,3,4,5
```
