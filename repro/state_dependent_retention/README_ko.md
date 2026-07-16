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
T=128 source-v6 angular integration, Adam LR 0.01, batch 64, 5,000 updates이며 state noise,
target noise, output dropout은 모두 0이다. Seeds는 0--2다.

실행 예:

```bash
PYTHONPATH=. /home/biadmin/ca_rnn/.conda-calru-p0/bin/python \
  -m repro.state_dependent_retention.run_writer_comparison \
  --stage main \
  --root /home/biadmin/ca_rnn/experiments/state_dependent_retention_writer_v1 \
  --bank /path/to/main_test.npz \
  --gpus 0,1,2,3,4,5
```

학습과 평가를 분리하려면 `--training-only`를 사용한다. 이 모드에서는 5,000번째
optimizer update 직후 `checkpoint_trained.pt`를 원자적으로 저장하고 task/blank/radial
평가는 수행하지 않는다. 일부 조건만 실행하려면 예를 들어 다음을 추가한다.

```text
--training-only \
--conditions gradient_only:recurrent,hybrid_rp:linear,hybrid_rp:input_nonlinear,hybrid_rp:recurrent
```

학습된 G-C checkpoint의 attractor 구조는 학습과 분리하여 다음처럼 분석한다.

```bash
PYTHONPATH=. python -m repro.state_dependent_retention.analyze_attractor \
  --root /path/to/training-only-campaign \
  --bank /path/to/main_test.npz \
  --output /path/to/attractor_analysis_g_c \
  --seeds 0,1,2 \
  --device cuda:0
```

H-C를 분석할 때는 별도 output 경로와 `--retention-mode hybrid_rp`를 지정한다.

이 분석은 slow-manifold reconstruction, output-projected flow와 flow reversal,
32개 anchor의 full local Jacobian, ambient/radial finite-kick recovery를 저장한다.
큰 kick이 발산하더라도 같은 seed의 reconstruction, flow, Jacobian과 더 작은 반지름의
recovery 결과는 보존한다.

Noise 조건을 논문용 model-comparison figure에서 제외하고 H-C를 추가하려면 기존
plotting command에 `--hc-training-root`, `--hc-analysis-root`, `--no-noise`
옵션을 함께 준다. Raw noise artifact는 삭제하지 않는다.
