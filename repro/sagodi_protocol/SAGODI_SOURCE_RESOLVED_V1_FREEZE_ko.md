# Ságodi 공개 코드 기반 재현 프로토콜 v1

이 프로토콜은 논문 본문의 문장을 하나의 공통 학습법으로 해석하지 않고,
Ságodi et al.이 공개한 코드의 실제 실행 경로를 기준으로 baseline을
재현한다. 업스트림은 다음 상태로 고정한다.

- 저장소: `catniplab/back_to_the_continuous_attractor`
- 커밋: `cbd7404e9baca4b2dc291560cfc6576bb7b1f078`
- 실행 계약: `sagodi_source_resolved_v1.json`

이 결과는 `Ságodi exact`가 아니라 **code-resolved reproduction with
documented deterministic repairs**라고 부른다. 공개 코드 자체에 초기화되지
않은 텐서와 학습/분석 경로 불일치가 있기 때문이다.

## 공통 task

- angular integration, `T=12.8`, `dt=0.1`, 128 loss-bearing steps
- RBF GP velocity, length scale 1
- `sparsity='variable'`: trial별 `s ~ Uniform(0,2)`를 뽑고
  `Uniform(0,1) < 1-s`인 velocity token을 0으로 만든다.
- 초기 각도는 `Uniform(-pi,pi)`이다.
- step `t`의 target은 velocity를 적용한 뒤의 `q_{t+1}`이다.
- 공개 코드와 같이 `target[:,0]`, 즉 post-update `q1`을 output-to-state
  map에 넣어 초기 recurrent state를 만든다.
- online task stream은 `(model_seed, update)`로 결정한다. 같은 seed와
  update의 모든 모델은 동일 데이터를 사용하고, 서로 다른 seed는 서로
  다른 online data stream을 사용한다.

## 모델별 학습 계약

| 모델 | LR | 학습 perturbation | dropout | recurrent WD | clip |
|---|---:|---|---:|---:|---:|
| tanh RNN, H=128 | 0.01 | state noise `sqrt(0.1)*0.1=0.0316228` | 0 | 0 | 없음 |
| GRU, H=128 | 0.01 | target noise 0.01 | 0.5 | 0.0001 | 100 |
| LSTM, H=64 | 0.001 | target noise 0.01 | 0 | 0.01 | 1 |

GRU recurrent weight는
`Uniform[-0.25/sqrt(H), 0.25/sqrt(H)]`, LSTM recurrent weight는
`Uniform[-1/sqrt(H), 1/sqrt(H)]`로 초기화한다. RNN은 공개
`models.RNN`의 gain 경로에 따라 input/output weight를
`Normal(0,1/sqrt(H))`, recurrent weight를
`Normal(0,1.5/sqrt(H))`, recurrent bias를
`Uniform[-sqrt(H),sqrt(H)]`로 초기화한다.

평가는 final-update checkpoint, clean target, dropout off, state noise off로
고정한다. 학습 perturbation과 deterministic evaluation을 섞어 보고하지
않는다.

## 명시적 repair

1. 공개 GRU/LSTM의 `output_to_hidden` 및 `output_to_cell`은
   `torch.Tensor`만 할당하고 초기화하지 않는다. 이를
   `Normal(0,1/sqrt(H))`로 명시적으로 초기화한다.
2. 공개 LSTM 학습 `forward`는 cell map을 계산한 뒤 `c0=tanh(h0)`로
   덮어쓰지만, 같은 파일의 sequence-analysis 경로는 cell map을 사용한다.
   학습과 분석이 동일한 모델을 사용하도록
   `c0=tanh(q1 @ output_to_cell)`로 고친다.
3. process-global RNG 대신 task, target noise, state noise를 독립된 keyed
   RNG stream으로 만든다.
4. GP covariance Cholesky에 diagonal jitter `1e-6`을 등록한다.

RNN의 source `h0` parameter는 `map_output_to_hidden=True`일 때도 state
dict에 남지만 `requires_grad=False`이고 forward에 쓰이지 않는다. 구현은
이 128개 frozen scalar를 보존한다. 따라서 RNN은 state-dict scalar가
17,282개지만 trainable parameter budget은 논문 비교에 쓰는 17,154개다.

## 실행 인터페이스

먼저 `source_angular_integration`으로 evaluation batch를 만들고
`tasks.save_fixed_bank`로 저장한다. 단일 seed worker는 다음과 같이
실행한다.

```bash
python -m repro.sagodi_protocol.source_resolved_worker \
  --run-id source__gru__seed0 \
  --model-id sagodi_gru_n128 \
  --model-seed 0 \
  --evaluation-bank /path/to/source_eval.npz \
  --output-dir /path/to/run \
  --device cuda:0
```

worker는 `run_manifest.json`, `training_trace.json`, `result.json`,
`checkpoint_final.pt`, `COMPLETE`, `completion_receipt.json`을 원자적으로
기록한다. 기존 v4의 paper-first 결과와 이 프로토콜의 결과를 한 집계표에
섞지 않는다.
