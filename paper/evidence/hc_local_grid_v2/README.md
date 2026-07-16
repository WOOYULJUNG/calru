# H-C local-grid v2 evidence

이 디렉터리는 state-dependent retention H-C의 개발용 국소 hyperparameter sweep과
checkpoint-only continuous-attractor 분석을 보존한다. 결과는 ring mechanism과 최종
후보를 고르는 **development/diagnostic evidence**이며, 독립 fresh-seed confirmatory
결과가 아니다.

## 실험 범위

- 새 학습: 8 cells × seeds 0, 1, 2 = 24 runs
- 재사용 reference: `a0p05_bm0p10`, seeds 0, 1, 2
- 공통 학습: width 52, Adam LR 0.01, batch 64, 5,000 updates
- noise: recurrent-state, target, output dropout 모두 0
- Retention Plasticity: warmup 1,500, interval 50, `eta_lambda=1000`,
  `damage_epsilon=3e-5`, blank ablation horizon 500
- screen: 2,048-step autonomous blank rollout
- full CA 분석: 1,024 task trajectories, 1,024-point periodic spline, 32 local
  Jacobian anchors, 2,048-step memory rollout, radial/ambient recovery through
  4,096 steps

## 파일

- `screen_summary.json`: 24개 새 run의 task/blank screening 결과
- `comparison.json`: reference를 포함한 seed-level 및 집계 CA 지표
- `comparison.md`: 사람이 읽을 수 있는 후보 비교표
- `selected_a0p05_bm0p10_analysis_summary.json`: 선택된 reference 후보의 full
  per-seed CA 분석
- `SELECTION_RATIONALE_ko.md`: 선택 기준, 대안과의 trade-off, 해석 한계
- `checksums.sha256`: 이 snapshot 파일의 SHA-256

대용량 checkpoint, training trace와 per-step NPZ는 포함하지 않는다. 이 snapshot만으로
집계값을 감사할 수 있지만 checkpoint-level 분석을 처음부터 다시 실행할 수는 없다.

## Provenance

- Git code before evidence import: `6471d5d`
- Grid config: `repro/state_dependent_retention/hc_local_grid_v2.json`
- Grid config SHA-256: `38503e773d7c7380ba5654d8b02d4d768ab19d16ae330f06b181383a3bbea70a`
- Fixed evaluation bank SHA-256:
  `e275acbcf2313061b5185d0fdd4927adbd084c70d1193bbace4edc2d81a445c7`
- Source screen summary SHA-256:
  `8b1555202d73a8c2649fa9ff83235eea17ad39bc10903736642a35bda684f552`
- Source comparison SHA-256:
  `4d7e522ee370292c7158b3e78d0e2917e26a3dbdf9bdea8b1530662e62678953`
- Selected analysis source SHA-256:
  `bf2ef8fbe8c975c67722885c789c7e1e660b6db09945cff75ddc23e539e0f268`

## 채택 결과

개발용 최종 후보는 다음과 같다.

\[
a=0.05,\qquad b_{\mathrm{gate}}=-0.10.
\]

이 조건은 3/3 seed stability를 만족하면서 비교 후보 중 가장 좋은 전체 균형을
보였다. 자세한 판단은 `SELECTION_RATIONALE_ko.md`를 따른다.
