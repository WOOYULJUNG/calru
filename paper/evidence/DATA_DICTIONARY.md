# Evidence data dictionary

이 문서는 [`raw/`](raw/)의 legacy key와 [`tables/`](tables/)의 paper-facing
column을 해석하는 기준이다. 수식과 task 생성의 상세한 설명은
[`paper/notes/methods_equations_data_ko.md`](../notes/methods_equations_data_ko.md)를
함께 본다.

## 1. 파일과 행의 단위

- Exp88/Exp72/Exp89 raw JSON 하나는 일반적으로 `task × model variant × seed`의
  한 최종 학습 run과 그 checkpoint에서 수행한 평가를 담는다.
- Ring transport raw CSV는 seed별로 model과 blank step에 따른 probe 행을 담는다.
- Aggregate CSV의 한 행은 identity column으로 정의한 조건에 해당하는 seed들의
  요약이다.
- 별도 표시가 없으면 기대 seed는 0, 1, 2다.

Aggregate column의 공통 의미는 다음과 같다.

| column 형태 | 의미 |
|---|---|
| `n_seeds` | 집계에 실제 포함된 seed 수 |
| `seed_ids` | 포함된 seed ID |
| `source_pattern` | `raw/`에 상대적인 입력 glob |
| `source_files` | 집계에 실제 사용한 seed-level 파일 목록 |
| `*_mean` | seed 산술평균 |
| `*_sd` | seed 표본 표준편차, `ddof=1` |

Aggregate table은 canonical header만 사용한다. 기존 raw key와 모델 tag는
`legacy_*`, `source_pattern`, `source_files`로 분리해 provenance를 보존한다.

## 2. Task ID와 자료형

| task ID | intrinsic coordinate | output dim | input dim | ID max velocity |
|---|---:|---:|---:|---:|
| `ring_hold` / `ring_integrate` | 1 | 2 | 6 | 3°/step |
| `torus_hold` / `torus_integrate` | 2 | 3 | 10 | 축별 2.5°/step |
| `complex_curve_hold` / `complex_curve_integrate` | 1 | 3 | 7 | 2.5°/step |
| `surface_hold` / `surface_integrate` | 2 | 3 | 6 | 0.018/step |
| `line_integrate` | 1, 2, 4, 8, 16 | d | task별 | 0.08/√d |
| `kway_hold_k16` | discrete K=16 | 16 logits | 17 | 해당 없음 |
| `flipflop_n4` | discrete 4 bits | 4 logits | 4 | 해당 없음 |

Main manifold training은 online fresh sampling, 10,000 optimizer steps, batch 256,
horizon 80–260, width 96, seed 0/1/2를 사용한다. Discrete scope control은 3,000
steps와 horizon 10–50을 사용한다. Retention scaling의 line protocol도 main Exp88
manifold battery와 별도 실험이다.

## 3. 선택된 RP threshold

| task | paper main epsilon |
|---|---:|
| `ring_hold` | `1e-4` |
| `ring_integrate` | `1e-4` |
| `torus_hold` | `3e-5` |
| `torus_integrate` | `3e-5` |
| `complex_curve_hold` | `3e-5` |
| `complex_curve_integrate` | `1e-4` |
| `surface_hold` | `3e-5` |
| `surface_integrate` | `3e-5` |

이 선택은 별도 held-out validation seed/data로 정한 것이 아니다. 저장된 epsilon
grid `0`, `3e-5`, `1e-4`에서 task별로 채택한 결과이므로, 원고에서
“validation-selected”라고 쓰면 안 된다. 전체 sensitivity는
`epsilon_sensitivity.csv`에 보존한다.

## 4. 연속 task metric

### 공통 RMSE

Raw `*_rmse`는 ambient output component RMSE다.

\[
\operatorname{RMSE}=
\sqrt{\frac{1}{B d_y}\sum_{b=1}^{B}
\lVert\hat y_b-y_b\rVert_2^2}.
\]

Raw `*_vec_rmse`는 batch별 vector norm을 사용하며 component RMSE의
`sqrt(output_dim)`배다. 어느 값도 angle error, geodesic error 또는 intrinsic
manifold-coordinate RMSE가 아니다.

| canonical metric | 대표 raw key | 정확한 의미 |
|---|---|---|
| `task_rmse` | `id_final_rmse` | ID horizon 260의 sequence endpoint target RMSE |
| `post_hold_500_rmse` | `id_postH500_rmse` | endpoint 뒤 500 blank 후 target 절대오차 |
| `post_hold_1000_rmse` | `id_postH1000_rmse` | endpoint 뒤 1000 blank 후 target 절대오차 |
| `normal_kick_1x_recovery_500_rmse` | `id_normal1_R500_rmse` | 1× state-RMS normal kick 후 500 blank target RMSE |
| `tangent_shift_recovery_500_rmse` | `id_tangentR500_rmse` | 인접한 clean memory를 다시 cue한 뒤 500 blank의 일관성 |
| `persistent_coordinate_count` | `lambda_gt_0p99` | carrier 96 좌표 중 λ>0.99인 수 |
| `retention_sum` | `lambda_sum` | carrier 좌표의 λ 합 |
| `trainable_parameter_count` | `params` | 학습 가능한 parameter 수 |
| `wall_time_seconds` | `seconds` | 공유 머신에서 관측한 run 시간 |

`post_hold`는 endpoint 대비 오차 증가량이 아니라 target에 대한 절대오차다.

### Normal kick

`normal{r}_R{R}`에서 `r ∈ {0.25, 0.5, 1}`은 absolute hidden norm이 아니라 각
모델 full diagnostic state의 RMS 배수다. Kick은 local tangent basis에 직교하도록
projection한 random direction에 적용한다. CA-LRU의 diagnostic state에는 carrier와
다음 step에 다시 계산되는 stream이 함께 들어가므로, main RMSE만으로 carrier의
순수 수축률을 주장할 수 없다.

`normal*_state_dev_ratio_R*`는 perturbed state와 clean rollout의 거리 비율이다.
Target RMSE보다 state recovery에 가깝지만 model별 state dimension/scale 차이를
고려해야 한다.

### Tangent shift

`id_tangentR*`는 기존 hidden state에 tangent vector를 더한 perturbation이 아니다.
Intrinsic coordinate를 norm 0.10만큼 옮긴 새 clean memory를 cue하고 같은 velocity
sequence를 다시 실행한 뒤 그 인접 기억이 유지되는지를 본다. 따라서 논문에서는
`tangent-shift consistency` 또는 `neighbor-memory consistency`라고 부른다.
`id_tangentOldR*`는 legacy 진단이며 main table에 사용하지 않는다.

### On-manifold motion과 PCA

| raw/canonical key | 의미 |
|---|---|
| `id_latent_tangent_motion_frac_mean` | moving step의 state displacement 중 local tangent projection 비율 평균 |
| `id_latent_tangent_motion_frac_min` | 같은 비율의 moving-step 최솟값 |
| `latent_pca_pr` / `pca_participation_ratio` | centered full-state PCA participation ratio |
| `latent_pca_dim90` / `pca_dimension_90_percent` | variance 90%를 설명하는 PC 수 |

Exp88 PCA는 CA-LRU carrier 96뿐 아니라 carrier+stream 192 전체를 사용한다.
반면 `lambda_gt_0p99`는 carrier 96에만 정의된다. 두 통계를 같은 vector space의
dimension으로 직접 동일시하지 않는다.

## 5. OOD metric

Prefix와 suffix를 조합해 평가 조건을 나타낸다.

| 형태 | 의미 |
|---|---|
| `id_*` | ID horizon 260 |
| `oodT500_*`, `oodT1000_*`, `oodT2000_*` | 더 긴 temporal OOD sequence |
| `vel1x_*`, `vel1.5x_*`, `vel2x_*` | max velocity scale을 바꾼 새 evaluation batch |
| `*_final_rmse` | 해당 sequence endpoint |
| `*_postH500_rmse` | 해당 endpoint 뒤 500 blank |

Temporal OOD는 길이만 늘리지 않는다. Integration에서 ordinary hold는 5–20에서
30–120, final hold는 20–80에서 100–250으로 변한다. Velocity scale별 결과도
동일 trajectory를 단순 rescale한 paired batch가 아니라 새로 샘플한 batch다.

`normal1_repeated{K}_H500`은 500 blank step 동안 K개의 normal kick을 균등하게
배치한다. K=100에서는 마지막 kick이 step 500에 있으므로 “100 kicks 후 500-step
recovery”가 아니라 마지막 kick 직후의 값이다.

## 6. Ring transport metric

Ring transport는 base angle 1.15 rad에서 3°/step을 5 step 입력해 15° 이동시킨 뒤,
최대 2,000 blank step 후 clean reference family와 비교한다. Reference grid는 1,440
points로 간격이 0.25°다.

| column | 의미 |
|---|---|
| `decoded_delta_deg` | readout이 나타낸 base 대비 각도 이동 |
| `dist_to_nearest_clean` | nearest clean reference까지 raw full-state Euclidean 거리 |
| `dist_to_target_clean` | 목표 각도의 clean reference까지 거리 |
| `nearest_angle_error_deg` | nearest clean reference angle과 목표 angle 차이 |

Distance는 manifold diameter나 local tangent scale로 정규화되지 않았다.
`nearest_angle_error_deg`는 grid 양자화의 영향을 받으며, 현재 결과는 세 seed지만
단일 base point만 사용한다.

## 7. Retention scaling metric

`retention_scaling_metrics.csv`는 Exp72 line-integration의 d=1,2,4,8,16을
집계한다. `rmse_T2000_postH500`은 독립 2,000-step random walk 뒤 500 blank target
RMSE다. PCA는 analysis horizon 50의 final full states에서 계산한다. 이 실험을
Exp88 manifold battery의 동일 protocol이라고 쓰지 않는다.

## 8. Discrete metric

| column | 의미 |
|---|---|
| `H2000_acc` / `hold_2000_accuracy` | K-way 2,000-step class accuracy |
| `H2000_full_acc` / `hold_2000_full_accuracy` | flip-flop 네 bit가 모두 맞은 sample 비율 |
| `basin_r1_*` | absolute Euclidean radius 1 perturbation 결과 |
| `postH500_*` | task endpoint 뒤 500 blank 결과 |
| `transition_after_full_acc` | single-bit transition 뒤 20 blank의 full-state accuracy |

Discrete 결과는 CA-LRU 우위를 주장하기 위한 main evidence가 아니라, 안정적인
point-attractor regime은 RNN/LSTM도 해결하며 논문의 target이 continuous memory임을
보이는 scope control이다.
