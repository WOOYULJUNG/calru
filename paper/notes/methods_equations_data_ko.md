# Learning Approximate Attractor Dynamics for Recurrent Memory

## 수식·방법론·데이터 마스터 정리

이 문서는 논문 본문 §3–§6과 Appendix를 채우기 위한 코드 감사본이다. 문장 구조는 첨부된 최신 Introduction/Related Work를 따르고, 수식·실험·수치는 저장소의 실제 코드와 raw JSON/CSV를 기준으로 한다. 기존 코드 태그 PAN, AM-LRU, CAMN은 provenance에만 남기고 논문에서는 최종 명칭인 **CA-LRU**와 **Retention Plasticity (RP)**를 사용한다.

집계값은 별도 표시가 없으면 seed 0/1/2의 mean ± sample SD이다. 원고의 기존 표는 평균만 보여 주지만, \(n=3\)이고 seed 변동이 작지 않은 행이 있으므로 최종 표에는 SD 또는 개별 seed 점을 함께 제시하는 것이 안전하다.

## 0. 먼저 확정할 것

### 0.1 최종 모델 매핑

| 논문 표기 | 실제 코드 variant | 핵심 update | 원자료 |
|---|---|---|---|
| **CA-LRU** | PAN-RNW-full | \(g([h;u])-g([h;0])\) | exp88_writer_sweep_results/ |
| CA-LRU, input-only update | PAN-NW-full | \(g(u)\), \(g(0)=0\) | exp88_writer_sweep_results/ |
| CA-LRU, linear update | PAN-full | \(Bu\) | exp88_manifold_main_results/ |
| LRU | lru-full | complex diagonal LRU | exp88_manifold_main_results/ |
| RNN/GRU/LSTM/SSM | 동명 | baseline | exp88_manifold_main_results/ |

최종 CA-LRU는 반드시 PAN-RNW-full이다. Table 1의 ring-hold \(0.0007\), ring-integrate \(0.0022\), persistent support 3/6은 이 variant에서 나온다. PAN-full은 최종 모델이 아니라 linear-update control이다.

### 0.2 증거 등급

**바로 사용 가능한 핵심 증거**

- 8개 manifold task의 CA-LRU 및 RNN/GRU/LSTM/SSM/LRU 결과: 모두 3 seeds.
- Ring integration의 temporal/velocity/repeated-kick OOD raw 결과.
- 최종 CA-LRU의 update-term 비교: state-dependent / input-only / linear.
- 최종 CA-LRU의 \(d=1,2,4,8,16\) retention-scaling 결과.
- Final CA-LRU scaffold에서 수행한 fixed-permutation/fresh-shuffle RP-score controls. 단, “count-matched”는 아님.

**Appendix에서 조건부로 사용 가능한 증거**

- Ring on-manifold transport: 3 seeds이지만 단일 base angle, 1440-point grid, raw full-state distance.
- Torus/curve on-manifold 그림: seed 0의 post-hoc probe.
- Local Jacobian과 selective-persistence 결과: 최종 CA-LRU가 아니라 예전 linear-update scaffold.
- Old line/ring protocol: final manifold battery와 data distribution/metric이 다름.

**현재 문장 그대로는 사용하면 안 되는 주장**

1. “no-RP/all-slow가 final CA-LRU와 identical scaffold” — 실제 control은 linear-update 모델이다.
2. “all-slow \(\lambda=0.99\)” — 실제 값은 \(\lambda=0.999\)이다.
3. “input-only torus error \(0.0055\)” — 실제 3-seed 평균은 \(0.01515\pm0.00517\)이다.
4. “모든 Jacobian이 blank-input Jacobian” — integrate 결과는 driven midpoint Jacobian이다.
5. “모든 manifold에서 다섯 진단을 3-seed로 통과” — on-manifold 정량 검사는 ring 중심이며 torus/curve는 seed 0, surface는 동등한 정량 표가 없다.
6. “3–9 of 96” — 선택된 8-task 결과의 seed별 실제 범위는 **3–8**, task 평균 범위는 **3.0–6.67**이다.
7. “CA-LRU가 discrete flip-flop도 완벽히 해결” — \(\epsilon=10^{-4}\)의 2000-step full-state accuracy가 \(0.445\pm0.479\)로 seed 불안정하다.
8. “모든 task에서 1–2 orders lower post-hold” — strongest baseline 대비 개선은 전체 task에서 \(4.2\times\)–\(116\times\); integration task만 보면 \(10.7\times\)–\(116\times\)이다.
9. “normal recovery 열은 normal contraction만 측정” — 실제 값은 blank retention/drift와 kick recovery가 섞인 target RMSE이다.
10. “task별 최적 \(\epsilon\)은 held-out validation으로 선택” — 별도 validation seed/data가 없다.

## 1. 기호 체계

| 기호 | 뜻 | 차원 |
|---|---|---:|
| \(x_t\) | task input | \(d_x\) |
| \(u_t=Ex_t\) | bias-free learned embedding | \(d=96\) |
| \(h_t\) | CA-LRU의 recurrent carrier state | \(N=96\) |
| \(v_t^{\rm str}\) | full block의 output/readout stream | \(d=96\) |
| \(s_t=(h_t,v_t^{\rm str})\) | diagnostic 코드가 사용하는 CA-LRU full state | 192 |
| \(q_t\) | manifold intrinsic coordinate | 1 또는 2 |
| \(y_t=\phi(q_t)\) | ambient target | 2 또는 3 |
| \(\hat y_t\) | decoded output | \(d_y\) |
| \(F_x\) | input \(x\) 아래 full-state one-step map | — |
| \(F_0\) | blank-input full-state map | — |
| \(A_j\) | carrier coordinate \(h_j\)를 0으로 만드는 ablation | — |
| \(D\) | full-state decoder | — |

\(h\)와 \(s\)를 반드시 구분해야 한다. \(h\)에 대한 blank dynamics는 diagonal decay이지만, 코드의 diagnostic state \(s\)에는 매 step 다시 계산되는 stream이 포함된다.

## 2. 논문에 넣을 수식

### 2.1 Manifold memory task

공통 latent dynamics는

\[
q_{t+1}=\operatorname{wrap}(q_t+\omega_t),
\qquad
y_t=\phi(q_t).
\]

Hold task에서는 \(\omega_t=0\)이고, integration task에서는 piecewise-constant velocity segment를 사용한다.

#### Ring

\[
q=\theta\in[0,2\pi),
\qquad
\phi_{\rm ring}(\theta)
=
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}.
\]

#### Torus

\[
q=(\theta,\psi)\in[0,2\pi)^2,
\quad R=1,
\quad r=0.35,
\]

\[
\phi_{\rm torus}(\theta,\psi)
=
\begin{bmatrix}
(R+r\cos\psi)\cos\theta\\
(R+r\cos\psi)\sin\theta\\
r\sin\psi
\end{bmatrix}.
\]

#### Complex closed curve

\[
q=s\in[0,2\pi),
\qquad
\rho(s)=1+0.25\cos(3s),
\]

\[
\phi_{\rm curve}(s)
=
\begin{bmatrix}
\rho(s)\cos s\\
\rho(s)\sin s\\
0.25\sin(3s)
\end{bmatrix}.
\]

#### Bounded surface

\[
q=(a,b)\in[-0.95,0.95]^2,
\]

\[
\tilde\phi_{\rm surf}(a,b)
=
\begin{bmatrix}
a\\
b\\
0.4\sin(\pi a)\sin(\pi b)
\end{bmatrix},
\qquad
\phi_{\rm surf}(a,b)=R_{123}\tilde\phi_{\rm surf}(a,b).
\]

\(R_{123}\)은 seed 123으로 생성한 \(3\times3\) orthogonal rotation이다. **Random rotation은 surface에만 적용**되며 ring/torus/curve에는 적용되지 않는다.

Surface boundary에서는

\[
q_{t+1}=\operatorname{clip}(q_t+\omega_t,-0.95,0.95),
\]

이고 모델에는 command가 아니라 clamp 뒤의 실제 변위 \(q_{t+1}-q_t\)가 입력된다. 따라서 “모델이 boundary collision을 스스로 추론한다”는 주장은 피해야 한다.

### 2.2 입력 구성

Angular task의 zero-preserving velocity feature는

\[
\nu(\omega)
=
\begin{bmatrix}
\omega\\
\sin\omega\\
\cos\omega-1
\end{bmatrix},
\qquad
\nu(0)=0.
\]

다차원 angular coordinate에는 이 식을 coordinate-wise 적용한다. Surface에서는 \(\nu(\omega)=\omega\)이다.

입력 sequence는

\[
x_0=[\phi(q_0);\,0;\,1],
\qquad
x_{t+1}=[0;\,\nu(\omega_t);\,0].
\]

마지막 scalar는 cue flag이다. 별도 query channel은 없으며 target은 모든 step에서 감독된다.

### 2.3 기본 LRU

LRU full block의 complex carrier \(c_t\in\mathbb C^N\)는

\[
c_{t+1,j}
=
\rho_j e^{i\varphi_j}c_{t,j}
+\gamma_j(Bu_t)_j,
\]

\[
\rho_j=\exp[-\exp(\nu_j)],
\qquad
\gamma_j=\sqrt{\max(1-\rho_j^2,10^{-6})}.
\]

Blank input에서는

\[
c_{t+H,j}
=
\rho_j^H e^{iH\varphi_j}c_{t,j}.
\]

즉 진폭 decay뿐 아니라 phase rotation도 일어난다.

### 2.4 CA-LRU carrier recurrence

각 coordinate의 retention과 write gain은

\[
\lambda_j=\sqrt{\sigma(\theta_j)},
\qquad
\gamma_j=\sigma(a_j),
\]

이고,

\[
h_{t+1}
=
\Lambda h_t+\Gamma w_\omega(h_t,u_t),
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N),
\quad
\Gamma=\operatorname{diag}(\gamma_1,\ldots,\gamma_N).
\]

최종 CA-LRU의 state-dependent update는

\[
w_\omega(h,u)
=
g_\omega([h;u])-g_\omega([h;0]),
\]

\[
g_\omega(z)
=
W_2\operatorname{GELU}(W_1z+b_1)+b_2.
\]

따라서 모든 \(h\)에 대해

\[
w_\omega(h,0)=0
\]

가 항등적으로 성립한다. \(\gamma_j\)는 \(\lambda_j\)와 tied되지 않으며 0.5로 초기화되어 task gradient로 학습된다.

두 update control은

\[
w_{\rm linear}(h,u)=Bu,
\qquad
w_{\rm input}(h,u)=g(u),\quad g(0)=0
\]

이다.

### 2.5 Full-block scaffold

한 layer 설정에서 carrier output과 stream을 다음처럼 쓸 수 있다.

\[
r_t=P h_{t+1}+b_P,
\]

\[
a_t
=
\operatorname{GLU}
\!\left(W_G\operatorname{GELU}(r_t)+b_G\right),
\]

\[
v_t^{\rm str}
=
\operatorname{LN}_{\rm out}(u_t+a_t),
\]

\[
\hat y_t
=
W_y\operatorname{LN}_{\rm head}(v_t^{\rm str})+b_y.
\]

여기서 \(\operatorname{GLU}([z_1;z_2])=z_1\odot\sigma(z_2)\)이다. 메인 설정은 carry_stream=False이므로 이전 stream은 다음 carrier update에 사용되지 않는다.

### 2.6 Task loss

Training horizon을 \(T\), batch를 \(B\), output dimension을 \(d_y\)라 하면

\[
\mathcal L_{\rm task}
=
\frac{1}{(T+1)Bd_y}
\sum_{t=0}^{T}\sum_{b=1}^{B}
\left\|
\hat y_{t,b}-y_{t,b}
\right\|_2^2.
\]

최종 CA-LRU에는 retention에 대한 gradient regularizer가 없다. \(\theta\)는 gradient path 밖에 있고 아래 RP rule로만 갱신된다.

### 2.7 Retention Plasticity

Probe sequence를 처리한 full state를 \(s_b\), final target을 \(z_b\), \(H_{\rm RP}=500\)이라 두자.

\[
\mathcal E_{\rm mem}(s;z)
=
\frac1B\sum_{b=1}^{B}
\left\|
D(F_0^{H_{\rm RP}}(s_b))-z_b
\right\|_2^2.
\]

Carrier coordinate \(j\)의 ablation damage는

\[
\Delta_j
=
\mathcal E_{\rm mem}(A_js;z)
-
\mathcal E_{\rm mem}(s;z).
\]

RP update는

\[
\theta_j
\leftarrow
\operatorname{clip}_{[-18,18]}
\left[
\theta_j+\eta_\lambda(\Delta_j-\epsilon)
\right],
\qquad
\eta_\lambda=3000.
\]

여기서

\[
\frac{\partial\lambda_j}{\partial\theta_j}
=
\frac12\lambda_j(1-\lambda_j^2),
\]

이므로 작은 update에서는

\[
\delta\lambda_j
\approx
\frac{\eta_\lambda}{2}
\lambda_j(1-\lambda_j^2)(\Delta_j-\epsilon).
\]

\(\Delta_j>\epsilon\)이면 retention이 증가하고, \(\Delta_j<\epsilon\)이면 감소한다. 이 rule은 coordinate-dependent heuristic이며 basis-invariant optimality를 보장하지 않는다. 또한 \(\Delta_j\)는 output dimension을 합산한 squared error이므로 fixed \(\epsilon\)의 의미가 task의 \(d_y\)에 따라 달라진다.

### 2.8 Proposition 1: blank-input carrier autonomy

Bias-free encoder 때문에 \(E0=0\)이고 difference update 때문에 \(w(h,0)=0\)이다. 따라서 blank input에서 carrier map은

\[
F_0^{\rm car}(h)=\Lambda h,
\qquad
(F_0^{\rm car})^H(h)=\Lambda^Hh,
\]

즉

\[
h_{t+H,j}=\lambda_j^Hh_{t,j}.
\]

이는 학습 결과가 아니라 아키텍처 항등식이다.

단, full state \(s=(h,v^{\rm str})\)에는 이 표현을 그대로 쓰면 안 된다. 어떤 nonlinear stream map \(\Phi\)에 대해

\[
F_0(h,v^{\rm str})
=
\bigl(\Lambda h,\Phi(\Lambda h)\bigr),
\]

\[
J_0(h,v^{\rm str})
=
\begin{bmatrix}
\Lambda & 0\\
D\Phi(\Lambda h)\Lambda & 0
\end{bmatrix}.
\]

이전 stream perturbation은 다음 step에 버려지지만, 새 stream은 carrier의 비선형 함수다.

### 2.9 Proposition 2: 조건부 local signature

Persistent coordinate 집합 \(P\), drained 집합 \(D\)가 다음을 만족한다고 하자.

\[
1-\delta\le\lambda_j\le1
\quad(j\in P),
\qquad
\lambda_j\le\kappa<1
\quad(j\in D).
\]

또 learned manifold의 carrier tangent space가 \(P\)에, 분석할 normal space가 \(D\)에 놓인다고 가정한다. Carrier blank Jacobian \(J_0^{\rm car}=\Lambda\)에 대해

\[
\operatorname{spec}(Q_T^\top J_0^{\rm car}Q_T)
\subset[1-\delta,1],
\]

\[
\left\|Q_N^\top J_0^{\rm car}Q_N\right\|_2
\le\kappa,
\qquad
Q_N^\top J_0^{\rm car}Q_T=0.
\]

\(H\) step에서는

\[
\|(\Lambda^H-I)v_T\|_2
\le
\left[1-(1-\delta)^H\right]\|v_T\|_2,
\]

\[
\|\Lambda^Hv_N\|_2
\le
\kappa^H\|v_N\|_2.
\]

이 명제는 **조건부**이다. RP가 tangent-coordinate alignment를 자동으로 보장한다는 theorem이 아니며, alignment는 실험적으로 확인해야 한다.

### 2.10 Retention time scale

각 coordinate의 \(1/e\) time constant와 half-life는

\[
\tau_j=-\frac1{\log\lambda_j},
\qquad
T_{1/2,j}=\frac{\log(1/2)}{\log\lambda_j}.
\]

\(\lambda>0.99\) count는 convenient support statistic일 뿐, 1000-step persistence를 직접 보장하는 threshold는 아니다. 실제 메인 CA-LRU에서 선택된 coordinate는 대부분 수치적으로 1 근처까지 polarization된다.

### 2.11 Baseline equations

Appendix에서 구현을 명확히 할 때 다음 정도면 충분하다.

**RNN**

\[
h_{t+1}=\tanh(W_xx_t+W_hh_t+b).
\]

**GRU**

\[
r_t=\sigma(W_rx_t+U_rh_t+b_r),
\quad
z_t=\sigma(W_zx_t+U_zh_t+b_z),
\]

\[
n_t=\tanh(W_nx_t+b_n+r_t\odot(U_nh_t+b'_n)),
\]

\[
h_{t+1}=z_t\odot h_t+(1-z_t)\odot n_t.
\]

**LSTM**은 PyTorch 표준 LSTMCell을 사용하며 decoder는 \(h_t\)만 읽는다.

**Diagonal SSM**

\[
h_{t+1}=a\odot h_t+Bx_t+b,
\qquad
a=\sigma(a_{\rm raw}),
\quad
a_{{\rm raw},j}=2.2\text{ at initialization}.
\]

이 SSM은 S4/S5/Mamba가 아니라 저장소 자체의 단순 diagonal recurrence다.

## 3. 다섯 진단의 정확한 정의

### 3.1 공통 RMSE

메인 표의 scalar는 ambient output component RMSE다.

\[
e(\hat z,z)
=
\sqrt{
\frac1{Bd_y}
\sum_{b=1}^{B}\|\hat z_b-z_b\|_2^2
}.
\]

별도 저장된 vector RMSE는

\[
e_{\rm vec}(\hat z,z)
=
\sqrt{\frac1B\sum_b\|\hat z_b-z_b\|_2^2}
=
\sqrt{d_y}\,e(\hat z,z).
\]

따라서 ring/torus 표를 angle error, geodesic error, 또는 manifold-coordinate RMSE라고 부르면 안 된다.

### 3.2 Task error

\[
E_{\rm task}=e(D(s_T),y_T).
\]

Training loss는 full sequence MSE이지만 이 진단은 ID horizon 260의 마지막 output만 측정한다.

### 3.3 Post-hold error

\[
E_{\rm post}(H)
=
e\left(D(F_0^H(s_T)),y_T\right),
\qquad H\in\{500,1000\}.
\]

이는 drift increment가 아니라 target에 대한 **절대 오차**다. 논문 column명은 post-hold error가 정확하다.

### 3.4 Local tangent basis

동일한 velocity sequence를 유지하면서 initial intrinsic coordinate의 \(k\)번째 축만 \(\pm\varepsilon\) 이동시킨 full state를 \(S(q_0\pm\varepsilon e_k;\omega)\)라 두면

\[
t_k
=
\frac{
S(q_0+\varepsilon e_k;\omega)
-S(q_0-\varepsilon e_k;\omega)
}{2\varepsilon},
\qquad
\varepsilon=10^{-3}.
\]

\[
Q_T=\operatorname{qr}([t_1,\ldots,t_{d_q}]).
\]

### 3.5 Normal recovery

Batch 전체의 state scale은

\[
\sigma_s
=
\sqrt{
\frac1{d_s}
\sum_{j=1}^{d_s}\operatorname{Var}_b(s_{b,j})
}.
\]

\(\xi\sim\mathcal N(0,I)\)를 tangent-orthogonalize하여

\[
n
=
\frac{(I-Q_TQ_T^\top)\xi}
{\|(I-Q_TQ_T^\top)\xi\|_2},
\]

\[
s^{\rm pert}=s_T+r\sigma_sn,
\qquad
r\in\{0.25,0.5,1\}.
\]

Main column은

\[
E_N(r=1,R=500)
=
e\left(D(F_0^{500}(s^{\rm pert})),y_T\right).
\]

여기서 radius 1은 absolute hidden norm 1이 아니라 **model-specific full-state RMS의 1배**이다. 또한 이 metric은 clean baseline drift와 perturbation recovery를 합친 target error다.

저장된 hidden recovery ratio는

\[
G_{\rm state}(R)
=
\frac{
\mathbb E_b\|F_0^R(s_b^{\rm pert})-F_0^R(s_b)\|_2
}{
\mathbb E_b\|s_b^{\rm pert}-s_b\|_2
}.
\]

CA-LRU의 attractor claim을 강화하려면 target RMSE와 함께 \(G_{\rm state}\), clean-subtracted excess error를 보고해야 한다.

### 3.6 Tangent-shift consistency

\[
\delta=0.10\frac{\xi}{\|\xi\|_2},
\qquad
s_\delta=S(\operatorname{wrap}(q_0+\delta);\omega),
\]

\[
E_T(R)
=
e\left(D(F_0^R(s_\delta)),y_\delta\right).
\]

이 실험은 기존 hidden state에 tangent vector를 더하는 kick가 아니다. Nearby intrinsic coordinate를 cue한 뒤 동일한 sequence를 처음부터 재실행하여 얻은 **nearby clean-manifold memory의 유지 검사**다. 본문에서는 tangent-shift consistency 또는 neighbor-memory consistency로 설명하는 것이 정확하다.

### 3.7 On-manifold movement

Clean reference set을

\[
\mathcal M_{\rm clean}
=
\{S_{\rm hold}(q):q\in\mathcal Q_{\rm grid}\}
\]

라 두고,

\[
d_{\mathcal M}(s)
=
\min_{q\in\mathcal Q_{\rm grid}}
\|s-S_{\rm hold}(q)\|_2,
\]

\[
q^\star(s)
=
\arg\min_q\|s-S_{\rm hold}(q)\|_2,
\]

\[
E_{\rm nearest}
=
\left|\operatorname{wrap}
(q^\star(s)-q_{\rm target})
\right|.
\]

Ring probe는 1440-point grid, base angle 1.15 rad, \(3^\circ\)/step를 5 steps, 이후 최대 2000 blank steps를 사용한다.

Raw full-state Euclidean distance는 모델별 state scale/dimension이 달라 cross-model 절댓값 비교가 완전히 정규화되지 않는다. 제출 전에는 clean-manifold diameter 또는 local tangent scale로 나눈 normalized distance를 추가하는 것이 좋다.

### 3.8 Per-step tangent motion fraction

Integration 중 state displacement \(\Delta s_t=s_{t+1}-s_t\)에 대해

\[
M_t
=
\frac{\|Q_{T,t}Q_{T,t}^\top\Delta s_t\|_2}
{\|\Delta s_t\|_2+10^{-8}}.
\]

코드는 moving step의 \(M_t\) 평균과 최솟값을 저장한다. 이는 clean-manifold nearest-distance와 다른 보조 진단이다.

### 3.9 Local Jacobian

\[
J_x(s)=\frac{\partial F_x(s)}{\partial s}.
\]

Input/output tangent basis가 다를 수 있으므로

\[
G_T=Q_{T,{\rm out}}^\top J_xQ_{T,{\rm in}},
\qquad
G_N=Q_{N,{\rm out}}^\top J_xQ_{N,{\rm in}}.
\]

Driven integrate map에서는 \(Q_{T,{\rm in}}\neq Q_{T,{\rm out}}\)일 수 있으므로 eigenvalue보다 singular value를 주 지표로 쓰는 편이 안전하다.

Random normal residue gain은

\[
g_N^{\rm rand}
=
\mathbb E_{\|a\|=1}
\left[
\|Q_{N,{\rm out}}^\top
J_xQ_{N,{\rm in}}a\|_2
\right].
\]

현재 저장된 Jacobian 결과는 final state-dependent CA-LRU가 아니라 linear-update scaffold다. Hold에서는 blank Jacobian, integrate에서는 midpoint의 actual velocity input을 사용한 driven Jacobian이다.

### 3.10 Retention support와 intrinsic dimension

\[
K_{0.99}=\sum_{j=1}^{N}\mathbf 1[\lambda_j>0.99],
\qquad
S_\lambda=\sum_{j=1}^{N}\lambda_j.
\]

Centered state matrix의 singular value를 \(\sigma_i\)라 하면 PCA participation ratio는

\[
\operatorname{PR}
=
\frac{(\sum_i\sigma_i^2)^2}{\sum_i\sigma_i^4}.
\]

Exp88의 PCA는 CA-LRU carrier 96뿐 아니라 carrier+stream 192 전체를 사용한다. Retention count는 carrier 96에만 정의되므로 두 statistic이 같은 vector space에서 계산된다고 쓰면 안 된다.

## 4. 재현 가능한 실험 방법론

### 4.1 Main manifold battery

| geometry | \(q_0\) | output dim | input dim | max velocity magnitude |
|---|---|---:|---:|---:|
| ring | \(U[0,2\pi)\) | 2 | 6 | \(3^\circ/\)step |
| torus | \(U[0,2\pi)^2\) | 3 | 10 | coordinate-wise \(2.5^\circ/\)step |
| complex curve | \(U[0,2\pi)\) | 3 | 7 | \(2.5^\circ/\)step |
| surface | \(U[-0.95,0.95]^2\) | 3 | 6 | 0.018/step |

속도는 위 값을 고정해 쓰는 것이 아니라 move segment마다 Uniform\((-{\rm max},+{\rm max})\)로 샘플한다.

Integration segment 구성:

| item | 값 |
|---|---:|
| hold length | 5–20 steps |
| move length | 3–10 steps |
| final hold | 20–80 steps |
| ring/curve modes | hold 0.25, move 0.75 |
| torus modes | hold/theta/psi/both 각 0.25 |
| surface modes | hold 0.25, \(a\) 0.30, \(b\) 0.30, both 0.15 |

각 batch sample은 독립적인 segment schedule을 가진다.

### 4.2 Training

| 항목 | 값 |
|---|---:|
| optimizer steps | 10,000 |
| batch | 256 |
| horizon | integer uniform 80–260; 실제 sequence length \(T+1\) |
| optimizer | AdamW |
| learning rate | \(10^{-3}\) |
| weight decay | \(10^{-5}\) |
| betas / optimizer epsilon | PyTorch default |
| gradient clipping | global norm 1.0 |
| scheduler / early stopping | 없음 |
| model width | 96 |
| recurrent width | 96 |
| layers | 1 |
| dropout | 0 |
| seeds | 0, 1, 2 |
| data | online fresh sampling |
| checkpoint | final step |

Python, NumPy, Torch, CUDA RNG를 같은 seed로 초기화한다. 그러나 architecture마다 initialization에서 소비하는 RNG 양이 달라 모델 간 train/eval sample은 paired되지 않는다.

### 4.3 RP schedule

| 항목 | 값 |
|---|---:|
| initial \(\lambda\) | deterministic linspace 0.999 → 0.90 |
| probe frequency | 100 optimizer steps |
| warm-up | first 30% = 3000 steps |
| first applied update | step 3100 |
| probe batch | 96 |
| probe task horizon | 260 |
| blank ablation horizon | 500 |
| \(\eta_\lambda\) | 3000 |
| \(\epsilon\) grid | \(0,3\times10^{-5},10^{-4}\) |
| threshold mode | fixed absolute |
| \(\theta\) clamp | \([-18,18]\) |

### 4.4 Main evaluation

| 진단 | 설정 |
|---|---|
| ID task | horizon 260, eval batch 256 |
| post-hold | +500, +1000 blank |
| tangent shift | intrinsic norm 0.10; recovery 0/20/100/500 |
| normal kick | state-RMS multiplier 0.25/0.5/1.0; recovery 0/20/100/500 |
| PCA | batch 96, horizon 260의 모든 full states |
| persistent count | \(\lambda>0.99\) |

각 seed당 대체로 metric별 단일 fresh eval batch를 사용한다. 고정 evaluation dataset과 별도 evaluation seed는 없다.

정상 perturbation은 모델의 전체 diagnostic state에 적용된다. CA-LRU에서는 carrier 96 + stream 96이고, complex LRU에서는 complex carrier의 real representation 192 + stream 96이다. Stream 좌표는 다음 step에 덮어써지므로 full-state random kick의 일부가 구조적으로 한 step 만에 사라진다. 공정한 carrier recovery를 주장하려면 carrier-only perturbation을 추가해야 한다.

### 4.5 Temporal/velocity OOD

Temporal OOD horizons는 500/1000/2000이다. Integration에서는 단순히 horizon만 늘리는 것이 아니라 distribution도 다음처럼 바뀐다.

| 항목 | ID | temporal OOD |
|---|---:|---:|
| ordinary hold | 5–20 | 30–120 |
| move | 3–10 | 3–10 |
| final hold | 20–80 | 100–250 |

따라서 T2000은 sequence length와 dwell-time distribution이 함께 바뀐 OOD다.

Velocity OOD는 maximum velocity를 1.0×/1.5×/2.0×로 바꾼다. 각 scale은 같은 trajectory의 rescaling이 아니라 새로운 batch다.

기존 원고 OOD 표는 endpoint가 섞여 있다.

- in-dist: ID sequence 마지막의 final RMSE.
- T2000: 2000-step temporal OOD 뒤 추가 500 blank RMSE.
- velocity: ID-length scaled-velocity sequence 뒤 추가 500 blank RMSE.
- normal/kicks: T2000 OOD state에서 측정.

최종 표는 final-to-final 또는 post-H500-to-post-H500 중 하나로 통일해야 한다.

### 4.6 Repeated kicks

500-step blank rollout 중 event time을

\[
\mathcal T_K
=
\operatorname{unique}
\left(
\operatorname{linspace}(1,500,K)
\right),
\quad K\in\{1,20,100\}
\]

로 잡는다. 각 event에서 blank update 후 kick을 더한다. \(K=100\)이면 마지막 kick이 step 500에 있어 최종 kick 후 recovery가 없다. 따라서 “100 consecutive kicks followed by 500-step recovery”가 아니라 “500 blank steps 동안 균등 분산된 100 kicks, 마지막 kick 직후 평가”이다.

각 kick은 최초 base state에서 계산한 tangent basis를 사용해 normal 방향으로 projection하며, rollout 중 basis를 다시 계산하지 않는다.

### 4.7 On-manifold ring transport

| 항목 | 값 |
|---|---:|
| base angle | 1.15 rad |
| drive | \(3^\circ/\)step × 5 |
| commanded displacement | \(15^\circ\) |
| clean grid | 1440 angles, spacing \(0.25^\circ\) |
| clean hold | 260 blank steps |
| post-drive blank | 0/1/5/20/100/500/1000/2000 |
| seeds | 0/1/2 |

현재 CSV probe는 drive state를 cue 직후부터 5 steps 이동시킨 뒤 clean-hold-260 manifold와 비교한다. 더 엄격한 재실험에서는 clean base state를 260 steps 안정화한 뒤 그 state에서 drive를 시작하는 것이 낫다.

### 4.8 Retention scaling

이 실험은 Exp88 manifold battery가 아니라 별도 Exp72 line-integration protocol이다.

| 항목 | 값 |
|---|---:|
| intrinsic dimension | 1, 2, 4, 8, 16 |
| model | final CA-LRU, PAN-RNW-full |
| \(\epsilon\) | \(10^{-4}\) |
| seeds | 0/1/2 |
| train horizon | 10–50 |
| \(z_0\) | \(U[-0.5,0.5]^d\) |
| velocity | iid \(U[-0.08/\sqrt d,0.08/\sqrt d]^d\) per step |
| train/eval batch | 256 / 512 |
| PCA | analysis horizon 50의 final full states, batch 96 |
| long OOD | independent 2000-step random walk + 500 blank |

### 4.9 Discrete scope control

Discrete task는 3000 training steps, horizon 10–50이다.

- K-way: \(K=16\), one-hot cue, full-sequence cross entropy.
- Flip-flop: 4 bits, bit마다 step당 probability 0.08의 set/reset pulse, full-sequence BCE.
- Eval horizons: 50/100/200/500/1000/2000.
- Basin \(r=1\)은 여기서는 absolute Euclidean radius 1.
- Transition probe는 16 prototypes × 4 single-bit flips의 64 경우를 즉시 및 20 blank 뒤 평가한다.

이 결과는 CA-LRU 우위가 아니라 continuous regime에 claim을 제한하는 근거로 사용한다.

### 4.10 Baseline와 parameter count

| model | 구현 | trainable params 범위 |
|---|---|---:|
| RNN | tanh RNNCell + 64-unit tanh MLP readout | 16.3k–16.8k |
| GRU | GRUCell + 같은 readout | 36.3k–37.5k |
| LSTM | LSTMCell, decoder reads \(h\) | 46.3k–47.9k |
| SSM | learned real diagonal recurrence | 7.1k–7.6k |
| LRU | complex diagonal full block | 56.9k–57.4k |
| CA-LRU | real diagonal + recurrent update full block | 57.0k–57.5k |
| input-only update | same outer block | 47.6k–48.1k |
| linear update | same outer block | 38.4k–38.9k |

LRU와 CA-LRU는 parameter count가 거의 같지만, RNN/GRU/LSTM/SSM은 width만 같고 scaffold/parameter count는 matched되지 않는다. LRU의 complex pole 하나는 real state 두 좌표에 해당하므로 LRU와 CA-LRU의 lambda count도 완전히 같은 state-coordinate 단위가 아니다.

Mamba, RG-LRU, LinOSS 같은 현대 baseline이 없고 SSM도 단순 diagonal recurrence라는 점은 현재 가장 큰 실험적 약점이다.

### 4.11 Hardware와 시간

기존 실험은 local NVIDIA TITAN RTX 24 GB 6장을 사용했다. Exp88 raw JSON의 wall-clock 평균은 task에 따라 크게 달라지지만, 대략 GRU보다 CA-LRU가 느리고 complex LRU보다는 비슷하거나 빠른 구간이 있다. GPU가 공유되었고 surface runs의 시간 분산이 매우 크므로 절대 throughput benchmark로 쓰지 말고, Appendix에서 “observed shared-machine wall time”으로만 보고한다.

### 4.12 재현성상 현재 빠진 것

- Train/validation/test split 또는 fixed validation/test set.
- Held-out rule에 따른 task별 \(\epsilon\) 선택.
- 모든 model이 공유하는 paired evaluation sequence.
- Git commit hash와 dependency lock.
- PyTorch/CUDA/cuDNN version.
- Deterministic algorithm 설정.
- Main table을 생성하는 단일 공식 script.

## 5. 실제 데이터

### 5.1 Main CA-LRU rows

선택된 task별 \(\epsilon\)은 ring hold/integrate 및 curve integrate에서 \(10^{-4}\), 나머지에서 \(3\times10^{-5}\)이다. 아래는 mean ± sample SD, \(n=3\)이다.

| task | task RMSE | post-H1000 | normal-1×R500 | tangent-R500 | \(\lambda>.99\) |
|---|---:|---:|---:|---:|---:|
| ring hold | 0.000735 ± 0.000414 | 0.000735 ± 0.000414 | 0.03138 ± 0.00259 | 0.000738 ± 0.000418 | 3.00 ± 0.00 |
| ring integrate | 0.002182 ± 0.002037 | 0.002182 ± 0.002037 | 0.01605 ± 0.00113 | 0.002160 ± 0.002022 | 6.00 ± 1.00 |
| torus hold | 0.002222 ± 0.000142 | 0.002222 ± 0.000142 | 0.02364 ± 0.000590 | 0.002226 ± 0.000139 | 3.67 ± 0.58 |
| torus integrate | 0.006904 ± 0.000789 | 0.006904 ± 0.000789 | 0.02365 ± 0.00394 | 0.006918 ± 0.000760 | 6.00 ± 0.00 |
| complex-curve hold | 0.003362 ± 0.00188 | 0.003362 ± 0.00188 | 0.02269 ± 0.00140 | 0.003363 ± 0.00188 | 4.33 ± 0.58 |
| complex-curve integrate | 0.003624 ± 0.000636 | 0.003624 ± 0.000636 | 0.01187 ± 0.000300 | 0.003384 ± 0.000365 | 6.67 ± 1.15 |
| surface hold | 0.001690 ± 0.000219 | 0.001690 ± 0.000219 | 0.01612 ± 0.00133 | 0.001702 ± 0.000226 | 4.33 ± 0.58 |
| surface integrate | 0.008057 ± 0.00177 | 0.008057 ± 0.00177 | 0.02332 ± 0.00110 | 0.007960 ± 0.00148 | 4.33 ± 0.58 |

모든 baseline의 동일 수치와 raw provenance는 `paper/evidence/tables/main_manifold_metrics.csv`에 있다.

### 5.2 Strongest evaluated baseline과의 정확한 비교

모든 축에서 strongest standard baseline은 GRU다.

| task | GRU task | GRU post-H1000 | GRU normal | GRU tangent |
|---|---:|---:|---:|---:|
| ring hold | 0.00238 | 0.0139 | **0.0149** | 0.00710 |
| ring integrate | 0.0199 | 0.254 | 0.111 | 0.103 |
| torus hold | **0.00193** | 0.0219 | **0.0156** | 0.0108 |
| torus integrate | 0.0510 | 0.268 | 0.160 | 0.159 |
| curve hold | **0.00155** | 0.0142 | **0.0156** | 0.00711 |
| curve integrate | 0.0214 | 0.264 | 0.145 | 0.149 |
| surface hold | 0.00550 | 0.0216 | 0.0162 | 0.0127 |
| surface integrate | 0.0168 | 0.0864 | 0.0525 | 0.0498 |

안전한 해석:

- CA-LRU는 모든 integration task에서 task/post-hold/normal-target/tangent-target error가 GRU보다 낮다.
- Hold에서는 GRU가 torus/curve의 ID task error와 ring/torus/curve의 normal-target error에서 더 낮다.
- CA-LRU의 post-H1000 및 tangent consistency는 모든 task에서 가장 낮다.
- “모든 개별 metric에서 best”가 아니라 “retention·neighbor-memory·integration을 동시에 만족하는 유일한 evaluated model” 정도가 데이터에 맞다. 단, formal pass threshold가 없으므로 “통과”라는 이산 판정은 정의가 더 필요하다.

### 5.3 Update-term ablation

각 task의 main CA-LRU와 같은 \(\epsilon\)을 사용한 task RMSE다.

| update | ring integrate | torus integrate | curve integrate | surface integrate |
|---|---:|---:|---:|---:|
| state-dependent | **0.00218** | **0.00690** | **0.00362** | **0.00806** |
| input-only | 0.00583 | 0.01515 | 0.01470 | 0.00975 |
| linear | 0.00738 | 0.01745 | 0.01852 | 0.00990 |

기존 원고의 torus input-only \(0.0055\)는 오류다. 이 정정 뒤에는 state-dependent update가 네 task 모두에서 가장 낮다.

### 5.4 RP evidence

#### Final CA-LRU scaffold에서 유효한 score-alignment control

아래 비교는 모두 \(\epsilon=3\times10^{-5}\), \(n=3\)이다.

| task/control | final | post-H1000 | \(\lambda>.99\) |
|---|---:|---:|---:|
| ring hold, aligned RP | 0.00169 | 0.00169 | 3.0 |
| ring hold, fixed permutation | 0.00490 | 0.00497 | 11.3 |
| ring hold, fresh shuffle | 0.00297 | 0.00299 | 94.7 |
| ring integrate, aligned RP | 0.00486 | 0.00486 | 6.3 |
| ring integrate, fixed permutation | 0.00947 | 0.0328 | 14.7 |
| ring integrate, fresh shuffle | 0.01145 | 0.01141 | 95.0 |
| torus integrate, aligned RP | 0.00690 | 0.00690 | 6.0 |
| torus integrate, fixed permutation | 0.00472 | 0.00729 | 27.3 |
| torus integrate, fresh shuffle | 0.00619 | 0.00619 | 96.0 |

Fixed permutation은 count-matched가 아니다. 정확한 결론은 “damage alignment가 더 작은 support로 flat post-hold를 얻는 데 도움이 된다”이다. Torus final error에서는 fixed permutation이 오히려 더 낮지만 더 큰 support와 post-hold drift를 보인다.

#### Linear-update scaffold에서만 유효한 no-RP/all-slow control

| linear variant, ring hold | task | post-H1000 | \(\lambda>.99\) |
|---|---:|---:|---:|
| RP, \(\epsilon=10^{-4}\) | 0.00380 | 0.00380 | 3.67 |
| no RP | 0.00932 | 0.4976 | 9.0 |
| all-slow \(\lambda=0.999\) | 0.00291 | 0.1402 | 96.0 |

이 표는 RP mechanism의 supporting control로 쓸 수 있지만 final CA-LRU의 causal ablation이라고 부르면 안 된다. 제출 전 PAN-RNW-full에서 no-RP/all-slow를 다시 실행해야 한다.

### 5.5 Ring on-manifold transport

Main \(\epsilon=10^{-4}\), blank 2000, 3 seeds의 aggregate다.

| model | decoded displacement | nearest-clean distance | nearest angle error |
|---|---:|---:|---:|
| CA-LRU | 14.96° ± 0.03° | 0.045 ± 0.026 | 0.140° ± 0.000° |
| input-only | 15.02° ± 0.16° | 1.191 ± 0.015 | 12.31° ± 0.95° |
| linear | 15.00° ± 0.20° | 1.185 ± 0.119 | 12.22° ± 0.14° |
| GRU | 14.10° ± 15.94° | 0.051 ± 0.043 | 10.79° ± 5.70° |
| LRU | 24.46° ± 138.19° | 8.48 ± 3.44 | 93.70° ± 63.65° |

CA-LRU의 nearest-angle 결과는 강하지만, \(0.140^\circ\)의 seed-identical 값은 \(0.25^\circ\) grid quantization과 단일 base point의 영향을 받는다. Raw seed 값은 `paper/evidence/tables/ring_transport_seed_values.csv`에 있다.

기존 원고의 seed-0 distance 0.067은 \(\epsilon=3\times10^{-5}\) checkpoint에서 왔고, 3-seed range 0.024–0.074는 \(\epsilon=10^{-4}\) checkpoint에서 왔다. 한 표에서 섞지 말고 main epsilon을 쓰면 seed-0 값은 0.0736이다.

### 5.6 OOD

Ring integration, \(n=3\). 아래는 endpoint가 일치하는 값이다.

| model | ID final | T2000 final | ID post-H500 | T2000 post-H500 | vel1.5 final | vel2 final |
|---|---:|---:|---:|---:|---:|---:|
| CA-LRU | 0.00218 | 0.00400 | 0.00218 | 0.00400 | 0.00347 | 0.00843 |
| linear update | 0.00858 | 0.106 | 0.00858 | 0.106 | 0.0536 | 0.163 |
| GRU | 0.0199 | 0.472 | 0.1067 | 0.540 | 0.0571 | 0.172 |
| LSTM | 0.0862 | 0.510 | 0.256 | 0.599 | 0.0858 | 0.123 |
| LRU | 0.0544 | 0.899 | 0.856 | 0.898 | 0.131 | 0.226 |

CA-LRU의 final-to-final T2000 degradation은 약 \(1.83\times\), GRU는 \(23.7\times\)다. Post-H500-to-post-H500에서는 CA-LRU 약 \(1.83\times\), GRU 약 \(5.1\times\)다.

T2000 OOD state에서:

| model | normal-1×R500 | repeated-100/H500 |
|---|---:|---:|
| CA-LRU | **0.0167** | **0.1666** |
| linear update | 0.108 | 0.263 |
| GRU | 0.540 | 0.559 |
| LSTM | 0.599 | 0.601 |
| LRU | 0.898 | 0.917 |

### 5.7 Retention scaling

| \(d\) | PCA PR | PCA dim90 | \(\lambda>.99\) | \(\sum\lambda\) | T2000+post500 RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.485 ± 0.116 | 2.00 ± 0.00 | 6.67 ± 1.53 | 6.679 ± 1.527 | 0.570 ± 0.150 |
| 2 | 2.820 ± 0.097 | 3.00 ± 0.00 | 8.00 ± 0.00 | 8.011 ± 0.0001 | 0.192 ± 0.012 |
| 4 | 5.958 ± 0.355 | 6.67 ± 0.58 | 8.33 ± 0.58 | 8.345 ± 0.577 | 0.0640 ± 0.0116 |
| 8 | 10.701 ± 0.301 | 11.00 ± 0.00 | 11.33 ± 0.58 | 11.344 ± 0.577 | 0.0383 ± 0.0025 |
| 16 | 18.741 ± 0.547 | 20.33 ± 0.58 | 13.00 ± 0.00 | 13.011 ± 0.0001 | 0.259 ± 0.0045 |

PCA PR/dim90가 \(d\)와 함께 증가하고 persistent count는 6.7→13으로 sublinear하게 증가한다는 descriptive statement는 가능하다. 다만 long-horizon RMSE가 \(d=1\)에서도 높고 \(d\)에 대해 비단조적이므로 \(d=16\)만을 새로운 capacity threshold로 단정하면 안 된다.

### 5.8 Discrete scope control

| model | K-way H2000 | K-way basin r=1 | flip-flop H2000 full | flip-flop transition-after |
|---|---:|---:|---:|---:|
| RNN | 1.000 | 0.997 | 1.000 | 1.000 |
| GRU | 0.497 ± 0.160 | 0.565 ± 0.108 | 1.000 | 1.000 |
| LSTM | 1.000 | 1.000 | 1.000 | 1.000 |
| LRU | 0.061 ± 0.009 | 0.089 ± 0.044 | 0.753 ± 0.017 | 0.344 ± 0.327 |
| CA-LRU, \(3\times10^{-5}\) | 1.000 | 0.988 ± 0.014 | 0.430 ± 0.330 | 0.578 ± 0.383 |
| CA-LRU, \(10^{-4}\) | 1.000 | 0.982 ± 0.025 | 0.445 ± 0.479 | 1.000 |

결론은 “RNN/LSTM이 discrete regime을 이미 안정적으로 해결하며 CA-LRU의 target regime은 continuous memory”여야 한다. CA-LRU가 flip-flop 전체를 완벽히 해결한다고 쓰면 안 된다.

### 5.9 Local Jacobian과 selective persistence

현재 결과는 linear-update scaffold에 한정된다.

| task | variant | tangent quantity | random normal gain |
|---|---|---:|---:|
| line hold | RP | 1.000 | 0.238 |
| line hold | all-slow | 0.999 | 0.768 |
| line hold | shuffle | 1.000 | 0.767 |
| ring hold | RP | 1.000 | 0.450 |
| ring hold | all-slow | 1.000 | 0.967 |
| ring hold | shuffle | 1.000 | 0.833 |
| ring integrate, driven | RP | 0.998 | 0.356 |
| ring integrate, driven | all-slow | 0.998 | 0.774 |
| ring integrate, driven | shuffle | 0.998 | 0.757 |

Normal eigenvalue maximum은 RP에서도 대략 1이므로 “모든 normal direction이 strict contraction”이라고 쓸 수 없다. Random normal residue가 평균적으로 작다는 결과다.

Line integrate의 step-500 selective-persistence 결과도 linear scaffold다.

| variant | nuisance residual | signal retained |
|---|---:|---:|
| RP | 0.161 | 0.824 |
| all-slow | 0.249 | 0.464 |
| shuffle | 0.651 | 0.920 |

이 두 표는 mechanism intuition에는 유용하지만 최종 CA-LRU 자체의 local proof로 쓰면 안 된다.

### 5.10 Old line protocol

Appendix D의 기존 line tables는 집계 방식이 일관되지 않는다. 예를 들어 linear-update \(d=1\) raw 5-seed mean은 task/post/normal/tangent가 0.4979/0.4975/0.00270/0.00196인데 원고는 0.47/0.47/0.0031/0.0024다. 앞쪽은 median, 뒤쪽은 초기 3-seed mean에 가깝다.

Old line results를 유지하려면 raw seed를 하나의 rule, 예를 들어 mean ± sample SD로 다시 집계해야 한다. 그렇지 않으면 final manifold battery와 retention-scaling만 남기는 편이 낫다.

## 6. 원고에 넣을 권장 주장

### 6.1 강하게 쓸 수 있는 주장

1. **Architectural identity:** blank task input에서 CA-LRU carrier update는 정확히 \(h^+=\Lambda h\)로 줄어든다.
2. **Long blank retention:** 모든 8 task에서 CA-LRU의 post-H1000 error가 strongest evaluated baseline보다 낮다.
3. **Integration:** 네 integration geometry 모두에서 CA-LRU가 evaluated standard baselines보다 낮은 task/post/normal-target/tangent-target error를 보인다.
4. **Update mechanism:** state-dependent update가 동일 task별 threshold에서 input-only 및 linear update보다 낮은 integration task error를 보인다.
5. **Ring transport:** decoded target만 맞추는 linear/input-only update와 달리 state-dependent CA-LRU는 ring clean-state family의 올바른 neighboring state에 도달한다.
6. **Compact retention:** selected main results에서 \(\lambda>0.99\) coordinate는 seed별 3–8/96이다.
7. **OOD:** ring integration에서 final-to-final T2000 및 velocity extrapolation의 degradation이 GRU/LRU보다 작다.

### 6.2 제한해서 쓸 주장

- Approximate continuous attractor는 exact invariant manifold가 아니라 finite-horizon behavioral/local sense로만 사용한다.
- Normal recovery는 “lower target error after a normal kick”로 쓰고 pure normal-contraction rate라고 단정하지 않는다.
- RP alignment는 fixed-permutation/fresh-shuffle 결과로 “compactness와 stability에 기여”한다고 쓰되 global optimality를 주장하지 않는다.
- Jacobian/variance-washout 결과는 “linear-update scaffold mechanism study”로 명시한다.
- On-manifold movement의 generality는 ring에서 정량 확인, torus/curve에서 qualitative seed-0 support로 제한한다.
- “유일하게 다섯 축을 통과”를 유지하려면 각 축의 pass threshold를 사전에 정의해야 한다.

### 6.3 제출 전 재실험 우선순위

1. **최종 CA-LRU matched RP ablation:** PAN-RNW-full에서 eta=0, all-slow \(\lambda=0.999\); 3–5 seeds.
2. **Final CA-LRU blank Jacobian:** carrier-only와 full-state를 분리하고 hold/integrate 모두 zero input에서 측정.
3. **Perturb recurrent carrier only:** overwrite되는 stream을 제외하고 kick; state scale을 model-comparable하게 정규화.
4. **Held-out hyperparameter selection:** train seeds와 별도의 validation seed/batch로 \(\epsilon\) 선택 후 untouched test seeds에 보고.
5. **Fixed evaluation set:** 모든 model/velocity scale에 같은 \(q_0\), segment schedule, velocity path를 사용.
6. **On-manifold test 확대:** 여러 base points, 양방향/multiple speed, 3–5 seeds, normalized manifold distance; torus/curve/surface 포함.
7. **Modern baselines:** 최소 RG-LRU 또는 Mamba 계열 하나와 stronger SSM/LinOSS 계열 하나.
8. **통계:** seed별 dots, mean±SD, 가능하면 5 seeds; task family 단위 paired comparison.

## 7. 파일과 재현 경로

### 7.1 이 감사에서 생성한 파일

- `scripts/build_evidence.py`: raw JSON/CSV를 다시 집계하는 표 생성기.
- `paper/evidence/tables/main_manifold_metrics.csv`: 8 tasks × CA-LRU/baselines.
- `paper/evidence/tables/update_ablation_metrics.csv`: state-dependent/input-only/linear.
- `paper/evidence/tables/ring_integrate_ood_metrics.csv`: endpoint별 OOD 값.
- `paper/evidence/tables/epsilon_sensitivity.csv`: 8 tasks × 3 thresholds.
- `paper/evidence/tables/rp_control_metrics.csv`: linear no-RP/all-slow 및 final-scaffold shuffle controls.
- `paper/evidence/tables/retention_scaling_metrics.csv`: \(d=1..16\).
- `paper/evidence/tables/ring_transport_seed_values.csv`: ring transport raw seed values.
- `paper/evidence/tables/ring_transport_summary.csv`: ring transport aggregate.
- `paper/evidence/tables/discrete_control_metrics.csv`: K-way/flip-flop scope controls.

재생성:

~~~bash
# repository root에서
python scripts/build_evidence.py
~~~

### 7.2 핵심 source code

아래 경로는 `repro/legacy_code/`에 보존한 실험 스냅샷을 기준으로 한다. 논문용 명칭은 정리했지만, 기존 checkpoint·result와의 호환성을 위해 실험 내부의 `PAN`/`CAMN` tag는 변경하지 않았다.

| 내용 | 파일/라인 |
|---|---|
| CA-LRU/LRU recurrence | pan_block.py:75–347 |
| full block/readout | pan_block.py:350–616 |
| variant mapping | exp71_pan_block_pulse_hold.py:1008–1210 |
| task geometry/input | exp88_manifold_attractor_tasks.py:38–307 |
| main diagnostics | exp88_manifold_attractor_tasks.py:337–567 |
| training loop | exp88_manifold_attractor_tasks.py:570–751 |
| RP damage/update | exp72_structured_attractor_tasks.py:515–594 |
| launcher hyperparameters | run_exp88_manifold_main.py:97–233 |
| writer variants | run_exp88_writer_sweep.py:14–46 |
| fixed shuffle | run_exp88_shuffle_matched.py:18–59 |
| ring transport | analyze_exp88_ring_transport.py:61–142 |
| Jacobian | analyze_latent_jacobian_controls.py:52–269 |
| retention scaling | run_e7_line_dim_rnw.py:17–46 |
| discrete control | run_e8_discrete_full.sh, exp74_discrete_attractor_tasks.py |

### 7.3 원자료 위치

표 재집계에 필요한 JSON/CSV 스냅샷은 `paper/evidence/raw/`에 포함했다. 아래는 이 스냅샷을 가져온 기존 작업 폴더의 상대 경로이며, checkpoint·trace·log는 새 저장소에 포함하지 않았다.

- Final CA-LRU/input-only: exp88_writer_sweep_results/
- Baselines/linear controls: exp88_manifold_main_results/
- Checkpoints: checkpoints_exp88_writer_sweep/, checkpoints_exp88_manifold_main/
- Lambda traces: traces_exp88_writer_sweep/, traces_exp88_manifold_main/
- Ring transport: analysis_exp88_ring_transport/
- Repeated kicks: analysis_exp88_repeated_kicks/
- OOD summary: figures_exp88_ood_summary/
- Jacobian: analysis_latent_jacobian_controls/
- Selective persistence: analysis_selective_persistence/
- Scaling: exp72_line_integrate_10k_d*_camn_rnw_eps1e-4_results/
- Discrete: exp89_discrete_kway_quick_results/

### 7.4 대표 재현 명령

~~~bash
# repository root에서
cd repro/legacy_code

python run_exp88_manifold_main.py --seeds 0,1,2
python run_exp88_surface_main.py --seeds 0,1,2
python run_exp88_writer_sweep.py --seeds 0,1,2 --include-surface
python run_exp88_shuffle_matched.py --seeds 0,1,2

python run_e7_line_dim_rnw.py --seeds 0,1,2
bash run_e8_discrete_full.sh 5
~~~

실행 전 launcher의 기본 GPU, output overwrite 옵션, 그리고 이미 존재하는 checkpoint 경로를 확인해야 한다.

## 8. 재현성 체크리스트

최종 Appendix에는 다음을 추가해야 한다.

- Git commit hash와 dirty-worktree 여부.
- Python/PyTorch/CUDA/cuDNN 버전 및 GPU model.
- Deterministic algorithm 설정 여부.
- 정확한 table-generation script와 column mapping.
- Model-selection rule 및 validation/test 분리.
- 모든 main row의 seed별 값 또는 mean±SD.
- Task metric이 ambient component RMSE임을 명시.
- CA-LRU carrier state와 diagnostic full state의 차원/구성.
- Normal radius가 state RMS multiplier임을 명시.
- Repeated-kick event timing과 마지막 kick 뒤 recovery가 없음을 명시.
- Temporal OOD가 horizon뿐 아니라 hold-length distribution도 바꾼다는 점.
- On-manifold distance normalization 및 grid resolution.

이 체크리스트를 충족하기 전에는 exact attractor, global stability, all five tests on every manifold, same-scaffold RP ablation을 쓰지 않는다.
