---
title: "Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"
date: 2026-07-24 10:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Motion Blur, Rolling Shutter, Visual-Inertial Odometry, Deblurring, ECCV]
description: "ECCV 2024"
math: true
toc: true
---

> [[Paper](https://arxiv.org/abs/2403.13327)] [[Page](https://spectacularai.github.io/3dgs-deblur/)] [[Github](https://github.com/SpectacularAI/3dgs-deblur)]
>
> Otto Seiskari, Jerry Ylilammi, Valtteri Kaatrasalo, Pekka Rantalankila, Matias Turkulainen, Juho Kannala, Esa Rahtu, Arno Solin

## Introduction

Gaussian Splatting(3DGS) 기반 고품질 scene reconstruction과 novel-view synthesis는 보통 흔들림 없는 고품질 사진을 요구한다. 그러나 handheld 카메라(특히 스마트폰)로 캡처하면 이 조건을 만족하기 어렵다. 움직이는 센서에서 얻은 이미지는 두 가지 왜곡에 취약하다.

- Motion blur: 셔터가 열려 있는 노출(exposure) 시간 동안 카메라와 장면의 상대적 움직임으로 발생한다.
- Rolling shutter(RS) distortion: 센서가 장면을 line-by-line(행 단위)으로 순차 스캔하기 때문에, 빠른 움직임에서 warping 왜곡이 생긴다.

이 두 효과는 reconstruction 품질을 떨어뜨릴 뿐 아니라, COLMAP 같은 photogrammetry 소프트웨어의 pose registration 실패 확률도 높인다.

기존 방법 대부분은 2D 입력 이미지를 직접 sharpening(classical 또는 deep learning 기반)하지만, 장면의 3D image formation model을 활용하지 못한다. NeRF 계열에서는 Deblur-NeRF, BAD-NeRF가 3D 표현으로 이를 풀었고, 3DGS 계열에서는 별개 문제인 de-focus blur(초점 흐림)를 다룬 연구가 있었다.

이 논문은 3DGS 프레임워크에서 motion blur와 rolling shutter를 함께 보정하는 방법을 제안한다. 핵심 차별점은 다음과 같다.

1. Blur kernel을 학습하는 대신(Deblur-NeRF 방식), camera motion과 rolling shutter를 포함한 물리적 image formation 과정을 직접 모델링한다.
2. Visual-Inertial Odometry(VIO) — IMU 데이터와 monocular video를 융합 — 로 추정한 velocity(선속도·각속도)를 활용한다. 노출 시간 동안 camera pose는 non-static으로 간주된다.
3. Screen space approximation을 도입한 differentiable rendering pipeline으로, 3DGS 연산을 반복 계산하지 않고 효율적으로 blur·RS 효과를 반영한다. MLP를 추가하지 않는다(Deblur-NeRF와 대조).

![Fig1](/assets/img/gsmove/fig1_cozyroom.png)
_Fig 1. Synthetic cozyroom 장면에서 motion blur / rolling shutter / pose noise 각 효과에 대해 보정 없이(Baseline)와 보정 적용(Ours) 비교. 우하단(pose noise 보정) 결과는 reference와 육안으로 구분이 어려울 정도다(PSNR 36.2)._

## Related Work

### Image Deblurring
Motion blur 보정은 single/multi-image 세팅에서 오래 연구됐다. Classical 방법은 sharp image와 blur kernel을 함께 recover하는 optimization이며, 여러 kernel이 같은 blurry image를 만들 수 있어 ill-posed 문제다. Deep learning 방법(SRN, MPR 등)은 대규모 데이터로 학습한 feature로 sharp image를 복원하며 spatially-varying blur를 더 잘 다룬다. Rolling shutter 보정은 전통적으로 행 단위 노출을 고려한 pixel/image warping 문제로 연구됐고, 최근에는 3D image formation과 camera motion 정보를 활용하거나 SfM·Visual-Inertial SLAM 맥락에서 다뤄졌다.

### Deblurring 3D Implicit Representations
Deblur-NeRF와 3DGS 기반 후속 연구는 blur 효과를 모델링하는 작은 MLP를 baseline에 추가하고, 학습 이미지를 시간상 고정된 것으로 취급한다. BAD-NeRF는 캡처 과정의 motion trajectory를 반영해 per-frame image formation을 모델링하며, 본 논문은 이 BAD-NeRF와 가장 유사하다(짧은 camera trajectory에 걸쳐 정보를 적분하여 blur를 explicit하게 모델링). 다만 위 방법들은 rolling shutter를 고려하지 않는다. RS는 NeRF 맥락에서 별도로 다뤄졌으나 역시 추가 learnable parameter가 필요하다.

본 논문의 차별점은 MLP를 추가하지 않고 linear·angular velocity로 local camera trajectory를 직접 모델링하며, 그 초기값을 VIO(IMU)로부터 얻는다는 점이다.

## Method

### Gaussian Splatting (Preliminary)

3DGS는 pixel color를 model parameter에 대해 미분 가능하게 렌더링한다. Output image $i$의 pixel $(x,y)$ color는 다음과 같고,

$$
\begin{equation}
C_i(x, y, P_i, \mathcal{G}) \in \mathcal{C}
\end{equation}
$$

이를 $N_{\text{img}}$개의 reference image $C_i'$와 비교하는 loss를 최소화한다.

$$
\begin{equation}
\mathcal{G} \mapsto \sum_{i=1}^{N_{\text{img}}} \mathcal{L}\left[C_i(\cdot, \cdot, P_i, \mathcal{G}), C_i'\right]
\end{equation}
$$

여기서 $P_i \in SE(3)$는 image $i$의 camera pose, model parameter는 Gaussian 집합 $\mathcal{G} = \{(\mu_j, \Sigma_j, \alpha_j, \theta_j)\}_j$ (mean, covariance, transparency, SH 계수 color)다.

### Blur and Rolling Shutter as Camera Motion

Motion blur와 rolling shutter를 모두 연속 trajectory $t \mapsto P(t)$를 따르는 camera motion으로 인한 동적 3D 효과로 통합 모델링한다. Rendering equation을 다음과 같이 바꾼다.

$$
\begin{equation}
C_i(x, y, \mathcal{G}) = g\left( \frac{1}{T_e} \int_{-\frac{1}{2}T_e}^{\frac{1}{2}T_e} C_i\left(x, y, P\left(t_i + t_e + (y/H - 1/2)T_{ro}\right), \mathcal{G}\right) dt_e \right)
\end{equation}
$$

여기서 $T_e$는 exposure time, $T_{ro}$는 rolling-shutter readout time, $t_i$는 frame 중심 timestamp, $g$는 gamma correction($\gamma=2.2$)이다. 핵심은 $(y/H - 1/2)T_{ro}$ 항으로, 이것이 pixel 행 위치 $y$에 따라 노출 시각을 다르게 만들어 rolling shutter를 표현한다(BAD-NeRF의 blur 적분식에서 한 걸음 더 나아간 부분).

Frame 중심 시각 $t_i$ 주변의 camera motion은 velocity로 파라미터화된다.

$$
\begin{equation}
P(t_i + \Delta t) = \left[ R_i \exp(\Delta t[\omega_i]_\times) \mid p_i + \Delta t \cdot R_i v_i \right]
\end{equation}
$$

$(v_i, \omega_i)$는 camera local 좌표계의 linear·angular velocity로, frame 구간 내에서 상수로 가정한다. 이 velocity를 optimizable parameter로 추가하고, 초기값을 VIO 추정값(있으면)으로 설정한다.

### Screen Space Approximation

렌더링 모델을 두 단계로 분해한다. (1) Gaussian parameter를 world 좌표에서 pixel 좌표로 변환하는 $p$, (2) pixel 좌표 parameter를 이미지에 rasterize하는 $r$. 핵심은 rasterization 단계 $r$이 camera pose $P_i$에 의존하지 않는다는 점이다.

![Fig3](/assets/img/gsmove/fig3_screenspace.png)
_Fig 3. Screen space approximation. Camera 움직임의 효과를 world 공간이 아니라 pixel 좌표(image plane)에서 근사한다. 이후 rasterization은 더 이상 camera pose에 의존하지 않는다._

노출 구간 동안의 camera motion을 pixel 좌표에서 다음과 같이 근사한다.

$$
\begin{equation}
\mu'_{i,j}(\Delta t) \approx \tilde{\mu}_{i,j}(\Delta t) := \mu'_{i,j}(0,0) + \Delta t \cdot v'_{i,j}
\end{equation}
$$

즉 (작은) camera motion이 covariance·depth·color 등 다른 view-dependent 변수에 주는 영향은 무시하고, Gaussian mean의 pixel 좌표 $\mu'_{i,j}$만 이동시킨다. 이때 새로 도입하는 pixel velocity는 다음과 같다.

$$
\begin{equation}
v'_{i,j} = -J_i(\omega_i \times \hat{\mu}_{i,j} + v_i)
\end{equation}
$$

$J_i = \text{diag}(f_x, f_y)/d_{i,j}$는 projective transform의 Jacobian이다.

### Rasterization with Pixel Velocities

식 (3)의 적분을 노출 구간의 $N_{\text{blur}}$개 uniform sample 합으로 근사한다.

$$
\begin{equation}
\tilde{C}_i(x,y,\mathcal{G}) := g\left( \frac{1}{N_{\text{blur}}} \sum_{k=1}^{N_{\text{blur}}} \sum_j T_j c_{i,j} \alpha_{i,j}(\tilde{\mu}'_{i,j}(\Delta t_k(y))) \right)
\end{equation}
$$

$$
\begin{equation}
\Delta t_k(y) := \left( \frac{k-1}{N_{\text{blur}}-1} - \frac{1}{2} \right) \cdot T_e + \left( \frac{y}{H} - \frac{1}{2} \right) \cdot T_{ro}
\end{equation}
$$

이 근사의 핵심 이득은, pixel-space에서 Gaussian mean에만 velocity를 적용하므로 blur sample마다 3DGS의 projective transform을 반복할 필요가 없다는 점이다. 따라서 world 공간에서 Gaussian velocity를 다루는 것보다 빠르다.

### Pose Optimization

Camera pose 성분에 대한 gradient도 Gaussian position derivative $\partial C_i / \partial \mu_j$로 근사한다(3DGS pipeline에서 이미 사용 가능).

$$
\begin{equation}
\frac{\partial C_i}{\partial p_i} \approx -\sum_j \frac{\partial C_i}{\partial \mu_j}, \qquad \frac{\partial C_i}{\partial \nu} \approx -\sum_j \frac{\partial C_i}{\partial \mu_j} \frac{\partial R_i}{\partial \nu} R_i^\top (\mu_j - p_i)
\end{equation}
$$

Reconstruction 안정화를 위해 pose가 초기 추정값에서 너무 벗어나지 않도록 간단한 $l_2$ penalty를 준다. 또한 gauge invariance(좌표계 회전·스케일·평행이동의 7 자유도 불확정성) 때문에 evaluation frame이 어긋날 수 있어, Gaussian은 고정한 채 evaluation frame의 pose·velocity만 최적화하는 evaluation 전략을 쓴다.

## Experiments

### Settings
- Nerfstudio의 Splatfacto(gsplat 기반)를 확장 구현. Baseline도 Splatfacto.
- Dataset: (1) BAD-NeRF가 re-render한 synthetic Deblur-NeRF 데이터셋, (2) 저자들이 재렌더한 motion blur / rolling shutter / pose noise 변형, (3) 스마트폰 실제 데이터(Samsung S20 FE, Google Pixel 5, iPhone 15 Pro) 11개 handheld 촬영.
- 실제 데이터는 Spectacular AI SDK로 VIO velocity $(v^{\text{VIO}}, \omega^{\text{VIO}})$와 VISLAM pose를 얻어 초기값으로 사용.
- $N_{\text{blur}} = 5$. Metric: PSNR / SSIM / LPIPS.

### Synthetic Results

![Table2](/assets/img/gsmove/table2_synthetic.png)
_Table 2. BAD-NeRF 변형 Deblur-NeRF 데이터셋에서의 novel view synthesis 정량 비교. Splatfacto·MPR·Deblur-NeRF·BAD-NeRF를 모두 능가한다._

BAD-NeRF 변형 Deblur-NeRF 데이터셋(Table 2)에서, 본 방법은 Splatfacto·MPR+Splatfacto·Deblur-NeRF·BAD-NeRF를 모두 능가한다. 예를 들어 Trolley 장면에서 PSNR 30.16(BAD-NeRF 28.25), Tanabata SSIM 0.912(BAD-NeRF 0.831)로, 특히 SSIM·LPIPS에서 큰 차이를 보인다.

![Table1](/assets/img/gsmove/table1_effects.png)
_Table 1. 재렌더한 각 효과(motion blur / rolling shutter / pose noise)별로 보정을 켠 Ours와 Splatfacto baseline의 정량 비교._

재렌더 변형 실험(Table 1)에서는 각 효과(motion blur / RS / pose noise)에 맞는 보정을 켜고 Splatfacto baseline과 비교한다. 모든 시나리오에서 baseline을 크게 앞선다. 예: rolling shutter cozyroom에서 PSNR 19.21 → 35.84, factory에서 15.29 → 35.27. Rolling shutter 왜곡이 심할수록 보정 효과가 극적이다.

![Fig2](/assets/img/gsmove/fig2_factory.png)
_Fig 2. Synthetic factory 장면에서의 3DGS 복원 비교. rolling shutter 등 효과에 대해 baseline 대비 Ours가 왜곡을 크게 줄인다._

![Fig4](/assets/img/gsmove/fig4_realdata.png)
_Fig 4. 스마트폰 실제 데이터 예시. 위에서부터 COLMAP pose로 motion blur 보정 없음(Baseline), 보정 적용(Ours), reference, $l_2$ error 기여도, ours와 baseline의 $l_2$ error 차이(빨강: error 증가, 파랑: 감소). 장면: lego1(iPhone), pots2(iPhone), table(Pixel), bike(S20)._

### Real Data & Ablation

![Table3](/assets/img/gsmove/table3_ablation.png)
_Table 3. 스마트폰 실데이터에서의 PSNR ablation. motion blur·rolling shutter·pose·velocity·VIO 초기화 등 구성 요소별 기여를 비교한다._

실제 스마트폰 데이터(Table 3)에서 ablation한 결과, motion blur 보정 / rolling shutter 보정 / pose 최적화 / velocity 최적화 / VIO velocity 초기화 모든 구성 요소가 PSNR에 긍정적으로 기여하며, 전체(Ours)가 최고 성능을 보인다.

CVR de-rolling 전처리와 비교하면 결과가 mixed다. Pixel 5에서는 CVR이 낫지만, S20에서는 나쁘고, RS readout이 작은 iPhone에서는 CVR이 확실히 더 나쁘다. CVR 출력에 artifact가 섞여 3DGS 성능을 해치기 때문으로, 정적 장면 RS 보정에서는 본 방법이 더 robust하다.

![Fig5](/assets/img/gsmove/fig5_mbrs.png)
_Fig 5. 실제 스마트폰 데이터(rolling shutter 효과 있음)에서의 ablation. 위에서부터 Baseline / Ours(MB, motion blur만) / Ours(MBRS, motion blur + rolling shutter) / Reference. MBRS가 특히 이미지 가장자리에서 더 선명하고 artifact 없는 reconstruction을 만든다._

### Timing (Table 4)
- Full method($N_{\text{blur}}=5$)는 baseline 대비 평균 약 5배 느림(NVIDIA A100 기준).
- Rolling shutter 보정만의 overhead는 낮음(약 26%). 대부분의 추가 비용은 motion blur 보정에서 온다.
- Memory 사용량은 baseline과 크게 다르지 않고, RTX 4060 Ti(16GB) 같은 consumer GPU에서도 학습 가능.

## Conclusion

이 논문은 motion blur와 rolling shutter를 3DGS 프레임워크에 효율적으로 통합하여, synthetic·real 데이터 모두에서 Splatfacto baseline을 능가하고 NeRF SOTA인 BAD-NeRF도 넘어섰다. 2D 입력을 전처리로 deblur/de-roll하는 방법(MPR, CVR)보다 3D 모델 생성에 직접 통합하는 편이 robust함을 보였다. Follow-up으로 local linear trajectory를 spline 기반의 더 복잡한 형태로 확장하는 것을 제시한다.
