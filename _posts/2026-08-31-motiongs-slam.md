---
title: "[논문리뷰] MotionGS-SLAM: Event-Modulated Gaussian Splatting for Motion-Blur Robust SLAM"
date: 2026-08-31 15:30:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, SLAM, Motion Blur, Event Camera, Adaptive Sampling, ICRA]
description: "ICRA 2026"
math: true
toc: true
compact_captions: true
---

> \[[Paper](https://arxiv.org/abs/2608.15024)\]
>
> Zhiqiang Hu, Shouren Huang, Masatoshi Ishikawa

## Introduction

3D Gaussian Splatting을 사용하는 SLAM은 camera pose와 photorealistic map을 함께 추정할 수 있지만, 입력 영상이 sharp하다는 가정에 크게 의존한다. 빠르게 움직이는 camera나 긴 exposure가 필요한 low-light 환경에서는 이 가정이 무너진다. Motion blur가 edge와 texture를 지우면서 tracking이 불안정해지고, 잘못된 pose와 관측 영상이 Gaussian map에 반영되어 blur와 ghosting이 남는다.

MotionGS-SLAM은 blur를 먼저 제거하는 대신, 노출 중 camera motion이 어떻게 blurry image를 만드는지 rendering 과정에서 모델링한다. 추정한 3D scene과 intra-exposure camera trajectory로 blurry image를 합성하고, 실제 관측과 일치하도록 최적화하는 방식이다. 이때 event camera가 제공하는 고시간해상도 brightness change를 trajectory 추정뿐 아니라 Gaussian kernel과 temporal sampling을 조절하는 데 사용한다.

![MotionGS-SLAM의 blurry input과 reconstructed frame](/assets/img/motiongs-slam/fig1_teaser.png)
_Fig 1. 위쪽은 motion blur가 포함된 입력 영상, 아래쪽은 MotionGS-SLAM으로 복원한 scene의 rendering이다._

핵심 구성은 다음과 같다.

1. Exposure 시작과 끝의 두 pose로 camera trajectory를 표현하고, intermediate rendering을 누적해 blur formation을 근사한다.
2. Event–Gaussian association으로 얻은 local motion statistics를 사용해 projected Gaussian의 spatial shape와 exposure integration의 temporal sample 수를 조절한다.
3. Tracking에서는 camera trajectory를, mapping에서는 keyframe trajectory와 Gaussian map을 함께 최적화한다. 입력은 blurry RGB frame과 event stream이며 depth sensor를 요구하지 않는다.

## Related Work

### Gaussian Splatting based SLAM

GS-SLAM, SplaTAM, MonoGS 등의 연구는 explicit하고 differentiable한 Gaussian representation을 tracking과 mapping에 도입했다. 논문은 Sgs-slam, Large spatial model, Compact 3d Gaussian splatting도 함께 언급하며, 3DGS 기반의 dense reconstruction과 scene representation 연구를 정리한다.

그러나 sharp하고 적절히 노출된 영상에 의존하는 시스템은 motion blur에 취약하다. I²-SLAM처럼 camera imaging process를 모델링하는 접근도 있지만, 저자들은 image-only 관측만으로는 긴 exposure 동안 손실된 motion 정보를 충분히 구분하기 어렵다고 설명한다. MotionGS-SLAM은 별도의 고시간해상도 modality인 event를 rendering model에 연결한다.

### Event-based 3D Gaussian Splatting/NeRF

E2NeRF와 Ev-DeblurNeRF는 blurry image와 event를 함께 사용해 NeRF reconstruction을 수행한다. Implicit event-RGBD neural SLAM은 event와 RGB-D 관측을 결합해 SLAM에서 event 활용 가능성을 보였다. EvaGaussians와 Event3DGS 계열 역시 event를 이용한 3D Gaussian reconstruction을 연구했다.

논문은 이러한 reconstruction 중심 접근과 달리, event-guided blur rendering을 online tracking front-end에 직접 넣는다는 점을 강조한다. 이미 주어진 camera pose를 사용해 scene만 복원하는 대신, blur가 포함된 관측으로부터 camera trajectory와 map을 함께 추정하는 것이 목표다.

## Method

### Preliminaries

Scene은 3D Gaussian의 집합 $\mathcal G$로 표현한다. 각 Gaussian은 mean, covariance, color, opacity를 가지며, camera pose가 주어지면 2D image plane으로 projection한 뒤 $\alpha$-blending으로 pixel color를 계산한다.

일반적인 rendering은 하나의 pose에서 순간적으로 관측한 sharp image에 대응한다. 반면 blurry frame은 exposure interval 동안 여러 pose에서 들어온 빛이 누적된 결과다. MotionGS-SLAM은 이 차이를 반영하기 위해 intra-exposure trajectory와 event-modulated kernel을 함께 사용한다.

![MotionGS-SLAM의 event association과 tracking 및 mapping pipeline](/assets/img/motiongs-slam/fig2_pipeline.png)
_Fig 2. Event–Gaussian association으로 motion statistics를 계산하고, spatial modulation과 temporal modulation을 적용해 blurry image를 합성한다. Tracking은 trajectory를 추정하고, mapping은 keyframe window의 pose와 Gaussian map을 함께 갱신한다._

### Motion Blur Image Formation Model

Frame $m$의 exposure duration을 $\Delta t_m$이라 하자. Exposure 동안 camera는 $$[t_{st},t_{ed}]$$ 구간의 trajectory를 따라 움직인다. 논문은 이 trajectory에서 렌더링한 $N$개의 latent sharp image를 평균하여 관측 blur를 근사한다.

$$
\begin{equation}
\bar{\mathbf C}_m(\mathbf u)
\approx \frac{1}{N}\sum_{k=1}^{N}
\mathbf C\bigl(\mathbf T(t_k),\mathbf u\bigr).
\tag{1}
\end{equation}
$$

여기서 $$\mathbf C(\mathbf T(t_k),\mathbf u)$$는 intermediate pose에서 렌더링한 pixel color다. 이 모델은 exposure integration을 finite temporal samples로 근사하며, 뒤에서 event statistics에 따라 sample 수를 결정한다.

#### Camera Motion Trajectory Modeling

Exposure 시작과 끝의 pose $$\mathbf T_{st},\mathbf T_{ed}\in\mathrm{SE}(3)$$를 최적화 변수로 둔다. 정규화된 시간 $$t\in[0,1]$$에서 pose는 다음과 같이 interpolation한다.

$$
\begin{equation}
\mathbf T(t)=\mathbf T_{st}\cdot
\exp\!\left(t\cdot\log\!\left(\mathbf T_{st}^{-1}\cdot\mathbf T_{ed}\right)\right).
\tag{2}
\end{equation}
$$

여기서 $\exp$와 $\log$는 SE(3)의 exponential map과 logarithmic map이다. 각 intermediate pose를 독립적으로 최적화하는 것이 아니라, 두 endpoint를 통해 exposure 내부의 continuous path를 정의한다. 식 (2)의 시간은 정규화된 값이며, 실제 exposure timestamp를 사용하는 뒤의 식 (10)과 구분해야 한다.

### Motion Blur Aware Tracker

#### Real-Time Event-Gaussian Association

각 visible Gaussian에 관련된 event를 모으기 위해 multi-resolution 4D hash grid를 사용한다. Event는 $e=(x,y,t,p)$로 표현하며, image coordinate, timestamp, polarity로 구성된다.

먼저 Gaussian을 image plane에 projection한다. Projected mean을 중심으로 한 $3\sigma$ axis-aligned bounding box와 exposure time window를 사용해 event 후보를 조회하고, 다음 Mahalanobis distance 조건으로 후보를 걸러낸다.

$$
\begin{equation}
(\mathbf p-\hat{\boldsymbol\mu}_j)^\top
\hat{\boldsymbol\Sigma}_j^{-1}
(\mathbf p-\hat{\boldsymbol\mu}_j)\leq\tau.
\tag{3}
\end{equation}
$$

식 (3)의 $\mathbf p$는 event의 2D 위치이며, event tuple의 polarity $p$와는 다른 표기다. 논문은 threshold의 예로 $\tau\approx9$를 사용한다. 저자들은 association의 연산량을 brute-force의 $$O(N_{\mathrm{gauss}}N_{\mathrm{event}})$$에서 hash query 기반의 $$O(N_{\mathrm{gauss}}K)$$로 줄인다고 설명한다. $K$는 Gaussian 하나당 조회되는 평균 event 수다.

이렇게 얻은 event 집합 $\mathcal E_j$에서 polarity-weighted displacement를 계산한다.

$$
\begin{equation}
\begin{aligned}
\mathbf v_j&=
\frac{\sum_{e_i\in\mathcal E_j}p_i(\mathbf x_i-\hat{\boldsymbol\mu}_j)}
{\sum_{e_i\in\mathcal E_j}|p_i|+\varepsilon},\\
\nu_j&=\|\mathbf v_j\|,\qquad
\phi_j=\operatorname{atan2}(v_y,v_x).
\end{aligned}
\tag{4}
\end{equation}
$$

Event polarity는 $$p_i\in\{-1,+1\}$$이고 $\varepsilon$는 division by zero를 막는 작은 상수다. Gaussian 중심에 대한 event 위치의 polarity-weighted 평균을 구하고, 그 크기와 방향을 local motion statistics로 사용한다. 논문은 $\nu_j$를 estimated motion speed로 부르며, 이 값이 이후 spatial stretch와 temporal sample 수를 제어한다.

#### Event-Modulated Gaussian Kernel

Spatial modulation은 image-plane Gaussian shape를, temporal modulation은 exposure integration에 사용하는 sample 수를 조절한다.

Spatial Kernel Modulation에서는 event로부터 다음 prior covariance를 구성한다.

$$
\begin{equation}
\begin{aligned}
\boldsymbol\Sigma^{2D}_{j,\mathrm{prior}}
&=\mathbf R(\phi_j)\operatorname{diag}
\left(\sigma_\perp^2,\kappa_j^2\sigma_\perp^2\right)
\mathbf R(\phi_j)^\top,\\
\kappa_j&=1+\beta\nu_j.
\end{aligned}
\tag{5}
\end{equation}
$$

$\mathbf R$은 2D rotation matrix이며, $$\sigma_\perp^2$$는 base variance다. 저자들이 의도하는 prior는 event에서 추정한 방향을 반영하고, motion magnitude가 커질수록 elongation이 증가하는 anisotropic Gaussian이다. Motion magnitude가 0이면 stretch factor가 1이 되어 isotropic prior로 돌아간다.

Projected covariance가 이 prior에 가까워지도록 shape loss를 사용한다.

$$
\begin{equation}
\mathcal L_{\mathrm{shape}}=
\frac{1}{|\mathcal G_{\mathrm{vis}}|}
\sum_{j\in\mathcal G_{\mathrm{vis}}}
\left\|\hat{\boldsymbol\Sigma}^{2D}_j-
\boldsymbol\Sigma^{2D}_{j,\mathrm{prior}}\right\|_F^2.
\tag{6}
\end{equation}
$$

$\mathcal G_{\mathrm{vis}}$는 현재 frame에서 보이는 Gaussian 집합이다. 식 (6)은 최적화 중인 3D Gaussian의 projected covariance와 event-derived prior의 Frobenius distance를 줄인다.

3D geometry와 blur에 따른 deformation을 나누기 위해 covariance는 다음 두 성분으로 표현한다.

$$
\begin{equation}
\begin{aligned}
\boldsymbol\Sigma_j&=\boldsymbol\Sigma_{\mathrm{base},j}
+\Delta\boldsymbol\Sigma_j,\\
\boldsymbol\Sigma_{\mathrm{base},j}&=
\mathbf R(q_j)\operatorname{diag}
\left(e^{\ell_{x,j}},e^{\ell_{y,j}},e^{\ell_{z,j}}\right)
\mathbf R(q_j)^\top.
\end{aligned}
\tag{7}
\end{equation}
$$

Base covariance는 exponential parameterization으로 positive-definite하게 구성하며 static geometry를 담당한다. Motion-induced term은 activation function으로 bounded하게 두고, shape prior와 photometric consistency를 통해 간접적으로 학습한다. 이는 underlying geometry와 blur effect를 분리하려는 설계다.

Temporal Kernel Modulation에서는 visible Gaussian들의 motion magnitude를 평균한다.

$$
\begin{equation}
\bar\nu_m=\frac{1}{|\mathcal G_{\mathrm{vis}}|}
\sum_{j\in\mathcal G_{\mathrm{vis}}}\nu_j.
\tag{8}
\end{equation}
$$

Frame별 temporal sample 수는 평균 motion과 exposure duration으로 결정한다.

$$
\begin{equation}
N_m=\operatorname{clamp}\!\left(
\operatorname{round}\!\left(n_0(1+\alpha\bar\nu_m\Delta t_m)\right),
N_{\min},N_{\max}\right).
\tag{9}
\end{equation}
$$

$n_0$는 base sample 수, $\alpha$는 motion에 대한 반응 정도를 정하는 계수다. 빠른 motion이나 긴 exposure에서는 sample 수를 늘리고, 결과를 상·하한 사이로 제한한다. Sample timestamp는 다음과 같이 균일하게 배치한다.

$$
\begin{equation}
t_k=t_{st}+k\frac{\Delta t_m}{N_m-1},
\qquad k\in\{0,1,\ldots,N_m-1\}.
\tag{10}
\end{equation}
$$

따라서 식 (8)–(10)의 adaptive sampling은 frame마다 sample 수를 바꾸는 방식이다. Gaussian별 local statistics를 집계하지만, 정해진 exposure interval 안에서는 uniform sampling을 사용한다. 본문의 photometric loss 설명에는 per-Gaussian adaptive samples라는 표현도 등장하지만, 여기서 명시적으로 정의한 sampling rule은 frame-level $N_m$이다.

#### Blur Aware Tracking Optimization

Tracking front-end는 map을 고정하고, 새 blurry frame의 exposure endpoint pose를 최적화한다.

$$
\begin{equation}
\min_{\mathbf T_{st},\mathbf T_{ed}}
\mathcal L_{\mathrm{photo}}+\lambda_{\mathrm{evt}}\mathcal L_{\mathrm{LI}}.
\tag{11}
\end{equation}
$$

Photometric loss는 exposure-integrated rendering과 실제 blurry observation 사이의 robust error다.

$$
\begin{equation}
\mathcal L_{\mathrm{photo}}=
\|\bar{\mathbf C}_m-\mathbf I_{\mathrm{obs}}\|_\rho.
\tag{12}
\end{equation}
$$

논문은 robust penalty $\rho$의 예로 Charbonnier loss를 든다. 이 loss는 sharp image와 blurry image를 직접 맞추는 것이 아니라, 추정한 scene과 trajectory가 합성하는 blur를 실제 관측과 맞춘다.

Event Alignment Loss는 rendered log-intensity change와 observed event polarity가 일치하도록 한다.

$$
\begin{equation}
\mathcal L_{\mathrm{LI}}=
\sum_{e_i\in\mathcal E}
\rho\!\left(p_i-
\frac{\log I(t_i)-\log I(t_i-\delta)}{\theta}\right).
\tag{13}
\end{equation}
$$

$I(t)$는 event timestamp에서의 rendered intensity이고, $\delta$는 작은 temporal offset, $\theta$는 event camera의 contrast threshold다. Exposure 전체를 평균한 frame과 달리, 이 항은 event timestamp 주변의 intensity difference를 비교한다. 이를 통해 camera trajectory가 event의 시간 정보와도 일치하도록 제약한다.

### Mapping with Keyframe Management

#### Keyframe Selection

새 keyframe은 다음 중 하나를 만족하면 삽입한다.

- 최신 keyframe과의 visible Gaussian overlap을 IoU로 측정했을 때 threshold보다 작아지는 경우.
- Camera translation이 average scene depth에 비례하는 threshold를 넘는 경우.

Window의 keyframe 수가 한도에 도달하면 neighbor overlap이 가장 큰 keyframe을 제거한다. 비슷한 view의 중복을 줄이면서 계산량을 제한하는 방식이다.

#### Bundle Adjustment

Mapping back-end는 sliding window $\mathcal W$에서 Gaussian map과 keyframe pose를 함께 최적화한다.

$$
\begin{equation}
\min_{\mathcal G,\{\mathbf T\}}
\sum_{w\in\mathcal W}\left[
\mathcal L_{\mathrm{photo}}^w+
\lambda_{\mathrm{evt}}\mathcal L_{\mathrm{LI}}^w\right]
+\lambda_{\mathrm{shape}}\mathcal L_{\mathrm{shape}}.
\tag{14}
\end{equation}
$$

Photometric loss와 event loss는 각 keyframe의 관측을 설명하도록 map과 trajectory를 맞춘다. Shape loss는 projected Gaussian이 event-derived motion prior를 따르도록 한다. Tracking이 현재 frame의 motion 추정에 집중한다면, mapping은 여러 view에 걸친 data consistency와 Gaussian shape를 함께 조정한다.

## Experiments

### Datasets and Metrics

EventReplica는 Replica scene을 기반으로 만든 synthetic benchmark다. Camera trajectory의 intermediate frame을 렌더링해 평균함으로써 long-exposure blur를 만들고, Poisson-Gaussian noise를 추가한다. ESIM으로 monochrome event를 생성하며 balanced contrast threshold는 $\Theta=0.2$다. Sharp ground-truth image를 이용해 reconstruction quality를 평가한다.

Real-world dataset은 Color-DAVIS346으로 촬영한 3개 scene으로 구성된다. Low-light handheld 촬영에서 $346\times260$ resolution의 color frame과 시간적으로 정렬된 event stream을 수집한다. Scene마다 여러 viewpoint와 다양한 blur 정도를 포함하는 30초 분량의 데이터를 취득한다.

Trajectory는 Absolute Trajectory Error(ATE, cm)로 평가한다. Synthetic reconstruction quality는 PSNR, SSIM, LPIPS로 측정한다.

### Implementation Details

구현은 MonoGS codebase를 기반으로 한 PyTorch 시스템이며, 단일 NVIDIA RTX 4080 GPU를 사용한다. Base temporal sample 수는 $n_0=9$, spatial stretch 제어 계수는 $\beta=0.05$다. Loss weight는 $$\lambda_{\mathrm{evt}}=2.0$$, $$\lambda_{\mathrm{photo}}=1.0$$, $$\lambda_{\mathrm{shape}}=0.2$$로 설정한다.

### Comparison with State-of-the-Art

#### Evaluation on Blurry-Replica with Event

Synthetic benchmark에서는 PhotoSLAM, MonoGS, EGS-SLAM과 비교한다. Table I에 보고된 MotionGS-SLAM의 평균 PSNR은 28.05 dB, SSIM은 0.842, LPIPS는 0.171이다. EGS-SLAM의 평균값은 각각 27.62 dB, 0.833, 0.182이며, image-only MonoGS는 23.12 dB, 0.728, 0.391이다.

![EventReplica rendering quality와 rendering FPS 비교](/assets/img/motiongs-slam/table1_rendering.png)
_Table I. EventReplica의 rendering quality와 rendering FPS. PSNR·SSIM은 높을수록, LPIPS는 낮을수록 좋다. ‘–’는 reinitialization이 성공하지 못한 sequence를 뜻한다._

Rendering FPS는 MotionGS-SLAM 1189.7, EGS-SLAM 1168.3, MonoGS 1102.1로 보고된다. 표의 수치는 rendering throughput이며, tracking과 mapping 전체를 포함한 end-to-end SLAM 처리 속도로 해석해서는 안 된다.

Tracking 결과에서 논문이 보고한 평균 ATE는 MotionGS-SLAM 4.44 cm, EGS-SLAM 5.17 cm, MonoGS 9.22 cm다. 개별 sequence에서도 MotionGS-SLAM이 가장 낮은 ATE를 기록한다. ORB-SLAM2와 PhotoSLAM은 여러 sequence에서 tracking failure를 보인다.

![EventReplica의 trajectory ATE 비교](/assets/img/motiongs-slam/table3_synthetic_tracking.png)
_Table III. EventReplica의 ATE(cm). L은 tracking failure를 나타낸다. 값이 낮을수록 좋다._

#### Evaluation on Real-World Data

Real-world dataset에서는 MotionGS-SLAM의 평균 ATE가 3.55 cm이며, EGS-SLAM은 4.83 cm, PhotoSLAM은 16.58 cm, ORB-SLAM3는 28.46 cm다. 같은 표의 단계별 ablation에서는 MonoGS baseline 6.65 cm에서 event를 추가하면 4.69 cm, spatial modulation까지 추가하면 4.12 cm, full model에서는 3.55 cm로 감소한다.

![Real-world dataset의 tracking 비교와 단계별 ablation](/assets/img/motiongs-slam/table2_real_tracking.png)
_Table II. Real-world 3개 scene의 ATE(cm)와 단계별 ablation. 위쪽은 기존 방법과의 비교, 아래쪽은 event 및 modulation 구성요소를 추가한 결과다._

#### Qualitative Results

Synthetic comparison에서는 image-only 방법에 남는 blur와 ghosting이 두드러진다. EGS-SLAM은 event를 사용해 개선되지만, 확대 영역의 texture와 object boundary에는 차이가 남는다. MotionGS-SLAM의 결과에서는 television 주변과 furniture edge 등의 detail이 더 선명하다.

![Synthetic scene reconstruction 정성 비교](/assets/img/motiongs-slam/fig3_synthetic_comparison.png)
_Fig 3. Severe motion blur 조건의 synthetic reconstruction 비교. 각 행은 Photo-SLAM, Mono-GS, EGS-SLAM, MotionGS-SLAM 순서이며, inset은 texture와 boundary detail을 보여준다._

Real-world comparison에서도 같은 경향을 제시한다. 저자들은 문자와 물체 윤곽에서 남는 blur를 비교하며, event-guided rendering으로 더 선명한 scene을 복원한다고 설명한다.

![Real-world scene reconstruction 정성 비교](/assets/img/motiongs-slam/fig4_real_comparison.png)
_Fig 4. Low-light handheld real-world sequence 비교. 왼쪽부터 input, Photo-SLAM, Mono-GS, EGS-SLAM, MotionGS-SLAM이다._

### Ablation Study

EventReplica의 room0와 office0에서 구성요소를 분리한다. Table IV에는 A0부터 A5까지 여섯 configuration이 제시된다.

- A0, Baseline: blurry image만 사용하는 MonoGS.
- A1, Blur Model Only: exposure integration renderer를 추가하되 event는 사용하지 않는다.
- A2, Event Association: event association과 event loss를 추가하지만 kernel modulation은 사용하지 않는다.
- A3, Spatial Modulation Only: A2에 spatial modulation을 추가하고 temporal sampling은 고정한다.
- A4, Temporal Modulation Only: A2에 adaptive temporal sampling을 추가하고 spatial kernel은 그대로 둔다.
- A5, Full: spatial modulation과 temporal modulation을 모두 사용한다.

![EventReplica room0와 office0의 구성요소별 ablation](/assets/img/motiongs-slam/table4_component_ablation.png)
_Table IV. Room0와 office0의 ablation. ATE·LPIPS는 낮을수록, PSNR·SSIM은 높을수록 좋다. Full model이 두 scene의 모든 평가 지표에서 가장 좋은 결과를 보인다._

#### The Necessity of Event Data

Room0에서 exposure integration만 추가한 A1은 ATE를 baseline의 12.76 cm에서 11.80 cm로 줄인다. Event association과 event loss를 추가한 A2에서는 6.10 cm로 감소한다. 이 비교는 blur model만으로 부족했던 trajectory 정보를 event가 보완한다는 주장을 뒷받침한다.

그러나 A2의 LPIPS는 0.340으로 full model의 0.214와 차이가 남는다. 논문은 event loss로 camera motion을 제약하는 것만으로는 blur를 rendering하는 방식까지 충분히 개선하지 못한다고 설명한다.

#### Dissecting the Dual-Modulation Kernel

Room0에서 spatial modulation을 사용하는 A3의 LPIPS는 0.250이며, A2의 0.340보다 낮다. Temporal modulation만 사용하는 A4는 ATE 5.10 cm, PSNR 23.10 dB를 기록한다. 두 modulation을 결합한 A5는 ATE 3.98 cm, PSNR 24.55 dB, SSIM 0.756, LPIPS 0.214로 두 단독 구성보다 좋은 결과를 보인다.

Office0에서도 full model은 ATE 3.01 cm, PSNR 32.31 dB를 기록한다. 저자들은 spatial blur shape와 temporal accumulation을 함께 모델링하는 것이 tracking과 reconstruction 모두에 유효하다고 해석한다.

![Event association 및 spatial modulation의 단계별 정성 효과](/assets/img/motiongs-slam/fig5_component_ablation.png)
_Fig 5. Input과 A2, A3, A5의 정성 비교. Spatial modulation을 추가하고 full model로 확장하면서 확대 영역의 texture와 edge가 더 선명해진다._

## Conclusion

MotionGS-SLAM은 blurry RGB frame과 event stream으로부터 camera trajectory와 3D Gaussian map을 함께 추정한다. Exposure endpoint interpolation과 temporal rendering average로 blur formation을 표현하고, event-derived motion statistics를 spatial covariance prior와 adaptive sample count에 연결한다.

핵심은 event를 별도의 loss에만 사용하는 것이 아니라 rendering 과정 자체를 조절하는 데 활용한다는 점이다. Synthetic 및 real-world 실험에서는 event association, spatial modulation, temporal modulation을 결합했을 때 tracking accuracy와 reconstruction quality가 함께 개선됨을 보여준다.
