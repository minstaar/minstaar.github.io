---
title: "[논문리뷰] EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering"
date: 2026-08-31 16:08:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Event Camera, Novel View Synthesis, Event Accumulation, Trajectory Interpolation, CVPR]
description: "CVPR 2025"
math: true
toc: true
compact_captions: true
---

> \[[Paper](https://arxiv.org/abs/2412.07293)\]
>
> Toshiya Yura, Ashkan Mirzaei, Igor Gilitschenski

## Introduction

Event camera는 일정한 frame rate로 absolute intensity를 기록하는 대신, 각 pixel의 log-intensity가 threshold 이상 변할 때 비동기적으로 event를 출력한다. 높은 temporal resolution과 dynamic range 덕분에 빠른 camera motion이나 어려운 조명 조건에서도 정보를 얻을 수 있지만, 이 출력을 RGB image처럼 바로 3D Gaussian Splatting의 supervision으로 사용할 수는 없다.

EventSplat은 움직이는 event camera의 관측으로 static scene을 3D Gaussians로 복원하고, 학습 후에는 rasterization으로 novel view를 실시간 렌더링하는 방법이다. Event-based NeRF의 높은 rendering 비용을 줄이면서, event가 측정하는 brightness change를 3DGS의 학습 신호로 연결하는 데 초점을 둔다.

![Event stream으로부터 복원한 synthetic 및 real scene의 novel view](/assets/img/eventsplat/fig1_teaser.png)
_Fig 1. EventSplat은 event sequence로부터 Gaussian scene representation을 학습하고, synthetic color scene과 real grayscale scene의 novel view를 렌더링한다._

논문의 기여는 세 부분으로 구성된다.

1. Event accumulation으로 만든 변화량 이미지와 두 rendering의 log-intensity difference를 비교하여 3DGS를 학습한다.
2. Event-to-video model이 생성한 영상에 SfM을 적용해 Gaussian의 초기 point cloud를 얻는다.
3. Cubic spline trajectory interpolation으로 비동기 event 구간의 endpoint에 대응하는 camera pose를 구한다.

여기서 real-time은 학습된 Gaussian scene의 rendering을 의미한다. 논문은 static scene별 reconstruction을 수행하며, dynamic object reconstruction이나 online SLAM을 제안하는 것은 아니다.

## Related Work

### Event Cameras for Vision Applications

Event camera의 낮은 latency, 높은 dynamic range와 temporal resolution은 tracking, detection, recognition, optical flow 등의 연구를 이끌었다. 그러나 비동기 event stream은 일반적인 frame과 구조가 다르므로, task에 맞는 representation과 처리 방법이 필요하다. EventSplat은 이 문제를 novel view synthesis와 real-time rendering 관점에서 다룬다.

### Volumetric Rendering for View Synthesis

NeRF는 neural scene representation과 volumetric rendering으로 높은 품질의 novel view synthesis를 가능하게 했다. 학습·렌더링 속도와 품질을 개선하는 연구가 이어졌지만, ray를 따라 sample을 평가하는 비용이 남는다.

3DGS는 scene을 3D Gaussian ellipsoid 집합으로 표현하고 differentiable rasterization으로 렌더링한다. EventSplat은 이 representation과 빠른 rasterizer를 유지하면서, RGB image 대신 event sequence로 최적화할 수 있도록 학습 과정을 바꾼다.

### Novel View Synthesis via Event Data

Ev-NeRF, E-NeRF, Robust e-NeRF, EventNeRF 등의 연구는 event supervision을 neural rendering에 활용했다. 논문은 E2GS, EvGGS, EAdeblur-GS, EV-GS, Event3DGS 등 event와 Gaussian representation을 결합하는 연구도 관련 연구로 언급한다.

EventSplat의 구성은 event accumulation을 통한 3DGS supervision, event-based point cloud initialization, camera trajectory interpolation을 함께 사용하는 것이다. Event-to-video 출력 자체를 최종 appearance target으로 사용하는 baseline과 달리, 주된 reconstruction supervision은 관측 event의 변화량에서 얻는다.

## Method

### Preliminaries

#### Event Camera Data

일반 camera가 intensity $I$를 관측한다면, event camera는 log-intensity $L=\log I$의 변화를 측정한다. 원문은 현재 시각과 직전 event 시각의 차이가 threshold를 넘는 조건을 다음과 같이 쓴다.

$$
\begin{equation}
l_{x,y}(t)-l_{x,y}(t_{\mathrm{recent}})>\Delta.
\tag{1}
\end{equation}
$$

식 (1)은 밝기가 증가하는 쪽의 threshold 조건이다. 실제 event는 $e=(t,x,y,p)$로 표현되며, polarity $$p\in\{-1,1\}$$로 감소와 증가를 구분한다. Contrast sensitivity $\Delta$는 positive와 negative event에 대해 서로 다를 수 있다.

Color event camera는 Bayer RGB pattern을 사용하므로, 각 pixel에서 발생한 event는 하나의 color channel에 대한 변화만 나타낸다. 이 성질 때문에 RGB rendering을 event와 비교할 때 remosaicing이 필요하다.

#### 3D Gaussian Splatting

3DGS는 scene을 Gaussian ellipsoid의 집합으로 표현하며 view-dependent color를 spherical harmonics로 저장한다. Camera viewing transformation $W$와 projection의 Jacobian $J$를 사용해 covariance를 image plane으로 투영한다.

$$
\begin{equation}
\Sigma'=JW\Sigma W^\top J^\top.
\tag{2}
\end{equation}
$$

투영된 Gaussians를 depth 순서로 정렬한 집합을 $N_s$라 하면, pixel intensity는 alpha compositing으로 계산한다.

$$
\begin{equation}
I=\sum_{i\in N_s}
\left(c_i\alpha_i'\prod_{j=1}^{i-1}(1-\alpha_j')\right).
\tag{3}
\end{equation}
$$

여기서 $$c_i,\alpha_i'$$는 color와 opacity 항이다. EventSplat은 이 image rasterization을 사용하되, 하나의 rendered image를 RGB ground truth와 비교하는 대신 두 시점 사이의 변화량을 비교한다.

### Method Overview

입력은 event sequence와 각 timestamp에 대응하는 camera pose다. Event마다 pose가 직접 주어지지 않으면, 일정한 rate로 얻은 pose들로부터 interpolation한다. 목표는 이 posed event sequence로 static Gaussian scene $\mathcal G$를 학습하는 것이다.

논문은 시간 구간 $$[a,b]$$의 log-intensity change를 모델링하려는 목표를 아래 식으로 제시한다.

$$
\begin{equation}
E(a,b):=\int_a^b\log\bigl(I'(t)\bigr)\,dt.
\tag{4}
\end{equation}
$$

식 (4)는 원문 표기를 그대로 옮겼다. 다만 일반적인 미분 표기에서 $\log(I'(t))$는 log-intensity의 시간 미분 $$\frac{d}{dt}\log I(t)$$와 다르다. 따라서 이 식을 endpoint log-difference와 동일한 항등식으로 읽어서는 안 된다. 실제 학습 구성은 뒤의 식 (5)의 accumulated event와 식 (6)의 두 log image 차이를 비교하는 형태로 명시되어 있다.

![Event accumulation과 두 endpoint rendering을 비교하는 EventSplat pipeline](/assets/img/eventsplat/fig2_pipeline.png)
_Fig 2. 선택한 event 구간의 누적 변화량 D와, 두 endpoint pose에서 렌더링한 log image의 차이를 비교한다. Event-to-video guided initialization과 cubic spline interpolation이 각각 초기 geometry와 endpoint pose를 제공한다._

### Event Accumulation

전체 trajectory에서 start와 endpoint를 무작위로 선택한다. 구간 길이도 무작위로 정하며, 원문은 최대 길이를 전체 event 수의 1%–10% 범위로 설정한다고 설명한다. 서로 다른 길이의 구간을 사용해 local 및 global geometry 정보를 학습에 반영한다.

선택한 구간에서 pixel별 polarity에 contrast threshold를 곱해 합산한다.

$$
\begin{equation}
D_{x,y}:=
\sum_{\substack{k\in\{k_{\mathrm{start}},\ldots,k_{\mathrm{end}}\}\\x_k=x,\ y_k=y}}
p_k\Delta.
\tag{5}
\end{equation}
$$

$D$는 RGB appearance가 아니라 선택한 구간의 quantized log-intensity change를 나타낸다. Positive와 negative threshold가 다르면 각 polarity에 맞는 threshold를 사용하도록 확장할 수 있다.

이 representation은 구간의 시작·끝 시각과 연결된 누적 변화량을 보존한다. 반면 구간 내부의 개별 event 순서와 timestamp를 모두 별도의 image channel에 저장하는 것은 아니다. 학습에 사용하는 temporal constraint는 선택한 구간과 그 양 끝의 camera pose에 연결된다.

### Training View Generation and Remosaicing

같은 start와 endpoint pose에서 Gaussian scene을 각각 렌더링한다. 두 결과를 log image로 변환한 뒤 차이를 취한다.

$$
\begin{equation}
\hat D:=\hat L_2-\hat L_1.
\tag{6}
\end{equation}
$$

Color event camera의 경우, rasterizer가 출력하는 RGB image와 event accumulation image의 channel 구조가 다르다. Rendered RGB image는 pixel마다 세 channel을 가지지만, event sensor는 Bayer pattern에 따라 pixel마다 한 channel만 측정한다. 이를 맞추기 위해 다음 remosaicing을 수행한다.

$$
\begin{equation}
\operatorname{Remosaicing}:\mathbb R^{H\times W\times3}
\longrightarrow\mathbb R^{H\times W}.
\tag{7}
\end{equation}
$$

각 $2\times2$ pixel block에서 RGB channel에 다음 mask를 Hadamard multiply한다.

$$
\begin{equation}
R=\begin{bmatrix}1&0\\0&0\end{bmatrix},\qquad
G=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad
B=\begin{bmatrix}0&0\\0&1\end{bmatrix}.
\tag{8}
\end{equation}
$$

Mask를 적용한 channel들은 서로 겹치지 않는 pixel 위치에 값을 가지므로, channel 방향으로 더하면 single-channel Bayer image가 된다. 여기에 element-wise logarithm을 적용해 $$\hat L_1,\hat L_2$$를 얻는다.

즉 RGB rendering을 관측 sensor의 Bayer 구조로 되돌리는 연산이다. Sparse하고 비동기적인 event를 먼저 완전한 RGB image로 demosaicing하는 방식과는 방향이 반대다.

### Event-to-Video Guided Initialization

Event는 brightness가 변한 pixel에서만 발생하므로, raw event supervision만으로 background와 geometry를 안정적으로 초기화하기 어렵다. 논문은 pretrained event-to-video model이 가진 prior를 이용한다.

먼저 event stream으로 영상을 생성하고, 이 영상들에 Structure from Motion을 적용해 Gaussian의 초기 위치를 얻는다. 생성 영상에는 noise가 있고 실제 RGB intensity를 정확히 재현하지 못할 수 있지만, texture와 background에 대한 정보가 초기 geometry를 얻는 데 도움이 된다.

이 단계의 역할은 appearance supervision을 대체하는 것이 아니라 optimization의 출발점을 제공하는 것이다. 이후에는 accumulated event와 rendered log-difference를 맞추며 scene을 개선한다.

### Event Camera Trajectory Interpolation

Event timestamp는 비동기적으로 나타나지만, camera tracking이나 motion estimation 시스템은 보통 일정한 rate로 pose를 출력한다. 따라서 event accumulation 구간의 endpoint와 주어진 pose timestamp가 정확히 일치하지 않을 수 있다.

EventSplat은 Cubic Spline Interpolation과 Spherical Cubic Spline Interpolation을 사용해 이 사이의 pose를 추정한다. Discrete pose 사이를 매끄럽게 연결하여 camera motion을 더 잘 근사하려는 목적이다.

이 interpolation은 식 (6)에 필요한 두 endpoint view를 얻기 위한 것이다. Exposure 전체의 RGB intensity를 적분하거나, trajectory 내부의 모든 시점에서 image를 렌더링해 평균하는 과정은 아니다. 또한 event-only pose estimation을 새로 수행하는 대신, 주어진 pose sample 사이의 trajectory를 보간한다.

### Loss Function

3DGS의 최적화에는 reconstruction 항과 SSIM 기반 항을 결합한다.

$$
\begin{equation}
\mathcal L=(1-\lambda)\mathcal L_1
+\lambda\mathcal L_{\mathrm{SSIM}}.
\tag{9}
\end{equation}
$$

비교 대상은 식 (5)의 observed event accumulation $D$와 식 (6)의 rendered log-difference $\hat D$다. Real-world data에서는 accumulated image를 만들 때 undistortion도 수행하며, 실제 event가 누적된 위치에 대해서만 loss를 계산한다고 설명한다.

## Experiments

### Datasets

실험은 camera intrinsics, lens distortion parameters, 일정한 rate로 샘플링된 camera poses에 접근할 수 있다고 가정한다. Pose는 임의의 event timestamp에서 정확한 interpolation이 가능하도록 충분히 높은 frequency로 제공되어야 한다.

#### Synthetic Scenes

Robust e-NeRF에서 사용한 ESIM 기반 synthetic event dataset을 이용한다. White background의 일곱 scene은 chair, drums, ficus, hotdog, lego, materials, mic이다. Camera가 object 주위를 움직이며 다양한 view를 관측한다.

#### Real-World Scenes

EDS에서는 다음 다섯 sequence를 사용한다.

- `03_rocket_earth_dark`
- `07_ziggy_and_fuzz_hdr`
- `08_peanuts_running`
- `11_all_characters`
- `13_airplane`

TUM-VIE에서는 `mocap-1d-trans`와 `mocap-desk2`를 사용한다. EDS는 Prophesee Gen 3, TUM-VIE는 Prophesee Gen 4 sensor로 기록되었다. Quantitative novel view synthesis 평가는 EDS에 대해 제시하고, TUM-VIE는 qualitative comparison에 포함한다.

### Baselines

비교 방법은 Robust-e-NeRF와 E2VID+3DGS다. Robust-e-NeRF는 InstantNGP 기반 event neural rendering 방법이다. E2VID+3DGS는 event-to-video model로 생성한 영상을 기존 3DGS의 학습 영상으로 사용한다.

EventSplat도 event-to-video prior를 활용하지만, generated frame 자체를 최종 supervision으로 삼는 baseline과 달리 event accumulation에 직접 맞추는 것이 차이다.

### Metrics

Novel view quality는 PSNR, SSIM, VGG16 기반 LPIPS로 평가하며, rendering time도 비교한다. PSNR·SSIM은 높을수록, LPIPS는 낮을수록 좋다.

Event는 absolute intensity가 아니라 log-intensity의 변화만 측정하기 때문에, 예측 intensity에는 관측만으로 결정되지 않는 offset이 남는다. 논문은 evaluation reference를 이용해 logarithmic domain에서 linear color transformation을 적용한 뒤 비교한다. 따라서 보고된 image quality는 이런 intensity alignment를 거친 결과다.

### Results and Evaluation

Scene마다 event sequence로 학습한 뒤 held-out view 또는 dataset이 제공하는 test view에서 평가한다. 학습 결과는 3D Gaussian 집합이며, inference에서는 이를 rasterization해 RGB 또는 grayscale image를 생성한다.

#### Synthetic Scenes

Table 1의 synthetic 실험은 random point initialization을 사용했다. EventSplat의 평균 PSNR은 28.14 dB, SSIM은 0.953, LPIPS는 0.051이다. Robust-e-NeRF는 각각 28.19 dB, 0.945, 0.057을 기록한다. EventSplat의 SSIM과 LPIPS가 더 좋지만, 평균 PSNR은 Robust-e-NeRF가 근소하게 높다.

![Synthetic 일곱 scene의 novel view synthesis 정량 비교](/assets/img/eventsplat/table1_synthetic.png)
_Table 1. Synthetic scene의 PSNR, SSIM, LPIPS 비교. 이 표의 EventSplat 결과는 random point initialization으로 학습했다._

E2VID+3DGS의 평균 PSNR은 19.29 dB이며, SSIM은 0.917, LPIPS는 0.118이다. EventSplat은 이 baseline보다 세 평균 지표 모두 좋은 결과를 보인다.

Fig 3의 rendering time은 chair에서 EventSplat 3.669 ms와 Robust-e-NeRF 73.147 ms, drums에서 3.278 ms와 90.556 ms, mic에서 4.499 ms와 64.563 ms다. E2VID+3DGS도 빠르게 렌더링하지만, 정성 결과에서는 background와 object appearance의 차이가 나타난다.

![Synthetic scene의 정성 결과와 rendering time 비교](/assets/img/eventsplat/fig3_synthetic_comparison.png)
_Fig 3. Chair, drums, mic의 정성 비교. 왼쪽부터 E2VID+3DGS, Robust-e-NeRF, EventSplat, ground truth이며, 각 rendering 아래에 소요 시간(ms)이 표시되어 있다._

#### Real Scenes

EDS 다섯 sequence의 평균값은 EventSplat이 PSNR 18.86 dB, SSIM 0.792, LPIPS 0.363이다. Robust-e-NeRF는 16.25 dB, 0.739, 0.543이며, E2VID+3DGS는 15.51 dB, 0.692, 0.375다.

![EDS real scene 다섯 sequence의 정량 비교](/assets/img/eventsplat/table2_real.png)
_Table 2. EDS의 03, 07, 08, 11, 13 sequence에 대한 정량 결과. EventSplat이 평균 PSNR·SSIM·LPIPS에서 가장 좋은 결과를 보인다._

평균 지표는 EventSplat이 가장 좋지만 모든 sequence의 모든 지표에서 최고인 것은 아니다. 예를 들어 03의 SSIM은 Robust-e-NeRF가, 08의 LPIPS는 E2VID+3DGS가 더 좋다.

정성 비교에서는 texture와 contrast, checkerboard 및 book cover의 detail을 보여준다. TUM-VIE의 `mocap-1d-trans`와 `mocap-desk2`에서는 uniform surface와 edge를 함께 비교한다. 저자들은 event accumulation이 aggregation을 통해 signal-to-noise ratio를 개선했을 가능성을 설명하며, 화면 경계의 artifact는 event camera의 좁은 field of view와 연결한다.

![EDS와 TUM-VIE의 real scene 정성 비교](/assets/img/eventsplat/fig4_real_comparison.png)
_Fig 4. 위쪽 세 행은 EDS의 07, 08, 11 sequence, 아래쪽 두 행은 TUM-VIE의 mocap-1d-trans와 mocap-desk2다. 열 순서는 E2VID+3DGS, Robust-e-NeRF, EventSplat, ground truth다._

Fig 4에 표시된 EventSplat의 rendering time은 3.725–6.087 ms다. 같은 그림의 Robust-e-NeRF는 1126.765–2669.292 ms를 기록한다. 이 수치는 novel view rendering 시간이며, initialization과 scene training을 포함한 전체 reconstruction 시간은 아니다.

### Ablations

Event-to-video guided initialization과 cubic spline interpolation의 효과를 단계적으로 비교한다.

![Guided initialization과 cubic spline interpolation의 ablation](/assets/img/eventsplat/table3_ablation.png){: width="430" }
_Table 3. Random initialization, guided initialization, guided initialization과 cubic spline을 함께 사용한 설정의 비교._

Random initialization에서는 PSNR 18.02 dB, SSIM 0.767, LPIPS 0.397이다. Guided initialization을 적용하면 18.75 dB, 0.788, 0.359로 세 지표가 모두 개선된다. Fig 5는 EDS 07 sequence의 background detail이 초기화에 따라 어떻게 달라지는지 보여준다.

![Random initialization과 event-to-video guided initialization의 background 비교](/assets/img/eventsplat/fig5_initialization.png)
_Fig 5. EDS 07 sequence에서 random initialization과 event-to-video guided initialization을 비교한다. Guided initialization은 확대 영역의 background detail을 더 잘 복원한다._

Cubic spline을 추가하면 PSNR과 SSIM은 각각 18.86 dB, 0.792로 상승한다. 다만 LPIPS는 0.359에서 0.363으로 소폭 증가하므로, interpolation의 개선 효과가 모든 지표에 동일하게 나타나는 것은 아니다.

### Limitations

논문은 두 가지 한계를 명시한다.

1. Static scene의 novel view synthesis만 대상으로 하며, independently moving object가 포함된 dynamic scene은 처리하지 못한다.
2. 두 viewpoint 사이의 relative intensity change로 학습하므로 absolute intensity를 직접 결정할 수 없다. Inference에서는 evaluation data를 reference로 하는 linear transformation이 필요하며, 이 transformation은 training에는 사용하지 않는다.

## Conclusion

EventSplat은 event의 brightness change를 Gaussian rasterization과 연결하여, 움직이는 event camera로부터 static scene의 novel view synthesis를 수행한다. 선택한 구간의 event accumulation과 두 endpoint rendering의 log-difference를 맞추고, event-to-video guided initialization과 spline interpolation으로 geometry 초기화와 pose 대응을 보완한다.

실험에서는 event-based NeRF와 비교해 높은 rendering speed를 보이면서, synthetic 및 real scene의 평균 perceptual quality를 개선했다. 동시에 absolute intensity ambiguity와 static scene이라는 적용 범위는 남아 있으며, 논문은 dynamic scene으로의 확장을 향후 과제로 제시한다.
