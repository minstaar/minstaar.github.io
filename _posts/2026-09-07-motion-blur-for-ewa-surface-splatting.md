---
title: "[논문리뷰] Motion Blur for EWA Surface Splatting"
date: 2026-09-07 01:47:00 +0900
categories: [논문리뷰, Computer Graphics]
tags: [EWA Surface Splatting, Point-Based Rendering, Motion Blur, Spatio-Temporal Filtering, GPU Rendering, Eurographics]
description: "Eurographics 2010"
math: true
toc: true
compact_captions: true
---

> \[[Paper](https://cgl.ethz.ch/Downloads/Publications/Papers/2010/Hein10/Hein10.pdf)\] \[[Page](https://cgl.ethz.ch/publications/papers/paperHein10.php)\]
>
> Simon Heinzle, Johanna Wolf, Yoshihiro Kanamori, Tim Weyrich, Tomoyuki Nishita, Markus Gross

## Introduction

Point-based geometry는 복잡한 surface를 triangle mesh 대신 조밀한 point sample로 표현한다. 특히 triangle의 projected area가 몇 pixel보다 작아지는 대규모 model에서는 triangle setup 비용과 rasterization 중 primitive가 사라지며 발생하는 aliasing이 문제가 된다. EWA surface splatting은 각 sample 주변의 surface를 planar elliptical Gaussian reconstruction kernel인 surface splat으로 근사하고, 이를 screen space에서 blending한다. Projected kernel에 low-pass filter를 결합하므로 hole과 aliasing을 줄이면서 point-sampled surface를 렌더링할 수 있다.

기존 EWA surface splatting은 한 시점의 정지 영상만 만든다. 하지만 실제 camera는 exposure 동안 들어온 빛을 적분하므로, camera나 object가 움직이면 image가 상대 운동 방향으로 퍼진다. 이를 정확히 합성하는 기본 방법은 시간축 supersampling이지만, 가장 빠른 motion의 Nyquist frequency를 만족하려면 scene 전체를 많은 시점에서 반복 렌더링해야 한다.

![EWA surface splatting으로 렌더링한 motion blur 예시](/assets/img/motion-blur-ewa-surface-splatting/fig1_teaser.png){: width="560" }
_Fig 1. 제안한 GPU implementation으로 렌더링한 point-based object의 motion blur 예시._

이 논문은 EWA surface splatting의 이론을 시간 차원으로 확장한다. 핵심은 움직이는 2D surface splat을 여러 시점에서 반복해서 그리는 대신, spatial footprint와 temporal motion을 하나로 묶는 3D Gaussian reconstruction kernel을 구성하는 것이다. 이렇게 얻은 ellipsoid를 viewing ray를 따라 적분하면 point-based object의 motion blur를 직접 계산할 수 있다.

논문의 기여는 세 부분으로 정리된다.

1. EWA surface splatting을 spatial-temporal domain으로 확장해 point-sampled dynamic scene의 motion blur를 연속적으로 표현한다.
2. Planar splat과 instantaneous velocity로 ellipsoidal reconstruction primitive를 만들고, 이를 viewing ray를 따라 적분하는 rendering algorithm을 유도한다.
3. Temporal visibility를 근사하는 six-pass GPU pipeline을 구현하고, 비슷한 visual quality의 temporal supersampling보다 적은 sample로 interactive rendering을 보인다.

## Related Work

### Point-Based Rendering

초기 point rendering은 point를 framebuffer로 forward mapping하는 방식에서 시작했다. 이후 연구는 point primitive의 anti-aliasing과 고품질 surface reconstruction에 초점을 맞췄다. EWA surface splatting은 Heckbert의 texture filtering을 point-sampled geometry에 적용하여, irregular sample로부터 연속 surface signal을 재구성하고 screen-space low-pass filtering을 수행한다.

관련 구현은 textured quad를 사용하거나 GPU의 programmable pipeline을 활용했으며, 전용 surface-splatting hardware도 제안되었다. Point set surface의 implicit reconstruction, ray tracing, dynamic upsampling도 point-based geometry를 렌더링하는 다른 흐름으로 소개된다.

### Motion Blur

Motion blur rendering은 크게 시간 적분을 직접 sample하는 방법과, sampling 전에 spatio-temporal signal을 bandlimit하는 방법으로 나뉜다. Distributed ray tracing과 accumulation buffer는 여러 시점의 sample을 합성하지만, noise나 banding을 피하려면 높은 sampling rate가 필요하다. Frequency-domain 분석을 이용한 연구는 space-time signal의 shear에 맞춘 sampling과 reconstruction filter를 제안했다.

다른 방법들은 constant shading이나 linear motion을 가정하거나, particle을 line segment로 바꾸고 새로운 semi-transparent geometry를 생성한다. Image-space post-processing은 rendered still image에 point spread function을 convolution하지만, local geometry와 복잡한 depth overlap을 정확히 다루기 어렵다.

이 논문과 가장 가까운 Guan과 Mueller의 방법은 2D object-space ellipse를 motion 방향으로 확장해 3D reconstruction kernel을 만든다. 그러나 static EWA rendering 위에 motion-hint ellipse를 합성하고 temporal visibility를 무시하므로, 높은 overlap에서 stripe와 overblending artifact가 나타난다. 본 논문은 kernel의 spatial contribution과 visible time interval을 함께 계산한다는 점을 차이로 둔다.

## Method

### Preliminaries

#### Motion Blur as Temporal Integration

시간 $t$에서 screen-space 위치 $\mathbf{x}$의 연속 signal을 $g(\mathbf{x},t)$라 하자. Exposure 구간 $T$에서 얻는 motion-blurred image $G_T$는 다음의 weighted temporal integration으로 정의된다.

$$
\begin{equation}
G_T(\mathbf{x})=\int_T a(t)g(\mathbf{x},t)\,dt.
\tag{1}
\end{equation}
$$

$a(t)$는 shutter와 recording medium의 영향을 나타내는 time-dependent weighting function이다. 따라서 motion blur 생성은 $g(\mathbf{x},t)$라는 spatio-temporal signal을 exposure 동안 resampling하는 문제로 볼 수 있다.

#### EWA Surface Reconstruction

Scene은 local surface parameter $\mathbf{u}$ 위의 연속 surface function $f(\mathbf{u},t)$로 표현한다. Projective mapping은 source space의 surface point를 screen-space 위치로 보낸다.

$$
\begin{equation}
m(\mathbf{u},t):\mathbb{R}^{2}\times\mathbb{R}
\rightarrow\mathbb{R}^{2}\times\mathbb{R}.
\tag{2}
\end{equation}
$$

고정된 $t$에서 $m$이 locally invertible이면 screen-space signal은 다음과 같다.

$$
\begin{equation}
g(\mathbf{x},t)=f\!\left(m^{-1}(\mathbf{x},t),t\right).
\tag{3}
\end{equation}
$$

Surface function은 time-dependent point sample 집합 $$\{P_k(t)\}$$으로 재구성한다. $k$번째 sample의 위치를 $\mathbf{u}_k(t)$, attribute를 $w_k(t)$, elliptical reconstruction kernel을 $r_k$라 하면,

$$
\begin{equation}
f(\mathbf{u},t)=\sum_{k\in\mathcal{N}}
w_k(t)r_k\!\left(\mathbf{u}-\mathbf{u}_k(t),t\right).
\tag{4}
\end{equation}
$$

이다. 이후 유도에서는 각 reconstruction kernel에 대한 mapping을 $m_k$로 표기한다.

### Time-Varying EWA Surface Splatting

움직이는 surface에는 위치뿐 아니라 visibility도 시간에 따라 변한다. $v_k(\mathbf{x},t)$를 $k$번째 sample의 temporal visibility function, $r'_k$를 screen space로 projection한 kernel이라 하면 식 (3)과 (4)는 다음으로 결합된다.

$$
\begin{equation}
g(\mathbf{x},t)=\sum_{k\in\mathcal{N}}
w_k(t)v_k(\mathbf{x},t)r'_k(\mathbf{x},t),
\qquad
r'_k(\mathbf{x},t)=r_k\!\left(m_k^{-1}(\mathbf{x},t)-\mathbf{u}_k(t),t\right).
\tag{5}
\end{equation}
$$

Spatio-temporal anti-aliasing filter $h(\mathbf{x},t)$로 signal을 bandlimit하면,

$$
\begin{equation}
\begin{aligned}
g'(\mathbf{x},t)
&=g(\mathbf{x},t)*h(\mathbf{x},t)\\
&=\int_T\int_{\mathbb{R}^2}
g(\boldsymbol{\xi},\tau)
h(\mathbf{x}-\boldsymbol{\xi},t-\tau)
\,d\boldsymbol{\xi}\,d\tau\\
&=\sum_{k\in\mathcal{N}}w_k(t)\rho_k(\mathbf{x},t),
\end{aligned}
\tag{6}
\end{equation}
$$

이며 filtered resampling kernel은

$$
\begin{equation}
\rho_k(\mathbf{x},t)=
\int_T\int_{\mathbb{R}^2}
v_k(\boldsymbol{\xi},\tau)r'_k(\boldsymbol{\xi},\tau)
h(\mathbf{x}-\boldsymbol{\xi},t-\tau)
\,d\boldsymbol{\xi}\,d\tau
\tag{7}
\end{equation}
$$

로 주어진다. 논문은 먼저 reconstruction kernel을 filtering한 뒤, filtered kernel을 기준으로 visibility를 결정한다.

### Temporal Reconstruction

기존 EWA가 surface의 local spatial patch를 2D kernel로 근사한다면, 제안 방법은 point trajectory도 시간축에서 sample하고 local linear trajectory patch로 근사한다. 시간 $t_k$에서의 2D elliptical Gaussian을 instantaneous velocity 방향의 1D Gaussian과 convolution하여 ellipsoidal 3D kernel $R_{k t_k}$를 만든다.

![움직이는 2D splat과 spatio-temporal 3D Gaussian kernel의 비교](/assets/img/motion-blur-ewa-surface-splatting/fig2_temporal_kernels.png)
_Fig 2. 왼쪽은 trajectory를 따라 움직이는 2D splat, 가운데는 spatial-temporal domain을 함께 갖는 3D Gaussian reconstruction kernels, 오른쪽은 2D kernel temporal supersampling이다._

이 kernel로 space와 time에 대해 연속적인 surface function을 재구성한다.

$$
\begin{equation}
f(\mathbf{u},t)=\sum_{k\in\mathcal{N}}\sum_{t_k}
w_k(t)R_{k t_k}\!\left(\mathbf{u}-\mathbf{u}_{k t_k}\right).
\tag{8}
\end{equation}
$$

Screen space로 mapping하고 visibility를 포함하면,

$$
\begin{equation}
g(\mathbf{x},t)=\sum_{k\in\mathcal{N}}\sum_{t_k}
w_k(t)v_{k t_k}(\mathbf{x},t)R'_{k t_k}(\mathbf{x})
\tag{9}
\end{equation}
$$

가 된다. $t_k$는 각 point의 motion과 shading change에 따라 선택되는 trajectory sample time이다. Bandlimited signal과 그 resampling kernel은 다음과 같다.

$$
\begin{equation}
g'(\mathbf{x},t)=\sum_{k\in\mathcal{N}}\sum_{t_k}
w_k(t)\hat{\rho}_{k t_k}(\mathbf{x},t),
\tag{10}
\end{equation}
$$

$$
\begin{equation}
\hat{\rho}_{k t_k}(\mathbf{x},t)=
\int_T\int_{\mathbb{R}^2}
v_{k t_k}(\boldsymbol{\xi},\tau)R'_{k t_k}(\boldsymbol{\xi},\tau)
h(\mathbf{x}-\boldsymbol{\xi},t_k-\tau)
\,d\boldsymbol{\xi}\,d\tau.
\tag{11}
\end{equation}
$$

실제 rendering에서는 A-Buffer 방식으로 pixel에 기여하는 filtered kernel과 visible interval을 구한 뒤, 이 visibility를 식 (1)의 temporal opacity contribution으로 사용한다.

### Rendering

#### 3D Spatio-Temporal Reconstruction Filter

원래 splat의 두 축 $\mathbf{a}_1,\mathbf{a}_2$는 local surface plane을 이룬다. 세 번째 축은 exposure sub-interval의 시작과 끝 사이 motion vector $\mathbf{m}$으로부터 $$\mathbf{a}_3=\frac{\alpha}{2}\mathbf{m}$$으로 구성하며, $\alpha$는 temporal variance를 조절한다.

![Surface splat과 motion vector로 구성한 3D kernel 좌표계](/assets/img/motion-blur-ewa-surface-splatting/fig3_kernel_construction.png){: width="540" }
_Fig 3. 두 surface-splat axis와 instantaneous motion vector로 3D reconstruction kernel을 만들고, transformation $T$를 통해 object space와 kernel parameter space를 연결한다._

Reconstruction kernel과 low-pass filter는 homogeneous coordinate $$\mathbf{x}=[x\ y\ z\ 1]^\top$$에서 정의한 3D ellipsoidal Gaussian이다.

$$
\begin{equation}
G_Q^3(\mathbf{x})=
\sqrt{\frac{\delta^3|\mathbf{Q}|}{\pi^3}}
\exp\!\left(-\delta\mathbf{x}^\top\mathbf{Q}\mathbf{x}\right).
\tag{12}
\end{equation}
$$

$\delta$는 Gaussian variance를 제어한다. Quadric matrix는 $$\mathbf{Q}=\mathbf{T}^{-\top}\mathbf{D}\mathbf{T}^{-1}$$로 분해하며, center $\mathbf{u}$와 세 axis로 transformation을 구성한다.

$$
\begin{equation}
\mathbf{T}=
\begin{bmatrix}
\mathbf{a}_1 & \mathbf{a}_2 & \mathbf{a}_3 & \mathbf{u}\\
0&0&0&1
\end{bmatrix}.
\tag{13}
\end{equation}
$$

$$
\begin{equation}
\mathbf{D}=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&0
\end{bmatrix}.
\tag{14}
\end{equation}
$$

Object space의 point에 $\mathbf{T}^{-1}$을 적용하면 Gaussian parameter space로 이동하며, 이 공간의 iso-level은 unit sphere로 해석할 수 있다. Screen-space bandlimiting은 pixel grid의 object-space spacing을 추정하고, projected axis가 filter width보다 작으면 filter radius까지 확대하는 방식으로 근사한다.

#### Sampling the Reconstruction Filter

한 pixel에 대한 3D Gaussian의 contribution은 exposure 동안 viewing ray를 따라 kernel을 적분해 계산한다. Window coordinate의 pixel을 $$\mathbf{p}=[x_w\ y_w\ 0\ 1]^\top$$, viewing direction을 $$\mathbf{d}=[0\ 0\ 1\ 0]^\top$$라 하면 ray는 $\mathbf{r}(s)=\mathbf{p}+s\mathbf{d}$다. Viewport, projection, modelview matrix를 각각 $\mathbf{V},\mathbf{P},\mathbf{M}$이라 할 때 parameter space ray는

$$
\begin{equation}
\tilde{\mathbf{r}}(s)=
(\mathbf{VPMT})^{-1}\mathbf{r}(s)
=\tilde{\mathbf{p}}+s\tilde{\mathbf{d}}
\tag{15}
\end{equation}
$$

로 변환된다. 이를 temporal coordinate로 정규화한 integration line은

$$
\begin{equation}
\mathbf{l}(t)=\mathbf{b}+\mathbf{f}t,
\tag{16}
\end{equation}
$$

$$
\begin{equation}
\mathbf{b}=\tilde{\mathbf{p}}'
-\frac{\tilde{p}'_z}{\tilde{d}'_z}\tilde{\mathbf{d}}',
\qquad
\mathbf{f}=-\frac{\tilde{\mathbf{d}}'}{\tilde{d}'_z}
\tag{17}
\end{equation}
$$

이다. Parameter space의 $z$축은 temporal dimension이며, 전체 exposure는 $$t\in[-1,1]$$에 대응한다.

![Viewing ray를 Gaussian parameter space의 integration line으로 변환하는 과정](/assets/img/motion-blur-ewa-surface-splatting/fig4_ray_integration.png){: width="540" }
_Fig 4. World-space viewing ray를 ellipsoid parameter space로 옮긴 뒤 exposure time에 대응하는 integration line으로 정규화한다. 실제 적분은 kernel이 보이는 $$[t_a,t_b]$$에서 수행한다._

Visible interval $$[t_a,t_b]$$에서 line integral은 spatial Gaussian과 temporal Gaussian의 곱으로 분리된다.

$$
\begin{equation}
\begin{aligned}
\int_{t_a}^{t_b}G^3(\mathbf{l}(t))\,dt
&=\int_{t_a}^{t_b}
e^{-\delta\mathbf{l}^\top(t)\mathbf{l}(t)}\,dt\\
&=\int_{t_a}^{t_b}
e^{-\delta(\mathbf{b}^\top\mathbf{b}
+2\mathbf{b}^\top\mathbf{f}t
+\mathbf{f}^\top\mathbf{f}t^2)}\,dt\\
&=\int_{t_a}^{t_b}
e^{-\delta\mathbf{l}_{xy}^\top(t)\mathbf{l}_{xy}(t)}
e^{-\delta t^2}\,dt.
\end{aligned}
\tag{18}
\end{equation}
$$

Finite integral의 closed form은 error function으로 표현된다.

$$
\begin{equation}
\int_{t_a}^{t_b}G^3(\mathbf{l}(t))\,dt
=\frac{\sqrt{\pi}
e^{-\delta\frac{(\mathbf{b}^\top\mathbf{b})(\mathbf{f}^\top\mathbf{f})-(\mathbf{b}^\top\mathbf{f})^2}{\mathbf{f}^\top\mathbf{f}}}}
{2\sqrt{\delta\mathbf{f}^\top\mathbf{f}}}
\left(\operatorname{erf}(F(t_b))-\operatorname{erf}(F(t_a))\right),
\tag{19}
\end{equation}
$$

$$
\begin{equation}
F(x)=\sqrt{\frac{\delta}{\mathbf{f}^\top\mathbf{f}}}
\left(\mathbf{f}^\top\mathbf{f}\,x+\mathbf{b}^\top\mathbf{f}\right),
\qquad
\operatorname{erf}(x)=\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}\,dt.
\tag{20}
\end{equation}
$$

GPU에서는 $\exp$와 $\operatorname{erf}$를 texture look-up table로 평가한다.

#### Bounding Volume and Visibility

Gaussian은 무한한 support를 가지지만 빠르게 감소하므로, 고정 cut-off에서 kernel을 3D tube로 제한한다. $$\mathbf{D}_{xy}=\operatorname{diag}(1,1,0,0)$$라 하면 viewing ray와 tube의 교차는 다음 quadratic relation으로 계산한다.

$$
\begin{equation}
\mathbf{r}(s)^\top(\mathbf{VPMT})^{-\top}
\mathbf{D}_{xy}(\mathbf{VPMT})^{-1}\mathbf{r}(s)
=\tilde{\mathbf{r}}_{xy}(s)^\top\tilde{\mathbf{r}}_{xy}(s)=1.
\tag{21}
\end{equation}
$$

Tube는 parameter-space의 $$\tilde r_z(s)=\pm1$$인 두 end ellipse로도 제한된다. Screen space에서는 두 end ellipse의 axis-aligned bounding box를 결합한 convex hull을 rasterization primitive로 사용하여 단순한 전체 bounding box보다 fragment 수를 줄인다.

정확한 visibility 처리는 각 kernel의 visible time interval과 depth interval을 저장하는 A-Buffer를 사용한다. Temporal overlap과 depth overlap이 모두 있으면 같은 surface의 kernel contribution을 누적하고, temporal overlap만 있으면 뒤쪽 interval에서 occluded 구간을 제거한다. 이렇게 얻은 $v_k(\mathbf{x},t)$에 shutter function $a(t)$를 적용해 최종 pixel contribution을 계산한다.

### GPU Implementation

논문의 GPU implementation은 정확한 A-Buffer 대신 six-pass approximation을 사용한다.

![Motion-blurred image를 합성하는 six-pass GPU pipeline](/assets/img/motion-blur-ewa-surface-splatting/fig6_gpu_pipeline.png){: width="560" }
_Fig 6. 시작·끝 visibility, earliest/latest visible time, kernel blending, normalization으로 이어지는 six-pass pipeline._

1. Exposure 시작 시점의 elliptical splat을 depth buffer에 렌더링한다.
2. Exposure 끝 시점의 splat을 별도 depth buffer에 렌더링한다.
3. 각 pixel에서 geometry가 보이기 시작하는 earliest time $t_{\min}$을 구한다.
4. 같은 방식으로 latest time $t_{\max}$를 구한다.
5. Visible reconstruction kernel의 line integral을 accumulation buffer에 blending한다.
6. 누적된 color를 kernel weight 합으로 normalization하고 background와 합성한다.

Pass 3–5의 vertex shader는 시작·끝 transformation에서 velocity vector와 두 시점의 normal을 계산한다. Geometry shader는 두 end ellipse의 bounding box를 convex octagon으로 결합한다. Fragment shader는 viewing ray와 bounding cylinder를 교차시켜 한 kernel의 integration interval을 얻는다. 시작·끝 depth buffer 중 하나에서 보이는 ellipsoid만 처리하여 불필요한 fragment 계산을 줄인다.

마지막 background blending weight는 $$\int_{t_{\min}}^{t_{\max}}e^{-t^2}dt$$다. 이 구현은 interval 내부의 정확한 visibility 변화 대신, 한 pixel에서 geometry가 보인 가장 이른 시점과 가장 늦은 시점만 사용한다.

### Sampling of the Motion Path

Kernel 하나는 local linear motion을 근사하므로 빠른 non-linear motion에서는 trajectory를 equidistant temporal subframe으로 나눈다. 각 subframe에 six-pass algorithm을 수행하고, 결과를 $$\int_{t_{\mathrm{subframe}_i}}^{t_{\mathrm{subframe}_{i+1}}}e^{-t^2}dt$$로 weighting해 accumulation texture에 합친다.

이는 temporal supersampling과 유사하지만, 각 sample이 단일 2D splat이 아니라 sub-interval의 motion을 포함하는 3D kernel이므로 일반적으로 훨씬 적은 sample을 사용한다. Sample 수를 늘리면 non-linear trajectory와 visibility approximation artifact가 줄지만 rendering cost는 증가한다.

## Experiments

### Rendering Quality

정성 비교는 제안한 GPU 3D-kernel rendering과 conventional EWA surface splatting의 temporal supersampling을 비슷한 visual quality에서 비교한다. Dragon은 depth axis 방향으로 회전하고, colored knot는 up 및 depth axis 주위로 회전한다. Face와 Igea heads는 서로 다른 depth에서 overlap하며, book은 camera 방향으로 이동한다.

![3D-kernel GPU rendering과 2D EWA temporal supersampling의 정성 비교](/assets/img/motion-blur-ewa-surface-splatting/fig7_quality_comparison.png)
_Fig 7. 위 행 (a–d)은 제안한 GPU implementation, 아래 행 (e–h)은 conventional EWA surface splatting의 temporal supersampling 결과다._

Volumetric kernel은 motion vector 길이에 맞춰 자동으로 늘어나므로, object speed가 변할 때 전체 animation을 같은 높은 frequency로 sample할 필요가 없다. 원문은 결과의 세부적인 temporal appearance를 accompanying video에서 확인하도록 안내한다.

### Performance

모든 성능은 NVIDIA GeForce GTX 280에서 측정했으며, Table 1의 `x`는 motion path를 나눈 temporal sample 수다. Supersampling factor는 각 사례에서 제안 방법과 비슷한 visual quality가 되도록 선택했다.

![3D kernel과 temporal supersampling의 성능 비교](/assets/img/motion-blur-ewa-surface-splatting/table1_performance.png){: width="600" }
_Table 1. 비슷한 visual quality에서 3D-kernel 방식과 conventional EWA temporal supersampling의 sample 수 및 throughput 비교._

제안 방법은 사례에 따라 1–12 samples를 사용한 반면 supersampling은 25–80 samples를 사용했다. Fig. 1(a)는 4 samples에서 1.80M points/s로, 30-sample supersampling의 1.44M points/s보다 높다. Fig. 1(b)는 2 samples에서 4.63M points/s이며, supersampling은 25 samples에서 2.91M points/s다.

복잡한 visibility 사례인 Fig. 7(b/f)와 7(c/g)에서는 제안 방법도 12 samples를 사용한다. 처리량은 각각 0.42M과 0.63M points/s로, 40-sample 및 80-sample supersampling의 0.46M과 0.65M points/s와 비슷하다. Book 장면인 Fig. 7(d/h)는 1 sample에서 1.65M points/s를 기록하며, 60-sample supersampling은 0.50M points/s다. 따라서 모든 장면에서 무조건 빠른 것은 아니지만, 유사한 품질에 필요한 temporal sample 수는 일관되게 적다.

### Limitations

이론적 algorithm의 A-Buffer software implementation은 kernel별 visible time과 depth interval을 다루지만, 당시 GPU implementation은 이를 정확히 구현하지 못한다. 시작과 끝의 depth image, 그리고 pixel별 $t_{\min}$과 $t_{\max}$만으로 binary visibility를 근사하기 때문이다. Backface culling도 시작·끝 normal에 의존하므로 kernel을 잘못 버리거나 받아들일 수 있다.

![Inter-object visibility approximation과 sample 수의 영향](/assets/img/motion-blur-ewa-surface-splatting/fig8_visibility_samples.png){: width="520" }
_Fig 8. (a)는 A-Buffer software implementation, (b–d)는 GPU implementation의 1, 4, 8 samples 결과다. Object 사이의 temporal-depth overlap은 sample 수가 낮을 때 부정확하다._

특히 geometry가 exposure 초반과 후반에만 잠깐 보이고 중간에는 사라지는 경우, earliest/latest approximation은 그 사이 전체가 덮였다고 잘못 판단한다. 비슷한 속도로 움직이는 object는 그럴듯하게 blending되지만, 빠른 object와 느린 object가 시간과 depth에서 동시에 overlap하면 artifact가 커진다.

![A-Buffer와 여러 sampling rate에서 나타나는 visibility artifact](/assets/img/motion-blur-ewa-surface-splatting/fig9_visibility_artifacts.png){: width="540" }
_Fig 9. 왼쪽은 A-Buffer와 GPU 3D-kernel 결과, 오른쪽은 2D-kernel supersampling이다. GPU approximation의 artifact는 temporal sample 수를 늘리면 완화되지만 성능이 감소한다._

또한 현재 구현은 frame마다 고정된 sample 수를 사용한다. 논문은 future work로 motion과 shading change에 따라 point별 sampling rate를 adaptive하게 정하는 방법과, 더 정확한 visibility를 위한 general-purpose GPU 또는 reduced A-Buffer hardware를 제안한다.

## Conclusion

Motion Blur for EWA Surface Splatting은 point-sampled dynamic geometry의 motion blur를 단순한 image-space blur나 많은 2D temporal sample의 합으로 처리하지 않고, EWA reconstruction 자체를 시간축으로 확장한다. Planar splat의 두 spatial axis와 motion vector로 ellipsoidal 3D kernel을 구성하고, visible exposure interval에서 viewing-ray integral을 계산한다.

이론적으로는 A-Buffer가 정확한 temporal visibility를 제공하지만, 실시간 GPU 구현은 시작·끝 depth와 earliest/latest visible time을 사용하는 six-pass approximation이다. 그 결과 유사한 visual quality의 conventional EWA supersampling보다 적은 temporal sample로 렌더링할 수 있지만, 서로 다른 속도의 object가 복잡하게 가리는 장면에서는 추가 sampling이 필요하다.
