---
title: "[논문리뷰] Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering"
date: 2026-09-16 21:50:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [Light Field Networks, Neural Scene Representation, Neural Rendering, Novel View Synthesis, Plücker Coordinates, Meta-Learning, NeurIPS]
description: "NeurIPS 2021 (Spotlight)"
math: true
toc: true
compact_captions: true
---

> \[[Paper](https://arxiv.org/abs/2106.02634)\] \[[Page](https://www.vincentsitzmann.com/lfns/)\] \[[Github](https://github.com/vsitzmann/light-field-networks)\]
>
> Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, Frédo Durand

## Introduction

2D image로부터 3D scene의 shape과 appearance를 추론하는 문제에서 neural implicit representation은 3D coordinate를 scene의 local property로 변환하고, differentiable renderer는 image supervision만으로 이 representation을 학습할 수 있게 한다. 그러나 3D 공간에 scene을 표현하면 rendering 과정에서 camera ray를 따라 surface를 찾아야 한다. Signed distance 또는 occupancy의 level set을 sphere tracing하거나, ray 위의 여러 지점에서 density와 color를 평가해 volumetric rendering해야 하므로 한 ray에 수십에서 수백 번의 network evaluation이 필요하다.

Light Field Networks(LFN)는 scene을 3D coordinate에 저장하는 대신, oriented camera ray를 그 ray에서 관측되는 radiance로 직접 mapping한다. Static scene과 fixed illumination에서 light field는 빈 공간을 통과하는 빛의 흐름을 나타내므로, rendering은 각 pixel ray에 대해 network를 한 번 평가하는 것으로 끝난다. 저자들은 이를 통해 volumetric method보다 rendering cost를 약 three orders of magnitude 줄이고, 약 400k parameters의 network에 360-degree light field를 저장한다.

![3D-structured neural renderer와 Light Field Network의 비교](/assets/img/light-field-networks/fig1_overview.png)
_Fig 1. 3D-structured representation은 ray 위의 여러 지점을 평가하지만, LFN은 Plücker coordinate로 표현한 ray를 한 번 평가해 color를 출력한다._

Light field는 appearance를 명시적으로 담지만, 같은 3D point를 지나는 ray들의 radiance 변화에는 geometry 정보도 포함된다. 논문은 neural field의 analytical derivative를 이용하여 LFN에서 sparse depth map을 추출한다. 또한 일반적인 4D function은 여러 view 사이의 consistency를 보장하지 않으므로, simple scene들의 분포에서 meta-learning으로 multi-view-consistent light field prior를 학습한다. 이 prior를 사용하면 single image observation만으로 새로운 scene의 360-degree light field를 복원할 수 있다.

논문의 기여는 네 가지로 정리된다.

1. 3D scene의 full 360-degree light field를 neural network로 직접 parameterize하여 single-evaluation rendering과 compact storage를 제공한다.
2. 4D ray space를 6D Plücker coordinates로 표현하여 방향의 singularity나 scene boundary 없이 continuous 360-degree light field를 구성한다.
3. Hypernetwork 기반 meta-learning으로 sparse 2D observation, 특히 single image에서 simple scene의 light field를 복원한다.
4. LFN의 derivative와 Epipolar Plane Image geometry를 이용해 sparse depth map을 계산한다.

논문이 다루는 범위는 single object와 simple room-scale environment 같은 simple scene의 reconstruction이다.

## Related Work

### Neural Scene Representations and Neural Rendering

Voxel grid는 명시적인 3D structure를 제공하지만 spatial resolution이 높아질수록 memory cost가 빠르게 증가한다. Neural field는 MLP가 3D coordinate를 occupancy, signed distance, density, radiance 같은 local property로 mapping하여 continuous scene representation을 만든다. Differentiable rendering을 결합하면 3D supervision 없이 image observation만으로 scene representation을 학습할 수 있다.

Sparse observation에서 reconstruction하기 위해 scene distribution에 대한 prior를 meta-learning하거나, input image의 local feature로 neural field를 condition하는 방법들이 사용되었다. 하지만 sphere tracing과 volumetric rendering은 ray마다 representation을 반복 평가하므로 training과 inference 모두에서 큰 시간과 memory cost를 요구한다. 일부 연구는 test-time rendering을 가속하지만 generalization을 지원하지 않거나 training과 inference에서 반복 evaluation 자체를 제거하지 못한다.

### Light Fields and Their Reconstruction

Light field는 novel-view synthesis와 computational photography에서 오래 사용된 scene representation이다. 4D light field의 2D slice를 추출하면 새로운 view를 직접 rendering할 수 있지만, discretely sampled light field는 storage cost가 크다. 또한 fronto-parallel two-plane 또는 cylindrical parameterization은 full 360-degree light field를 표현하기 어렵고, cubical two-plane 구성은 경계에서 continuous하지 않다.

기존 light-field reconstruction은 Fourier 또는 shearlet domain의 sparsity prior를 사용하거나, CNN으로 sparse view 사이를 in-paint하고 extrapolate했다. 그러나 이러한 방법은 주로 fronto-parallel novel-view synthesis를 대상으로 한다. LFN은 360-degree ray space를 continuous neural field로 parameterize하고, scene class에 대한 prior를 학습해 sparse observation으로부터 scene 단위의 compact representation을 추론한다.

## Method

### Preliminaries

#### 3D-Structured Neural Scene Representations

3D-structured neural scene representation $$\Phi^{3D}$$는 3D coordinate를 그 위치의 scene property $$\mathbf{v}$$로 mapping한다.

$$
\begin{equation}
\Phi^{3D}:\mathbb{R}^{3}\rightarrow\mathbb{R}^{n},
\qquad
\mathbf{x}\mapsto\Phi^{3D}(\mathbf{x})=\mathbf{v}.
\tag{1}
\end{equation}
$$

Differentiable renderer $m$은 ray $\mathbf{r}$과 representation을 받아 관측 color를 계산한다.

$$
\begin{equation}
m(\mathbf{r},\Phi^{3D})=\mathbf{c}(\mathbf{r})\in\mathbb{R}^{3}.
\tag{2}
\end{equation}
$$

Sphere-tracing renderer와 volumetric renderer는 $\mathbf{c}(\mathbf{r})$를 얻기 위해 ray 위에서 $$\Phi^{3D}$$를 수십 또는 수백 번 평가한다. Training에서는 이 반복 계산을 통과해 gradient를 backpropagate해야 하므로 memory와 time complexity가 함께 증가한다.

### The Light Field Network Scene Representation

LFN은 oriented ray의 4D 공간 $\mathcal{L}$을 RGB radiance로 직접 mapping하는 MLP $$\Phi_{\phi}$$다.

$$
\begin{equation}
\Phi_{\phi}:\mathcal{L}\rightarrow\mathbb{R}^{3},
\qquad
\mathbf{r}\mapsto\Phi_{\phi}(\mathbf{r})=\mathbf{c}(\mathbf{r}).
\tag{3}
\end{equation}
$$

Static scene과 fixed illumination에서 light field는 unobstructed space를 지나는 모든 ray의 radiance를 나타낸다. 따라서 pixel ray의 color를 얻기 위해 ray marching이나 alpha compositing을 수행할 필요가 없으며, LFN을 한 번 평가하면 된다.

### Implicit Representations for 360-Degree Light Fields

#### Plücker Ray Parameterization

논문은 모든 oriented ray를 singular direction 없이 균일하게 표현하기 위해 6D Plücker coordinates를 사용한다. Point $\mathbf{p}$를 지나고 normalized direction $\mathbf{d}$를 갖는 ray는 다음과 같다.

$$
\begin{equation}
\mathbf{r}=(\mathbf{d},\mathbf{m})\in\mathbb{R}^{6},
\qquad
\mathbf{m}=\mathbf{p}\times\mathbf{d},
\qquad
\mathbf{d}\in\mathbb{S}^{2},\ \mathbf{p}\in\mathbb{R}^{3}.
\tag{4}
\end{equation}
$$

$\mathbf{m}$은 origin과 ray가 이루는 plane의 normal이며, 그 크기는 ray와 origin 사이의 거리를 나타낸다. Plücker coordinate는 여섯 개의 실수로 쓰이지만 유효한 ray들은 curved 4D subspace $\mathcal{L}$ 위에 놓인다. Neural field는 continuous function이므로 conventional light field처럼 최소 차원의 discrete grid에 제한될 필요가 없다. 이 표현은 fronto-parallel 또는 cylindrical parameterization과 달리 full 360-degree ray를 다루며, two-sphere parameterization처럼 scene이 bounded해야 한다는 조건도 요구하지 않는다.

#### Rendering LFNs

Camera extrinsic을 $$\mathbf{E}=[\mathbf{R}\mid\mathbf{t}]\in SE(3)$$, intrinsic matrix를 $$\mathbf{K}\in\mathbb{R}^{3\times3}$$라 하자. Pixel $(u,v)$의 ray direction과 Plücker coordinate는 다음과 같이 계산한다.

$$
\begin{equation}
\begin{aligned}
\mathbf{d}_{u,v}
&=\mathbf{R}\mathbf{K}^{-1}
\begin{pmatrix}u\\v\\1\end{pmatrix}+\mathbf{t},\\
\mathbf{r}_{u,v}
&=\frac{(\mathbf{d}_{u,v},\mathbf{t}\times\mathbf{d}_{u,v})}
{\lVert\mathbf{d}_{u,v}\rVert}.
\end{aligned}
\tag{5}
\end{equation}
$$

각 pixel color는 $$\mathbf{c}_{u,v}=\Phi(\mathbf{r}_{u,v})$$로 얻는다. 논문은 LFN parameter $$\phi\in\mathbb{R}^{\ell}$$와 camera를 image로 연결하는 rendering function을 다음과 같이 정의한다.

$$
\begin{equation}
\Theta^{\Phi}_{\mathbf{E},\mathbf{K}}:
\mathbb{R}^{\ell}\rightarrow\mathbb{R}^{H\times W\times3}.
\tag{6}
\end{equation}
$$

![LFN의 360-degree light field와 novel-view rendering](/assets/img/light-field-networks/fig3_360_light_field.png)
_Fig 3. LFN에서 얻은 room-scale light-field slice와 object의 local light field, novel RGB views 및 sparse depth maps._

### The Geometry of Light Field Networks

#### Epipolar Plane Images

LFN이 Lambertian scene을 정확히 표현한다고 가정하면, 같은 surface point를 보는 unobstructed rays는 동일한 color를 가진다. 이 성질을 분석하기 위해 ray $\mathbf{r}$ 위의 두 point $$\mathbf{x},\mathbf{x}'$$를 선택하고, ray와 평행하지 않은 unit direction $\mathbf{d}$로 두 parallel line을 만든다.

$$
\mathbf{a}(s)=\mathbf{x}+s\mathbf{d},
\qquad
\mathbf{b}(t)=\mathbf{x}'+t\mathbf{d}.
$$

두 line의 point를 잇는 ray를 LFN에 입력하면 2D light-field slice인 Epipolar Plane Image(EPI)를 얻는다.

$$
\begin{equation}
\begin{aligned}
\mathbf{c}(s,t)&=\Phi\bigl(\mathbf{r}(s,t)\bigr),\\
\mathbf{r}(s,t)
&=\overrightarrow{\mathbf{a}(s)\mathbf{b}(t)}\\
&=\left(
\frac{\mathbf{b}(t)-\mathbf{a}(s)}
{\lVert\mathbf{b}(t)-\mathbf{a}(s)\rVert},
\frac{\mathbf{a}(s)\times\mathbf{b}(t)}
{\lVert\mathbf{b}(t)-\mathbf{a}(s)\rVert}
\right).
\end{aligned}
\tag{7}
\end{equation}
$$

![Light Field Network의 EPI geometry](/assets/img/light-field-networks/fig2_epi_geometry.png)
_Fig 2. 한 ray를 포함하는 2D scene slice와 EPI. 같은 surface point $\mathbf{p}$를 지나는 ray family는 EPI에서 일정한 color를 갖는 line $L_p$를 이룬다._

3D point $\mathbf{p}$를 지나는 ray family는 EPI에서 line $L_p$를 이루고, Lambertian surface에서는 이 line을 따라 color가 일정하다. Object depth에 따라 parallax가 달라지므로 $L_p$의 slope가 point의 3D 위치를 결정한다. Tangent ray에서는 object 뒤쪽이 disocclude되면서 EPI color가 바뀐다.

#### Extracting Depth Maps

두 parallel line 사이의 거리를 $D$라 하면, ray $$\mathbf{r}=\overrightarrow{\mathbf{a}(s)\mathbf{b}(t)}$$를 따라 $$\mathbf{a}(s)$$에서 surface point까지의 depth는 local color gradient로 계산된다.

$$
\begin{equation}
d(\mathbf{r})=
D\frac{\partial_t\mathbf{c}(s,t)}
{\partial_s\mathbf{c}(s,t)+\partial_t\mathbf{c}(s,t)}.
\tag{8}
\end{equation}
$$

Neural implicit representation은 analytically differentiable하므로 finite difference 없이 이 derivative를 얻을 수 있다. 다만 식 (8)은 light-field derivative가 nonzero인 ray에서만 meaningful하다. 실제 구현은 ray 주변의 작은 $(s,t)$ neighborhood를 sample하고 gradient variance가 크면 depth를 invalid로 처리한다. 따라서 결과는 dense depth map이 아니라 confidence가 높은 위치의 sparse depth map이다.

### Meta-Learning with Conditional Light Field Networks

#### Dataset and Hypernetwork

Dataset $\mathcal{D}$는 $N$개의 3D scene으로 구성되며, 각 scene에는 image와 camera extrinsic 및 intrinsic이 $K$개씩 주어진다.

$$
\begin{equation}
S_i=
\left\{(\mathbf{I}_j,\mathbf{E}_j,\mathbf{K}_j)\right\}_{j=1}^{K}
\in
\mathbb{R}^{H\times W\times3}\times SE(3)\times\mathbb{R}^{3\times3},
\qquad i=1,\ldots,N.
\tag{9}
\end{equation}
$$

일반적인 4D function은 같은 3D point를 보는 ray마다 서로 다른 color를 출력할 수 있어 multi-view consistency가 구조적으로 보장되지 않는다. 논문은 natural scene의 multi-view-consistent light field가 이루는 manifold에 prior를 학습하여 이 문제를 완화한다.

각 scene $S_i$에는 latent vector $$\mathbf{z}_i\in\mathbb{R}^{k}$$를 할당하고, hypernetwork $$\Psi_{\psi}$$가 이 latent를 해당 scene의 LFN parameters $$\phi_i\in\mathbb{R}^{\ell}$$로 변환한다.

$$
\begin{equation}
\Psi:\mathbb{R}^{k}\rightarrow\mathbb{R}^{\ell},
\qquad
\Psi_{\psi}(\mathbf{z}_i)=\phi_i.
\tag{10}
\end{equation}
$$

#### Training and Test-Time Reconstruction

Training에서는 모든 scene latent와 hypernetwork parameters를 공동 최적화한다.

$$
\begin{equation}
\underset{\{\mathbf{z}_i\},\psi}{\operatorname{argmin}}
\sum_i\sum_j
\left\lVert
\Theta^{\Phi}_{\mathbf{E}_j,\mathbf{K}_j}
\bigl(\Psi_{\psi}(\mathbf{z}_i)\bigr)-\mathbf{I}_j
\right\rVert_2^2
+\lambda_{\mathrm{lat}}\lVert\mathbf{z}_i\rVert_2^2.
\tag{11}
\end{equation}
$$

첫 번째 항은 rendered image와 observation 사이의 $\ell_2$ error이며, 두 번째 항은 zero-mean diagonal Gaussian latent prior를 적용한다. Test time에는 hypernetwork를 고정하고, 새로운 scene의 single observation에 맞는 latent만 gradient descent로 찾는다.

$$
\begin{equation}
\mathbf{z}_{S}=\underset{\mathbf{z}}{\operatorname{argmin}}
\left\lVert
\Theta^{\Phi}_{\mathbf{E},\mathbf{K}}
\bigl(\Psi_{\psi}(\mathbf{z})\bigr)-\mathbf{I}
\right\rVert_2^2
+\lambda_{\mathrm{lat}}\lVert\mathbf{z}\rVert_2^2.
\tag{12}
\end{equation}
$$

이 방식은 하나의 latent로 scene 전체를 나타내는 global conditioning이다. PixelNeRF처럼 image의 local feature로 coordinate network를 condition하는 방법은 stronger generalization을 보이지만, test-time context가 계속 필요하고 observation 수에 따라 conditioning representation의 크기가 증가한다. LFN은 compact global scene representation을 얻는 대신, 현재 구성에서는 locally conditioned pixelNeRF보다 reconstruction quality가 낮다.

## Experiments

### Implementation Details

모든 실험에서 LFN은 6-layer ReLU MLP, hypernetwork는 3-layer ReLU MLP로 구성하며 두 network 모두 layer normalization을 사용한다. Optimization에는 Adam을 사용하고 learning rate는 $$10^{-4}$$로 설정한다.

### Reconstructing Appearance and Geometry

논문은 ShapeNet cars의 single-object scene과 simple room-scale environment에서 360-degree light field를 학습한다. Cars는 object당 50 observations를 사용하고 room-scale scene은 15 views로 reconstruction한다. LFN은 arbitrary camera perspective의 novel RGB view를 real-time frame rate로 rendering하며, EPI와 식 (8)을 이용해 sparse depth map도 계산한다. Rendering은 network를 ray당 한 번 평가하고, depth extraction은 network와 그 gradient를 평가하지만 ray casting은 수행하지 않는다.

### Multi-Class Single-View Reconstruction

13개 largest ShapeNet categories를 하나의 model로 학습하고, single input view에서 scene latent를 auto-decode하여 novel view를 평가한다. Baseline은 globally conditioned Scene Representation Networks(SRN)와 Differentiable Volumetric Rendering(DVR)이다. DVR은 추가적인 foreground-background mask supervision을 사용한다.

![ShapeNet multi-class single-shot reconstruction 정성 비교](/assets/img/light-field-networks/fig4_multiclass_results.png)
_Fig 4. 13개 ShapeNet class에서 input, DVR, SRN, LFN 및 ground truth를 비교한다._

![ShapeNet 13개 class의 single-shot reconstruction 정량 결과](/assets/img/light-field-networks/table1_multiclass.png)
_Table 1. Multi-class single-view reconstruction의 class별 PSNR과 SSIM. LFN은 평균 PSNR 24.95 dB와 SSIM 0.870을 기록한다._

LFN의 평균 PSNR은 24.95 dB로 DVR의 22.70 dB와 SRN의 23.28 dB보다 높다. 평균 SSIM은 LFN 0.870, DVR 0.860, SRN 0.849다. LFN은 13개 class 중 두 class를 제외한 대부분에서 global-conditioning baselines보다 높은 결과를 보이며, 저자들은 LFN reconstruction이 정성적으로도 더 선명한 경우가 많다고 설명한다.

### Class-Specific Single-View Reconstruction

Cars와 chairs를 각각 학습하는 class-specific setting에서는 LFN이 SRN과 대체로 비슷한 성능을 보인다. Chairs의 PSNR/SSIM은 SRN이 22.89/0.89, LFN이 22.26/0.90이며, cars에서는 SRN이 22.25/0.89, LFN이 22.42/0.89다. 저자들은 multi-class 결과보다 성능이 낮아진 원인을 smaller training set에 따른 multi-view inconsistency로 해석한다.

![Class-specific single-shot reconstruction 비교](/assets/img/light-field-networks/fig5_class_specific.png)
_Fig 5. Chairs와 cars에서 LFN과 SRN의 class-specific single-shot reconstruction 결과._

### Global and Local Conditioning

Locally conditioned pixelNeRF와 비교하면 LFN은 class-specific 평균에서 약 1 dB, multi-class 평균에서 약 2 dB 낮다. Class-specific PSNR/SSIM은 LFN 22.34/0.90, pixelNeRF 23.45/0.91이며, multi-class는 각각 24.95/0.87과 26.80/0.91이다. LFN은 compact global scene representation을 추론하지만 pixelNeRF는 input image의 local feature를 계속 참조하므로 두 방식의 conditioning 목적과 representation 성격이 다르다.

![Global conditioning LFN과 local conditioning pixelNeRF 비교](/assets/img/light-field-networks/fig6_global_local.png)
_Fig 6. LFN은 일부 object에서 pixelNeRF와 비슷하지만 다른 예에서는 object class를 혼동한다._

### Rendering Time and Storage

256×256 image를 NVIDIA RTX 6000 GPU에서 rendering했을 때 LFN은 ray당 network evaluation 1회와 2.1 ms를 기록한다. SRN은 ray당 11회와 120 ms, pixelNeRF는 ray당 192회와 30,000 ms가 필요하다. 논문은 이를 volumetric 및 ray-marching rendering 대비 약 three orders of magnitude의 compute 감소로 정리한다.

![LFN, SRN 및 pixelNeRF의 rendering complexity](/assets/img/light-field-networks/table2_rendering_complexity.png){: width="520" }
_Table 2. 256×256 image의 ray당 network evaluation 수와 rendering time 비교._

Single LFN은 약 400k parameters, 약 1.6 MB의 storage를 사용한다. 반면 six-plane Lumigraph configuration에서 $$256\times256\times17\times17$$ resolution의 360-degree light field는 146 MB가 필요하다.

### Evaluation of Reconstructed Geometry

Class-specific single-shot reconstruction의 각 sample에서 sparse depth map을 추출하고, 네 view의 depth를 backprojection하여 point cloud를 만든다. Valid depth estimate에서 mean $L_1$ error를 SRN과 비교한 결과, LFN은 cars에서 0.059, chairs에서 0.130을 기록하여 SRN의 0.071과 0.20보다 낮다.

![LFN에서 추출한 sparse depth의 3D point cloud reconstruction](/assets/img/light-field-networks/fig7_geometry.png){: width="620" }
_Fig 7. LFN의 sparse depth를 backprojection한 point cloud와 valid depth에서의 mean $L_1$ error._

이 비교는 LFN이 high-confidence라고 판단한 sparse depth만 평가하므로 LFN에 다소 유리하다. 논문은 geometry reconstruction 전용 방법과 경쟁한다고 주장하지 않으며, LFN derivative에서 유효한 depth를 추출할 수 있음을 보이는 실험으로 한정한다.

### Limitations

논문은 세 가지 한계를 명시한다.

1. Light field는 oriented ray마다 하나의 color만 저장하므로, 서로를 가리는 object 사이에 camera를 배치한 view를 rendering하기 어렵다.
2. Globally conditioned baseline보다 좋은 결과를 보이지만 locally conditioned pixelNeRF의 reconstruction quality에는 미치지 못한다.
3. 3D-structured representation과 달리 strict multi-view consistency를 구조적으로 보장하지 않으며, training dataset이 작으면 inconsistency가 나타날 수 있다.

## Conclusion

Light Field Networks는 3D coordinate를 반복 평가하는 대신 full 360-degree 4D light field를 neural implicit representation으로 직접 parameterize한다. Plücker coordinates로 oriented ray를 표현하고 ray당 한 번의 MLP evaluation으로 color를 얻으므로, volumetric 또는 ray-marching method보다 rendering time과 memory consumption을 크게 줄인다.

Meta-learned prior는 simple scene의 single image observation에서 compact LFN을 추론하고, EPI geometry와 analytical derivative는 ray casting 없이 sparse depth를 제공한다. 실험에서는 globally conditioned single-shot novel-view synthesis baseline보다 높은 평균 품질과 real-time rendering을 보였지만, local conditioning과 strict multi-view consistency, occluded space 내부의 camera placement는 해결되지 않은 범위로 남는다.
