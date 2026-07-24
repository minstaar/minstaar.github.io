---
title: "BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting"
date: 2026-07-22 12:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Deblurring, Motion Blur, Bundle Adjustment, Novel View Synthesis, ECCV]
description: "ECCV 2024"
math: true
toc: true
---

> [[Paper](https://arxiv.org/abs/2403.11831)] [[Page](https://lingzhezhao.github.io/BAD-Gaussians/)] [[Github](https://github.com/WU-CVGL/BAD-Gaussians)]
>
> Lingzhe Zhao, Peng Wang, Peidong Liu

## Introduction

Neural rendering은 3D scene reconstruction과 novel-view synthesis에서 뛰어난 성능을 보였지만, 두 가지 전제에 크게 의존한다. (1) high-quality sharp image, (2) 정확한 camera pose(보통 COLMAP으로 계산)이다. 그러나 현실에서는 저조도(low-light)나 긴 노출(long-exposure) 환경에서 motion blur가 흔하게 발생한다. 이런 blurry image는 위 두 전제를 모두 무너뜨린다.

Motion blur가 NeRF와 3D-GS에 던지는 문제는 크게 세 가지다.

- (a) Supervision 문제: NeRF와 3D-GS는 sharp image를 supervision으로 가정하는데, blurry image는 multi-view 간 geometry가 부정확해져 정확한 3D 표현을 얻기 어렵다.
- (b) Pose 문제: 정확한 camera pose가 필수인데, blurry image로부터 COLMAP으로 정확한 pose를 복원하는 것 자체가 어렵다.
- (c) Initialization 문제: 3D-GS는 COLMAP의 sparse point cloud를 Gaussian 초기화에 쓰는데, blurry image에서는 feature matching이 부정확해 point가 적게 생성되어 초기화가 더 나빠진다.

기존에 이 문제를 다룬 implicit NeRF 계열(Deblur-NeRF, DP-NeRF, BAD-NeRF)이 있었지만, implicit MLP 표현의 한계로 (1) 심한 motion blur에서 정교한 detail 복원이 어렵고 (2) real-time rendering이 불가능하다.

저자들은 이를 해결하기 위해 3D-GS 기반 최초의 motion deblur 프레임워크인 BAD-Gaussians (Bundle Adjusted Deblur Gaussian Splatting)를 제안한다. 핵심 아이디어는 motion blur의 물리적 image formation 과정을 3D-GS 학습에 직접 통합하고, 노출 시간 동안의 camera motion trajectory와 Gaussian 파라미터를 joint optimization하는 것이다.

논문의 기여는 다음과 같다.

1. Motion-blurred image를 위한 photometric bundle adjustment formulation을 제안하여, 3D-GS 프레임워크에서 최초로 blurry image로부터 real-time rendering을 달성한다.
2. 이 formulation으로 blurry image 집합에서 high-quality 3D scene representation을 얻을 수 있음을 보인다.
3. 심한 motion blur를 성공적으로 deblur하고, 더 높은 품질의 novel view를 합성하며, real-time rendering까지 달성하여 기존 implicit deblur 방법을 능가한다.

## Related Work

### NeRF for Camera Optimization
BARF는 coarse-to-fine bundle adjustment 전략으로 camera pose와 NeRF를 동시에 최적화한 선구적 연구이며, 이후 SC-NeRF, CamP 등이 camera model과 파라미터 최적화를 개선했다. 다만 이들은 sharp image의 pose 최적화에 초점을 둔다. BAD-Gaussians는 여기서 더 나아가 각 blurry image의 노출 시간 내 trajectory 전체를 복원한다.

### NeRF for Deblurring
Deblur-NeRF는 canonical kernel을 공간 위치별로 변형하는 deformable sparse kernel로 blur 과정을 시뮬레이션한다. DP-NeRF는 motion-blur 획득 과정의 physical prior를 Deblur-NeRF에 통합한다. BAD-NeRF는 motion blur의 물리 과정을 모델링하고 노출 시간 내 camera trajectory와 radiance field를 joint optimization하며, BAD-Gaussians의 직접적인 NeRF 버전 선행연구다.

이들 공통 한계는 implicit MLP 구조 때문에 real-time rendering이 어렵고 정교한 detail 복원이 힘들다는 점이다. 특히 Deblur-NeRF와 DP-NeRF는 학습 중 부정확한 pose를 고정해버려 성능이 떨어진다.

### Image Deblurring
전통적 2D deblurring은 (1) blur kernel과 latent sharp image를 함께 추정하는 optimization 방식과, (2) deep learning 기반 end-to-end 학습(SRN, MPR 등)으로 나뉜다. 그러나 이런 2D 방법들은 multi-view 간 3D scene geometry를 활용하지 못해 시점 간 일관성(view consistency)을 보장하지 못한다.

## Method

![Fig1](/assets/img/badgs/fig1_pipeline.png)
_Fig 1. BAD-Gaussians의 pipeline. COLMAP에서 얻은 부정확한 pose와 sparse point cloud로 Gaussian을 초기화한다. Forward projection과 differentiable Gaussian rasterization으로 노출 시간 내 여러 virtual sharp image를 렌더링하고, 이를 평균(averaging)하여 blurry image를 합성한다. Virtual camera pose는 SE(3) 공간의 continuous trajectory로 보간되며, 합성 blurry image와 실제 blurry image 사이의 photometric loss를 최소화하여 Gaussian과 camera trajectory를 joint optimization한다._

BAD-Gaussians의 목표는 COLMAP에서 얻은 부정확한 pose·sparse point cloud와 motion-blurred image 시퀀스가 주어졌을 때, camera motion trajectory와 Gaussian 파라미터를 함께 학습하여 sharp 3D scene을 복원하는 것이다.

### Preliminary: 3D Gaussian Splatting

각 Gaussian $G$는 mean position $\mu \in \mathbb{R}^3$, 3D covariance $\Sigma \in \mathbb{R}^{3\times3}$, opacity $o \in \mathbb{R}$, color $c \in \mathbb{R}^3$로 정의된다.

$$
\begin{equation}
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}
\end{equation}
$$

Covariance $\Sigma$가 positive semi-definite를 유지하도록, scale $S$와 rotation matrix $R$(quaternion $q$로 저장)로 분해한다.

$$
\begin{equation}
\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^{T}\mathbf{R}^{T}
\end{equation}
$$

주어진 camera pose $\mathbf{T}_c = \{\mathbf{R}_c, \mathbf{t}_c\}$에서 3D Gaussian을 2D로 projection한 covariance는 다음과 같다.

$$
\begin{equation}
\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{R}_c\boldsymbol{\Sigma}\mathbf{R}_c^T\mathbf{J}^{T}
\end{equation}
$$

여기서 $\mathbf{J}$는 projective transformation의 affine approximation에 대한 Jacobian이다. 각 pixel color는 depth 정렬된 $N$개의 2D Gaussian을 $\alpha$-blending하여 렌더링된다.

$$
\begin{equation}
\mathbf{C} = \sum_{i}^{N} \mathbf{c}_i \alpha_i \prod_{j}^{i-1}(1-\alpha_j)
\end{equation}
$$

$$
\begin{equation}
\alpha_i = o_i \cdot \exp(-\sigma_i), \quad \sigma_i = \frac{1}{2}\Delta_i^T \boldsymbol{\Sigma}'^{-1} \Delta_i
\end{equation}
$$

여기서 핵심은, 렌더링된 pixel color $\mathbf{C}$가 모든 learnable Gaussian $G$와 camera pose $\mathbf{T}_c$에 대해 미분 가능하다는 점이다. 이것이 blurry image와 부정확한 pose를 다루는 bundle adjustment의 토대가 된다.

### Physical Motion Blur Image Formation Model

디지털 카메라의 image formation은 노출 시간 동안 광자를 모아 전하로 변환하는 과정이며, 수학적으로는 노출 시간 동안의 virtual latent sharp image들을 적분하는 것으로 표현된다.

$$
\begin{equation}
\mathbf{B}(\mathbf{u}) = \phi \int_{0}^{\tau} \mathbf{C}_{\mathrm{t}}(\mathbf{u}) \, \mathrm{dt}
\end{equation}
$$

여기서 $\mathbf{B}(\mathbf{u})$는 실제 촬영된 blurry image, $\mathbf{u}$는 pixel 위치, $\phi$는 normalization factor, $\tau$는 노출 시간, $\mathbf{C}_t(\mathbf{u})$는 시각 $t$에 촬영된 virtual latent sharp image다. 이를 $n$개의 discrete sample로 근사하면 다음과 같다.

$$
\begin{equation}
\mathbf{B}(\mathbf{u}) \approx \frac{1}{n} \sum_{i=0}^{n-1} \mathbf{C}_i(\mathbf{u})
\end{equation}
$$

즉 blurry image는 노출 시간 동안의 virtual sharp image들을 평균한 것이다. 카메라가 빠르게 움직이거나 노출이 짧으면 blur가 적고, 느리게 움직이거나 노출이 길면(저조도) blur가 심해진다. 또한 $\mathbf{B}(\mathbf{u})$는 각 virtual sharp image $\mathbf{C}_i(\mathbf{u})$에 대해 미분 가능하다.

### Camera Motion Trajectory Modeling

식 (7)을 풀려면 각 virtual sharp image $\mathbf{C}_i$가 필요하고, $\mathbf{C}_i$는 특정 pose $\mathbf{T}_i$에서 렌더링할 수 있다. 노출 시간이 짧다고 가정하면, 노출 시작 pose $\mathbf{T}_{\text{start}}$와 종료 pose $\mathbf{T}_{\text{end}}$ 두 개만으로 trajectory를 표현하고 그 사이를 보간할 수 있다. 시각 $t$($0 \le t \le \tau$)의 virtual camera pose는 SE(3) 공간에서 linear interpolation으로 표현된다.

$$
\begin{equation}
\mathbf{T}_t = \mathbf{T}_{\text{start}} \cdot \exp\left(\frac{t}{\tau} \cdot \log(\mathbf{T}_{\text{start}}^{-1} \cdot \mathbf{T}_{\text{end}})\right)
\end{equation}
$$

$\frac{t}{\tau}$는 $\frac{i}{n-1}$로 discretize된다($i$번째 virtual sharp image). $\mathbf{T}_i$는 $\mathbf{T}_{\text{start}}$와 $\mathbf{T}_{\text{end}}$ 모두에 대해 미분 가능하다. BAD-Gaussians의 목표는 각 프레임의 $\mathbf{T}_{\text{start}}$, $\mathbf{T}_{\text{end}}$와 Gaussian 파라미터 $G_\theta$를 함께 추정하는 것이다.

### Loss Function

$K$개의 blurry image로부터 Gaussian 파라미터 $\theta$($\mu, \Sigma, o, c$)와 각 이미지의 camera trajectory($\mathbf{T}_{\text{start}}, \mathbf{T}_{\text{end}}$)를 함께 추정한다. 합성 blurry image $\mathbf{B}_k$와 실제 blurry image $\mathbf{B}_k^{gt}$ 사이의 L1 loss와 D-SSIM term을 최소화한다.

$$
\begin{equation}
\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}
\end{equation}
$$

Gaussian 파라미터 $\theta$와 pose $\mathbf{T}$의 gradient는 다음과 같이 chain rule로 흐른다.

$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{k=0}^{K-1} \frac{\partial \mathcal{L}}{\partial \mathbf{B}_k} \cdot \frac{1}{n}\sum_{i=0}^{n-1} \frac{\partial \mathbf{B}_k}{\partial \mathbf{C}_i} \frac{\partial \mathbf{C}_i}{\partial \theta}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \mathbf{T}} = \sum_{k=0}^{K-1} \frac{\partial \mathcal{L}}{\partial \mathbf{B}_k} \cdot \frac{1}{n}\sum_{i=0}^{n-1} \frac{\partial \mathbf{B}_k}{\partial \mathbf{C}_i} \frac{\partial \mathbf{C}_i}{\partial \theta} \frac{\partial \theta}{\partial \mathbf{T}}
\end{equation}
$$

$\mathbf{T}_{\text{start}}$와 $\mathbf{T}_{\text{end}}$는 각각 SE(3)의 Lie algebra(6D vector)로 파라미터화된다. Gaussian의 pose에 대한 Jacobian $\frac{\partial \theta}{\partial \mathbf{T}}$에서, color $c$와 opacity $o$는 pose와 독립이고 효율성을 위해 $\frac{\partial \Sigma'}{\partial \mathbf{T}_i}$는 무시(GS-SLAM 방식)하여, 사실상 mean position $\mu$의 pose에 대한 Jacobian만 해석적으로 계산한다.

## Experiments

### Experimental Settings
- PyTorch로 3D-GS 프레임워크 위에 구현. Gaussian과 camera pose 모두 Adam으로 최적화.
- Camera pose learning rate는 $1\times10^{-3} \to 1\times10^{-5}$로 exponential decay.
- Virtual camera pose 개수 $n = 10$ (성능-효율 trade-off).
- COLMAP 추정값으로 초기화, NVIDIA RTX 4090에서 실험.
- Dataset: Deblur-NeRF(synthetic + real), MBA-VO(등속이 아닌 가속 포함 실제 trajectory 기반). Metric: PSNR, SSIM, LPIPS, 그리고 pose 정확도는 ATE(Absolute Trajectory Error).

### Ablation Study

Virtual camera pose 개수 $n$: $n$을 3에서 20으로 늘리며 실험한 결과, 심한 blur(Tanabata)에서는 $n$이 커질수록 성능이 오르지만 약한 blur(Cozy2room)에서는 개선이 미미하다. 성능과 효율의 균형점으로 $n=10$을 선택했다.

Trajectory 표현: linear interpolation과 cubic B-spline(4개 control knot)을 비교했다. Synthetic 장면에서는 둘이 비슷하지만, 실제 장면에서는 노출 시간이 길어 cubic B-spline이 더 우수하다. 따라서 synthetic에는 linear, real data에는 cubic B-spline을 사용한다.

![Table1](/assets/img/badgs/table1_ablation.png)
_Table 1·2. Virtual camera pose 개수 $n$(Table 1)과 trajectory 표현 방식(Table 2)에 대한 ablation. $n=10$, 그리고 synthetic엔 linear·real엔 cubic B-spline이 균형점이다._

### Results

![Fig2](/assets/img/badgs/fig2_synthetic.png)
_Fig 2. Synthetic 데이터셋에서 방법별 novel view synthesis 정성 비교. Ground-truth pose로 학습한 Deblur-NeRF*, DP-NeRF*보다도, 부정확한 pose로 학습한 BAD-Gaussians가 더 높은 품질로 장면을 복원한다._

![Table3](/assets/img/badgs/table3_synthetic.png)
_Table 3·4. Synthetic Deblur-NeRF 데이터셋에서의 deblurring(Table 3)·novel view synthesis(Table 4) 정량 비교(PSNR/SSIM/LPIPS)._

Synthetic 데이터셋(Deblur-NeRF)에서 BAD-Gaussians는 deblurring과 novel view synthesis 모두에서 이전 SOTA 대비 평균 PSNR 3.6dB / 1.7dB 향상을 보인다. NeRF와 3D-GS는 blurry image에서 성능이 크게 떨어지고, 단일 이미지 방법(MPR, SRN)은 multi-view geometry를 활용하지 못해 뒤처진다. Deblur-NeRF와 DP-NeRF는 부정확한 pose를 고정하는 탓에 성능이 낮으며, GT pose로 학습(*)하면 향상되는 것이 오히려 pose 정확도 민감성을 방증한다.

렌더링 효율 면에서 BAD-Gaussians는 200 FPS 이상을 달성하는 반면 Deblur-NeRF·DP-NeRF·BAD-NeRF는 1 FPS 미만이다. 학습 시간도 약 30분으로, 10시간 이상 걸리는 다른 방법보다 훨씬 빠르다.

한 가지 예외로 Factory 장면에서는 BAD-NeRF보다 낮은데, 이는 하늘(sky)을 Gaussian splatting으로 표현하는 능력이 NeRF보다 떨어지기 때문이다. 그럼에도 평균적으로는 모든 이전 방법을 큰 차이로 앞선다.

MBA-VO 데이터셋(등속이 아닌 가속 포함 trajectory)에서도 BAD-Gaussians가 최고 성능을 달성하여, 실제와 가까운 복잡한 카메라 움직임에도 강건함을 보인다.

![Fig3](/assets/img/badgs/fig3_real.png)
_Fig 3. 실제(real) 데이터셋에서 방법별 novel view synthesis 정성 비교. BAD-Gaussians가 실제 데이터에서도 정교한 detail을 복원한다. 반면 BAD-NeRF는 실제 데이터에서 성능이 떨어지고 synthetic에서만 만족스럽다._

실제 데이터에서도 정량(Table 6)·정성(Fig 3) 모두 BAD-Gaussians가 우수하다. Deblur-NeRF와 DP-NeRF는 blur kernel을 렌더링 이미지에 convolution하는 방식이라 depth 변화가 큰 영역을 잘 모델링하지 못한다.

Pose Estimation(Table 7): ATE 기준으로, blurry image에 COLMAP을 직접 쓴 경우(COLMAP-blur)보다 BAD-Gaussians가 훨씬 정확한 pose를 복원한다. 즉 blur를 제대로 모델링하는 것이 pose 복원 정확도로 이어진다.

![Table7](/assets/img/badgs/table7_pose.png)
_Table 7. Pose estimation 정확도 비교(ATE). COLMAP·BAD-NeRF 대비 BAD-Gaussians가 더 정확한 trajectory를 복원한다._

## Conclusion

BAD-Gaussians는 부정확한 pose를 가진 motion-blurred image 집합으로부터 Gaussian splatting을 학습하는 최초의 파이프라인이다. 3D scene representation과 camera motion trajectory를 joint optimization하며, 물리적 blur formation model을 explicit Gaussian 표현에 통합하여 high-quality novel view와 real-time rendering을 동시에 달성했다. Implicit NeRF 기반 deblur 방법들의 속도·품질 한계를 explicit representation으로 극복한 점이 핵심 기여다.
