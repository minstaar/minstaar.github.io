---
title: "[논문리뷰] 3D Gaussian Splatting for Real-Time Radiance Field Rendering"
date: 2026-07-20 13:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Radiance Field, Novel View Synthesis, Neural Rendering, SIGGRAPH]
description: "3D Gaussian Splatting 논문 리뷰 (SIGGRAPH 2023)"
math: true
toc: true
---

> SIGGRAPH 2023. [[Paper](https://arxiv.org/abs/2308.04079)] [[Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)] [[Github](https://github.com/graphdeco-inria/gaussian-splatting)]
>
> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
>
> Inria, Université Côte d'Azur | Max-Planck-Institut für Informatik
>
> 8 Aug 2023

## Introduction

Radiance field 기반 방법들은 여러 장의 사진이나 영상으로 촬영한 장면의 **novel-view synthesis**를 크게 발전시켰다. 하지만 높은 visual quality를 얻으려면 학습과 렌더링 비용이 큰 neural network가 필요하고, 최근의 빠른 방법들은 speed를 위해 quality를 희생하는 경향이 있다. 특히 unbounded하고 완전한 장면(고립된 물체가 아닌)에 대해 1080p 해상도로 렌더링할 때, 실시간(real-time) display rate을 달성하는 방법은 아직 없었다.

![Fig1](/assets/img/3dgs/fig1_teaser.png)
_Fig 1. 본 방법은 이전 최고 품질 방법(Mip-NeRF360)과 동등한 quality를 유지하면서도, 가장 빠른 방법들과 경쟁 가능한 학습 시간만 요구한다. 핵심은 새로운 3D Gaussian scene representation과 real-time differentiable renderer이다._

저자들은 state-of-the-art visual quality를 유지하면서 경쟁력 있는 학습 시간을 확보하고, 무엇보다 1080p 해상도에서 고품질 real-time($\geq$ 30fps) novel-view synthesis를 가능하게 하는 세 가지 핵심 요소를 제안한다.

1. **3D Gaussian**을 flexible하고 expressive한 scene representation으로 도입한다. Structure-from-Motion(SfM)으로 얻은 sparse point에서 시작하며, 대부분의 point-based 방법과 달리 continuous volumetric radiance field의 좋은 성질을 보존하면서도 불필요한 빈 공간(empty space) 계산을 피한다.
2. 3D Gaussian에 대한 **optimization**과 **adaptive density control**을 번갈아 수행한다. 특히 anisotropic covariance를 최적화하여 장면을 정확하게 표현한다.
3. Visibility-aware **rendering** 알고리즘을 개발하여 anisotropic splatting을 지원하고, 학습을 가속하는 동시에 real-time rendering을 가능하게 한다.

## Related Work

### Traditional Scene Reconstruction and Rendering
초기 novel-view synthesis는 light field에 기반했으며, 이후 Structure-from-Motion(SfM)과 Multi-View Stereo(MVS)의 발전으로 여러 장의 사진에서 3D 장면을 재구성할 수 있게 되었다. 이러한 MVS 기반 geometry는 재투영(reproject)과 blending을 통해 새로운 시점을 합성하는 데 쓰였지만, 재구성되지 않은 영역이나 over-reconstruction에서 발생하는 artifact를 완전히 극복하지는 못했다.

### Neural Rendering and Radiance Fields
**Neural Radiance Fields(NeRF)**는 MLP로 장면을 표현하고 volumetric ray-marching으로 렌더링하여 높은 품질을 달성했다. 하지만 원래 NeRF는 학습과 렌더링이 매우 느리다. 이후 **Mip-NeRF360**은 quality 측면에서 SOTA를 달성했으나 학습에 최대 48시간이 걸린다. 반면 **InstantNGP**는 hash grid를, **Plenoxels**는 sparse voxel grid를 사용하여 학습을 크게 가속했지만, unbounded 장면의 최고 품질에는 미치지 못하거나 real-time rendering이 어렵다.

### Point-Based Rendering and Radiance Fields
Point-based 방법은 GPU에서 효율적으로 렌더링되지만, point sampling에 따른 hole이나 aliasing 문제를 겪는다. **Differentiable point-based rendering**은 이러한 point를 최적화 가능한 primitive로 다루며, 본 논문의 3D Gaussian은 이 계열을 발전시킨 것이다. 3D Gaussian은 미분 가능하고 rasterize하기 쉬우면서도, continuous representation의 장점을 유지한다.

## Overview

![Fig2](/assets/img/3dgs/fig2_pipeline.png)
_Fig 2. Optimization은 sparse SfM point cloud에서 시작하여 3D Gaussian 집합을 생성한다. 이후 이 Gaussian 집합의 density를 최적화하고 adaptive하게 제어한다. Optimization 중에는 fast tile-based renderer를 사용하여 SOTA 대비 경쟁력 있는 학습 시간을 확보한다._

전체 파이프라인은 다음과 같다. SfM으로 얻은 sparse point cloud로 3D Gaussian들을 초기화한 뒤, 각 Gaussian의 위치·covariance·opacity·색상(SH 계수)을 최적화한다. Optimization 과정에서 differentiable tile-based rasterizer로 이미지를 렌더링하고, ground-truth view와 비교한 loss로 gradient를 역전파한다. 동시에 **adaptive density control**로 Gaussian을 추가(densify)하거나 제거(prune)하여 장면을 효율적으로 채운다.

## Differentiable 3D Gaussian Splatting

목표는 normal(법선)이 없는 sparse SfM point 집합으로부터, 고품질 novel-view synthesis가 가능한 장면 표현을 최적화하는 것이다. 이를 위해 미분 가능하고 2D splat으로 쉽게 projection할 수 있는 **3D Gaussian**을 primitive로 선택한다.

3D Gaussian은 world space에서 정의된 full 3D covariance matrix $\Sigma$와 중심점(mean) $\mu$로 정의된다.

$$
\begin{equation}
G(x) = e^{-\frac{1}{2}(x)^T \Sigma^{-1} (x)}
\end{equation}
$$

이 Gaussian을 렌더링하려면 3D에서 2D로 projection해야 한다. 시점 변환(viewing transformation) $W$가 주어졌을 때, camera coordinate에서의 covariance matrix $\Sigma'$는 다음과 같다.

$$
\begin{equation}
\Sigma' = J W \Sigma W^T J^T
\end{equation}
$$

여기서 $J$는 projective transformation의 affine approximation에 대한 Jacobian이다.

**핵심 문제**는 covariance matrix $\Sigma$를 직접 최적화하면, physically meaningful하려면 positive semi-definite해야 하는데 gradient descent로는 이 제약을 쉽게 보장할 수 없다는 점이다. 따라서 저자들은 $\Sigma$를 직접 다루는 대신, 타원체(ellipsoid)의 configuration을 나타내는 더 직관적이면서 동등하게 표현력 있는 형태로 분해한다. 즉, scaling matrix $S$와 rotation matrix $R$을 사용하여 다음과 같이 covariance를 구성한다.

$$
\begin{equation}
\Sigma = R S S^T R^T
\end{equation}
$$

이렇게 하면 scaling은 3D vector $s$로, rotation은 quaternion $q$로 독립적으로 저장하고 최적화할 수 있어 항상 유효한 covariance를 유지한다. 실제로는 학습 중에 이러한 요소들의 gradient를 명시적으로 유도하여 autodiff의 오버헤드를 피한다.

![Fig3](/assets/img/3dgs/fig3_anisotropic.png)
_Fig 3. Optimization 이후 3D Gaussian을 60% 축소(shrinking)하여 시각화한 것(far right). 최적화된 3D Gaussian들이 복잡한 geometry를 anisotropic한 형태로 compact하게 표현함을 보여준다._

## Optimization with Adaptive Density Control of 3D Gaussians

### Optimization

Optimization은 렌더링 결과를 학습 뷰(training view)와 비교하는 과정을 반복하며 진행된다. 3D를 2D로 projection하는 과정의 모호함 때문에 geometry가 잘못 배치될 수 있으므로, geometry를 생성·제거·이동할 수 있어야 한다.

Optimization은 SGD 기반이며, 일부 연산은 custom CUDA kernel로 구현했다. Opacity $\alpha$에는 sigmoid activation을, covariance의 scale에는 exponential activation을 사용한다. 초기 covariance는 가장 가까운 세 point까지의 평균 거리로부터 isotropic Gaussian으로 추정한다.

Loss는 **L1** loss와 **D-SSIM** term을 결합한 형태이다.

$$
\begin{equation}
\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
\end{equation}
$$

모든 실험에서 $\lambda = 0.2$를 사용한다.

### Adaptive Control of Gaussians

![Fig4](/assets/img/3dgs/fig4_densification.png)
_Fig 4. Adaptive Gaussian densification 방식. **위 행(under-reconstruction)**: small-scale geometry가 충분히 표현되지 않은 경우 해당 Gaussian을 **clone**한다. **아래 행(over-reconstruction)**: 하나의 Gaussian이 넓은 영역을 덮는 경우 이를 **split**한다._

SfM에서 얻은 초기 sparse point 집합만으로는 고품질 radiance field를 표현하기에 부족하다. 따라서 저자들은 100 iteration마다 Gaussian을 densify하고, 거의 투명한(opacity가 임계값 $\epsilon_\alpha$ 미만) Gaussian을 제거(prune)한다.

Densification은 view-space positional gradient의 크기가 임계값 $\tau_{\text{pos}}$(= 0.0002)를 넘는 Gaussian을 대상으로 한다. 이는 **under-reconstruction**과 **over-reconstruction** 두 경우를 모두 다룬다.

- **Under-reconstruction (missing geometry)**: 작은 Gaussian이 있는 영역에서는 같은 크기의 Gaussian을 **clone**하여 positional gradient 방향으로 이동시킨다. 이는 전체 volume과 Gaussian 수를 늘린다.
- **Over-reconstruction (하나의 Gaussian이 넓은 영역을 덮는 경우)**: 큰 Gaussian을 두 개의 작은 Gaussian으로 **split**한다. 이때 scale을 실험적으로 정한 factor $\phi = 1.6$으로 나눈다.

또한 floater(입력 카메라 근처에 떠 있는 불필요한 Gaussian)가 과도하게 증식하는 것을 막기 위해, 3000 iteration마다 opacity $\alpha$를 0에 가깝게 **reset**한다. 이후 optimization이 필요한 Gaussian의 $\alpha$를 다시 키우고, 그렇지 않은 것은 prune된다.

## Fast Differentiable Rasterizer for Gaussians

빠른 렌더링과 sorting을 위해, 임의 개수의 blended Gaussian에 대해 gradient를 계산할 수 있는 **tile-based rasterizer**를 설계했다. 이는 pixel당 정렬 비용을 피하기 위해 전체 화면을 **16×16 tile**로 나눈다.

렌더링 과정은 다음과 같다.

1. 각 3D Gaussian을 view frustum에 대해 **culling**한다. 99% confidence interval 밖에 있는 Gaussian을 제거하고, 가까운 near plane이나 view frustum 밖의 Gaussian도 reject한다.
2. 각 Gaussian을 겹치는 tile 개수만큼 instantiate하고, view space depth와 tile ID를 결합한 key를 부여한다.
3. GPU Radix sort로 Gaussian을 한 번에 정렬한다(pixel당 정렬이 아님). 이 정렬 순서를 그대로 사용하여 $\alpha$-blending을 수행한다.
4. 각 tile에 대해 정렬된 Gaussian을 front-to-back으로 누적하며 색상과 $\alpha$를 blending한다. 한 tile의 target saturation($\alpha = 1$)에 도달하면 해당 thread를 종료한다.

Backward pass에서는 forward에서 정렬한 리스트를 다시 활용하여, 어떤 Gaussian이 각 pixel에 기여했는지 back-to-front로 순회하며 gradient를 계산한다. 이로써 임의 깊이(depth)의 blending에 대해서도 효율적인 미분이 가능하다.

## Implementation, Results and Evaluation

### Implementation Details
- Optimization은 GPU에서 수행되며 일부는 custom CUDA kernel로 구현했다.
- 저해상도에서 시작하는 **warm-up**을 사용한다. 초기에는 이미지 해상도를 1/4로 낮추어 최적화를 시작하고, 250 및 500 iteration에서 두 번 upsampling하여 최종 해상도로 올린다.
- Spherical Harmonics(SH) 계수는 view-dependent한 색상을 표현한다. 1000 iteration마다 SH의 band를 하나씩 늘려가며 최대 degree까지 확장한다. 이렇게 하면 angular한 정보가 점진적으로 추가되어 학습이 안정된다.

### Results

![Fig5](/assets/img/3dgs/fig5_comparison.png)
_Fig 5. 이전 방법들 및 ground truth와의 비교. 위에서부터 Mip-NeRF360 데이터셋의 Bicycle, Garden, Stump, Counter, Room, Deep Blending 데이터셋의 Playroom, DrJohnson, Tanks&Temples의 Truck과 Train. 미세한 품질 차이는 화살표/inset으로 표시._

Mip-NeRF360, Tanks&Temples, Deep Blending 데이터셋에서 실험한 결과, 본 방법은 **Mip-NeRF360**과 동등하거나 더 나은 quality를 달성하면서, 학습 시간은 InstantNGP나 Plenoxels 수준으로 빠르다. 특히 real-time(≥30fps, 최대 134fps) 렌더링을 1080p에서 달성한 것은 이전 어떤 방법도 하지 못한 것이다.

주요 수치로, Mip-NeRF360 데이터셋에서 **Ours-30K**는 SSIM 0.815, PSNR 27.21을 기록하며 평균 134fps로 렌더링한다.

![Fig6](/assets/img/3dgs/fig6_iterations.png)
_Fig 6. 일부 장면(위)에서는 7K iteration(약 5분)만으로도 train 장면을 잘 포착한다. 30K iteration(약 35분)에서는 background artifact가 크게 줄어든다. 다른 장면(아래)에서는 그 차이가 거의 보이지 않으며, 7K iteration(약 8분)만으로도 이미 매우 높은 품질을 보인다._

### Ablations

저자들은 각 설계 요소의 기여를 검증하기 위해 ablation을 수행했다.

- **SfM 초기화 vs random 초기화**: SfM point로 초기화하면 학습이 더 안정적이고 최종 품질도 향상된다. 다만 Synthetic NeRF처럼 배경이 단순한 경우 100K random point로 초기화해도 좋은 품질을 얻을 수 있다.
- **Densification (clone & split)**: clone과 split을 모두 사용하는 것이 under/over-reconstruction을 각각 잘 처리하여 품질을 높인다.
- **Anisotropic covariance**: isotropic Gaussian으로 제한하면 복잡한 geometry(예: 얇은 구조)를 표현하기 어렵다. Anisotropic covariance가 표현력을 크게 높인다.
- **Unlimited depth / gradient**: backward pass에서 blending에 기여하는 Gaussian 수를 제한하지 않아야(즉 unlimited) 품질 저하가 없다.

### Limitations
- 잘 관측되지 않은(sparsely observed) 영역에서는 artifact가 남을 수 있다.
- Gaussian이 지나치게 커지는 경우(특히 view-dependent appearance가 있을 때) popping artifact가 생길 수 있다.
- 다른 방법 대비 메모리 사용량이 크다. 완전한 장면을 표현하려면 수백만 개의 Gaussian이 필요할 수 있다.

## Conclusion

본 논문은 **3D Gaussian**을 명시적(explicit) scene representation으로 사용하고, **adaptive density control**과 **tile-based differentiable rasterizer**를 결합하여, radiance field의 품질을 유지하면서도 real-time rendering을 최초로 달성했다. 이는 이후 수많은 후속 연구(dynamic scene, compression, editing 등)의 기반이 된 중요한 연구이다.
