---
title: "Gaussian Grouping: Segment and Edit Anything in 3D Scenes"
date: 2026-07-21 01:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [Gaussian Splatting, 3D Segmentation, Scene Editing, SAM, Open-World, ECCV]
description: "ECCV 2024"
math: true
toc: true
---

> [[Paper](https://arxiv.org/abs/2312.00732)] [[Github](https://github.com/lkeab/gaussian-grouping)]
>
> Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke
>
> ETH Zürich

## Introduction

3D Gaussian Splatting(3DGS)은 고품질의 real-time novel-view synthesis를 달성했지만, 장면의 appearance와 geometry만 모델링할 뿐 object-level의 fine-grained scene understanding은 다루지 못한다. 즉, 렌더링은 잘하지만 "이 장면 안에 어떤 객체들이 있는가"를 알지 못한다.

![Fig1](/assets/img/gaussian-grouping/fig1_teaser.png)
_Fig 1. Gaussian Grouping은 open-world 3D 장면을 재구성하는 동시에 segment한다. grouping된 Gaussian을 이용하면 object removal, inpainting, style transfer, recomposition 등 다양한 3D editing을 세밀하게 수행할 수 있다._

저자들은 이를 해결하기 위해 Gaussian Splatting을 확장한 Gaussian Grouping을 제안한다. 핵심 기여는 다음과 같다.

1. 각 Gaussian에 compact한 Identity Encoding을 추가하여, Gaussian들이 3D 장면 안의 object instance 또는 stuff 단위로 grouping되도록 한다.
2. 값비싼 3D label 대신, Segment Anything Model(SAM)의 2D mask 예측을 differentiable rendering을 통해 supervision으로 활용하고, 여기에 3D spatial consistency regularization을 더한다.
3. Grouping된 discrete Gaussian을 기반으로 Local Gaussian Editing 기법을 제안하여, object removal·inpainting·colorization·style transfer·scene recomposition 등 다양한 편집을 고품질로 지원한다.

## Related Work

### Radiance Fields and 3D Gaussian Splatting
NeRF 계열은 implicit MLP로 장면을 표현하여 고품질 novel-view synthesis를 달성했고, 3D Gaussian Splatting은 explicit한 Gaussian 집합과 tile-based rasterizer로 real-time 렌더링을 가능하게 했다. 하지만 두 계열 모두 기본적으로 appearance와 geometry에 집중하며, object 단위 이해는 포함하지 않는다.

### 3D Scene Understanding and Editing
LERF나 Distilled Feature Fields(DFF)는 CLIP·DINO 같은 2D feature를 NeRF에 distill하여 open-world semantics를 얻으려 했고, SA3D 등은 SAM mask를 NeRF와 결합했지만 대개 단일 객체에 초점을 맞춘다. 이들 implicit 표현은 편집이 어렵고 느리다. Gaussian Grouping은 discrete Gaussian을 grouping함으로써 segmentation과 editing을 모두 효율적으로 지원한다는 점에서 차별화된다.

## Method

![Fig2](/assets/img/gaussian-grouping/fig2_pipeline.png)
_Fig 2. 전체 파이프라인. (a) 각 view에 대해 SAM을 everything 모드로 적용하여 mask를 독립적으로 생성한다. (b) view마다 다른 mask ID를 일관되게 맞추기 위해, universal temporal propagation(zero-shot tracker)으로 mask label을 연결하여 multi-view에서 일관된 segmentation을 얻는다. (c) 준비된 입력으로 Gaussian의 모든 속성과 group Identity Encoding을 differentiable rendering으로 함께 학습한다. Identity Encoding은 2D Identity Loss와 3D Regularization Loss로 supervise된다._

### 3D Gaussian with Identity Encoding

기본 3DGS에서 각 Gaussian은 centroid $\mathbf{p} \in \mathbb{R}^3$, scale $\mathbf{s} \in \mathbb{R}^3$, rotation quaternion $\mathbf{q} \in \mathbb{R}^4$, opacity $\alpha$, 그리고 색상(SH 계수)으로 표현된다. Gaussian Grouping은 여기에 compact하고 저차원(기본 16차원)인 학습 가능한 Identity Encoding $\mathbf{e}_i$를 추가한다. 이 encoding이 각 Gaussian이 어떤 object에 속하는지를 나타낸다.

### Rendering Identity by Splatting

색상을 splatting하는 것과 동일하게, Identity Encoding도 $\alpha$-blending으로 2D에 렌더링한다. 한 pixel에 기여하는 정렬된 Gaussian들에 대해 identity feature는 다음과 같이 누적된다.

$$
\begin{equation}
E_{id} = \sum_{i} \mathbf{e}_i\, \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
\end{equation}
$$

이렇게 얻은 2D rendered identity feature에 별도의 linear layer를 붙여, 각 pixel 위치에서 어떤 object에 속하는지를 분류한다.

### 2D Identity Loss

Cross-view로 일관되게 정리된 mask label을 정답으로 사용하여, 위 분류 결과에 standard cross-entropy를 적용한다. 이 loss $\mathcal{L}_{2d}$가 rendered identity feature가 올바른 object ID를 가리키도록 학습시킨다.

### 3D Regularization Loss

2D supervision만으로는 object 내부나 크게 가려진(occluded) Gaussian이 충분히 학습되지 않는다. 이를 보완하기 위해, 각 Gaussian의 identity 분포가 3D 상에서 가까운 이웃과 비슷해지도록 하는 unsupervised 3D Regularization Loss를 도입한다. 어떤 Gaussian의 softmax identity 분포 $P$와 그 $K$개의 nearest-neighbor Gaussian 분포 $P_j$ 사이의 KL divergence를 최소화한다.

$$
\begin{equation}
\mathcal{L}_{3d} = \frac{1}{K}\sum_{j=1}^{K} D_{KL}\!\left(P \,\|\, P_j\right)
\end{equation}
$$

이는 spatial consistency를 강제하여, 직접 관측되지 못한 Gaussian도 이웃을 통해 supervise되도록 한다.

### Mask Cross-view Association

SAM은 각 이미지에 대해 독립적으로 mask를 만들기 때문에, 같은 객체라도 view마다 다른 ID가 부여된다. Gaussian Grouping은 이를 하나의 video처럼 다루어, universal temporal propagation 모델(zero-shot tracker)로 view 간 mask를 연결한다. 그 결과 전체 장면에 대해 일관된 object ID 집합을 얻고, 이를 2D Identity Loss의 정답으로 사용한다.

## Local Gaussian Editing

Grouping이 끝나면 각 Gaussian은 group ID를 가지므로, 특정 객체에 해당하는 Gaussian만 선택적으로 조작할 수 있다. 전체 장면 구조를 훼손하지 않고 개별 객체를 다루는 것이 핵심이다.

- Object removal: 선택한 group의 Gaussian을 삭제한다.
- Inpainting: 객체를 제거해 생긴 빈 공간을, 2D inpainting guidance와 함께 새로 추가한 소수의 Gaussian을 국소적으로 최적화하여 자연스럽게 채운다.
- Colorization / Style transfer: 선택한 group의 색상·스타일만 바꾼다.
- Scene recomposition: 특정 객체 group을 복제·이동하거나 다른 장면에 합성한다.

![Fig3](/assets/img/gaussian-grouping/fig3_editing.png)
_Fig 3. Local Gaussian Editing 예시. removal, inpainting, style transfer, recomposition 등에서 편집 대상 객체만 정확히 조작되고 나머지 장면은 그대로 유지된다._

## Experiments

![Table1](/assets/img/gaussian-grouping/table1_lerf_mask.png)
_Table 1. LERF-Mask 벤치마크에서의 open-world 3D segmentation 정량 비교(mIoU·mBIoU). Gaussian Grouping이 LERF, SA3D 등 기존 방법을 능가한다._

저자들은 open-world 3D segmentation 평가를 위해 LERF-Mask annotation을 새로 구축했다. 이 벤치마크에서 Gaussian Grouping은 LERF, SA3D 등 기존 방법을 mIoU·mBIoU 모두에서 큰 폭으로 능가한다. 또한 3DGS의 discrete 표현을 그대로 활용하므로, implicit NeRF 기반 방법 대비 학습·렌더링이 훨씬 빠르고 편집 결과의 시각적 품질과 granularity도 우수하다.

### Ablations

- Mask cross-view association: view 간 mask ID를 일관되게 맞추는 단계가 grouping 정확도에 결정적이다. 이 연결 없이 SAM의 per-view mask를 그대로 쓰면 성능이 크게 떨어진다.
- 3D Regularization Loss: KNN 기반 spatial consistency를 추가하면, 가려진 영역이나 객체 내부의 Gaussian까지 안정적으로 grouping되어 품질이 향상된다.
- Identity Encoding 차원: 지나치게 낮으면 많은 객체를 구분하기 어렵고, 필요 이상으로 높이면 이득 없이 비용만 증가한다. 저차원의 compact한 encoding으로도 충분한 표현력을 얻는다.

## Limitations

- 성능이 SAM과 cross-view association(tracker)의 품질에 의존한다. 2D mask나 view 간 연결이 부정확하면 grouping 오류로 이어질 수 있다(저자들은 association error를 보정하는 방법도 함께 제시한다).
- 매우 복잡하거나 모호한 경계를 가진 객체에서는 여전히 분리가 어려울 수 있다.
- 3DGS를 기반으로 하므로, Gaussian 수 증가에 따른 메모리 부담 등 3DGS의 근본적 한계를 공유한다.

## Conclusion

Gaussian Grouping은 각 Gaussian에 Identity Encoding을 부여하고, SAM의 2D mask와 3D spatial regularization으로 이를 학습시켜, 3D 장면의 reconstruction과 open-world segmentation을 하나의 표현에서 동시에 달성한다. 나아가 grouping된 discrete Gaussian 덕분에 removal·inpainting·style transfer·recomposition 같은 다양한 편집을 효율적이고 정밀하게 수행할 수 있다. 이는 3DGS를 단순한 렌더링을 넘어 object-level의 이해·편집으로 확장한 대표적인 후속 연구다.
