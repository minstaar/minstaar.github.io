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

## Introduction

3D Gaussian Splatting(3DGS)은 고품질의 real-time novel-view synthesis를 달성했지만, appearance와 geometry만 모델링할 뿐 장면을 object 단위로 이해하지는 못한다. 저자들은 3DGS를 확장하여, 재구성과 동시에 open-world 3D 장면을 segment하는 Gaussian Grouping을 제안한다.

핵심 아이디어는, 각 Gaussian에 색상(SH)과 유사한 형태의 Identity Encoding을 추가하여 Gaussian들이 object instance 또는 stuff 단위로 grouping되도록 하는 것이다. 값비싼 3D label 대신 SAM의 2D mask를 supervision으로 쓰되, 이를 differentiable rendering과 3D regularization으로 3D에 일관되게 lift한다.

![Fig1](/assets/img/gaussian-grouping/fig1_teaser.png)
_Fig 1. Gaussian Grouping은 open-world 3D 장면을 재구성(a)하는 동시에 segment(b)한다. grouping된 discrete Gaussian을 이용하면 object를 쉽게 제거·편집할 수 있다._

기여를 정리하면 다음과 같다.

1. 3D Gaussian에 학습 가능한 Identity Encoding을 추가하여, 장면을 appearance·geometry와 함께 instance/stuff identity까지 표현한다.
2. 학습 시 cost-based linear assignment 대신 zero-shot tracker로 multi-view mask를 일관되게 연결하여, 값싸고 견고한 supervision을 얻는다.
3. Grouping된 discrete Gaussian을 기반으로 Local Gaussian Editing을 제안하여 removal·inpainting·colorization·style transfer·recomposition을 지원한다.

## Preliminaries: 3D Gaussian Splatting

3DGS는 장면을 다수의 3D Gaussian 집합으로 표현한다. 각 Gaussian은 centroid $\mathbf{p} \in \mathbb{R}^3$, 표준편차 형태의 3D size $\mathbf{s} \in \mathbb{R}^3$, rotation quaternion $\mathbf{q} \in \mathbb{R}^4$, opacity $\alpha \in \mathbb{R}$, 그리고 SH로 표현되는 색상 $\mathbf{c}$로 정의된다($S_{\Theta_i} = \{\mathbf{p}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i, \mathbf{c}_i\}$). 이 3D Gaussian들을 2D 이미지 평면에 투영하여 pixel별로 미분 가능한 렌더링을 수행한다.

## 3D Gaussian Grouping

Gaussian Grouping의 핵심은, Gaussian의 기존 속성(위치·색·opacity·크기)은 그대로 두고 색상 모델링과 유사한 형태의 Identity Encoding을 새로 추가하는 것이다. 이를 통해 각 Gaussian이 장면 안의 어떤 instance/stuff에 속하는지를 나타낸다.

![Fig2](/assets/img/gaussian-grouping/fig2_pipeline.png)
_Fig 2. 파이프라인 3단계. (a) 각 view에 SAM을 everything 모드로 적용해 mask를 독립적으로 생성한다. (b) view 간 mask ID를 일치시키기 위해 universal temporal propagation 모델(zero-shot tracker)로 mask를 연결하여 일관된 multi-view segmentation을 만든다. (c) 준비된 입력으로 Gaussian의 모든 속성과 Identity Encoding을 differentiable rendering으로 함께 학습한다. Identity Encoding은 2D Identity Loss와 3D Regularization Loss로 supervise된다._

### Anything Mask Input

먼저 multi-view 이미지 각각에 SAM을 적용하여 everything 모드로 mask를 생성한다. 이 2D mask는 이미지마다 독립적으로 만들어지므로, 같은 객체라도 view마다 다른 ID를 갖는다.

### Identity Consistency across Views

학습 중 cost-based linear assignment를 반복 계산하는 대신, multi-view 이미지를 시점이 점진적으로 변하는 하나의 video로 간주하여, 잘 학습된 zero-shot tracker로 mask를 propagate·association한다. 이 과정에서 장면 전체의 총 instance 개수 $K$도 얻는다. 저자들은 이 방식이 linear assignment 대비 60배 이상 빠르고, SAM의 dense·overlapping mask 상황에서 성능도 더 좋다고 보고한다.

### Identity Encoding and Rendering

각 Gaussian에 학습 가능한 길이 16의 compact한 Identity Encoding $e_i$를 추가한다. instance ID는 시점에 무관하므로 SH degree를 0(DC 성분만)으로 두어 view-independent하게 모델링한다. 색상(SH)을 splatting하는 것과 동일하게, Identity Encoding도 $\alpha'$-blending으로 2D에 렌더링한다.

$$
\begin{equation}
E_{\text{id}} = \sum_{i \in \mathcal{N}} e_i\, \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j)
\end{equation}
$$

여기서 각 Gaussian의 영향 계수 $\alpha'_i$는 2D covariance $\Sigma^{2D}$와 per-point opacity로 계산되며, 3D covariance는 다음과 같이 2D로 투영된다.

$$
\begin{equation}
\Sigma^{2D} = J W \Sigma^{3D} W^T J^T
\end{equation}
$$

$J$는 3D-2D projection의 affine approximation에 대한 Jacobian, $W$는 world-to-camera 변환이다.

### Grouping Loss

Grouping loss $\mathcal{L}_{\text{id}}$는 두 항으로 구성된다.

2D Identity Loss: mask label이 2D이므로, 렌더링된 $E_{\text{id}}$에 linear layer $f$를 붙여 차원을 $K$로 되돌린 뒤 $\text{softmax}(f(E_{\text{id}}))$로 identity를 분류하고, $K$개 카테고리에 대한 standard cross-entropy $\mathcal{L}_{\text{2d}}$를 적용한다.

3D Regularization Loss: 2D supervision은 간접적이라, object 내부나 크게 가려진 Gaussian은 충분히 학습되지 않는다. 이를 보완하기 위해, 각 Gaussian의 identity 분포가 3D 상에서 가장 가까운 $k$개 이웃과 가까워지도록 하는 KL divergence loss를 도입한다($F$는 linear layer $f$ 뒤의 softmax).

$$
\begin{equation}
\mathcal{L}_{\text{3d}} = \frac{1}{m}\sum_{j=1}^{m} D_{\text{KL}}(P \,\|\, Q)
= \frac{1}{mk}\sum_{j=1}^{m}\sum_{i=1}^{k} F(e_j)\log\!\left(\frac{F(e_j)}{F(e'_i)}\right)
\end{equation}
$$

$P$는 샘플링한 Gaussian의 Identity Encoding $e$, $Q = \{e'_1, \dots, e'_k\}$는 3D Euclidean 공간에서의 $k$-nearest neighbor이다(실험에서 $k=5$). 최종 loss는 기존 reconstruction loss와 결합된다.

$$
\begin{equation}
\mathcal{L}_{\text{render}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{id}}
= \mathcal{L}_{\text{rec}} + \lambda_{\text{2d}}\mathcal{L}_{\text{2d}} + \lambda_{\text{3d}}\mathcal{L}_{\text{3d}}
\end{equation}
$$

## Local Gaussian Editing

학습이 끝나면 각 Gaussian이 group ID를 가지므로, 특정 group의 Gaussian만 선택적으로 조작하여 전체 장면을 훼손하지 않고 편집할 수 있다.

![Fig3](/assets/img/gaussian-grouping/fig3_editing.png)
_Fig 3. 학습 후 grouping된 3D Gaussian은 instance/stuff 단위로 완전히 분리되며, group 단위 연산으로 다양한 편집을 지원한다._

- 3D Object Removal: 선택한 group의 Gaussian을 삭제한다.
- 3D Object Inpainting: 객체를 삭제한 뒤 새 Gaussian을 추가하여 빈 영역을 채운다.
- 3D Object Colorization: 선택 group의 색상(SH)을 변경한다.
- 3D Object Style Transfer: 선택 group의 색상(SH)을 fine-tuning한다.
- 3D Scene Re-composition: group들의 위치를 교환·재배치한다.

## Experiments

저자들은 open-world 3D segmentation을 평가하기 위해 LERF-Localization의 세 장면(figurines, ramen, teatime)에 정확한 mask를 직접 annotation한 LERF-Mask 데이터셋을 구축했다(장면당 평균 7.7개의 text query).

![Table1](/assets/img/gaussian-grouping/table1_lerf_mask.png)
_Table 3. LERF-Mask에서의 open-vocabulary 3D segmentation 비교(mIoU·mBIoU). mask ID 선택에는 Grounding DINO를 사용한다._

Table 3에서 Gaussian Grouping은 figurines 69.7/67.9, ramen 77.0/68.7, teatime 71.7/66.1(mIoU/mBIoU)을 기록하여, LERF(33.5/30.6, 28.3/14.7, 49.7/42.6)와 SA3D를 큰 폭으로 능가하고 LangSplat도 앞선다. Fig 8의 정성 비교에서도 더 선명하고 정확한 경계를 보인다. 또한 Panoptic Lifting과의 panoptic segmentation 비교(Table 4)에서 성능·속도 모두 우위를 보이며, 편집 품질은 CLIP Text-Image Direction Similarity(Table 5)로 평가하여 inpainting(0.153 vs SPIn-NeRF 0.126), style transfer(0.178 vs Instruct-NeRF2NeRF 0.171), removal(0.183 vs DFF 0.166)에서 각 SOTA를 앞선다.

### Ablations

- Identity Consistency(mask association): cost-based linear assignment 대신 tracker 기반 연결을 쓰면 학습이 빠르고 성능도 좋다. SAM+tracker가 일부 view에서 실패해도, 공유된 3D Gaussian 표현 덕분에 mask 오류가 렌더링 과정에서 교정된다(Fig 5).
- 3D Regularization의 $k$: object removal 등에서 $k=5$가 품질과 효율의 균형이 가장 좋다.
- Grouping Loss: 2D Identity Loss와 3D Regularization Loss를 함께 쓰는 joint supervision이 가장 정확한 grouping을 만든다.

## Limitation

동적(dynamic) 모델링과 시간에 따른 갱신이 없어, 현재 방법은 static 3D 장면에 한정된다. 저자들은 향후 완전 unsupervised 3D Gaussian grouping을 탐구할 여지를 남긴다.

## Conclusion

Gaussian Grouping은 3D Gaussian 기반으로 open-world 장면을 재구성과 동시에 segment하는 최초의 방법이다. 각 Gaussian에 Identity Encoding을 부여하고 SAM 2D mask와 3D spatial regularization으로 학습시켜, 하나의 표현에서 reconstruction과 segmentation을 함께 달성한다. 나아가 grouping된 discrete Gaussian 덕분에 removal·inpainting·style transfer·recomposition 같은 편집을 효율적이고 정밀하게 수행할 수 있다.
