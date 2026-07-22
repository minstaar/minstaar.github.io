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

Open-world 3D scene understanding은 로보틱스, AR/VR, 자율주행 등에 중요한 문제다. Posed RGB 이미지 집합이 주어졌을 때, 저자들의 목표는 3D 장면을 재구성하는 동시에 그 안의 모든 것을 segment하고, 나아가 다양한 downstream 편집(제거·inpainting·recomposition 등)을 쉽게 지원하는 3D 표현을 학습하는 것이다.

2D에서는 SAM 등으로 큰 진전이 있었지만, 3D로의 확장은 제약이 컸다. 3D 장면 데이터셋을 만드는 비용이 크고, 기존 방법은 수작업 라벨이나 정밀 스캔 point cloud에 의존해 real-world 일반화가 어려웠다. NeRF 기반 방법들은 multi-view 캡처로부터 2D mask를 lift하거나 CLIP/DINO feature를 distill했지만, implicit·continuous 표현 특성상 최적화 비용이 크고, MLP가 장면의 각 부분을 쉽게 분해하지 못해 국소 편집에 부적합하다.

3DGS는 explicit한 3D Gaussian으로 빠르고 고품질 재구성을 이루지만, object instance나 semantic 이해는 담지 못한다. 저자들은 이를 확장하여, 재구성과 동시에 open-world 장면을 segment하는 Gaussian Grouping을 제안한다. 핵심은 각 Gaussian에 색상(SH)과 유사한 형태의 학습 가능한 Identity Encoding을 부여하여 Gaussian들을 instance/stuff 단위로 grouping하는 것이며, SAM의 zero-shot 2D 이해 능력을 3D로 일관되게 확장한다.

![Fig1](/assets/img/gaussian-grouping/fig1_teaser.png)
_Fig 1. Gaussian Grouping은 open-world 3D 장면을 재구성(a)하는 동시에 segment(b)한다. grouping된 discrete Gaussian을 이용하면 object를 쉽게 제거·편집하거나 장면을 재구성할 수 있다._

기여를 정리하면 다음과 같다.

1. 3D Gaussian에 학습 가능한 Identity Encoding을 추가하여, 장면을 appearance·geometry와 함께 instance/stuff identity까지 표현하는 최초의 Gaussian 기반 open-world 3D 이해 방법을 제안한다.
2. 학습 시 cost-based linear assignment 대신 zero-shot tracker로 multi-view mask를 일관되게 연결하여, 값싸고 견고한 supervision을 얻는다.
3. Grouping된 discrete Gaussian을 기반으로 Local Gaussian Editing을 제안하여 removal·inpainting·colorization·style transfer·recomposition을 효율적으로 지원한다.

## Related Work

### 3D Gaussian Models
3DGS는 real-time radiance field 렌더링으로 3D 장면을 재구성하는 강력한 방법으로 등장했고, 이후 dynamic 장면 확장이나 diffusion과 결합한 3D 생성 등 여러 후속 연구가 나왔다. 그러나 기존 Gaussian Splatting 계열 중 object/stuff 단위나 semantic 이해를 지원하는 것은 없었다. Gaussian Grouping은 이를 open-world·fine-grained 이해로 확장한다.

### Radiance-based Open-World Scene Understanding
Semantic-NeRF는 NeRF에 semantic 정보를 결합했고, 이후 instance 모델링이나 visual feature 인코딩으로 확장되었지만 대개 in-domain에 한정되어 open-world 일반화가 어렵다. CLIP/DINO feature를 distill하는 Distilled Feature Fields나 language field를 학습하는 LERF는 open-world를 다루지만, semantic segmentation에 국한되어 같은 카테고리의 유사 객체를 분리하기 어렵고 정밀한 mask를 내지 못한다.

### SAM in 3D
SAM은 zero-shot 2D segmentation의 foundation model이다. SAM의 2D mask를 NeRF나 point cloud를 통해 3D로 lift하는 연구들이 있으나, 대개 장면의 단일/소수 객체에만 집중한다. Gaussian Grouping은 everything 모드로 동작하여 장면 전체의 모든 instance/stuff에 대한 holistic 이해를 얻는다.

### Radiance-based Scene Editing
Radiance field 상의 장면 편집은 어렵다. Clip-NeRF, Object-NeRF, LaTeRF 등은 NeRF로 표현된 객체를 변경·완성하지만 단순 객체에 한정되고, bounding box 지정이나 물리 시뮬레이션 결합 방식도 있다. Gaussian Grouping은 discrete Gaussian의 group 단위 연산으로 clutter가 많은 장면에서도 효율적 편집을 지원한다.

## Preliminaries: 3D Gaussian Splatting

3DGS는 장면을 다수의 3D Gaussian 집합으로 표현한다. 각 Gaussian은 centroid $\mathbf{p} \in \mathbb{R}^3$, 표준편차 형태의 3D size $\mathbf{s} \in \mathbb{R}^3$, rotation quaternion $\mathbf{q} \in \mathbb{R}^4$, opacity $\alpha \in \mathbb{R}$, 그리고 SH로 표현되는 색상 $\mathbf{c}$로 정의된다($S_{\Theta_i} = \{\mathbf{p}_i, \mathbf{s}_i, \mathbf{q}_i, \alpha_i, \mathbf{c}_i\}$). 이 3D Gaussian들을 2D 이미지 평면에 투영하여 pixel별로 미분 가능한 $\alpha$-blending 렌더링을 수행한다.

## 3D Gaussian Grouping

Gaussian Grouping의 핵심은, Gaussian의 기존 속성(위치·색·opacity·크기)은 그대로 두고 색상 모델링과 유사한 형태의 Identity Encoding을 새로 추가하는 것이다. 이를 통해 각 Gaussian이 장면 안의 어떤 instance/stuff에 속하는지를 나타낸다.

![Fig2](/assets/img/gaussian-grouping/fig2_pipeline.png)
_Fig 2. 파이프라인 3단계. (a) 각 view에 SAM을 everything 모드로 적용해 mask를 독립적으로 생성한다. (b) view 간 mask ID를 일치시키기 위해 universal temporal propagation 모델(zero-shot tracker)로 mask를 연결하여 일관된 multi-view segmentation을 만든다. (c) 준비된 입력으로 Gaussian의 모든 속성과 Identity Encoding을 differentiable rendering으로 함께 학습한다. Identity Encoding은 2D Identity Loss와 3D Regularization Loss로 supervise된다._

### Anything Mask Input (a)

먼저 multi-view 이미지 각각에 SAM을 적용하여 everything 모드로 mask를 생성한다. 이 2D mask는 이미지마다 독립적으로 만들어지므로, 같은 객체라도 view마다 다른 ID를 갖는다. 따라서 이들을 3D 장면 상에서 동일 identity끼리 연결하고, 장면 전체의 총 instance 개수 $K$를 구해야 한다.

### Identity Consistency across Views (b)

학습 중 cost-based linear assignment를 반복 계산하는 대신, multi-view 이미지를 시점이 점진적으로 변하는 하나의 video로 간주하여, 잘 학습된 zero-shot tracker로 mask를 propagate·association한다. 이 과정에서 총 mask identity 수 $K$도 얻는다. 저자들은 이 방식이 linear assignment 대비 60배 이상 빠르고, SAM의 dense·overlapping mask 상황에서 성능도 더 좋으며, tracker의 연결에 오류가 있어도 공유된 3D 표현 덕분에 학습 중 교정된다고 보고한다.

### Identity Encoding and Rendering (c)

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

### Grouping Loss (d)

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

학습이 끝나면 각 Gaussian이 group ID를 가지므로, 특정 group의 Gaussian만 선택적으로 조작하여 전체 장면을 훼손하지 않고 편집할 수 있다. 핵심은 잘 학습된 대부분의 Gaussian은 freeze하고, 편집 대상과 관련된 소수의 기존/신규 Gaussian만 조정하는 것이다.

![Fig3](/assets/img/gaussian-grouping/fig3_editing.png)
_Fig 3. 학습 후 grouping된 3D Gaussian은 instance/stuff 단위로 완전히 분리되며, group 삭제·추가·SH 변경·위치 교환 같은 단순 연산(Gaussian Operation List)으로 다양한 편집을 지원한다._

- 3D Object Removal: 편집 대상 group의 Gaussian을 삭제한다(파라미터 튜닝 불필요).
- 3D Scene Re-composition: 두 group의 3D 위치를 교환한다(튜닝 불필요).
- 3D Object Inpainting: 대상 Gaussian을 삭제한 뒤 소수의 새 Gaussian을 추가하고, 렌더링 시 LaMa의 2D inpainting 결과로 이를 supervise하여 빈 영역을 채운다.
- 3D Object Colorization: geometry를 보존하기 위해 해당 group의 색상(SH) 파라미터만 튜닝한다.
- 3D Object Style Transfer: 더 사실적인 결과를 위해 색상뿐 아니라 3D 위치·크기까지 unfreeze하여 튜닝한다.

세밀한 mask 모델링 덕분에, 전체 장면을 재학습하지 않고도 여러 국소 편집을 서로 간섭 없이 동시에 수행할 수 있다.

![Fig9](/assets/img/gaussian-grouping/fig9_removal.png)
_Fig 9. Tanks & Temples에서의 3D object removal 비교. Distilled Feature Fields 대비 대상 객체를 더 깔끔하게 제거하고 배경을 자연스럽게 유지한다._

## Experiments

저자들은 open-world 3D segmentation을 평가하기 위해 LERF-Localization의 세 장면(figurines, ramen, teatime)에 coarse bounding box 대신 정확한 mask를 직접 annotation한 LERF-Mask 데이터셋을 구축했다(장면당 평균 7.7개의 text query). SAM은 language prompt를 지원하지 않으므로, SA3D와 본 방법 모두 Grounding DINO로 2D에서 mask ID를 식별한다.

![Table1](/assets/img/gaussian-grouping/table1_lerf_mask.png)
_Table 3. LERF-Mask에서의 open-vocabulary 3D segmentation 비교(mIoU·mBIoU)._

Table 3에서 Gaussian Grouping은 figurines 69.7/67.9, ramen 77.0/68.7, teatime 71.7/66.1(mIoU/mBIoU)을 기록하여, LERF(33.5/30.6, 28.3/14.7, 49.7/42.6)와 SA3D를 큰 폭으로 능가하고 LangSplat도 앞선다.

![Fig8](/assets/img/gaussian-grouping/fig8_seg_compare.png)
_Fig 8. LERF와의 세그멘테이션 정성 비교. Gaussian Grouping의 mask가 훨씬 선명하고 정확한 경계를 가지며, 유사한 색의 객체(예: green apple)도 잘 구분한다._

정성 비교(Fig 8)에서도 더 선명하고 정확한 경계를 보이며, 유사한 색의 객체 구분에도 강하다. 또한 Panoptic Lifting과의 panoptic segmentation 비교(Table 4)에서 성능·속도 모두 우위를 보인다. 편집 품질은 CLIP Text-Image Direction Similarity(Table 5)로 평가한다.

![Table5](/assets/img/gaussian-grouping/table5_editing.png)
_Table 5. 세 가지 편집 task(inpainting·style transfer·removal)에 대한 CLIP Text-Image Direction Similarity 비교._

inpainting(0.153 vs SPIn-NeRF 0.126), style transfer(0.178 vs Instruct-NeRF2NeRF 0.171), removal(0.183 vs DFF 0.166)에서 각 task의 SOTA를 앞선다.

### Ablations

- Identity Consistency(mask association): cost-based linear assignment 대신 tracker 기반 연결을 쓰면 학습이 빠르고 성능도 좋다(Fig 4). SAM+tracker가 일부 view에서 실패해도 공유된 3D Gaussian 표현이 mask 오류를 렌더링 과정에서 교정한다(Fig 5).
- 3D Regularization의 $k$: object removal 등에서 $k=5$가 품질과 효율의 균형이 가장 좋다(Fig 6, Table 2).
- Grouping Loss: 2D Identity Loss와 3D Regularization Loss를 함께 쓰는 joint supervision이 가장 정확한 grouping을 만든다(Fig 7).

## Limitation

동적(dynamic) 모델링과 시간에 따른 갱신이 없어, 현재 방법은 static 3D 장면에 한정된다. 저자들은 향후 완전 unsupervised 3D Gaussian grouping을 탐구할 여지를 남긴다.

## Conclusion

Gaussian Grouping은 3D Gaussian 기반으로 open-world 장면을 재구성과 동시에 segment하는 최초의 방법이다. 각 Gaussian에 Identity Encoding을 부여하고 SAM 2D mask와 3D spatial regularization으로 학습시켜, 하나의 표현에서 reconstruction과 segmentation을 함께 달성한다. 나아가 grouping된 discrete Gaussian 덕분에 removal·inpainting·style transfer·recomposition 같은 편집을 효율적이고 정밀하게 수행할 수 있다.
