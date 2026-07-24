---
title: "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
date: 2026-07-24 12:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Dynamic Scene, 6-DOF Tracking, Point Tracking, Novel View Synthesis, 3DV]
description: "3DV 2024"
math: true
toc: true
---

> [[Paper](https://arxiv.org/abs/2308.09713)] [[Page](https://dynamic3dgaussians.github.io/)] [[Github](https://github.com/JonathonLuiten/Dynamic3DGaussians)]
>
> Jonathon Luiten, Georgios Kopanas, Bastian Leibe, Deva Ramanan

## Introduction

3D Gaussian Splatting (3DGS) 은 static scene의 novel-view synthesis에서 놀라운 품질과 real-time rendering 속도를 달성했다. 하지만 현실 세계는 대부분 dynamic하다. 이 논문은 3DGS를 dynamic scene으로 확장하여, dynamic scene의 novel-view synthesis와 함께 scene을 구성하는 모든 dense element의 6-DOF (6 자유도) tracking을 동시에 수행하는 방법을 제안한다.

핵심 아이디어는 analysis-by-synthesis framework이다. 즉, 별도의 tracking supervision(optical flow, correspondence label, pose skeleton 등) 없이 오직 입력 이미지를 rendering으로 재현하도록 최적화하는 것만으로, dense correspondence tracking이 자연스럽게 emergent하게 나타난다. 이 논문에서 추적된 dense 6-DOF trajectory는 아래 teaser 그림에서 확인할 수 있다.

![fig1](/assets/img/dynamic3dgs/fig1_teaser.png){: width="100%"}
_Figure 1. Persistent Dynamic Novel-View Synthesis and Tracking. 각 scene에 대해 novel-view synthesis 결과(위쪽)와 depth map(아래쪽), 그리고 물체를 구성하는 dense Gaussian들이 시간에 따라 움직인 3D trajectory를 함께 보여준다._

이 방법의 핵심 insight는 dynamic scene을 oriented particle의 집합으로 모델링하고, 각 particle을 local rigid-body transformation을 통해 움직이는 것으로 물리적으로 다루는 것이다. 구체적으로, 각 Gaussian의 attribute를 다음과 같이 나눈다.

- 시간에 대해 persistent(불변)한 property: number(Gaussian 개수), color, opacity, size. 이들은 첫 timestep에서 결정된 후 고정된다. 즉, 각 Gaussian은 항상 동일한 물리적 공간의 조각을 나타낸다.
- 시간에 따라 변하는 property: position(center)과 orientation(rotation). 이 두 가지만 timestep마다 갱신되며, 이것이 곧 dense 6-DOF motion을 정의한다.

이렇게 property를 나누는 것이 dense correspondence를 얻는 핵심이다. 각 Gaussian이 항상 같은 물체 조각을 나타내므로, 그 Gaussian의 center를 시간에 따라 따라가는 것만으로 correspondence tracking이 완성되는 것이다. 이 논문의 tracking은 어떤 별도의 correspondence 입력도 없이 rendering objective만으로 emergent하게 얻어진다.

## Related Work

기존 dynamic scene 표현 방법(주로 dynamic NeRF 계열)은 크게 다섯 가지로 분류할 수 있다.

- (a) per-timestep: 각 timestep을 독립적인 static scene으로 표현. temporal consistency가 없다.
- (b) Eulerian 4D grid: 4차원(공간 3 + 시간 1) grid에 값을 저장. 공간의 고정된 위치에서 시간에 따른 값을 본다.
- (c) canonical + deformation field: canonical space를 하나 두고, 각 timestep을 canonical space로부터의 deformation field로 표현.
- (d) template-guided: human body 같은 template model을 사전에 알고 있다고 가정.
- (e) point-based (Lagrangian): scene을 particle의 집합으로 보고, 각 particle을 시간에 따라 따라간다.

이 논문은 (e) oriented particle 접근에 해당한다. (a)~(d)와 달리, Lagrangian 관점은 물체의 각 조각을 명시적으로 추적하므로 dense correspondence를 자연스럽게 제공한다.

가장 유사한 연구는 OmniMotion이다. 그러나 OmniMotion은 monocular video를 입력으로 받고 dense optical flow를 입력으로 요구하며, 그 계산 비용이 frame 수의 제곱($$\text{frames}^2$$)으로 증가한다. 반면 이 논문은 어떤 correspondence 입력도 필요로 하지 않는다.

## Method

이 논문의 방법은 test-time optimization 방식이다. 추가 training data 없이, 대상 scene의 multi-camera 영상만으로 최적화한다. 최적화는 temporally online 방식으로, 한 번에 한 timestep씩 진행하며 각 timestep은 직전 timestep의 결과로 초기화된다. 첫 timestep에서는 모든 property를 최적화한 뒤 이후로는 고정하고, 이후 timestep에서는 motion(position, rotation)을 정의하는 파라미터만 갱신한다.

### Dynamic 3D Gaussians

Dynamic scene representation $$\mathcal{S}$$ 는 Dynamic 3D Gaussian들의 집합으로 파라미터화되며, 각 Gaussian은 다음 파라미터를 가진다.

1. 각 timestep의 3D center $$(x_t, y_t, z_t)$$
2. 각 timestep의 3D rotation을 나타내는 quaternion $$(qw_t, qx_t, qy_t, qz_t)$$
3. 모든 timestep에 걸쳐 일정한 3D size (standard deviation) $$(s_x, s_y, s_z)$$
4. 모든 timestep에 걸쳐 일정한 color $$(r, g, b)$$
5. 모든 timestep에 걸쳐 일정한 opacity logit $$(o)$$
6. 모든 timestep에 걸쳐 일정한 background logit $$(bg)$$

즉, Gaussian 하나당 총 $$7t + 8$$ 개의 파라미터를 가진다($$t$$ 는 timestep 수). 실험에서 scene은 20만~30만 개의 Gaussian으로 표현되며, 이 중 static background가 아닌 것은 보통 3만~10만 개이다. (코드에는 spherical harmonics로 view-dependent color를 표현하는 기능이 있으나, 실험에서는 단순화를 위해 끈다.)

각 Gaussian은 3D 공간의 한 점 $$p$$ 에 대해 opacity로 가중된 표준 (unnormalized) Gaussian 식에 따라 영향을 미친다.

$$
f_{i,t}(p) = \text{sigm}(o_i) \exp\left( -\frac{1}{2}(p - \mu_{i,t})^T \Sigma_{i,t}^{-1} (p - \mu_{i,t}) \right)
$$

여기서 $$\mu_{i,t} = [x_{i,t}, y_{i,t}, z_{i,t}]^T$$ 는 timestep $$t$$ 에서 Gaussian $$i$$ 의 center이고, covariance matrix $$\Sigma_{i,t}$$ 는 scaling과 rotation 성분을 결합해 구성된다.

$$
\Sigma_{i,t} = R_{i,t} S_i S_i^T R_{i,t}^T
$$

이때 $$S_i = \text{diag}([s_{x_i}, s_{y_i}, s_{z_i}])$$ 는 scaling 성분(시간 불변), $$R_{i,t} = \text{q2R}([qw_{i,t}, qx_{i,t}, qy_{i,t}, qz_{i,t}])$$ 는 rotation 성분(시간 가변)이며, $$\text{q2R}()$$ 는 quaternion으로부터 rotation matrix를 구성하는 함수다.

각 Gaussian의 influence $$f$$ 는 본질적으로 local(작은 공간 영역을 표현)이면서 동시에 이론적으로 무한한 extent를 가진다. 이 무한 extent 성질이 중요한데, 아직 잘못된 위치에 있는 Gaussian이라도 멀리 있는 gradient를 받아 올바른 위치로 이동할 수 있게 해주기 때문이다. 이는 gradient 기반의 tracking-by-differentiable-rendering에서 필수적이다.

시간에 따라 size/opacity/color를 고정함으로써, 각 Gaussian은 공간이 dynamic하게 움직이더라도 항상 동일한 물리적 조각을 표현하게 된다. 이 motion을 표현하기 위해 각 Gaussian은 시간에 따라 움직일 수 있는 center와 rotation을 가지며, 이것이 곧 scene 전체의 dense non-rigid 6-DOF tracking을 가능하게 한다.

![fig2](/assets/img/dynamic3dgs/fig2_centers.png){: width="70%"}
_Figure 2. Gaussian Centers. 연속된 두 timestep에서 colored center들의 point cloud. Gaussian center들이 scene geometry를 모델링하면서 시간에 따라 어떻게 움직이는지 보여준다._

![fig3](/assets/img/dynamic3dgs/fig3_rotation.png){: width="70%"}
_Figure 3. Relative Rotation Tracking. 1번째 panel: 첫 frame에서 3%의 Gaussian에 왼쪽을 향하는 colored vector를 부착. 2, 3번째 panel: 이 vector들이 부착된 Gaussian과 함께 이동하고 회전하며, 이 방법이 6-DOF motion을 올바르게 모델링함을 보여준다._

### Differentiable Rendering via Gaussian 3D Splatting

Gaussian의 파라미터를 최적화해 scene을 표현하려면, Gaussian을 이미지로 differentiable하게 rendering해야 한다. 이 논문은 원본 3DGS의 differentiable 3D Gaussian renderer를 그대로 사용하고, 이를 dynamic scene으로 확장한다. Gaussian center는 표준 point rendering 식으로 splatting된다.

$$
\mu^{2D} = K \left( \frac{E\mu}{(E\mu)_z} \right)
$$

여기서 3D center $$\mu$$ 는 world-to-camera extrinsic matrix $$E$$ 로 변환되고, z-normalization을 거친 뒤 intrinsic projection matrix $$K$$ 로 곱해져 2D 이미지로 projection된다. 3D covariance는 다음 식으로 2D로 splatting된다.

$$
\Sigma^{2D} = J E \Sigma E^T J^T
$$

여기서 $$J$$ 는 위 projection 식의 Jacobian $$\partial \mu^{2D} / \partial \mu$$ 이다. 이제 각 pixel에서 각 Gaussian의 2D influence $$f^{2D}$$ 를 계산할 수 있고, 모든 Gaussian을 depth 순으로 정렬한 뒤 front-to-back volume rendering으로 합친다.

$$
C_{\text{pix}} = \sum_{i \in \mathcal{S}} c_i f_{i,\text{pix}}^{2D} \prod_{j=1}^{i-1} (1 - f_{j,\text{pix}}^{2D})
$$

최종 pixel color $$C_{\text{pix}}$$ 는 각 Gaussian의 color $$c_i$$ 를 그 pixel에 대한 influence $$f_{i,\text{pix}}^{2D}$$ 로 가중하고, 앞쪽 Gaussian들의 occlusion(transmittance) 항으로 down-weight한 가중합이다. 원본 3DGS 구현의 그래픽스·CUDA 최적화 덕분에 이 논문의 scene에서 850 FPS의 매우 빠른 rendering 속도를 달성하며, 이는 곧 매우 빠른 training으로도 이어진다.

### Physically-Based Priors

단순히 color/opacity/size를 고정하는 것만으로는 long-term persistent track을 얻기에 부족하다. 특히 색이 거의 균일한 넓은 영역에서는, 그 영역 안에서 Gaussian이 자유롭게 떠돌아다녀도 rendering loss에 아무 제약이 없기 때문이다. 물리적으로 움직이는 scene을 모델링하고 있으므로, non-rigid physical modeling에서 영감을 얻어 세 가지 regularization loss를 도입한다.

(1) Local-Rigidity Loss ($$\mathcal{L}^{\text{rigid}}$$) — 세 loss 중 가장 중요하다. 각 Gaussian $$i$$ 에 대해, 이웃 Gaussian $$j$$ 는 timestep 사이에서 $$i$$ 의 coordinate system의 rigid-body transform을 따라 움직여야 한다는 제약이다.

$$
\mathcal{L}^{\text{rigid}}_{i,j} = w_{i,j} \left\| (\mu_{j,t-1} - \mu_{i,t-1}) - R_{i,t-1} R_{i,t}^{-1} (\mu_{j,t} - \mu_{i,t}) \right\|_2
$$

$$
\mathcal{L}^{\text{rigid}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \text{knn}_{i;k}} \mathcal{L}^{\text{rigid}}_{i,j}
$$

online optimization이므로 $$\mu_{i,t-1}, R_{i,t-1}, \mu_{j,t-1}$$ 은 고정되어 있고, $$\mu_{i,t}, R_{i,t}, \mu_{j,t}$$ 를 최적화하여 $$i$$ 자신의 coordinate system 변화가 정의하는 rigid-body transform과 일치하도록 만든다. 예를 들어 $$i$$ 가 회전하면, $$j$$ 는 $$i$$ 의 중심을 축으로 회전한 것과 동등하게 (자신의 coordinate system에서) translation해야 한다. 이 loss는 반대 방향으로도 작용하여 translation이 rotation과 일치하도록 강제하며, 그 결과 rendering만 맞춰 최적화함에도 모든 dense point에 대해 정확한 rotation(6-DOF) tracking을 얻는다. 이는 일반적인 point-based 표현으로는 불가능한 일이다.

![fig4](/assets/img/dynamic3dgs/fig4_rigidity.png){: width="80%"}
_Figure 4. Local Rigidity Loss. 각 Gaussian $$i$$ 에 대해, 이웃 Gaussian $$j$$ 는 timestep 사이에서 $$i$$ 의 coordinate system의 rigid-body transform을 따라 움직여야 한다._

이웃 집합 $$j$$ 는 $$i$$ 의 k-nearest-neighbour ($$k=20$$)로 제한되며, loss는 Gaussian pair에 대한 weighting factor로 가중된다.

$$
w_{i,j} = \exp\left( -\lambda_w \|\mu_{j,0} - \mu_{i,0}\|_2^2 \right)
$$

이는 (unnormalized) isotropic Gaussian weighting factor이다. $$\lambda_w = 2000$$ 으로 설정하면 standard deviation이 약 2.2cm가 되며, 첫 timestep에서의 Gaussian center 간 거리로 계산해 이후 고정한다. 이로써 rigidity loss는 local하게만 강제되고, global하게는 여전히 non-rigid reconstruction이 가능하다.

(2) Local Rotation-Similarity Loss ($$\mathcal{L}^{\text{rot}}$$) — rigidity loss 하나만으로도 충분히 좋은 결과를 얻지만, 이웃 Gaussian들이 시간에 따라 같은 rotation을 갖도록 명시적으로 강제하면 수렴이 더 잘 되었다.

$$
\mathcal{L}^{\text{rot}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \text{knn}_{i;k}} w_{i,j} \left\| \hat{q}_{j,t} \hat{q}_{j,t-1}^{-1} - \hat{q}_{i,t} \hat{q}_{i,t-1}^{-1} \right\|_2
$$

여기서 $$\hat{q}$$ 는 각 Gaussian rotation의 normalized quaternion 표현으로, smooth한 최적화를 가능하게 한다. 동일한 k-nearest-neighbour와 weighting function을 사용한다.

(3) Long-Term Local-Isometry Loss ($$\mathcal{L}^{\text{iso}}$$) — 위 두 loss는 현재 timestep과 직전 timestep 사이에서만(short-term) 적용되므로, 시간이 지나면 scene의 요소들이 서서히 떨어져 나갈(drift) 수 있다. 이를 막기 위해 long-term에 걸쳐 세 번째 loss를 적용한다.

$$
\mathcal{L}^{\text{iso}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \text{knn}_{i;k}} w_{i,j} \left| \|\mu_{j,0} - \mu_{i,0}\|_2 - \|\mu_{j,t} - \mu_{i,t}\|_2 \right|
$$

이는 rigidity loss보다 약한 제약으로, 두 Gaussian 사이의 위치를 동일하게 강제하는 대신 그 거리만 동일하게 유지하도록 한다.

### Optimization Details

각 timestep은 하나씩 순차적으로 최적화된다. 첫 timestep에서는 모든 파라미터를 최적화하지만, 이후에는 size, color, opacity, background logit을 고정하고 position과 rotation만 갱신한다. 따라서 이 방법은 첫 frame의 static reconstruction을 먼저 수행한 뒤, 이후 frame에 대해 long-term dense 6-DOF tracking을 수행하는 것으로 볼 수 있다.

원본 3DGS를 따라, 첫 timestep에서는 coarse point cloud로 scene을 초기화한다(colmap으로 얻을 수도 있으나, 여기서는 depth camera의 sparse sample을 사용). 단, 이 depth 값은 첫 timestep의 sparse point cloud 초기화에만 쓰이고 최적화 과정에서는 전혀 사용되지 않는다. 첫 timestep에서는 densification을 적용해 Gaussian 밀도를 높이지만, 이후 frame에서는 Gaussian 개수를 고정하고 densification을 끈다.

첫 frame은 27개 training camera로 10000 iteration(약 4분) 최적화하고, 이후 각 timestep은 2000 iteration(약 50초)씩 최적화하여 150 timestep 전체에 대해 총 2시간이 걸린다. 각 새 timestep의 시작에서는, 현재 position에서 이전 position을 뺀 velocity를 이용한 forward estimate로 Gaussian의 center와 rotation quaternion을 초기화한다(quaternion은 재정규화). 이 forward propagation이 좋은 결과를 얻는 데 매우 중요했다. 또한 각 timestep 시작 시 Adam optimizer의 1차·2차 momentum을 reset한다.

테스트 scene에서 피험자의 셔츠 색(회색)이 배경과 매우 비슷해, 셔츠가 배경과 혼동되어 mis-tracking되는 경우가 있었다. 이를 위해 foreground/background mask를 rendering하고 pseudo-ground-truth background mask에 대해 background segmentation loss ($$\mathcal{L}^{\text{Bg}}$$) 를 적용한다(mask는 foreground 물체가 없는 frame과의 차분으로 쉽게 얻는다). 또한 background point는 움직이거나 회전하지 않도록 직접 loss를 걸고, 위의 rigidity·rotation·isometry loss는 foreground point에 대해서만 적용해 효율을 높인다.

마지막으로, 27개 camera마다 white balance·exposure·sensor sensitivity·color calibration이 다르므로, 각 camera·각 color channel마다 scale과 offset 파라미터를 최적화해 이 차이를 모델링한다(첫 timestep에서만 최적화 후 고정).

### Tracking with Dynamic 3D Gaussians

학습이 끝나면 3D scene의 임의의 점에 대해 dense correspondence를 얻을 수 있다. 3D 공간의 임의의 점 $$p$$ 의 timestep 간 correspondence는, 그 점에 대해 가장 큰 influence $$f(p)$$ 를 갖는 Gaussian의 coordinate system 안에서의 위치를 취함으로써(모든 Gaussian에 대해 $$f(p) < 0.5$$ 이면 static background coordinate system) motion-space를 linearize하여 구한다. 이렇게 하면 모든 timestep에 걸쳐 모든 점에 대해 잘 정의된 일대일·가역 mapping, 즉 dense correspondence가 얻어진다.

같은 아이디어로 임의의 (입력 또는 novel) view의 2D pixel을 다른 timestep이나 view로 추적할 수 있다. 먼저 입력 pixel에 대응하는 3D 점을 결정하는데, color 대신 Gaussian center의 depth를 사용해(원본 3DGS의 rendering 식 재활용) depth map을 rendering한 뒤 unprojection한다. 그다음 가장 influence가 큰 Gaussian을 찾아 추적하고, 새 camera frame으로 projection한다.

## Experiments

### Dataset

PanopticSports 라는 dataset을 준비했다. Panoptic Studio dataset의 sports sequence에서 6개 sub-sequence(juggle, box, softball, tennis, football, basketball)를 취한다. 각 sequence는 30 FPS로 150 frame이며, 31개 HD camera 중 27개를 training, 4개(cam 0, 10, 15, 30)를 testing으로 나눈다. Camera들은 capture studio dome 중앙의 관심 영역을 반구 형태로 둘러싸고 있으며, 정확한 intrinsic·extrinsic이 제공된다. 이미지는 640×360으로 resize한다. 첫 timestep의 Gaussian 초기화를 위해 10개 depth camera의 point를 사용한다. Ground-truth 3D/2D trajectory는 scene에 제공된 고품질 facial·hand keypoint annotation으로 준비하여, 총 21개의 ground-truth 3D trajectory와 371개의 2D ground-truth track을 얻는다.

### Metrics & Comparisons

Novel-view synthesis는 표준 PSNR, SSIM, LPIPS 로 평가한다. 2D long-term point tracking은 PointOdyssey benchmark의 지표(median trajectory error(MTE), position accuracy($$\delta$$), survival rate)를 사용한다. 3D long-term point tracking은 선행 연구가 없어, 2D 지표를 3D domain으로(normalized-pixel 대신 cm 단위) 확장하여 사용한다.

세 task(view synthesis, 3D tracking, 2D tracking) 모두에서 원본 3DGS를 online mode로 돌린 3GS-O 를 baseline으로 비교한다. 2D tracking은 long-term 2D point tracking의 대표적 학습 기반 방법인 PIPs 와도 비교한다. (완전히 공정한 비교는 아니다. 이 방법은 27개 training camera를 모두 보지만 PIPs는 추적 대상 camera 하나만 본다. 반대로 PIPs는 ground-truth track이 주어진 13085개 training video로 학습된 반면, 이 방법은 어떤 ground-truth track도, 다른 어떤 video data도 본 적이 없다.)

### PanopticSports Results

| Task | Metric | Method | Juggle | Boxes | Softball | Tennis | Football | Basketball | **Mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| View Synthesis | PSNR↑ | 3GS-O | 28.19 | 28.74 | 28.77 | 28.03 | 28.49 | 27.02 | 28.21 |
| | | **Ours** | 29.48 | 29.46 | 28.43 | 28.11 | 28.49 | 28.22 | **28.7** |
| | SSIM↑ | 3GS-O | 0.91 | 0.91 | 0.91 | 0.90 | 0.90 | 0.89 | 0.90 |
| | | **Ours** | 0.92 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | **0.91** |
| | LPIPS↓ | 3GS-O | 0.15 | 0.15 | 0.14 | 0.16 | 0.16 | 0.18 | **0.16** |
| | | Ours | 0.15 | 0.17 | 0.19 | 0.17 | 0.19 | 0.18 | 0.17 |
| 3D Tracking | 3D MTE↓ | 3GS-O | 32.81 | 39.95 | 64.94 | 75.54 | 45.57 | 76.71 | 55.9 |
| | | **Ours** | 1.90 | 1.97 | 2.02 | 2.33 | 2.45 | 2.56 | **2.21** |
| | 3D δ↑ | 3GS-O | 13.6 | 3.5 | 5.9 | 4.2 | 9.8 | 3.5 | 6.8 |
| | | **Ours** | 77.2 | 75.9 | 70.3 | 69.0 | 69.4 | 66.3 | **71.4** |
| | 3D Surv↑ | 3GS-O | 56.3 | 60.8 | 37.2 | 16.9 | 59.6 | 31.9 | 43.8 |
| | | **Ours** | 100 | 100 | 100 | 100 | 100 | 100 | **100** |
| 2D Tracking | 2D MTE↓ | 3GS-O | 23.86 | 29.88 | 51.6 | 58.15 | 35.15 | 64.29 | 43.8 |
| | | PIPs | 5.76 | 8.42 | 13.3 | 21.0 | 23.2 | 22.6 | 15.7 |
| | | **Ours** | 1.54 | 1.42 | 1.69 | 1.36 | 1.48 | 1.93 | **1.57** |
| | 2D δ↑ | 3GS-O | 17.1 | 10.5 | 8.9 | 6.5 | 15.0 | 7.2 | 10.9 |
| | | PIPs | 55.9 | 39.5 | 37.0 | 28.4 | 43.5 | 33.2 | 39.6 |
| | | **Ours** | 80.4 | 82.5 | 77.3 | 80.2 | 79.7 | 73.9 | **78.4** |
| | 2D Surv↑ | 3GS-O | 71.3 | 74.4 | 42.7 | 23.0 | 69.6 | 47.1 | 54.7 |
| | | PIPs | 91.6 | 61.3 | 88.6 | 72.2 | 79.8 | 77.6 | 79.0 |
| | | **Ours** | 100 | 100 | 100 | 100 | 100 | 100 | **100** |

_Table 1. PanopticSports dataset 결과._

Novel-view synthesis 에서 이 방법은 최종 PSNR 28.7을 달성하며, dynamic scene의 temporal consistency를 올바르게 모델링함으로써 원본 3DGS보다 PSNR·SSIM에서 모든 scene에 걸쳐 더 나은 점수를 얻는다(LPIPS는 약간 낮음).

3D tracking 에서는 모든 scene의 모든 trajectory에 걸쳐 median trajectory error가 단 2.21cm 인 뛰어난 결과를 얻는다. 이는 극도로 복잡하고 빠른 motion을 150 timestep 동안 추적하면서도 손목 폭보다 작은 오차이다. 또한 survival rate 100%(추적 대상을 한 번도 놓치지 않음)를 달성한다. 원본 3DGS는 55.9cm의 훨씬 큰 오차로 3D tracking을 전혀 제대로 수행하지 못한다.

2D tracking 에서 이 방법이 진가를 발휘한다. SOTA tracker인 PIPs 대비 median trajectory error를 10배 낮춘 1.57 pixel(PIPs 15.7), trajectory accuracy 78.4(PIPs 39.6), survival rate 100%(PIPs 79%)를 달성한다.

![fig6](/assets/img/dynamic3dgs/fig6_gt.png){: width="90%"}
_Figure 6. Ground-truth Comparison. 이 방법의 결과(파란색)와 ground-truth(빨간색)의 비교. Ground-truth가 noisy한 부분에서는 오히려 이 방법의 결과가 더 정확할 수 있다._

### Particle-NeRF Dataset Results

더 단순한 synthetic scene으로 구성된 Particle-NeRF dataset(20 train, 10 test camera)에서도 비교했다. 이 dataset에서 이 방법은 PSNR·SSIM·LPIPS 모두에서 거의 완벽에 가까운 점수를 달성한다.

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|:---:|:---:|:---:|:---:|
| TiNeuVox-S | 26.64 | 0.92 | 0.14 |
| TiNeuVox | 27.28 | 0.91 | 0.13 |
| InstantNGP | 24.69 | 0.91 | 0.12 |
| Particle-NeRF | 27.47 | 0.94 | 0.08 |
| **Ours** | **39.49** | **0.99** | **0.02** |

_Table 2. Particle-NeRF dataset 결과._

![fig5](/assets/img/dynamic3dgs/fig5_comparison.png){: width="60%"}
_Figure 5. Visual comparison. Particle-NeRF dataset에서 Particle-NeRF(왼쪽)와 Ours(오른쪽) 비교._

### Ablation Study

Juggle sequence에서 방법의 각 구성 요소를 제거하는 ablation을 수행했다. 원본 3DGS 대비 추가된 6개 핵심 요소(rigidity loss, rotation loss, isometry loss, background loss, parameter fixing, forward propagation)를 하나씩, 그리고 모두 제거한 경우를 평가한다.

| Exp | Description | PSNR↑ | 3D MTE↓ | 3D δ↑ | 2D MTE↓ | 2D δ↑ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | **Ours - Full** | **29.48** | **1.90** | **77.2** | **1.54** | **80.4** |
| 1 | No $$\mathcal{L}^{\text{rigid}}$$ | 28.51 | 4.32 | 55.2 | 3.80 | 58.7 |
| 2 | No $$\mathcal{L}^{\text{rot}}$$ | 29.43 | 1.91 | 76.6 | 1.55 | 79.8 |
| 3 | No $$\mathcal{L}^{\text{iso}}$$ | 29.36 | 1.93 | 76.7 | 1.72 | 79.3 |
| 4 | No $$\mathcal{L}^{\text{Bg}}$$ | 24.14 | 8.46 | 60.0 | 6.40 | 63.2 |
| 5 | No Param Fixing | 27.14 | 30.7 | 57.7 | 19.15 | 58.8 |
| 6 | No Forward Prop | 28.48 | 6.32 | 54.87 | 5.4 | 57.7 |
| 7 | 3GS-O | 28.19 | 32.81 | 13.6 | 23.86 | 17.1 |

_Table 3. PanopticSports Juggle scene에 대한 ablation 결과._

View synthesis에서는 원본 3GS-O도 이미 매우 잘 작동하지만, scene 요소의 motion을 올바르게 모델링함으로써 full 방법은 PSNR을 1.3 향상시킨다. Tracking에서는 원본이 전혀 정확하게 추적하지 못하는 반면 이 방법은 매우 정확하다. 핵심적으로 rigidity loss, background segmentation loss, parameter fixing, forward propagation 네 가지가 좋은 tracking 결과와 view-synthesis 향상 모두에 필수적이다. Rotation loss와 isometry loss는 측정 지표상 개선폭은 작지만, 시각적으로는 둘 다 켰을 때 reconstruction이 훨씬 일관되고 정확했다.

### Further Applications

Gaussian이 서로 독립적이므로, 일부 subset을 scene에서 손쉽게 추가·제거하여 다양한 효과를 만들 수 있다. 또한 6-DOF tracking을 활용해 edit(예: 표면에 추가한 로고)을 시간에 걸쳐 자동으로 전파하거나, camera를 특정 Gaussian에 부착해 그 Gaussian의 움직임·회전을 따라가는 first-person view를 만들 수 있다.

![fig7](/assets/img/dynamic3dgs/fig7_applications.png){: width="100%"}
_Figure 7. Dynamic 3D Gaussians가 가능하게 하는 augmented reality 응용. 왼쪽: dynamic 물체를 scene에서 제거·복제하거나 다른 scene에 추가. 중앙 왼쪽: 이미지 edit을 3D로 lift한 뒤 시간에 걸쳐 자동 전파. 중앙 오른쪽: camera view를 dynamic Gaussian에 부착(위: first-person view, 아래: juggling ball의 view). 오른쪽: 물체를 scan해 scene을 따라 움직이도록 추가(예: 모자가 handstand를 하는 사람 위에 올바른 translation·rotation으로 유지)._

## Conclusion

이 논문은 dynamic 3D scene modeling, view synthesis, 6-DOF tracking을 한 번에 수행하는 새로운 방법을 제안했다. Gaussian element로 dynamic scene을 모델링함으로써 물리적 성질과 일관되게 움직임과 회전을 포착하며, entertainment, robotics, VR, AR 등 다양한 domain에 응용될 수 있다. 이 접근은 efficiency와 accuracy를 겸비하여, real-time rendering과 creative scene editing에 대한 새로운 가능성을 제시하며 3D modeling·tracking 분야의 향후 연구에 유망한 방향을 제시한다.

### Limitations

이 방법은 뛰어난 결과를 달성하지만 한계도 존재한다.

- 첫 frame에 보이는 부분만 tracking 가능: 설계상 이 방법은 초기 frame에 보이는 scene 부분만 추적할 수 있다. 따라서 scene에 새로 등장하는 물체는 전혀 재구성하지 못한다.
- multi-camera setup 필요: 이 방법은 multi-camera setup을 요구하며, monocular video 에서는 그대로 작동하지 않는다.

저자들은 이 한계들이 오히려 Dynamic 3D Gaussian 표현을 확장하는 흥미로운 향후 연구 방향의 씨앗이라고 본다.
