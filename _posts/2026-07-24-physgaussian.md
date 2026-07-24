---
title: "PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics"
date: 2026-07-24 11:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Material Point Method, Continuum Mechanics, Physics Simulation, Generative Dynamics, CVPR]
description: "CVPR 2024"
math: true
toc: true
---

> [[Paper](https://arxiv.org/abs/2311.12198)] [[Page](https://xpandora.github.io/PhysGaussian/)] [[Github](https://github.com/XPandora/PhysGaussian)]
>
> Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, Chenfanfu Jiang

## Introduction

NeRF와 3D Gaussian Splatting(3DGS)은 novel-view synthesis에서 큰 진전을 이뤘지만, 새로운 dynamics(움직임)를 생성하는 응용에는 여전히 큰 격차가 있다. NeRF에 새로운 pose를 만드는 시도들은 대개 quasi-static geometry editing에 국한되고, meshing이나 tetrahedron 같은 coarse proxy mesh에 visual geometry를 embedding해야 한다.

전통적인 physics-based 콘텐츠 생성 pipeline은 여러 단계로 이루어진 번거로운 과정이다. geometry 구성 → simulation-ready 변환(tetrahedralization 등) → physics simulation → rendering. 이 다단계 과정은 simulation geometry와 최종 rendering geometry 사이에 불일치(discrepancy)를 만든다. 자연계에서는 물체의 물리적 거동과 시각적 외형이 본질적으로 얽혀 있는데, 이 pipeline은 그것을 인위적으로 분리한다.

이 논문은 이 두 측면을 하나로 통합하는 "what you see is what you simulate"(WS2) 철학을 제안한다. 핵심 아이디어는 simulation과 rendering이 동일한 3D Gaussian kernel을 discrete representation으로 공유하는 것이다.

![Fig1](/assets/img/physgaussian/fig1_teaser.png)
_Fig 1. PhysGaussian은 3D Gaussians와 continuum mechanics에 기반한 통합 simulation-rendering pipeline이다. 위: fox(elastic), 아래: 꽃병(다양한 dynamics). 우측: "What You See"(rendering)와 "What You Simulate"(내부 particle)이 하나의 표현을 공유함을 보여준다._

이를 위해 저자들은 PhysGaussian을 제안한다. 3D Gaussian kernel에 velocity·strain 같은 kinematic attribute와 elastic energy·stress·plasticity 같은 mechanical property를 부여한다. continuum mechanics 원리와 custom Material Point Method(MPM)를 통해, physical simulation과 visual rendering이 모두 3D Gaussians로 구동된다. 이렇게 하면 embedding mechanism이 필요 없어져 simulated와 rendered 사이의 resolution mismatch가 사라진다.

주요 기여는 다음과 같다.

- Continuum Mechanics for 3D Gaussian Kinematics: physical PDE로 구동되는 displacement field 안에서 3D Gaussian kernel과 그 spherical harmonics를 진화시키는 continuum mechanics 기반 전략.
- Unified Simulation-Rendering Pipeline: 통합된 3D Gaussian representation으로 explicit meshing 없이 motion을 생성하는 효율적 pipeline.
- Versatile Benchmarking: elastic object, plastic metal, non-Newtonian fluid, granular material 등 다양한 재질에 대한 종합 벤치마크. 간단한 dynamics에서는 real-time 성능 달성.

## Related Work

### Radiance Fields Rendering for View Synthesis
NeRF는 fully-connected network로 하나의 scene을 모델링하고 volume rendering으로 이미지를 생성한다. 반면 point-based 방법 중 SOTA인 3D Gaussian Splatting은 implicit neural representation과 달리 explicit·unstructured representation을 써서 post-manipulation으로 확장이 쉽고, fast visibility-aware rendering으로 real-world dynamics 생성도 가능하다.

### Dynamic Neural Radiance Field
NeRF에 temporal 차원을 더해 dynamic scene을 표현하는 방향이다. 일부 연구는 time-dependent field를 inverse displacement field와 canonical time-invariant field로 분해하고, physics-based deformation을 NeRF에 넣기도 했지만 대개 NeRF에서 추출한 mesh에 의존한다. 이 논문은 explicit한 3D GS ellipsoid를 physics·graphics 공통 표현으로 쓰며, 기존 dynamic GS(kernel 모양을 유지하거나 학습으로 수정)와 달리 displacement map의 first-order 정보(deformation gradient)를 활용해 kernel을 변형시킨다.

### Material Point Method
MPM은 elastic object·fluid·sand·snow 등 다양한 multi-physics 현상에 널리 쓰이는 simulation framework다. topology 변화와 frictional interaction을 자연스럽게 다룰 수 있고, GPU 가속이 잘 입증되어 있다. 이 논문은 GS와 particle representation을 공유하며 MPM으로 latent physical dynamics를 지원한다.

## Method

PhysGaussian은 continuum mechanics와 3D GS에 기반한 통합 simulation-rendering framework다. 먼저 static scene의 GS representation을 복원하고(선택적으로 over-skinny kernel을 억제하는 anisotropic loss 사용), 이 Gaussians를 simulation 대상 continuum의 discretization으로 본다. 새로운 kinematics 하에서 변형된 Gaussians를 직접 splat하여 photo-realistic rendering을 얻는다. 물리적 정합을 위해 선택적으로 물체 내부 영역을 particle로 채운다(internal filling).

![Fig2](/assets/img/physgaussian/fig2_overview.png)
_Fig 2. Method Overview. Input Images로 3D Gaussian Splatting을 수행(선택적 anisotropic loss + kernel filling) → Gaussian ellipsoid를 continuum으로 간주 → Physics Integration(Kinematics: Gaussian evolution·harmonics transform / Dynamics: continuum mechanics·time integration) → 여러 시점에서 physics-grounded novel motion을 렌더링._

### 3D Gaussian Splatting

3DGS는 NeRF를 unstructured 3D Gaussian kernel 집합 $\{x_p, \sigma_p, A_p, C_p\}_{p \in \mathcal{P}}$ (center, opacity, covariance, spherical harmonic 계수)로 reparameterize한다. 3D Gaussian을 image plane에 2D Gaussian으로 projection하며, pixel color는 다음과 같이 계산된다.

$$
\begin{equation}
C = \sum_{k \in \mathcal{P}} \alpha_k \text{SH}(d_k; \mathcal{C}_k) \prod_{j=1}^{k-1}(1 - \alpha_j)
\end{equation}
$$

여기서 $\alpha_k$는 z-depth 순서로 정렬된 effective opacity, $d_k$는 camera에서 $x_k$로의 view direction이다. $L_1$ loss와 SSIM loss로 최적화한다. 이 explicit representation은 학습·렌더링 속도를 크게 높이고 scene을 직접 조작할 수 있게 한다.

### Continuum Mechanics

Continuum mechanics는 undeformed material space $\Omega^0$와 deformed world space $\Omega^t$ 사이의 time-dependent 연속 deformation map $x = \phi(X, t)$로 움직임을 기술한다. deformation gradient $F(X, t) = \nabla_X \phi(X, t)$는 stretch·rotation·shear를 포함한 local transformation을 인코딩한다. deformation의 진화는 질량 보존과 운동량 보존으로 지배된다. 질량 보존은 임의의 미소 영역 안의 mass가 시간에 대해 일정함을 보장한다.

$$
\begin{equation}
\int_{B_\epsilon^t} \rho(x, t) \equiv \int_{B_\epsilon^0} \rho(\phi^{-1}(x, t), 0)
\end{equation}
$$

velocity field를 $v(x, t)$라 하면 운동량 보존은 다음과 같다.

$$
\begin{equation}
\rho(x, t)\dot{v}(x, t) = \nabla \cdot \sigma(x, t) + f^{\text{ext}}
\end{equation}
$$

여기서 $\sigma = \frac{1}{\det(F)}\frac{\partial \Psi}{\partial F}(F^E){F^E}^T$는 hyperelastic energy density $\Psi(F)$에 대응하는 Cauchy stress tensor, $f^{\text{ext}}$는 단위 부피당 external force다. total deformation gradient는 elastic part와 plastic part로 분해된다($F = F^E F^P$). 이는 plasticity로 인한 영구적 rest shape 변화를 지원한다.

### Material Point Method

MPM은 위 지배 방정식을 Lagrangian particle과 Eulerian grid의 강점을 결합해 푼다. Continuum을 particle 집합으로 discretize하고, 각 particle이 position $x_p$·velocity $v_p$·deformation gradient $F_p$를 추적한다. 질량 보존은 Lagrangian particle에서 자연스럽고, 운동량 보존은 Eulerian representation에서 자연스럽다(mesh 구성 불필요). $C^1$ 연속 B-spline kernel로 두 표현 사이를 two-way transfer한다. forward Euler로 discretize한 운동량 보존은 다음과 같다.

$$
\begin{equation}
\frac{m_i}{\Delta t}(v_i^{n+1} - v_i^n) = -\sum_p V_p^0 \frac{\partial \Psi}{\partial F}(F_p^{E,n}){F_p^{E,n}}^T \nabla w_{ip}^n + f_i^{\text{ext}}
\end{equation}
$$

여기서 $i$는 Eulerian grid, $p$는 Lagrangian particle의 field, $w_{ip}^n$은 B-spline kernel, $V_p^0$는 초기 대표 부피, $\Delta t$는 time step이다. 갱신된 grid velocity $v_i^{n+1}$을 particle로 전달해 위치를 $x_p^{n+1} = x_p^n + \Delta t\, v_p^{n+1}$로 갱신한다. $F^E$는 $F_p^{E,n+1} = (I + \Delta t \nabla v_p)F_p^{E,n}$로 갱신하고, plasticity를 위해 return mapping $Z(\cdot)$로 regularize한다.

### Physics-Integrated 3D Gaussians

Gaussian kernel을 simulated continuum을 공간 discretize하는 particle cloud로 취급하고, continuum이 변형되면 Gaussian kernel도 함께 변형시킨다. material space에서 $X_p$에 정의된 Gaussian kernel $G_p(X) = e^{-\frac{1}{2}(X - X_p)^T A_p^{-1}(X - X_p)}$에 deformation map $\phi(X, t)$를 적용하면

$$
\begin{equation}
G_p(x, t) = e^{-\frac{1}{2}(\phi^{-1}(x, t) - X_p)^T A_p^{-1}(\phi^{-1}(x, t) - X_p)}
\end{equation}
$$

인데, 이것은 world space에서 반드시 Gaussian은 아니다(splatting 조건 위반). 다행히 particle이 first-order 근사로 local affine transformation을 겪는다고 가정하면

$$
\begin{equation}
\tilde{\phi}_p(X, t) = x_p + F_p(X - X_p)
\end{equation}
$$

변형된 kernel은 다시 Gaussian이 된다.

$$
\begin{equation}
G_p(x, t) = e^{-\frac{1}{2}(x - x_p)^T (F_p A_p F_p^T)^{-1}(x - x_p)}
\end{equation}
$$

이 변환은 3D GS framework에 time-dependent 버전의 $x_p$와 $A_p$를 자연스럽게 제공한다.

$$
\begin{equation}
x_p(t) = \phi(X_p, t), \qquad a_p(t) = F_p(t)A_p F_p(t)^T
\end{equation}
$$

요약하면, static scene의 3D GS $\{X_p, A_p, \sigma_p, C_p\}$가 주어지면 simulation으로 이를 dynamic Gaussians $\{x_p(t), a_p(t), \sigma_p, C_p\}$로 진화시킨다. opacity와 SH 계수는 시간에 대해 불변으로 가정하되 harmonics는 rotation된다(다음 절). 각 particle의 대표 부피 $V_p^0$는 background cell 부피를 포함된 particle 수로 나눠 초기화하고, mass는 사용자 지정 density $\rho_p$로부터 $m_p = \rho_p V_p^0$로 추론한다. Gaussians 자체가 continuum의 discretization이므로 직접 simulation되고, 변형된 Gaussians는 splatting으로 직접 rendering되어 상용 rendering 소프트웨어가 필요 없다. 무엇보다 real data로 복원한 scene을 직접 simulation하여 WS2를 달성한다.

### Evolving Orientations of Spherical Harmonics

world-space 3D Gaussians를 렌더링하는 것만으로도 고품질 결과를 얻는다. 그러나 물체가 rotation을 겪을 때, spherical harmonic basis는 여전히 material space에서 표현되므로, 물체 기준 view direction이 고정되어도 appearance가 달라지는 문제가 생긴다. 해법은 간단하다. ellipsoid가 회전하면 그 spherical harmonics의 orientation도 함께 회전시킨다. basis는 GS framework 내부에 hard-coded되어 있으므로, view direction에 inverse rotation을 적용해 동일한 효과를 낸다. local rotation은 deformation gradient $F_p$에서 polar decomposition $F_p = R_p S_p$로 얻는다. material space의 SH basis $f^0(d)$에 대해 rotation된 basis는 다음과 같다.

$$
\begin{equation}
f^t(d) = f^0(R^T d)
\end{equation}
$$

### Incremental Evolution of Gaussians

total deformation gradient $F$에 대한 의존을 피하는 대안적 Gaussian kinematics도 제안한다. 이는 $F$를 strain measure로 쓰지 않는 물리 material model로 가는 길을 열어준다. computational fluid dynamics 관례를 따라, world-space covariance $a$의 rate form $\dot{a} = (\nabla v)a + a(\nabla v)^T$를 discretize하면 다음과 같다.

$$
\begin{equation}
a_p^{n+1} = a_p^n + \Delta t(\nabla v_p\, a_p^n + a_p^n \nabla v_p^T)
\end{equation}
$$

이 방식은 $F_p$ 없이 $t^n$에서 $t^{n+1}$로 kernel 모양을 incremental 갱신한다. SH basis의 rotation matrix $R_p$도 유사하게, $R_p^0 = I$에서 시작해 $(I + \Delta t \nabla v_p)R_p^n$을 polar decomposition하여 $R_p^{n+1}$을 추출한다.

### Internal Filling

복원된 Gaussians는 물체 표면 근처에 분포하는 경향이 있어 내부는 비어 있고, volumetric 물체의 거동이 부정확해진다. 내부 빈 영역을 particle로 채우기 위해, 3D Gaussians의 opacity로부터 3D opacity field를 빌려온다.

$$
\begin{equation}
d(x) = \sum_p \sigma_p \exp\left(-\frac{1}{2}(x - x_p)^T A_p^{-1}(x - x_p)\right)
\end{equation}
$$

이 연속 field를 3D grid에 discretize한다. 사용자 지정 threshold $\sigma_{th}$로 intersection(낮은 opacity grid에서 높은 opacity grid로 ray가 통과)을 정의하고, 6축으로 ray를 쏴 intersection을 검사(condition 1)해 candidate grid를 찾은 뒤, 추가 ray로 intersection 수를 평가(condition 2)해 선택을 정교화한다. 채워진 particle은 가장 가까운 Gaussian kernel의 $\sigma_p, C_p$를 상속하고, covariance는 $\text{diag}(r_p^2, r_p^2, r_p^2)$로 초기화한다($r_p = (3V_p^0/4\pi)^{1/3}$).

### Anisotropy Regularizer

Gaussian kernel의 anisotropy는 3D representation 효율을 높이지만, 지나치게 가느다란 kernel은 큰 변형에서 표면 밖으로 삐져나와 plush(털뭉치) artifact를 만든다. 이를 억제하기 위해 3D Gaussian 복원 시 다음 loss를 제안한다.

$$
\begin{equation}
\mathcal{L}_{aniso} = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}}\max\{\max(S_p)/\min(S_p), r\} - r
\end{equation}
$$

$S_p$는 3D Gaussian의 scaling이며, 이 loss는 major axis와 minor axis 길이의 비가 $r$을 넘지 않도록 제약한다.

## Experiments

### Evaluation of Generative Dynamics

Datasets. synthetic sofa suite(BlenderNeRF)에 더해 Instant-NGP의 fox, Nerfstudio의 plane, DroneDeploy NeRF의 ruins를 사용한다. 또 iPhone으로 실제 데이터 두 개(toast, jam)를 수집했고 각 scene당 사진 150장이다. 초기 point cloud와 camera parameter는 COLMAP으로 얻는다.

Simulation Setups. Zong et al.의 MPM을 기반으로 한다. simulation 영역을 수동 선택해 edge 길이 2인 cube로 정규화하고, 필요시 internal filling을 수행한다. 특정 particle의 velocity를 선택적으로 수정해 controlled movement를 유도하고, 나머지는 물리 법칙에 따라 자연스럽게 움직인다. 실험은 Intel i9-10920X + Nvidia RTX 3090에서 수행한다.

![Fig3](/assets/img/physgaussian/fig3_materials.png)
_Fig 3. Material Versatility. fox(elastic entity), plane(plastic metal), toast(fracture), ruins(granular material), jam(viscoplastic material), sofa suite(collision) 등 다양한 재질에서의 versatility. 좌측이 static, 우측이 physics-based dynamics 시퀀스._

Results. 다양한 physics-based dynamics를 simulation한다(Fig 3).

- Elasticity: 변형 중 rest shape이 불변인 성질(가장 단순한 일상 dynamics).
- Metal: 영구적 rest shape 변화를 겪음. von-Mises plasticity model을 따름.
- Fracture: MPM에서 자연스럽게 지원됨. 큰 변형이 particle을 여러 그룹으로 분리.
- Sand: Drucker-Prager plasticity model. 입자 수준의 frictional 효과를 포착.
- Paste: viscoplastic non-Newtonian fluid로 모델링. Herschel-Bulkley plasticity model.
- Collision: grid time integration으로 자동 처리되는 MPM의 핵심 기능.

Explicit MPM은 GPU에서 고도로 최적화 가능하며, 1/24초 frame duration 기준 일부 케이스는 real-time을 달성한다: plane(30 FPS), toast(25 FPS), jam(36 FPS). FEM으로 elasticity simulation을 더 빠르게 할 수 있지만, mesh 추출 단계가 추가되고 inelasticity에서 MPM의 일반성을 잃는다.

각 demo scene이 어떤 constitutive model로 simulation되는지는 아래와 같다.

| Scene | Figure | Constitutive Model |
|---|---|---|
| Vasedeck | Fig 1 | Fixed corotated |
| Ficus | Fig 2 | Fixed corotated |
| Fox | Fig 3 | Fixed corotated |
| Plane | Fig 3 | von Mises |
| Toast | Fig 3 | Fixed corotated |
| Ruins | Fig 3 | Drucker-Prager |
| Jam | Fig 3 | Herschel-Bulkley |
| Sofa Suite | Fig 3 | Fixed corotated |
| Microphone | Fig 7 | Neo-Hookean |
| Bread / Cake / Can / Wolf | Fig 9 | Fixed corotated / Herschel-Bulkley / von Mises / Drucker-Prager |

_Table 2. Model Settings. 각 장면에 적용한 constitutive(물리) 모델._

constitutive model에서 쓰이는 탄성 파라미터의 정의와 $E, \nu$로부터의 관계는 다음과 같다.

| 표기 | 의미 | 관계 |
|---|---|---|
| $E$ | Young's modulus (강성) | — |
| $\nu$ | Poisson's ratio | — |
| $\mu$ | Shear modulus | $\mu = \frac{E}{2(1+\nu)}$ |
| $\lambda$ | Lamé modulus | $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ |
| $\kappa$ | Bulk modulus | $\kappa = \frac{E}{3(1-2\nu)}$ |

_Table 3. Material Parameters. 탄성 파라미터 표기와 $E$·$\nu$로부터 유도되는 관계._

### Lattice Deformation Benchmarks

Dataset. post-deformation의 ground truth가 없으므로, BlenderNeRF로 여러 scene을 합성해 lattice deformation tool로 bending·twisting을 적용한다. 각 scene마다 undeformed 상태의 multi-view rendering 100장(학습용)과 각 deformed 상태 100장(ground truth)을 만든다.

Comparisons. 수동 deformation을 지원하는 SOTA NeRF framework와 비교한다. NeRF-Editing(추출한 surface mesh로 deform), Deforming-NeRF(cage mesh 이용), PAC-NeRF(초기 particle 조작).

![Fig4](/assets/img/physgaussian/fig4_comparisons.png)
_Fig 4. Comparisons. 각 benchmark 케이스마다 한 test 시점을 선택해 모든 비교를 시각화. 배경 간섭을 피하려 검은 배경 사용. 좌→우: Ground Truth / Ours / Deforming-NeRF / NeRF-Editing / PAC-NeRF._

NeRF-Editing은 scene representation으로 NeuS를 써서 surface reconstruction에는 적합하나 rendering 품질은 3DGS보다 낮고, deformation이 추출한 surface mesh와 dilated cage mesh의 정밀도에 크게 의존한다. Deforming-NeRF는 깨끗한 rendering을 주지만 모든 cage vertex에서 smooth interpolation을 하므로 fine local detail이 사라지고 lattice deformation을 정확히 맞추지 못한다. PAC-NeRF는 단순 물체·텍스처의 system identification용이라 rendering fidelity가 낮다. 본 방법은 각 lattice cell의 zero-order(deformation map)와 first-order(deformation gradient) 정보를 모두 활용하여 모든 케이스에서 최고 성능이다(Table 1).

| Test Case | Wolf (Bend) | Wolf (Twist) | Stool (Bend) | Stool (Twist) | Plant (Bend) | Plant (Twist) |
|---|---|---|---|---|---|---|
| NeRF-Editing | 26.74 | 24.37 | 25.00 | 21.10 | 19.85 | 19.08 |
| Deforming-NeRF | 21.65 | 21.72 | 22.32 | 21.16 | 17.90 | 18.63 |
| PAC-NeRF | 26.91 | 25.27 | 21.83 | 21.26 | 18.50 | 17.78 |
| Fixed Covariance | 26.77 | 26.02 | 29.94 | 25.31 | 23.95 | 23.09 |
| Rigid Covariance | 26.84 | 26.16 | 30.28 | 25.70 | 24.09 | 23.53 |
| Fixed Harmonics | 26.83 | 26.02 | 30.87 | 25.75 | 25.09 | 23.69 |
| **Ours** | **26.96** | **26.46** | **31.15** | **26.15** | **25.81** | **23.87** |

_Table 1. Lattice deformation benchmark의 PSNR(높을수록 좋음). Ours가 모든 케이스에서 우위._

Ablation Studies. Gaussian kernel과 spherical harmonics kinematics의 필요성을 검증한다. Fixed Covariance(kernel을 translation만), Rigid Covariance(rigid transformation만), Fixed Harmonics(SH orientation을 회전하지 않음)와 비교한다.

![Fig5](/assets/img/physgaussian/fig5_ablation.png)
_Fig 5. Ablation Studies. Non-extensible Gaussians는 변형 중 severe artifact를 유발한다. 변형된 kernel을 직접 렌더링해도 좋은 결과를 얻지만, spherical harmonics에 rotation을 추가하면 rendering 품질이 향상된다. 좌→우: Ground Truth / Ours / Fixed Cov. / Rigid Cov. / Fixed Harmonics._

kernel이 non-extensible하면 변형 후 표면을 제대로 덮지 못해 심각한 artifact가 생긴다. SH rotation을 켜면 ground truth와의 일관성이 조금 향상된다. Table 1이 이 모든 요소가 최고 성능에 필요함을 보인다.

### Additional Qualitative Studies

Internal Filling. 3DGS는 표면 appearance에 집중하고 내부 구조를 놓쳐 물체 내부가 껍데기처럼 비게 된다. 재구성한 density field 기반 internal filling으로 이를 해결한다. 내부 particle이 없는 물체는 material parameter와 무관하게 중력에 무너지지만, internal filling을 적용하면 재질 특성에 맞는 세밀한 dynamics 제어가 가능하다.

![Fig6](/assets/img/physgaussian/fig6_filling.png)
_Fig 6. Internal filling은 더 사실적인 simulation을 가능케 한다. 큰 Young's modulus $E$는 높은 stiffness를, 큰 Poisson ratio $\nu$는 더 나은 volume preservation을 의미한다._

Volume Conservation. 기존 NeRF 조작은 물리 특성 없이 geometric 조정에 치중한다. 실제 물체의 핵심 속성인 volume 보존을 본 방법은 정확히 포착한다. surface As-Rigid-As-Possible(ARAP) deformation을 쓰는 NeRF-Editing과 달리, 본 방법은 변형된 물체의 부피를 유지한다.

![Fig7](/assets/img/physgaussian/fig7_volume.png)
_Fig 7. Volume Conservation. geometry-based editing(NeRF-Editing)과 달리 물리 기반 방법인 Ours는 volumetric behavior를 포착해 더 사실적인 dynamics를 만든다. 늘렸을 때(stretch) Ours는 부피 보존으로 가운데가 잘록해지는 반면 NeRF-Editing은 그대로 늘어난다._

Anisotropy Regularizer. 지나치게 slender한 Gaussian kernel은 큰 변형에서 burr(가시) artifact를 만든다. Eq. (12)의 anisotropy 제약 loss가 이 artifact를 효과적으로 완화한다.

![Fig8](/assets/img/physgaussian/fig8_anisotropy.png)
_Fig 8. Anisotropy Regularizer. slender한 kernel이 유발하는 burr artifact를, anisotropy 제약 loss로 억제한 결과 비교._

## Conclusion

PhysGaussian은 physics-based dynamics와 photo-realistic rendering을 동시에·seamless하게 생성하는 통합 simulation-rendering pipeline이다. 2D 입력을 별도로 전처리하는 대신 3D 모델 생성에 물리를 직접 통합하고, 3D Gaussian이라는 하나의 표현으로 simulation과 rendering을 공유해 embedding·meshing의 불일치를 제거했다.

### Limitation

shadow의 진화는 고려되지 않으며 material parameter는 수동 설정된다. GS segmentation과 differentiable MPM simulator를 결합해 video로부터 parameter를 자동 할당하는 방향, geometry-aware 3DGS 복원 결합, liquid 등 더 다양한 재질과 LLM 기반의 직관적 user control이 future work로 제시된다.
