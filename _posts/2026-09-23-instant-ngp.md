---
title: "[논문리뷰] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
date: 2026-09-23 04:20:00 +0900
categories: [논문리뷰, Computer Graphics]
tags: [Neural Graphics Primitives, Multiresolution Hash Encoding, Neural Rendering, NeRF, Signed Distance Function, CUDA, SIGGRAPH]
description: "SIGGRAPH 2022"
math: true
toc: true
compact_captions: true
---

> \[[Paper](https://arxiv.org/abs/2201.05989)\] \[[Page](https://nvlabs.github.io/instant-ngp/)\] \[[Github](https://github.com/NVlabs/instant-ngp)\]
>
> Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller

## Introduction

Neural graphics primitive는 image, signed distance function(SDF), radiance field처럼 graphics에서 사용하는 함수를 fully connected neural network로 표현한다. 이러한 representation은 continuous하고 compact하지만, high-frequency local detail을 학습하려면 큰 MLP와 긴 optimization 시간이 필요하다. 특히 NeRF는 positional encoding을 사용하더라도 ray마다 network를 반복 평가하므로 training과 rendering cost가 크다.

논문은 작은 MLP 앞에 trainable multiresolution hash encoding을 배치한다. Scene의 detail은 여러 해상도의 hash table에 저장하고, MLP는 interpolated feature들을 결합하여 최종 출력을 예측한다. Hash table은 tree처럼 pruning, splitting, merging을 수행하지 않으며 lookup에 control flow가 필요하지 않다. 저자들은 이를 fully-fused CUDA implementation과 결합하여 single GPU에서 neural graphics primitive를 수 초 안에 학습한다.

![여러 neural graphics primitive의 빠른 학습 결과](/assets/img/instant-ngp/fig1_overview.png)
_Fig 1. Multiresolution hash encoding을 gigapixel image, SDF, neural radiance caching(NRC), NeRF에 적용한 결과. 각 행은 짧은 학습 시간에도 reference의 local detail에 빠르게 접근하는 과정을 보여준다._

이 논문의 대상은 NeRF 하나로 한정되지 않는다. 동일한 encoding과 대부분의 hyperparameter를 다음 네 task에 적용하고, task에 따라 hash table 크기만 조절한다.

1. 2D coordinate를 RGB color로 mapping하는 gigapixel image approximation
2. 3D coordinate를 surface까지의 signed distance로 mapping하는 neural SDF
3. Path tracer의 lighting calculation을 예측하는 neural radiance caching
4. Image와 camera pose로부터 density와 view-dependent radiance를 학습하는 NeRF

논문의 기여는 task-independent한 multiresolution hash encoding, hash collision을 network optimization에 포함하는 representation, 그리고 GPU의 cache와 parallelism을 고려한 implementation으로 정리된다. 이 결합은 high-quality neural graphics primitive의 training 시간을 수시간에서 수초 또는 수분으로 단축한다.

## Related Work

### Frequency Encodings

MLP는 raw coordinate에서 high-frequency function을 학습하는 데 어려움이 있다. Positional encoding은 scalar coordinate $x$를 여러 frequency의 sine과 cosine으로 변환하여 network가 세부 변화를 표현하도록 한다.

$$
\begin{equation}
\operatorname{enc}(x)=
\left(
\sin(2^0x),\ldots,\sin(2^{L-1}x),
\cos(2^0x),\ldots,\cos(2^{L-1}x)
\right).
\tag{1}
\end{equation}
$$

NeRF는 spatial coordinate와 view direction을 이러한 frequency encoding으로 변환한다. Encoding 자체에는 trainable parameter가 없으므로 scene의 detail은 MLP weights에 저장된다. 따라서 높은 approximation quality를 위해 비교적 큰 network가 필요하다.

### Parametric Encodings

Parametric encoding은 MLP 외부의 grid 또는 tree에 trainable feature vector를 저장한다. Input coordinate 주변의 feature만 lookup하고 interpolation하므로 하나의 sample이 update하는 parameter 수는 작다. 더 많은 parameter를 사용하면서도 MLP를 줄일 수 있어 dense MLP보다 빠르게 수렴한다.

Dense feature grid는 전체 domain에 같은 양의 memory를 할당한다. 하지만 3D scene의 visible surface는 전체 volume에서 작은 부분만 차지하므로 empty space에도 feature가 저장된다. Grid resolution이 $N$일 때 storage는 $O(N^3)$로 증가하지만 surface area는 대체로 $O(N^2)$로 증가한다.

### Sparse Parametric Encodings

Octree와 sparse voxel grid는 occupied region에만 feature를 할당하여 dense grid의 낭비를 줄인다. 그러나 NeRF에서는 geometry가 training 중에 나타나므로 coarse-to-fine refinement와 periodic pruning 같은 구조 변경이 필요하다. 이러한 update는 training procedure를 복잡하게 만들고, GPU에서는 branch divergence와 pointer chasing을 발생시킬 수 있다.

Instant-NGP는 여러 resolution의 dense grid vertex를 고정 크기 array에 mapping한다. Coarse level은 collision-free dense grid로 동작하고, fine level은 spatial hash table로 동작한다. 별도의 sparse structure를 유지하지 않으면서 memory를 제한하고 O(1) lookup을 제공한다.

## Method

### Preliminaries

Coordinate-based neural representation을 $m(\mathbf{y};\Phi)$라 하자. 기존 frequency encoding에서는 입력 coordinate $\mathbf{x}$를 fixed function으로 변환하지만, 이 논문은 trainable parameter $\theta$를 가진 encoding을 사용한다.

$$
\mathbf{y}=\operatorname{enc}(\mathbf{x};\theta),
\qquad
\hat{\mathbf{v}}=m(\mathbf{y};\Phi).
$$

$\theta$는 multiresolution hash table의 feature vector이고, $\Phi$는 작은 MLP의 weight다. 두 parameter set은 task loss에서 전달되는 gradient로 함께 학습된다.

![서로 다른 input encoding의 NeRF reconstruction 비교](/assets/img/instant-ngp/fig2_encoding_comparison.png)
_Fig 2. 같은 NeRF implementation에서 input encoding과 MLP 크기를 바꾼 비교. Hash table $T=2^{14}$는 frequency encoding과 비슷한 parameter 수로 8배 이상 빠르게 학습하고, $T=2^{19}$는 training time을 거의 늘리지 않으면서 가장 높은 PSNR을 얻는다._

No encoding을 사용한 MLP는 smooth한 approximation만 학습한다. Frequency encoding은 detail을 개선하지만 큰 MLP가 필요하다. Single-resolution dense grid는 빠르지만 multi-scale structure를 활용하지 못하며, multiresolution dense grid는 높은 품질을 얻는 대신 많은 parameter를 사용한다. Multiresolution hash encoding은 fixed memory budget 안에서 여러 spatial scale의 feature를 제공한다.

### Multiresolution Hash Encoding

Encoding은 $L$개의 독립적인 level로 구성된다. 각 level은 최대 $T$개의 $F$-dimensional feature vector를 저장한다. Level $l$의 grid resolution은 coarsest resolution과 finest resolution 사이의 geometric progression으로 정한다.

$$
\begin{equation}
N_l:=\left\lfloor N_{\min}\cdot b^l\right\rfloor.
\tag{2}
\end{equation}
$$

$$
\begin{equation}
b:=\exp\left(
\frac{\ln N_{\max}-\ln N_{\min}}{L-1}
\right).
\tag{3}
\end{equation}
$$

논문의 기본 설정은 $L=16$, $F=2$, $$N_{\min}=16$$이다. Task에 따라 hash table 크기 $T$는 $2^{14}$부터 $2^{24}$, finest resolution $$N_{\max}$$는 512부터 524288 사이에서 선택한다. 실제로 조절해야 하는 주요 값은 $T$와 $$N_{\max}$$이며, 논문의 task에서 geometric growth factor $b$는 1.26부터 2 사이이다.

Input coordinate $\mathbf{x}\in\mathbb{R}^d$를 level resolution으로 scale하면 $\mathbf{x}_l=\mathbf{x}N_l$을 얻는다. $\lfloor\mathbf{x}_l\rfloor$와 $\lceil\mathbf{x}_l\rceil$이 만드는 voxel에는 $2^d$개의 integer vertex가 있다. 각 vertex coordinate를 level별 feature array의 index로 mapping한다.

Coarse level에서 필요한 entry 수가 $T$보다 작으면 모든 vertex를 서로 다른 entry에 대응시킨다. Fine level에서는 spatial hash function을 사용한다.

$$
\begin{equation}
h(\mathbf{x})=
\left(
\bigoplus_{i=1}^{d} x_i\pi_i
\right)\bmod T,
\tag{4}
\end{equation}
$$

$\oplus$는 bit-wise XOR이고, $\pi_i$는 dimension마다 다른 큰 prime number다. 구현에서는 cache coherence를 위해 $\pi_1=1$을 사용하고, 3D coordinate의 나머지 값으로 2654435761과 805459861을 사용한다.

![Multiresolution hash encoding의 처리 과정](/assets/img/instant-ngp/fig3_hash_encoding_pipeline.png)
_Fig 3. 입력 coordinate에서 voxel vertex hashing, feature lookup, linear interpolation, level별 feature concatenation, MLP evaluation으로 이어지는 multiresolution hash encoding pipeline._

각 level에서 lookup한 corner feature들은 voxel 내부의 상대 위치
$\mathbf{w}_l=\mathbf{x}_l-\lfloor\mathbf{x}_l\rfloor$에 따라 $d$-linear interpolation된다. $L$개 level의 interpolated feature와 auxiliary input $\boldsymbol{\xi}\in\mathbb{R}^E$를 concatenate하여 MLP input을 만든다.

$$
\mathbf{y}\in\mathbb{R}^{LF+E}.
$$

Auxiliary input에는 NRC의 material feature나 NeRF의 encoded view direction처럼 spatial hash encoding의 대상이 아닌 값이 들어간다. Gradient는 MLP, concatenation, interpolation을 거쳐 해당 sample에서 lookup한 feature vector에만 누적된다.

### Implicit Hash Collision Resolution

논문은 collision을 probing, bucket, chaining으로 명시적으로 해결하지 않는다. Coarse level은 collision이 없어 low-frequency structure를 안정적으로 구분하고, fine level은 collision을 허용하면서 high-frequency detail을 저장한다. 서로 다른 두 point가 모든 level에서 동시에 충돌할 가능성은 낮다.

같은 entry를 참조하는 sample의 gradient는 평균된다. Visible surface처럼 loss에 크게 기여하는 point는 empty space보다 큰 gradient를 만들기 때문에, 중요한 detail이 collision update를 지배한다. 이후의 MLP는 여러 level에서 얻은 feature를 함께 해석하여 남은 ambiguity를 줄인다.

Input distribution이 특정 region으로 집중되면 그 region이 사용하는 fine-level entry의 collision도 감소한다. 따라서 explicit tree update 없이 training data의 분포에 맞춰 representation capacity가 이동한다.

### Linear Interpolation and Continuity

Feature를 interpolation하지 않고 직접 lookup하면 grid boundary에 discontinuity가 생긴다. 논문은 각 level에서 $d$-linear interpolation을 사용하여 encoding과 network output을 continuous하게 만든다. 하지만 derivative까지 continuous하지는 않으므로 SDF surface normal에 작은 discontinuity가 남을 수 있다.

Appendix에서는 interpolation weight에 smoothstep을 적용하는 방법을 제시하지만, 모든 실험은 reconstruction quality가 조금 낮아지는 것을 피하기 위해 기본 linear interpolation을 사용한다.

### Implementation

Encoding은 CUDA로 구현되고 tiny-cuda-nn의 fully-fused MLP와 결합된다. Hash table entry는 half precision으로 저장하고, stable parameter update를 위해 full-precision master copy를 유지한다.

Batch의 모든 input에 대해 첫 번째 level을 처리한 뒤 다음 level로 이동하는 level-wise execution을 사용한다. 이 순서는 동시에 필요한 hash table 영역을 줄여 GPU cache utilization을 높인다. RTX 3090의 6 MB L2 cache에서는 $T\leq2^{19}$까지 encoding performance가 대체로 일정하고, 이 크기를 넘으면 cache oversubscription으로 속도가 감소한다.

NeRF를 제외한 task의 MLP는 width 64의 hidden layer 두 개를 사용한다. Adam의 parameter는 $\beta_1=0.9$, $\beta_2=0.99$, $\epsilon=10^{-15}$이며, gradient가 정확히 0인 hash entry에는 Adam update를 수행하지 않는다.

### NeRF Architecture and Ray Marching

NeRF model은 density MLP와 color MLP로 구성된다. Density MLP는 hash-encoded position을 16개 값으로 변환하고, 첫 번째 값을 log-space density로 사용한다. Color MLP는 이 16개 값과 degree 4까지의 spherical harmonics로 표현한 view direction을 받아 RGB를 출력한다. Density MLP는 width 64의 hidden layer 한 개, color MLP는 같은 width의 hidden layer 두 개를 사용한다.

NeRF의 속도에는 encoding 외의 ray-marching optimization도 포함된다. 논문은 occupancy grid로 empty space를 건너뛰고, large scene에서는 distance에 비례해 step size를 늘린다. Ray transmittance가 $10^{-4}$보다 작아지면 march를 종료하며, 유효 sample을 dense buffer로 compact하여 network를 효율적으로 실행한다.

## Experiments

### Hash Table Size and Encoding Configuration

Hash table 크기 $T$는 memory, quality, performance의 직접적인 trade-off를 제공한다. $T$가 커지면 collision이 감소해 품질은 높아지지만 memory usage가 증가하고, GPU cache를 넘은 뒤에는 training과 inference가 느려진다.

![Hash table 크기에 따른 training time과 reconstruction quality](/assets/img/instant-ngp/fig4_hash_size_tradeoff.png)
_Fig 4. Gigapixel image, SDF, NeRF에서 hash table 크기 $T$에 따른 test error와 training time. 큰 $T$는 품질을 높이지만 RTX 3090의 cache가 포화되는 $T>2^{19}$ 부근에서 performance가 급격히 감소한다._

Feature dimension $F$와 level 수 $L$을 비슷한 parameter budget에서 비교한 결과, $F=2$와 $L=16$이 네 task 전반에서 quality-performance Pareto optimum에 가까웠다. 논문은 이를 기본 설정으로 사용한다.

### Gigapixel Image Approximation

Gigapixel image task는 2D coordinate를 RGB color로 mapping한다. Tokyo panorama에서 ACORN은 36.9시간 후 PSNR 38.59 dB를 기록한다. Hash table 크기 $T=2^{24}$를 사용한 본 방법은 2.5분 만에 같은 PSNR에 도달하고, 4분 후 41.9 dB를 기록한다.

저자들은 fully-fused CUDA kernel이 약 10배의 차이를 만들고, 더 작은 MLP를 사용할 수 있게 하는 input encoding이 나머지 속도 향상의 큰 부분을 만든다고 설명한다. 또한 ACORN이 adaptive subdivision과 learning curriculum을 사용하는 것과 달리, hash encoding에는 training 중의 structure update가 필요하지 않다.

### Signed Distance Functions

SDF에서는 multiresolution hash encoding을 frequency encoding과 NGLOD에 비교한다. NGLOD는 ground-truth mesh에 맞춘 octree를 사용하므로 가장 높은 visual quality를 보인다. Hash encoding은 비슷한 parameter 수에서 NGLOD에 가까운 IoU를 기록하고, octree 내부뿐 아니라 training volume 전체에서 SDF를 평가할 수 있다.

![SDF reconstruction의 frequency encoding, NGLOD 및 hash encoding 비교](/assets/img/instant-ngp/fig7_sdf_comparison.png)
_Fig 7. 11,000 step 동안 학습한 neural SDF 비교. Hash encoding은 NGLOD와 비슷한 IoU를 얻지만 random hash collision에서 비롯된 surface roughness가 나타난다._

정성 결과에서는 finest grid scale의 grainy microstructure가 관찰된다. 논문은 collision-free analog인 NGLOD에는 같은 artifact가 없다는 점을 근거로 hash collision을 원인으로 본다. 이 artifact는 training 시간을 늘려도 사라지지 않는다.

### Neural Radiance Caching

NRC는 real-time path tracer가 생성하는 sparse light path로부터 online supervision을 받는다. Camera와 scene이 움직이는 동안 network는 현재 관측되는 shape과 lighting에 계속 적응해야 하며, training budget은 frame당 1 ms이다.

Multiresolution hash encoding은 기존 triangle-wave encoding보다 intricate shadow와 close-up detail을 선명하게 학습한다. 1920×1080에서 triangle-wave encoding은 147 FPS, hash encoding은 133 FPS를 기록한다. 약 0.7 ms의 overhead에는 encoding의 training과 inference가 모두 포함된다.

### Neural Radiance and Density Fields

NeRF 실험은 synthetic NeRF dataset의 Mic, Ficus, Chair, Hotdog, Materials, Drums, Ship, Lego를 사용한다. 입력은 RGB image와 known camera pose이며, differentiable ray marcher를 통해 density와 view-dependent color를 학습한다.

![Synthetic NeRF dataset의 PSNR 비교](/assets/img/instant-ngp/table2_nerf_psnr.png)
_Table 2. Multiresolution hash encoding의 1초부터 5분까지의 PSNR과 NeRF, mip-NeRF, NSVF 및 optimized frequency-encoding baseline 비교._

Ours: Hash의 8-scene 평균 PSNR은 1초에 21.202 dB, 5초에 29.261 dB, 15초에 31.407 dB, 1분에 32.635 dB, 5분에 33.176 dB다. 논문에 인용된 수시간 학습 결과는 mip-NeRF 33.090 dB, NSVF 31.739 dB, NeRF 31.005 dB이다.

Hash encoding은 Ficus, Drums, Ship, Lego처럼 geometric detail이 많은 scene에서 높은 결과를 얻는다. 반면 Materials처럼 복잡한 view-dependent reflection이 있는 scene에서는 mip-NeRF와 NSVF가 더 높은 PSNR을 기록한다. 논문은 빠른 실행을 위해 사용한 작은 MLP가 복잡한 reflection을 표현하는 데 불리하다고 설명한다.

![Instant-NGP로 학습한 실제 장면 NeRF](/assets/img/instant-ngp/fig12_nerf_results.png){: width="520" }
_Fig 12. Modular synthesizer와 large natural 360 scene의 NeRF reconstruction. 왼쪽 결과는 1080p에서 128 samples를 5초 동안 accumulate했으며, 오른쪽 장면은 같은 RTX 3090에서 10 FPS로 interactive rendering된다._

같은 optimized implementation에서 hash encoding을 frequency encoding으로 바꾸고 MLP를 원래 NeRF와 비슷한 크기로 늘린 baseline은 5분 후 평균 30.056 dB를 기록한다. Ours: Hash는 5–15초 만에 이를 넘어선다. 논문은 이 차이를 hash encoding과 작은 MLP에서 얻은 20–60배의 improvement로 해석한다. 전체적인 orders-of-magnitude speedup에는 fully-fused CUDA implementation과 accelerated ray marching도 함께 기여한다.

### Limitations

Hash collision은 SDF에서 가장 뚜렷한 grainy microstructure를 만든다. MLP가 collision을 완전히 보상하지 못하며, 논문은 filtered lookup이나 additional smoothness prior가 필요할 수 있다고 설명한다.

Generative model은 보통 regular dense grid의 feature를 CNN으로 생성한다. Hash encoding의 feature는 spatial point와 bijective한 regular arrangement를 갖지 않으므로, 별도의 generator로 hash feature를 생성하는 setting에는 추가적인 설계가 필요하다.

NeRF에서는 작은 MLP가 complex view-dependent reflection의 품질을 제한한다. 또한 hash table 크기가 GPU cache capacity를 넘으면 performance가 급격히 감소하므로, $T$를 늘려 얻는 품질 향상에는 hardware-dependent한 비용이 따른다.

## Conclusion

Instant Neural Graphics Primitives는 low-dimensional coordinate function을 위한 trainable multiresolution hash encoding을 제안한다. Coarse level의 collision-free structure와 fine level의 hashed detail을 결합하고, interpolation된 feature를 작은 MLP가 해석하도록 하여 fixed memory budget에서 high-frequency function을 빠르게 학습한다.

이 encoding은 task-specific pruning이나 tree update 없이 gigapixel image, SDF, neural radiance caching, NeRF에 동일하게 적용된다. Fully-fused CUDA kernel과 GPU-friendly memory access, 작은 MLP, task별 rendering optimization을 결합하여 single GPU에서 수초 단위의 training과 interactive rendering을 달성한다. 실험은 encoding이 빠른 convergence를 제공함을 보여주는 동시에, hash collision에 의한 surface microstructure와 작은 MLP의 view-dependent appearance 한계를 함께 제시한다.
