---
title: "[논문리뷰] 3D Gaussian Splatting for Real-Time Radiance Field Rendering"
date: 2026-07-20 13:00:00 +0900
categories: [논문리뷰, 3D Vision]
tags: [3D Gaussian Splatting, Radiance Field, Real-Time Rendering]
---

## 논문 정보

- 제목: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- 저자: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria)
- 학회: SIGGRAPH 2023
- 링크: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>

## 요약

3D 장면을 수백만 개의 3D 가우시안으로 표현하고, 이를 화면에 투영·정렬해 빠르게 렌더링(splatting)하는 방법이다. NeRF 계열이 볼륨 렌더링으로 느렸던 것과 달리, 미분 가능한 래스터라이저를 사용해 실시간(1080p 기준 100+ FPS) 렌더링을 달성하면서도 최고 수준의 화질을 낸다.

## 핵심 아이디어

각 가우시안은 위치, 공분산(모양·방향), 불투명도, 색상(구면 조화 계수)을 파라미터로 갖는다. SfM 포인트로 초기화한 뒤, 학습 과정에서 가우시안을 분할·복제·제거하며 장면에 적응적으로 밀도를 조절한다.

## 리뷰

여기에 개인적인 코멘트와 인상 깊었던 점을 적습니다.
