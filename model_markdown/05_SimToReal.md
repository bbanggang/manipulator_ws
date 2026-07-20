# 05. Sim-to-Real — 4모델별 구체 실행 계획 (Isaac Sim, SO-101)

> 확정 결정(D1~D7, 2026-07-20 — [setup/sim/README.md](../setup/sim/README.md) §3) 기반의
> **모델별 실행 절차서**. 각 결정·설계의 근거(웹/논문/자체 실측)를 명시한다.
> 전제: GR00T 실기 완결 후 착수(D7). 실기 실험에서 확정된 처방을 sim에도 선적용한다.

---

## 0. 공통 기반

### 0.1 스택과 근거

| 요소 | 선택 | 근거 |
|---|---|---|
| 시뮬레이터 | Isaac Sim + Isaac Lab (Docker) | SO-101이 Isaac Sim [공식 내장 자산](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html); [공식 워크숍](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)이 RTX 5090(Blackwell) 테스트 완료 |
| 태스크 | `Lerobot-So101-Teleop-Vials-To-Rack(-DR)` → 이후 T1 유사 환경 | D1. 워크숍이 씬·DR·평가 환경([-Eval/-DR-Eval](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop))을 완비 |
| 데이터 수집 | 실물 리더암 teleop, 50ep, DR 켬 | D3·D4. 리더암 teleop은 워크숍 `lerobot_agent` 공식 지원. 50ep은 SmolVLA 논문의 task당 50 demo 프로토콜([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) 및 우리 실기 v2/v3와 규모 일치(비교 가능성). DR은 sim-to-real 전이의 표준 처방([Tobin et al. 2017, arXiv:1703.06907](https://arxiv.org/abs/1703.06907)) |
| 데이터 포맷 | LeRobot 포맷 (sim 출력) | 워크숍 `lerobot_push_dataset` 제공 → **4모델 기존 학습 파이프라인 재사용** (선정의 결정 근거) |
| 수집 규칙 | 성공 종결·idle 최소화·위치 다양화 | **자체 실측**: GR00T idle 어트랙터([report/GR00T_report.md](../report/GR00T_report.md) §3 — 시연 초반 2.8s 정지가 배포 시 정지 모드로 복제됨), SmolVLA 위치 다양화 효과([report/SmolVLA_report.md](../report/SmolVLA_report.md) §11 — T2 v3 개선) |
| 전이 실험 | zero-shot + co-training 둘 다 | D6. sim+real co-training이 실데이터 단독 대비 평균 +38% ([arXiv:2503.24361](https://arxiv.org/abs/2503.24361)) |

### 0.2 데이터 흐름

```
[로컬] sim teleop(리더암) → LeRobot 데이터셋 → HF Hub(private)
  ├─ ACT·SmolVLA 학습 (로컬 5070Ti)
  └─ rsync/Hub → 5090: π0 학습, GR00T(v3→v2 변환 후) 학습
```

- sim 데이터가 v3 포맷이면 GR00T용만 v2 변환: `setup/gr00t/`의 변환·stats수정·트리밍 도구 재사용
- **idle 트리밍을 전 모델 데이터에 공통 적용** 검토 (GR00T에서 실증, ACT/SmolVLA도 이득 가능)

### 0.3 sim 평가 공통 인프라 (D5)

- **GR00T**: 워크숍 `lerobot_eval` + `-Eval`/`-DR-Eval` 환경 (공식 경로)
- **ACT/SmolVLA/π0**: lerobot `async_inference.policy_server`가 세 정책 모두 지원
  (**자체 확인**: lerobot 0.6.1 `SUPPORTED_POLICIES = [act, smolvla, ..., pi0, ...]`)
  → 정책 서버는 컨테이너 밖(5070Ti 또는 5090), sim 컨테이너 안에 **공용 클라이언트 어댑터 1개**
  (워크숍 `lerobot_eval`의 GR00T 클라이언트 자리를 lerobot gRPC 프로토콜로 교체)
- 지표: sim SR n회(초기 위치·시드 변화), 실기와 동일하게 10회 기준

---

## 1. ACT (from-scratch, 로컬 5070Ti)

**계획**
1. sim 50ep로 `lerobot-train --policy.type=act` (실기와 동일 레시피: batch 8 / 100k steps —
   `setup/act/train_act_t1.sh` 변형)
2. sim 평가: policy_server(ACT, 로컬) + sim 어댑터 → SR 10회
3. 실기 전이: 기존 `eval_act_*.sh` 경로로 zero-shot 10회
4. co-training: sim 50ep + 실기 50ep 혼합 재학습 → 실기 재측정

**모델별 유의점과 근거**
- ACT는 언어 미사용·단일 태스크 암기형이라([Zhao et al. 2023, arXiv:2304.13705](https://arxiv.org/abs/2304.13705))
  **시각 도메인 갭에 가장 취약**할 것으로 예상 — DR 데이터 수집이 특히 중요
- 실기 실측에서 ACT는 T1 80%로 최강 기준선([report/ACT_report.md](../report/ACT_report.md)) —
  sim-to-real에서도 기준선 역할
- 카메라 키: sim 데이터셋의 키를 그대로 학습하므로 별도 매핑 불필요 (ACT는 임의 키 허용)

## 2. SmolVLA (FT, 로컬 5070Ti)

**계획**
1. `smolvla_base` FT: batch 16 / steps는 **epoch ≈ 20~25 기준으로 산정** (steps = epoch × frames / batch)
2. sim 평가: policy_server(SmolVLA) + 공용 어댑터 → SR 10회
3. 실기 전이 zero-shot 10회 → co-training 재학습 → 재측정

**모델별 유의점과 근거**
- steps를 epoch 기준으로 잡는 이유: **자체 실측** — 동일 steps에서 데이터가 커지면 epoch이
  반감되어 성능 급락(T1 v2 30%), epoch 통제(v3)로 원인 분리함
  ([report/SmolVLA_report.md](../report/SmolVLA_report.md) §10~11). sim 데이터 프레임 수 확인 후 steps 계산
- FT 레시피 근거: SmolVLA 논문 50-demo 프로토콜 + 커뮤니티 SO-101 사례(ggando.com, RTX3090/batch64/20k)
- 언어 명령은 sim 태스크 문장으로 고정(단일 문장 — 실기와 동일 관례)

## 3. π0 (expert-only FT, 5090)

**계획**
1. `lerobot/pi0_base`에서 **`train_expert_only=true` + `freeze_vision_encoder=true`** FT
   (`setup/pi0/remote_5090/pi0-train.sh` 변형, bf16 + grad checkpointing, batch 8 / 50k)
2. sim 평가: 기존 **async 정책 서버**(5090, 포트 8080) + sim 어댑터
3. 실기 전이: 기존 `infer_pi0_t1_remote_ft.sh` 경로 (클램프·계측 로깅 포함)
4. co-training 재학습 → 재측정

**모델별 유의점과 근거**
- expert-only 선택 근거: **자체 실측** — LoRA(1.4M)는 loss 0.074 정체·배회, expert-only(578M)는
  0.04까지 하락하며 부분 인식·접근 도달 ([report/Pi0_report.md](../report/Pi0_report.md) §3~4).
  π0.5 공식 문서도 동일 방식을 메모리 절감 경로로 권장([HF docs pi05](https://huggingface.co/docs/lerobot/pi05))
- **주의**: 실기에서 π0는 frozen vision으로 과제 완결에 미달했음 — sim에서도 동일 한계 가능성.
  sim 결과가 나쁘면 π0는 "sim에서도 재현된 한계"로 기록하고 무리한 튜닝은 생략 (실기 보류와 일관)
- zero-shot은 시도하지 않음: 카메라 3키·32차원 불일치로 불가 판정 완료 (동 보고서 §1)

## 4. GR00T (5090) — sim은 워크숍 스택(최신), 실기는 N1.5

**계획**
1. **sim 경로는 워크숍 공식 그대로**: `gr00t_finetune.py --data-config so100_dualcam
   --embodiment-tag new_embodiment` (워크숍 추론 컨테이너의 최신 Isaac-GR00T 사용, D2)
2. 학습 전 데이터 처리: idle 트리밍(`trim_idle_v2.py`) + (v3 데이터면) v2 변환 + stats 수정
3. **`tune_visual` 필수 적용, 유효 batch 32 확보**(grad accumulation)
4. sim 평가: 워크숍 `lerobot_eval` + `-Eval`/`-DR-Eval`
5. 실기 전이: 기존 zmq 서버 경로 — 단 **버전 차이(N1.7계 sim vs N1.5 실기)를 보고서에 명시**하고,
   sim 체크포인트의 실기 전이는 워크숍 추론 컨테이너(동일 버전)로 수행해 버전 혼선 방지

**모델별 유의점과 근거**
- tune_visual 근거: **자체 실측** — vision freeze 시 큐브 위치 무관 모드 고정(pan -77),
  open-loop는 완벽하나 closed-loop 시각 그라운딩 부재 ([report/GR00T_report.md](../report/GR00T_report.md) §3)
- 유효 batch 근거: batch 4×10k(1.6 epoch)는 loss 0.041로 학습부족 — 유효 32(accum 8) 필요 (동 §3 v3/v4)
- 튜토리얼 레시피 근거: [HF 블로그 GR00T-N1.5 SO-101 튜닝](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
  (10k steps, so100_dualcam) + 워크숍 [교육 문서](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)
- 워크숍 eval 환경이 DR-Eval까지 제공하므로 **DR 유무별 sim SR을 분리 측정** (전이 예측력 확인)

---

## 5. 평가 매트릭스 (Phase 5 산출물)

| 모델 | sim SR | real zero-shot 전이 SR | sim+real co-train SR | (기준) real-only SR |
|---|---|---|---|---|
| ACT | | | | T1 80% (30ep v1) |
| SmolVLA | | | | T1 20~50% (v1~v3) |
| π0 | | | | 보류(부분 접근) |
| GR00T | | | | (vis2 판정 후) |

- 각 칸 10회 측정, 실기 전이는 기존 5지점/goto_home/개입금지 프로토콜 유지
- 해석 축: ① sim SR vs real 전이 SR 갭(도메인 갭 크기) ② co-training 개선폭(+38% 근거 대비)
  ③ 모델별 갭 민감도 (가설: ACT > SmolVLA ≈ GR00T > π0 순으로 시각 갭 민감)

## 6. 일정·순서 (D7)

1. [선행] GR00T 실기 완결 (vis2 → dry-run 판정 → 측정)
2. Phase 0 마무리: sim 이미지 빌드 완료 → 컨테이너 기동·씬 로드 확인 (양 PC)
3. Phase 1: 리더암 sim teleop 검증 → DR 태스크 50ep 수집
4. Phase 2~3: 4모델 학습 + sim 평가 (어댑터 제작 1~2일 포함)
5. Phase 4: 실기 전이 → co-training → 재측정
6. Phase 5: 매트릭스 완성·종합 보고서

## 7. 근거 자료 총람

**공식 문서/코드**
- Sim-to-Real SO-101 Workshop (NVIDIA·Isaac Sim 팀): https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- NVIDIA 교육 문서(Sim-to-Real SO-101): https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- Isaac Sim 공식 로봇 자산(SO-100/101 포함): https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html
- GR00T N1.5 SO-101 튜닝 블로그(HF·NVIDIA): https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
- LeIsaac(SO-101 sim teleop→LeRobot): https://wiki.seeedstudio.com/simulate_soarm101_by_leisaac/
- π0.5 공식 문서(train_expert_only 권장): https://huggingface.co/docs/lerobot/pi05

**논문**
- SmolVLA (50-demo 프로토콜): arXiv:2506.01844
- ACT/ALOHA (Zhao et al. 2023): arXiv:2304.13705
- Domain Randomization (Tobin et al. 2017): arXiv:1703.06907
- Sim+Real co-training +38%: arXiv:2503.24361

**자체 실측 (이 저장소)**
- GR00T idle 어트랙터·모드 고정·tune_visual 필요성: report/GR00T_report.md
- π0 LoRA 용량부족·expert-only 효과: report/Pi0_report.md
- SmolVLA epoch 통제·eval 프로토콜 교란: report/SmolVLA_report.md
- ACT 기준선(T1 80%): report/ACT_report.md
