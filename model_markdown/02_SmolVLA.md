# 02. SmolVLA (450M) — Zero-shot 시도 → **Few-episode FT로 전환**

> **역할 변경 (2026-07-09)**: 애초 "SO-ARM 계열 사전학습이라 zero-shot 유리"로 기대했으나,
> 심층 조사 결과 **SO-101 데이터는 사전학습에 전혀 포함되지 않았고**, SmolVLA 논문 자체가
> zero-shot 수치를 단 하나도 보고하지 않음이 확인됨. 7회 이상의 물리 테스트 + 논문/커뮤니티
> 조사를 종합해 **"zero-shot pick-place는 이 모델의 설계·검증 범위 밖"**으로 결론.
> → SmolVLA는 이제 **ACT처럼 few-episode 파인튜닝 후 평가**하는 쪽으로 역할을 바꾼다.
> (§7 "Zero-shot 불가 최종 판정과 근거", §8 "Few-episode FT 방향" 참조)

## 모델 개요

| 항목 | 내용 |
|---|---|
| 파라미터 | 450M |
| 사전학습 체크포인트 | ✅ `lerobot/smolvla_base` |
| 사전학습 데이터 | **SO-100 전용** — 481개 커뮤니티 데이터셋, ~23K 에피소드, ~10.6M 프레임. **SO-101 데이터 0%** (논문 원문: *"SmolVLA is not pretrained on any datasets recorded for the SO101"*) |
| 학습 입력 | 이미지 + state + **언어** (LeRobotDataset) |
| 추론 입력 | 카메라 프레임 + joint state + 자연어 명령 |
| 출력 | flow-matching action chunk (joint) |
| 특성 | 컨슈머 GPU 설계, 언어 조건부, **vision encoder는 파인튜닝 시에도 동결**(`freeze_vision_encoder=True`) — action expert만 학습 |
| 5070 Ti 16GB | ✅ Zero-shot 추론 및 Full FT 모두 가능 (로컬 완결) |

## 데이터 요구사항 (향후 FT 대비)

- **포맷**: LeRobotDataset + **에피소드별 자연어 task 설명 필드** (수집 시 포함하면 끝)
- 참고: SmolVLA 문서가 "25 에피소드로는 성능 나쁨"을 명시 — 향후 few-ep FT 비교(zero-shot / 25ep / 50ep+)의 기준점

## 진행 절차 (D5, zero-shot 경로 — 완료·전환됨)

- [x] **1. 셋업**
  - [x] `lerobot/smolvla_base` 로드 — 3캠(camera1/2/3, 256×256) 기대, 우리는 2캠(top/wrist) 640×480
    - 해상도는 `resize_imgs_with_padding=(512,512)`로 자동 처리, 카메라 개수는 `empty_cameras=0`이라 2캠만으로 자동 추론(3번째 슬롯 생략) — 코드 확인
    - ⚠️ `--dataset.rename_map`은 `LeRobotDataset.create()`가 반영 안 해 `make_policy()` 검증 실패 → **로봇 카메라 자체를 `camera1`/`camera2`로 명명**해 해결(실측 확인)
  - [x] 카메라 key / `observation.state` shape 정합 확인 — 통과
  - [x] dry-run — **심각한 이슈 발견, §5~7에서 근본 원인까지 규명**
- [x] **2. Zero-shot 측정 시도** → **불가 판정으로 중단** (§7.5 최종 결론). 대신 few-episode FT로 전환(§8)
- [ ] **3. (신규) Few-episode FT 경로** — §8 절차대로 진행 → 완료 후 π0로

## 5. Dry-run 초기 진단 (2026-07-09) — Zero-shot 심각한 실패

**증상**: T1 dry-run 시작 후 ~25% 지점(약 14초)에서 팔이 **작업 공간 반대쪽(왼쪽 빈 시트)으로 크게 회전한 뒤, 남은 시간(60초) 내내 완전히 고정**. 물체 방향으로 접근 시도 자체가 없음.

**검증 절차 5회 (모두 물리적 재현·개입 테스트, 2026-07-09)**:
1. **재현성 확인** (동일 셋업 2회 반복) → **완전히 동일한 실패** (같은 지점에서 같은 방향으로 고정)
2. **직접 개입 테스트** (빨간 큐브를 wrist 카메라 코앞에 들이댐, q1~q3 구간 유지) → **그리퍼가 픽셀 단위로 전혀 반응 없음** — 시각 입력 변화에 완전 무감각
3. **카메라 매핑 스왑 테스트** (`wrist→camera1, top→camera2`로 반대 매핑, `eval_smolvla_t1_camswap.sh`) → **동일 실패 재현** — camera1/2 슬롯 순서 문제 배제
4. **`--policy.empty_cameras=1` 적용 #1** (camera3를 검은 이미지로 패딩, smolvla_aloha_sim과 동일 패턴) → **예외적으로 팔이 테이블을 정상적으로 응시** — 원인 규명된 줄 알았으나
5. **`--policy.empty_cameras=1` 적용 #2 (동일 조건 재시행)** → **다시 좌측 고정으로 원복**. 4번 결과는 배경 요소(리더 팔 등 화면 구성 차이) 또는 우연이었을 가능성이 높음 — **empty_cameras도 근본 원인 아님**으로 정정

**5회 중 4회가 "좌측 고정"으로 압도적 다수** → 카메라 매핑·empty_cameras 설정 모두 배제됨. **SO-101 하드웨어·카메라 구도가 SmolVLA 사전학습 분포에서 심하게 벗어난(OOD) 진짜 zero-shot 실패**로 결론. 시각 인코더가 낯선 로봇 외형에서 의미 있는 특징을 못 뽑아내 행동이 붕괴하는 것으로 추정("cube" 어휘 문제 가능성은 낮게 평가 — 개입 테스트에서 시각 변화 자체에 무반응이었으므로 어휘보다 시각 인코딩 단계 문제를 시사).

**부가 발견**: 제어 루프 목표 30Hz 대비 **~9~10Hz**로 저하 (SmolVLA 450M ≫ ACT 51.6M, 추론 지연 큼).

## 6. 정규화 버그 수정 후 재검증 (2026-07-09, `diag_fixed_norm.py`)

`lerobot-record` CLI 우회, 코드 직접 분석으로 진짜 버그 하나를 찾아 수정:

- **버그**: `smolvla_base` 체크포인트에 내장된 정규화 stats가 `so100-blue.buffer.action` 등
  임바디먼트 접두사 키인데, 실제 조회 키는 평범한 `"action"`/`"observation.state"` — 매칭 실패로
  **정규화가 통째로 스킵**됨(`_apply_transform`: `key not in self._tensor_stats` → identity 통과).
  게다가 `lerobot-record`는 SmolVLA 등 비-GR00T 정책에 대해 우리 데이터셋 stats를 정규화에
  전혀 반영하지 않음(코드 확인, `factory.py`) — CLI로는 고칠 방법이 없어 커스텀 스크립트 작성.
- **수정**: `make_pre_post_processors()`를 직접 호출해 우리 실측 데이터(`so101_t1_pickplace`)의
  mean/std를 평범한 키로 주입 (`setup/smolvla/diag_fixed_norm.py`).
- **효과**: 수정 전 "고정된 이상한 자세로 붕괴" → 수정 후 **"정상 범위 값으로 부드럽게 계속 움직임"**.
  다만 **여전히 큐브를 향한 목적성 있는 접근은 전혀 없음** — 박스 근처에서 배회만 함.
- 부가로 `max_relative_target=8.0` 안전 클램프 필요(미설정 시 스텝당 최대 93도 급점프 실측 —
  temporal_ensemble 미적용으로 청크 경계에서 발생), rerun 실시간 시각화 추가.

**결론**: 정규화는 "붕괴 방지" 조건이었을 뿐 "task 수행 능력"과는 무관했다. 스케일을 고쳐도
목표 인식·접근 자체가 안 된다는 게 최종 확인됐다.

## 7. Zero-shot 불가 최종 판정과 근거 (2026-07-09 심층 조사)

정규화 수정 후에도 실패가 반복되자, SmolVLA 논문·커뮤니티 사례를 전면 재조사했다.

### 7.1 사전학습 데이터에 SO-101이 있는가 → **없다**

논문(arXiv:2506.01844) 원문: *"Critically, SmolVLA is not pretrained on any datasets
recorded for the SO101."* 사전학습은 **SO-100 단일 임바디먼트**(481 데이터셋, ~23K 에피소드,
~10.6M 프레임)로만 이루어졌다. 논문의 Limitations 섹션도 스스로 인정: *"Our pretraining
currently uses datasets collected from a single robot type (SO100)... we argue incorporating
training data from multiple robot embodiments is likely to prove critical in enhancing the
model's ability to generalize to new robotic platforms."* — **SO-101(우리 로봇)은 사전학습
시점에 모델이 한 번도 본 적 없는 새 임바디먼트**다. "SO-ARM 계열이라 유사할 것"이라는
애초 가정이 틀렸다 — SO-100과 SO-101은 형태는 비슷해도 모델 입장에선 완전히 다른 로봇이다.

### 7.2 Zero-shot 성공 사례가 정말 없는가 → **exhaustive 재조사 결과 없음**

HuggingFace 포럼·Spaces, Reddit, X, YouTube 설명란, GitHub Discussions, 후속 논문까지
전방위로 재검색했으나 **`smolvla_base`를 파인튜닝 없이 pick-place에 성공시킨 사례는
단 하나도 발견되지 않았다.** 오히려 우리와 정확히 같은 실패를 보고하는 사례들이 확인됨:

- **Henry Hu (Medium)**: Franka 시뮬레이션에서 `smolvla_base` zero-shot 테스트 — *"the arm
  would simply lift straight up in a generic motion, making no attempt to reach for the
  cube"*, 10회 중 접촉 0% — **우리가 관찰한 "목적 없는 배회"와 정확히 일치**.
- **Beegbrain (HF 블로그)**: 실제 SO-101 하드웨어로 zero-shot 시도 → 우리가 발견한 것과
  **동일한 정규화 stats 버그**에 막혀 행동 테스트 자체를 못 하고 파인튜닝으로 전환.
- GitHub #1239: SO-100에서 SmolVLA zero-shot "제대로 작동 안 함" 영상 보고(원인 미규명).

**"성공 사례를 찾으면 그에 맞게 task를 바꾼다"는 조건은 성립하지 않는다 — 맞춰 따라할
성공 사례 자체가 존재하지 않는다.**

### 7.3 논문은 zero-shot을 어떻게 다루는가 → **zero-shot 수치를 아예 보고하지 않는다**

논문의 모든 정량 결과(시뮬레이션: LIBERO 1,693ep/40task, Meta-World 2,500ep/50task; 실로봇:
SO-100 Table 3, SO-101 Table 4 "Pick-Place-Lego")는 **전부 task당 50개 시연**(5개 시작
위치 × 10 궤적)으로 **파인튜닝한 뒤**의 결과다. 사전학습만으로 측정한 수치, 즉 zero-shot
결과는 **논문 어디에도, 단 하나의 표에도 없다.** "zero-shot"이라는 단어 자체가 SmolVLA의
자체 능력을 설명하는 헤드라인 결과로 등장하지 않는다. 공식 문서·블로그의 행동 지침도
"Finetune SmolVLA on your own data"이지 "그대로 배포하라"가 아니다.

### 7.4 스케일을 고쳤는데도 안 되는 기술적 이유

1. **사전학습 목적 자체가 "task 완수"가 아니다** — flow-matching action expert는 다양한
   행동 패턴의 폭넓은 사전 지식(prior)을 학습할 뿐, 특정 task를 완수하도록 최적화되지
   않았다. 논문은 파인튜닝을 "새 로봇에 적응시키는 필수 단계"로 명시적으로 프레이밍한다.
2. **최소 시연 수 요구는 "선택"이 아니라 "기본 전제"** — 모든 실전 가이드가 "10~50 에피소드
   수집 후 파인튜닝"을 표준 워크플로로 다루며, 베이스 체크포인트를 배포 가능 상태로 취급하는
   자료는 없다.
3. **카메라·임바디먼트 캘리브레이션 민감성** — GitHub #1763, #2259가 카메라 뷰-사전학습
   컨벤션 정합성을 파인튜닝 후에도 성능에 영향을 주는 요인으로 지목.
4. **저자 스스로 명시한 일반화 한계** (§7.1 인용) — 새 로봇에 적응하려면 데이터가 필요하다는
   것이 저자들의 공식 입장.

### 7.5 최종 재검토 결론

> **우리 환경에서 SmolVLA zero-shot으로 T1을 수행하는 것은 현재 불가능하며, 이는 설정
> 오류가 아니라 모델의 설계·검증 범위를 정확히 벗어난 것이다.** 카메라 매핑, 카메라 개수,
> 정규화 스케일, 안전 클램프, 언어 전달까지 기술적으로 확인·수정 가능한 모든 요인을
> 소진했고, 남은 원인(SO-101 미학습 임바디먼트, task-완수 미최적화)은 **코드 수정으로
> 해결 불가능한 모델 자체의 특성**이다. 논문 자체도 zero-shot을 검증·주장하지 않는다.

## 8. Few-episode Fine-tuning 방향 분석

논문이 검증한 유일한 경로(50 demos/task)로 SmolVLA를 재평가하는 것이 합리적이다.

| 항목 | 내용 |
|---|---|
| 논문 프로토콜 | task당 **50개 시연**(5개 시작 위치 × 10 궤적), SO-100 Table 3 / SO-101 Table 4(Pick-Place-Lego)에서 이 방식으로 유의미한 성공률 보고 |
| 우리 기존 자산 | `heongyu/so101_t1_pickplace` **30ep**(ACT용으로 이미 수집) — 논문 기준(50ep)보다 적지만 즉시 재사용 가능, 부족분(20ep)만 추가 수집하면 됨 |
| 학습 비용 | vision encoder 동결(`freeze_vision_encoder=True`), action expert만 학습(`train_expert_only=True`) → **ACT보다 가볍지 않지만 5070 Ti 16GB에서 로컬 완결 가능**(모델 비교 보고서에서 이미 "Full FT 가능" 판정) |
| 예상 시간 | ACT(3시간/task)보다 다소 김 — VLM 백본 forward 비용 때문. 야간 슬롯 활용 권장 |
| 비교 프레임 변화 | SmolVLA가 "zero-shot 대표"에서 빠지면서 **ACT·SmolVLA(FT) 2종 vs π0·GR00T(zero-shot 시도) 2종** 구도로 재편됨 — π0/GR00T도 SmolVLA와 동일하게 "특정 임바디먼트 미학습 시 zero-shot 실패" 위험이 있으므로, 이번 SmolVLA 사례가 **사전 경고 신호**로 작용 (D6·D7 zero-shot 시도 시 같은 정규화 버그·미학습 임바디먼트 이슈를 먼저 점검할 것) |

**권장 진행**:
1. `record_t1.sh 10 resume`으로 T1을 50ep까지 보강 (또는 30ep 그대로 시작해 비교)
2. `train_act_t1.sh`를 참고해 `train_smolvla_t1.sh` 작성 (policy.type=smolvla, freeze_vision_encoder 등 반영)
3. 학습된 체크포인트로 `eval_smolvla_t1.sh`류 스크립트 재사용해 측정 → ACT와 "FT vs FT" 비교(대신 사전학습 유무 효과 확인 가능 — ACT는 from-scratch, SmolVLA는 SO-100 사전학습 위에 FT)

## 9. T1 Fine-tuning 실행 및 결과 (2026-07-09)

### 9.1 학습

기존 30ep(`heongyu/so101_t1_pickplace`) 그대로 재사용. `lerobot/smolvla_base`에서 시작해
`--policy.pretrained_path`로 파인튜닝(`setup/smolvla/train_smolvla_t1.sh`).

| 항목 | 값 |
|---|---|
| batch_size / steps | 16 / 20,000 (커뮤니티 SO-101 FT 사례를 5070 Ti 16GB에 맞춰 축소) |
| lr | SmolVLAConfig 기본값 그대로(1e-4 cosine decay, warmup 1000) |
| 최종 loss | 0.064(step200) → **0.006~0.007**, grad norm 0.14로 안정적 수렴 |
| epoch | 24.55 (30ep 데이터 기준) |
| 결과물 | `heongyu/smolvla_so101_t1`, 로드 시 `observation.images.top/wrist` 그대로 기대(카메라 리네이밍 불필요 — zero-shot과 다른 점) |

### 9.2 T1 측정 (10회, 그리드 1→2→3 순환)

**성공률: 5/10 (50%)**, VRAM 피크 2.48GB.

| 시행 | 그리드 | 길이(s) | SR | 실패유형 | 메모 |
|---|---|---|---|---|---|
| 0 | 1 | 56.7 | ❌ | 미접촉 | 접근만, 파지 시도 없음 |
| 1 | 2 | 56.9 | ❌ | 미접촉 | 〃 |
| 2 | 3 | 17.4 | ✅ | — | |
| 3 | 1 | 56.7 | ❌ | 미접촉 | 큐브 전혀 안 건드림 |
| 4 | 2 | 12.5 | ✅ | — | |
| 5 | 3 | 13.5 | ✅ | — | |
| 6 | 1 | 56.9 | ❌ | **파지 성공, 배치 실패** | 큐브 쥔 채 테이블 위에서 진행 정지 |
| 7 | 2 | 12.9 | ✅ | — | |
| 8 | 3 | 19.7 | ✅ | — | **1차 파지 실패 → 재접근 → 2차 파지 성공 → 박스 안착** 확인(영상) |
| 9 | 1 | 56.9 | ❌ | **파지 성공, 배치 실패** | 6번과 동일 양상 |

대표 클립: `report/clips/smolvla_ft_t1_success_trial8.mp4`(재시도 후 성공),
`smolvla_ft_t1_fail_trial6.mp4`(파지 성공·배치 실패)

### 9.3 ACT와의 비교 (동일 30ep 데이터 기준)

| 모델 | T1 SR |
|---|---|
| ACT (from-scratch) | 80% |
| SmolVLA-FT (SO-100 사전학습 위) | 50% |

이 데이터 규모(30ep, 논문 기준 50ep보다 적음)에서는 **ACT가 수치상 더 높다** — SmolVLA의
SO-100 사전학습 이점이 적은 데이터에서 충분히 발휘되지 못했을 가능성. 다만 **zero-shot(사실상
0%) 대비는 압도적 개선**이며, 실패 양상도 질적으로 다르다:

- **zero-shot**: 물체 인식·접근 자체가 없음 (완전 붕괴)
- **FT**: 실패해도 ①물체 근처까지 접근하거나 ②파지에는 성공하는 등 **"거의 됨" 수준**.
  특히 시행8은 **1차 파지 실패 후 스스로 재시도해 성공**하는 회복 행동을 보임 — ACT가
  보인 "실패 시 완전 정지"와 대비되는 특징.

### 9.4 시사점

- Few-episode FT는 zero-shot의 근본적 한계(§7)를 확실히 해결한다 — SO-101이 사전학습에
  없었어도 30ep 파인튜닝만으로 "미접촉→파지 성공"까지 능력이 생김.
- 다만 30ep은 논문 기준(50ep)에 못 미쳐 **ACT보다 낮은 SR**을 보임 — 데이터를 50ep로
  보강하면 역전 가능성 있음(향후 실험 후보).
- "파지 성공, 배치 실패"(시행6·9) 유형은 ACT에선 관찰 안 된 SmolVLA 특유의 실패 모드 —
  가능성 있게는 언어 조건부 모델이라 "배치" 서브골 자체를 덜 명확히 학습했을 수 있음.

## 완결 체크리스트 (게이트) — FT 경로 기준

- [x] T1 FT 데이터 확보(기존 30ep 재사용) + 학습 완료
- [x] T1 측정 10회 (SR 50%, 위 §9.2)
- [ ] T2 FT 데이터 수집 + 학습 완료
- [ ] T2 측정 10회
- [x] T1 대표 클립 (성공 1 + 실패 1)
- [ ] T2 대표 클립
- [ ] 모델 요약 카드 최종본 (T1+T2 통합, ACT 대비 비교 포함)
- [x] 로그·클립 백업

## 리스크 & 대응

| 리스크 | 대응 |
|---|---|
| 학습 시간이 ACT보다 김(VLM 백본) | 야간 슬롯 활용, 필요 시 스텝 수 조정 |
| FT 후에도 성능이 ACT 이하일 가능성 | **실측 확인됨**(50% vs 80%) — 정상적인 결과로 취급, "사전학습이 항상 유리하진 않다"도 유효한 결론 |
| 데이터 부족(30ep < 논문 50ep) | T2 진행 전 T1을 50ep로 보강할지 판단 필요 |

## 참고 자료

- SmolVLA 논문: arXiv:2506.01844 — "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics" (https://arxiv.org/html/2506.01844v1)
- 블로그: https://huggingface.co/blog/smolvla
- 공식 예제 데이터셋: `lerobot/svla_so101_pickplace`
- Henry Hu, "GenAI for Robotics: Fine-Tuning SmolVLA to Pick and Place" — zero-shot 실패 사례: https://medium.com/@henryhu1607/genai-for-robotics-fine-tuning-smolvla-to-pick-and-place-940b485e6c9b
- Beegbrain, "Weekly Robotics June #2" — SO-101 zero-shot 시도 중 정규화 버그 보고: https://huggingface.co/blog/Beegbrain/daily-robotics-june-2
- GitHub Issues: #1239(zero-shot 오동작), #1763(카메라 셋업 논의), #2259(임바디먼트 파인튜닝 이슈)
