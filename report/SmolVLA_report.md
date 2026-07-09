# SmolVLA 모델 진행 보고서 — Zero-shot 시도 및 불가 판정

> 작성일: 2026-07-09 | SmolVLA zero-shot 진단 완료, **Few-episode FT로 전환 결정**
> 프로젝트: 단일 SO-ARM101 매니퓰레이터로 4개 모델(ACT / SmolVLA / π0 / GR00T N1.5)의
> 학습 및 zero-shot 성능을 비교한다. 본 문서는 두 번째 모델 **SmolVLA**의 zero-shot
> 시도 전 과정과, 왜 이 경로가 불가능하다고 결론 내렸는지를 정리한다.

---

## 1. 개요 — 무엇을, 왜 했는가

계획 단계에서 SmolVLA는 **"SO-ARM 계열 데이터로 사전학습된 유일한 모델"**이라는 근거로
zero-shot 3종(SmolVLA/π0/GR00T) 중 가장 성공 가능성이 높다고 예상했다. ACT가 매번
데이터를 모아 처음부터 학습해야 하는 것과 달리, SmolVLA는 `lerobot/smolvla_base`라는
사전학습 체크포인트가 있어 **학습 없이 바로 우리 로봇에서 추론**을 시도할 수 있었다.

**Task**: T1 Pick-and-Place — 빨간 큐브를 집어 검은 박스에 넣는다 (ACT와 동일 물리 셋업).

이 보고서는 "왜 이 낙관적 예상이 빗나갔는가"를 처음부터 끝까지 추적한 기록이다 —
단순한 실패 보고가 아니라, **표면적 실패(이상 동작)에서 시작해 코드 레벨 버그를 찾아
수정하고, 그래도 안 되는 이유를 논문·커뮤니티 근거로 규명하기까지의 진단 과정** 자체가
이 보고서의 핵심 내용이다.

---

## 2. 실험 셋업

- **로봇/카메라**: ACT와 동일 — SO-ARM101 follower, `top`(전경)·`wrist`(그리퍼) 2캠, MJPG 640×480
- **정책**: `lerobot/smolvla_base` (450M 파라미터, flow-matching action expert, VLM 백본
  `SmolVLM2-500M-Video-Instruct`의 16개 층만 사용)
- **언어 명령**: `"Pick up the red cube and place it in the black box"`
- **카메라 슬롯 매핑**: SmolVLA는 `camera1/camera2/camera3` 3개 슬롯을 기대. 우리는 2캠뿐이라
  로봇 카메라 자체를 `camera1`(top)`/camera2`(wrist)로 명명해 스키마를 직접 일치시킴
  (`--dataset.rename_map`은 `LeRobotDataset.create()`가 반영하지 않아 무용지물임을 코드로 확인)

---

## 3. Zero-shot 시도 — 초기 실패와 5단계 검증

### 3.1 증상

Dry-run 시작 후 **~25% 지점(약 14초)에서 팔이 작업 공간 반대쪽으로 크게 회전한 뒤,
남은 시간(최대 60초) 내내 완전히 고정**됐다. 물체 방향으로 접근 시도 자체가 없었다.

### 3.2 5단계 원인 배제 과정

| # | 검증 | 방법 | 결과 |
|---|---|---|---|
| 1 | 재현성 | 동일 셋업 2회 반복 | **완전히 동일한 실패** — 같은 지점에서 같은 방향으로 고정. 노이즈 아님 |
| 2 | 직접 개입 | 빨간 큐브를 wrist 카메라 코앞에 들이댐 | **그리퍼 완전 무반응** — 시각 입력 변화에 무감각 |
| 3 | 카메라 매핑 스왑 | `wrist↔top` 슬롯 순서 반전 | **동일 실패 재현** — 슬롯 순서 문제 아님, 배제 |
| 4 | `empty_cameras=1` 적용 | camera3를 검은 패딩으로 채움 (라이브러리 공식 패턴) | 1회 예외적 정상 동작 → 재현 시도 |
| 5 | 4번 재현 시도 | 동일 조건 재시행 | **다시 원래 실패로 복귀** — 4번은 우연/배경 요소였던 것으로 정정 |

5회 중 4회가 "동일 방향 고정"으로 압도적이었다. 카메라 매핑·개수 모두 원인에서 배제됐다.

---

## 4. 근본 원인 규명 — 정규화 버그

라이브러리 소스 코드를 직접 추적해 실제 버그를 발견했다.

### 4.1 발견

`smolvla_base` 체크포인트에 내장된 정규화 통계(mean/std)가 **`so100-blue.buffer.action`,
`so100-red.buffer.action`, `so100.buffer.action`처럼 임바디먼트별 접두사가 붙은 키**로
저장되어 있었다. 그런데 실제 정규화 코드(`UnnormalizerProcessorStep._apply_transform`)는
평범한 `"action"`/`"observation.state"` 키로 조회한다:

```python
if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
    return tensor   # 키가 안 맞으면 정규화를 통째로 건너뛴다
```

즉 **키가 매칭되지 않아 입력 정규화와 출력 역정규화가 전부 조용히 스킵**되고 있었다.
모델은 학습 때와 완전히 다른 스케일의 원시 관절값을 입력받고, 출력도 정규화 안 된 채
그대로 로봇에 전달된 것이다.

### 4.2 CLI로는 고칠 수 없음

`lerobot-record`는 GR00T를 제외한 모든 정책(SmolVLA 포함)에 대해 `--dataset.rename_map`이나
우리 데이터셋의 통계를 정규화 파이프라인에 **전혀 반영하지 않는다**(`factory.py`
`make_pre_post_processors` 코드 확인). 따라서 표준 CLI로는 이 버그를 우회할 방법이 없어,
`lerobot-record`를 거치지 않는 **커스텀 진단 스크립트**(`setup/smolvla/diag_fixed_norm.py`)를
직접 작성했다 — 정책·프로세서·로봇·카메라를 Python으로 직접 조립하고, 우리 실측 데이터
(`so101_t1_pickplace`)의 mean/std를 평범한 키로 주입했다.

---

## 5. 버그 수정 후 재검증

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 관절값 스케일 | 비정상 (붕괴) | **실제 로봇 범위(도 단위)와 일치** |
| 동작 패턴 | 한 자세로 완전 고정 | **15~60초 내내 부드럽게 계속 움직임** |
| 큐브를 향한 접근 | 없음 | **여전히 없음** |

추가로 두 가지를 더 고쳤다:
- **안전 클램프**(`max_relative_target=8.0`): 미설정 시 한 스텝에 최대 93도까지 급점프하는
  위험한 동작이 실측됨(청크 경계에서 `temporal_ensemble` 미적용으로 발생) — 관절당 8도로 제한
- **rerun 실시간 시각화** 추가: 카메라 뷰 + state/action 시계열을 실시간으로 확인 가능하게 함

**결론**: 정규화 수정은 "위험한 붕괴를 막는" 효과는 있었지만, **목표 물체를 인식하고
다가가는 능력과는 무관**했다. 스케일을 고쳐도 여전히 task를 수행하지 못했다.

---

## 6. 심층 조사 — 왜 그래도 안 되는가

정규화를 고쳤는데도 실패가 반복되자, SmolVLA 논문과 커뮤니티 사례를 전면 재조사했다.

### 6.1 사전학습 데이터에 SO-101이 있는가 → **없다**

논문(arXiv:2506.01844) 원문:

> *"Critically, SmolVLA is not pretrained on any datasets recorded for the SO101."*

사전학습은 **SO-100 단일 임바디먼트만**(481개 데이터셋, ~23K 에피소드, ~10.6M 프레임)
사용했다. 논문 스스로도 한계로 인정한다:

> *"Our pretraining currently uses datasets collected from a single robot type (SO100)...
> we argue incorporating training data from multiple robot embodiments is likely to prove
> critical in enhancing the model's ability to generalize to new robotic platforms."*

**"SO-ARM 계열이라 비슷할 것"이라는 최초 가정이 틀렸다** — SO-100과 SO-101은 외형은
비슷해도 모델 입장에선 한 번도 본 적 없는 완전히 다른 로봇이다.

### 6.2 Zero-shot 성공 사례가 정말 없는가 → **exhaustive 재조사 결과 없음**

HuggingFace 포럼·Spaces, Reddit, X, YouTube, GitHub Discussions, 후속 논문까지 전방위로
검색했으나 **`smolvla_base`를 파인튜닝 없이 pick-place에 성공시킨 사례는 하나도 없었다.**
오히려 우리와 동일한 실패를 보고하는 사례들만 확인됐다:

- **Henry Hu (Medium)**: Franka 시뮬레이션에서 zero-shot 테스트 — *"팔이 그냥 위로 들리는
  일반적인 동작만 하고, 큐브에 접근하려는 시도가 전혀 없었다"*, 10회 중 접촉 0% —
  **우리가 관찰한 "목적 없는 배회"와 정확히 일치**
- **Beegbrain (HF 블로그)**: 실제 SO-101 하드웨어로 zero-shot 시도 → 우리가 발견한 것과
  **동일한 정규화 stats 버그**에 막혀 행동 테스트조차 못 하고 파인튜닝으로 전환
- GitHub #1239: SO-100에서 SmolVLA zero-shot 오동작 영상 보고 (원인 미규명 상태로 방치)

### 6.3 논문은 zero-shot을 어떻게 다루는가 → **수치를 아예 보고하지 않는다**

논문의 모든 정량 결과(시뮬레이션 LIBERO·Meta-World, 실로봇 SO-100·SO-101)는 **전부
task당 50개 시연(5개 시작 위치 × 10 궤적)으로 파인튜닝한 뒤**의 결과다. 사전학습만으로
측정한 zero-shot 수치는 **논문 어디에도 없다.** "zero-shot"이라는 단어가 SmolVLA 자체
능력을 설명하는 헤드라인 결과로 등장하지 않는다. 공식 문서의 지침도 "그대로 배포하라"가
아니라 "Finetune SmolVLA on your own data"이다.

---

## 7. 결론 — 최종 판정

> **우리 환경에서 SmolVLA zero-shot으로 pick-place를 수행하는 것은 현재 불가능하며,
> 이는 설정 오류가 아니라 모델의 설계·검증 범위를 정확히 벗어난 것이다.**

카메라 매핑, 카메라 개수, 정규화 스케일, 안전 클램프, 언어 전달까지 **기술적으로
확인·수정 가능한 모든 요인을 소진**했다. 남은 원인 — SO-101이라는 미학습 임바디먼트,
그리고 애초에 task 완수를 목표로 최적화되지 않은 사전학습 목적 — 은 **코드 수정으로
해결할 수 없는 모델 자체의 특성**이다. 논문 저자들조차 zero-shot을 검증하거나
주장하지 않는다.

이 결론은 실패가 아니라 **철저히 검증된 성과**다 — "설정 문제인지 근본적 한계인지"를
5단계 물리 검증 + 코드 레벨 버그 수정 + 논문/커뮤니티 대조까지 거쳐 명확히 가려냈다.

---

## 8. 다음 진행 계획 — Few-episode Fine-tuning으로 전환

논문이 유일하게 검증한 경로(task당 50개 시연)로 SmolVLA를 재평가한다.

| 항목 | 내용 |
|---|---|
| 논문 프로토콜 | task당 50개 시연 (5개 시작 위치 × 10 궤적) |
| 우리 기존 자산 | `heongyu/so101_t1_pickplace` **30ep** (ACT용으로 이미 수집) — 즉시 재사용 가능 |
| 학습 비용 | vision encoder 동결, action expert만 학습 → 5070 Ti 16GB에서 로컬 완결 가능 |
| 비교 프레임 변화 | ACT(from-scratch FT) vs SmolVLA(SO-100 사전학습 위 FT) — **"사전학습이 실제로 도움이 되는가"**를 직접 비교하는 흥미로운 축이 새로 생김 |
| 프로젝트 전체 시사점 | π0·GR00T도 특정 임바디먼트 미학습 시 동일한 zero-shot 실패 위험 — 이번 사례가 사전 경고 신호. D6·D7 진행 시 정규화 버그를 먼저 점검할 것 |

**절차**:
1. `record_t1.sh 10 resume`으로 T1을 50ep까지 보강 (또는 기존 30ep로 우선 시작)
2. `train_act_t1.sh`를 참고해 `train_smolvla_t1.sh` 작성 (`policy.type=smolvla`)
3. 학습된 체크포인트로 평가 스크립트 재사용해 측정 → ACT와 나란히 비교

---

*관련 문서: [model_markdown/02_SmolVLA.md](../model_markdown/02_SmolVLA.md) (상세 진단 로그·체크리스트),
[report/ACT_report.md](ACT_report.md) (동일 형식의 ACT 보고서),
[model_markdown/README.md](../model_markdown/README.md) (전체 프로젝트 가이드라인)*
