#!/usr/bin/env bash
# GR00T sim 평가 (5090 headless) — 워크숍 공식 경로: real-robot 서버 + teleop-docker 클라이언트.
# **이 스크립트는 5090에서 실행한다** (scp로 배포 후 ssh 5090에서 bash).
#
# 사용법(5090에서):  ~/sim_eval_gr00t.sh <MODEL_SUBPATH> [num_episodes] [eval|dr]
#   예: ~/sim_eval_gr00t.sh grootn16_sim75/checkpoint-10000 10 eval
#       ~/sim_eval_gr00t.sh grootn16_sim75/checkpoint-10000 10 dr    # DR-Eval(조명 변동)
#
# 동작: (1) real-robot 컨테이너로 GR00T 서버(5555) 백그라운드 기동 → ready 대기
#       (2) teleop-docker 컨테이너로 lerobot_eval 클라이언트 headless 실행 → SR 출력
#       (3) 종료 시 서버 컨테이너 정리
#
# 근거: NVIDIA 워크숍 sim 평가(11-sim-evaluation) — 75ep sim-only 학습 시 SR 50~70% 기대.
#   우리는 사전학습 체크포인트를 그대로 평가해 파이프라인 검증 + 기준선 재현(옵션 A).
set -uo pipefail

MODEL="${1:?MODEL 서브경로 필요 (예: grootn16_sim75/checkpoint-10000)}"
NUM="${2:-10}"
MODE="${3:-eval}"
TASK="Lerobot-So101-Teleop-Vials-To-Rack-Eval"
[ "$MODE" = "dr" ] && TASK="Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval"

WORKSHOP="$HOME/Sim-to-Real-SO-101-Workshop"
cd "$WORKSHOP"

RENAME='{"external_D455": "front", "ego": "wrist"}'
LANG="Pick up the vial and place it in the yellow rack"
SRV_LOG="$HOME/sim_eval_server.log"
CLI_LOG="$HOME/sim_eval_client_$(date +%Y%m%d_%H%M%S).log"

cleanup() { docker rm -f gr00t-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "▶ [1/2] GR00T 서버 기동 (real-robot 컨테이너, 모델: $MODEL)"
docker run -d --name gr00t-srv --rm --network host --privileged --gpus all \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOME/gr00tn16_ws/checkpoints:/workspace/models" \
  -v "$WORKSHOP/docker/real/scripts:/workspace/Isaac-GR00T/gr00t/eval/real_robot/SO100" \
  real-robot \
  bash -c "cd /Isaac-GR00T && python3 gr00t/eval/run_gr00t_server.py --model-path /workspace/models/$MODEL" \
  > /dev/null
docker logs -f gr00t-srv > "$SRV_LOG" 2>&1 &

echo "   서버 로딩 대기 (모델 로드 ~1-2분)..."
for i in $(seq 1 120); do
  grep -q "Server is ready" "$SRV_LOG" 2>/dev/null && { echo "   ✅ 서버 준비 완료"; break; }
  if ! docker ps -q --filter name=gr00t-srv | grep -q .; then
    echo "   ❌ 서버 컨테이너가 종료됨. 로그:"; tail -20 "$SRV_LOG"; exit 1
  fi
  sleep 3
done

echo "▶ [2/2] sim 평가 클라이언트 실행 (teleop-docker, $TASK, ${NUM}ep, headless)"
docker run --name teleop-eval --rm --privileged --gpus all \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --network host \
  -v "$WORKSHOP/docker/env:/root/env" \
  -v "$WORKSHOP/source:/workspace/Sim-to-Real-SO-101-Workshop/source" \
  -v "$WORKSHOP/outputs:/workspace/Sim-to-Real-SO-101-Workshop/outputs" \
  teleop-docker:latest \
  bash -c "lerobot_eval --task $TASK --num_envs 1 --num_episodes $NUM \
    --rename_map '$RENAME' --action_horizon 16 \
    --lang_instruction '$LANG' --headless" 2>&1 | tee "$CLI_LOG"

echo
echo "=== 결과 요약 ==="
grep -i "Success Rate" "$CLI_LOG" || echo "(성공률 라인 미검출 — 클라이언트 로그 확인: $CLI_LOG)"
