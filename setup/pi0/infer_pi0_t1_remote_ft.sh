#!/usr/bin/env bash
# π0 LoRA FT 추론 클라이언트 (T1) — 5090의 파인튜닝 체크포인트 서빙
#
# 구조(async inference): 5090에서 policy_server가 π0를 GPU에 로드하고, 이 로컬 클라이언트가
#   실로봇 관측을 gRPC로 보내고 액션 청크를 받아 로봇에 실행한다.
#   서버 먼저 실행 필요 → 5090에서:  ~/pi0_remote/pi0-server.sh start   (포트 8080)
#   방화벽 허용 필요   → 5090에서:  sudo ufw allow from 192.168.0.11 to any port 8080 proto tcp
#
# 사용법: ./setup/pi0/infer_pi0_t1_remote.sh
#
# ⚠️ 실로봇이 실제로 움직이는 명령이다. 물체·그리드 배치 후 사람이 지켜보며 실행할 것.
# ⚠️ FT 체크포인트는 top/wrist·6차원에 적응됨 — zero-shot 키/차원 불일치 없음.
# ⚠️ max_relative_target=8: 스텝당 관절 이동을 8도로 제한(안전 클램프). 미설정 시 큰 점프로
#    위험한 급동작 발생(2026-07-14 실측) → SmolVLA 때와 동일하게 클램프 필수.
#    동작이 너무 굼뜨면 값을 키우고, 여전히 위험하면 낮춰 재시도.
LAUNCHER="$(cd "$(dirname "$0")" && pwd)/_robot_client_launcher.py"
cd "$(dirname "$0")/../../envs/lerobot" || exit 1

SERVER="${SERVER:-192.168.0.56:8080}"

CAMS='{ top:   {type: opencv, index_or_path: /dev/cam_top,   width: 640, height: 480, fps: 30, fourcc: MJPG},
        wrist: {type: opencv, index_or_path: /dev/cam_wrist, width: 640, height: 480, fps: 30, fourcc: MJPG}}'

# NOTE: `python -m lerobot.async_inference.robot_client`는 so_follower 미등록으로
# --robot.type 선택지가 비어 에러남 → 로봇 config를 먼저 등록하는 런처로 우회.
exec uv run python "$LAUNCHER" \
  --server_address="$SERVER" \
  --robot.type=so101_follower --robot.port=/dev/ttyFOLLOWER --robot.id=follower \
  --robot.max_relative_target=8.0 \
  --robot.cameras="$CAMS" \
  --task="Pick up the red cube and place it in the black box" \
  --policy_type=pi0 \
  --pretrained_name_or_path=/home/user_lerobot/work/train/pi0_t1/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --fps=30 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=true
