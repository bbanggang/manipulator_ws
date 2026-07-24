#!/usr/bin/env bash
# T1 커스텀 태스크 파일을 워크숍 소스에 배포(idempotent). 로컬 검증 → 5090 적용 워크플로용.
# 사용법: ./deploy_t1.sh local      # 로컬(5070ti) 워크숍에 배포
#         ./deploy_t1.sh 5090       # 5090 워크숍에 배포(ssh)
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${1:?사용법: ./deploy_t1.sh <local|5090>}"

# 배포 로직(원격/로컬 공통) — 워크숍 소스 경로를 인자로 받아 실행
read -r -d '' PYDEPLOY <<'PY' || true
import sys, shutil, os
W = sys.argv[1]                       # .../source/sim_to_real_so101
usd = f"{W}/assets/usd"
# 1) 검은 박스 에셋
tb = f"{usd}/tray_black.usda"
if not os.path.exists(tb):
    s = open(f"{usd}/tray.usda").read()
    old = "color3f inputs:diffuse_color_constant = (0.03931828, 0.13806282, 0.2850575)"
    new = "color3f inputs:diffuse_color_constant = (0.02, 0.02, 0.02)"
    assert old in s, "tray diffuse line not found"
    open(tb, "w").write(s.replace(old, new)); print("  tray_black.usda 생성")
else:
    print("  tray_black.usda 존재")
# 2) env cfg (stdin 경유로 별도 복사됨 — 여기선 존재만 확인)
# 3) reset_cube_box → resets.py + ROBOT_COLORS에 purple 추가
rp = f"{W}/mdp/resets.py"; s = open(rp).read()
if "def reset_cube_box" not in s:
    s += open("/tmp/_t1_reset.py").read(); print("  resets.py: reset_cube_box append")
else: print("  resets.py: reset_cube_box 이미 있음")
if '"purple"' not in s:
    s = s.replace("ROBOT_COLORS = {\n", 'ROBOT_COLORS = {\n    "purple": (0.5, 0.1, 0.75),\n', 1)
    print("  resets.py: ROBOT_COLORS purple 추가")
else: print("  resets.py: purple 이미 있음")
open(rp, "w").write(s)
# 4) gym.register → __init__.py
ip = f"{W}/tasks/__init__.py"; s = open(ip).read()
if "Lerobot-So101-T1-CubeBox" not in s:
    s += open("/tmp/_t1_gym.py").read(); open(ip, "w").write(s); print("  __init__.py: T1 4종 append")
else: print("  __init__.py: 이미 있음")
PY

if [ "$TARGET" = "local" ]; then
  W="$HOME/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101"
  cp t1_cube_box_env_cfg.py "$W/tasks/t1_cube_box_env_cfg.py"
  cp open_box.usda "$W/assets/usd/open_box.usda"
  cp reset_cube_box_snippet.py /tmp/_t1_reset.py
  cp gym_register_t1_snippet.py /tmp/_t1_gym.py
  echo "[local] 배포:"; python3 -c "$PYDEPLOY" "$W"
elif [ "$TARGET" = "5090" ]; then
  W="~/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101"
  scp -q t1_cube_box_env_cfg.py "5090:$W/tasks/t1_cube_box_env_cfg.py"
  scp -q open_box.usda "5090:$W/assets/usd/open_box.usda"
  scp -q reset_cube_box_snippet.py 5090:/tmp/_t1_reset.py
  scp -q gym_register_t1_snippet.py 5090:/tmp/_t1_gym.py
  echo "[5090] 배포:"; ssh 5090 "python3 -c \"\$(cat)\" $W" <<< "$PYDEPLOY"
else
  echo "알 수 없는 타깃: $TARGET (local | 5090)"; exit 1
fi
echo "완료: $TARGET"
