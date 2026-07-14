#!/bin/bash
# PC3 전체 스택 한 방에 기동.
#
#   ./start.sh          rosbridge + fleet_fsm + safety_alert_bridge + 웹캠 감지
#   ./start.sh --stop   전부 종료
#
# 감지 프로그램은 자체 대시보드 창이 뜨고 'q' 로 종료한다. 그래서 포그라운드로
# 돌리고 나머지 3개는 백그라운드에 둔다.
#
# ⚠ 감지 창은 반드시 'q' 로 종료할 것. kill -9 하면 cap.release() 가 안 돌아서
#   Jieli USB 카메라(cam1)가 먹통이 된다 (복구: sudo usbreset 4c4a:4a55).

set +u   # ROS 의 setup.bash 는 set -u 와 함께 못 쓴다 (AMENT_TRACE_SETUP_FILES)

BASE="$HOME/turtlebot4_ws/final_project"
LOG="$BASE/.logs"
mkdir -p "$LOG"

# ── ROS 환경 ────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$HOME/turtlebot4_ws/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=6
export ROS_DISCOVERY_SERVER=";;192.168.107.102:11811;;;;;;;192.168.107.109:11811"
export ROS_SUPER_CLIENT=True
#   ROS_DISCOVERY_SERVER 형식: ';' 로 구분한 자리 = 서버 인덱스
#     ;;192.168.107.102:11811;;;;;;;192.168.107.109:11811
#      0 1        2=robot2      3 4 5      6=robot9
#   ROS_SUPER_CLIENT=True 필수:
#     일반 client 는 자기가 구독할 토픽만 discovery 한다. 그런데 fleet_fsm 은
#     로봇의 Nav2 생존 여부를 '노드 그래프 조회'로 판단하므로, SUPER_CLIENT 가
#     아니면 살아있는 planner_server 를 못 보고 NO_NAV2 로 오판해 출동을 안 시킨다.

# ── 프로세스 패턴 ───────────────────────────────────────────────────
# pkill -f 는 '모든' 프로세스의 cmdline 을 본다. 호출하는 셸의 명령줄에 패턴이
# 들어 있으면 자기 자신을 죽인다. 그래서 변수에 담아 이 파일 안에서만 쓴다.
P_RB='rosbridge_websocket'
P_FSM='lib/fp_amr_fsm/fleet_fsm'
P_BRG='lib/fp_amr_fsm/safety_alert_bridge'
P_CAM='12_dual_camera_entry_yolo_tracking_modular'
P_WEB='http.server 8000'

stop_all() {
  pkill -INT -f "$P_CAM" 2>/dev/null    # 감지는 SIGINT 로 (카메라 해제)
  sleep 3
  pkill -f "$P_RB"  2>/dev/null
  pkill -f "$P_FSM" 2>/dev/null
  pkill -f "$P_BRG" 2>/dev/null
  sleep 2
  pkill -f "$P_WEB" 2>/dev/null
  pkill -9 -f "$P_RB"  2>/dev/null
  pkill -9 -f "$P_FSM" 2>/dev/null
  pkill -9 -f "$P_BRG" 2>/dev/null
  pkill -9 -f "$P_WEB" 2>/dev/null
  # 감지에는 -9 를 쓰지 않는다. cap.release() 가 안 돌면 UVC 카메라가
  # 물려서(select() timeout) 다음 실행 때 한 프레임도 못 읽는다.
  # 그렇게 되면 sudo usbreset <vendor:product> 로 USB 를 리셋해야 한다.
  sleep 1
}

if [ "$1" = "--stop" ]; then
  echo "종료 중..."
  stop_all
  echo "완료."
  exit 0
fi

# ── 중복 제거 ──────────────────────────────────────────────────────
# rosbridge 가 2개 뜨면 하나만 포트 9090 을 잡고 나머지는 좀비가 된다.
# 웹이 좀비 쪽에 붙으면 화면이 영원히 'connecting' 이다. 반드시 먼저 정리.
echo "기존 프로세스 정리..."
stop_all

# ── 카메라 자동 탐지 ────────────────────────────────────────────────
# USB 를 다시 꽂으면 /dev/videoN 번호가 바뀐다. 이름으로 찾는다.
cam0=$(v4l2-ctl --list-devices 2>/dev/null | grep -A1 "Web Camera"     | grep -o "/dev/video[0-9]*" | head -1)
cam1=$(v4l2-ctl --list-devices 2>/dev/null | grep -A1 "USB Composite"  | grep -o "/dev/video[0-9]*" | head -1)
if [ -z "$cam0" ] || [ -z "$cam1" ]; then
  echo "❌ 카메라를 못 찾음 (cam0=$cam0  cam1=$cam1)"
  echo "   v4l2-ctl --list-devices 로 확인하세요."
  exit 1
fi

echo "════════════════════════════════════════════════"
echo " ROS_DOMAIN_ID    = $ROS_DOMAIN_ID"
echo " DISCOVERY_SERVER = $ROS_DISCOVERY_SERVER"
echo " ROS_SUPER_CLIENT = $ROS_SUPER_CLIENT"
echo " cam0 = $cam0 (Web Camera)   cam1 = $cam1 (USB Composite)"
echo "════════════════════════════════════════════════"

# ── 1. rosbridge (웹 ↔ ROS, 포트 9090) ─────────────────────────────
echo "[1/4] rosbridge"
setsid ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
       > "$LOG/rosbridge.log" 2>&1 < /dev/null &
for i in $(seq 1 20); do
  sleep 1
  ss -tln 2>/dev/null | grep -q ':9090' && break
done
if ! ss -tln 2>/dev/null | grep -q ':9090'; then
  echo "❌ rosbridge 기동 실패 - $LOG/rosbridge.log"
  exit 1
fi
echo "      ✅ 포트 9090"

# ── 2. fleet_fsm (관제탑: 로봇 선정 + /fleet/status) ────────────────
echo "[2/4] fleet_fsm"
setsid ros2 run fp_amr_fsm fleet_fsm \
       > "$LOG/fleet_fsm.log" 2>&1 < /dev/null &

# ── 3. safety_alert_bridge (/safety/* → /alert/*) ──────────────────
echo "[3/4] safety_alert_bridge"
setsid ros2 run fp_amr_fsm safety_alert_bridge \
       > "$LOG/alert_bridge.log" 2>&1 < /dev/null &
sleep 5

[ "$(pgrep -cf "$P_FSM")" -ge 1 ] && echo "      ✅ fleet_fsm" \
  || echo "      ❌ fleet_fsm 실패 - $LOG/fleet_fsm.log"
[ "$(pgrep -cf "$P_BRG")" -ge 1 ] && echo "      ✅ safety_alert_bridge" \
  || echo "      ❌ bridge 실패 - $LOG/alert_bridge.log"

# ── 4. 웹 대시보드 서빙 (PC4 등 다른 PC 에서 접속하려면 필수) ───────
# file:// 로 직접 열면 그 PC 의 localhost:9090 을 보므로 PC4 에서는 못 붙는다.
# http 로 서빙하면 html 이 '서빙한 호스트'의 9090 을 자동으로 찾아간다.
echo "[4/5] 웹 대시보드 (포트 8000)"
setsid python3 -m http.server 8000 \
       --directory "$BASE/fp_amr_fsm_connec_vision/fp_amr_fsm/web" \
       > "$LOG/webserver.log" 2>&1 < /dev/null &
sleep 1
MYIP=$(ip -4 addr show wlo1 2>/dev/null | grep -o 'inet [0-9.]*' | awk '{print $2}')
ss -tln 2>/dev/null | grep -q ':8000' && echo "      ✅ http://${MYIP:-localhost}:8000/fleet_monitor.html" \
  || echo "      ❌ 웹서버 기동 실패 - $LOG/webserver.log"

# ── 5. 웹캠 감지 (포그라운드) ───────────────────────────────────────
echo "[5/5] 웹캠 감지  (모델 로딩 20~30초)"
echo
echo "  웹(이 PC / PC4 공통): http://${MYIP:-localhost}:8000/fleet_monitor.html"
echo "  종료: 감지 창에서 'q'  →  그 다음 ./start.sh --stop"
echo

cd "$BASE/detection_final"
exec python3 12_dual_camera_entry_yolo_tracking_modular.py \
     --cam0-id "${cam0#/dev/video}" \
     --cam1-id "${cam1#/dev/video}" \
     --publish-ros
