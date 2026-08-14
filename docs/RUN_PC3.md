# PC3 전체 실행 — 로봇 2대까지 (시나리오 검증용)

FSM 을 PC4 로 이식하기 전에, **PC3 한 대에서 전 사이클을 돌려 검증**한다.

```
PC3 (이 PC 하나가 전부)
├─ 감지            카메라 2대 → 쓰러짐/안전모/침입 판정
├─ safety_alert_bridge   /safety/* → /alert/*
├─ fleet_fsm       관제탑: 어느 로봇을 보낼지 결정          ← 나중에 PC4 로 이식
├─ rosbridge       웹 대시보드
├─ localization    robot2 / robot9   (amcl - 로봇이 자기 위치 앎)
├─ nav2            robot2 / robot9   (경로계획 + 주행)
└─ amr_patrol      robot2 / robot9   (fleet_fsm 명령 받아 Nav2 로 주행)
```

로봇 본체(`turtlebot4_node`, 라이다)는 **로봇 위에서** 돌고 있어야 한다. 나머지는 PC3.

---

## 0. 공통 ROS 환경 — **모든 터미널에서**

```bash
source /opt/ros/humble/setup.bash
source ~/rokey_turtlebot4_final_project/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=6
export ROS_DISCOVERY_SERVER=";;192.168.107.102:11811;;;;;;;192.168.107.109:11811"
export ROS_SUPER_CLIENT=True
```

`.bashrc` 맨 아래에 넣어두면 편하다.

### 먼저 확인: 로봇이 보이나

```bash
ping -c2 192.168.107.102     # robot2
ping -c2 192.168.107.109     # robot9
```

**둘 다 응답 없으면 WiFi 가 로봇망이 아니다.** Discovery Server 모드는 멀티캐스트를
꺼버리므로, 서버에 못 닿으면 **같은 PC 안의 노드끼리도 서로 못 본다.** 이 상태로
아무리 띄워도 `ros2 topic list` 가 비어 있고 웹도 안 붙는다.

```bash
ros2 topic list      # 첫 조회가 비면 한 번 더 (데몬이 discovery 동기화에 몇 초 걸림)
```
`/robot2/battery_state`, `/robot9/battery_state` 가 보이면 로봇과 통신되는 것.

---

## 1. 로봇별 localization (터미널 2개)

**map 경로가 핵심이다.** 감지(PC3)와 로봇이 **같은 맵**을 써야 좌표가 맞는다.

```bash
# 터미널 A — robot2
ros2 launch turtlebot4_navigation localization.launch.py \
  namespace:=/robot2 \
  map:=$HOME/rokey_turtlebot4_final_project/src/fp_amr_fsm/maps/final_project.yaml
```

```bash
# 터미널 B — robot9
ros2 launch turtlebot4_navigation localization.launch.py \
  namespace:=/robot9 \
  map:=$HOME/rokey_turtlebot4_final_project/src/fp_amr_fsm/maps/final_project.yaml
```

### ⚠️ initial pose 를 반드시 잡아줄 것

AMCL 은 처음엔 자기 위치를 모른다. 안 잡아주면 `amcl_pose` 가 `(0, 0)` 으로 나오고,
**Nav2 가 엉뚱한 출발점에서 경로를 짜서 벽으로 간다.**

- 로봇을 **도크에 올려두고 시작**하거나
- RViz 에서 **2D Pose Estimate** 로 실제 위치를 찍어준다

확인:
```bash
ros2 topic echo -1 /robot2/amcl_pose --qos-durability transient_local
ros2 topic echo -1 /robot9/amcl_pose --qos-durability transient_local
```
실제 위치와 맞는 좌표가 나와야 한다. `(0,0)` 이면 아직 안 잡힌 것.

---

## 2. 로봇별 nav2 (터미널 2개)

```bash
# 터미널 C — robot2
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot2
```
```bash
# 터미널 D — robot9
ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot9
```

확인 — `fleet_fsm` 은 이 노드들이 **그래프에 있는지**로 Nav2 생존을 판단한다:
```bash
ros2 node list | grep -E "robot2|robot9" | grep -E "planner_server|smoother_server"
```
4개(로봇당 2개)가 다 나와야 `fleet_fsm` 이 출동을 배정한다. 하나라도 없으면 `NO_NAV2`.

---

## 3. 로봇별 실행부 (터미널 2개)

`fleet_fsm` 의 명령을 받아 Nav2 로 실제 주행하는 노드.

```bash
# 터미널 E
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot2
```
```bash
# 터미널 F
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot9
```

기동 시 첫 줄에 `[amr_patrol] 네임스페이스 = /robot2   도크 = [...]` 가 찍힌다.
**여기가 robot2 로 나오는지 반드시 확인할 것.**

> 원래 `ROBOT_NAMESPACE = 'robot9'` 이 하드코딩이었고 `main()` 이
> `rclpy.init(args=['-r','__ns:=/robot9'])` 로 직접 박아서 커맨드라인을 무시했다.
> `robot_id` 필터도 같은 상수를 봐서 **robot2 는 자기 앞으로 온 명령을 전부 버렸다.**
> `--robot` 인자로 고쳐놨다.

### ⚠️ robot2 도크 좌표

`DOCK_POSITIONS` 에서 robot2 가 아직 robot9 과 같은 `[3.8, 6.06]` 이다.
**두 대가 같은 자리로 복귀하려다 충돌한다.** 실측해서 넣어야 한다:

```bash
# robot2 를 도크에 올린 뒤
ros2 topic echo -1 /robot2/amcl_pose --qos-durability transient_local
```
나온 x, y 를 `fp_amr_fsm/fp_amr_fsm/amr_patrol_emer_helmet.py` 의
`DOCK_POSITIONS['robot2']` 에 넣고 다시 빌드.

---

## 4. 감지 + 관제 + 웹 (터미널 1개)

```bash
cd ~/rokey_turtlebot4_final_project
./start.sh
```
rosbridge + fleet_fsm + safety_alert_bridge + 웹캠 감지 를 한 번에 띄운다.
감지 창은 **`q`** 로 종료 (강제종료 금지 — cam1 이 먹통이 된다).

**웹:** `src/fp_amr_fsm/web/fleet_monitor.html` 을 크롬에 끌어다 놓기.

---

## 5. 시나리오 검증

### 로봇이 준비됐는지
```bash
ros2 topic echo -1 /fleet/status
```
두 로봇 다 `"display_state": "IDLE"`, `"stack_fault": null` 이어야 한다.

| `stack_fault` | 뜻 | 조치 |
|---|---|---|
| `OFFLINE` | battery_state 가 안 옴 | 로봇 본체 전원/네트워크 |
| `NO_LOCALIZATION` | amcl_pose 미수신 | **1번** — initial pose 안 잡힘 |
| `NO_NAV2` | planner/smoother 노드 없음 | **2번** — nav2 안 뜸 |
| `null` | 정상 | 출동 가능 ✅ |

### 쓰러짐 → 출동
카메라 앞에서 눕는다. 5초 후 EMERGENCY.

```bash
# PC3 가 감지했나
ros2 topic echo /safety/emergency_state --qos-durability transient_local

# fleet_fsm 이 어느 로봇을 골랐나
ros2 topic echo /robot2/emergency_goal --qos-durability transient_local
ros2 topic echo /robot9/emergency_goal --qos-durability transient_local
```

payload:
```json
{"x": 1.71, "y": 4.07, "yaw": 56.3, "reason": "EMERGENCY",
 "person_x": 2.10, "person_y": 4.65, "robot_id": "robot2", "timestamp": ...}
```
`x, y` 는 **사람 좌표가 아니라 접근점**이다. `fleet_fsm` 이 사람을 밟지 않도록
0.7 m 떨어져서 사람을 바라보는 위치와 각도를 계산해 보낸다.

### 손 흔들기 → 상황 해제
```bash
ros2 topic echo /robot2/emergency_clear --qos-durability transient_local
```

---

## 6. 전부 종료

```bash
cd ~/rokey_turtlebot4_final_project && ./start.sh --stop
```
localization / nav2 / amr_patrol 은 각 터미널에서 `Ctrl+C`.

---

## 자주 걸리는 것

**`ros2 topic list` 가 비어 있다**
데몬이 Discovery Server 동기화 전에 조회한 것. **한 번 더 치면 나온다.**
환경변수를 바꿨다면 `ros2 daemon stop` 후 다시.
급하면 `ros2 topic list --no-daemon --spin-time 8`.
→ 이건 **CLI 만의 문제**다. rosbridge / fleet_fsm / 감지 노드는 목록 조회를 안 하므로 영향 없다.

**웹이 계속 `connecting`**
`pgrep -c rosbridge_websocket` → **2 가 정상** (launch 래퍼 + 노드). **4 면 중복이다.**
rosbridge 가 2개면 하나만 포트 9090 을 잡고 나머지는 좀비가 되어, 웹이 좀비 쪽에 붙으면
영원히 connecting 이다. `pkill -f rosbridge_websocket` 후 하나만 다시.

**배너가 `CONNECTING…` 에서 안 바뀐다**
그건 rosbridge 연결 표시가 아니라 **fleet 상황** 표시다. `/fleet/status` 가 와야 바뀐다.
`fleet_fsm` 이 떠 있는지 확인. (rosbridge 연결 여부는 오른쪽 초록 점을 봐라)

**로봇이 명령을 무시한다**
`amr_patrol` 기동 로그의 `네임스페이스 = /robotN` 이 맞는지 확인.
`--robot` 을 안 주면 기본값 robot9 으로 뜬다.

**로봇이 켜지자마자 엉뚱한 곳으로 출발한다**
명령 토픽이 `TRANSIENT_LOCAL`(latched) 이라, 로봇이 나중에 떠도 **마지막 goal 이 즉시 배달**된다.
`GOAL_MAX_AGE_SEC = 30` 으로 30 초 지난 명령은 버리게 막아뒀다.
