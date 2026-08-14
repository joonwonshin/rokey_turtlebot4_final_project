# PC4 세팅 & 토픽 검증

PC3(감지) → PC4(관제/웹) 연동용. 로봇(robot2/robot9)과 같은 망에 있어야 한다.

---

## 1. PC4 ROS 환경 (터미널마다 실행 / `~/.bashrc` 맨 아래에 넣어도 됨)

```bash
source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=6
export ROS_DISCOVERY_SERVER=";;192.168.107.102:11811;;;;;;;192.168.107.109:11811"
export ROS_SUPER_CLIENT=True
```

### `ROS_DISCOVERY_SERVER` 형식

`;` 로 구분한 자리 = 서버 인덱스.

```
   ;    ;   192.168.107.102:11811   ;  ;  ;  ;   192.168.107.109:11811
   ^0   ^1          ^2 = robot2      3  4  5           ^6 = robot9
```

로봇 IP 가 바뀌면 이 줄만 고치면 된다.

### ⚠️ `ROS_SUPER_CLIENT=True` 는 반드시 그대로 둘 것

로봇팀이 쓰는 아래 줄을 **복사하지 마라**:

```bash
[ -t 0 ] && export ROS_SUPER_CLIENT=True || export ROS_SUPER_CLIENT=False   # ✗ 쓰지 말 것
```

`-t 0` 은 "터미널에서 직접 실행했나"를 본다. **스크립트·systemd·launch 파일로 노드를 띄우면 False** 가 된다.

Discovery Server 에서 일반 client 는 *자기가 구독할 토픽만* discovery 한다. 그런데
`fleet_fsm` 은 로봇의 Nav2 생존 여부를 **노드 그래프 조회**(`get_node_names_and_namespaces`)
로 판단한다. SUPER_CLIENT 가 아니면 그래프가 부분적으로만 보여서, **살아있는
`planner_server`/`smoother_server` 를 못 보고 `NO_NAV2` 로 오판 → 출동을 영원히 안 시킨다.**
(실측 확인함: False 로 두면 멀쩡한 robot9 도 NO_NAV2 로 나온다)

---

## 2. 토픽 확인 — `ros2 topic list` 가 비어 보인다면

**정상이다. 토픽이 없는 게 아니라 CLI 가 못 보는 것이다.**

`ros2` CLI 는 백그라운드 **데몬**을 통해 그래프를 조회하는데, Discovery Server 는
연결에 몇 초가 걸린다. 데몬은 기다리지 않고 즉시 빈 목록을 돌려준다.

```bash
# ✗ 빈 목록이 나온다
ros2 topic list

# ✓ 8초 기다리게 하면 전부 보인다
ros2 topic list --no-daemon --spin-time 8
ros2 node  list --no-daemon --spin-time 8
```

**중요:** 이건 **CLI 만의 문제**다. `rosbridge`, `fleet_fsm`, 우리가 짠 구독 노드들은
목록 조회 없이 토픽 이름으로 바로 구독하므로 **아무 영향 없다.**

---

## 3. PC3 가 발행하는 토픽

| 토픽 | 타입 | QoS | 내용 |
|---|---|---|---|
| `/safety/emergency_state` | `std_msgs/String` | RELIABLE + **TRANSIENT_LOCAL** | `EMERGENCY` / `EMERGENCY_CLEAR` |
| `/safety/emergency_goal` | `geometry_msgs/PoseStamped` | RELIABLE + VOLATILE | 쓰러진 사람 **발 위치** (map) |
| `/safety/helmet_state` | `std_msgs/String` | RELIABLE + **TRANSIENT_LOCAL** | `NO_HELMET` / `HELMET_CLEAR` |
| `/safety/helmet_goal` | `geometry_msgs/PoseStamped` | RELIABLE + VOLATILE | 안전모 미착용자 위치 |
| `/safety/unauthorized_state` | `std_msgs/String` | RELIABLE + **TRANSIENT_LOCAL** | 침입자 |
| `/safety/unauthorized_person` | `geometry_msgs/PoseStamped` | RELIABLE + VOLATILE | 침입자 위치 |
| `/safety/persons_json` | `std_msgs/String` | RELIABLE + VOLATILE | 전원 상태 JSON (웹용, 5 Hz) |
| `/safety/persons` | `visualization_msgs/MarkerArray` | **BEST_EFFORT** | RViz 용 마커 |
| `/safety/cam0/image/compressed` | `sensor_msgs/CompressedImage` | **BEST_EFFORT** | cam0 화면 (15 fps) |
| `/safety/cam1/image/compressed` | `sensor_msgs/CompressedImage` | **BEST_EFFORT** | cam1 화면 |

**QoS 를 안 맞추면 조용히 연결이 안 된다.** 특히:
- `TRANSIENT_LOCAL` 발행 → **`TRANSIENT_LOCAL` 로 구독**해야 latch 된 마지막 값을 받는다
- `BEST_EFFORT` 발행 → `RELIABLE` 로 구독하면 **아예 안 붙는다**

```bash
# 터미널에서 볼 때도 QoS 를 맞춰야 한다
ros2 topic echo /safety/emergency_state --qos-durability transient_local
ros2 topic echo /safety/persons --qos-reliability best_effort
```

---

## 4. 검증 순서 (PC4 에서)

```bash
# ① 토픽이 오는가
ros2 topic list --no-daemon --spin-time 8 | grep -E "^/safety/|^/fleet/"

# ② 카메라 프레임이 오는가 (hz 가 찍히면 성공)
ros2 topic hz /safety/cam0/image/compressed
ros2 topic hz /safety/cam1/image/compressed

# ③ 사람 감지 결과
ros2 topic echo /safety/persons_json

# ④ 쓰러짐 이벤트 (PC3 앞에서 실제로 누워본다)
ros2 topic echo /safety/emergency_state --qos-durability transient_local
ros2 topic echo /safety/emergency_goal

# ⑤ fleet_fsm 이 로봇에게 보낸 명령
ros2 topic echo /robot2/emergency_goal --qos-durability transient_local
ros2 topic echo /robot9/emergency_goal --qos-durability transient_local

# ⑥ 관제 상태 (로봇 위치/배터리/결함)
ros2 topic echo /fleet/status
```

**PC3 에서 한 방에 전부 확인:**
```bash
cd ~/rokey_turtlebot4_final_project
python3 check_web.py --cam
```
(rosbridge 웹소켓으로 확인 — 브라우저가 보는 것과 완전히 동일한 경로)

---

## 5. PC4 에서 띄울 것

```bash
# 관제 FSM (로봇 선정: 응급=가까운놈 / 안전모=순찰 안 하고 노는놈)
ros2 run fp_amr_fsm fleet_fsm

# PC3 의 /safety/* 를 fleet_fsm 이 먹는 /alert/* 로 중계
ros2 run fp_amr_fsm safety_alert_bridge

# 웹 대시보드용 (포트 9090)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

웹: `src/fp_amr_fsm/web/fleet_monitor.html` 을 크롬으로 **그냥 열면 된다** (로컬 서버 불필요).
`ws://localhost:9090` 으로 붙으므로 **rosbridge 와 같은 PC** 에서 열어야 한다.
다른 PC 에서 열려면 HTML 의 `CONFIG.ROSBRIDGE_URL` 을 `ws://<PC4-IP>:9090` 으로 고친다.

---

## 6. 데이터 흐름

```
PC3 (카메라 2대)
  YOLO-pose 사람 + best.pt 안전모 → 쓰러짐/안전모/침입 판정
        │  /safety/emergency_goal, /safety/emergency_state ...
        ▼
  safety_alert_bridge          (/safety/*  →  /alert/*)
        │  /alert/emergency, /alert/emergency_clear
        ▼
  fleet_fsm                    (어느 로봇을 보낼지 결정)
        │  /robot2/emergency_goal, /robot9/emergency_goal  (접근점 + yaw)
        ▼
  amr_patrol_emer_helmet       (로봇 위에서 실행, Nav2 로 주행)
```

`fleet_fsm` 은 사람 좌표를 그대로 주지 않는다. **사람을 밟지 않도록 0.7 m 떨어진
접근점과 사람을 바라보는 yaw 를 계산해서** 보낸다.

---

## 7. 로봇 쪽 실행 (담당자 확인 후)

```bash
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot2
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot9
```

> 예전에는 `ROBOT_NAMESPACE = 'robot9'` 이 하드코딩이었고 `main()` 이
> `rclpy.init(args=['-r', '__ns:=/robot9'])` 로 직접 박아서, `--ros-args -r __ns:=/robot2`
> 를 줘도 무시하고 robot9 으로 떴다. 게다가 `robot_id` 필터도 같은 상수를 봐서
> **robot2 는 자기 앞으로 온 명령을 전부 버렸다.** `--robot` 인자로 고쳐놨다.

### 아직 남은 것

- **robot2 `DOCK_POSITION`** 이 robot9 과 같은 `[3.8, 6.06]` 이다 → 두 대가 같은 자리로
  복귀하려다 충돌한다. `amr_patrol_emer_helmet.py` 의 `DOCK_POSITIONS` 에 실측값을 넣어야 한다:
  ```bash
  # robot2 를 도크에 올린 뒤
  ros2 topic echo -1 /robot2/amcl_pose --qos-durability transient_local
  ```
- **robot2 initial pose** — 지금 `amcl_pose` 가 `(-0.02, 0.00)` = 원점이다. AMCL 이 측위를
  못 잡은 상태다. 이대로 출동시키면 Nav2 가 **틀린 출발점에서 경로를 짜서 벽으로 간다.**
  RViz 의 2D Pose Estimate 로 실제 위치를 찍어주거나 도크에서 시작해야 한다.
- **robot2 Nav2** — `planner_server` / `controller_server` 가 안 떠 있다 (localization 만 있음).
