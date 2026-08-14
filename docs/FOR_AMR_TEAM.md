# AMR 관리 PC 담당자에게

PC3(비전)는 감지 + `fleet_fsm`(관제탑) + `rosbridge` + 웹을 돌립니다.
**AMR PC 는 로봇 2대의 `localization` + `nav2` + `amr_patrol` 을 돌려주세요.**

```
PC3                                AMR PC
├─ 카메라 감지                      ├─ localization  robot2 / robot9
├─ safety_alert_bridge             ├─ nav2          robot2 / robot9
├─ fleet_fsm  ← 어느 로봇 갈지 결정  └─ amr_patrol    robot2 / robot9   ← 명령 받아 실제 주행
├─ rosbridge
└─ 웹 대시보드
```

---

## ⚠️ 1. `amr_patrol_emer_helmet.py` 를 **반드시 새 버전으로** 받아주세요

기존 파일은 **robot9 전용 하드코딩**이라 **robot2 를 띄울 수가 없습니다.**

```python
ROBOT_NAMESPACE = 'robot9'          # 모듈 상수
...
def main():
    rclpy.init(args=['--ros-args', '-r', f'__ns:=/{ROBOT_NAMESPACE}', ...])   # 직접 박음
```

`main()` 이 `__ns` 를 직접 박기 때문에 **`--ros-args -r __ns:=/robot2` 를 줘도 무시하고 robot9 으로 뜹니다.**
게다가 `robot_id` 필터도 같은 상수를 봅니다:

```python
target = data.get('robot_id')
if target and target != ROBOT_NAMESPACE:   # robot2 로 띄워도 'robot9' 과 비교
    return                                  # → 자기 앞으로 온 명령을 전부 버림
```

**고친 버전은 `--robot` 인자를 받습니다:**

```bash
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot2
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot9
```

기동 시 첫 줄에 이게 찍힙니다. **여기가 맞는지 꼭 확인하세요:**
```
[amr_patrol] 네임스페이스 = /robot2   도크 = [3.8, 6.06]
```

---

## ⚠️ 2. robot2 도크 좌표를 실측해서 넣어주세요

지금 **두 로봇의 도크 좌표가 같습니다** (`[3.8, 6.06]`). 이대로면 **두 대가 같은 자리로
복귀하려다 충돌합니다.**

```bash
# robot2 를 도크에 올린 뒤
ros2 topic echo -1 /robot2/amcl_pose --qos-durability transient_local
```

나온 x, y 를 `fp_amr_fsm/fp_amr_fsm/amr_patrol_emer_helmet.py` 에 넣고 다시 빌드:

```python
DOCK_POSITIONS = {
    'robot9': [3.8, 6.06],
    'robot2': [??, ??],   # ← 여기
}
```

---

## ⚠️ 3. 맵은 **PC3 와 같은 파일**을 써야 합니다

카메라가 계산한 사람 좌표와 로봇 좌표가 **같은 맵 기준**이어야 로봇이 사람에게 갑니다.
다른 맵이면 좌표가 어긋나 엉뚱한 곳으로 갑니다.

```
fp_amr_fsm/maps/final_project.yaml
  image: final_project.pgm
  resolution: 0.05
  origin: [-5.23, -1.38, 0]
```

```bash
ros2 launch turtlebot4_navigation localization.launch.py \
  namespace:=/robot2 \
  map:=<경로>/fp_amr_fsm/maps/final_project.yaml
```

---

## ⚠️ 4. initial pose 를 반드시 잡아주세요

AMCL 은 처음에 자기 위치를 모릅니다. 안 잡아주면 `amcl_pose` 가 `(0, 0)` 으로 나오고,
**Nav2 가 엉뚱한 출발점에서 경로를 짜서 벽으로 갑니다.**

- 로봇을 **도크에 올려두고 시작**하거나
- RViz 에서 **2D Pose Estimate** 로 실제 위치를 찍어주세요

확인:
```bash
ros2 topic echo -1 /robot2/amcl_pose --qos-durability transient_local
```
`(0, 0)` 이 아니라 실제 위치가 나와야 합니다.

---

## 5. 실행

```bash
# 공통 환경
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=6
export ROS_DISCOVERY_SERVER=";;192.168.107.102:11811;;;;;;;192.168.107.109:11811"
export ROS_SUPER_CLIENT=True
```

```bash
# 로봇당 3개 터미널
MAP=~/turtlebot4_ws/src/fp_amr_fsm/maps/final_project.yaml

ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot2 map:=$MAP
ros2 launch turtlebot4_navigation nav2.launch.py         namespace:=/robot2
ros2 run    fp_amr_fsm amr_patrol_emer_helmet --robot robot2

ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot9 map:=$MAP
ros2 launch turtlebot4_navigation nav2.launch.py         namespace:=/robot9
ros2 run    fp_amr_fsm amr_patrol_emer_helmet --robot robot9
```

### ❗ 중복 금지

`localization` / `nav2` / `amr_patrol` 은 **로봇당 딱 1개**여야 합니다.
ROS 2 는 ROS 1 과 달리 **같은 이름의 노드 중복을 막지 않습니다.** 조용히 다 돌면서 망가집니다.

- **AMCL 2개** → `map→odom` TF 가 두 개. 로봇 위치가 두 추정치 사이에서 떨림 → Nav2 가 쓰레기 TF 로 경로를 짬
- **Nav2 2개** → `controller_server` 두 개가 `/robotN/cmd_vel` 에 동시 발행 → 로봇이 상충하는 속도 명령을 받아 덜덜 떨거나 엉뚱하게 감
- **amr_patrol 2개** → 같은 로봇에 서로 다른 목적지를 쏨

띄우기 전에 PC3 의 `check_dup.sh` 로 확인하거나:
```bash
ros2 node list --no-daemon --spin-time 8 | grep -E "amcl|planner_server|controller_server"
```

---

## 6. 확인

`fleet_fsm` 이 로봇을 쓸 수 있다고 판단해야 출동시킵니다.

```bash
ros2 topic echo -1 /fleet/status
```

두 로봇 다 `"stack_fault": null`, `"display_state": "IDLE"` 이어야 합니다.

| `stack_fault` | 뜻 | 조치 |
|---|---|---|
| `OFFLINE` | `battery_state` 미수신 | 로봇 전원 / 네트워크 |
| `NO_LOCALIZATION` | `amcl_pose` 미수신 | **initial pose 안 잡힘** (4번) |
| `NO_NAV2` | `planner_server`/`smoother_server` 노드 없음 | nav2 안 뜸 |
| `null` | 정상 | 출동 가능 ✅ |

### `ROS_SUPER_CLIENT=True` 를 반드시 그대로 두세요

로봇팀 스니펫의 이 줄을 **복사하지 마세요**:
```bash
[ -t 0 ] && export ROS_SUPER_CLIENT=True || export ROS_SUPER_CLIENT=False   # ✗
```

`-t 0` 은 "터미널에서 직접 실행했나"를 봅니다. **스크립트/launch 로 띄우면 False** 가 됩니다.
Discovery Server 에서 일반 client 는 자기가 구독할 토픽만 discovery 합니다. 그런데
`fleet_fsm` 은 로봇의 Nav2 생존을 **노드 그래프 조회**로 판단하므로, SUPER_CLIENT 가 아니면
**살아있는 `planner_server` 를 못 보고 `NO_NAV2` 로 오판해 출동을 영영 안 시킵니다.**

---

## 7. 동작 흐름

```
사람 쓰러짐
  → PC3 감지            /safety/emergency_goal  (사람 발 위치)
  → safety_alert_bridge /alert/emergency
  → fleet_fsm           가장 가까운 로봇 선정 + 접근점 계산
                        /robot2/emergency_goal
  → amr_patrol          Nav2 로 주행                       ← AMR PC
```

`/robot2/emergency_goal` payload:
```json
{"x": 1.71, "y": 4.07, "yaw": 56.3, "reason": "EMERGENCY",
 "person_x": 2.10, "person_y": 4.65, "robot_id": "robot2", "timestamp": 1783912581.78}
```

- `x, y` 는 **사람 좌표가 아니라 접근점**입니다. `fleet_fsm` 이 사람을 밟지 않도록
  **0.7 m 떨어져서 사람을 바라보는** 위치와 각도를 계산해 보냅니다.
- `timestamp` — 명령 토픽이 `TRANSIENT_LOCAL`(latched) 이라 로봇이 나중에 떠도 **마지막 goal 이
  즉시 배달**됩니다. 그래서 이미 끝난 상황의 좌표로 출발하는 사고가 납니다.
  새 `amr_patrol` 은 **30초 지난 명령을 버립니다** (`GOAL_MAX_AGE_SEC`).
