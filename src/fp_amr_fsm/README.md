# fp_amr_fsm

TurtleBot4 AMR 관제(FSM) 패키지 — PC4에서 실행되는 **fleet 관제 노드**, **PC3 감지 시스템 연동 브릿지**, **로봇 구동(순찰/출동) 노드**를 담는다.

(구 패키지명 `fp_amr_vision` 에서 변경. vision 처리는 별도 `vision_pc3/` 가 담당하고, 이 패키지는 상태머신/관제가 중심이므로 fsm 으로 개명.)

## 시스템 구성

```
[PC3: vision_pc3]                [PC4: fp_amr_fsm]                   [로봇: robot2/robot9]
 웹캠 2대 YOLO 감지        /safety/*   safety_alert_bridge    /alert/*     amr_patrol_emer_helmet
 (쓰러짐/헬멧/무단침입) ──PoseStamped──▶ (타입/이름 변환)  ──String JSON──▶ fleet_fsm ──/{robot}/*──▶ (순찰·출동 실행)
```

## 노드

### 1. `fleet_fsm` — 관제 FSM 노드 (PC4)

로봇별 상태 추적 + 위급/안전모 이벤트 배정 + 지도 관제 화면(OpenCV) + 터미널 상태 테이블.

- **로봇 구동 상태 판단**: 로봇 자체 보고 대신 표준 raw 토픽을 직접 구독해 판단
  (`dock_status`, `navigate_to_pose/_action/status`, `amcl_pose`, `battery_state`).
  하트비트 단절 시 OFFLINE, battery만 오면 NO_LOCALIZATION, nav2 노드 부재 시 NO_NAV2 진단.
- **출동 배정**: 위급(EMERGENCY)은 최우선 — 안전모 확인 이동 중 로봇 선점 허용, 최단거리 선정.
  안전모(HELMET)는 idle 로봇만. 실패(abort) 시 최대 3회 재배정.
- **자동 순찰**: `PATROL_PERIOD`(기본 180초)마다 배터리 최고 idle 로봇에게 순찰 자동 배정.
  0 이하로 두면 비활성(수동 주입만).
- **순찰 교대**: 순찰 중 로봇의 `amr_status`가 PATROL → RETURNING(배터리 부족 자체 복귀)으로
  전이하면 다른 가용 로봇에게 순찰을 재배정.
- **배터리 히스테리시스**: `BATTERY_LOW = 0.30` (로봇 노드의 자체 복귀 임계값과 일치 필수),
  `BATTERY_DISPATCH_MIN = 0.40` (이 미만이면 출동/순찰 후보 제외 — 복귀 직후 재배정 핑퐁 방지),
  `BATTERY_RESUME = 0.90` (충전 완료 후 IDLE 복귀).

| 구분 | 토픽 | 타입 | 설명 |
|---|---|---|---|
| 구독 | `/alert/emergency` | String | JSON `{"x","y"[,"robot_id"]}` 위급 출동 요청 |
| 구독 | `/alert/helmet` | String | JSON `{"x","y"}` 안전모 확인 이동 요청 |
| 구독 | `/alert/patrol` | String | JSON `{}` 또는 `{"robot_id"}` 순찰 시작 |
| 구독 | `/alert/emergency_clear` | String | JSON `{}` 또는 `{"robot_id"}` 응급 조치 완료 |
| 구독 | `/alert/helmet_clear` | String | JSON `{}` 또는 `{"robot_id"}` 안전모 착용 확인 (미배정 큐 폐기 포함) |
| 구독 | `/{robot}/battery_state` 등 | (표준) | 로봇 raw 상태 (위 4종 + `amr_status`) |
| 발행 | `/{robot}/emergency_goal` | String | JSON `{"x","y","yaw","reason","person_x","person_y","robot_id","timestamp"}` — x,y는 사람 위치가 아니라 **0.7m 물러난 접근점**, yaw[deg]는 사람을 바라보는 방향 |
| 발행 | `/{robot}/helmet_goal` | String | 〃 (로봇이 idle일 때만 수행) |
| 발행 | `/{robot}/patrol_cmd` | String | 순찰 시작 명령 |
| 발행 | `/{robot}/emergency_clear` | Bool | 현장 대기 종료 신호 |
| 발행 | `/{robot}/helmet_clear` | Bool | 안전모 상황 해제 신호 |
| 발행 | `/fleet/status` | String | 관제 전체 상태 JSON (1Hz) |

명령 토픽 QoS: RELIABLE + TRANSIENT_LOCAL(latched), depth 1.

### 2. `safety_alert_bridge` — PC3 연동 브릿지 (PC4)

vision_pc3(PC3)의 `/safety/*` (PoseStamped/String)를 fleet_fsm의 `/alert/*` (String JSON)로 변환.

| 입력 (PC3) | 출력 (fleet_fsm) | 비고 |
|---|---|---|
| `/safety/emergency_goal` (PoseStamped) | `/alert/emergency` | 엣지 1회, 그대로 중계 |
| `/safety/helmet_goal` (PoseStamped, 2Hz follow) | `/alert/helmet` | **에피소드당 첫 goal 1회만** (큐 중복 방지) |
| `/safety/emergency_state` = "EMERGENCY_CLEAR" | `/alert/emergency_clear` | |
| `/safety/helmet_state` = "HELMET_CLEAR" | `/alert/helmet_clear` | goal을 중계했던 에피소드만 |

helmet 에피소드 = `/safety/helmet_state`의 NO_HELMET ~ HELMET_CLEAR 구간.

### 3. `amr_patrol_emer_helmet` — 로봇 구동 노드 (로봇별, 네임스페이스 필수)

fleet_fsm의 명령 토픽을 구독해 순찰 waypoint 주행, 위급 출동(순찰 즉시 선점),
안전모 배달(idle에서만)을 수행. 배터리 30% 미만이면 순찰을 중단하고 자체 도킹 복귀.
자체 상태를 `/{robot}/amr_status` (IDLE/PATROL/EMERGENCY/HELMET/RETURNING, latched)로 보고.

### 4. 웹 관제 모니터 — `web/fleet_monitor.html`

rosbridge(`ws://localhost:9090`) 로 `/fleet/status` JSON 을 구독해 브라우저에서
지도·로봇 상태·큐를 표시하는 단일 HTML 대시보드. 배경 지도는 같은 폴더의
`map.png` (final_project.pgm 변환본).

```bash
# rosbridge 실행 (미설치 시: sudo apt install ros-humble-rosbridge-suite)
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
# html 서빙 (설치된 share 폴더 기준)
cd $(ros2 pkg prefix fp_amr_fsm)/share/fp_amr_fsm/web
python3 -m http.server 8000   # → http://localhost:8000/fleet_monitor.html
```

### 기타

- `maps/`: fleet_fsm 이 사용하는 지도(final_project.pgm/yaml). share 에 설치되며,
  fleet_fsm 은 설치된 share/maps 를 우선 사용하고 없으면 `~/rokey_ws/maps` 폴백.
- `waypoint_patrol`, `patrol_fsm`, `amr_agent`: 이전 단계 실험/개발용 노드.
- `aruco_*`: ArUco 마커 감지 도구.

## 접속 환경 (PC3 연동 전제조건 — PC3_ROS_INTERFACE.md §0)

| 항목 | 값 |
|---|---|
| `ROS_DOMAIN_ID` | **6** |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| Discovery | Fast DDS Discovery Server (`ROS_DISCOVERY_SERVER`) |
| `ROS_SUPER_CLIENT` | `True` (Discovery Server 모드에서 `ros2` CLI가 토픽을 보려면 필수) |

자주 겪는 함정:
- 환경변수를 바꾸면 `ros2 daemon stop` — CLI 데몬은 처음 뜬 환경을 기억한다.
- Discovery Server 접속 확인: `nc -vz <server_ip> 11811`
- PC3↔PC4 연동 작업 중에는 `ROS_DISCOVERY_SERVER`를 절대 unset 하지 말 것.

## 빌드 및 실행

```bash
cd ~/rokey_turtlebot4_final_project
colcon build --packages-select fp_amr_fsm --symlink-install
source install/setup.bash

# PC4 (관제)
ros2 run fp_amr_fsm fleet_fsm
ros2 run fp_amr_fsm safety_alert_bridge

# 로봇 (robot2 예시)
ros2 run fp_amr_fsm amr_patrol_emer_helmet --ros-args -r __ns:=/robot2
```

PC3에서는 `vision_pc3/12_dual_camera_entry_yolo_tracking_modular.py` 실행 (ROS 발행 기본 ON).

## 시연/디버깅용 수동 주입

```bash
# 위급상황 발생
ros2 topic pub -1 /alert/emergency std_msgs/String '{data: "{\"x\": 1.2, \"y\": -0.5}"}'
# 안전모 미착용 지점 확인 이동
ros2 topic pub -1 /alert/helmet std_msgs/String '{data: "{\"x\": -1.0, \"y\": 2.0}"}'
# 순찰 시작 (robot_id 생략 시 배터리 최고 idle 로봇)
ros2 topic pub -1 /alert/patrol std_msgs/String '{data: "{}"}'
# 응급 조치 완료 / 안전모 착용 확인
ros2 topic pub -1 /alert/emergency_clear std_msgs/String '{data: "{}"}'
ros2 topic pub -1 /alert/helmet_clear std_msgs/String '{data: "{}"}'
```

## 조정 포인트 (`fleet_fsm.py` 클래스 상수)

| 상수 | 기본값 | 설명 |
|---|---|---|
| `ROBOTS` | `['robot2', 'robot9']` | 관제 대상 네임스페이스 (로봇 노드 `__ns` 와 일치 필수) |
| `MAP_PGM` / `MAP_YAML` | `~/rokey_ws/maps/final_project.*` | 지도 파일 (없으면 빈 캔버스로 동작) |
| `PATROL_PERIOD` | 180.0 | 자동 순찰 주기 [s], 0 이하 = 비활성 |
| `BATTERY_LOW` | 0.30 | 로봇 자체 복귀 임계값과 일치 필수 |
| `BATTERY_DISPATCH_MIN` | 0.40 | 출동/순찰 배정 가능 최소 배터리 |
| `BATTERY_RESUME` | 0.90 | 충전 완료 판정 |
| `OFFLINE_TIMEOUT` | 5.0 | 하트비트 단절 → OFFLINE [s] |
| `MAX_DISPATCH_ATTEMPTS` | 3 | 출동 실패 재배정 상한 |

## 알려진 미구현 / 주의

- 접근점의 costmap free 확인 + 각도 스캔(명세 §3)은 미구현 — 접근점이 장애물 위면
  Nav2 abort 후 fleet_fsm 재배정(최대 3회)으로만 대응한다.
- `/safety/unauthorized_*` (침입자 경보)는 브릿지/관제 화면 모두 미표시.
- 로봇 노드는 `/{robot}/helmet_clear`를 아직 구독하지 않음 (배달 후 고정 시간 대기).
- headless(SSH) 환경에서는 지도 창이 자동 비활성화되고 터미널 로그만 출력.
- `irobot_create_msgs`, `turtlebot4_navigation` 설치 필요.
