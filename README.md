# 산업안전 AMR 관제 시스템 — 전체 소스코드

천장 웹캠 2대가 작업자의 **쓰러짐 / 안전모 미착용 / 무단침입**을 감지하고,
관제 FSM이 AMR 2대(robot2 · robot9) 중 **가장 가까운 로봇**을 골라 출동시킨다.
순찰 중 소화기의 ArUco 마커를 인식해 점검 결과를 DB에 기록하고 웹으로 조회한다.

```
                    ┌──────────────── 감지 (PC3) ────────────────┐
   웹캠 cam0 ──┐    │  YOLO11-pose + 안전모 검출                 │
   웹캠 cam1 ──┴───▶│  호모그래피 + Z캘리브 → map 좌표           │
                    │  쓰러짐/안전모/침입 판정                    │
                    └───────────────┬───────────────────────────┘
                                    │ /safety/*  (관찰)
                                    ▼
                          safety_alert_bridge   (타입/이름 변환)
                                    │ /alert/*   (요청)
                                    ▼
                    ┌──────────── 관제 (fleet_fsm) ─────────────┐
                    │  로봇 선정(최근접) · 접근점 계산 · 큐 관리 │
                    │  상황 상태머신 (NORMAL ↔ EMERGENCY)        │
                    └───────────────┬───────────────────────────┘
                                    │ /robotN/*  (명령)
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             amr_patrol (robot2)            amr_patrol (robot9)
             Nav2 주행 · 순찰 · 도킹 · ArUco 소화기 점검
                                    │
                                    ▼
                            sqlite3db  (점검 결과 DB + Flask 웹)
```

---

## 폴더 구성

| 폴더 | 내용 | 실행 위치 |
|---|---|---|
| `1_vision_pc3/` | 웹캠 감지 (단독 Python, ROS 패키지 아님) | 비전 PC |
| `2_ros2_packages/` | colcon 빌드 대상 ROS 2 패키지 3개 | 관제 PC / AMR PC |
| `3_calibration_tools/` | 호모그래피 · Z캘리브레이션 제작 도구 | 최초 1회 (셋업용) |
| `4_docs/` | 실행 절차서 · 흐름도 | — |
| `start.sh` | 감지 + 관제 + 웹 일괄 기동 | 비전 PC |

---

## 1. `1_vision_pc3/` — 웹캠 감지

ROS 패키지가 아니라 **단독 Python 프로그램**이다. 스크립트가 상대경로로
모델·캘리브레이션을 참조하므로 **이 폴더 안에서 실행**해야 한다.

```
1_vision_pc3/
├── 12_dual_camera_entry_yolo_tracking_modular.py   # 메인 (진입점)
├── safety_lib/
│   ├── vision_core.py       # 카메라 루프, YOLO 추론, 좌표 변환
│   ├── safety_logic.py      # ★ 쓰러짐/안전모 판정 + 트랙별 상태기계
│   ├── base_utils.py        # 호모그래피, Z캘리브(P행렬), 키포인트 유틸
│   ├── global_fusion.py     # 두 카메라 간 동일 인물 통합 (global_id)
│   ├── ros_bridge.py        # /safety/* 토픽 발행 (엣지 판정 + CLEAR 디바운스)
│   ├── dashboard_ui.py      # OpenCV 대시보드
│   └── state_io.py          # 상태 JSON 저장
├── calibration/             # 실측 캘리브레이션 결과 (필수)
│   ├── cam0_to_map.npz      #   호모그래피 (픽셀 → map 좌표)
│   ├── cam0_z_calib.npz     #   3×4 projection P (머리 높이 역산용)
│   ├── cam1_to_map.npz
│   ├── cam1_z_calib.npz
│   └── entry_roi.json       #   출입 통제 구역 폴리곤
├── yolo_experiments/best.pt # 안전모 검출 모델 (직접 학습)
├── yolo11s-pose.pt          # 자세 추정 모델 (17 keypoints)
├── final_project.pgm/.yaml  # SLAM 맵 (좌표계 기준)
└── PC3_ROS_INTERFACE.md     # 발행 토픽 명세
```

**실행**
```bash
cd 1_vision_pc3
python3 12_dual_camera_entry_yolo_tracking_modular.py \
    --cam0-id 4 --cam1-id 0 --publish-ros
```
종료는 **감지 창에서 `q`**. `kill -9` 하면 `cap.release()`가 안 돌아 UVC 카메라가
물린다(복구: `sudo usbreset <vendor:product>`).

**의존성**: `ultralytics`, `opencv-python`, `numpy`, `rclpy`

---

## 2. `2_ros2_packages/` — ROS 2 패키지

```bash
# colcon 워크스페이스에 심볼릭 링크 또는 복사
cp -r 2_ros2_packages/* ~/ros2_ws/src/
cd ~/ros2_ws && colcon build && source install/setup.bash
```

### 2-1. `fp_amr_fsm/` — 관제탑 + 웹 대시보드 ★ 핵심

| 노드 | 역할 |
|---|---|
| `fleet_fsm` | **관제탑.** 로봇 선정(최근접), 접근점 계산, 큐 관리, 상황 상태머신 |
| `safety_alert_bridge` | `/safety/*` (PoseStamped) → `/alert/*` (JSON String) 변환 |
| `amr_patrol_emer_helmet` | **로봇 실행부** — Nav2 주행 (아래 ⚠️ 참조) |

`web/fleet_monitor.html` — rosbridge 기반 관제 웹. 지도 위 로봇/사람 실시간 표시,
**출동 로봇 선정 근거**(로봇별 거리·배터리·제외 사유), 이벤트 주입 버튼, 토픽 로그.

```bash
ros2 run fp_amr_fsm fleet_fsm
ros2 run fp_amr_fsm safety_alert_bridge
ros2 launch rosbridge_server rosbridge_websocket_launch.xml   # 웹용
```

### 2-2. `amr_aruco/` — AMR 실행부 + ArUco 소화기 점검

| 노드 | 역할 |
|---|---|
| `aruco_detect` | OAK-D 카메라로 소화기의 ArUco 마커 인식 → `/robotN/aruco_marker_id` |
| `amr_patrol_emer_helmet` | 실행부 (모듈 분리 리팩터링 + 아루코 연동) |

`config/amr_aruco_params.yaml`로 파라미터화, `patrol_points.json`으로 순찰 경로 관리.

### ⚠️ 두 패키지에 `amr_patrol_emer_helmet` 이 중복 존재한다

| | `fp_amr_fsm` 판 | `amr_aruco` 판 |
|---|---|---|
| 라인 수 | 861 | 354 (모듈 분리) |
| ArUco 소화기 점검 | ❌ | ✅ |
| `helmet_clear` 수신 (안전모 해제) | ✅ | ❌ |
| clear가 **미실행 goal 취소** | ✅ | ❌ |
| 현장 대기 5분 상한 | ✅ | ❌ |
| 도킹 **전** 안전모 배달 | ✅ | ❌ |
| 응급 시 배터리 무시 | ✅ | ❌ |

**→ FSM 로직은 `fp_amr_fsm` 판이 최신이다. 실행부는 이쪽을 쓴다:**
```bash
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot2
ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robot9
```
`amr_aruco`는 **ArUco 인식 노드**로 사용한다:
```bash
ros2 run amr_aruco aruco_detect
```
> 두 판의 FSM 로직 통합(= `amr_aruco`의 모듈 구조 + `fp_amr_fsm`의 최신 수정)은
> 남은 과제다. 현재는 두 소스를 모두 보존해 이력을 남긴다.

### 2-3. `sqlite3db/` — 점검 결과 DB + Flask 웹

| 노드 | 역할 |
|---|---|
| `create_db` | DB/테이블 생성, CSV·JSON·XLSX 초기 데이터 적재 |
| `ros2_db_node` | ROS 토픽 ↔ DB 연동 (로봇 카메라 프레임 저장 포함) |
| `db_update` | ArUco 마커 ID를 받아 **소화기 점검 결과 갱신** + 스냅샷 저장 |
| `app` | Flask 웹 — 점검 현황 조회 |

```bash
ros2 run sqlite3db create_db      # 최초 1회
ros2 launch sqlite3db monitoring.launch.py
```
`camera_frames/`, `camera_frames/snapshots/` 는 **런타임에 이미지가 쌓이는 폴더**다
(빈 채로 제출). `fire_db.db`는 `create_db`로 재생성할 수 있다.

---

## 3. `3_calibration_tools/` — 캘리브레이션 제작 (셋업용)

`1_vision_pc3/calibration/*.npz` 를 만들어낸 도구들. 카메라를 옮기면 다시 돌려야 한다.
번호가 곧 작업 순서다.

| 스크립트 | 역할 |
|---|---|
| `00_capture_ref.py` · `01_dual_camera_capture.py` | 기준 프레임 촬영 |
| `02_make_homography_pairwise.py` | 픽셀 ↔ map 대응점 → **호모그래피** `camN_to_map.npz` |
| `03` · `04` | 호모그래피 시각 검증 |
| `05_guided_single_camera_capture.py` | 높이별 대응점 수집 |
| `08_z_height_calibration_test.py` | **3×4 projection P** 산출 → `camN_z_calib.npz` |
| `06` · `07` · `09` ~ `12` | 판정 알고리즘 개발 이력 (bbox → z → pose) |

---

## 4. 실행 순서 (전체 시스템)

```bash
# 공통 ROS 환경 (모든 터미널)
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=6
export ROS_DISCOVERY_SERVER=";;<robot2_ip>:11811;;;;;;;<robot9_ip>:11811"
export ROS_SUPER_CLIENT=True      # ★ fleet_fsm 이 노드 그래프로 Nav2 생존을 판단
```

| 순서 | 실행 | 위치 |
|---|---|---|
| 1 | `ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robotN map:=...` | AMR PC |
| 2 | `ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robotN` | AMR PC |
| 3 | `ros2 run fp_amr_fsm amr_patrol_emer_helmet --robot robotN` | AMR PC |
| 4 | `ros2 run amr_aruco aruco_detect` | AMR PC |
| 5 | `./start.sh` (rosbridge + fleet_fsm + bridge + 감지 + 웹서버) | 비전 PC |
| 6 | `ros2 launch sqlite3db monitoring.launch.py` | 관제 PC |
| 7 | 웹: `http://<비전PC_IP>:8000/fleet_monitor.html` | 임의 PC |

> `ROS_SUPER_CLIENT=True` 는 선택이 아니다. `fleet_fsm` 은 `planner_server` 가
> **노드 그래프에 존재하는지**로 Nav2 생존을 판단하므로, 이 값이 `False` 면
> 살아있는 Nav2 를 못 보고 `NO_NAV2` 로 오판해 **출동을 배정하지 않는다.**
> 시스템 스크립트가 대화형 셸일 때만 `True` 로 두는 경우가 있으니 확인할 것.

---

## 5. 토픽 계약 (3계층)

| 계층 | 토픽 | 의미 |
|---|---|---|
| **관찰** | `/safety/emergency_state`, `/safety/helmet_state`, `/safety/persons_json` … | 카메라가 본 것. 로봇 개념 없음. 좌표 = **사람의 발 위치** |
| **요청** | `/alert/emergency`, `/alert/helmet`, `/alert/*_clear`, `/alert/patrol`, `/alert/queue_clear` | 관제탑 접수창구. 감지가 넣든 **웹 버튼**이 넣든 동일 처리 |
| **명령** | `/robotN/emergency_goal`, `/robotN/helmet_goal`, `/robotN/patrol_cmd`, `/robotN/*_clear` | 그 로봇에게. 좌표 = 사람에서 **0.7m 물러난 접근점 + 사람을 바라보는 yaw** |

명령 토픽 QoS: `RELIABLE` + `TRANSIENT_LOCAL`(latched) — 한 번만 발행해도 유실되지
않고, 나중에 뜬 로봇에게도 배달된다. 대신 옛 명령이 뒤늦게 배달되는 것을 막기 위해
payload 의 `timestamp` 로 30초 이상 지난 명령은 로봇이 버린다.

---

## 6. 우선순위 정책

> **응급(EMERGENCY) > 안전모(HELMET) > AMR 자기 역할(순찰)**

- **응급은 배터리를 보지 않는다.** 충전 중인 로봇도 끌어낸다.
  물리적으로 갈 수 없는 경우(`OFFLINE`, `NO_NAV2`, 측위 미실행)만 제외한다.
- **안전모는 순찰을 선점한다.** 배달 후 **도킹하지 않고** 중단된 waypoint 부터 순찰 재개.
- 로봇을 명시(`robot_id`)하면 **다른 로봇으로 대체하지 않는다** (응급만 예외).

---

## 제외한 것

빌드 산출물(`build/`, `install/`, `log/`, `*.egg-info`), 캐시(`__pycache__`, `*.pyc`),
백업(`*.bak`), 학습 데이터셋, 실험용 모델 가중치, 런타임 산출물(스냅샷 이미지, 로그).
**실행에 필요한 모델·캘리브레이션·맵은 모두 포함**했다.
