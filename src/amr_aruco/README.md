# amr_aruco

TurtleBot4 순찰 로봇용 ArUco 마커 인식 + 순찰/응급출동/안전모배달 FSM 패키지.

## 패키지 구조

```
amr_aruco/
├── amr_aruco/
│   ├── amr_patrol_emer_helmet.py  # 순찰/응급/안전모 실행부 노드 (메인 FSM 루프)
│   ├── patrol_navigator.py        # PatrolNavigator: 상태 플래그 + 토픽 콜백 + 파라미터 선언
│   ├── aruco_detect.py            # OAK-D 카메라 ArUco 마커 인식 노드
│   ├── aruco_vision.py            # OpenCV ArUco 검출/표시 유틸 (ROS 의존성 없음)
│   ├── nav_utils.py               # 웨이포인트/포즈 생성 유틸 (patrol_points.json 로드 포함)
│   ├── common.py                  # 토픽 이름/QoS 등 fleet_fsm 과의 프로토콜 상수
│   └── patrol_points.json         # 순찰 웨이포인트 좌표 목록
├── config/
│   └── amr_aruco_params.yaml      # 두 노드의 파라미터 기본값 (의미/제약 주석 포함)
└── launch/
    └── amr_aruco.launch.py        # 두 노드를 함께 실행
```

- `amr_patrol_emer_helmet` : `patrol_points.json`의 웨이포인트를 순찰하면서
  fleet_fsm(상태머신, 별도 패키지)이 내려주는 명령에 따라 순찰/응급출동/안전모배달을
  수행하는 실행부 노드. 로봇 선정(어떤 로봇을 보낼지) 자체는 담당하지 않는다.
  TurtleBot4Navigator 기반이라 **노드 이름은 `basic_navigator`** 다 (파라미터
  yaml/`ros2 param` 대상 이름에 주의).
- `aruco_detect` : OAK-D 카메라(CompressedImage)에서 ArUco 마커를 인식하는 노드.
  `aruco_scan_enable` 토픽이 `True`인 동안에만 카메라 토픽을 구독/검출한다 -
  주행 중에는 카메라 스트림 자체를 끊어 오탐 방지와 함께 CPU/WiFi 부하로
  Nav2(AMCL tf 갱신)가 밀리는 것을 막는다. 노드 이름은 `aruco_marker_detector`.
- `patrol_points.json` : 순찰 웨이포인트 좌표 목록. `yaw` 필드가 있는 지점은
  소화기 포인트로 간주되어 도착 후 ArUco 대조 완료 신호를 기다린다. RViz에서
  좌표를 자주 조정하므로 코드가 아닌 별도 파일로 분리했다(symlink-install
  환경에서는 재빌드 없이 수정 가능). 다른 파일을 쓰려면 `patrol_points_file`
  파라미터로 경로를 지정한다.

## 두 노드의 연동 관계

```
amr_patrol_emer_helmet          aruco_detect
  --- aruco_scan_enable (Bool) ------->   (소화기 포인트 도착 시에만 True)
  <-- aruco_check_done (Bool) ---------   (마커 대조 완료를 알려주는 노드가 별도로 필요)
```

`aruco_detect`는 인식된 마커 ID 목록을 `.../aruco/detection/ids`
(Int32MultiArray)로 발행하기만 하며, 이를 보고 "맞는 소화기인지" 판정해
`aruco_check_done`(Bool)을 발행하는 것은 이 패키지에 포함되어 있지 않다(별도
판정 노드가 필요, 혹은 테스트 시 아래처럼 수동 발행). 소화기 포인트에서
`aruco_check_timeout`(기본 5초) 안에 대조 완료 신호가 오지 않으면 대조를
포기하고 다음 지점으로 이동한다.

두 노드 모두 **노드 네임스페이스**로 로봇을 구분한다. launch 실행 시
`namespace` 인자(기본 `robot2`)가 두 노드에 함께 적용되므로 어긋날 일이 없고,
`ros2 run`으로 단독 실행하면 `__ns` 리맵이 없을 때 기본 네임스페이스
`robot2`(`common.DEFAULT_ROBOT_NAMESPACE`)가 자동 적용된다.

## 빌드

워크스페이스 루트(`e2_tutlebot`)에서:

```bash
cd ~/e2_tutlebot
colcon build --packages-select amr_aruco --symlink-install
source install/setup.bash
```

## 실행

1. TurtleBot4 브링업 및 Nav2(경로계획)가 먼저 떠 있어야 한다 (별도 launch,
   예: `turtlebot4_navigation nav2.launch.py`, `turtlebot4_navigation
   slam.launch.py` 또는 저장된 맵으로 `localization.launch.py` 등 기존 환경
   구성을 따른다).

2. launch 로 두 노드를 함께 실행한다:

   ```bash
   ros2 launch amr_aruco amr_aruco.launch.py
   # 다른 로봇에서 실행할 때
   ros2 launch amr_aruco amr_aruco.launch.py namespace:=robot6
   # 파라미터 yaml 교체
   ros2 launch amr_aruco amr_aruco.launch.py params_file:=/path/to/my_params.yaml
   ```

   단독 실행(개별 노드 + `-p` 오버라이드)도 가능하다:

   ```bash
   ros2 run amr_aruco aruco_detect --ros-args -p aruco_dict:=DICT_5X5_100 -p show_window:=false
   ros2 run amr_aruco amr_patrol_emer_helmet --ros-args -p battery_low_threshold:=0.4
   # 네임스페이스를 직접 줄 때 (amr_patrol_emer_helmet 은 tf 리맵도 필요)
   ros2 run amr_aruco amr_patrol_emer_helmet --ros-args -r __ns:=/robot6 \
     -r /tf:=/robot6/tf -r /tf_static:=/robot6/tf_static
   ```

   기동 후 도킹 상태를 확인하고 초기 pose를 설정한 뒤, Nav2가 활성화되면
   `patrol_cmd` / `emergency_goal` / `helmet_goal` 명령을 기다리는 대기(IDLE)
   상태가 된다.

3. (선택) 로봇 선정을 담당하는 `fleet_fsm`(다른 패키지, 예: `fp_amr_vision`)을
   실행해 실제 순찰/응급/안전모 명령을 이 노드로 내려준다. `fleet_fsm` 없이
   단독으로 동작을 확인하려면 터미널에서 아래처럼 직접 명령을 흉내낼 수 있다:

   ```bash
   # 순찰 시작
   ros2 topic pub -1 --qos-reliability reliable --qos-durability transient_local \
     /robot2/patrol_cmd std_msgs/msg/String \
     '{data: "{\"robot_id\": \"robot2\"}"}'

   # 소화기 포인트에서 아루코 대조 완료를 수동으로 알림 (판정 노드가 없을 때)
   ros2 topic pub -1 /robot2/aruco_check_done std_msgs/msg/Bool "{data: true}"

   # 응급 출동
   ros2 topic pub -1 --qos-reliability reliable --qos-durability transient_local \
     /robot2/emergency_goal std_msgs/msg/String \
     '{data: "{\"x\": 1.0, \"y\": 2.0, \"reason\": \"TEST\"}"}'

   # 응급 상황 종료
   ros2 topic pub -1 --qos-reliability reliable --qos-durability transient_local \
     /robot2/emergency_clear std_msgs/msg/Bool "{data: true}"

   # 안전모 배달 (도킹/idle 상태일 때만 시작됨)
   ros2 topic pub -1 --qos-reliability reliable --qos-durability transient_local \
     /robot2/helmet_goal std_msgs/msg/String \
     '{data: "{\"x\": 1.0, \"y\": 2.0, \"robot_id\": \"robot2\"}"}'
   ```

## 파라미터

기본값과 상세 주석은 [config/amr_aruco_params.yaml](config/amr_aruco_params.yaml)
참고. 토픽 이름/QoS 는 fleet_fsm 과의 프로토콜이라 파라미터가 아닌 코드 상수
(`amr_aruco/common.py`)로 관리한다(바꿔야 하면 리맵핑 `-r` 사용).

### amr_patrol_emer_helmet (노드 이름: `basic_navigator`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `battery_low_threshold` | `0.3` | 배터리 잔량(0.0~1.0)이 이 값 미만이면 순찰 중단 후 도킹 복귀 |
| `dock_position` | `[0.0, 0.0]` | 도킹 스테이션 좌표 [x, y] (map 기준, 초기 pose 설정에도 사용) |
| `dock_approach_position` | `[-0.5, 0.0]` | dock() 전에 먼저 이동해 둘 여유 지점 [x, y] |
| `dock_direction` | `0` | 도킹 방향 (TurtleBot4Directions 도 단위: NORTH=0, WEST=90, ...) |
| `helmet_idle_delay` | `3.0` | idle 상태 유지 후 안전모 배달을 시작하기까지의 유예 시간(초) |
| `helmet_delivery_hold` | `5.0` | 안전모 배달 지점 도착 후 현장 대기 시간(초) |
| `aruco_settle_time` | `3.0` | 소화기 포인트 도착 후 완전 정지를 기다리는 시간(초, 모션 블러 오검출 방지) |
| `aruco_check_timeout` | `5.0` | 아루코 대조 완료 신호를 기다리는 최대 시간(초, 초과 시 다음 지점으로) |
| `patrol_points_file` | `""` | 순찰 웨이포인트 json 경로 (빈 값이면 패키지 내 patrol_points.json) |

### aruco_detect (노드 이름: `aruco_marker_detector`)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `aruco_dict` | `"DICT_4X4_50"` | 사용할 아루코 딕셔너리 (cv2.aruco 상수명, 현장 마커와 일치해야 함) |
| `show_window` | `true` | OpenCV 창(imshow) 표시 여부 (headless 환경에서는 false) |

## 주요 토픽 (네임스페이스 `/robot2` 기준)

| 토픽 | 타입 | 발행 → 구독 |
|---|---|---|
| `/robot2/oakd/rgb/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 드라이버 → `aruco_detect` |
| `/robot2/aruco_scan_enable` | `std_msgs/Bool` | `amr_patrol_emer_helmet` → `aruco_detect` |
| `/robot2/aruco/detection/compressed` | `sensor_msgs/CompressedImage` | `aruco_detect` → (디버그 뷰어) |
| `/robot2/aruco/detection/ids` | `std_msgs/Int32MultiArray` | `aruco_detect` → (마커 판정 로직) |
| `/robot2/aruco_check_done` | `std_msgs/Bool` | (마커 판정 로직/수동) → `amr_patrol_emer_helmet` |
| `/robot2/patrol_cmd` | `std_msgs/String` | `fleet_fsm`/수동 → `amr_patrol_emer_helmet` |
| `/robot2/emergency_goal` | `std_msgs/String` | `fleet_fsm`/수동 → `amr_patrol_emer_helmet` |
| `/robot2/emergency_clear` | `std_msgs/Bool` | `fleet_fsm`/수동 → `amr_patrol_emer_helmet` |
| `/robot2/helmet_goal` | `std_msgs/String` | `fleet_fsm`/수동 → `amr_patrol_emer_helmet` |
| `/robot2/amr_status` | `std_msgs/String` | `amr_patrol_emer_helmet` → `fleet_fsm` |

명령/상태 토픽(`patrol_cmd`, `emergency_*`, `helmet_goal`, `amr_status`,
`aruco_scan_enable`)은 RELIABLE + TRANSIENT_LOCAL(latched) QoS
(`common.COMMAND_QOS`)를 쓰므로, 터미널 테스트 시에도 위 예시처럼 QoS 옵션을
맞춰야 한다.

## 의존성

`rclpy`, `ros2launch`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `cv_bridge`,
`nav2_simple_commander`, `turtlebot4_navigation`, OpenCV(`python3-opencv`) —
모두 `package.xml`에 선언되어 있으며, TurtleBot4/Nav2 표준 설치 환경에
포함되어 있어야 한다.
