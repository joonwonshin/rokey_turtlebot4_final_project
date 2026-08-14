# PC3 → PC4 ROS 2 인터페이스 명세

PC3(웹캠 안전 감시)가 발행하는 토픽 9종의 계약서.
PC4(fleet manager / dashboard)는 이 문서만 보고 구독하면 된다.

- 노드 이름: `pc3_safety_bridge`
- 좌표 프레임: `map` (아래 "좌표계" 절 참고)
- 실행: `python3 12_dual_camera_entry_yolo_tracking_modular.py --mode run --cam0-id 4 --cam1-id 0 --device 0`

---

## 0. 접속 환경 (제일 먼저 확인할 것)

| 항목 | 값 |
|---|---|
| `ROS_DOMAIN_ID` | **6** |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| Discovery | Fast DDS **Discovery Server** (`ROS_DISCOVERY_SERVER`) |

### ⚠ 자주 겪는 함정 3가지

1. **`ros2 topic list`가 비어 있는데 RViz는 잘 뜬다**
   → `ros2` CLI는 백그라운드 **데몬**에게 묻는다. 데몬은 처음 뜰 때의 환경변수를 물고 산다.
   환경변수를 바꿨으면 반드시:
   ```bash
   ros2 daemon stop      # 다음 ros2 명령 때 새 환경으로 재기동
   # 또는
   ros2 topic list --no-daemon
   ```
   RViz·rclpy 노드는 데몬을 쓰지 않으므로 이 문제와 무관하다.

2. **Discovery Server가 안 잡히면 같은 PC 안에서도 통신이 안 된다**
   서버가 살아있고 **같은 서브넷**인지 확인.
   ```bash
   nc -vz <discovery_server_ip> 11811
   ```
   서버 없이 로컬 단독 테스트만 할 거면 `unset ROS_DISCOVERY_SERVER`.
   (**PC3↔PC4 연동 시에는 절대 unset 하지 말 것**)

3. **Discovery Server 모드에서 `ros2` CLI가 토픽을 못 본다**
   → `export ROS_SUPER_CLIENT=True` 필요.

---

## 1. 토픽 요약

| # | 토픽 | 타입 | QoS | 발행 패턴 |
|---|---|---|---|---|
| 1 | `/safety/emergency_state` | `std_msgs/String` | RELIABLE, **TRANSIENT_LOCAL**, depth 1 | 엣지 (상태 변화 시 1회) |
| 2 | `/safety/emergency_goal` | `geometry_msgs/PoseStamped` | RELIABLE, VOLATILE, depth 1 | 엣지 (진입 시 **딱 1회**) |
| 3 | `/safety/helmet_state` | `std_msgs/String` | RELIABLE, **TRANSIENT_LOCAL**, depth 1 | 엣지 |
| 4 | `/safety/helmet_goal` | `geometry_msgs/PoseStamped` | RELIABLE, VOLATILE, depth 1 | **Follow, 최대 2 Hz** |
| 5 | `/safety/unauthorized_state` | `std_msgs/String` | RELIABLE, **TRANSIENT_LOCAL**, depth 1 | 엣지 |
| 6 | `/safety/unauthorized_person` | `geometry_msgs/PoseStamped` | RELIABLE, VOLATILE, depth 1 | 엣지 (1회) |
| 7 | `/safety/persons` | `visualization_msgs/MarkerArray` | **BEST_EFFORT**, VOLATILE, depth 1 | 매 프레임 (~15 Hz) |
| 8 | `/safety/cam0/image/compressed` | `sensor_msgs/CompressedImage` | **BEST_EFFORT**, VOLATILE, depth 1 | ~15 Hz, JPEG q75 |
| 9 | `/safety/cam1/image/compressed` | `sensor_msgs/CompressedImage` | **BEST_EFFORT**, VOLATILE, depth 1 | ~15 Hz, JPEG q75 |

### QoS를 반드시 맞출 것

QoS가 안 맞으면 **구독이 아예 연결되지 않는다** (에러도 안 난다).

- `*_state` → `TRANSIENT_LOCAL`. 늦게 붙은 구독자도 **마지막 상태를 즉시** 받는다.
  PC4 대시보드가 재시작해도 현재 EMERGENCY 여부를 바로 안다.
- `*_goal`, `unauthorized_person` → `VOLATILE`. **절대 latch 하지 않는다.**
  latch 하면 나중에 붙은 구독자가 **이미 처리된 옛 goal**을 받아 AMR이 재출동한다.
- `persons`, `image` → `BEST_EFFORT`. RELIABLE로 구독하면 연결되지 않는다.

---

## 2. 상태 토픽 (1, 3, 5)

`std_msgs/String`, `data` 필드에 라벨.

| 토픽 | ON | OFF |
|---|---|---|
| `emergency_state` | `EMERGENCY` | `EMERGENCY_CLEAR` |
| `helmet_state` | `NO_HELMET` | `HELMET_CLEAR` |
| `unauthorized_state` | `UNAUTHORIZED` | `UNAUTHORIZED_CLEAR` |

**엣지 발행**: 상태가 바뀌는 순간에만 1회. 지속 중에는 침묵.
PC3 종료 시 진행 중이던 상태에 대해 `*_CLEAR`를 한 번 뿌린다 (stale 방지).

---

## 3. 좌표 토픽 (2, 4, 6)

`geometry_msgs/PoseStamped`

```
header.frame_id = "map"
header.stamp    = PC3 의 ROS clock
pose.position.x = 사람의 발 위치 X [m]
pose.position.y = 사람의 발 위치 Y [m]
pose.position.z = 0.0
pose.orientation = (0, 0, 0, 1)   ← 항등. 의미 없음. 아래 참고.
```

### 🔴 orientation은 자리표시자다 (PC4가 반드시 재계산)

PC3는 **어느 AMR이 배정될지 모른다.** 따라서 접근 각도(yaw)를 계산할 수 없다.
`(0,0,0,0)`은 무효한 쿼터니언이라 최소한 `w=1`만 채웠다.

### 🔴 position은 "사람의 발 위치"다 (goal 이 아니다)

로봇이 사람 위로 갈 수는 없다. PC4가 **접근점 + yaw**를 계산해야 한다.

```python
import math

STANDOFF = 0.7   # 사람 앞 몇 m 에 설 것인가

def approach_pose(person_xy, robot_xy, standoff=STANDOFF):
    px, py = person_xy
    rx, ry = robot_xy
    dx, dy = px - rx, py - ry
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return px, py, 0.0
    ux, uy = dx / d, dy / d
    ax, ay = px - ux * standoff, py - uy * standoff   # 접근점
    yaw = math.atan2(uy, ux)                          # 접근점에서 사람을 바라봄
    return ax, ay, yaw
```

### ⚠ Nav2가 goal을 거부할 수 있다

`final_project.pgm` 기준 장애물 여유(distance transform) 실측:

| 구역 | 여유 ≥ 0.35 m 인 비율 |
|---|---|
| cam0 감시 영역 | **57 %** |
| cam1 감시 영역 | **55 %** |
| Entry ROI 꼭짓점 | 0.00 ~ 0.31 m (**전부 위험**) |

TurtleBot4 반경 ≈ 0.18 m + Nav2 inflation을 더하면, 벽 근처 사람에 대한 접근점은
경로가 안 나온다. **접근점이 costmap에서 free인지 확인하고, 막히면 사람 주변을 각도로 스캔**할 것.

```python
for off in (0, 30, -30, 60, -60, 90, -90, 120, -120, 180):
    ax, ay = person + standoff * unit(yaw + radians(off))
    if costmap_is_free(ax, ay):
        break
```

---

## 4. 발행 패턴별 의미

### `emergency_goal` — 엣지, 딱 1회

쓰러진 사람은 **안 움직인다.** 진입 순간 좌표를 한 번 보내면 충분하다.
지속 중 재발행하면 Nav2가 계속 replan 한다.

- `emergency_state = EMERGENCY` 와 **동시에** 1회 발행
- 해제 시에는 `EMERGENCY_CLEAR` 만 발행 (goal 재전송 없음)

> **PC4는 goal 을 받은 뒤 스스로 기억해야 한다.** 다시 안 온다.

### `helmet_goal` — Follow, 최대 2 Hz

헬멧 없이 **걸어다니는** 사람을 추적한다.

- EMA 스무딩 (α = 0.3)
- 발행 상한 **2 Hz** (`--helmet-goal-fps`)
- 직전 발행 위치에서 **10 cm 이상** 이동했을 때만 재발행
- 대상 인물이 바뀌어도 rate limit 준수

### `unauthorized_person` — 엣지, 1회

**AMR 출동 없음.** 관리자 대시보드 경보용 좌표.

---

## 5. 🔴 helmet_goal 억제 규칙 (안전상 중요)

다음 상태의 사람에게는 **`helmet_alert`가 켜지지 않고, 따라서 `helmet_goal`이 발행되지 않는다.**

| 상태 | 이유 |
|---|---|
| `lying_candidate` / `emergency` | 누우면 머리가 가려져 헬멧이 검출 안 된다. 그걸 '미착용'으로 읽고 배달 로봇을 보내면 **긴급 구조가 필요한 사람 옆으로 엉뚱한 로봇이 움직인다.** 긴급이 항상 우선. |
| `unauthorized` | 침입자에게 헬멧을 배달할 이유가 없다. 관리자 경보로만 다룬다. |

PC3 내부에서 3중으로 막는다 (상태기계 / 융합 후 / 발행 직전).
**PC4도 방어적으로 한 번 더 확인하는 것을 권장한다.**

---

## 6. `/safety/persons` (MarkerArray)

RViz 시각화용. 매 프레임 발행.

- `markers[0]` = `action: DELETEALL` (이전 프레임 잔재 제거)
- 이후 사람 1명당 `SPHERE` 하나

```
ns        = "safety_persons"
id        = 프레임 내 0,1,2...  (프레임마다 재할당됨. 영속 ID 아님)
type      = SPHERE
scale     = 0.35 m
position  = (map_x, map_y, 0.2)   # 발 좌표 위 20cm
lifetime  = 1.0 s
text      = "#<display_id>|<global_id>"
```

### 색상 = 상태 (우선순위 순)

| 색 | RGBA | 상태 |
|---|---|---|
| 빨강 | 1.0, 0.0, 0.0 | `emergency` |
| 주황 | 1.0, 0.55, 0.0 | `helmet_alert` |
| **보라** | 0.78, 0.0, 1.0 | `unauthorized` (침입자) |
| 초록 | 0.0, 1.0, 0.0 | `access_granted` |
| 하늘 | 0.2, 0.8, 1.0 | 정상 |

### 중복 제거

한 사람이 두 카메라에 동시에 잡혀도 **마커는 하나**다.
`global_id`(카메라 간 인격 통합) → 위치 근접(0.5 m) 2단 dedup 적용.

### RViz 설정

`MarkerArray` display의 **Reliability를 `Best Effort`로** 바꿔야 보인다 (기본값 `Reliable`).

---

## 7. 카메라 영상 (8, 9)

`sensor_msgs/CompressedImage`

```
header.frame_id = "cam0" | "cam1"
format          = "jpeg"
data            = JPEG bytes (quality 75)
```

- 상한 **15 fps** (`--ros-image-fps`). 루프가 더 빨라도 이 이상 안 나간다.
- 이미 **오버레이(bbox / 스켈레톤 / 상태 라벨 / Entry ROI)가 그려진** 프레임이다.
- 720p 기준 30~80 KB/frame → 약 0.5~1.2 MB/s per camera.

```python
import cv2, numpy as np
img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
```

---

## 8. 좌표계

PC3는 `final_project.yaml` 의 map 좌표계를 그대로 쓴다.

| | |
|---|---|
| resolution | 0.05 m/px |
| origin | (-5.23, -1.38, 0) |
| size | 201 × 241 px = 10.05 × 12.05 m |
| map 범위 | x ∈ [-5.23, 4.82], y ∈ [-1.38, 10.67] |

**로봇이 같은 pgm/yaml을 로드하면 두 `map` 프레임은 정의상 동일하다.**

검증법: 로봇의 `/map` 헤더 비교
```bash
ros2 topic echo /map --once --field info
# resolution / origin.position 이 위 값과 같아야 한다
```

### 좌표 정확도 (재투영 오차, 실측)

| 카메라 | 호모그래피 평균 오차 |
|---|---|
| cam0 (입구) | **3.7 cm** (inlier 15/16) |
| cam1 (작업장) | **6.7 cm** (7점) |

키포인트 노이즈까지 포함한 실사용 오차는 **10 cm 내외**로 보면 된다.

---

## 9. 타이밍 (상태 전이)

| 이벤트 | 진입 | 해제 |
|---|---|---|
| Emergency | 쓰러짐 **5 s** 지속 | 회복 **3 s** 지속 |
| Helmet Alert | 미착용 **3 s** 지속 | 착용 **2 s** 지속 |
| Access Granted | ROI 안 + 손 듦 + 헬멧 **3 s** | — |
| Track Hold | — | 검출 끊김 **0.8 s** 유지 |

해제에도 시간 조건을 둔 것은 **히스테리시스**다. 경계에서 상태가 떨리는 것을 막는다.

---

## 10. PC4 구현 체크리스트

- [ ] QoS를 표대로 맞춘다 (특히 `persons`/`image` = **BEST_EFFORT**)
- [ ] `*_goal` 은 **VOLATILE로 구독**한다 (latch 금지)
- [ ] `emergency_goal` 은 **한 번만 온다.** 받으면 저장할 것
- [ ] `position` 은 **사람의 발 위치**다. **접근점 + yaw를 직접 계산**할 것
- [ ] 접근점이 costmap에서 free인지 확인하고, 막히면 각도 스캔
- [ ] `orientation` 은 무시하고 재계산할 것
- [ ] `helmet_goal` 대상이 `emergency`/`unauthorized` 가 아닌지 방어적으로 재확인
- [ ] `unauthorized_person` 에는 **AMR을 보내지 않는다** (대시보드 경보만)
- [ ] PC3 종료 시 `*_CLEAR` 가 오므로 그때 상태를 리셋할 것
- [ ] 환경변수 바꾸면 `ros2 daemon stop`

---

## 11. 빠른 확인 명령

```bash
export ROS_DOMAIN_ID=6
ros2 daemon stop                      # 환경 바꿨으면 필수

ros2 topic list | grep safety         # 9개
ros2 topic hz  /safety/persons        # ~15 Hz
ros2 topic hz  /safety/cam0/image/compressed   # ~15 Hz
ros2 topic echo /safety/emergency_state        # TRANSIENT_LOCAL 이라 즉시 마지막 값
ros2 topic echo /safety/emergency_goal         # 엣지라 사건 없으면 조용함
ros2 topic info /safety/persons --verbose      # QoS 확인
```
