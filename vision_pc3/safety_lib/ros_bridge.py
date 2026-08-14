"""
safety_lib.ros_bridge
=====================

PC3 (webcam detection) → PC4 (fleet manager / dashboard) ROS 2 bridge.

이 모듈 하나가 PC3에서 나가는 모든 ROS 2 발행을 담당한다.
메인 스크립트는 매 프레임 SafetyRosBridge의 세 함수만 부르면 된다:

    bridge.publish_camera_frame(cam_name, bgr)   # cam0 / cam1 오버레이 스트림
    bridge.publish_state_and_goals(all_dets)     # 상태·goal (엣지 / follow)
    bridge.publish_persons_markers(all_dets)     # RViz 시각화

발행 토픽 (총 9개):
    /safety/emergency_state       std_msgs/String                (엣지)
    /safety/emergency_goal        geometry_msgs/PoseStamped      (엣지 1회)
    /safety/helmet_state          std_msgs/String                (엣지)
    /safety/helmet_goal           geometry_msgs/PoseStamped      (follow 2Hz)
    /safety/unauthorized_state    std_msgs/String                (엣지)
    /safety/unauthorized_person   geometry_msgs/PoseStamped      (엣지 1회)
    /safety/persons               visualization_msgs/MarkerArray (매 프레임)
    /safety/cam0/image/compressed sensor_msgs/CompressedImage    (스트림 15Hz)
    /safety/cam1/image/compressed sensor_msgs/CompressedImage    (스트림 15Hz)
    /safety/persons_json          std_msgs/String (JSON)         (스트림 5Hz)
                                  웹 대시보드(rosbridge)용. MarkerArray 를
                                  파싱하지 않고 바로 그릴 수 있게 사람별
                                  좌표+상태를 JSON 으로 낸다. RELIABLE 이라
                                  rosbridge 기본 QoS 로 그냥 붙는다.

세 가지 발행 패턴:
    ┌────────────┬──────────────────────────────────────────────┐
    │ 엣지       │ 상태 진입/해제 순간 한 번만 발행. 지속 중엔  │
    │            │ 침묵. Nav2 replan 폭발 방지.                 │
    │            │ emergency(정지 대상), unauthorized(경보만).   │
    ├────────────┼──────────────────────────────────────────────┤
    │ Follow     │ EMA 스무딩 + 발행 주기 상한 + 이동 임계값.   │
    │            │ 사람이 걸어다니는 대상 (helmet_goal).         │
    │            │ raw 흔들림은 EMA로 흡수, 미세 이동은 skip.   │
    ├────────────┼──────────────────────────────────────────────┤
    │ Stream     │ 매 프레임(또는 상한 fps)마다 발행.           │
    │            │ 카메라 오버레이 영상, RViz 마커.             │
    └────────────┴──────────────────────────────────────────────┘

rclpy 임포트는 이 모듈 안에서만 발생하므로, ROS 미설치 환경(Windows)에서
메인 스크립트를 --no-publish-ros로 실행하면 이 파일 자체가 로드되지 않아
ImportError를 회피할 수 있다.
"""
import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
# QoS 프로파일 3종
# ============================================================
# ROS 2의 QoS는 4개 축(Reliability/Durability/History/Depth) 조합으로
# 통신 특성을 결정한다. 세 종류로 나눠 재사용.

def _qos_state():
    """
    상태 라벨용 (emergency_state, helmet_state, unauthorized_state).

    - RELIABLE       : 놓치면 안 되는 이벤트. TCP 유사.
    - TRANSIENT_LOCAL: publisher가 마지막 값을 저장. 나중에 붙는
                       subscriber(예: PC4 대시보드가 뒤늦게 재시작)에게
                       현재 상태를 즉시 전달. latched topic 개념.
    - KEEP_LAST(1)   : 마지막 1개만 유지 → publish() 블록 방지.
    """
    return QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


def _qos_goal():
    """
    좌표 명령용 (emergency_goal, helmet_goal, unauthorized_person).

    - RELIABLE : 명령 손실 = AMR 미출동. 절대 손실 금지.
    - VOLATILE : latch 안 함. 옛 goal이 나중 구독자에게 재전송되면
                 처리 완료된 위치로 AMR이 재출동하는 사고 발생.
                 그래서 반드시 VOLATILE.
    - KEEP_LAST(1)
    """
    return QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


def _qos_stream():
    """
    시각화·영상 스트림용 (persons MarkerArray, cam*/image).

    - BEST_EFFORT: UDP 유사. 재전송 없음. 스트림은 한 프레임 놓쳐도
                   다음 프레임이 새로 옴. RELIABLE 걸면 늦은 구독자
                   때문에 publisher가 대기해서 loop rate가 무너짐.
    - VOLATILE
    - KEEP_LAST(1)
    """
    return QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
    )


# ============================================================
# SafetyRosBridge
# ============================================================

class SafetyRosBridge:
    """
    PC3의 모든 ROS 2 발행을 담당하는 단일 진입점.

    Usage:
        bridge = SafetyRosBridge(cam_names=("cam0","cam1"))
        while running:
            bridge.publish_camera_frame("cam0", cam0_view)
            bridge.publish_camera_frame("cam1", cam1_view)
            bridge.publish_state_and_goals(all_detections)
            bridge.publish_persons_markers(all_detections)
            bridge.spin_some()
        bridge.shutdown()
    """

    def __init__(
        self,
        cam_names=("cam0", "cam1"),
        node_name="pc3_safety_bridge",
        map_frame="map",
        jpeg_quality=75,
        image_fps=15.0,
        marker_lifetime_sec=1.0,
        emergency_state_on="EMERGENCY",
        emergency_state_off="EMERGENCY_CLEAR",
        helmet_state_on="NO_HELMET",
        helmet_state_off="HELMET_CLEAR",
        unauthorized_state_on="UNAUTHORIZED",
        unauthorized_state_off="UNAUTHORIZED_CLEAR",
        dedup_dist=0.5,
        helmet_goal_fps=2.0,
        helmet_goal_ema_alpha=0.3,
        helmet_goal_min_move=0.10,
        persons_json_fps=5.0,
        clear_debounce_sec=1.5,
    ):
        """
        Args:
            cam_names            : CompressedImage 발행 대상 카메라 이름.
            node_name            : ROS 2 노드 이름.
            map_frame            : PoseStamped/Marker의 frame_id.
            jpeg_quality         : cv2.imencode JPEG 품질 (1~100).
            image_fps            : cam*/image 상한 fps. loop가 30fps라도
                                    이 값으로 rate limit.
            marker_lifetime_sec  : RViz marker 유지 시간.
            *_state_on/off       : 상태 라벨 문자열.
            dedup_dist           : 위치 근접 dedup 임계값 [m].
                                    GlobalFuser의 첫 프레임 co-observation
                                    한계를 커버하는 안전망.
            helmet_goal_fps      : helmet_goal follow 모드 발행 상한 [Hz].
            helmet_goal_ema_alpha: helmet_goal EMA 스무딩 (0<α≤1).
            helmet_goal_min_move : 이 거리 이하 이동은 goal 재발행 skip.
            clear_debounce_sec   : 상태 해제(CLEAR) 디바운스 [s]. 대상이
                                    "0명인 상태"가 이 시간 이상 연속으로
                                    유지돼야 CLEAR 를 발행한다. 아래 참조.
        """
        # rclpy 컨텍스트 초기화 — 이 모듈이 여러 번 임포트되어도 중복
        # init 방지. rclpy.ok()가 False일 때만 init.
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)
        self.map_frame = str(map_frame)
        self.jpeg_quality = int(jpeg_quality)

        # image_fps → period 로 변환. 나중에 (now - last) < period 이면 skip.
        self.image_period = 1.0 / max(1e-3, float(image_fps))
        self.marker_lifetime_sec = float(marker_lifetime_sec)

        # 상태 라벨
        self.emergency_state_on = str(emergency_state_on)
        self.emergency_state_off = str(emergency_state_off)
        self.helmet_state_on = str(helmet_state_on)
        self.helmet_state_off = str(helmet_state_off)
        self.unauthorized_state_on = str(unauthorized_state_on)
        self.unauthorized_state_off = str(unauthorized_state_off)

        # dedup / follow 파라미터
        self.dedup_dist = float(dedup_dist)
        self.helmet_goal_period = 1.0 / max(1e-3, float(helmet_goal_fps))
        self.helmet_goal_ema_alpha = float(helmet_goal_ema_alpha)
        self.helmet_goal_min_move = float(helmet_goal_min_move)
        self.persons_json_period = 1.0 / max(1e-3, float(persons_json_fps))
        self.persons_json_last = 0.0

        # --------------------------------------------------------------
        # Publisher 생성
        # --------------------------------------------------------------
        state_qos = _qos_state()
        goal_qos = _qos_goal()
        stream_qos = _qos_stream()

        # 3 종류 × 2 (state + goal) = 6 개 이벤트 토픽
        self.pub_emg_state = self.node.create_publisher(
            String, "/safety/emergency_state", state_qos
        )
        self.pub_emg_goal = self.node.create_publisher(
            PoseStamped, "/safety/emergency_goal", goal_qos
        )
        self.pub_hlm_state = self.node.create_publisher(
            String, "/safety/helmet_state", state_qos
        )
        self.pub_hlm_goal = self.node.create_publisher(
            PoseStamped, "/safety/helmet_goal", goal_qos
        )
        self.pub_unauth_state = self.node.create_publisher(
            String, "/safety/unauthorized_state", state_qos
        )
        self.pub_unauth_person = self.node.create_publisher(
            PoseStamped, "/safety/unauthorized_person", goal_qos
        )

        # RViz용 통합 시각화
        self.pub_persons = self.node.create_publisher(
            MarkerArray, "/safety/persons", stream_qos
        )

        # 웹 대시보드용 JSON. rosbridge 는 QoS 지정이 어려워 BEST_EFFORT 토픽에
        # 붙지 못하므로, 브라우저가 바로 먹을 수 있는 RELIABLE String 으로 낸다.
        self.pub_persons_json = self.node.create_publisher(
            String, "/safety/persons_json", goal_qos
        )

        # 카메라별 CompressedImage — dict로 관리해 카메라 추가 확장 용이
        self.image_pubs = {}
        self.image_last_pub = {}  # rate limit용 마지막 발행 시각

        for name in cam_names:
            topic = f"/safety/{name}/image/compressed"
            self.image_pubs[name] = self.node.create_publisher(
                CompressedImage, topic, stream_qos
            )
            self.image_last_pub[name] = 0.0

        # --------------------------------------------------------------
        # 상태 기억용 필드
        # --------------------------------------------------------------
        # 엣지 트리거 판단은 "이전 프레임 상태"와 "현재 프레임 상태"의
        # 비교로 이뤄진다. 세 상태 각각 prev 플래그 유지.
        self.prev_emergency = False
        self.prev_helmet_alert = False
        self.prev_unauthorized = False

        # ── CLEAR 디바운스 ───────────────────────────────────────────
        # WorkerStateStore 는 트랙 1개 단위로는 히스테리시스가 충분하다
        # (쓰러짐 5초 지속 -> emergency, 일어섬 3초 지속 -> 해제).
        # 그런데 여기서 쓰는 emg_now 는 "응급인 사람이 한 명이라도 있나" 라는
        # 집계 신호이고, 여기엔 히스테리시스가 전혀 없었다. 그래서
        #   * 사람이 막 쓰러지는 순간 YOLO 트랙 ID 가 한 프레임 튀거나
        #   * 누운 자세에서 발끝/키 추정이 한 프레임 무효가 되어 map_xy 가 없거나
        # 하면 그 프레임만 대상이 0명이 되고, 즉시 EMERGENCY_CLEAR 가 나갔다.
        # 실측: 쓰러진 직후 1초 동안 ON/CLEAR 가 0.1초 간격으로 5번 뒤집힘.
        #
        # 이건 그냥 지저분한 로그가 아니라 위험하다. 이 상태 토픽은
        # safety_alert_bridge -> fleet_fsm -> 로봇으로 이어지므로, 로봇이
        # 출동 시작 -> 즉시 취소 -> 복귀 -> 재출동을 반복하게 된다.
        #
        # 그래서 "대상 0명" 이 clear_debounce_sec 동안 연속으로 유지될 때만
        # CLEAR 를 낸다. 진짜 회복은 트랙 상태기계에서 이미 3초를 쓰므로,
        # 이 디바운스가 정상 해제를 놓치는 일은 없다.
        # (반대로 ON 은 디바운스하지 않는다 - 응급은 늦게 알리면 안 된다)
        self.clear_debounce_sec = float(clear_debounce_sec)
        self.emg_absent_since = None      # emergency 대상이 0명이 된 시각
        self.hlm_absent_since = None
        self.uauth_absent_since = None

        # emergency 대상 좌표 기록 (사고 대응 후 재검증용, 실사용은 로그만)
        self.last_emergency_target = None

        # helmet follow 모드 상태 — 사람이 걸어다니는 대상 추적
        self.helmet_smoothed_xy = None       # EMA로 스무딩된 좌표
        self.helmet_last_published_xy = None # 마지막으로 goal에 실려나간 좌표
        self.helmet_last_pub_time = 0.0      # rate limit 기준 시각
        self.helmet_current_target_key = None  # 대상 인물 식별자

        self.node.get_logger().info(
            f"SafetyRosBridge up. cams={list(cam_names)} "
            f"jpeg_q={self.jpeg_quality} image_fps={1.0/self.image_period:.1f} "
            f"helmet_goal_fps={helmet_goal_fps} ema_alpha={helmet_goal_ema_alpha}"
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _stamp_now(self):
        """ROS clock 기반의 현재 시각을 msg 헤더에 넣을 형태로 반환."""
        return self.node.get_clock().now().to_msg()

    def _map_xy_of(self, det):
        """
        detection dict에서 map 좌표(x, y) [m] 추출.

        GlobalFuser가 붙여놓은 통합 좌표(global_map_xy)를 우선 사용하고,
        없거나 무효면 원본 map_xy로 폴백. NaN/inf 방어.
        """
        for key in ("global_map_xy", "map_xy"):
            v = det.get(key)
            if v is None:
                continue
            try:
                arr = np.array(v, dtype=np.float64).reshape(-1)[:2]
            except Exception:
                continue
            if arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
                continue
            return float(arr[0]), float(arr[1])
        return None

    def _dedup_key(self, det):
        """
        detection의 dedup 식별자 반환.

        우선순위:
          1) global_id  — GlobalFuser 매핑 성공 시 카메라 넘어 동일 인격
          2) state_key  — WorkerStateStore 로컬 track key ("cam0:7")
          3) 조합 폴백  — 위 둘 다 없을 때
        """
        gid = det.get("global_id")
        if gid is not None:
            return f"G:{gid}"
        sk = det.get("state_key")
        if sk is not None:
            return f"S:{sk}"
        cam = det.get("camera", "cam")
        yid = det.get("yolo_track_id")
        if yid is None:
            return None
        return f"L:{cam}:{int(yid)}"

    def _collect_flagged(self, detections, flag_key):
        """
        flag_key(예: "emergency", "helmet_alert", "unauthorized")가 True인
        detection을 뽑아 dedup된 [(det, xy), ...] 리스트로 반환.

        2단계 dedup:
          1) global_id / state_key 기반 (같은 인격은 하나만)
          2) 위치 근접 (dedup_dist 이내 → 우선순위 높은 쪽만 유지)
             → GlobalFuser가 첫 프레임 co-observation을 병합 못하는 한계
                를 커버하는 belt-and-suspenders 안전망.

        우선순위 기준:
          emergency    → fall_elapsed (오래 쓰러진 사람 우선)
          helmet_alert → helmet_alert_elapsed (오래 미착용 사람 우선)
          그 외        → conf (검출 신뢰도 높은 쪽)
        """
        # ------- 1st pass: identity dedup -------
        picked = {}
        for det in detections:
            if not bool(det.get(flag_key, False)):
                continue

            # helmet_goal 은 "이 사람에게 헬멧을 배달하라"는 AMR 출동 명령이다.
            # 쓰러진 사람 / 긴급 상태 / 무단 침입자에게는 절대 보내면 안 된다.
            # WorkerStateStore 가 이미 helmet_alert 를 끄지만, 발행 직전에
            # 한 번 더 막는다 (AMR 이 움직이는 명령이므로 belt-and-suspenders).
            if flag_key == "helmet_alert" and (
                det.get("emergency", False)
                or det.get("lying_candidate", False)
                or det.get("unauthorized", False)
            ):
                continue
            xy = self._map_xy_of(det)
            if xy is None:
                continue
            key = self._dedup_key(det)
            if key is None:
                continue
            if key in picked:
                continue
            picked[key] = (det, xy)

        entries = list(picked.values())

        # dedup_dist 비활성이거나 후보가 1개 이하면 그대로 반환
        if self.dedup_dist <= 0.0 or len(entries) <= 1:
            return entries

        # ------- 2nd pass: proximity dedup -------
        def _priority(det):
            if flag_key == "emergency":
                return float(det.get("fall_elapsed", 0.0))
            if flag_key == "helmet_alert":
                return float(det.get("helmet_alert_elapsed", 0.0))
            return float(det.get("conf", 0.0))

        # 우선순위 내림차순 정렬 → 앞에서부터 근접 클러스터를 흡수
        sorted_entries = sorted(entries, key=lambda e: -_priority(e[0]))
        kept = []
        dist_sq = self.dedup_dist * self.dedup_dist

        for det, xy in sorted_entries:
            duplicate = False
            for _kept_det, kept_xy in kept:
                dx = xy[0] - kept_xy[0]
                dy = xy[1] - kept_xy[1]
                if dx * dx + dy * dy <= dist_sq:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((det, xy))

        return kept

    def _make_pose(self, xy):
        """
        (x, y) [m] → PoseStamped in map frame.

        ── orientation은 의도적으로 identity(w=1)로 남긴다 ──
        AMR 접근 방향(yaw)은 PC3의 관심사가 아니다. 이유:
          1) PC3는 관측(perception)만 담당한다. AMR 배정·경로·접근각도는
             PC4의 fleet_manager / mission_orchestrator 소관.
          2) 여러 AMR(AMR1, AMR2) 중 어느 로봇이 배정될지 PC3는 모른다.
             그러니 AMR 위치 기반 yaw 계산이 원천적으로 불가.
          3) PC4에서 이 goal을 받으면:
               - 가까운 AMR 선택 → 그 AMR의 /amcl_pose 확인
               - 사람 앞 0.7m 접근점 + yaw = atan2(person - approach) 계산
               - 완성된 PoseStamped를 해당 AMR의 Nav2 goal action으로 전달
          4) quaternion (0,0,0,0)은 무효라서 최소한 w=1은 채워야 함.
             이 값은 다운스트림이 무시하고 재계산하는 placeholder.
        """
        p = PoseStamped()
        p.header.stamp = self._stamp_now()
        p.header.frame_id = self.map_frame
        p.pose.position.x = float(xy[0])
        p.pose.position.y = float(xy[1])
        p.pose.position.z = 0.0
        # 아래 4줄은 placeholder — PC4가 AMR 위치 기반으로 재계산할 것.
        p.pose.orientation.x = 0.0
        p.pose.orientation.y = 0.0
        p.pose.orientation.z = 0.0
        p.pose.orientation.w = 1.0
        return p

    def _publish_string(self, pub, data):
        """String 메시지 래퍼."""
        m = String()
        m.data = str(data)
        pub.publish(m)

    def _debounce_clear(self, raw_now, prev, absent_attr):
        """
        해제(True → False) 방향에만 디바운스를 건다.

          raw_now=True                 → 즉시 True (응급/경보는 늦추면 안 된다)
          raw_now=False, prev=False    → False (이미 꺼져 있음)
          raw_now=False, prev=True     → clear_debounce_sec 동안은 True 를 유지.
                                          그 시간이 지나도록 계속 비어 있으면 False.

        "대상 0명" 이 한 프레임만 스쳐도 CLEAR 가 나가던 것을 막는다.
        절대 놓치면 안 되는 '진짜 해제'는 트랙 상태기계가 이미 3초를 쓰고
        오므로, 1.5초 디바운스로 지연되는 것 외의 부작용은 없다.
        """
        t = time.monotonic()
        if raw_now:
            setattr(self, absent_attr, None)
            return True
        if not prev:
            return False
        since = getattr(self, absent_attr)
        if since is None:
            setattr(self, absent_attr, t)   # 방금 비었다 - 유예 시작
            return True
        if (t - since) < self.clear_debounce_sec:
            return True                     # 유예 중 - 아직 켜진 것으로 본다
        setattr(self, absent_attr, None)
        return False                        # 유예 내내 비어 있었다 - 진짜 해제

    # ------------------------------------------------------------------
    # 메인 발행: 상태 + goal
    # ------------------------------------------------------------------

    def publish_state_and_goals(self, all_detections):
        """
        매 프레임 호출.

        - Emergency    : 엣지. 진입 순간 state + goal 각 1회, 해제 순간
                          state clear 1회. 사람이 쓰러져 정지 상태.
        - Helmet       : Follow. EMA 스무딩 + 2Hz 상한 + 10cm 이동 임계.
                          걸어다니는 대상 추적.
                          쓰러진 사람 / 침입자는 대상에서 제외 (AMR 오출동 방지).
        - Unauthorized : 엣지. 진입 순간 state + person 좌표 1회.
                          AMR 출동 없음 (대시보드 경보만).
        """
        # 세 상태 각각 dedup 처리한 대상 리스트 취득
        emg_list = self._collect_flagged(all_detections, "emergency")
        hlm_list = self._collect_flagged(all_detections, "helmet_alert")
        uauth_list = self._collect_flagged(all_detections, "unauthorized")

        # 이번 프레임의 raw 신호. 그대로 쓰면 한 프레임만 대상이 비어도
        # 즉시 CLEAR 가 나가므로(트랙 ID 순간 소실 / map_xy 일시 무효),
        # 해제 방향에만 디바운스를 건다. 켜지는 방향은 즉시 통과.
        emg_raw = len(emg_list) > 0
        hlm_raw = len(hlm_list) > 0
        uauth_raw = len(uauth_list) > 0

        emg_now = self._debounce_clear(
            emg_raw, self.prev_emergency, 'emg_absent_since')
        hlm_now = self._debounce_clear(
            hlm_raw, self.prev_helmet_alert, 'hlm_absent_since')
        uauth_now = self._debounce_clear(
            uauth_raw, self.prev_unauthorized, 'uauth_absent_since')

        # =============================================================
        # EMERGENCY : 정적 대상, edge only
        # =============================================================
        # 쓰러진 사람은 안 움직이므로 진입 순간 goal 한 번만 보내면 됨.
        # 지속 중 재발행은 Nav2 replan 유발 → 금지.
        if emg_now and not self.prev_emergency:
            # False → True 엣지: 상태 알림 + goal 좌표
            det, xy = emg_list[0]
            self._publish_string(self.pub_emg_state, self.emergency_state_on)
            self.pub_emg_goal.publish(self._make_pose(xy))
            self.last_emergency_target = xy
            self.node.get_logger().warn(
                f"EMERGENCY ON at map=({xy[0]:.2f},{xy[1]:.2f}) "
                f"cam={det.get('camera')} track={det.get('yolo_track_id')}"
            )
        elif (not emg_now) and self.prev_emergency:
            # True → False 엣지: 해제 알림만 (goal 재전송 X)
            self._publish_string(self.pub_emg_state, self.emergency_state_off)
            self.last_emergency_target = None
            self.node.get_logger().info("EMERGENCY CLEAR")

        # =============================================================
        # HELMET : 이동 대상, follow mode
        # =============================================================
        # 헬멧 없이 걸어다니는 사람을 추적. 매 프레임 raw 좌표 → EMA로
        # 흔들림 줄이고, 2Hz 상한으로 Nav2 replan 부하 조절.
        if hlm_now and not hlm_list:
            # 디바운스 유예 중 (대상이 한두 프레임 사라짐). 상태는 유지하고
            # follow 갱신만 건너뛴다. hlm_list 가 비었으므로 아래 블록의
            # hlm_list[0] 을 그대로 타면 IndexError 가 난다.
            pass
        elif hlm_now:
            det, raw_xy = hlm_list[0]
            target_key = self._dedup_key(det)

            # ---- 스무딩 상태 관리 ----
            if (target_key != self.helmet_current_target_key
                    or self.helmet_smoothed_xy is None):
                # 처음 감지 or 대상 인물 교체 → smoothed 초기화 (raw 사용)
                self.helmet_smoothed_xy = (float(raw_xy[0]), float(raw_xy[1]))
                self.helmet_current_target_key = target_key
                new_target = True
            else:
                # 같은 사람이 계속 잡히는 중 → EMA 업데이트
                # smoothed = α · raw + (1-α) · smoothed_prev
                # α=0.3이면 raw 30% + 과거값 70% → 부드럽게 뒤따라감
                a = self.helmet_goal_ema_alpha
                sx, sy = self.helmet_smoothed_xy
                self.helmet_smoothed_xy = (
                    a * float(raw_xy[0]) + (1.0 - a) * sx,
                    a * float(raw_xy[1]) + (1.0 - a) * sy,
                )
                new_target = False

            # ---- 발행 여부 판단 (3중 조건) ----
            now_t = time.time()
            should_publish = False

            if not self.prev_helmet_alert:
                # 조건 1: alert 자체가 이번에 처음 켜짐 → 엣지, 즉시 발행
                self._publish_string(self.pub_hlm_state, self.helmet_state_on)
                should_publish = True
                self.node.get_logger().warn(
                    f"HELMET ALERT ON at map=({self.helmet_smoothed_xy[0]:.2f},"
                    f"{self.helmet_smoothed_xy[1]:.2f}) "
                    f"cam={det.get('camera')} track={det.get('yolo_track_id')}"
                )
            elif new_target and (now_t - self.helmet_last_pub_time) >= self.helmet_goal_period:
                # 조건 2: 지속 중인데 대상 사람이 바뀜 → 새 goal.
                # 단 rate limit 은 지킨다. 미착용자가 둘이고 우선순위가
                # 엎치락뒤치락하면 target_key 가 매 프레임 바뀌어 goal 이
                # 루프 속도(15~25Hz)로 쏟아진다.
                should_publish = True
            elif (now_t - self.helmet_last_pub_time) >= self.helmet_goal_period:
                # 조건 3: rate limit 통과 시점에 min_move 이상 움직였을 때만
                if self.helmet_last_published_xy is None:
                    should_publish = True
                else:
                    dx = self.helmet_smoothed_xy[0] - self.helmet_last_published_xy[0]
                    dy = self.helmet_smoothed_xy[1] - self.helmet_last_published_xy[1]
                    if (dx * dx + dy * dy) >= (self.helmet_goal_min_move ** 2):
                        should_publish = True

            if should_publish:
                # smoothed 좌표를 goal로 발행 (raw 아님에 주의)
                self.pub_hlm_goal.publish(
                    self._make_pose(self.helmet_smoothed_xy)
                )
                self.helmet_last_published_xy = self.helmet_smoothed_xy
                self.helmet_last_pub_time = now_t
        else:
            # alert 해제
            if self.prev_helmet_alert:
                self._publish_string(self.pub_hlm_state, self.helmet_state_off)
                self.node.get_logger().info("HELMET ALERT CLEAR")

            # follow 상태 리셋 → 다음 alert 때 다시 raw로 시작
            self.helmet_smoothed_xy = None
            self.helmet_last_published_xy = None
            self.helmet_current_target_key = None

        # =============================================================
        # UNAUTHORIZED : 관리자 경보용, edge only
        # =============================================================
        # AMR 출동 없이 대시보드 알림만. 최초 감지 위치만 보내주면 되므로
        # emergency와 동일한 엣지 패턴.
        if uauth_now and not self.prev_unauthorized:
            det, xy = uauth_list[0]
            self._publish_string(self.pub_unauth_state, self.unauthorized_state_on)
            self.pub_unauth_person.publish(self._make_pose(xy))
            self.node.get_logger().warn(
                f"UNAUTHORIZED ENTRY at map=({xy[0]:.2f},{xy[1]:.2f}) "
                f"cam={det.get('camera')} track={det.get('yolo_track_id')}"
            )
        elif (not uauth_now) and self.prev_unauthorized:
            self._publish_string(self.pub_unauth_state, self.unauthorized_state_off)
            self.node.get_logger().info("UNAUTHORIZED CLEAR")

        # 다음 프레임 엣지 판단을 위해 현재 상태 저장
        self.prev_emergency = emg_now
        self.prev_helmet_alert = hlm_now
        self.prev_unauthorized = uauth_now

    # ------------------------------------------------------------------
    # RViz 시각화용 마커
    # ------------------------------------------------------------------

    def publish_persons_markers(self, all_detections):
        """
        전체 인원을 map 프레임 상의 sphere marker로 발행.

        - DELETEALL로 이전 프레임 마커 전부 지운 뒤 새로 그림
          (RViz에서 사람이 사라졌을 때 잔재가 남지 않도록).
        - dedup은 goal과 동일하게 identity + proximity 2단계.
        - 색상은 상태 우선순위: emergency(빨강) > helmet_alert(주황) >
          unauthorized(보라) > access_granted(초록) > 기본(하늘).
        """
        ma = MarkerArray()

        # 이전 프레임의 모든 마커 삭제 신호
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = self._stamp_now()
        clear.ns = "safety_persons"
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # dedup 준비
        seen_keys = set()
        kept_xy = []
        dist_sq = (self.dedup_dist * self.dedup_dist) if self.dedup_dist > 0.0 else -1.0
        idx = 0

        for det in all_detections:
            xy = self._map_xy_of(det)
            if xy is None:
                continue

            # 1차: identity 중복 제거
            key = self._dedup_key(det)
            if key is None or key in seen_keys:
                continue

            # 2차: 위치 근접 중복 제거
            if dist_sq > 0.0:
                duplicate = False
                for kx, ky in kept_xy:
                    dx = xy[0] - kx
                    dy = xy[1] - ky
                    if dx * dx + dy * dy <= dist_sq:
                        duplicate = True
                        break
                if duplicate:
                    continue

            seen_keys.add(key)
            kept_xy.append(xy)

            # ---- Sphere marker 생성 ----
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = clear.header.stamp
            m.ns = "safety_persons"
            m.id = idx  # 프레임 내 고유 인덱스 (DELETEALL 후 새로 할당)
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            # 발 좌표 위 20cm에 sphere 배치 → 지면과 겹치지 않음
            m.pose.position.x = float(xy[0])
            m.pose.position.y = float(xy[1])
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0

            # 지름 35cm
            m.scale.x = 0.35
            m.scale.y = 0.35
            m.scale.z = 0.35

            # 상태 우선순위 → 색상 매핑
            c = ColorRGBA()
            if det.get("emergency", False):
                c.r, c.g, c.b, c.a = 1.0, 0.0, 0.0, 1.0   # 빨강
            elif det.get("helmet_alert", False):
                c.r, c.g, c.b, c.a = 1.0, 0.55, 0.0, 1.0  # 주황
            elif det.get("unauthorized", False):
                c.r, c.g, c.b, c.a = 0.78, 0.0, 1.0, 1.0  # 보라 (침입자)
            elif det.get("access_granted", False):
                c.r, c.g, c.b, c.a = 0.0, 1.0, 0.0, 0.9   # 초록
            else:
                c.r, c.g, c.b, c.a = 0.2, 0.8, 1.0, 0.7   # 하늘 (안전)
            m.color = c

            # 다음 프레임에 갱신 안 되면 자동 소멸 → 잔재 방지 이중 안전장치
            m.lifetime.sec = int(self.marker_lifetime_sec)
            m.lifetime.nanosec = int(
                (self.marker_lifetime_sec - int(self.marker_lifetime_sec)) * 1e9
            )

            # 라벨: display_id + global_id (RViz에서 hover 시 확인용)
            did = det.get("display_id")
            gid = det.get("global_id")
            m.text = f"#{did if did is not None else 'X'}|{gid if gid is not None else '-'}"

            ma.markers.append(m)
            idx += 1

        self.pub_persons.publish(ma)

    def _dedup_persons(self, detections):
        """
        한 사람 = 한 항목. publish_persons_markers 와 동일한 2단 dedup.
          1) global_id (없으면 state_key)
          2) 위치 근접 (dedup_dist)
        우선순위: live > held > emergency > unauthorized > helmet_alert > conf
        """
        picked = {}
        for det in detections:
            xy = self._map_xy_of(det)
            if xy is None:
                continue
            key = self._dedup_key(det)
            if key is None or key in picked:
                continue
            picked[key] = (det, xy)

        entries = list(picked.values())
        if self.dedup_dist <= 0.0 or len(entries) <= 1:
            return entries

        def _prio(d):
            return (not bool(d.get("is_held")), bool(d.get("emergency")),
                    bool(d.get("unauthorized")), bool(d.get("helmet_alert")),
                    float(d.get("conf", 0.0)))

        entries.sort(key=lambda e: _prio(e[0]), reverse=True)
        kept = []
        d2 = self.dedup_dist ** 2
        for det, xy in entries:
            if any((xy[0] - kx) ** 2 + (xy[1] - ky) ** 2 <= d2 for _, (kx, ky) in kept):
                continue
            kept.append((det, xy))
        return kept

    @staticmethod
    def _person_state(det):
        """대시보드/RViz 와 동일한 색상 우선순위."""
        if det.get("emergency", False):
            return "EMERGENCY"
        if det.get("unauthorized", False):
            return "INTRUDER"
        if det.get("helmet_alert", False):
            return "HELMET_ALERT"
        if det.get("access_granted", False):
            return "GRANTED"
        if det.get("lying_candidate", False):
            return "LYING"
        if det.get("helmet_suspicious", False) or det.get("helmet_status") == "SUSPICIOUS":
            return "SUSPICIOUS"
        return "NORMAL"

    def publish_persons_json(self, all_detections):
        """
        웹 대시보드가 지도에 사람을 그릴 수 있도록 좌표+상태를 JSON 으로.
        persons_json_fps 로 발행 상한. 상태가 바뀌면 즉시 발행(rate limit 무시).
        """
        now = time.time()
        persons = []

        for det, xy in self._dedup_persons(all_detections):
            persons.append({
                "id": det.get("global_id") or det.get("state_key"),
                "camera": det.get("camera"),
                "x": round(float(xy[0]), 3),
                "y": round(float(xy[1]), 3),
                "state": self._person_state(det),
                "posture": det.get("posture"),
                "helmet": det.get("helmet_status"),
                "entry_state": det.get("entry_state"),
                "emergency": bool(det.get("emergency", False)),
                "unauthorized": bool(det.get("unauthorized", False)),
                "helmet_alert": bool(det.get("helmet_alert", False)),
                "helmet_suppressed": bool(det.get("helmet_suppressed", False)),
                "access_granted": bool(det.get("access_granted", False)),
                "lying": bool(det.get("lying_candidate", False)),
                "held": bool(det.get("is_held", False)),
                "fall_elapsed": round(float(det.get("fall_elapsed", 0.0)), 1),
                "entry_progress": round(float(det.get("entry_progress", 0.0)), 1),
            })

        counts = {
            "persons": len(persons),
            "emergency": sum(1 for p in persons if p["emergency"]),
            "intruder": sum(1 for p in persons if p["unauthorized"]),
            "helmet_alert": sum(1 for p in persons if p["helmet_alert"]),
            "granted": sum(1 for p in persons if p["access_granted"]),
            "lying": sum(1 for p in persons if p["lying"]),
        }

        sig = tuple(sorted((p["id"], p["state"]) for p in persons))
        changed = sig != getattr(self, "_persons_sig", None)

        if (not changed) and (now - self.persons_json_last) < self.persons_json_period:
            return

        self._persons_sig = sig
        self.persons_json_last = now

        m = String()
        m.data = json.dumps({"t": now, "counts": counts, "persons": persons},
                            ensure_ascii=False)
        self.pub_persons_json.publish(m)

    # ------------------------------------------------------------------
    # 카메라 오버레이 스트림
    # ------------------------------------------------------------------

    def publish_camera_frame(self, cam_name, bgr_frame):
        """
        오버레이가 이미 그려진 BGR 프레임을 JPEG로 인코딩해 발행.

        - 카메라별 rate limit (image_period)로 초당 발행 상한 적용.
          → cv2.imencode CPU 소비를 loop rate로부터 격리해 V4L2 버퍼
             오버플로우(카메라 스톨)를 방지.
        - 이 상한은 loop rate와 무관하므로 loop가 30fps로 돌아도
          영상은 15fps로 나감. 대역폭·subscriber 처리 부하 조절.
        """
        pub = self.image_pubs.get(cam_name)
        if pub is None or bgr_frame is None:
            return

        # 마지막 발행 후 image_period 미달이면 이번엔 skip
        now = time.time()
        last = self.image_last_pub.get(cam_name, 0.0)
        if (now - last) < self.image_period:
            return

        # JPEG 인코딩 — 720p 기준 5~15ms, 30~80KB
        ok, buf = cv2.imencode(
            ".jpg",
            bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self._stamp_now()
        msg.header.frame_id = str(cam_name)
        msg.format = "jpeg"  # image_transport가 이 값으로 디코딩 방식 판단
        msg.data = buf.tobytes()

        pub.publish(msg)
        self.image_last_pub[cam_name] = now

    # ------------------------------------------------------------------
    # 생명주기
    # ------------------------------------------------------------------

    def spin_some(self):
        """
        루프 안에서 매 프레임 호출. rclpy 내부 워커에 CPU를 잠깐 넘김.
        발행만 하고 콜백(subscriber)이 없으므로 timeout=0.0으로 non-block.
        """
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            pass

    def shutdown(self):
        """
        메인 루프 종료 시 호출. 진행 중이던 상태에 대해 CLEAR을 한 번
        뿌려서 subscriber(PC4 대시보드)가 stale 상태로 남지 않게 함.
        """
        try:
            if self.prev_emergency:
                self._publish_string(self.pub_emg_state, self.emergency_state_off)
            if self.prev_helmet_alert:
                self._publish_string(self.pub_hlm_state, self.helmet_state_off)
            if self.prev_unauthorized:
                self._publish_string(self.pub_unauth_state, self.unauthorized_state_off)
        except Exception:
            pass

        try:
            self.node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
