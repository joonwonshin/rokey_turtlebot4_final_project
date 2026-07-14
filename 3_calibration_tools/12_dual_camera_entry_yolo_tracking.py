import argparse
import importlib.util
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Load V09 utility module
# ------------------------------------------------------------
# 기존 09_safety_dashboard_z.py에 있는 유틸 함수들을 재사용한다.
# - map loading
# - homography loading
# - pose parsing
# - helmet 판단
# - z height 추정
# - drawing helper
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
V09_PATH = SCRIPT_DIR / "09_safety_dashboard_z.py"

if not V09_PATH.exists():
    raise FileNotFoundError(f"09_safety_dashboard_z.py not found: {V09_PATH}")

spec09 = importlib.util.spec_from_file_location("safety_v09", str(V09_PATH))
v09 = importlib.util.module_from_spec(spec09)
spec09.loader.exec_module(v09)


# ============================================================
# COCO pose keypoint indices
# ------------------------------------------------------------
# YOLO pose는 COCO 17 keypoints 기준이다.
# v09에 상수가 있으면 그걸 쓰고, 없으면 기본값을 쓴다.
# ============================================================

NOSE = getattr(v09, "NOSE", 0)
LEFT_EYE = getattr(v09, "LEFT_EYE", 1)
RIGHT_EYE = getattr(v09, "RIGHT_EYE", 2)
LEFT_EAR = getattr(v09, "LEFT_EAR", 3)
RIGHT_EAR = getattr(v09, "RIGHT_EAR", 4)
LEFT_SHOULDER = getattr(v09, "LEFT_SHOULDER", 5)
RIGHT_SHOULDER = getattr(v09, "RIGHT_SHOULDER", 6)
LEFT_ELBOW = getattr(v09, "LEFT_ELBOW", 7)
RIGHT_ELBOW = getattr(v09, "RIGHT_ELBOW", 8)
LEFT_WRIST = getattr(v09, "LEFT_WRIST", 9)
RIGHT_WRIST = getattr(v09, "RIGHT_WRIST", 10)
LEFT_HIP = getattr(v09, "LEFT_HIP", 11)
RIGHT_HIP = getattr(v09, "RIGHT_HIP", 12)
LEFT_KNEE = getattr(v09, "LEFT_KNEE", 13)
RIGHT_KNEE = getattr(v09, "RIGHT_KNEE", 14)
LEFT_ANKLE = getattr(v09, "LEFT_ANKLE", 15)
RIGHT_ANKLE = getattr(v09, "RIGHT_ANKLE", 16)


# ============================================================
# Basic helpers
# ============================================================

def get_class_name(names, cls_id):
    """Ultralytics class id를 class name으로 변환한다."""
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), str(cls_id)))
        return str(names[int(cls_id)])
    except Exception:
        return str(cls_id)


def box_area(xyxy):
    """xyxy bbox 면적 계산."""
    x1, y1, x2, y2 = xyxy
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def map_image_pixel_to_meter_local(px, py, map_info):
    """
    map image pixel 좌표를 map meter 좌표로 변환한다.
    v09에 동일 함수가 있으면 우선 사용한다.
    """
    if hasattr(v09, "map_image_pixel_to_meter"):
        return v09.map_image_pixel_to_meter(px, py, map_info)

    resolution = float(map_info["resolution"])
    origin = map_info["origin"]

    ox = float(origin[0])
    oy = float(origin[1])
    h = int(map_info["height"])

    mx = ox + float(px) * resolution
    my = oy + float(h - py) * resolution

    return mx, my


def get_visible_point_safe(pose_item, idx, conf_thres=0.25):
    """
    v09.get_visible_point를 안전하게 감싼 함수.
    """
    if pose_item is None:
        return None

    if hasattr(v09, "get_visible_point"):
        try:
            return v09.get_visible_point(pose_item, idx, conf_thres)
        except Exception:
            return None

    return None


def point_in_polygon(point, polygon):
    """
    point가 polygon 내부에 있는지 판단한다.
    point: [map_x, map_y]
    polygon: [[x1,y1], [x2,y2], ...]
    """
    if point is None or polygon is None or len(polygon) < 3:
        return False

    x, y = float(point[0]), float(point[1])

    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )

        if intersect:
            inside = not inside

        j = i

    return inside


def blink_on(rate_hz=2.5):
    """
    map emergency 표시 전용 점멸 함수.
    status panel에는 이 점멸을 쓰지 않는다.
    """
    return int(time.time() * rate_hz * 2) % 2 == 0


def emergency_map_color():
    """
    map 위 emergency 점멸용 색상.
    켜질 때 빨강, 꺼질 때 어두운 색.
    """
    if blink_on():
        return (0, 0, 255)
    return (30, 30, 30)


# ============================================================
# YOLO tracking result extraction
# ------------------------------------------------------------
# best.pt.track() 결과에서 person / helmet을 분리한다.
#
# person:
# - YOLO track id를 가진다.
#
# helmet:
# - detection 결과만 사용한다.
# - helmet 자체 track id는 중요하지 않다.
# - 중요한 건 "해당 person의 머리 근처에 helmet box가 있는가"이다.
# ============================================================

def extract_yolo_track_boxes(
    result,
    names,
    image_area,
    person_conf_thres=0.45,
    helmet_conf_thres=0.25,
    person_min_area_ratio=0.002,
    person_min_height=35,
):
    persons = []
    helmets = []

    if result.boxes is None:
        return persons, helmets

    boxes = result.boxes

    if boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
        return persons, helmets

    xyxys = boxes.xyxy.detach().cpu().numpy().astype(float)
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy().astype(float)

    if boxes.id is not None:
        ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxys)

    for i in range(len(xyxys)):
        cls_id = int(clss[i])
        conf = float(confs[i])
        xyxy = xyxys[i].tolist()
        track_id = ids[i]

        cls_name = get_class_name(names, cls_id).lower()

        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        area_ratio = box_area(xyxy) / max(image_area, 1)

        if cls_name == "person":
            if conf < person_conf_thres:
                continue

            if area_ratio < person_min_area_ratio:
                continue

            if max(bh, bw) < person_min_height:
                continue

            persons.append(
                {
                    "box": xyxy,
                    "conf": conf,
                    "track_id": track_id,
                    "source": "yolo_track",
                }
            )

        elif cls_name == "helmet":
            if conf < helmet_conf_thres:
                continue

            helmets.append(
                {
                    "box": xyxy,
                    "conf": conf,
                    "track_id": track_id,
                }
            )

    return persons, helmets


# ============================================================
# Entry ROI functions
# ============================================================

def load_entry_roi(path):
    """
    calibration/entry_roi.json에서 입구 ROI polygon을 불러온다.
    ROI는 map meter 좌표 기준으로 저장된다.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"entry ROI json not found: {path}\n"
            "먼저 --mode set_roi 로 입구 ROI를 저장해야 함."
        )

    data = json.loads(path.read_text())
    pts = data.get("points_map", [])

    if len(pts) < 3:
        raise RuntimeError(f"entry ROI needs at least 3 points: {path}")

    return [[float(x), float(y)] for x, y in pts], data


def draw_entry_roi_on_map(map_img, map_info, roi_points_map):
    """
    map panel 위에 entry ROI 표시.
    점/글자 크기를 작게 해서 map이 가려지지 않도록 한다.
    """
    out = map_img.copy()

    if roi_points_map is None or len(roi_points_map) < 3:
        return out

    pts = []

    for mx, my in roi_points_map:
        px, py = v09.map_meter_to_image_pixel(mx, my, map_info)
        pts.append([px, py])

    pts_np = np.array(pts, dtype=np.int32)

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts_np], (0, 180, 255))
    cv2.addWeighted(overlay, 0.16, out, 0.84, 0, out)

    cv2.polylines(out, [pts_np], True, (0, 180, 255), 2)

    for i, (px, py) in enumerate(pts):
        cv2.circle(out, (px, py), 2, (0, 180, 255), -1)
        v09.draw_text(out, f"E{i}", (px + 4, py - 4), (0, 180, 255), 0.30, 1)

    return out


def project_map_points_to_camera(H_inv, map_info, roi_points_map, homography_output):
    """
    map ROI를 camera image 좌표로 역투영한다.

    기존 homography H:
    camera pixel -> map coordinate

    여기서는 H_inv:
    map coordinate -> camera pixel
    """
    if H_inv is None or roi_points_map is None or len(roi_points_map) < 3:
        return None

    pts = []

    for mx, my in roi_points_map:
        if homography_output == "map_meters":
            src = np.array([float(mx), float(my), 1.0], dtype=np.float64)
        else:
            map_px, map_py = v09.map_meter_to_image_pixel(mx, my, map_info)
            src = np.array([float(map_px), float(map_py), 1.0], dtype=np.float64)

        uvw = H_inv @ src

        if abs(float(uvw[2])) < 1e-9:
            continue

        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])

        if np.isfinite(u) and np.isfinite(v):
            pts.append([int(round(u)), int(round(v))])

    if len(pts) < 3:
        return None

    return np.array(pts, dtype=np.int32)


def draw_entry_roi_on_camera(frame, H_inv, map_info, roi_points_map, homography_output, cam_name):
    """
    webcam 화면에 entry ROI 표시.
    이 함수는 cam0에서만 호출된다.
    cam1에서는 호출하지 않는다.
    """
    out = frame.copy()

    pts_np = project_map_points_to_camera(
        H_inv,
        map_info,
        roi_points_map,
        homography_output,
    )

    if pts_np is None:
        return out

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts_np], (0, 180, 255))
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    cv2.polylines(out, [pts_np], True, (0, 180, 255), 2)

    x0, y0 = pts_np[0]
    v09.draw_text(
        out,
        f"{cam_name} ENTRY ROI",
        (int(x0) + 6, max(20, int(y0) - 6)),
        (0, 180, 255),
        0.58,
        2,
    )

    return out


def set_roi_mode(args):
    """
    ROI 설정 모드.

    map 이미지 위에서 입구 영역을 polygon으로 클릭해서 저장한다.

    조작:
    - left click: 점 추가
    - u: undo
    - r: reset
    - s: save
    - q: quit
    """
    map_info = v09.load_map_from_yaml(args.map_yaml)
    img = map_info["img"].copy()

    state = {
        "points_px": [],
        "points_map": [],
    }

    window = "SET ENTRY ROI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < map_info["width"] and 0 <= y < map_info["height"]):
                print("[WARN] click outside map")
                return

            mx, my = map_image_pixel_to_meter_local(x, y, map_info)

            state["points_px"].append([int(x), int(y)])
            state["points_map"].append([float(mx), float(my)])

            print(
                f"[ROI POINT {len(state['points_px'])}] "
                f"px=({x},{y}) map=({mx:.3f},{my:.3f})"
            )

    cv2.setMouseCallback(window, mouse_cb)

    print("======================================")
    print("SET ENTRY ROI MODE")
    print("--------------------------------------")
    print("입구 영역을 map 위에서 polygon으로 클릭")
    print("left click : 점 추가")
    print("u          : undo")
    print("r          : reset")
    print("s          : save")
    print("q          : quit")
    print("======================================")

    while True:
        show = img.copy()

        if len(state["points_px"]) >= 1:
            pts = np.array(state["points_px"], dtype=np.int32)

            for i, (px, py) in enumerate(state["points_px"]):
                cv2.circle(show, (px, py), 4, (0, 180, 255), -1)
                v09.draw_text(show, f"{i}", (px + 6, py - 6), (0, 180, 255), 0.48, 1)

            if len(pts) >= 2:
                cv2.polylines(show, [pts], False, (0, 180, 255), 2)

            if len(pts) >= 3:
                overlay = show.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 255))
                cv2.addWeighted(overlay, 0.20, show, 0.80, 0, show)
                cv2.polylines(show, [pts], True, (0, 180, 255), 2)

        overlay = show.copy()
        cv2.rectangle(overlay, (10, 10), (760, 88), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, show, 0.50, 0, show)

        v09.draw_text(
            show,
            "SET ENTRY ROI | click entrance polygon on map",
            (24, 40),
            (255, 255, 255),
            0.62,
            2,
        )
        v09.draw_text(
            show,
            f"points: {len(state['points_px'])} | s:save u:undo r:reset q:quit",
            (24, 70),
            (255, 255, 255),
            0.52,
            1,
        )

        cv2.imshow(window, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("u"):
            if state["points_px"]:
                removed_px = state["points_px"].pop()
                removed_map = state["points_map"].pop()
                print("[UNDO]", removed_px, removed_map)

        if key == ord("r"):
            state["points_px"] = []
            state["points_map"] = []
            print("[RESET] ROI points cleared")

        if key == ord("s"):
            if len(state["points_map"]) < 3:
                print("[WARN] Need at least 3 points")
                continue

            out_path = Path(args.entry_roi_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "map_yaml": args.map_yaml,
                "points_px": state["points_px"],
                "points_map": state["points_map"],
            }

            out_path.write_text(json.dumps(data, indent=2))

            print("")
            print("======================================")
            print("[SAVED] entry ROI")
            print(f"path: {out_path}")
            print("points_map:")
            for p in state["points_map"]:
                print(f"  - {p[0]:.3f}, {p[1]:.3f}")
            print("======================================")

    cv2.destroyAllWindows()


# ============================================================
# Pose logic
# ============================================================

def judge_hand_raise(pose_item, conf_thres=0.25, margin_px=20):
    """
    한 손 들기 판정.

    이미지 좌표계에서는 위로 갈수록 y가 작다.
    따라서 wrist_y < shoulder_y - margin 이면 손을 들었다고 판단한다.
    """
    if pose_item is None:
        return False, "NONE", {}

    lw = get_visible_point_safe(pose_item, LEFT_WRIST, conf_thres)
    rw = get_visible_point_safe(pose_item, RIGHT_WRIST, conf_thres)
    ls = get_visible_point_safe(pose_item, LEFT_SHOULDER, conf_thres)
    rs = get_visible_point_safe(pose_item, RIGHT_SHOULDER, conf_thres)

    left_raise = False
    right_raise = False

    if lw is not None and ls is not None:
        left_raise = float(lw[1]) < float(ls[1]) - margin_px

    if rw is not None and rs is not None:
        right_raise = float(rw[1]) < float(rs[1]) - margin_px

    if left_raise and right_raise:
        side = "BOTH"
    elif left_raise:
        side = "LEFT"
    elif right_raise:
        side = "RIGHT"
    else:
        side = "NONE"

    return left_raise or right_raise, side, {
        "left_raise": left_raise,
        "right_raise": right_raise,
    }


def judge_posture_z_priority(
    person_box,
    pose_item,
    head_z,
    z_residual,
    image_h,
    user_height_m=1.80,
    pose_conf=0.25,
    lying_height_thres=0.78,
    very_low_height_thres=0.55,
    residual_thres=160.0,
):
    """
    Z-priority lying logic.

    기존 v09 lying logic이 너무 보수적인 느낌이 있어서,
    이번 버전에서는 z calibration 결과를 더 강하게 반영한다.
    """
    old_lying, old_posture, debug = v09.judge_posture_v09(
        person_box,
        pose_item,
        head_z=head_z,
        z_residual=z_residual,
        image_h=image_h,
        user_height_m=user_height_m,
        pose_conf=pose_conf,
        lying_height_thres=lying_height_thres,
        very_low_height_thres=very_low_height_thres,
        residual_thres=residual_thres,
    )

    x1, y1, x2, y2 = person_box
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph

    try:
        torso_horizontal, _, _ = v09.get_torso_horizontal(pose_item, pose_conf)
    except Exception:
        torso_horizontal = False

    try:
        kpt_ratio, visible_kpts = v09.get_keypoint_spread_ratio(pose_item, pose_conf)
    except Exception:
        kpt_ratio, visible_kpts = 0.0, 0

    valid_z = (
        head_z is not None
        and np.isfinite(head_z)
        and -0.20 <= head_z <= user_height_m + 1.2
    )

    reasons = list(debug.get("reasons", []))
    posture = old_posture
    lying = old_lying

    if valid_z:
        if head_z < very_low_height_thres:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_very_low")

        elif head_z < lying_height_thres:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_low")

        elif head_z < user_height_m * 0.55 and (
            aspect > 1.10 or torso_horizontal or kpt_ratio > 1.15
        ):
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_mid_low_with_shape")

        elif old_lying:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("v09_old_lying_kept")

        else:
            if head_z < user_height_m * 0.65:
                posture = "LOW_POSTURE"
                lying = False
            else:
                posture = "NORMAL"
                lying = False

    else:
        lying = old_lying
        posture = old_posture
        reasons.append("z_unavailable_fallback_v09")

    debug["lying_mode"] = "z_priority_v12"
    debug["reasons"] = reasons
    debug["head_z"] = None if head_z is None else float(head_z)
    debug["z_residual"] = None if z_residual is None else float(z_residual)
    debug["aspect"] = float(aspect)
    debug["torso_horizontal"] = bool(torso_horizontal)
    debug["kpt_ratio"] = float(kpt_ratio)
    debug["visible_kpts"] = int(visible_kpts)

    return lying, posture, debug


# ============================================================
# Worker state store
# ------------------------------------------------------------
# YOLO track id를 기준으로 상태를 누적한다.
#
# 화면 표시용 id:
# - YOLO 원본 id는 Y137처럼 커질 수 있다.
# - 화면에는 현재 보이는 사람 기준 #0~#3으로 압축 표시한다.
#
# entry_enabled:
# - cam0: True  -> 입구 검사 수행
# - cam1: False -> MONITOR_ONLY, 입구 검사 없음
# ============================================================

class WorkerStateStore:
    def __init__(
        self,
        camera_name,
        entry_roi_points,
        required_check_sec=3.0,
        emergency_sec=5.0,
        max_age_sec=3.0,
        history_len=6,
        max_workers=4,
        entry_enabled=True,
    ):
        self.camera_name = camera_name
        self.entry_roi_points = entry_roi_points
        self.required_check_sec = required_check_sec
        self.emergency_sec = emergency_sec
        self.max_age_sec = max_age_sec
        self.history_len = history_len
        self.max_workers = max_workers
        self.entry_enabled = entry_enabled
        self.states = {}

    def _new_state(self, key, now):
        return {
            "key": key,
            "first_seen": now,
            "last_seen": now,

            "helmet_history": deque(maxlen=self.history_len),
            "posture_history": deque(maxlen=self.history_len),
            "lying_history": deque(maxlen=self.history_len),
            "hand_history": deque(maxlen=self.history_len),

            "fall_start": None,
            "fall_elapsed": 0.0,
            "emergency": False,

            "entry_seen": False,
            "entry_check_start": None,
            "entry_progress": 0.0,
            "entry_state": "OUTSIDE",
            "access_granted": False,
            "access_granted_time": None,
            "unauthorized": False,
        }

    def _majority(self, hist, default):
        if not hist:
            return default

        counts = {}

        for x in hist:
            counts[x] = counts.get(x, 0) + 1

        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _smooth_bool(self, hist):
        if not hist:
            return False

        return sum(1 for x in hist if x) >= max(2, len(hist) // 2)

    def update_one(self, det, now):
        yolo_id = det.get("yolo_track_id")

        if yolo_id is None:
            det["state_key"] = f"{self.camera_name}:NO_ID"
            det["entry_state"] = "NO_ID"
            det["entry_progress"] = 0.0
            det["entry_progress_ratio"] = 0.0
            det["access_granted"] = False
            det["unauthorized"] = False
            det["fall_elapsed"] = 0.0
            det["emergency"] = False
            return det

        key = f"{self.camera_name}:{int(yolo_id)}"

        if key not in self.states:
            self.states[key] = self._new_state(key, now)

        st = self.states[key]
        st["last_seen"] = now

        raw_helmet = det.get("helmet_status", "UNKNOWN")
        raw_posture = det.get("posture", "NORMAL")
        raw_lying = bool(det.get("lying_candidate", False))
        raw_hand = bool(det.get("hand_raised", False))

        st["helmet_history"].append(raw_helmet)
        st["posture_history"].append(raw_posture)
        st["lying_history"].append(raw_lying)
        st["hand_history"].append(raw_hand)

        smooth_helmet = self._majority(st["helmet_history"], "UNKNOWN")
        smooth_posture = self._majority(st["posture_history"], "NORMAL")
        smooth_lying = self._smooth_bool(st["lying_history"])
        smooth_hand = self._smooth_bool(st["hand_history"])

        det["helmet_status_raw"] = raw_helmet
        det["posture_raw"] = raw_posture
        det["hand_raised_raw"] = raw_hand

        det["helmet_status"] = smooth_helmet
        det["posture"] = smooth_posture
        det["lying_candidate"] = smooth_lying
        det["hand_raised"] = smooth_hand

        # ----------------------------------------------------
        # Lying / emergency
        # ----------------------------------------------------
        # lying이 emergency_sec 이상 유지되면 emergency.
        # 사람이 일어나서 lying이 False가 되면 emergency도 종료된다.
        if smooth_lying:
            if st["fall_start"] is None:
                st["fall_start"] = now

            st["fall_elapsed"] = now - st["fall_start"]

            if st["fall_elapsed"] >= self.emergency_sec:
                st["emergency"] = True
        else:
            st["fall_start"] = None
            st["fall_elapsed"] = 0.0
            st["emergency"] = False

        # ----------------------------------------------------
        # cam1은 입구 검사를 하지 않는다.
        # ----------------------------------------------------
        if not self.entry_enabled:
            det["in_entry_roi"] = False
            det["entry_state"] = "MONITOR_ONLY"
            det["entry_progress"] = 0.0
            det["entry_progress_ratio"] = 0.0
            det["access_granted"] = False
            det["unauthorized"] = False

            det["state_key"] = key
            det["fall_elapsed"] = st["fall_elapsed"]
            det["emergency"] = st["emergency"]

            return det

        # ----------------------------------------------------
        # cam0 entry state machine
        # ----------------------------------------------------
        map_xy = det.get("map_xy")
        in_entry = point_in_polygon(map_xy, self.entry_roi_points)
        det["in_entry_roi"] = in_entry

        check_condition = in_entry and smooth_hand and smooth_helmet == "HELMET_ON"

        if in_entry:
            st["entry_seen"] = True

            if st["access_granted"]:
                st["entry_state"] = "ACCESS_GRANTED"

            elif check_condition:
                if st["entry_check_start"] is None:
                    st["entry_check_start"] = now

                st["entry_progress"] = now - st["entry_check_start"]

                if st["entry_progress"] >= self.required_check_sec:
                    st["access_granted"] = True
                    st["access_granted_time"] = now
                    st["unauthorized"] = False
                    st["entry_state"] = "ACCESS_GRANTED"
                else:
                    st["entry_state"] = "CHECKING"

            else:
                st["entry_check_start"] = None
                st["entry_progress"] = 0.0

                if smooth_helmet != "HELMET_ON" and smooth_hand:
                    st["entry_state"] = "CHECK_FAIL_NO_HELMET"
                elif smooth_helmet == "HELMET_ON" and not smooth_hand:
                    st["entry_state"] = "WAIT_HAND_RAISE"
                else:
                    st["entry_state"] = "WAIT_CHECK"

        else:
            st["entry_check_start"] = None
            st["entry_progress"] = 0.0

            if st["access_granted"]:
                st["entry_state"] = "ENTERED"
            elif st["entry_seen"]:
                st["unauthorized"] = True
                st["entry_state"] = "UNAUTHORIZED_ENTRY"
            else:
                st["entry_state"] = "OUTSIDE"

        det["state_key"] = key
        det["fall_elapsed"] = st["fall_elapsed"]
        det["emergency"] = st["emergency"]

        det["entry_state"] = st["entry_state"]
        det["entry_progress"] = min(st["entry_progress"], self.required_check_sec)
        det["entry_progress_ratio"] = min(
            1.0,
            st["entry_progress"] / max(0.1, self.required_check_sec),
        )
        det["access_granted"] = bool(st["access_granted"])
        det["unauthorized"] = bool(st["unauthorized"])

        return det

    def assign_compact_display_ids(self, detections):
        """
        현재 카메라 안에 보이는 사람에게 #0~#3 display id를 부여한다.
        """
        valid_dets = []

        for det in detections:
            key = det.get("state_key")

            if key is None or key.endswith("NO_ID"):
                det["display_id"] = None
                continue

            if key in self.states:
                valid_dets.append(det)

        valid_dets.sort(
            key=lambda d: self.states[d["state_key"]]["first_seen"]
        )

        for i, det in enumerate(valid_dets):
            if i < self.max_workers:
                det["display_id"] = i
            else:
                det["display_id"] = None

        return detections

    def cleanup(self, now):
        dead = []

        for key, st in self.states.items():
            if now - st["last_seen"] > self.max_age_sec:
                dead.append(key)

        for key in dead:
            del self.states[key]

    def reset(self):
        self.states = {}


# ============================================================
# Camera processor
# ============================================================

class CameraEntryTracker:
    def __init__(
        self,
        cam_id,
        cam_name,
        det_model_path,
        homography_path,
        z_calib_path,
        map_info,
        homography_output,
        worker_store,
        tracker_yaml,
        width=1280,
        height=720,
        fps=15,
    ):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.map_info = map_info
        self.homography_output = homography_output
        self.worker_store = worker_store
        self.tracker_yaml = tracker_yaml

        # 카메라별 YOLO 객체를 따로 만든다.
        # cam0/cam1 tracker state가 섞이면 안 되기 때문.
        self.det_model = YOLO(str(det_model_path))
        self.det_names = self.det_model.names

        self.H, self.H_key = v09.load_homography(homography_path)

        try:
            self.H_inv = np.linalg.inv(self.H)
        except Exception as e:
            print(f"[WARN] {cam_name} inverse homography failed: {e}")
            self.H_inv = None

        self.P = v09.load_z_calibration(z_calib_path)

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open {cam_name}, camera id={cam_id}")

        self.frame_idx = 0
        self.prev_time = time.time()
        self.fps = 0.0

        self.last_preview = None
        self.last_tracked = []
        self.last_metrics = {}

        print(f"[INFO] {cam_name}")
        print(f"  camera id  : {cam_id}")
        print(f"  H          : {homography_path}")
        print(f"  H key      : {self.H_key}")
        print(f"  z calib    : {z_calib_path}")
        print(f"  tracker    : {tracker_yaml}")
        print(f"  entry      : {self.worker_store.entry_enabled}")

    def release(self):
        self.cap.release()

    def process(self, pose_model, args, now):
        ret, frame = self.cap.read()

        if not ret:
            print(f"[WARN] {self.cam_name} frame read failed")

            if self.last_preview is not None:
                return self.last_preview, self.last_tracked, self.last_metrics

            empty = np.zeros((720, 1280, 3), dtype=np.uint8)
            v09.draw_text(
                empty,
                f"{self.cam_name} frame read failed",
                (40, 80),
                (0, 0, 255),
                1.0,
                2,
            )
            return empty, [], {}

        self.frame_idx += 1

        image_h, image_w = frame.shape[:2]
        image_area = image_h * image_w

        run_inference = (
            self.frame_idx % args.process_every == 0
        ) or self.last_preview is None

        if not run_inference:
            return self.last_preview, self.last_tracked, self.last_metrics

        # ----------------------------------------------------
        # 1. YOLO tracking
        # ----------------------------------------------------
        track_result = self.det_model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=args.device,
            persist=True,
            tracker=self.tracker_yaml,
            verbose=False,
        )[0]

        persons, helmets = extract_yolo_track_boxes(
            track_result,
            self.det_names,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            helmet_conf_thres=args.helmet_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        # ----------------------------------------------------
        # 2. YOLO pose
        # ----------------------------------------------------
        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_items = v09.extract_pose_items(pose_result)

        # ----------------------------------------------------
        # 3. Base frame + ROI
        # ----------------------------------------------------
        preview = frame.copy()

        # Entry ROI는 cam0에만 표시한다.
        if self.worker_store.entry_enabled:
            preview = draw_entry_roi_on_camera(
                preview,
                self.H_inv,
                self.map_info,
                self.worker_store.entry_roi_points,
                self.homography_output,
                self.cam_name,
            )

        for helmet in helmets:
            v09.draw_box(
                preview,
                helmet["box"],
                (0, 255, 255),
                f"helmet {helmet['conf']:.2f}",
                1,
            )

        tracked_outputs = []

        # ----------------------------------------------------
        # 4. Per-person logic
        # ----------------------------------------------------
        for person in persons:
            pbox = person["box"]
            yolo_track_id = person.get("track_id")

            pose_item, pose_iou = v09.match_pose_to_person(pbox, pose_items)

            nearest_helmet, _ = v09.find_nearest_related_helmet(
                pbox,
                helmets,
                image_w,
                image_h,
            )
            nearest_helmet_box = nearest_helmet["box"] if nearest_helmet is not None else None

            head_px, head_src = v09.get_head_point(
                pose_item,
                person_box=pbox,
                helmet_box=nearest_helmet_box,
                conf_thres=args.pose_conf,
            )

            foot_px, foot_src = v09.get_foot_point(
                pose_item,
                pbox,
                conf_thres=args.pose_conf,
            )

            foot_map = v09.camera_pixel_to_map_meter(
                self.H,
                foot_px,
                self.map_info,
                self.homography_output,
            )

            head_z = None
            z_residual = None
            projected_uv = None

            if head_px is not None and foot_map is not None:
                head_z, z_residual, projected_uv = v09.estimate_z_on_vertical_line(
                    self.P,
                    foot_map[0],
                    foot_map[1],
                    head_px,
                )

            helmet_status, head_radius, related_helmets = v09.judge_helmet_status(
                pbox,
                helmets,
                head_px,
                head_src,
                image_w,
                image_h,
                head_radius_scale=args.head_radius_scale,
                helmet_expand_ratio=args.helmet_expand,
            )

            lying_candidate, posture, debug = judge_posture_z_priority(
                pbox,
                pose_item,
                head_z=head_z,
                z_residual=z_residual,
                image_h=image_h,
                user_height_m=args.user_height_m,
                pose_conf=args.pose_conf,
                lying_height_thres=args.lying_height_thres,
                very_low_height_thres=args.very_low_height_thres,
                residual_thres=args.residual_thres,
            )

            hand_raised, hand_side, hand_debug = judge_hand_raise(
                pose_item,
                conf_thres=args.pose_conf,
                margin_px=args.hand_raise_margin,
            )

            det_item = {
                "camera": self.cam_name,
                "yolo_track_id": yolo_track_id,

                "image_point": foot_px,
                "map_xy": foot_map,

                "helmet_status": helmet_status,
                "posture": posture,
                "lying_candidate": lying_candidate,

                "hand_raised": hand_raised,
                "hand_side": hand_side,
                "hand_debug": hand_debug,

                "debug": debug,
                "person_box": pbox,
                "source": person["source"],
                "conf": person["conf"],

                "head_px": head_px,
                "head_src": head_src,
                "head_radius": head_radius,
                "foot_px": foot_px,
                "foot_src": foot_src,
                "projected_uv": projected_uv,
            }

            det_item = self.worker_store.update_one(det_item, now)
            tracked_outputs.append(det_item)

        tracked_outputs = self.worker_store.assign_compact_display_ids(tracked_outputs)
        self.worker_store.cleanup(now)

        # ----------------------------------------------------
        # 5. Draw people
        # ----------------------------------------------------
        for det in tracked_outputs:
            draw_person_overlay(preview, det, pose_items, args, image_h)

        # ----------------------------------------------------
        # 6. Header
        # ----------------------------------------------------
        overlay = preview.copy()
        cv2.rectangle(overlay, (8, 8), (760, 54), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, preview, 0.50, 0, preview)

        mode_text = "ENTRY" if self.worker_store.entry_enabled else "MONITOR_ONLY"

        v09.draw_text(
            preview,
            f"{self.cam_name} | {mode_text} | YOLO TRACK | persons:{len(tracked_outputs)} helmets:{len(helmets)} FPS:{self.fps:.1f}",
            (18, 40),
            (255, 255, 255),
            0.62,
            2,
        )

        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            self.fps = self.fps * 0.90 + (1.0 / dt) * 0.10

        metrics = calc_metrics(tracked_outputs, helmets)
        metrics["fps"] = self.fps
        metrics["cam_name"] = self.cam_name

        self.last_preview = preview
        self.last_tracked = tracked_outputs
        self.last_metrics = metrics

        return preview, tracked_outputs, metrics


# ============================================================
# Drawing helpers
# ============================================================

def state_color(det):
    """
    상태별 색상.
    주의: 여기서는 점멸하지 않는다.
    emergency 점멸은 map에서만 한다.
    """
    if det.get("emergency", False):
        return (0, 0, 255)

    if det.get("unauthorized", False):
        return (0, 0, 255)

    if det.get("entry_state") in ["CHECK_FAIL_NO_HELMET", "WAIT_HAND_RAISE", "WAIT_CHECK"]:
        return (0, 165, 255)

    if det.get("entry_state") == "CHECKING":
        return (0, 255, 255)

    if det.get("access_granted", False):
        return (0, 255, 0)

    return v09.color_for_state(
        det.get("helmet_status", "UNKNOWN"),
        det.get("posture", "NORMAL"),
        det.get("emergency", False),
    )


def display_id_text(det):
    did = det.get("display_id")
    if did is None:
        return "X"
    return str(int(did))


def draw_progress_bar(img, x1, y1, x2, y2, ratio, color, text):
    """
    bbox 위에 3초 검사 progress bar 표시.
    """
    ratio = max(0.0, min(1.0, float(ratio)))

    bar_w = max(120, min(240, int(x2 - x1)))
    bar_h = 16

    bx1 = int(x1)
    by1 = int(max(0, y1 - 34))
    bx2 = bx1 + bar_w
    by2 = by1 + bar_h

    cv2.rectangle(img, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

    fill_w = int(bar_w * ratio)
    cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), color, -1)

    v09.draw_text(img, text, (bx1, max(16, by1 - 5)), color, 0.52, 1)


def draw_person_overlay(preview, det, pose_items, args, image_h):
    """
    사람 bbox, 상태, display id, progress bar를 크게 표시한다.
    """
    pbox = det["person_box"]
    x1, y1, x2, y2 = map(int, pbox)

    color = state_color(det)

    did = display_id_text(det)
    yolo_id = det.get("yolo_track_id")
    yolo_text = "NO_ID" if yolo_id is None else f"Y{int(yolo_id)}"

    label = f"#{did} ({yolo_text}) {det.get('helmet_status')} {det.get('posture')}"

    if det.get("entry_state"):
        label += f" | {det.get('entry_state')}"

    if det.get("emergency", False):
        label += " | EMERGENCY"
    elif det.get("lying_candidate", False):
        label += f" | lying {det.get('fall_elapsed', 0):.1f}/{args.emergency_sec:.0f}s"

    head_z = det.get("debug", {}).get("head_z")
    if head_z is not None:
        label += f" z={head_z:.2f}"

    line_thick = 3 if det.get("emergency", False) else 2
    v09.draw_box(preview, pbox, color, label, line_thick)

    # 큰 display id 박스
    cv2.rectangle(preview, (x1, y1), (x1 + 48, y1 + 48), color, -1)
    v09.draw_text(preview, f"{did}", (x1 + 12, y1 + 37), (0, 0, 0), 1.10, 3)

    # progress / result bar
    entry_state = det.get("entry_state", "OUTSIDE")
    progress_ratio = det.get("entry_progress_ratio", 0.0)
    progress = det.get("entry_progress", 0.0)

    if entry_state == "CHECKING":
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            progress_ratio,
            (0, 255, 255),
            f"checking {progress:.1f}/{args.entry_check_sec:.0f}s",
        )

    elif det.get("access_granted", False):
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (0, 255, 0),
            "ACCESS GRANTED",
        )

    elif det.get("unauthorized", False):
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (0, 0, 255),
            "UNAUTHORIZED ENTRY",
        )

    # head circle
    head_px = det.get("head_px")
    head_radius = det.get("head_radius", 40)

    if head_px is not None:
        hp = tuple(head_px.astype(int))
        cv2.circle(preview, hp, int(head_radius), (255, 0, 255), 1)
        cv2.circle(preview, hp, 4, (255, 0, 255), -1)

    # foot point
    foot_px = det.get("foot_px")
    if foot_px is not None:
        fp = tuple(foot_px.astype(int))
        cv2.circle(preview, fp, 6, (255, 255, 0), -1)

    # z projected point
    projected_uv = det.get("projected_uv")
    if projected_uv is not None and np.all(np.isfinite(projected_uv)):
        pp = tuple(projected_uv.astype(int))
        cv2.circle(preview, pp, 5, (0, 165, 255), -1)

        if head_px is not None:
            cv2.line(preview, tuple(head_px.astype(int)), pp, (0, 165, 255), 1)

    # torso line
    pose_item_tmp, _ = v09.match_pose_to_person(pbox, pose_items)

    try:
        torso_horizontal, shoulder_mid, hip_mid = v09.get_torso_horizontal(
            pose_item_tmp,
            args.pose_conf,
        )
    except Exception:
        shoulder_mid, hip_mid = None, None

    if shoulder_mid is not None and hip_mid is not None:
        cv2.line(
            preview,
            tuple(shoulder_mid.astype(int)),
            tuple(hip_mid.astype(int)),
            (255, 0, 0),
            2,
        )

    # debug text, 더 크게
    line_y = min(image_h - 10, y2 + 24)

    v09.draw_text(
        preview,
        f"hand={det.get('hand_raised')} side={det.get('hand_side')} entry={det.get('in_entry_roi')}",
        (x1, line_y),
        color,
        0.50,
        1,
    )

    line_y += 22
    v09.draw_text(
        preview,
        f"raw H/P={det.get('helmet_status_raw')}/{det.get('posture_raw')}",
        (x1, min(image_h - 10, line_y)),
        color,
        0.44,
        1,
    )

    map_xy = det.get("map_xy")
    if map_xy is not None:
        line_y += 20
        v09.draw_text(
            preview,
            f"map=({map_xy[0]:.2f},{map_xy[1]:.2f})",
            (x1, min(image_h - 10, line_y)),
            color,
            0.44,
            1,
        )


def calc_metrics(tracked, helmets):
    return {
        "person_count": len(tracked),
        "helmet_count": len(helmets),
        "no_helmet_count": sum(1 for d in tracked if d.get("helmet_status") == "NO_HELMET"),
        "unknown_helmet_count": sum(1 for d in tracked if d.get("helmet_status") == "UNKNOWN"),
        "checking_count": sum(1 for d in tracked if d.get("entry_state") == "CHECKING"),
        "granted_count": sum(1 for d in tracked if d.get("access_granted")),
        "unauthorized_count": sum(1 for d in tracked if d.get("unauthorized")),
        "lying_count": sum(1 for d in tracked if d.get("posture") == "LYING_CANDIDATE"),
        "emergency_count": sum(1 for d in tracked if d.get("emergency")),
    }


def draw_combined_map(map_img, all_detections, map_info, entry_roi_points):
    """
    map panel.
    emergency 점멸은 여기서만 발생한다.
    """
    out = draw_entry_roi_on_map(map_img, map_info, entry_roi_points)

    for det in all_detections:
        map_xy = det.get("map_xy")
        if map_xy is None:
            continue

        mx, my = map_xy
        px, py = v09.map_meter_to_image_pixel(mx, my, map_info)

        if not (0 <= px < map_info["width"] and 0 <= py < map_info["height"]):
            continue

        emergency = bool(det.get("emergency", False))
        unauthorized = bool(det.get("unauthorized", False))

        if emergency:
            color = emergency_map_color()
            radius = 8 if blink_on() else 4

            cv2.circle(out, (px, py), radius, color, -1)
            cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 1)

            if blink_on():
                cv2.circle(out, (px, py), 16, (0, 0, 255), 2)

        else:
            color = state_color(det)
            radius = 6 if unauthorized else 5

            cv2.circle(out, (px, py), radius, color, -1)
            cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 1)

        label = f"{det.get('camera')}-#{display_id_text(det)}"

        if det.get("unauthorized"):
            label += " UNAUTH"
        elif det.get("access_granted"):
            label += " OK"
        elif det.get("entry_state") == "CHECKING":
            label += " CHECK"
        elif det.get("emergency"):
            label += " EMG"
        elif det.get("posture") == "LYING_CANDIDATE":
            label += " LYING"

        v09.draw_text(out, label, (px + 8, py - 8), color, 0.38, 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (330, 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    v09.draw_text(
        out,
        "MAP | Entry ROI | EMG blink",
        (16, 32),
        (255, 255, 255),
        0.48,
        1,
    )

    return out


def draw_legend_item(panel, x, y, color, text):
    cv2.circle(panel, (x, y - 5), 6, color, -1)
    v09.draw_text(panel, text, (x + 18, y), (190, 190, 190), 0.42, 1)


def draw_status_panel(width, height, metrics0, metrics1, all_detections, overall_fps, state_json_path, args):
    """
    우측 하단 status panel.
    이 영역에서는 emergency 점멸을 하지 않는다.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    total_persons = metrics0.get("person_count", 0) + metrics1.get("person_count", 0)
    total_no_helmet = metrics0.get("no_helmet_count", 0) + metrics1.get("no_helmet_count", 0)
    total_checking = metrics0.get("checking_count", 0) + metrics1.get("checking_count", 0)
    total_granted = metrics0.get("granted_count", 0) + metrics1.get("granted_count", 0)
    total_unauth = metrics0.get("unauthorized_count", 0) + metrics1.get("unauthorized_count", 0)
    total_lying = metrics0.get("lying_count", 0) + metrics1.get("lying_count", 0)
    total_emg = metrics0.get("emergency_count", 0) + metrics1.get("emergency_count", 0)

    def put(text, x, y, color=(235, 235, 235), scale=0.48, thick=1):
        v09.draw_text(panel, text, (x, y), color, scale, thick)

    # Header
    put("V12 ENTRY + YOLO TRACKING", 18, 34, (255, 255, 255), 0.62, 2)
    put(
        f"FPS {overall_fps:.1f} | {args.tracker} | GPU {args.device} | EMG {args.emergency_sec:.0f}s",
        18,
        62,
        (180, 180, 180),
        0.42,
        1,
    )

    # Left summary
    lx = 24
    y = 105
    put("TOTAL", lx, y, (255, 255, 255), 0.56, 2)

    y += 30
    put(f"Persons      {total_persons}", lx, y, (255, 255, 255), 0.50, 1)

    y += 28
    put(
        f"No Helmet    {total_no_helmet}",
        lx,
        y,
        (0, 0, 255) if total_no_helmet > 0 else (0, 255, 0),
        0.50,
        1,
    )

    y += 28
    put(
        f"Checking     {total_checking}",
        lx,
        y,
        (0, 255, 255) if total_checking > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 28
    put(
        f"Granted      {total_granted}",
        lx,
        y,
        (0, 255, 0) if total_granted > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 28
    put(
        f"Unauthorized {total_unauth}",
        lx,
        y,
        (0, 0, 255) if total_unauth > 0 else (0, 255, 0),
        0.52,
        1,
    )

    y += 28
    put(
        f"Lying        {total_lying}",
        lx,
        y,
        (0, 165, 255) if total_lying > 0 else (0, 255, 0),
        0.50,
        1,
    )

    y += 32
    put(
        f"Emergency    {total_emg}",
        lx,
        y,
        (0, 0, 255) if total_emg > 0 else (0, 255, 0),
        0.62,
        2,
    )

    # Right worker list
    rx = width // 2 + 15
    y2 = 105
    put("WORKERS", rx, y2, (255, 255, 255), 0.56, 2)
    y2 += 30

    for det in all_detections[:6]:
        color = state_color(det)

        line1 = f"{det.get('camera')}-#{display_id_text(det)} {det.get('entry_state')}"
        put(line1, rx, y2, color, 0.45, 1)
        y2 += 23

        line2 = (
            f"H={det.get('helmet_status')} "
            f"hand={det.get('hand_raised')} "
            f"L={det.get('lying_candidate')}"
        )

        if det.get("entry_state") == "CHECKING":
            line2 += f" {det.get('entry_progress', 0):.1f}/{args.entry_check_sec:.0f}s"

        put(line2, rx + 8, y2, (190, 190, 190), 0.40, 1)
        y2 += 27

        if y2 > height - 105:
            break

    # Bottom legend
    ly = height - 82
    draw_legend_item(panel, 24, ly, (0, 255, 0), "safe / granted")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 255, 255), "checking")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 165, 255), "wait / low")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 0, 255), "no helmet / emergency")

    ly2 = height - 82
    put("cam0: ENTRY ROI only", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)
    ly2 += 22
    put("cam1: MONITOR ONLY", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)
    ly2 += 22
    put("Map emergency blinks only", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)

    return panel


# ============================================================
# JSON save
# ============================================================

def save_state_json(path, timestamp, detections0, detections1, metrics0, metrics1):
    all_detections = list(detections0) + list(detections1)

    persons = []
    access_events = []
    emergency_targets = []
    unauthorized_entries = []

    for det in all_detections:
        map_xy = det.get("map_xy")
        debug = det.get("debug", {})

        item = {
            "camera": det.get("camera"),
            "display_id": det.get("display_id"),
            "yolo_track_id": None if det.get("yolo_track_id") is None else int(det.get("yolo_track_id")),
            "state_key": det.get("state_key"),

            "helmet_status": det.get("helmet_status"),
            "helmet_status_raw": det.get("helmet_status_raw"),

            "posture": det.get("posture"),
            "posture_raw": det.get("posture_raw"),

            "hand_raised": bool(det.get("hand_raised", False)),
            "hand_side": det.get("hand_side"),

            "in_entry_roi": bool(det.get("in_entry_roi", False)),
            "entry_state": det.get("entry_state"),
            "entry_progress": float(det.get("entry_progress", 0.0)),
            "entry_progress_ratio": float(det.get("entry_progress_ratio", 0.0)),
            "access_granted": bool(det.get("access_granted", False)),
            "unauthorized": bool(det.get("unauthorized", False)),

            "lying_candidate": bool(det.get("lying_candidate", False)),
            "emergency": bool(det.get("emergency", False)),
            "fall_elapsed": float(det.get("fall_elapsed", 0.0)),

            "head_z": debug.get("head_z"),
            "z_residual": debug.get("z_residual"),

            "map_xy": None if map_xy is None else [float(map_xy[0]), float(map_xy[1])],
        }

        persons.append(item)

        if item["access_granted"]:
            access_events.append(item)

        if item["unauthorized"]:
            unauthorized_entries.append(item)

        if item["emergency"] and item["map_xy"] is not None:
            emergency_targets.append(item)

    state = {
        "timestamp": timestamp,
        "mode": "v12_entry_yolo_tracking_cam0_only_roi",
        "metrics": {
            "cam0": metrics0,
            "cam1": metrics1,
            "total_person_count": metrics0.get("person_count", 0) + metrics1.get("person_count", 0),
            "total_unauthorized_count": metrics0.get("unauthorized_count", 0) + metrics1.get("unauthorized_count", 0),
            "total_emergency_count": metrics0.get("emergency_count", 0) + metrics1.get("emergency_count", 0),
        },
        "persons": persons,
        "access_events": access_events,
        "unauthorized_entries": unauthorized_entries,
        "emergency_targets": emergency_targets,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    return state


# ============================================================
# Run mode
# ============================================================

def run_mode(args):
    map_info = v09.load_map_from_yaml(args.map_yaml)
    entry_roi_points, roi_data = load_entry_roi(args.entry_roi_json)

    print("======================================")
    print("Dual Camera Entry YOLO Tracking V12")
    print("--------------------------------------")
    print(f"cam0 id      : {args.cam0_id}")
    print(f"cam1 id      : {args.cam1_id}")
    print(f"det model    : {args.det_model}")
    print(f"pose model   : {args.pose_model}")
    print(f"tracker      : {args.tracker}")
    print(f"device       : {args.device}")
    print(f"imgsz        : {args.imgsz}")
    print(f"entry ROI    : {args.entry_roi_json}")
    print(f"entry check  : {args.entry_check_sec}s")
    print(f"emergency    : {args.emergency_sec}s")
    print("cam0         : ENTRY ROI ENABLED")
    print("cam1         : MONITOR ONLY")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()

    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    pose_model = YOLO(args.pose_model)

    # cam0: 입구 검사 수행
    store0 = WorkerStateStore(
        camera_name=args.cam0_name,
        entry_roi_points=entry_roi_points,
        required_check_sec=args.entry_check_sec,
        emergency_sec=args.emergency_sec,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
        max_workers=args.max_workers,
        entry_enabled=True,
    )

    # cam1: 입구 없음. 내부 감시 전용.
    store1 = WorkerStateStore(
        camera_name=args.cam1_name,
        entry_roi_points=None,
        required_check_sec=args.entry_check_sec,
        emergency_sec=args.emergency_sec,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
        max_workers=args.max_workers,
        entry_enabled=False,
    )

    cam0 = CameraEntryTracker(
        cam_id=args.cam0_id,
        cam_name=args.cam0_name,
        det_model_path=det_model_path,
        homography_path=args.cam0_homography,
        z_calib_path=args.cam0_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        worker_store=store0,
        tracker_yaml=args.tracker,
    )

    cam1 = CameraEntryTracker(
        cam_id=args.cam1_id,
        cam_name=args.cam1_name,
        det_model_path=det_model_path,
        homography_path=args.cam1_homography,
        z_calib_path=args.cam1_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        worker_store=store1,
        tracker_yaml=args.tracker,
    )

    window = "V12 Entry YOLO Tracking Dashboard"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_loop_time = time.time()
    overall_fps = 0.0

    try:
        while True:
            now = time.time()

            cam0_view, cam0_tracked, cam0_metrics = cam0.process(pose_model, args, now)
            cam1_view, cam1_tracked, cam1_metrics = cam1.process(pose_model, args, now)

            all_detections = list(cam0_tracked) + list(cam1_tracked)

            save_state_json(
                args.state_json,
                timestamp=now,
                detections0=cam0_tracked,
                detections1=cam1_tracked,
                metrics0=cam0_metrics,
                metrics1=cam1_metrics,
            )

            map_view = draw_combined_map(
                map_info["img"],
                all_detections,
                map_info,
                entry_roi_points,
            )

            dt = now - prev_loop_time
            prev_loop_time = now

            if dt > 0:
                overall_fps = overall_fps * 0.90 + (1.0 / dt) * 0.10

            status_panel = draw_status_panel(
                width=args.display_w // 2,
                height=args.display_h // 2,
                metrics0=cam0_metrics,
                metrics1=cam1_metrics,
                all_detections=all_detections,
                overall_fps=overall_fps,
                state_json_path=args.state_json,
                args=args,
            )

            q_w = args.display_w // 2
            q_h = args.display_h // 2

            cam0_panel = v09.resize_keep_ratio(cam0_view, q_w, q_h)
            cam1_panel = v09.resize_keep_ratio(cam1_view, q_w, q_h)
            map_panel = v09.resize_keep_ratio(map_view, q_w, q_h)
            status_panel = v09.resize_keep_ratio(status_panel, q_w, q_h)

            top = np.hstack([cam0_panel, cam1_panel])
            bottom = np.hstack([map_panel, status_panel])
            dashboard = np.vstack([top, bottom])

            cv2.imshow(window, dashboard)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                store0.reset()
                store1.reset()
                print("[RESET] worker states reset")

    finally:
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="run", choices=["run", "set_roi"])

    # 현재 사용 기준:
    # cam0 = 입구 카메라
    # cam1 = 내부 감시 카메라
    parser.add_argument("--cam0-id", type=int, default=0)
    parser.add_argument("--cam0-name", type=str, default="cam0")
    parser.add_argument("--cam0-homography", type=str, default="calibration/cam0_to_map.npz")
    parser.add_argument("--cam0-z-calib", type=str, default="calibration/cam0_z_calib.npz")

    parser.add_argument("--cam1-id", type=int, default=2)
    parser.add_argument("--cam1-name", type=str, default="cam1")
    parser.add_argument("--cam1-homography", type=str, default="calibration/cam1_to_map.npz")
    parser.add_argument("--cam1-z-calib", type=str, default="calibration/cam1_z_calib.npz")

    parser.add_argument("--map-yaml", type=str, default="final_project.yaml")
    parser.add_argument("--homography-output", type=str, default="map_meters", choices=["map_meters", "map_pixels"])

    parser.add_argument("--entry-roi-json", type=str, default="calibration/entry_roi.json")
    parser.add_argument("--entry-check-sec", type=float, default=3.0)

    parser.add_argument("--det-model", type=str, default="yolo_experiments/v8n_640_e50/weights/best.pt")
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--person-conf", type=float, default=0.45)
    parser.add_argument("--helmet-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    parser.add_argument("--head-radius-scale", type=float, default=0.20)
    parser.add_argument("--helmet-expand", type=float, default=1.00)

    parser.add_argument("--user-height-m", type=float, default=1.80)

    # Z-priority lying logic
    parser.add_argument("--lying-height-thres", type=float, default=0.78)
    parser.add_argument("--very-low-height-thres", type=float, default=0.55)
    parser.add_argument("--residual-thres", type=float, default=160.0)

    parser.add_argument("--hand-raise-margin", type=float, default=20.0)

    # emergency 5초
    parser.add_argument("--emergency-sec", type=float, default=5.0)

    parser.add_argument("--track-max-age", type=float, default=3.0)
    parser.add_argument("--history-len", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=4)

    parser.add_argument("--display-w", type=int, default=1600)
    parser.add_argument("--display-h", type=int, default=900)
    parser.add_argument("--process-every", type=int, default=1)

    parser.add_argument("--state-json", type=str, default="outputs/safety_state_entry_v12.json")

    args = parser.parse_args()

    if args.mode == "set_roi":
        set_roi_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()