import argparse
import json
import math
import time
from pathlib import Path

# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# COCO Pose Keypoints
# ==============================
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


# ==============================
# Basic Geometry
# ==============================
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_bottom_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def point_to_box_distance(point, box):
    x, y = point
    x1, y1, x2, y2 = box

    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)

    return float(math.sqrt(dx * dx + dy * dy))


def expand_box(box, ratio, image_w, image_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    ex = w * ratio
    ey = h * ratio

    nx1 = max(0, x1 - ex)
    ny1 = max(0, y1 - ey)
    nx2 = min(image_w - 1, x2 + ex)
    ny2 = min(image_h - 1, y2 + ey)

    return [nx1, ny1, nx2, ny2]


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


# ==============================
# Map / Homography
# ==============================
def read_yaml_minimal(path):
    """
    map yaml에서 image, resolution, origin 정도만 읽기 위한 최소 파서.
    PyYAML 없어도 동작하게 만들었음.
    """
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "image":
                result[key] = value
            elif key == "resolution":
                result[key] = float(value)
            elif key == "origin":
                value = value.replace("[", "").replace("]", "")
                parts = [float(x.strip()) for x in value.split(",")]
                result[key] = parts

    return result


def load_map_from_yaml(map_yaml_path):
    map_yaml_path = Path(map_yaml_path).expanduser()
    if not map_yaml_path.exists():
        raise FileNotFoundError(f"map yaml not found: {map_yaml_path}")

    info = read_yaml_minimal(map_yaml_path)

    image_name = info.get("image")
    if image_name is None:
        raise RuntimeError(f"image field not found in map yaml: {map_yaml_path}")

    map_img_path = Path(image_name)
    if not map_img_path.is_absolute():
        map_img_path = map_yaml_path.parent / map_img_path

    resolution = float(info.get("resolution", 0.05))
    origin = info.get("origin", [0.0, 0.0, 0.0])

    gray = cv2.imread(str(map_img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"failed to read map image: {map_img_path}")

    # occupancy map은 흑백이라 BGR로 변환
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return {
        "yaml": str(map_yaml_path),
        "image": str(map_img_path),
        "img": bgr,
        "resolution": resolution,
        "origin": origin,
        "height": bgr.shape[0],
        "width": bgr.shape[1],
    }


def load_homography(npz_path):
    npz_path = Path(npz_path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(f"homography npz not found: {npz_path}")

    data = np.load(str(npz_path))

    candidate_keys = [
        "H",
        "homography",
        "camera_to_map",
        "cam_to_map",
        "M",
        "matrix",
    ]

    for k in candidate_keys:
        if k in data:
            H = data[k]
            if H.shape == (3, 3):
                return H.astype(np.float64), k

    # key 이름을 모를 때 3x3 행렬 자동 탐색
    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.shape == (3, 3):
            return arr.astype(np.float64), k

    raise RuntimeError(f"No 3x3 homography matrix found in {npz_path}. keys={data.files}")


def apply_homography(H, pixel_xy):
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p

    if abs(q[2]) < 1e-9:
        return None

    q = q / q[2]
    return float(q[0]), float(q[1])


def map_meter_to_image_pixel(map_x, map_y, map_info):
    """
    ROS map 좌표 meter -> map image pixel
    ROS map origin은 보통 이미지 좌하단 기준.
    OpenCV 이미지는 좌상단 기준이라 y를 뒤집음.
    """
    res = map_info["resolution"]
    ox, oy, _ = map_info["origin"]
    h = map_info["height"]

    px = int(round((map_x - ox) / res))
    py_from_bottom = int(round((map_y - oy) / res))
    py = h - py_from_bottom

    return px, py


def map_image_pixel_to_meter(px, py, map_info):
    res = map_info["resolution"]
    ox, oy, _ = map_info["origin"]
    h = map_info["height"]

    map_x = ox + px * res
    map_y = oy + (h - py) * res

    return float(map_x), float(map_y)


def camera_pixel_to_map_meter(H, pixel_xy, map_info, homography_output):
    """
    homography_output:
    - map_meters: H 결과가 ROS map meter 좌표
    - map_pixels: H 결과가 map image pixel 좌표
    """
    out = apply_homography(H, pixel_xy)
    if out is None:
        return None

    a, b = out

    if homography_output == "map_meters":
        return float(a), float(b)

    if homography_output == "map_pixels":
        return map_image_pixel_to_meter(int(round(a)), int(round(b)), map_info)

    raise ValueError(f"unknown homography_output: {homography_output}")


# ==============================
# Detection
# ==============================
def extract_detection_boxes(
    result,
    names,
    image_area,
    person_conf_thres=0.35,
    helmet_conf_thres=0.25,
    person_min_area_ratio=0.002,
    person_min_height=35,
):
    persons = []
    helmets = []

    if result.boxes is None:
        return persons, helmets

    for b in result.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
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
                    "source": "det",
                }
            )

        elif cls_name == "helmet":
            if conf < helmet_conf_thres:
                continue

            helmets.append(
                {
                    "box": xyxy,
                    "conf": conf,
                }
            )

    return persons, helmets


# ==============================
# Pose
# ==============================
def extract_pose_items(pose_result):
    items = []

    if pose_result.boxes is None or pose_result.keypoints is None:
        return items

    if pose_result.boxes.xyxy is None:
        return items

    boxes = pose_result.boxes.xyxy.detach().cpu().numpy()

    if pose_result.boxes.conf is not None:
        box_confs = pose_result.boxes.conf.detach().cpu().numpy()
    else:
        box_confs = np.ones(len(boxes), dtype=np.float32)

    kpts_xy = pose_result.keypoints.xy
    if kpts_xy is None:
        return items

    kpts_xy = kpts_xy.detach().cpu().numpy()

    kpts_conf = pose_result.keypoints.conf
    if kpts_conf is not None:
        kpts_conf = kpts_conf.detach().cpu().numpy()

    for i in range(len(boxes)):
        items.append(
            {
                "box": boxes[i].astype(float).tolist(),
                "conf": float(box_confs[i]),
                "kpts_xy": kpts_xy[i],
                "kpts_conf": None if kpts_conf is None else kpts_conf[i],
            }
        )

    return items


def get_visible_point(kpts_xy, kpts_conf, idx, conf_thres=0.25):
    x, y = kpts_xy[idx]

    if x <= 1 and y <= 1:
        return None

    if kpts_conf is not None and kpts_conf[idx] < conf_thres:
        return None

    return np.array([x, y], dtype=np.float32)


def get_visible_points(pose_item, indices, conf_thres=0.25):
    pts = []
    for idx in indices:
        p = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], idx, conf_thres)
        if p is not None:
            pts.append(p)
    return pts


def get_head_center_and_source(pose_item, conf_thres=0.25):
    face_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    face_pts = get_visible_points(pose_item, face_indices, conf_thres)

    if len(face_pts) >= 1:
        return np.mean(np.stack(face_pts, axis=0), axis=0), "face"

    ls = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_SHOULDER, conf_thres)
    rs = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_SHOULDER, conf_thres)
    lh = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_HIP, conf_thres)
    rh = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_HIP, conf_thres)

    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0

        torso_vec = hip_mid - shoulder_mid
        torso_len = float(np.linalg.norm(torso_vec))

        if torso_len > 5:
            unit = torso_vec / torso_len
            inferred_head = shoulder_mid - unit * torso_len * 0.40
            return inferred_head, "shoulder_inferred"

    return None, None


def get_torso_horizontal(pose_item, conf_thres=0.25):
    ls = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_SHOULDER, conf_thres)
    rs = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_SHOULDER, conf_thres)
    lh = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_HIP, conf_thres)
    rh = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_HIP, conf_thres)

    if ls is None or rs is None or lh is None or rh is None:
        return False, None, None

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    horizontal = abs(dx) > abs(dy) * 1.05

    return horizontal, shoulder_mid, hip_mid


def get_keypoint_spread_ratio(pose_item, conf_thres=0.25):
    pts = get_visible_points(pose_item, list(range(17)), conf_thres)

    if len(pts) < 4:
        return 0.0, len(pts)

    arr = np.stack(pts, axis=0)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)

    kw = max(1.0, x_max - x_min)
    kh = max(1.0, y_max - y_min)

    return kw / kh, len(pts)


def match_pose_to_person(person_box, pose_items):
    best_iou = 0.0
    best_item = None

    for item in pose_items:
        iou = box_iou(person_box, item["box"])
        if iou > best_iou:
            best_iou = iou
            best_item = item

    if best_iou >= 0.10:
        return best_item, best_iou

    person_c = box_center(person_box)

    for item in pose_items:
        pose_box = item["box"]
        pose_c = box_center(pose_box)

        if point_in_box(pose_c, person_box) or point_in_box(person_c, pose_box):
            return item, box_iou(person_box, pose_box)

    return None, 0.0


def add_pose_only_persons(persons, pose_items, image_area, min_pose_area_ratio=0.002):
    new_persons = list(persons)

    for pose_item in pose_items:
        pose_box = pose_item["box"]
        pose_area_ratio = box_area(pose_box) / max(1, image_area)

        if pose_area_ratio < min_pose_area_ratio:
            continue

        already_matched = False
        for p in persons:
            if box_iou(p["box"], pose_box) > 0.15:
                already_matched = True
                break

        if not already_matched:
            new_persons.append(
                {
                    "box": pose_box,
                    "conf": pose_item["conf"],
                    "source": "pose",
                }
            )

    return new_persons


# ==============================
# Helmet Logic
# ==============================
def judge_helmet_status(
    person_box,
    helmets,
    head_center,
    head_source,
    image_w,
    image_h,
    head_radius_scale=0.20,
    helmet_expand_ratio=1.00,
):
    px1, py1, px2, py2 = person_box
    person_w = px2 - px1
    person_h = py2 - py1
    person_size = math.sqrt(max(1.0, person_w * person_h))

    expanded_person = expand_box(person_box, 0.08, image_w, image_h)
    related_helmets = []

    for h in helmets:
        hb = h["box"]
        hc = box_center(hb)

        if (
            point_in_box(hc, expanded_person)
            or box_iou(hb, expanded_person) > 0.02
            or point_to_box_distance(hc, expanded_person) < 35
        ):
            related_helmets.append(h)

    head_radius = int(np.clip(person_size * head_radius_scale, 35, 110))

    if head_center is not None:
        for h in related_helmets:
            expanded_helmet = expand_box(h["box"], helmet_expand_ratio, image_w, image_h)

            if point_in_box(head_center, expanded_helmet):
                return "HELMET_ON", head_radius, related_helmets

            dist = point_to_box_distance(head_center, expanded_helmet)
            if dist <= head_radius:
                return "HELMET_ON", head_radius, related_helmets

        if head_source in ["face", "shoulder_inferred"]:
            return "NO_HELMET", head_radius, related_helmets

        return "UNKNOWN", head_radius, related_helmets

    aspect = person_w / person_h if person_h > 1 else 0.0

    if aspect < 0.95:
        head_zone = [px1, py1, px2, py1 + person_h * 0.40]

        for h in related_helmets:
            hc = box_center(h["box"])
            if point_in_box(hc, head_zone):
                return "HELMET_ON", head_radius, related_helmets

    return "UNKNOWN", head_radius, related_helmets


# ==============================
# Fall Logic
# ==============================
def judge_fall_candidate(
    person_box,
    pose_item,
    image_h,
    pose_conf=0.25,
    floor_y_ratio=0.45,
    fall_score_thres=3,
):
    x1, y1, x2, y2 = person_box

    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph

    center_y = (y1 + y2) / 2.0
    bottom_y = y2

    score = 0
    reasons = []

    if aspect > 1.15:
        score += 1
        reasons.append("wide_box")

    if aspect > 1.45:
        score += 1
        reasons.append("very_wide_box")

    if aspect > 1.85:
        score += 1
        reasons.append("extreme_wide_box")

    torso_horizontal = False
    shoulder_mid = None
    hip_mid = None
    kpt_ratio = 0.0
    visible_count = 0

    if pose_item is not None:
        torso_horizontal, shoulder_mid, hip_mid = get_torso_horizontal(pose_item, pose_conf)

        if torso_horizontal:
            score += 2
            reasons.append("horizontal_torso")

        kpt_ratio, visible_count = get_keypoint_spread_ratio(pose_item, pose_conf)

        if visible_count >= 5 and kpt_ratio > 1.20:
            score += 1
            reasons.append("wide_keypoints")

        if visible_count >= 5 and kpt_ratio > 1.60:
            score += 1
            reasons.append("very_wide_keypoints")

    floor_area = center_y > image_h * floor_y_ratio or bottom_y > image_h * 0.62

    if floor_area and aspect > 1.10:
        score += 1
        reasons.append("floor_area")

    if aspect < 0.75 and pose_item is not None and not torso_horizontal:
        score -= 1
        reasons.append("upright_penalty")

    candidate = score >= fall_score_thres

    if aspect > 1.90 and floor_area:
        candidate = True
        reasons.append("bbox_only_fall")

    debug = {
        "aspect": aspect,
        "score": score,
        "reasons": reasons,
        "torso_horizontal": torso_horizontal,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "kpt_ratio": kpt_ratio,
        "visible_kpts": visible_count,
        "floor_area": floor_area,
    }

    return candidate, debug


# ==============================
# Lightweight Track Manager
# ==============================
class TrackManager:
    def __init__(self, max_distance_px=90, max_age_sec=2.0, emergency_sec=10.0):
        self.max_distance_px = max_distance_px
        self.max_age_sec = max_age_sec
        self.emergency_sec = emergency_sec
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now):
        """
        detections item:
        {
          "image_point": np.array([u,v]),
          "map_xy": (x,y) or None,
          "helmet_status": str,
          "fall_candidate": bool,
          ...
        }
        """
        assigned_track_ids = set()

        for det in detections:
            p = det["image_point"]

            best_id = None
            best_dist = 1e9

            for tid, tr in self.tracks.items():
                if tid in assigned_track_ids:
                    continue

                age = now - tr["last_seen"]
                if age > self.max_age_sec:
                    continue

                prev = tr["image_point"]
                dist = float(np.linalg.norm(p - prev))

                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None and best_dist <= self.max_distance_px:
                tid = best_id
            else:
                tid = self.next_id
                self.next_id += 1

                self.tracks[tid] = {
                    "id": tid,
                    "first_seen": now,
                    "last_seen": now,
                    "image_point": p,
                    "map_xy": None,
                    "fall_start": None,
                    "emergency": False,
                    "helmet_status": "UNKNOWN",
                }

            tr = self.tracks[tid]
            tr["last_seen"] = now
            tr["image_point"] = p
            tr["map_xy"] = det.get("map_xy")
            tr["helmet_status"] = det.get("helmet_status", "UNKNOWN")
            tr["fall_candidate"] = bool(det.get("fall_candidate", False))
            tr["fall_debug"] = det.get("fall_debug", {})
            tr["person_box"] = det.get("person_box")
            tr["source"] = det.get("source", "det")
            tr["conf"] = det.get("conf", 0.0)

            if tr["fall_candidate"]:
                if tr["fall_start"] is None:
                    tr["fall_start"] = now

                fall_elapsed = now - tr["fall_start"]
                tr["fall_elapsed"] = fall_elapsed

                if fall_elapsed >= self.emergency_sec:
                    tr["emergency"] = True
            else:
                tr["fall_start"] = None
                tr["fall_elapsed"] = 0.0
                tr["emergency"] = False

            det["track_id"] = tid
            det["fall_elapsed"] = tr.get("fall_elapsed", 0.0)
            det["emergency"] = tr["emergency"]

            assigned_track_ids.add(tid)

        # 오래된 track 제거
        dead = []
        for tid, tr in self.tracks.items():
            if now - tr["last_seen"] > self.max_age_sec:
                dead.append(tid)

        for tid in dead:
            del self.tracks[tid]

        return detections

    def get_active_tracks(self):
        return list(self.tracks.values())


# ==============================
# Drawing
# ==============================
def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        cv2.putText(
            img,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_dashboard_panel(
    width,
    height,
    person_count,
    helmet_count,
    no_helmet_count,
    unknown_helmet_count,
    fall_candidate_count,
    emergency_count,
    fps,
    cam_name,
    state_json_path,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    y = 35
    draw_text(panel, f"LOCAL SAFETY DASHBOARD - {cam_name}", (20, y), (255, 255, 255), 0.75, 2)

    y += 45
    draw_text(panel, f"FPS: {fps:.1f}", (20, y), (200, 200, 200), 0.65, 2)

    y += 40
    draw_text(panel, f"Persons: {person_count}", (20, y), (255, 255, 255), 0.70, 2)

    y += 40
    draw_text(panel, f"Helmets: {helmet_count}", (20, y), (255, 255, 255), 0.70, 2)

    y += 40
    color = (0, 0, 255) if no_helmet_count > 0 else (0, 255, 0)
    draw_text(panel, f"No Helmet: {no_helmet_count}", (20, y), color, 0.70, 2)

    y += 40
    draw_text(panel, f"Helmet Unknown: {unknown_helmet_count}", (20, y), (0, 165, 255), 0.70, 2)

    y += 40
    color = (0, 165, 255) if fall_candidate_count > 0 else (0, 255, 0)
    draw_text(panel, f"Lying Candidate: {fall_candidate_count}", (20, y), color, 0.70, 2)

    y += 40
    color = (0, 0, 255) if emergency_count > 0 else (0, 255, 0)
    draw_text(panel, f"EMERGENCY: {emergency_count}", (20, y), color, 0.80, 2)

    y += 60
    draw_text(panel, "Map dot color:", (20, y), (255, 255, 255), 0.60, 2)

    y += 32
    cv2.circle(panel, (35, y - 5), 7, (0, 255, 0), -1)
    draw_text(panel, "helmet on", (55, y), (200, 200, 200), 0.55, 1)

    y += 28
    cv2.circle(panel, (35, y - 5), 7, (0, 0, 255), -1)
    draw_text(panel, "no helmet", (55, y), (200, 200, 200), 0.55, 1)

    y += 28
    cv2.circle(panel, (35, y - 5), 7, (0, 165, 255), -1)
    draw_text(panel, "unknown / lying candidate", (55, y), (200, 200, 200), 0.55, 1)

    y += 28
    cv2.circle(panel, (35, y - 5), 9, (255, 0, 255), -1)
    draw_text(panel, "emergency target", (55, y), (200, 200, 200), 0.55, 1)

    y = height - 65
    draw_text(panel, "q: quit | r: reset", (20, y), (255, 255, 255), 0.60, 2)

    y += 30
    draw_text(panel, f"state: {state_json_path}", (20, y), (180, 180, 180), 0.45, 1)

    return panel


def color_for_status(helmet_status, fall_candidate, emergency):
    if emergency:
        return (255, 0, 255)
    if fall_candidate:
        return (0, 165, 255)
    if helmet_status == "NO_HELMET":
        return (0, 0, 255)
    if helmet_status == "HELMET_ON":
        return (0, 255, 0)
    return (0, 165, 255)


def draw_map_points(map_img, detections, map_info):
    out = map_img.copy()

    for det in detections:
        map_xy = det.get("map_xy")
        if map_xy is None:
            continue

        mx, my = map_xy
        px, py = map_meter_to_image_pixel(mx, my, map_info)

        if not (0 <= px < map_info["width"] and 0 <= py < map_info["height"]):
            continue

        color = color_for_status(
            det.get("helmet_status", "UNKNOWN"),
            det.get("fall_candidate", False),
            det.get("emergency", False),
        )

        radius = 7
        if det.get("emergency", False):
            radius = 12

        cv2.circle(out, (px, py), radius, color, -1)
        cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 2)

        label = f"ID{det.get('track_id', '?')}"
        if det.get("emergency", False):
            label += " EMG"
        elif det.get("helmet_status") == "NO_HELMET":
            label += " NO_HELMET"
        elif det.get("fall_candidate", False):
            label += " LYING"

        draw_text(out, label, (px + 10, py - 10), color, 0.45, 1)

    return out


def resize_keep_ratio(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized

    return canvas


def save_state_json(path, timestamp, cam_name, detections, tracks):
    emergency_targets = []

    persons = []

    for det in detections:
        map_xy = det.get("map_xy")
        item = {
            "track_id": det.get("track_id"),
            "helmet_status": det.get("helmet_status"),
            "fall_candidate": bool(det.get("fall_candidate", False)),
            "emergency": bool(det.get("emergency", False)),
            "fall_elapsed": float(det.get("fall_elapsed", 0.0)),
            "source": det.get("source"),
            "conf": float(det.get("conf", 0.0)),
            "map_xy": None if map_xy is None else [float(map_xy[0]), float(map_xy[1])],
        }
        persons.append(item)

        if item["emergency"] and item["map_xy"] is not None:
            emergency_targets.append(item)

    state = {
        "timestamp": timestamp,
        "camera": cam_name,
        "person_count": len(detections),
        "persons": persons,
        "emergency_count": len(emergency_targets),
        "emergency_targets": emergency_targets,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    return state


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--cam-name", type=str, default="cam0")

    parser.add_argument(
        "--det-model",
        type=str,
        default=str(
            VISION_DIR
            / "yolo_experiments/best.pt"
        ),
    )
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument(
        "--map-yaml",
        type=str,
        default=str(VISION_DIR / "final_project.yaml"),
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=str(VISION_DIR / "calibration/cam0_to_map.npz"),
    )
    parser.add_argument(
        "--homography-output",
        type=str,
        default="map_meters",
        choices=["map_meters", "map_pixels"],
    )

    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--helmet-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    parser.add_argument("--head-radius-scale", type=float, default=0.20)
    parser.add_argument("--helmet-expand", type=float, default=1.00)

    parser.add_argument("--emergency-sec", type=float, default=10.0)
    parser.add_argument("--fall-score-thres", type=int, default=3)
    parser.add_argument("--floor-y-ratio", type=float, default=0.45)

    parser.add_argument("--track-max-distance", type=float, default=90.0)
    parser.add_argument("--track-max-age", type=float, default=2.0)

    parser.add_argument(
        "--state-json",
        type=str,
        default=str(VISION_DIR / "outputs/safety_state.json"),
    )

    parser.add_argument("--display-w", type=int, default=1280)
    parser.add_argument("--display-h", type=int, default=720)

    args = parser.parse_args()

    print("======================================")
    print("Local Safety Dashboard")
    print("--------------------------------------")
    print(f"camera      : {args.camera}")
    print(f"cam_name    : {args.cam_name}")
    print(f"det_model   : {args.det_model}")
    print(f"pose_model  : {args.pose_model}")
    print(f"device      : {args.device}")
    print(f"map_yaml    : {args.map_yaml}")
    print(f"homography  : {args.homography}")
    print(f"state_json  : {args.state_json}")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()
    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    map_info = load_map_from_yaml(args.map_yaml)
    H, H_key = load_homography(args.homography)

    print(f"[INFO] map image: {map_info['image']}")
    print(f"[INFO] map resolution: {map_info['resolution']}")
    print(f"[INFO] map origin: {map_info['origin']}")
    print(f"[INFO] homography key: {H_key}")
    print(f"[INFO] homography output mode: {args.homography_output}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(args.pose_model)
    det_names = det_model.names

    print("[INFO] detection class names:", det_names)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {args.camera}")

    track_manager = TrackManager(
        max_distance_px=args.track_max_distance,
        max_age_sec=args.track_max_age,
        emergency_sec=args.emergency_sec,
    )

    window_name = "Local Safety Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame read failed")
            continue

        now = time.time()
        image_h, image_w = frame.shape[:2]
        image_area = image_h * image_w

        # ==============================
        # Detection
        # ==============================
        det_result = det_model(
            frame,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=args.device,
            verbose=False,
        )[0]

        persons, helmets = extract_detection_boxes(
            det_result,
            det_names,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            helmet_conf_thres=args.helmet_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        # ==============================
        # Pose
        # ==============================
        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_items = extract_pose_items(pose_result)

        persons = add_pose_only_persons(
            persons,
            pose_items,
            image_area=image_area,
            min_pose_area_ratio=args.person_min_area_ratio,
        )

        # ==============================
        # Person state
        # ==============================
        detections_for_tracks = []

        preview = frame.copy()

        # draw helmet boxes
        for h in helmets:
            draw_box(preview, h["box"], (0, 255, 255), f"helmet {h['conf']:.2f}", 2)

        for person in persons:
            pbox = person["box"]

            pose_item, pose_iou = match_pose_to_person(pbox, pose_items)

            head_center = None
            head_source = None

            if pose_item is not None:
                head_center, head_source = get_head_center_and_source(pose_item, args.pose_conf)

            helmet_status, head_radius, related_helmets = judge_helmet_status(
                pbox,
                helmets,
                head_center,
                head_source,
                image_w,
                image_h,
                head_radius_scale=args.head_radius_scale,
                helmet_expand_ratio=args.helmet_expand,
            )

            fall_candidate, fall_debug = judge_fall_candidate(
                pbox,
                pose_item,
                image_h=image_h,
                pose_conf=args.pose_conf,
                floor_y_ratio=args.floor_y_ratio,
                fall_score_thres=args.fall_score_thres,
            )

            foot_px = box_bottom_center(pbox)

            map_xy = camera_pixel_to_map_meter(
                H,
                foot_px,
                map_info,
                homography_output=args.homography_output,
            )

            det_item = {
                "image_point": foot_px,
                "map_xy": map_xy,
                "helmet_status": helmet_status,
                "fall_candidate": fall_candidate,
                "fall_debug": fall_debug,
                "person_box": pbox,
                "source": person["source"],
                "conf": person["conf"],
                "head_center": head_center,
                "head_radius": head_radius,
                "head_source": head_source,
            }

            detections_for_tracks.append(det_item)

        tracked = track_manager.update(detections_for_tracks, now)

        # ==============================
        # Draw camera preview
        # ==============================
        for det in tracked:
            box = det["person_box"]
            tid = det["track_id"]
            helmet_status = det["helmet_status"]
            fall_candidate = det["fall_candidate"]
            emergency = det["emergency"]

            color = color_for_status(helmet_status, fall_candidate, emergency)

            label = f"ID{tid} {helmet_status}"
            if emergency:
                label += " EMERGENCY"
            elif fall_candidate:
                label += f" LYING {det['fall_elapsed']:.1f}/{args.emergency_sec:.0f}s"

            draw_box(preview, box, color, label, 2)

            head_center = det.get("head_center")
            head_radius = det.get("head_radius", 40)

            if head_center is not None:
                hc = tuple(head_center.astype(int))
                cv2.circle(preview, hc, int(head_radius), (255, 0, 255), 2)
                cv2.circle(preview, hc, 4, (255, 0, 255), -1)

            map_xy = det.get("map_xy")
            if map_xy is not None:
                x1, y1, x2, y2 = map(int, box)
                draw_text(
                    preview,
                    f"map({map_xy[0]:.2f}, {map_xy[1]:.2f})",
                    (x1, min(image_h - 10, y2 + 22)),
                    color,
                    0.48,
                    1,
                )

        # ==============================
        # Metrics
        # ==============================
        person_count = len(tracked)
        helmet_count = len(helmets)
        no_helmet_count = sum(1 for d in tracked if d["helmet_status"] == "NO_HELMET")
        unknown_helmet_count = sum(1 for d in tracked if d["helmet_status"] == "UNKNOWN")
        fall_candidate_count = sum(1 for d in tracked if d["fall_candidate"])
        emergency_count = sum(1 for d in tracked if d["emergency"])

        # ==============================
        # Save JSON state
        # ==============================
        save_state_json(
            args.state_json,
            timestamp=now,
            cam_name=args.cam_name,
            detections=tracked,
            tracks=track_manager.get_active_tracks(),
        )

        # ==============================
        # Map view
        # ==============================
        map_view = draw_map_points(map_info["img"], tracked, map_info)

        # ==============================
        # Dashboard layout
        # ==============================
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = fps * 0.90 + (1.0 / dt) * 0.10

        panel_w = 360
        total_w = args.display_w
        total_h = args.display_h

        left_w = (total_w - panel_w) // 2
        right_w = total_w - panel_w - left_w

        cam_panel = resize_keep_ratio(preview, left_w, total_h)
        map_panel = resize_keep_ratio(map_view, right_w, total_h)

        info_panel = make_dashboard_panel(
            panel_w,
            total_h,
            person_count=person_count,
            helmet_count=helmet_count,
            no_helmet_count=no_helmet_count,
            unknown_helmet_count=unknown_helmet_count,
            fall_candidate_count=fall_candidate_count,
            emergency_count=emergency_count,
            fps=fps,
            cam_name=args.cam_name,
            state_json_path=args.state_json,
        )

        dashboard = np.hstack([cam_panel, map_panel, info_panel])

        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            track_manager = TrackManager(
                max_distance_px=args.track_max_distance,
                max_age_sec=args.track_max_age,
                emergency_sec=args.emergency_sec,
            )
            print("[RESET] track manager reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
