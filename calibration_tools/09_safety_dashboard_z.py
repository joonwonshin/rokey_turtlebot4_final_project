import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# COCO 17 Pose Keypoints
# ==============================
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


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
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def point_to_box_distance(point, box):
    x, y = float(point[0]), float(point[1])
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
                result[key] = [float(x.strip()) for x in value.split(",")]

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

    gray = cv2.imread(str(map_img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"failed to read map image: {map_img_path}")

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return {
        "yaml": str(map_yaml_path),
        "image": str(map_img_path),
        "img": bgr,
        "resolution": float(info.get("resolution", 0.05)),
        "origin": info.get("origin", [0.0, 0.0, 0.0]),
        "height": bgr.shape[0],
        "width": bgr.shape[1],
    }


def load_homography(npz_path):
    npz_path = Path(npz_path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(f"homography npz not found: {npz_path}")

    data = np.load(str(npz_path))

    candidate_keys = ["H", "homography", "camera_to_map", "cam_to_map", "M", "matrix"]

    for k in candidate_keys:
        if k in data:
            H = data[k]
            if H.shape == (3, 3):
                return H.astype(np.float64), k

    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.shape == (3, 3):
            return arr.astype(np.float64), k

    raise RuntimeError(f"No 3x3 homography found in {npz_path}. keys={data.files}")


def apply_homography(H, pixel_xy):
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p

    if abs(q[2]) < 1e-9:
        return None

    q = q / q[2]
    return float(q[0]), float(q[1])


def map_meter_to_image_pixel(map_x, map_y, map_info):
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
# Z Height Estimation
# ==============================
def load_z_calibration(npz_path):
    npz_path = Path(npz_path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(f"z calibration npz not found: {npz_path}")

    data = np.load(str(npz_path))
    if "P" not in data:
        raise RuntimeError(f"P not found in z calibration npz: {npz_path}")

    return data["P"].astype(np.float64)


def project_points(P, world_points):
    world_points = np.asarray(world_points, dtype=np.float64)

    if world_points.ndim == 1:
        world_points = world_points.reshape(1, 3)

    ones = np.ones((world_points.shape[0], 1), dtype=np.float64)
    Xh = np.hstack([world_points, ones])

    x = (P @ Xh.T).T

    uv = np.full((world_points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.abs(x[:, 2]) > 1e-9

    uv[valid, 0] = x[valid, 0] / x[valid, 2]
    uv[valid, 1] = x[valid, 1] / x[valid, 2]

    return uv


def estimate_z_on_vertical_line(P, map_x, map_y, image_uv):
    """
    Assume target point lies on vertical line:
    world = [map_x, map_y, Z]

    Given image pixel [u,v], solve best Z.
    """
    u, v = float(image_uv[0]), float(image_uv[1])

    r1 = P[0]
    r2 = P[1]
    r3 = P[2]

    a1 = r1[0] * map_x + r1[1] * map_y + r1[3]
    b1 = r1[2]

    a2 = r2[0] * map_x + r2[1] * map_y + r2[3]
    b2 = r2[2]

    a3 = r3[0] * map_x + r3[1] * map_y + r3[3]
    b3 = r3[2]

    A = np.array(
        [
            [u * b3 - b1],
            [v * b3 - b2],
        ],
        dtype=np.float64,
    )

    b = np.array(
        [
            a1 - u * a3,
            a2 - v * a3,
        ],
        dtype=np.float64,
    )

    if np.linalg.norm(A) < 1e-12:
        return None, None, None

    z_hat = float(np.linalg.lstsq(A, b, rcond=None)[0][0])

    projected = project_points(P, np.array([[map_x, map_y, z_hat]], dtype=np.float64))[0]
    residual = float(np.linalg.norm(projected - np.array([u, v], dtype=np.float64)))

    return z_hat, residual, projected


# ==============================
# Detection
# ==============================
def extract_detection_boxes(
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


def get_visible_point(pose_item, idx, conf_thres=0.25):
    if pose_item is None:
        return None

    kpts_xy = pose_item["kpts_xy"]
    kpts_conf = pose_item["kpts_conf"]

    x, y = kpts_xy[idx]

    if x <= 1 and y <= 1:
        return None

    if kpts_conf is not None and kpts_conf[idx] < conf_thres:
        return None

    return np.array([x, y], dtype=np.float32)


def get_visible_points(pose_item, indices, conf_thres=0.25):
    pts = []
    for idx in indices:
        p = get_visible_point(pose_item, idx, conf_thres)
        if p is not None:
            pts.append(p)
    return pts


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


def get_head_point(pose_item, person_box=None, helmet_box=None, conf_thres=0.25):
    """
    1순위: face keypoints
    2순위: shoulder/hip inferred
    3순위: helmet center
    4순위: bbox top center
    """
    if pose_item is not None:
        face_pts = get_visible_points(
            pose_item,
            [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR],
            conf_thres,
        )

        if len(face_pts) >= 1:
            return np.mean(np.stack(face_pts, axis=0), axis=0), "face"

        ls = get_visible_point(pose_item, LEFT_SHOULDER, conf_thres)
        rs = get_visible_point(pose_item, RIGHT_SHOULDER, conf_thres)
        lh = get_visible_point(pose_item, LEFT_HIP, conf_thres)
        rh = get_visible_point(pose_item, RIGHT_HIP, conf_thres)

        if ls is not None and rs is not None and lh is not None and rh is not None:
            shoulder_mid = (ls + rs) / 2.0
            hip_mid = (lh + rh) / 2.0

            torso_vec = hip_mid - shoulder_mid
            torso_len = float(np.linalg.norm(torso_vec))

            if torso_len > 5:
                unit = torso_vec / torso_len
                inferred_head = shoulder_mid - unit * torso_len * 0.40
                return inferred_head, "shoulder_inferred"

    if helmet_box is not None:
        return box_center(helmet_box), "helmet_center"

    if person_box is not None:
        x1, y1, x2, y2 = person_box
        return np.array([(x1 + x2) / 2.0, y1], dtype=np.float32), "bbox_top"

    return None, None


def get_foot_point(pose_item, person_box, conf_thres=0.25):
    """
    1순위: ankle midpoint
    2순위: bbox bottom center
    """
    if pose_item is not None:
        ankles = get_visible_points(pose_item, [LEFT_ANKLE, RIGHT_ANKLE], conf_thres)
        if len(ankles) >= 1:
            return np.mean(np.stack(ankles, axis=0), axis=0), "ankle"

    return box_bottom_center(person_box), "bbox_bottom"


def get_torso_horizontal(pose_item, conf_thres=0.25):
    if pose_item is None:
        return False, None, None

    ls = get_visible_point(pose_item, LEFT_SHOULDER, conf_thres)
    rs = get_visible_point(pose_item, RIGHT_SHOULDER, conf_thres)
    lh = get_visible_point(pose_item, LEFT_HIP, conf_thres)
    rh = get_visible_point(pose_item, RIGHT_HIP, conf_thres)

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


def get_standing_order_score(pose_item, conf_thres=0.25):
    """
    standing이면 y순서가 대체로:
    head < shoulder < hip < ankle
    """
    if pose_item is None:
        return 0, {}

    head_pts = get_visible_points(
        pose_item,
        [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR],
        conf_thres,
    )
    shoulder_pts = get_visible_points(
        pose_item,
        [LEFT_SHOULDER, RIGHT_SHOULDER],
        conf_thres,
    )
    hip_pts = get_visible_points(
        pose_item,
        [LEFT_HIP, RIGHT_HIP],
        conf_thres,
    )
    ankle_pts = get_visible_points(
        pose_item,
        [LEFT_ANKLE, RIGHT_ANKLE],
        conf_thres,
    )

    info = {}
    score = 0

    head_y = float(np.mean([p[1] for p in head_pts])) if head_pts else None
    shoulder_y = float(np.mean([p[1] for p in shoulder_pts])) if shoulder_pts else None
    hip_y = float(np.mean([p[1] for p in hip_pts])) if hip_pts else None
    ankle_y = float(np.mean([p[1] for p in ankle_pts])) if ankle_pts else None

    info["head_y"] = head_y
    info["shoulder_y"] = shoulder_y
    info["hip_y"] = hip_y
    info["ankle_y"] = ankle_y

    if head_y is not None and shoulder_y is not None and head_y < shoulder_y:
        score += 1

    if shoulder_y is not None and hip_y is not None and shoulder_y < hip_y:
        score += 1

    if hip_y is not None and ankle_y is not None and hip_y < ankle_y:
        score += 1

    return score, info


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
def find_nearest_related_helmet(person_box, helmets, image_w, image_h):
    expanded_person = expand_box(person_box, 0.08, image_w, image_h)

    related = []
    for h in helmets:
        hb = h["box"]
        hc = box_center(hb)

        if (
            point_in_box(hc, expanded_person)
            or box_iou(hb, expanded_person) > 0.02
            or point_to_box_distance(hc, expanded_person) < 35
        ):
            related.append(h)

    if not related:
        return None, related

    pc = box_center(person_box)
    related_sorted = sorted(related, key=lambda h: float(np.linalg.norm(box_center(h["box"]) - pc)))
    return related_sorted[0], related


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

    nearest_helmet, related_helmets = find_nearest_related_helmet(person_box, helmets, image_w, image_h)

    head_radius = int(np.clip(person_size * head_radius_scale, 35, 110))

    if head_center is not None:
        for h in related_helmets:
            expanded_helmet = expand_box(h["box"], helmet_expand_ratio, image_w, image_h)

            if point_in_box(head_center, expanded_helmet):
                return "HELMET_ON", head_radius, related_helmets

            dist = point_to_box_distance(head_center, expanded_helmet)
            if dist <= head_radius:
                return "HELMET_ON", head_radius, related_helmets

        if head_source in ["face", "shoulder_inferred", "bbox_top"]:
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
# Posture / Lying Logic
# ==============================
def judge_posture_v09(
    person_box,
    pose_item,
    head_z,
    z_residual,
    image_h,
    user_height_m=1.80,
    pose_conf=0.25,
    lying_height_thres=0.70,
    very_low_height_thres=0.48,
    residual_thres=140.0,
):
    """
    v09 logic:
    - head_z만 낮다고 lying 아님.
    - low head_z + horizontal torso / wide bbox / wide keypoints 조합이 필요.
    - crouch, kneel, bow는 머리는 낮아져도 bbox가 세로형이고 torso가 수평이 아니므로 lying에서 제외.
    """
    x1, y1, x2, y2 = person_box

    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph

    center_y = (y1 + y2) / 2.0
    bottom_y = y2
    floor_area = center_y > image_h * 0.45 or bottom_y > image_h * 0.62

    torso_horizontal, shoulder_mid, hip_mid = get_torso_horizontal(pose_item, pose_conf)
    kpt_ratio, visible_kpts = get_keypoint_spread_ratio(pose_item, pose_conf)
    standing_order, standing_info = get_standing_order_score(pose_item, pose_conf)

    lying_score = 0
    upright_score = 0
    reasons = []
    upright_reasons = []

    # ------------------------------
    # Z height evidence
    # ------------------------------
    if head_z is not None and np.isfinite(head_z):
        # 비정상적으로 튀는 값 방지
        if head_z < -0.20 or head_z > user_height_m + 1.20:
            reasons.append("z_outlier_ignored")
        else:
            if head_z < very_low_height_thres:
                lying_score += 3
                reasons.append("very_low_head_z")

            elif head_z < lying_height_thres:
                lying_score += 2
                reasons.append("low_head_z")

            elif head_z < user_height_m * 0.55:
                lying_score += 1
                reasons.append("mid_low_head_z")

            if head_z > user_height_m * 0.62:
                upright_score += 2
                upright_reasons.append("head_high_enough")

            elif head_z > user_height_m * 0.48:
                upright_score += 1
                upright_reasons.append("head_mid_height")

    # ------------------------------
    # Shape evidence
    # ------------------------------
    if aspect > 1.15:
        lying_score += 1
        reasons.append("wide_box")

    if aspect > 1.45:
        lying_score += 1
        reasons.append("very_wide_box")

    if aspect > 1.85:
        lying_score += 1
        reasons.append("extreme_wide_box")

    if aspect < 0.85:
        upright_score += 1
        upright_reasons.append("vertical_box")

    # ------------------------------
    # Pose evidence
    # ------------------------------
    if torso_horizontal:
        lying_score += 2
        reasons.append("horizontal_torso")
    else:
        if pose_item is not None:
            upright_score += 1
            upright_reasons.append("not_horizontal_torso")

    if visible_kpts >= 5 and kpt_ratio > 1.25:
        lying_score += 1
        reasons.append("wide_keypoints")

    if visible_kpts >= 5 and kpt_ratio > 1.65:
        lying_score += 1
        reasons.append("very_wide_keypoints")

    if standing_order >= 2:
        upright_score += 2
        upright_reasons.append("standing_order")

    if standing_order >= 3:
        upright_score += 1
        upright_reasons.append("strong_standing_order")

    # ------------------------------
    # Residual evidence
    # ------------------------------
    # z residual은 단독 lying 근거로 쓰지 않음.
    # vertical-line assumption이 안 맞는다는 보조 신호로만 사용.
    if z_residual is not None and np.isfinite(z_residual):
        if z_residual > residual_thres and (torso_horizontal or aspect > 1.20 or kpt_ratio > 1.25):
            lying_score += 1
            reasons.append("high_z_residual_with_pose_shape")

    # ------------------------------
    # Floor evidence
    # ------------------------------
    if floor_area and (aspect > 1.10 or torso_horizontal or kpt_ratio > 1.25):
        lying_score += 1
        reasons.append("floor_area")

    # ------------------------------
    # Anti false-positive logic
    # ------------------------------
    # 무릎 꿇기/쭈그림/인사는 head_z가 낮아도 보통 세로형 bbox + torso not horizontal
    crouch_bow_like = False
    if head_z is not None and np.isfinite(head_z):
        if (
            head_z >= very_low_height_thres
            and aspect < 1.10
            and not torso_horizontal
            and kpt_ratio < 1.25
        ):
            crouch_bow_like = True
            upright_score += 2
            upright_reasons.append("crouch_bow_guard")

    # ------------------------------
    # Final decision
    # ------------------------------
    posture = "NORMAL"
    lying_candidate = False

    adjusted_score = lying_score - upright_score

    strong_low_z_and_shape = (
        head_z is not None
        and np.isfinite(head_z)
        and head_z < lying_height_thres
        and (torso_horizontal or aspect > 1.20 or kpt_ratio > 1.25)
    )

    very_low_z_with_floor = (
        head_z is not None
        and np.isfinite(head_z)
        and head_z < very_low_height_thres
        and floor_area
        and not crouch_bow_like
    )

    shape_only_lying = (
        head_z is None
        and (
            aspect > 1.85
            or (aspect > 1.35 and torso_horizontal)
            or (torso_horizontal and kpt_ratio > 1.40)
        )
    )

    if crouch_bow_like:
        posture = "CROUCH_BOW"
        lying_candidate = False

    elif (lying_score >= 5 and adjusted_score >= 2) or strong_low_z_and_shape or very_low_z_with_floor or shape_only_lying:
        posture = "LYING_CANDIDATE"
        lying_candidate = True

    elif head_z is not None and head_z < user_height_m * 0.62:
        posture = "LOW_POSTURE"
        lying_candidate = False

    else:
        posture = "NORMAL"
        lying_candidate = False

    debug = {
        "posture": posture,
        "lying_score": lying_score,
        "upright_score": upright_score,
        "adjusted_score": adjusted_score,
        "reasons": reasons,
        "upright_reasons": upright_reasons,
        "aspect": aspect,
        "head_z": None if head_z is None else float(head_z),
        "z_residual": None if z_residual is None else float(z_residual),
        "torso_horizontal": torso_horizontal,
        "kpt_ratio": kpt_ratio,
        "visible_kpts": visible_kpts,
        "standing_order": standing_order,
        "floor_area": floor_area,
        "crouch_bow_like": crouch_bow_like,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }

    return lying_candidate, posture, debug


# ==============================
# Tracking
# ==============================
class TrackManager:
    def __init__(self, max_distance_px=90, max_age_sec=2.0, emergency_sec=10.0):
        self.max_distance_px = max_distance_px
        self.max_age_sec = max_age_sec
        self.emergency_sec = emergency_sec
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, now):
        assigned = set()

        for det in detections:
            p = det["image_point"]

            best_id = None
            best_dist = 1e9

            for tid, tr in self.tracks.items():
                if tid in assigned:
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
                    "fall_start": None,
                    "emergency": False,
                }

            tr = self.tracks[tid]
            tr["last_seen"] = now
            tr["image_point"] = p
            tr["map_xy"] = det.get("map_xy")
            tr["helmet_status"] = det.get("helmet_status", "UNKNOWN")
            tr["posture"] = det.get("posture", "NORMAL")
            tr["lying_candidate"] = bool(det.get("lying_candidate", False))
            tr["debug"] = det.get("debug", {})
            tr["person_box"] = det.get("person_box")
            tr["source"] = det.get("source", "det")
            tr["conf"] = det.get("conf", 0.0)

            if tr["lying_candidate"]:
                if tr["fall_start"] is None:
                    tr["fall_start"] = now

                tr["fall_elapsed"] = now - tr["fall_start"]

                if tr["fall_elapsed"] >= self.emergency_sec:
                    tr["emergency"] = True
            else:
                tr["fall_start"] = None
                tr["fall_elapsed"] = 0.0
                tr["emergency"] = False

            det["track_id"] = tid
            det["fall_elapsed"] = tr.get("fall_elapsed", 0.0)
            det["emergency"] = tr["emergency"]

            assigned.add(tid)

        dead = []
        for tid, tr in self.tracks.items():
            if now - tr["last_seen"] > self.max_age_sec:
                dead.append(tid)

        for tid in dead:
            del self.tracks[tid]

        return detections

    def reset(self):
        self.next_id = 1
        self.tracks = {}


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
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.58, thickness=2):
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


def color_for_state(helmet_status, posture, emergency):
    if emergency:
        return (255, 0, 255)

    if posture == "LYING_CANDIDATE":
        return (0, 165, 255)

    if helmet_status == "NO_HELMET":
        return (0, 0, 255)

    if helmet_status == "HELMET_ON":
        return (0, 255, 0)

    return (0, 165, 255)


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

        color = color_for_state(
            det.get("helmet_status", "UNKNOWN"),
            det.get("posture", "NORMAL"),
            det.get("emergency", False),
        )

        radius = 12 if det.get("emergency", False) else 7

        cv2.circle(out, (px, py), radius, color, -1)
        cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 2)

        label = f"ID{det.get('track_id', '?')}"
        if det.get("emergency", False):
            label += " EMG"
        elif det.get("posture") == "LYING_CANDIDATE":
            label += " LYING"
        elif det.get("helmet_status") == "NO_HELMET":
            label += " NO_HELMET"

        draw_text(out, label, (px + 10, py - 10), color, 0.45, 1)

    return out


def make_dashboard_panel(
    width,
    height,
    person_count,
    helmet_count,
    no_helmet_count,
    unknown_helmet_count,
    low_posture_count,
    lying_count,
    emergency_count,
    fps,
    cam_name,
    state_json_path,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    y = 35
    draw_text(panel, f"SAFETY DASHBOARD V09 - {cam_name}", (20, y), (255, 255, 255), 0.72, 2)

    y += 45
    draw_text(panel, f"FPS: {fps:.1f}", (20, y), (200, 200, 200), 0.62, 2)

    y += 38
    draw_text(panel, f"Persons: {person_count}", (20, y), (255, 255, 255), 0.68, 2)

    y += 38
    draw_text(panel, f"Helmets: {helmet_count}", (20, y), (255, 255, 255), 0.68, 2)

    y += 38
    color = (0, 0, 255) if no_helmet_count > 0 else (0, 255, 0)
    draw_text(panel, f"No Helmet: {no_helmet_count}", (20, y), color, 0.68, 2)

    y += 38
    draw_text(panel, f"Helmet Unknown: {unknown_helmet_count}", (20, y), (0, 165, 255), 0.62, 2)

    y += 38
    draw_text(panel, f"Low posture: {low_posture_count}", (20, y), (0, 165, 255), 0.62, 2)

    y += 38
    color = (0, 165, 255) if lying_count > 0 else (0, 255, 0)
    draw_text(panel, f"Lying Candidate: {lying_count}", (20, y), color, 0.68, 2)

    y += 38
    color = (0, 0, 255) if emergency_count > 0 else (0, 255, 0)
    draw_text(panel, f"EMERGENCY: {emergency_count}", (20, y), color, 0.78, 2)

    y += 55
    draw_text(panel, "V09 Lying Logic", (20, y), (255, 255, 255), 0.62, 2)

    y += 28
    draw_text(panel, "low head_z alone != lying", (20, y), (180, 180, 180), 0.48, 1)

    y += 24
    draw_text(panel, "needs: low_z + horizontal/wide", (20, y), (180, 180, 180), 0.48, 1)

    y += 24
    draw_text(panel, "crouch/kneel/bow guard ON", (20, y), (180, 180, 180), 0.48, 1)

    y = height - 88
    draw_text(panel, "q: quit | r: reset tracks", (20, y), (255, 255, 255), 0.58, 2)

    y += 28
    draw_text(panel, "map color:", (20, y), (255, 255, 255), 0.50, 1)

    y += 24
    draw_text(panel, "green=safe red=no helmet orange=lying magenta=emergency", (20, y), (180, 180, 180), 0.42, 1)

    y += 24
    draw_text(panel, f"state: {state_json_path}", (20, y), (160, 160, 160), 0.38, 1)

    return panel


def save_state_json(path, timestamp, cam_name, detections):
    persons = []
    emergency_targets = []

    for det in detections:
        map_xy = det.get("map_xy")
        debug = det.get("debug", {})

        item = {
            "track_id": det.get("track_id"),
            "helmet_status": det.get("helmet_status"),
            "posture": det.get("posture"),
            "lying_candidate": bool(det.get("lying_candidate", False)),
            "emergency": bool(det.get("emergency", False)),
            "fall_elapsed": float(det.get("fall_elapsed", 0.0)),
            "source": det.get("source"),
            "conf": float(det.get("conf", 0.0)),
            "head_z": debug.get("head_z"),
            "z_residual": debug.get("z_residual"),
            "lying_score": debug.get("lying_score"),
            "upright_score": debug.get("upright_score"),
            "reasons": debug.get("reasons", []),
            "upright_reasons": debug.get("upright_reasons", []),
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
        default=str(Path.home() / "turtlebot4_ws/final_project/yolo_experiments/v8n_640_e50/weights/best.pt"),
    )
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument(
        "--map-yaml",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/final_project.yaml"),
    )
    parser.add_argument(
        "--homography",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/calibration/cam0_to_map.npz"),
    )
    parser.add_argument(
        "--z-calib",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/calibration/cam0_z_calib.npz"),
    )
    parser.add_argument(
        "--homography-output",
        type=str,
        default="map_meters",
        choices=["map_meters", "map_pixels"],
    )

    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--person-conf", type=float, default=0.45)
    parser.add_argument("--helmet-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    parser.add_argument("--head-radius-scale", type=float, default=0.20)
    parser.add_argument("--helmet-expand", type=float, default=1.00)

    parser.add_argument("--user-height-m", type=float, default=1.80)
    parser.add_argument("--lying-height-thres", type=float, default=0.70)
    parser.add_argument("--very-low-height-thres", type=float, default=0.48)
    parser.add_argument("--residual-thres", type=float, default=140.0)
    parser.add_argument("--emergency-sec", type=float, default=10.0)

    parser.add_argument("--track-max-distance", type=float, default=90.0)
    parser.add_argument("--track-max-age", type=float, default=2.0)

    parser.add_argument(
        "--state-json",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/outputs/safety_state_v09.json"),
    )

    parser.add_argument("--display-w", type=int, default=1280)
    parser.add_argument("--display-h", type=int, default=720)

    args = parser.parse_args()

    print("======================================")
    print("Safety Dashboard V09: Detection + Pose + Z Height")
    print("--------------------------------------")
    print(f"camera      : {args.camera}")
    print(f"cam_name    : {args.cam_name}")
    print(f"det_model   : {args.det_model}")
    print(f"pose_model  : {args.pose_model}")
    print(f"device      : {args.device}")
    print(f"map_yaml    : {args.map_yaml}")
    print(f"homography  : {args.homography}")
    print(f"z_calib     : {args.z_calib}")
    print(f"user_height : {args.user_height_m}")
    print(f"person_conf : {args.person_conf}")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()
    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    map_info = load_map_from_yaml(args.map_yaml)
    H, H_key = load_homography(args.homography)
    P = load_z_calibration(args.z_calib)

    print(f"[INFO] map image: {map_info['image']}")
    print(f"[INFO] H key: {H_key}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(args.pose_model)
    det_names = det_model.names

    print("[INFO] detection names:", det_names)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {args.camera}")

    tracker = TrackManager(
        max_distance_px=args.track_max_distance,
        max_age_sec=args.track_max_age,
        emergency_sec=args.emergency_sec,
    )

    window_name = "Safety Dashboard V09"
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

        det_result = det_model(
            frame,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
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

        pose_items = extract_pose_items(pose_result)

        persons = add_pose_only_persons(
            persons,
            pose_items,
            image_area=image_area,
            min_pose_area_ratio=args.person_min_area_ratio,
        )

        preview = frame.copy()

        for helmet in helmets:
            draw_box(preview, helmet["box"], (0, 255, 255), f"helmet {helmet['conf']:.2f}", 2)

        detections_for_tracks = []

        for person in persons:
            pbox = person["box"]
            pose_item, pose_iou = match_pose_to_person(pbox, pose_items)

            nearest_helmet, _ = find_nearest_related_helmet(pbox, helmets, image_w, image_h)
            nearest_helmet_box = nearest_helmet["box"] if nearest_helmet is not None else None

            head_px, head_src = get_head_point(
                pose_item,
                person_box=pbox,
                helmet_box=nearest_helmet_box,
                conf_thres=args.pose_conf,
            )

            foot_px, foot_src = get_foot_point(
                pose_item,
                pbox,
                conf_thres=args.pose_conf,
            )

            foot_map = camera_pixel_to_map_meter(
                H,
                foot_px,
                map_info,
                args.homography_output,
            )

            head_z = None
            z_residual = None
            projected_uv = None

            if head_px is not None and foot_map is not None:
                head_z, z_residual, projected_uv = estimate_z_on_vertical_line(
                    P,
                    foot_map[0],
                    foot_map[1],
                    head_px,
                )

            helmet_status, head_radius, related_helmets = judge_helmet_status(
                pbox,
                helmets,
                head_px,
                head_src,
                image_w,
                image_h,
                head_radius_scale=args.head_radius_scale,
                helmet_expand_ratio=args.helmet_expand,
            )

            lying_candidate, posture, debug = judge_posture_v09(
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

            det_item = {
                "image_point": foot_px,
                "map_xy": foot_map,
                "helmet_status": helmet_status,
                "posture": posture,
                "lying_candidate": lying_candidate,
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

            detections_for_tracks.append(det_item)

        tracked = tracker.update(detections_for_tracks, now)

        # ==============================
        # Draw camera preview
        # ==============================
        for det in tracked:
            pbox = det["person_box"]
            tid = det["track_id"]
            helmet_status = det["helmet_status"]
            posture = det["posture"]
            emergency = det["emergency"]
            debug = det["debug"]

            color = color_for_state(helmet_status, posture, emergency)

            label = f"ID{tid} {helmet_status} {posture}"
            if emergency:
                label += " EMERGENCY"
            elif det["lying_candidate"]:
                label += f" {det['fall_elapsed']:.1f}/{args.emergency_sec:.0f}s"

            head_z = debug.get("head_z")
            z_residual = debug.get("z_residual")
            if head_z is not None:
                label += f" z={head_z:.2f}"
            if z_residual is not None:
                label += f" r={z_residual:.0f}"

            draw_box(preview, pbox, color, label, 2)

            head_px = det.get("head_px")
            head_radius = det.get("head_radius", 40)

            if head_px is not None:
                hp = tuple(head_px.astype(int))
                cv2.circle(preview, hp, int(head_radius), (255, 0, 255), 2)
                cv2.circle(preview, hp, 4, (255, 0, 255), -1)
                draw_text(preview, f"head:{det.get('head_src')}", (hp[0] + 8, hp[1] - 8), (255, 0, 255), 0.45, 1)

            foot_px = det.get("foot_px")
            if foot_px is not None:
                fp = tuple(foot_px.astype(int))
                cv2.circle(preview, fp, 6, (255, 255, 0), -1)
                draw_text(preview, f"foot:{det.get('foot_src')}", (fp[0] + 8, fp[1] + 16), (255, 255, 0), 0.45, 1)

            projected_uv = det.get("projected_uv")
            if projected_uv is not None and np.all(np.isfinite(projected_uv)):
                pp = tuple(projected_uv.astype(int))
                cv2.circle(preview, pp, 5, (0, 165, 255), -1)
                if head_px is not None:
                    cv2.line(preview, tuple(head_px.astype(int)), pp, (0, 165, 255), 2)

            torso_horizontal, shoulder_mid, hip_mid = get_torso_horizontal(
                match_pose_to_person(pbox, pose_items)[0],
                args.pose_conf,
            )
            if shoulder_mid is not None and hip_mid is not None:
                cv2.line(preview, tuple(shoulder_mid.astype(int)), tuple(hip_mid.astype(int)), (255, 0, 0), 2)

            x1, y1, x2, y2 = map(int, pbox)
            reason_text = ",".join(debug.get("reasons", [])[:3])
            guard_text = ",".join(debug.get("upright_reasons", [])[:2])
            draw_text(
                preview,
                f"score L/U={debug.get('lying_score')}/{debug.get('upright_score')} ar={debug.get('aspect'):.2f} kpt={debug.get('kpt_ratio'):.2f}",
                (x1, min(image_h - 10, y2 + 22)),
                color,
                0.44,
                1,
            )
            if reason_text:
                draw_text(
                    preview,
                    f"L: {reason_text}",
                    (x1, min(image_h - 10, y2 + 42)),
                    color,
                    0.42,
                    1,
                )
            if guard_text:
                draw_text(
                    preview,
                    f"U: {guard_text}",
                    (x1, min(image_h - 10, y2 + 62)),
                    (0, 255, 255),
                    0.42,
                    1,
                )

            map_xy = det.get("map_xy")
            if map_xy is not None:
                draw_text(
                    preview,
                    f"map=({map_xy[0]:.2f},{map_xy[1]:.2f})",
                    (x1, min(image_h - 10, y2 + 82)),
                    color,
                    0.42,
                    1,
                )

        # ==============================
        # Counts
        # ==============================
        person_count = len(tracked)
        helmet_count = len(helmets)
        no_helmet_count = sum(1 for d in tracked if d["helmet_status"] == "NO_HELMET")
        unknown_helmet_count = sum(1 for d in tracked if d["helmet_status"] == "UNKNOWN")
        low_posture_count = sum(1 for d in tracked if d["posture"] in ["LOW_POSTURE", "CROUCH_BOW"])
        lying_count = sum(1 for d in tracked if d["posture"] == "LYING_CANDIDATE")
        emergency_count = sum(1 for d in tracked if d["emergency"])

        save_state_json(
            args.state_json,
            timestamp=now,
            cam_name=args.cam_name,
            detections=tracked,
        )

        map_view = draw_map_points(map_info["img"], tracked, map_info)

        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = fps * 0.90 + (1.0 / dt) * 0.10

        panel_w = 380
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
            low_posture_count=low_posture_count,
            lying_count=lying_count,
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
            tracker.reset()
            print("[RESET] tracks reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
