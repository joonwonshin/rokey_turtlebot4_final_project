import argparse
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
# Geometry Utils
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


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def point_to_box_distance(point, box):
    x, y = point
    x1, y1, x2, y2 = box

    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)

    return float(np.sqrt(dx * dx + dy * dy))


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
# Detection Parsing
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
    """
    Detection model classes:
    - helmet
    - person

    Return:
    persons: [{"box": [x1,y1,x2,y2], "conf": float, "source": "det"}]
    helmets: [{"box": [x1,y1,x2,y2], "conf": float}]
    """
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

            # 너무 작은 person 박스는 TurtleBot/배경 오탐일 가능성이 높음
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
# Pose Parsing
# ==============================
def extract_pose_items(pose_result):
    """
    Return list of:
    {
      "box": [x1,y1,x2,y2],
      "conf": float,
      "kpts_xy": np.ndarray shape (17, 2),
      "kpts_conf": np.ndarray shape (17,) or None
    }
    """
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

    # 안 잡힌 keypoint가 0,0 근처로 오는 경우 제거
    if x <= 1 and y <= 1:
        return None

    if kpts_conf is not None:
        if kpts_conf[idx] < conf_thres:
            return None

    return np.array([x, y], dtype=np.float32)


def get_visible_points(pose_item, indices, conf_thres=0.25):
    kpts_xy = pose_item["kpts_xy"]
    kpts_conf = pose_item["kpts_conf"]

    pts = []
    for idx in indices:
        p = get_visible_point(kpts_xy, kpts_conf, idx, conf_thres)
        if p is not None:
            pts.append(p)

    return pts


def get_midpoint_from_pair(pose_item, idx_a, idx_b, conf_thres=0.25):
    pts = get_visible_points(pose_item, [idx_a, idx_b], conf_thres)
    if len(pts) == 2:
        return (pts[0] + pts[1]) / 2.0
    return None


def get_head_center_and_source(pose_item, conf_thres=0.25):
    """
    Return:
    head_center: np.array([x, y]) or None
    source: "face_keypoints" / "shoulder_inferred" / None

    1순위: nose/eye/ear 평균
    2순위: shoulder와 hip을 이용해서 머리 방향 추정
    """
    face_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    face_pts = get_visible_points(pose_item, face_indices, conf_thres)

    if len(face_pts) >= 1:
        return np.mean(np.stack(face_pts, axis=0), axis=0), "face_keypoints"

    left_shoulder = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_SHOULDER, conf_thres)
    right_shoulder = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_SHOULDER, conf_thres)
    left_hip = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_HIP, conf_thres)
    right_hip = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_HIP, conf_thres)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        if left_hip is not None and right_hip is not None:
            hip_mid = (left_hip + right_hip) / 2.0
            torso_vec = hip_mid - shoulder_mid
            torso_len = float(np.linalg.norm(torso_vec))

            if torso_len > 5:
                # 어깨에서 엉덩이 방향의 반대쪽이 머리 방향
                unit = torso_vec / torso_len
                inferred_head = shoulder_mid - unit * torso_len * 0.40
                return inferred_head, "shoulder_inferred"

    return None, None


def get_torso_horizontal(pose_item, conf_thres=0.25):
    """
    어깨 중심 - 엉덩이 중심 벡터가 수평에 가까우면 누운 자세 후보.
    """
    left_shoulder = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_SHOULDER, conf_thres)
    right_shoulder = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_SHOULDER, conf_thres)
    left_hip = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], LEFT_HIP, conf_thres)
    right_hip = get_visible_point(pose_item["kpts_xy"], pose_item["kpts_conf"], RIGHT_HIP, conf_thres)

    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return False, None, None

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    # abs(dx)가 abs(dy)보다 크면 수평에 가까움
    horizontal = abs(dx) > abs(dy) * 1.05

    return horizontal, shoulder_mid, hip_mid


def get_keypoint_spread_ratio(pose_item, conf_thres=0.25):
    """
    보이는 keypoint들의 전체 분포가 가로로 긴지 확인.
    Return:
    ratio, visible_count
    """
    pts = get_visible_points(
        pose_item,
        list(range(17)),
        conf_thres,
    )

    if len(pts) < 4:
        return 0.0, len(pts)

    arr = np.stack(pts, axis=0)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)

    kw = max(1.0, x_max - x_min)
    kh = max(1.0, y_max - y_min)

    return kw / kh, len(pts)


def match_pose_to_person(person_box, pose_items):
    """
    Detection person box와 pose person box를 매칭.
    1순위: IoU
    2순위: 중심점 포함
    """
    best_iou = 0.0
    best_item = None

    for item in pose_items:
        iou = box_iou(person_box, item["box"])
        if iou > best_iou:
            best_iou = iou
            best_item = item

    if best_iou >= 0.10:
        return best_item, best_iou

    # fallback: 중심점 포함
    person_c = box_center(person_box)

    for item in pose_items:
        pose_box = item["box"]
        pose_c = box_center(pose_box)

        if point_in_box(pose_c, person_box) or point_in_box(person_c, pose_box):
            return item, box_iou(person_box, pose_box)

    return None, 0.0


def add_pose_only_persons(persons, pose_items, image_area, min_pose_area_ratio=0.002):
    """
    detection 모델이 누운 사람을 person으로 못 잡아도,
    pose 모델이 사람을 잡으면 fall 판단용 person으로 추가한다.

    단, pose-only person은 helmet 착용 판단은 약하게만 사용한다.
    """
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
    head_radius_scale=0.18,
    helmet_expand_ratio=0.85,
):
    """
    Return:
    HELMET_ON / NO_HELMET / UNKNOWN

    핵심:
    - 머리 keypoint가 있으면 head circle과 helmet box의 거리로 판단
    - 머리 keypoint가 없으면 person 상단으로 약하게 판단
    - 머리 자체가 불확실하면 NO_HELMET으로 확정하지 않고 UNKNOWN
    """
    px1, py1, px2, py2 = person_box
    person_w = px2 - px1
    person_h = py2 - py1
    person_size = np.sqrt(max(1.0, person_w * person_h))

    related_helmets = []

    # person box를 조금 확장해서 근처 helmet 후보 포함
    expanded_person = expand_box(person_box, 0.08, image_w, image_h)

    for h in helmets:
        hb = h["box"]
        hc = box_center(hb)

        if (
            point_in_box(hc, expanded_person)
            or box_iou(hb, expanded_person) > 0.02
            or point_to_box_distance(hc, expanded_person) < 30
        ):
            related_helmets.append(h)

    # pose로 머리 위치를 아는 경우
    if head_center is not None:
        # 너무 작게 잡으면 helmet_on이 잘 안 뜸.
        # 너무 크게 잡으면 근처 헬멧을 착용으로 오인함.
        head_radius = int(np.clip(person_size * head_radius_scale, 35, 95))

        for h in related_helmets:
            hb = h["box"]
            expanded_helmet = expand_box(hb, helmet_expand_ratio, image_w, image_h)

            # 1. head center가 확장 helmet box 안에 들어오는 경우
            if point_in_box(head_center, expanded_helmet):
                return "HELMET_ON", head_radius, related_helmets

            # 2. head circle이 helmet box와 가까운 경우
            dist = point_to_box_distance(head_center, expanded_helmet)
            if dist <= head_radius:
                return "HELMET_ON", head_radius, related_helmets

        # face keypoint 또는 shoulder 기반 추정이 있는데 helmet이 없으면 미착용으로 판단 가능
        if head_source in ["face_keypoints", "shoulder_inferred"]:
            return "NO_HELMET", head_radius, related_helmets

        return "UNKNOWN", head_radius, related_helmets

    # head_center가 없는 경우: 서 있는 사람만 상단 영역으로 약하게 추정
    aspect = person_w / person_h if person_h > 1 else 0

    # 서 있는 사람처럼 세로가 길 때만 상단 40%를 머리 후보로 사용
    if aspect < 0.95:
        head_zone = [px1, py1, px2, py1 + person_h * 0.40]

        for h in related_helmets:
            hc = box_center(h["box"])
            if point_in_box(hc, head_zone):
                return "HELMET_ON", int(max(35, person_size * head_radius_scale)), related_helmets

    return "UNKNOWN", int(max(35, person_size * head_radius_scale)), related_helmets


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
    """
    쓰러짐/누움 후보 판단.

    점수 기반:
    - bbox가 가로로 길다
    - pose torso가 수평이다
    - keypoint 분포가 가로로 길다
    - 바닥 영역에 있다
    """
    x1, y1, x2, y2 = person_box

    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph

    center_y = (y1 + y2) / 2.0
    bottom_y = y2

    score = 0
    reasons = []

    # 1. bbox 비율
    if aspect > 1.15:
        score += 1
        reasons.append("wide_box")

    if aspect > 1.45:
        score += 1
        reasons.append("very_wide_box")

    if aspect > 1.85:
        score += 1
        reasons.append("extreme_wide_box")

    # 2. pose torso 방향
    torso_horizontal = False
    shoulder_mid = None
    hip_mid = None

    if pose_item is not None:
        torso_horizontal, shoulder_mid, hip_mid = get_torso_horizontal(pose_item, pose_conf)

        if torso_horizontal:
            score += 2
            reasons.append("horizontal_torso")

        # 3. keypoint 전체 분포
        kpt_ratio, visible_count = get_keypoint_spread_ratio(pose_item, pose_conf)

        if visible_count >= 5 and kpt_ratio > 1.20:
            score += 1
            reasons.append("wide_keypoints")

        if visible_count >= 5 and kpt_ratio > 1.60:
            score += 1
            reasons.append("very_wide_keypoints")
    else:
        kpt_ratio = 0.0
        visible_count = 0

    # 4. 바닥 영역 보조 조건
    floor_area = center_y > image_h * floor_y_ratio or bottom_y > image_h * 0.62

    if floor_area and aspect > 1.10:
        score += 1
        reasons.append("floor_area")

    # 5. 서 있는 사람으로 보이면 감점
    # 세로로 긴 bbox + torso가 수직이면 fall 가능성 낮음
    if aspect < 0.75 and pose_item is not None and not torso_horizontal:
        score -= 1
        reasons.append("upright_penalty")

    # 최종 후보 판단
    candidate = False

    if score >= fall_score_thres:
        candidate = True

    # pose가 없어도 bbox가 매우 가로로 길면 후보
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
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_small_text(img, text, pos, color=(255, 255, 255), scale=0.55, thickness=1):
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


def draw_panel(img, lines, x=20, y=20, w=720, line_h=30):
    h = 24 + line_h * len(lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, img, 0.42, 0, img)

    cy = y + 32
    for text, color in lines:
        cv2.putText(
            img,
            text,
            (x + 15, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        cy += line_h


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument(
        "--det-model",
        type=str,
        default=str(
            Path.home()
            / "turtlebot4_ws/final_project/yolo_experiments/v8n_640_e50/weights/best.pt"
        ),
    )
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")

    parser.add_argument("--device", type=str, default="cpu", help="cpu or 0")
    parser.add_argument("--imgsz", type=int, default=640)

    # raw YOLO confidence. 이 값보다 낮은 박스는 YOLO에서 애초에 버림.
    # helmet을 놓치지 않으려면 det-conf는 낮게 두고, class별 후처리 conf로 거르는 게 좋음.
    parser.add_argument("--det-conf", type=float, default=0.20)

    # class별 후처리 threshold
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--helmet-conf", type=float, default=0.25)

    parser.add_argument("--pose-conf", type=float, default=0.25)

    # TurtleBot/배경 오탐 감소용
    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    # helmet-head matching
    parser.add_argument("--head-radius-scale", type=float, default=0.18)
    parser.add_argument("--helmet-expand", type=float, default=0.85)

    # fall / emergency
    parser.add_argument("--emergency-sec", type=float, default=10.0)
    parser.add_argument("--fall-score-thres", type=int, default=3)
    parser.add_argument("--floor-y-ratio", type=float, default=0.45)

    args = parser.parse_args()

    print("======================================")
    print("Detection + Pose + Helmet + Fall Demo")
    print("--------------------------------------")
    print(f"camera        : {args.camera}")
    print(f"det_model     : {args.det_model}")
    print(f"pose_model    : {args.pose_model}")
    print(f"device        : {args.device}")
    print(f"det_conf      : {args.det_conf}")
    print(f"person_conf   : {args.person_conf}")
    print(f"helmet_conf   : {args.helmet_conf}")
    print(f"emergency_sec : {args.emergency_sec}")
    print("======================================")

    det_model_path = Path(args.det_model)
    if not det_model_path.exists():
        print(f"[ERROR] detection model not found: {det_model_path}")
        print("먼저 v8n_640_e50 학습 결과 best.pt 경로를 확인해.")
        return

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(args.pose_model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")

    det_names = det_model.names
    print("[INFO] detection class names:", det_names)

    window_name = "Safety Monitor: Detection + Pose + Fall"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fall_start_time = None
    fall_confirmed = False
    fall_elapsed = 0.0

    prev_time = time.time()
    fps = 0.0

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame read failed")
            continue

        frame_index += 1
        image_h, image_w = frame.shape[:2]
        image_area = image_w * image_h

        now = time.time()

        # ==============================
        # 1. Detection
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
        # 2. Pose
        # ==============================
        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_items = extract_pose_items(pose_result)

        # detection이 놓친 누운 사람도 pose가 잡았으면 fall 후보로 사용
        persons = add_pose_only_persons(
            persons,
            pose_items,
            image_area=image_area,
            min_pose_area_ratio=args.person_min_area_ratio,
        )

        show = frame.copy()

        # ==============================
        # 3. Draw helmet boxes
        # ==============================
        for helmet in helmets:
            draw_box(
                show,
                helmet["box"],
                (0, 255, 255),
                f"helmet {helmet['conf']:.2f}",
                thickness=2,
            )

        # ==============================
        # 4. Per-person logic
        # ==============================
        any_fall_candidate = False
        best_fall_debug = None

        for idx, person in enumerate(persons):
            pbox = person["box"]
            source = person["source"]

            px1, py1, px2, py2 = pbox
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            aspect = pw / ph

            pose_item, pose_iou = match_pose_to_person(pbox, pose_items)

            # ==============================
            # Head position
            # ==============================
            head_center = None
            head_source = None
            head_radius = 40
            related_helmets = []

            if pose_item is not None:
                head_center, head_source = get_head_center_and_source(
                    pose_item,
                    conf_thres=args.pose_conf,
                )

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

            # ==============================
            # Fall judgement
            # ==============================
            fall_candidate, fall_debug = judge_fall_candidate(
                pbox,
                pose_item,
                image_h=image_h,
                pose_conf=args.pose_conf,
                floor_y_ratio=args.floor_y_ratio,
                fall_score_thres=args.fall_score_thres,
            )

            if fall_candidate:
                any_fall_candidate = True
                best_fall_debug = fall_debug

            # ==============================
            # Drawing person
            # ==============================
            if fall_candidate:
                person_color = (0, 0, 255)
                fall_text = f"LYING_CAND score={fall_debug['score']} ar={fall_debug['aspect']:.2f}"
            else:
                person_color = (0, 255, 0)
                fall_text = f"NORMAL ar={aspect:.2f}"

            if source == "pose":
                person_color = (0, 120, 255)

            label = f"{source}_person {person['conf']:.2f} | {helmet_status} | {fall_text}"
            draw_box(show, pbox, person_color, label, thickness=2)

            # head circle
            if head_center is not None:
                hc = tuple(head_center.astype(int))
                cv2.circle(show, hc, int(head_radius), (255, 0, 255), 2)
                cv2.circle(show, hc, 5, (255, 0, 255), -1)

                draw_small_text(
                    show,
                    f"head zone ({head_source})",
                    (hc[0] + 8, hc[1] - 8),
                    color=(255, 0, 255),
                    scale=0.55,
                    thickness=2,
                )

            # torso line
            if pose_item is not None:
                torso_horizontal, shoulder_mid, hip_mid = get_torso_horizontal(
                    pose_item,
                    args.pose_conf,
                )

                if shoulder_mid is not None and hip_mid is not None:
                    cv2.line(
                        show,
                        tuple(shoulder_mid.astype(int)),
                        tuple(hip_mid.astype(int)),
                        (255, 0, 0),
                        3,
                    )

                    draw_small_text(
                        show,
                        "torso",
                        tuple(((shoulder_mid + hip_mid) / 2 + np.array([5, -5])).astype(int)),
                        color=(255, 0, 0),
                        scale=0.5,
                        thickness=1,
                    )

            # fall reasons
            if fall_candidate:
                reasons = ",".join(fall_debug["reasons"][:3])
                draw_small_text(
                    show,
                    f"fall reasons: {reasons}",
                    (int(px1), min(image_h - 10, int(py2) + 22)),
                    color=(0, 0, 255),
                    scale=0.55,
                    thickness=2,
                )

        # ==============================
        # 5. Emergency timer
        # ==============================
        if any_fall_candidate:
            if fall_start_time is None:
                fall_start_time = now

            fall_elapsed = now - fall_start_time

            if fall_elapsed >= args.emergency_sec:
                fall_confirmed = True
        else:
            fall_start_time = None
            fall_confirmed = False
            fall_elapsed = 0.0

        # FPS
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.90 * fps + 0.10 * (1.0 / dt)

        # ==============================
        # 6. Status panel
        # ==============================
        if fall_confirmed:
            emergency_text = "EMERGENCY"
            emergency_color = (0, 0, 255)
        elif any_fall_candidate:
            emergency_text = f"FALL CANDIDATE {fall_elapsed:.1f}/{args.emergency_sec:.1f}s"
            emergency_color = (0, 165, 255)
        else:
            emergency_text = "Monitoring"
            emergency_color = (0, 255, 255)

        lines = [
            (f"FPS: {fps:.1f} | Frame: {frame_index}", (255, 255, 255)),
            (f"Persons: {len(persons)} | Helmets: {len(helmets)} | Pose persons: {len(pose_items)}", (255, 255, 255)),
            (f"Helmet logic: head circle + helmet box distance", (255, 0, 255)),
            (f"Fall logic: bbox aspect + torso angle + keypoint spread + {args.emergency_sec:.0f}s duration", (255, 255, 255)),
            (f"Status: {emergency_text}", emergency_color),
        ]

        if best_fall_debug is not None:
            reason_text = ",".join(best_fall_debug["reasons"])
            lines.append(
                (
                    f"Fall debug: score={best_fall_debug['score']} ar={best_fall_debug['aspect']:.2f} kpt_ar={best_fall_debug['kpt_ratio']:.2f} [{reason_text}]",
                    (0, 165, 255),
                )
            )

        draw_panel(show, lines, x=20, y=20, w=950, line_h=30)

        # bottom guide
        draw_small_text(
            show,
            "q: quit | r: reset emergency timer",
            (20, image_h - 20),
            color=(255, 255, 255),
            scale=0.65,
            thickness=2,
        )

        cv2.imshow(window_name, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            fall_start_time = None
            fall_confirmed = False
            fall_elapsed = 0.0
            print("[RESET] emergency timer reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()