import argparse
import json
import math
import time
from pathlib import Path

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
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# ==============================
# Basic Utils
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


def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        draw_text(img, label, (x1, max(22, y1 - 8)), color, 0.55, 2)


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


def load_map_info(map_yaml_path):
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

    return {
        "yaml": str(map_yaml_path),
        "image": str(map_img_path),
        "resolution": float(info.get("resolution", 0.05)),
        "origin": info.get("origin", [0.0, 0.0, 0.0]),
        "width": gray.shape[1],
        "height": gray.shape[0],
    }


def load_homography(npz_path):
    npz_path = Path(npz_path).expanduser()

    if not npz_path.exists():
        raise FileNotFoundError(f"homography not found: {npz_path}")

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
        if k in data and data[k].shape == (3, 3):
            return data[k].astype(np.float64), k

    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.shape == (3, 3):
            return arr.astype(np.float64), k

    raise RuntimeError(f"No 3x3 homography matrix found. keys={data.files}")


def apply_homography(H, pixel_xy):
    u, v = float(pixel_xy[0]), float(pixel_xy[1])
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p

    if abs(q[2]) < 1e-9:
        return None

    q = q / q[2]
    return float(q[0]), float(q[1])


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
# Projection Matrix Calibration
# ==============================
def estimate_projection_matrix_dlt(world_points, image_points):
    """
    Estimate 3x4 projection matrix P.

    world_points: Nx3, [X,Y,Z]
    image_points: Nx2, [u,v]

    Requires at least 6 points.
    Better: 8~20 points.
    """
    world_points = np.asarray(world_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    if len(world_points) < 6:
        raise ValueError("Need at least 6 3D-2D points for DLT.")

    A = []

    for (X, Y, Z), (u, v) in zip(world_points, image_points):
        row1 = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        row2 = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]
        A.append(row1)
        A.append(row2)

    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # scale normalization
    norm = np.linalg.norm(P)
    if norm > 1e-12:
        P = P / norm

    return P


def project_points(P, world_points):
    world_points = np.asarray(world_points, dtype=np.float64)

    if world_points.ndim == 1:
        world_points = world_points.reshape(1, 3)

    ones = np.ones((world_points.shape[0], 1), dtype=np.float64)
    Xh = np.hstack([world_points, ones])

    x = (P @ Xh.T).T

    valid = np.abs(x[:, 2]) > 1e-9
    uv = np.zeros((world_points.shape[0], 2), dtype=np.float64)
    uv[:] = np.nan

    uv[valid, 0] = x[valid, 0] / x[valid, 2]
    uv[valid, 1] = x[valid, 1] / x[valid, 2]

    return uv


def reprojection_error(P, world_points, image_points):
    pred = project_points(P, world_points)
    image_points = np.asarray(image_points, dtype=np.float64)

    errors = np.linalg.norm(pred - image_points, axis=1)

    return {
        "mean": float(np.nanmean(errors)),
        "max": float(np.nanmax(errors)),
        "per_point": errors.tolist(),
    }


def estimate_z_on_vertical_line(P, map_x, map_y, image_uv):
    """
    Assume target point lies on vertical line:
    world = [map_x, map_y, Z]

    Given image pixel [u,v], solve best Z.

    Return:
    z_hat, reprojection_residual_px, projected_uv
    """
    u, v = float(image_uv[0]), float(image_uv[1])

    r1 = P[0]
    r2 = P[1]
    r3 = P[2]

    # row dot [X,Y,Z,1] = a + b*Z
    a1 = r1[0] * map_x + r1[1] * map_y + r1[3]
    b1 = r1[2]

    a2 = r2[0] * map_x + r2[1] * map_y + r2[3]
    b2 = r2[2]

    a3 = r3[0] * map_x + r3[1] * map_y + r3[3]
    b3 = r3[2]

    # u = (a1+b1Z)/(a3+b3Z)
    # v = (a2+b2Z)/(a3+b3Z)
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
# Detection / Pose
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

            persons.append({"box": xyxy, "conf": conf})

        elif cls_name == "helmet":
            if conf < helmet_conf_thres:
                continue

            helmets.append({"box": xyxy, "conf": conf})

    return persons, helmets


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

    return None, 0.0


def get_head_point(pose_item, person_box=None, conf_thres=0.25):
    """
    1순위: nose/eye/ear 평균
    2순위: shoulder/hip으로 머리 방향 추정
    3순위: bbox top center
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
                head = shoulder_mid - unit * torso_len * 0.40
                return head, "shoulder_inferred"

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
        return False

    ls = get_visible_point(pose_item, LEFT_SHOULDER, conf_thres)
    rs = get_visible_point(pose_item, RIGHT_SHOULDER, conf_thres)
    lh = get_visible_point(pose_item, LEFT_HIP, conf_thres)
    rh = get_visible_point(pose_item, RIGHT_HIP, conf_thres)

    if ls is None or rs is None or lh is None or rh is None:
        return False

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    return abs(dx) > abs(dy) * 1.05


# ==============================
# Camera
# ==============================
def open_camera(camera_id):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera {camera_id}")

    return cap


# ==============================
# Calibration Mode
# ==============================
def build_world_image_points(click_pairs, height_m):
    world_points = []
    image_points = []

    for pair in click_pairs:
        mx, my = pair["map_xy"]

        bottom_px = pair["bottom_pixel"]
        top_px = pair["top_pixel"]

        world_points.append([mx, my, 0.0])
        image_points.append(bottom_px)

        world_points.append([mx, my, height_m])
        image_points.append(top_px)

    return np.asarray(world_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)


def save_z_calibration(out_npz, out_json, click_pairs, height_m, P, error_info, args):
    out_npz = Path(out_npz).expanduser()
    out_json = Path(out_json).expanduser()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    world_points, image_points = build_world_image_points(click_pairs, height_m)

    np.savez(
        str(out_npz),
        P=P,
        world_points=world_points,
        image_points=image_points,
        height_m=np.array([height_m], dtype=np.float64),
        reproj_mean=np.array([error_info["mean"]], dtype=np.float64),
        reproj_max=np.array([error_info["max"]], dtype=np.float64),
    )

    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": args.camera,
        "cam_name": args.cam_name,
        "height_m": height_m,
        "map_yaml": args.map_yaml,
        "homography": args.homography,
        "homography_output": args.homography_output,
        "out_npz": str(out_npz),
        "reprojection_error": {
            "mean_px": error_info["mean"],
            "max_px": error_info["max"],
        },
        "pairs": click_pairs,
        "P": P.tolist(),
    }

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print("")
    print("======================================")
    print("[SAVED] Z calibration")
    print(f"npz  : {out_npz}")
    print(f"json : {out_json}")
    print(f"mean reprojection error: {error_info['mean']:.2f}px")
    print(f"max  reprojection error: {error_info['max']:.2f}px")
    print("======================================")


def run_calibrate(args):
    map_info = load_map_info(args.map_yaml)
    H, H_key = load_homography(args.homography)

    print("======================================")
    print("Z Height Calibration Mode")
    print("--------------------------------------")
    print(f"camera      : {args.camera}")
    print(f"height_m    : {args.height_m}")
    print(f"homography  : {args.homography}")
    print(f"H key       : {H_key}")
    print(f"H output    : {args.homography_output}")
    print("--------------------------------------")
    print("Click order:")
    print("  1st click: bottom point touching floor")
    print("  2nd click: top point at known height")
    print("")
    print("Keys:")
    print("  s: save calibration")
    print("  u: undo last pair")
    print("  r: reset current bottom click")
    print("  q: quit")
    print("======================================")

    cap = open_camera(args.camera)

    state = {
        "current_bottom": None,
        "click_pairs": [],
        "mouse_pos": None,
        # space 로 화면을 얼려 bottom/top 두 점을 천천히 정확히 찍는다.
        # 실시간 영상 위에서 정수리를 찍으면 사람이 미세하게 흔들려 오차가 커진다.
        "frozen": None,
    }

    window = "08 Z Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["mouse_pos"] = [x, y]

        if event == cv2.EVENT_LBUTTONDOWN:
            if state["current_bottom"] is None:
                state["current_bottom"] = [x, y]
                print(f"[BOTTOM] pixel=({x}, {y})")
            else:
                bottom = state["current_bottom"]
                top = [x, y]

                map_xy = camera_pixel_to_map_meter(
                    H,
                    bottom,
                    map_info,
                    args.homography_output,
                )

                if map_xy is None:
                    print("[WARN] homography failed for bottom point")
                    state["current_bottom"] = None
                    return

                pair = {
                    "bottom_pixel": [float(bottom[0]), float(bottom[1])],
                    "top_pixel": [float(top[0]), float(top[1])],
                    "map_xy": [float(map_xy[0]), float(map_xy[1])],
                }

                state["click_pairs"].append(pair)
                state["current_bottom"] = None

                print(
                    f"[PAIR {len(state['click_pairs'])}] "
                    f"bottom={pair['bottom_pixel']} top={pair['top_pixel']} "
                    f"map=({map_xy[0]:.3f}, {map_xy[1]:.3f})"
                )

                if len(state["click_pairs"]) >= 4:
                    world_points, image_points = build_world_image_points(
                        state["click_pairs"],
                        args.height_m,
                    )
                    P = estimate_projection_matrix_dlt(world_points, image_points)
                    err = reprojection_error(P, world_points, image_points)
                    print(f"  current reproj error: mean={err['mean']:.2f}px max={err['max']:.2f}px")

    cv2.setMouseCallback(window, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] frame read failed")
            continue

        # 얼린 상태면 방금 읽은 프레임은 버리고(V4L2 버퍼만 비움) 정지 화면을 쓴다
        if state["frozen"] is not None:
            frame = state["frozen"]

        show = frame.copy()

        # draw existing pairs
        for i, pair in enumerate(state["click_pairs"], start=1):
            b = tuple(map(int, pair["bottom_pixel"]))
            t = tuple(map(int, pair["top_pixel"]))

            cv2.circle(show, b, 6, (0, 255, 0), -1)
            cv2.circle(show, t, 6, (0, 0, 255), -1)
            cv2.line(show, b, t, (255, 0, 255), 2)
            draw_text(show, f"{i}", (b[0] + 8, b[1] - 8), (0, 255, 0), 0.6, 2)

        # current bottom
        if state["current_bottom"] is not None:
            b = tuple(map(int, state["current_bottom"]))
            cv2.circle(show, b, 8, (255, 255, 0), -1)
            draw_text(show, "bottom selected - click top", (b[0] + 10, b[1]), (255, 255, 0), 0.6, 2)

        # status panel
        overlay = show.copy()
        cv2.rectangle(overlay, (15, 15), (760, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, show, 0.45, 0, show)

        draw_text(show, f"Z Calibration | known height: {args.height_m:.2f}m", (30, 45), (255, 255, 255), 0.7, 2)
        draw_text(show, f"pairs: {len(state['click_pairs'])} / recommended 6~10", (30, 78), (255, 255, 255), 0.65, 2)
        draw_text(show, "click bottom(feet) -> top(head) | SPACE:freeze s:save u:undo r:reset q:quit", (30, 112), (255, 255, 255), 0.6, 2)

        if state["frozen"] is not None:
            cv2.rectangle(show, (0, 0), (show.shape[1] - 1, show.shape[0] - 1), (0, 200, 255), 6)
            draw_text(show, "FROZEN  (SPACE to resume)", (show.shape[1] - 430, 40), (0, 200, 255), 0.8, 2)

        if len(state["click_pairs"]) >= 4:
            world_points, image_points = build_world_image_points(state["click_pairs"], args.height_m)
            P = estimate_projection_matrix_dlt(world_points, image_points)
            err = reprojection_error(P, world_points, image_points)
            draw_text(show, f"reproj error mean={err['mean']:.2f}px max={err['max']:.2f}px", (30, 140), (0, 255, 255), 0.55, 2)

        cv2.imshow(window, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            if state["frozen"] is None:
                state["frozen"] = frame.copy()
                print("[FREEZE] 화면 정지 — 발밑/정수리를 천천히 클릭하세요")
            else:
                state["frozen"] = None
                print("[RESUME] 실시간 복귀")

        if key == ord("r"):
            state["current_bottom"] = None
            print("[RESET] current bottom cleared")

        if key == ord("u"):
            if state["click_pairs"]:
                removed = state["click_pairs"].pop()
                print("[UNDO]", removed)
            else:
                print("[UNDO] no pair to remove")

        if key == ord("s"):
            if len(state["click_pairs"]) < 4:
                print("[WARN] Need at least 4 pairs. Recommended 6~10 pairs.")
                continue

            world_points, image_points = build_world_image_points(state["click_pairs"], args.height_m)
            P = estimate_projection_matrix_dlt(world_points, image_points)
            err = reprojection_error(P, world_points, image_points)

            save_z_calibration(
                args.out,
                args.json_out,
                state["click_pairs"],
                args.height_m,
                P,
                err,
                args,
            )

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# Test Mode
# ==============================
def run_test(args):
    map_info = load_map_info(args.map_yaml)
    H, H_key = load_homography(args.homography)

    z_npz = Path(args.z_calib).expanduser()
    if not z_npz.exists():
        raise FileNotFoundError(f"z calibration npz not found: {z_npz}")

    z_data = np.load(str(z_npz))
    P = z_data["P"].astype(np.float64)

    print("======================================")
    print("Z Height Test Mode")
    print("--------------------------------------")
    print(f"camera      : {args.camera}")
    print(f"det_model   : {args.det_model}")
    print(f"pose_model  : {args.pose_model}")
    print(f"z_calib     : {args.z_calib}")
    print(f"homography  : {args.homography}")
    print(f"H key       : {H_key}")
    print(f"H output    : {args.homography_output}")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()
    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(args.pose_model)

    names = det_model.names
    print("[INFO] detection names:", names)

    cap = open_camera(args.camera)

    window = "08 Z Height Test"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

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
            names,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            helmet_conf_thres=args.helmet_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        pose_items = extract_pose_items(pose_result)

        show = frame.copy()

        for h in helmets:
            draw_box(show, h["box"], (0, 255, 255), f"helmet {h['conf']:.2f}", 2)

        for person in persons:
            pbox = person["box"]
            pose_item, pose_iou = match_pose_to_person(pbox, pose_items)

            head_px, head_src = get_head_point(pose_item, pbox, args.pose_conf)
            foot_px, foot_src = get_foot_point(pose_item, pbox, args.pose_conf)

            foot_map = camera_pixel_to_map_meter(
                H,
                foot_px,
                map_info,
                args.homography_output,
            )

            z_hat = None
            residual = None
            projected_uv = None

            if head_px is not None and foot_map is not None:
                z_hat, residual, projected_uv = estimate_z_on_vertical_line(
                    P,
                    foot_map[0],
                    foot_map[1],
                    head_px,
                )

            x1, y1, x2, y2 = pbox
            pw = max(1.0, x2 - x1)
            ph = max(1.0, y2 - y1)
            aspect = pw / ph
            torso_h = get_torso_horizontal(pose_item, args.pose_conf)

            # 임시 lying 판단: z값 확인용
            lying_candidate = False
            reasons = []

            if z_hat is not None:
                if z_hat < args.lying_height_thres:
                    lying_candidate = True
                    reasons.append("low_head_z")

                if residual is not None and residual > args.residual_thres and torso_h:
                    lying_candidate = True
                    reasons.append("high_residual_torso_horizontal")

            if aspect > 1.55:
                lying_candidate = True
                reasons.append("wide_bbox")

            color = (0, 0, 255) if lying_candidate else (0, 255, 0)

            label = f"person {person['conf']:.2f}"
            if z_hat is not None:
                label += f" | z={z_hat:.2f}m res={residual:.1f}px"
            else:
                label += " | z=N/A"

            if lying_candidate:
                label += " | LYING_CAND"

            draw_box(show, pbox, color, label, 2)

            if head_px is not None:
                hp = tuple(head_px.astype(int))
                cv2.circle(show, hp, 7, (255, 0, 255), -1)
                draw_text(show, f"head:{head_src}", (hp[0] + 8, hp[1] - 8), (255, 0, 255), 0.5, 1)

            if foot_px is not None:
                fp = tuple(foot_px.astype(int))
                cv2.circle(show, fp, 7, (255, 255, 0), -1)
                draw_text(show, f"foot:{foot_src}", (fp[0] + 8, fp[1] + 16), (255, 255, 0), 0.5, 1)

            if projected_uv is not None and np.all(np.isfinite(projected_uv)):
                pp = tuple(projected_uv.astype(int))
                cv2.circle(show, pp, 6, (0, 165, 255), -1)
                cv2.line(show, tuple(head_px.astype(int)), pp, (0, 165, 255), 2)
                draw_text(show, "reproj", (pp[0] + 8, pp[1] - 8), (0, 165, 255), 0.5, 1)

            if foot_map is not None:
                draw_text(
                    show,
                    f"map=({foot_map[0]:.2f},{foot_map[1]:.2f}) aspect={aspect:.2f} torso_h={torso_h}",
                    (int(x1), min(image_h - 10, int(y2) + 22)),
                    color,
                    0.48,
                    1,
                )

                if reasons:
                    draw_text(
                        show,
                        "reasons: " + ",".join(reasons),
                        (int(x1), min(image_h - 10, int(y2) + 44)),
                        color,
                        0.48,
                        1,
                    )

        dt = now - prev_time
        prev_time = now

        if dt > 0:
            fps = fps * 0.90 + (1.0 / dt) * 0.10

        overlay = show.copy()
        cv2.rectangle(overlay, (15, 15), (900, 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, show, 0.45, 0, show)

        draw_text(show, f"08 Z Height Test | FPS {fps:.1f}", (30, 45), (255, 255, 255), 0.75, 2)
        draw_text(show, f"green=normal candidate, red=lying candidate", (30, 78), (255, 255, 255), 0.62, 2)
        draw_text(show, f"threshold: lying_height<{args.lying_height_thres:.2f}m residual>{args.residual_thres:.0f}px", (30, 108), (255, 255, 255), 0.60, 2)

        cv2.imshow(window, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["calibrate", "test"])

    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--cam-name", type=str, default="cam0")

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
        "--homography-output",
        type=str,
        default="map_meters",
        choices=["map_meters", "map_pixels"],
    )

    # calibration
    parser.add_argument("--height-m", type=float, default=0.40)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/calibration/cam0_z_calib.npz"),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/calibration/cam0_z_points.json"),
    )

    # test
    parser.add_argument(
        "--z-calib",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/calibration/cam0_z_calib.npz"),
    )
    parser.add_argument(
        "--det-model",
        type=str,
        default=str(Path.home() / "turtlebot4_ws/final_project/yolo_experiments/v8n_640_e50/weights/best.pt"),
    )
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--helmet-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    parser.add_argument("--lying-height-thres", type=float, default=0.70)
    parser.add_argument("--residual-thres", type=float, default=120.0)

    args = parser.parse_args()

    if args.mode == "calibrate":
        run_calibrate(args)
    elif args.mode == "test":
        run_test(args)


if __name__ == "__main__":
    main()
