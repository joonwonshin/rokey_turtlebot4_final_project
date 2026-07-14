"""
safety_lib.base_utils
=====================

기하학/좌표 변환/포즈 유틸 라이브러리. 순수 함수 위주.

구성:
  * COCO 17 Pose Keypoint 상수 (LEFT_WRIST 등)
  * Map / Homography     : occupancy grid ↔ 미터 좌표, H 로드/적용
  * Z Calibration        : 3x4 projection P 로드, 머리 높이 head_z 역산
  * BBox / Geometry      : IoU, 중심, 확장, point-in-box, circle-box
  * Pose 유틸            : keypoint 추출, IoU 매칭, head/foot/torso 계산
  * Helmet 로직          : 사람과 관련된 helmet 찾기 + 4단계 상태 판정
  * Posture (v09 폴백)   : head_z 없을 때 형상 기반 쓰러짐 판정
  * Drawing              : draw_text/box, resize, 상태별 색상

vision_core / safety_logic이 여기 함수들을 조합해서 프레임을 처리.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


# ============================================================
# COCO 17 Pose Keypoint Index
# ============================================================
# Ultralytics YOLO pose가 반환하는 keypoint 순서 (COCO 표준).
# get_visible_point(pose_item, idx, ...) 형태로 조회.

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


# ============================================================
# Map / Homography
# ============================================================

def load_map_from_yaml(yaml_path):
    """
    ROS 표준 occupancy grid yaml + pgm을 읽어 map_info dict 생성.

    yaml 필수 필드:
      image      : 상대/절대 경로
      resolution : m/px (예: 0.05)
      origin     : [ox, oy, θ] map 좌표계 원점

    Returns:
        dict {img, gray, width, height, resolution, origin, yaml_path, image_path}
    """
    yaml_path = Path(yaml_path).expanduser().resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"map yaml not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    image_rel = data.get("image")
    if image_rel is None:
        raise RuntimeError(f"'image' key not found in map yaml: {yaml_path}")

    image_path = Path(image_rel)
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"map image not found: {image_path}")

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"failed to read map image: {image_path}")

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    resolution = float(data.get("resolution", 0.05))
    origin = data.get("origin", [0.0, 0.0, 0.0])

    return {
        "img": img,
        "gray": gray,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "resolution": resolution,
        "origin": origin,
        "yaml_path": str(yaml_path),
        "image_path": str(image_path),
    }


def map_meter_to_image_pixel(mx, my, map_info):
    """
    map 미터 좌표 → pgm 이미지 픽셀 좌표.

    ROS 표준 규약: y축이 뒤집혀 있음 (map은 위로 +y, 이미지는 아래로 +y).
      px = (mx - ox) / res
      py = h - (my - oy) / res
    """
    resolution = float(map_info["resolution"])
    origin = map_info["origin"]

    ox = float(origin[0])
    oy = float(origin[1])
    h = int(map_info["height"])

    px = int(round((float(mx) - ox) / resolution))
    py = int(round(h - (float(my) - oy) / resolution))

    return px, py


def map_image_pixel_to_meter(px, py, map_info):
    """
    pgm 이미지 픽셀 좌표 → map 미터 좌표. map_meter_to_image_pixel의 역함수.
    Entry ROI 지정 모드(set_roi_mode)에서 마우스 클릭 좌표 변환에 사용.
    """
    resolution = float(map_info["resolution"])
    origin = map_info["origin"]

    ox = float(origin[0])
    oy = float(origin[1])
    h = int(map_info["height"])

    mx = ox + float(px) * resolution
    my = oy + float(h - py) * resolution

    return mx, my


def load_homography(path):
    """
    캘리브레이션 .npz에서 3×3 호모그래피 행렬 H 로드.

    npz 안의 여러 키 중 우선순위 순으로 검색: H > homography > camera_to_map > matrix.
    없으면 npz의 다른 배열 중 shape (3,3)인 첫 것을 사용.

    현재 프로젝트의 H는 카메라 픽셀 → **map 미터**로 피팅되어 있음.
    (검증: 재투영 오차 ~7cm, pixel 좌표와는 200px 이상 어긋남)

    Returns:
        (H, key_used) — key_used는 어느 키에서 읽었는지 디버깅용
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"homography file not found: {path}")

    data = np.load(str(path), allow_pickle=True)

    preferred_keys = ["H", "homography", "camera_to_map", "matrix"]

    for key in preferred_keys:
        if key in data:
            arr = np.array(data[key], dtype=np.float64)
            if arr.shape == (3, 3):
                return arr, key

    for key in data.files:
        arr = np.array(data[key])
        if arr.shape == (3, 3):
            return arr.astype(np.float64), key

    raise RuntimeError(f"no 3x3 homography matrix found in {path}")


def apply_homography(H, point):
    """
    호모그래피로 2D 점 변환. 동차좌표(homogeneous)에서 3번째 성분으로 나눠 정규화.

      [x' y' w']^T = H @ [x y 1]^T
      out = (x'/w', y'/w')

    w' ≈ 0 (평면상 점이 카메라 뒤 등)이면 None 반환.
    """
    if point is None:
        return None

    x, y = float(point[0]), float(point[1])
    src = np.array([x, y, 1.0], dtype=np.float64)

    dst = H @ src

    if abs(float(dst[2])) < 1e-9:
        return None

    ox = float(dst[0] / dst[2])
    oy = float(dst[1] / dst[2])

    if not np.isfinite(ox) or not np.isfinite(oy):
        return None

    return np.array([ox, oy], dtype=np.float64)


def camera_pixel_to_map_meter(H, pixel, map_info, homography_output="map_meters"):
    """
    카메라 픽셀 → map 미터 좌표.

    homography_output:
        "map_meters" (기본, 권장): H가 미터로 피팅되어 있으므로 그대로 반환.
        "map_pixels" (⚠ 사용 금지): H를 픽셀로 착각해 다시 변환. 결과가 어긋남.

    항상 "map_meters"를 사용해야 함.
    """
    out = apply_homography(H, pixel)

    if out is None:
        return None

    if homography_output == "map_meters":
        return out

    if homography_output == "map_pixels":
        mx, my = map_image_pixel_to_meter(out[0], out[1], map_info)
        return np.array([mx, my], dtype=np.float64)

    raise ValueError(f"unknown homography_output: {homography_output}")


# ============================================================
# Z Calibration
# ============================================================

def load_z_calibration(path):
    """
    카메라 3×4 projection matrix P 로드.
    P: world (X,Y,Z,1)^T → image (u*w, v*w, w)^T 매핑.

    npz 안에서 P > projection > projection_matrix > camera_matrix 순으로 탐색.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"z calibration file not found: {path}")

    data = np.load(str(path), allow_pickle=True)

    preferred_keys = ["P", "projection", "projection_matrix", "camera_matrix"]

    for key in preferred_keys:
        if key in data:
            arr = np.array(data[key], dtype=np.float64)
            if arr.shape == (3, 4):
                return arr

    for key in data.files:
        arr = np.array(data[key])
        if arr.shape == (3, 4):
            return arr.astype(np.float64)

    raise RuntimeError(f"no 3x4 projection matrix found in {path}")


def project_world_to_image(P, X, Y, Z):
    """
    world (X, Y, Z) → image (u, v) 순방향 투영.
    검증용 (head_z 추정 결과가 얼마나 정확히 head_px에 재투영되는지 확인).
    """
    xyz1 = np.array([float(X), float(Y), float(Z), 1.0], dtype=np.float64)
    uvw = P @ xyz1

    if abs(float(uvw[2])) < 1e-9:
        return None

    u = float(uvw[0] / uvw[2])
    v = float(uvw[1] / uvw[2])

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return np.array([u, v], dtype=np.float64)


def estimate_z_on_vertical_line(P, map_x, map_y, head_px, z_min=-0.5, z_max=3.5):
    """
    발이 서 있는 (X, Y) 위에서 z를 변수로 두고 head_px에 맞는 z를 역산.

    투영식: [u*w, v*w, w]^T = P @ [X, Y, Z, 1]^T
    → u = (p0X + p1Y + p2Z + p3) / (p8X + p9Y + p10Z + p11)
      v = (p4X + p5Y + p6Z + p7) / (p8X + p9Y + p10Z + p11)

    미지수는 Z 하나인데 식이 둘(u식, v식)이라 답이 두 개 나온다.

    ── 두 답을 단순 평균하면 안 된다 ──
    머리가 위아래로 움직여도 가로 픽셀 u는 거의 안 변한다. 즉 u식은
    높이 정보를 거의 담고 있지 않다. 실측(cam1)에서 Z 1m당
        v: 360 px 이동      u: 40 px 이동
    이라 u식으로 푼 Z는 v식보다 9배 부정확하고, 화면 가장자리에서는
    거의 퇴화한다 (u식이 -4.57 m 같은 값을 뱉음).

    그래서 각 행의 z 민감도 s = |d(coord)/dz| 를 유한차분으로 구해
    역분산 가중(w = s^2)으로 합친다. s_v >> s_u 이면 자동으로 v식이 된다.

    유효 범위(z_min ~ z_max) 밖의 해는 버린다. 둘 다 버려지면
    "머리 높이를 못 구함"이므로 None 을 반환한다.
    (예전에는 버린 값들을 다시 평균해서 되살렸고, 누운 사람이
     6.00 m 와 -0.97 m 의 평균인 2.51 m 로 '서 있는 사람'이 되었다.)

    Returns:
        (z_est, residual, projected_uv)
          z_est        : 추정 머리 높이 [m]. 실패 시 None.
          residual     : 추정 z로 재투영한 픽셀과 head_px 차이 [px].
                         두 식이 얼마나 어긋났는지를 재는 값이며,
                         "머리가 발 위에 있다"는 가정이 깨지면 급증한다.
                         호출부는 이 값으로 z 를 신뢰할지 결정해야 한다.
          projected_uv : 재투영 결과 픽셀 좌표
    """
    if head_px is None:
        return None, None, None

    u = float(head_px[0])
    v = float(head_px[1])
    X = float(map_x)
    Y = float(map_y)

    solutions = []  # (row_idx, z)

    for row_idx, coord in [(0, u), (1, v)]:
        prow = P[row_idx]
        pden = P[2]

        row_fixed = prow[0] * X + prow[1] * Y + prow[3]
        den_fixed = pden[0] * X + pden[1] * Y + pden[3]

        denom = coord * pden[2] - prow[2]
        numer = row_fixed - coord * den_fixed

        if abs(float(denom)) > 1e-9:
            z = numer / denom
            if np.isfinite(z):
                solutions.append((row_idx, float(z)))

    valid = [(r, z) for r, z in solutions if z_min <= z <= z_max]

    if not valid:
        return None, None, None

    if len(valid) == 1:
        z_est = valid[0][1]
    else:
        z_nom = float(np.mean([z for _, z in valid]))

        p0 = project_world_to_image(P, X, Y, z_nom)
        p1 = project_world_to_image(P, X, Y, z_nom + 0.01)

        if p0 is None or p1 is None:
            z_est = z_nom
        else:
            # s[0] = du/dz, s[1] = dv/dz  (px per meter)
            s = np.abs(p1 - p0) / 0.01
            w = np.array([s[r] ** 2 for r, _ in valid], dtype=np.float64)

            if float(w.sum()) <= 1e-12:
                z_est = z_nom
            else:
                zs = np.array([z for _, z in valid], dtype=np.float64)
                z_est = float(np.sum(w * zs) / w.sum())

    projected = project_world_to_image(P, X, Y, z_est)

    if projected is None:
        return z_est, None, None

    residual = float(np.linalg.norm(projected - np.array([u, v], dtype=np.float64)))

    return z_est, residual, projected


# ============================================================
# BBox / Geometry
# ============================================================

def box_area(box):
    """bbox [x1,y1,x2,y2] 면적 [px^2]. 음수 방어."""
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def box_iou(a, b):
    """
    두 bbox의 IoU (Intersection over Union). pose ↔ person 매칭에 사용.
    교집합 없으면 0, 완전 일치면 1.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def box_center(box):
    """bbox 중심점 [cx, cy]."""
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def expand_box(box, ratio, image_w=None, image_h=None):
    """
    bbox를 중심 기준으로 ratio 배 확장. 옵션으로 이미지 경계 clip.
    helmet 매칭 시 사람 bbox 주변 여유 영역을 잡기 위해 사용.
    """
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1) * float(ratio)
    h = (y2 - y1) * float(ratio)

    nx1 = cx - w * 0.5
    ny1 = cy - h * 0.5
    nx2 = cx + w * 0.5
    ny2 = cy + h * 0.5

    if image_w is not None:
        nx1 = max(0.0, min(float(image_w - 1), nx1))
        nx2 = max(0.0, min(float(image_w - 1), nx2))

    if image_h is not None:
        ny1 = max(0.0, min(float(image_h - 1), ny1))
        ny2 = max(0.0, min(float(image_h - 1), ny2))

    return [nx1, ny1, nx2, ny2]


def point_in_box(point, box):
    """점이 axis-aligned bbox 내부에 있는지 판정. None 안전."""
    if point is None:
        return False

    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, box)

    return x1 <= x <= x2 and y1 <= y <= y2


def circle_intersects_box(center, radius, box):
    """
    원(center, radius)이 bbox와 교차하는지 판정.
    헬멧 착용 판정에서 "머리 원이 helmet box에 겹치나?" 검사에 사용.
    bbox 내부에서 center에 가장 가까운 점과의 거리로 판정.
    """
    if center is None:
        return False

    cx, cy = float(center[0]), float(center[1])
    x1, y1, x2, y2 = map(float, box)

    nearest_x = min(max(cx, x1), x2)
    nearest_y = min(max(cy, y1), y2)

    dist_sq = (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2

    return dist_sq <= float(radius) ** 2


# ============================================================
# Pose
# ============================================================

def extract_pose_items(result):
    """
    Ultralytics pose 결과를 dict 리스트로 변환.
    각 항목: {'keypoints': (17,2), 'conf': (17,), 'box': [x1,y1,x2,y2]}
    box가 없으면 visible keypoint의 min/max로 자동 생성.
    """
    items = []

    if result is None:
        return items

    if result.keypoints is None:
        return items

    try:
        xy = result.keypoints.xy.detach().cpu().numpy()
    except Exception:
        return items

    try:
        conf = result.keypoints.conf.detach().cpu().numpy()
    except Exception:
        conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    boxes = None
    try:
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
    except Exception:
        boxes = None

    for i in range(xy.shape[0]):
        kpts = xy[i].astype(np.float64)
        kconf = conf[i].astype(np.float64)

        if boxes is not None and i < len(boxes):
            box = boxes[i].tolist()
        else:
            visible = kpts[kconf > 0.1]
            if len(visible) > 0:
                x1, y1 = visible.min(axis=0)
                x2, y2 = visible.max(axis=0)
                box = [float(x1), float(y1), float(x2), float(y2)]
            else:
                box = [0.0, 0.0, 0.0, 0.0]

        items.append(
            {
                "keypoints": kpts,
                "conf": kconf,
                "box": box,
            }
        )

    return items


def get_visible_point(pose_item, idx, conf_thres=0.25):
    """
    특정 keypoint idx의 (x, y) 반환. conf가 낮거나 NaN이면 None.
    자세/손 판정 로직에서 개별 keypoint 조회할 때 사용.
    """
    if pose_item is None:
        return None

    kpts = pose_item.get("keypoints")
    conf = pose_item.get("conf")

    if kpts is None or conf is None:
        return None

    if idx < 0 or idx >= len(kpts):
        return None

    if float(conf[idx]) < float(conf_thres):
        return None

    p = np.array(kpts[idx], dtype=np.float64)

    if not np.all(np.isfinite(p)):
        return None

    return p


def match_pose_to_person(person_box, pose_items):
    """
    person YOLO detection에 가장 어울리는 pose_item을 IoU 기준으로 선택.
    두 모델(track vs pose)이 독립 추론이라 매 프레임 매칭이 필요.
    """
    if not pose_items:
        return None, 0.0

    best_item = None
    best_iou = 0.0

    for item in pose_items:
        iou = box_iou(person_box, item.get("box", [0, 0, 0, 0]))

        if iou > best_iou:
            best_iou = iou
            best_item = item

    if best_item is None:
        return None, 0.0

    return best_item, float(best_iou)


def get_head_point(pose_item, person_box=None, helmet_box=None, conf_thres=0.25):
    """
    머리 픽셀 좌표 반환. 3단 폴백:
      1) 얼굴 5개 키포인트(코/눈/귀) 평균 → "pose_head" (최우선, 안정적)
      2) 관련 helmet box 중앙 → "helmet_box"
      3) person bbox 상단 12% 지점 → "person_top" (최후 폴백, 불안정)

    반환된 source(head_src)는 helmet 판정에서 신뢰도 판단에 사용.
    """
    head_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    pts = []

    for idx in head_indices:
        p = get_visible_point(pose_item, idx, conf_thres)
        if p is not None:
            pts.append(p)

    if pts:
        return np.mean(np.stack(pts, axis=0), axis=0), "pose_head"

    if helmet_box is not None:
        x1, y1, x2, y2 = map(float, helmet_box)
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64), "helmet_box"

    if person_box is not None:
        x1, y1, x2, y2 = map(float, person_box)
        return np.array([(x1 + x2) * 0.5, y1 + (y2 - y1) * 0.12], dtype=np.float64), "person_top"

    return None, "none"


def get_foot_point(pose_item, person_box, conf_thres=0.25):
    """
    발 픽셀 좌표 반환. 2단 폴백:
      1) 좌/우 발목 keypoint 평균 → "ankle" (호모그래피 입력으로 정확)
      2) person bbox 하단 중앙 → "bbox_bottom" (발목 안 보일 때)

    이 지점을 H로 변환한 결과가 map_xy — Nav2 goal의 최종 좌표.
    """
    pts = []

    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        p = get_visible_point(pose_item, idx, conf_thres)
        if p is not None:
            pts.append(p)

    if pts:
        return np.mean(np.stack(pts, axis=0), axis=0), "ankle"

    x1, y1, x2, y2 = map(float, person_box)
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float64), "bbox_bottom"


def get_torso_horizontal(pose_item, conf_thres=0.25):
    """
    몸통이 수평인지 판정 (쓰러짐 형상 조건).

    어깨 중점(양쪽 shoulder 평균) → 엉덩이 중점(양쪽 hip 평균) 벡터의
    |dx| > |dy| * 1.15면 몸통이 옆으로 누워있다고 봄.
    (1.15 마진은 서있어도 약간 기운 자세를 오탐 안 하기 위함)

    Returns:
        (is_horizontal: bool, shoulder_mid, hip_mid) — 미드 포인트는 그리기 용
    """
    if pose_item is None:
        return False, None, None

    ls = get_visible_point(pose_item, LEFT_SHOULDER, conf_thres)
    rs = get_visible_point(pose_item, RIGHT_SHOULDER, conf_thres)
    lh = get_visible_point(pose_item, LEFT_HIP, conf_thres)
    rh = get_visible_point(pose_item, RIGHT_HIP, conf_thres)

    if ls is None or rs is None or lh is None or rh is None:
        return False, None, None

    shoulder_mid = (ls + rs) * 0.5
    hip_mid = (lh + rh) * 0.5

    vec = hip_mid - shoulder_mid
    dx = abs(float(vec[0]))
    dy = abs(float(vec[1]))

    torso_horizontal = dx > dy * 1.15

    return bool(torso_horizontal), shoulder_mid, hip_mid


def get_keypoint_spread_ratio(pose_item, conf_thres=0.25):
    """
    visible keypoint의 bounding rect 종횡비(w/h) 반환.
    누워있으면 키포인트가 가로로 넓게 퍼짐 → w/h > 1.
    쓰러짐 판정의 형상 보조 지표.
    """
    if pose_item is None:
        return 0.0, 0

    kpts = pose_item.get("keypoints")
    conf = pose_item.get("conf")

    if kpts is None or conf is None:
        return 0.0, 0

    mask = conf >= conf_thres
    visible = kpts[mask]

    if len(visible) < 3:
        return 0.0, int(len(visible))

    x1, y1 = visible.min(axis=0)
    x2, y2 = visible.max(axis=0)

    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    return float(w / h), int(len(visible))


# ============================================================
# Body axis analysis  (pose keypoints + z calibration)
# ============================================================

def _mid(pose_item, idx_a, idx_b, conf_thres):
    """두 키포인트의 중점. 하나만 보이면 그것, 둘 다 없으면 None."""
    a = get_visible_point(pose_item, idx_a, conf_thres)
    b = get_visible_point(pose_item, idx_b, conf_thres)

    if a is not None and b is not None:
        return (a + b) * 0.5
    return a if a is not None else b


def analyze_body_axis(P, foot_map, foot_px, pose_item, conf_thres=0.25, user_height_m=1.80):
    """
    캘리브레이션이 알려주는 '월드 수직 방향'을 기준으로 몸의 자세를 분석한다.

    ── 왜 머리 하나만 보면 안 되는가 ──
    카메라가 가파르게 내려다보면 코·눈·귀가 정수리에 가려진다. 그러면
    get_head_point 가 매 프레임 pose_head / helmet_box / person_top 사이를
    오가며 머리 픽셀이 튄다. cam0 에서 사람이 363px 로 작게 보이고 v 민감도가
    155 px/m 라, 키포인트가 몇 픽셀만 흔들려도 head_z 가 크게 흔들린다.

    반면 어깨·엉덩이는 위에서 봐도 가려지지 않고 pose 모델이 가장 안정적으로
    잡는 부위다. 그래서 머리 대신 몸 전체의 z 사다리를 본다.

    ── 월드 수직 방향 e_v ──
    발이 선 (X,Y) 위로 1m 올라간 점을 P 로 투영하면, 그 위치에서 '위'가
    화면상 어느 방향인지 알 수 있다. 이미지 세로축이 아니라 이것을 기준으로
    삼아야 카메라 기울기와 원근에 영향을 받지 않는다.

    Returns:
        dict 또는 None (계산 불가 시)
          e_v, e_perp     : 화면상 '위' 방향과 그 수직 방향 (단위벡터)
          px_per_m        : e_v 방향으로 1m 가 몇 px 인지
          z_hip/z_shoulder/z_head : 각 부위의 추정 높이 [m] (None 가능)
          ladder_m        : z_shoulder - z_hip. 서면 ~0.5, 누우면 ~0
          along_spread_m  : 키포인트가 수직 방향으로 퍼진 길이 [m]
          perp_spread_m   : 수직선에서 옆으로 퍼진 길이 [m]
          spread_ratio    : perp / along.  서면 ~0.3, 옆으로 누우면 >1.5
          torso_angle_deg : 어깨→엉덩이 벡터가 수직에서 벗어난 각 [deg]
          body_angle_deg  : 어깨→발목 벡터가 수직에서 벗어난 각 [deg]
          visible_kpts
    """
    if P is None or foot_map is None or foot_px is None or pose_item is None:
        return None

    X, Y = float(foot_map[0]), float(foot_map[1])

    p0 = project_world_to_image(P, X, Y, 0.0)
    p1 = project_world_to_image(P, X, Y, 1.0)

    if p0 is None or p1 is None:
        return None

    up = p1 - p0
    px_per_m = float(np.linalg.norm(up))

    if px_per_m < 1e-6:
        return None

    e_v = up / px_per_m                      # 화면상 '위'
    e_perp = np.array([-e_v[1], e_v[0]])     # 그 수직 방향

    kpts = pose_item.get("keypoints")
    conf = pose_item.get("conf")

    if kpts is None or conf is None:
        return None

    mask = np.asarray(conf) >= conf_thres
    visible = np.asarray(kpts)[mask]

    if len(visible) < 4:
        return None

    # 발 픽셀을 원점으로, (e_v, e_perp) 좌표계로 키포인트를 옮긴다
    d = visible - np.asarray(foot_px, dtype=np.float64)
    along = d @ e_v / px_per_m               # [m] 지면 위 높이 (선형 근사)
    perp = d @ e_perp / px_per_m             # [m] 수직선에서 옆으로

    along_spread = float(along.max() - along.min())
    perp_spread = float(perp.max() - perp.min())
    spread_ratio = perp_spread / max(along_spread, 1e-3)

    def z_of(point):
        if point is None:
            return None, None
        z, res, _ = estimate_z_on_vertical_line(P, X, Y, point)
        return z, res

    hip_mid = _mid(pose_item, LEFT_HIP, RIGHT_HIP, conf_thres)
    sh_mid = _mid(pose_item, LEFT_SHOULDER, RIGHT_SHOULDER, conf_thres)
    knee_mid = _mid(pose_item, LEFT_KNEE, RIGHT_KNEE, conf_thres)
    ank_mid = _mid(pose_item, LEFT_ANKLE, RIGHT_ANKLE, conf_thres)
    head_pt, _src = get_head_point(pose_item, conf_thres=conf_thres)

    z_hip, r_hip = z_of(hip_mid)
    z_sh, r_sh = z_of(sh_mid)
    z_head, r_head = z_of(head_pt)

    ladder = None
    if z_hip is not None and z_sh is not None:
        ladder = float(z_sh - z_hip)

    def angle_from_up(vec):
        if vec is None:
            return None
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            return None
        c = float(np.clip(np.dot(vec / n, -e_v), -1.0, 1.0))  # 어깨→엉덩이는 '아래'
        return float(np.degrees(np.arccos(abs(c))))

    torso_angle = None
    if sh_mid is not None and hip_mid is not None:
        torso_angle = angle_from_up(hip_mid - sh_mid)

    body_angle = None
    if sh_mid is not None and ank_mid is not None:
        body_angle = angle_from_up(ank_mid - sh_mid)

    # ── 다리 각도: 쓰러짐 판정의 핵심 ──────────────────────────────
    #
    # 다리를 네 마디(좌허벅지/좌정강이/우허벅지/우정강이)로 쪼개고
    # 그 중 '가장 수직인 마디'의 각도를 본다 = min(4마디).
    #
    # 이렇게 해야 아래가 전부 걸러진다:
    #   걷기(한발 들기) : 든 다리는 수평이지만 디딤발은 수직 -> min 작음
    #   딥 스쿼트       : 허벅지는 수평이지만 정강이는 수직   -> min 작음
    #   무릎 꿇기       : 정강이는 수평이지만 허벅지는 수직   -> min 작음
    #   의자 앉기       : 허벅지 수평, 정강이 수직            -> min 작음
    #   누움            : 네 마디 전부 수평                  -> min 큼
    #
    # (엉덩이중점→발목중점 하나만 쓰면 걷기·스쿼트에서 30~47도가 나와 오탐)
    def seg_angle(a_idx, b_idx):
        a = get_visible_point(pose_item, a_idx, conf_thres)
        b = get_visible_point(pose_item, b_idx, conf_thres)
        if a is None or b is None:
            return None
        return angle_from_up(b - a)

    segments = [
        seg_angle(LEFT_HIP, LEFT_KNEE),
        seg_angle(LEFT_KNEE, LEFT_ANKLE),
        seg_angle(RIGHT_HIP, RIGHT_KNEE),
        seg_angle(RIGHT_KNEE, RIGHT_ANKLE),
    ]

    # ── 다리별로 묶어서 판단한다 ─────────────────────────────────
    #
    # 한쪽 다리가 '누웠다' = 그 다리의 허벅지와 정강이가 둘 다 수평이다.
    #   leg_L = min(허벅지L, 정강이L)
    #   leg_R = min(허벅지R, 정강이R)
    # 그리고 '한쪽 다리라도 통째로 누웠으면' 누운 것으로 본다.
    #   leg_angle = max(leg_L, leg_R)
    #
    # 왜 4마디 전체의 min 이 아닌가:
    #   사람은 쓰러질 때 다리를 편 채로 눕지 않는다. 보통 한쪽 무릎을 굽힌다.
    #   실측 예) segs = [50, 6, 29, 36]  (허벅L 정강L 허벅R 정강R)
    #     min(4마디) = 6   -> 임계 20 미달 -> 놓침
    #     max(다리별) = min(29,36) = 29 -> 검출
    #   굽힌 왼쪽 정강이(6도) 때문에 전체가 죽는 것을 막는다.
    #
    # 오탐이 늘지 않는 이유:
    #   걷기/스쿼트/무릎꿇기/의자앉기는 어느 다리를 봐도 한 마디는 수직이라
    #   leg_L, leg_R 이 둘 다 작다. 게다가 몸통 조건(torso > 25도)이 AND 로
    #   붙어 있어 이 자세들은 그쪽에서도 걸러진다.
    def _leg(thigh, shank):
        seen = [x for x in (thigh, shank) if x is not None]
        return float(min(seen)) if seen else None

    leg_left = _leg(segments[0], segments[1])
    leg_right = _leg(segments[2], segments[3])
    legs = [x for x in (leg_left, leg_right) if x is not None]
    leg_angle = float(max(legs)) if legs else None

    # 가장 수직인 다리. "다리가 서 있음을 증명" 하는 데 쓴다.
    leg_min = float(min(legs)) if legs else None

    # 엉덩이가 (더 낮은 쪽) 발목보다 얼마나 위인가 [m].
    #   서있음/걷기/굽힘 0.83~0.91 / 의자앉기 0.40 / 스쿼트 0.25 / 무릎꿇기 0.28
    hip_above_ankle = None
    if hip_mid is not None:
        fp = np.asarray(foot_px, dtype=np.float64)
        ank_alongs = []
        for idx in (LEFT_ANKLE, RIGHT_ANKLE):
            a = get_visible_point(pose_item, idx, conf_thres)
            if a is not None:
                ank_alongs.append(float((a - fp) @ e_v / px_per_m))
        if ank_alongs:
            hip_along = float((hip_mid - fp) @ e_v / px_per_m)
            hip_above_ankle = hip_along - min(ank_alongs)

    # ── 원근압축 지표: 몸이 카메라 축 방향으로 누우면 '짧아 보인다' ──────
    #
    # 각도만 보면 절대 못 잡는 사각지대가 있다.
    # 카메라를 정면으로 향해 누우면 몸통이 화면상 수직선과 나란해져서
    # torso_angle 이 서 있는 사람과 똑같이 0~15도로 나온다. (실측 13.3, 14.2)
    #
    # 하지만 길이는 못 속인다. px_per_m 은 그 지점에서 '월드 1m 가 몇 px' 인지를
    # 호모그래피(P)에서 직접 뽑은 값이므로, 픽셀 길이를 m 로 되돌릴 수 있다.
    #
    #   사람 몸통(어깨→엉덩이)은 무조건 약 0.52 m 다.
    #   서든 앉든 쪼그리든 무릎꿇든 '허리는 줄어들지 않는다'.
    #   원근압축으로 누우면 0.25~0.33 m 로 보인다. 그런 사람은 없다.
    #
    #   torso_len_m  : 어깨→엉덩이 겉보기 길이 [m]  (실제 0.52)
    #   body_len_m   : 발목→머리   겉보기 길이 [m]  (실제 1.65)
    #
    # body_len_m 은 게이트로 쓴다. 스쿼트/바닥앉기는 몸이 '접혀서' 진짜로
    # 짧아지므로(0.5~0.8m) 원근압축과 헷갈린다. 누운 사람은 몸이 펴져 있어
    # 압축돼도 0.9m 이상은 유지된다. 이걸로 둘을 가른다.
    def _seg_len_m(a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.asarray(b) - np.asarray(a)) / px_per_m)

    torso_len_m = _seg_len_m(sh_mid, hip_mid)
    ank_ref = ank_mid if ank_mid is not None else np.asarray(foot_px, dtype=np.float64)
    body_len_m = _seg_len_m(head_pt, ank_ref)

    return {
        "e_v": e_v,
        "e_perp": e_perp,
        "px_per_m": px_per_m,
        "torso_len_m": torso_len_m,
        "body_len_m": body_len_m,
        "z_hip": z_hip, "z_shoulder": z_sh, "z_head": z_head,
        "res_hip": r_hip, "res_shoulder": r_sh, "res_head": r_head,
        "ladder_m": ladder,
        "along_spread_m": along_spread,
        "perp_spread_m": perp_spread,
        "spread_ratio": float(spread_ratio),
        "torso_angle_deg": torso_angle,
        "body_angle_deg": body_angle,
        "leg_angle_deg": leg_angle,
        "leg_min_deg": leg_min,
        "leg_left_deg": leg_left,
        "leg_right_deg": leg_right,
        "leg_segments_deg": segments,
        "hip_above_ankle_m": hip_above_ankle,
        "visible_kpts": int(len(visible)),
    }


# ============================================================
# Helmet logic
# ============================================================

def find_nearest_related_helmet(person_box, helmets, image_w, image_h):
    """
    사람 bbox와 관련된 helmet 중 가장 유력한 하나를 선택.

    후보 조건: helmet 중심이 person 확장 bbox(±25% w, ±15% h) 내부.
    점수 = 사람 중심과의 거리 - IoU * 100 → 거리 짧고 겹칠수록 낮은 값(우선).
    """
    if not helmets:
        return None, None

    x1, y1, x2, y2 = map(float, person_box)
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)

    person_expanded = [
        max(0.0, x1 - pw * 0.25),
        max(0.0, y1 - ph * 0.15),
        min(float(image_w - 1), x2 + pw * 0.25),
        min(float(image_h - 1), y2 + ph * 0.15),
    ]

    pcenter = box_center(person_box)

    best = None
    best_score = float("inf")

    for helmet in helmets:
        hbox = helmet.get("box")
        if hbox is None:
            continue

        hc = box_center(hbox)

        if not point_in_box(hc, person_expanded):
            continue

        dist = float(np.linalg.norm(hc - pcenter))
        score = dist - box_iou(person_box, hbox) * 100.0

        if score < best_score:
            best_score = score
            best = helmet

    return best, best_score if best is not None else None


def judge_helmet_status(
    person_box,
    helmets,
    head_px,
    head_src,
    image_w,
    image_h,
    head_radius_scale=0.20,
    helmet_expand_ratio=1.00,
):
    """
    HELMET_ON
    - 머리 주변에 helmet box 있음

    NO_HELMET_RELATED
    - helmet box는 보임
    - 하지만 머리 주변이 아니라 사람 bbox 내부/근처에 있음
    - 즉 헬멧을 들고 있거나 바닥/몸 근처에 있는 상황

    NO_HELMET_MISSING
    - 사람 머리/상체는 어느 정도 보임
    - 그런데 helmet box가 아예 안 보임
    - 즉 헬멧 미착용으로 판단

    SUSPICIOUS
    - 사람은 보임
    - 그런데 머리 위치도 불안정하고 helmet도 안 보임
    - 의심 상황
    """
    x1, y1, x2, y2 = map(float, person_box)
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)

    head_radius = max(24.0, min(90.0, pw * float(head_radius_scale)))

    head_stable = head_px is not None and head_src in ["pose_head", "helmet_box"]
    head_unstable = head_px is None or head_src in ["person_top", "none"]

    # helmet box가 아예 안 보이는 경우
    if not helmets:
        if head_stable:
            return "NO_HELMET_MISSING", head_radius, []

        if head_unstable:
            return "SUSPICIOUS", head_radius, []

        return "UNKNOWN", head_radius, []

    expanded_person = [
        max(0.0, x1 - pw * 0.35),
        max(0.0, y1 - ph * 0.20),
        min(float(image_w - 1), x2 + pw * 0.35),
        min(float(image_h - 1), y2 + ph * 0.20),
    ]

    # helmet은 있는데 head 위치가 불안정한 경우
    if head_px is None:
        related_near = []

        for helmet in helmets:
            hbox = helmet.get("box")
            if hbox is None:
                continue

            hc = box_center(hbox)

            if point_in_box(hc, expanded_person):
                related_near.append(helmet)

        if related_near:
            return "SUSPICIOUS", head_radius, related_near

        return "UNKNOWN", head_radius, []

    head_related = []
    body_related = []
    near_related = []

    for helmet in helmets:
        hbox = helmet.get("box")
        if hbox is None:
            continue

        expanded_hbox = expand_box(
            hbox,
            helmet_expand_ratio,
            image_w=image_w,
            image_h=image_h,
        )

        # 머리 근처 helmet이면 착용
        if circle_intersects_box(head_px, head_radius, expanded_hbox):
            head_related.append(helmet)

        hc = box_center(hbox)

        # 사람 bbox 내부 helmet
        if point_in_box(hc, person_box):
            body_related.append(helmet)

        # 사람 bbox 근처 helmet
        elif point_in_box(hc, expanded_person):
            near_related.append(helmet)

    if head_related:
        return "HELMET_ON", head_radius, head_related

    # 헬멧은 보이는데 머리에 없음
    if body_related or near_related:
        return "NO_HELMET_RELATED", head_radius, body_related + near_related

    # 헬멧은 프레임에 있지만 이 사람과 관련 없음
    if head_stable:
        return "NO_HELMET_MISSING", head_radius, []

    if head_unstable:
        return "SUSPICIOUS", head_radius, []

    return "UNKNOWN", head_radius, []

# ============================================================
# Posture fallback
# ============================================================

def judge_posture_v09(
    person_box,
    pose_item,
    head_z=None,
    z_residual=None,
    image_h=None,
    user_height_m=1.80,
    pose_conf=0.25,
    lying_height_thres=0.78,
    very_low_height_thres=0.55,
    residual_thres=160.0,
):
    x1, y1, x2, y2 = map(float, person_box)
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph

    torso_horizontal, _, _ = get_torso_horizontal(pose_item, pose_conf)
    kpt_ratio, visible_kpts = get_keypoint_spread_ratio(pose_item, pose_conf)

    reasons = []
    lying = False
    posture = "NORMAL"

    valid_z = (
        head_z is not None
        and np.isfinite(head_z)
        and -0.20 <= float(head_z) <= user_height_m + 1.2
    )

    if valid_z:
        if head_z < very_low_height_thres:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("very_low_head_z")

        elif head_z < lying_height_thres and (aspect > 1.05 or torso_horizontal or kpt_ratio > 1.10):
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("low_head_z_with_shape")

        elif head_z < user_height_m * 0.65:
            posture = "LOW_POSTURE"
            reasons.append("low_posture_z")

    if not lying:
        if aspect > 1.35:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("wide_bbox")

        elif torso_horizontal and kpt_ratio > 1.0:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("horizontal_torso")

    debug = {
        "reasons": reasons,
        "aspect": float(aspect),
        "torso_horizontal": bool(torso_horizontal),
        "kpt_ratio": float(kpt_ratio),
        "visible_kpts": int(visible_kpts),
        "head_z": None if head_z is None else float(head_z),
        "z_residual": None if z_residual is None else float(z_residual),
    }

    return lying, posture, debug


# ============================================================
# Drawing
# ============================================================

def draw_text(img, text, org, color=(255, 255, 255), scale=0.5, thickness=1):
    """
    가독성을 위한 검정 외곽선 + 컬러 본문 텍스트 렌더링.
    같은 텍스트를 두 번 (외곽선 두껍게 + 본문 얇게) 그림.
    """
    x, y = int(org[0]), int(org[1])

    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (0, 0, 0),
        int(thickness) + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, int(thickness))

    if label:
        draw_text(img, label, (x1, max(18, y1 - 6)), color, 0.45, 1)


def resize_keep_ratio(img, target_w, target_h):
    """
    가로세로 비율 유지 리사이즈. 남는 영역은 검정 패딩.
    대시보드 4분할 조립 시 각 패널 크기 맞추는 데 사용.
    """
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2

    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas


def color_for_state(helmet_status, posture, emergency=False):
    """
    간단 상태 → BGR 색상 매핑. dashboard_ui.state_color()의 폴백.
    우선순위: emergency(빨) > lying(주) > helmet_on(초) > no_helmet(빨주) > suspicious(노) > 기본
    """
    if emergency:
        return (0, 0, 255)

    if posture == "LYING_CANDIDATE":
        return (0, 165, 255)

    if helmet_status == "HELMET_ON":
        return (0, 255, 0)

    if helmet_status == "NO_HELMET":
        return (0, 80, 255)

    if helmet_status == "SUSPICIOUS":
        return (0, 200, 255)

    return (0, 255, 255)