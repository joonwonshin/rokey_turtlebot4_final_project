import cv2
import numpy as np
import yaml
import json
import sys
from pathlib import Path


# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE_DIR = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물

CAPTURE_DIR = BASE_DIR / "captures"
CALIB_DIR = VISION_DIR / "calibration"
PREVIEW_DIR = CALIB_DIR / "preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

MAP_YAML = VISION_DIR / "final_project.yaml"

MAP_DISPLAY_SCALE = 3.0


def load_map_info(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    image_path = yaml_path.parent / data["image"]
    resolution = float(data["resolution"])
    origin = data["origin"]

    map_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"Map image not found: {image_path}")

    map_bgr = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    return map_bgr, resolution, origin


def ros_to_map_pixel(x_ros, y_ros, map_height, resolution, origin):
    origin_x, origin_y, _ = origin

    u = (x_ros - origin_x) / resolution
    v = map_height - ((y_ros - origin_y) / resolution)

    return int(round(u)), int(round(v))


def apply_homography(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def draw_points_on_camera(cam_img, camera_points):
    out = cam_img.copy()

    for idx, (x, y) in enumerate(camera_points):
        x = int(round(x))
        y = int(round(y))

        cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            out,
            str(idx + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        "CAMERA POINTS",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    return out


def draw_points_on_map(map_img, map_pixel_points, projected_map_pixels=None):
    out = map_img.copy()

    for idx, (x, y) in enumerate(map_pixel_points):
        x = int(round(x))
        y = int(round(y))

        # 실제 찍은 map point: 파란색
        cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(
            out,
            str(idx + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if projected_map_pixels is not None:
        for idx, (x, y) in enumerate(projected_map_pixels):
            x = int(round(x))
            y = int(round(y))

            # 호모그래피로 다시 예측한 위치: 초록색
            cv2.circle(out, (x, y), 3, (0, 255, 0), 1)

            # 실제점과 예측점 사이 선
            tx, ty = map_pixel_points[idx]
            tx = int(round(tx))
            ty = int(round(ty))
            cv2.line(out, (tx, ty), (x, y), (0, 255, 255), 1)

    cv2.putText(
        out,
        "MAP POINTS: blue=clicked, green=projected",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def make_side_by_side(cam_vis, map_vis):
    target_h = 720

    cam_resized = resize_to_height(cam_vis, target_h)

    map_big = cv2.resize(
        map_vis,
        None,
        fx=MAP_DISPLAY_SCALE,
        fy=MAP_DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    map_resized = resize_to_height(map_big, target_h)

    canvas = np.hstack([cam_resized, map_resized])
    return canvas


def visualize(cam_name):
    cam_img_path = CAPTURE_DIR / f"{cam_name}_ref.jpg"
    json_path = CALIB_DIR / f"{cam_name}_points.json"
    npz_path = CALIB_DIR / f"{cam_name}_to_map.npz"

    if not cam_img_path.exists():
        raise FileNotFoundError(f"Camera image not found: {cam_img_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Point json not found: {json_path}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Homography npz not found: {npz_path}")

    cam_img = cv2.imread(str(cam_img_path))
    if cam_img is None:
        raise FileNotFoundError(f"Failed to load: {cam_img_path}")

    map_img, resolution, origin = load_map_info(MAP_YAML)
    map_height = map_img.shape[0]

    with open(json_path, "r") as f:
        data = json.load(f)

    npz = np.load(str(npz_path))
    H = npz["H"]

    camera_points = data["camera_points"]
    map_pixel_points = data["map_pixel_points"]

    projected_map_pixels = []
    errors_px = []

    for cam_pt, map_pt in zip(camera_points, map_pixel_points):
        cx, cy = cam_pt

        x_ros, y_ros = apply_homography(H, cx, cy)
        pu, pv = ros_to_map_pixel(
            x_ros,
            y_ros,
            map_height=map_height,
            resolution=resolution,
            origin=origin,
        )

        projected_map_pixels.append((pu, pv))

        tx, ty = map_pt
        err_px = np.sqrt((pu - tx) ** 2 + (pv - ty) ** 2)
        errors_px.append(err_px)

    cam_vis = draw_points_on_camera(cam_img, camera_points)
    map_vis = draw_points_on_map(map_img, map_pixel_points, projected_map_pixels)

    side_by_side = make_side_by_side(cam_vis, map_vis)

    cam_out = PREVIEW_DIR / f"{cam_name}_camera_points.jpg"
    map_out = PREVIEW_DIR / f"{cam_name}_map_points_check.jpg"
    side_out = PREVIEW_DIR / f"{cam_name}_side_by_side.jpg"

    cv2.imwrite(str(cam_out), cam_vis)
    cv2.imwrite(str(map_out), map_vis)
    cv2.imwrite(str(side_out), side_by_side)

    print("======================================")
    print(f"[{cam_name}] Visualization saved")
    print("--------------------------------------")
    print(f"Camera points image : {cam_out}")
    print(f"Map check image     : {map_out}")
    print(f"Side by side image  : {side_out}")
    print("--------------------------------------")
    print(f"Point count         : {len(camera_points)}")
    print(f"Mean error px       : {np.mean(errors_px):.2f} px")
    print(f"Max error px        : {np.max(errors_px):.2f} px")
    print("======================================")

    cv2.imshow(f"{cam_name} camera points", resize_to_height(cam_vis, 720))

    map_big = cv2.resize(
        map_vis,
        None,
        fx=MAP_DISPLAY_SCALE,
        fy=MAP_DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow(f"{cam_name} map check", map_big)

    cv2.imshow(f"{cam_name} side by side", side_by_side)

    print("Press any key on image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 scripts/03_visualize_homography.py cam0")
        print("  python3 scripts/03_visualize_homography.py cam1")
        print("  python3 scripts/03_visualize_homography.py all")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "all":
        visualize("cam0")
        visualize("cam1")
    elif target in ["cam0", "cam1"]:
        visualize(target)
    else:
        print("[ERROR] target must be cam0, cam1, or all")
        sys.exit(1)


if __name__ == "__main__":
    main()