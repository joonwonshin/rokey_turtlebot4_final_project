import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE_DIR = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물

MAP_YAML = VISION_DIR / "final_project.yaml"
CALIB_DIR = VISION_DIR / "calibration"
OUTPUT_DIR = VISION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAM0_ID = 2
CAM1_ID = 4

MAP_DISPLAY_SCALE = 3.0

# 클릭 기록 저장
events = []


def load_map_info(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    image_path = yaml_path.parent / data["image"]
    resolution = float(data["resolution"])
    origin = data["origin"]

    map_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if map_gray is None:
        raise FileNotFoundError(f"Map image not found: {image_path}")

    map_bgr = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)
    return map_bgr, resolution, origin


def load_homography(cam_name):
    npz_path = CALIB_DIR / f"{cam_name}_to_map.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Homography file not found: {npz_path}")

    data = np.load(str(npz_path))
    H = data["H"]
    return H


def camera_pixel_to_ros(H, px, py):
    p = np.array([px, py, 1.0], dtype=np.float64)
    q = H @ p
    q = q / q[2]
    return float(q[0]), float(q[1])


def ros_to_map_pixel(x_ros, y_ros, map_height, resolution, origin):
    origin_x, origin_y, _ = origin

    u = (x_ros - origin_x) / resolution
    v = map_height - ((y_ros - origin_y) / resolution)

    return int(round(u)), int(round(v))


def draw_map_with_events(map_original, resolution, origin):
    out = map_original.copy()
    map_h = out.shape[0]

    for idx, ev in enumerate(events):
        cam_name = ev["camera"]
        x_ros = ev["x_ros"]
        y_ros = ev["y_ros"]

        u, v = ros_to_map_pixel(
            x_ros,
            y_ros,
            map_height=map_h,
            resolution=resolution,
            origin=origin,
        )

        # CAM0 = red, CAM1 = blue
        if cam_name == "cam0":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.circle(out, (u, v), 4, color, -1)
        cv2.putText(
            out,
            f"{idx + 1}:{cam_name}",
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        "MAP VIEW | red=cam0, blue=cam1 | c=clear, s=save, q=quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    map_disp = cv2.resize(
        out,
        None,
        fx=MAP_DISPLAY_SCALE,
        fy=MAP_DISPLAY_SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    return map_disp, out


def draw_camera_overlay(frame, cam_name):
    out = frame.copy()

    if cam_name == "cam0":
        color = (0, 0, 255)
        label = "CAM0 / video2"
    else:
        color = (255, 0, 0)
        label = "CAM1 / video4"

    cv2.putText(
        out,
        f"{label} | click floor point",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    # 해당 카메라에서 찍은 점 표시
    for idx, ev in enumerate(events):
        if ev["camera"] != cam_name:
            continue

        px = int(ev["px"])
        py = int(ev["py"])

        cv2.circle(out, (px, py), 7, color, -1)
        cv2.putText(
            out,
            str(idx + 1),
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def make_mouse_callback(cam_name, H):
    def callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x_ros, y_ros = camera_pixel_to_ros(H, x, y)

        ev = {
            "camera": cam_name,
            "px": x,
            "py": y,
            "x_ros": x_ros,
            "y_ros": y_ros,
        }
        events.append(ev)

        print(
            f"[{cam_name.upper()} CLICK] "
            f"pixel=({x}, {y}) -> map=({x_ros:.3f}, {y_ros:.3f})"
        )

    return callback


def open_camera(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera id {cam_id}")

    return cap


def main():
    map_original, resolution, origin = load_map_info(MAP_YAML)

    H0 = load_homography("cam0")
    H1 = load_homography("cam1")

    cap0 = open_camera(CAM0_ID)
    cap1 = open_camera(CAM1_ID)

    cv2.namedWindow("CAM0 / video2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("CAM1 / video4", cv2.WINDOW_NORMAL)
    cv2.namedWindow("MAP CLICK TEST", cv2.WINDOW_AUTOSIZE)

    cv2.setMouseCallback("CAM0 / video2", make_mouse_callback("cam0", H0))
    cv2.setMouseCallback("CAM1 / video4", make_mouse_callback("cam1", H1))

    print("======================================")
    print("Dual Camera Click To Map Test")
    print("--------------------------------------")
    print("CAM0 window click : cam0 pixel -> map coord")
    print("CAM1 window click : cam1 pixel -> map coord")
    print("c : clear points")
    print("s : save map preview")
    print("q : quit")
    print("======================================")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if ret0:
            cam0_vis = draw_camera_overlay(frame0, "cam0")
            cv2.imshow("CAM0 / video2", cam0_vis)

        if ret1:
            cam1_vis = draw_camera_overlay(frame1, "cam1")
            cv2.imshow("CAM1 / video4", cam1_vis)

        map_disp, map_save = draw_map_with_events(map_original, resolution, origin)
        cv2.imshow("MAP CLICK TEST", map_disp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            events.clear()
            print("[CLEAR] all clicked points removed")

        elif key == ord("s"):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"click_map_test_{now}.jpg"
            cv2.imwrite(str(out_path), map_save)
            print(f"[SAVE] {out_path}")

        elif key == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()