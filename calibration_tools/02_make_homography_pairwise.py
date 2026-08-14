import cv2
import numpy as np
import yaml
import json
import sys
from pathlib import Path


# =========================
# Path settings
# =========================
# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE_DIR = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물

CAPTURE_DIR = BASE_DIR / "captures"
CALIB_DIR = VISION_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

MAP_YAML = VISION_DIR / "final_project.yaml"


# =========================
# Display settings
# =========================
# 카메라 이미지는 1280x720이라 조금 줄여서 보여줌
CAM_DISPLAY_SCALE = 0.75

# 맵은 원본이 작아서 확대해서 보여줌
# 너무 크면 2.0, 더 크게 보고 싶으면 3.0~4.0
MAP_DISPLAY_SCALE = 3.0

CAM_POINT_RADIUS = 5
MAP_POINT_RADIUS = 5

CAM_FONT_SCALE = 0.6
MAP_FONT_SCALE = 0.6

CAM_THICKNESS = 2
MAP_THICKNESS = 2


def load_map_info(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    image_path = yaml_path.parent / data["image"]
    resolution = float(data["resolution"])
    origin = data["origin"]  # [x, y, theta]

    map_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"Map image not found: {image_path}")

    map_bgr = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

    return map_bgr, resolution, origin, image_path


def map_pixel_to_ros(u, v, map_height, resolution, origin):
    """
    map image pixel 좌표를 ROS map 좌표로 변환.

    이미지 좌표:
      u: 오른쪽 +
      v: 아래쪽 +

    ROS map 좌표:
      x: 오른쪽 +
      y: 위쪽 +

    그래서 y는 map_height - v 로 뒤집어야 함.
    """
    origin_x, origin_y, _ = origin

    x = origin_x + u * resolution
    y = origin_y + (map_height - v) * resolution

    return x, y


def draw_label_box(img, lines, x=10, y=25, font_scale=0.6, thickness=2):
    """
    화면 위쪽 설명 글씨를 잘 보이게 박스 배경과 함께 그림.
    """
    line_height = int(28 * font_scale / 0.6)
    box_h = line_height * len(lines) + 16
    box_w = 760

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        yy += line_height


class PairwiseHomographyCollector:
    def __init__(self, cam_img, map_img, cam_name):
        self.cam_name = cam_name

        self.cam_original = cam_img.copy()
        self.map_original = map_img.copy()

        self.cam_h, self.cam_w = self.cam_original.shape[:2]
        self.map_h, self.map_w = self.map_original.shape[:2]

        self.cam_points = []  # original camera pixel points
        self.map_points = []  # original map pixel points

        self.expect = "camera"  # camera -> map -> camera -> map ...
        self.finished = False

        self.cam_window = f"{cam_name.upper()} CAMERA IMAGE"
        self.map_window = "MAP IMAGE"

        cv2.namedWindow(self.cam_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.map_window, cv2.WINDOW_AUTOSIZE)

        cv2.setMouseCallback(self.cam_window, self.on_camera_click)
        cv2.setMouseCallback(self.map_window, self.on_map_click)

    def on_camera_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.expect != "camera":
            print("[INFO] 지금은 MAP에서 대응점을 찍어야 함")
            return

        # display 좌표 -> 원본 camera pixel 좌표
        original_x = x / CAM_DISPLAY_SCALE
        original_y = y / CAM_DISPLAY_SCALE

        if not (0 <= original_x < self.cam_w and 0 <= original_y < self.cam_h):
            print("[WARN] 카메라 이미지 범위 밖 클릭")
            return

        self.cam_points.append((original_x, original_y))
        self.expect = "map"

        print(f"[CAM ] Point {len(self.cam_points)} display=({x}, {y}) original=({original_x:.1f}, {original_y:.1f})")
        self.redraw()

    def on_map_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.expect != "map":
            print("[INFO] 지금은 CAMERA에서 점을 찍어야 함")
            return

        # display 좌표 -> 원본 map pixel 좌표
        # 정수로 반올림하면 지도 1픽셀(=5cm) 격자에 스냅되어 최대 3.5cm 오차가
        # 그냥 얹힌다. subpixel 로 보관해 재투영 오차를 줄인다.
        original_x = x / MAP_DISPLAY_SCALE
        original_y = y / MAP_DISPLAY_SCALE

        if not (0 <= original_x < self.map_w and 0 <= original_y < self.map_h):
            print("[WARN] 맵 이미지 범위 밖 클릭")
            return

        self.map_points.append((original_x, original_y))
        self.expect = "camera"

        print(f"[MAP ] Point {len(self.map_points)} display=({x}, {y}) original=({original_x:.2f}, {original_y:.2f})")
        self.redraw()

    def make_display_images(self):
        # 원본을 display용으로 resize
        cam_disp = cv2.resize(
            self.cam_original,
            None,
            fx=CAM_DISPLAY_SCALE,
            fy=CAM_DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )

        map_disp = cv2.resize(
            self.map_original,
            None,
            fx=MAP_DISPLAY_SCALE,
            fy=MAP_DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST,
        )

        # camera points draw
        for idx, (x, y) in enumerate(self.cam_points):
            dx = int(round(x * CAM_DISPLAY_SCALE))
            dy = int(round(y * CAM_DISPLAY_SCALE))

            cv2.circle(cam_disp, (dx, dy), CAM_POINT_RADIUS, (0, 0, 255), -1)
            cv2.putText(
                cam_disp,
                str(idx + 1),
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                CAM_FONT_SCALE,
                (0, 0, 255),
                CAM_THICKNESS,
                cv2.LINE_AA,
            )

        # map points draw
        for idx, (x, y) in enumerate(self.map_points):
            dx = int(round(x * MAP_DISPLAY_SCALE))
            dy = int(round(y * MAP_DISPLAY_SCALE))

            cv2.circle(map_disp, (dx, dy), MAP_POINT_RADIUS, (255, 0, 0), -1)
            cv2.putText(
                map_disp,
                str(idx + 1),
                (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                MAP_FONT_SCALE,
                (255, 0, 0),
                MAP_THICKNESS,
                cv2.LINE_AA,
            )

        cam_count = len(self.cam_points)
        map_count = len(self.map_points)
        pair_count = min(cam_count, map_count)

        next_idx = pair_count + 1
        next_target = "CAMERA" if self.expect == "camera" else "MAP"

        common_lines = [
            f"Next pair #{next_idx}: click on {next_target}",
            f"Pairs: {pair_count} | CAM points: {cam_count} | MAP points: {map_count}",
            "Keys: u=undo pair, r=reset, f=finish/save, q=quit",
        ]

        cam_lines = [f"{self.cam_name.upper()} CAMERA"] + common_lines
        map_lines = ["MAP IMAGE"] + common_lines

        draw_label_box(
            cam_disp,
            cam_lines,
            x=15,
            y=25,
            font_scale=CAM_FONT_SCALE,
            thickness=CAM_THICKNESS,
        )

        draw_label_box(
            map_disp,
            map_lines,
            x=15,
            y=25,
            font_scale=MAP_FONT_SCALE,
            thickness=MAP_THICKNESS,
        )

        return cam_disp, map_disp

    def redraw(self):
        cam_disp, map_disp = self.make_display_images()
        cv2.imshow(self.cam_window, cam_disp)
        cv2.imshow(self.map_window, map_disp)

    def undo_last_pair(self):
        # camera 점만 찍고 map 점을 아직 안 찍은 상태
        if len(self.cam_points) > len(self.map_points):
            removed_cam = self.cam_points.pop()
            self.expect = "camera"
            print(f"[UNDO CAM ONLY] cam={removed_cam}")

        # 완성된 pair가 있는 상태
        elif len(self.cam_points) == len(self.map_points) and len(self.cam_points) > 0:
            removed_cam = self.cam_points.pop()
            removed_map = self.map_points.pop()
            self.expect = "camera"
            print(f"[UNDO PAIR] cam={removed_cam}, map={removed_map}")

        else:
            print("[UNDO] nothing to undo")

        self.redraw()

    def reset_all(self):
        self.cam_points = []
        self.map_points = []
        self.expect = "camera"
        print("[RESET] all points cleared")
        self.redraw()

    def run(self):
        print("======================================")
        print("Pairwise Homography Point Selection")
        print("--------------------------------------")
        print("순서:")
        print("1) CAMERA에서 바닥 기준점 클릭")
        print("2) MAP에서 같은 대응점 클릭")
        print("3) 반복")
        print("--------------------------------------")
        print("u : 마지막 pair 취소")
        print("r : 전체 reset")
        print("f : finish/save")
        print("q : quit")
        print("======================================")
        print("")
        print("중요:")
        print("- 최소 4쌍 필요")
        print("- 가능하면 8쌍 이상 추천")
        print("- 바닥에 닿는 코너만 찍기")
        print("- 박스 윗면, 벽 위쪽, 창문, 로봇 위쪽 찍지 말기")
        print("======================================")

        self.redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u"):
                self.undo_last_pair()

            elif key == ord("r"):
                self.reset_all()

            elif key == ord("f"):
                if len(self.cam_points) != len(self.map_points):
                    print("[ERROR] camera/map point 개수가 다름. 아직 pair가 완성되지 않음.")
                    continue

                if len(self.cam_points) < 4:
                    print("[ERROR] 최소 4쌍 필요")
                    continue

                self.finished = True
                print("[FINISH] point selection completed")
                break

            elif key == ord("q"):
                print("[QUIT]")
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.cam_points, self.map_points


def compute_reprojection_error(H, src_pts, dst_pts):
    """
    camera point를 H로 변환했을 때 실제 map 좌표와 얼마나 차이나는지 계산.
    단위는 meter, 즉 ROS map 좌표 기준.
    """
    src_h = cv2.convertPointsToHomogeneous(src_pts).reshape(-1, 3).T
    projected = H @ src_h
    projected = projected[:2] / projected[2]

    projected = projected.T
    errors = np.linalg.norm(projected - dst_pts, axis=1)

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 scripts/02_make_homography_pairwise.py cam0")
        print("  python3 scripts/02_make_homography_pairwise.py cam1")
        sys.exit(1)

    cam_name = sys.argv[1].lower()

    if cam_name not in ["cam0", "cam1"]:
        print("[ERROR] cam_name must be cam0 or cam1")
        sys.exit(1)

    cam_img_path = CAPTURE_DIR / f"{cam_name}_ref.jpg"

    if not cam_img_path.exists():
        raise FileNotFoundError(f"Camera reference image not found: {cam_img_path}")

    cam_img = cv2.imread(str(cam_img_path))
    if cam_img is None:
        raise FileNotFoundError(f"Failed to load camera image: {cam_img_path}")

    map_img, resolution, origin, map_img_path = load_map_info(MAP_YAML)
    map_height = map_img.shape[0]

    print("======================================")
    print(f"Making homography for {cam_name}")
    print("--------------------------------------")
    print(f"Base dir     : {BASE_DIR}")
    print(f"Camera image : {cam_img_path}")
    print(f"Map yaml     : {MAP_YAML}")
    print(f"Map image    : {map_img_path}")
    print(f"Resolution   : {resolution}")
    print(f"Origin       : {origin}")
    print(f"Cam scale    : {CAM_DISPLAY_SCALE}")
    print(f"Map scale    : {MAP_DISPLAY_SCALE}")
    print("======================================")

    collector = PairwiseHomographyCollector(cam_img, map_img, cam_name)
    cam_points, map_pixel_points = collector.run()

    src_pts = np.array(cam_points, dtype=np.float32)

    dst_ros_points = []
    for u, v in map_pixel_points:
        x_ros, y_ros = map_pixel_to_ros(
            u,
            v,
            map_height=map_height,
            resolution=resolution,
            origin=origin,
        )
        dst_ros_points.append((x_ros, y_ros))

    dst_pts = np.array(dst_ros_points, dtype=np.float32)

    # dst_pts 단위가 meter 이므로 ransacReprojThreshold 도 meter 여야 한다.
    # 기본값 3.0 을 그냥 쓰면 "3m 이내면 정상"이 되어 RANSAC 이 아무것도
    # 걸러내지 못한다 (대응이 뒤바뀐 점도 inlier 로 통과).
    RANSAC_THRES_M = 0.15

    H, mask = cv2.findHomography(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRES_M
    )

    if H is None:
        print("[ERROR] Failed to compute homography.")
        sys.exit(1)

    inliers = mask.ravel().astype(bool)

    # inlier 만으로 최소자승 재적합 → RANSAC 이 고른 모델보다 정밀
    if inliers.sum() >= 4:
        H_refined, _ = cv2.findHomography(src_pts[inliers], dst_pts[inliers], method=0)
        if H_refined is not None:
            H = H_refined

    errors = compute_reprojection_error(H, src_pts, dst_pts)

    print("")
    print("======================================")
    print("점별 재투영 오차")
    print("--------------------------------------")
    print(" idx  cam pixel        map(clicked)     error")
    for i, err in enumerate(errors):
        tag = "" if inliers[i] else "   <<< OUTLIER (대응 확인!)"
        print(
            f" {i:3d}  ({src_pts[i][0]:6.1f},{src_pts[i][1]:6.1f})  "
            f"({dst_pts[i][0]:5.2f},{dst_pts[i][1]:5.2f})  {err * 100:6.1f} cm{tag}"
        )
    print("--------------------------------------")
    print(f"inlier {int(inliers.sum())}/{len(errors)}  (RANSAC {RANSAC_THRES_M * 100:.0f}cm)")

    if not inliers.all():
        print("")
        print("[WARN] OUTLIER 가 있습니다. 카메라 점과 지도 점이 서로 다른 코너에")
        print("       찍혔을 가능성이 큽니다. 다시 실행해 해당 쌍을 정확히 찍으세요.")
    print("======================================")

    npz_path = CALIB_DIR / f"{cam_name}_to_map.npz"
    json_path = CALIB_DIR / f"{cam_name}_points.json"

    np.savez(
        str(npz_path),
        H=H,
        camera_points=src_pts,
        map_ros_points=dst_pts,
        map_pixel_points=np.array(map_pixel_points, dtype=np.float32),
        resolution=resolution,
        origin=np.array(origin, dtype=np.float32),
        map_height=np.array([map_height], dtype=np.float32),
    )

    with open(json_path, "w") as f:
        json.dump(
            {
                "camera": cam_name,
                "camera_image": str(cam_img_path),
                "map_yaml": str(MAP_YAML),
                "map_image": str(map_img_path),
                "camera_points": cam_points,
                "map_pixel_points": map_pixel_points,
                "map_ros_points": dst_ros_points,
                "resolution": resolution,
                "origin": origin,
                "H": H.tolist(),
                "reprojection_error_m": errors.tolist(),
                "mean_error_m": float(np.mean(errors)),
                "max_error_m": float(np.max(errors)),
            },
            f,
            indent=2,
        )

    print("")
    print("======================================")
    print("[SUCCESS] Homography saved")
    print("--------------------------------------")
    print(f"NPZ  : {npz_path}")
    print(f"JSON : {json_path}")
    print("--------------------------------------")
    print(f"Point pairs      : {len(cam_points)}")
    print(f"Mean error       : {np.mean(errors):.4f} m")
    print(f"Max error        : {np.max(errors):.4f} m")
    print("======================================")
    print("")
    print("H =")
    print(H)
    print("")


if __name__ == "__main__":
    main()