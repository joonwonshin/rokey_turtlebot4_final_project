"""
호모그래피 최종 육안 검증.

지도(pgm)의 벽 셀을 H 의 역행렬로 카메라 화면에 되찍는다.
빨간 점이 실제 골판지 벽 '밑동'에 정확히 얹히면 H 가 맞는 것이다.

숫자(재투영 오차)는 찍은 점들에 대한 값이라, 그 점들 근처만 맞아도 작게 나온다.
이 도구는 화면 전체에서 맞는지를 본다. 이게 진짜 판정이다.

  python3 wallcheck.py --cam cam0 --device 2
  python3 wallcheck.py --cam cam1 --device 0
  python3 wallcheck.py --cam cam0 --device 2 --live   # 실시간

키(--live):  q 종료
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물

ap = argparse.ArgumentParser()
ap.add_argument("--cam", required=True, choices=["cam0", "cam1"])
ap.add_argument("--device", type=int, required=True)
ap.add_argument("--live", action="store_true")
ap.add_argument("--detection", action="store_true",
                help="구 옵션. 캘리브 폴더가 vision_pc3/calibration 하나로 통합되어 무시된다")
cli = ap.parse_args()

CAL = VISION_DIR / "calibration"   # 구 detection_final/ = 현 vision_pc3/ 로 통합됨

y = yaml.safe_load(open(VISION_DIR / "final_project.yaml"))
res = float(y["resolution"])
ox, oy, _ = y["origin"]
g = cv2.imread(str(VISION_DIR / "final_project.pgm"), cv2.IMREAD_GRAYSCALE)
h, w = g.shape

occ = np.argwhere(g < 100)                       # 벽 셀 (row, col)
walls = np.stack([ox + occ[:, 1] * res,
                  oy + (h - occ[:, 0]) * res], axis=1)

d = np.load(CAL / f"{cli.cam}_to_map.npz")
H = d["H"].astype(np.float64)
Hi = np.linalg.inv(H)
cam_pts = d["camera_points"].astype(np.float32)
map_pts = d["map_ros_points"].astype(np.float32)
_, mask = cv2.findHomography(cam_pts, map_pts, cv2.RANSAC, 0.15)
inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(cam_pts), bool)

src = np.hstack([walls, np.ones((len(walls), 1))])
proj = (Hi @ src.T).T
ok = np.abs(proj[:, 2]) > 1e-9
uv = proj[ok][:, :2] / proj[ok][:, 2:3]
vis = (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 720)
uv = uv[vis].astype(int)

cap = cv2.VideoCapture(cli.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
if not cap.isOpened():
    raise SystemExit(f"[ERROR] /dev/video{cli.device} 열기 실패")

WIN = f"WALL CHECK  {cli.cam}  (/dev/video{cli.device})"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

print("=" * 66)
print(f"{cli.cam}  H: {CAL}/{cli.cam}_to_map.npz")
print(f"  캘리브 점 {len(cam_pts)}개 (inlier {inl.sum()})")
print(f"  화면 안에 들어온 지도 벽 셀 {len(uv)}개")
print("-" * 66)
print("  빨간 점이 실제 벽 '밑동' 에 얹히면 H 가 맞는 것")
print("  허공에 뜨거나 벽에서 벗어나면 재캘리브 필요")
print("=" * 66)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    view = frame.copy()
    for u, v in uv:
        cv2.circle(view, (u, v), 2, (0, 0, 255), -1)

    # 캘리브에 쓴 점 (초록=inlier, 파랑=이상치)
    for i, (u, v) in enumerate(cam_pts):
        c = (0, 220, 0) if inl[i] else (255, 100, 0)
        cv2.circle(view, (int(u), int(v)), 5, c, -1)
        cv2.circle(view, (int(u), int(v)), 7, (0, 0, 0), 1)

    for i, s in enumerate([
        f"{cli.cam}: red = map walls reprojected",
        "green = calib inliers,  blue = outliers",
    ]):
        o = (16, 34 + i * 28)
        cv2.putText(view, s, o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, s, o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WIN, view)

    if not cli.live:
        p = BASE / f"captures/wallcheck_{cli.cam}.jpg"
        cv2.imwrite(str(p), view)
        print(f"[SAVED] {p}")
        cv2.waitKey(0)
        break

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
