"""
캘리브레이션 검증. H / P / Entry ROI 를 한 번에 점검한다.

  python3 verify_calib.py              # calibration/ (작업본)
  python3 verify_calib.py --detection  # (구 옵션 — 이제 무시된다)
  python3 verify_calib.py --cam cam0

검사 항목:
  [1] H 재투영 오차 [cm]     — inlier 평균 5cm 이하면 좋음
  [2] Entry ROI 가 신뢰 영역(캘리브 inlier 볼록껍질) 안에 있는가
  [3] P 재투영 오차 [px]     — 평균 5px 이하면 좋음
  [4] z 학습 구간이 판정 임계값(0.55/0.78/1.17m)을 포함하는가
  [5] z 민감도 — v(세로)가 u(가로)보다 훨씬 커야 정상

※ '외삽' 경고는 "여기 밖은 검증한 적 없다"는 뜻이지 "틀렸다"가 아니다.
   진짜 판정은 wallcheck.py 로 지도 벽을 카메라에 되찍어 눈으로 확인할 것.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물

ap = argparse.ArgumentParser()
ap.add_argument("--detection", action="store_true",
                help="구 옵션. 캘리브 폴더가 vision_pc3/calibration 하나로 통합되어 무시된다")
ap.add_argument("--cam", nargs="+", default=["cam0", "cam1"])
cli = ap.parse_args()

CAL = VISION_DIR / "calibration"   # 구 detection_final/ = 현 vision_pc3/ 로 통합됨
ROI = VISION_DIR / "calibration" / "entry_roi.json"

print(f"검사 대상: {CAL}")

for cam in cli.cam:
    hp = CAL / f"{cam}_to_map.npz"
    if not hp.exists():
        print(f"\n═══ {cam} ═══\n  ❌ {hp} 없음")
        continue

    print(f"\n═══ {cam} ═══")

    d = np.load(hp)
    H = d["H"]
    cp = d["camera_points"].astype(np.float32)
    mp = d["map_ros_points"].astype(np.float32)

    src = np.hstack([cp, np.ones((len(cp), 1), np.float32)])
    dst = (H @ src.T).T
    dst = dst[:, :2] / dst[:, 2:3]
    err = np.linalg.norm(dst - mp, axis=1)

    _, mask = cv2.findHomography(cp, mp, cv2.RANSAC, 0.15)
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(cp), bool)
    out = np.where(~inl)[0]

    print("[1] Homography H")
    print(f"    점 {len(cp)}개 (inlier {inl.sum()})  "
          f"{'✅' if inl.sum() >= 10 else '⚠ 10점 이상 권장'}")
    print(f"    inlier 오차: 평균 {err[inl].mean()*100:.1f} cm / 최대 {err[inl].max()*100:.1f} cm  "
          f"{'✅' if err[inl].mean() < 0.05 else '⚠ 5cm 초과' if err[inl].mean() < 0.10 else '❌ 다시'}")
    if len(out):
        print(f"    제외된 이상치: {out.tolist()}  ("
              + ", ".join(f"#{i}={err[i]*100:.0f}cm" for i in out) + ")")

    # ---- Entry ROI (cam0 만 의미 있음) ----
    if cam == "cam0" and ROI.exists():
        roi = np.array(json.loads(ROI.read_text())["points_map"], np.float32)
        hull = cv2.convexHull(mp[inl].reshape(-1, 1, 2))
        Hi = np.linalg.inv(H)
        n_in = n_vis = 0
        print("[2] Entry ROI")
        for i, p in enumerate(roi):
            in_hull = cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0
            w = Hi @ np.array([p[0], p[1], 1.0])
            u, v = w[:2] / w[2]
            vis = 0 <= u < 1280 and 0 <= v < 720
            n_in += in_hull
            n_vis += vis
            print(f"    R{i} map({p[0]:5.2f},{p[1]:5.2f}) -> cam({u:7.1f},{v:6.1f})  "
                  f"{'내삽 ✅' if in_hull else '외삽 ⚠ '}  {'화면안 ✅' if vis else '화면밖 ❌'}")
        print(f"    → 내삽 {n_in}/{len(roi)}, 화면 안 {n_vis}/{len(roi)}")

    # ---- P ----
    zp = CAL / f"{cam}_z_calib.npz"
    if not zp.exists():
        print("[3] z calibration: 아직 없음")
        continue

    z = np.load(zp)
    P = z["P"]
    wp = z["world_points"]
    ip = z["image_points"]

    xyz1 = np.hstack([wp, np.ones((len(wp), 1))])
    uvw = (P @ xyz1.T).T
    pred = uvw[:, :2] / uvw[:, 2:3]
    perr = np.linalg.norm(pred - ip, axis=1)

    # 재투영 오차를 u/v 로 분해한다.
    # 높이 정보는 v(세로)에만 실린다. u(가로)는 z 에 거의 반응하지 않는 축이라
    # 거기 오차가 커도 head_z 추정에는 영향이 없다. 전체 norm 만 보면
    # "오차 61px" 같은 숫자에 놀라게 되지만 실제 z 는 3cm 이내로 정확할 수 있다.
    du = np.abs(pred[:, 0] - ip[:, 0])
    dv = np.abs(pred[:, 1] - ip[:, 1])

    # 실제 판정 코드로 head_z 를 다시 풀어 참값(height_m)과 비교 = 진짜 지표
    sys.path.insert(0, str(VISION_DIR))
    from safety_lib import base_utils as bu
    hm = float(z["height_m"][0])
    zs = []
    for i in range(len(wp) // 2):
        ze, _r, _ = bu.estimate_z_on_vertical_line(P, wp[2*i][0], wp[2*i][1], ip[2*i+1])
        if ze is not None:
            zs.append(abs(ze - hm))
    zs = np.array(zs) if zs else np.array([9.9])

    print("[3] Projection P")
    print(f"    {len(wp)//2}쌍  재투영 평균 {perr.mean():.2f} px")
    print(f"      u(가로) {du.mean():5.2f} px   v(세로) {dv.mean():5.2f} px   "
          f"{'✅' if dv.mean() < 6.0 else '⚠ v 가 크다'}   <- 높이는 v 에만 실린다")
    print(f"    head_z 오차 (참값 {hm:.2f}m): 평균 {zs.mean()*100:.1f} cm / 최대 {zs.max()*100:.1f} cm  "
          f"{'✅' if zs.max() < 0.08 else '⚠ 8cm 초과' if zs.max() < 0.15 else '❌ 다시'}   <- 진짜 지표")

    zmin, zmax = wp[:, 2].min(), wp[:, 2].max()
    print(f"[4] z 학습 구간 [{zmin:.2f}, {zmax:.2f}] m  (height_m = {float(z['height_m'][0]):.2f})")
    for name, thr in [("very_low 0.55", 0.55), ("lying 0.78", 0.78), ("low_posture 1.17", 1.17)]:
        print(f"      {name:18s} {'✅ 구간 안' if zmin <= thr <= zmax else '❌ 구간 밖 (외삽)'}")

    X, Y = float(np.median(wp[:, 0])), float(np.median(wp[:, 1]))

    def proj(Z):
        w = P @ np.array([X, Y, Z, 1.0])
        return w[:2] / w[2]

    du = abs(proj(1.0)[0] - proj(0.0)[0])
    dv = abs(proj(1.0)[1] - proj(0.0)[1])
    print(f"[5] z 민감도 @ ({X:.2f},{Y:.2f}):  u {du:.1f} px/m,  v {dv:.1f} px/m  "
          f"(v/u = {dv/max(du,1e-6):.1f}배)  {'✅' if dv > du else '⚠ v 가 더 커야 정상'}")
