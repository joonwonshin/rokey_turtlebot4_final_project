"""
쓰러짐 판정 실시간 진단.

메인과 똑같은 CameraEntryTracker 를 쓰되, 판정 근거를 전부 화면에 띄운다.
어떤 지표가 임계값을 못 넘겨서 놓치는지 눈으로 본다.

  python3 debug_posture.py --cam cam0 --device 2
  python3 debug_posture.py --cam cam1 --device 0
  python3 debug_posture.py --cam cam0 --device 2 --log fall.csv

조작:
  마우스 우클릭 : 화면 정지 / 해제  (정지 중에도 판정값은 그대로 보임)
  q : 종료
  l : CSV 로그 on/off
  s : 현재 화면 저장
"""
import argparse
import csv
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

BASE = Path.home() / "turtlebot4_ws" / "final_project"
DET = BASE / "detection_final"
sys.path.insert(0, str(DET))

from safety_lib import base_utils as utils          # noqa: E402
from safety_lib.safety_logic import WorkerStateStore  # noqa: E402
from safety_lib.vision_core import CameraEntryTracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cam", required=True, choices=["cam0", "cam1"])
ap.add_argument("--device", type=int, required=True)
ap.add_argument("--imgsz", type=int, default=None)
ap.add_argument("--device-gpu", type=str, default="0")
ap.add_argument("--leg-angle-thres", type=float, default=20.0)
ap.add_argument("--torso-angle-thres", type=float, default=25.0)
ap.add_argument("--hip-low-thres", type=float, default=0.70)
ap.add_argument("--leg-upright-deg", type=float, default=8.0)
ap.add_argument("--hip-upright-m", type=float, default=0.55)
ap.add_argument("--torso-short-m", type=float, default=0.34)
ap.add_argument("--body-len-min-m", type=float, default=0.85)
ap.add_argument("--hip-invert-m", type=float, default=-0.35)
ap.add_argument("--log", type=str, default="")
cli = ap.parse_args()

imgsz = cli.imgsz or (1280 if cli.cam == "cam0" else 960)

args = Namespace(
    imgsz=imgsz, device=cli.device_gpu,
    det_conf=0.20, person_conf=0.40, helmet_conf=0.25, pose_conf=0.25,
    person_min_area_ratio=0.002, person_min_height=35,
    head_radius_scale=0.20, helmet_expand=1.00,
    user_height_m=1.80,
    lying_height_thres=0.78, very_low_height_thres=0.55, residual_thres=90.0,
    leg_angle_thres=cli.leg_angle_thres,
    torso_angle_thres=cli.torso_angle_thres,
    hip_low_thres=cli.hip_low_thres,
    leg_upright_deg=cli.leg_upright_deg,
    hip_upright_m=cli.hip_upright_m,
    torso_short_m=cli.torso_short_m,
    body_len_min_m=cli.body_len_min_m,
    hip_invert_m=cli.hip_invert_m,
    hand_raise_margin=20.0, process_every=1,
    emergency_sec=5.0, recovery_sec=3.0, entry_check_sec=3.0,
)

map_info = utils.load_map_from_yaml(str(DET / "final_project.yaml"))
store = WorkerStateStore(
    camera_name=cli.cam, entry_roi_points=None,
    hold_sec=0.8, confirm_frames=2, max_age_sec=3.0, entry_enabled=False,
)
tracker = CameraEntryTracker(
    cam_id=cli.device, cam_name=cli.cam,
    det_model_path=DET / "yolo_experiments/best.pt",
    pose_model_path=DET / "yolo11s-pose.pt",
    homography_path=str(DET / f"calibration/{cli.cam}_to_map.npz"),
    z_calib_path=str(DET / f"calibration/{cli.cam}_z_calib.npz"),
    map_info=map_info, homography_output="map_meters",
    worker_store=store, tracker_yaml="bytetrack.yaml", imgsz=imgsz,
)

writer = fp = None
if cli.log:
    fp = open(cli.log, "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(["t", "posture", "lying", "leg", "torso", "hip_h",
                     "segs", "aspect", "kpt_ratio", "visible_kpts",
                     "head_z", "z_residual", "reasons"])
logging_on = False

# 우클릭으로 화면을 얼린다. 정지 중에도 마지막 판정 결과를 계속 그려서
# 어떤 지표가 임계값을 못 넘겼는지 천천히 볼 수 있다.
state = {"frozen": False}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        state["frozen"] = not state["frozen"]
        print("[FROZEN]" if state["frozen"] else "[RESUME]")


WIN = f"POSTURE DEBUG  {cli.cam}"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_mouse)


def txt(img, s, org, color, sc=0.5, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)


print("=" * 70)
print(f"{cli.cam}  /dev/video{cli.device}  imgsz={imgsz}")
print(f"  LEG > {cli.leg_angle_thres}  AND  TORSO > {cli.torso_angle_thres}  ->  LYING")
print("  q 종료 / l 로그 / SPACE 정지")
print("=" * 70)

try:
    last = None

    while True:
        now = time.time()

        if state["frozen"] and last is not None:
            # 카메라는 계속 읽어서 V4L2 버퍼가 밀리지 않게 하되,
            # 화면과 판정값은 얼린 프레임 것을 쓴다.
            tracker.cap.grab()
            view, dets, metrics = last
        else:
            view, dets, metrics = tracker.process(args, now)
            last = (view, dets, metrics)

        view = view.copy()
        h, w = view.shape[:2]

        panel = np.zeros((h, 520, 3), np.uint8)
        y = 30
        txt(panel, f"{cli.cam}  FPS {metrics.get('fps', 0):.1f}  persons {len(dets)}",
            (12, y), (255, 255, 255), 0.6, 2)

        for det in dets[:2]:
            d = det.get("debug", {})
            leg = d.get("leg_angle_deg")
            tor = d.get("torso_angle_deg")
            hip = d.get("hip_above_ankle_m")
            segs = d.get("leg_segments_deg")

            y += 34
            cv2.line(panel, (8, y - 16), (512, y - 16), (70, 70, 70), 1)
            lying = det.get("lying_candidate", False)
            emg = det.get("emergency", False)
            col = (0, 0, 255) if emg else ((0, 165, 255) if lying else (0, 255, 0))
            txt(panel, f"#{det.get('display_id')}  {det.get('posture')}"
                       f"{'  EMERGENCY' if emg else ''}", (12, y), col, 0.62, 2)

            f = lambda v, n=5, p=1: (f"%{n}.{p}f" % v) if v is not None else " " * (n - 1) + "-"

            y += 26
            c = (0, 255, 0) if (leg is not None and leg > cli.leg_angle_thres) else (0, 80, 255)
            txt(panel, f"LEG   {f(leg)} deg  > {cli.leg_angle_thres:.0f} ?", (18, y), c, 0.55, 2)
            y += 24
            c = (0, 255, 0) if (tor is not None and tor > cli.torso_angle_thres) else (0, 80, 255)
            txt(panel, f"TORSO {f(tor)} deg  > {cli.torso_angle_thres:.0f} ?  <- 주지표",
                (18, y), c, 0.55, 2)
            y += 24
            up = d.get("legs_upright")
            txt(panel, f"legs_upright(거부권) : {up}", (18, y),
                (0, 80, 255) if up else (0, 255, 0), 0.5, 2)

            # ── 사각지대용 길이 지표 ──────────────────────────────
            # 각도(TORSO)가 죽는 방향에서 이 둘 중 하나가 살아난다.
            y += 28
            tl = d.get("torso_len_m")
            bl = d.get("body_len_m")
            ext = bl is not None and bl > cli.body_len_min_m
            txt(panel, f"몸통보임 {f(tl, 5, 2)} m  (실제 0.52, < {cli.torso_short_m:.2f} = 압축)",
                (18, y), (0, 255, 0) if d.get("torso_foreshortened") else (170, 170, 170), 0.46, 1)
            y += 20
            txt(panel, f"몸길이보임 {f(bl, 5, 2)} m  (실제 1.65, > {cli.body_len_min_m:.2f} = 펴짐)",
                (18, y), (0, 255, 0) if ext else (0, 80, 255), 0.46, 1)

            y += 26
            a1 = d.get("angle_says_lying")
            a2 = d.get("torso_foreshortened")
            a3 = d.get("hip_inverted")
            for lbl, v in (("(1) 몸통 기울어짐", a1),
                           ("(2) 몸통 원근압축", a2),
                           ("(3) 엉덩이 반전", a3)):
                txt(panel, f"  {lbl} : {v}", (18, y),
                    (0, 0, 255) if v else (110, 110, 110), 0.44, 1)
                y += 19
            lying_now = det.get("lying_candidate", False)
            y += 4
            txt(panel, f"LYING = (1) or (2) or (3) : {lying_now}", (18, y),
                (0, 0, 255) if lying_now else (150, 150, 150), 0.55, 2)

            y += 26
            if segs:
                txt(panel, "segs " + " ".join("  - " if v is None else "%4.0f" % v for v in segs),
                    (18, y), (170, 170, 170), 0.44)
                y += 18
                txt(panel, "      허벅L 정강L 허벅R 정강R", (18, y), (120, 120, 120), 0.38)
                y += 18
                lL = d.get("leg_left_deg")
                lR = d.get("leg_right_deg")
                lmin = d.get("leg_min_deg")
                txt(panel, f"legL {f(lL)}  legR {f(lR)}   min {f(lmin)}"
                           f"  (< {cli.leg_upright_deg:.0f} = 서있음)",
                    (18, y), (170, 170, 170), 0.42)
            y += 24
            txt(panel, f"hip_h {f(hip, 5, 2)} m  (< {cli.hip_low_thres:.2f} -> LOW)",
                (18, y), (170, 170, 170), 0.44)
            y += 22
            txt(panel, f"aspect {d.get('aspect', 0):.2f}  kpt {d.get('kpt_ratio', 0):.2f}"
                       f"  kpts {d.get('visible_kpts', 0)}", (18, y), (170, 170, 170), 0.44)
            y += 22
            hz = d.get("head_z")
            rz = d.get("z_residual")
            txt(panel, f"head_z {f(hz, 5, 2)}  resid {f(rz, 6, 0)}  trusted {d.get('z_trusted')}",
                (18, y), (150, 150, 150), 0.42)
            y += 22
            for r in d.get("reasons", [])[-2:]:
                txt(panel, f"- {r}", (18, y), (140, 200, 140), 0.40)
                y += 18

            if writer and logging_on:
                writer.writerow([
                    f"{now:.2f}", det.get("posture"), lying,
                    "" if leg is None else f"{leg:.1f}",
                    "" if tor is None else f"{tor:.1f}",
                    "" if hip is None else f"{hip:.2f}",
                    "|".join("" if v is None else f"{v:.0f}" for v in (segs or [])),
                    f"{d.get('aspect', 0):.2f}", f"{d.get('kpt_ratio', 0):.2f}",
                    d.get("visible_kpts"),
                    "" if hz is None else f"{hz:.2f}",
                    "" if rz is None else f"{rz:.0f}",
                    "|".join(d.get("reasons", [])),
                ])
                fp.flush()

        if not dets:
            txt(panel, "no person", (14, 70), (120, 120, 120), 0.6)

        if writer:
            txt(panel, "REC" if logging_on else "log off", (430, 30),
                (0, 0, 255) if logging_on else (100, 100, 100), 0.6, 2)

        if state["frozen"]:
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 200, 255), 6)
            txt(view, "FROZEN  (right-click to resume)", (16, 40), (0, 200, 255), 0.8, 2)

        out = np.hstack([view, panel])
        cv2.imshow(WIN, out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("l") and writer:
            logging_on = not logging_on
            print(f"[LOG] {'시작' if logging_on else '정지'}")
        if k == ord("s"):
            p = BASE / f"captures/posture_{cli.cam}_{int(now)}.jpg"
            cv2.imwrite(str(p), out)
            print(f"[SAVED] {p}")
finally:
    tracker.release()
    cv2.destroyAllWindows()
    if fp:
        fp.close()
        print(f"[LOG] 저장: {cli.log}")
