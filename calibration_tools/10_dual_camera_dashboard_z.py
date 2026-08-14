import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# Load V09 module dynamically
# ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
V09_PATH = SCRIPT_DIR / "09_safety_dashboard_z.py"

if not V09_PATH.exists():
    raise FileNotFoundError(f"09_safety_dashboard_z.py not found: {V09_PATH}")

spec = importlib.util.spec_from_file_location("safety_v09", str(V09_PATH))
v09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v09)


# ==============================
# Camera Processor
# ==============================
class CameraProcessor:
    def __init__(
        self,
        cam_id,
        cam_name,
        homography_path,
        z_calib_path,
        map_info,
        homography_output,
        tracker,
        width=1280,
        height=720,
        fps=15,
    ):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.map_info = map_info
        self.homography_output = homography_output
        self.tracker = tracker

        self.H, self.H_key = v09.load_homography(homography_path)
        self.P = v09.load_z_calibration(z_calib_path)

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open {cam_name}, camera id={cam_id}")

        self.frame_idx = 0
        self.prev_time = time.time()
        self.fps = 0.0

        self.last_preview = None
        self.last_tracked = []
        self.last_helmets = []
        self.last_metrics = {}

        print(f"[INFO] {cam_name}")
        print(f"  camera id  : {cam_id}")
        print(f"  H          : {homography_path}")
        print(f"  H key      : {self.H_key}")
        print(f"  z calib    : {z_calib_path}")

    def release(self):
        self.cap.release()

    def process(
        self,
        det_model,
        pose_model,
        det_names,
        args,
        now,
    ):
        ret, frame = self.cap.read()
        if not ret:
            print(f"[WARN] {self.cam_name} frame read failed")
            if self.last_preview is not None:
                return self.last_preview, self.last_tracked, self.last_metrics
            empty = np.zeros((720, 1280, 3), dtype=np.uint8)
            v09.draw_text(empty, f"{self.cam_name} frame read failed", (40, 80), (0, 0, 255), 1.0, 2)
            return empty, [], {}

        self.frame_idx += 1

        image_h, image_w = frame.shape[:2]
        image_area = image_h * image_w

        # inference every N frames
        run_inference = (self.frame_idx % args.process_every == 0) or self.last_preview is None

        if not run_inference:
            return self.last_preview, self.last_tracked, self.last_metrics

        # ==============================
        # Detection
        # ==============================
        det_result = det_model(
            frame,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=args.device,
            verbose=False,
        )[0]

        persons, helmets = v09.extract_detection_boxes(
            det_result,
            det_names,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            helmet_conf_thres=args.helmet_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        # ==============================
        # Pose
        # ==============================
        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_items = v09.extract_pose_items(pose_result)

        persons = v09.add_pose_only_persons(
            persons,
            pose_items,
            image_area=image_area,
            min_pose_area_ratio=args.person_min_area_ratio,
        )

        preview = frame.copy()

        # Draw helmet boxes
        for helmet in helmets:
            v09.draw_box(
                preview,
                helmet["box"],
                (0, 255, 255),
                f"helmet {helmet['conf']:.2f}",
                2,
            )

        detections_for_tracks = []

        # ==============================
        # Per person logic
        # ==============================
        for person in persons:
            pbox = person["box"]
            pose_item, pose_iou = v09.match_pose_to_person(pbox, pose_items)

            nearest_helmet, _ = v09.find_nearest_related_helmet(
                pbox,
                helmets,
                image_w,
                image_h,
            )

            nearest_helmet_box = nearest_helmet["box"] if nearest_helmet is not None else None

            head_px, head_src = v09.get_head_point(
                pose_item,
                person_box=pbox,
                helmet_box=nearest_helmet_box,
                conf_thres=args.pose_conf,
            )

            foot_px, foot_src = v09.get_foot_point(
                pose_item,
                pbox,
                conf_thres=args.pose_conf,
            )

            foot_map = v09.camera_pixel_to_map_meter(
                self.H,
                foot_px,
                self.map_info,
                self.homography_output,
            )

            head_z = None
            z_residual = None
            projected_uv = None

            if head_px is not None and foot_map is not None:
                head_z, z_residual, projected_uv = v09.estimate_z_on_vertical_line(
                    self.P,
                    foot_map[0],
                    foot_map[1],
                    head_px,
                )

            helmet_status, head_radius, related_helmets = v09.judge_helmet_status(
                pbox,
                helmets,
                head_px,
                head_src,
                image_w,
                image_h,
                head_radius_scale=args.head_radius_scale,
                helmet_expand_ratio=args.helmet_expand,
            )

            lying_candidate, posture, debug = v09.judge_posture_v09(
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
                "camera": self.cam_name,
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

        tracked = self.tracker.update(detections_for_tracks, now)

        # ==============================
        # Draw tracked results
        # ==============================
        for det in tracked:
            pbox = det["person_box"]
            tid = det["track_id"]
            helmet_status = det["helmet_status"]
            posture = det["posture"]
            emergency = det["emergency"]
            debug = det["debug"]

            color = v09.color_for_state(helmet_status, posture, emergency)

            label = f"{self.cam_name}-ID{tid} {helmet_status} {posture}"

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

            v09.draw_box(preview, pbox, color, label, 2)

            # head circle
            head_px = det.get("head_px")
            head_radius = det.get("head_radius", 40)

            if head_px is not None:
                hp = tuple(head_px.astype(int))
                cv2.circle(preview, hp, int(head_radius), (255, 0, 255), 2)
                cv2.circle(preview, hp, 4, (255, 0, 255), -1)
                v09.draw_text(
                    preview,
                    f"head:{det.get('head_src')}",
                    (hp[0] + 8, hp[1] - 8),
                    (255, 0, 255),
                    0.42,
                    1,
                )

            # foot point
            foot_px = det.get("foot_px")
            if foot_px is not None:
                fp = tuple(foot_px.astype(int))
                cv2.circle(preview, fp, 6, (255, 255, 0), -1)

            # projected head
            projected_uv = det.get("projected_uv")
            if projected_uv is not None and np.all(np.isfinite(projected_uv)):
                pp = tuple(projected_uv.astype(int))
                cv2.circle(preview, pp, 5, (0, 165, 255), -1)
                if head_px is not None:
                    cv2.line(preview, tuple(head_px.astype(int)), pp, (0, 165, 255), 2)

            # torso line
            pose_item_tmp, _ = v09.match_pose_to_person(pbox, pose_items)
            torso_horizontal, shoulder_mid, hip_mid = v09.get_torso_horizontal(
                pose_item_tmp,
                args.pose_conf,
            )

            if shoulder_mid is not None and hip_mid is not None:
                cv2.line(
                    preview,
                    tuple(shoulder_mid.astype(int)),
                    tuple(hip_mid.astype(int)),
                    (255, 0, 0),
                    2,
                )

            x1, y1, x2, y2 = map(int, pbox)
            reason_text = ",".join(debug.get("reasons", [])[:2])
            guard_text = ",".join(debug.get("upright_reasons", [])[:2])

            v09.draw_text(
                preview,
                f"L/U={debug.get('lying_score')}/{debug.get('upright_score')} ar={debug.get('aspect'):.2f}",
                (x1, min(image_h - 10, y2 + 20)),
                color,
                0.40,
                1,
            )

            if reason_text:
                v09.draw_text(
                    preview,
                    f"L:{reason_text}",
                    (x1, min(image_h - 10, y2 + 38)),
                    color,
                    0.38,
                    1,
                )

            if guard_text:
                v09.draw_text(
                    preview,
                    f"U:{guard_text}",
                    (x1, min(image_h - 10, y2 + 56)),
                    (0, 255, 255),
                    0.38,
                    1,
                )

            map_xy = det.get("map_xy")
            if map_xy is not None:
                v09.draw_text(
                    preview,
                    f"map=({map_xy[0]:.2f},{map_xy[1]:.2f})",
                    (x1, min(image_h - 10, y2 + 74)),
                    color,
                    0.38,
                    1,
                )

        # Camera label panel
        overlay = preview.copy()
        cv2.rectangle(overlay, (15, 15), (420, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, preview, 0.45, 0, preview)

        v09.draw_text(
            preview,
            f"{self.cam_name} | persons:{len(tracked)} helmets:{len(helmets)} FPS:{self.fps:.1f}",
            (28, 50),
            (255, 255, 255),
            0.70,
            2,
        )

        # FPS update
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            self.fps = self.fps * 0.90 + (1.0 / dt) * 0.10

        metrics = calc_metrics(tracked, helmets)
        metrics["fps"] = self.fps
        metrics["cam_name"] = self.cam_name

        self.last_preview = preview
        self.last_tracked = tracked
        self.last_helmets = helmets
        self.last_metrics = metrics

        return preview, tracked, metrics


# ==============================
# Metrics / Status
# ==============================
def calc_metrics(tracked, helmets):
    return {
        "person_count": len(tracked),
        "helmet_count": len(helmets),
        "no_helmet_count": sum(1 for d in tracked if d["helmet_status"] == "NO_HELMET"),
        "unknown_helmet_count": sum(1 for d in tracked if d["helmet_status"] == "UNKNOWN"),
        "low_posture_count": sum(1 for d in tracked if d["posture"] in ["LOW_POSTURE", "CROUCH_BOW"]),
        "lying_count": sum(1 for d in tracked if d["posture"] == "LYING_CANDIDATE"),
        "emergency_count": sum(1 for d in tracked if d["emergency"]),
    }


def draw_combined_map(map_img, all_detections, map_info):
    out = map_img.copy()

    for det in all_detections:
        map_xy = det.get("map_xy")
        if map_xy is None:
            continue

        mx, my = map_xy
        px, py = v09.map_meter_to_image_pixel(mx, my, map_info)

        if not (0 <= px < map_info["width"] and 0 <= py < map_info["height"]):
            continue

        color = v09.color_for_state(
            det.get("helmet_status", "UNKNOWN"),
            det.get("posture", "NORMAL"),
            det.get("emergency", False),
        )

        radius = 12 if det.get("emergency", False) else 7

        cv2.circle(out, (px, py), radius, color, -1)
        cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 2)

        label = f"{det.get('camera', '?')}-ID{det.get('track_id', '?')}"
        if det.get("emergency", False):
            label += " EMG"
        elif det.get("posture") == "LYING_CANDIDATE":
            label += " LYING"
        elif det.get("helmet_status") == "NO_HELMET":
            label += " NO_HELMET"

        v09.draw_text(out, label, (px + 10, py - 10), color, 0.45, 1)

    # map title
    overlay = out.copy()
    cv2.rectangle(overlay, (15, 15), (520, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    v09.draw_text(
        out,
        "MAP VIEW | combined cam0 + cam1 targets",
        (28, 50),
        (255, 255, 255),
        0.65,
        2,
    )

    return out


def draw_status_panel(
    width,
    height,
    metrics0,
    metrics2,
    all_detections,
    overall_fps,
    state_json_path,
    args,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    total_persons = metrics0.get("person_count", 0) + metrics2.get("person_count", 0)
    total_helmets = metrics0.get("helmet_count", 0) + metrics2.get("helmet_count", 0)
    total_no_helmet = metrics0.get("no_helmet_count", 0) + metrics2.get("no_helmet_count", 0)
    total_lying = metrics0.get("lying_count", 0) + metrics2.get("lying_count", 0)
    total_emergency = metrics0.get("emergency_count", 0) + metrics2.get("emergency_count", 0)

    y = 35
    v09.draw_text(panel, "DUAL CAMERA STATUS V10", (20, y), (255, 255, 255), 0.72, 2)

    y += 38
    v09.draw_text(panel, f"Overall FPS: {overall_fps:.1f} | Device: {args.device}", (20, y), (200, 200, 200), 0.55, 1)

    y += 32
    v09.draw_text(panel, f"Model: {Path(args.det_model).name}", (20, y), (160, 160, 160), 0.48, 1)

    y += 42
    v09.draw_text(panel, "TOTAL", (20, y), (255, 255, 255), 0.65, 2)

    y += 32
    v09.draw_text(panel, f"Persons: {total_persons}", (35, y), (255, 255, 255), 0.58, 2)

    y += 28
    v09.draw_text(panel, f"Helmets: {total_helmets}", (35, y), (255, 255, 255), 0.58, 2)

    y += 28
    color = (0, 0, 255) if total_no_helmet > 0 else (0, 255, 0)
    v09.draw_text(panel, f"No Helmet: {total_no_helmet}", (35, y), color, 0.58, 2)

    y += 28
    color = (0, 165, 255) if total_lying > 0 else (0, 255, 0)
    v09.draw_text(panel, f"Lying Candidate: {total_lying}", (35, y), color, 0.58, 2)

    y += 32
    color = (0, 0, 255) if total_emergency > 0 else (0, 255, 0)
    v09.draw_text(panel, f"EMERGENCY: {total_emergency}", (35, y), color, 0.72, 2)

    y += 50
    draw_camera_metrics(panel, "CAM0", metrics0, 20, y)
    y += 150
    draw_camera_metrics(panel, "CAM1", metrics2, 20, y)

    y += 165
    v09.draw_text(panel, "EMERGENCY TARGETS", (20, y), (255, 255, 255), 0.58, 2)
    y += 28

    emergency_targets = [d for d in all_detections if d.get("emergency", False) and d.get("map_xy") is not None]

    if not emergency_targets:
        v09.draw_text(panel, "None", (35, y), (0, 255, 0), 0.55, 2)
        y += 28
    else:
        for det in emergency_targets[:5]:
            mx, my = det["map_xy"]
            line = f"{det.get('camera')}-ID{det.get('track_id')} -> ({mx:.2f}, {my:.2f})"
            v09.draw_text(panel, line, (35, y), (255, 0, 255), 0.47, 1)
            y += 24

    y = height - 130
    v09.draw_text(panel, "Legend", (20, y), (255, 255, 255), 0.52, 2)
    y += 24

    draw_legend_item(panel, 35, y, (0, 255, 0), "helmet on / normal")
    y += 22
    draw_legend_item(panel, 35, y, (0, 0, 255), "no helmet")
    y += 22
    draw_legend_item(panel, 35, y, (0, 165, 255), "lying candidate / low posture")
    y += 22
    draw_legend_item(panel, 35, y, (255, 0, 255), "emergency")

    y = height - 28
    v09.draw_text(panel, f"state: {state_json_path}", (20, y), (150, 150, 150), 0.35, 1)

    return panel


def draw_camera_metrics(panel, title, m, x, y):
    v09.draw_text(panel, title, (x, y), (255, 255, 255), 0.58, 2)

    y += 27
    v09.draw_text(panel, f"FPS: {m.get('fps', 0):.1f}", (x + 15, y), (180, 180, 180), 0.45, 1)

    y += 23
    v09.draw_text(panel, f"Persons: {m.get('person_count', 0)}", (x + 15, y), (255, 255, 255), 0.47, 1)

    y += 23
    v09.draw_text(panel, f"Helmets: {m.get('helmet_count', 0)}", (x + 15, y), (255, 255, 255), 0.47, 1)

    y += 23
    color = (0, 0, 255) if m.get("no_helmet_count", 0) > 0 else (0, 255, 0)
    v09.draw_text(panel, f"No Helmet: {m.get('no_helmet_count', 0)}", (x + 15, y), color, 0.47, 1)

    y += 23
    color = (0, 165, 255) if m.get("lying_count", 0) > 0 else (0, 255, 0)
    v09.draw_text(panel, f"Lying: {m.get('lying_count', 0)}", (x + 15, y), color, 0.47, 1)

    y += 23
    color = (0, 0, 255) if m.get("emergency_count", 0) > 0 else (0, 255, 0)
    v09.draw_text(panel, f"Emergency: {m.get('emergency_count', 0)}", (x + 15, y), color, 0.47, 1)


def draw_legend_item(panel, x, y, color, text):
    cv2.circle(panel, (x, y - 5), 6, color, -1)
    v09.draw_text(panel, text, (x + 18, y), (180, 180, 180), 0.38, 1)


def save_dual_state_json(path, timestamp, detections0, detections2, metrics0, metrics2):
    all_detections = list(detections0) + list(detections2)

    persons = []
    emergency_targets = []

    for det in all_detections:
        map_xy = det.get("map_xy")
        debug = det.get("debug", {})

        item = {
            "camera": det.get("camera"),
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
        "mode": "dual_camera_v10",
        "metrics": {
            "cam0": metrics0,
            "cam1": metrics2,
            "total_person_count": metrics0.get("person_count", 0) + metrics2.get("person_count", 0),
            "total_emergency_count": metrics0.get("emergency_count", 0) + metrics2.get("emergency_count", 0),
        },
        "persons": persons,
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

    parser.add_argument("--cam0-id", type=int, default=2)
    parser.add_argument("--cam0-name", type=str, default="cam0")
    parser.add_argument("--cam0-homography", type=str, default="calibration/cam0_to_map.npz")
    parser.add_argument("--cam0-z-calib", type=str, default="calibration/cam0_z_calib.npz")

    parser.add_argument("--cam1-id", type=int, default=4)
    parser.add_argument("--cam1-name", type=str, default="cam1")
    parser.add_argument("--cam1-homography", type=str, default="calibration/cam1_to_map.npz")
    parser.add_argument("--cam1-z-calib", type=str, default="calibration/cam1_z_calib.npz")

    parser.add_argument("--map-yaml", type=str, default="final_project.yaml")
    parser.add_argument("--homography-output", type=str, default="map_meters", choices=["map_meters", "map_pixels"])

    parser.add_argument("--det-model", type=str, default="yolo_experiments/v8n_640_e50/weights/best.pt")
    parser.add_argument("--pose-model", type=str, default="yolo11n-pose.pt")

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)

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

    parser.add_argument("--display-w", type=int, default=1600)
    parser.add_argument("--display-h", type=int, default=900)

    parser.add_argument("--process-every", type=int, default=1)

    parser.add_argument(
        "--state-json",
        type=str,
        default="outputs/safety_state_dual_v10.json",
    )

    args = parser.parse_args()

    print("======================================")
    print("Dual Camera Safety Dashboard V10")
    print("--------------------------------------")
    print(f"cam0 id     : {args.cam0_id}")
    print(f"cam1 id     : {args.cam1_id}")
    print(f"det model   : {args.det_model}")
    print(f"pose model  : {args.pose_model}")
    print(f"device      : {args.device}")
    print(f"imgsz       : {args.imgsz}")
    print(f"process every N frames: {args.process_every}")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()
    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    map_info = v09.load_map_from_yaml(args.map_yaml)

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(args.pose_model)
    det_names = det_model.names

    tracker0 = v09.TrackManager(
        max_distance_px=args.track_max_distance,
        max_age_sec=args.track_max_age,
        emergency_sec=args.emergency_sec,
    )

    tracker2 = v09.TrackManager(
        max_distance_px=args.track_max_distance,
        max_age_sec=args.track_max_age,
        emergency_sec=args.emergency_sec,
    )

    cam0 = CameraProcessor(
        cam_id=args.cam0_id,
        cam_name=args.cam0_name,
        homography_path=args.cam0_homography,
        z_calib_path=args.cam0_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        tracker=tracker0,
    )

    cam1 = CameraProcessor(
        cam_id=args.cam1_id,
        cam_name=args.cam1_name,
        homography_path=args.cam1_homography,
        z_calib_path=args.cam1_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        tracker=tracker2,
    )

    window_name = "Dual Camera Safety Dashboard V10"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_loop_time = time.time()
    overall_fps = 0.0

    try:
        while True:
            now = time.time()

            cam0_view, cam0_tracked, cam0_metrics = cam0.process(
                det_model,
                pose_model,
                det_names,
                args,
                now,
            )

            cam1_view, cam1_tracked, cam1_metrics = cam1.process(
                det_model,
                pose_model,
                det_names,
                args,
                now,
            )

            all_detections = list(cam0_tracked) + list(cam1_tracked)

            save_dual_state_json(
                args.state_json,
                timestamp=now,
                detections0=cam0_tracked,
                detections2=cam1_tracked,
                metrics0=cam0_metrics,
                metrics2=cam1_metrics,
            )

            map_view = draw_combined_map(
                map_info["img"],
                all_detections,
                map_info,
            )

            dt = now - prev_loop_time
            prev_loop_time = now

            if dt > 0:
                overall_fps = overall_fps * 0.90 + (1.0 / dt) * 0.10

            status_panel = draw_status_panel(
                width=args.display_w // 2,
                height=args.display_h // 2,
                metrics0=cam0_metrics,
                metrics2=cam1_metrics,
                all_detections=all_detections,
                overall_fps=overall_fps,
                state_json_path=args.state_json,
                args=args,
            )

            q_w = args.display_w // 2
            q_h = args.display_h // 2

            cam0_panel = v09.resize_keep_ratio(cam0_view, q_w, q_h)
            cam1_panel = v09.resize_keep_ratio(cam1_view, q_w, q_h)
            map_panel = v09.resize_keep_ratio(map_view, q_w, q_h)
            status_panel = v09.resize_keep_ratio(status_panel, q_w, q_h)

            top = np.hstack([cam0_panel, cam1_panel])
            bottom = np.hstack([map_panel, status_panel])
            dashboard = np.vstack([top, bottom])

            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                tracker0.reset()
                tracker2.reset()
                print("[RESET] all tracks reset")

    finally:
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
