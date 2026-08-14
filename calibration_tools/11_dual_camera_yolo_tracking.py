import argparse
import importlib.util
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# Load V09 utility functions
# ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
V09_PATH = SCRIPT_DIR / "09_safety_dashboard_z.py"

if not V09_PATH.exists():
    raise FileNotFoundError(f"09_safety_dashboard_z.py not found: {V09_PATH}")

spec = importlib.util.spec_from_file_location("safety_v09", str(V09_PATH))
v09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v09)


# ==============================
# YOLO Tracking Extraction
# ==============================
def extract_yolo_track_boxes(
    result,
    names,
    image_area,
    person_conf_thres=0.45,
    helmet_conf_thres=0.25,
    person_min_area_ratio=0.002,
    person_min_height=35,
):
    persons = []
    helmets = []

    if result.boxes is None:
        return persons, helmets

    boxes = result.boxes

    if boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
        return persons, helmets

    xyxys = boxes.xyxy.detach().cpu().numpy().astype(float)
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy().astype(float)

    if boxes.id is not None:
        ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxys)

    for i in range(len(xyxys)):
        cls_id = int(clss[i])
        conf = float(confs[i])
        xyxy = xyxys[i].tolist()
        track_id = ids[i]

        cls_name = v09.get_class_name(names, cls_id).lower()

        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1
        area_ratio = v09.box_area(xyxy) / max(image_area, 1)

        if cls_name == "person":
            if conf < person_conf_thres:
                continue

            if area_ratio < person_min_area_ratio:
                continue

            if max(bh, bw) < person_min_height:
                continue

            persons.append(
                {
                    "box": xyxy,
                    "conf": conf,
                    "track_id": track_id,
                    "source": "yolo_track",
                }
            )

        elif cls_name == "helmet":
            if conf < helmet_conf_thres:
                continue

            helmets.append(
                {
                    "box": xyxy,
                    "conf": conf,
                    "track_id": track_id,
                }
            )

    return persons, helmets


# ==============================
# Track State Store
# YOLO가 ID를 만들고,
# 이 클래스는 ID별 상태만 저장한다.
# ==============================
class TrackStateStore:
    def __init__(self, emergency_sec=10.0, max_age_sec=3.0, history_len=8):
        self.emergency_sec = emergency_sec
        self.max_age_sec = max_age_sec
        self.history_len = history_len
        self.states = {}

    def _new_state(self, key, now):
        return {
            "key": key,
            "first_seen": now,
            "last_seen": now,
            "fall_start": None,
            "fall_elapsed": 0.0,
            "emergency": False,
            "helmet_history": deque(maxlen=self.history_len),
            "posture_history": deque(maxlen=self.history_len),
            "lying_history": deque(maxlen=self.history_len),
            "last_map_xy": None,
        }

    def majority_helmet(self, hist):
        if not hist:
            return "UNKNOWN"

        counts = {}
        for x in hist:
            counts[x] = counts.get(x, 0) + 1

        return max(counts.items(), key=lambda kv: kv[1])[0]

    def majority_posture(self, hist):
        if not hist:
            return "NORMAL"

        counts = {}
        for x in hist:
            counts[x] = counts.get(x, 0) + 1

        return max(counts.items(), key=lambda kv: kv[1])[0]

    def smoothed_lying(self, hist):
        if not hist:
            return False

        # 최근 history_len 중 절반 이상이 lying이면 lying 유지
        return sum(1 for x in hist if x) >= max(2, len(hist) // 2)

    def update_detection(self, det, now):
        camera = det.get("camera", "cam")
        yolo_id = det.get("yolo_track_id")

        if yolo_id is None:
            # YOLO track id가 없으면 상태 누적이 불가능하므로 임시 ID로 표시만 한다.
            det["state_key"] = f"{camera}:NO_ID"
            det["fall_elapsed"] = 0.0
            det["emergency"] = False
            det["helmet_status_raw"] = det.get("helmet_status", "UNKNOWN")
            det["posture_raw"] = det.get("posture", "NORMAL")
            return det

        key = f"{camera}:{int(yolo_id)}"

        if key not in self.states:
            self.states[key] = self._new_state(key, now)

        st = self.states[key]
        st["last_seen"] = now
        st["last_map_xy"] = det.get("map_xy")

        raw_helmet = det.get("helmet_status", "UNKNOWN")
        raw_posture = det.get("posture", "NORMAL")
        raw_lying = bool(det.get("lying_candidate", False))

        st["helmet_history"].append(raw_helmet)
        st["posture_history"].append(raw_posture)
        st["lying_history"].append(raw_lying)

        smooth_helmet = self.majority_helmet(st["helmet_history"])
        smooth_posture = self.majority_posture(st["posture_history"])
        smooth_lying = self.smoothed_lying(st["lying_history"])

        det["helmet_status_raw"] = raw_helmet
        det["posture_raw"] = raw_posture

        det["helmet_status"] = smooth_helmet
        det["posture"] = smooth_posture
        det["lying_candidate"] = smooth_lying

        if smooth_lying:
            if st["fall_start"] is None:
                st["fall_start"] = now

            st["fall_elapsed"] = now - st["fall_start"]

            if st["fall_elapsed"] >= self.emergency_sec:
                st["emergency"] = True
        else:
            st["fall_start"] = None
            st["fall_elapsed"] = 0.0
            st["emergency"] = False

        det["state_key"] = key
        det["fall_elapsed"] = st["fall_elapsed"]
        det["emergency"] = st["emergency"]

        return det

    def cleanup(self, now):
        dead = []

        for key, st in self.states.items():
            if now - st["last_seen"] > self.max_age_sec:
                dead.append(key)

        for key in dead:
            del self.states[key]

    def reset(self):
        self.states = {}


# ==============================
# Camera Processor using YOLO Track
# ==============================
class CameraYOLOTracker:
    def __init__(
        self,
        cam_id,
        cam_name,
        det_model_path,
        homography_path,
        z_calib_path,
        map_info,
        homography_output,
        state_store,
        tracker_yaml,
        width=1280,
        height=720,
        fps=15,
    ):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.map_info = map_info
        self.homography_output = homography_output
        self.state_store = state_store
        self.tracker_yaml = tracker_yaml

        # 중요: 카메라별 YOLO 객체 분리
        self.det_model = YOLO(str(det_model_path))

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
        self.last_metrics = {}

        print(f"[INFO] {cam_name}")
        print(f"  camera id  : {cam_id}")
        print(f"  H          : {homography_path}")
        print(f"  H key      : {self.H_key}")
        print(f"  z calib    : {z_calib_path}")
        print(f"  tracker    : {tracker_yaml}")

    def release(self):
        self.cap.release()

    def process(
        self,
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

        run_inference = (self.frame_idx % args.process_every == 0) or self.last_preview is None

        if not run_inference:
            return self.last_preview, self.last_tracked, self.last_metrics

        # ==============================
        # YOLO Track
        # ==============================
        track_result = self.det_model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.det_conf,
            device=args.device,
            persist=True,
            tracker=self.tracker_yaml,
            verbose=False,
        )[0]

        persons, helmets = extract_yolo_track_boxes(
            track_result,
            det_names,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            helmet_conf_thres=args.helmet_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        # ==============================
        # Pose Predict
        # ==============================
        pose_result = pose_model(
            frame,
            imgsz=args.imgsz,
            conf=args.pose_conf,
            device=args.device,
            verbose=False,
        )[0]

        pose_items = v09.extract_pose_items(pose_result)

        preview = frame.copy()

        # Draw helmet
        for helmet in helmets:
            v09.draw_box(
                preview,
                helmet["box"],
                (0, 255, 255),
                f"helmet {helmet['conf']:.2f}",
                2,
            )

        tracked_outputs = []

        for person in persons:
            pbox = person["box"]
            yolo_track_id = person.get("track_id")

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
                "yolo_track_id": yolo_track_id,
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

            det_item = self.state_store.update_detection(det_item, now)
            tracked_outputs.append(det_item)

        self.state_store.cleanup(now)

        # ==============================
        # Draw persons
        # ==============================
        for det in tracked_outputs:
            pbox = det["person_box"]
            yolo_id = det.get("yolo_track_id")
            helmet_status = det["helmet_status"]
            posture = det["posture"]
            emergency = det["emergency"]
            debug = det["debug"]

            color = v09.color_for_state(helmet_status, posture, emergency)

            if yolo_id is None:
                id_label = "NO_ID"
            else:
                id_label = f"ID{int(yolo_id)}"

            label = f"{self.cam_name}-{id_label} {helmet_status} {posture}"

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

            head_px = det.get("head_px")
            head_radius = det.get("head_radius", 40)

            if head_px is not None:
                hp = tuple(head_px.astype(int))
                cv2.circle(preview, hp, int(head_radius), (255, 0, 255), 2)
                cv2.circle(preview, hp, 4, (255, 0, 255), -1)

            foot_px = det.get("foot_px")
            if foot_px is not None:
                fp = tuple(foot_px.astype(int))
                cv2.circle(preview, fp, 6, (255, 255, 0), -1)

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
                f"raw={det.get('helmet_status_raw')}/{det.get('posture_raw')} L/U={debug.get('lying_score')}/{debug.get('upright_score')}",
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

        # Camera label
        overlay = preview.copy()
        cv2.rectangle(overlay, (15, 15), (580, 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, preview, 0.45, 0, preview)

        v09.draw_text(
            preview,
            f"{self.cam_name} | YOLO TRACK | persons:{len(tracked_outputs)} helmets:{len(helmets)} FPS:{self.fps:.1f}",
            (28, 52),
            (255, 255, 255),
            0.66,
            2,
        )

        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            self.fps = self.fps * 0.90 + (1.0 / dt) * 0.10

        metrics = calc_metrics(tracked_outputs, helmets)
        metrics["fps"] = self.fps
        metrics["cam_name"] = self.cam_name

        self.last_preview = preview
        self.last_tracked = tracked_outputs
        self.last_metrics = metrics

        return preview, tracked_outputs, metrics


# ==============================
# Metrics / Drawing
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

        yolo_id = det.get("yolo_track_id")
        if yolo_id is None:
            id_label = "NO_ID"
        else:
            id_label = f"ID{int(yolo_id)}"

        label = f"{det.get('camera', '?')}-{id_label}"

        if det.get("emergency", False):
            label += " EMG"
        elif det.get("posture") == "LYING_CANDIDATE":
            label += " LYING"
        elif det.get("helmet_status") == "NO_HELMET":
            label += " NO_HELMET"

        v09.draw_text(out, label, (px + 10, py - 10), color, 0.45, 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (15, 15), (560, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    v09.draw_text(
        out,
        "MAP VIEW | YOLO TRACK combined cam0 + cam1",
        (28, 50),
        (255, 255, 255),
        0.63,
        2,
    )

    return out


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


def draw_status_panel(
    width,
    height,
    metrics0,
    metrics1,
    all_detections,
    overall_fps,
    state_json_path,
    args,
):
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    total_persons = metrics0.get("person_count", 0) + metrics1.get("person_count", 0)
    total_helmets = metrics0.get("helmet_count", 0) + metrics1.get("helmet_count", 0)
    total_no_helmet = metrics0.get("no_helmet_count", 0) + metrics1.get("no_helmet_count", 0)
    total_lying = metrics0.get("lying_count", 0) + metrics1.get("lying_count", 0)
    total_emergency = metrics0.get("emergency_count", 0) + metrics1.get("emergency_count", 0)

    y = 35
    v09.draw_text(panel, "DUAL YOLO TRACK STATUS V11", (20, y), (255, 255, 255), 0.70, 2)

    y += 38
    v09.draw_text(panel, f"Overall FPS: {overall_fps:.1f} | Device: {args.device}", (20, y), (200, 200, 200), 0.55, 1)

    y += 30
    v09.draw_text(panel, f"Tracker: {args.tracker}", (20, y), (160, 160, 160), 0.48, 1)

    y += 30
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
    draw_camera_metrics(panel, "CAM1", metrics1, 20, y)

    y += 165
    v09.draw_text(panel, "EMERGENCY TARGETS", (20, y), (255, 255, 255), 0.58, 2)
    y += 28

    emergency_targets = [
        d for d in all_detections
        if d.get("emergency", False) and d.get("map_xy") is not None
    ]

    if not emergency_targets:
        v09.draw_text(panel, "None", (35, y), (0, 255, 0), 0.55, 2)
        y += 28
    else:
        for det in emergency_targets[:5]:
            mx, my = det["map_xy"]

            yolo_id = det.get("yolo_track_id")
            if yolo_id is None:
                id_label = "NO_ID"
            else:
                id_label = f"ID{int(yolo_id)}"

            line = f"{det.get('camera')}-{id_label} -> ({mx:.2f}, {my:.2f})"
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


def save_state_json(path, timestamp, detections0, detections1, metrics0, metrics1):
    all_detections = list(detections0) + list(detections1)

    persons = []
    emergency_targets = []

    for det in all_detections:
        map_xy = det.get("map_xy")
        debug = det.get("debug", {})

        item = {
            "camera": det.get("camera"),
            "yolo_track_id": det.get("yolo_track_id"),
            "state_key": det.get("state_key"),
            "helmet_status": det.get("helmet_status"),
            "helmet_status_raw": det.get("helmet_status_raw"),
            "posture": det.get("posture"),
            "posture_raw": det.get("posture_raw"),
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
        "mode": "dual_camera_yolo_tracking_v11",
        "metrics": {
            "cam0": metrics0,
            "cam1": metrics1,
            "total_person_count": metrics0.get("person_count", 0) + metrics1.get("person_count", 0),
            "total_emergency_count": metrics0.get("emergency_count", 0) + metrics1.get("emergency_count", 0),
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

    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
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

    parser.add_argument("--track-max-age", type=float, default=3.0)
    parser.add_argument("--history-len", type=int, default=8)

    parser.add_argument("--display-w", type=int, default=1600)
    parser.add_argument("--display-h", type=int, default=900)

    parser.add_argument("--process-every", type=int, default=1)

    parser.add_argument(
        "--state-json",
        type=str,
        default="outputs/safety_state_yolo_tracking_v11.json",
    )

    args = parser.parse_args()

    print("======================================")
    print("Dual Camera YOLO Tracking Dashboard V11")
    print("--------------------------------------")
    print(f"cam0 id     : {args.cam0_id}")
    print(f"cam1 id     : {args.cam1_id}")
    print(f"det model   : {args.det_model}")
    print(f"pose model  : {args.pose_model}")
    print(f"tracker     : {args.tracker}")
    print(f"device      : {args.device}")
    print(f"imgsz       : {args.imgsz}")
    print(f"process every N frames: {args.process_every}")
    print("======================================")

    det_model_path = Path(args.det_model).expanduser()
    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    map_info = v09.load_map_from_yaml(args.map_yaml)

    # pose는 tracking state가 없으므로 공유 가능
    pose_model = YOLO(args.pose_model)

    # det_names용 임시 모델. 실제 tracking model은 카메라별로 따로 생성.
    names_model = YOLO(str(det_model_path))
    det_names = names_model.names
    del names_model

    state_store0 = TrackStateStore(
        emergency_sec=args.emergency_sec,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
    )

    state_store1 = TrackStateStore(
        emergency_sec=args.emergency_sec,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
    )

    cam0 = CameraYOLOTracker(
        cam_id=args.cam0_id,
        cam_name=args.cam0_name,
        det_model_path=det_model_path,
        homography_path=args.cam0_homography,
        z_calib_path=args.cam0_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        state_store=state_store0,
        tracker_yaml=args.tracker,
    )

    cam1 = CameraYOLOTracker(
        cam_id=args.cam1_id,
        cam_name=args.cam1_name,
        det_model_path=det_model_path,
        homography_path=args.cam1_homography,
        z_calib_path=args.cam1_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        state_store=state_store1,
        tracker_yaml=args.tracker,
    )

    window_name = "Dual Camera YOLO Tracking Dashboard V11"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_loop_time = time.time()
    overall_fps = 0.0

    try:
        while True:
            now = time.time()

            cam0_view, cam0_tracked, cam0_metrics = cam0.process(
                pose_model=pose_model,
                det_names=det_names,
                args=args,
                now=now,
            )

            cam1_view, cam1_tracked, cam1_metrics = cam1.process(
                pose_model=pose_model,
                det_names=det_names,
                args=args,
                now=now,
            )

            all_detections = list(cam0_tracked) + list(cam1_tracked)

            save_state_json(
                args.state_json,
                timestamp=now,
                detections0=cam0_tracked,
                detections1=cam1_tracked,
                metrics0=cam0_metrics,
                metrics1=cam1_metrics,
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
                metrics1=cam1_metrics,
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
                state_store0.reset()
                state_store1.reset()
                print("[RESET] all track states reset")

    finally:
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()