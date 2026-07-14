"""
safety_lib.state_io
===================

파일 I/O 유틸.

담당:
  * resolve_path      : 상대 경로를 base_dir 기준 절대 경로로 변환
  * load_entry_roi    : 저장된 Entry ROI JSON 로드
  * set_roi_mode      : map 위에 mouse로 폴리곤 찍어 ROI 저장 (--mode set_roi)
  * save_state_json   : 매 프레임 전체 상태를 outputs/*.json 에 덤프
                         (외부 감시 도구가 이 파일을 tail 해서 상태 확인)
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import base_utils as utils


def resolve_path(base_dir, path_str):
    """
    상대 경로 → base_dir 기준 절대 경로.
    스크립트 CWD와 무관하게 파일을 찾도록 하기 위한 헬퍼.
    이미 절대 경로면 그대로 반환.
    """
    p = Path(path_str).expanduser()

    if p.is_absolute():
        return p

    return Path(base_dir).resolve() / p


def load_entry_roi(path):
    """
    저장된 Entry ROI JSON 파일 로드.

    JSON 구조:
        points_px  : [[x,y], ...] map 이미지 픽셀 좌표 (참고용)
        points_map : [[mx,my], ...] map 미터 좌표 (실제 사용)

    반환:
        (points_map, raw_data) — points_map은 point_in_polygon에 그대로 넣음
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"entry ROI json not found: {path}\n"
            "먼저 --mode set_roi 로 입구 ROI를 저장해야 합니다."
        )

    data = json.loads(path.read_text())
    pts = data.get("points_map", [])

    if len(pts) < 3:
        raise RuntimeError(f"entry ROI needs at least 3 points: {path}")

    return [[float(x), float(y)] for x, y in pts], data


def set_roi_mode(args):
    """
    --mode set_roi 진입점.

    map pgm을 창에 띄우고 사용자가 마우스 클릭으로 폴리곤 꼭짓점 지정.
      L click : 점 추가
      u       : 마지막 점 취소 (undo)
      r       : 전부 리셋
      s       : entry_roi.json으로 저장
      q       : 종료

    저장되는 좌표는 픽셀(참고)과 미터(사용) 둘 다. 이후 --mode run에서
    load_entry_roi()로 다시 읽어 point_in_polygon 판정에 사용.
    """
    map_info = utils.load_map_from_yaml(args.map_yaml)
    img = map_info["img"].copy()

    state = {
        "points_px": [],
        "points_map": [],
    }

    window = "SET ENTRY ROI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        # 좌클릭 시 map 이미지 픽셀 → 미터로 변환해 함께 저장
        if event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < map_info["width"] and 0 <= y < map_info["height"]):
                print("[WARN] click outside map")
                return

            mx, my = utils.map_image_pixel_to_meter(x, y, map_info)

            state["points_px"].append([int(x), int(y)])
            state["points_map"].append([float(mx), float(my)])

            print(
                f"[ROI POINT {len(state['points_px'])}] "
                f"px=({x},{y}) map=({mx:.3f},{my:.3f})"
            )

    cv2.setMouseCallback(window, mouse_cb)

    print("======================================")
    print("SET ENTRY ROI MODE")
    print("--------------------------------------")
    print("입구 영역을 map 위에서 polygon으로 클릭")
    print("left click : 점 추가")
    print("u          : undo")
    print("r          : reset")
    print("s          : save")
    print("q          : quit")
    print("======================================")

    while True:
        show = img.copy()

        if len(state["points_px"]) >= 1:
            pts = np.array(state["points_px"], dtype=np.int32)

            for i, (px, py) in enumerate(state["points_px"]):
                cv2.circle(show, (px, py), 4, (0, 180, 255), -1)
                utils.draw_text(show, f"{i}", (px + 6, py - 6), (0, 180, 255), 0.48, 1)

            if len(pts) >= 2:
                cv2.polylines(show, [pts], False, (0, 180, 255), 2)

            if len(pts) >= 3:
                overlay = show.copy()
                cv2.fillPoly(overlay, [pts], (0, 180, 255))
                cv2.addWeighted(overlay, 0.20, show, 0.80, 0, show)
                cv2.polylines(show, [pts], True, (0, 180, 255), 2)

        overlay = show.copy()
        cv2.rectangle(overlay, (10, 10), (760, 88), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, show, 0.50, 0, show)

        utils.draw_text(
            show,
            "SET ENTRY ROI | click entrance polygon on map",
            (24, 40),
            (255, 255, 255),
            0.62,
            2,
        )
        utils.draw_text(
            show,
            f"points: {len(state['points_px'])} | s:save u:undo r:reset q:quit",
            (24, 70),
            (255, 255, 255),
            0.52,
            1,
        )

        cv2.imshow(window, show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("u"):
            if state["points_px"]:
                removed_px = state["points_px"].pop()
                removed_map = state["points_map"].pop()
                print("[UNDO]", removed_px, removed_map)

        if key == ord("r"):
            state["points_px"] = []
            state["points_map"] = []
            print("[RESET] ROI points cleared")

        if key == ord("s"):
            if len(state["points_map"]) < 3:
                print("[WARN] Need at least 3 points")
                continue

            out_path = Path(args.entry_roi_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "map_yaml": args.map_yaml,
                "points_px": state["points_px"],
                "points_map": state["points_map"],
            }

            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

            print("")
            print("======================================")
            print("[SAVED] entry ROI")
            print(f"path: {out_path}")
            print("points_map:")
            for p in state["points_map"]:
                print(f"  - {p[0]:.3f}, {p[1]:.3f}")
            print("======================================")

    cv2.destroyAllWindows()


def save_state_json(path, timestamp, detections0, detections1, metrics0, metrics1):
    """
    매 프레임 전체 상태를 JSON 파일로 덤프.

    ROS 발행과 별개의 로컬 감시 채널. 외부 감시 스크립트가 이 파일을
    tail 해서 emergency/helmet_alert/unauthorized 상태를 확인할 수 있음.
    현재는 주로 디버깅/데모용.

    JSON 최상위 구조:
      timestamp, mode, metrics, persons (전체),
      emergency_targets, helmet_targets, suspicious_targets,
      unauthorized_entries, access_events

    각 *_targets 리스트는 이미 플래그로 필터링된 상태 (bridge와 동일 목록).
    """
    all_detections = list(detections0) + list(detections1)

    # 카테고리별 필터링 결과 컨테이너
    persons = []
    access_events = []
    emergency_targets = []
    unauthorized_entries = []
    helmet_targets = []
    suspicious_targets = []

    for det in all_detections:
        map_xy = det.get("map_xy")
        debug = det.get("debug", {})

        item = {
            "camera": det.get("camera"),
            "display_id": det.get("display_id"),
            "yolo_track_id": None if det.get("yolo_track_id") is None else int(det.get("yolo_track_id")),
            "state_key": det.get("state_key"),

            "helmet_status": det.get("helmet_status"),
            "helmet_status_raw": det.get("helmet_status_raw"),
            "helmet_alert": bool(det.get("helmet_alert", False)),
            "helmet_alert_elapsed": float(det.get("helmet_alert_elapsed", 0.0)),
            "helmet_recover_elapsed": float(det.get("helmet_recover_elapsed", 0.0)),
            "helmet_suspicious": bool(det.get("helmet_suspicious", False)),

            "posture": det.get("posture"),
            "posture_raw": det.get("posture_raw"),

            "hand_raised": bool(det.get("hand_raised", False)),
            "hand_side": det.get("hand_side"),

            "in_entry_roi": bool(det.get("in_entry_roi", False)),
            "entry_state": det.get("entry_state"),
            "entry_progress": float(det.get("entry_progress", 0.0)),
            "entry_progress_ratio": float(det.get("entry_progress_ratio", 0.0)),
            "access_granted": bool(det.get("access_granted", False)),
            "unauthorized": bool(det.get("unauthorized", False)),

            "lying_candidate": bool(det.get("lying_candidate", False)),
            "emergency": bool(det.get("emergency", False)),
            "fall_elapsed": float(det.get("fall_elapsed", 0.0)),
            "recovery_elapsed": float(det.get("recovery_elapsed", 0.0)),

            "head_z": debug.get("head_z"),
            "z_residual": debug.get("z_residual"),

            "map_xy": None if map_xy is None else [float(map_xy[0]), float(map_xy[1])],
        }

        persons.append(item)

        if item["access_granted"]:
            access_events.append(item)

        if item["unauthorized"]:
            unauthorized_entries.append(item)

        if item["emergency"] and item["map_xy"] is not None:
            emergency_targets.append(item)

        if item["helmet_alert"] and item["map_xy"] is not None:
            helmet_targets.append(item)

        if item["helmet_suspicious"] and item["map_xy"] is not None:
            suspicious_targets.append(item)

    state = {
        "timestamp": timestamp,
        "mode": "v12_modular_entry_yolo_tracking_helmet_alert_update",
        "metrics": {
            "cam0": metrics0,
            "cam1": metrics1,
            "total_person_count": metrics0.get("person_count", 0) + metrics1.get("person_count", 0),
            "total_unauthorized_count": metrics0.get("unauthorized_count", 0) + metrics1.get("unauthorized_count", 0),
            "total_emergency_count": metrics0.get("emergency_count", 0) + metrics1.get("emergency_count", 0),
            "total_helmet_alert_count": metrics0.get("helmet_alert_count", 0) + metrics1.get("helmet_alert_count", 0),
            "total_suspicious_count": metrics0.get("suspicious_count", 0) + metrics1.get("suspicious_count", 0),
        },
        "persons": persons,
        "access_events": access_events,
        "unauthorized_entries": unauthorized_entries,
        "emergency_targets": emergency_targets,
        "helmet_targets": helmet_targets,
        "suspicious_targets": suspicious_targets,
    }

    # 부모 디렉토리가 없으면 자동 생성 (첫 실행 시 outputs/ 자동 생성)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 원자적 쓰기: 임시 파일에 쓰고 rename.
    # 매 프레임 덮어쓰므로, 외부 감시 스크립트가 tail 하다가 반쯤 쓰인
    # JSON 을 읽고 파싱 에러를 내는 걸 막는다. os.replace 는 원자적이다.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

    return state