"""
safety_lib.dashboard_ui
=======================

로컬 대시보드 렌더러 (PC3 화면에 붙어 있는 오퍼레이터용).

패널 조립 순서 (메인 스크립트의 run_mode 안에서):
    ┌──────────────┬──────────────┐
    │ cam0_preview │ cam1_preview │   ← CameraEntryTracker.process가 반환
    ├──────────────┼──────────────┤
    │ map_view     │ status_panel │
    └──────────────┴──────────────┘
    map_view    = draw_combined_map(...)
    status_panel = draw_status_panel(...)

주요 함수:
  * draw_entry_roi_on_camera : 카메라 화면에 map ROI를 H_inv로 역투영
  * draw_person_overlay      : 사람별 bbox/keypoint/텍스트 오버레이
  * draw_combined_map        : map 위에 인원 위치 점 찍기
  * draw_status_panel        : 좌측 통계 + 우측 워커 목록
  * state_color              : 상태 우선순위 → BGR 색상
  * blink_on                 : 깜빡임 (emergency/helmet_alert 강조용)
"""
import time

import cv2
import numpy as np

from . import base_utils as utils

# COCO 17 keypoint 연결 (어깨-팔꿈치-손목, 어깨-엉덩이, 엉덩이-무릎-발목)
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def blink_on(rate_hz=2.5):
    """
    time.time()에서 파생한 boolean toggle.
    rate_hz=2.5면 초당 5번 상태 변경 → 깜빡깜빡 효과.
    emergency/helmet_alert 마커 강조에 사용.
    """
    return int(time.time() * rate_hz * 2) % 2 == 0


def emergency_map_color():
    """Emergency 점 색상 (빨강↔어두운 회색 blink). BGR."""
    if blink_on():
        return (0, 0, 255)
    return (30, 30, 30)


def helmet_map_color():
    """Helmet alert 점 색상 (주황↔어두운 회색 blink). BGR."""
    if blink_on():
        return (0, 140, 255)  # BGR orange
    return (30, 30, 30)


def unauthorized_map_color():
    """무단 침입자 점 색상 (보라↔어두운 회색 blink). BGR."""
    if blink_on():
        return (255, 0, 200)  # BGR purple/magenta
    return (30, 30, 30)


def draw_entry_roi_on_map(map_img, map_info, roi_points_map):
    """
    map 이미지 위에 Entry ROI 폴리곤을 반투명 오렌지로 그림.
    map 좌표(미터) → 이미지 픽셀로 변환 후 fillPoly + polylines.
    """
    out = map_img.copy()

    if roi_points_map is None or len(roi_points_map) < 3:
        return out

    pts = []

    for mx, my in roi_points_map:
        px, py = utils.map_meter_to_image_pixel(mx, my, map_info)
        pts.append([px, py])

    pts_np = np.array(pts, dtype=np.int32)

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts_np], (0, 180, 255))
    cv2.addWeighted(overlay, 0.16, out, 0.84, 0, out)

    cv2.polylines(out, [pts_np], True, (0, 180, 255), 2)

    for i, (px, py) in enumerate(pts):
        cv2.circle(out, (px, py), 2, (0, 180, 255), -1)
        utils.draw_text(out, f"E{i}", (px + 4, py - 4), (0, 180, 255), 0.30, 1)

    return out


def project_map_points_to_camera(H_inv, map_info, roi_points_map, homography_output):
    """
    map ROI 폴리곤을 H_inv로 역투영해 카메라 픽셀 좌표로.
    카메라 화면 위에 ROI를 그리기 위함. H_inv가 없으면 None.
    """
    if H_inv is None or roi_points_map is None or len(roi_points_map) < 3:
        return None

    pts = []

    for mx, my in roi_points_map:
        if homography_output == "map_meters":
            src = np.array([float(mx), float(my), 1.0], dtype=np.float64)
        else:
            map_px, map_py = utils.map_meter_to_image_pixel(mx, my, map_info)
            src = np.array([float(map_px), float(map_py), 1.0], dtype=np.float64)

        uvw = H_inv @ src

        if abs(float(uvw[2])) < 1e-9:
            continue

        u = float(uvw[0] / uvw[2])
        v = float(uvw[1] / uvw[2])

        if np.isfinite(u) and np.isfinite(v):
            pts.append([int(round(u)), int(round(v))])

    if len(pts) < 3:
        return None

    return np.array(pts, dtype=np.int32)


def draw_entry_roi_on_camera(frame, H_inv, map_info, roi_points_map, homography_output, cam_name):
    """
    카메라 preview에 Entry ROI 오렌지 영역 오버레이.
    cam0에서만 호출 (worker_store.entry_enabled=True인 경우).
    """
    out = frame.copy()

    pts_np = project_map_points_to_camera(
        H_inv,
        map_info,
        roi_points_map,
        homography_output,
    )

    if pts_np is None:
        return out

    overlay = out.copy()
    cv2.fillPoly(overlay, [pts_np], (0, 180, 255))
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    cv2.polylines(out, [pts_np], True, (0, 180, 255), 2)

    x0, y0 = pts_np[0]
    utils.draw_text(
        out,
        f"{cam_name} ENTRY ROI",
        (int(x0) + 6, max(20, int(y0) - 6)),
        (0, 180, 255),
        0.58,
        2,
    )

    return out


def state_color(det):
    """
    detection의 현재 상태를 BGR 색상으로. 상태기계 우선순위대로 판정.

    우선순위 (위에서부터 매칭되는 첫 케이스):
      emergency         → 빨강
      unauthorized      → 빨강
      helmet_alert      → 빨주
      NO_HELMET_*       → 빨주
      SUSPICIOUS        → 노랑
      entry WAIT/FAIL   → 주황
      entry CHECKING    → 노랑
      access_granted    → 초록
      기타              → base_utils.color_for_state 폴백

    ros_bridge의 marker 색상과 대응됨 (일관된 UX).
    """
    if det.get("emergency", False):
        return (0, 0, 255)

    # 무단 침입자는 빨강(긴급)이 아니라 보라. 헬멧 미착용과도 구별해야 한다.
    if det.get("unauthorized", False):
        return (255, 0, 200)

    if det.get("helmet_alert", False):
        return (0, 80, 255)

    if str(det.get("helmet_status", "")).startswith("NO_HELMET"):
        return (0, 80, 255)

    if det.get("helmet_suspicious", False) or det.get("helmet_status") == "SUSPICIOUS":
        return (0, 200, 255)

    if det.get("entry_state") in ["CHECK_FAIL_NO_HELMET", "WAIT_HAND_RAISE", "WAIT_CHECK"]:
        return (0, 165, 255)

    if det.get("entry_state") == "CHECKING":
        return (0, 255, 255)

    if det.get("access_granted", False):
        return (0, 255, 0)

    return utils.color_for_state(
        det.get("helmet_status", "UNKNOWN"),
        det.get("posture", "NORMAL"),
        det.get("emergency", False),
    )


def display_id_text(det):
    """compact display_id를 문자열로. None이면 'X'."""
    did = det.get("display_id")
    if did is None:
        return "X"
    return str(int(did))


def draw_progress_bar(img, x1, y1, x2, y2, ratio, color, text):
    """
    bbox 상단에 3초 카운트다운 진행 바.
    entry CHECKING 중 시각적 피드백에 사용.
    """
    ratio = max(0.0, min(1.0, float(ratio)))

    bar_w = max(120, min(240, int(x2 - x1)))
    bar_h = 16

    bx1 = int(x1)
    by1 = int(max(0, y1 - 34))
    bx2 = bx1 + bar_w
    by2 = by1 + bar_h

    cv2.rectangle(img, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

    fill_w = int(bar_w * ratio)
    cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), color, -1)

    utils.draw_text(img, text, (bx1, max(16, by1 - 5)), color, 0.52, 1)


def draw_person_overlay(preview, det, pose_items, args, image_h):
    """
    한 사람에 대해 오버레이 전부 그림:
      - bbox + 라벨 (#id, helmet_status, posture, emergency 여부 등)
      - 좌상단 compact id 사각형
      - Entry 상태별 progress bar / ACCESS GRANTED / UNAUTHORIZED 태그
      - 머리 원 + 발 원 + head_z 재투영 검증 점
      - torso 라인 (파랑, get_torso_horizontal 시각화)
      - 손/entry 상태 텍스트

    상태 우선순위별로 색상과 문구가 바뀜.
    """
    pbox = det["person_box"]
    x1, y1, x2, y2 = map(int, pbox)

    color = state_color(det)

    did = display_id_text(det)
    yolo_id = det.get("yolo_track_id")
    yolo_text = "NO_ID" if yolo_id is None else f"Y{int(yolo_id)}"

    label = f"#{did} ({yolo_text}) {det.get('helmet_status')} {det.get('posture')}"

    if det.get("is_held", False):
        label += f" | HOLD {det.get('hold_elapsed', 0.0):.1f}s"

    if det.get("helmet_suspicious", False) or det.get("helmet_status") == "SUSPICIOUS":
        label += " | SUSPICIOUS"

    if det.get("helmet_alert", False):
        label += " | HELMET_ALERT"

    if det.get("entry_state"):
        label += f" | {det.get('entry_state')}"

    if det.get("emergency", False):
        if det.get("lying_candidate", False):
            label += f" | EMERGENCY {det.get('fall_elapsed', 0):.1f}s"
        else:
            label += f" | RECOVERY {det.get('recovery_elapsed', 0):.1f}/{args.recovery_sec:.0f}s"

    elif det.get("lying_candidate", False):
        label += f" | lying {det.get('fall_elapsed', 0):.1f}/{args.emergency_sec:.0f}s"

    head_z = det.get("debug", {}).get("head_z")
    if head_z is not None:
        label += f" z={head_z:.2f}"

    line_thick = 3 if det.get("emergency", False) else 2
    utils.draw_box(preview, pbox, color, label, line_thick)

    # Display compact id box
    cv2.rectangle(preview, (x1, y1), (x1 + 48, y1 + 48), color, -1)
    utils.draw_text(preview, f"{did}", (x1 + 12, y1 + 37), (0, 0, 0), 1.10, 3)

    # Entry / alert progress bar
    entry_state = det.get("entry_state", "OUTSIDE")
    progress_ratio = det.get("entry_progress_ratio", 0.0)
    progress = det.get("entry_progress", 0.0)

    if entry_state == "CHECKING":
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            progress_ratio,
            (0, 255, 255),
            f"checking {progress:.1f}/{args.entry_check_sec:.0f}s",
        )

    elif det.get("access_granted", False):
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (0, 255, 0),
            "ACCESS GRANTED",
        )

    elif det.get("unauthorized", False):
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (255, 0, 200),
            "UNAUTHORIZED INTRUDER",
        )

    elif det.get("helmet_alert", False):
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (0, 80, 255),
            "HELMET DELIVERY REQUEST",
        )

    elif det.get("helmet_suspicious", False) or det.get("helmet_status") == "SUSPICIOUS":
        draw_progress_bar(
            preview,
            x1,
            y1,
            x2,
            y2,
            1.0,
            (0, 200, 255),
            "SUSPICIOUS HELMET STATE",
        )

    # Head circle
    head_px = det.get("head_px")
    head_radius = det.get("head_radius", 40)

    if head_px is not None:
        hp = tuple(head_px.astype(int))
        cv2.circle(preview, hp, int(head_radius), (255, 0, 255), 1)
        cv2.circle(preview, hp, 4, (255, 0, 255), -1)

    # Foot point
    foot_px = det.get("foot_px")
    if foot_px is not None:
        fp = tuple(foot_px.astype(int))
        cv2.circle(preview, fp, 6, (255, 255, 0), -1)

    # Z projected point
    projected_uv = det.get("projected_uv")
    if projected_uv is not None and np.all(np.isfinite(projected_uv)):
        pp = tuple(projected_uv.astype(int))
        cv2.circle(preview, pp, 5, (0, 165, 255), -1)

        if head_px is not None:
            cv2.line(preview, tuple(head_px.astype(int)), pp, (0, 165, 255), 1)

    # COCO 17 skeleton (pose 모델이 준 키포인트를 그대로 그림)
    pi = det.get("pose_item")
    if pi is not None:
        kp = pi.get("keypoints")
        kc = pi.get("conf")
        if kp is not None:
            for a, b in COCO_SKELETON:
                if kc is None or (kc[a] > args.pose_conf and kc[b] > args.pose_conf):
                    cv2.line(preview, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                             (255, 180, 0), 2, cv2.LINE_AA)
            for j in range(len(kp)):
                if kc is None or kc[j] > args.pose_conf:
                    cv2.circle(preview, tuple(kp[j].astype(int)), 3, (0, 200, 255), -1)

    # Torso line
    pose_item_tmp, _ = utils.match_pose_to_person(pbox, pose_items)

    try:
        torso_horizontal, shoulder_mid, hip_mid = utils.get_torso_horizontal(
            pose_item_tmp,
            args.pose_conf,
        )
    except Exception:
        shoulder_mid, hip_mid = None, None

    if shoulder_mid is not None and hip_mid is not None:
        cv2.line(
            preview,
            tuple(shoulder_mid.astype(int)),
            tuple(hip_mid.astype(int)),
            (255, 0, 0),
            2,
        )

    # Debug text
    line_y = min(image_h - 10, y2 + 24)

    utils.draw_text(
        preview,
        f"hand={det.get('hand_raised')} side={det.get('hand_side')} entry={det.get('in_entry_roi')}",
        (x1, line_y),
        color,
        0.50,
        1,
    )

    line_y += 22
    utils.draw_text(
        preview,
        f"raw H/P={det.get('helmet_status_raw')}/{det.get('posture_raw')}",
        (x1, min(image_h - 10, line_y)),
        color,
        0.44,
        1,
    )

    map_xy = det.get("map_xy")
    if map_xy is not None:
        line_y += 20
        utils.draw_text(
            preview,
            f"map=({map_xy[0]:.2f},{map_xy[1]:.2f})",
            (x1, min(image_h - 10, line_y)),
            color,
            0.44,
            1,
        )


def calc_metrics(tracked, helmets):
    """
    한 카메라의 이번 프레임 통계를 dict로. status_panel 표시용.
    person_count, helmet_count, no_helmet_count, emergency_count 등.
    save_state_json에도 그대로 실려서 외부에서 확인 가능.
    """
    return {
        "person_count": len(tracked),
        "helmet_count": len(helmets),
        "no_helmet_count": sum(
            1 for d in tracked
            if str(d.get("helmet_status", "")).startswith("NO_HELMET")
        ),
        "unknown_helmet_count": sum(
            1 for d in tracked
            if d.get("helmet_status") == "UNKNOWN"
        ),
        "suspicious_count": sum(
            1 for d in tracked
            if d.get("helmet_suspicious") or d.get("helmet_status") == "SUSPICIOUS"
        ),
        "helmet_alert_count": sum(1 for d in tracked if d.get("helmet_alert")),
        "checking_count": sum(1 for d in tracked if d.get("entry_state") == "CHECKING"),
        "granted_count": sum(1 for d in tracked if d.get("access_granted")),
        "unauthorized_count": sum(1 for d in tracked if d.get("unauthorized")),
        "lying_count": sum(1 for d in tracked if d.get("posture") == "LYING_CANDIDATE"),
        "emergency_count": sum(1 for d in tracked if d.get("emergency")),
        "hold_count": sum(1 for d in tracked if d.get("is_held")),
    }


def dedup_detections(all_detections, dedup_dist=0.5):
    """
    같은 사람을 두 번 그리지 않도록 detection 을 걸러낸다.

    cam0 과 cam1 의 시야는 일부 겹친다. 한 사람이 두 카메라에 동시에 잡히면
    detection 이 두 개 나오고, GlobalFuser 가 같은 global_id 로 묶어준다.
    그런데 지도 그리기와 인원수는 그 global_id 를 안 보고 로컬 map_xy 를
    그대로 그려서 점이 두 개 찍혔다.

    ros_bridge 와 똑같은 2단 dedup 을 쓴다:
      1) global_id (없으면 state_key) 기준
      2) 위치 근접 (dedup_dist [m]) — 융합이 실패한 첫 프레임의 안전망

    Returns:
        [(det, (mx, my)), ...]  중복 제거된 목록
    """
    picked = {}

    for det in all_detections:
        xy = det.get("global_map_xy")
        if xy is None:
            xy = det.get("map_xy")
        if xy is None:
            continue

        key = det.get("global_id") or det.get("state_key")
        if key is None:
            continue

        if key in picked:
            continue

        picked[key] = (det, (float(xy[0]), float(xy[1])))

    entries = list(picked.values())

    if dedup_dist <= 0.0 or len(entries) <= 1:
        return entries

    def _priority(det):
        # live > held > (emergency > unauthorized > helmet_alert) > conf
        return (
            not bool(det.get("is_held")),
            bool(det.get("emergency")),
            bool(det.get("unauthorized")),
            bool(det.get("helmet_alert")),
            float(det.get("conf", 0.0)),
        )

    entries.sort(key=lambda e: _priority(e[0]), reverse=True)

    kept = []
    d2 = dedup_dist * dedup_dist

    for det, xy in entries:
        if any((xy[0] - kx) ** 2 + (xy[1] - ky) ** 2 <= d2 for _, (kx, ky) in kept):
            continue
        kept.append((det, xy))

    return kept


def draw_combined_map(map_img, all_detections, map_info, entry_roi_points, dedup_dist=0.5):
    """
    map pgm 위에 Entry ROI + 인원 위치 점 그리기.

    한 사람에 점 하나. GlobalFuser 의 융합 좌표(global_map_xy)를 우선 쓰고,
    같은 인격은 dedup_detections 로 한 번만 그린다.

    emergency 빨강 blink / unauthorized 보라 blink / helmet_alert 주황 blink /
    suspicious 노랑 고정 / 나머지는 상태 색상. is_held 면 회색 링.
    """
    out = draw_entry_roi_on_map(map_img, map_info, entry_roi_points)

    for det, (mx, my) in dedup_detections(all_detections, dedup_dist):
        px, py = utils.map_meter_to_image_pixel(mx, my, map_info)

        if not (0 <= px < map_info["width"] and 0 <= py < map_info["height"]):
            continue

        emergency = bool(det.get("emergency", False))
        unauthorized = bool(det.get("unauthorized", False))
        helmet_alert = bool(det.get("helmet_alert", False))
        helmet_suspicious = bool(det.get("helmet_suspicious", False)) or det.get("helmet_status") == "SUSPICIOUS"
        is_held = bool(det.get("is_held", False))

        if emergency:
            color = emergency_map_color()
            radius = 8 if blink_on() else 4
            cv2.circle(out, (px, py), radius, color, -1)
            cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 1)
            if blink_on():
                cv2.circle(out, (px, py), 16, (0, 0, 255), 2)

        elif unauthorized:
            # 침입자: 보라 blink. 헬멧 경보(주황)와 헷갈리면 안 된다.
            color = unauthorized_map_color()
            radius = 7 if blink_on() else 4
            cv2.circle(out, (px, py), radius, color, -1)
            cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 1)
            if blink_on():
                cv2.circle(out, (px, py), 15, (255, 0, 200), 2)

        elif helmet_alert:
            color = helmet_map_color()
            radius = 7 if blink_on() else 4
            cv2.circle(out, (px, py), radius, color, -1)
            cv2.circle(out, (px, py), radius + 2, (0, 0, 0), 1)
            if blink_on():
                cv2.circle(out, (px, py), 14, (0, 140, 255), 2)

        elif helmet_suspicious:
            color = (0, 200, 255)
            cv2.circle(out, (px, py), 5, color, -1)
            cv2.circle(out, (px, py), 7, (0, 0, 0), 1)

        else:
            color = state_color(det)
            cv2.circle(out, (px, py), 5, color, -1)
            cv2.circle(out, (px, py), 7, (0, 0, 0), 1)

        if is_held:
            cv2.circle(out, (px, py), 11, (180, 180, 180), 1)

        gid = det.get("global_id")
        label = f"{gid}" if gid else f"{det.get('camera')}-#{display_id_text(det)}"

        if emergency:
            label += " EMG"
        elif unauthorized:
            label += " INTRUDER"
        elif helmet_alert:
            label += " HELMET"
        elif helmet_suspicious:
            label += " SUSP"
        elif is_held:
            label += " HOLD"
        elif det.get("access_granted"):
            label += " OK"
        elif det.get("entry_state") == "CHECKING":
            label += " CHECK"
        elif det.get("posture") == "LYING_CANDIDATE":
            label += " LYING"

        utils.draw_text(out, label, (px + 8, py - 8), color, 0.38, 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (460, 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    utils.draw_text(
        out,
        "MAP | EMG red | INTRUDER purple | HELMET orange",
        (16, 32),
        (255, 255, 255),
        0.42,
        1,
    )

    return out


def draw_legend_item(panel, x, y, color, text):
    cv2.circle(panel, (x, y - 5), 6, color, -1)
    utils.draw_text(panel, text, (x + 18, y), (190, 190, 190), 0.42, 1)


def draw_status_panel(width, height, metrics0, metrics1, all_detections, overall_fps, state_json_path, args, dedup_dist=0.5):
    """
    4분할 대시보드의 우하단 통계 패널.

    좌측: 두 카메라 합산 카운트 (Persons/Hold/No Helmet/Suspicious/
          Helmet Goal/Checking/Granted/Unauthorized/Lying/Emergency)
    우측: 최대 6명 워커 상세 (상태, 헬멧, 자세, 진행률)
    하단: 색상 범례 + cam0/cam1 역할 표시
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    # 인원수는 카메라별 합이 아니라 '중복 제거된 사람 수'여야 한다.
    # 한 사람이 두 카메라에 잡히면 metrics 합은 2, 실제는 1명이다.
    unique = dedup_detections(all_detections, dedup_dist)
    total_persons = len(unique)
    total_no_helmet = metrics0.get("no_helmet_count", 0) + metrics1.get("no_helmet_count", 0)
    total_suspicious = metrics0.get("suspicious_count", 0) + metrics1.get("suspicious_count", 0)
    total_checking = metrics0.get("checking_count", 0) + metrics1.get("checking_count", 0)
    total_granted = metrics0.get("granted_count", 0) + metrics1.get("granted_count", 0)
    total_emg = sum(1 for d, _ in unique if d.get("emergency"))
    total_unauth = sum(1 for d, _ in unique if d.get("unauthorized"))
    total_helmet_alert = sum(1 for d, _ in unique if d.get("helmet_alert"))
    total_lying = sum(1 for d, _ in unique if d.get("posture") == "LYING_CANDIDATE")
    total_hold = sum(1 for d, _ in unique if d.get("is_held"))
    total_suppressed = sum(1 for d, _ in unique if d.get("helmet_suppressed"))

    def put(text, x, y, color=(235, 235, 235), scale=0.48, thick=1):
        utils.draw_text(panel, text, (x, y), color, scale, thick)

    put("V12 MODULAR ENTRY + YOLO TRACKING", 18, 34, (255, 255, 255), 0.58, 2)
    put(
        f"FPS {overall_fps:.1f} | {args.tracker} | GPU {args.device} | "
        f"EMG {args.emergency_sec:.0f}s | REC {args.recovery_sec:.0f}s | HOLD {args.hold_sec:.1f}s",
        18,
        62,
        (180, 180, 180),
        0.38,
        1,
    )

    lx = 24
    y = 105
    put("TOTAL", lx, y, (255, 255, 255), 0.56, 2)

    y += 28
    put(f"Persons      {total_persons}", lx, y, (255, 255, 255), 0.50, 1)

    y += 26
    put(
        f"Hold         {total_hold}",
        lx,
        y,
        (180, 180, 180) if total_hold > 0 else (120, 120, 120),
        0.48,
        1,
    )

    y += 26
    put(
        f"No Helmet    {total_no_helmet}",
        lx,
        y,
        (0, 80, 255) if total_no_helmet > 0 else (0, 255, 0),
        0.50,
        1,
    )

    y += 26
    put(
        f"Suspicious   {total_suspicious}",
        lx,
        y,
        (0, 200, 255) if total_suspicious > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 26
    put(
        f"Helmet Goal  {total_helmet_alert}"
        + (f"   (suppressed {total_suppressed})" if total_suppressed else ""),
        lx,
        y,
        (0, 140, 255) if total_helmet_alert > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 26
    put(
        f"Checking     {total_checking}",
        lx,
        y,
        (0, 255, 255) if total_checking > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 26
    put(
        f"Granted      {total_granted}",
        lx,
        y,
        (0, 255, 0) if total_granted > 0 else (170, 170, 170),
        0.50,
        1,
    )

    y += 26
    put(
        f"Intruder     {total_unauth}",
        lx,
        y,
        (255, 0, 200) if total_unauth > 0 else (0, 255, 0),
        0.52,
        1,
    )

    y += 26
    put(
        f"Lying        {total_lying}",
        lx,
        y,
        (0, 165, 255) if total_lying > 0 else (0, 255, 0),
        0.50,
        1,
    )

    y += 30
    put(
        f"Emergency    {total_emg}",
        lx,
        y,
        (0, 0, 255) if total_emg > 0 else (0, 255, 0),
        0.62,
        2,
    )

    rx = width // 2 + 15
    y2 = 105
    put("WORKERS", rx, y2, (255, 255, 255), 0.56, 2)
    y2 += 30

    for det, _xy in unique[:6]:
        color = state_color(det)

        hold_text = ""
        if det.get("is_held", False):
            hold_text = f" HOLD {det.get('hold_elapsed', 0.0):.1f}s"

        line1 = f"{det.get('camera')}-#{display_id_text(det)} {det.get('entry_state')}{hold_text}"
        put(line1, rx, y2, color, 0.43, 1)
        y2 += 23

        line2 = (
            f"H={det.get('helmet_status')} "
            f"HG={det.get('helmet_alert')} "
            f"SUSP={det.get('helmet_suspicious')} "
            f"L={det.get('lying_candidate')}"
        )

        if det.get("entry_state") == "CHECKING":
            line2 += f" {det.get('entry_progress', 0):.1f}/{args.entry_check_sec:.0f}s"

        put(line2, rx + 8, y2, (190, 190, 190), 0.34, 1)
        y2 += 27

        if y2 > height - 105:
            break

    ly = height - 82
    draw_legend_item(panel, 24, ly, (0, 255, 0), "safe / granted")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 255, 255), "checking")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 200, 255), "suspicious")
    ly += 22
    draw_legend_item(panel, 24, ly, (0, 140, 255), "helmet alert blink")
    ly += 22
    draw_legend_item(panel, 24, ly, (255, 0, 200), "intruder blink (no helmet goal)")

    ly2 = height - 82
    put("cam0: ENTRY ROI only", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)
    ly2 += 22
    put("cam1: MONITOR ONLY", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)
    ly2 += 22
    put("EMG red / Intruder purple / Helmet orange", width // 2 + 15, ly2, (190, 190, 190), 0.40, 1)

    return panel