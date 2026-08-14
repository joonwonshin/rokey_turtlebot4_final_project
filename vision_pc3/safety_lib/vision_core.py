"""
safety_lib.vision_core
======================

카메라 1대의 실시간 처리 전 과정을 감싸는 CameraEntryTracker.

책임:
    - VideoCapture 오픈 + 매 프레임 read()
    - YOLO track (person + helmet, ByteTrack)
    - YOLO pose (17 keypoints)
    - person ↔ pose IoU 매칭
    - homography H로 발 → map 좌표 변환
    - projection P로 머리 높이 head_z 추정
    - helmet/posture/hand_raise 3종 판정
    - WorkerStateStore로 상태 갱신
    - preview 프레임에 오버레이 그리기

메인 스크립트는 CameraEntryTracker × 2 (cam0/cam1)만 생성하고
매 프레임 process()만 호출하면 됨.
"""
import time

import cv2
import numpy as np
from ultralytics import YOLO

from . import base_utils as utils
from .safety_logic import judge_hand_raise, judge_posture_z_priority
from .dashboard_ui import (
    draw_entry_roi_on_camera,
    draw_person_overlay,
    calc_metrics,
)


def get_class_name(names, cls_id):
    """
    Ultralytics 모델의 names(dict 또는 list)에서 class id → 이름 변환.
    실패 시 문자열화한 id 반환 (안전 폴백).
    """
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), str(cls_id)))
        return str(names[int(cls_id)])
    except Exception:
        return str(cls_id)


def extract_helmet_boxes(result, names, helmet_conf_thres=0.25):
    """
    best.pt 결과에서 helmet box 만 뽑는다.

    person 클래스는 더 이상 쓰지 않는다. 이유:
      학습 데이터의 person 라벨 220개 중 w/h > 1.0 인 것이 0개 → 누운 사람을
      단 한 장도 본 적이 없다. 그래서 쓰러진 사람을 아예 검출하지 못한다.
      사람 검출은 COCO-Pose 로 사전학습된 pose 모델이 담당한다.
    """
    helmets = []

    if result.boxes is None:
        return helmets

    boxes = result.boxes
    if boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
        return helmets

    xyxys = boxes.xyxy.detach().cpu().numpy().astype(float)
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy().astype(float)

    for i in range(len(xyxys)):
        if get_class_name(names, int(clss[i])).lower() != "helmet":
            continue

        conf = float(confs[i])
        if conf < helmet_conf_thres:
            continue

        helmets.append({"box": xyxys[i].tolist(), "conf": conf, "track_id": None})

    return helmets


def extract_pose_persons(
    result,
    image_area,
    person_conf_thres=0.45,
    person_min_area_ratio=0.002,
    person_min_height=35,
    keypoint_conf_fallback=1.0,
):
    """
    yolo11s-pose 의 track 결과에서 person + 17 keypoints + track_id 를 뽑는다.

    pose 모델을 사람 검출기로 쓰는 이유:
      * COCO Keypoints (약 15만 person 인스턴스) 로 사전학습 → 눕기/앉기/가림 모두 포함
      * 17개 부위를 개별로 맞혀야 하므로 통짜 실루엣이 아니라 '해부학'을 학습 →
        사람이 90도 돌아누워도 부위 검출기가 그대로 발화한다
      * 사람당 박스가 정확히 하나 → 조각난 중복 박스가 생기지 않는다
      * 키포인트가 박스에 이미 묶여 나오므로 IoU 매칭 자체가 불필요하다

    Returns:
        [{box, conf, track_id, pose_item, source}, ...]
        pose_item = {'keypoints': (17,2), 'conf': (17,), 'box': [x1,y1,x2,y2]}
    """
    persons = []

    if result.boxes is None or result.boxes.xyxy is None:
        return persons

    xyxys = result.boxes.xyxy.detach().cpu().numpy().astype(float)
    confs = result.boxes.conf.detach().cpu().numpy().astype(float)

    if result.boxes.id is not None:
        ids = result.boxes.id.detach().cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxys)

    kxy = None
    kcf = None
    if result.keypoints is not None:
        try:
            kxy = result.keypoints.xy.detach().cpu().numpy()
        except Exception:
            kxy = None
        try:
            kcf = result.keypoints.conf.detach().cpu().numpy()
        except Exception:
            kcf = None

    for i in range(len(xyxys)):
        conf = float(confs[i])
        if conf < person_conf_thres:
            continue

        xyxy = xyxys[i].tolist()
        x1, y1, x2, y2 = xyxy
        bw = x2 - x1
        bh = y2 - y1

        if utils.box_area(xyxy) / max(image_area, 1) < person_min_area_ratio:
            continue

        # 누운 사람은 bh 가 작고 bw 가 크다. 그래서 max(bh, bw) 로 본다.
        if max(bh, bw) < person_min_height:
            continue

        if kxy is not None and i < len(kxy):
            kpts = kxy[i].astype(np.float64)
            kconf = (kcf[i].astype(np.float64) if kcf is not None and i < len(kcf)
                     else np.full(len(kpts), float(keypoint_conf_fallback)))
            pose_item = {"keypoints": kpts, "conf": kconf, "box": xyxy}
        else:
            pose_item = None

        persons.append(
            {
                "box": xyxy,
                "conf": conf,
                "track_id": ids[i],
                "pose_item": pose_item,
                "source": "yolo_pose_track",
            }
        )

    return persons


class CameraEntryTracker:
    """
    카메라 1대의 실시간 감지·판정·시각화를 담당하는 orchestrator.

    구조:
        __init__ : 모델 로드 + 캘리브 로드 + VideoCapture 오픈
        process  : 매 프레임 호출, (preview, tracked, metrics) 반환
        release  : 카메라 릴리스

    process()의 단계:
        1. cap.read()로 프레임 취득
        2. YOLO track (person + helmet, ByteTrack persist)
        3. YOLO pose (17 keypoints)
        4. Per-person:
           - pose ↔ person IoU 매칭
           - 발/머리 픽셀 좌표 추출
           - H로 발 → map_xy [m]
           - P로 head_z [m] 추정
           - helmet_status / posture / hand_raise 판정
           - WorkerStateStore.update_one()
        5. hold_sec 이내 소실 track 재삽입 (add_held_detections)
        6. display_id 부여 (compact 0~3)
        7. preview에 오버레이 그리기
        8. metrics 계산 (person_count, emergency_count, ...)

    args.process_every로 프레임 스킵 가능 (기본 1 = 매 프레임).
    스킵되는 프레임은 이전 preview/tracked/metrics 그대로 반환.
    """

    def __init__(
        self,
        cam_id,
        cam_name,
        det_model_path,
        pose_model_path,
        homography_path,
        z_calib_path,
        map_info,
        homography_output,
        worker_store,
        tracker_yaml,
        imgsz=None,
        width=1280,
        height=720,
        fps=15,
    ):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.map_info = map_info
        self.homography_output = homography_output
        self.worker_store = worker_store
        self.tracker_yaml = tracker_yaml

        # 카메라별 추론 해상도 오버라이드. cam0 는 사람이 절반 크기로 보여
        # (1.8m 사람이 363px vs cam1 650px) 축소 손실이 키포인트를 흔든다.
        self.imgsz = imgsz

        # ---- helmet 검출 모델 (best.pt) ----
        # person 클래스는 쓰지 않는다 (extract_helmet_boxes 주석 참고).
        # 추적이 필요 없으므로 predict 만 하며, 카메라 간 상태 공유 문제도 없다.
        self.det_model = YOLO(str(det_model_path))
        self.det_names = self.det_model.names

        # ---- 사람 검출 + 키포인트 + 추적 (yolo11s-pose) ----
        # 카메라마다 독립 인스턴스여야 한다. .track(persist=True) 는 모델
        # 인스턴스 안에 ByteTrack 상태를 들고 있어서, 하나를 두 카메라가
        # 공유하면 cam0 의 track_id 와 cam1 의 track_id 가 뒤섞인다.
        self.pose_model = YOLO(str(pose_model_path))

        # ---- 캘리브레이션 로드 ----
        # H (3x3): 카메라 픽셀 → map 미터 좌표
        self.H, self.H_key = utils.load_homography(homography_path)

        # H_inv: map → 카메라 픽셀 역변환 (Entry ROI 시각화용)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except Exception as e:
            print(f"[WARN] {cam_name} inverse homography failed: {e}")
            self.H_inv = None

        # P (3x4): world (X,Y,Z) → 카메라 픽셀 (head_z 추정용)
        self.P = utils.load_z_calibration(z_calib_path)

        # ---- 카메라 오픈 ----
        # V4L2 백엔드 + MJPG 사용 → USB 대역폭 절약 + 프레임 지연 감소
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open {cam_name}, camera id={cam_id}")

        # ---- 런타임 상태 ----
        self.frame_idx = 0        # 총 프레임 카운터 (process_every 스킵용)
        self.prev_time = time.time()
        self.fps = 0.0             # EMA로 갱신되는 실측 fps

        # 스킵되는 프레임엔 이전 결과를 재사용 → 대시보드 프리즈 방지
        self.last_preview = None
        self.last_tracked = []
        self.last_metrics = {}

        print(f"[INFO] {cam_name}")
        print(f"  camera id  : {cam_id}")
        print(f"  H          : {homography_path}")
        print(f"  H key      : {self.H_key}")
        print(f"  z calib    : {z_calib_path}")
        print(f"  tracker    : {tracker_yaml}")
        print(f"  imgsz      : {imgsz if imgsz else '(global)'}")
        print(f"  entry      : {self.worker_store.entry_enabled}")

    def release(self):
        """카메라 하드웨어 자원 릴리스. 종료 시 반드시 호출."""
        self.cap.release()

    def process(self, args, now):
        """
        매 프레임 처리 메인 진입점.

        Args:
            args : CLI argparse namespace
            now  : time.time()
        Returns:
            (preview_bgr, tracked_detections, metrics_dict)
        """
        ret, frame = self.cap.read()

        # 프레임 읽기 실패 (V4L2 오버플로우, 케이블 문제 등) → 이전 값 재사용
        # 완전 첫 프레임에서 실패한 경우엔 검은 화면 + 에러 텍스트로 폴백
        if not ret:
            print(f"[WARN] {self.cam_name} frame read failed")

            if self.last_preview is not None:
                return self.last_preview, self.last_tracked, self.last_metrics

            empty = np.zeros((720, 1280, 3), dtype=np.uint8)
            utils.draw_text(
                empty,
                f"{self.cam_name} frame read failed",
                (40, 80),
                (0, 0, 255),
                1.0,
                2,
            )
            return empty, [], {}

        self.frame_idx += 1

        image_h, image_w = frame.shape[:2]
        image_area = image_h * image_w

        imgsz = self.imgsz or args.imgsz

        # process_every로 프레임 스킵. GPU 부하 조절용. 기본 1 (매 프레임).
        # 스킵 프레임은 캐시된 last_* 반환 → 상태기계 시간 축은 그대로 흐름.
        run_inference = (
            self.frame_idx % args.process_every == 0
        ) or self.last_preview is None

        if not run_inference:
            return self.last_preview, self.last_tracked, self.last_metrics

        # ================================================================
        # 1. YOLO pose track: 사람 bbox + 17 keypoints + ByteTrack ID
        # ================================================================
        # 사람 검출을 pose 모델이 담당한다. persist=True 로 track_id 를 이어가고,
        # WorkerStateStore 가 그 ID 로 상태기계를 이어간다.
        # 키포인트가 박스에 묶여 나오므로 pose ↔ person IoU 매칭이 필요 없다.
        pose_result = self.pose_model.track(
            frame,
            imgsz=imgsz,
            conf=args.person_conf,
            device=args.device,
            persist=True,
            tracker=self.tracker_yaml,
            verbose=False,
        )[0]

        persons = extract_pose_persons(
            pose_result,
            image_area=image_area,
            person_conf_thres=args.person_conf,
            person_min_area_ratio=args.person_min_area_ratio,
            person_min_height=args.person_min_height,
        )

        # 오버레이(torso 선 등)가 참조하는 pose_item 목록
        pose_items = [p["pose_item"] for p in persons if p.get("pose_item") is not None]

        # ================================================================
        # 2. helmet 검출 (best.pt). 추적 불필요 → predict 만.
        # ================================================================
        det_result = self.det_model(
            frame,
            imgsz=imgsz,
            conf=args.det_conf,
            device=args.device,
            verbose=False,
        )[0]

        helmets = extract_helmet_boxes(
            det_result,
            self.det_names,
            helmet_conf_thres=args.helmet_conf,
        )

        # ================================================================
        # 3. Preview 준비: 원본 프레임 복사 + Entry ROI 오버레이 (cam0만)
        # ================================================================
        preview = frame.copy()

        # cam0만 Entry ROI 표시
        if self.worker_store.entry_enabled:
            preview = draw_entry_roi_on_camera(
                preview,
                self.H_inv,
                self.map_info,
                self.worker_store.entry_roi_points,
                self.homography_output,
                self.cam_name,
            )

        # helmet boxes
        for helmet in helmets:
            utils.draw_box(
                preview,
                helmet["box"],
                (0, 255, 255),
                f"helmet {helmet['conf']:.2f}",
                1,
            )

        tracked_outputs = []

        # ================================================================
        # 4. Per person: 매칭 → 좌표 변환 → 판정 → 상태 업데이트
        # ================================================================
        for person in persons:
            pbox = person["box"]
            yolo_track_id = person.get("track_id")

            # 4-1) 키포인트는 pose track 결과에 이미 묶여 있다 (IoU 매칭 불필요)
            pose_item = person.get("pose_item")

            # 4-2) 이 사람과 가장 관련 있는 helmet box 찾기 (근접+IoU 점수)
            nearest_helmet, _ = utils.find_nearest_related_helmet(
                pbox,
                helmets,
                image_w,
                image_h,
            )
            nearest_helmet_box = nearest_helmet["box"] if nearest_helmet is not None else None

            # 4-3) 머리 픽셀 좌표: 얼굴 키포인트 평균 → 안 되면 헬멧 → person top
            head_px, head_src = utils.get_head_point(
                pose_item,
                person_box=pbox,
                helmet_box=nearest_helmet_box,
                conf_thres=args.pose_conf,
            )

            # 4-4) 발 픽셀 좌표: 발목 키포인트 → 안 되면 bbox 하단 중앙
            foot_px, foot_src = utils.get_foot_point(
                pose_item,
                pbox,
                conf_thres=args.pose_conf,
            )

            # 4-5) 발 픽셀 → map 미터 좌표 (호모그래피 H)
            # 지면 평면(z=0) 가정. 결과는 (x, y) [m] in map frame.
            foot_map = utils.camera_pixel_to_map_meter(
                self.H,
                foot_px,
                self.map_info,
                self.homography_output,
            )

            head_z = None
            z_residual = None
            projected_uv = None

            # 4-6) 머리 높이 head_z 추정 (수직선 상에서 P 역투영)
            # 발이 서 있는 (X,Y) 위에서 z를 변수로 두고, head_px에 맞는 z를 풀이.
            if head_px is not None and foot_map is not None:
                head_z, z_residual, projected_uv = utils.estimate_z_on_vertical_line(
                    self.P,
                    foot_map[0],
                    foot_map[1],
                    head_px,
                )

            # 4-7) 헬멧 상태 판정: 머리 주변 helmet box 존재 여부
            #      HELMET_ON / NO_HELMET_MISSING / NO_HELMET_RELATED / SUSPICIOUS
            helmet_status, head_radius, related_helmets = utils.judge_helmet_status(
                pbox,
                helmets,
                head_px,
                head_src,
                image_w,
                image_h,
                head_radius_scale=args.head_radius_scale,
                helmet_expand_ratio=args.helmet_expand,
            )

            # 4-7b) 몸통 축 분석: 캘리브의 '월드 수직' 기준으로 자세 각도 계산
            # 머리 하나가 아니라 어깨/엉덩이/전체 키포인트를 본다. 위에서
            # 내려다보는 카메라에서 얼굴 키포인트가 가려져 head_px 가 튀어도
            # 어깨·엉덩이는 안정적이다.
            body_axis = utils.analyze_body_axis(
                self.P,
                foot_map,
                foot_px,
                pose_item,
                conf_thres=args.pose_conf,
                user_height_m=args.user_height_m,
            )

            # 4-8) 자세 판정: 몸통 축 각도 우선 → head_z 보조
            lying_candidate, posture, debug = judge_posture_z_priority(
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
                body_axis=body_axis,
                leg_angle_thres=args.leg_angle_thres,
                torso_angle_thres=args.torso_angle_thres,
                hip_low_thres=args.hip_low_thres,
                leg_upright_deg=args.leg_upright_deg,
                hip_upright_m=args.hip_upright_m,
                torso_short_m=getattr(args, "torso_short_m", 0.34),
                body_len_min_m=getattr(args, "body_len_min_m", 0.85),
                hip_invert_m=getattr(args, "hip_invert_m", -0.35),
            )

            # 4-9) 손 들기 판정 (ACCESS_GRANTED 조건 중 하나)
            hand_raised, hand_side, hand_debug = judge_hand_raise(
                pose_item,
                conf_thres=args.pose_conf,
                margin_px=args.hand_raise_margin,
            )

            # 4-10) 이 사람의 모든 결과를 dict으로 묶어 WorkerStateStore로
            det_item = {
                "camera": self.cam_name,
                "yolo_track_id": yolo_track_id,

                "image_point": foot_px,
                "map_xy": foot_map,

                "helmet_status": helmet_status,
                "posture": posture,
                "lying_candidate": lying_candidate,

                "hand_raised": hand_raised,
                "hand_side": hand_side,
                "hand_debug": hand_debug,

                "debug": debug,
                "person_box": pbox,
                "pose_item": pose_item,
                "source": person["source"],
                "conf": person["conf"],

                "head_px": head_px,
                "head_src": head_src,
                "head_radius": head_radius,
                "foot_px": foot_px,
                "foot_src": foot_src,
                "projected_uv": projected_uv,
            }

            # 4-11) WorkerStateStore가 스무딩 + 3개 상태기계 갱신
            # det_item에 emergency/helmet_alert/entry_state 등 필드 추가됨
            det_item = self.worker_store.update_one(det_item, now)
            tracked_outputs.append(det_item)

        # ================================================================
        # 5. 트랙 유지 (hold_sec) + display_id + 만료 track 정리
        # ================================================================
        # 5-1) 이번 프레임에 detection 없는 confirmed track을 hold_sec 유지
        tracked_outputs = self.worker_store.add_held_detections(tracked_outputs, now)
        # 5-2) 대시보드 표시용 작은 번호(0~3) 부여
        tracked_outputs = self.worker_store.assign_compact_display_ids(tracked_outputs)
        # 5-3) max_age_sec 넘은 트랙 삭제
        self.worker_store.cleanup(now)

        # ================================================================
        # 6. 오버레이 그리기 (사람별 bbox, 상태, keypoint, ROI 등)
        # ================================================================
        for det in tracked_outputs:
            draw_person_overlay(preview, det, pose_items, args, image_h)

        # ================================================================
        # 7. 상단 헤더 (카메라 이름, 모드, 인원수, FPS)
        # ================================================================
        overlay = preview.copy()
        cv2.rectangle(overlay, (8, 8), (850, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, preview, 0.50, 0, preview)

        mode_text = "ENTRY" if self.worker_store.entry_enabled else "MONITOR_ONLY"

        utils.draw_text(
            preview,
            f"{self.cam_name} | {mode_text} | YOLO TRACK | persons:{len(tracked_outputs)} helmets:{len(helmets)} FPS:{self.fps:.1f}",
            (18, 42),
            (255, 255, 255),
            0.62,
            2,
        )

        # 8. FPS 계산 (EMA 스무딩)
        # 실시간성 확인용. 값이 급락하면 GPU 부하 or V4L2 문제 의심.
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            self.fps = self.fps * 0.90 + (1.0 / dt) * 0.10

        # 9. 대시보드용 메트릭 집계 (person/helmet/emergency 카운트 등)
        metrics = calc_metrics(tracked_outputs, helmets)
        metrics["fps"] = self.fps
        metrics["cam_name"] = self.cam_name

        self.last_preview = preview
        self.last_tracked = tracked_outputs
        self.last_metrics = metrics

        return preview, tracked_outputs, metrics