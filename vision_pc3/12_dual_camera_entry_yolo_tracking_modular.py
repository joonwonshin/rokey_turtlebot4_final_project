"""
12_dual_camera_entry_yolo_tracking_modular.py
=============================================

PC3 메인 스크립트. 두 대의 USB 웹캠(cam0/cam1)을 실시간 처리해
안전 상태(쓰러짐/헬멧 미착용/무단침입/입장허가)를 판정하고,
결과를 PC4로 ROS 2 토픽으로 발행한다.

파이프라인 요약:
    ┌─ cam0 ─┐  ┌──── CameraEntryTracker (pose track + helmet det)
    ├─ cam1 ─┤  ├──── WorkerStateStore   (3개 상태기계)
    │        │  ├──── GlobalFuser        (카메라 간 인격 통합)
    │        │  ├──── save_state_json    (로컬 상태 덤프)
    │        │  ├──── SafetyRosBridge    (PC4로 발행)
    │        │  ├──── draw_combined_map  (map 시각화)
    │        │  └──── 대시보드 조립 + imshow
    └────────┘

실행 모드:
    --mode set_roi : 지도 위에 입구 polygon을 마우스로 찍어 저장
    --mode run     : 실시간 감시 (기본)

주요 CLI:
    --no-publish-ros           : Windows 등 rclpy 미설치 환경에서 로컬 테스트
    --dedup-dist 0.5           : 위치 근접 중복 제거 임계 [m]
    --helmet-goal-fps 2.0      : helmet_goal follow 모드 발행 상한 [Hz]
    --helmet-goal-ema-alpha 0.3: helmet_goal 위치 스무딩 계수
    --ros-image-fps 15         : cam*/image/compressed 상한 fps
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from safety_lib import base_utils as utils
from safety_lib.state_io import resolve_path, load_entry_roi, set_roi_mode, save_state_json
from safety_lib.safety_logic import WorkerStateStore, apply_post_fusion_suppression
from safety_lib.vision_core import CameraEntryTracker
from safety_lib.dashboard_ui import draw_combined_map, draw_status_panel
from safety_lib.global_fusion import GlobalFuser


def prepare_paths(args, base_dir):
    """
    CLI에서 받은 상대 경로를 스크립트 위치 기준 절대 경로로 정규화한다.
    이렇게 하면 어느 디렉토리에서 스크립트를 실행해도 calibration/,
    yolo_experiments/ 등을 정확히 찾을 수 있음.
    """
    args.map_yaml = str(resolve_path(base_dir, args.map_yaml))
    args.cam0_homography = str(resolve_path(base_dir, args.cam0_homography))
    args.cam0_z_calib = str(resolve_path(base_dir, args.cam0_z_calib))
    args.cam1_homography = str(resolve_path(base_dir, args.cam1_homography))
    args.cam1_z_calib = str(resolve_path(base_dir, args.cam1_z_calib))
    args.entry_roi_json = str(resolve_path(base_dir, args.entry_roi_json))
    args.det_model = str(resolve_path(base_dir, args.det_model))
    args.pose_model = str(resolve_path(base_dir, args.pose_model))
    args.state_json = str(resolve_path(base_dir, args.state_json))

    return args


def run_mode(args):
    """
    실시간 감시 메인 루프. 카메라 오픈부터 종료까지 전 과정 담당.

    단계:
      1) map/ROI 로드
      2) 모델 경로 확인 (실제 로드는 CameraEntryTracker 내부, 카메라별 독립)
      3) WorkerStateStore × 2 (cam0=entry ROI, cam1=monitor only)
      4) CameraEntryTracker × 2 (카메라 오픈 + inference wrapper)
      5) GlobalFuser (카메라 간 인격 통합)
      6) SafetyRosBridge (rclpy 노드, 조건부 초기화)
      7) 메인 루프:
           - 카메라별 process()
           - GlobalFuser.update()
           - save_state_json()
           - bridge publish (image / state+goal / markers)
           - map/status 시각화 조립
           - imshow + key handling
      8) 종료: 카메라 릴리스 + bridge shutdown
    """
    # ------------------------------------------------------------------
    # 1) 지도 정보 + 입구 ROI 로드
    # ------------------------------------------------------------------
    map_info = utils.load_map_from_yaml(args.map_yaml)
    entry_roi_points, roi_data = load_entry_roi(args.entry_roi_json)

    print("======================================")
    print("Dual Camera Entry YOLO Tracking V12 Modular")
    print("--------------------------------------")
    print(f"cam0 id          : {args.cam0_id}")
    print(f"cam1 id          : {args.cam1_id}")
    print(f"det model        : {args.det_model}")
    print(f"pose model       : {args.pose_model}")
    print(f"tracker          : {args.tracker}")
    print(f"device           : {args.device}")
    print(f"imgsz            : {args.imgsz}")
    print(f"entry ROI        : {args.entry_roi_json}")
    print(f"entry check      : {args.entry_check_sec}s")
    print(f"helmet alert     : {args.helmet_alert_sec}s")
    print(f"helmet recovery  : {args.helmet_recovery_sec}s")
    print(f"track hold       : {args.hold_sec}s")
    print(f"confirm frames   : {args.confirm_frames}")
    print(f"emergency        : {args.emergency_sec}s")
    print(f"recovery         : {args.recovery_sec}s")
    print("cam0             : ENTRY ROI ENABLED")
    print("cam1             : MONITOR ONLY")
    print("======================================")

    # ------------------------------------------------------------------
    # 2) YOLO 모델 로드
    # ------------------------------------------------------------------
    # 사람 검출/추적/키포인트는 pose 모델(yolo11s-pose.pt)이 전담한다.
    # best.pt 는 helmet 만 담당 (person 클래스는 누운 사람을 학습한 적이 없어 사용 안 함).
    # 두 모델 모두 카메라별 독립 인스턴스로 CameraEntryTracker 안에서 로드된다.
    # pose 모델은 .track(persist=True) 로 ByteTrack 상태를 인스턴스에 들고 있으므로
    # 공유하면 cam0/cam1 의 track_id 가 뒤섞인다.
    det_model_path = Path(args.det_model).expanduser()
    pose_model_path = Path(args.pose_model).expanduser()

    if not det_model_path.exists():
        raise FileNotFoundError(f"det model not found: {det_model_path}")

    if not pose_model_path.exists():
        raise FileNotFoundError(f"pose model not found: {pose_model_path}")

    # ------------------------------------------------------------------
    # 3) WorkerStateStore × 2 — 카메라별 상태기계 저장소
    # ------------------------------------------------------------------
    # cam0은 entry ROI 판정 활성화 → ACCESS_GRANTED / UNAUTHORIZED 상태 사용.
    # cam1은 모니터링 전용 → emergency/helmet만 판정.
    store0 = WorkerStateStore(
        camera_name=args.cam0_name,
        entry_roi_points=entry_roi_points,
        required_check_sec=args.entry_check_sec,
        emergency_sec=args.emergency_sec,
        recovery_sec=args.recovery_sec,
        helmet_alert_sec=args.helmet_alert_sec,
        helmet_recovery_sec=args.helmet_recovery_sec,
        hold_sec=args.hold_sec,
        confirm_frames=args.confirm_frames,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
        max_workers=args.max_workers,
        entry_enabled=True,
    )

    store1 = WorkerStateStore(
        camera_name=args.cam1_name,
        entry_roi_points=None,
        required_check_sec=args.entry_check_sec,
        emergency_sec=args.emergency_sec,
        recovery_sec=args.recovery_sec,
        helmet_alert_sec=args.helmet_alert_sec,
        helmet_recovery_sec=args.helmet_recovery_sec,
        hold_sec=args.hold_sec,
        confirm_frames=args.confirm_frames,
        max_age_sec=args.track_max_age,
        history_len=args.history_len,
        max_workers=args.max_workers,
        entry_enabled=False,
    )

    # ------------------------------------------------------------------
    # 4) CameraEntryTracker × 2 — 카메라 오픈 + inference wrapper
    # ------------------------------------------------------------------
    # 각 tracker가 자기 카메라를 VideoCapture로 열고, 매 프레임:
    #   - YOLO pose track → person bbox + 17 keypoints + ByteTrack ID
    #   - best.pt        → helmet bbox
    #   - homography H → 발 map_xy [m]
    #   - projection P → head_z 추정
    #   - helmet/posture/hand_raise 판정
    #   - WorkerStateStore에 결과 전달 (스무딩 + 상태기계)
    cam0 = CameraEntryTracker(
        cam_id=args.cam0_id,
        cam_name=args.cam0_name,
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        homography_path=args.cam0_homography,
        z_calib_path=args.cam0_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        worker_store=store0,
        tracker_yaml=args.tracker,
        imgsz=args.cam0_imgsz,
    )

    cam1 = CameraEntryTracker(
        cam_id=args.cam1_id,
        cam_name=args.cam1_name,
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        homography_path=args.cam1_homography,
        z_calib_path=args.cam1_z_calib,
        map_info=map_info,
        homography_output=args.homography_output,
        worker_store=store1,
        tracker_yaml=args.tracker,
        imgsz=args.cam1_imgsz,
    )

    # ------------------------------------------------------------------
    # 5) GlobalFuser — 카메라 간 동일 인격 통합
    # ------------------------------------------------------------------
    # cam0의 "cam0:7"과 cam1의 "cam1:3"이 map 좌표 상 65cm 이내면
    # 같은 사람으로 판단해 동일 global_id 부여.
    # → save_state_json/bridge 모두 global_id로 dedup 가능.
    global_fuser = GlobalFuser(
        match_dist=args.fuse_match_dist,
        rebind_dist=args.fuse_rebind_dist,
        max_age_sec=args.fuse_max_age,
        ema_alpha=args.fuse_ema_alpha,
        merge_dist=args.fuse_merge_dist,
        handoff_dist=args.fuse_handoff_dist,
        reclaim_dist=args.fuse_reclaim_dist,
    )

    # ------------------------------------------------------------------
    # 6) SafetyRosBridge — PC4로 발행
    # ------------------------------------------------------------------
    # rclpy 임포트는 이 블록 안에서만 발생하도록 지연 임포트.
    # --no-publish-ros 또는 rclpy 미설치 시 bridge=None으로 유지.
    bridge = None
    if args.publish_ros:
        try:
            from safety_lib.ros_bridge import SafetyRosBridge

            bridge = SafetyRosBridge(
                cam_names=(args.cam0_name, args.cam1_name),
                node_name=args.ros_node_name,
                map_frame=args.ros_map_frame,
                jpeg_quality=args.jpeg_quality,
                image_fps=args.ros_image_fps,
                dedup_dist=args.dedup_dist,
                helmet_goal_fps=args.helmet_goal_fps,
                helmet_goal_ema_alpha=args.helmet_goal_ema_alpha,
                helmet_goal_min_move=args.helmet_goal_min_move,
                persons_json_fps=args.persons_json_fps,
            )
            print(f"[ROS] SafetyRosBridge enabled (image_fps={args.ros_image_fps})")
        except Exception as e:
            print(f"[ROS] disabled — bridge init failed: {e}")
            bridge = None
    else:
        print("[ROS] disabled by --no-publish-ros")

    window = "V12 Modular Entry YOLO Tracking Dashboard"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_loop_time = time.time()
    overall_fps = 0.0

    # ------------------------------------------------------------------
    # 7) 메인 루프
    # ------------------------------------------------------------------
    try:
        while True:
            now = time.time()

            # ─── 카메라별 처리 ─────────────────────────────────────────
            # process()는 (오버레이 그려진 preview, detection list, metrics)
            # 3-튜플을 반환. detection에는 이미 map_xy, helmet_status,
            # posture, emergency, helmet_alert, entry_state 등이 채워짐.
            cam0_view, cam0_tracked, cam0_metrics = cam0.process(args, now)
            cam1_view, cam1_tracked, cam1_metrics = cam1.process(args, now)

            # ─── 두 카메라 결과 병합 ────────────────────────────────────
            all_detections = list(cam0_tracked) + list(cam1_tracked)

            # ─── GlobalFuser로 카메라 간 인격 통합 ─────────────────────
            # 각 det에 global_id, global_map_xy 등 추가 필드가 붙는다.
            # 첫 프레임 co-observation은 병합 못하지만 ros_bridge의
            # 위치 근접 dedup(0.5m)이 안전망 역할.
            all_detections, _fused_summary = global_fuser.update(all_detections, now)

            # GlobalFuser 가 sticky unauthorized 를 사람 단위로 되돌려 쓴 뒤,
            # 헬멧 goal 억제를 다시 적용한다. cam1 의 상태기계는 그 사람이
            # cam0 입구에서 침입했다는 걸 모르기 때문.
            apply_post_fusion_suppression(all_detections)

            # ─── 로컬 JSON 상태 덤프 (디버깅/외부 감시 용도) ───────────
            save_state_json(
                args.state_json,
                timestamp=now,
                detections0=cam0_tracked,
                detections1=cam1_tracked,
                metrics0=cam0_metrics,
                metrics1=cam1_metrics,
            )

            # ─── ROS 2 발행 (bridge가 살아있을 때만) ──────────────────
            # 순서: 영상 → 상태/goal → 마커 → spin. spin은 논블록.
            if bridge is not None:
                bridge.publish_camera_frame(args.cam0_name, cam0_view)
                bridge.publish_camera_frame(args.cam1_name, cam1_view)
                bridge.publish_state_and_goals(all_detections)
                bridge.publish_persons_markers(all_detections)
                bridge.publish_persons_json(all_detections)
                bridge.spin_some()

            # ─── 로컬 대시보드 시각화 ─────────────────────────────────
            # PC3 본체에 붙어있는 오퍼레이터가 즉시 확인용. PC4로 나가는
            # 영상과는 별개의 로컬 창.
            map_view = draw_combined_map(
                map_info["img"],
                all_detections,
                map_info,
                entry_roi_points,
                dedup_dist=args.dedup_dist,
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
                dedup_dist=args.dedup_dist,
            )

            q_w = args.display_w // 2
            q_h = args.display_h // 2

            cam0_panel = utils.resize_keep_ratio(cam0_view, q_w, q_h)
            cam1_panel = utils.resize_keep_ratio(cam1_view, q_w, q_h)
            map_panel = utils.resize_keep_ratio(map_view, q_w, q_h)
            status_panel = utils.resize_keep_ratio(status_panel, q_w, q_h)

            top = np.hstack([cam0_panel, cam1_panel])
            bottom = np.hstack([map_panel, status_panel])
            dashboard = np.vstack([top, bottom])

            cv2.imshow(window, dashboard)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                store0.reset()
                store1.reset()
                # GlobalFuser 도 같이 비워야 한다. 안 그러면 local_to_global 에
                # 죽은 state_key 가 남아 새 트랙이 옛 global_id 로 REBIND 된다.
                global_fuser.reset()
                print("[RESET] worker states + global fuser reset")

    finally:
        cam0.release()
        cam1.release()
        cv2.destroyAllWindows()

        if bridge is not None:
            try:
                bridge.shutdown()
            except Exception as e:
                print(f"[ROS] shutdown warning: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="run", choices=["run", "set_roi"])

    # Camera
    parser.add_argument("--cam0-id", type=int, default=0)
    parser.add_argument("--cam0-name", type=str, default="cam0")
    parser.add_argument("--cam0-homography", type=str, default="calibration/cam0_to_map.npz")
    parser.add_argument("--cam0-z-calib", type=str, default="calibration/cam0_z_calib.npz")

    parser.add_argument("--cam1-id", type=int, default=2)
    parser.add_argument("--cam1-name", type=str, default="cam1")
    parser.add_argument("--cam1-homography", type=str, default="calibration/cam1_to_map.npz")
    parser.add_argument("--cam1-z-calib", type=str, default="calibration/cam1_z_calib.npz")

    # Map / ROI
    parser.add_argument("--map-yaml", type=str, default="final_project.yaml")
    parser.add_argument("--homography-output", type=str, default="map_meters", choices=["map_meters", "map_pixels"])

    parser.add_argument("--entry-roi-json", type=str, default="calibration/entry_roi.json")
    parser.add_argument("--entry-check-sec", type=float, default=3.0)

    # Models
    parser.add_argument("--det-model", type=str, default="yolo_experiments/best.pt")
    parser.add_argument("--pose-model", type=str, default="yolo11s-pose.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")

    # Inference
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--cam0-imgsz", type=int, default=1280,
                        help="cam0 는 1.8m 사람이 363px 로 작게 보인다. 960 으로 축소하면 272px 가 되어 "
                             "키포인트가 흔들린다. 원본 해상도 1280 권장.")
    parser.add_argument("--cam1-imgsz", type=int, default=960,
                        help="cam1 은 사람이 650px 로 크게 보여 960 으로 충분하다.")

    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--person-conf", type=float, default=0.45)
    parser.add_argument("--helmet-conf", type=float, default=0.25)
    parser.add_argument("--pose-conf", type=float, default=0.25)

    parser.add_argument("--person-min-area-ratio", type=float, default=0.002)
    parser.add_argument("--person-min-height", type=float, default=35)

    # Helmet / head
    parser.add_argument("--head-radius-scale", type=float, default=0.20)
    parser.add_argument("--helmet-expand", type=float, default=1.00)

    # Human / posture
    parser.add_argument("--user-height-m", type=float, default=1.80)

    parser.add_argument("--lying-height-thres", type=float, default=0.78)
    parser.add_argument("--very-low-height-thres", type=float, default=0.55)
    parser.add_argument("--leg-angle-thres", type=float, default=20.0,
                        help="다리 네 마디(좌우 허벅지/정강이) 중 가장 수직인 마디가 "
                             "'월드 수직'에서 이 각도 이상 벗어나야 누움 [deg]. "
                             "걷기·스쿼트·무릎꿇기는 항상 수직인 마디가 하나 이상 남는다. "
                             "합성검증 중앙값: 서있음 1.1, 걷기 2.0, 스쿼트 10.0, 누움 50.4.")
    parser.add_argument("--torso-angle-thres", type=float, default=25.0,
                        help="어깨→엉덩이 벡터가 '월드 수직'에서 이 각도 이상 벗어나야 누움 [deg]. "
                             "다리 조건과 AND. 허리굽힘(몸통 55deg, 다리 1deg)과 "
                             "바닥앉기(다리 48deg, 몸통 2deg)를 둘 다 걸러낸다.")
    parser.add_argument("--leg-upright-deg", type=float, default=8.0,
                        help="다리가 '서 있음을 증명' 하는 각도 [deg]. 한쪽 다리가 통째로 이보다 "
                             "수직이고 엉덩이도 높으면, 몸통이 수평이어도 쓰러짐으로 보지 않는다 "
                             "(허리 굽힘 방어).")
    parser.add_argument("--hip-upright-m", type=float, default=0.55,
                        help="다리 거부권의 엉덩이 높이 조건 [m]. 발목보다 이만큼 위에 있어야 "
                             "'서 있다' 고 인정한다.")
    parser.add_argument("--hip-low-thres", type=float, default=0.70,
                        help="엉덩이가 발목보다 이만큼 위에 없으면 LOW_POSTURE [m]. "
                             "서있음/숙임 0.85~0.89, 무릎꿇기 0.61, 웅크림 0.44.")
    parser.add_argument("--torso-short-m", type=float, default=0.34,
                        help="몸통(어깨→엉덩이)이 이보다 짧게 보이면 원근압축 = 카메라 축 방향 "
                             "누움 [m]. 사람 몸통은 실제 0.52m 이고, 서든 앉든 쪼그리든 "
                             "허리는 줄지 않는다. 각도로는 못 잡는 사각지대를 여기서 잡는다.")
    parser.add_argument("--body-len-min-m", type=float, default=0.85,
                        help="발목→머리가 이보다 짧게 보이면 '몸이 접힌 자세'(스쿼트/바닥앉기)로 "
                             "보고 위 두 규칙을 적용하지 않는다 [m]. 실제 1.65m. "
                             "누운 사람은 압축돼도 0.9m 이상 유지된다.")
    parser.add_argument("--hip-invert-m", type=float, default=-0.35,
                        help="엉덩이가 발목보다 이만큼 '아래' 로 보이면 인체 불가능 = 누움 [m]. "
                             "머리가 카메라 반대쪽을 향해 누우면 화면상 상하 순서가 뒤집힌다.")
    parser.add_argument("--residual-thres", type=float, default=90.0,
                        help="재투영 잔차가 이 값을 넘으면 head_z 를 버리고 형상 기반 판정으로 "
                             "강등 [px]. 정상 프레임 잔차 99.5%%: cam0 ~33px, cam1 ~89px. "
                             "누운 사람(수직선 가정 붕괴)은 120~1100px. cam1 의 H 오차(6.7cm)가 "
                             "커서 90 이 필요하다. cam1 H 를 다시 잡으면 40 까지 조일 수 있다.")

    parser.add_argument("--hand-raise-margin", type=float, default=20.0)

    # State timers
    parser.add_argument("--helmet-alert-sec", type=float, default=3.0)
    parser.add_argument("--helmet-recovery-sec", type=float, default=2.0)
    parser.add_argument("--emergency-sec", type=float, default=5.0)
    parser.add_argument("--recovery-sec", type=float, default=3.0)

    # Track persistence
    parser.add_argument("--hold-sec", type=float, default=0.8)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--track-max-age", type=float, default=3.0)
    parser.add_argument("--history-len", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=4)

    # UI
    parser.add_argument("--display-w", type=int, default=1600)
    parser.add_argument("--display-h", type=int, default=900)
    parser.add_argument("--process-every", type=int, default=1)

    # Output state
    parser.add_argument("--state-json", type=str, default="outputs/safety_state_entry_v12.json")

    # Global fusion (cam0/cam1 좌표 기반 중복 제거)
    parser.add_argument("--fuse-match-dist", type=float, default=0.65,
                        help="새 detection을 기존 global track에 붙일 최대 map 거리 [m]")
    parser.add_argument("--fuse-rebind-dist", type=float, default=1.20,
                        help="이미 같은 local_key로 붙어있던 track을 유지할 최대 거리 [m]")
    parser.add_argument("--fuse-max-age", type=float, default=8.0,
                        help="global track 을 삭제하기 전 대기 시간 [s]. 카메라 간 이동 중 "
                             "잠깐 아무 카메라에도 안 잡히는 구간이 있으므로 넉넉해야 한다. "
                             "짧으면 침입자 딱지(sticky_unauthorized)를 잃는다.")
    parser.add_argument("--fuse-reclaim-dist", type=float, default=3.00,
                        help="그 카메라가 지금 안 보고 있는 global track 을 이 거리 [m] 이내에서 "
                             "되찾는다. cam1 에 오래 머문 뒤 cam0 로 돌아올 때 ByteTrack 이 새 id 를 "
                             "주고 두 카메라 시야가 겹치지 않아 REBIND/NEAREST 가 모두 실패하는 것을 "
                             "막는다. 0 이면 비활성.")
    parser.add_argument("--fuse-ema-alpha", type=float, default=0.40,
                        help="융합 좌표 EMA 스무딩. 1.0=끔. 낮출수록 부드럽지만 지연이 생긴다. "
                             "카메라가 바뀔 때 좌표가 튀는 것을 완화한다.")
    parser.add_argument("--fuse-handoff-dist", type=float, default=1.50,
                        help="카메라 간 핸드오프 [m]. 두 카메라 시야가 겹치지 않으면 "
                             "cam1->cam0 이동 시 cam1 의 hold 잔영(0.8s)이 두 번째 점으로 남는다. "
                             "held detection 이 '다른 카메라' 의 live 와 이 거리 이내면 잔영으로 보고 버린다. "
                             "0 이면 비활성.")
    parser.add_argument("--fuse-merge-dist", type=float, default=0.60,
                        help="서로 다른 카메라만 보고 있고 이 거리 [m] 이내인 두 global track 을 "
                             "하나로 병합. 한 프레임의 나쁜 좌표로 갈라진 gid 를 되돌린다. "
                             "0 이면 병합 안 함.")

    # ROS bridge
    parser.add_argument("--publish-ros", dest="publish_ros", action="store_true",
                        default=True, help="PC4로 ROS 2 토픽 발행 (기본 ON)")
    parser.add_argument("--no-publish-ros", dest="publish_ros", action="store_false",
                        help="ROS 발행 비활성화 (rclpy 미설치 환경용)")
    parser.add_argument("--ros-node-name", type=str, default="pc3_safety_bridge")
    parser.add_argument("--ros-map-frame", type=str, default="map",
                        help="PoseStamped header.frame_id")
    parser.add_argument("--persons-json-fps", type=float, default=5.0,
                        help="/safety/persons_json 발행 상한 [Hz]. 웹 대시보드용. "
                             "상태가 바뀌면 상한 무시하고 즉시 발행한다.")
    parser.add_argument("--ros-image-fps", type=float, default=15.0,
                        help="cam0/cam1 CompressedImage 발행 상한 fps")
    parser.add_argument("--jpeg-quality", type=int, default=75,
                        help="CompressedImage JPEG 품질 (1~100)")
    parser.add_argument("--dedup-dist", type=float, default=0.5,
                        help="goal/marker 발행 시 위치 근접 dedup 임계값 [m]. "
                             "GlobalFuser가 첫 프레임 co-observation을 못 합치는 경우의 안전망. "
                             "0.0으로 두면 이 2차 dedup 비활성.")

    # Helmet goal follow-mode
    parser.add_argument("--helmet-goal-fps", type=float, default=2.0,
                        help="helmet_goal 재발행 최대 주기 [Hz]. Nav2 replan 부하 조절.")
    parser.add_argument("--helmet-goal-ema-alpha", type=float, default=0.3,
                        help="helmet_goal 위치 EMA 스무딩 계수. 1.0=raw, 0.3=부드럽게.")
    parser.add_argument("--helmet-goal-min-move", type=float, default=0.10,
                        help="이전 goal 대비 이 거리 [m] 이상 움직였을 때만 재발행. "
                             "미세한 흔들림 필터.")

    args = parser.parse_args()

    detection_dir = Path(__file__).resolve().parent
    args = prepare_paths(args, detection_dir)

    if args.mode == "set_roi":
        set_roi_mode(args)
    else:
        run_mode(args)


if __name__ == "__main__":
    main()