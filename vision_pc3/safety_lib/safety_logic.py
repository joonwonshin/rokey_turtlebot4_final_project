"""
safety_lib.safety_logic
=======================

WorkerStateStore + 판정 스무딩 로직.

이 파일이 담당하는 것:
  1) point_in_polygon         — 입구 ROI 폴리곤 내부 판단 (ray casting)
  2) judge_hand_raise         — 손 들었는지 (wrist_y < shoulder_y)
  3) judge_posture_z_priority — 쓰러짐 판정 (head_z 우선 + 형상 보조)
  4) WorkerStateStore         — 트랙별 상태기계 3개
       ├─ Emergency  : 쓰러짐 5초 지속 → True, 회복 3초 지속 → False
       ├─ HelmetAlert: NO_HELMET 3초 지속 → True, HELMET_ON 2초 지속 → False
       └─ Entry      : ROI 안 + 손 듦 + 헬멧 3초 → ACCESS_GRANTED
                       ROI에서 벗어남 → UNAUTHORIZED / ENTERED / OUTSIDE

핵심 개념:
  * 히스토리 다수결 스무딩 — 매 프레임 흔들리는 raw 판정을 최근 N프레임
    다수결로 부드럽게. history_len=6이면 최근 6프레임 중 과반이면 True.
  * hold_sec — YOLO track이 순간적으로 놓쳐도 confirmed track은
    hold_sec(0.8s) 동안 이전 상태를 유지 (깜빡임 방지).
  * 3개 상태기계는 각자 독립. 하나가 True여도 다른 것에 영향 없음.
"""
from collections import deque

import numpy as np

from . import base_utils as utils


def point_in_polygon(point, polygon):
    """
    Ray casting (crossing number) 알고리즘으로 점이 폴리곤 내부에 있는지 판단.

    입구 ROI 판정에 사용. polygon은 map 좌표계의 [(x,y), ...] 리스트.
    점에서 +x 방향으로 반직선을 쐈을 때 폴리곤 변과 교차한 횟수가 홀수면 내부.

    Args:
        point   : (x, y) or None
        polygon : [(x1,y1), (x2,y2), ...] 3점 이상
    Returns:
        bool. 점이 None이거나 폴리곤이 3점 미만이면 False. 다각형을 생성하지 못한다는 뜻
    """
    if point is None or polygon is None or len(polygon) < 3:
        return False

    x, y = float(point[0]), float(point[1])

    inside = False
    j = len(polygon) - 1  # 이전 정점 인덱스 (초기값: 마지막 정점)

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # 변 (i, j)가 수평선 y를 위/아래로 걸치고,    ----> 쉽게 말해 겹치는 부분구하는 방법에 대한 코드
        # 그 교차점이 점의 x보다 오른쪽에 있으면 카운트 토글
        # (yj - yi + 1e-12): 수평 변 division-by-zero 방지
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi # 수평방지 1e-12
        )

        if intersect:
            inside = not inside

        j = i

    return inside


def judge_hand_raise(pose_item, conf_thres=0.25, margin_px=20):
    """
    손을 들었는지 판단. 입구 ACCESS_GRANTED 조건 중 하나.

    기준: wrist(손목) y좌표가 shoulder(어깨)보다 margin_px 픽셀만큼 위에 있으면 든 것.
    (이미지 좌표계는 y가 아래로 증가하므로 wrist_y < shoulder_y가 위쪽)

    Args:
        pose_item  : YOLO pose 결과 dict {'keypoints', 'conf', 'box'}
        conf_thres : 키포인트 신뢰도 임계값
        margin_px  : 흔들림 방지 마진 (20 px)
    Returns:
        (raised: bool, side: 'LEFT'|'RIGHT'|'BOTH'|'NONE', debug: dict)
    """
    if pose_item is None:
        return False, "NONE", {}

    # COCO 17 keypoint 인덱스는 base_utils에 상수로 정의됨
    lw = utils.get_visible_point(pose_item, utils.LEFT_WRIST, conf_thres)
    rw = utils.get_visible_point(pose_item, utils.RIGHT_WRIST, conf_thres)
    ls = utils.get_visible_point(pose_item, utils.LEFT_SHOULDER, conf_thres)
    rs = utils.get_visible_point(pose_item, utils.RIGHT_SHOULDER, conf_thres)

    left_raise = False
    right_raise = False

    # 좌우 각각 독립적으로 판정 → 한쪽만 들어도 인정
    if lw is not None and ls is not None:
        left_raise = float(lw[1]) < float(ls[1]) - margin_px

    if rw is not None and rs is not None:
        right_raise = float(rw[1]) < float(rs[1]) - margin_px

    if left_raise and right_raise:
        side = "BOTH"
    elif left_raise:
        side = "LEFT"
    elif right_raise:
        side = "RIGHT"
    else:
        side = "NONE"

    return left_raise or right_raise, side, {
        "left_raise": left_raise,
        "right_raise": right_raise,
    }


def judge_posture_z_priority(
    person_box,
    pose_item,
    head_z,
    z_residual,
    image_h,
    user_height_m=1.80,
    pose_conf=0.25,
    lying_height_thres=0.78,
    very_low_height_thres=0.55,
    residual_thres=160.0,
    body_axis=None,
    leg_angle_thres=20.0,
    torso_angle_thres=25.0,
    hip_low_thres=0.70,
    leg_upright_deg=8.0,
    hip_upright_m=0.55,   # 미사용 (누운 자세에서 hip 높이는 신뢰 불가)
    torso_short_m=0.34,   # 몸통이 이보다 짧게 보이면 원근압축 = 카메라축 누움
    body_len_min_m=0.85,  # 몸이 이보다 짧게 보이면 '접힌 자세'. 압축과 구분하는 게이트
    hip_invert_m=-0.35,   # 엉덩이가 발목보다 이만큼 아래 = 인체 불가능
):
    """
    쓰러짐 판정 v13 — 몸통 축 각도를 주 지표로, head_z 를 보조로 쓴다.

    ── 왜 z 를 무조건 믿으면 안 되는가 ──
    head_z 는 "머리는 발 바로 위에 있다"는 가정 위에서 역산된다. 그런데
    사람이 누우면 머리가 발에서 1.7m 떨어져 그 가정이 깨진다. 하필
    검출하려는 바로 그 상황에서 전제가 무너진다.

    다행히 가정이 깨진 정도를 재는 값이 있다: 재투영 잔차 z_residual.
    u식과 v식이 일치하면 0 에 가깝고, 어긋날수록 커진다.
      정상 프레임 (서있음/웅크림) : 5 ~ 20 px   (99% 상한 70px)
      누운 사람 (가정 붕괴)       : 120 ~ 1100 px
    그래서 z_residual > residual_thres 면 head_z 를 버리고 형상 기반
    판정(judge_posture_v09)으로 강등한다. 누운 사람은 bbox 가 가로로
    길어서 그쪽에서 잡힌다.

    판정 계층 (z 를 믿을 때만):
      head_z < very_low_height_thres(0.55m)         → LYING (형상 불필요)
      head_z < lying_height_thres(0.78m)   + 형상   → LYING
      head_z < user_height_m * 0.55        + 형상   → LYING
      head_z < user_height_m * 0.65                 → LOW_POSTURE (쪼그림)
      그 외                                          → NORMAL

    0.78m 구간에 형상 조건을 AND 로 건 이유: 무릎 꿇고 작업하는 사람의
    머리가 0.7m 대로 내려온다. "낮으면서 + 가로로 누워야" 쓰러짐이다.
    (judge_posture_v09 는 원래 이 조건을 요구했는데 v12 에서 빠져 있었다.)

    Args:
        person_box       : [x1,y1,x2,y2] 사람 bbox (px)
        pose_item        : YOLO pose dict
        head_z           : 추정 머리 높이 [m] or None
        z_residual       : 재투영 잔차 [px]. None 이면 신뢰 불가로 처리.
        image_h          : 프레임 높이 (미사용, 인터페이스 호환용)
        user_height_m    : 표준 신장 [m]
        pose_conf        : 키포인트 conf 임계값
        lying_height_thres    : 이 이하 + 형상이면 LYING [m]
        very_low_height_thres : 이 이하면 형상 없이 LYING [m]
        residual_thres   : 이 값을 넘으면 head_z 를 버린다 [px]
    Returns:
        (lying: bool, posture: str, debug: dict)
    """
    # bbox 형상 지표
    x1, y1, x2, y2 = person_box
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    aspect = pw / ph  # >1이면 가로로 긴 bbox = 누워있을 가능성

    # 몸통 수평 여부 (어깨-엉덩이 벡터의 dx/dy 비교)
    try:
        torso_horizontal, _, _ = utils.get_torso_horizontal(pose_item, pose_conf)
    except Exception:
        torso_horizontal = False

    # 키포인트 분포 종횡비 (넓게 퍼져 있으면 누움)
    try:
        kpt_ratio, visible_kpts = utils.get_keypoint_spread_ratio(pose_item, pose_conf)
    except Exception:
        kpt_ratio, visible_kpts = 0.0, 0

    shape_says_lying = aspect > 1.10 or torso_horizontal or kpt_ratio > 1.15

    # ── 몸의 축 분석 (pose 키포인트 + z 캘리브의 '월드 수직' 방향) ──────
    #
    # 왜 다리인가:
    #   사람은 허리를 90도 굽혀도 다리는 수직으로 남는다. 반면 누우면 다리도
    #   눕는다. 몸통 각도(어깨→엉덩이)는 '숙임'과 '누움'을 구별하지 못한다.
    #
    #   합성 검증(키포인트 노이즈 포함) leg_angle 95% 상한:
    #       서있음 3.0  살짝숙임20 3.8  많이숙임45 2.9  허리굽힘70 3.3  웅크림 5.4
    #   누움 5% 하한 6.3~6.9, 중앙값 56~60.
    #   임계 20도에서 오탐 0%, 누움 검출 85~88%.
    #
    # 나머지 8~15% 미검출은 몸이 카메라 시선축과 정확히 정렬된 각도로,
    # 단일 카메라의 원리적 한계다. 5초 확진 창(약 125프레임) 동안 사람이
    # 그 각도를 유지할 확률은 낮다.
    leg_angle = None
    leg_min = None
    leg_segments = None
    torso_angle = None
    body_angle = None
    spread_ratio = None
    along_spread = None
    hip_above_ankle = None
    legs_upright = False
    axis_says_lying = False
    axis_available = False
    torso_len = None
    body_len = None
    angle_says_lying = False
    torso_foreshortened = False
    hip_inverted = False

    if body_axis is not None:
        leg_angle = body_axis.get("leg_angle_deg")
        leg_min = body_axis.get("leg_min_deg")
        leg_segments = body_axis.get("leg_segments_deg")
        torso_angle = body_axis.get("torso_angle_deg")
        body_angle = body_axis.get("body_angle_deg")
        spread_ratio = body_axis.get("spread_ratio")
        along_spread = body_axis.get("along_spread_m")
        hip_above_ankle = body_axis.get("hip_above_ankle_m")

        axis_available = torso_angle is not None

        # ── 몸통이 주 지표, 다리는 거부권 ────────────────────────────
        #
        # 예전에는 "다리도 눕고 몸통도 누워야 쓰러짐" (AND) 이었다.
        # 그런데 사람은 쓰러질 때 무릎을 굽힌다. 실측:
        #     segs = [59, 14, 18, 9]  TORSO = 47.2  -> 양 무릎을 다 세움
        # 다리 각도가 어떻게 나오든 몸통이 47도 기울었으면 쓰러진 것이다.
        #
        # 그래서 뒤집는다:
        #   몸통이 수평이면 일단 쓰러짐으로 본다.
        #   단, 다리가 '서 있음을 적극적으로 증명' 하면 거부한다.
        #
        # 다리가 서 있음을 증명한다 =
        #   * 한쪽 다리가 통째로 수직이고 (leg_min < leg_upright_deg)
        #   * 엉덩이가 발목보다 충분히 높다 (hip_above_ankle > hip_upright_m)
        #
        # 이 거부권이 잡아내는 것:
        #   허리굽힘 70도  : 몸통은 수평이지만 다리 2도, 엉덩이 0.90m -> 서 있음
        #   바닥앉기       : 다리는 수평이지만 몸통 16도 -> 애초에 몸통 조건 미달
        #   딥스쿼트/무릎꿇기/의자앉기 : 몸통 15~17도 -> 몸통 조건 미달
        # hip_above_ankle 은 거부권에 쓰지 않는다.
        # 그 값은 "발이 서 있는 수직선 위에 몸이 있다" 는 가정으로 계산되는데,
        # 누우면 그 가정이 깨져 헛값이 된다 (실측: 누운 사람인데 0.56m).
        # 다리 각도만으로 판단한다.
        legs_upright = bool(
            leg_min is not None
            and leg_min < leg_upright_deg
        )

        angle_says_lying = bool(
            axis_available
            and torso_angle > torso_angle_thres
            and not legs_upright
        )

        # ── 각도로 절대 못 잡는 사각지대: 카메라 축 방향 누움 ────────────
        #
        # 몸이 카메라를 정면으로 향해 누우면 몸통이 화면상 수직선과 나란해진다.
        # 그래서 torso_angle 이 서 있는 사람과 똑같이 나온다 (실측 13.3, 14.2도).
        # 이건 각도의 결함이 아니라 투영의 성질이다. 각도로는 원리적으로 불가능.
        #
        # 대신 '길이'를 본다. px_per_m 은 호모그래피에서 나온 값이라
        # 겉보기 픽셀 길이를 실제 m 로 되돌릴 수 있다.
        #
        # 두 가지 인체 불가능 조건:
        #   (a) 몸통이 0.34m 미만으로 보인다
        #       사람 몸통은 0.52m. 서든 앉든 쪼그리든 허리는 안 줄어든다.
        #       원근압축으로 누웠을 때만 0.25~0.33m 가 나온다.
        #       (머리가 카메라 쪽을 향해 누운 경우 — yaw 270~315)
        #
        #   (b) 엉덩이가 발목보다 0.35m 넘게 아래에 있다
        #       머리가 카메라 반대쪽을 향해 누우면 화면상 상하 순서가 뒤집힌다.
        #       서 있는 어떤 자세로도 엉덩이가 발목 아래로 갈 수 없다.
        #       (yaw 105~120)
        #
        # 게이트 — body_len (발목→머리 겉보기 길이, 실제 1.65m):
        #   딥스쿼트/바닥앉기는 몸이 '접혀서' 진짜로 짧아진다 (0.5~0.8m).
        #   압축과 헷갈리므로 반드시 제외해야 한다.
        #   누운 사람은 몸이 펴져 있어 압축돼도 0.9m 이상 유지된다.
        torso_len = body_axis.get("torso_len_m")
        body_len = body_axis.get("body_len_m")

        body_extended = bool(body_len is not None and body_len > body_len_min_m)

        torso_foreshortened = bool(
            body_extended
            and torso_len is not None and torso_len < torso_short_m
            and hip_above_ankle is not None and hip_above_ankle < hip_low_thres + 0.05
        )
        hip_inverted = bool(
            body_extended
            and hip_above_ankle is not None and hip_above_ankle < hip_invert_m
        )

        axis_says_lying = bool(angle_says_lying or torso_foreshortened or hip_inverted)

    # head_z 신뢰성 = 값의 범위 + 재투영 잔차
    #   -0.20m : 지면보다 살짝 아래도 허용 (측정 오차)
    #   user_height_m + 1.2m : 상한 (팔 뻗은 상태 등)
    range_ok = (
        head_z is not None
        and np.isfinite(head_z)
        and -0.20 <= head_z <= user_height_m + 1.2
    )
    residual_ok = (
        z_residual is not None
        and np.isfinite(z_residual)
        and float(z_residual) <= float(residual_thres)
    )
    valid_z = bool(range_ok and residual_ok)

    # 신뢰 못 하는 head_z 는 폴백에도 넘기지 않는다 (형상만으로 판정)
    trusted_head_z = head_z if valid_z else None

    old_lying, old_posture, debug = utils.judge_posture_v09(
        person_box,
        pose_item,
        head_z=trusted_head_z,
        z_residual=z_residual,
        image_h=image_h,
        user_height_m=user_height_m,
        pose_conf=pose_conf,
        lying_height_thres=lying_height_thres,
        very_low_height_thres=very_low_height_thres,
        residual_thres=residual_thres,
    )

    reasons = list(debug.get("reasons", []))
    posture = old_posture
    lying = old_lying

    def _finish(is_lying, post, extra_reason):
        debug["lying_mode"] = "leg_axis_v13"
        debug["reasons"] = reasons + [extra_reason]
        debug["head_z"] = None if head_z is None else float(head_z)
        debug["z_residual"] = None if z_residual is None else float(z_residual)
        debug["z_trusted"] = valid_z
        debug["residual_thres"] = float(residual_thres)
        debug["aspect"] = float(aspect)
        debug["torso_horizontal"] = bool(torso_horizontal)
        debug["kpt_ratio"] = float(kpt_ratio)
        debug["visible_kpts"] = int(visible_kpts)
        debug["shape_says_lying"] = bool(shape_says_lying)
        debug["leg_angle_deg"] = leg_angle
        debug["leg_min_deg"] = leg_min
        debug["legs_upright"] = bool(legs_upright)
        debug["leg_segments_deg"] = leg_segments
        debug["torso_angle_deg"] = torso_angle
        debug["body_angle_deg"] = body_angle
        debug["spread_ratio"] = spread_ratio
        debug["along_spread_m"] = along_spread
        debug["hip_above_ankle_m"] = hip_above_ankle
        debug["torso_len_m"] = torso_len
        debug["body_len_m"] = body_len
        debug["angle_says_lying"] = bool(angle_says_lying)
        debug["torso_foreshortened"] = bool(torso_foreshortened)
        debug["hip_inverted"] = bool(hip_inverted)
        debug["axis_says_lying"] = bool(axis_says_lying)
        debug["axis_available"] = bool(axis_available)
        return is_lying, post, debug

    # ── 다리 각도를 쓸 수 있으면 그것만으로 결론 ────────────────────────
    if axis_available:
        if angle_says_lying:
            return _finish(True, "LYING_CANDIDATE", "torso_horizontal")
        if torso_foreshortened:
            # 카메라 축 방향 누움 — 각도로는 서 있는 것과 구별 불가
            return _finish(True, "LYING_CANDIDATE", "torso_foreshortened_impossible")
        if hip_inverted:
            return _finish(True, "LYING_CANDIDATE", "hip_below_ankle_impossible")

        if hip_above_ankle is not None and hip_above_ankle < hip_low_thres:
            return _finish(False, "LOW_POSTURE", "low_posture_by_hip_height")

        return _finish(False, "NORMAL", "leg_axis_upright")

    # ── 다리가 안 보이면 (가림/화면 밖) head_z + 형상 폴백 ──────────────
    if valid_z:
        if head_z < very_low_height_thres:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_very_low")

        elif head_z < lying_height_thres and shape_says_lying:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_low_with_shape")

        elif head_z < user_height_m * 0.55 and shape_says_lying:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("z_priority_mid_low_with_shape")

        elif old_lying:
            lying = True
            posture = "LYING_CANDIDATE"
            reasons.append("base_old_lying_kept")

        else:
            if head_z < user_height_m * 0.65:
                posture = "LOW_POSTURE"
                lying = False
            else:
                posture = "NORMAL"
                lying = False

    else:
        # z 를 못 믿음 → 형상 기반 판정 결과를 그대로 사용
        lying = old_lying
        posture = old_posture

        if head_z is None:
            reasons.append("z_unavailable_fallback_shape")
        elif not range_ok:
            reasons.append("z_out_of_range_fallback_shape")
        else:
            reasons.append("z_residual_too_high_fallback_shape")

        # 몸이 서 있지는 않지만 눕지도 않았을 때 (z 를 못 믿는 상황)
        # 키포인트가 수직 방향으로 퍼진 길이로 웅크림을 잡는다.
        # 서있음 ~2.0m, 무릎꿇기 ~1.3m, 웅크림 ~1.0m
        if (not lying) and along_spread is not None and along_spread < 1.35:
            posture = "LOW_POSTURE"
            reasons.append("low_posture_by_along_spread")

    debug["lying_mode"] = "z_priority_v13"
    debug["reasons"] = reasons
    debug["head_z"] = None if head_z is None else float(head_z)
    debug["z_residual"] = None if z_residual is None else float(z_residual)
    debug["z_trusted"] = valid_z
    debug["residual_thres"] = float(residual_thres)
    debug["aspect"] = float(aspect)
    debug["torso_horizontal"] = bool(torso_horizontal)
    debug["kpt_ratio"] = float(kpt_ratio)
    debug["visible_kpts"] = int(visible_kpts)
    debug["shape_says_lying"] = bool(shape_says_lying)
    debug["torso_angle_deg"] = torso_angle
    debug["body_angle_deg"] = body_angle
    debug["spread_ratio"] = spread_ratio
    debug["along_spread_m"] = along_spread
    debug["axis_says_lying"] = False

    return lying, posture, debug


def apply_post_fusion_suppression(detections):
    """
    GlobalFuser 가 sticky 신분(unauthorized)을 되돌려 쓴 뒤에 호출한다.

    WorkerStateStore 는 카메라 단위라서, cam1 의 상태기계는 그 사람이
    cam0 입구에서 무단 침입했다는 사실을 모른다. 그래서 cam1 에서
    헬멧 미착용으로 보이면 helmet_alert 를 켜버린다.

    융합 후에는 det["unauthorized"] 가 사람 단위로 정확하므로,
    여기서 helmet_alert 를 한 번 더 정리한다.

      쓰러짐 / 긴급 / 무단침입  →  helmet_goal 금지 (AMR 오출동 방지)
    """
    for det in detections:
        if (
            det.get("emergency", False)
            or det.get("lying_candidate", False)
            or det.get("unauthorized", False)
        ):
            if det.get("helmet_alert", False):
                det["helmet_alert"] = False
                det["helmet_alert_elapsed"] = 0.0
            det["helmet_suppressed"] = True

    return detections


class WorkerStateStore:
    """
    카메라 1대의 트랙별 상태를 관리하는 저장소.

    입력: 매 프레임의 raw detection (helmet_status/posture/hand_raised 라벨)
    출력: smoothed 라벨 + emergency/helmet_alert/entry_state 3종 상태기계 결과

    핵심 메커니즘:
      - track_id별로 state dict을 self.states에 유지
      - 매 프레임 3가지 history deque(maxlen=6) 갱신 → 다수결 스무딩
      - hold_sec(0.8s) 동안은 detection이 끊겨도 이전 상태 유지
      - max_age_sec(3s) 지나면 track 제거 (사람이 완전히 사라진 경우)

    3개 상태기계 (독립 동작):
      Emergency    : smoothed_lying True 5초 지속 → emergency=True
                    smoothed_lying False 3초 지속 → emergency=False
      HelmetAlert  : NO_HELMET_* 3초 지속 → helmet_alert = True
                    HELMET_ON 2초 지속 → helmet_alert = False
      Entry (cam0) : ROI 안 + 손 듦 + HELMET_ON 3초 → ACCESS_GRANTED
                    ROI 벗어남 + 이전에 진입 시도 있었음 → UNAUTHORIZED
    """
    def __init__(
        self,
        camera_name,
        entry_roi_points,
        required_check_sec=3.0,
        emergency_sec=5.0,
        recovery_sec=3.0,
        helmet_alert_sec=3.0,
        helmet_recovery_sec=2.0,
        hold_sec=0.8,
        confirm_frames=2,
        max_age_sec=3.0,
        history_len=6,
        max_workers=4,
        entry_enabled=True,
    ):
        """
        Args:
            camera_name        : 'cam0' / 'cam1' (state_key prefix로 사용)
            entry_roi_points   : map 미터 좌표 폴리곤. cam0만 유의미.
            required_check_sec : ACCESS 판정 요구 지속 시간 [s]
            emergency_sec      : 쓰러짐 확진 시간 [s]
            recovery_sec       : 회복 확진 시간 [s]
            helmet_alert_sec   : 헬멧 미착용 확진 시간 [s]
            helmet_recovery_sec: 헬멧 착용 확진 시간 [s]
            hold_sec           : detection 끊긴 후 상태 유지 시간 [s]
            confirm_frames     : 트랙 확정에 필요한 프레임 수
            max_age_sec        : 트랙 완전 제거까지 대기 [s]
            history_len        : 히스토리 deque 크기 (다수결 스무딩)
            max_workers        : compact display id 최대치 (#0~#3)
            entry_enabled      : False면 entry ROI 판정 스킵 (cam1)
        """
        self.camera_name = camera_name
        self.entry_roi_points = entry_roi_points
        self.required_check_sec = required_check_sec
        self.emergency_sec = emergency_sec
        self.recovery_sec = recovery_sec
        self.helmet_alert_sec = helmet_alert_sec
        self.helmet_recovery_sec = helmet_recovery_sec
        self.hold_sec = hold_sec
        self.confirm_frames = confirm_frames
        self.max_age_sec = max_age_sec
        self.history_len = history_len
        self.max_workers = max_workers
        self.entry_enabled = entry_enabled
        self.states = {}

    def _new_state(self, key, now):
        """
        새 트랙에 대한 상태 dict 초기화.

        각 트랙마다 이 dict 하나를 self.states[key]에 유지.
        3개 상태기계의 시작/경과/결과 플래그가 여기 모두 담김.
        """
        return {
            "key": key,                    # "cam0:7" 형식
            "first_seen": now,
            "last_seen": now,

            "hit_count": 0,                 # confirm_frames 도달 카운트
            "track_confirmed": False,
            "last_det": None,               # hold_sec 유지용 이전 detection

            # ----- 스무딩용 히스토리 (다수결 판정 대상) -----
            "helmet_history": deque(maxlen=self.history_len),
            "posture_history": deque(maxlen=self.history_len),
            "lying_history": deque(maxlen=self.history_len),
            "hand_history": deque(maxlen=self.history_len),

            # ----- 상태기계 1: Emergency (쓰러짐) -----
            "fall_start": None,           # 쓰러짐 시작 시각
            "fall_elapsed": 0.0,           # 쓰러짐 경과 시간
            "emergency": False,            # 5s 넘으면 True

            "recovery_start": None,        # 회복 시작 시각
            "recovery_elapsed": 0.0,

            # ----- 상태기계 2: Helmet Alert -----
            "no_helmet_start": None,      # NO_HELMET 시작 시각
            "helmet_alert_elapsed": 0.0,
            "helmet_alert": False,         # 3s 넘으면 True

            "helmet_recover_start": None, # HELMET_ON 복귀 시작
            "helmet_recover_elapsed": 0.0,
            "helmet_suspicious": False,    # SUSPICIOUS 상태 표시용
            "helmet_suppressed": False,    # 쓰러짐/침입 중이라 경보 억제됨

            # ----- 상태기계 3: Entry ROI (cam0 전용) -----
            "entry_seen": False,           # 한 번이라도 ROI 안에 들어옴
            "entry_check_start": None,     # 검사 시작 시각
            "entry_progress": 0.0,
            "entry_state": "OUTSIDE",
            "access_granted": False,       # 3s 조건 충족
            "access_granted_time": None,
            "unauthorized": False,         # 진입 시도 후 ROI 벗어남
        }

    def _copy_det(self, det):
        copied = {}

        for k, v in det.items():
            try:
                if isinstance(v, np.ndarray):
                    copied[k] = v.copy()
                elif isinstance(v, list):
                    copied[k] = list(v)
                elif isinstance(v, dict):
                    copied[k] = dict(v)
                else:
                    copied[k] = v
            except Exception:
                copied[k] = v

        return copied

    def _remember_det(self, st, det):
        st["last_det"] = self._copy_det(det)

    def _majority(self, hist, default):
        """
        히스토리에서 가장 많이 나온 값 반환. helmet_status / posture 같은
        문자열 라벨 스무딩용 (다수결).
        """
        if not hist:
            return default

        counts = {}

        for x in hist:
            counts[x] = counts.get(x, 0) + 1

        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _smooth_bool(self, hist):
        """
        불리언 히스토리 스무딩. 최소 2번 이상 AND 과반 이상일 때 True.
        1회 반짝 True로 상태가 튀는 걸 방지.
        """
        if not hist:
            return False

        return sum(1 for x in hist if x) >= max(2, len(hist) // 2)

    def _is_no_helmet_state(self, helmet_status):
        return helmet_status in [
            "NO_HELMET",
            "NO_HELMET_RELATED",
            "NO_HELMET_MISSING",
        ]

    def _update_emergency_state(self, st, smooth_lying, now):
        """
        상태기계 1: Emergency.

        전이:
          smooth_lying=True :
              fall_start 없으면 지금 시각으로 시작
              fall_elapsed 5s 넘으면 emergency=True
          smooth_lying=False + emergency=False :
              모든 타이머 리셋 (평상시)
          smooth_lying=False + emergency=True :
              recovery_start부터 시간 재기 시작
              recovery_elapsed 3s 넘으면 emergency=False로 해제

        핵심: 회복(True→False)에도 3초 확인 시간을 두어서 사람이 몸을
              뒤척여 잠깐 서있는 것처럼 보여도 emergency를 유지함.
        """
        if smooth_lying:
            # 쓰러진 상태 지속 → 회복 타이머 리셋
            st["recovery_start"] = None
            st["recovery_elapsed"] = 0.0

            if st["fall_start"] is None:
                st["fall_start"] = now

            st["fall_elapsed"] = now - st["fall_start"]

            # 5초 지속 → emergency 확진
            if st["fall_elapsed"] >= self.emergency_sec:
                st["emergency"] = True

        else:
            if st["emergency"]:
                # emergency 중인데 지금은 서 있음 → 회복 관찰 중
                if st["recovery_start"] is None:
                    st["recovery_start"] = now

                st["recovery_elapsed"] = now - st["recovery_start"]

                # 3초 서 있음 → 회복 확진
                if st["recovery_elapsed"] >= self.recovery_sec:
                    st["emergency"] = False
                    st["fall_start"] = None
                    st["fall_elapsed"] = 0.0
                    st["recovery_start"] = None
                    st["recovery_elapsed"] = 0.0
            else:
                # 평상시 → 모든 타이머 리셋
                st["fall_start"] = None
                st["fall_elapsed"] = 0.0
                st["recovery_start"] = None
                st["recovery_elapsed"] = 0.0

    def _update_helmet_alert_state(self, st, smooth_helmet, now):
        """
        상태기계 2: Helmet Alert.

        전이:
          helmet_alert=False (평상시):
            NO_HELMET_* 3초 지속 → helmet_alert=True
            SUSPICIOUS → suspicious 플래그만 세팅
            HELMET_ON → 타이머 리셋

          helmet_alert=True (경보 중):
            HELMET_ON 2초 지속 → helmet_alert=False로 해제
            그 외 → 회복 타이머 리셋 (경보 유지)

        회복 시간(2s)이 진입 시간(3s)보다 짧은 이유:
          경보 후 착용은 "이미 인지한 상태에서의 응답"이라 빠르게 확인.

        SUSPICIOUS 상태는 alert를 직접 발동시키지 않고, 대시보드에
        "의심 상황" 표시만 하는 용도.
        """
        st["helmet_suspicious"] = smooth_helmet == "SUSPICIOUS"

        if st["helmet_alert"]:
            if smooth_helmet == "HELMET_ON":
                if st["helmet_recover_start"] is None:
                    st["helmet_recover_start"] = now

                st["helmet_recover_elapsed"] = now - st["helmet_recover_start"]

                if st["helmet_recover_elapsed"] >= self.helmet_recovery_sec:
                    st["helmet_alert"] = False
                    st["no_helmet_start"] = None
                    st["helmet_alert_elapsed"] = 0.0
                    st["helmet_recover_start"] = None
                    st["helmet_recover_elapsed"] = 0.0
            else:
                st["helmet_recover_start"] = None
                st["helmet_recover_elapsed"] = 0.0

            return

        if self._is_no_helmet_state(smooth_helmet):
            st["helmet_recover_start"] = None
            st["helmet_recover_elapsed"] = 0.0

            if st["no_helmet_start"] is None:
                st["no_helmet_start"] = now

            st["helmet_alert_elapsed"] = now - st["no_helmet_start"]

            if st["helmet_alert_elapsed"] >= self.helmet_alert_sec:
                st["helmet_alert"] = True

        else:
            st["no_helmet_start"] = None
            st["helmet_alert_elapsed"] = 0.0

            if smooth_helmet == "HELMET_ON":
                st["helmet_recover_start"] = None
                st["helmet_recover_elapsed"] = 0.0

    def _apply_helmet_suppression(self, st, smooth_lying, now):
        """쓰러짐/긴급/무단침입 중이면 helmet_alert 를 강제로 끄고 타이머를 리셋."""
        suppress = bool(smooth_lying or st["emergency"] or st["unauthorized"])

        st["helmet_suppressed"] = suppress

        if suppress:
            st["helmet_alert"] = False
            st["no_helmet_start"] = None
            st["helmet_alert_elapsed"] = 0.0
            st["helmet_recover_start"] = None
            st["helmet_recover_elapsed"] = 0.0

        return suppress

    def _attach_helmet_fields(self, det, st):
        det["helmet_alert"] = st["helmet_alert"]
        det["helmet_alert_elapsed"] = st["helmet_alert_elapsed"]
        det["helmet_recover_elapsed"] = st["helmet_recover_elapsed"]
        det["helmet_suspicious"] = st["helmet_suspicious"]
        det["helmet_suppressed"] = bool(st.get("helmet_suppressed", False))

    def update_one(self, det, now):
        """
        detection 하나를 상태기계에 통과시키고 결과 필드를 붙여 반환.

        절차:
          1) yolo_track_id 없으면 NO_ID 라벨로 폴백 (state key 안 만듦)
          2) state_key = "{camera}:{track_id}"로 조회/생성
          3) confirm_frames 도달하면 track_confirmed=True
          4) helmet/posture/lying/hand 스무딩
          5) 3개 상태기계 각자 갱신
          6) det에 결과 필드 attach (state_key, entry_state, emergency, ...)
          7) last_det에 이번 det의 복사본 저장 (hold_sec 유지용)
        """
        yolo_id = det.get("yolo_track_id")

        if yolo_id is None:
            det["state_key"] = f"{self.camera_name}:NO_ID"
            det["track_state"] = "NO_ID"
            det["track_confirmed"] = False
            det["is_held"] = False
            det["hold_elapsed"] = 0.0

            det["entry_state"] = "NO_ID"
            det["entry_progress"] = 0.0
            det["entry_progress_ratio"] = 0.0
            det["access_granted"] = False
            det["unauthorized"] = False

            det["fall_elapsed"] = 0.0
            det["recovery_elapsed"] = 0.0
            det["emergency"] = False

            det["helmet_alert"] = False
            det["helmet_alert_elapsed"] = 0.0
            det["helmet_recover_elapsed"] = 0.0
            det["helmet_suspicious"] = False

            return det

        key = f"{self.camera_name}:{int(yolo_id)}"

        if key not in self.states:
            self.states[key] = self._new_state(key, now)

        st = self.states[key]
        st["last_seen"] = now
        st["hit_count"] += 1

        if st["hit_count"] >= self.confirm_frames:
            st["track_confirmed"] = True

        raw_helmet = det.get("helmet_status", "UNKNOWN")
        raw_posture = det.get("posture", "NORMAL")
        raw_lying = bool(det.get("lying_candidate", False))
        raw_hand = bool(det.get("hand_raised", False))

        st["helmet_history"].append(raw_helmet)
        st["posture_history"].append(raw_posture)
        st["lying_history"].append(raw_lying)
        st["hand_history"].append(raw_hand)

        smooth_helmet = self._majority(st["helmet_history"], "UNKNOWN")
        smooth_posture = self._majority(st["posture_history"], "NORMAL")
        smooth_lying = self._smooth_bool(st["lying_history"])
        smooth_hand = self._smooth_bool(st["hand_history"])

        det["state_key"] = key
        det["track_confirmed"] = st["track_confirmed"]
        det["track_state"] = "CONFIRMED" if st["track_confirmed"] else "CANDIDATE"
        det["is_held"] = False
        det["hold_elapsed"] = 0.0

        det["helmet_status_raw"] = raw_helmet
        det["posture_raw"] = raw_posture
        det["hand_raised_raw"] = raw_hand

        det["helmet_status"] = smooth_helmet
        det["posture"] = smooth_posture
        det["lying_candidate"] = smooth_lying
        det["hand_raised"] = smooth_hand

        self._update_emergency_state(st, smooth_lying, now)
        self._update_helmet_alert_state(st, smooth_helmet, now)

        # ── helmet_alert 억제 ──────────────────────────────────────────
        # helmet_alert 는 "이 사람에게 헬멧을 배달하라"는 AMR 출동 신호다.
        # 다음 두 경우엔 절대 울리면 안 된다.
        #
        #   1) 쓰러진 사람 (lying / emergency)
        #      누우면 머리가 가려져 헬멧이 검출 안 되는 일이 잦다. 그걸
        #      '미착용'으로 읽고 배달 로봇을 보내면, 긴급 구조가 필요한
        #      사람 옆으로 엉뚱한 로봇이 움직인다. 긴급이 항상 우선.
        #
        #   2) 무단 침입자 (unauthorized)
        #      침입자는 헬멧을 안 줘야 한다. 그냥 침입자다.
        #      관리자 경보(unauthorized_state)로만 다뤄야 한다.
        self._apply_helmet_suppression(st, smooth_lying, now)

        if not self.entry_enabled:
            self._attach_helmet_fields(det, st)
            det["in_entry_roi"] = False
            det["entry_state"] = "MONITOR_ONLY"
            det["entry_progress"] = 0.0
            det["entry_progress_ratio"] = 0.0
            det["access_granted"] = False
            det["unauthorized"] = False

            det["fall_elapsed"] = st["fall_elapsed"]
            det["recovery_elapsed"] = st["recovery_elapsed"]
            det["emergency"] = st["emergency"]

            self._remember_det(st, det)
            return det

        # 상태기계 3: Entry ROI (cam0 전용)
        # 판정 조건: ROI 안 + 손 들었음 + 헬멧 착용 → 3초 지속되면 허가
        map_xy = det.get("map_xy")
        in_entry = point_in_polygon(map_xy, self.entry_roi_points)
        det["in_entry_roi"] = in_entry

        check_condition = in_entry and smooth_hand and smooth_helmet == "HELMET_ON"

        if in_entry:
            st["entry_seen"] = True

            if st["access_granted"]:
                st["entry_state"] = "ACCESS_GRANTED"

            elif check_condition:
                if st["entry_check_start"] is None:
                    st["entry_check_start"] = now

                st["entry_progress"] = now - st["entry_check_start"]

                if st["entry_progress"] >= self.required_check_sec:
                    st["access_granted"] = True
                    st["access_granted_time"] = now
                    st["unauthorized"] = False
                    st["entry_state"] = "ACCESS_GRANTED"
                else:
                    st["entry_state"] = "CHECKING"

            else:
                st["entry_check_start"] = None
                st["entry_progress"] = 0.0

                if smooth_helmet != "HELMET_ON" and smooth_hand:
                    st["entry_state"] = "CHECK_FAIL_NO_HELMET"
                elif smooth_helmet == "HELMET_ON" and not smooth_hand:
                    st["entry_state"] = "WAIT_HAND_RAISE"
                else:
                    st["entry_state"] = "WAIT_CHECK"

        else:
            st["entry_check_start"] = None
            st["entry_progress"] = 0.0

            if st["access_granted"]:
                st["entry_state"] = "ENTERED"
            elif st["entry_seen"]:
                st["unauthorized"] = True
                st["entry_state"] = "UNAUTHORIZED_ENTRY"
            else:
                st["entry_state"] = "OUTSIDE"

        # entry 상태기계가 unauthorized 를 세운 뒤 다시 한 번 억제 적용
        self._apply_helmet_suppression(st, smooth_lying, now)
        self._attach_helmet_fields(det, st)

        det["fall_elapsed"] = st["fall_elapsed"]
        det["recovery_elapsed"] = st["recovery_elapsed"]
        det["emergency"] = st["emergency"]

        det["entry_state"] = st["entry_state"]
        det["entry_progress"] = min(st["entry_progress"], self.required_check_sec)
        det["entry_progress_ratio"] = min(
            1.0,
            st["entry_progress"] / max(0.1, self.required_check_sec),
        )
        det["access_granted"] = bool(st["access_granted"])
        det["unauthorized"] = bool(st["unauthorized"])

        self._remember_det(st, det)
        return det

    def add_held_detections(self, detections, now, held_iou_suppress=0.35):
        """
        이번 프레임 detection에 없는 confirmed track을 hold_sec 동안 유지.

        YOLO track이 순간적으로 놓친 경우(occlusion, 낮은 conf)에도 대시보드
        표시가 깜빡이지 않도록, 마지막 detection의 복사본을 이번 프레임 결과
        리스트에 다시 끼워 넣음. is_held=True 플래그로 표시.

        ⚠ ByteTrack 이 같은 사람에게 새 track_id 를 주면 (id switch)
          옛 트랙은 '소실'로 보여 held 로 되살아나고, 새 트랙은 live 로 들어와
          한 사람이 두 개가 된다. 그러면 GlobalFuser 에서 held 가 먼저
          (gid, camera) 를 차지해 live 쪽에 새 global_id 가 생기고, 지도에
          점이 두 개 찍힌다 (최대 hold_sec 동안).

          그래서 live detection 과 bbox 가 크게 겹치는 held 는 되살리지 않는다.
          겹친다 = 같은 사람이 id 만 바뀐 것.

        hold_sec(0.8s) 경과하면 held를 그만 넣음. max_age_sec(3.0s) 지나면
        cleanup에서 state 자체가 삭제됨.
        """
        output = []
        current_keys = set()
        live_boxes = []

        for det in detections:
            key = det.get("state_key")

            if not det.get("track_confirmed", False):
                # 아직 confirm_frames 를 못 채운 새 트랙. 화면에 내보내지 않는다.
                # held 억제 판단에도 쓰지 않는다 — 안 그러면 id switch 직후
                # (새 트랙 미확정 + held 억제) 한 프레임 동안 사람이 사라진다.
                continue

            output.append(det)

            if key is not None:
                current_keys.add(key)

            box = det.get("person_box")
            if box is not None:
                live_boxes.append(box)

        for key, st in self.states.items():
            if key in current_keys:
                continue

            if not st.get("track_confirmed", False):
                continue

            last_det = st.get("last_det")
            if last_det is None:
                continue

            hold_elapsed = now - st["last_seen"]

            if not (0.0 < hold_elapsed <= self.hold_sec):
                continue

            # id switch 감지: 살아있는 사람과 겹치면 그건 유령이다
            hbox = last_det.get("person_box")
            if hbox is not None and live_boxes:
                if max(utils.box_iou(hbox, lb) for lb in live_boxes) > held_iou_suppress:
                    continue

            held = self._copy_det(last_det)
            held["is_held"] = True
            held["hold_elapsed"] = hold_elapsed
            held["track_state"] = "HOLD"
            held["track_confirmed"] = True

            output.append(held)

        return output

    def assign_compact_display_ids(self, detections):
        """
        YOLO track_id는 시간이 지나면 커지므로 (예: 137번),
        대시보드 표시용으로 첫 등장 순서에 따라 0~max_workers-1의 작은 번호를 부여.
        max_workers 초과 인원은 display_id=None (박스에 'X' 표시).
        """
        valid_dets = []

        for det in detections:
            key = det.get("state_key")

            if key is None or key.endswith("NO_ID"):
                det["display_id"] = None
                continue

            if key in self.states:
                valid_dets.append(det)

        valid_dets.sort(
            key=lambda d: self.states[d["state_key"]]["first_seen"]
        )

        for i, det in enumerate(valid_dets):
            if i < self.max_workers:
                det["display_id"] = i
            else:
                det["display_id"] = None

        return detections

    def cleanup(self, now):
        """
        max_age_sec 이상 안 보인 트랙 상태를 완전히 삭제.
        메모리 누적 방지 + 새로 등장한 사람이 같은 track_id를 재사용해도
        옛 상태가 잘못 이어붙지 않도록 함.
        """
        dead = []

        for key, st in self.states.items():
            if now - st["last_seen"] > self.max_age_sec:
                dead.append(key)

        for key in dead:
            del self.states[key]

    def reset(self):
        """'r' 키로 전체 상태 초기화. 테스트/데모 재시작용."""
        self.states = {}