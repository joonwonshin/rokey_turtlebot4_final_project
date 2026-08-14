#!/usr/bin/env python3
"""
Fleet FSM 관제 노드 (PC4 의 fleet_manager + mission_orchestrator + dashboard 역할).

AMR 구동 노드(patrol_fire.py)와 분리된 '판단 + 관제 표시' 전용 노드.
patrol_fire.py 는 로봇마다 독립 실행되는 standalone 스크립트로, 하드코딩된
waypoint 순찰과 배터리부족 자동 복귀를 전부 자체적으로 수행한다. 이 노드는
그 순찰 자체에는 관여하지 않고(명령 불필요), 오직 웹캠이 감지한 응급/안전모
이벤트가 들어왔을 때 어떤 로봇을 보낼지 "선정"만 하고, 선정된 로봇의 전용
토픽(f'/{robot_id}/emergency_goal')에 목표 좌표만 발행한다. patrol_fire.py
코드는 수정하지 않는다는 전제이므로, 이 노드는 patrol_fire.py 가 구독하기로
한 그 토픽 계약(스키마)만 지키면 된다.

각 로봇의 '구동 상태'(도킹/이동/대기)는 로봇의 자체 보고를 신뢰하지 않고,
로봇이 실제로 발행하는 표준 ROS 토픽을 이 노드가 직접 구독해서 판단한다
(day3/3_1_e_mail_delivery.py 가 TurtleBot4Navigator의 getDockedStatus()/
goToPose 처럼 로봇의 실제 동작 primitive 를 기준으로 삼는 것과 같은 방식).
patrol_fire.py 는 순찰/점검/배터리복귀 결과를 별도 이벤트로 보고하지 않으므로,
출동 완료 여부도 raw 토픽(Nav2 액션 상태의 navigating True->False 전이)만으로
추론한다.

    구동 상태 판단 근거 (raw 토픽, 표준 ROS 인터페이스만 사용):
      DOCKED     <- /<robot>/dock_status        (irobot_create_msgs/DockStatus.is_docked)
      NAVIGATING <- /<robot>/navigate_to_pose/_action/status (Nav2 액션 상태,
                     ACCEPTED/EXECUTING 이면 이동 중)
      IDLE       <- 위 둘 다 아닐 때
      위치       <- /<robot>/amcl_pose
      배터리     <- /<robot>/battery_state
      하트비트   <- 위 네 토픽 중 아무거나 수신되면 '살아있음'으로 간주
                     (OFFLINE_TIMEOUT 동안 전혀 없으면 OFFLINE)

================================ README ================================
1) 실행 전 준비
   - 이 노드는 patrol_fire.py 가 로봇마다 하나씩 떠서 자체 순찰 중이라고
     가정한다 (fleet_fsm 은 순찰을 명령하지 않는다).
   - ROBOTS 상수의 이름과 patrol_fire.py 실행 시 네임스페이스(__ns)가 반드시
     일치해야 한다. 그 네임스페이스로 dock_status/battery_state/amcl_pose/
     navigate_to_pose 액션이 발행되어야 이 노드가 로봇을 인식한다.
   - patrol_fire.py 가 f'/{robot_id}/emergency_goal' (std_msgs/String, JSON
     {"x","y","reason"}) 토픽을 구독하도록 연동되어 있어야 실제 출동 명령이
     전달된다 (현재 patrol_fire.py 에는 아직 미구현 - 통합 시 추가 필요).
   - MAP_PGM / MAP_YAML 경로를 실제 지도 파일로 맞춘다 (nav2 map_server 에 쓰는
     것과 동일한 .pgm + .yaml 쌍). 없으면 회색 빈 캔버스로 대체되어 계속 동작한다.
   - irobot_create_msgs 가 설치되어 있어야 한다 (TurtleBot4/Create3 표준 패키지,
     `/robotN/dock_status` 구독에 사용).

2) 실행
     colcon build --packages-select fp_amr_fsm --symlink-install
     source install/setup.bash
     ros2 run fp_amr_fsm fleet_fsm
   DISPLAY 가 없는 환경(SSH 등)이면 지도 창은 자동으로 비활성화되고 터미널
   로그만 출력된다 (5초 주기 상태 테이블 + 이벤트 즉시 로그).

3) 실습/시연용 이벤트 주입 (실제 웹캠 감지 노드가 아직 없을 때)
     # 위급상황(웹캠이 사람 위치를 map 좌표로 변환해 보냄) 발생시키기
     ros2 topic pub -1 /alert/emergency std_msgs/String \\
       '{data: "{\\"x\\": 1.2, \\"y\\": -0.5}"}'
     # 특정 로봇을 지정해서 출동시키고 싶으면 robot_id 를 함께 보낸다
     ros2 topic pub -1 /alert/emergency std_msgs/String \\
       '{data: "{\\"x\\": 1.2, \\"y\\": -0.5, \\"robot_id\\": \\"robot9\\"}"}'
     # 안전모 미착용 확인 이동 (평시 유지, IDLE 로봇만 배정)
     ros2 topic pub -1 /alert/helmet std_msgs/String \\
       '{data: "{\\"x\\": -1.0, \\"y\\": 2.0}"}'
   실제 연동 시에는 PC3 의 helmet_detector.py/fall_detector.py +
   homography_localizer.py 가 픽셀 좌표를 map 좌표로 변환해 이 두 토픽에
   발행하도록 연결하면 된다.

4) 모니터링 방법 (두 채널, 동시 동작)
   - 터미널: 로그에 5초마다 상태 테이블(로봇 상태 한글 표기 · 배터리 · 위치)이
     찍히고, 미션 배정/도착/점검/충전 등 이벤트는 발생 즉시 로그로 남는다.
   - 지도 창(OpenCV, "Fleet Monitor (map)"): map.pgm 위에 로봇을 상태별 색상
     원으로 표시하고, 상단 배너로 NORMAL(평시)/EMERGENCY(위급상황)를 크게
     표시하며, 출동/안전모 지점과 로봇-목표 연결선, 우측에 로봇별 상세 패널과
     색상 범례를 그린다. 창에서 'q' 키를 누르면 창만 닫히고 노드는 계속 동작.

5) 조정 포인트 (클래스 상단 상수)
   - ROBOTS: 관제 대상 로봇 네임스페이스 목록
   - MAP_PGM / MAP_YAML / VIZ_SCALE: 지도 파일 경로와 확대 배율
   - BATTERY_LOW / BATTERY_RESUME: 표시/선정 판단용 배터리 임계값(0.0~1.0)
     (patrol_fire.py 자체 임계값과는 독립적 - 표시/로봇 선정 배제용으로만 사용)

── 로봇별 상태머신 (robot2, robot9) ─────────────────────────────
    OFFLINE ─(raw 토픽 수신)→ IDLE
    IDLE ─(출동 배정)→ DISPATCHING ─(도착)→ DISPATCHING+현장대기
                     ─(*_clear 수신)→ IDLE
    배터리 < LOW ─→ (표시만) RETURNING_LOW_BATTERY → (dock_status.is_docked) → CHARGING
    CHARGING ─(배터리 ≥ RESUME)→ IDLE
    하트비트(raw 토픽) 끊김 ─→ OFFLINE
    순찰(PATROLLING) 은 로봇 노드가 자체적으로 수행하므로 이 FSM 은
    순찰 시작/종료를 표시하지 않는다. IDLE 상태에서 로봇이 실제로는 순찰
    중일 수 있으며, 이는 raw 구동 상태(NAVIGATING)로만 보인다.

    ★ 도착 != 상황 종료. 로봇 노드는 목표 지점에 도착한 뒤 *_clear 를
    받을 때까지 현장에서 대기하다가 그때 비로소 복귀한다 (응급·안전모 동일,
    고정 시간 자동 복귀 없음). 그래서 fleet 도 Nav2 액션의 navigating
    True→False 전이(=도착)로는 DISPATCHING 을 풀지 않고 '현장 대기'
    (dispatch_arrived)만 표시한다. IDLE 복귀와 situation 의 NORMAL 복귀는
    오직 /alert/emergency_clear · /alert/helmet_clear 수신 시에만 일어난다.
    (도착 즉시 IDLE 로 돌리면, 로봇이 아직 현장에 서 있는데 관제 화면만
    먼저 평시로 바뀌는 불일치가 생긴다.)
    단, goal 이 실패(ABORTED/CANCELED)로 끝난 경우는 현장 대기 대상이
    아니므로 즉시 IDLE 로 풀고 이벤트를 재큐잉한다.

── 상황(situation) 상태머신 ─────────────────────────────────────
    NORMAL(평시): 로봇들은 patrol_fire.py 로 자체 순찰 중 + 안전모 미착용
        지점 확인 이동(/alert/helmet)만 fleet_fsm 이 배정
    EMERGENCY(위급): 웹캠이 감지한 위치(/alert/emergency {x,y[,robot_id]})로
        지정/최적 로봇에게 즉시 이동 명령. 모든 위급 출동이 끝나면 NORMAL 복귀

── 관제 표시 ────────────────────────────────────────────────────
    1) 터미널: 주기적 상태 테이블 (로봇 상태 한국어 표기, 상황, 배터리, 위치)
    2) map.pgm 창(OpenCV): 지도 위 로봇 위치/상태 색상, 상황 배너,
       위급/안전모 지점 마커, 우측 상태 패널
       (한글은 OpenCV 폰트가 지원하지 않아 화면은 영문, 터미널은 한국어)

발행: /<robot>/emergency_goal (로봇별, JSON {x, y, reason})
구독: /alert/emergency, /alert/helmet,
      /<robot>/battery_state, /<robot>/dock_status, /<robot>/amcl_pose,
      /<robot>/navigate_to_pose/_action/status
==========================================================================
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from irobot_create_msgs.msg import DockStatus
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_profile_action_status_default, qos_profile_sensor_data)
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String

# 명령/상태 토픽용 QoS: RELIABLE + TRANSIENT_LOCAL(latched) depth 1.
# - RELIABLE: DDS 레벨 재전송으로 유실 방지 (best-effort 는 한 번 놓치면 끝)
# - TRANSIENT_LOCAL: publish 시점에 아직 discovery/매칭이 끝나지 않은
#   구독자(막 뜬 로봇 노드 등)에게도 마지막 메시지 1개를 보관했다가 전달.
#   "한 번만 보내면 가끔 수신 안 되는" 문제의 주원인이 이 discovery 경합이다.
# 주의: 양쪽(발행/구독) 모두 이 QoS 를 써야 매칭된다. 터미널 테스트 시:
#   ros2 topic pub -1 --qos-reliability reliable \
#     --qos-durability transient_local <topic> <type> <msg>
COMMAND_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


# 터미널 표기용 한국어 상태명
# NAVIGATING 은 raw 구동 상태로, patrol_fire.py 의 자체 순찰 이동도 포함한다
# (fleet_fsm 은 순찰을 명령하지 않으므로 별도 PATROLLING 상태를 두지 않는다).
KR_STATE = {
    'OFFLINE': '오프라인',
    'NO_LOCALIZATION': '측위 미실행',
    'NO_NAV2': 'Nav2 미실행',
    'IDLE': '대기',
    'CHARGING': '도킹(충전 중)',
    'DOCKED': '도킹',
    'NAVIGATING': '이동 중(순찰 포함)',
    'DISPATCHING_EMERGENCY': '위급 출동 중',
    'DISPATCHING_HELMET': '안전모 확인 이동 중',
    # 목표 지점에 도착했지만 아직 clear 를 못 받아 현장에서 대기 중.
    # 로봇은 clear 를 받아야만 복귀하므로 fleet 도 이 구간을 '출동 중'으로
    # 유지한다 (도착 즉시 IDLE 로 돌리면 상황 배너가 먼저 평시로 바뀐다).
    'ONSCENE_EMERGENCY': '현장 대기(조치 중)',
    'ONSCENE_HELMET': '현장 대기(안전모 확인)',
    'RETURNING_LOW_BATTERY': '배터리부족 복귀 중'
}

# 지도 표시용 상태 색상 (BGR)
STATE_COLOR = {
    'OFFLINE': (110, 110, 110),
    'NO_LOCALIZATION': (200, 100, 180),
    'NO_NAV2': (160, 60, 140),
    'IDLE': (230, 230, 230),
    'NAVIGATING': (200, 200, 80),
    'DISPATCHING_EMERGENCY': (60, 60, 255),
    'DISPATCHING_HELMET': (0, 165, 255),
    'ONSCENE_EMERGENCY': (120, 60, 255),
    'ONSCENE_HELMET': (60, 200, 255),
    'RETURNING_LOW_BATTERY': (60, 220, 220),
    'CHARGING': (255, 190, 80),
    'DOCKED': (255, 190, 80),
}


class RobotContext:
    """ 관제가 추적하는 로봇 1대의 상태 """

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.fsm_state = 'OFFLINE'     # 관제 FSM 상태 (이 노드가 판단/배정한 상태)
        # 로봇의 실제 "구동 상태" 는 로봇의 자체 보고를 신뢰하지 않고
        # 각 로봇이 실제로 발행하는 표준 ROS 토픽에서 직접 판단한다:
        #   docked      <- /<robot>/dock_status (irobot_create_msgs/DockStatus.is_docked)
        #   navigating  <- /<robot>/navigate_to_pose/_action/status (Nav2 액션 상태)
        # reported_state 는 이 둘을 조합한 'DOCKED' | 'NAVIGATING' | 'IDLE' 중 하나
        self.docked = False
        self.navigating = False
        self.reported_state = None
        self.pose = None               # [x, y] (amcl_pose)
        self.battery = None            # 0.0 ~ 1.0
        self.last_seen = None          # 위 raw 토픽 중 하나라도 수신한 마지막 시각
        self.seen = {}                 # 토픽 소스별 마지막 수신 시각 (스택 세분 진단용)
        # 스택 세분 진단 결과 (tick 에서 갱신):
        #   None              정상
        #   NO_LOCALIZATION   battery_state 는 오는데 amcl_pose 미수신
        #                     -> localization 노드가 안 떠 있음
        #   NO_NAV2           battery+amcl 은 정상인데 planner_server/
        #                     smoother_server 노드가 그래프에 없음 -> nav2 미실행
        self.stack_fault = None
        self.dispatch_goal = None      # 진행 중 출동 좌표 [x, y]
        self.dispatch_reason = None    # 'EMERGENCY' | 'HELMET'
        self.dispatch_item = None      # 배정 원본 이벤트 (실패 시 재큐잉용)
        self.charging = False          # 배터리부족 복귀~충전완료 구간 (출동 배정 제외)
        # 출동 완료 보고가 없으므로(patrol_fire.py 는 이벤트를 보내지 않음),
        # Nav2 액션 상태의 navigating True->False 전이로 도착을 추론한다.
        # 명령 도달 전의 우연한 False(예: 이전 순찰 goal 잔상)를 도착으로
        # 오판하지 않으려면 '출동 후 한 번이라도 navigating=True 를
        # 관측했는지'를 먼저 확인해야 한다.
        self.dispatch_seen_moving = False
        # 목표 지점에 도착했지만 아직 *_clear 를 못 받은 '현장 대기' 구간.
        # 로봇 노드는 도착해도 clear 를 받아야만 복귀하므로(응급/안전모 동일),
        # 도착을 곧 상황 종료로 보면 안 된다. 도착 시 DISPATCHING 을 유지한 채
        # 이 플래그만 세우고, IDLE 복귀는 *_clear 수신 시에만 한다.
        self.dispatch_arrived = False
        # 로봇 구동 노드(amr_patrol_emer_helmet.py)가 amr_status 토픽으로
        # 보고하는 자체 상태 (IDLE/PATROL/EMERGENCY/HELMET/RETURNING).
        # 순찰/안전모 배정 시 "정말 idle 인지" 판단하는 근거로 쓴다.
        self.amr_status = None


    def update_reported_state(self):
        """ raw 토픽(docked/navigating) 조합으로 '구동 상태' 갱신 """
        if self.docked:
            self.reported_state = 'DOCKED'
        elif self.navigating:
            self.reported_state = 'NAVIGATING'
        else:
            self.reported_state = 'IDLE'

    def display_state(self):
        """ 표시용 상태 키: 관제 FSM(무엇을 하라고 시켰는지) 우선,
        그 외에는 raw 토픽 기반 구동 상태(DOCKED/NAVIGATING/IDLE) 표시 """
        if self.fsm_state == 'OFFLINE':
            return 'OFFLINE'
        if self.stack_fault:
            return self.stack_fault
        if self.fsm_state == 'DISPATCHING':
            reason = self.dispatch_reason or 'EMERGENCY'
            prefix = 'ONSCENE' if self.dispatch_arrived else 'DISPATCHING'
            return f'{prefix}_{reason}'
        if self.fsm_state in ('RETURNING_LOW_BATTERY', 'CHARGING'):
            return self.fsm_state
        return self.reported_state or 'IDLE'


class FleetFSMNode(Node):

    # === 관제 대상 로봇 (id=2, 9). 로봇 구동 노드의 네임스페이스와 일치해야 함 ===
    # 바꿀 때 같이 고칠 것:
    #   - web/fleet_monitor.html 의 ROBOTS
    #   - amr_patrol_emer_helmet.py 의 DOCK_POSITIONS 키
    #   - start.sh / check_dup.sh 의 ROS_DISCOVERY_SERVER (서버 인덱스 = 로봇 번호)
    ROBOTS = ['robot2', 'robot9']

    # === 지도 파일 ===
    # 패키지 share/maps 에 설치된 지도를 우선 사용하고, 미설치 환경(소스
    # 직접 실행 등)을 위해 기존 rokey_ws 경로를 폴백으로 유지한다.
    try:
        from ament_index_python.packages import get_package_share_directory
        _MAP_DIR = get_package_share_directory('fp_amr_fsm') + '/maps'
    except Exception:
        _MAP_DIR = '/home/rokey/rokey_ws/maps'
    MAP_PGM = _MAP_DIR + '/final_project.pgm'
    MAP_YAML = _MAP_DIR + '/final_project.yaml'
    VIZ_SCALE = 6            # 지도 확대 배율 (128x118 -> 768x708)
    PANEL_W = 400            # 우측 상태 패널 폭(px)

    # === 배터리 기준 ===
    # BATTERY_LOW 는 로봇 구동 노드(amr_patrol_emer_helmet.py)의
    # BATTERY_LOW_THRESHOLD(0.30)와 반드시 일치해야 한다. 과거 0.25 로 두었을
    # 때 로봇은 30% 미만이면 자체 복귀하는데 fleet 은 25% 까지 정상으로 봐서,
    # 31% 로봇에게 순찰을 배정하면 29% 에서 곧바로 복귀하는 핑퐁이 발생했다.
    # 배터리부족 판정 임계값.
    # ★ 로봇 노드(amr_patrol_emer_helmet.py)의 BATTERY_LOW_THRESHOLD 와 반드시
    #   같아야 한다. 로봇은 이 값 미만이면 순찰을 중단하고 스스로 도크로
    #   복귀하는데, fleet 쪽 값이 더 낮으면 fleet 은 "아직 정상" 으로 보고
    #   순찰을 배정하고, 로봇은 받자마자 복귀하는 핑퐁이 생긴다.
    BATTERY_LOW = 0.10
    # 충전 완료로 보고 다시 임무에 투입하는 기준.
    # 0.90 이었는데, 30% 밑으로 한 번 떨어진 로봇은 90% 까지 충전되기 전에는
    # charging 플래그가 안 풀려서 사실상 '영구 제외' 상태가 됐다 (충전에만
    # 한참 걸린다). 60% 면 로봇 자체 복귀 임계값(30%)보다 충분히 높아서
    # 한 번 출동했다고 곧바로 재복귀하지 않는다.
    BATTERY_RESUME = 0.60
    # 출동/순찰 배정 가능 최소 배터리.
    #
    # 주의 - 이 값이 BATTERY_LOW(0.30) 보다 낮으면 사실상 무의미하다.
    # 배터리가 BATTERY_LOW 밑으로 떨어지는 순간 update_battery_state() 가
    # charging=True 로 만들고, available() 이 charging 로봇을 무조건 제외하기
    # 때문이다. 즉 30% 미만 로봇은 이 값과 무관하게 이미 배제된다.
    # (30% 미만 로봇도 출동시키고 싶다면 BATTERY_LOW 를 같이 내려야 하는데,
    #  그러면 로봇 노드의 BATTERY_LOW_THRESHOLD(0.30)와 어긋나 '보냈는데
    #  로봇이 곧장 복귀'하는 핑퐁이 생긴다. 두 값은 함께 움직여야 한다.)
    BATTERY_DISPATCH_MIN = 0.20

    # === 출동 접근 거리 [m]. goal 좌표는 '사람의 발 위치'(PC3 명세)이므로
    # 이만큼 물러난 접근점 + 사람을 바라보는 yaw 로 보정해 로봇에 보낸다. ===
    APPROACH_STANDOFF = 0.7

    # 이 시간 동안 로봇의 raw 토픽(battery/dock/amcl/nav)이 하나도 안 오면 OFFLINE.
    #
    # 5초였는데, 실측한 WiFi 가 RTT 평균 1000ms · 패킷 손실 33% 라
    # 1Hz 로 오는 battery_state 에 5초 넘는 구멍이 예사로 뚫린다. 그러면 멀쩡히
    # 순찰 중인 로봇이 OFFLINE 로 깜빡이고(실측: 하트비트 끊김 8회 / 재접속 8회),
    # 하필 그 순간 응급이 뜨면 후보에서 빠져 출동을 못 한다.
    # 진짜 죽은 로봇을 늦게 알아채는 대가보다, 살아있는 로봇을 배제하는 손해가 크다.
    OFFLINE_TIMEOUT = 15.0
    # 출동 실패(ABORTED 등) 시 재배정 상한.
    #
    # 1 = 재시도 없음 = "한 이벤트당 goal 은 딱 한 번만 발행".
    # 재시도를 켜면(2 이상) 같은 좌표로 emergency_goal 이 다시 나가는데,
    # 로봇 노드는 새 goal 을 받으면 진행 중이던 것을 취소하고 그리로 다시
    # 출발한다. 그래서 로봇이 복귀/순찰을 시작한 뒤에도 옛 응급 좌표로
    # 끌려나오는 일이 생긴다 (실측). 이동에 실패해도 로봇은 그 자리에서
    # clear 를 기다리므로, 재시도의 실익보다 이 부작용이 크다.
    MAX_DISPATCH_ATTEMPTS = 1
    TICK_PERIOD = 1.0
    TABLE_LOG_PERIOD = 5.0   # 터미널 상태 테이블 출력 주기(초)
    # /fleet/status 발행 주기(초). 판단 주기(TICK_PERIOD)와 분리한다.
    #
    # 이 토픽은 웹 대시보드가 로봇을 그리는 유일한 입력이다. tick() 에 묶어
    # 1Hz 로 내보내면 OpenCV 지도 창(gui_loop, 5Hz 로 메모리를 직접 읽음)은
    # 부드러운데 웹만 1초에 한 칸씩 끊겨 보인다 (실측으로 확인).
    # 판단은 1초마다면 충분하지만 표시는 그렇지 않으므로 별도 타이머로 뺀다.
    STATUS_PERIOD = 0.2      # 5Hz - OpenCV 창과 동일

    # === 자동 순찰 주기 (초). 이 주기마다 idle 로봇에게 순찰을 자동 배정한다.
    # /alert/patrol 수동 주입 없이도 평시 순찰이 반복되도록 하기 위한 설정.
    # 0 이하로 두면 자동 순찰 비활성 (기존처럼 수동 주입만 동작). ===
    PATROL_PERIOD = 300.0

    def __init__(self):
        super().__init__('fleet_fsm_node')

        self.lock = threading.Lock()
        self.robots = {rid: RobotContext(rid) for rid in self.ROBOTS}
        self.emergency_queue = []   # [{'x','y','robot_id'?}] 미배정 위급 이벤트
        self.helmet_queue = []      # [{'x','y'}] 미배정 안전모 확인 지점
        self.patrol_queue = []      # [{'robot_id'?}] 미배정 순찰 시작 명령
        self.situation = 'NORMAL'   # NORMAL(평시) | EMERGENCY(위급상황)
        # 마지막 출동 로봇 선정의 '계산 과정' 스냅샷. /fleet/status 로 발행해
        # 웹이 그대로 표시한다.
        #
        # 웹이 로봇 좌표를 받아 스스로 거리를 다시 계산하게 두면, fleet 이
        # 실제로 판단한 시점의 좌표와 웹이 보고 있는 좌표가 달라서 화면 숫자와
        # 실제 선정 결과가 어긋나 보인다. 판단 주체가 자기 계산을 그대로
        # 내보내는 것이 유일한 진실이다.
        self.last_selection = None
        self._last_table_log = 0.0
        # 노드 기동 시각 기준으로 첫 자동 순찰은 PATROL_PERIOD 후 배정
        # (기동 직후에는 로봇 discovery/상태 수신이 아직 안 끝나 있을 수 있음)
        self._last_auto_patrol = time.monotonic()

        self.load_map()

        # 로봇별 전용 출동 좌표 토픽. patrol_fire.py 가 이 토픽을 구독해
        # 새 goal 을 받으면 순찰을 중단하고 즉시 이동하도록 연동하는 것이
        # 통합 단계의 몫이며, 이 노드는 발행까지만 책임진다.
        self.emergency_goal_pubs = {
            rid: self.create_publisher(String, f'/{rid}/emergency_goal', COMMAND_QOS)
            for rid in self.ROBOTS
        }
        # 안전모 배달은 emergency_goal 이 아니라 로봇 노드가 구독하는
        # 전용 helmet_goal 토픽으로 보내야 idle 상태에서만 처리된다.
        self.helmet_goal_pubs = {
            rid: self.create_publisher(String, f'/{rid}/helmet_goal', COMMAND_QOS)
            for rid in self.ROBOTS
        }
        # 순찰 시작 명령 (robot_id 미지정 시 배터리 최고 idle 로봇 선정)
        self.patrol_cmd_pubs = {
            rid: self.create_publisher(String, f'/{rid}/patrol_cmd', COMMAND_QOS)
            for rid in self.ROBOTS
        }
        # 응급 조치 완료 신호. 로봇은 이 신호를 받아야 현장 대기를 끝낸다.
        self.emergency_clear_pubs = {
            rid: self.create_publisher(Bool, f'/{rid}/emergency_clear', COMMAND_QOS)
            for rid in self.ROBOTS
        }
        # 안전모 상황 해제 신호 (착용 확인). emergency_clear 와 동일 계약:
        # 로봇 노드는 안전모 지점에 도착한 뒤 이 신호를 받아야만 복귀한다.
        self.helmet_clear_pubs = {
            rid: self.create_publisher(Bool, f'/{rid}/helmet_clear', COMMAND_QOS)
            for rid in self.ROBOTS
        }
        self.status_pub = self.create_publisher(String, '/fleet/status', 10)

        self.create_subscription(String, '/alert/emergency', self.emergency_callback, 10)
        self.create_subscription(String, '/alert/helmet', self.helmet_callback, 10)
        # 디버깅/시연용 명령 주입 토픽:
        #   순찰 시작 (robot_id 생략 시 배터리 최고 idle 로봇):
        #     ros2 topic pub -1 /alert/patrol std_msgs/String '{data: "{}"}'
        #     ros2 topic pub -1 /alert/patrol std_msgs/String \
        #       '{data: "{\"robot_id\": \"robot9\"}"}'
        #   응급 조치 완료 (robot_id 생략 시 위급 출동 중인 모든 로봇에게):
        #     ros2 topic pub -1 /alert/emergency_clear std_msgs/String '{data: "{}"}'
        self.create_subscription(String, '/alert/patrol', self.patrol_callback, 10)
        self.create_subscription(
            String, '/alert/emergency_clear', self.emergency_clear_callback, 10)
        #   안전모 착용 확인(상황 해제): ros2 topic pub -1 /alert/helmet_clear \
        #     std_msgs/String '{data: "{}"}'  (safety_alert_bridge 가
        #     /safety/helmet_state 의 HELMET_CLEAR 를 이 토픽으로 중계한다)
        self.create_subscription(
            String, '/alert/helmet_clear', self.helmet_clear_callback, 10)
        #   큐 전체 초기화 (시나리오 리셋): 웹의 '큐 삭제' 버튼이 쓴다.
        #     ros2 topic pub -1 /alert/queue_clear std_msgs/String '{data: "{}"}'
        self.create_subscription(
            String, '/alert/queue_clear', self.queue_clear_callback, 10)
        for rid in self.ROBOTS:
            # 로봇의 '구동 상태'는 로봇의 자체 보고(String)를 거치지 않고
            # 각 로봇이 실제로 발행하는 표준 ROS 토픽에서 직접 구독해 판단한다.
            self.create_subscription(
                BatteryState, f'/{rid}/battery_state',
                lambda m, r=rid: self.battery_callback(r, m), qos_profile_sensor_data)
            self.create_subscription(
                DockStatus, f'/{rid}/dock_status',
                lambda m, r=rid: self.dock_status_callback(r, m), qos_profile_sensor_data)
            # amcl_pose 는 반드시 TRANSIENT_LOCAL(latched)로 구독해야 한다.
            #
            # Nav2 의 AMCL 은 amcl_pose 를 depth=1 / RELIABLE / TRANSIENT_LOCAL
            # 로 발행하고, '로봇이 움직여 파티클 필터가 갱신될 때만' 발행한다.
            # 도크에 가만히 서 있는 로봇은 아무것도 새로 안 내보낸다.
            #
            # 여기서 VOLATILE(= 그냥 10)로 구독하면 '구독 이후에 새로 오는 것'
            # 만 받으므로, 정지해 있는 로봇의 위치를 영원히 못 받는다.
            # -> pose=None -> NO_LOCALIZATION -> 출동 대상에서 제외 -> 아무
            #    로봇도 안 움직인다. (실측으로 확인한 버그)
            #
            # TRANSIENT_LOCAL 로 구독하면 latch 된 '마지막 위치'를 즉시 받는다.
            self.create_subscription(
                PoseWithCovarianceStamped, f'/{rid}/amcl_pose',
                lambda m, r=rid: self.amcl_callback(r, m), COMMAND_QOS)
            self.create_subscription(
                GoalStatusArray, f'/{rid}/navigate_to_pose/_action/status',
                lambda m, r=rid: self.nav_status_callback(r, m),
                qos_profile_action_status_default)
            # 로봇 구동 노드가 latched(TRANSIENT_LOCAL)로 발행하는 자체 상태
            self.create_subscription(
                String, f'/{rid}/amr_status',
                lambda m, r=rid: self.amr_status_callback(r, m), COMMAND_QOS)

        self.create_timer(self.TICK_PERIOD, self.tick)
        # 표시용 상태 발행은 판단 주기와 분리 (웹이 끊겨 보이지 않도록)
        self.create_timer(self.STATUS_PERIOD, self.publish_status)

        # 지도 표시 스레드 (DISPLAY 없으면 자동으로 터미널 전용으로 전환)
        self.gui_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        self.get_logger().info(
            f'Fleet FSM started. Robots: {self.ROBOTS}, situation=NORMAL(평시)')

    # ------------------------------------------------------------------
    # 지도 로드 / 좌표 변환
    # ------------------------------------------------------------------
    def load_map(self):
        self.map_img = None
        self.map_res = 0.05
        self.map_origin = (0.0, 0.0)
        if os.path.exists(self.MAP_PGM) and os.path.exists(self.MAP_YAML):
            img = cv2.imread(self.MAP_PGM, cv2.IMREAD_GRAYSCALE)
            with open(self.MAP_YAML, 'r') as f:
                meta = yaml.safe_load(f)
            if img is not None:
                self.map_img = img
                self.map_res = float(meta['resolution'])
                self.map_origin = (float(meta['origin'][0]), float(meta['origin'][1]))
                self.get_logger().info(
                    f'Map loaded: {self.MAP_PGM} {img.shape[1]}x{img.shape[0]}px, '
                    f'res={self.map_res}, origin={self.map_origin}')
                return
        self.get_logger().warn(
            f'Map not found ({self.MAP_PGM}). Using blank canvas.')
        self.map_img = np.full((240, 240), 205, dtype=np.uint8)
        self.map_origin = (-6.0, -6.0)

    def world_to_px(self, x, y):
        """ map 좌표(m) -> 확대된 지도 이미지 픽셀.
        지도 범위 밖 좌표는 가장자리에 클램프 (로봇이 화면에서 사라지지 않도록) """
        h, w = self.map_img.shape[:2]
        px = (x - self.map_origin[0]) / self.map_res
        py = h - 1 - (y - self.map_origin[1]) / self.map_res
        px = min(max(px, 0), w - 1)
        py = min(max(py, 0), h - 1)
        return int(px * self.VIZ_SCALE), int(py * self.VIZ_SCALE)

    # ------------------------------------------------------------------
    # 수신 콜백
    # ------------------------------------------------------------------
    def mark_alive(self, robot, source):
        """ raw 토픽 수신 = 로봇 스택이 살아있다는 증거 (커스텀 하트비트 불필요).
        모든 raw 콜백에서 호출. OFFLINE -> IDLE 전이도 여기서 처리.
        source 별 수신 시각은 tick 의 스택 세분 진단(어느 노드가 안 떠
        있는지 구분)에 쓴다. """
        robot.last_seen = time.monotonic()
        robot.seen[source] = robot.last_seen
        robot.update_reported_state()
        if robot.fsm_state == 'OFFLINE':
            robot.fsm_state = 'IDLE'
            self.get_logger().info(
                f'[{robot.robot_id}] ONLINE (구동 상태={robot.reported_state})')

    def battery_callback(self, rid, msg):
        with self.lock:
            robot = self.robots.get(rid)
            if robot is None:
                return
            if msg.percentage is not None:
                robot.battery = float(msg.percentage)
            self.mark_alive(robot, 'battery')

    def dock_status_callback(self, rid, msg):
        """ 도킹 여부는 로봇 자체 보고가 아니라 실제 센서 토픽(is_docked) 기준
        (충전 상태 전이 및 '구동 상태' 표시의 근거) """
        with self.lock:
            robot = self.robots.get(rid)
            if robot is None:
                return
            robot.docked = bool(msg.is_docked)
            self.mark_alive(robot, 'dock')

    def nav_status_callback(self, rid, msg):
        """ Nav2 NavigateToPose 액션 상태로 '이동 중' 여부 판단.
        어떤 노드가 goal 을 보냈는지와 무관하게 액션 서버가 직접 발행하는
        표준 토픽이므로, 로봇 노드의 자체 보고 없이도 신뢰할 수 있다.
        로봇 노드는 DISPATCH 완료를 이벤트로 알려주지 않으므로, 여기서
        navigating True->False 전이를 직접 감지해 출동 종료를 추론한다.

        ★ 이 추론은 '출동 goal 이 아직 진행 중일 때'만 유효하다. 도착한
        뒤(dispatch_arrived)에는 절대 보면 안 된다 - 로봇은 현장에서 clear 를
        기다리는 동안에도 다른 Nav2 활동(순찰 재개, 도킹 복귀 등)을 하고,
        그 goal 의 종료가 '우리 출동이 실패했다'로 오독되어 재배정 → 같은
        좌표로 goal 재발행 → 로봇이 집에 갔다가 다시 끌려나오는 루프가 된다
        (실측으로 확인한 버그). 도착 이후의 해제는 오직 *_clear 뿐이다. """
        with self.lock:
            robot = self.robots.get(rid)
            if robot is None:
                return
            was_navigating = robot.navigating
            latest = None
            if msg.status_list:
                # status_list 에는 이 액션 서버가 처리한 goal 들이 쌓여 있다.
                # 하나라도 활동 중이면 '이동 중'으로 본다.
                #
                # CANCELING 을 활동 중에 포함해야 한다. 이걸 빼면 '취소 진행
                # 중'이 곧 '이동 끝'으로 읽혀서, 새 goal 을 보내 이전 태스크가
                # 취소될 때마다 출동 실패로 오판하고 재배정한다 (goal 재발행
                # 루프의 두 번째 원인).
                # 진짜 종료는 SUCCEEDED / CANCELED / ABORTED 뿐이다.
                robot.navigating = any(
                    s.status in (GoalStatus.STATUS_ACCEPTED,
                                 GoalStatus.STATUS_EXECUTING,
                                 GoalStatus.STATUS_CANCELING)
                    for s in msg.status_list)
                latest = msg.status_list[-1].status
            self.mark_alive(robot, 'nav')

            if robot.fsm_state == 'DISPATCHING' and not robot.dispatch_arrived:
                if robot.navigating:
                    robot.dispatch_seen_moving = True
                elif was_navigating and robot.dispatch_seen_moving:
                    # True -> False 전이 = 출동 goal 종료. 종료 status 로
                    # 성공(도착)과 실패(접근 불가 좌표, 장애물로 인한 abort,
                    # 취소)를 구분한다.
                    goal, reason = robot.dispatch_goal, robot.dispatch_reason
                    item = robot.dispatch_item
                    robot.dispatch_seen_moving = False
                    if latest == GoalStatus.STATUS_SUCCEEDED:
                        # 도착 != 상황 종료. 로봇 노드는 목표 지점에 도착한 뒤
                        # *_clear 를 받을 때까지 현장에서 대기하고, 그때 비로소
                        # 복귀한다 (응급/안전모 동일). 그러므로 fleet 도 여기서
                        # IDLE 로 돌리면 안 된다 - 돌리면 tick() 의 상황 복귀
                        # 조건이 곧바로 참이 되어, 로봇이 아직 현장에 서 있는데
                        # 상황 배너와 지도 마커가 먼저 평시로 바뀐다.
                        # DISPATCHING 을 유지한 채 '현장 대기'만 표시하고,
                        # IDLE 복귀는 *_clear_callback 이 담당한다.
                        robot.dispatch_arrived = True
                        self.get_logger().info(
                            f'[{rid}] 현장 도착({reason}): {goal}. '
                            f'clear 신호까지 현장 대기.')
                    else:
                        # 실패한 goal 은 현장 대기 대상이 아니다. 즉시 놓아주고
                        # 이벤트를 재큐잉해 다른 로봇/다음 시도에 맡긴다.
                        self._release_dispatch(robot)
                        self._handle_dispatch_failure(rid, reason, goal, item, latest)

    def _release_dispatch(self, robot):
        """ 출동 종료(clear 수신 또는 goal 실패) -> 대기 상태로 되돌린다.
        도착만으로는 호출되지 않는다 (로봇이 현장에서 clear 를 기다리므로).
        주의: 호출자가 lock 을 잡은 상태에서 부른다. """
        robot.fsm_state = 'IDLE'
        robot.dispatch_goal = None
        robot.dispatch_reason = None
        robot.dispatch_item = None
        robot.dispatch_seen_moving = False
        robot.dispatch_arrived = False

    def _handle_dispatch_failure(self, rid, reason, goal, item, status):
        """ 출동 goal 이 SUCCEEDED 외의 종료 status(ABORTED=접근 불가/장애물,
        CANCELED 등)로 끝난 경우. 이벤트를 버리지 않고 큐 맨 앞에 되돌려
        다음 tick 에 재배정한다 (같은 좌표가 계속 실패하면 무한 루프가 되므로
        MAX_DISPATCH_ATTEMPTS 회 초과 시 포기하고 에러 로그만 남긴다).
        주의: nav_status_callback 의 lock 안에서 호출되므로 lock 을 잡지 않는다. """
        item = dict(item) if item else {'x': goal[0], 'y': goal[1]}
        item['attempts'] = item.get('attempts', 1) + 1
        # ponytail: 같은 로봇이 재선정될 수 있음(장애물이 치워지면 성공).
        # 좌표 자체가 접근 불가면 attempts 상한이 루프를 끊는다.
        if item['attempts'] <= self.MAX_DISPATCH_ATTEMPTS:
            queue = (self.emergency_queue if reason == 'EMERGENCY'
                     else self.helmet_queue)
            queue.insert(0, item)
            self.get_logger().error(
                f'[{rid}] 출동 실패({reason}, status={status}): {goal}. '
                f'재배정 예정 (시도 {item["attempts"]}/{self.MAX_DISPATCH_ATTEMPTS})')
        else:
            self.get_logger().error(
                f'[{rid}] 출동 실패({reason}, status={status}): {goal}. '
                f'{self.MAX_DISPATCH_ATTEMPTS}회 시도 후 포기 - 좌표 접근 불가로 판단.')

    def amcl_callback(self, rid, msg):
        p = msg.pose.pose.position
        with self.lock:
            robot = self.robots.get(rid)
            if robot is None:
                return
            robot.pose = [round(p.x, 3), round(p.y, 3)]
            self.mark_alive(robot, 'amcl')

    def emergency_callback(self, msg):
        """ 위급상황: 웹캠 감지 위치 {x, y[, robot_id]} -> 상황 EMERGENCY 전환 + 큐 """
        try:
            data = json.loads(msg.data)
            item = {'x': float(data['x']), 'y': float(data['y'])}
            if data.get('robot_id'):
                item['robot_id'] = data['robot_id']  # 특정 로봇 지정 출동
        except Exception as e:
            self.get_logger().warn(f'Invalid emergency message: {e}')
            return
        with self.lock:
            self.emergency_queue.append(item)
        self.set_situation('EMERGENCY',
                           f"웹캠 감지 위치 ({item['x']:.2f}, {item['y']:.2f})")

    def helmet_callback(self, msg):
        """ 평시 이벤트: 안전모 미착용 감지 위치 {x, y[, robot_id]} -> 확인 이동 큐.

        robot_id 는 emergency 와 동일하게 '이 로봇을 보내라' 는 지정이다.
        예전에는 여기서 robot_id 를 파싱조차 하지 않아서, 웹에서 robot2 를
        지정해도 자동 선정으로 빠져 robot9 이 나가는 일이 있었다. """
        try:
            data = json.loads(msg.data)
            item = {'x': float(data['x']), 'y': float(data['y'])}
            if data.get('robot_id'):
                item['robot_id'] = data['robot_id']
        except Exception as e:
            self.get_logger().warn(f'Invalid helmet message: {e}')
            return
        with self.lock:
            self.helmet_queue.append(item)
        self.get_logger().warn(
            f"안전모 미착용 감지: ({item['x']:.2f}, {item['y']:.2f}) - 확인 이동 배정 예정")

    def amr_status_callback(self, rid, msg):
        handover = False
        with self.lock:
            robot = self.robots.get(rid)
            if robot is None:
                return
            prev = robot.amr_status
            robot.amr_status = msg.data
            self.mark_alive(robot, 'amr')
            # ── 순찰 교대 감지 ──
            # PATROL → RETURNING 전이는 로봇 노드가 순찰 중 배터리 부족
            # (BATTERY_LOW_THRESHOLD 미만)으로 순찰을 중단하고 자체 복귀를
            # 시작했다는 뜻 (위급 선점이면 PATROL → EMERGENCY 로 가므로
            # 이 전이와 구분된다). 복귀하는 로봇 대신 다른 가용 로봇에게
            # 순찰을 이어받게 큐에 넣는다. 대체 로봇 선정은 assign_patrol/
            # select_patrol_robot 이 담당하며, 복귀 중인 이 로봇은
            # amr_status=RETURNING + 배터리 < BATTERY_DISPATCH_MIN 이라
            # 후보에서 자동 제외된다. 가용 로봇이 없으면 큐에 남아 있다가
            # 생기는 즉시 배정된다.
            if prev == 'PATROL' and msg.data == 'RETURNING':
                if self.situation == 'NORMAL' and not self.patrol_queue:
                    self.patrol_queue.append({})
                    handover = True
        if handover:
            batt = (f'{robot.battery*100:.0f}%'
                    if robot.battery is not None else '?')
            self.get_logger().warn(
                f'[{rid}] 순찰 중 배터리 부족 복귀 감지(배터리 {batt}) - '
                f'다른 가용 로봇에게 순찰 교대 배정 예정')

    def patrol_callback(self, msg):
        """ 순찰 시작 명령 주입: {} 또는 {robot_id}.
        robot_id 미지정이면 배터리가 가장 높은 idle 로봇을 선정해 보낸다. """
        item = {}
        if msg.data.strip():
            try:
                data = json.loads(msg.data)
                if data.get('robot_id'):
                    item['robot_id'] = data['robot_id']
            except Exception as e:
                self.get_logger().warn(f'Invalid patrol message: {e}')
                return
        with self.lock:
            self.patrol_queue.append(item)
        self.get_logger().info(
            f"순찰 시작 요청 수신 (robot_id={item.get('robot_id', '자동(배터리 최고)')})")

    def emergency_clear_callback(self, msg):
        """ 응급 조치 완료: {} 또는 {robot_id}. robot_id 미지정이면 위급 출동
        중(DISPATCHING/EMERGENCY)인 모든 로봇에게 clear 를 내려보낸다.

        아직 배정되지 않은 위급 큐도 함께 폐기한다 (helmet_clear 와 대칭).
        비우지 않으면 두 가지가 깨진다:
          1) 가용 로봇이 없는 동안 발생한 위급이 큐에 남아, tick() 의
             상황 복귀 조건(emergency_queue 가 비어야 함)이 영원히 거짓 →
             situation 이 EMERGENCY 에 갇히고 관제 화면의 빨간 마커도 안 사라짐.
          2) 사람이 일어선 뒤 로봇이 idle 이 되는 순간, 이미 해제된 옛 좌표로
             뒤늦게 출동한다.
        """
        target_id = None
        if msg.data.strip():
            try:
                data = json.loads(msg.data)
                target_id = data.get('robot_id')
            except Exception as e:
                self.get_logger().warn(f'Invalid emergency_clear message: {e}')
                return
        with self.lock:
            dropped = len(self.emergency_queue)
            self.emergency_queue.clear()
            if (self.last_selection or {}).get('reason') == 'EMERGENCY':
                self.last_selection = None   # 선정 근거 패널도 비운다
            targets = self._clear_targets(target_id, 'EMERGENCY')
            # 현장 대기(또는 이동) 중이던 위급 출동을 여기서 종료시킨다.
            # 도착만으로는 DISPATCHING 이 풀리지 않기 때문에, clear 가
            # 유일한 해제 지점이다.
            for rid in targets:
                robot = self.robots.get(rid)
                if (robot is not None and robot.fsm_state == 'DISPATCHING'
                        and robot.dispatch_reason == 'EMERGENCY'):
                    self._release_dispatch(robot)
        clear = Bool()
        clear.data = True
        for rid in targets:
            self.emergency_clear_pubs[rid].publish(clear)
            self.get_logger().info(f'[{rid}] 응급 조치 완료(emergency_clear) 발행')
        if not targets:
            self.get_logger().info(
                '응급 조치 완료 - 위급 대응 중인 로봇이 없어 clear 발행 생략')
        if dropped:
            self.get_logger().info(
                f'응급 조치 완료 - 미배정 emergency 이벤트 {dropped}건 폐기')

    def _clear_targets(self, target_id, reason):
        """ *_clear 를 실제로 내려보낼 로봇을 고른다 (lock 안에서 호출).

        예전에는 전 로봇에게 브로드캐스트했다. 상황과 무관한 로봇까지 clear 를
        받는 게 지저분하고, 응급이 2건이라 두 로봇이 각각 출동 중일 때는
        한쪽의 clear 가 다른 쪽 출동까지 취소시키는 실제 버그였다.

        그렇다고 "fleet 이 DISPATCHING 으로 아는 로봇" 만 고르면, goal 이
        ABORT 되어 fleet 은 IDLE 로 되돌렸지만 로봇은 그 자리에서 clear 를
        기다리는 경우(amr_patrol 은 이동 실패해도 현장 대기한다) 그 로봇이
        영영 안 돌아온다. 그래서 로봇의 자체 보고(amr_status)도 함께 본다.

          fleet 의 판단  : fsm_state == DISPATCHING and dispatch_reason == reason
          로봇의 자체 보고: amr_status == reason  ('EMERGENCY' | 'HELMET')

        둘 중 하나라도 해당하면 대상. 아무도 없으면 빈 리스트(발행 생략).
        """
        if target_id:
            return [target_id] if target_id in self.robots else []
        return [r.robot_id for r in self.robots.values()
                if (r.fsm_state == 'DISPATCHING' and r.dispatch_reason == reason)
                or r.amr_status == reason]

    def helmet_clear_callback(self, msg):
        """ 안전모 착용 확인(상황 해제): {} 또는 {robot_id}.
        emergency_clear 와 동일 패턴 + 아직 배정 전인 helmet 큐도 함께
        폐기한다 (착용이 확인된 지점으로 뒤늦게 확인 이동할 이유가 없음). """
        target_id = None
        if msg.data.strip():
            try:
                data = json.loads(msg.data)
                target_id = data.get('robot_id')
            except Exception as e:
                self.get_logger().warn(f'Invalid helmet_clear message: {e}')
                return
        with self.lock:
            dropped = len(self.helmet_queue)
            self.helmet_queue.clear()
            if (self.last_selection or {}).get('reason') == 'HELMET':
                self.last_selection = None
            targets = self._clear_targets(target_id, 'HELMET')
            # 안전모도 응급과 같은 계약이다: 로봇은 도착 후 helmet_clear 를
            # 받아야만 복귀한다 (고정 시간 자동 복귀 없음). 따라서 현장 대기
            # 중인 HELMET 출동을 여기서 종료시킨다.
            for rid in targets:
                robot = self.robots.get(rid)
                if (robot is not None and robot.fsm_state == 'DISPATCHING'
                        and robot.dispatch_reason == 'HELMET'):
                    self._release_dispatch(robot)
        if dropped:
            self.get_logger().info(
                f'안전모 착용 확인 - 미배정 helmet 이벤트 {dropped}건 폐기')
        clear = Bool()
        clear.data = True
        for rid in targets:
            self.helmet_clear_pubs[rid].publish(clear)
            self.get_logger().info(f'[{rid}] 안전모 해제(helmet_clear) 발행')
        if not targets:
            self.get_logger().info(
                '안전모 착용 확인 - 안전모 대응 중인 로봇이 없어 clear 발행 생략')

    def queue_clear_callback(self, msg):
        """ 큐 전체 초기화 (시나리오 리셋용, 웹의 '큐 삭제' 버튼).

        관제탑의 세 큐를 비우는 것만으로는 부족하다. 이미 로봇에게 배달된
        goal 은 로봇의 pending_* 에 남아 있어서, 나중에 그리로 출동해버린다.
        그래서 clear 신호(Bool)를 전 로봇에 내려보내 로봇 쪽 pending 까지
        지우게 한다 (로봇의 *_clear_callback 이 pending goal 을 폐기한다).

        여기서는 의도적으로 '전 로봇' 브로드캐스트다 - 리셋이니까.
        """
        with self.lock:
            n = (len(self.emergency_queue) + len(self.helmet_queue)
                 + len(self.patrol_queue))
            self.emergency_queue.clear()
            self.helmet_queue.clear()
            self.patrol_queue.clear()
            self.last_selection = None
            released = [r.robot_id for r in self.robots.values()
                        if r.fsm_state == 'DISPATCHING']
            for r in self.robots.values():
                if r.fsm_state == 'DISPATCHING':
                    self._release_dispatch(r)
            all_ids = list(self.robots.keys())

        clear = Bool()
        clear.data = True
        for rid in all_ids:
            self.emergency_clear_pubs[rid].publish(clear)
            self.helmet_clear_pubs[rid].publish(clear)
        self.get_logger().warn(
            f'●●● 큐 초기화: 미배정 이벤트 {n}건 폐기, '
            f'출동 해제 {released or "없음"}, 전 로봇에 clear 발행')
        self.set_situation('NORMAL', '(큐 초기화)')

    def set_situation(self, situation, reason=''):
        with self.lock:
            if self.situation == situation:
                return
            self.situation = situation
        kr = '위급상황' if situation == 'EMERGENCY' else '평시'
        msg = f'●●● 상황 전환: {situation}({kr}) {reason}'
        
        # 아예 error() 하나로 통일 (위급상황의 심각도가 높으니 ERROR로 항상 출력하는 것도 가능)
        if situation == 'EMERGENCY':
            self.get_logger().error(msg)
        else:
            self.get_logger().info(msg)

    # ------------------------------------------------------------------
    # FSM 판단 주기
    # ------------------------------------------------------------------
    def tick(self):
        now = time.monotonic() # 일정하게 흐름. 신뢰 가능 <-> 리얼타임
        with self.lock:
            # 1) 하트비트 기반 OFFLINE 판정
            for robot in self.robots.values():
                if (robot.fsm_state != 'OFFLINE' and robot.last_seen is not None
                        and now - robot.last_seen > self.OFFLINE_TIMEOUT):
                    robot.fsm_state = 'OFFLINE'
                    self.get_logger().error(
                        f'[{robot.robot_id}] 하트비트 끊김 -> OFFLINE')

            # 2) 순찰 시작/종료는 patrol_fire.py 가 자체 처리하므로 이 FSM 은
            # 관여하지 않는다. 여기서는 하트비트 단절(OFFLINE)만 감지한다.

        # 2.5) 로봇 스택 세분 진단: 어떤 토픽/노드가 살아있는지 조합해
        # "무엇이 안 떠 있는지"를 구분한다 (출동 선정에서도 제외).
        #   battery O + amcl X                       -> NO_LOCALIZATION
        #   battery O + amcl O + planner/smoother X  -> NO_NAV2
        self.diagnose_stacks(now)

        # 3) 배터리 상태 갱신 (표시/선정 배제용. 복귀 명령은 내리지 않음 -
        # patrol_fire.py 가 스스로 배터리 부족을 감지해 복귀·도킹한다)
        self.update_battery_state()

        # 4) 위급 출동 배정 (최우선, 안전모 확인 이동 중인 로봇 선점 허용)
        self.assign_dispatch(self.emergency_queue, 'EMERGENCY', preempt_helmet=True)

        # 5) 평시: 안전모 확인 이동 배정 (IDLE 로봇만)
        self.assign_dispatch(self.helmet_queue, 'HELMET', preempt_helmet=False)

        # 5.3) 자동 순찰 주입: PATROL_PERIOD 마다 순찰 큐에 자동으로 이벤트를
        # 넣는다 (기존에는 /alert/patrol 수동 주입이 있어야만 순찰 시작).
        self.auto_patrol(now)

        # 5.5) 순찰 시작 배정 (robot_id 미지정 시 배터리 최고 idle 로봇)
        self.assign_patrol()

        # 6) 상황 복귀 판정: 위급 큐 비었고 위급 출동 중 로봇 없음 -> NORMAL
        with self.lock:
            emergency_active = bool(self.emergency_queue) or any(
                r.fsm_state == 'DISPATCHING' and r.dispatch_reason == 'EMERGENCY'
                for r in self.robots.values())
        if not emergency_active:
            self.set_situation('NORMAL', '(위급 대응 완료)')

        # 7) 터미널 상태 테이블 (주기 출력)
        if now - self._last_table_log >= self.TABLE_LOG_PERIOD:
            self._last_table_log = now
            self.log_status_table()
        # publish_status 는 STATUS_PERIOD 전용 타이머가 담당한다 (여기서 또
        # 부르면 tick 주기마다 중복 발행됨).

    def diagnose_stacks(self, now):
        """ 로봇별 스택 세분 진단.
        - battery_state 는 Create3 베이스가 항상 주기 발행하므로 '베이스 생존'
          신호로 쓴다 (OFFLINE_TIMEOUT 내 수신 = 신선).
        - amcl_pose 는 로봇이 움직여 필터가 갱신될 때만 발행되므로 신선도가
          아니라 '한 번이라도 수신했는가'로 판단한다 (정지 로봇 오진 방지).
        - planner_server/smoother_server 는 토픽이 없으므로 ROS 노드 그래프
          (get_node_names_and_namespaces)에서 해당 네임스페이스 노드 존재로
          확인한다. """
        running = set(self.get_node_names_and_namespaces())
        with self.lock:
            for robot in self.robots.values():
                fault = None
                if robot.fsm_state != 'OFFLINE':
                    batt_fresh = (robot.seen.get('battery') is not None
                                  and now - robot.seen['battery'] <= self.OFFLINE_TIMEOUT)
                    amcl_seen = 'amcl' in robot.seen
                    ns = f'/{robot.robot_id}'
                    if batt_fresh and not amcl_seen:
                        fault = 'NO_LOCALIZATION'
                    elif batt_fresh and amcl_seen and not (
                            ('planner_server', ns) in running
                            and ('smoother_server', ns) in running):
                        fault = 'NO_NAV2'
                if fault != robot.stack_fault:
                    if fault:
                        detail = ('localization(amcl) 노드 미실행 추정: '
                                  'battery_state 는 수신되나 amcl_pose 미수신'
                                  if fault == 'NO_LOCALIZATION' else
                                  'nav2 미실행 추정: planner_server/smoother_server '
                                  '노드가 그래프에 없음')
                        self.get_logger().error(f'[{robot.robot_id}] {fault} - {detail}')
                    else:
                        self.get_logger().info(
                            f'[{robot.robot_id}] 스택 진단 정상 복귀')
                    robot.stack_fault = fault

    def update_battery_state(self):
        """ 배터리 임계값에 따라 표시 상태만 갱신한다. patrol_fire.py 가
        복귀/도킹을 자체 수행하므로 이 노드는 명령을 내리지 않고, 그 결과를
        raw 토픽(dock_status/battery_state)으로 관찰해 표시에만 반영한다. """
        for robot in self.robots.values():
            with self.lock:
                b = robot.battery
                state = robot.fsm_state
                docked = robot.docked
                charging = robot.charging
            if b is None or state == 'OFFLINE':
                continue

            with self.lock:
                if (not charging and b < self.BATTERY_LOW
                        and state not in ('RETURNING_LOW_BATTERY', 'CHARGING', 'DISPATCHING')):
                    robot.fsm_state = 'RETURNING_LOW_BATTERY'
                    robot.charging = True
                    self.get_logger().warn(
                        f'[{robot.robot_id}] 배터리 {b*100:.0f}% < '
                        f'{self.BATTERY_LOW*100:.0f}% - 자체 복귀 예상 (표시만 갱신)')
                    continue

                # 도킹 완료 -> 충전 중.
                # IDLE 도 포함해야 한다: 위급 출동이 충전 중이던 로봇을 끌어내면
                # (응급은 배터리를 무시한다) 조치 완료 후 fsm_state 는 IDLE 로
                # 돌아오는데, charging 플래그는 True 로 남는다. 그 로봇이 도크로
                # 복귀해도 RETURNING_LOW_BATTERY 가 아니므로 CHARGING 으로 못
                # 넘어가고, 결국 배터리가 차도 charging 이 안 풀려 안전모/순찰에서
                # 영원히 배제된다.
                if robot.charging and docked and robot.fsm_state in (
                        'RETURNING_LOW_BATTERY', 'IDLE'):
                    robot.fsm_state = 'CHARGING'
                # 충전 완료 -> 대기(순찰은 로봇이 자체 재개)
                if robot.charging and robot.fsm_state == 'CHARGING' \
                        and b >= self.BATTERY_RESUME:
                    robot.charging = False
                    robot.fsm_state = 'IDLE'
                    self.get_logger().info(
                        f'[{robot.robot_id}] 충전 완료({b*100:.0f}%). 대기 상태로 전환.')

    def publish_status(self):
        """ render_frame() 이 OpenCV 로 그리던 상태를 JSON 토픽으로도 발행.
        lock 안에서는 dict 구성(복사)만 하고, publish 는 lock 밖에서 수행 """
        now = time.monotonic()
        with self.lock:
            payload = {
                'situation': self.situation,
                'emergency_queue': list(self.emergency_queue),
                'helmet_queue': list(self.helmet_queue),
                'patrol_queue': list(self.patrol_queue),
                # 출동 로봇 선정 계산 (웹이 그대로 표시 - 재계산하지 않는다)
                'selection': self.last_selection,
                'robots': {
                    r.robot_id: {
                        'fsm_state': r.fsm_state,
                        'display_state': r.display_state(),
                        'reported_state': r.reported_state,
                        'stack_fault': r.stack_fault,
                        'amr_status': r.amr_status,
                        'docked': r.docked,
                        'navigating': r.navigating,
                        'dispatch_seen_moving': r.dispatch_seen_moving,
                        'battery': r.battery,
                        'pose': r.pose,
                        'dispatch_goal': r.dispatch_goal,
                        'dispatch_reason': r.dispatch_reason,
                        'dispatch_arrived': r.dispatch_arrived,
                        'last_seen_age': (now - r.last_seen) if r.last_seen else None,
                    } for r in self.robots.values()
                },
            }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def assign_dispatch(self, queue, reason, preempt_helmet):
        with self.lock:
            item = queue.pop(0) if queue else None
        if item is None:
            return
        robot = self.select_dispatch_robot(item, reason, preempt_helmet)
        # 선정 과정을 그대로 기록해 /fleet/status 로 내보낸다 (웹이 표시).
        self.record_selection(item, reason, robot)
        if robot is None:
            # 로봇을 명시했는데 그 로봇이 못 가는 경우(응급 제외)는 큐에 되돌리지
            # 않는다. 되돌리면 매 tick 재시도하며 경고가 도배되고, 몇 분 뒤 그
            # 로봇이 idle 이 되는 순간 이미 잊혀진 요청이 뒤늦게 실행된다.
            # 사용자가 지정한 수동 요청이므로 한 번 알리고 폐기하는 편이 예측 가능하다.
            if item.get('robot_id') and reason != 'EMERGENCY':
                self.get_logger().warn(
                    f'{reason} 요청 폐기 - 지정 로봇 {item["robot_id"]} 사용 불가.')
                return
            with self.lock:
                queue.insert(0, item)  # 가용 로봇 생길 때까지 재시도
            return
        self.publish_goal(robot, item['x'], item['y'], reason)
        with self.lock:
            robot.fsm_state = 'DISPATCHING'
            robot.dispatch_goal = [item['x'], item['y']]
            robot.dispatch_reason = reason
            robot.dispatch_item = item  # 실패 시 재큐잉용 원본 보관
            robot.dispatch_seen_moving = False

    # 응급 출동은 배터리를 보지 않는다.
    #
    # 사람이 쓰러져 있는데 "배터리가 낮아서 안 갑니다" 는 말이 안 된다.
    # 충전 중이거나 배터리부족 복귀 중인 로봇이라도 끌어내서 보낸다 -
    # 충전은 조치가 끝난 뒤 다시 하면 되지만, 응급은 다시 할 수 없다.
    # 로봇 노드도 같은 정책이다 (run_emergency_dispatch 는 배터리를 안 본다.
    # 순찰만 BATTERY_LOW_THRESHOLD 로 중단한다). 즉 지금까지는 fleet 쪽에서만
    # 막고 있었다.
    #
    # 반대로 안전모/순찰은 급하지 않으므로 배터리 게이트를 유지한다.
    # 충전 중인 로봇을 굳이 끌어낼 이유가 없다.
    BATTERY_OVERRIDE_REASONS = ('EMERGENCY',)

    # 응급 시 끌어낼 수 있는 '충전 계열' 상태들
    CHARGING_STATES = ('CHARGING', 'RETURNING_LOW_BATTERY')

    def exclusion_reason(self, r, reason):
        """ 이 로봇이 출동 후보에서 빠지는 이유. 후보면 None.
        select_dispatch_robot 의 필터와 같은 순서로 검사한다
        (여기가 어긋나면 화면 설명과 실제 판단이 달라지므로 함께 고칠 것). """
        override = reason in self.BATTERY_OVERRIDE_REASONS

        # 물리적으로 갈 수 없는 경우 - 응급이라도 어쩔 수 없다
        if r.fsm_state == 'OFFLINE':
            return '오프라인'
        if r.stack_fault == 'NO_NAV2':
            return 'Nav2 미실행'
        if r.stack_fault == 'NO_LOCALIZATION':
            return '측위 미실행'
        if r.stack_fault:
            return r.stack_fault
        if r.pose is None:
            return '위치 미수신'
        if r.fsm_state == 'DISPATCHING':
            return f'이미 출동 중({r.dispatch_reason})'

        # 배터리 관련 - 응급이면 무시한다
        if not override:
            if r.charging:
                return '배터리부족 복귀/충전 중'
            if r.battery is not None and r.battery < self.BATTERY_DISPATCH_MIN:
                return (f'배터리 {r.battery*100:.0f}% < '
                        f'{self.BATTERY_DISPATCH_MIN*100:.0f}%')
        if r.fsm_state != 'IDLE' and not (
                override and r.fsm_state in self.CHARGING_STATES):
            return r.fsm_state

        if reason == 'HELMET' and r.amr_status not in (
                None, 'IDLE', 'PATROL', 'RETURNING'):
            return f'amr_status={r.amr_status}'
        # 순찰은 '지금 아무것도 안 하는' 로봇에게만 시작시킨다
        # (이미 순찰 중인 로봇에게 또 시키면 처음부터 다시 돈다).
        if reason == 'PATROL' and r.amr_status not in (None, 'IDLE'):
            return f'amr_status={r.amr_status}'
        return None

    def record_selection(self, item, reason, chosen):
        """ 선정 계산을 스냅샷으로 남긴다 (웹의 '출동 로봇 선정 근거' 패널용).

        거리는 직선거리(유클리드)다. Nav2 경로 길이가 아니라는 점을 명시한다 -
        벽을 사이에 두면 직선이 가까워도 실제 주행거리는 더 길 수 있다. """
        gx, gy = item['x'], item['y']
        with self.lock:
            rows = {}
            for r in self.robots.values():
                d = (math.hypot(r.pose[0] - gx, r.pose[1] - gy)
                     if r.pose else None)
                why = self.exclusion_reason(r, reason)
                rows[r.robot_id] = {
                    'distance': round(d, 3) if d is not None else None,
                    'eligible': why is None,
                    'reason': why,
                    'pose': list(r.pose) if r.pose else None,
                    'battery': r.battery,
                }
            self.last_selection = {
                'reason': reason,
                'person': [round(gx, 3), round(gy, 3)],
                'selected': chosen.robot_id if chosen else None,
                'metric': 'euclidean',   # 직선거리 (Nav2 경로 길이 아님)
                'standoff': self.APPROACH_STANDOFF,
                'approach': None,        # publish_goal 에서 채운다
                'yaw': None,
                'robots': rows,
                'timestamp': time.time(),
            }

    def select_dispatch_robot(self, item, reason, preempt_helmet):
        """ 출동 로봇 선정.

        후보 판정은 exclusion_reason() 하나만 쓴다. 예전에는 여기에 별도의
        available() 필터가 있어서, 화면에 표시되는 '제외 사유' 와 실제 선정
        기준이 따로 놀 위험이 있었다. 판단은 한 곳에서만 한다.

          EMERGENCY : 배터리를 보지 않는다 (충전 중인 로봇도 끌어낸다).
                      가용 로봇이 아예 없으면 안전모 확인 이동 중인 로봇을 선점.
          HELMET    : 배터리 게이트 유지. 순찰/복귀 중인 로봇에게도 배정하되
                      (로봇이 순찰을 선점해 먼저 배달한다) 진짜 idle 로봇이
                      있으면 그 쪽을 우선한다.
          선정 기준 : 후보 중 사람 좌표까지 직선거리 최소
        """
        gx, gy = item['x'], item['y']

        def distance(r):
            if r.pose is None:
                return float('inf')
            return math.hypot(r.pose[0] - gx, r.pose[1] - gy)

        with self.lock:
            forced_id = item.get('robot_id')
            if forced_id:
                robot = self.robots.get(forced_id)
                why = ('관제 대상이 아님' if robot is None
                       else self.exclusion_reason(robot, reason))
                # 위급은 이미 출동 중인 로봇도 재지정할 수 있다 (최신 목표 우선)
                if robot is not None and (
                        why is None or (reason == 'EMERGENCY'
                                        and why.startswith('이미 출동 중'))):
                    return robot
                if reason == 'EMERGENCY':
                    # 응급만 예외 - 지정 로봇이 못 가도 누군가는 반드시 가야 한다.
                    self.get_logger().warn(
                        f'[{forced_id}] 지정됐으나 사용 불가 ({why}). '
                        f'응급이므로 자동 선정으로 전환한다.')
                else:
                    # 안전모/순찰은 사용자가 로봇을 명시한 것을 존중한다.
                    # 조용히 다른 로봇을 보내면 "robot2 를 눌렀는데 robot9 이 간다".
                    self.get_logger().warn(
                        f'[{forced_id}] {reason} 배정 불가 ({why}). '
                        f'다른 로봇으로 대체하지 않는다 (로봇을 지정했으므로).')
                    return None

            pool = [r for r in self.robots.values()
                    if self.exclusion_reason(r, reason) is None]

            if pool:
                if reason == 'HELMET':
                    # 지금 당장 갈 수 있는 로봇(진짜 idle)을 우선, 그 다음 거리
                    return min(pool, key=lambda r: (
                        0 if r.amr_status in (None, 'IDLE') else 1, distance(r)))
                chosen = min(pool, key=distance)
                if chosen.charging or (chosen.battery is not None
                                       and chosen.battery < self.BATTERY_LOW):
                    b = (f'{chosen.battery*100:.0f}%'
                         if chosen.battery is not None else '?')
                    self.get_logger().warn(
                        f'[{chosen.robot_id}] 배터리 {b} 이지만 위급이므로 출동시킨다 '
                        f'(응급은 배터리보다 우선).')
                return chosen

            if preempt_helmet:
                # 위급인데 가용 로봇이 없다 - 안전모 확인 이동 중인 로봇을 선점
                helmet_pool = [r for r in self.robots.values()
                               if r.fsm_state == 'DISPATCHING'
                               and r.dispatch_reason == 'HELMET'
                               and r.stack_fault is None
                               and r.pose is not None]
                if helmet_pool:
                    return min(helmet_pool, key=distance)
        return None

    def approach_pose(self, person_xy, robot_xy):
        """ PC3 인터페이스 명세: goal 좌표는 '사람의 발 위치'이므로 로봇이
        그 위로 진입하면 안 된다. 사람에서 로봇 방향으로 APPROACH_STANDOFF
        만큼 물러난 접근점과, 접근점에서 사람을 바라보는 yaw(도)를 계산한다.
        로봇 위치를 모르면(amcl 미수신) 원 좌표 + yaw 0 폴백. """
        px, py = person_xy
        if robot_xy is None:
            return px, py, 0.0
        dx, dy = px - robot_xy[0], py - robot_xy[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return px, py, 0.0
        ux, uy = dx / d, dy / d
        # 로봇이 이미 standoff 안쪽에 있으면 후퇴점이 로봇 뒤로 가므로
        # 현재 위치에서 사람만 바라보게 한다.
        if d <= self.APPROACH_STANDOFF:
            return robot_xy[0], robot_xy[1], math.degrees(math.atan2(uy, ux))
        ax = px - ux * self.APPROACH_STANDOFF
        ay = py - uy * self.APPROACH_STANDOFF
        return ax, ay, math.degrees(math.atan2(uy, ux))

    def publish_goal(self, robot, x, y, reason):
        """ 로봇 전용 토픽에 좌표 발행. 위급은 emergency_goal, 안전모는
        helmet_goal 로 보낸다 (로봇 노드가 우선순위를 다르게 처리:
        emergency 는 즉시 선점, helmet 은 idle 상태에서만 실행).
        (x, y)는 감지된 사람의 발 위치이므로 접근점 + yaw 로 보정해 보낸다. """
        with self.lock:
            robot_xy = list(robot.pose) if robot.pose else None
        ax, ay, yaw = self.approach_pose((x, y), robot_xy)
        payload = {'x': ax, 'y': ay, 'yaw': round(yaw, 1), 'reason': reason,
                   'person_x': x, 'person_y': y,
                   'robot_id': robot.robot_id, 'timestamp': time.time()}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pubs = self.helmet_goal_pubs if reason == 'HELMET' else self.emergency_goal_pubs
        pubs[robot.robot_id].publish(msg)
        # 실제로 로봇에게 보낸 접근점/각도를 선정 스냅샷에 채워 넣는다.
        with self.lock:
            if self.last_selection is not None:
                self.last_selection['approach'] = [round(ax, 3), round(ay, 3)]
                self.last_selection['yaw'] = round(yaw, 1)
        self.get_logger().info(
            f'[{robot.robot_id}] 출동 좌표 발행: 사람 ({x:.2f}, {y:.2f}) → '
            f'접근점 ({ax:.2f}, {ay:.2f}, yaw={yaw:.0f}°) reason={reason}')

    def auto_patrol(self, now):
        """ PATROL_PERIOD 주기 자동 순찰 배정.
        조건이 하나라도 어긋나면 이번 주기는 건너뛴다 (큐에 쌓아두지 않음 -
        수동 주입과 달리 자동 주입은 밀린 것을 나중에 몰아서 처리할 이유가 없다):
          - 평시(NORMAL) 일 것 (위급상황 중 순찰 배정 금지)
          - 순찰 큐가 비어 있을 것 (수동 주입 대기분과 중복 방지)
          - 이미 순찰 중(amr_status=PATROL)인 로봇이 없을 것 """
        if self.PATROL_PERIOD <= 0:
            return
        if now - self._last_auto_patrol < self.PATROL_PERIOD:
            return
        with self.lock:
            if (self.situation != 'NORMAL' or self.patrol_queue
                    or any(r.amr_status == 'PATROL' for r in self.robots.values())):
                # 주기는 소모하되 이벤트는 넣지 않는다. 다음 주기에 재평가.
                self._last_auto_patrol = now
                return
            self.patrol_queue.append({})
            self._last_auto_patrol = now
        self.get_logger().info(
            f'자동 순찰 주기 도래({self.PATROL_PERIOD:.0f}s) - '
            f'배터리 최고 idle 로봇에게 순찰 배정 예정')

    def assign_patrol(self):
        with self.lock:
            item = self.patrol_queue.pop(0) if self.patrol_queue else None
        if item is None:
            return
        robot = self.select_patrol_robot(item.get('robot_id'))
        if robot is None:
            if item.get('robot_id'):
                # 지정 로봇이 못 가면 폐기한다 (다른 로봇으로 대체하지 않는다).
                self.get_logger().warn(
                    f'순찰 요청 폐기 - 지정 로봇 {item["robot_id"]} 사용 불가.')
                return
            with self.lock:
                self.patrol_queue.insert(0, item)  # 가용 로봇 생길 때까지 재시도
            return
        payload = {'robot_id': robot.robot_id, 'timestamp': time.time()}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.patrol_cmd_pubs[robot.robot_id].publish(msg)
        battery = f'{robot.battery*100:.0f}%' if robot.battery is not None else '?'
        self.get_logger().info(
            f'[{robot.robot_id}] 순찰 시작 명령 발행 (배터리 {battery})')

    def select_patrol_robot(self, forced_id=None):
        """ 순찰 로봇 선정.

        - robot_id 지정 시: ★ 반드시 그 로봇. 조건에 안 맞으면 아무도 안 보낸다.
          예전에는 지정 로봇이 조건에 안 맞으면 "자동 선정으로 전환" 해서
          다른 로봇을 보냈다. 웹에서 robot2 카드의 [순찰 시작] 을 눌렀는데
          robot9 이 나가는 황당한 동작이 이것 때문이었다. 사용자가 로봇을
          명시했다는 것은 "그 로봇을 보내라" 는 뜻이지 "아무나 보내라" 가 아니다.
          못 보내면 이유를 로그로 남기고 그만둔다.
        - 미지정 시: idle 로봇 중 배터리가 가장 높은 로봇.
        """
        with self.lock:
            if forced_id:
                robot = self.robots.get(forced_id)
                if robot is None:
                    self.get_logger().warn(
                        f'순찰 지정 로봇 {forced_id} 는 관제 대상이 아니다.')
                    return None
                why = self.exclusion_reason(robot, 'PATROL')
                if why is None:
                    return robot
                self.get_logger().warn(
                    f'[{forced_id}] 순찰 배정 불가 ({why}). '
                    f'다른 로봇으로 대체하지 않는다 (로봇을 지정했으므로).')
                return None

            pool = [r for r in self.robots.values()
                    if self.exclusion_reason(r, 'PATROL') is None]
            if pool:
                # 배터리 미보고 로봇(-1 취급)보다 보고된 로봇을 우선한다.
                return max(pool, key=lambda r: r.battery
                           if r.battery is not None else -1.0)
        return None

    # ------------------------------------------------------------------
    # 터미널 상태 테이블
    # ------------------------------------------------------------------
    def log_status_table(self):
        with self.lock:
            situation = self.situation
            n_emg, n_helmet = len(self.emergency_queue), len(self.helmet_queue)
            rows = []
            for r in self.robots.values():
                batt = f'{r.battery*100:5.1f}%' if r.battery is not None else '  ?  '
                pose = (f'({r.pose[0]:6.2f},{r.pose[1]:6.2f})'
                        if r.pose else '(  ?  ,  ?  )')
                kr = KR_STATE.get(r.display_state(), r.display_state())
                rows.append(f'  {r.robot_id:8s} {kr:14s} 배터리 {batt}  위치 {pose}')
        kr_sit = '위급상황' if situation == 'EMERGENCY' else '평시'
        lines = [f'━━ 관제 상태: {situation}({kr_sit})'
                 f'  위급대기 {n_emg} / 안전모대기 {n_helmet} ━━'] + rows
        self.get_logger().info('\n'.join(lines))

    # ------------------------------------------------------------------
    # 지도 화면 (OpenCV)
    # ------------------------------------------------------------------
    def render_frame(self):
        """ map.pgm 위에 로봇/상황/이벤트 지점을 그린 프레임 생성 """
        base = cv2.resize(
            self.map_img, None, fx=self.VIZ_SCALE, fy=self.VIZ_SCALE,
            interpolation=cv2.INTER_NEAREST)
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        with self.lock:
            situation = self.situation
            emg_pts = [(e['x'], e['y']) for e in self.emergency_queue]
            helmet_pts = [(h['x'], h['y']) for h in self.helmet_queue]
            robots = []
            for r in self.robots.values():
                robots.append({
                    'id': r.robot_id, 'pose': r.pose,
                    'state': r.display_state(),
                    'battery': r.battery,
                    'goal': list(r.dispatch_goal) if r.dispatch_goal else None,
                    'reason': r.dispatch_reason,
                })

        # 출동 목표 지점 + 로봇-목표 연결선
        for rb in robots:
            if rb['goal'] is not None:
                gp = self.world_to_px(*rb['goal'])
                color = (60, 60, 255) if rb['reason'] == 'EMERGENCY' else (0, 165, 255)
                cv2.drawMarker(canvas, gp, color, cv2.MARKER_TILTED_CROSS, 22, 3)
                if rb['pose'] is not None:
                    cv2.line(canvas, self.world_to_px(*rb['pose']), gp, color, 1,
                             cv2.LINE_AA)

        # 미배정 이벤트 지점
        for x, y in emg_pts:
            cv2.drawMarker(canvas, self.world_to_px(x, y), (60, 60, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 3)
        for x, y in helmet_pts:
            cv2.circle(canvas, self.world_to_px(x, y), 10, (0, 165, 255), 2)

        # 로봇
        for rb in robots:
            if rb['pose'] is None:
                continue
            p = self.world_to_px(*rb['pose'])
            color = STATE_COLOR.get(rb['state'], (255, 255, 255))
            cv2.circle(canvas, p, 12, color, -1)
            cv2.circle(canvas, p, 12, (30, 30, 30), 2)
            label = rb['id'].replace('robot', 'R')
            cv2.putText(canvas, label, (p[0] - 12, p[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(canvas, label, (p[0] - 12, p[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 상황 배너 (지도 상단)
        banner_color = (0, 0, 200) if situation == 'EMERGENCY' else (0, 130, 0)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), banner_color, -1)
        banner = ('EMERGENCY - dispatching to webcam-detected position'
                  if situation == 'EMERGENCY' else 'NORMAL - routine patrol')
        cv2.putText(canvas, banner, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 우측 상태 패널
        panel = np.full((canvas.shape[0], self.PANEL_W, 3), 30, dtype=np.uint8)
        y = 40
        cv2.putText(panel, 'FLEET MONITOR', (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 40
        for rb in robots:
            color = STATE_COLOR.get(rb['state'], (255, 255, 255))
            batt = f"{rb['battery']*100:.0f}%" if rb['battery'] is not None else '?'
            pose = (f"({rb['pose'][0]:.2f}, {rb['pose'][1]:.2f})"
                    if rb['pose'] else '(?, ?)')
            cv2.circle(panel, (24, y - 6), 8, color, -1)
            cv2.putText(panel, f"{rb['id']}  batt {batt}", (42, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 26
            cv2.putText(panel, rb['state'], (42, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 24
            cv2.putText(panel, pose, (42, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 34
        y += 6
        cv2.putText(panel, f'emergency queue: {len(emg_pts)}', (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2)
        y += 26
        cv2.putText(panel, f'helmet queue: {len(helmet_pts)}', (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        y += 40
        # 범례
        for state in ('NAVIGATING', 'DISPATCHING_EMERGENCY', 'ONSCENE_EMERGENCY',
                      'DISPATCHING_HELMET', 'ONSCENE_HELMET',
                      'RETURNING_LOW_BATTERY', 'CHARGING',
                      'IDLE', 'NO_LOCALIZATION', 'NO_NAV2', 'OFFLINE'):
            cv2.circle(panel, (24, y - 5), 6, STATE_COLOR[state], -1)
            cv2.putText(panel, state, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 22

        return np.hstack([canvas, panel])

    def gui_loop(self):
        # 중요: DISPLAY 가 없으면 cv2.namedWindow() 진입 자체를 막아야 한다.
        # 이 OpenCV 빌드의 Qt 백엔드는 플러그인 로드 실패 시 파이썬 예외가 아니라
        # Qt 내부에서 qFatal()/abort() 를 호출해 프로세스 전체가 죽는다
        # (SIGABRT). try/except 로는 잡히지 않으므로, SSH/systemd 등 headless
        # 환경에서 관제 노드 전체가 함께 죽는 것을 막기 위해 사전 차단한다.
        if not os.environ.get('DISPLAY'):
            self.get_logger().warn(
                'DISPLAY not set. Map window disabled - terminal monitoring only.')
            return
        window = 'Fleet Monitor (map)'
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except Exception as e:
            self.get_logger().warn(
                f'Map window unavailable ({e}). Terminal monitoring only.')
            return
        while not self.gui_stop.is_set():
            try:
                cv2.imshow(window, self.render_frame())
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            except Exception as e:
                self.get_logger().warn(
                    f'Map window failed ({e}). Terminal monitoring only.')
                return
        cv2.destroyAllWindows()

    def destroy_node(self):
        self.gui_stop.set()
        self.gui_thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = FleetFSMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
