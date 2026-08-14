#!/usr/bin/env python3
"""
PC3(vision_pc3) → fleet_fsm 토픽 브릿지 노드.

vision_pc3 의 SafetyRosBridge 는 map 좌표를 geometry_msgs/PoseStamped 로
/safety/* 토픽에 발행하지만, fleet_fsm 은 std_msgs/String(JSON {"x","y"}) 을
/alert/* 토픽으로 기대한다. 양쪽 코드를 건드리지 않고 이 노드가 중간에서
타입/이름을 변환한다.

    /safety/emergency_goal  (PoseStamped)          → /alert/emergency       (String JSON)
    /safety/helmet_goal     (PoseStamped, 2Hz follow) → /alert/helmet       (String JSON, 에피소드당 1회)
    /safety/emergency_state (String "EMERGENCY_CLEAR") → /alert/emergency_clear (String "{}")
    /safety/helmet_state    (String "HELMET_CLEAR")    → /alert/helmet_clear    (String "{}")

helmet_goal 은 PC3 가 follow 모드로 같은 사람에 대해 2Hz 로 재발행하는데,
fleet_fsm 의 helmet_callback 은 수신마다 큐에 append 하므로 그대로 중계하면
goal 이 계속 쌓여 로봇이 중복 배정된다. 그래서 /safety/helmet_state 의
NO_HELMET ~ HELMET_CLEAR 를 하나의 에피소드로 보고, 에피소드당 첫 goal
한 번만 /alert/helmet 으로 넘긴다 (이후 follow 갱신 좌표는 무시 - 로봇은
최초 감지 지점으로 확인 이동하면 충분).

실행:
    ros2 run fp_amr_fsm safety_alert_bridge
"""

import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

# PC3 goal 토픽과 동일한 QoS (RELIABLE + VOLATILE, depth 1).
# goal 은 절대 latch 되면 안 된다 - 처리 완료된 옛 좌표로 재출동 방지.
GOAL_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# PC3 state 토픽과 동일한 QoS (RELIABLE + TRANSIENT_LOCAL, depth 1).
# 브릿지가 늦게 떠도 현재 상태(latched)를 즉시 받는다.
STATE_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class SafetyAlertBridge(Node):

    # goal 이 state 보다 먼저 도착하는 경합 판별 윈도우 [s].
    # 같은 에피소드의 NO_HELMET 이 이 시간 안에 오면 goal 재중계를 막는다.
    RACE_WINDOW_SEC = 2.0

    def __init__(self):
        super().__init__('safety_alert_bridge')

        # fleet_fsm 의 /alert/* 구독은 기본 QoS(depth 10) 이므로 그대로 맞춘다.
        self.pub_emergency = self.create_publisher(String, '/alert/emergency', 10)
        self.pub_helmet = self.create_publisher(String, '/alert/helmet', 10)
        self.pub_emergency_clear = self.create_publisher(
            String, '/alert/emergency_clear', 10)
        self.pub_helmet_clear = self.create_publisher(
            String, '/alert/helmet_clear', 10)

        self.create_subscription(
            PoseStamped, '/safety/emergency_goal', self.emergency_goal_cb, GOAL_QOS)
        self.create_subscription(
            PoseStamped, '/safety/helmet_goal', self.helmet_goal_cb, GOAL_QOS)
        self.create_subscription(
            String, '/safety/emergency_state', self.emergency_state_cb, STATE_QOS)
        self.create_subscription(
            String, '/safety/helmet_state', self.helmet_state_cb, STATE_QOS)

        # helmet 에피소드 게이트: NO_HELMET 수신 후 goal 1회만 통과.
        # 브릿지가 helmet_state 보다 goal 을 먼저 받는 경합도 있으므로
        # (둘 다 RELIABLE 이지만 토픽 간 순서 보장은 없음) goal 이 먼저 와도
        # 에피소드가 열려 있지 않으면 새 에피소드로 간주해 통과시킨다.
        self.helmet_episode_open = False   # NO_HELMET ~ HELMET_CLEAR 구간
        self.helmet_forwarded = False      # 이번 에피소드에서 goal 전달했는지
        self.helmet_last_forward = 0.0     # 마지막 goal 중계 시각 (경합 판별용)
        # PC3 명세 §5: emergency 상태에서 helmet_goal 은 PC3 가 3중으로
        # 막지만, PC4 측에서도 방어적으로 한 번 더 확인할 것을 권장.
        self.emergency_active = False

        self.get_logger().info(
            'safety_alert_bridge up: /safety/* (PoseStamped) → /alert/* (String JSON)')

    # ------------------------------------------------------------------
    @staticmethod
    def _xy_json(pose_msg):
        return json.dumps({
            'x': round(float(pose_msg.pose.position.x), 3),
            'y': round(float(pose_msg.pose.position.y), 3),
        })

    def emergency_goal_cb(self, msg):
        """ emergency 는 PC3 쪽에서 이미 엣지 1회 발행이므로 그대로 중계. """
        out = String()
        out.data = self._xy_json(msg)
        self.pub_emergency.publish(out)
        self.get_logger().warn(f'/alert/emergency 중계: {out.data}')

    def helmet_goal_cb(self, msg):
        """ follow 모드 재발행(2Hz)은 무시하고 에피소드당 첫 goal 만 중계. """
        if self.emergency_active:
            # 방어적 게이트: 긴급 대응이 항상 우선. PC3 가 정상이라면 이
            # 조합은 오지 않지만, 온 경우 helmet 확인 이동을 보류한다.
            self.get_logger().warn(
                'emergency 진행 중 helmet_goal 수신 - 중계 보류 (명세 §5 방어 게이트)')
            return
        if self.helmet_episode_open and self.helmet_forwarded:
            return
        self.helmet_episode_open = True
        self.helmet_forwarded = True
        self.helmet_last_forward = time.monotonic()
        out = String()
        out.data = self._xy_json(msg)
        self.pub_helmet.publish(out)
        self.get_logger().warn(f'/alert/helmet 중계 (에피소드 첫 goal): {out.data}')

    def emergency_state_cb(self, msg):
        """ EMERGENCY_CLEAR → /alert/emergency_clear "{}" (위급 출동 중인
        모든 로봇에게 clear - fleet_fsm 이 대상을 알아서 고른다). """
        self.emergency_active = (msg.data == 'EMERGENCY')
        if msg.data == 'EMERGENCY_CLEAR':
            out = String()
            out.data = '{}'
            self.pub_emergency_clear.publish(out)
            self.get_logger().info('/alert/emergency_clear 중계 (EMERGENCY_CLEAR)')

    def helmet_state_cb(self, msg):
        if msg.data == 'NO_HELMET':
            # NO_HELMET 은 엣지 발행이라 같은 에피소드 안에서 재수신되지
            # 않는다. 다만 두 경우를 구분해야 한다:
            #  (a) 같은 에피소드의 goal 이 state 보다 먼저 도착한 경합
            #      → 방금(RACE_WINDOW 이내) forwarded 됐음. 리셋하면 같은
            #        goal 이 한 번 더 나가 로봇이 중복 배정되므로 유지.
            #  (b) PC3 가 CLEAR 없이 비정상 종료(crash) 후 재시작한 새 에피소드
            #      → 마지막 중계가 오래전. stale 게이트를 리셋해 새 goal 통과.
            if (time.monotonic() - self.helmet_last_forward) > self.RACE_WINDOW_SEC:
                self.helmet_forwarded = False
            self.helmet_episode_open = True
        else:  # HELMET_CLEAR (또는 기타 해제 라벨)
            # goal 을 실제로 중계했던 에피소드가 닫힐 때만 clear 를 중계한다
            # (브릿지 기동 시 latched HELMET_CLEAR 수신 등 무의미한 clear 방지).
            forwarded = self.helmet_forwarded
            self.helmet_episode_open = False
            self.helmet_forwarded = False
            if forwarded:
                out = String()
                out.data = '{}'
                self.pub_helmet_clear.publish(out)
                self.get_logger().info(
                    '/alert/helmet_clear 중계 (HELMET_CLEAR) - 에피소드 종료')
            else:
                self.get_logger().info('helmet 에피소드 종료 - 다음 감지 시 새 goal 중계')


def main():
    rclpy.init()
    node = SafetyAlertBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
