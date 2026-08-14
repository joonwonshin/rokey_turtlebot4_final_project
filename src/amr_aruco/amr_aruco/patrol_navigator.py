"""순찰/응급/안전모 실행 노드의 상태 보관 + 토픽 입출력 담당 네비게이터 클래스.

노드 로직(순찰 루프, 응급 출동 절차 등)은 amr_patrol_emer_helmet.py 에 있고,
이 모듈은 fleet_fsm/aruco_detect 와 주고받는 토픽 콜백과 그 상태 플래그,
그리고 ROS 파라미터 선언만 담당한다.
"""
import json

from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, String

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator

from amr_aruco.common import (
    ARUCO_CHECK_TOPIC,
    ARUCO_SCAN_ENABLE_TOPIC,
    COMMAND_QOS,
    EMERGENCY_CLEAR_TOPIC,
    EMERGENCY_GOAL_TOPIC,
    HELMET_GOAL_TOPIC,
    PATROL_CMD_TOPIC,
    STATUS_TOPIC,
)


class PatrolNavigator(TurtleBot4Navigator):
    """순찰 중에도 배터리 상태를 계속 구독하는 네비게이터."""

    def __init__(self):
        super().__init__()

        # === ROS 파라미터 (기본값/의미/제약은 config/amr_aruco_params.yaml 참고) ===
        # 참조 이름은 기존 상수명(소문자)을 그대로 유지한다.
        self.declare_parameter('battery_low_threshold', 0.3)
        self.declare_parameter('dock_position', [0.0, 0.0])
        self.declare_parameter('dock_approach_position', [-0.5, 0.0])
        self.declare_parameter('dock_direction', 0)
        self.declare_parameter('helmet_idle_delay', 3.0)
        self.declare_parameter('helmet_delivery_hold', 5.0)
        self.declare_parameter('aruco_settle_time', 3.0)
        self.declare_parameter('aruco_check_timeout', 5.0)
        self.declare_parameter('patrol_points_file', '')

        self.battery_low_threshold = self.get_parameter('battery_low_threshold').value
        self.dock_position = list(self.get_parameter('dock_position').value)
        self.dock_approach_position = list(self.get_parameter('dock_approach_position').value)
        self.dock_direction = TurtleBot4Directions(self.get_parameter('dock_direction').value)
        self.helmet_idle_delay = self.get_parameter('helmet_idle_delay').value
        self.helmet_delivery_hold = self.get_parameter('helmet_delivery_hold').value
        self.aruco_settle_time = self.get_parameter('aruco_settle_time').value
        self.aruco_check_timeout = self.get_parameter('aruco_check_timeout').value
        self.patrol_points_file = self.get_parameter('patrol_points_file').value

        # robot_id 검증용 자기 이름. 토픽 스코프의 근거인 노드 네임스페이스에서
        # 직접 얻으므로 별도 파라미터와 어긋날 일이 없다 (예: '/robot2' -> 'robot2').
        self.robot_namespace = self.get_namespace().strip('/')

        self.battery_percentage = None
        self.aruco_check_done = False
        # fleet_fsm 이 로봇을 선정한 뒤 내려보내는 응급 출동 좌표. 콜백과
        # 메인 루프가 동일 스레드(rclpy.spin_once 기반)에서 돌기 때문에
        # 별도 락 없이도 안전하다 (배터리/아루코 상태와 동일한 방식).
        self.pending_emergency = None
        self.emergency_clear_requested = False
        self.pending_helmet = None
        self.patrol_requested = False

        self.status_pub = self.create_publisher(String, STATUS_TOPIC, COMMAND_QOS)
        # aruco_detect 가 소화기 포인트 도착 시점보다 늦게 켜져도 마지막 값을
        # 받을 수 있도록 VOLATILE(기본) 대신 latched QoS(COMMAND_QOS) 사용.
        self.aruco_scan_pub = self.create_publisher(Bool, ARUCO_SCAN_ENABLE_TOPIC, COMMAND_QOS)

        self.create_subscription(
            BatteryState,
            'battery_state',
            self._battery_callback,
            10,
        )

        self.create_subscription(
            Bool,
            ARUCO_CHECK_TOPIC,
            self._aruco_check_callback,
            10,
        )

        self.create_subscription(
            String,
            EMERGENCY_GOAL_TOPIC,
            self._emergency_callback,
            COMMAND_QOS,
        )

        self.create_subscription(
            Bool,
            EMERGENCY_CLEAR_TOPIC,
            self._emergency_clear_callback,
            COMMAND_QOS,
        )

        self.create_subscription(
            String,
            HELMET_GOAL_TOPIC,
            self._helmet_callback,
            COMMAND_QOS,
        )

        self.create_subscription(
            String,
            PATROL_CMD_TOPIC,
            self._patrol_cmd_callback,
            COMMAND_QOS,
        )

    def publish_status(self, state: str):
        msg = String()
        msg.data = state
        self.status_pub.publish(msg)
        self.info(f'상태 변경: {state}')

    def _battery_callback(self, msg: BatteryState):
        self.battery_percentage = msg.percentage

    def _aruco_check_callback(self, msg: Bool):
        if msg.data:
            self.aruco_check_done = True

    def _emergency_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            item = {
                'x': float(data['x']),
                'y': float(data['y']),
                'reason': data.get('reason', 'EMERGENCY'),
            }
        except Exception as e:
            self.error(f'Invalid emergency_goal message: {e}')
            return
        # fleet_fsm 이 로봇 전용 토픽(f'/{robot_id}/emergency_goal')으로만
        # 발행하므로 원래 토픽 이름 자체가 이미 자기 id 로 스코프돼 있지만,
        # payload 에 robot_id 가 실려 오는 경우 한 번 더 검증해 다른 로봇 앞으로
        # 온 메시지를 잘못 수신(오배선/디버그 pub 실수 등)했을 때 무시한다.
        target = data.get('robot_id')
        if target and target != self.robot_namespace:
            self.info(f'다른 로봇({target}) 대상 응급 신호 - 무시합니다.')
            return
        # 최신 목표 1개만 유지 - 순찰 중 이동/대조 대기 도중에 와도 즉시
        # 눈에 띄어야 하므로 이전에 처리 못한 목표가 있어도 덮어쓴다.
        self.pending_emergency = item
        self.info(
            f"응급 출동 신호 수신 (reason={item['reason']}): "
            f"({item['x']:.2f}, {item['y']:.2f})"
        )

    def _emergency_clear_callback(self, msg: Bool):
        if msg.data:
            self.emergency_clear_requested = True
            self.info('응급 상황 종료 신호 수신.')

    def _helmet_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            item = {'x': float(data['x']), 'y': float(data['y'])}
        except Exception as e:
            self.error(f'Invalid helmet_goal message: {e}')
            return
        target = data.get('robot_id')
        if target and target != self.robot_namespace:
            self.info(f'다른 로봇({target}) 대상 안전모 신호 - 무시합니다.')
            return
        self.pending_helmet = item
        self.info(f"안전모 배달 신호 수신: ({item['x']:.2f}, {item['y']:.2f})")

    def _patrol_cmd_callback(self, msg: String):
        # payload 없이 빈 문자열로 와도 순찰 시작으로 처리한다.
        data = {}
        if msg.data.strip():
            try:
                data = json.loads(msg.data)
            except Exception as e:
                self.error(f'Invalid patrol_cmd message: {e}')
                return
        target = data.get('robot_id')
        if target and target != self.robot_namespace:
            self.info(f'다른 로봇({target}) 대상 순찰 명령 - 무시합니다.')
            return
        self.patrol_requested = True
        self.info('순찰 시작 명령 수신.')

    def take_patrol_cmd(self):
        requested = self.patrol_requested
        self.patrol_requested = False
        return requested

    def has_pending_emergency(self):
        return self.pending_emergency is not None

    def take_emergency(self):
        item = self.pending_emergency
        self.pending_emergency = None
        return item

    def take_emergency_clear(self):
        cleared = self.emergency_clear_requested
        self.emergency_clear_requested = False
        return cleared

    def has_pending_helmet(self):
        return self.pending_helmet is not None

    def take_helmet(self):
        item = self.pending_helmet
        self.pending_helmet = None
        return item

    def is_battery_low(self):
        return (
            self.battery_percentage is not None
            and self.battery_percentage < self.battery_low_threshold
        )

    def set_aruco_scan(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.aruco_scan_pub.publish(msg)
