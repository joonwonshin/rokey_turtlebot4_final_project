"""amr_aruco 패키지 공용 상수/QoS/헬퍼.

두 노드(amr_patrol_emer_helmet, aruco_detect)와 fleet_fsm 사이의 "프로토콜"에
해당하는 것들(토픽 이름, QoS)을 한곳에 모아 서로 어긋나지 않게 한다.
토픽 이름은 ROS 파라미터가 아닌 상수로 유지한다 - 노드별로 값을 바꾸면
fleet_fsm/상대 노드와 매칭이 깨지므로, 정말 바꿔야 할 때는 ROS 표준
리맵핑(--ros-args -r)을 쓰는 것이 맞다.
"""
import sys

from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

# 두 노드가 기본으로 사용하는 로봇 네임스페이스. __ns 리맵 없이 ros2 run 으로
# 단독 실행했을 때만 사용된다 (launch 파일에서는 namespace 런치 인자가 우선).
# ROBOT_NAMESPACE = 'robot6'  # 실제 로봇 네임스페이스로 변경 (ros2 topic list로 확인)
DEFAULT_ROBOT_NAMESPACE = 'robot2'

# fleet_fsm 과의 명령/상태 토픽용 QoS (fleet_fsm.py 의 COMMAND_QOS 와 동일해야
# 매칭됨): RELIABLE 재전송 + TRANSIENT_LOCAL(latched) 로 discovery 경합 시에도
# 마지막 명령 1개가 유실되지 않게 한다. 터미널 테스트 시에도 QoS 를 맞춰야 한다:
#   ros2 topic pub -1 --qos-reliability reliable \
#     --qos-durability transient_local /robot6/emergency_clear \
#     std_msgs/msg/Bool "{data: true}"
#
# aruco_scan_enable 게이트에도 같은 프로파일을 쓴다: amr_patrol_emer_helmet 이
# 소화기 포인트 도착 시점에 딱 한 번 발행하는데, aruco_detect 가 그보다 늦게
# 켜지면 VOLATILE(기본 QoS)로는 그 값을 영영 못 받는다. TRANSIENT_LOCAL 로
# 맞춰서 늦게 구독해도 마지막 발행값을 즉시 받도록 한다 (발행/구독 양쪽이
# 반드시 이 동일 프로파일을 써야 함).
COMMAND_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

# fleet_fsm 이 순찰을 지시하는 토픽. 노드는 시작 후 도킹 상태로 대기하다가
# 이 명령을 받아야만 순찰을 시작한다. payload 는 JSON 문자열이며 robot_id 는
# 선택(실려 오면 자기 것인지 검증). 터미널 테스트:
#   ros2 topic pub -1 /robot6/patrol_cmd std_msgs/msg/String \
#     '{data: "{\"robot_id\": \"robot6\"}"}'
PATROL_CMD_TOPIC = 'patrol_cmd'

# 이 노드가 fleet_fsm 에 현재 상태를 알리는 토픽. 로봇 선정 시
# "idle 인 로봇 중에서" 를 판단하는 근거가 된다.
# 발행 값: IDLE / PATROL / EMERGENCY / HELMET / RETURNING
STATUS_TOPIC = 'amr_status'

# fleet_fsm 이 응급 상황 발생 시 로봇을 선정한 뒤 좌표만 발행하는 로봇 전용
# 토픽. 도킹 대기 중이든 순찰 중이든 이 토픽으로 목표가 들어오면 즉시
# undock 후 이동한다 (실제 로봇 선정/우선순위 판단은 fleet_fsm 이 담당하고,
# 이 노드는 좌표를 받아 이동만 수행하는 실행부).
EMERGENCY_GOAL_TOPIC = 'emergency_goal'

# 실습/디버그용: 응급 상황 종료(조치 완료)를 수동으로 알리는 토픽. 실제
# 연동 전까지는 터미널에서 아래처럼 흉내낼 수 있다:
#   ros2 topic pub -1 /robot6/emergency_clear std_msgs/msg/Bool "{data: true}"
# 목표 지점 도착 후에는 이 신호를 받을 때까지 정해진 시간 없이 그 자리에서
# 무기한 대기한다 (충분한 응급조치가 이루어질 때까지). 이동 중에 받으면
# 즉시 취소하고 그 자리에서 대기 상태로 전환한다.
EMERGENCY_CLEAR_TOPIC = 'emergency_clear'

# 응급/순찰보다 낮은 우선순위: 도킹/idle 상태일 때만 안전모 배달 지점으로
# 이동한다. fleet_fsm 이 가장 가까운 idle 로봇을 선정해 발행하며, robot_id 는
# 선택(실려 오면 자기 것인지 검증). 실제 연동 전 터미널 테스트:
#   ros2 topic pub -1 /robot6/helmet_goal std_msgs/msg/String \
#     '{data: "{\"x\": 1.0, \"y\": 2.0, \"robot_id\": \"robot6\"}"}'
HELMET_GOAL_TOPIC = 'helmet_goal'

# 소화기 포인트에서 아루코 마커 인식/대조가 끝났는지 알려주는 디버그용 토픽.
# 실제 인식 노드가 준비되기 전까지는 터미널에서 아래처럼 흉내낼 수 있다:
#   ros2 topic pub -1 /robot2/aruco_check_done std_msgs/msg/Bool "{data: true}"
ARUCO_CHECK_TOPIC = 'aruco_check_done'

# aruco_detect.py 의 검출 게이트. 소화기 포인트에서 대조 결과를 기다리는
# 동안에만 True 를 발행해, 순찰 중 이동하며 마커가 우연히 잡혀도 검출/발행하지
# 않게 한다(aruco_detect.py 의 scan_enabled 참고).
ARUCO_SCAN_ENABLE_TOPIC = 'aruco_scan_enable'


def default_namespace_args(args=None, remap_tf=False):
    """rclpy.init 에 넘길 인자를 만든다.

    launch 파일이나 사용자가 이미 __ns 리맵을 준 경우(sys.argv 에 포함)는
    그대로 두고, 리맵 없이 ros2 run 으로 단독 실행한 경우에만 기존(리팩토링
    전 하드코딩) 동작을 유지하도록 DEFAULT_ROBOT_NAMESPACE 리맵을 주입한다.
    remap_tf=True 면 /tf, /tf_static 도 네임스페이스 아래로 리맵한다
    (Nav2/TurtleBot4Navigator 를 쓰는 노드에 필요).
    """
    argv = list(sys.argv) if args is None else list(args)
    if any('__ns:=' in arg for arg in argv):
        return argv

    extra = ['--ros-args', '-r', f'__ns:=/{DEFAULT_ROBOT_NAMESPACE}']
    if remap_tf:
        extra += [
            '-r', f'/tf:=/{DEFAULT_ROBOT_NAMESPACE}/tf',
            '-r', f'/tf_static:=/{DEFAULT_ROBOT_NAMESPACE}/tf_static',
        ]
    return argv + extra
