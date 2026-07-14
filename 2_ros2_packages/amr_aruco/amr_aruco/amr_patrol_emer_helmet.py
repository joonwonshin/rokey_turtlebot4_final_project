#!/usr/bin/env python3
"""순찰/응급출동/안전모배달 실행부 노드 (메인 FSM 루프).

- 상태/토픽 입출력: patrol_navigator.PatrolNavigator
- 웨이포인트/포즈 유틸: nav_utils
- 토픽 이름/QoS 등 fleet_fsm 과의 프로토콜: common
- 파라미터 기본값/의미/제약: config/amr_aruco_params.yaml

=== fleet_fsm 연동 개요 ===
이 노드는 로봇 선정 판단을 하지 않는 "실행부"다. fleet_fsm(상태머신)이
 - 순찰: 배터리가 가장 높은 idle 로봇 선정
 - 응급/안전모: 목표 지점에서 가장 가까운 로봇 선정
을 담당하고, 선정된 로봇의 전용 토픽(f'/{robot_id}/...')으로만 명령을
발행한다. 선정에 필요한 정보는 fleet_fsm 이 각 로봇 네임스페이스의
 - /robotX/battery_state (sensor_msgs/BatteryState): 배터리 잔량
 - /robotX/amcl_pose (geometry_msgs/PoseWithCovarianceStamped): 현재 위치
 - /robotX/amr_status (std_msgs/String, common.STATUS_TOPIC): IDLE/PATROL/... 상태
를 구독해서 얻는다.
"""
import signal
import time

import rclpy
from rclpy.executors import ExternalShutdownException

from nav2_simple_commander.robot_navigator import TaskResult

from amr_aruco.common import default_namespace_args
from amr_aruco.nav_utils import build_patrol_waypoints, get_pose_with_yaw, load_patrol_points
from amr_aruco.patrol_navigator import PatrolNavigator


def _cancel_and_wait(navigator):
    """ cancelTask()는 취소 "요청"이 접수됐는지만 확인하고 반환하므로,
    이전 작업이 서버에서 완전히 종료될 때까지 기다린 뒤 다음 goToPose를
    보내야 한다. 그렇지 않으면 새 goToPose가 이전 취소 상태와 뒤섞여
    이동도 하기 전에 즉시 완료된 것으로 오판될 수 있다. """
    navigator.cancelTask()
    while not navigator.isTaskComplete():
        pass


def run_patrol(navigator, waypoints, start_index=0):
    """
    웨이포인트를 하나씩 goToPose로 순찰한다. start_index 부터 시작하므로
    응급 출동으로 중단됐던 지점을 그대로 이어서 재개할 수 있다.

    소화기 포인트(wait_for_aruco=True)에 도착하면 aruco_check_done 토픽으로
    대조 완료 신호가 올 때까지 대기했다가 다음 지점으로 이동한다.

    배터리가 낮아지거나 응급 출동 신호가 들어오면 즉시 태스크를 취소하고
    (result, 중단된 waypoint index) 를 반환한다 ('battery_low' / 'emergency').
    끝까지 완주하면 ('completed', None).
    """
    for idx in range(start_index, len(waypoints)):
        pose, wait_for_aruco = waypoints[idx]
        navigator.goToPose(pose)

        while not navigator.isTaskComplete():
            if navigator.has_pending_emergency():
                navigator.info('응급 출동 신호 감지 - 순찰을 중단합니다.')
                _cancel_and_wait(navigator)
                return 'emergency', idx
            if navigator.is_battery_low():
                navigator.info(
                    f'배터리 잔량 {navigator.battery_percentage * 100:.1f}% '
                    f'({navigator.battery_low_threshold * 100:.0f}% 미만) - '
                    '순찰을 중단하고 복귀합니다.'
                )
                _cancel_and_wait(navigator)
                return 'battery_low', idx

        if wait_for_aruco:
            navigator.info(
                f'소화기 포인트 도착. 완전히 정지할 때까지 '
                f'{navigator.aruco_settle_time:.1f}초 대기합니다...'
            )
            # 도착 직전 관성으로 흔들리는 프레임 기반의 오검출을 걸러내기 위해,
            # 완전히 멈출 때까지 잠깐 그대로 대기한 뒤에만 aruco_check_done 을 본다.
            time.sleep(navigator.aruco_settle_time)

            navigator.info('아루코 마커 대조 결과를 기다립니다...')
            navigator.set_aruco_scan(True)
            try:
                navigator.aruco_check_done = False
                wait_start = time.monotonic()
                while not navigator.aruco_check_done:
                    rclpy.spin_once(navigator, timeout_sec=0.1)
                    if navigator.has_pending_emergency():
                        navigator.info('응급 출동 신호 감지 - 순찰을 중단합니다.')
                        return 'emergency', idx
                    if navigator.is_battery_low():
                        navigator.info(
                            f'배터리 잔량 {navigator.battery_percentage * 100:.1f}% '
                            f'({navigator.battery_low_threshold * 100:.0f}% 미만) - '
                            '순찰을 중단하고 복귀합니다.'
                        )
                        return 'battery_low', idx
                    if time.monotonic() - wait_start >= navigator.aruco_check_timeout:
                        navigator.info(
                            f'아루코 대조 결과가 {navigator.aruco_check_timeout:.1f}초 동안 '
                            '오지 않아 대조를 포기하고 다음 지점으로 이동합니다.'
                        )
                        navigator.aruco_check_done = True
                        break
                else:
                    navigator.info('아루코 마커 대조 완료. 다음 지점으로 이동합니다.')
            finally:
                # 정상 완료든 응급/배터리 부족으로 중단되든, 이 지점을 벗어나면
                # 반드시 스캔을 꺼서 이동 중 우연히 잡히는 마커까지 발행되지 않게 한다.
                navigator.set_aruco_scan(False)

    return 'completed', None


def run_emergency_dispatch(navigator):
    """
    대기 중인 응급 출동 좌표로 이동한다. 기존 순찰 웨이포인트 이동과 동일한
    get_pose_with_yaw + goToPose 로직을 재사용하되, yaw 는 지정되지 않으므로
    0.0 으로 둔다. 도킹 중이면 먼저 undock 하고, 이동 중 더 최신 응급 신호가
    들어오면 현재 이동을 취소하고 새 목표로 즉시 갱신한다.

    목표 지점 도착 후에는 emergency_clear(디버그용 조치 완료 신호)를 받을
    때까지 정해진 시간 없이 그 자리에서 무기한 대기한다 - 실제 응급조치가
    끝나는 시점은 현장 상황에 따라 다르므로 fixed hold 로 임의 종료하지
    않는다. 대기 중 더 최신 응급 신호가 오면 그 목표로 즉시 갱신한다.
    """
    # 이전 응급 상황에서 온(또는 latched 로 남아 있던) clear 신호가 새 응급의
    # 대기를 즉시 끝내버리지 않도록, 출동 시작 시점에 잔여 플래그를 버린다.
    navigator.take_emergency_clear()

    while navigator.has_pending_emergency():
        item = navigator.take_emergency()
        x, y, reason = item['x'], item['y'], item['reason']
        navigator.info(f'응급 출동 시작(reason={reason}): ({x:.2f}, {y:.2f})')

        if navigator.getDockedStatus():
            navigator.info('도킹 상태 - undock 후 출동합니다.')
            navigator.undock()

        pose = get_pose_with_yaw(navigator, x, y, 0.0)
        navigator.goToPose(pose)

        while not navigator.isTaskComplete():
            if navigator.take_emergency_clear():
                navigator.info('이동 중 상황 종료 신호 수신 - 출동을 중단합니다.')
                _cancel_and_wait(navigator)
                return
            if navigator.has_pending_emergency():
                navigator.info('더 최신 응급 신호 수신 - 목표를 갱신합니다.')
                _cancel_and_wait(navigator)
                break
        else:
            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                navigator.info(
                    f'응급 출동 지점 도착: ({x:.2f}, {y:.2f}). '
                    '조치 완료(emergency_clear) 신호를 기다립니다.'
                )
            else:
                navigator.error(
                    f'응급 출동 이동 실패 (result={result}). '
                    '그 자리에서 조치 완료 신호를 기다립니다.'
                )

            # 응급조치가 충분히 이루어질 때까지 그 자리에서 무기한 대기한다.
            while not navigator.take_emergency_clear():
                if navigator.has_pending_emergency():
                    break
                rclpy.spin_once(navigator, timeout_sec=0.1)
            else:
                navigator.info('응급 상황 종료 신호 수신 - 순찰을 이어서 진행합니다.')
                return
            # break 로 빠져나온 경우(더 최신 응급 신호) 바깥 while 이 이어서 처리


def run_helmet_delivery(navigator):
    """
    응급/순찰보다 낮은 우선순위로 안전모 배달 지점까지 이동한다. 도킹/idle
    상태(main() 의 대기 루프)에서만 호출되므로 순찰이나 응급 출동을 방해하지
    않는다. 이동 중 응급 신호가 들어오면 즉시 중단하고 응급 대응에 양보한다.
    """
    while navigator.has_pending_helmet():
        if navigator.has_pending_emergency():
            navigator.info('응급 신호 우선 - 안전모 배달을 시작하지 않습니다.')
            return

        item = navigator.take_helmet()
        x, y = item['x'], item['y']
        navigator.info(f'안전모 배달 시작: ({x:.2f}, {y:.2f})')

        if navigator.getDockedStatus():
            navigator.undock()

        pose = get_pose_with_yaw(navigator, x, y, 0.0)
        navigator.goToPose(pose)

        while not navigator.isTaskComplete():
            if navigator.has_pending_emergency():
                navigator.info('안전모 배달 중 응급 신호 수신 - 중단합니다.')
                _cancel_and_wait(navigator)
                return
        else:
            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                navigator.info(f'안전모 배달 지점 도착: ({x:.2f}, {y:.2f})')
            else:
                navigator.error(f'안전모 배달 이동 실패 (result={result})')

            deadline = time.monotonic() + navigator.helmet_delivery_hold
            while time.monotonic() < deadline:
                rclpy.spin_once(navigator, timeout_sec=0.1)
                if navigator.has_pending_emergency():
                    break

    navigator.info('안전모 배달 완료. 도킹 스테이션으로 복귀합니다.')
    _return_to_dock(navigator)


def _return_to_dock(navigator):
    """ 순찰 중단 지점은 도킹 스테이션과 멀 수 있으므로, dock()을 바로 부르지
    않고 도킹 스테이션 앞 여유 지점까지 먼저 이동한 뒤 dock() 한다.

    dock() 전에는 도킹 스테이션 좌표(dock_position)로 바로 가지 않고
    dock_approach_position 여유 지점까지만 이동한다. dock_position은 코스트맵상
    도킹 스테이션 본체와 겹쳐 있어 곧장 goToPose하면 경로 계획이 실패할 수 있고,
    마지막 정밀 접속은 IR 센서 기반의 dock()에 맡겨야 하기 때문이다. """
    dock_approach_pose = navigator.getPoseStamped(
        navigator.dock_approach_position, navigator.dock_direction)
    navigator.goToPose(dock_approach_pose)
    while not navigator.isTaskComplete():
        pass

    approach_result = navigator.getResult()
    if approach_result != TaskResult.SUCCEEDED:
        navigator.error(
            f'도킹 스테이션 앞 지점 이동 실패 (result={approach_result}). '
            '현재 위치에서 바로 도킹을 시도합니다.'
        )

    navigator.dock()


def _run(navigator):
    """노드 초기화(도킹/초기 pose/Nav2 대기) 후 메인 FSM 루프를 돈다."""
    # Start on dock
    if not navigator.getDockedStatus():
        navigator.info('Docking before intialising pose')
        navigator.dock()

    # Set initial pose
    initial_pose = navigator.getPoseStamped(navigator.dock_position, navigator.dock_direction)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2
    navigator.waitUntilNav2Active()

    patrol_points = load_patrol_points(navigator.patrol_points_file)
    waypoints = build_patrol_waypoints(navigator, patrol_points, navigator.dock_position)

    navigator.info('fleet_fsm 명령 대기를 시작합니다 (patrol_cmd / emergency_goal / helmet_goal).')

    # 응급 출동으로 중단된 순찰 웨이포인트 index. 응급 조치가 끝나면 처음부터
    # 다시 돌지 않고 이 지점부터 이어서 순찰한다.
    patrol_index = 0
    # 순찰은 fleet_fsm 의 patrol_cmd 를 받아야만 시작한다. 응급 출동으로
    # 중단된 경우에도 이 플래그가 살아 있으므로 조치 후 이어서 순찰한다.
    patrol_active = False

    navigator.publish_status('IDLE')
    idle_since = time.monotonic()

    # 메인 루프: 우선순위는 응급 > 순찰(명령 기반) > 안전모 배달.
    # 도킹 대기 중이든 순찰 중이든 emergency_goal 이 오면 최우선으로 반응한다.
    while rclpy.ok():
        if navigator.has_pending_emergency():
            navigator.publish_status('EMERGENCY')
            run_emergency_dispatch(navigator)
            if not patrol_active:
                # 재개할 순찰이 없으면 도킹 스테이션으로 복귀해 대기한다.
                navigator.publish_status('RETURNING')
                _return_to_dock(navigator)
                navigator.publish_status('IDLE')
                idle_since = time.monotonic()
            continue

        if navigator.take_patrol_cmd():
            patrol_active = True
            patrol_index = 0

        if patrol_active:
            if navigator.getDockedStatus():
                navigator.undock()
            navigator.publish_status('PATROL')
            result, idx = run_patrol(navigator, waypoints, start_index=patrol_index)

            if result == 'emergency':
                # 응급 신호는 run_patrol 이 이미 취소만 해두고 반환했으므로,
                # 다음 루프 최상단에서 run_emergency_dispatch 로 처리된다.
                # 중단된 지점을 기억해뒀다가 응급 조치가 끝나면 이어서 순찰한다.
                patrol_index = idx
                continue

            if result == 'battery_low':
                navigator.info('배터리 부족으로 순찰을 중단했습니다. 도킹 스테이션으로 복귀합니다.')
            else:
                navigator.info('순찰 완료. 도킹 스테이션으로 복귀합니다.')
            patrol_active = False
            patrol_index = 0
            navigator.publish_status('RETURNING')
            _return_to_dock(navigator)
            navigator.publish_status('IDLE')
            navigator.info('대기 모드: 도킹 상태에서 다음 명령을 기다립니다.')
            idle_since = time.monotonic()
            continue

        # 대기 모드: 도킹 상태에서 fleet_fsm 명령을 기다린다 (spin_once 로만
        # 대기하므로 CPU 를 거의 쓰지 않는다). 안전모 배달은 helmet_idle_delay
        # 초 동안 idle 상태가 유지된 뒤에만 시작한다 (막 도킹한 로봇이 곧바로
        # 재출동하지 않도록 하는 유예 시간).
        rclpy.spin_once(navigator, timeout_sec=0.2)
        if (navigator.has_pending_helmet()
                and time.monotonic() - idle_since >= navigator.helmet_idle_delay):
            navigator.publish_status('HELMET')
            run_helmet_delivery(navigator)
            if not navigator.has_pending_emergency():
                navigator.publish_status('IDLE')
            idle_since = time.monotonic()


def main(args=None):
    # __ns 리맵 없이 단독 실행하면 기본 네임스페이스(robot2)로 리맵해 기존
    # 동작을 유지한다. launch 파일이 namespace/tf 리맵을 준 경우는 그대로 사용.
    rclpy.init(args=default_namespace_args(args, remap_tf=True))

    navigator = None
    try:
        navigator = PatrolNavigator()
        _run(navigator)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ros2 launch 의 SIGINT 종료 시 트레이스백 없이 조용히 내려가도록 처리
        pass
    finally:
        # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고, ros2 launch 가
        # 자식에게 SIGINT 를 한 번 더 전달하므로 정리 도중 두 번째
        # KeyboardInterrupt 가 터질 수 있다. 이후 SIGINT 는 무시하고 정리한다.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if navigator is not None:
            navigator.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
