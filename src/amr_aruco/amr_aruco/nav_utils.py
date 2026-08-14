"""순찰 웨이포인트/포즈 생성 유틸 (노드 로직과 분리 - 단독 테스트 가능)."""
import json
import math
import os

from geometry_msgs.msg import PoseStamped, Quaternion

# === 순찰 웨이포인트 (patrol_points.json 에서 로드) ===
# "yaw" 필드가 있으면 소화기 포인트로 간주해 그 yaw를 바라보고 도착한 뒤
# aruco_check_done 토픽에서 대조 완료 신호를 받을 때까지 대기했다가 다음 지점으로 이동한다.
# 좌표는 RViz에서 클릭해가며 자주 조정하므로, 코드가 아닌 별도 json 파일로 분리해
# (symlink-install 환경에서는) 재빌드 없이 수정할 수 있게 했다.
DEFAULT_PATROL_POINTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'patrol_points.json')


def load_patrol_points(path=''):
    """patrol_points.json 을 읽는다. path 가 비어 있으면 패키지 기본 파일 사용."""
    with open(path or DEFAULT_PATROL_POINTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_pose_with_yaw(navigator, x, y, yaw_deg):
    """TurtleBot4Directions(4방향)로는 표현 안 되는 임의의 yaw(도 단위)로 PoseStamped 생성"""
    yaw = math.radians(yaw_deg)
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    pose.pose.orientation = Quaternion(
        x=0.0, y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    return pose


def build_patrol_waypoints(navigator, patrol_points, dock_position):
    """순찰 지점들을 (PoseStamped, wait_for_aruco) 튜플 리스트로 변환.

    도킹 스테이션 복귀는 순찰 완료 후 _return_to_dock(여유 지점 이동 -> dock())이
    담당하므로 웨이포인트에는 포함하지 않는다. dock_position 은 마지막 순찰
    지점의 도착 yaw(복귀 방향 바라보기) 계산에만 쓰인다.
    """
    goal_pose = []

    for i, point in enumerate(patrol_points):
        x, y = point['x'], point['y']
        # yaw 필드가 있는 지점은 소화기 포인트로 간주해 대조 완료를 기다린다.
        wait_for_aruco = 'yaw' in point

        if wait_for_aruco:
            yaw_deg = point['yaw']
        else:
            # yaw가 없으면 다음 지점을 바라보며 도착하도록 자동 계산한다.
            if i < len(patrol_points) - 1:
                nx, ny = patrol_points[i + 1]['x'], patrol_points[i + 1]['y']
            else:
                nx, ny = dock_position  # 마지막 순찰 지점은 복귀 지점 방향을 봄
            yaw_deg = math.degrees(math.atan2(ny - y, nx - x))

        pose = get_pose_with_yaw(navigator, x, y, yaw_deg)
        goal_pose.append((pose, wait_for_aruco))

    return goal_pose
