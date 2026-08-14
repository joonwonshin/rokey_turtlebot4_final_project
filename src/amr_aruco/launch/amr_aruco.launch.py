"""amr_aruco 패키지의 두 노드(순찰 FSM + 아루코 인식)를 함께 실행하는 launch.

사용 예:
  ros2 launch amr_aruco amr_aruco.launch.py
  ros2 launch amr_aruco amr_aruco.launch.py namespace:=robot6
  ros2 launch amr_aruco amr_aruco.launch.py params_file:=/path/to/my_params.yaml

TurtleBot4 브링업 + Nav2 는 별도 launch 로 먼저 떠 있어야 한다 (README 참고).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='robot2',
            description='로봇 네임스페이스 (fleet_fsm 이 명령을 내리는 robot_id 와 동일해야 함)',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('amr_aruco'), 'config', 'amr_aruco_params.yaml']),
            description='두 노드의 파라미터 yaml 경로 (기본: 패키지 config)',
        ),

        # 순찰/응급출동/안전모배달 실행부. TurtleBot4Navigator(Nav2)가 tf 를
        # 네임스페이스 아래(/robot2/tf)에서 찾으므로 /tf 리맵이 필요하다.
        Node(
            package='amr_aruco',
            executable='amr_patrol_emer_helmet',
            namespace=namespace,
            parameters=[params_file],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static'),
            ],
            output='screen',
        ),

        # 아루코 마커 인식 노드 (같은 네임스페이스에서 실행).
        Node(
            package='amr_aruco',
            executable='aruco_detect',
            namespace=namespace,
            parameters=[params_file],
            output='screen',
        ),
    ])
