from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def persistent_process(executable):
    # 예기치 않게 종료된 프로세스는 2초 후 자동으로 다시 시작합니다.
    return Node(
        package="sqlite3db",
        executable=executable,
        output="screen",
        respawn=True,
        respawn_delay=2.0,
    )


def generate_launch_description():
    # launch 시작 시 DB 테이블을 한 번 준비하는 일회성 프로세스입니다.
    create_db = Node(
        package="sqlite3db",
        executable="create_db",
        output="screen",
    )

    persistent_nodes = [
        persistent_process("ros2_db_node"),
        persistent_process("db_update"),
        persistent_process("app"),
    ]

    # create_db가 종료된 뒤에만 세 개의 지속 프로세스를 시작합니다.
    start_persistent_nodes = RegisterEventHandler(
        OnProcessExit(
            target_action=create_db,
            on_exit=persistent_nodes,
        )
    )

    return LaunchDescription(
        [
            start_persistent_nodes,
            create_db,
        ]
    )
