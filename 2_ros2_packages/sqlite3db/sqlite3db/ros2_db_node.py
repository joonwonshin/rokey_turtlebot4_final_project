# 1_ros2_db_node.py
# ROS2와 DB 연동 노드

from pathlib import Path
import sqlite3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from irobot_create_msgs.msg import DockStatus
from sensor_msgs.msg import BatteryState, CompressedImage


# ROS2 메시지에서 받은 최신 상태를 저장할 SQLite DB 파일입니다.
DB_PATH = Path(__file__).with_name("fire_db.db")
# 압축 이미지 메시지를 jpg 파일로 저장할 폴더입니다.
IMAGE_DIR = Path(__file__).with_name("camera_frames")
# 배터리 상태를 구독할 로봇 이름과 토픽 이름 목록입니다.
BATTERY_TOPICS = [
    ("Robot 2", "/robot2/battery_state"),
    ("Robot 9", "/robot9/battery_state"),
]
# 도킹 상태를 구독할 로봇 이름과 토픽 이름 목록입니다.
DOCK_TOPICS = [
    ("Robot 2", "/robot2/dock_status"),
    ("Robot 9", "/robot9/dock_status"),
]
# 카메라 압축 이미지 토픽과 저장할 파일명을 함께 정의합니다.
CAMERA_TOPICS = [
    ("Robot 2", "/robot2/oakd/rgb/image_raw/compressed", "robot2_rgb.jpg"),
    ("Robot 9", "/robot9/oakd/rgb/image_raw/compressed", "robot9_rgb.jpg"),
    ("Safety Cam 0", "/safety/cam0/image/compressed", "safety_cam0.jpg"),
    ("Safety Cam 1", "/safety/cam1/image/compressed", "safety_cam1.jpg"),
]


def get_connection():
    # SQLite DB에 연결한 connection 객체를 반환합니다.
    return sqlite3.connect(DB_PATH)


def initialize_battery_rows():
    # 웹 화면에서 ROS2 데이터 수신 전에도 로봇 행이 보이도록 기본 행을 준비합니다.
    # 여기서는 실제 배터리 값을 넣지 않고 robot_name/topic_name만 먼저 넣습니다.
    # 실제 percentage, voltage, current 값은 배터리 토픽이 들어올 때 update_battery_status()에서 갱신합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        robot_names = [robot_name for robot_name, _ in BATTERY_TOPICS]
        placeholders = ", ".join("?" for _ in robot_names)
        cursor.execute(
            f"DELETE FROM robot_battery_status "
            f"WHERE robot_name NOT IN ({placeholders})",
            robot_names,
        )
        cursor.executemany(
            """
            INSERT INTO robot_battery_status (
                robot_name,
                topic_name
            )
            VALUES (?, ?)
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name
            """,
            # robot_name이 이미 있으면 중복 INSERT 오류를 내지 않고 topic_name만 최신 설정으로 맞춥니다.
            BATTERY_TOPICS,
        )


def initialize_dock_rows():
    # 도킹 상태 테이블에도 로봇별 기본 행을 미리 만들어 둡니다.
    # 실제 dock_visible, is_docked 값은 도킹 토픽이 들어온 뒤 update_dock_status()에서 저장합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        robot_names = [robot_name for robot_name, _ in DOCK_TOPICS]
        placeholders = ", ".join("?" for _ in robot_names)
        cursor.execute(
            f"DELETE FROM robot_dock_status "
            f"WHERE robot_name NOT IN ({placeholders})",
            robot_names,
        )
        cursor.executemany(
            """
            INSERT INTO robot_dock_status (
                robot_name,
                topic_name
            )
            VALUES (?, ?)
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name
            """,
            # robot_name 기본 키 충돌 시 토픽 이름만 갱신합니다.
            DOCK_TOPICS,
        )


def initialize_camera_rows():
    # 카메라 상태 테이블에 로봇별 토픽과 이미지 파일명을 등록합니다.
    # 실제 카메라 프레임 수신 시간은 이미지 토픽이 들어올 때 update_camera_status()에서 갱신합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        camera_names = [robot_name for robot_name, _, _ in CAMERA_TOPICS]
        placeholders = ", ".join("?" for _ in camera_names)
        # 설정에서 빠진 이전 카메라 행은 화면에 계속 남지 않도록 정리합니다.
        cursor.execute(
            f"DELETE FROM robot_camera_status WHERE robot_name NOT IN ({placeholders})",
            camera_names,
        )
        cursor.executemany(
            """
            INSERT INTO robot_camera_status (
                robot_name,
                topic_name,
                image_filename
            )
            VALUES (?, ?, ?)
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name,
                image_filename = excluded.image_filename
            """,
            # 카메라 파일명까지 함께 저장해야 웹 앱이 이미지를 찾을 수 있습니다.
            CAMERA_TOPICS,
        )


def update_battery_status(robot_name, topic_name, msg):
    # BatteryState 메시지의 잔량, 전압, 전류를 DB의 최신 상태로 저장합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO robot_battery_status (
                robot_name,
                topic_name,
                percentage,
                voltage,
                current,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, datetime('now', '+9 hours'))
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name,
                percentage = excluded.percentage,
                voltage = excluded.voltage,
                current = excluded.current,
                updated_at = datetime('now', '+9 hours')
            """,
            # msg.percentage는 0.0~1.0 범위의 배터리 비율입니다.
            (
                robot_name,
                topic_name,
                msg.percentage,
                msg.voltage,
                msg.current,
            ),
        )


def update_dock_status(robot_name, topic_name, msg):
    # DockStatus 메시지의 불리언 값을 SQLite에 저장하기 위해 0/1 정수로 변환합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO robot_dock_status (
                robot_name,
                topic_name,
                dock_visible,
                is_docked,
                updated_at
            )
            VALUES (?, ?, ?, ?, datetime('now', '+9 hours'))
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name,
                dock_visible = excluded.dock_visible,
                is_docked = excluded.is_docked,
                updated_at = datetime('now', '+9 hours')
            """,
            # SQLite에는 bool 타입이 따로 없으므로 int로 변환해서 저장합니다.
            (
                robot_name,
                topic_name,
                int(msg.dock_visible),
                int(msg.is_docked),
            ),
        )


def update_camera_status(robot_name, topic_name, image_filename, msg):
    # 카메라 프레임 저장 폴더가 없으면 생성합니다.
    IMAGE_DIR.mkdir(exist_ok=True)
    image_path = IMAGE_DIR / image_filename
    temporary_path = image_path.with_suffix(".tmp")
    # Flask가 쓰는 중간 프레임을 읽지 않도록 임시 파일을 완성한 뒤 원자적으로 교체합니다.
    temporary_path.write_bytes(bytes(msg.data))
    temporary_path.replace(image_path)

    # DB에는 실제 이미지 데이터 대신 파일명과 마지막 수신 시간만 기록합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO robot_camera_status (
                robot_name,
                topic_name,
                image_filename,
                updated_at
            )
            VALUES (?, ?, ?, datetime('now', '+9 hours'))
            ON CONFLICT(robot_name) DO UPDATE SET
                topic_name = excluded.topic_name,
                image_filename = excluded.image_filename,
                updated_at = datetime('now', '+9 hours')
            """,
            (robot_name, topic_name, image_filename),
        )


class BatteryDbNode(Node):
    def __init__(self):
        # ROS2 노드 이름을 battery_db_node로 초기화합니다.
        super().__init__("battery_db_node")
        # subscription 객체가 가비지 컬렉션으로 사라지지 않도록 리스트에 보관합니다.
        self.subscriptions_by_topic = []

        # 배터리 토픽마다 콜백을 만들어 구독합니다.
        for robot_name, topic_name in BATTERY_TOPICS:
            subscription = self.create_subscription(
                BatteryState,
                topic_name,
                self.create_battery_callback(robot_name, topic_name),
                qos_profile_sensor_data,
            )
            self.subscriptions_by_topic.append(subscription)
            self.get_logger().info(f"Subscribed: {topic_name}")

        # 도킹 상태 토픽마다 콜백을 만들어 구독합니다.
        for robot_name, topic_name in DOCK_TOPICS:
            subscription = self.create_subscription(
                DockStatus,
                topic_name,
                self.create_dock_callback(robot_name, topic_name),
                qos_profile_sensor_data,
            )
            self.subscriptions_by_topic.append(subscription)
            self.get_logger().info(f"Subscribed: {topic_name}")

        # 카메라 이미지 토픽마다 콜백을 만들어 구독합니다.
        for robot_name, topic_name, image_filename in CAMERA_TOPICS:
            subscription = self.create_subscription(
                CompressedImage,
                topic_name,
                self.create_camera_callback(robot_name, topic_name, image_filename),
                qos_profile_sensor_data,
            )
            self.subscriptions_by_topic.append(subscription)
            self.get_logger().info(f"Subscribed: {topic_name}")

    def create_battery_callback(self, robot_name, topic_name):
        # 반복문 안에서 robot_name/topic_name 값을 고정하기 위해 콜백 생성 함수를 사용합니다.
        def callback(msg):
            try:
                # 메시지를 받을 때마다 DB의 배터리 상태를 갱신합니다.
                update_battery_status(robot_name, topic_name, msg)
                self.get_logger().info(f"{topic_name} updated")
            except sqlite3.Error as error:
                # DB 오류가 나도 노드가 바로 종료되지 않도록 로그만 남깁니다.
                self.get_logger().error(f"{topic_name} DB update failed: {error}")

        return callback

    def create_dock_callback(self, robot_name, topic_name):
        # 도킹 상태 토픽용 콜백을 생성합니다.
        def callback(msg):
            try:
                # 메시지를 받을 때마다 DB의 도킹 상태를 갱신합니다.
                update_dock_status(robot_name, topic_name, msg)
                self.get_logger().info(f"{topic_name} updated")
            except sqlite3.Error as error:
                # DB 오류가 나도 노드가 계속 동작하도록 로그만 남깁니다.
                self.get_logger().error(f"{topic_name} DB update failed: {error}")

        return callback

    def create_camera_callback(self, robot_name, topic_name, image_filename):
        # 카메라 토픽용 콜백을 생성합니다.
        def callback(msg):
            try:
                # 프레임 파일 저장과 DB 상태 갱신을 함께 수행합니다.
                update_camera_status(robot_name, topic_name, image_filename, msg)
                self.get_logger().info(f"{topic_name} updated")
            except (OSError, sqlite3.Error) as error:
                # 이미지 파일 또는 DB 오류는 ROS2 로그로 확인할 수 있게 남깁니다.
                self.get_logger().error(f"{topic_name} camera update failed: {error}")

        return callback


def main(args=None):
    # 노드 시작 전에 각 상태 테이블에 기본 행을 준비합니다.
    initialize_battery_rows()
    initialize_dock_rows()
    initialize_camera_rows()

    # ROS2 클라이언트 라이브러리를 초기화하고 노드를 생성합니다.
    rclpy.init(args=args)
    node = BatteryDbNode()

    try:
        # 콜백이 계속 실행될 수 있도록 노드를 spin 상태로 유지합니다.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 종료는 정상 종료 흐름으로 처리합니다.
        pass
    finally:
        # 종료 시 노드와 ROS2 리소스를 정리합니다.
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
