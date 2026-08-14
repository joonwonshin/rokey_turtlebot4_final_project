# 3_db_update.py
# Aruco marker id를 받아 소화기 DB 점검 결과를 업데이트하는 ROS2 노드

from datetime import date
from pathlib import Path
import sqlite3
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray


# 현재 파이썬 파일과 같은 폴더의 DB 파일을 사용합니다.
DB_PATH = Path(__file__).with_name("fire_db.db")
IMAGE_DIR = Path(__file__).with_name("camera_frames")
# 로봇별 Aruco marker id 구독 토픽과 점검 완료 publish 토픽입니다.
ROBOT_ARUCO_TOPICS = {
    "robot2": ("/robot2/aruco/detection/ids", "/robot2/aruco_check_done"),
    "robot9": ("/robot9/aruco/detection/ids", "/robot9/aruco_check_done"),
}
# 같은 마커가 카메라에 계속 보일 때 DB를 너무 자주 쓰지 않도록 제한하는 시간입니다.
MARKER_UPDATE_COOLDOWN_SECONDS = 1.0
# 점검 결과로 사용할 문자열 상수입니다.
PASS_RESULT = "PASS"
FAIL_RESULT = "FAIL"
# 제조일 기준 내용연수입니다. 10년이 지나면 FAIL로 판정합니다.
LIFETIME_YEARS = 10


def get_connection():
    # SQLite DB에 연결한 connection 객체를 반환합니다.
    return sqlite3.connect(DB_PATH)


def get_camera_robot_name(robot_name):
    # Aruco 네임스페이스 robot2를 카메라 DB의 Robot 2 형식으로 변환합니다.
    suffix = robot_name.removeprefix("robot")
    return f"Robot {suffix}" if suffix else robot_name


def capture_inspection_snapshot(connection, marker_id, robot_name):
    # 점검 시점의 해당 로봇 최신 프레임을 별도 파일로 보존합니다.
    row = connection.execute(
        """
        SELECT image_filename
        FROM robot_camera_status
        WHERE robot_name = ?
        """,
        (get_camera_robot_name(robot_name),),
    ).fetchone()
    if row is None or not row[0]:
        return None

    source_path = IMAGE_DIR / row[0]
    if not source_path.is_file():
        return None

    snapshot_directory = IMAGE_DIR / "snapshots"
    snapshot_name = f"snapshots/{robot_name}_marker{marker_id}_{time.time_ns()}.jpg"
    snapshot_path = IMAGE_DIR / snapshot_name
    try:
        snapshot_directory.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(source_path.read_bytes())
    except OSError:
        return None
    return snapshot_name


def get_fire_extinguisher_by_marker_id(marker_id):
    # marker_id로 소화기 한 개의 현재 DB 정보를 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                marker_id,
                location_x,
                location_y,
                manufacture_date,
                last_inspection_date,
                pressure_status,
                result
            FROM fire_extinguisher
            WHERE marker_id = ?
            """,
            (marker_id,),
        )
        return cursor.fetchone()


def get_lifetime_limit(today=None):
    # 테스트에서 날짜를 고정할 수 있도록 today를 선택적으로 받습니다.
    if today is None:
        today = date.today()

    try:
        # 오늘 날짜에서 내용연수만큼 뺀 날짜가 사용 가능 한계일입니다.
        return today.replace(year=today.year - LIFETIME_YEARS)
    except ValueError:
        # 2월 29일은 평년에는 존재하지 않으므로 2월 28일로 보정합니다.
        return today.replace(year=today.year - LIFETIME_YEARS, day=28)


def is_expired(manufacture_date, today=None):
    # ISO 형식(YYYY-MM-DD)의 제조일을 date 객체로 바꿔 만료 여부를 판단합니다.
    manufactured_on = date.fromisoformat(manufacture_date)
    return manufactured_on < get_lifetime_limit(today)


def judge_inspection_result(extinguisher, today=None):
    # DB 조회 결과 튜플에서 제조일과 압력 상태 컬럼을 꺼냅니다.
    manufacture_date = extinguisher[3]
    pressure_status = extinguisher[5]

    # 내용연수가 지났으면 압력 상태와 관계없이 FAIL입니다.
    if is_expired(manufacture_date, today):
        return FAIL_RESULT

    # 압력이 normal이 아니면 점검 실패로 처리합니다.
    if pressure_status != "normal":
        return FAIL_RESULT

    # 만료되지 않았고 압력이 정상인 경우에만 PASS입니다.
    return PASS_RESULT


def update_inspection_result(marker_id, result, robot_name=None):
    # 의도하지 않은 문자열이 DB에 들어가지 않도록 허용값을 제한합니다.
    if result not in {PASS_RESULT, FAIL_RESULT}:
        raise ValueError("result는 'PASS' 또는 'FAIL'만 사용할 수 있습니다.")

    # 점검 결과와 마지막 점검 시간을 함께 갱신합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE fire_extinguisher
            SET
                result = ?,
                last_inspection_date = datetime('now', '+9 hours')
            WHERE marker_id = ?
            """,
            (result, marker_id),
        )
        updated = cursor.rowcount == 1

        if updated and robot_name is not None:
            # 현재 상태를 갱신한 트랜잭션에서 점검 이력도 함께 남깁니다.
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fire_extinguisher_inspection_history(
                inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                marker_id INTEGER NOT NULL,
                robot_name TEXT NOT NULL,
                result TEXT NOT NULL,
                snapshot_filename TEXT,
                inspected_at TEXT NOT NULL DEFAULT (datetime('now', '+9 hours'))
            )
            """)
            history_columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(fire_extinguisher_inspection_history)"
                )
            }
            if "snapshot_filename" not in history_columns:
                cursor.execute(
                    "ALTER TABLE fire_extinguisher_inspection_history "
                    "ADD COLUMN snapshot_filename TEXT"
                )
            snapshot_filename = capture_inspection_snapshot(
                connection, marker_id, robot_name
            )
            cursor.execute(
                """
                INSERT INTO fire_extinguisher_inspection_history (
                    marker_id,
                    robot_name,
                    result,
                    snapshot_filename
                )
                VALUES (?, ?, ?, ?)
                """,
                (marker_id, robot_name, result, snapshot_filename),
            )
            # 최근 50건만 남기고 더 오래된 점검 이력과 사진은 정리합니다.
            stale_snapshots = cursor.execute(
                """
                SELECT snapshot_filename
                FROM fire_extinguisher_inspection_history
                ORDER BY inspection_id DESC
                LIMIT -1 OFFSET 50
                """
            ).fetchall()
            cursor.execute("""
            DELETE FROM fire_extinguisher_inspection_history
            WHERE inspection_id NOT IN (
                SELECT inspection_id
                FROM fire_extinguisher_inspection_history
                ORDER BY inspection_id DESC
                LIMIT 50
            )
            """)
            for (stale_snapshot,) in stale_snapshots:
                if stale_snapshot:
                    (IMAGE_DIR / stale_snapshot).unlink(missing_ok=True)

        return updated


def inspect_marker(marker_id, robot_name=None):
    # marker_id에 해당하는 소화기가 없으면 None을 반환합니다.
    extinguisher = get_fire_extinguisher_by_marker_id(marker_id)
    if extinguisher is None:
        return None

    # 현재 DB 값을 기준으로 PASS/FAIL을 판정한 뒤 DB에 저장합니다.
    result = judge_inspection_result(extinguisher)
    update_inspection_result(marker_id, result, robot_name)
    # 갱신된 최신 행을 다시 조회해서 반환합니다.
    return get_fire_extinguisher_by_marker_id(marker_id)


class FireExtinguisherInspectionNode(Node):
    def __init__(self):
        # Aruco marker id를 받아 DB를 업데이트하는 ROS2 노드입니다.
        super().__init__("fire_extinguisher_inspection_node")
        # 로봇와 marker_id별 마지막 처리 시간을 저장합니다.
        self.last_update_time_by_robot_and_marker_id = {}
        self.marker_id_subscriptions = []
        self.check_done_publishers = {}

        for robot_name, (marker_topic, check_done_topic) in ROBOT_ARUCO_TOPICS.items():
            subscription = self.create_subscription(
                Int32MultiArray,
                marker_topic,
                lambda msg, name=robot_name: self.handle_marker_ids(msg, name),
                qos_profile_sensor_data,
            )
            self.marker_id_subscriptions.append(subscription)
            self.check_done_publishers[robot_name] = self.create_publisher(
                Bool, check_done_topic, 10
            )
            self.get_logger().info(f"Subscribed: {marker_topic}")
            self.get_logger().info(f"Publish check done: {check_done_topic}")

    def should_skip_marker(self, robot_name, marker_id):
        # 같은 로봇에서 같은 marker_id가 짧은 시간 안에 반복되면 건너뜁니다.
        now = time.monotonic()
        marker_key = (robot_name, marker_id)
        last_update_time = self.last_update_time_by_robot_and_marker_id.get(marker_key)

        if (
            last_update_time is not None
            and now - last_update_time < MARKER_UPDATE_COOLDOWN_SECONDS
        ):
            return True

        self.last_update_time_by_robot_and_marker_id[marker_key] = now
        return False

    def handle_marker_ids(self, msg, robot_name):
        # std_msgs/Int32MultiArray 메시지의 data 필드가 Aruco marker id 목록입니다.
        for marker_id in msg.data:
            self.handle_marker_id(int(marker_id), robot_name)

    def handle_marker_id(self, marker_id, robot_name):
        # 인식 실패를 -1 같은 음수 값으로 publish하는 경우는 DB 조회 없이 무시합니다.
        if marker_id < 0:
            return

        if self.should_skip_marker(robot_name, marker_id):
            return

        try:
            updated_extinguisher = inspect_marker(marker_id, robot_name)
        except (sqlite3.Error, ValueError) as error:
            # DB 오류나 날짜 형식 오류가 발생해도 노드는 계속 동작하게 로그만 남깁니다.
            self.get_logger().error(
                f"marker_id {marker_id} inspection update failed: {error}"
            )
            return

        if updated_extinguisher is None:
            # DB에 등록되지 않은 marker id가 들어오면 경고만 남깁니다.
            self.get_logger().warn(f"marker_id {marker_id} not found in DB")
            return

        # 업데이트된 행에서 점검 결과와 마지막 점검 시간을 꺼내 로그로 확인합니다.
        result = updated_extinguisher[6]
        last_inspection_date = updated_extinguisher[4]
        self.get_logger().info(
            f"marker_id {marker_id} updated: result={result}, "
            f"last_inspection_date={last_inspection_date}"
        )
        self.publish_check_done(robot_name)

    def publish_check_done(self, robot_name):
        # DB 업데이트를 요청한 로봇에게만 완료 신호를 보냅니다.
        message = Bool()
        message.data = True
        self.check_done_publishers[robot_name].publish(message)


def main(args=None):
    # ROS2를 초기화하고 Aruco marker id 구독 노드를 실행합니다.
    rclpy.init(args=args)
    node = FireExtinguisherInspectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
