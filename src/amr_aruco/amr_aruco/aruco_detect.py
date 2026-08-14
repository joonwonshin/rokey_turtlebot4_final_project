#!/usr/bin/env python3
"""
TurtleBot4 OAK-D 카메라 (CompressedImage 토픽) 아루코 마커 인식 노드
ROS2 Humble / Ubuntu 22.04 / TurtleBot4 환경 기준

토픽은 모두 노드 네임스페이스 기준 상대 이름이다. launch 파일의 namespace
인자(기본 robot2)나 __ns 리맵으로 네임스페이스가 정해지며, 리맵 없이
ros2 run 으로 단독 실행하면 기본 네임스페이스(robot2)가 적용된다
(common.default_namespace_args 참고 - 기존 input() 방식 대체).

구독:
  - oakd/rgb/image_raw/compressed        (aruco_scan_enable=True 인 동안에만 구독을 생성 -
                                         주행 중에는 카메라 스트림 자체를 끊어 CPU/WiFi 부하를 줄인다)
  - aruco_scan_enable                    (Bool, amr_patrol_emer_helmet 이 소화기 포인트에서
                                         대기하는 동안에만 True 로 발행 - 그 외 순찰 중 이동 구간에서는
                                         마커가 잡혀도 검출/발행을 하지 않기 위한 게이트)
발행:
  - aruco/detection/compressed           (마커 표시된 디버그 영상, CompressedImage)
  - aruco/detection/ids                  (탐지된 모든 마커 ID 목록, Int32MultiArray.
                                         aruco_scan_enable 이 True 인 동안에만 발행됨)
  - aruco/detection/marker               (마커 1개씩, ArucoMarker 커스텀 msg 없이도 쓰도록
                                         id + 픽셀 중심좌표를 PointStamped로 발행,
                                         point.z 에 marker_id를 담아 전달 - 현재 주석 처리됨)
"""

import signal

import cv2

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32MultiArray
# from geometry_msgs.msg import PointStamped  # marker 토픽
from cv_bridge import CvBridge

from amr_aruco import aruco_vision
from amr_aruco.common import (
    ARUCO_SCAN_ENABLE_TOPIC,
    COMMAND_QOS,
    default_namespace_args,
)


class ArucoMarkerDetector(Node):
    def __init__(self):
        super().__init__('aruco_marker_detector')
        self.bridge = CvBridge()

        # === ROS 파라미터 (기본값/의미/제약은 config/amr_aruco_params.yaml 참고) ===
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('show_window', True)
        aruco_dict_name = self.get_parameter('aruco_dict').value
        self.show_window = self.get_parameter('show_window').value

        # amr_patrol_emer_helmet 이 소화기 포인트에서 대기를 시작할 때만 True 로
        # 켜주는 게이트. 이동 중에는 마커가 화면에 잡혀도 검출/발행하지 않는다.
        self.scan_enabled = False

        self.detector = aruco_vision.create_detector(aruco_dict_name)
        self.get_logger().info(f'ArUco dictionary: {aruco_dict_name}')

        # 카메라 구독은 scan_enabled 일 때만 생성한다. 구독이 살아 있으면
        # 검출을 안 해도 OAK-D 압축 영상 스트림이 계속 들어와 매 프레임
        # JPEG 디코드/재인코드/imshow 로 CPU 와 WiFi 대역폭을 잡아먹고,
        # Nav2/AMCL 이 밀리면 map->odom 이 갱신을 멈춰
        # "Transform data too old" 로 goal 이 abort 될 수 있다.
        # 마커 확인 구간은 로봇이 정지해 있으므로 그때만 스트림을 열어도 충분하다.
        self.camera_topic = 'oakd/rgb/image_raw/compressed'
        self.subscription = None

        self.debug_publisher = self.create_publisher(
            CompressedImage,
            'aruco/detection/compressed',
            10)

        self.ids_publisher = self.create_publisher(
            Int32MultiArray,
            'aruco/detection/ids',
            10)

        # amr_patrol_emer_helmet 쪽 발행과 QoS(latched)가 동일해야 한다
        # (common.COMMAND_QOS 주석 참고).
        self.scan_enable_subscription = self.create_subscription(
            Bool,
            ARUCO_SCAN_ENABLE_TOPIC,
            self._scan_enable_callback,
            COMMAND_QOS)

        # marker 토픽
        # self.marker_publisher = self.create_publisher(
        #     PointStamped,
        #     '/aruco/detection/marker',
        #     10)

    def _scan_enable_callback(self, msg):
        self.scan_enabled = msg.data
        self.get_logger().info(f"아루코 스캔 {'시작' if msg.data else '중지'}")

        if msg.data and self.subscription is None:
            self.get_logger().info(f"Subscribing to: {self.camera_topic}")
            self.subscription = self.create_subscription(
                CompressedImage,
                self.camera_topic,
                self.listener_callback,
                qos_profile_sensor_data)  # (BEST EFFORT, 5, VOLATILE)
        elif not msg.data and self.subscription is not None:
            # 구독을 끊어 주행 구간에서는 카메라 스트림이 아예 흐르지 않게 한다.
            self.destroy_subscription(self.subscription)
            self.subscription = None
            if self.show_window:
                cv2.destroyAllWindows()

    def listener_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        stamp = self.get_clock().now().to_msg()

        # 소화기 포인트에서 대기 중일 때(scan_enabled=True)만 검출/발행한다.
        # 순찰 중 이동하며 우연히 마커가 잡히는 것까지 발행하면 오탐/불필요한
        # 트래픽이 생기므로, 그 외 구간에서는 아예 검출도 수행하지 않는다.
        if self.scan_enabled:
            # ArUco 탐지는 흑백 패턴만 보면 되므로 그레이스케일로 변환 (컬러 정보 불필요)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = aruco_vision.detect_markers(self.detector, gray)

            if ids is not None and len(ids) > 0:
                # 아이디 1차원으로
                ids_flat = [int(i) for i in ids.flatten()]

                aruco_vision.draw_markers(img, corners, ids)

                # marker 토픽(좌표 발행) - 재활성화 시 numpy(np) import 필요
                # for marker_corners, marker_id in zip(corners, ids_flat):
                #     pts = marker_corners.reshape(4, 2)
                #     cx = float(np.mean(pts[:, 0]))
                #     cy = float(np.mean(pts[:, 1]))
                #
                #     self.get_logger().info(
                #         f"Marker detected: id={marker_id}, center=({cx:.1f}, {cy:.1f})")
                #
                #     marker_msg = PointStamped()
                #     marker_msg.header.stamp = stamp
                #     marker_msg.point.x = cx
                #     marker_msg.point.y = cy
                #     marker_msg.point.z = float(marker_id)  # id를 z에 실어 전달
                #     self.marker_publisher.publish(marker_msg)

                ids_msg = Int32MultiArray()  # 아이디 여러개
                ids_msg.data = ids_flat
                self.ids_publisher.publish(ids_msg)
            else:
                # 탐지된 마커가 없으면 빈 배열 발행 (구독 측에서 "없음"을 알 수 있도록)
                self.ids_publisher.publish(Int32MultiArray(data=[]))

        try:
            out_msg = self.bridge.cv2_to_compressed_imgmsg(img, dst_format='jpg')
            out_msg.header.stamp = stamp
            self.debug_publisher.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish failed: {e}")

        if self.show_window:
            cv2.imshow("Aruco Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via 'q'")
                rclpy.shutdown()


def main(args=None):
    # __ns 리맵 없이 단독 실행하면 기본 네임스페이스(robot2)로 리맵해 기존
    # 동작(항상 robot2 토픽 사용)을 유지한다. launch 파일 실행 시에는
    # namespace 런치 인자가 우선한다.
    rclpy.init(args=default_namespace_args(args))
    node = ArucoMarkerDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # ros2 launch 의 SIGINT 종료 시 트레이스백 없이 조용히 내려가도록 처리
        pass
    finally:
        # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고, ros2 launch 가
        # 자식에게 SIGINT 를 한 번 더 전달하므로 정리 도중 두 번째
        # KeyboardInterrupt 가 터질 수 있다. 이후 SIGINT 는 무시하고 정리한다.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
