#!/usr/bin/env python3
"""
TurtleBot4 OAK-D 카메라 (CompressedImage 토픽) 아루코 마커 인식 노드
ROS2 Humble / Ubuntu 22.04 / TurtleBot4 환경 기준

구독: /{robot_namespace}/oakd/rgb/image_raw/compressed
발행:
  - /aruco/detection/compressed        (마커 표시된 디버그 영상, CompressedImage)
  - /aruco/detection/ids               (탐지된 모든 마커 ID 목록, Int32MultiArray)
  - /aruco/detection/marker            (마커 1개씩, ArucoMarker 커스텀 msg 없이도 쓰도록
                                         id + 픽셀 중심좌표를 PointStamped로 발행,
                                         point.z 에 marker_id를 담아 전달)
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


# 사용할 아루코 딕셔너리 (필요에 맞게 변경: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250 등)
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50


class ArucoMarkerDetector(Node):
    def __init__(self, robot_namespace='robot9'):
        super().__init__('aruco_marker_detector')
        self.bridge = CvBridge()

        # OpenCV 4.9.0 기준 ArUco API (ArucoDetector 클래스 방식)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        topic_name = f'/{robot_namespace}/oakd/rgb/image_raw/compressed'
        self.get_logger().info(f"Subscribing to: {topic_name}")
        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            qos_profile_sensor_data)

        self.debug_publisher = self.create_publisher(
            CompressedImage,
            '/aruco/detection/compressed',
            10)

        self.ids_publisher = self.create_publisher(
            Int32MultiArray,
            '/aruco/detection/ids',
            10)

        self.marker_publisher = self.create_publisher(
            PointStamped,
            '/aruco/detection/marker',
            10)

    def detect_markers(self, gray):
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids

    def listener_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_markers(gray)

        stamp = self.get_clock().now().to_msg()

        if ids is not None and len(ids) > 0:
            ids_flat = [int(i) for i in ids.flatten()]

            # 디버그 영상에 마커 테두리/ID 그리기
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            for marker_corners, marker_id in zip(corners, ids_flat):
                pts = marker_corners.reshape(4, 2)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))

                self.get_logger().info(f"Marker detected: id={marker_id}, center=({cx:.1f}, {cy:.1f})")

                marker_msg = PointStamped()
                marker_msg.header.stamp = stamp
                marker_msg.point.x = cx
                marker_msg.point.y = cy
                marker_msg.point.z = float(marker_id)  # id를 z에 실어 전달
                self.marker_publisher.publish(marker_msg)

            ids_msg = Int32MultiArray()
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

        cv2.imshow("Aruco Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Shutdown requested via 'q'")
            rclpy.shutdown()


def main():
    robot_namespace = input("Enter robot namespace (e.g. robot2): ").strip() or 'robot2'

    rclpy.init()
    node = ArucoMarkerDetector(robot_namespace)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()