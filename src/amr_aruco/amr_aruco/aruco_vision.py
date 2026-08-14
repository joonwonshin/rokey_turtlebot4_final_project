"""OpenCV ArUco 검출/표시 유틸 (ROS 의존성 없음 - 단독 테스트 가능)."""
import cv2


def create_detector(dict_name: str):
    """딕셔너리 이름 문자열로 ArucoDetector 를 만든다.

    dict_name: 사용할 아루코 딕셔너리 이름
    (필요에 맞게 변경: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250 등)
    """
    # OpenCV 4.9.0 기준 ArUco API (ArucoDetector 클래스 방식)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def detect_markers(detector, gray):
    # corners: 인식된 마커들의 4개 코너 좌표, ids: 각 마커의 ID
    # rejected: 사각형 후보였지만 유효한 마커로 디코딩되지 못해 걸러진 것들
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids


def draw_markers(img, corners, ids):
    """디버그 영상에 마커 테두리/ID 그리기 (img 를 제자리에서 수정)."""
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    return img
