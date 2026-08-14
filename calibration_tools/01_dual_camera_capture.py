import cv2
from datetime import datetime
from pathlib import Path

# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE_DIR = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

cam0_id = 2
cam1_id = 4

cap0 = cv2.VideoCapture(cam0_id, cv2.CAP_V4L2)
cap1 = cv2.VideoCapture(cam1_id, cv2.CAP_V4L2)

for cap in [cap0, cap1]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

print("====================================")
print("Dual Camera Capture")
print("CAM0 = /dev/video2")
print("CAM1 = /dev/video4")
print("------------------------------------")
print("s : save cam0_ref.jpg, cam1_ref.jpg")
print("q : quit")
print("====================================")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        show0 = frame0.copy()
        cv2.putText(show0, "CAM0 / video2", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("CAM0", show0)

    if ret1:
        show1 = frame1.copy()
        cv2.putText(show1, "CAM1 / video4", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("CAM1", show1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ret0:
            cv2.imwrite(str(CAPTURE_DIR / "cam0_ref.jpg"), frame0)
            cv2.imwrite(str(CAPTURE_DIR / f"cam0_ref_{now}.jpg"), frame0)
            print("[SAVE] cam0_ref.jpg")

        if ret1:
            cv2.imwrite(str(CAPTURE_DIR / "cam1_ref.jpg"), frame1)
            cv2.imwrite(str(CAPTURE_DIR / f"cam1_ref_{now}.jpg"), frame1)
            print("[SAVE] cam1_ref.jpg")

    elif key == ord("q"):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
