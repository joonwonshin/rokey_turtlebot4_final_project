"""
캘리브레이션용 참조 사진 촬영.

02_make_homography_pairwise.py 는 실시간 영상이 아니라 captures/{cam}_ref.jpg
정지 사진 위에서 클릭을 받는다. 그래서 H 를 다시 잡기 전에 반드시 지금 카메라가
보고 있는 장면을 새로 찍어야 한다.

  python3 00_capture_ref.py cam0 --device 2
  python3 00_capture_ref.py cam1 --device 0

장치 번호는 USB 재열거로 바뀐다. 실행 전에 확인할 것:
  v4l2-ctl --list-devices

키:
  s : 저장 후 종료
  q : 저장 없이 종료
"""
import argparse
import sys
from pathlib import Path

import cv2

BASE = Path.home() / "turtlebot4_ws" / "final_project"

ap = argparse.ArgumentParser()
ap.add_argument("camera", choices=["cam0", "cam1"])
ap.add_argument("--device", type=int, required=True, help="/dev/videoN 의 N")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
cli = ap.parse_args()

OUT = BASE / "captures" / f"{cli.camera}_ref.jpg"

cap = cv2.VideoCapture(cli.device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cli.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cli.height)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    sys.exit(f"[ERROR] /dev/video{cli.device} 를 열 수 없음. "
             f"다른 프로그램이 잡고 있는지 확인: fuser /dev/video{cli.device}")

print("=" * 60)
print(f"{cli.camera}  <-  /dev/video{cli.device}")
print(f"out: {OUT}")
print("-" * 60)
print("바닥 기준점(벽/박스가 바닥에 닿는 코너)이 화면에 골고루 보이는지 확인")
print("  s: 저장   q: 취소")
print("=" * 60)

win = f"{cli.camera} ref capture  (/dev/video{cli.device})"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] frame read failed")
        continue

    show = frame.copy()
    h, w = show.shape[:2]

    # 3x3 격자 — 클릭점을 화면 전체에 퍼뜨리는 데 참고
    for i in (1, 2):
        cv2.line(show, (w * i // 3, 0), (w * i // 3, h), (60, 60, 60), 1)
        cv2.line(show, (0, h * i // 3), (w, h * i // 3), (60, 60, 60), 1)

    txt = f"{cli.camera}  /dev/video{cli.device}  {w}x{h}   s:save  q:quit"
    cv2.putText(show, txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(show, txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(win, show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[CANCEL] 저장하지 않음")
        break

    if key == ord("s"):
        if (w, h) != (cli.width, cli.height):
            print(f"[ERROR] {w}x{h} 로 열렸다. 캘리브레이션은 "
                  f"{cli.width}x{cli.height} 기준이라 좌표가 틀어진다. 저장 안 함.")
            break
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT), frame)
        print(f"[SAVED] {OUT}  ({w}x{h})")
        break

cap.release()
cv2.destroyAllWindows()
