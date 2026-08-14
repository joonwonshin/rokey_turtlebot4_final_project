import cv2
import argparse
import time
import os
import re
import shutil
from pathlib import Path


BASE_DIR = Path.home() / "turtlebot4_ws" / "final_project"
DATASET_DIR = BASE_DIR / "dataset" / "raw_auto"

CAMERA_IDS = {
    "cam0": 2,
    "cam1": 4,
}


class AutoCapture:
    def __init__(self, cam_name, interval, target, clear=False):
        self.cam_name = cam_name
        self.cam_id = CAMERA_IDS[cam_name]
        self.interval = interval
        self.target = target

        self.save_dir = DATASET_DIR / cam_name

        if clear and self.save_dir.exists():
            print(f"[CLEAR] removing old images in {self.save_dir}")
            shutil.rmtree(self.save_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.window_name = f"{cam_name} auto capture"

        self.cap = self.open_camera(self.cam_id)

        self.auto_save = False
        self.last_save_time = 0.0
        self.latest_frame = None

        self.capture_start_time = None
        self.capture_end_time = None

        # 현재 실행 중 저장한 파일만 u로 삭제
        self.history = []

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def open_camera(self, cam_id):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera id {cam_id}")

        return cap

    def current_count(self):
        return len(list(self.save_dir.glob("*.jpg")))

    def get_next_index(self):
        pattern = re.compile(rf"{self.cam_name}_free_(\d+)\.jpg$")

        max_idx = 0
        for path in self.save_dir.glob("*.jpg"):
            m = pattern.search(path.name)
            if m:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)

        return max_idx + 1

    def next_file_path(self):
        idx = self.get_next_index()
        filename = f"{self.cam_name}_free_{idx:06d}.jpg"
        return self.save_dir / filename

    def format_time(self, seconds):
        seconds = int(max(0, seconds))
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"

    def save_frame(self):
        if self.latest_frame is None:
            return

        count_before = self.current_count()

        if count_before >= self.target:
            self.auto_save = False
            if self.capture_end_time is None:
                self.capture_end_time = time.time()
            print("[TARGET REACHED] Auto save stopped.")
            return

        path = self.next_file_path()
        cv2.imwrite(str(path), self.latest_frame)
        self.history.append(path)

        count = self.current_count()
        print(f"[SAVE] {path}  ({count}/{self.target})")

        if count >= self.target:
            self.auto_save = False
            self.capture_end_time = time.time()
            print("======================================")
            print(f"[TARGET REACHED] {self.cam_name}: {count}/{self.target}")
            print("Auto save stopped. Press q to quit.")
            print("======================================")

    def undo_last(self):
        if not self.history:
            print("[UNDO] No image saved in this session.")
            return

        path = self.history.pop()

        if path.exists():
            os.remove(path)
            print(f"[DELETE] {path}  ({self.current_count()}/{self.target})")
        else:
            print(f"[WARN] File already missing: {path}")

        if self.current_count() < self.target:
            self.capture_end_time = None

    def get_elapsed_seconds(self):
        if self.capture_start_time is None:
            return 0.0

        if self.capture_end_time is not None:
            return self.capture_end_time - self.capture_start_time

        return time.time() - self.capture_start_time

    def get_timing_info(self):
        count = self.current_count()
        elapsed = self.get_elapsed_seconds()

        expected_total = self.target * self.interval
        remaining_by_interval = max(0, self.target - count) * self.interval

        if count > 0 and elapsed > 0:
            actual_sec_per_image = elapsed / count
            remaining_by_actual = max(0, self.target - count) * actual_sec_per_image
        else:
            actual_sec_per_image = self.interval
            remaining_by_actual = remaining_by_interval

        return {
            "elapsed": elapsed,
            "expected_total": expected_total,
            "remaining_by_interval": remaining_by_interval,
            "remaining_by_actual": remaining_by_actual,
            "actual_sec_per_image": actual_sec_per_image,
        }

    def draw_overlay(self, frame):
        show = frame.copy()
        count = self.current_count()
        timing = self.get_timing_info()

        status = "AUTO ON" if self.auto_save else "AUTO OFF"
        status_color = (0, 255, 0) if self.auto_save else (0, 0, 255)

        elapsed_sec = int(timing["elapsed"])
        elapsed_str = self.format_time(timing["elapsed"])
        expected_total_str = self.format_time(timing["expected_total"])
        eta_interval_str = self.format_time(timing["remaining_by_interval"])
        eta_actual_str = self.format_time(timing["remaining_by_actual"])

        h, w = show.shape[:2]

        # 오른쪽 하단 작은 정보 패널
        panel_w = 540
        panel_h = 200
        margin = 18

        x1 = w - panel_w - margin
        y1 = h - panel_h - margin
        x2 = w - margin
        y2 = h - margin

        # 반투명 검정 패널
        overlay = show.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, show, 0.45, 0, show)

        # 테두리
        cv2.rectangle(show, (x1, y1), (x2, y2), (180, 180, 180), 1)

        lines = [
            f"{self.cam_name.upper()} / video{self.cam_id}",
            f"{status} | {count}/{self.target} | interval {self.interval:.2f}s",
            f"Elapsed: {elapsed_str} ({elapsed_sec}s)",
            f"ETA: {eta_interval_str} | Actual ETA: {eta_actual_str}",
            f"Expected total: {expected_total_str}",
            "SPACE start/pause | u undo | s save | q quit",
        ]

        tx = x1 + 15
        ty = y1 + 28

        for i, line in enumerate(lines):
            if i == 0:
                color = (0, 255, 255)
                scale = 0.72
                thickness = 2
            elif i == 1:
                color = status_color
                scale = 0.72
                thickness = 2
            elif i == 5:
                color = (255, 255, 255)
                scale = 0.54
                thickness = 1
            else:
                color = (255, 255, 255)
                scale = 0.60
                thickness = 1

            cv2.putText(
                show,
                line,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            ty += 27

        # 작은 progress bar
        bar_x = x1 + 15
        bar_y = y2 - 28
        bar_w = 330
        bar_h = 16

        ratio = min(count / self.target, 1.0) if self.target > 0 else 0.0

        if count >= self.target:
            bar_color = (0, 255, 0)
        elif ratio >= 0.7:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 120, 255)

        cv2.rectangle(
            show,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (180, 180, 180),
            1,
        )

        fill_w = int(bar_w * ratio)
        cv2.rectangle(
            show,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            bar_color,
            -1,
        )

        cv2.putText(
            show,
            f"{count}/{self.target}",
            (bar_x + bar_w + 12, bar_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bar_color,
            2,
            cv2.LINE_AA,
        )

        if count >= self.target:
            cv2.putText(
                show,
                "TARGET REACHED",
                (x1 + 15, y2 - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return show

    def run(self):
        print("======================================")
        print("Auto Single Camera Dataset Capture")
        print("--------------------------------------")
        print(f"Camera   : {self.cam_name}")
        print(f"Device   : /dev/video{self.cam_id}")
        print(f"Save dir : {self.save_dir}")
        print(f"Interval : {self.interval} sec")
        print(f"Target   : {self.target} images")
        print(f"Expected : {self.format_time(self.target * self.interval)}")
        print("--------------------------------------")
        print("SPACE : start/pause auto capture")
        print("u     : undo/delete last saved image")
        print("s     : save one image manually")
        print("q     : quit")
        print("======================================")
        print("")
        print("촬영 팁:")
        print("- 헬멧 쓰기/벗기 반복")
        print("- 서기/걷기/쪼그리기/숙이기/눕기 섞기")
        print("- 중앙/끝쪽/박스 근처/벽 근처 이동")
        print("- 카메라는 절대 건드리지 말기")
        print("======================================")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[WARN] frame read failed")
                continue

            self.latest_frame = frame.copy()

            now = time.time()

            if self.auto_save and (now - self.last_save_time >= self.interval):
                self.save_frame()
                self.last_save_time = now

            show = self.draw_overlay(frame)
            cv2.imshow(self.window_name, show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if self.current_count() >= self.target:
                    print("[INFO] Target already reached. Increase --target if you want more.")
                    self.auto_save = False
                else:
                    self.auto_save = not self.auto_save
                    self.last_save_time = 0.0

                    if self.auto_save:
                        if self.capture_start_time is None:
                            self.capture_start_time = time.time()
                        self.capture_end_time = None

                    print(f"[AUTO] {'ON' if self.auto_save else 'OFF'}")

            elif key == ord("u"):
                self.undo_last()

            elif key == ord("s"):
                if self.capture_start_time is None:
                    self.capture_start_time = time.time()
                self.save_frame()

            elif key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        timing = self.get_timing_info()

        print("")
        print("======================================")
        print(f"[SUMMARY] {self.cam_name}")
        print(f"Saved        : {self.current_count()}/{self.target}")
        print(f"Elapsed      : {self.format_time(timing['elapsed'])} ({int(timing['elapsed'])} sec)")
        print(f"Expected     : {self.format_time(timing['expected_total'])}")
        print(f"Actual speed : {timing['actual_sec_per_image']:.3f} sec/image")
        print(f"Dir          : {self.save_dir}")
        print("======================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", choices=["cam0", "cam1"], help="cam0 or cam1")
    parser.add_argument("--interval", type=float, default=0.3, help="capture interval seconds")
    parser.add_argument("--target", type=int, default=1000, help="target image count for this camera")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="delete existing images for this camera before capture",
    )
    args = parser.parse_args()

    app = AutoCapture(
        cam_name=args.camera,
        interval=args.interval,
        target=args.target,
        clear=args.clear,
    )
    app.run()


if __name__ == "__main__":
    main()