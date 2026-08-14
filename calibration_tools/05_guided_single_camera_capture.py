import cv2
import argparse
from pathlib import Path
import os
import re


# 저장소 루트 기준으로 잡는다. 이 스크립트는 calibration_tools/ 안에 있다.
# (예전에는 ~/turtlebot4_ws/final_project 가 하드코딩돼 있어 다른 PC 에서 깨졌다.)
REPO_DIR = Path(__file__).resolve().parents[1]
VISION_DIR = REPO_DIR / "vision_pc3"   # 모델·캘리브레이션·맵이 사는 곳
BASE_DIR = REPO_DIR                      # captures/ dataset/ 등 작업용 산출물
DATASET_DIR = BASE_DIR / "dataset" / "raw"

CAMERA_IDS = {
    "cam0": 2,
    "cam1": 4,
}

TOTAL_TARGET_BOTH_CAMERAS = 1000


# target = 카메라 1대 기준 목표 촬영 장수
# cam0 500장 + cam1 500장 = 총 1000장
SCENARIOS = [
    {
        "name": "01_empty_area",
        "title": "STEP 1 / Empty Area",
        "guide": "No person. Clean background only.",
        "target": 15,
    },

    {
        "name": "02_one_standing_helmet",
        "title": "STEP 2 / One Standing Person + Helmet",
        "guide": "One person with helmet. Stand/walk in central visible area.",
        "target": 40,
    },
    {
        "name": "03_one_standing_no_helmet",
        "title": "STEP 3 / One Standing Person + No Helmet",
        "guide": "One person without helmet. Stand/walk/turn head.",
        "target": 40,
    },

    {
        "name": "04_one_crouching_helmet",
        "title": "STEP 4 / One Crouching Person + Helmet",
        "guide": "Crouch/squat/bend with helmet visible.",
        "target": 30,
    },
    {
        "name": "05_one_crouching_no_helmet",
        "title": "STEP 5 / One Crouching Person + No Helmet",
        "guide": "Crouch/squat/bend without helmet. Head visible.",
        "target": 30,
    },

    {
        "name": "06_edge_standing_partial_head",
        "title": "STEP 6 / Edge Area Standing + Partial Head",
        "guide": "Stand near camera edge. Head/helmet may be partially hidden.",
        "target": 30,
    },
    {
        "name": "07_edge_crouching_helmet",
        "title": "STEP 7 / Edge Area Crouching + Helmet",
        "guide": "Crouch near edge so helmet becomes visible.",
        "target": 30,
    },
    {
        "name": "08_edge_crouching_no_helmet",
        "title": "STEP 8 / Edge Area Crouching + No Helmet",
        "guide": "Crouch near edge without helmet. Head visible.",
        "target": 30,
    },

    {
        "name": "09_two_person_both_helmet",
        "title": "STEP 9 / Two Persons + Both Helmet",
        "guide": "Two people both wearing helmets. Far/close/crossing.",
        "target": 40,
    },
    {
        "name": "10_two_person_mixed_helmet",
        "title": "STEP 10 / Two Persons + Mixed Helmet",
        "guide": "One with helmet, one without. Swap front/back.",
        "target": 60,
    },
    {
        "name": "11_two_person_no_helmet",
        "title": "STEP 11 / Two Persons + No Helmet",
        "guide": "Two people without helmets. Far/close/overlap.",
        "target": 45,
    },

    {
        "name": "12_occlusion_box_wall",
        "title": "STEP 12 / Occlusion Near Box or Wall",
        "guide": "Person partly hidden by box/wall. Partial head/helmet.",
        "target": 35,
    },

    {
        "name": "13_fall_lying_pose",
        "title": "STEP 13 / Lying Pose",
        "guide": "Lie down, side lying, face down, near wall/box.",
        "target": 50,
    },
    {
        "name": "14_fall_transition",
        "title": "STEP 14 / Fall Transition",
        "guide": "Standing to lying, crouching to lying, getting up.",
        "target": 25,
    },
]


class GuidedCapture:
    def __init__(self, cam_name):
        self.cam_name = cam_name
        self.cam_id = CAMERA_IDS[cam_name]

        self.step_idx = 0
        self.latest_frame = None

        # u로 삭제하기 위한 현재 실행 세션의 저장 기록
        self.history = []

        self.window_name = f"{cam_name} guided capture"

        self.save_root = DATASET_DIR / cam_name
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.cap = self.open_camera(self.cam_id)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def open_camera(self, cam_id):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera id {cam_id}")

        return cap

    def current_scenario(self):
        return SCENARIOS[self.step_idx]

    def scenario_by_name(self, scenario_name):
        for s in SCENARIOS:
            if s["name"] == scenario_name:
                return s
        return None

    def save_dir_for(self, cam_name, scenario_name):
        save_dir = DATASET_DIR / cam_name / scenario_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def current_save_dir(self):
        scenario = self.current_scenario()
        return self.save_dir_for(self.cam_name, scenario["name"])

    def count_images_in_dir(self, save_dir):
        if not save_dir.exists():
            return 0
        return len(list(save_dir.glob("*.jpg")))

    def count_for_camera_scenario(self, cam_name, scenario_name):
        save_dir = DATASET_DIR / cam_name / scenario_name
        return self.count_images_in_dir(save_dir)

    def count_for_current_step_current_camera(self):
        return self.count_for_camera_scenario(
            self.cam_name,
            self.current_scenario()["name"],
        )

    def count_for_current_step_both_cameras(self):
        scenario_name = self.current_scenario()["name"]
        return (
            self.count_for_camera_scenario("cam0", scenario_name)
            + self.count_for_camera_scenario("cam1", scenario_name)
        )

    def total_count_for_camera(self, cam_name):
        root = DATASET_DIR / cam_name
        if not root.exists():
            return 0
        return len(list(root.glob("*/*.jpg")))

    def total_count_current_camera(self):
        return self.total_count_for_camera(self.cam_name)

    def total_count_both_cameras(self):
        return self.total_count_for_camera("cam0") + self.total_count_for_camera("cam1")

    def target_for_one_camera(self):
        return sum(s["target"] for s in SCENARIOS)

    def target_for_both_cameras(self):
        return self.target_for_one_camera() * 2

    def target_for_current_step_both_cameras(self):
        return self.current_scenario()["target"] * 2

    def get_next_index(self, save_dir, scenario_name):
        pattern = re.compile(rf"{self.cam_name}_{re.escape(scenario_name)}_(\d+)\.jpg$")

        max_idx = 0
        for path in save_dir.glob("*.jpg"):
            m = pattern.search(path.name)
            if m:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)

        return max_idx + 1

    def next_file_path(self):
        scenario = self.current_scenario()
        save_dir = self.current_save_dir()

        next_idx = self.get_next_index(save_dir, scenario["name"])
        filename = f"{self.cam_name}_{scenario['name']}_{next_idx:06d}.jpg"

        return save_dir / filename

    def save_current_frame(self):
        if self.latest_frame is None:
            print("[WARN] No frame to save yet.")
            return

        path = self.next_file_path()
        cv2.imwrite(str(path), self.latest_frame)

        self.history.append(
            {
                "path": path,
                "step_idx": self.step_idx,
                "scenario": self.current_scenario()["name"],
                "cam_name": self.cam_name,
            }
        )

        scenario = self.current_scenario()
        current_step_count = self.count_for_current_step_current_camera()
        current_step_target = scenario["target"]

        both_step_count = self.count_for_current_step_both_cameras()
        both_step_target = self.target_for_current_step_both_cameras()

        cam_total = self.total_count_current_camera()
        cam_target = self.target_for_one_camera()

        both_total = self.total_count_both_cameras()
        both_target = self.target_for_both_cameras()

        print(f"[SAVE] {path}")
        print(f"[STEP/CAM]  {scenario['name']}: {current_step_count}/{current_step_target}")
        print(f"[STEP/ALL]  {scenario['name']}: {both_step_count}/{both_step_target}")
        print(f"[CAM TOTAL] {self.cam_name}: {cam_total}/{cam_target}")
        print(f"[ALL TOTAL] cam0+cam1: {both_total}/{both_target}")

        if current_step_count >= current_step_target:
            print("======================================")
            print(f"[TARGET REACHED] {scenario['title']} for {self.cam_name}")
            print("Press n for next step, or keep shooting if you need more.")
            print("======================================")

    def undo_last_save(self):
        if not self.history:
            print("[UNDO] No saved image to delete in this session.")
            return

        last = self.history.pop()
        path = last["path"]

        if path.exists():
            os.remove(path)
            print(f"[DELETE] {path}")
        else:
            print(f"[WARN] File already missing: {path}")

        self.step_idx = last["step_idx"]
        scenario = self.current_scenario()

        print(f"[STEP] Moved to deleted image step: {scenario['name']}")
        print(f"[STEP/CAM] {scenario['name']}: {self.count_for_current_step_current_camera()}/{scenario['target']}")
        print(f"[ALL TOTAL] cam0+cam1: {self.total_count_both_cameras()}/{self.target_for_both_cameras()}")

    def next_step(self):
        if self.step_idx < len(SCENARIOS) - 1:
            self.step_idx += 1
            self.print_current_step("NEXT STEP")
        else:
            print("[INFO] Already at last step.")

    def prev_step(self):
        if self.step_idx > 0:
            self.step_idx -= 1
            self.print_current_step("PREV STEP")
        else:
            print("[INFO] Already at first step.")

    def print_current_step(self, label):
        scenario = self.current_scenario()

        print("======================================")
        print(f"[{label}] {scenario['title']}")
        print(f"[GUIDE] {scenario['guide']}")
        print("--------------------------------------")
        print(f"[STEP/CAM] {self.count_for_current_step_current_camera()}/{scenario['target']}")
        print(f"[STEP/ALL] {self.count_for_current_step_both_cameras()}/{self.target_for_current_step_both_cameras()}")
        print(f"[CAM TOTAL] {self.cam_name}: {self.total_count_current_camera()}/{self.target_for_one_camera()}")
        print(f"[ALL TOTAL] cam0+cam1: {self.total_count_both_cameras()}/{self.target_for_both_cameras()}")
        print("======================================")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_current_frame()

    def progress_color(self, count, target):
        if count >= target:
            return (0, 255, 0)
        ratio = count / target if target > 0 else 0
        if ratio >= 0.7:
            return (0, 200, 255)
        return (0, 120, 255)

    def draw_progress_bar(self, img, x, y, w, h, count, target, label):
        ratio = min(count / target, 1.0) if target > 0 else 0.0
        color = self.progress_color(count, target)

        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 2)

        fill_w = int(w * ratio)
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)

        cv2.putText(
            img,
            f"{label}: {count}/{target}",
            (x + w + 15, y + h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_overlay(self, frame):
        show = frame.copy()

        scenario = self.current_scenario()

        step_cam_count = self.count_for_current_step_current_camera()
        step_cam_target = scenario["target"]

        step_all_count = self.count_for_current_step_both_cameras()
        step_all_target = self.target_for_current_step_both_cameras()

        cam_total_count = self.total_count_current_camera()
        cam_total_target = self.target_for_one_camera()

        all_total_count = self.total_count_both_cameras()
        all_total_target = self.target_for_both_cameras()

        target_reached = step_cam_count >= step_cam_target

        overlay = show.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 310), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, show, 0.35, 0, show)

        title_color = (0, 255, 0) if target_reached else (0, 255, 255)

        lines = [
            f"{self.cam_name.upper()} / video{self.cam_id}",
            f"{scenario['title']}   ({self.step_idx + 1}/{len(SCENARIOS)})",
            f"Guide: {scenario['guide']}",
            "Mouse Left Click: SAVE | u: undo last | n: next | b: back | q: quit",
        ]

        y = 35
        for i, line in enumerate(lines):
            if i == 1:
                color = title_color
                scale = 0.85
                thickness = 2
            elif i == 3:
                color = (255, 255, 255)
                scale = 0.72
                thickness = 2
            else:
                color = (0, 255, 0)
                scale = 0.75
                thickness = 2

            cv2.putText(
                show,
                line,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += 35

        bar_x = 30
        bar_w = 420
        bar_h = 22

        self.draw_progress_bar(
            show,
            bar_x,
            155,
            bar_w,
            bar_h,
            step_cam_count,
            step_cam_target,
            "Current step / this cam",
        )

        self.draw_progress_bar(
            show,
            bar_x,
            190,
            bar_w,
            bar_h,
            step_all_count,
            step_all_target,
            "Current step / both cams",
        )

        self.draw_progress_bar(
            show,
            bar_x,
            225,
            bar_w,
            bar_h,
            cam_total_count,
            cam_total_target,
            f"{self.cam_name} total",
        )

        self.draw_progress_bar(
            show,
            bar_x,
            260,
            bar_w,
            bar_h,
            all_total_count,
            all_total_target,
            "All total",
        )

        if target_reached:
            cv2.putText(
                show,
                "TARGET REACHED - press n for next step",
                (590, 157),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return show

    def print_summary(self):
        print("")
        print("======================================")
        print(f"[SUMMARY] {self.cam_name}")
        print("--------------------------------------")

        for scenario in SCENARIOS:
            name = scenario["name"]
            cam_count = self.count_for_camera_scenario(self.cam_name, name)
            step_all_count = (
                self.count_for_camera_scenario("cam0", name)
                + self.count_for_camera_scenario("cam1", name)
            )

            target = scenario["target"]
            target_all = target * 2

            status = "OK" if cam_count >= target else "LOW"
            print(f"{name}: this_cam {cam_count}/{target} [{status}] | both {step_all_count}/{target_all}")

        print("--------------------------------------")
        print(f"{self.cam_name} total: {self.total_count_current_camera()}/{self.target_for_one_camera()}")
        print(f"cam0+cam1 total: {self.total_count_both_cameras()}/{self.target_for_both_cameras()}")
        print("======================================")
        print("")

    def run(self):
        print("======================================")
        print("Guided Single Camera Dataset Capture")
        print("--------------------------------------")
        print(f"Camera    : {self.cam_name}")
        print(f"Device    : /dev/video{self.cam_id}")
        print(f"Save root : {self.save_root}")
        print("--------------------------------------")
        print(f"Target this camera : {self.target_for_one_camera()} images")
        print(f"Target both cameras: {self.target_for_both_cameras()} images")
        print("--------------------------------------")
        print("Mouse Left Click : save current frame")
        print("u                : delete previous saved image")
        print("n                : next step")
        print("b                : previous step")
        print("q                : quit")
        print("======================================")
        self.print_current_step("START")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[WARN] frame read failed")
                continue

            self.latest_frame = frame.copy()

            show = self.draw_overlay(frame)
            cv2.imshow(self.window_name, show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("u"):
                self.undo_last_save()

            elif key == ord("n"):
                self.next_step()

            elif key == ord("b"):
                self.prev_step()

            elif key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera", choices=["cam0", "cam1"], help="cam0 or cam1")
    args = parser.parse_args()

    app = GuidedCapture(args.camera)
    app.run()


if __name__ == "__main__":
    main()