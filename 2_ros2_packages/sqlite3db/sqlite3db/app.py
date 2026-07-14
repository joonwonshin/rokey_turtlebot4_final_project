# 2_app.py
# Flask 웹 애플리케이션

from datetime import datetime, timedelta
from html import escape
from pathlib import Path
import sqlite3
from urllib.parse import quote

from flask import Flask, abort, jsonify, redirect, request, send_file

from sqlite3db.create_db import import_fire_extinguishers as import_fire_extinguisher_file


# Flask 애플리케이션 객체입니다. 이 객체에 URL 라우트를 등록합니다.
app = Flask(__name__)
# 잘못된 대용량 파일로 웹 서버 메모리가 과도하게 사용되는 것을 막습니다.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
# DB와 카메라 이미지 폴더는 현재 파이썬 파일과 같은 폴더를 기준으로 사용합니다.
DB_PATH = Path(__file__).with_name("fire_db.db")
IMAGE_DIR = Path(__file__).with_name("camera_frames")
# 이 시간(초) 안에 새 데이터가 들어오면 로봇이 연결된 것으로 표시합니다.
CONNECTION_TIMEOUT_SECONDS = 5
# 각 화면에 표시할 카메라 이름을 구분합니다.
WEBCAM_CAMERA_NAMES = {"Safety Cam 0", "Safety Cam 1"}
# Flask 개발 서버 실행 주소와 포트입니다.
HOST = "127.0.0.1"
PORT = 5000

# fleet_monitor.html의 관제 화면 팔레트를 Flask 전체 화면에서 공유합니다.
FLEET_MONITOR_THEME = """
:root {
    --bg: #12161d;
    --panel: #1a2029;
    --panel-raised: #202733;
    --panel-edge: #2a3342;
    --text: #d9dee7;
    --dim: #7d8794;
    --accent: #50c8c8;
    --accent-strong: #32a9ad;
    --success: #6fdb8d;
    --danger: #ff6464;
    --warning: #ffc45b;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { color-scheme: dark; }
body {
    margin: 0 !important;
    padding: 18px !important;
    min-height: 100vh;
    background:
        radial-gradient(circle at 92% 0, rgba(80, 200, 200, .09), transparent 28rem),
        var(--bg) !important;
    color: var(--text);
    font-family: 'Malgun Gothic', 'Noto Sans KR', sans-serif !important;
    font-size: 14px;
}
body > h1 {
    margin: 0 0 6px;
    font-size: clamp(24px, 3vw, 34px);
    letter-spacing: -.6px;
}
body > h1::before { content: '●'; color: var(--accent); margin-right: 10px; font-size: .58em; }
h2 {
    margin: 30px 0 14px;
    color: var(--text);
    font-size: 20px;
    letter-spacing: .8px;
    text-transform: uppercase;
}
h3 { margin: 0 0 4px; font-size: 16px; }
p { color: var(--dim); }
a { color: var(--accent); text-decoration: none; }
a:hover { color: #8be4e4; }
.back-link {
    display: inline-flex !important;
    align-items: center;
    gap: 6px;
    margin: 0 0 18px !important;
    padding: 6px 10px;
    border: 1px solid var(--panel-edge);
    border-radius: 5px;
    background: var(--panel);
    font-size: 12px;
}
.back-link::before { content: '←'; }
.monitor-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 18px 0;
}
.monitor-nav a {
    padding: 8px 12px;
    border: 1px solid var(--panel-edge);
    border-radius: 5px;
    background: var(--panel);
    color: var(--text);
}
.monitor-nav a:hover { border-color: var(--accent); color: var(--accent); }
.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    max-width: 1180px;
}
.dashboard-card, .camera-card, .csv-import, .status-table-scroll, .history-scroll {
    border: 1px solid var(--panel-edge) !important;
    border-radius: 8px;
    background: var(--panel) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, .14);
}
.dashboard-card {
    min-height: 170px;
    padding: 18px;
    transition: border-color .2s, transform .2s;
}
.dashboard-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.dashboard-card .card-kicker {
    color: var(--accent);
    font: 700 11px ui-monospace, 'Cascadia Mono', Consolas, monospace;
    letter-spacing: 1.2px;
}
.dashboard-card h2 { margin: 18px 0 8px; color: var(--text); font-size: 18px; letter-spacing: 0; }
.dashboard-card p { margin: 0; line-height: 1.65; }
.dashboard-card .card-link { display: inline-block; margin-top: 18px; font-weight: 700; }
.camera-grid { gap: 12px !important; }
.camera-card { padding: 12px !important; overflow: hidden; }
.camera-card p { margin: 5px 0; font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 12px; }
.camera-frame, .camera-card img {
    border: 1px solid #303a49;
    border-radius: 5px;
    background: #0c0f14 !important;
}
.placeholder {
    display: grid;
    min-height: 220px;
    place-items: center;
    border: 1px dashed #384456 !important;
    border-radius: 5px;
    background: #141922;
    color: var(--dim);
}
.camera-selector { color: var(--dim); font-size: 12px; }
select, input[type='file'] {
    padding: 7px 9px;
    border: 1px solid var(--panel-edge);
    border-radius: 5px;
    background: #0c0f14;
    color: var(--text);
}
button {
    padding: 8px 14px;
    border: 0;
    border-radius: 5px;
    background: var(--accent-strong);
    color: #fff;
    font-weight: 700;
    cursor: pointer;
}
button:hover { filter: brightness(1.15); }
table { border-collapse: collapse; width: 100%; background: transparent; }
th, td {
    padding: 10px 12px !important;
    border: 0 !important;
    border-bottom: 1px solid #252e3b !important;
    text-align: left;
}
th {
    background: #202733 !important;
    color: var(--dim);
    font-size: 11px;
    letter-spacing: .7px;
    text-transform: uppercase;
}
tbody tr:hover { background: rgba(80, 200, 200, .055); }
.connected, .result-pass { color: var(--success) !important; }
.disconnected, .result-fail { color: var(--danger) !important; }
.connection-status::before { content: '● '; font-size: 10px; }
.csv-import { padding: 16px !important; }
.csv-import h2 { margin: 0 0 14px !important; font-size: 20px; }
.csv-import-message { color: var(--success) !important; }
.history-title { margin-top: 28px !important; }
.history-row.has-snapshot:hover { background: rgba(80, 200, 200, .1) !important; }
.snapshot-panel { background: var(--panel) !important; border: 1px solid var(--panel-edge); }
.status-table-scroll, .history-scroll { overflow: auto; }
@media (max-width: 900px) {
    body { padding: 12px !important; }
    .dashboard-grid { grid-template-columns: 1fr; }
    .camera-grid { grid-template-columns: 1fr !important; }
    th, td { white-space: nowrap; }
}
"""


def get_connection():
    # SQLite DB에 연결한 connection 객체를 반환합니다.
    return sqlite3.connect(DB_PATH)


def get_fire_extinguisher_entries():
    # 소화기 테이블의 모든 행을 marker_id 순서로 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT
            marker_id,
            location_x,
            location_y,
            manufacture_date,
            last_inspection_date,
            pressure_status,
            result
        FROM fire_extinguisher
        ORDER BY marker_id
        """)
        return cursor.fetchall()


def get_fire_extinguisher_history_entries(limit=50):
    # 최근 점검 이력을 최신순으로 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    inspection_id,
                    marker_id,
                    robot_name,
                    result,
                    snapshot_filename,
                    inspected_at
                FROM fire_extinguisher_inspection_history
                ORDER BY inspection_id DESC
                LIMIT ?
                """,
                (limit,),
            )
        except sqlite3.OperationalError as error:
            # 기존 DB에 이력 테이블이 아직 없으면 빈 이력으로 표시합니다.
            if "no such table" in str(error):
                return []
            raise
        return cursor.fetchall()


def get_last_import_filename():
    # 소화기 페이지에 표시할 마지막 Import 원본 파일명을 조회합니다.
    with get_connection() as connection:
        try:
            row = connection.execute(
                """
                SELECT metadata_value
                FROM system_metadata
                WHERE metadata_key = 'last_import_filename'
                """
            ).fetchone()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error):
                return ""
            raise
    return row[0] if row else ""


def get_latest_fire_extinguisher_update(rows):
    # 소화기 점검 행들 중 가장 최근 업데이트 시간을 찾습니다.
    updated_values = [row[4] for row in rows if row[4]]

    if not updated_values:
        return ""

    # 시간 문자열 형식이 동일하므로 문자열 max로도 최신 시간이 선택됩니다.
    return max(updated_values)


def get_current_display_time():
    # 웹 화면이 최신 데이터를 다시 읽은 시간을 표시하기 위한 현재 시간 문자열입니다.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_fire_extinguisher_payload():
    # 브라우저의 주기적 갱신 요청에 응답할 소화기 DB JSON 데이터를 구성합니다.
    rows = get_fire_extinguisher_entries()
    history_rows = get_fire_extinguisher_history_entries()
    last_import_filename = get_last_import_filename()

    return {
        "updated_at": get_current_display_time(),
        "rows": [
            {
                "marker_id": marker_id,
                "location_x": location_x,
                "location_y": location_y,
                "manufacture_date": manufacture_date,
                "last_inspection_date": last_inspection_date or "",
                "pressure_status": pressure_status or "",
                "result": result or "",
                "result_class": get_result_class(result),
            }
            for (
                marker_id,
                location_x,
                location_y,
                manufacture_date,
                last_inspection_date,
                pressure_status,
                result,
            ) in rows
        ],
        "history": [
            {
                "inspection_id": inspection_id,
                "marker_id": marker_id,
                "robot_name": robot_name,
                "result": result,
                "result_class": get_result_class(result),
                "snapshot_url": (
                    f"/inspection_snapshot/{inspection_id}"
                    if snapshot_filename
                    else ""
                ),
                "inspected_at": inspected_at,
            }
            for (
                inspection_id,
                marker_id,
                robot_name,
                result,
                snapshot_filename,
                inspected_at,
            )
            in history_rows
        ],
    }


def get_robot_battery_entries():
    # 로봇 배터리 상태를 DB에서 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT
            robot_name,
            topic_name,
            percentage,
            voltage,
            current,
            updated_at
        FROM robot_battery_status
        ORDER BY robot_name
        """)
        rows = cursor.fetchall()

    if rows:
        return rows

    # 아직 ROS2 노드가 데이터를 넣지 않았을 때도 화면에 기본 행을 보여줍니다.
    return [
        ("Robot 2", "/robot2/battery_state", None, None, None, None),
        ("Robot 9", "/robot9/battery_state", None, None, None, None),
    ]


def get_robot_dock_entries():
    # 로봇 도킹 상태를 DB에서 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT
            robot_name,
            topic_name,
            dock_visible,
            is_docked,
            updated_at
        FROM robot_dock_status
        ORDER BY robot_name
        """)
        rows = cursor.fetchall()

    if rows:
        return rows

    # DB에 행이 없을 때 표시할 기본 도킹 토픽 정보입니다.
    return [
        ("Robot 2", "/robot2/dock_status", None, None, None),
        ("Robot 9", "/robot9/dock_status", None, None, None),
    ]


def get_robot_camera_entries():
    # 로봇 카메라 상태와 이미지 파일명을 DB에서 조회합니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT
            robot_name,
            topic_name,
            image_filename,
            updated_at
        FROM robot_camera_status
        ORDER BY robot_name
        """)
        rows = cursor.fetchall()

    if rows:
        return rows

    # DB에 행이 없을 때 표시할 기본 카메라 토픽 정보입니다.
    return [
        ("Robot 2", "/robot2/oakd/rgb/image_raw/compressed", None, None),
        ("Robot 9", "/robot9/oakd/rgb/image_raw/compressed", None, None),
        ("Safety Cam 0", "/safety/cam0/image/compressed", None, None),
        ("Safety Cam 1", "/safety/cam1/image/compressed", None, None),
    ]


def format_percentage(value):
    # 배터리 데이터가 없으면 대기 문구를 표시합니다.
    if value is None:
        return "Waiting for ROS2 data"
    # BatteryState.percentage는 0~1 값이므로 백분율로 변환합니다.
    return f"{value * 100:.1f}%"


def format_number(value, unit):
    # 값이 없으면 빈 칸으로 표시해 테이블을 깔끔하게 유지합니다.
    if value is None:
        return ""
    return f"{value:.2f} {unit}"


def format_dock_status(value):
    # 도킹 데이터가 아직 없으면 대기 문구를 표시합니다.
    if value is None:
        return "Waiting for ROS2 data"
    return "Dock" if value else "Undock"


def format_visible_status(value):
    # dock_visible 값이 없으면 빈 칸으로 표시합니다.
    if value is None:
        return ""
    return "Visible" if value else "Not visible"


def get_result_class(result):
    # 소화기 점검 결과에 따라 CSS 클래스를 선택합니다.
    if result == "PASS":
        return "result-pass"
    if result == "FAIL":
        return "result-fail"
    return ""


def is_connection_alive(updated_at):
    # 마지막 업데이트 후 경과 시간이 제한 시간 이하면 연결 상태로 판단합니다.
    return get_seconds_since_update(updated_at) <= CONNECTION_TIMEOUT_SECONDS


def get_seconds_since_update(updated_at):
    # 업데이트 시간이 없으면 아주 오래된 데이터처럼 처리합니다.
    if not updated_at:
        return float("inf")

    try:
        # DB에는 'YYYY-MM-DD HH:MM:SS' 형식 문자열로 시간이 저장됩니다.
        updated_time = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # 알 수 없는 시간 형식은 연결 끊김으로 판단되도록 무한대로 처리합니다.
        return float("inf")

    return (datetime.now() - updated_time).total_seconds()


def get_robot_connection_map(battery_rows, dock_rows, camera_rows):
    # 로봇별로 배터리/도킹/카메라 중 가장 최근 업데이트 시간을 모읍니다.
    latest_updates = {}

    for robot_name, _, _, _, _, updated_at in battery_rows:
        if updated_at:
            latest_updates[robot_name] = min(
                latest_updates.get(robot_name, float("inf")),
                get_seconds_since_update(updated_at),
            )

    for robot_name, _, _, _, updated_at in dock_rows:
        if updated_at:
            latest_updates[robot_name] = min(
                latest_updates.get(robot_name, float("inf")),
                get_seconds_since_update(updated_at),
            )

    for robot_name, _, _, updated_at in camera_rows:
        if updated_at:
            latest_updates[robot_name] = min(
                latest_updates.get(robot_name, float("inf")),
                get_seconds_since_update(updated_at),
            )

    # 제한 시간 안에 하나라도 최신 데이터가 있으면 해당 로봇을 연결 상태로 표시합니다.
    return {
        robot_name: seconds <= CONNECTION_TIMEOUT_SECONDS
        for robot_name, seconds in latest_updates.items()
    }


def get_latest_robot_update(robot_name, battery_rows, dock_rows, camera_rows):
    # 특정 로봇의 배터리/도킹/카메라 업데이트 시간 중 가장 최신 값을 찾습니다.
    updated_values = []

    for row in battery_rows:
        if row[0] == robot_name and row[5]:
            updated_values.append(row[5])

    for row in dock_rows:
        if row[0] == robot_name and row[4]:
            updated_values.append(row[4])

    for row in camera_rows:
        if row[0] == robot_name and row[3]:
            updated_values.append(row[3])

    if not updated_values:
        return ""

    # 시간 문자열 형식이 동일하므로 문자열 max로도 최신 시간이 선택됩니다.
    return max(updated_values)


def is_camera_frame_alive(updated_at):
    # 카메라 프레임이 없으면 화면에서 숨길 수 있도록 False를 반환합니다.
    if not updated_at:
        return False

    try:
        # 마지막 카메라 프레임 수신 시간을 datetime으로 변환합니다.
        updated_time = datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False

    # 지정 시간 안에 받은 프레임만 살아 있는 프레임으로 봅니다.
    return datetime.now() - updated_time <= timedelta(seconds=CONNECTION_TIMEOUT_SECONDS)


def get_robot_status_payload():
    # 브라우저의 주기적 갱신 요청에 응답할 JSON 데이터를 구성합니다.
    battery_rows = get_robot_battery_entries()
    dock_rows = get_robot_dock_entries()
    camera_rows = get_robot_camera_entries()
    robot_connection_map = get_robot_connection_map(battery_rows, dock_rows, camera_rows)

    return {
        # 배터리 테이블에 렌더링할 데이터입니다.
        "battery": [
            {
                "robot_name": robot_name,
                "topic_name": topic_name,
                "percentage": format_percentage(percentage),
                "voltage": format_number(voltage, "V"),
                "current": format_number(current, "A"),
                "connection_alive": robot_connection_map.get(robot_name, False),
                "updated_at": updated_at or "",
            }
            for (
                robot_name,
                topic_name,
                percentage,
                voltage,
                current,
                updated_at,
            ) in battery_rows
        ],
        # 도킹 테이블에 렌더링할 데이터입니다.
        "dock": [
            {
                "robot_name": robot_name,
                "topic_name": topic_name,
                "dock_status": format_dock_status(is_docked),
                "dock_visible": format_visible_status(dock_visible),
                "connection_alive": robot_connection_map.get(robot_name, False),
                "updated_at": updated_at or "",
            }
            for (
                robot_name,
                topic_name,
                dock_visible,
                is_docked,
                updated_at,
            ) in dock_rows
        ],
        # 카메라 카드와 이미지 갱신에 사용할 데이터입니다.
        "camera": [
            {
                "robot_name": robot_name,
                "topic_name": topic_name,
                "image_url": f"/camera_frame/{quote(robot_name)}" if image_filename else "",
                "updated_at": updated_at or "",
                "connection_alive": robot_connection_map.get(robot_name, False),
                "camera_frame_alive": is_camera_frame_alive(updated_at),
                "latest_robot_update": get_latest_robot_update(
                    robot_name,
                    battery_rows,
                    dock_rows,
                    camera_rows,
                ),
            }
            for robot_name, topic_name, image_filename, updated_at in camera_rows
        ],
    }


@app.route("/fleet-theme.css")
def fleet_theme():
    # 모든 Flask 화면이 동일한 관제 UI 테마를 사용하도록 CSS를 제공합니다.
    return app.response_class(FLEET_MONITOR_THEME, mimetype="text/css")


@app.route("/")
def index():
    # 홈 화면에서 각 모니터링 페이지 링크를 제공합니다.
    return """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>RAPID Monitoring</title>
        <link rel="stylesheet" href="/fleet-theme.css">
    </head>
    <body>
        <h1>RAPID Monitoring</h1>
        <p>ROS 2 fleet operations · live database console</p>
        <nav class="monitor-nav">
            <a href="/">Overview</a>
            <a href="/robot_status">Robot Status</a>
            <a href="/webcam_cctv">Webcam CCTV</a>
            <a href="/fire_extinguishers">Fire Extinguishers</a>
        </nav>
        <main class="dashboard-grid">
            <article class="dashboard-card">
                <span class="card-kicker">01 · FLEET</span>
                <h2>Robot Status</h2>
                <p>로봇의 배터리, 도킹, 연결 상태, 카메라 피드를 확인합니다.</p>
                <a class="card-link" href="/robot_status">Open monitor →</a>
            </article>
            <article class="dashboard-card">
                <span class="card-kicker">02 · CCTV</span>
                <h2>Safety Cameras</h2>
                <p>두 안전 카메라의 최신 프레임과 ROS 2 수신 상태를 실시간으로 확인합니다.</p>
                <a class="card-link" href="/webcam_cctv">Open CCTV →</a>
            </article>
            <article class="dashboard-card">
                <span class="card-kicker">03 · INSPECTION</span>
                <h2>Fire Extinguishers</h2>
                <p>소화기 기준 정보, PASS/FAIL 결과, 최근 점검 이력과 스냅샷을 관리합니다.</p>
                <a class="card-link" href="/fire_extinguishers">Open database →</a>
            </article>
        </main>
    </body>
    </html>
    """


@app.route("/fire_extinguishers/import", methods=["POST"])
def import_fire_extinguishers():
    # 브라우저에서 선택한 CSV/JSON/XLSX를 메모리에서 검증하고 DB에 반영합니다.
    uploaded_file = request.files.get("data_file")
    if uploaded_file is None or not uploaded_file.filename:
        return _data_import_result_page("선택된 데이터 파일이 없습니다.", False), 400

    try:
        import_fire_extinguisher_file(uploaded_file.read(), uploaded_file.filename)
    except (ValueError, sqlite3.Error) as error:
        return _data_import_result_page(str(error), False), 400

    # 성공 알림 페이지를 거치지 않고 갱신된 소화기 테이블로 바로 이동합니다.
    return redirect("/fire_extinguishers?imported=1")


def _data_import_result_page(message, success):
    # 결과 문구는 escape해 업로드된 내용이 HTML로 해석되지 않게 합니다.
    title = "Data Import Complete" if success else "Data Import Failed"
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link rel="stylesheet" href="/fleet-theme.css">
    </head>
    <body>
        <h1>{title}</h1>
        <p>{escape(message)}</p>
        <nav class="monitor-nav">
            <a href="/">Back to Home</a>
            <a href="/fire_extinguishers">Fire Extinguishers</a>
        </nav>
    </body>
    </html>
    """


@app.route("/webcam_cctv")
def webcam_cctv_page():
    # 안전 카메라의 최신 압축 이미지와 연결 상태를 표시합니다.
    return """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Webcam CCTV</title>
        <link rel="stylesheet" href="/fleet-theme.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 32px;
            }
            .back-link {
                display: inline-block;
                margin-bottom: 16px;
            }
            .camera-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
            }
            .camera-view {
                min-width: 0;
            }
            .camera-selector {
                margin-bottom: 12px;
            }
            .camera-card {
                border: 1px solid #ccc;
                padding: 12px;
            }
            .camera-frame {
                display: block;
                width: 100%;
                max-height: 360px;
                background: #111;
                object-fit: contain;
            }
            .placeholder {
                border: 1px solid #ccc;
                padding: 16px;
                margin-bottom: 12px;
            }
            .connection-status {
                font-weight: 700;
                margin: 8px 0;
            }
            .connected {
                color: #16803c;
            }
            .disconnected {
                color: #c62828;
            }
            @media (max-width: 900px) {
                .camera-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <h1>Webcam CCTV</h1>
        <a class="back-link" href="/">Back to Home</a>

        <div class="camera-grid">
          <div class="camera-view">
            <div class="camera-selector">
                <label for="webcam-select-1">Camera 1: </label>
                <select id="webcam-select-1">
                    <option value="Safety Cam 0">Safety Cam 0</option>
                    <option value="Safety Cam 1">Safety Cam 1</option>
                </select>
            </div>
            <div class="camera-card">
                <h3 id="webcam-name-1">Safety Cam 0</h3>
                <p id="webcam-topic-1">/safety/cam0/image/compressed</p>
                <img id="webcam-image-1" class="camera-frame" alt="Selected safety camera 1" hidden>
                <div id="webcam-placeholder-1" class="placeholder">Waiting for camera frame</div>
                <p id="webcam-status-1" class="connection-status disconnected">Disconnected</p>
                <p>updated_at: <span id="webcam-updated-at-1"></span></p>
            </div>
          </div>
          <div class="camera-view">
            <div class="camera-selector">
                <label for="webcam-select-2">Camera 2: </label>
                <select id="webcam-select-2">
                    <option value="Safety Cam 0">Safety Cam 0</option>
                    <option value="Safety Cam 1" selected>Safety Cam 1</option>
                </select>
            </div>
            <div class="camera-card">
                <h3 id="webcam-name-2">Safety Cam 1</h3>
                <p id="webcam-topic-2">/safety/cam1/image/compressed</p>
                <img id="webcam-image-2" class="camera-frame" alt="Selected safety camera 2" hidden>
                <div id="webcam-placeholder-2" class="placeholder">Waiting for camera frame</div>
                <p id="webcam-status-2" class="connection-status disconnected">Disconnected</p>
                <p>updated_at: <span id="webcam-updated-at-2"></span></p>
            </div>
          </div>
        </div>
        <script>
            const CAMERA_REFRESH_INTERVAL_MS = 100;
            const STATUS_REFRESH_INTERVAL_MS = 1000;
            let latestWebcamRows = [];

            function refreshFrame(slot) {
                const image = document.getElementById(`webcam-image-${slot}`);
                if (image.dataset.loading === "true" || image.dataset.connected !== "true" || !image.dataset.cameraSrc) {
                    return;
                }

                image.dataset.loading = "true";
                const nextImage = new Image();
                nextImage.onload = () => {
                    image.src = nextImage.src;
                    image.hidden = false;
                    image.dataset.loading = "false";
                };
                nextImage.onerror = () => {
                    image.dataset.loading = "false";
                };
                nextImage.src = `${image.dataset.cameraSrc}?t=${Date.now()}`;
            }

            function renderWebcam(slot) {
                const select = document.getElementById(`webcam-select-${slot}`);
                const camera = latestWebcamRows.find(
                    (item) => item.robot_name === select.value
                );
                const image = document.getElementById(`webcam-image-${slot}`);
                const placeholder = document.getElementById(`webcam-placeholder-${slot}`);
                const status = document.getElementById(`webcam-status-${slot}`);

                if (!camera) {
                    image.hidden = true;
                    placeholder.hidden = false;
                    status.textContent = "Disconnected";
                    status.classList.remove("connected");
                    status.classList.add("disconnected");
                    return;
                }

                document.getElementById(`webcam-name-${slot}`).textContent = camera.robot_name;
                document.getElementById(`webcam-topic-${slot}`).textContent = camera.topic_name;
                document.getElementById(`webcam-updated-at-${slot}`).textContent = camera.updated_at;
                image.dataset.cameraSrc = camera.image_url;
                image.dataset.connected = camera.camera_frame_alive ? "true" : "false";
                image.hidden = !camera.camera_frame_alive;
                placeholder.hidden = camera.camera_frame_alive;
                status.textContent = camera.camera_frame_alive ? "Connected" : "Disconnected";
                status.classList.toggle("connected", camera.camera_frame_alive);
                status.classList.toggle("disconnected", !camera.camera_frame_alive);
            }

            function updateWebcamSelectors(cameras) {
                latestWebcamRows = cameras;
                const names = new Set(cameras.map((camera) => camera.robot_name));
                [1, 2].forEach((slot) => {
                    const select = document.getElementById(`webcam-select-${slot}`);
                    const previousSelection = select.value;
                    Array.from(select.options).forEach((option) => {
                        if (!names.has(option.value)) option.remove();
                    });
                    cameras.forEach((camera) => {
                        if (!Array.from(select.options).some(
                            (option) => option.value === camera.robot_name
                        )) {
                            select.add(new Option(camera.robot_name, camera.robot_name));
                        }
                    });
                    if (names.has(previousSelection)) {
                        select.value = previousSelection;
                    } else if (cameras.length > 0) {
                        select.value = cameras[Math.min(slot - 1, cameras.length - 1)].robot_name;
                    }
                    renderWebcam(slot);
                });
            }

            async function refreshCameraStatus() {
                try {
                    const response = await fetch(`/webcam_status_data?t=${Date.now()}`, {
                        cache: "no-store"
                    });
                    if (!response.ok) return;

                    const cameras = await response.json();
                    updateWebcamSelectors(cameras);
                } catch (error) {
                    document.querySelectorAll(".connection-status").forEach((status) => {
                        status.textContent = "Flask connection error";
                        status.classList.remove("connected");
                        status.classList.add("disconnected");
                    });
                }
            }

            [1, 2].forEach((slot) => {
                document.getElementById(`webcam-select-${slot}`).addEventListener(
                    "change", () => renderWebcam(slot)
                );
            });

            refreshCameraStatus();
            setInterval(refreshCameraStatus, STATUS_REFRESH_INTERVAL_MS);
            setInterval(() => {
                [1, 2].forEach(refreshFrame);
            }, CAMERA_REFRESH_INTERVAL_MS);
        </script>
    </body>
    </html>
    """


@app.route("/fire_extinguishers")
def fire_extinguishers_page():
    # 소화기 DB 데이터를 조회해 HTML 테이블 행으로 변환합니다.
    rows = get_fire_extinguisher_entries()
    history_rows = get_fire_extinguisher_history_entries()
    last_import_filename = get_last_import_filename()
    updated_at = get_current_display_time()
    import_completed = request.args.get("imported") == "1"
    import_message = (
        '<span id="csv-import-message" class="csv-import-message">DB Overwrited!</span>'
        if import_completed
        else ""
    )

    table_rows = "\n".join(
        f"""
        <tr>
            <td>{escape(str(marker_id))}</td>
            <td>{escape(str(location_x))}</td>
            <td>{escape(str(location_y))}</td>
            <td>{escape(str(manufacture_date))}</td>
            <td>{escape(str(last_inspection_date or ""))}</td>
            <td>{escape(str(pressure_status or ""))}</td>
            <td class="{escape(get_result_class(result))}">{escape(str(result or ""))}</td>
        </tr>
        """
        for (
            marker_id,
            location_x,
            location_y,
            manufacture_date,
            last_inspection_date,
            pressure_status,
            result,
        ) in rows
    )

    # 조회 결과가 없을 때는 빈 테이블 대신 안내 행을 표시합니다.
    if not table_rows:
        table_rows = '<tr><td colspan="7">No data</td></tr>'

    history_table_rows = "\n".join(
        f"""
        <tr class="history-row{' has-snapshot' if snapshot_filename else ''}"
            data-inspection-id="{escape(str(inspection_id))}"
            data-snapshot-url="{'/inspection_snapshot/' + str(inspection_id) if snapshot_filename else ''}">
            <td>{escape(str(inspection_id))}</td>
            <td>{escape(str(marker_id))}</td>
            <td>{escape(str(robot_name))}</td>
            <td class="{escape(get_result_class(result))}">{escape(str(result))}</td>
            <td>{escape(str(inspected_at))}</td>
        </tr>
        """
        for (
            inspection_id,
            marker_id,
            robot_name,
            result,
            snapshot_filename,
            inspected_at,
        )
        in history_rows
    )
    if not history_table_rows:
        history_table_rows = '<tr><td colspan="5">No inspection history</td></tr>'

    # escape()로 DB 문자열을 HTML에 안전하게 출력합니다.
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Fire Extinguisher DB</title>
        <link rel="stylesheet" href="/fleet-theme.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            .history-title {{
                margin-top: 32px;
            }}
            .history-scroll {{
                max-height: 780px;
                overflow-y: auto;
                border-bottom: 1px solid #ccc;
                scroll-behavior: auto;
            }}
            .history-scroll table {{
                margin: 0;
            }}
            .history-scroll thead th {{
                position: sticky;
                top: 0;
                z-index: 1;
            }}
            .history-row.has-snapshot {{
                cursor: pointer;
            }}
            .history-row.has-snapshot:hover {{
                background: #eef6ff;
            }}
            .inspection-snapshot {{
                display: block;
                max-width: min(900px, 90vw);
                max-height: 80vh;
                object-fit: contain;
                background: #111;
            }}
            .snapshot-overlay {{
                position: fixed;
                inset: 0;
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 24px;
                background: rgba(0, 0, 0, 0.72);
            }}
            .snapshot-overlay[hidden] {{
                display: none;
            }}
            .snapshot-panel {{
                position: relative;
                padding: 16px;
                border-radius: 8px;
                background: #fff;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
            }}
            .snapshot-close {{
                position: absolute;
                top: 4px;
                right: 6px;
                width: 32px;
                height: 32px;
                border: 0;
                border-radius: 50%;
                color: #fff;
                background: rgba(0, 0, 0, 0.65);
                font-size: 22px;
                cursor: pointer;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px 10px;
                text-align: left;
            }}
            th {{
                background: #f2f2f2;
            }}
            .back-link {{
                display: inline-block;
                margin-bottom: 16px;
            }}
            .result-pass {{
                color: #16803c;
                font-weight: 700;
            }}
            .result-fail {{
                color: #c62828;
                font-weight: 700;
            }}
            .csv-import {{
                border: 1px solid #ccc;
                padding: 16px;
                margin-bottom: 24px;
            }}
            .csv-import h2 {{
                margin-top: 0;
            }}
            .csv-import-message {{
                color: #16803c;
                font-weight: 700;
                margin-left: 12px;
                transition: opacity 0.3s ease;
            }}
            .connection-status {{
                font-weight: 700;
            }}
            .connected {{
                color: #16803c;
            }}
            .disconnected {{
                color: #c62828;
            }}
        </style>
    </head>
    <body>
        <h1>Fire Extinguisher DB</h1>
        <a class="back-link" href="/">Back to Home</a>
        <p>updated_at: <span id="fire-db-updated-at">{escape(str(updated_at))}</span></p>
        <section class="csv-import">
            <h2>Data Import</h2>
            <p>CSV, JSON, XLSX 파일을 선택하면 기존 소화기 목록을 파일 내용으로 교체합니다.</p>
            <p>Imported file: <strong>{escape(last_import_filename or "None")}</strong></p>
            <form action="/fire_extinguishers/import" method="post" enctype="multipart/form-data">
                <input type="file" name="data_file" accept=".csv,.json,.xlsx,text/csv,application/json" required>
                <button type="submit">Import Data</button>
                {import_message}
            </form>
        </section>
        <table>
            <thead>
                <tr>
                    <th>marker_id</th>
                    <th>location_x</th>
                    <th>location_y</th>
                    <th>manufacture_date</th>
                    <th>last_inspection_date</th>
                    <th>pressure_status</th>
                    <th>result</th>
                </tr>
            </thead>
            <tbody id="fire-extinguisher-body">
                {table_rows}
            </tbody>
        </table>
        <h2 class="history-title">Inspection History</h2>
        <p>Latest 50 inspections (about 20 rows visible)</p>
        <div class="history-scroll">
            <table>
                <thead>
                    <tr>
                        <th>inspection_id</th>
                        <th>marker_id</th>
                        <th>robot_name</th>
                        <th>result</th>
                        <th>inspected_at</th>
                    </tr>
                </thead>
                <tbody id="inspection-history-body">
                    {history_table_rows}
                </tbody>
            </table>
        </div>
        <div id="snapshot-overlay" class="snapshot-overlay" hidden>
            <div class="snapshot-panel" role="dialog" aria-modal="true" aria-label="Inspection snapshot">
                <button id="snapshot-close" class="snapshot-close" type="button" aria-label="Close">&times;</button>
                <img id="inspection-snapshot-image" class="inspection-snapshot" alt="Inspection snapshot">
            </div>
        </div>
        <script>
            const FIRE_DB_REFRESH_INTERVAL_MS = 1000;

            const importMessage = document.getElementById("csv-import-message");
            if (importMessage) {{
                setTimeout(() => {{
                    importMessage.style.opacity = "0";
                    setTimeout(() => importMessage.remove(), 300);
                }}, 3000);
            }}

            function setCellText(row, text) {{
                // 테이블 행에 텍스트 셀을 하나 추가합니다.
                const cell = document.createElement("td");
                cell.textContent = text;
                row.appendChild(cell);
            }}

            function renderFireExtinguisherRows(rows) {{
                // JSON으로 받은 소화기 DB 데이터를 테이블에 다시 렌더링합니다.
                const tableBody = document.getElementById("fire-extinguisher-body");
                tableBody.replaceChildren();

                if (rows.length === 0) {{
                    const row = document.createElement("tr");
                    const cell = document.createElement("td");
                    cell.colSpan = 7;
                    cell.textContent = "No data";
                    row.appendChild(cell);
                    tableBody.appendChild(row);
                    return;
                }}

                rows.forEach((item) => {{
                    const row = document.createElement("tr");
                    setCellText(row, item.marker_id);
                    setCellText(row, item.location_x);
                    setCellText(row, item.location_y);
                    setCellText(row, item.manufacture_date);
                    setCellText(row, item.last_inspection_date);
                    setCellText(row, item.pressure_status);

                    const resultCell = document.createElement("td");
                    resultCell.textContent = item.result;
                    if (item.result_class) {{
                        resultCell.classList.add(item.result_class);
                    }}
                    row.appendChild(resultCell);

                    tableBody.appendChild(row);
                }});
            }}

            let renderedHistorySignature = "";

            function renderInspectionHistory(rows) {{
                // 점검할 때마다 누적된 최근 이력을 테이블에 다시 그립니다.
                const tableBody = document.getElementById("inspection-history-body");
                const historyScroll = document.querySelector(".history-scroll");
                const previousScrollTop = historyScroll.scrollTop;
                tableBody.replaceChildren();

                if (rows.length === 0) {{
                    const row = document.createElement("tr");
                    const cell = document.createElement("td");
                    cell.colSpan = 5;
                    cell.textContent = "No inspection history";
                    row.appendChild(cell);
                    tableBody.appendChild(row);
                    return;
                }}

                rows.forEach((item) => {{
                    const row = document.createElement("tr");
                    row.classList.add("history-row");
                    row.dataset.inspectionId = item.inspection_id;
                    row.dataset.snapshotUrl = item.snapshot_url;
                    if (item.snapshot_url) {{
                        row.classList.add("has-snapshot");
                    }}
                    setCellText(row, item.inspection_id);
                    setCellText(row, item.marker_id);
                    setCellText(row, item.robot_name);

                    const resultCell = document.createElement("td");
                    resultCell.textContent = item.result;
                    if (item.result_class) {{
                        resultCell.classList.add(item.result_class);
                    }}
                    row.appendChild(resultCell);

                    setCellText(row, item.inspected_at);
                    tableBody.appendChild(row);
                }});

                // tbody 교체 후에도 사용자가 보고 있던 스크롤 위치를 유지합니다.
                historyScroll.scrollTop = previousScrollTop;
                requestAnimationFrame(() => {{
                    historyScroll.scrollTop = previousScrollTop;
                }});
            }}

            const snapshotOverlay = document.getElementById("snapshot-overlay");
            const snapshotImage = document.getElementById("inspection-snapshot-image");
            let openedInspectionId = "";

            function closeSnapshot() {{
                snapshotOverlay.hidden = true;
                snapshotImage.removeAttribute("src");
                openedInspectionId = "";
            }}

            document.getElementById("inspection-history-body").addEventListener(
                "click",
                (event) => {{
                    const historyRow = event.target.closest(".history-row.has-snapshot");
                    if (!historyRow) {{
                        return;
                    }}
                    if (
                        !snapshotOverlay.hidden &&
                        openedInspectionId === historyRow.dataset.inspectionId
                    ) {{
                        closeSnapshot();
                        return;
                    }}
                    snapshotImage.src = historyRow.dataset.snapshotUrl;
                    snapshotImage.alt = `Inspection ${{historyRow.dataset.inspectionId}} snapshot`;
                    openedInspectionId = historyRow.dataset.inspectionId;
                    snapshotOverlay.hidden = false;
                }}
            );

            document.getElementById("snapshot-close").addEventListener("click", closeSnapshot);
            snapshotOverlay.addEventListener("click", (event) => {{
                if (event.target === snapshotOverlay) {{
                    closeSnapshot();
                }}
            }});
            document.addEventListener("keydown", (event) => {{
                if (event.key === "Escape" && !snapshotOverlay.hidden) {{
                    closeSnapshot();
                }}
            }});

            function updateFireDbStatus(data) {{
                // 테이블 위쪽의 최신 업데이트 시간 텍스트를 갱신합니다.
                const updatedAt = document.getElementById("fire-db-updated-at");

                updatedAt.textContent = data.updated_at;
            }}

            async function refreshFireExtinguishers() {{
                // Flask JSON API에서 최신 소화기 DB 상태를 받아옵니다.
                const response = await fetch(`/fire_extinguishers_data?t=${{Date.now()}}`, {{
                    cache: "no-store"
                }});

                if (!response.ok) {{
                    return;
                }}

                const data = await response.json();
                updateFireDbStatus(data);
                renderFireExtinguisherRows(data.rows);
                const historySignature = JSON.stringify(data.history);
                if (historySignature !== renderedHistorySignature) {{
                    renderInspectionHistory(data.history);
                    renderedHistorySignature = historySignature;
                }}
            }}

            refreshFireExtinguishers();
            setInterval(refreshFireExtinguishers, FIRE_DB_REFRESH_INTERVAL_MS);
        </script>
    </body>
    </html>
    """


@app.route("/fire_extinguishers_data")
def fire_extinguishers_data():
    # JavaScript가 주기적으로 호출하는 소화기 DB JSON API입니다.
    return jsonify(get_fire_extinguisher_payload())


@app.route("/inspection_snapshot/<int:inspection_id>")
def inspection_snapshot(inspection_id):
    # 이력에 저장된 점검 순간의 마커 스냅샷을 반환합니다.
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT snapshot_filename
            FROM fire_extinguisher_inspection_history
            WHERE inspection_id = ?
            """,
            (inspection_id,),
        ).fetchone()
    if row is None or not row[0]:
        abort(404)

    snapshot_path = (IMAGE_DIR / row[0]).resolve()
    image_root = IMAGE_DIR.resolve()
    if image_root not in snapshot_path.parents or not snapshot_path.is_file():
        abort(404)
    return send_file(snapshot_path, mimetype="image/jpeg", max_age=0)


@app.route("/camera_frame/<robot_name>")
def camera_frame(robot_name):
    # robot_name에 해당하는 최신 카메라 이미지 파일명을 DB에서 찾습니다.
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT image_filename
            FROM robot_camera_status
            WHERE robot_name = ?
            """,
            (robot_name,),
        )
        row = cursor.fetchone()

    # DB에 이미지 정보가 없으면 404를 반환합니다.
    if row is None or row[0] is None:
        abort(404)

    image_path = IMAGE_DIR / row[0]
    # DB에는 파일명이 있지만 실제 파일이 없으면 404를 반환합니다.
    if not image_path.exists():
        abort(404)

    # 브라우저 캐시를 막아 항상 최신 프레임을 요청하게 합니다.
    response = send_file(image_path, mimetype="image/jpeg", max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/robot_status_data")
def robot_status_data():
    # 로봇 상태 화면에서는 별도 CCTV 화면용 안전 카메라를 제외합니다.
    payload = get_robot_status_payload()
    payload["camera"] = [
        camera
        for camera in payload["camera"]
        if camera["robot_name"] not in WEBCAM_CAMERA_NAMES
    ]
    return jsonify(payload)


@app.route("/webcam_status_data")
def webcam_status_data():
    # Webcam CCTV 화면에는 외부 안전 카메라 상태만 반환합니다.
    camera_rows = get_robot_status_payload()["camera"]
    return jsonify(
        [camera for camera in camera_rows if camera["robot_name"] in WEBCAM_CAMERA_NAMES]
    )


@app.route("/robot_status")
def robot_status_page():
    # 첫 화면 렌더링에 사용할 현재 DB 상태를 조회합니다.
    battery_rows = get_robot_battery_entries()
    dock_rows = get_robot_dock_entries()
    all_camera_rows = get_robot_camera_entries()
    # Safety Cam은 Webcam CCTV 화면에서만 표시합니다.
    camera_rows = [
        row for row in all_camera_rows if row[0] not in WEBCAM_CAMERA_NAMES
    ]
    robot_connection_map = get_robot_connection_map(
        battery_rows,
        dock_rows,
        all_camera_rows,
    )

    # 배터리 상태 테이블의 초기 HTML 행을 만듭니다.
    battery_topic_rows = "\n".join(
        f"""
        <tr>
            <td>{escape(robot_name)}</td>
            <td>{escape(topic_name)}</td>
            <td>{escape(format_percentage(percentage))}</td>
            <td>{escape(format_number(voltage, "V"))}</td>
            <td>{escape(format_number(current, "A"))}</td>
            <td class="connection-status {'connected' if robot_connection_map.get(robot_name, False) else 'disconnected'}">
                {'Connected' if robot_connection_map.get(robot_name, False) else 'Disconnected'}
            </td>
            <td>{escape(str(updated_at or ""))}</td>
        </tr>
        """
        for (
            robot_name,
            topic_name,
            percentage,
            voltage,
            current,
            updated_at,
        ) in battery_rows
    )

    # 도킹 상태 테이블의 초기 HTML 행을 만듭니다.
    dock_topic_rows = "\n".join(
        f"""
        <tr>
            <td>{escape(robot_name)}</td>
            <td>{escape(topic_name)}</td>
            <td>{escape(format_dock_status(is_docked))}</td>
            <td>{escape(format_visible_status(dock_visible))}</td>
            <td class="connection-status {'connected' if robot_connection_map.get(robot_name, False) else 'disconnected'}">
                {'Connected' if robot_connection_map.get(robot_name, False) else 'Disconnected'}
            </td>
            <td>{escape(str(updated_at or ""))}</td>
        </tr>
        """
        for (
            robot_name,
            topic_name,
            dock_visible,
            is_docked,
            updated_at,
        ) in dock_rows
    )

    camera_options_1 = "\n".join(
        f'<option value="{escape(robot_name)}">{escape(robot_name)}</option>'
        for robot_name, _, _, _ in camera_rows
    )
    camera_options_2 = "\n".join(
        f'<option value="{escape(robot_name)}"'
        f'{" selected" if index == 1 else ""}>{escape(robot_name)}</option>'
        for index, (robot_name, _, _, _) in enumerate(camera_rows)
    )

    # 이후 상태 변화는 아래 JavaScript가 /robot_status_data를 호출해 갱신합니다.
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Robot status</title>
        <link rel="stylesheet" href="/fleet-theme.css">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 28px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px 10px;
                text-align: left;
            }}
            th {{
                background: #f2f2f2;
            }}
            .back-link {{
                display: inline-block;
                margin-bottom: 16px;
            }}
            .placeholder {{
                border: 1px solid #ccc;
                padding: 16px;
                margin-bottom: 12px;
            }}
            .status-table-scroll {{
                max-height: 260px;
                overflow: auto;
                margin-bottom: 28px;
            }}
            .status-table-scroll table {{
                margin-bottom: 0;
            }}
            .status-table-scroll thead th {{
                position: sticky;
                top: 0;
                z-index: 1;
            }}
            .camera-selector {{
                margin-bottom: 12px;
            }}
            .camera-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
            }}
            .camera-view {{
                min-width: 0;
            }}
            .camera-card {{
                border: 1px solid #ccc;
                padding: 12px;
            }}
            .camera-card img {{
                display: block;
                width: 100%;
                max-height: 360px;
                object-fit: contain;
                background: #111;
            }}
            @media (max-width: 900px) {{
                .camera-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            .connection-status {{
                font-weight: 700;
                margin: 8px 0;
            }}
            .connected {{
                color: #16803c;
            }}
            .disconnected {{
                color: #c62828;
            }}
        </style>
    </head>
    <body>
        <h1>Robot status</h1>
        <a class="back-link" href="/">Back to Home</a>

        <h2>Battery state</h2>
        <div class="status-table-scroll">
          <table>
            <thead>
                <tr>
                    <th>robot</th>
                    <th>topic</th>
                    <th>battery</th>
                    <th>voltage</th>
                    <th>current</th>
                    <th>connection</th>
                    <th>updated_at</th>
                </tr>
            </thead>
            <tbody id="battery-status-body">
                {battery_topic_rows}
            </tbody>
          </table>
        </div>

        <h2>Dock status</h2>
        <div class="status-table-scroll">
          <table>
            <thead>
                <tr>
                    <th>robot</th>
                    <th>topic</th>
                    <th>dock status</th>
                    <th>dock visible</th>
                    <th>connection</th>
                    <th>updated_at</th>
                </tr>
            </thead>
            <tbody id="dock-status-body">
                {dock_topic_rows}
            </tbody>
          </table>
        </div>

        <h2>Camera</h2>
        <div class="camera-grid">
          <div class="camera-view">
            <div class="camera-selector">
                <label for="camera-select-1">Camera 1: </label>
                <select id="camera-select-1">{camera_options_1}</select>
            </div>
            <div class="camera-card">
                <h3 id="selected-camera-name-1">Select a camera</h3>
                <p id="selected-camera-topic-1"></p>
                <img id="selected-camera-image-1" class="live-camera" alt="Selected camera 1" hidden>
                <div id="selected-camera-placeholder-1" class="placeholder">Waiting for camera frame</div>
                <p id="selected-camera-status-1" class="connection-status disconnected">Disconnected</p>
                <p>updated_at: <span id="selected-camera-updated-at-1"></span></p>
            </div>
          </div>
          <div class="camera-view">
            <div class="camera-selector">
                <label for="camera-select-2">Camera 2: </label>
                <select id="camera-select-2">{camera_options_2}</select>
            </div>
            <div class="camera-card">
                <h3 id="selected-camera-name-2">Select a camera</h3>
                <p id="selected-camera-topic-2"></p>
                <img id="selected-camera-image-2" class="live-camera" alt="Selected camera 2" hidden>
                <div id="selected-camera-placeholder-2" class="placeholder">Waiting for camera frame</div>
                <p id="selected-camera-status-2" class="connection-status disconnected">Disconnected</p>
                <p>updated_at: <span id="selected-camera-updated-at-2"></span></p>
            </div>
          </div>
        </div>

        <script>
            // 카메라 이미지는 빠르게 갱신해 영상처럼 보이도록 합니다.
            const CAMERA_REFRESH_INTERVAL_MS = 100;
            // 배터리/도킹/연결 상태는 1초마다 갱신합니다.
            const STATUS_REFRESH_INTERVAL_MS = 1000;

            function setCellText(row, text) {{
                // 테이블 행에 텍스트 셀을 하나 추가합니다.
                const cell = document.createElement("td");
                cell.textContent = text;
                row.appendChild(cell);
            }}

            function setConnectionCell(row, connected) {{
                // 연결 여부에 따라 문구와 CSS 클래스를 함께 설정합니다.
                const cell = document.createElement("td");
                cell.textContent = connected ? "Connected" : "Disconnected";
                cell.classList.add("connection-status");
                cell.classList.toggle("connected", connected);
                cell.classList.toggle("disconnected", !connected);
                row.appendChild(cell);
            }}

            function renderBatteryRows(rows) {{
                // JSON으로 받은 배터리 데이터를 테이블에 다시 렌더링합니다.
                const tableBody = document.getElementById("battery-status-body");
                tableBody.replaceChildren();

                rows.forEach((item) => {{
                    // 각 로봇의 배터리 상태를 한 행으로 만듭니다.
                    const row = document.createElement("tr");
                    setCellText(row, item.robot_name);
                    setCellText(row, item.topic_name);
                    setCellText(row, item.percentage);
                    setCellText(row, item.voltage);
                    setCellText(row, item.current);
                    setConnectionCell(row, item.connection_alive);
                    setCellText(row, item.updated_at);
                    tableBody.appendChild(row);
                }});
            }}

            function renderDockRows(rows) {{
                // JSON으로 받은 도킹 데이터를 테이블에 다시 렌더링합니다.
                const tableBody = document.getElementById("dock-status-body");
                tableBody.replaceChildren();

                rows.forEach((item) => {{
                    // 각 로봇의 도킹 상태를 한 행으로 만듭니다.
                    const row = document.createElement("tr");
                    setCellText(row, item.robot_name);
                    setCellText(row, item.topic_name);
                    setCellText(row, item.dock_status);
                    setCellText(row, item.dock_visible);
                    setConnectionCell(row, item.connection_alive);
                    setCellText(row, item.updated_at);
                    tableBody.appendChild(row);
                }});
            }}

            function replaceCameraFrameWhenLoaded(image, imageUrl, updatedAt = "") {{
                // 이전 이미지 요청이 아직 끝나지 않았으면 중복 요청을 만들지 않습니다.
                if (image.dataset.loading === "true") {{
                    return;
                }}

                image.dataset.loading = "true";
                const nextImage = new Image();
                nextImage.onload = () => {{
                    // 새 이미지가 완전히 로드된 뒤에만 화면 이미지를 교체합니다.
                    image.src = nextImage.src;
                    if (updatedAt) {{
                        image.dataset.updatedAt = updatedAt;
                    }}
                    image.hidden = false;
                    image.dataset.loading = "false";
                }};
                nextImage.onerror = () => {{
                    // 이미지 로드 실패 시 다음 주기에 다시 시도할 수 있도록 상태를 해제합니다.
                    image.dataset.loading = "false";
                }};
                // t 쿼리값을 붙여 브라우저 캐시 대신 최신 파일을 가져오게 합니다.
                nextImage.src = `${{imageUrl}}?t=${{Date.now()}}`;
            }}

            let latestCameraRows = [];

            function renderSelectedCamera(slot) {{
                const select = document.getElementById(`camera-select-${{slot}}`);
                const item = latestCameraRows.find(
                    (camera) => camera.robot_name === select.value
                );
                const image = document.getElementById(`selected-camera-image-${{slot}}`);
                const placeholder = document.getElementById(
                    `selected-camera-placeholder-${{slot}}`
                );
                const status = document.getElementById(`selected-camera-status-${{slot}}`);

                if (!item) {{
                    image.hidden = true;
                    placeholder.hidden = false;
                    status.textContent = "Disconnected";
                    status.classList.remove("connected");
                    status.classList.add("disconnected");
                    return;
                }}

                document.getElementById(`selected-camera-name-${{slot}}`).textContent = item.robot_name;
                document.getElementById(`selected-camera-topic-${{slot}}`).textContent = item.topic_name;
                document.getElementById(`selected-camera-updated-at-${{slot}}`).textContent = item.updated_at;
                status.textContent = item.camera_frame_alive ? "Connected" : "Disconnected";
                status.classList.toggle("connected", item.camera_frame_alive);
                status.classList.toggle("disconnected", !item.camera_frame_alive);
                image.dataset.cameraSrc = item.image_url;
                image.dataset.connectionAlive = item.camera_frame_alive ? "true" : "false";
                image.hidden = !item.camera_frame_alive;
                placeholder.hidden = item.camera_frame_alive;
            }}

            function updateCameraStatus(rows) {{
                latestCameraRows = rows;
                const names = new Set(rows.map((item) => item.robot_name));

                [1, 2].forEach((slot) => {{
                    const select = document.getElementById(`camera-select-${{slot}}`);
                    const previousSelection = select.value;
                    Array.from(select.options).forEach((option) => {{
                        if (!names.has(option.value)) {{
                            option.remove();
                        }}
                    }});
                    rows.forEach((item) => {{
                        if (!Array.from(select.options).some(
                            (option) => option.value === item.robot_name
                        )) {{
                            select.add(new Option(item.robot_name, item.robot_name));
                        }}
                    }});
                    if (names.has(previousSelection)) {{
                        select.value = previousSelection;
                    }} else if (rows.length > 0) {{
                        select.value = rows[Math.min(slot - 1, rows.length - 1)].robot_name;
                    }}
                    renderSelectedCamera(slot);
                }});
            }}

            function refreshSelectedCameraFrames() {{
                [1, 2].forEach((slot) => {{
                    const image = document.getElementById(`selected-camera-image-${{slot}}`);
                    if (
                        image.dataset.connectionAlive === "true" &&
                        image.dataset.cameraSrc
                    ) {{
                        replaceCameraFrameWhenLoaded(image, image.dataset.cameraSrc);
                    }}
                }});
            }}

            [1, 2].forEach((slot) => {{
                document.getElementById(`camera-select-${{slot}}`).addEventListener(
                    "change", () => renderSelectedCamera(slot)
                );
            }});

            async function refreshRobotStatus() {{
                // Flask JSON API에서 최신 로봇 상태를 받아옵니다.
                const response = await fetch(`/robot_status_data?t=${{Date.now()}}`, {{
                    cache: "no-store"
                }});

                // 응답 오류가 있으면 이번 갱신만 건너뜁니다.
                if (!response.ok) {{
                    return;
                }}

                // 받은 JSON으로 테이블과 카메라 상태를 갱신합니다.
                const data = await response.json();
                renderBatteryRows(data.battery);
                renderDockRows(data.dock);
                updateCameraStatus(data.camera);
            }}

            // 페이지 로드 직후 한 번 갱신하고, 이후에는 정해진 주기로 반복합니다.
            refreshRobotStatus();
            refreshSelectedCameraFrames();
            setInterval(refreshRobotStatus, STATUS_REFRESH_INTERVAL_MS);
            setInterval(refreshSelectedCameraFrames, CAMERA_REFRESH_INTERVAL_MS);
        </script>
    </body>
    </html>
    """


def main():
    # 이 파일을 직접 실행하면 Flask 개발 서버를 시작합니다.
    app.run(host=HOST, port=PORT, debug=True)


if __name__ == "__main__":
    main()
