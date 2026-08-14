"""
safety_lib.global_fusion
========================

카메라 간 동일 인격 통합 (track association).

문제:
    cam0에서 "cam0:7", cam1에서 "cam1:3"이라 부르는 두 detection이
    실제로는 같은 사람일 수 있다. map 좌표 상에서 가까이 있으면 같은 사람.

해결:
    매 프레임 detection들에 global_id를 부여해서 카메라 넘어 동일 인격
    으로 인식. GlobalFuser는 3단계 매칭으로 이걸 판정한다.

    1) REBIND  : 이전 프레임에 이미 매핑된 local_key는 그대로 유지 (sticky).
                 local_key 가 같으면 = 같은 카메라의 같은 ByteTrack 트랙이면
                 거리와 무관하게 유지한다. 발 좌표는 지평선 근처에서 한 프레임에
                 1m 넘게 튈 수 있는데, 그때마다 인격이 갈리면 지도에 점이 두 개
                 찍히고 침입자 딱지도 날아간다.
    2) NEAREST : 나머지는 기존 global track과 거리(match_dist=0.65m) 검사
    3) NEW     : 매칭 실패한 detection에 신규 global_id 부여

    이후 같은 global_id로 매칭된 detection들의 map_xy를 가중 평균으로 융합.

    그리고 unauthorized / access_granted 는 카메라가 아니라 '그 사람'의
    신분이므로 global track 에 sticky 로 기억한다 (카메라 전환에도 유지).

파라미터:
    match_dist  : 새 detection과 기존 track 간 매칭 최대 거리 [m]
                  캘리브 오차 ~7cm의 9배 여유
    rebind_dist : 이 거리를 넘는 REBIND 는 좌표가 튄 것으로 보고 로그용
                  플래그(rebind_jump)만 남긴다. gid 는 그대로 유지한다.
    max_age_sec : 소실 후 track 유지 시간 [s]. 카메라 간 이동 중 잠깐 안 보이는
                  구간이 있으므로 넉넉해야 한다 (침입자 딱지를 잃지 않으려면).
    ema_alpha   : 위치 EMA 스무딩 (1.0=off, 0.3=부드럽게)

    4) MERGE   : 서로 다른 카메라만 보고 있고 merge_dist 이내인 두 global track
                 을 하나로 합침. 한 프레임의 나쁜 좌표(가장자리 외삽 등)로
                 갈라진 gid 를 되돌린다. 이게 없으면 REBIND 가 sticky 라서
                 영원히 두 개로 남는다.

    4b) RECLAIM: 그 카메라가 지금 안 보고 있는 global track 이 reclaim_dist(3m)
                 이내에 있으면 되찾는다. "입구로 들어온 사람이 없으니 지금 나타난
                 이 사람은 아까 그 사람이다" 라는 추론. 이게 없으면 cam1 에 오래
                 머문 뒤 cam0 로 돌아올 때 침입자 딱지가 날아간다.

    5) HANDOFF : 두 카메라 시야가 겹치지 않는 배치에서, 사람이 cam1 -> cam0 으로
                 넘어갈 때 cam1 의 hold 잔영(0.8s)이 지도에 두 번째 점으로 남는다.
                 held detection 이 '다른 카메라' 의 live detection 과 handoff_dist
                 이내면 그 잔영을 버린다.

한계:
    * 첫 프레임 co-observation 은 NEW pass 가 순차 처리라 그 프레임에는 못 합치지만,
      다음 프레임의 MERGE pass 가 즉시 합친다.
    * ros_bridge / dashboard 의 위치 근접 dedup(0.5m)이 추가 안전망.
"""
import numpy as np


class GlobalFuser:
    """
    카메라 간 인격 통합기.

    Usage:
        fuser = GlobalFuser()
        while running:
            all_detections, summary = fuser.update(all_detections, now)
            # 각 det에 global_id / global_map_xy 필드가 붙어서 나옴
    """
    def __init__(
        self,
        match_dist=0.65,
        rebind_dist=1.20,
        max_age_sec=8.0,
        ema_alpha=0.40,
        merge_dist=0.60,
        handoff_dist=1.50,
        reclaim_dist=3.00,
    ):
        """
        Args:
            match_dist  : NEAREST pass에서 신규 매칭 허용 최대 거리 [m]
            rebind_dist : REBIND pass에서 sticky 유지 최대 거리 [m]
            max_age_sec : 소실 track 삭제까지 대기 [s]
            ema_alpha   : 융합 좌표 EMA 계수. 1.0=raw 그대로, 0<α<1이면
                          이전값과 α:(1-α) 비율로 블렌딩 (스무딩).
        """
        self.match_dist = float(match_dist)
        self.rebind_dist = float(rebind_dist)
        self.max_age_sec = float(max_age_sec)
        self.ema_alpha = float(ema_alpha)
        self.merge_dist = float(merge_dist)
        self.handoff_dist = float(handoff_dist)
        self.reclaim_dist = float(reclaim_dist)

        # 현재 살아있는 전역 트랙: {gid: {gid, map_xy, source_keys, ...}}
        self.tracks = {}
        # 로컬 track_id → 전역 gid 매핑 캐시 (REBIND용)
        self.local_to_global = {}
        # 다음에 부여할 gid 카운터 (증가만 함)
        self.next_gid = 0

    def _new_track(self, map_xy, now):
        """신규 전역 track 하나 생성. gid는 monotonically increasing."""
        gid = self.next_gid
        self.next_gid += 1

        self.tracks[gid] = {
            "gid": gid,
            "global_id": f"G{gid}",           # 문자열 라벨 ("G0", "G1", ...)
            "first_seen": now,
            "last_seen": now,
            "map_xy": np.array(map_xy, dtype=np.float64),         # 융합 좌표
            "last_raw_map_xy": np.array(map_xy, dtype=np.float64), # 이번 프레임 raw
            "source_keys": set(),      # 이 track에 붙은 로컬 keys
            "cameras": set(),          # 이 track이 관찰된 카메라들
            "last_assigned_keys": [],

            # ── 사람에게 붙는 sticky 신분 ────────────────────────────
            # unauthorized 는 카메라가 아니라 '그 사람'의 속성이다.
            # cam0 의 WorkerStateStore 에 두면, 침입자가 cam1 구역으로
            # 넘어가는 순간 cam0 트랙이 만료되고(max_age 3s) 돌아왔을 때
            # 새 트랙이 되어 침입 이력이 사라진다.
            # global track 은 카메라를 넘어 살아있으므로 여기에 기억한다.
            "sticky_unauthorized": False,
            "sticky_access_granted": False,
            "unauthorized_since": None,
        }

        return gid

    def _local_key(self, det):
        """detection의 로컬 식별자 반환. WorkerStateStore의 state_key와 동일."""
        state_key = det.get("state_key")
        if state_key is not None:
            return str(state_key)

        cam = det.get("camera", "cam")
        yid = det.get("yolo_track_id")

        if yid is None:
            return None

        return f"{cam}:{int(yid)}"

    def _valid_map_xy(self, det):
        """detection의 map_xy를 float64 (2,) ndarray로 정규화. NaN/inf 방어."""
        map_xy = det.get("map_xy")

        if map_xy is None:
            return None

        arr = np.array(map_xy, dtype=np.float64).reshape(-1)

        if len(arr) < 2:
            return None

        arr = arr[:2]

        if not np.all(np.isfinite(arr)):
            return None

        return arr

    def _distance(self, a, b):
        return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))

    def _cleanup(self, now):
        """
        max_age_sec 넘게 안 보인 전역 track 제거 + 고아 로컬 매핑 정리.
        메모리 누적 방지.
        """
        dead_gids = []

        for gid, tr in self.tracks.items():
            if now - tr["last_seen"] > self.max_age_sec:
                dead_gids.append(gid)

        for gid in dead_gids:
            del self.tracks[gid]

        # 삭제된 gid를 가리키던 로컬 매핑도 함께 정리
        dead_local = []

        for local_key, gid in self.local_to_global.items():
            if gid not in self.tracks:
                dead_local.append(local_key)

        for key in dead_local:
            del self.local_to_global[key]

    def _merge_close_tracks(self, assignments, frame_cameras, now):
        """
        서로 가까운 전역 track 을 하나로 합친다.

        ── 왜 필요한가 ──
        cam0 가 처음 그 사람을 잡는 프레임에서 좌표가 match_dist(0.65m) 밖으로
        어긋나면 (화면 가장자리 외삽, 발 키포인트 흔들림) NEW pass 가 새 gid 를
        만든다. 그 뒤 좌표가 0.1m 로 수렴해도 REBIND 가 sticky 라서 두 gid 가
        영원히 분리된 채 남는다 → 지도에 점이 두 개.

        그래서 매 프레임, 이번 프레임에 갱신된 track 들 중
          * 거리가 merge_dist 이내이고
          * 서로 다른 카메라에서만 관측된 (카메라 집합이 겹치지 않는)
        쌍을 같은 사람으로 보고 합친다.

        같은 카메라가 둘 다 보고 있으면 실제로 두 사람이 가까이 선 것이므로
        합치지 않는다.
        """
        if self.merge_dist <= 0.0:
            return assignments

        live = [gid for gid in self.tracks if gid in frame_cameras]

        merged = True
        while merged:
            merged = False
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    a, b = live[i], live[j]
                    if a not in self.tracks or b not in self.tracks:
                        continue
                    if frame_cameras[a] & frame_cameras[b]:
                        continue   # 같은 카메라가 둘 다 봄 → 다른 사람
                    if self._distance(self.tracks[a]["map_xy"], self.tracks[b]["map_xy"]) > self.merge_dist:
                        continue

                    keep, drop = (a, b) if a < b else (b, a)   # 오래된 gid 유지
                    tk, td = self.tracks[keep], self.tracks[drop]

                    tk["map_xy"] = (tk["map_xy"] + td["map_xy"]) * 0.5
                    tk["source_keys"] |= td["source_keys"]
                    tk["cameras"] |= td["cameras"]
                    tk["first_seen"] = min(tk["first_seen"], td["first_seen"])
                    tk["last_seen"] = max(tk["last_seen"], td["last_seen"])

                    # sticky 신분은 OR 로 흡수 (침입자 딱지를 잃으면 안 된다)
                    tk["sticky_unauthorized"] = tk["sticky_unauthorized"] or td["sticky_unauthorized"]
                    tk["sticky_access_granted"] = tk["sticky_access_granted"] or td["sticky_access_granted"]
                    if tk["unauthorized_since"] is None:
                        tk["unauthorized_since"] = td["unauthorized_since"]

                    frame_cameras[keep] |= frame_cameras.pop(drop, set())

                    for lk, g in list(self.local_to_global.items()):
                        if g == drop:
                            self.local_to_global[lk] = keep
                    for idx, a_ in assignments.items():
                        if a_["gid"] == drop:
                            a_["gid"] = keep
                            a_["mode"] = "MERGED"

                    del self.tracks[drop]
                    live = [g for g in live if g != drop]
                    merged = True
                    break
                if merged:
                    break

        return assignments

    def update(self, detections, now):
        """
        매 프레임 호출되는 메인 진입점.

        Args:
            detections : list of dict (cam0_tracked + cam1_tracked)
            now        : time.time()
        Returns:
            (detections, summary)
              detections는 각 dict에 global_id/global_map_xy/global_track_state
              등이 추가된 상태로 그대로 반환됨 (in-place mutation).
              summary는 현재 살아있는 전역 track의 리스트.
        """
        # 0. 오래된 track 제거
        self._cleanup(now)

        # 1. 유효 detection만 valid에 담고, 나머지엔 NO_GLOBAL 라벨
        valid = []

        for idx, det in enumerate(detections):
            map_xy = self._valid_map_xy(det)
            local_key = self._local_key(det)

            # map_xy 없거나 local_key 없으면 융합 대상 아님
            if map_xy is None or local_key is None:
                det["global_id"] = None
                det["global_numeric_id"] = None
                det["global_map_xy"] = None
                det["global_track_state"] = "NO_GLOBAL"
                det["global_match_dist"] = None
                continue

            valid.append(
                {
                    "idx": idx,
                    "det": det,
                    "map_xy": map_xy,
                    "local_key": local_key,
                    "camera": det.get("camera", "cam"),
                    "is_held": bool(det.get("is_held", False)),
                }
            )

        # live detection 을 held 보다 먼저 처리한다.
        # held 는 이미 소실된 트랙의 잔영이므로, (gid, camera) 슬롯을 두고
        # 경쟁하면 살아있는 쪽이 이겨야 한다. (id switch 시 유령이 gid 를
        # 선점해 진짜에게 새 global_id 가 생기는 것을 막는다)
        valid.sort(key=lambda it: it["is_held"])

        # ---------- 1-b. 카메라 간 핸드오프: held 유령 제거 ----------
        #
        # 두 카메라의 시야가 겹치지 않으면 (실측: 겹치는 면적 0 m2) 사람이
        # cam1 -> cam0 으로 넘어갈 때 "동시에 보이는" 프레임이 없다.
        #
        #   cam1 에서 사라짐  →  (공백)  →  cam0 에서 새로 등장
        #
        # 그런데 WorkerStateStore 가 hold_sec(0.8s) 동안 마지막 detection 을
        # 되살려 넣으므로, 그 0.8초간 지도에 점이 두 개 찍힌다:
        #   cam1 의 유령(HELD, 마지막 위치)  +  cam0 의 진짜(live)
        #
        # MERGE pass 는 '동시에 보일 때' 만 작동하므로 이걸 못 잡는다.
        # add_held_detections 의 IoU 억제도 같은 카메라 안에서만 유효하다
        # (다른 카메라는 화면 좌표계가 달라 bbox 비교가 무의미).
        #
        # 그래서 여기서 처리한다: held detection 이 '다른 카메라'의 live
        # detection 과 handoff_dist 이내면 그건 넘어간 사람의 잔영이므로 버린다.
        # 같은 카메라 안의 occlusion hold 는 건드리지 않는다.
        live_by_cam = {}
        for item in valid:
            if not item["is_held"]:
                live_by_cam.setdefault(item["camera"], []).append(item["map_xy"])

        if live_by_cam and self.handoff_dist > 0.0:
            kept = []
            for item in valid:
                if item["is_held"]:
                    ghost = False
                    for cam, pts in live_by_cam.items():
                        if cam == item["camera"]:
                            continue   # 같은 카메라 = 진짜 occlusion hold
                        if any(self._distance(item["map_xy"], p) <= self.handoff_dist
                               for p in pts):
                            ghost = True
                            break
                    if ghost:
                        det = detections[item["idx"]]
                        det["global_id"] = None
                        det["global_numeric_id"] = None
                        det["global_map_xy"] = None
                        det["global_track_state"] = "HANDOFF_GHOST"
                        det["global_match_dist"] = None
                        det["is_handoff_ghost"] = True
                        continue
                kept.append(item)
            valid = kept

        # assignments[idx] = {"gid":..., "dist":..., "mode": REBIND/NEAREST/NEW}
        assignments = {}
        # 같은 (gid, camera) 쌍이 두 번 배정되지 않도록 (한 카메라가 같은
        # 사람에 대해 detection을 두 개 낸 병리적 케이스 방어)
        used_global_camera = set()

        # ---------- 2. REBIND pass ----------
        # 이전 프레임에 이미 local_key → gid 매핑이 있으면 거리만 확인 후 유지.
        # rebind_dist(1.2m)까지 허용해서 순간적인 좌표 튐을 흡수.
        for item in valid:
            local_key = item["local_key"]
            camera = item["camera"]
            map_xy = item["map_xy"]

            if local_key not in self.local_to_global:
                continue

            gid = self.local_to_global[local_key]

            if gid not in self.tracks:
                continue

            tr = self.tracks[gid]
            dist = self._distance(map_xy, tr["map_xy"])

            if (gid, camera) in used_global_camera:
                continue

            # ── local_key 가 같으면 거리와 무관하게 REBIND 한다 ──
            #
            # local_key = "cam1:3" = 같은 카메라의 같은 ByteTrack 트랙.
            # ByteTrack 이 같은 id 를 유지한다는 것은 그 카메라가 그 사람을
            # 끊김 없이 추적하고 있다는 뜻이므로, 확실히 같은 사람이다.
            #
            # 예전에는 rebind_dist(1.2m) 를 넘으면 gid 를 버리고 새로 만들었다.
            # 그런데 발 좌표는 지평선 근처나 가장자리에서 한 프레임에 1m 넘게
            # 튈 수 있다. 그때마다 인격이 갈리면서:
            #   * 지도에 점이 두 개 (옛 track 이 max_age 동안 남음)
            #   * sticky_unauthorized(침입자 딱지)가 날아감
            #   * cam1 -> cam0 복귀 시 cam0 가 '처음 보는 사람' 이 됨
            #
            # 좌표가 튀었다는 이유로 사람의 정체성을 바꾸면 안 된다.
            # 대신 튄 좌표는 EMA 가 흡수한다.
            if dist > self.rebind_dist:
                tr["rebind_jump"] = float(dist)

            assignments[item["idx"]] = {
                "gid": gid,
                "dist": dist,
                "mode": "REBIND" if dist <= self.rebind_dist else "REBIND_JUMP",
            }
            used_global_camera.add((gid, camera))

        # ---------- 3. NEAREST match pass ----------
        # REBIND 안 된 detection에 대해 모든 기존 track과 거리 계산 →
        # match_dist(0.65m) 이하 후보를 모두 수집, 거리 오름차순 정렬 후
        # 앞에서부터 배정 (탐욕적 최근접 배정).
        candidates = []

        for item in valid:
            if item["idx"] in assignments:
                continue

            camera = item["camera"]
            map_xy = item["map_xy"]

            for gid, tr in self.tracks.items():
                if (gid, camera) in used_global_camera:
                    continue

                dist = self._distance(map_xy, tr["map_xy"])

                if dist <= self.match_dist:
                    candidates.append((dist, item["idx"], gid))

        candidates.sort(key=lambda x: x[0])  # 가까운 것 우선

        assigned_indices = set(assignments.keys())

        for dist, idx, gid in candidates:
            if idx in assigned_indices:
                continue

            det = detections[idx]
            camera = det.get("camera", "cam")

            if (gid, camera) in used_global_camera:
                continue

            assignments[idx] = {
                "gid": gid,
                "dist": dist,
                "mode": "NEAREST",
            }
            assigned_indices.add(idx)
            used_global_camera.add((gid, camera))

        # ---------- 3-b. RECLAIM pass ----------
        #
        # "입구로 들어온 사람이 없으니, 지금 나타난 이 사람은 아까 그 사람이다"
        #
        # 사람이 cam1 구역에 오래 머무르면 cam0 의 WorkerStateStore 트랙이
        # max_age(3s) 로 만료된다. 그 뒤 cam0 로 돌아오면 ByteTrack 이 새 id 를
        # 주므로 local_key("cam0:7")가 local_to_global 에 없다 → REBIND 불가.
        # 그리고 두 카메라 시야가 거의 안 겹쳐서 복귀 지점과 마지막 관측 위치가
        # 1.5m 이상 떨어져 있다 → NEAREST(0.65m)도 실패 → NEW.
        #
        # 그러면 침입자 딱지(sticky_unauthorized)가 통째로 날아간다.
        #
        # 그래서 마지막 방어선: 지금 이 카메라가 보고 있지 않은 global track 중
        # reclaim_dist 이내에 있는 것을 되찾는다. 그 카메라가 그 track 을
        # 이번 프레임에 안 보고 있다는 건, 그 사람이 방금 이 카메라 시야로
        # 넘어왔다는 뜻이다.
        #
        # 같은 카메라가 이미 보고 있는 track 은 건드리지 않는다 (다른 사람이다).
        if self.reclaim_dist > 0.0:
            reclaim_cands = []

            for item in valid:
                if item["idx"] in assignments:
                    continue

                camera = item["camera"]
                map_xy = item["map_xy"]

                for gid, tr in self.tracks.items():
                    if (gid, camera) in used_global_camera:
                        continue
                    # 이번 프레임에 이 카메라가 그 track 을 이미 보고 있으면 스킵
                    if camera in tr["cameras"] and any(
                        a["gid"] == gid for a in assignments.values()
                    ):
                        continue

                    dist = self._distance(map_xy, tr["map_xy"])
                    if dist <= self.reclaim_dist:
                        reclaim_cands.append((dist, item["idx"], gid))

            reclaim_cands.sort(key=lambda x: x[0])

            for dist, idx, gid in reclaim_cands:
                if idx in assignments:
                    continue
                camera = detections[idx].get("camera", "cam")
                if (gid, camera) in used_global_camera:
                    continue

                assignments[idx] = {"gid": gid, "dist": dist, "mode": "RECLAIM"}
                used_global_camera.add((gid, camera))

        # ---------- 4. NEW pass ----------
        # 여전히 매칭 안 된 detection → 신규 gid 생성.
        # ⚠ 여기서 방금 만든 track을 뒤 detection이 참조 못 함 (loop 내부).
        #    이게 "첫 프레임 co-observation 병합 못 함" 한계의 원인.
        for item in valid:
            idx = item["idx"]

            if idx in assignments:
                continue

            # held detection 에는 새 global_id 를 만들지 않는다.
            # 잔영일 뿐이므로 붙을 track 이 없으면 그냥 버린다.
            if item["is_held"]:
                det = detections[idx]
                det["global_id"] = None
                det["global_numeric_id"] = None
                det["global_map_xy"] = None
                det["global_track_state"] = "HELD_ORPHAN"
                det["global_match_dist"] = None
                continue

            gid = self._new_track(item["map_xy"], now)

            assignments[idx] = {
                "gid": gid,
                "dist": 0.0,
                "mode": "NEW",
            }

            used_global_camera.add((gid, item["camera"]))

        # ---------- 5. 같은 gid로 배정된 detection들 좌표 융합 ----------
        # 두 카메라가 같은 사람을 잡아 같은 gid에 붙었다면 두 좌표의
        # 가중 평균으로 융합 (held track은 신뢰도 낮음 → weight 0.45).
        grouped = {}

        for item in valid:
            idx = item["idx"]

            if idx not in assignments:
                continue

            gid = assignments[idx]["gid"]
            grouped.setdefault(gid, []).append(item)

        for gid, items in grouped.items():
            if gid not in self.tracks:
                continue

            tr = self.tracks[gid]

            weighted_points = []
            weights = []
            source_keys = []
            cameras = []

            for item in items:
                map_xy = item["map_xy"]

                # held detection은 이미 소실된 track의 잔영이므로 weight 낮춤
                w = 0.45 if item["is_held"] else 1.0

                weighted_points.append(map_xy)
                weights.append(w)
                source_keys.append(item["local_key"])
                cameras.append(item["camera"])

            pts = np.stack(weighted_points, axis=0)
            ws = np.array(weights, dtype=np.float64)
            ws = ws / max(1e-9, ws.sum())  # weight 정규화

            fused_raw = np.sum(pts * ws[:, None], axis=0)  # 가중 평균

            # 선택적 EMA로 이전 융합값과 블렌딩 (진동 감소)
            prev = tr["map_xy"]
            alpha = self.ema_alpha

            if alpha >= 0.999:
                fused = fused_raw  # EMA off (기본)
            else:
                fused = alpha * fused_raw + (1.0 - alpha) * prev

            tr["last_seen"] = now
            tr["last_raw_map_xy"] = fused_raw
            tr["map_xy"] = fused
            tr["source_keys"] = set(source_keys)
            tr["cameras"] = set(cameras)
            tr["last_assigned_keys"] = source_keys

            # ── sticky 신분 갱신 ──────────────────────────────────
            # 어느 한 카메라라도 unauthorized 를 봤으면 그 사람은 침입자다.
            # 입장 허가(access_granted)를 받으면 침입자 딱지가 풀린다.
            for item in items:
                d = item["det"]
                if d.get("access_granted"):
                    tr["sticky_access_granted"] = True
                    tr["sticky_unauthorized"] = False
                    tr["unauthorized_since"] = None
                elif d.get("unauthorized"):
                    if not tr["sticky_unauthorized"]:
                        tr["unauthorized_since"] = now
                    tr["sticky_unauthorized"] = True

            # local_to_global 매핑 갱신 (다음 프레임 REBIND용)
            for local_key in source_keys:
                self.local_to_global[local_key] = gid

        # ---------- 5-b. 가까운 전역 track 병합 ----------
        # 한 프레임의 나쁜 좌표가 만든 중복 gid 를 여기서 되돌린다.
        frame_cameras = {}
        for item in valid:
            idx = item["idx"]
            if idx in assignments:
                frame_cameras.setdefault(assignments[idx]["gid"], set()).add(item["camera"])

        assignments = self._merge_close_tracks(assignments, frame_cameras, now)

        # ---------- 6. detection dict에 결과 필드 attach ----------
        # save_state_json / ros_bridge가 이 필드들을 읽어서 dedup 판단.
        for item in valid:
            idx = item["idx"]

            if idx not in assignments:
                continue

            det = detections[idx]
            gid = assignments[idx]["gid"]

            if gid not in self.tracks:
                continue

            tr = self.tracks[gid]

            det["global_numeric_id"] = int(gid)
            det["global_id"] = tr["global_id"]                      # "G0", "G1", ...
            det["global_map_xy"] = tr["map_xy"].copy()               # 융합 좌표
            det["global_raw_map_xy"] = tr["last_raw_map_xy"].copy()  # 융합 전 raw
            det["global_track_state"] = assignments[idx]["mode"]     # REBIND/NEAREST/NEW
            det["global_match_dist"] = float(assignments[idx]["dist"])
            det["global_sources"] = sorted(list(tr["source_keys"]))
            det["global_cameras"] = sorted(list(tr["cameras"]))

            # sticky 신분을 detection 에 되돌려 쓴다.
            # cam1(MONITOR_ONLY) 의 detection 도 이제 침입자로 보인다.
            det["unauthorized"] = bool(tr["sticky_unauthorized"])
            det["access_granted"] = bool(tr["sticky_access_granted"])
            det["unauthorized_elapsed"] = (
                0.0 if tr["unauthorized_since"] is None else float(now - tr["unauthorized_since"])
            )

        return detections, self.summary(now)

    def summary(self, now):
        """살아있는 전역 track 목록을 JSON serializable 형태로."""
        out = []

        for gid in sorted(self.tracks.keys()):
            tr = self.tracks[gid]

            out.append(
                {
                    "global_id": tr["global_id"],
                    "global_numeric_id": int(gid),
                    "map_xy": [float(tr["map_xy"][0]), float(tr["map_xy"][1])],
                    "raw_map_xy": [float(tr["last_raw_map_xy"][0]), float(tr["last_raw_map_xy"][1])],
                    "age_sec": float(now - tr["last_seen"]),
                    "first_seen": float(tr["first_seen"]),
                    "last_seen": float(tr["last_seen"]),
                    "unauthorized": bool(tr["sticky_unauthorized"]),
                    "access_granted": bool(tr["sticky_access_granted"]),
                    "source_keys": sorted(list(tr["source_keys"])),
                    "cameras": sorted(list(tr["cameras"])),
                    "last_assigned_keys": list(tr["last_assigned_keys"]),
                }
            )

        return out

    def reset(self):
        """모든 전역 track과 매핑 초기화. 데모 재시작용."""
        self.tracks = {}
        self.local_to_global = {}
        self.next_gid = 0
