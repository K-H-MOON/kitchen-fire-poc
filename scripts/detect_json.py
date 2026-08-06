"""탐지 JSON 생성 — 파이프라인 5단계.

프레임 폴더 또는 영상을 받아 규격(docs/DETECT_SCHEMA.md)에 맞는 JSON Lines 를 냄.

  python scripts/detect_json.py --frames eval_nonhyeon --weights round10_best.pt \
      --camera gaewon-01 --stations stations/gaewon-01.json --out log.jsonl

화구 영역 파일이 없으면 station_id 는 전부 null 이 되며, 그래도 동작함
(손으로 든 카메라처럼 고정 영역을 정의할 수 없는 경우).
"""
import os, json, glob, argparse
from datetime import datetime, timedelta, timezone

SCHEMA = 'kitchen-fire-detect/1.0'


def load_stations(path, frame_wh):
    """화구 영역 로드. 프레임 크기가 다르면 비례 축소"""
    if not path or not os.path.exists(path):
        return [], None
    cfg = json.load(open(path, encoding='utf-8'))
    fw, fh = cfg.get('frame_size', frame_wh)
    sx, sy = frame_wh[0] / fw, frame_wh[1] / fh
    out = []
    for s in cfg['stations']:
        x0, y0, x1, y1 = s['bbox_xyxy']
        out.append({**s, 'bbox_xyxy': [x0 * sx, y0 * sy, x1 * sx, y1 * sy]})
    return out, cfg


def assign_station(box, stations):
    """박스 중심이 속한 화구. 어디에도 안 속하면 (None, 0.0)"""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    best, best_ov = None, 0.0
    for s in stations:
        a, b, c, d = s['bbox_xyxy']
        ov = (max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))) / area
        if a <= cx <= c and b <= cy <= d and ov > best_ov:
            best, best_ov = s['id'], ov
    return best, round(best_ov, 3)


class EventTracker:
    """탐지가 이어지는 동안 하나의 사건으로 묶음. gap 안에 다시 잡히면 같은 사건"""

    def __init__(self, camera_id, gap_s=10.0):
        self.cam, self.gap = camera_id, gap_s
        self.eid = self.first_ts = None
        self.last_hit = None
        self.max_conf = self.max_area = 0.0

    def update(self, t, dets):
        if not dets:
            if self.eid and self.last_hit is not None and (t - self.last_hit) > self.gap:
                self.eid = None
                self.max_conf = self.max_area = 0.0
            return None
        if self.eid is None:
            self.first_ts = t
            self.eid = f'{self.cam}-{iso(t).replace("-", "").replace(":", "")[:15]}'
            self.max_conf = self.max_area = 0.0
        self.last_hit = t
        self.max_conf = max(self.max_conf, max(d['conf'] for d in dets))
        self.max_area = max(self.max_area, max(d['area_ratio'] for d in dets))
        return {
            'raised': True,
            'event_id': self.eid,
            'first_seen_ts': iso(self.first_ts),
            'duration_s': round(t - self.first_ts, 2),
            'max_conf_in_event': round(self.max_conf, 3),
            'max_area_ratio_in_event': round(self.max_area, 4),
        }


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(t):
    return (T0 + timedelta(seconds=t)).isoformat().replace('+00:00', 'Z')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', help='프레임 폴더 (파일명 순서대로 읽음)')
    ap.add_argument('--video', help='영상 파일 (--fps 로 추출)')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--camera', default='cam-01')
    ap.add_argument('--stations', help='화구 영역 json. 없으면 station_id 전부 null')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--fps', type=float, default=1.0, help='프레임 간 시간 간격 계산용')
    ap.add_argument('--start-ts', type=float, default=0.0, help='첫 프레임의 초')
    ap.add_argument('--gap', type=float, default=10.0, help='사건 병합 허용 공백(초)')
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    if a.video:
        tmp = '/tmp/_dj_frames'
        os.system(f'rm -rf {tmp} && mkdir -p {tmp}')
        os.system(f'ffmpeg -v error -i "{a.video}" -vf fps={a.fps} -q:v 2 {tmp}/%06d.jpg')
        paths = sorted(glob.glob(f'{tmp}/*.jpg'))
    else:
        paths = sorted(glob.glob(f'{a.frames}/*.jpg') + glob.glob(f'{a.frames}/*.png'))
    assert paths, '프레임을 찾을 수 없습니다'

    h, w = cv2.imread(paths[0]).shape[:2]
    stations, _ = load_stations(a.stations, (w, h))
    print(f'프레임 {len(paths)}장 · {w}x{h} · 화구 {len(stations)}개 '
          f'{[s["id"] for s in stations]}')

    m = YOLO(a.weights)
    tracker = EventTracker(a.camera, a.gap)
    model_info = {'weights': os.path.basename(a.weights), 'imgsz': a.imgsz,
                  'conf_th': a.conf}

    n_frame_hit = n_alarm_start = 0
    seen_events = set()
    with open(a.out, 'w', encoding='utf-8') as fp:
        for i, p in enumerate(paths):
            t = a.start_ts + i / a.fps
            r = m.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            dets = []
            for b, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = [float(v) for v in b]
                sid, ov = assign_station((x0, y0, x1, y1), stations)
                dets.append({
                    'class': 'fire',
                    'conf': round(float(cf), 3),
                    'bbox_xyxy': [round(x0), round(y0), round(x1), round(y1)],
                    'bbox_norm': [round(x0 / w, 4), round(y0 / h, 4),
                                  round(x1 / w, 4), round(y1 / h, 4)],
                    'area_ratio': round((x1 - x0) * (y1 - y0) / (w * h), 4),
                    'station_id': sid,
                    'station_overlap': ov,
                })
            if dets:
                n_frame_hit += 1
            alarm = tracker.update(t, dets) or {'raised': False}
            if alarm.get('event_id') and alarm['event_id'] not in seen_events:
                seen_events.add(alarm['event_id']); n_alarm_start += 1
            rec = {
                'schema': SCHEMA, 'camera_id': a.camera, 'ts': iso(t),
                'frame_seq': i, 'model': model_info, 'detections': dets,
                'temporal': {'rule': '1-of-1', 'window_n': 1,
                             'hits_in_window': 1 if dets else 0},
                'alarm': alarm,
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'-> {a.out}')
    print(f'탐지된 프레임 {n_frame_hit}/{len(paths)} · 사건 {n_alarm_start}건')
    if stations:
        from collections import Counter
        c = Counter()
        for line in open(a.out, encoding='utf-8'):
            for d in json.loads(line)['detections']:
                c[d['station_id'] or '(화구 밖)'] += 1
        print('화구별 탐지 수:', dict(c))


if __name__ == '__main__':
    main()
