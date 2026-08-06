# ===== 파이프라인 5단계 — 탐지 JSON 예시 로그 생성 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요. 약 3분.
# 업로드할 파일 4개:
#   round10_best.pt · kitchen-fire-poc.zip · assets_1_bases.zip · mofn_seq.zip
#
# 규격: docs/DETECT_SCHEMA.md
#
# 두 가지 로그를 만듦
#   (1) 개원중 CCTV 정상 조리 75장 — 고정 카메라, 화구 영역 정의됨
#   (2) 화재 사건 evt5 (29초) — 화구 영역 없음(다른 카메라), station_id 는 null

!pip -q install ultralytics==8.3.*

import zipfile, os, glob, json
from google.colab import files

up = files.upload()
os.makedirs('/content/work', exist_ok=True)
for n in [k for k in up if k.endswith('.zip')]:
    zipfile.ZipFile(n).extractall('/content/work')
W = [k for k in up if k.endswith('.pt')][0]
os.chdir('/content/work')
print('가중치:', W)
print('개원중 CCTV', len(glob.glob('assets/eval_neg/*.jpg')), '장')
print('화재 evt5   ', len(glob.glob('mofn/fire/evt5/*.jpg')), '장')

# (1) 정상 조리 — 화구 영역 적용
print('\n' + '=' * 70)
print('[1] 개원중 CCTV 정상 조리 (고정 카메라, 화구 영역 정의됨)')
print('=' * 70)
!python scripts/detect_json.py --frames assets/eval_neg \
    --weights "/content/work/{W}" --camera gaewon-01 \
    --stations stations/gaewon-01.json --out /content/log_normal.jsonl --fps 1

# (2) 화재 사건 — 화구 영역 없음
print('\n' + '=' * 70)
print('[2] 화재 사건 evt5 (다른 카메라이므로 화구 영역 없음)')
print('=' * 70)
!python scripts/detect_json.py --frames mofn/fire/evt5 \
    --weights "/content/work/{W}" --camera kfire01 \
    --out /content/log_fire.jsonl --fps 1

# 예시 한 줄씩 보기 좋게 출력
def show(path, title, want_alarm):
    print('\n' + '=' * 70); print(title); print('=' * 70)
    lines = [json.loads(l) for l in open(path, encoding='utf-8')]
    pick = next((r for r in lines if r['alarm']['raised'] == want_alarm), lines[0])
    print(json.dumps(pick, ensure_ascii=False, indent=2))

show('/content/log_normal.jsonl', '정상 조리 — 탐지 없는 프레임 (평상시 로그)', False)
show('/content/log_normal.jsonl', '정상 조리 — 오탐이 난 프레임 (있다면)', True)
show('/content/log_fire.jsonl', '화재 — 경보가 올라간 프레임', True)

# 사건 요약 — 뒤 단계(VLM)가 실제로 받게 될 단위
print('\n' + '=' * 70); print('사건 요약 (VLM 이 받는 단위)'); print('=' * 70)
for path, tag in (('/content/log_normal.jsonl', '정상 조리'), ('/content/log_fire.jsonl', '화재')):
    ev = {}
    for l in open(path, encoding='utf-8'):
        r = json.loads(l)
        a = r['alarm']
        if a.get('raised'):
            ev.setdefault(a['event_id'], []).append(r)
    print(f'\n{tag} — 사건 {len(ev)}건')
    for eid, rs in ev.items():
        last = rs[-1]['alarm']
        st = {d['station_id'] for r in rs for d in r['detections']}
        print(f"  {eid}  지속 {last['duration_s']}초 · 최고신뢰도 "
              f"{last['max_conf_in_event']} · 최대크기 {last['max_area_ratio_in_event']} · "
              f"화구 {st}")

files.download('/content/log_normal.jsonl')
files.download('/content/log_fire.jsonl')
