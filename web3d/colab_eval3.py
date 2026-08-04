# ===== 웹3D 탐지 박스 위치 정확도(IoU) 측정 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요.
# 업로드할 파일 3개: best.pt · web3d_proc.zip · web3d_atlas.zip
#   (zip 안에 labels.json 이 있어야 함 — 정답 박스 포함 버전 scene.html로 캡처)
#
# 사전 등록 기준 (캡처 전 고정, 2026-08-04)
#   조건 P 기준, 유지율 = (IoU 0.5 기준 D) / (이미지 단위 D)
#     >= 0.80  → 좌표 사용 가능. 파이프라인 5단계(탐지 JSON → VLM) 진행
#     0.50~0.80 → 조건부. 좌표를 보조 정보로만 사용
#     <  0.50  → 좌표 사용 불가. VLM에 이미지 전체를 넘기는 설계로 변경
#   운용 임계값 conf 0.10 · IoU 0.5

!pip -q install ultralytics==8.3.*

import os, glob, json, zipfile, numpy as np, cv2
from google.colab import files
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

CONF, IOU_T = 0.10, 0.50
PASS_KEEP, PARTIAL_KEEP = 0.80, 0.50

up = files.upload()
WEIGHTS = [k for k in up if k.endswith('.pt')][0]
ZIPS    = sorted([k for k in up if k.endswith('.zip')])
m = YOLO(WEIGHTS)

def unpack(zf, dest):
    z = zipfile.ZipFile(zf)
    for n in z.namelist():
        p = n.replace('\\', '/')
        if p.endswith('/'): continue
        os.makedirs(os.path.join(dest, os.path.dirname(p)) or dest, exist_ok=True)
        open(os.path.join(dest, p), 'wb').write(z.read(n))
    return dest

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

RES = {}
for zf in ZIPS:
    tag = 'proc' if 'proc' in zf else ('atlas' if 'atlas' in zf else zf)
    d = unpack(zf, '/content/iou_' + tag)
    lab = json.load(open(f'{d}/labels.json'))['boxes']
    rows = []
    fire = sorted(glob.glob(f'{d}/fire/*.jpg'))
    for p in fire:
        fn = os.path.basename(p)
        gt = lab.get(fn)
        r = m.predict(p, conf=0.03, verbose=False)[0]
        bs = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        cs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        keep = cs >= CONF
        bs, cs = bs[keep], cs[keep]
        best = max((iou(b, gt) for b in bs), default=0.0) if gt else None
        rows.append(dict(f=p, gt=gt, n=len(bs), best=best,
                         boxes=bs, confs=cs))
    RES[tag] = rows
    print(f'{tag}: 발화 {len(fire)}장 · 정답 박스 {sum(r["gt"] is not None for r in rows)}장')

print('\n' + '=' * 70)
print(f'conf {CONF} · IoU {IOU_T}')
print(f'{"조건":<7}{"이미지 단위 D":>14}{"IoU 기준 D":>13}{"유지율":>9}{"정탐 평균 IoU":>14}')
KEEP = {}
for tag, rows in RES.items():
    v = [r for r in rows if r['gt'] is not None]
    img = sum(r['n'] > 0 for r in v)
    loc = sum(r['best'] >= IOU_T for r in v)
    keep = loc/img if img else 0
    KEEP[tag] = keep
    mi = np.mean([r['best'] for r in v if r['best'] >= IOU_T]) if loc else 0
    print(f'{tag:<7}{img}/{len(v)} = {img/len(v)*100:5.1f}%'
          f'{loc}/{len(v)} = {loc/len(v)*100:5.1f}%{keep*100:8.1f}%{mi:13.3f}')

if 'proc' in KEEP:
    k = KEEP['proc']
    v = ('좌표 사용 가능 — 5단계 진행' if k >= PASS_KEEP else
         '조건부 — 좌표는 보조 정보로만' if k >= PARTIAL_KEEP else
         '좌표 사용 불가 — 설계 변경 필요')
    print(f'\n사전 등록 판정 (조건 P 유지율 {k*100:.1f}%): {v}')

print('\nIoU 분포 (정답 박스가 있는 발화 프레임)')
bins = [0, .1, .3, .5, .7, .9, 1.01]
print(f'{"구간":<12}' + ''.join(f'{t:>9}' for t in RES))
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    row = f'{lo:.1f}~{hi if hi<=1 else 1.0:.1f}'.ljust(12)
    for tag, rows in RES.items():
        v = [r['best'] for r in rows if r['gt'] is not None]
        row += f'{sum(lo <= x < hi for x in v):>9}'
    print(row)

def draw(rows, sel, title, n=8, cols=4):
    sel = [r for r in rows if sel(r)][:n]
    if not sel: print(f'\n{title} — 없음'); return
    print(f'\n■ {title} ({n}장 이내)')
    ims = []
    for r in sel:
        im = cv2.imread(r['f'])
        g = r['gt']
        cv2.rectangle(im, (g[0], g[1]), (g[2], g[3]), (255, 128, 0), 2)   # 정답 파랑
        for b, c in zip(r['boxes'], r['confs']):
            b = b.astype(int)
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)  # 예측 초록
            cv2.putText(im, f'{c:.2f}', (b[0], max(12, b[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.putText(im, f'IoU {r["best"]:.2f}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        ims.append(cv2.resize(im, (400, 225)))
    while len(ims) % cols: ims.append(np.zeros((225, 400, 3), np.uint8))
    cv2_imshow(np.vstack([np.hstack(ims[i:i+cols]) for i in range(0, len(ims), cols)]))

for tag, rows in RES.items():
    v = [r for r in rows if r['gt'] is not None]
    draw(v, lambda r: r['best'] >= IOU_T, f'{tag} — 위치까지 맞음 (IoU>={IOU_T})')
    draw(v, lambda r: 0 < r['n'] and r['best'] < IOU_T, f'{tag} — 잡았으나 위치 어긋남')
    draw(v, lambda r: r['n'] == 0, f'{tag} — 미탐')
