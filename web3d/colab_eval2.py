# ===== 웹3D 렌더 화면 채점 v2 — 절차적 화염 vs 실사 아틀라스 짝 비교 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요.
# 업로드할 파일 3개: best.pt · web3d_proc.zip · web3d_atlas.zip
#
# 사전 등록 기준 (캡처 전 고정, 2026-08-04)
#   절차적 화염 D >= 50%  → 전이 성립
#   절차적 화염 D 25~50%  → 부분 전이
#   절차적 화염 D <  25%  → 실사 텍스처 의존
#   운용 임계값 conf 0.10 · 정상 오탐 E <= 15%

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, math, numpy as np, cv2
from google.colab import files
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

PASS_D, PARTIAL_D, MAX_E, CONF = 0.50, 0.25, 0.15, 0.10

up = files.upload()
WEIGHTS = [k for k in up if k.endswith('.pt')][0]
ZIPS    = sorted([k for k in up if k.endswith('.zip')])
print('가중치:', WEIGHTS, '| zip:', ZIPS, '\n')

def unpack(zf, dest):
    z = zipfile.ZipFile(zf)
    for n in z.namelist():
        p = n.replace('\\', '/')
        if p.endswith('/'): continue
        os.makedirs(os.path.join(dest, os.path.dirname(p)), exist_ok=True)
        open(os.path.join(dest, p), 'wb').write(z.read(n))
    return dest

m = YOLO(WEIGHTS)

def maxconf(paths, batch=16):
    out = []
    for i in range(0, len(paths), batch):
        for r in m.predict(paths[i:i+batch], conf=0.03, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100

SETS = {}
for zf in ZIPS:
    tag = 'proc' if 'proc' in zf else ('atlas' if 'atlas' in zf else zf)
    d = unpack(zf, '/content/' + tag)
    SETS[tag] = {k: sorted(glob.glob(f'{d}/{k}/*.jpg')) for k in ('fire', 'smoke', 'normal')}
    print(f'{tag}: 발화 {len(SETS[tag]["fire"])} · 연기 {len(SETS[tag]["smoke"])} · 정상 {len(SETS[tag]["normal"])}')

RES = {}
for tag, s in SETS.items():
    RES[tag] = {k: maxconf(v) for k, v in s.items()}

print('\n' + '=' * 62)
print(f'운용 기준선 conf {CONF}')
print(f'{"세트":<8}{"D 발화":>18}{"S 연기":>12}{"E 정상":>12}')
for tag in RES:
    c = RES[tag]
    nD, kD = len(c['fire']), int((c['fire'] >= CONF).sum())
    lo, hi = wilson(kD, nD)
    S = (c['smoke'] >= CONF).mean() * 100 if len(c['smoke']) else 0
    E = (c['normal'] >= CONF).mean() * 100 if len(c['normal']) else 0
    print(f'{tag:<8}{kD}/{nD} = {kD/nD*100:5.1f}%  [{lo:.1f}~{hi:.1f}]{S:>10.1f}%{E:>11.1f}%')

if 'proc' in RES:
    c = RES['proc']; D = (c['fire'] >= CONF).mean(); E = (c['normal'] >= CONF).mean()
    verdict = ('전이 성립' if D >= PASS_D else '부분 전이' if D >= PARTIAL_D else '실사 텍스처 의존')
    if E > MAX_E: verdict += ' · 단 정상 오탐 초과'
    print(f'\n사전 등록 기준 판정 (절차적): {verdict}')
if 'proc' in RES and 'atlas' in RES:
    dp = (RES['proc']['fire'] >= CONF).mean() * 100
    da = (RES['atlas']['fire'] >= CONF).mean() * 100
    print(f'텍스처 계열 효과: 아틀라스 {da:.1f}% → 절차적 {dp:.1f}%  (차 {da-dp:+.1f}p)')

print('\n임계값 스윕')
hdr = f'{"conf":>6}' + ''.join(f'{t+" D":>12}' for t in RES) + ''.join(f'{t+" E":>12}' for t in RES)
print(hdr)
for c0 in (0.03, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
    row = f'{c0:>6.2f}'
    for t in RES: row += f'{(RES[t]["fire"]>=c0).mean()*100:>11.1f}%'
    for t in RES: row += f'{(RES[t]["normal"]>=c0).mean()*100:>11.1f}%'
    print(row)

def grid(paths, title, cols=4, n=8):
    if not paths: print(f'\n{title} — 없음'); return
    print(f'\n■ {title} ({len(paths)}장 중 최대 {n}장)')
    ims = []
    for p in paths[:n]:
        im = cv2.imread(p)
        for r in m.predict(p, conf=CONF, verbose=False):
            for b in r.boxes.xyxy.cpu().numpy().astype(int):
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        ims.append(cv2.resize(im, (400, 225)))
    while len(ims) % cols: ims.append(np.zeros((225, 400, 3), np.uint8))
    cv2_imshow(np.vstack([np.hstack(ims[i:i+cols]) for i in range(0, len(ims), cols)]))

for tag in RES:
    f = SETS[tag]['fire']; c = RES[tag]['fire']
    grid([f[i] for i in np.where(c >= CONF)[0]], f'{tag} 발화 — 잡은 것')
    grid([f[i] for i in np.where(c <  CONF)[0]], f'{tag} 발화 — 놓친 것')
    nm = SETS[tag]['normal']; cn = RES[tag]['normal']
    grid([nm[i] for i in np.where(cn >= CONF)[0]], f'{tag} 정상 — 오탐')
