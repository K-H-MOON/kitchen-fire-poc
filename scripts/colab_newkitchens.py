# ===== 신규 급식실 3곳 오탐 점검 (10회차 사전 조사) =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요. 약 3분.
# 업로드할 파일 3개: round5_best.pt · round9_best.pt · newkitchens.zip
#
# 답하려는 질문
#   9회차가 급식실 오탐 69.3%를 낸 것은 개원중 한 곳의 특성인가,
#   아니면 급식실 전반의 문제인가?
#
#   영동중(국탕, 주황 표시등) · 원촌중(튀김솥, 밝은 스테인리스가 화면 절반)
#   논현중(튀김솥, 밝은 노란 조리복 — 평가 전용 예정)
#
# 세 곳 모두 화재가 없는 정상 조리 영상이므로, 반응하면 전부 오탐임.

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, math, numpy as np, cv2
from google.colab import files
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

CONF = 0.10
up = files.upload()
for Z in [k for k in up if k.endswith('.zip')]:
    zipfile.ZipFile(Z).extractall('/content/NK')
W5 = [k for k in up if '5' in k and k.endswith('.pt')][0]
W9 = [k for k in up if '9' in k and k.endswith('.pt')][0]
print('5회차 가중치:', W5, '/ 9회차 가중치:', W9)

SITES = ['yeongdong', 'wonchon', 'nonhyeon']
paths = {s: sorted(glob.glob(f'/content/NK/newneg/{s}/*.jpg')) for s in SITES}
for s in SITES:
    print(f'{s:10s} {len(paths[s])}장')


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def scan(m, ps, batch=32):
    out = []
    for i in range(0, len(ps), batch):
        for r in m.predict(ps[i:i + batch], conf=0.03, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)


res = {}
for tag, w in (('5회차', W5), ('9회차', W9)):
    m = YOLO(w)
    print(f'\n{tag} 모델 클래스: {m.names}')
    res[tag] = {s: scan(m, paths[s]) for s in SITES}

print('\n' + '=' * 70)
print('운용 기준선 conf 0.10 — 전부 정상 조리이므로 반응 = 오탐')
print(f"{'급식실':12s}{'5회차':>16s}{'9회차':>16s}")
for s in SITES:
    a = res['5회차'][s]; b = res['9회차'][s]
    ka, kb = int((a >= CONF).sum()), int((b >= CONF).sum())
    print(f'{s:12s}{ka:5d}/{len(a):3d} = {ka/len(a)*100:5.1f}%'
          f'{kb:9d}/{len(b):3d} = {kb/len(b)*100:5.1f}%')
alla = np.concatenate([res['5회차'][s] for s in SITES])
allb = np.concatenate([res['9회차'][s] for s in SITES])
ka, kb = int((alla >= CONF).sum()), int((allb >= CONF).sum())
lo, hi = wilson(ka, len(alla)); lo2, hi2 = wilson(kb, len(allb))
print(f"{'합계':12s}{ka:5d}/{len(alla):3d} = {ka/len(alla)*100:5.1f}%"
      f"{kb:9d}/{len(allb):3d} = {kb/len(allb)*100:5.1f}%")
print(f'  95%CI  5회차 {lo:.1f}~{hi:.1f}   9회차 {lo2:.1f}~{hi2:.1f}')
print('  참고 — 개원중(평가군 B): 5회차 2.7% / 9회차 69.3%')

print('\n기준선별')
print(f"{'conf':>6s}{'5회차':>10s}{'9회차':>10s}")
for t in (0.03, 0.05, 0.10, 0.15, 0.25, 0.40):
    print(f'{t:6.2f}{(alla>=t).mean()*100:9.1f}%{(allb>=t).mean()*100:9.1f}%')

# 오탐 사진 — 무엇에 반응했는지가 10회차 설계의 근거
m9 = YOLO(W9)
for s in SITES:
    idx = np.where(res['9회차'][s] >= CONF)[0]
    print(f'\n■ {s} — 9회차 오탐 {len(idx)}장 (앞 8장)')
    if not len(idx):
        print('없음'); continue
    tiles = []
    for i in idx[:8]:
        im = cv2.imread(paths[s][i])
        for r in m9.predict(paths[s][i], conf=CONF, verbose=False):
            for b, c in zip(r.boxes.xyxy.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(im, f'{c:.2f}', (b[0], max(14, b[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (400, 225)))
    while len(tiles) % 4:
        tiles.append(np.zeros((225, 400, 3), np.uint8))
    cv2_imshow(np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
