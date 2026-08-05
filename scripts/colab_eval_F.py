# ===== 평가군 F(주방 화재) 채점 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요. 약 3분.
# 업로드할 파일 3개: round5_best.pt · eval_F_fire.zip · eval_F_nofire.zip
#
# 이 측정이 답하는 질문
#   지금까지 A(D-Fire)는 산불·차량화재라 "급식실 화재를 잡는가"에 답하지 못했다.
#   F는 소방 실험 영상에서 뽑은 조리기구 위 식용유 화염이다. 처음으로 대상에 맞는 측정이 된다.
#
# F-fire   화염 있음  → 인식해야 함 (인식률)
# F-nofire 화염 없음  → 반응하면 안 됨 (오탐률). 같은 실험·같은 배경이라
#                       배경이 아니라 화염에 반응하는지 직접 확인됨

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, math, numpy as np, cv2
from google.colab import files
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

FIRE = 0
CONF = 0.10

up = files.upload()
W = [k for k in up if k.endswith('.pt')][0]
for Z in [k for k in up if k.endswith('.zip')]:
    zipfile.ZipFile(Z).extractall('/content/F')

P = sorted(glob.glob('/content/F/eval_F/F_fire/*.jpg'))
N = sorted(glob.glob('/content/F/eval_F/F_nofire/*.jpg'))
print(f'F 화염 {len(P)}장 · F 화염없음 {len(N)}장\n')

m = YOLO(W)
print('모델 클래스:', m.names, '\n')


def scan(paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        for r in m.predict(paths[i:i + batch], conf=0.03, verbose=False):
            f = 0.0
            if len(r.boxes):
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                if (cl == FIRE).any():
                    f = float(cf[cl == FIRE].max())
            out.append(f)
    return np.array(out)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


zP, zN = scan(P), scan(N)

print('=' * 64)
for t in (0.03, 0.05, 0.10, 0.15, 0.25, 0.40):
    p = (zP >= t).mean(); n = (zN >= t).mean()
    ratio = (p / n) if n > 0 else float('inf')
    mark = '  <- 운용' if abs(t - CONF) < 1e-9 else ''
    print(f'conf {t:4.2f} | F 인식률 {p*100:5.1f}%  F 오탐률 {n*100:5.1f}%  판별비 {ratio:6.2f}{mark}')

kP = int((zP >= CONF).sum()); kN = int((zN >= CONF).sum())

# ---- 주 지표: 출처별 macro 평균 ----
# kfire01이 화염 264장 중 127장을 차지하나 한 영상·한 실험장에서 나온 것이라
# 독립 관측이 아님. 장수로 합산하면 이 출처가 결과를 좌우함.
TAGS = sorted({os.path.basename(q).split('_')[0] for q in P + N})
rows = []
for tag in TAGS:
    ip = [i for i, q in enumerate(P) if os.path.basename(q).startswith(tag)]
    inn = [i for i, q in enumerate(N) if os.path.basename(q).startswith(tag)]
    rec = float((zP[ip] >= CONF).mean()) if ip else None
    fpr = float((zN[inn] >= CONF).mean()) if inn else None
    rows.append((tag, len(ip), rec, len(inn), fpr))

mrec = float(np.mean([r for _, _, r, _, _ in rows if r is not None]))
mfpr = float(np.mean([f for _, _, _, _, f in rows if f is not None]))

print('\n' + '=' * 64)
print('주 지표 — 출처별 macro 평균')
print(f'  F 인식률 (macro) : {mrec*100:.1f}%')
print(f'  F 오탐률 (macro) : {mfpr*100:.1f}%')
print(f'  판별비 F         : {mrec/mfpr:.2f}' if mfpr > 0 else '  판별비 F         : 오탐 0건')

print('\n보조 — micro (장수 합산)')
lo, hi = wilson(kP, len(P)); lo2, hi2 = wilson(kN, len(N))
print(f'  인식률 {kP}/{len(P)} = {kP/len(P)*100:.1f}%  95%CI {lo:.1f}~{hi:.1f}')
print(f'  오탐률 {kN}/{len(N)} = {kN/len(N)*100:.1f}%  95%CI {lo2:.1f}~{hi2:.1f}')
if abs(mrec - kP/len(P)) > 0.10:
    print('  !! macro와 micro가 10%p 이상 갈림 — 특정 출처가 전체를 끌고 있음')

print('\n출처별 원수치')
for tag, np_, rec, nn_, fpr in rows:
    a = f'{round(rec*np_)}/{np_} = {rec*100:.1f}%' if rec is not None else '-'
    b = f'{round(fpr*nn_)}/{nn_} = {fpr*100:.1f}%' if fpr is not None else '-'
    print(f'  {tag:10s} 화염 {a:>16s}   화염없음 {b:>16s}')

# 화염 크기별 — 작은 화염을 놓치는지가 조기 경보의 핵심
print('\n화염 면적비 구간별 인식률')
def warm_frac(p):
    im = cv2.imread(p); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255., hsv[..., 2] / 255.
    return float((((h < 28) | (h > 170)) & (s > 0.35) & (v > 0.62)).mean())
ff = np.array([warm_frac(p) for p in P])
for lo_, hi_ in ((0, .02), (.02, .05), (.05, .15), (.15, 1.0)):
    idx = np.where((ff >= lo_) & (ff < hi_))[0]
    if len(idx):
        k = int((zP[idx] >= CONF).sum())
        print(f'  {lo_*100:4.0f}~{hi_*100:4.0f}%  {k:3d}/{len(idx):3d} = {k/len(idx)*100:5.1f}%')

NAME = {0: 'fire', 1: 'smoke'}; COL = {0: (0, 255, 0), 1: (255, 180, 0)}


def grid(paths, title, n=12, cols=4):
    ims = []
    for p in paths[:n]:
        im = cv2.imread(p)
        for r in m.predict(p, conf=CONF, verbose=False):
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int)):
                cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), COL[k], 2)
                cv2.putText(im, f'{NAME[k]} {c:.2f}', (b[0], max(14, b[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, COL[k], 2)
        ims.append(cv2.resize(im, (400, 225)))
    print(f'\n■ {title} ({len(ims)}장)')
    if not ims:
        print('없음'); return
    while len(ims) % cols:
        ims.append(np.zeros((225, 400, 3), np.uint8))
    cv2_imshow(np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]))


grid([P[i] for i in np.where(zP < CONF)[0]], 'F 화염 — 놓친 것')
grid([P[i] for i in np.where(zP >= CONF)[0]], 'F 화염 — 잡은 것')
grid([N[i] for i in np.where(zN >= CONF)[0]], 'F 화염없음 — 잘못 반응한 것')
