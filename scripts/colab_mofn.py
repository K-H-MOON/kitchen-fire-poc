# ===== 시간축 M-of-N 판정 측정 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요. 약 5분.
# 업로드할 파일 4개: round10_best.pt · mofn_seq.zip · seq_a.zip · seq_b.zip
#
# 사전 등록: docs/PREREGISTER_TIME.md
#
# 답하려는 질문
#   지금까지의 수치는 전부 사진 한 장 기준임.
#   "최근 N장 중 M장 이상"으로 누적 판정하면 사건 단위 성능이 얼마가 되는가?
#   그리고 우리 오탐은 산발적인가, 지속적인가?

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, math, numpy as np, cv2
from google.colab import files
from ultralytics import YOLO

CONF = 0.10          # 프레임 임계값 — 운용값 고정
FPS = 1.0            # 모든 시퀀스를 1fps 로 맞춤
COOLDOWN = 30        # 경보 후 쿨다운(초). 한 번 울린 뒤 연속 카운트를 막음

up = files.upload()
for Z in [k for k in up if k.endswith('.zip')]:
    zipfile.ZipFile(Z).extractall('/content/T')
W = [k for k in up if k.endswith('.pt')][0]
print('가중치:', W)

# 논현중 1fps 시퀀스는 seq_a / seq_b 두 개로 나뉘어 있으므로 합쳐서 순서대로 읽음
non = sorted(glob.glob('/content/T/seq_a/*.jpg') + glob.glob('/content/T/seq_b/*.jpg'),
             key=lambda p: os.path.basename(p))

FIRE_SEQ = {os.path.basename(d): sorted(glob.glob(d + '/*.jpg'))
            for d in sorted(glob.glob('/content/T/mofn/fire/*'))}
NORM_SEQ = {os.path.basename(d): sorted(glob.glob(d + '/*.jpg'))
            for d in sorted(glob.glob('/content/T/mofn/normal/*'))}
NORM_SEQ['nonhyeon(미사용)'] = non

print(f'\n화재 사건 {len(FIRE_SEQ)}개 · 총 {sum(len(v) for v in FIRE_SEQ.values())}프레임')
for k, v in FIRE_SEQ.items():
    print(f'  {k:10s} {len(v):3d}장 = {len(v)/FPS:.0f}초')
print(f'정상 시퀀스 {len(NORM_SEQ)}개 · 총 {sum(len(v) for v in NORM_SEQ.values())}프레임')
for k, v in NORM_SEQ.items():
    print(f'  {k:18s} {len(v):3d}장 = {len(v)/FPS/60:.1f}분')

m = YOLO(W)
print('\n모델 클래스:', m.names)


def detect_series(paths, batch=32):
    """각 프레임에서 탐지 여부(0/1)"""
    out = []
    for i in range(0, len(paths), batch):
        for r in m.predict(paths[i:i + batch], conf=CONF, verbose=False):
            out.append(1 if len(r.boxes) else 0)
    return np.array(out)


print('\n프레임 단위 탐지 중...')
D_fire = {k: detect_series(v) for k, v in FIRE_SEQ.items()}
D_norm = {k: detect_series(v) for k, v in NORM_SEQ.items()}


def alarms(d, M, N):
    """M-of-N 규칙으로 경보가 울린 프레임 인덱스 목록 (쿨다운 적용)"""
    idx, last = [], -10 ** 9
    for t in range(len(d)):
        w = d[max(0, t - N + 1):t + 1]
        if w.sum() >= M and (t - last) * (1 / FPS) >= COOLDOWN:
            idx.append(t); last = t
    return idx


RULES = [(1, 1), (2, 3), (2, 5), (3, 5), (3, 10), (5, 10), (7, 10)]

print('\n' + '=' * 78)
print(f"{'규칙':>8s}{'사건 탐지':>12s}{'지연(초) 중앙':>14s}"
      f"{'허위경보/시간 전체':>20s}{'논현중 단독':>14s}")
print('-' * 78)

rows = []
for M, N in RULES:
    hit, delay = 0, []
    for k, d in D_fire.items():
        a = alarms(d, M, N)
        if a:
            hit += 1; delay.append(a[0] / FPS)
    # 허위 경보 — 전체 정상 시퀀스, 그리고 논현중 단독
    fa_all = fa_non = 0
    hr_all = hr_non = 0.0
    for k, d in D_norm.items():
        n = len(alarms(d, M, N)); hrs = len(d) / FPS / 3600
        fa_all += n; hr_all += hrs
        if '논현중' in k or 'nonhyeon' in k:
            fa_non += n; hr_non += hrs
    r_all = fa_all / hr_all if hr_all else float('nan')
    r_non = fa_non / hr_non if hr_non else float('nan')
    md = np.median(delay) if delay else float('nan')
    rows.append((M, N, hit, md, r_all, r_non))
    tag = '  <- 현재' if (M, N) == (1, 1) else ''
    print(f'{M}-of-{N:<4d}{hit:>6d}/{len(D_fire)}{md:>13.0f}'
          f'{r_all:>19.1f}{r_non:>13.1f}{tag}')

print('\n' + '=' * 78)
print('사전 등록 성공선 — 사건 탐지 7/8 이상 · 논현중 허위경보 시간당 6회 이하')
ok = [r for r in rows if r[2] >= 7 and r[5] <= 6]
if ok:
    b = min(ok, key=lambda r: r[5])
    print(f'  성립 — {b[0]}-of-{b[1]} (탐지 {b[2]}/8, 지연 {b[3]:.0f}초, 논현중 {b[5]:.1f}회/시간)')
else:
    base = [r for r in rows if (r[0], r[1]) == (1, 1)][0]
    half = [r for r in rows if r[2] >= 7 and r[5] <= base[5] / 2]
    if half:
        b = min(half, key=lambda r: r[5])
        print(f'  부분 — {b[0]}-of-{b[1]} 에서 허위경보가 절반 이하 '
              f'({base[5]:.1f} → {b[5]:.1f}회/시간), 탐지 {b[2]}/8 유지')
    else:
        print('  실패 — 어떤 규칙에서도 탐지율을 지키며 허위경보가 절반으로 줄지 않음')
        print('        → 오탐이 산발적이 아니라 지속적이라는 뜻')

# 오탐이 산발적인지 지속적인지 직접 확인
print('\n' + '=' * 78)
print('오탐의 지속성 — 탐지된 프레임이 얼마나 이어지는가')
for k, d in D_norm.items():
    if d.sum() == 0:
        print(f'  {k:18s} 탐지 0건'); continue
    runs, cur = [], 0
    for v in d:
        if v: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    print(f'  {k:18s} 탐지 {int(d.sum()):3d}/{len(d):3d}프레임 · '
          f'연속 구간 {len(runs)}개 · 최장 {max(runs)}초 · 중앙 {int(np.median(runs))}초')

print('\n참고 — 화재 쪽 탐지 지속성')
for k, d in D_fire.items():
    print(f'  {k:10s} {int(d.sum())}/{len(d)}프레임 탐지')
