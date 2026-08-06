# ===== 사후 탐색 — 신뢰도 임계값 x M-of-N 격자 =====
# 앞의 colab_mofn.py 를 실행한 노트북에 이어서 붙여넣고 실행.
# 업로드 불필요. 이미 로드된 시퀀스를 재사용함.
#
# !! 이것은 사전 등록에 없던 사후 탐색임 (docs/PREREGISTER_TIME.md 는 conf 0.10 고정).
#    여기서 나온 최적 조합은 "이 자료에서 가장 좋아 보이는 값"이며,
#    같은 자료로 고르고 같은 자료로 평가한 것이므로 낙관적으로 치우쳐 있음.
#    별도 자료로 재검증하기 전까지 성능 근거로 인용하지 않음.

import numpy as np, glob, os
from ultralytics import YOLO

CONFS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
RULES = [(1, 1), (2, 3), (2, 5), (3, 5), (3, 10), (5, 10), (7, 10)]
FPS, COOLDOWN = 1.0, 30

m = YOLO(W)


def conf_series(paths, batch=32):
    """각 프레임의 최고 신뢰도 (0 = 탐지 없음)"""
    out = []
    for i in range(0, len(paths), batch):
        for r in m.predict(paths[i:i + batch], conf=0.03, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)


print('신뢰도 재수집 중...')
Zf = {k: conf_series(v) for k, v in FIRE_SEQ.items()}
Zn = {k: conf_series(v) for k, v in NORM_SEQ.items()}


def alarms(d, M, N):
    idx, last = [], -10 ** 9
    for t in range(len(d)):
        if d[max(0, t - N + 1):t + 1].sum() >= M and (t - last) / FPS >= COOLDOWN:
            idx.append(t); last = t
    return idx


def evaluate(conf, M, N):
    hit, delay = 0, []
    for k, z in Zf.items():
        a = alarms((z >= conf).astype(int), M, N)
        if a: hit += 1; delay.append(a[0] / FPS)
    fa = hr = 0.0
    for k, z in Zn.items():
        if 'nonhyeon' not in k:
            continue
        fa += len(alarms((z >= conf).astype(int), M, N)); hr += len(z) / FPS / 3600
    return hit, (np.median(delay) if delay else float('nan')), (fa / hr if hr else float('nan'))


print('\n' + '=' * 74)
print('논현중(학습 미사용) 허위 경보 회/시간 — 괄호는 사건 탐지 n/8')
print('=' * 74)
hdr = ''.join(f'{f"{M}-of-{N}":>11s}' for M, N in RULES)
print(f"{'conf':>6s}{hdr}")
best = []
for c in CONFS:
    cells = []
    for M, N in RULES:
        hit, md, rate = evaluate(c, M, N)
        cells.append(f'{rate:6.1f}({hit})')
        best.append((rate, hit, md, c, M, N))
    print(f'{c:6.2f}' + ''.join(f'{x:>11s}' for x in cells))

print('\n' + '=' * 74)
print('사전 등록 성공선을 사후 적용 — 사건 탐지 7/8 이상 · 논현중 시간당 6회 이하')
ok = [b for b in best if b[1] >= 7 and b[0] <= 6]
if ok:
    b = min(ok, key=lambda x: x[0])
    print(f'  조합 존재 — conf {b[3]:.2f} · {b[4]}-of-{b[5]} '
          f'(탐지 {b[1]}/8, 지연 {b[2]:.0f}초, 허위경보 {b[0]:.1f}회/시간)')
    print('  !! 사후 탐색이므로 이 조합의 성능은 별도 자료로 재검증해야 함')
else:
    k7 = [b for b in best if b[1] >= 7]
    if k7:
        b = min(k7, key=lambda x: x[0])
        print(f'  없음 — 탐지 7/8 을 지키는 조합 중 최저는 conf {b[3]:.2f} · {b[4]}-of-{b[5]} '
              f'에서 {b[0]:.1f}회/시간')
    k6 = [b for b in best if b[0] <= 6]
    if k6:
        b = max(k6, key=lambda x: x[1])
        print(f'  허위경보 6회 이하를 만족하는 조합 중 최고 탐지는 conf {b[3]:.2f} · '
              f'{b[4]}-of-{b[5]} 에서 {b[1]}/8')

# 사건별로 어디서 떨어져 나가는지 — 조기 경보의 약점 확인
print('\n' + '=' * 74)
print('사건별 최고 신뢰도 (이 값 이상으로 임계값을 올리면 그 사건은 놓침)')
for k, z in sorted(Zf.items()):
    print(f'  {k:8s} 최고 {z.max():.2f} · 0.10 이상 프레임 {int((z >= 0.10).sum())}/{len(z)}')
