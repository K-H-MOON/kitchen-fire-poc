#!/usr/bin/env python3
"""D-Fire 를 걸러 평가셋 A(실화재)·C(배경)와 학습용 배경 목록을 만든다.

  python scripts/dfire_eval_set.py --dfire /path/to/d-fire --out eval

선별 규칙은 성능을 보기 전에 확정한 것으로, 이후 수정하지 않는다.
  · 화재 박스 면적이 화면의 1~35%      -> 근중거리 중소형 화재
  · 박스 중심이 화면 하단 2/3          -> 항공·원경 제외
  · 평균 밝기 >= 70, 최대 박스 면적 >= 3%  -> 야간 원거리 감시영상 제외
"""
import argparse, glob, os
from collections import Counter
import cv2, numpy as np

BRIGHT_MIN, AREA_MIN = 70, 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfire', required=True, help='D-Fire 루트 (하위에 images/labels 가 있으면 됨)')
    ap.add_argument('--out', default='eval')
    ap.add_argument('--n-pos', type=int, default=150)
    ap.add_argument('--n-neg', type=int, default=75)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 폴더 구조를 가정하지 않고 라벨에서 역으로 이미지를 찾는다
    labs = [p for p in glob.glob(f'{a.dfire}/**/*.txt', recursive=True)
            if os.path.basename(p).lower() not in ('classes.txt', 'readme.txt')]
    imgs = {}
    for e in ('*.jpg', '*.jpeg', '*.png'):
        for p in glob.glob(f'{a.dfire}/**/{e}', recursive=True):
            imgs.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    print(f'라벨 {len(labs)}개 / 이미지 {len(imgs)}장')

    pairs, cls = [], Counter()
    for lp in labs:
        ip = imgs.get(os.path.splitext(os.path.basename(lp))[0])
        if ip is None:
            continue
        rows = [r.split() for r in open(lp).read().strip().splitlines() if r.strip()]
        pairs.append((ip, rows))
        for r in rows:
            cls[int(r[0])] += 1
    assert pairs, 'D-Fire 경로가 잘못되었습니다'
    # D-Fire 원본 기준 fire 14,692 / smoke 11,865 -> 박스가 더 많은 쪽이 fire
    FIRE = max(cls, key=cls.get) if len(cls) > 1 else 0
    print(f'짝이 맞는 이미지 {len(pairs)} | 클래스별 박스 {dict(cls)} -> fire = {FIRE}')

    cand, bg = [], []
    for ip, rows in pairs:
        if not rows:
            bg.append(ip); continue
        for r in rows:
            if int(r[0]) != FIRE:
                continue
            _, xc, yc, w, h = map(float, r)
            if 0.01 <= w * h <= 0.35 and yc > 0.33:
                cand.append((ip, rows)); break
    print(f'1차 통과 화재 {len(cand)}장 / 배경 {len(bg)}장')

    sel = []
    for ip, rows in cand:
        im = cv2.imread(ip, cv2.IMREAD_REDUCED_COLOR_8)   # 1/8 축소 디코딩
        if im is None:
            continue
        areas = [float(r[3]) * float(r[4]) for r in rows if int(r[0]) == FIRE]
        if im.mean() >= BRIGHT_MIN and max(areas or [0]) >= AREA_MIN:
            sel.append(ip)
    print(f'2차(밝기·근접) 통과 {len(sel)}장')

    take = lambda L, n: [L[round(i * (len(L) - 1) / max(n - 1, 1))]
                         for i in range(min(n, len(L)))]
    pos, dneg = take(sorted(sel), a.n_pos), take(sorted(set(bg)), a.n_neg)
    used = set(pos) | set(dneg)
    train_bg = [p for p in sorted(set(bg)) if p not in used]   # 학습용 배경(평가와 완전 분리)

    for name, lst in (('eval_pos.txt', pos), ('eval_dneg.txt', dneg),
                      ('train_bg.txt', train_bg)):
        open(f'{a.out}/{name}', 'w').write('\n'.join(lst))
        print(f'{name:16s} {len(lst)}장')


if __name__ == '__main__':
    main()
