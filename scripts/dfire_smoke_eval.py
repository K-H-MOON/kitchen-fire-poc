#!/usr/bin/env python3
"""D-Fire 에서 smoke 평가셋(A-smoke)을 만든다. 학습에는 절대 쓰지 않는다.

  python scripts/dfire_smoke_eval.py --dfire /path/to/d-fire --out eval

원칙은 화염과 동일하다. D-Fire 의 실제 연기 이미지는 평가에만 쓰고 학습에는 배경만
쓴다. 그래야 "실제 연기 이미지 0장 학습"이라는 주장이 화염과 같은 구조로 성립한다.

선별 규칙 (성능을 보기 전에 확정, 이후 수정 금지)
  · smoke 박스만 있고 fire 박스는 없는 이미지  -> 발화 이전 상태만 남김
  · 최대 smoke 박스 면적이 화면의 2~60%
  · 박스 중심이 화면 하단 3/4              -> 항공·원경 제외
  · 평균 밝기 >= 70                        -> 야간 원거리 감시영상 제외
"""
import argparse, glob, os
from collections import Counter
import cv2

BRIGHT_MIN = 70
AREA_MIN, AREA_MAX = 0.02, 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfire', required=True)
    ap.add_argument('--out', default='eval')
    ap.add_argument('--n', type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    labs = [p for p in glob.glob(f'{a.dfire}/**/*.txt', recursive=True)
            if os.path.basename(p).lower() not in ('classes.txt', 'readme.txt')]
    imgs = {}
    for e in ('*.jpg', '*.jpeg', '*.png'):
        for p in glob.glob(f'{a.dfire}/**/{e}', recursive=True):
            imgs.setdefault(os.path.splitext(os.path.basename(p))[0], p)

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
    FIRE = max(cls, key=cls.get)
    SMOKE = min(cls, key=cls.get)      # fire 14,692 > smoke 11,865
    print(f'클래스별 박스 {dict(cls)} -> fire={FIRE}, smoke={SMOKE}')

    cand = []
    for ip, rows in pairs:
        if not rows:
            continue
        ids = {int(r[0]) for r in rows}
        if FIRE in ids or SMOKE not in ids:
            continue                                   # 불이 함께 있으면 제외
        areas = []
        ok_pos = False
        for r in rows:
            if int(r[0]) != SMOKE:
                continue
            _, xc, yc, w, h = map(float, r)
            areas.append(w * h)
            if yc > 0.25:
                ok_pos = True
        if ok_pos and areas and AREA_MIN <= max(areas) <= AREA_MAX:
            cand.append(ip)
    print(f'1차 통과(연기만) {len(cand)}장')

    sel = []
    for ip in cand:
        im = cv2.imread(ip, cv2.IMREAD_REDUCED_COLOR_8)
        if im is not None and im.mean() >= BRIGHT_MIN:
            sel.append(ip)
    print(f'2차(밝기) 통과 {len(sel)}장')

    take = lambda L, n: [L[round(i * (len(L) - 1) / max(n - 1, 1))]
                         for i in range(min(n, len(L)))]
    out = take(sorted(sel), a.n)
    open(f'{a.out}/eval_smoke.txt', 'w').write('\n'.join(out))
    print(f'eval_smoke.txt   {len(out)}장')

    # 학습용 배경 목록에서 평가 이미지를 배제 — 누수 차단
    bgp = f'{a.out}/train_bg.txt'
    if os.path.exists(bgp):
        bg = [l.strip() for l in open(bgp) if l.strip()]
        keep = [p for p in bg if p not in set(out)]
        if len(keep) != len(bg):
            open(bgp, 'w').write('\n'.join(keep))
            print(f'train_bg.txt 에서 {len(bg)-len(keep)}장 제거(평가 중복 차단)')


if __name__ == '__main__':
    main()
