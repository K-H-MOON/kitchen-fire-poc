#!/usr/bin/env python3
"""A/B/C 3그룹 채점 + 신뢰도 기준선 곡선.

  python scripts/eval_gate.py --weights runs/gate/weights/best.pt \
      --eval-dir eval --cctv assets/eval_neg

A 실화재(D-Fire)   합성 학습이 실사에 전이되는가
B 주방 오탐(CCTV)  실제 배치 환경에서 오작동하지 않는가   <- 학습에 미사용
C 배경 오탐(D-Fire) 화재를 배웠는가, 데이터셋 도메인을 외웠는가
"""
import argparse, glob, os
import numpy as np
from ultralytics import YOLO

# 판정 기준 — 실행 전 확정, 결과를 보고 수정하지 않는다
PASS_RECALL, PARTIAL_RECALL, MAX_FP_CCTV = .50, .20, .15


def max_conf(model, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], conf=0.03, verbose=False):
            out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--eval-dir', default='eval')
    ap.add_argument('--cctv', default='assets/eval_neg')
    ap.add_argument('--conf', type=float, default=0.10, help='운용 기준선')
    a = ap.parse_args()

    read = lambda n: [l.strip() for l in open(f'{a.eval_dir}/{n}') if l.strip()]
    A, C = read('eval_pos.txt'), read('eval_dneg.txt')
    B = sorted(glob.glob(f'{a.cctv}/*.jpg'))
    print(f'평가셋  A {len(A)} / B {len(B)} / C {len(C)}\n')

    m = YOLO(a.weights)
    cA, cB, cC = max_conf(m, A), max_conf(m, B), max_conf(m, C)

    r = (cA >= a.conf).mean(); fb = (cB >= a.conf).mean(); fc = (cC >= a.conf).mean()
    print(f'conf {a.conf}')
    print(f'  A 실화재 인식률   : {int((cA>=a.conf).sum())}/{len(A)} = {r*100:.1f}%')
    print(f'  B 주방 오탐률     : {int((cB>=a.conf).sum())}/{len(B)} = {fb*100:.1f}%')
    print(f'  C 배경 오탐률     : {int((cC>=a.conf).sum())}/{len(C)} = {fc*100:.1f}%')
    print(f'  판별비 A/B        : {r/fb:.2f}' if fb else '  판별비 A/B        : -')

    v = ('통과 -> 확장' if r >= PASS_RECALL else
         '부분 -> 보강 후 재시도' if r >= PARTIAL_RECALL else
         '실패 -> 합성 단독 경로 중단')
    if fb > MAX_FP_CCTV:
        v += '  (단, 배치 환경 오탐 과다 — 보강 필요)'
    print(f'\n판정: {v}\n')

    print(f'{"conf":>5} {"A 실화재":>9} {"B 주방오탐":>10} {"C 배경오탐":>10}')
    for t in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.40):
        print(f'{t:>5.2f} {(cA>=t).mean()*100:8.1f}% '
              f'{(cB>=t).mean()*100:9.1f}% {(cC>=t).mean()*100:9.1f}%')

    print('\n주의: B는 고정 카메라 영상이라 실효 표본 수가 프레임 수보다 작습니다.')


if __name__ == '__main__':
    main()
