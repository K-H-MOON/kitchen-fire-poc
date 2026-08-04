#!/usr/bin/env python3
"""6회차 채점 — fire·smoke 2클래스. 5회차 대비 화염 회귀 여부를 함께 본다.

  python scripts/eval_gate6.py --weights runs/r6/weights/best.pt \
      --eval-dir eval --cctv assets/eval_neg --steam assets/eval_steam \
      --base assets/weights/round5_best.pt

평가군
  A-fire  D-Fire 실화재 150   화염 인식률 (5회차와 같은 이미지)
  A-smoke D-Fire 실연기 150   연기 인식률   <- 학습에 실제 연기 0장
  B       급식실 CCTV 75      배치 환경 오탐 (학습 미사용)
  C       D-Fire 배경 75      도메인 암기 진단
  D       수증기 352          김을 연기로 오인하는가   <- 이번 회차 신설

판정 기준은 web3d/PREREGISTER_SMOKE.md 에 캡처·학습 전에 확정했다.
"""
import argparse, glob
import numpy as np
from ultralytics import YOLO

PASS_SMOKE, PARTIAL_SMOKE = .40, .20      # A-smoke 통과선
MAX_FP_STEAM = .25                        # D 수증기 오탐 상한
MAX_FIRE_DROP = .08                       # A-fire 가 5회차 대비 8%p 넘게 떨어지면 회귀
MAX_FP_CCTV = .15

FIRE, SMOKE = 0, 1


def scan(model, paths, batch=32, conf=0.03):
    """이미지별 (fire 최대 conf, smoke 최대 conf)"""
    out = []
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], conf=conf, verbose=False):
            f = s = 0.0
            if len(r.boxes):
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                if (cl == FIRE).any():
                    f = float(cf[cl == FIRE].max())
                if (cl == SMOKE).any():
                    s = float(cf[cl == SMOKE].max())
            out.append((f, s))
    return np.array(out) if out else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--base', help='5회차 가중치 — 화염 회귀 비교용 (선택)')
    ap.add_argument('--eval-dir', default='eval')
    ap.add_argument('--cctv', default='assets/eval_neg')
    ap.add_argument('--steam', default='assets/eval_steam')
    ap.add_argument('--conf', type=float, default=0.10)
    a = ap.parse_args()

    read = lambda n: [l.strip() for l in open(f'{a.eval_dir}/{n}') if l.strip()]
    A_f, A_s = read('eval_pos.txt'), read('eval_smoke.txt')
    C = read('eval_dneg.txt')
    B = sorted(glob.glob(f'{a.cctv}/*.jpg'))
    D = sorted(glob.glob(f'{a.steam}/*.jpg'))
    print(f'평가셋  A-fire {len(A_f)} / A-smoke {len(A_s)} / '
          f'B {len(B)} / C {len(C)} / D 수증기 {len(D)}\n')

    m = YOLO(a.weights)
    zAf, zAs, zB, zC, zD = (scan(m, x) for x in (A_f, A_s, B, C, D))
    t = a.conf

    r_fire = (zAf[:, FIRE] >= t).mean()
    r_smoke = (zAs[:, SMOKE] >= t).mean()
    fp_b = (zB.max(1) >= t).mean()
    fp_c = (zC.max(1) >= t).mean()
    fp_d_any = (zD.max(1) >= t).mean()
    fp_d_smoke = (zD[:, SMOKE] >= t).mean()

    print(f'conf {t}')
    print(f'  A-fire  화염 인식률 : {int((zAf[:,FIRE]>=t).sum())}/{len(A_f)} = {r_fire*100:.1f}%')
    print(f'  A-smoke 연기 인식률 : {int((zAs[:,SMOKE]>=t).sum())}/{len(A_s)} = {r_smoke*100:.1f}%')
    print(f'  B 주방 오탐률       : {fp_b*100:.1f}%')
    print(f'  C 배경 오탐률       : {fp_c*100:.1f}%')
    print(f'  D 수증기 오탐률     : {fp_d_any*100:.1f}%  (smoke 클래스만 {fp_d_smoke*100:.1f}%)')

    base_fire = None
    if a.base:
        mb = YOLO(a.base)
        zb = scan(mb, A_f)
        base_fire = (zb[:, FIRE] >= t).mean()
        print(f'\n5회차 A-fire {base_fire*100:.1f}% -> 6회차 {r_fire*100:.1f}% '
              f'({(r_fire-base_fire)*100:+.1f}p)')

    v = ('연기 학습 성립' if r_smoke >= PASS_SMOKE else
         '부분 성립' if r_smoke >= PARTIAL_SMOKE else '연기 학습 실패')
    notes = []
    if fp_d_any > MAX_FP_STEAM:
        notes.append(f'수증기 오탐 초과({fp_d_any*100:.1f}% > {MAX_FP_STEAM*100:.0f}%)')
    if fp_b > MAX_FP_CCTV:
        notes.append(f'주방 오탐 초과({fp_b*100:.1f}%)')
    if base_fire is not None and (base_fire - r_fire) > MAX_FIRE_DROP:
        notes.append(f'화염 회귀({(r_fire-base_fire)*100:.1f}p) — 되돌림 검토')
    print(f'\n판정: {v}' + (' · ' + ' · '.join(notes) if notes else ''))

    print(f'\n{"conf":>5} {"A-fire":>9} {"A-smoke":>9} {"B":>8} {"C":>8} {"D 수증기":>10}')
    for x in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.40):
        print(f'{x:>5.2f} {(zAf[:,FIRE]>=x).mean()*100:8.1f}% '
              f'{(zAs[:,SMOKE]>=x).mean()*100:8.1f}% '
              f'{(zB.max(1)>=x).mean()*100:7.1f}% {(zC.max(1)>=x).mean()*100:7.1f}% '
              f'{(zD.max(1)>=x).mean()*100:9.1f}%')

    print('\n주의: B는 고정 카메라라 실효 표본 수가 프레임 수보다 작습니다. '
          'D는 급식실이 아닌 스톡 영상이므로 도메인이 다릅니다.')


if __name__ == '__main__':
    main()
