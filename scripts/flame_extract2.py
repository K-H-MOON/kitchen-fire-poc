"""화염 소재 알파 추출 v2 — 백열(white-hot) 코어를 살림.

v1의 결함
    a *= clip((sat - 0.18) / 0.25, 0, 1)
  연기·회색 안개를 걸러내려고 채도로 알파를 깎았는데, 하얗게 포화된 화염 중심부는
  채도가 0에 가까워 함께 지워졌음. 390종 중 354종에서 백열 코어의 평균 알파가 0.22.
  (화염 몸통 평균 0.83) → 학습이 백열 화염을 한 번도 보지 못함.

v2의 게이트
    keep = max(채도 램프, 고휘도 램프)
  연기는 회색이면서 어둡고, 백열 화염은 회색이면서 아주 밝음. 휘도가 둘을 가름.
  채도가 낮아도 휘도가 임계 이상이면 화염으로 보고 알파를 유지함.

배경 임계 lo 는 테두리 휘도에서 자동 추정 (v1은 영상마다 수동 지정이었음).
"""
import os, glob, argparse
import numpy as np, cv2

SAT_LO, SAT_W = 0.18, 0.25     # v1과 동일한 채도 램프
HOT_LO, HOT_W = 0.80, 0.12     # v2 신규 — 이 휘도 이상이면 채도 무관하게 유지
MIN_PIX, MIN_H, MIN_W = 800, 70, 50
HASH_DIST = 10


def auto_lo(lum):
    """테두리(배경) 휘도에서 키잉 임계를 추정"""
    H, W = lum.shape
    bh, bw = max(2, H // 20), max(2, W // 20)
    border = np.concatenate([lum[:bh].ravel(), lum[-bh:].ravel(),
                             lum[:, :bw].ravel(), lum[:, -bw:].ravel()])
    return float(np.clip(np.median(border) + 0.085, 0.09, 0.38))


def key(bgr, lo=None, v1=False):
    im = bgr.astype(np.float32) / 255.
    lum = im.max(2)
    if lo is None:
        lo = auto_lo(lum)
    a = np.clip((lum - lo) / 0.28, 0, 1)
    a[a < 0.06] = 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.
    sat_ramp = np.clip((sat - SAT_LO) / SAT_W, 0, 1)
    if v1:
        keep = sat_ramp                                   # 기존 방식
    else:
        hot_ramp = np.clip((lum - HOT_LO) / HOT_W, 0, 1)  # 백열 보존
        keep = np.maximum(sat_ramp, hot_ramp)
    a *= keep

    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    if (a > 0.15).sum() < MIN_PIX:
        return None
    ys, xs = np.where(a > 0.12)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < MIN_H or (x1 - x0) < MIN_W:
        return None
    return im[y0:y1, x0:x1], a[y0:y1, x0:x1]


def dhash(a, s=8):
    g = cv2.resize(a, (s + 1, s))
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def core_alpha(rgb, a):
    """백열 코어(아주 밝고 채도 낮음) 영역의 평균 알파 — 구멍 진단용"""
    bgr = (rgb * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1] / 255., hsv[..., 2] / 255.
    core = (v > 0.92) & (s < 0.30)
    return (float(a[core].mean()), int(core.sum())) if core.sum() > 30 else (np.nan, 0)


def run(src_glob, out, tag_of, v1=False, lo_map=None):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(src_glob))
    by = {}
    for p in files:
        by.setdefault(tag_of(p), []).append(p)
    tot, cores = 0, []
    for tag, ps in sorted(by.items()):
        hs, n = [], 0
        for p in ps:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            r = key(bgr, None if lo_map is None else lo_map.get(tag), v1=v1)
            if r is None:
                continue
            rgb, a = r
            h = dhash(a)
            if any(int(np.unpackbits(h ^ k).sum()) <= HASH_DIST for k in hs):
                continue
            hs.append(h)
            n += 1
            ca, cn = core_alpha(rgb, a)
            if cn:
                cores.append(ca)
            rgba = np.dstack([(rgb * 255).astype(np.uint8), (a * 255).astype(np.uint8)])
            cv2.imwrite(f'{out}/{tag}_{n:03d}.png', rgba)
        print(f'  {tag}: {n}종')
        tot += n
    cores = np.array([c for c in cores if not np.isnan(c)])
    print(f'합계 {tot}종 · 백열 코어 평균 알파 {cores.mean():.2f} (1.0=온전, 0=구멍)')
    return tot


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--v1', action='store_true')
    a = ap.parse_args()
    run(a.src, a.out, lambda p: os.path.basename(p).split('_')[0], v1=a.v1)
