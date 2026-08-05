#!/usr/bin/env python3
"""실주방 프레임 + 실화염 소재 -> 합성 화재 학습셋 (YOLO 라벨 자동 생성).

  python scripts/synthesize.py --assets assets --out ds \
      --dfire-bg-list dfire_bg.txt --dfire-bg-count 600

--dfire-bg-* 를 주면 주방 밖 배경에도 같은 화염을 합성해 배경 다양성을 확보한다(4회차 구성).
생략하면 주방 프레임만 사용하는 3회차 구성이 된다.
"""
import argparse, glob, json, os, shutil
import cv2, numpy as np


def bg_noise(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F))) * 0.9)


def synth(bg, lib, cx, cy, tw_frac, rng, haze_p=0.0):
    """cx,cy = 불꽃 밑동 위치(정규화). tw_frac = 불꽃 폭(화면 폭 대비).
    반환: (합성 이미지, bbox) 또는 (None, None)."""
    H, W = bg.shape[:2]
    r = cv2.imread(rng.choice(lib), cv2.IMREAD_UNCHANGED)
    fl, al = r[..., :3].astype(np.float32) / 255, r[..., 3].astype(np.float32) / 255
    if rng.random() < .5:
        fl, al = fl[:, ::-1].copy(), al[:, ::-1].copy()

    s = tw_frac * W / fl.shape[1]
    fw, fh = max(8, int(fl.shape[1] * s)), max(8, int(fl.shape[0] * s))
    if fh > H * 3:
        return None, None
    fl, al = cv2.resize(fl, (fw, fh)), cv2.resize(al, (fw, fh))

    # --- 아티팩트 무작위화: 합성 흔적이 "일관된" 단서가 되지 못하게 한다.
    #     사실성을 높이는 것보다 이쪽이 실사 전이에 유효하다.
    al = np.clip(al, 0, 1) ** rng.uniform(.65, 1.6)          # 경계 단단함
    k = rng.uniform(0, 2.2)
    if k > .3:                                                # 경계 흐림
        al = cv2.GaussianBlur(al, (0, 0), k)
        fl = cv2.GaussianBlur(fl, (0, 0), k * .7)
    fl = np.clip(fl * np.array([rng.uniform(.85, 1.10),
                                rng.uniform(.92, 1.06), 1.], np.float32)
                 * rng.uniform(.85, 1.12), 0, 1)              # 색온도·노출

    # --- 대기 산란(선택) : 대낮·원거리·연기 속의 흐려진 화염을 모사
    #     소재가 전부 검은 배경의 선명한 화염이라 생기는 분포 편향을 보정한다.
    if haze_p and rng.random() < haze_p:
        h = rng.uniform(.15, .50)
        gray = fl.mean(2, keepdims=True)
        fl = fl * (1 - .45 * h) + gray * (.45 * h)                    # 채도 저하
        fl = fl * (1 - h) + np.array([.84, .86, .88], np.float32) * h  # 흰 베일
        al = al * rng.uniform(.58, .90)                                # 투과
        fl = cv2.GaussianBlur(fl, (0, 0), rng.uniform(.6, 2.0))

    bx, by = int(cx * W - fw / 2), int(cy * H - fh)
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(W, bx + fw), min(H, by + fh)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None, None
    fl = fl[y0 - by:y1 - by, x0 - bx:x1 - bx]
    al = al[y0 - by:y1 - by, x0 - bx:x1 - bx]

    img = bg.astype(np.float32) / 255
    # 광원 번짐 — 불이 주변을 밝히는 실제 물리 현상이므로 전 positive 에 적용
    glow = np.zeros((H, W), np.float32); glow[y0:y1, x0:x1] = al
    glow = cv2.GaussianBlur(glow, (0, 0), max(H, W) * rng.uniform(.02, .05))
    glow /= glow.max() + 1e-6
    img += glow[..., None] * np.array([.30, .55, .95], np.float32) * rng.uniform(.25, .65)

    roi = img[y0:y1, x0:x1]
    op = np.clip(al * rng.uniform(1.2, 1.7), 0, 1)[..., None]
    img[y0:y1, x0:x1] = roi * (1 - op) + fl * op + fl * (al[..., None] * rng.uniform(.2, .55))
    img = np.clip(img, 0, 1)

    # 센서 특성 정합 — 배경과 같은 입자감·압축 이력을 갖게 한다.
    sig = bg_noise(bg) / 255 * rng.uniform(.8, 1.4)
    n = rng.normal(0, sig, (y1 - y0, x1 - x0, 3)).astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] + n * (al[..., None] > .05), 0, 1)
    out = (img * 255).astype(np.uint8)
    q = int(rng.integers(55, 93))
    out = cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

    ys, xs = np.where(al > .18)
    if len(ys) == 0:
        return None, None
    return out, (x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max())


def write(root, img, bb, name):
    H, W = img.shape[:2]
    cv2.imwrite(f'{root}/images/train/{name}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    p = f'{root}/labels/train/{name}.txt'
    if bb is None:
        open(p, 'w').close()
    else:
        x0, y0, x1, y1 = bb
        open(p, 'w').write(f'0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} '
                           f'{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', default='assets')
    ap.add_argument('--flamelib', default='flamelib', help='소재 폴더명 (9회차: flamelib2)')
    ap.add_argument('--out', default='ds')
    ap.add_argument('--variants', type=int, default=13, help='주방 베이스 1장당 합성 횟수')
    ap.add_argument('--dfire-bg-list', help='주방 밖 배경 이미지 경로 목록 (한 줄에 하나)')
    ap.add_argument('--dfire-bg-count', type=int, default=600)
    ap.add_argument('--haze-prob', type=float, default=0.0,
                    help='대기 산란을 적용할 positive 비율 (0=끔, 5회차는 0.5)')
    ap.add_argument('--seed', type=int, default=20260803)
    a = ap.parse_args()

    lib = sorted(glob.glob(f'{a.assets}/{a.flamelib}/*.webp'))
    anch = json.load(open(f'{a.assets}/anchors.json'))
    assert lib and anch, '소재를 찾을 수 없습니다 — --assets 경로를 확인하세요'
    rng = np.random.default_rng(a.seed)

    shutil.rmtree(a.out, ignore_errors=True)
    for d in ('images/train', 'labels/train'):
        os.makedirs(f'{a.out}/{d}', exist_ok=True)

    # (1) 주방 베이스 — 앵커(조리기구 위치·폭) 기준으로 배치
    npos_k = 0
    for bi, (n, (cx, cy, vw, _grp)) in enumerate(anch.items()):
        bg = cv2.imread(f'{a.assets}/bases/{n}')
        for v in range(a.variants):
            out, bb = synth(bg, lib,
                            cx + rng.uniform(-.15, .15) * vw,
                            cy + rng.uniform(-.02, .02),
                            vw * rng.uniform(.35, 1.05), rng, a.haze_prob)
            if out is None:
                continue
            write(a.out, out, bb, f'fire_k{bi:03d}_{v:02d}'); npos_k += 1

    # (2) 주방 밖 배경 — 같은 배경의 "불 없음" 짝을 함께 넣어 배경이 단서가 되지 않게 한다
    npos_d = nneg_d = 0
    if a.dfire_bg_list:
        pool = [l.strip() for l in open(a.dfire_bg_list) if l.strip()]
        rng.shuffle(pool)
        for i, p in enumerate(pool[:a.dfire_bg_count]):
            bg = cv2.imread(p)
            if bg is None or min(bg.shape[:2]) < 200:
                continue
            out, bb = synth(bg, lib, rng.uniform(.2, .8), rng.uniform(.55, .95),
                            rng.uniform(.06, .35), rng, a.haze_prob)
            if out is None:
                continue
            write(a.out, out, bb, f'fire_d{i:04d}'); npos_d += 1
            write(a.out, bg, None, f'norm_d{i:04d}'); nneg_d += 1

    # (3) negative — 주방 원본 프레임, 가공 없음
    nneg_k = 0
    for p in sorted(glob.glob(f'{a.assets}/negsrc/*.jpg')):
        s = 'norm_k_' + os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, f'{a.out}/images/train/{s}.jpg')
        open(f'{a.out}/labels/train/{s}.txt', 'w').close(); nneg_k += 1

    open(f'{a.out}/data.yaml', 'w').write(
        f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/train\n"
        f"nc: 1\nnames: ['fire']\n")
    print(f'positive  주방 {npos_k} + 주방밖 {npos_d} = {npos_k + npos_d}')
    print(f'negative  주방 {nneg_k} + 주방밖 {nneg_d} = {nneg_k + nneg_d}')
    print(f'-> {a.out}/data.yaml')


if __name__ == '__main__':
    main()
