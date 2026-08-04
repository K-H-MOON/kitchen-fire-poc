#!/usr/bin/env python3
"""6회차 — 실주방 프레임 + 실화염/실연기 소재 -> fire·smoke 2클래스 합성 학습셋.

  python scripts/synthesize_smoke.py --assets assets --out ds6 \
      --dfire-bg-list eval/train_bg.txt --dfire-bg-count 600 --haze-prob 0.5

1~5회차용 scripts/synthesize.py 는 그대로 두었다(재현성 보존).
이 스크립트는 거기에 연기 레이어와 smoke 라벨 자동 생성을 더한 것이다.

장면 구성 — 급식실 튀김 조리의 실제 진행 순서를 따른다
    과열   : 연기만            -> smoke        (발화 이전 = 조기 경보 대상)
    발화   : 화염 + 상승 연기   -> fire + smoke
    초기발화: 화염만            -> fire
화염 positive 총량은 5회차와 같게 유지한다(variants 를 늘려 보정).
그래야 "클래스 추가로 화염 성능이 나빠졌는가"를 데이터 양과 분리해 볼 수 있다.

수작업 라벨링 0건 — 합성 좌표를 알고 있으므로 박스가 자동 생성된다.
"""
import argparse, glob, json, os, shutil
import cv2, numpy as np

FIRE, SMOKE = 0, 1


def bg_noise(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.median(np.abs(cv2.Laplacian(g, cv2.CV_32F))) * 0.9)


def _fit(mat, tw, rng, stretch=(1.0, 1.0)):
    """소재를 목표 폭에 맞춰 리사이즈. stretch = 세로 신장 범위."""
    rgb, al = mat[..., :3].astype(np.float32) / 255, mat[..., 3].astype(np.float32) / 255
    if rng.random() < .5:
        rgb, al = rgb[:, ::-1].copy(), al[:, ::-1].copy()
    s = tw / rgb.shape[1]
    sy = s * rng.uniform(*stretch)
    fw, fh = max(8, int(rgb.shape[1] * s)), max(8, int(rgb.shape[0] * sy))
    return cv2.resize(rgb, (fw, fh)), cv2.resize(al, (fw, fh))


def _paste(img, rgb, al, bx, by):
    """img(float, HxWx3)에 알파 합성. 반환: 화면 좌표 (x0,y0,x1,y1,al_crop) 또는 None."""
    H, W = img.shape[:2]
    fh, fw = al.shape
    x0, y0 = max(0, bx), max(0, by)
    x1, y1 = min(W, bx + fw), min(H, by + fh)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    rgb = rgb[y0 - by:y1 - by, x0 - bx:x1 - bx]
    al = al[y0 - by:y1 - by, x0 - bx:x1 - bx]
    return x0, y0, x1, y1, rgb, al


def place_flame(img, lib, cx, cy, tw_frac, rng, haze_p=0.0):
    """5회차와 동일한 화염 배치. img 를 제자리에서 수정하고 bbox 를 반환."""
    H, W = img.shape[:2]
    mat = cv2.imread(rng.choice(lib), cv2.IMREAD_UNCHANGED)
    fl, al = _fit(mat, tw_frac * W, rng)
    if al.shape[0] > H * 3:
        return None

    al = np.clip(al, 0, 1) ** rng.uniform(.65, 1.6)
    k = rng.uniform(0, 2.2)
    if k > .3:
        al = cv2.GaussianBlur(al, (0, 0), k)
        fl = cv2.GaussianBlur(fl, (0, 0), k * .7)
    fl = np.clip(fl * np.array([rng.uniform(.85, 1.10),
                                rng.uniform(.92, 1.06), 1.], np.float32)
                 * rng.uniform(.85, 1.12), 0, 1)

    if haze_p and rng.random() < haze_p:
        h = rng.uniform(.15, .50)
        gray = fl.mean(2, keepdims=True)
        fl = fl * (1 - .45 * h) + gray * (.45 * h)
        fl = fl * (1 - h) + np.array([.84, .86, .88], np.float32) * h
        al = al * rng.uniform(.58, .90)
        fl = cv2.GaussianBlur(fl, (0, 0), rng.uniform(.6, 2.0))

    r = _paste(img, fl, al, int(cx * W - al.shape[1] / 2), int(cy * H - al.shape[0]))
    if r is None:
        return None
    x0, y0, x1, y1, fl, al = r

    # 광원 번짐 — 불이 주변을 밝히는 실제 물리 현상
    glow = np.zeros((H, W), np.float32); glow[y0:y1, x0:x1] = al
    glow = cv2.GaussianBlur(glow, (0, 0), max(H, W) * rng.uniform(.02, .05))
    glow /= glow.max() + 1e-6
    img += glow[..., None] * np.array([.30, .55, .95], np.float32) * rng.uniform(.25, .65)

    op = np.clip(al * rng.uniform(1.2, 1.7), 0, 1)[..., None]
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * (1 - op) + fl * op
                         + fl * (al[..., None] * rng.uniform(.2, .55)))
    np.clip(img, 0, 1, out=img)

    ys, xs = np.where(al > .18)
    if len(ys) == 0:
        return None
    return x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()


def _vertical_crop(mat, rng):
    """가로로 퍼진 소재는 세로 조각으로 잘라 '피어오르는 기둥' 형태로 만든다."""
    h, w = mat.shape[:2]
    if w <= h * 1.25:
        return mat
    cw = int(h * rng.uniform(0.55, 1.15))
    cw = min(cw, w)
    x0 = int(rng.integers(0, max(1, w - cw)))
    return mat[:, x0:x0 + cw]


def place_smoke(img, lib, cx, cy, tw_frac, rng, sooty_p=0.45):
    """연기 기둥 배치. 밑동을 (cx,cy)에 두고 위로 뻗는다.

    배경 밝기에 따라 검댕/흰 연기를 고르고, 합성 후 실제로 보이는지 검사한다.
    보이지 않는 연기에 라벨을 붙이면 모델에 잡음을 가르치게 되므로 그런 장면은 버린다.
    """
    H, W = img.shape[:2]
    mat = _vertical_crop(cv2.imread(rng.choice(lib), cv2.IMREAD_UNCHANGED), rng)
    sm, al = _fit(mat, tw_frac * W, rng, stretch=(1.1, 2.4))   # 연기는 세로로 길다

    # 아티팩트 무작위화 — 화염과 같은 취지
    al = np.clip(al, 0, 1) ** rng.uniform(.75, 1.6)
    k = rng.uniform(0.4, 2.4)                                   # 연기는 경계가 더 흐리다
    al = cv2.GaussianBlur(al, (0, 0), k)
    sm = cv2.GaussianBlur(sm, (0, 0), k * .8)

    r = _paste(img, sm, al, int(cx * W - al.shape[1] / 2), int(cy * H - al.shape[0]))
    if r is None:
        return None
    x0, y0, x1, y1, sm, al = r

    # 배경이 밝으면 검댕 연기 쪽으로 기운다 — 흰 연기는 흰 주방에서 보이지 않는다
    roi = img[y0:y1, x0:x1]
    bgl = float(roi.mean())
    p_soot = np.clip(sooty_p + (bgl - 0.45) * 1.1, 0.05, 0.95)
    if rng.random() < p_soot:
        sm = sm * rng.uniform(.16, .52)                         # 검댕
        sm = np.clip(sm * np.array([rng.uniform(.94, 1.06),
                                    rng.uniform(.96, 1.04), 1.], np.float32), 0, 1)
        al = al * rng.uniform(.50, .95)
    else:
        sm = np.clip(sm * rng.uniform(.98, 1.14), 0, 1)         # 흰 연기
        al = al * rng.uniform(.55, .95)

    op = np.clip(al, 0, 1)[..., None]
    before = roi.copy()
    img[y0:y1, x0:x1] = roi * (1 - op) + sm * op
    np.clip(img, 0, 1, out=img)

    # 가시성 검사 — 라벨 박스 안에서 실제로 화면이 바뀌었는가
    diff = np.abs(img[y0:y1, x0:x1] - before).mean(2)
    m = al > .10
    if m.sum() < 40 or diff[m].mean() < 0.035:
        img[y0:y1, x0:x1] = before
        return None

    ys, xs = np.where(m & (diff > 0.02))                        # 눈에 보이는 부분만 박스로
    if len(ys) < 40:
        img[y0:y1, x0:x1] = before
        return None
    return x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()


def finalize(img, bg, rng):
    """센서 특성 정합 + JPEG 재압축 — 배경과 같은 입자감·압축 이력을 갖게 한다."""
    sig = bg_noise(bg) / 255 * rng.uniform(.8, 1.4)
    img = np.clip(img + rng.normal(0, sig, img.shape).astype(np.float32), 0, 1)
    out = (img * 255).astype(np.uint8)
    q = int(rng.integers(55, 93))
    return cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


def write(root, img, boxes, name):
    H, W = img.shape[:2]
    cv2.imwrite(f'{root}/images/train/{name}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(f'{root}/labels/train/{name}.txt', 'w') as f:
        for cls, (x0, y0, x1, y1) in boxes:
            f.write(f'{cls} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} '
                    f'{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n')


def scene(bg, flib, slib, cx, cy, vw, rng, haze_p, mode):
    """mode: 'smoke' | 'fire_smoke' | 'fire'"""
    img = bg.astype(np.float32) / 255
    boxes = []
    if mode in ('fire', 'fire_smoke'):
        bb = place_flame(img, flib, cx + rng.uniform(-.15, .15) * vw,
                         cy + rng.uniform(-.02, .02),
                         vw * rng.uniform(.35, 1.05), rng, haze_p)
        if bb is None:
            return None, None
        boxes.append((FIRE, bb))
    if mode in ('smoke', 'fire_smoke'):
        # 발화 시엔 화염 위에서, 과열 시엔 조리기구 표면에서 피어오른다
        lift = rng.uniform(.03, .10) if mode == 'fire_smoke' else rng.uniform(-.01, .03)
        bb = place_smoke(img, slib, cx + rng.uniform(-.12, .12) * vw, cy - lift,
                         vw * rng.uniform(.45, 1.25), rng,
                         sooty_p=.55 if mode == 'fire_smoke' else .35)
        if bb is None:
            return None, None
        boxes.append((SMOKE, bb))
    return finalize(img, bg, rng), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', default='assets')
    ap.add_argument('--out', default='ds6')
    ap.add_argument('--variants', type=int, default=20,
                    help='주방 베이스 1장당 합성 횟수 (화염 총량을 5회차와 맞추기 위해 13->20)')
    ap.add_argument('--dfire-bg-list')
    ap.add_argument('--dfire-bg-count', type=int, default=600)
    ap.add_argument('--haze-prob', type=float, default=0.5)
    ap.add_argument('--p-smoke', type=float, default=0.35, help='연기만 있는 장면 비율')
    ap.add_argument('--p-fire-smoke', type=float, default=0.40, help='화염+연기 비율')
    ap.add_argument('--seed', type=int, default=20260804)
    a = ap.parse_args()

    flib = sorted(glob.glob(f'{a.assets}/flamelib/*.webp'))
    slib = sorted(glob.glob(f'{a.assets}/smokelib/*.webp'))
    anch = json.load(open(f'{a.assets}/anchors.json'))
    assert flib and slib and anch, '소재를 찾을 수 없습니다 — --assets 경로를 확인하세요'
    rng = np.random.default_rng(a.seed)

    def pick():
        u = rng.random()
        if u < a.p_smoke:
            return 'smoke'
        if u < a.p_smoke + a.p_fire_smoke:
            return 'fire_smoke'
        return 'fire'

    shutil.rmtree(a.out, ignore_errors=True)
    for d in ('images/train', 'labels/train'):
        os.makedirs(f'{a.out}/{d}', exist_ok=True)

    cnt = {'fire': 0, 'smoke': 0, 'img': 0}
    for bi, (n, (cx, cy, vw, _grp)) in enumerate(anch.items()):
        bg = cv2.imread(f'{a.assets}/bases/{n}')
        for v in range(a.variants):
            m = pick()
            out, boxes = scene(bg, flib, slib, cx, cy, vw, rng, a.haze_prob, m)
            if out is None:
                continue
            write(a.out, out, boxes, f'k{bi:03d}_{v:02d}_{m}')
            cnt['img'] += 1
            for c, _ in boxes:
                cnt['fire' if c == FIRE else 'smoke'] += 1

    nneg_d = 0
    if a.dfire_bg_list:
        pool = [l.strip() for l in open(a.dfire_bg_list) if l.strip()]
        rng.shuffle(pool)
        for i, p in enumerate(pool[:a.dfire_bg_count]):
            bg = cv2.imread(p)
            if bg is None or min(bg.shape[:2]) < 200:
                continue
            m = pick()
            out, boxes = scene(bg, flib, slib, rng.uniform(.2, .8), rng.uniform(.55, .95),
                               rng.uniform(.10, .40), rng, a.haze_prob, m)
            if out is None:
                continue
            write(a.out, out, boxes, f'd{i:04d}_{m}')
            cnt['img'] += 1
            for c, _ in boxes:
                cnt['fire' if c == FIRE else 'smoke'] += 1
            # 같은 배경의 "아무 일 없음" 짝 — 배경 자체가 단서가 되지 않게 한다
            write(a.out, bg, [], f'norm_d{i:04d}'); nneg_d += 1

    # negative — 주방 원본 프레임. 김(수증기)이 다수 포함돼 있어 smoke 의 hard negative 로 작동한다.
    nneg_k = 0
    for p in sorted(glob.glob(f'{a.assets}/negsrc/*.jpg')):
        s = 'norm_k_' + os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, f'{a.out}/images/train/{s}.jpg')
        open(f'{a.out}/labels/train/{s}.txt', 'w').close(); nneg_k += 1

    open(f'{a.out}/data.yaml', 'w').write(
        f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/train\n"
        f"nc: 2\nnames: ['fire', 'smoke']\n")
    print(f"positive 이미지 {cnt['img']}장 · fire 박스 {cnt['fire']} · smoke 박스 {cnt['smoke']}")
    print(f'negative  주방 {nneg_k} + 주방밖 {nneg_d} = {nneg_k + nneg_d}')
    print(f'-> {a.out}/data.yaml')


if __name__ == '__main__':
    main()
