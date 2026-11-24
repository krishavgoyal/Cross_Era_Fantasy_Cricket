"""Steps 2 and 3: colours taken from the reference photo, then held steady over time.

Step 2, colour transfer (per region: the player, and the scene around him, each matched to the same region of
the reference image, blended with a feathered mask so there is no seam):
  - black-and-white photo (Bradman): the clip is made monochrome too (MusePose had tinted it), and the
    luminance distribution of each region is matched to the photo's
  - colour photo (McGrath): a linear colour transfer (Monge-Kantorovich, Pitie & Kokaram 2007) in Lab from
    the WHOLE clip's colour statistics to the photo's: one transform for all frames, so the colours that
    vary over time keep varying the same way, only the overall cast and saturation change
  - palette constraint: player pixels whose colour is far from every colour in the photo (k-means palette,
    24 colours) have their chroma pulled towards the nearest photo colour; luminance is untouched, so
    shading and gradients stay
Step 3, temporal stabilisation:
  - each region's mean colour is smoothed over time (Gaussian, sigma 2 frames) and every frame is shifted
    to it: no colour flicker between frames
  - the static scene is calmed: background pixels are blended with their temporal median over +-4 frames,
    using only frames where the player does not cover that pixel (no ghosts)
  - all processing in float32; output with a +-0.5 level dither, so smooth gradients do not band
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from refine.score import palette, player_mask


def to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def to_bgr(lab, dither=True):
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0
    if dither:
        out = out + np.random.default_rng(0).uniform(-0.5, 0.5, out.shape).astype(np.float32)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def feather(mask, r=7):
    m = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), r)[..., None]


def _sqrtm(C):
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.clip(w, 1e-8, None))) @ V.T


def mkl(Cs, Ct):
    """Linear map A with A Cs A^T = Ct (closest to identity in the Monge-Kantorovich sense)."""
    Ss = _sqrtm(Cs)
    Ssi = np.linalg.inv(Ss)
    return Ssi @ _sqrtm(Ss @ Ct @ Ss) @ Ssi


def _stats(samples):
    x = np.concatenate(samples).reshape(-1, 3).astype(np.float64)
    return x.mean(0), np.cov(x.T) + np.eye(3) * 1e-3


def lum_match(src_vals, ref_vals):
    """Monotone luminance mapping (quantile matching, 256 knots) from src_vals to ref_vals."""
    q = np.linspace(0, 100, 257)
    return np.percentile(src_vals, q), np.percentile(ref_vals, q)


def masks_for(frames):
    out = []
    for f in frames:
        m = player_mask(f)
        out.append(np.zeros(f.shape[:2], bool) if m is None else m)
    return out


def transfer(frames, ref, masks, ref_mask, pal_k=24, foreign_thr=12.0):
    """Step 2. frames: list of BGR uint8; ref: BGR uint8 (same size). Returns frames (BGR uint8), info."""
    rl = to_lab(ref)
    mono = float(np.hypot(rl[..., 1], rl[..., 2]).mean()) < 2.0
    labs = [to_lab(f) for f in frames]
    rng = np.random.default_rng(0)
    sub = lambda a: a[rng.choice(len(a), min(len(a), 4000), replace=False)] if len(a) else a
    out, info = [], {"mode": "monochrome (photo is black-and-white)" if mono else "colour transfer + palette"}
    if mono:
        knots = {}
        for name, sel_src, sel_ref in (("player", lambda l, m: l[m], rl[ref_mask]), ("scene", lambda l, m: l[~m], rl[~ref_mask])):
            src = np.concatenate([sub(sel_src(l, m)[:, 0]) for l, m in zip(labs, masks)])
            knots[name] = lum_match(src, sel_ref[:, 0])
        for l, m in zip(labs, masks):
            w = feather(m)[..., 0]
            Lp = np.interp(l[..., 0], *knots["player"])
            Ls = np.interp(l[..., 0], *knots["scene"])
            L = w * Lp + (1 - w) * Ls
            o = np.zeros_like(l)
            o[..., 0] = L
            out.append(o)
        info["tint_removed_mean_chroma"] = round(float(np.mean([np.hypot(l[..., 1], l[..., 2]).mean() for l in labs])), 2)
        return out, info
    T = {}
    for name, pick, rsel in (("player", lambda l, m: l[m], rl[ref_mask]), ("scene", lambda l, m: l[~m], rl[~ref_mask])):
        ms, Cs = _stats([sub(pick(l, m)) for l, m in zip(labs, masks)])
        mt, Ct = _stats([rsel])
        A = mkl(Cs, Ct)
        T[name] = (A, ms, mt)
    pal = palette(rl, ref_mask) if ref_mask.sum() > 200 else palette(rl, np.ones_like(ref_mask))
    moved = []
    for l, m in zip(labs, masks):
        w = feather(m)
        x = l.reshape(-1, 3)
        res = []
        for name in ("player", "scene"):
            A, ms, mt = T[name]
            res.append(((x - ms) @ A.T + mt).reshape(l.shape))
        o = w * res[0] + (1 - w) * res[1]
        # palette constraint on the player: chroma towards the nearest photo colour if it is foreign
        pm = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        px = o[pm]
        if len(px):
            d = np.linalg.norm(px[:, None, :] - pal[None], axis=2)
            j = d.argmin(1)
            dm = d[np.arange(len(px)), j]
            s = np.clip((dm - foreign_thr) / foreign_thr, 0, 1)[:, None] * 0.85
            px[:, 1:] = px[:, 1:] + s * (pal[j, 1:] - px[:, 1:])
            moved.append(float((s > 0).mean()))
            o[pm] = px
        o[..., 0] = np.clip(o[..., 0], 0, 100)
        out.append(o)
    info["foreign_pixels_pulled_share"] = round(float(np.mean(moved)), 4) if moved else 0.0
    info["player_matrix_det"] = round(float(np.linalg.det(T["player"][0])), 3)
    return out, info


def skeleton_cover(frame_tg, shape, thick=None):
    """Mask of where the target says the player, bat and ball are (thick skeleton lines): fast limbs and the
    bat are not always inside the person segmentation, and must never be blended with other frames."""
    from animate.skeleton import BODY, LIMBS
    m = np.zeros(shape, np.uint8)
    b = frame_tg["body"]
    pts = [b.get(j) for j in BODY]
    if thick is None:
        nk, rh, lh = b.get("neck"), b.get("right_hip"), b.get("left_hip")
        tor = np.hypot(nk[0] - (rh[0] + lh[0]) / 2, nk[1] - (rh[1] + lh[1]) / 2) if nk and rh and lh else 60
        thick = max(9, int(0.35 * tor))
    for a, c in LIMBS:
        if pts[a] and pts[c]:
            cv2.line(m, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[c]), 1, thick)
    for side in ("left_hand", "right_hand"):
        for q in frame_tg.get(side) or []:
            if q:
                cv2.circle(m, tuple(int(v) for v in q), thick // 2, 1, -1)
    if frame_tg.get("bat") and all(frame_tg["bat"]):
        cv2.line(m, tuple(int(v) for v in frame_tg["bat"][0]), tuple(int(v) for v in frame_tg["bat"][1]), 1, thick)
    if frame_tg.get("ball"):
        cv2.circle(m, tuple(int(v) for v in frame_tg["ball"]), 14, 1, -1)
    return m > 0


def stabilise(labs, masks, sigma=2.0, bg_window=4, bg_blend=0.6, target_frames=None):
    """Step 3 on Lab frames (from step 2). Returns BGR uint8 frames, info."""
    n = len(labs)
    means = {"player": [], "scene": []}
    for l, m in zip(labs, masks):
        means["player"].append(l[m].mean(0) if m.sum() > 50 else np.full(3, np.nan))
        means["scene"].append(l[~m].mean(0))
    sm = {}
    for k, v in means.items():
        v = np.array(v)
        for c in range(3):
            col = v[:, c]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                col = np.interp(np.arange(n), np.where(ok)[0], col[ok])
            v[:, c] = col
        sm[k] = (v, gaussian_filter1d(v, sigma, axis=0, mode="nearest"))
    shifted = []
    for t, (l, m) in enumerate(zip(labs, masks)):
        w = feather(m)
        dp = np.nan_to_num(sm["player"][1][t] - sm["player"][0][t]) * np.array([0.7, 1, 1])
        ds = (sm["scene"][1][t] - sm["scene"][0][t]) * np.array([0.7, 1, 1])
        shifted.append(l + w * dp + (1 - w) * ds)
    # calm the static scene: temporal median of uncovered background pixels
    cover = [cv2.dilate(m.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0 for m in masks]
    if target_frames is not None:
        cover = [c | cv2.dilate(skeleton_cover(tf, c.shape).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                 for c, tf in zip(cover, target_frames)]
    out = []
    for t in range(n):
        ks = range(max(0, t - bg_window), min(n, t + bg_window + 1))
        stack = np.stack([shifted[k] for k in ks])
        valid = np.stack([~cover[k] for k in ks])[..., None]
        med = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
        have = ~np.isnan(med[..., 0]) & ~cover[t]
        o = shifted[t].copy()
        o[have] = (1 - bg_blend) * o[have] + bg_blend * med[have]
        out.append(to_bgr(o))
    return out, {"colour_smoothing_sigma_frames": sigma, "background_median_window": 2 * bg_window + 1,
                 "background_blend": bg_blend}


def noise_sigma(gray):
    """Noise/grain level: robust std of the fine detail (image minus its 3x3 median) in smooth regions."""
    g = gray.astype(np.float32)
    hp = g - cv2.medianBlur(gray, 3).astype(np.float32)
    grad = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F)
    smooth = np.abs(grad) < np.percentile(np.abs(grad), 40)
    v = hp[smooth]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def grain(frames, ref, rho=0.5, seed=0):
    """Step 4: add luminance grain so the clip's fine texture matches the photo's (generated frames are
    smoother than a real photo, which shows as flat, stepped gradients). Grain std = sqrt(photo^2 - clip^2),
    slightly soft (sigma 0.6 px), correlated over time (AR(1), rho) so it does not boil."""
    s_ref = noise_sigma(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    s_vid = float(np.median([noise_sigma(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]))
    add = float(np.sqrt(max(0.0, s_ref ** 2 - s_vid ** 2)))
    info = {"photo_grain": round(s_ref, 2), "clip_grain": round(s_vid, 2), "added": round(add, 2)}
    if add < 0.3:
        return frames, info
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    n = rng.standard_normal((h, w)).astype(np.float32)
    out = []
    for f in frames:
        n = rho * n + np.sqrt(1 - rho ** 2) * rng.standard_normal((h, w)).astype(np.float32)
        g = cv2.GaussianBlur(n, (0, 0), 0.6)
        g = g / max(float(g.std()), 1e-6) * add
        out.append(np.clip(f.astype(np.float32) + g[..., None], 0, 255).round().astype(np.uint8))
    return out, info
