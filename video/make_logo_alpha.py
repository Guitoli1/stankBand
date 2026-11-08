"""Stellt das Stank!-Logo frei: weißer Hintergrund -> transparent (Flood-Fill vom Rand)."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

src, dst = sys.argv[1], sys.argv[2]
im = np.array(Image.open(src).convert("RGB")).astype(np.float32)
whiteness = 255 - im.min(axis=2)          # 0 = reines Weiß
near_white = whiteness < 40
lab, _ = ndimage.label(near_white)
# Hintergrund = Randflächen + eingeschlossene, große reinweiße Flächen (z.B. im "S")
edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
pure = whiteness < 4
sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(lab.max() + 1))
pure_frac = ndimage.mean(pure, lab, index=np.arange(lab.max() + 1))
keep = {i for i in range(1, lab.max() + 1) if sizes[i] > 400 and pure_frac[i] > 0.85}
bg = np.isin(lab, list(edge | keep))
# Weiche Kante: Alpha aus Weißabstand, nur in Randnähe des Hintergrunds
alpha = np.where(bg, np.clip((whiteness - 4) / 36.0, 0, 1), 1.0)
alpha = ndimage.gaussian_filter(alpha, 0.6)
# Weißen Saum entfernen (Un-Premultiply gegen Weiß)
a3 = np.maximum(alpha[..., None], 1e-3)
rgb = np.clip((im - 255 * (1 - a3)) / a3, 0, 255)
out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
Image.fromarray(out, "RGBA").save(dst)
print("ok", out.shape)
