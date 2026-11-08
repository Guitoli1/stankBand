#!/usr/bin/env python3
"""Baut das Stank!-Intro-Video: animiertes Logo, Bandvorstellung, Sprecher und Blues-Begleitung.

Aufruf:  python3 build.py --voice /pfad/zu/de-karlsson-low.onnx [--out stank-intro.mp4]

Benötigt: ffmpeg (mit rubberband), Python-Pakete piper-tts, numpy, pillow, scipy.
Der Sprecher ist eine generische, tief gepitchte deutsche Piper-Stimme ("Karlsson"),
im Stil eines rauen Gangsterfilm-Erzählers. Keine Nachbildung einer realen Person.
"""
import argparse
import os
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, "assets", *p)

W, H, FPS, SR = 1920, 1080, 30, 48000

# ---------------------------------------------------------------- Drehbuch ----
# Jeder Eintrag: (Szene, Text, Pause danach in s). [[...]] = rohe Phoneme für Piper.
STANK = "[[ sˈtɛŋk ]]"
SCRIPT = [
    ("dark",   "Hör zu.", 0.6),
    ("dark",   "Es gibt Bands.", 0.7),
    ("dark",   "Und dann gibt es ...", 0.5),
    ("hit",    f"{STANK}.", 1.2),
    ("logo",   "Soul und Blues. Ehrlich. Dreckig. Direkt aus dem Bauch.", 0.5),
    ("logo",   "Ich stell dir mal die Jungs vor.", 0.8),
    ("role:0", "An der Gitarre:", 0.25),
    ("name:0", "Hardy. Der redet nicht viel. Muss er auch nicht. Das macht seine Gitarre für ihn.", 0.9),
    ("role:1", "Am Mikro. Und auch an der Gitarre:", 0.25),
    ("name:1", "Oli. Wenn der singt, hört der ganze Laden zu. Freiwillig.", 0.9),
    ("role:2", "Am Schlagzeug:", 0.25),
    ("name:2", "[[ ʃtˈɛfan ]]. Er gibt den Takt vor. Und keiner widerspricht ihm.", 0.9),
    ("role:3", "Und am Bass:", 0.25),
    ("name:3", "[[ tˈoːmas ]]. Seinen Groove spürst du erst im Magen. Dann ist es zu spät.", 1.0),
    ("outro",  f"Das ist {STANK}.", 0.4),
    ("outro",  "Soul und Blues.", 0.5),
    ("outro",  "Merk dir den Namen.", 3.0),
]
MEMBERS = [
    ("HARDY", "GITARRE"),
    ("OLI", "GESANG & GITARRE"),
    ("STEFAN", "SCHLAGZEUG"),
    ("THOMAS", "BASS"),
]
LEAD_IN = 1.6  # Sekunden Dunkelheit + Knistern vor dem ersten Satz

GOLD_HI = np.array([255, 240, 200], np.float32)
GOLD_LO = np.array([214, 140, 40], np.float32)


# ------------------------------------------------------------------ Audio -----
def read_wav(path):
    with wave.open(path) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    x = x.reshape(-1, ch).mean(axis=1)
    assert sr == SR, sr
    return x


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def synth_voice(voice_path, tmp):
    """Erzeugt pro Zeile eine bearbeitete Sprachspur (tief, warm, etwas Raum)."""
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(voice_path)
    cfg = SynthesisConfig(length_scale=1.22, noise_scale=0.55, noise_w_scale=0.7)
    clips = []
    for i, (_, text, _) in enumerate(SCRIPT):
        raw = os.path.join(tmp, f"v{i}_raw.wav")
        out = os.path.join(tmp, f"v{i}.wav")
        with wave.open(raw, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=cfg)
        af = ",".join([
            "aresample=48000",
            "rubberband=pitch=0.84:formant=preserved",  # tiefer, Stimme bleibt natürlich
            "highpass=f=55",
            "lowshelf=f=180:g=5",
            "equalizer=f=2600:t=q:w=1.2:g=2.5",     # Präsenz / Kratzen
            "lowpass=f=7000",
            "acompressor=threshold=-20dB:ratio=4:attack=5:release=120:makeup=4",
            "aecho=0.8:0.6:38|63:0.12|0.07",        # kleiner Raum
            "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB,areverse",
            "apad=pad_dur=0.25",
        ])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", af, "-ac", "1", out], check=True)
        clips.append(read_wav(out))
    return clips


def adsr(n, a, d, s, r):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    env = np.full(n, s, np.float32)
    env[:a] = np.linspace(0, 1, a)
    env[a:a + d] = np.linspace(1, s, len(env[a:a + d]))
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def synth_band(total, rng):
    """Langsamer 12-Takt-Shuffle-Blues in E, komplett synthetisch."""
    bpm = 66
    beat = 60 / bpm
    out = np.zeros((int(total * SR) + SR, 2), np.float32)
    music_start = LEAD_IN + 0.0

    def add(sig, t, pan=0.0, gain=1.0):
        i = int(t * SR)
        if i >= len(out):
            return
        sig = sig[: len(out) - i] * gain
        out[i:i + len(sig), 0] += sig * (1 - pan) * 0.5 * 2 ** 0.5
        out[i:i + len(sig), 1] += sig * (1 + pan) * 0.5 * 2 ** 0.5

    def tone(f, dur, harm=(1, .5, .25), pluck=True):
        n = int(dur * SR)
        t = np.arange(n) / SR
        s = sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harm))
        env = np.exp(-t * 3.5) if pluck else adsr(n, .02, .2, .7, .1)
        return (s * env).astype(np.float32)

    def kick():
        t = np.arange(int(.4 * SR)) / SR
        f = 45 + 70 * np.exp(-t * 30)
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)).astype(np.float32)

    def brush(dur=.25, bright=0.5):
        n = int(dur * SR)
        t = np.arange(n) / SR
        x = rng.standard_normal(n).astype(np.float32)
        x = np.diff(x, prepend=0) * bright + x * (1 - bright)
        return x * np.exp(-t * 18) * 0.3

    E, A_, B = 41.2, 55.0, 61.74  # E1, A1, B1
    prog = [E, A_, E, E, A_, A_, E, E, B, A_, E, B]
    t = music_start
    bar = 0
    while t < total - 2:
        root = prog[bar % 12]
        for b in range(4):
            tb = t + b * beat
            sw = beat * 2 / 3  # Shuffle-Achtel
            # Walking-Bass: Grundton, Terz, Quinte, Sexte
            steps = [1, 5 / 4, 3 / 2, 5 / 3]
            add(tone(root * 2 * steps[b], beat * .95, (1, .6, .3, .15)), tb, -0.15, 0.32)
            add(kick(), tb, 0, 0.35 if b in (0, 2) else 0.18)
            if b in (1, 3):
                add(brush(.35, 0.3), tb, 0.1, 0.55)
            add(brush(.12, 0.9), tb, 0.35, 0.18)
            add(brush(.08, 0.9), tb + sw, 0.35, 0.12)
        # Orgel-Pad: Dominantseptakkord, leichtes Tremolo
        n = int(4 * beat * SR)
        tt = np.arange(n) / SR
        chord = [root * 4, root * 5, root * 6, root * 7 * 1.0]
        pad = sum(np.sin(2 * np.pi * f * tt) + .3 * np.sin(4 * np.pi * f * tt) for f in chord)
        pad *= (0.8 + 0.2 * np.sin(2 * np.pi * 5.5 * tt)) * adsr(n, .3, .3, .8, .4)
        add(pad.astype(np.float32), t, 0.25, 0.035)
        t += 4 * beat
        bar += 1

    # Rumpeln + Becken beim Logo-Einschlag wird separat hinzugefügt
    # Weicher Ausklang
    fade = int(3 * SR)
    end = int((total) * SR)
    out[end - fade:end] *= np.linspace(1, 0, fade)[:, None]
    out[end:] = 0
    # sanfte Tiefpass-Wärme
    from scipy.signal import butter, sosfilt
    out = sosfilt(butter(2, 5000, fs=SR, output="sos"), out, axis=0).astype(np.float32)
    return out[: int(total * SR)]


def impact(rng):
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t * 12)) / SR) * np.exp(-t * 2.2)
    from scipy.signal import butter, sosfilt
    cym = sosfilt(butter(2, 4000, "hp", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t * 1.6) * 0.25
    return (boom * 0.9 + cym).astype(np.float32)


def crackle(n, rng):
    x = np.zeros(n, np.float32)
    idx = rng.integers(0, n, n // 2500)
    x[idx] = rng.uniform(-1, 1, len(idx)) * rng.uniform(0.1, 0.6, len(idx))
    from scipy.signal import butter, sosfilt
    hiss = sosfilt(butter(2, [800, 6000], "bp", fs=SR, output="sos"), rng.standard_normal(n)) * 0.012
    return (np.convolve(x, np.hanning(24), "same") * 0.25 + hiss).astype(np.float32)


# ------------------------------------------------------------------ Video -----
def ease_out(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def gold_text(text, font, spacing=0, outline=10, shadow=(14, 14)):
    """Text im Logo-Look: creme-goldener Verlauf, schwarze Kontur, harter Schatten."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0] + spacing * max(len(text) - 1, 0)
    th = bbox[3] - bbox[1]
    pad = outline + max(shadow) + 10
    size = (tw + 2 * pad, th + 2 * pad)
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    x = pad - bbox[0]
    for ch in text:
        d.text((x, pad - bbox[1]), ch, font=font, fill=255)
        x += font.getlength(ch) + spacing
    m = np.array(mask, np.float32) / 255
    outl = np.array(mask.filter(ImageFilter.MaxFilter(outline * 2 + 1)), np.float32) / 255
    sh = np.roll(np.roll(outl, shadow[1], 0), shadow[0], 1)
    grad = np.linspace(0, 1, size[1])[:, None, None]
    fill = GOLD_HI * (1 - grad) + GOLD_LO * grad
    rgba = np.zeros((size[1], size[0], 4), np.float32)
    rgba[..., 3] = np.maximum(sh * 0.85, outl)
    rgba[..., :3] = 0
    rgba[..., :3] = rgba[..., :3] * (1 - m[..., None]) + fill * m[..., None]
    rgba[..., 3] = np.maximum(rgba[..., 3], m) * 255
    return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")


def role_text(text, font):
    """Untertitel im Stil von '—— SOUL & BLUES ——'."""
    t = gold_text(text, font, spacing=14, outline=5, shadow=(5, 5))
    line = 160
    img = Image.new("RGBA", (t.width + 2 * line + 40, t.height), 0)
    d = ImageDraw.Draw(img)
    y = t.height // 2 - 2
    col = (214, 150, 50, 255)
    for x0 in (8, t.width + line + 32):
        d.rectangle([x0 - 3, y - 6, x0 + line - 8 + 3, y + 6 + 4], fill=(0, 0, 0, 255))
        d.rectangle([x0, y - 3, x0 + line - 8, y + 4], fill=col)
    img.alpha_composite(t, (line + 20, 0))
    return img


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.42) / (H * 0.75)) ** 2)
    glow = np.clip(1 - r, 0, 1) ** 1.6
    base = np.array([10, 6, 4], np.float32)
    warm = np.array([92, 52, 18], np.float32)
    bg = base + warm * glow[..., None]
    return bg


def spotlight(cx, strength):
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    # Lichtkegel von oben
    half = 140 + yy * 0.55
    d = np.abs(xx - cx) / half
    cone = np.clip(1 - d, 0, 1) ** 1.5 * (0.25 + 0.75 * (yy / H))
    cone = np.array(Image.fromarray((cone * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return cone[..., None] * np.array([120, 80, 35], np.float32) * strength


class Renderer:
    def __init__(self, timeline, total):
        self.tl = timeline
        self.total = total
        self.bg = make_background()
        self.logo = Image.open(A("stank-logo.png")).convert("RGBA")
        self.logo_cache = {}
        rng = np.random.default_rng(7)
        self.grain = [rng.normal(0, 6, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vignette = (1 - np.clip(v - 0.55, 0, 1) * 0.9)[..., None]
        name_font = ImageFont.truetype(A("BebasNeue-Regular.ttf"), 300)
        role_font = ImageFont.truetype(A("BebasNeue-Regular.ttf"), 78)
        self.names = [gold_text(n, name_font, spacing=10, outline=12, shadow=(18, 18)) for n, _ in MEMBERS]
        self.roles = [role_text(r, role_font) for _, r in MEMBERS]
        self.spots = [spotlight(W * 0.5, 1.0)]
        self.dust = rng.uniform(0, 1, (60, 4))

    def logo_at(self, scale):
        key = round(scale, 3)
        if key not in self.logo_cache:
            if len(self.logo_cache) > 400:
                self.logo_cache.clear()
            w = max(2, int(self.logo.width * scale))
            h = max(2, int(self.logo.height * scale))
            self.logo_cache[key] = self.logo.resize((w, h), Image.LANCZOS)
        return self.logo_cache[key]

    def frame(self, fi):
        t = fi / FPS
        tl = self.tl
        img = self.bg.copy()
        hit, members_start, outro = tl["hit"], tl["members_start"], tl["outro"]

        # Bühnenlicht: im Dunkel-Teil fast aus
        light = ease_io((t - hit + 0.1) / 0.4) if t < members_start else 1.0
        if t >= outro:
            light = 1.0
        img = img * (0.15 + 0.85 * light)

        # Spotlight auf der Bühne während Vorstellung
        sp = 0.0
        for (st, en) in tl["member_spans"]:
            sp = max(sp, ease_io((t - st) / 0.5) * (1 - ease_io((t - en + 0.5) / 0.5)))
        if sp > 0:
            img = img + self.spots[0] * sp

        frame = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert("RGBA")

        # ---- Logo
        logo_alpha, scale, cx, cy, shake = 0.0, 0.62, W / 2, H / 2, 0.0
        if t >= hit:
            k = (t - hit) / 0.55
            scale = 0.62 * (1.35 - 0.35 * ease_out(k)) * (1 + 0.035 * min(1, (t - hit) / 12))
            logo_alpha = min(1, (t - hit) / 0.12)
            shake = 22 * np.exp(-(t - hit) * 6)
        if t >= members_start - 0.8:
            k = ease_io((t - members_start + 0.8) / 0.8)
            scale = scale * (1 - k) + 0.2 * k
            cx = W / 2 * (1 - k) + 270 * k
            cy = H / 2 * (1 - k) + 175 * k
        if t >= outro:
            k = ease_out((t - outro) / 1.0)
            scale = 0.2 * (1 - k) + 0.62 * k * (1 + 0.03 * (t - outro) / 6)
            cx = 270 * (1 - k) + W / 2 * k
            cy = 175 * (1 - k) + H / 2 * k
            shake = 10 * np.exp(-(t - outro) * 5)
        fade_end = self.total - 1.4
        if t > fade_end:
            logo_alpha *= 1 - ease_io((t - fade_end) / 1.2)
        if logo_alpha > 0:
            lg = self.logo_at(scale)
            if logo_alpha < 1:
                lg = lg.copy()
                lg.putalpha(lg.getchannel("A").point(lambda v: int(v * logo_alpha)))
            rng = np.random.default_rng(fi)
            dx, dy = rng.uniform(-shake, shake, 2)
            frame.alpha_composite(lg, (int(cx - lg.width / 2 + dx), int(cy - lg.height / 2 + dy)))
            # Glanzstreifen über das Logo kurz nach dem Einschlag und im Outro
            for t0 in (hit + 1.0, outro + 1.2):
                p = (t - t0) / 1.1
                if 0 < p < 1:
                    self.sheen(frame, lg, cx + dx, cy + dy, p)

        # ---- Bandmitglieder
        for i, ((role_t, name_t, end_t)) in enumerate(tl["members"]):
            if not (role_t - 0.1 <= t <= end_t + 0.6):
                continue
            out_k = ease_io((t - end_t) / 0.5)
            # Rolle: blendet ein und gleitet leicht nach oben
            rk = ease_out((t - role_t) / 0.6)
            role = self.roles[i]
            a = rk * (1 - out_k)
            ry = int(H * 0.66 + 40 * (1 - rk) - 60 * out_k)
            self.paste(frame, role, W / 2 - out_k * 300, ry, a)
            # Name: knallt von groß auf normal rein
            if t >= name_t:
                nk = ease_out((t - name_t) / 0.35)
                sc = 1.6 - 0.6 * nk
                nm = self.names[i]
                nm_s = nm.resize((int(nm.width * sc), int(nm.height * sc)), Image.BILINEAR) if sc != 1 else nm
                jitter = 14 * np.exp(-(t - name_t) * 9)
                rng = np.random.default_rng(fi + 99)
                jx, jy = rng.uniform(-jitter, jitter, 2)
                self.paste(frame, nm_s, W / 2 + jx - out_k * 300, H * 0.46 + jy, min(1, nk * 1.5) * (1 - out_k))
        # ---- Blitz beim Einschlag
        out = np.array(frame.convert("RGB"), np.float32)
        for tf, s in ((hit, 0.9), (outro, 0.35)):
            if tf <= t < tf + 0.35:
                f = s * (1 - (t - tf) / 0.35) ** 2
                out = out + (np.array([255, 230, 190], np.float32) - out) * f

        # Staubflocken im Licht
        out = self.add_dust(out, t, light)
        # Filmkorn + Vignette + Flackern
        g = self.grain[fi % len(self.grain)]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
        flicker = 1 + 0.025 * np.sin(t * 23.0) * np.sin(t * 7.3)
        out = (out * flicker + g) * self.vignette
        # Gesamtfade am Ende und Anfang
        if t > self.total - 1.2:
            out *= 1 - ease_io((t - self.total + 1.2) / 1.1)
        return out.clip(0, 255).astype(np.uint8)

    def paste(self, frame, im, cx, cy, alpha):
        if alpha <= 0.01:
            return
        if alpha < 0.999:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
        frame.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2))) if (
            0 <= int(cx - im.width / 2) and int(cx + im.width / 2) <= W
        ) else self.safe_paste(frame, im, int(cx - im.width / 2), int(cy - im.height / 2))

    @staticmethod
    def safe_paste(frame, im, x, y):
        layer = Image.new("RGBA", frame.size, 0)
        layer.paste(im, (x, y), im)
        frame.alpha_composite(layer)

    def sheen(self, frame, lg, cx, cy, p):
        w, h = lg.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        pos = (xx + yy * 0.6) / (w + h * 0.6)
        band = np.clip(1 - np.abs(pos - (p * 1.4 - 0.2)) / 0.06, 0, 1) ** 2
        a = np.array(lg.getchannel("A"), np.float32) / 255 * band * 0.55
        layer = np.zeros((h, w, 4), np.uint8)
        layer[..., :3] = 255
        layer[..., 3] = (a * 255).astype(np.uint8)
        frame.alpha_composite(Image.fromarray(layer, "RGBA"), (int(cx - w / 2), int(cy - h / 2)))

    def add_dust(self, out, t, light):
        if light < 0.05:
            return out
        for (x0, y0, sp, sz) in self.dust:
            x = int((x0 * W + np.sin(t * 0.3 + y0 * 9) * 40) % W)
            y = int((y0 * H - t * (8 + sp * 20)) % H)
            r = 1 + int(sz * 2)
            br = 60 * light * (0.4 + 0.6 * sp)
            out[max(0, y - r):y + r, max(0, x - r):x + r] += br
        return out


# ------------------------------------------------------------------ Main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "stank-intro.mp4"))
    ap.add_argument("--preview", type=float, nargs="*", help="Nur Standbilder zu diesen Zeiten rendern")
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    with tempfile.TemporaryDirectory() as tmp:
        clips = synth_voice(args.voice, tmp)

        # Zeitleiste aufbauen
        t = LEAD_IN
        starts = []
        for clip, (_, _, pause) in zip(clips, SCRIPT):
            starts.append(t)
            t += len(clip) / SR + pause
        total = t
        scene_t = lambda name: [s for s, (sc, _, _) in zip(starts, SCRIPT) if sc == name]
        members = []
        for i in range(len(MEMBERS)):
            role_t = scene_t(f"role:{i}")[0]
            name_t = scene_t(f"name:{i}")[0]
            nxt = SCRIPT.index(next(x for x in SCRIPT if x[0] == f"name:{i}")) + 1
            end_t = starts[nxt] - 0.35
            members.append((role_t, name_t, end_t))
        tl = {
            "hit": scene_t("hit")[0],
            "members_start": members[0][0],
            "outro": scene_t("outro")[0],
            "members": members,
            "member_spans": [(r, e) for r, _, e in members],
        }
        print(f"Länge: {total:.1f}s", {k: v for k, v in tl.items() if not isinstance(v, list)})

        # Audio mischen
        n = int(total * SR)
        voice = np.zeros(n, np.float32)
        for clip, s in zip(clips, starts):
            i = int(s * SR)
            voice[i:i + len(clip)] += clip[: n - i]
        voice *= 0.95 / max(1e-6, np.abs(voice).max())
        band = synth_band(total, rng)
        # Musik beginnt erst mit dem Logo-Einschlag
        gate = np.clip((np.arange(n) / SR - tl["hit"]) / 0.05, 0, 1).astype(np.float32)
        band *= gate[:, None]
        # Ducking unter der Stimme
        env = np.convolve(np.abs(voice), np.ones(int(0.25 * SR)) / (0.25 * SR), "same")
        duck = 1 - 0.55 * np.clip(env / 0.05, 0, 1)
        band *= duck[:, None]
        mix = np.stack([voice, voice], 1) * 0.9 + band * 0.55
        imp = impact(rng)
        for tt, g in ((tl["hit"], 0.7), (tl["outro"], 0.35)):
            i = int(tt * SR)
            seg = imp[: n - i]
            mix[i:i + len(seg)] += seg[:, None] * g
        mix += crackle(n, rng)[:, None]
        mix *= 0.89 / np.abs(mix).max()
        audio = os.path.join(tmp, "mix.wav")
        write_wav(audio, mix)

        r = Renderer(tl, total)
        if args.preview is not None:
            for pt in args.preview:
                Image.fromarray(r.frame(int(pt * FPS))).save(os.path.join(HERE, f"preview_{pt:05.1f}.jpg"), quality=85)
            return

        nframes = int(total * FPS)
        enc = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", audio,
            "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-tune", "grain", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out,
        ], stdin=subprocess.PIPE)
        for fi in range(nframes):
            enc.stdin.write(r.frame(fi).tobytes())
            if fi % 150 == 0:
                print(f"  Frame {fi}/{nframes}", flush=True)
        enc.stdin.close()
        enc.wait()
        print("Fertig:", args.out)


if __name__ == "__main__":
    main()
