# Stank! – Intro-Video

`stank-intro.mp4` (1920×1080, 30 fps, ca. 54 s): Das Logo schlägt ein, danach stellt ein rauer Erzähler die Band vor.

| Wer | Was |
|---|---|
| Hardy | Gitarre |
| Oli | Gesang & Gitarre |
| Stefan | Schlagzeug |
| Thomas | Bass |

## Neu bauen

```bash
pip install piper-tts numpy pillow scipy        # ffmpeg mit rubberband wird zusätzlich benötigt
curl -LO https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-de-karlsson-low.tar.gz
tar xzf voice-de-karlsson-low.tar.gz
python3 make_logo_alpha.py assets/stank-logo.webp assets/stank-logo.png   # Logo freistellen
python3 build.py --voice de-karlsson-low.onnx                             # -> stank-intro.mp4
python3 build.py --voice de-karlsson-low.onnx --preview 7 20 48           # nur Standbilder
```

Den Sprechertext, die Pausen und die Besetzung legst du oben in `build.py` fest (`SCRIPT`, `MEMBERS`).
Mit `[[ ... ]]` gibst du Lautschrift direkt vor, zum Beispiel `[[ sˈtɛŋk ]]` für „Stänk“.

**Stimme:** Das ist die freie deutsche Piper-Stimme „Karlsson“. Sie ist tiefer gepitcht und hat Kompression und etwas Raumhall bekommen.
Sie ist keine Nachbildung einer realen Person.
**Musik:** Ein langsamer 12-Takt-Shuffle in E, komplett im Skript synthetisiert (Bass, Besen, Orgel, Knistern).
**Schriften:** Bebas Neue (SIL OFL), siehe `assets/OFL-BebasNeue.txt`.
