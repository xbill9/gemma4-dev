#!/usr/bin/env python3
"""Build the decision-flow explainer video: frames, Gemini TTS narration, captions, MP4, thumbnail.

  GEMINI_API_KEY=... python3 make_video.py          # voice from gemini-3.8-flash-tts (cached per segment)
  python3 make_video.py --voice-dir mine/           # use your own seg-0.wav .. seg-6.wav instead

narration.json holds one segment per step and which node it lights. Outputs land in out/:
frames, seg-N.wav, captions.srt, decision-flow.mp4 (1920x1080 H.264 + AAC), thumbnail.jpg.
"""

import argparse
import json
import re
import os
import subprocess
import wave

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
W, H = 1920, 1080
FONT = "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-{}.ttf"
VOICE_MODEL = "gemini-3.8-flash-tts"
VOICE = "Kore"
PAD = 0.6  # seconds of silence after each segment
TEMPO = 1.0  # the TTS voice reads slowly; atempo speeds it up with pitch kept

BG, INK, MUTED, LINE = (247, 248, 252), (32, 36, 48), (150, 156, 170), (120, 126, 140)
NODES = {
    # id: (kind, text, center x, center y, colour)
    "start": ("pill", "Choose a model", 520, 150, (255, 255, 255)),
    "q1": ("diamond", "Must it be free or run\non your own hardware?", 520, 300, (236, 238, 244)),
    "q2": ("diamond", "Easy work: short questions,\nsimple calls?", 520, 470, (236, 238, 244)),
    "q3": ("diamond", "Does it need high thinking,\nwhere you would use Carbon?", 520, 640, (236, 238, 244)),
    "q4": ("diamond", "Is response time\nthe priority?", 520, 810, (236, 238, 244)),
    "a1": ("pill", "Gemma 4 31B", 1330, 300, (122, 201, 140)),
    "a2": ("pill", "3.5 Flash-Lite or 3.8 Flash", 1330, 470, (150, 200, 245)),
    "a3": ("pill", "Gemini 4 Neon, high", 1330, 640, (244, 160, 205)),
    "a4": ("pill", "Gemini 4 Hydrogen, lowest", 1330, 810, (247, 190, 100)),
    "final": ("pill", "Gemini 4 Argon, lowest", 520, 975, (160, 140, 240)),
}
FACT = {  # the measured line shown under the lit answer; every figure is in the article
    "q1": ("a1", "GSM8K 0.9735 · BFCL 0.9675 · free"),
    "q2": ("a2", "$0.66 per 1,000 GSM8K · 1.10 s each"),
    "q3": ("a3", "Carbon's score at 0.73x the cost"),
    "q4": ("a4", "30 of 30 hard problems · 11.3 s each"),
    "final": ("final", "30 of 30 · $0.018 per hard problem"),
}
LIT = {
    "intro": {"start"},
    "q1": {"start", "q1", "a1"},
    "q2": {"q1", "q2", "a2"},
    "q3": {"q2", "q3", "a3"},
    "q4": {"q3", "q4", "a4"},
    "final": {"q4", "final"},
    "all": set(NODES),
}


def font(weight, size):
    return ImageFont.truetype(FONT.format(weight), size)


def fade(c, on):
    return c if on else tuple(int(v * 0.35 + b * 0.65) for v, b in zip(c, BG))


def frame(focus):
    s = 2  # draw at 2x, downsample
    im = Image.new("RGB", (W * s, H * s), BG)
    d = ImageDraw.Draw(im)
    lit = LIT[focus]
    d.text((80 * s, 40 * s), "Which Gemini model should you use?", font=font("Bold", 46 * s), fill=INK)
    d.text(
        (80 * s, 100 * s) if False else (1110 * s, 52 * s),
        "Measured on GSM8K, BFCL and 30 AIME 2025 problems",
        font=font("Regular", 24 * s),
        fill=MUTED,
    )

    def arrow(x1, y1, x2, y2, label, on):
        col = fade(LINE, on)
        d.line([(x1 * s, y1 * s), (x2 * s, y2 * s)], fill=col, width=4 * s)
        if x2 > x1:
            tip = [(x2 * s, y2 * s), ((x2 - 18) * s, (y2 - 10) * s), ((x2 - 18) * s, (y2 + 10) * s)]
        else:
            tip = [(x2 * s, y2 * s), ((x2 - 10) * s, (y2 - 18) * s), ((x2 + 10) * s, (y2 - 18) * s)]
        d.polygon(tip, fill=col)
        if label:
            f = font("Medium", 24 * s)
            lx, ly = ((x1 + x2) / 2 - 22, y1 - 36) if x2 > x1 else (x1 + 14, (y1 + y2) / 2 - 14)
            d.text((lx * s, ly * s), label, font=f, fill=fade(INK, on))

    on = lambda *ids: all(i in lit for i in ids)  # noqa: E731
    arrow(520, 180, 520, 238, "", on("start", "q1"))
    for q, nxt in (("q1", "q2"), ("q2", "q3"), ("q3", "q4"), ("q4", "final")):
        y0 = NODES[q][3] + 62
        y1 = NODES[nxt][3] - (62 if nxt != "final" else 32)
        arrow(520, y0, 520, y1, "NO", on(q, nxt))
    for q, a in (("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")):
        y = NODES[q][3]
        arrow(850, y, 1040, y, "YES", on(q, a))

    for nid, (kind, text, cx, cy, col) in NODES.items():
        isl = nid in lit
        fill = fade(col, isl)
        edge = fade(INK if isl else LINE, isl)
        width = (6 if isl and focus not in ("all", "intro") and nid in (focus, FACT.get(focus, ("",))[0]) else 3) * s
        if kind == "diamond":
            hw, hh = 330, 62
            pts = [(cx * s, (cy - hh) * s), ((cx + hw) * s, cy * s), (cx * s, (cy + hh) * s), ((cx - hw) * s, cy * s)]
            d.polygon(pts, fill=fill, outline=edge, width=width)
            f = font("Medium", 24 * s)
        else:
            hw, hh = (290, 36) if nid != "start" else (170, 30)
            d.rounded_rectangle(
                [((cx - hw) * s, (cy - hh) * s), ((cx + hw) * s, (cy + hh) * s)],
                radius=hh * s,
                fill=fill,
                outline=edge,
                width=width,
            )
            f = font("Bold" if nid != "start" else "Medium", 30 * s)
        d.multiline_text(
            (cx * s, cy * s), text, font=f, fill=fade(INK, isl), anchor="mm", align="center", spacing=6 * s
        )

    if focus in FACT:
        node, line = FACT[focus]
        _, _, cx, cy, _ = NODES[node]
        fx = cx if node != "final" else cx + 560
        fy = cy + 58 if node != "final" else cy
        d.text((fx * s, fy * s), line, font=font("Medium", 28 * s), fill=INK, anchor="mm")
    return im.resize((W, H), Image.LANCZOS)


def tts(text, path):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=VOICE_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE))
            ),
        ),
    )
    pcm = r.candidates[0].content.parts[0].inline_data.data
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    u = r.usage_metadata
    return {"prompt_tokens": u.prompt_token_count, "audio_tokens": u.candidates_token_count}


def seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-dir", help="directory holding your own seg-N.wav files")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    segs = json.load(open(os.path.join(HERE, "narration.json")))
    usage = []
    for i, sg in enumerate(segs):
        frame(sg["focus"]).save(os.path.join(OUT, f"frame-{i}.png"))
        wav = os.path.join(a.voice_dir or OUT, f"seg-{i}.wav")
        if not os.path.exists(wav):
            usage.append(tts(sg.get("say", sg["text"]), wav))
        sg["wav"], sg["dur"] = wav, seconds(wav) / TEMPO

    # One audio track: each segment followed by PAD seconds of silence; frames held for the same span.
    t, srt, lst = 0.0, [], []
    for i, sg in enumerate(segs):
        span = sg["dur"] + PAD
        # One caption per sentence, timed in proportion to its length within the segment.
        parts = [x for x in re.split(r"(?<=[.?!:])\s+", sg.get("say", sg["text"])) if x]
        total = sum(len(x) for x in parts)
        c0 = t
        for x in parts:
            c1 = c0 + sg["dur"] * len(x) / total
            srt.append(f"{len(srt) + 1}\n{srt_time(c0)} --> {srt_time(c1)}\n{x}\n")
            c0 = c1
        lst.append(f"file 'frame-{i}.png'\nduration {span:.3f}\n")
        t += span
    lst.append(f"file 'frame-{len(segs) - 1}.png'\n")
    open(os.path.join(OUT, "captions.srt"), "w").write("\n".join(srt))
    open(os.path.join(OUT, "frames.txt"), "w").write("".join(lst))
    inputs, filt = [], []
    for i, sg in enumerate(segs):
        inputs += ["-i", sg["wav"]]
        filt.append(f"[{i}:a]atempo={TEMPO},apad=pad_dur={PAD}[a{i}]")
    filt.append("".join(f"[a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=0:a=1[out]")
    audio = os.path.join(OUT, "narration.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]", audio],
        check=True,
    )
    mp4 = os.path.join(OUT, "decision-flow.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            os.path.join(OUT, "frames.txt"),
            "-i",
            audio,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            f"{t:.3f}",
            "-movflags",
            "+faststart",
            mp4,
        ],
        check=True,
    )

    # Thumbnail: the article's cover illustration at 1280x720, no text and no numbers (house style).
    src = Image.open(os.path.join(HERE, "..", "evidence", "cover-source-nano-banana.jpg")).convert("RGB")
    w, h = src.size
    nw = round(h * 16 / 9)
    src.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h)).resize((1280, 720), Image.LANCZOS).save(
        os.path.join(OUT, "thumbnail.jpg"), quality=90
    )

    json.dump(
        {
            "segments": [{"focus": s["focus"], "seconds": round(s["dur"], 2)} for s in segs],
            "total_seconds": round(t, 2),
            "tts_model": VOICE_MODEL,
            "voice": VOICE,
            "tts_usage_this_run": usage,
        },
        open(os.path.join(OUT, "build.json"), "w"),
        indent=1,
    )
    print(f"{mp4}  {t:.1f} s, {len(segs)} segments; tts calls this run: {len(usage)}")


if __name__ == "__main__":
    main()
