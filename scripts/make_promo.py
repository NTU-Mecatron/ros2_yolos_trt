#!/usr/bin/env python3
# Copyright 2026 YOLOs-TRT Team
# SPDX-License-Identifier: AGPL-3.0
"""Build the LinkedIn promo video from the recorded demo GIFs.

Renders title/stat/CTA cards with Pillow, crops the rviz chrome off the demo
GIFs so only the two image panels remain, composites the pair into a designed
frame, and cross-fades the whole thing into a single 4:5 MP4 sized for the
LinkedIn feed.

The numbers burned into the video come from benchmarks/results.json, so the
video cannot drift from the README's published table.

Requires: ffmpeg (with libx264) and Pillow.

Usage:
  ./scripts/make_promo.py                       # -> promo/ros2_yolos_trt_promo.mp4
  ./scripts/make_promo.py --aspect square       # 1080x1080 re-cut
  ./scripts/make_promo.py --keep-work           # leave intermediates for inspection
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- design tokens

BG = (10, 12, 16)
PANEL = (18, 22, 29)
RULE = (38, 46, 58)
FG = (242, 245, 248)
MUTED = (138, 151, 166)
DIM = (99, 110, 123)
GREEN = (118, 185, 0)        # NVIDIA
BLUE = (74, 158, 255)        # ROS
WARN = (255, 176, 46)

FONTS = {
    "black": "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "reg": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "mono": "/System/Library/Fonts/Menlo.ttc",
}

ASPECTS = {"portrait": (1080, 1350), "square": (1080, 1080), "landscape": (1920, 1080)}
FPS = 30
XFADE = 0.3

# Crop geometry of the recorded GIFs (560x634 rviz captures). Both image panels
# sit at x=64 w=420; the upper panel's image starts at y=132, the lower at
# y=413. rviz stacks them in a different order per clip, so each task records
# which offset holds the camera feed and which holds the annotated output.
GIF_X, GIF_W, GIF_H = 64, 420, 220
PANEL_TOP, PANEL_BOTTOM = 132, 413

TASKS = [
    {"key": "detection", "title": "OBJECT DETECTION", "model": "yolo11n · 640",
     "cam": PANEL_TOP, "out": PANEL_BOTTOM, "start": 1.2,
     "msg": "vision_msgs/Detection2DArray"},
    {"key": "segmentation", "title": "INSTANCE SEGMENTATION", "model": "yolo11n-seg · 640",
     "cam": PANEL_TOP, "out": PANEL_BOTTOM, "start": 1.2,
     "msg": "Detection2DArray + mono8 label mask"},
    {"key": "pose", "title": "POSE ESTIMATION", "model": "yolo11n-pose · 640",
     "cam": PANEL_BOTTOM, "out": PANEL_TOP, "start": 2.0,
     "msg": "Detection2DArray + KeyPoint2DArray"},
    {"key": "depth", "title": "MONOCULAR METRIC DEPTH", "model": "yolo26n-depth · 768",
     "cam": PANEL_BOTTOM, "out": PANEL_TOP, "start": 2.0,
     "msg": "sensor_msgs/Image — 32FC1, metres"},
]

CLIP_SECONDS = 4.0


def font(kind, size):
    return ImageFont.truetype(FONTS[kind], size)


def wh(f, s):
    x0, y0, x1, y1 = f.getbbox(s)
    return x1 - x0, y1 - y0


def tracked(draw, xy, s, f, fill, track=0):
    """draw.text with manual letter-spacing (Pillow has none)."""
    x, y = xy
    for ch in s:
        draw.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + track
    return x - xy[0] - track


def tracked_width(f, s, track=0):
    return sum(f.getlength(c) + track for c in s) - track


# ------------------------------------------------------------------ card canvas

def canvas(size, glow=GREEN, glow_at=(0.16, 0.10), strength=0.20):
    """Dark ground with one soft off-centre glow and a hairline frame."""
    w, h = size
    im = Image.new("RGB", size, BG)
    r = int(max(w, h) * 1.5)
    mask = ImageOps.invert(Image.radial_gradient("L").resize((r, r), Image.LANCZOS))
    mask = mask.point(lambda v: int((v / 255.0) ** 2.2 * 255 * strength))
    wash = Image.new("RGB", (r, r), glow)
    im.paste(wash, (int(w * glow_at[0]) - r // 2, int(h * glow_at[1]) - r // 2), mask)
    d = ImageDraw.Draw(im)
    m = int(w * 0.037)
    d.rectangle([m, m, w - m - 1, h - m - 1], outline=RULE, width=1)
    return im


def wordmark(d, size, y, accent=GREEN):
    """Small repo wordmark, centred."""
    w, _ = size
    f = font("mono", int(w * 0.0213))
    s = "Geekgineer/ros2_yolos_trt"
    tw = tracked_width(f, s, 1)
    tracked(d, ((w - tw) / 2, y), s, f, accent, 1)


def kicker(d, size, y, s, colour=MUTED):
    w, _ = size
    f = font("bold", int(w * 0.0176))
    tw = tracked_width(f, s, int(w * 0.0037))
    tracked(d, ((w - tw) / 2, y), s, f, colour, int(w * 0.0037))


def star(d, centre, r, colour, points=5):
    """Five-point star — Arial Black carries no U+2605 glyph."""
    import math
    cx, cy = centre
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * 0.42
        a = -math.pi / 2 + i * math.pi / points
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    d.polygon(pts, fill=colour)


def rule(d, size, y, width_frac=0.10, colour=GREEN, thick=4):
    w, _ = size
    half = w * width_frac / 2
    d.rectangle([w / 2 - half, y, w / 2 + half, y + thick], fill=colour)


# ------------------------------------------------------------------------ cards

def card_hook(size):
    w, h = size
    im = canvas(size)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    kicker(d, size, h * 0.16, "ROS 2   \u00b7   TENSORRT   \u00b7   C++17")
    lines = [("SIX YOLO TASKS.", FG), ("ONE ROS 2 PACKAGE.", FG),
             ("ZERO CPU", GREEN), ("PREPROCESSING.", GREEN)]
    f = font("black", int(74 * s))
    lh = int(92 * s)
    y = h * 0.285
    for text, colour in lines:
        tw, _ = wh(f, text)
        d.text(((w - tw) / 2, y), text, font=f, fill=colour)
        y += lh
    rule(d, size, y + int(22 * s))
    f2 = font("reg", int(31 * s))
    sub = "Detection \u00b7 Segmentation \u00b7 Pose \u00b7 OBB \u00b7 Classification \u00b7 Depth"
    tw, _ = wh(f2, sub)
    d.text(((w - tw) / 2, y + int(62 * s)), sub, font=f2, fill=MUTED)

    # the pipeline itself, as one line of plumbing
    box = [w * 0.075, h * 0.705, w * 0.925, h * 0.775]
    d.rounded_rectangle(box, radius=int(10 * s), fill=(13, 16, 21), outline=RULE, width=1)
    fm = font("mono", int(19 * s))
    flow = [("/camera/image_raw", MUTED), ("  ->  ", DIM),
            ("one CUDA kernel", GREEN), ("  ->  ", DIM),
            ("TensorRT", GREEN), ("  ->  ", DIM), ("vision_msgs", MUTED)]
    total = sum(fm.getlength(t) for t, _ in flow)
    while total > (box[2] - box[0]) * 0.92:
        fm = font("mono", fm.size - 1)
        total = sum(fm.getlength(t) for t, _ in flow)
    x = (w - total) / 2
    ty = (box[1] + box[3]) / 2 - int(12 * s)
    for t, colour in flow:
        d.text((x, ty), t, font=fm, fill=colour)
        x += fm.getlength(t)

    wordmark(d, size, h * 0.845)
    return im


def card_tasks(size, rows):
    w, h = size
    im = canvas(size, glow=BLUE, glow_at=(0.85, 0.12), strength=0.16)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    kicker(d, size, h * 0.10, "ONE PACKAGE · SIX LIFECYCLE NODES")
    f = font("black", int(60 * s))
    t = "ros2_yolos_trt"
    tw, _ = wh(f, t)
    d.text(((w - tw) / 2, h * 0.145), t, font=f, fill=FG)
    f2 = font("reg", int(30 * s))
    t2 = "TensorRT-native YOLO inference for ROS 2"
    tw, _ = wh(f2, t2)
    d.text(((w - tw) / 2, h * 0.205), t2, font=f2, fill=MUTED)

    fname = font("bold", int(30 * s))
    fmsg = font("mono", int(20 * s))
    x0, x1 = w * 0.085, w * 0.915
    top = h * 0.275
    row_h = (h * 0.86 - top) / len(rows)
    for i, (name, msg, hot) in enumerate(rows):
        y = top + i * row_h
        d.rounded_rectangle([x0, y, x1, y + row_h - int(12 * s)], radius=int(10 * s),
                            fill=PANEL, outline=GREEN if hot else RULE, width=2 if hot else 1)
        d.rectangle([x0, y + int(14 * s), x0 + int(5 * s), y + row_h - int(26 * s)],
                    fill=GREEN if hot else BLUE)
        d.text((x0 + int(28 * s), y + int(16 * s)), name, font=fname, fill=GREEN if hot else FG)
        d.text((x0 + int(28 * s), y + int(56 * s)), msg, font=fmsg, fill=MUTED)
        if hot:
            tag = "NOT IN THE ONNX SIBLING"
            ft = font("bold", int(17 * s))
            tgw = tracked_width(ft, tag, 1)
            d.rounded_rectangle([x1 - tgw - int(34 * s), y + int(18 * s), x1 - int(16 * s), y + int(46 * s)],
                                radius=int(6 * s), fill=(26, 42, 10))
            tracked(d, (x1 - tgw - int(25 * s), y + int(24 * s)), tag, ft, GREEN, 1)
    wordmark(d, size, h - h * 0.075)
    return im


def demo_background(size, task, stat, geo):
    """Static frame the two video panels are overlaid onto."""
    w, h = size
    im = canvas(size, glow=GREEN if task["key"] != "depth" else WARN,
                glow_at=(0.80, 0.85), strength=0.14)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    px, pw, ph = geo["x"], geo["w"], geo["h"]
    ya, yb = geo["ya"], geo["yb"]

    kicker(d, size, geo["kicker_y"], "LIVE rviz2 CAPTURE · REAL VIDEO")
    f = font("black", int(geo["title_size"] * s))
    tw, _ = wh(f, task["title"])
    while tw > w * 0.86:
        f = font("black", f.size - 2)
        tw, _ = wh(f, task["title"])
    d.text(((w - tw) / 2, geo["title_y"]), task["title"], font=f, fill=FG)
    f2 = font("mono", int(24 * s))
    tw, _ = wh(f2, task["model"])
    d.text(((w - tw) / 2, geo["model_y"]), task["model"], font=f2, fill=MUTED)

    for y in (ya, yb):
        d.rectangle([px - 1, y - 1, px + pw, y + ph], outline=RULE, width=1)

    # footer: measured latency, with the qualifier the README insists on
    fnum = font("black", int(geo["num_size"] * s))
    flab = font("bold", int(26 * s))
    num, lab = f"{stat['inference_ms_mean']:.2f} ms", "  mean inference"
    nw, _ = wh(fnum, num)
    lw, _ = wh(flab, lab)
    x = (w - (nw + lw)) / 2
    d.text((x, geo["num_y"]), num, font=fnum, fill=GREEN)
    d.text((x + nw, geo["num_y"] + int(geo["num_size"] * s * 0.42)), lab, font=flab, fill=MUTED)
    fq = font("mono", int(19 * s))
    q = f"RTX 4090 · fp16 · p95 {stat['inference_ms_p95']:.1f} ms · node-reported on ~/timing"
    tw, _ = wh(fq, q)
    d.text(((w - tw) / 2, geo["qual_y"]), q, font=fq, fill=DIM)
    return im


def demo_overlay(size, task, geo):
    """Transparent layer of corner pills that sit on top of the video panels."""
    w, h = size
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    f = font("bold", int(20 * s))
    for y, label, colour in ((geo["ya"], "CAMERA IN", (210, 220, 232)),
                             (geo["yb"], "NODE OUT", GREEN)):
        tw = tracked_width(f, label, 1)
        x, yy = geo["x"] + int(14 * s), y + int(14 * s)
        d.rounded_rectangle([x, yy, x + tw + int(26 * s), yy + int(32 * s)],
                            radius=int(6 * s), fill=(8, 10, 14, 215))
        tracked(d, (x + int(13 * s), yy + int(7 * s)), label, f, colour, 1)
    return im


def card_stats(size, rows):
    w, h = size
    im = canvas(size, glow=BLUE, glow_at=(0.12, 0.88), strength=0.16)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    kicker(d, size, h * 0.085, "REPRODUCIBLE, NOT MARKETING")
    f = font("black", int(62 * s))
    for i, line in enumerate(("MEASURED.", "NOT ESTIMATED.")):
        tw, _ = wh(f, line)
        d.text(((w - tw) / 2, h * 0.125 + i * int(76 * s)), line, font=f,
               fill=FG if i == 0 else GREEN)

    x0, x1 = w * 0.085, w * 0.915
    top = h * 0.295
    fh = font("bold", int(20 * s))
    cols = [(0.00, "TASK", "l"), (0.52, "MEAN ms", "r"), (0.76, "p95 ms", "r"), (1.00, "FPS", "r")]
    for frac, label, align in cols:
        x = x0 + (x1 - x0) * frac
        lw, _ = wh(fh, label)
        d.text((x - (lw if align == "r" else 0), top), label, font=fh, fill=DIM)
    d.rectangle([x0, top + int(34 * s), x1, top + int(35 * s)], fill=RULE)

    fk = font("bold", int(31 * s))
    fv = font("mono", int(29 * s))
    fvb = font("black", int(31 * s))
    row_h = int(88 * s)
    y = top + int(56 * s)
    for r in rows:
        d.text((x0, y), r["task"].capitalize(), font=fk, fill=FG)
        for frac, val, fnt, colour in ((0.52, f"{r['inference_ms_mean']:.2f}", fv, FG),
                                       (0.76, f"{r['inference_ms_p95']:.1f}", fv, MUTED),
                                       (1.00, f"{r['inference_fps']:.0f}", fvb, GREEN)):
            x = x0 + (x1 - x0) * frac
            vw, _ = wh(fnt, val)
            d.text((x - vw, y), val, font=fnt, fill=colour)
        y += row_h
        d.rectangle([x0, y - int(22 * s), x1, y - int(21 * s)], fill=(24, 29, 37))

    fq = font("reg", int(24 * s))
    qual = [
        "NVIDIA RTX 4090 \u00b7 TensorRT 10.16 \u00b7 ROS 2 Jazzy \u00b7 fp16 engines.",
        "Latency is what the nodes report on ~/timing \u2014 inference plus",
        "message conversion. FPS is 1000/mean: one node, one GPU, nothing",
        "else running. It is not a model-capacity figure.",
    ]
    yq = y + int(16 * s)
    for line in qual:
        lw, _ = wh(fq, line)
        d.text(((w - lw) / 2, yq), line, font=fq, fill=MUTED)
        yq += int(35 * s)

    fn = font("bold", int(23 * s))
    note = "Segmentation's tail is CPU mask postprocessing \u2014 documented, not hidden."
    lw, _ = wh(fn, note)
    while lw > w * 0.86:
        fn = font("bold", fn.size - 1)
        lw, _ = wh(fn, note)
    d.text(((w - lw) / 2, yq + int(10 * s)), note, font=fn, fill=WARN)

    # flows below the qualifier so the shorter aspects cannot collide with it
    box_top = max(yq + int(46 * s), h * 0.845)
    box = [w * 0.085, box_top, w * 0.915, box_top + int(74 * s)]
    d.rounded_rectangle(box, radius=int(10 * s), fill=(13, 16, 21), outline=RULE, width=1)
    fm = font("mono", int(20 * s))
    cmd = "$ ./scripts/run_benchmarks.sh /models video.mp4 250"
    cw, _ = wh(fm, cmd)
    while cw > (box[2] - box[0]) * 0.90:
        fm = font("mono", fm.size - 1)
        cw, _ = wh(fm, cmd)
    d.text(((w - cw) / 2, (box[1] + box[3]) / 2 - int(13 * s)), cmd, font=fm, fill=GREEN)
    wordmark(d, size, h - h * 0.052)
    return im


def card_depth(size):
    w, h = size
    im = canvas(size, glow=WARN, glow_at=(0.20, 0.18), strength=0.17)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    kicker(d, size, h * 0.115, "THE REASON THIS PACKAGE EXISTS")
    f = font("black", int(74 * s))
    for i, line in enumerate(("MONOCULAR", "METRIC DEPTH")):
        tw, _ = wh(f, line)
        d.text(((w - tw) / 2, h * 0.165 + i * int(90 * s)), line, font=f, fill=FG)
    f2 = font("black", int(52 * s))
    t = "32FC1. In metres."
    tw, _ = wh(f2, t)
    d.text(((w - tw) / 2, h * 0.355), t, font=f2, fill=WARN)
    rule(d, size, h * 0.435, colour=WARN)

    f3 = font("reg", int(29 * s))
    blurb = ["Not a normalised 0-1 map. Real distance,",
             "in the unit ROS already expects \u2014 so it",
             "drops into existing tooling unchanged:"]
    for i, line in enumerate(blurb):
        tw, _ = wh(f3, line)
        d.text(((w - tw) / 2, h * 0.485 + i * int(40 * s)), line, font=f3, fill=MUTED)

    box = [w * 0.075, h * 0.632, w * 0.925, h * 0.752]
    d.rounded_rectangle(box, radius=int(12 * s), fill=(13, 16, 21), outline=RULE, width=1)
    fm = font("mono", int(21 * s))
    code = ["$ ros2 run depth_image_proc point_cloud_xyz \\",
            "    --ros-args -r image_rect:=\\",
            "      /yolos_trt_depth/depth/image"]
    for i, line in enumerate(code):
        x, y = box[0] + int(26 * s), box[1] + int(24 * s) + i * int(33 * s)
        if i == 0:
            d.text((x, y), "$ ", font=fm, fill=GREEN)
            x += fm.getlength("$ ")
            line = line[2:]
        d.text((x, y), line, font=fm, fill=(198, 208, 220))
    f4 = font("bold", int(27 * s))
    t4 = "Point cloud out of one RGB camera."
    tw, _ = wh(f4, t4)
    d.text(((w - tw) / 2, h * 0.795), t4, font=f4, fill=FG)
    wordmark(d, size, h - h * 0.085)
    return im


def card_cta(size):
    w, h = size
    im = canvas(size, glow=GREEN, glow_at=(0.5, 0.5), strength=0.22)
    d = ImageDraw.Draw(im)
    s = w / 1080.0
    kicker(d, size, h * 0.145, "OPEN SOURCE \u00b7 AGPL-3.0")
    f = font("black", int(58 * s))
    for i, line in enumerate(("BUILD IT", "INTO YOUR STACK")):
        tw, _ = wh(f, line)
        d.text(((w - tw) / 2, h * 0.195 + i * int(72 * s)), line, font=f, fill=FG)
    rule(d, size, h * 0.355)

    box = [w * 0.07, h * 0.415, w * 0.93, h * 0.505]
    d.rounded_rectangle(box, radius=int(12 * s), fill=PANEL, outline=GREEN, width=2)
    fm = font("bold", int(31 * s))
    t = "github.com/Geekgineer/ros2_yolos_trt"
    tw, _ = wh(fm, t)
    while tw > (box[2] - box[0]) * 0.90:
        fm = font("bold", fm.size - 1)
        tw, _ = wh(fm, t)
    d.text(((w - tw) / 2, (box[1] + box[3]) / 2 - int(20 * s)), t, font=fm, fill=GREEN)

    f3 = font("reg", int(27 * s))
    for i, line in enumerate(["ROS 2 Humble & Jazzy · TensorRT 10.x · CUDA 12+",
                              "Point it at an .onnx — the engine builds and caches itself.",
                              "Six composable lifecycle nodes. 51 unit tests, no GPU needed."]):
        tw, _ = wh(f3, line)
        d.text(((w - tw) / 2, h * 0.565 + i * int(43 * s)), line, font=f3, fill=MUTED)

    f4 = font("black", int(34 * s))
    t4 = "Star it if it saves you a week."
    tw, _ = wh(f4, t4)
    ty = h * 0.725
    star(d, ((w - tw) / 2 - int(46 * s), ty + int(17 * s)), int(20 * s), GREEN)
    d.text(((w - tw) / 2, ty), t4, font=f4, fill=FG)
    wordmark(d, size, h * 0.855)
    return im


# ------------------------------------------------------------------------ build

def geometry(size):
    """Panel + text positions, derived so every aspect ratio stays composed."""
    w, h = size
    s = w / 1080.0
    gap = int(14 * s)
    head = h * 0.186          # kicker + task title + model line
    foot = h * 0.142          # latency figure + its qualifier
    ph = (h - head - foot - gap) / 2
    pw = ph * GIF_W / GIF_H
    if pw > w * 0.845:        # square/landscape: width is the binding constraint
        pw = w * 0.845
        ph = pw * GIF_H / GIF_W
    block = 2 * ph + gap
    ya = head + (h - head - foot - block) / 2
    foot_top = ya + block
    return {
        "x": int((w - pw) / 2), "w": int(pw), "h": int(ph),
        "ya": int(ya), "yb": int(ya + ph + gap),
        "kicker_y": head * 0.24, "title_y": head * 0.35,
        "model_y": head * 0.68, "title_size": 54,
        "num_size": 52,
        "num_y": foot_top + (h - foot_top) * 0.13,
        "qual_y": foot_top + (h - foot_top) * 0.60,
    }


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(" ".join(str(c) for c in cmd) + "\n" + r.stderr[-3000:] + "\n")
        raise SystemExit(f"ffmpeg failed ({r.returncode})")


def still(png, seconds, out, size):
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", f"{seconds}", "-i", png,
         "-vf", f"scale={size[0]}:{size[1]},fps={FPS},setsar=1,format=yuv420p",
         "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-video_track_timescale", "90000",
         out])


def demo(gif, bg, ov, task, geo, seconds, out, size):
    g = geo
    chain = (
        f"[1:v]trim=start={task['start']}:duration={seconds},setpts=PTS-STARTPTS,fps={FPS},"
        f"crop={GIF_W}:{GIF_H}:{GIF_X}:{task['cam']},scale={g['w']}:{g['h']}:flags=lanczos,setsar=1[cam];"
        f"[2:v]trim=start={task['start']}:duration={seconds},setpts=PTS-STARTPTS,fps={FPS},"
        f"crop={GIF_W}:{GIF_H}:{GIF_X}:{task['out']},scale={g['w']}:{g['h']}:flags=lanczos,setsar=1[out];"
        f"[0:v]scale={size[0]}:{size[1]},fps={FPS},setsar=1[bg];"
        f"[bg][cam]overlay={g['x']}:{g['ya']}:shortest=1[s1];"
        f"[s1][out]overlay={g['x']}:{g['yb']}[s2];"
        f"[s2][3:v]overlay=0:0,format=yuv420p[v]"
    )
    run(["ffmpeg", "-v", "error", "-y",
         "-loop", "1", "-t", f"{seconds}", "-i", bg,
         "-ignore_loop", "0", "-i", gif,
         "-ignore_loop", "0", "-i", gif,
         "-loop", "1", "-t", f"{seconds}", "-i", ov,
         "-filter_complex", chain, "-map", "[v]", "-t", f"{seconds}",
         "-c:v", "libx264", "-crf", "17", "-preset", "medium",
         "-video_track_timescale", "90000", out])


def stitch(segments, durations, out):
    inputs = []
    for s in segments:
        inputs += ["-i", str(s)]
    parts, prev, offset = [], "0:v", 0.0
    for i in range(1, len(segments)):
        offset += durations[i - 1] - XFADE
        label = f"x{i}"
        parts.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}[{label}]")
        prev = label
    graph = ";".join(parts) if parts else None
    cmd = ["ffmpeg", "-v", "error", "-y"] + inputs
    if graph:
        cmd += ["-filter_complex", graph + f";[{prev}]format=yuv420p[v]", "-map", "[v]"]
    cmd += ["-c:v", "libx264", "-crf", "19", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.2", "-r", str(FPS),
            "-movflags", "+faststart", str(out)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", default="portrait", choices=sorted(ASPECTS))
    ap.add_argument("--out", default=str(ROOT / "promo"))
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    size = ASPECTS[args.aspect]
    outdir = pathlib.Path(args.out)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    stats = {r["task"]: r for r in json.loads((ROOT / "benchmarks/results.json").read_text())}
    order = ["detection", "segmentation", "pose", "depth"]
    rows = [stats[t] for t in order if t in stats]

    geo = geometry(size)
    plan = []

    def add_card(name, image, seconds):
        p = work / f"{name}.png"
        image.save(p)
        seg = work / f"{name}.mp4"
        still(str(p), seconds, str(seg), size)
        plan.append((seg, seconds))
        return p

    hook = add_card("00_hook", card_hook(size), 2.8)
    add_card("01_tasks", card_tasks(size, [
        ("Detection", "vision_msgs/Detection2DArray", False),
        ("Segmentation", "Detection2DArray + mono8 mask", False),
        ("Pose", "Detection2DArray + KeyPoint2DArray", False),
        ("Oriented boxes", "OBBDetection2DArray", False),
        ("Classification", "vision_msgs/Classification", False),
        ("Metric depth", "sensor_msgs/Image — 32FC1, metres", True),
    ]), 3.4)

    for i, task in enumerate(TASKS):
        gif = ROOT / "assets" / f"{task['key']}.gif"
        if not gif.exists():
            sys.stderr.write(f"skip {task['key']}: {gif} missing\n")
            continue
        st = stats.get(task["key"])
        if st is None:
            sys.stderr.write(f"skip {task['key']}: no benchmark row\n")
            continue
        bg = work / f"1{i}_{task['key']}_bg.png"
        ov = work / f"1{i}_{task['key']}_ov.png"
        demo_background(size, task, st, geo).save(bg)
        demo_overlay(size, task, geo).save(ov)
        seg = work / f"1{i}_{task['key']}.mp4"
        demo(str(gif), str(bg), str(ov), task, geo, CLIP_SECONDS, str(seg), size)
        plan.append((seg, CLIP_SECONDS))

    add_card("20_stats", card_stats(size, rows), 5.0)
    add_card("21_depth", card_depth(size), 4.2)
    add_card("22_cta", card_cta(size), 3.6)

    final = outdir / f"ros2_yolos_trt_promo_{args.aspect}.mp4"
    stitch([p for p, _ in plan], [d for _, d in plan], final)
    shutil.copy(hook, outdir / f"poster_{args.aspect}.png")

    total = sum(d for _, d in plan) - XFADE * (len(plan) - 1)
    print(f"{final}  ({size[0]}x{size[1]}, {total:.1f}s, {final.stat().st_size/1e6:.1f} MB)")
    print(f"{outdir / f'poster_{args.aspect}.png'}  (poster / thumbnail)")
    if not args.keep_work:
        shutil.rmtree(work)


if __name__ == "__main__":
    sys.exit(main())
