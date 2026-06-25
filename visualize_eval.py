#!/usr/bin/env python3
"""Beautiful eval visualizations for LocateAnything-3B fine-tuning.

Reads `eval_preds.json` (produced by eval_compare.py) and renders:
  1. eval_vis/panel_finetuned.png  — grid of fine-tuned predictions, boxes
     color-coded TP (green) / FP (red) / FN missed-GT (amber), drawn with
     supervision's rounded-box annotator on a dark theme.
  2. eval_vis/base_vs_finetuned.png — side-by-side base vs fine-tuned for the
     most object-rich val images (shows the improvement at a glance).
  3. eval_vis/metrics.png / .html   — plotly grouped bars, F1@0.5 & @0.95,
     base vs fine-tuned per category.
"""
import argparse
import json
import os

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# github-dark vibrant palette
BG = (13, 17, 23)
PANEL = (22, 27, 34)
FG = (230, 237, 243)
SUB = (139, 148, 158)
C_TP = (63, 185, 80)      # green
C_FP = (248, 81, 73)      # red
C_FN = (210, 153, 34)     # amber
C_GT = (88, 166, 255)     # blue

# Optional filename-substring -> (label, color) chips shown on each card, e.g.
# for split / difficulty buckets. Empty by default; customize for your dataset.
TAGS = ()


def _font(size, bold=True):
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(base, name), size)
    except Exception:
        return ImageFont.load_default()


def classify(preds, gts, thr=0.5):
    """Greedy match -> (tp_boxes, fp_boxes, fn_boxes)."""
    used = [False] * len(gts)
    tp, fp = [], []
    for p in preds:
        best, bi = thr, -1
        for i, g in enumerate(gts):
            if used[i]:
                continue
            v = iou(p, g)
            if v >= best:
                best, bi = v, i
        if bi >= 0:
            used[bi] = True
            tp.append(p)
        else:
            fp.append(p)
    fn = [g for i, g in enumerate(gts) if not used[i]]
    return tp, fp, fn


def _scale_boxes(boxes, w, h):
    return np.array([[b[0]/1000*w, b[1]/1000*h, b[2]/1000*w, b[3]/1000*h]
                     for b in boxes], dtype=float).reshape(-1, 4)


def annotate(img_rgb, groups, thick=None):
    """groups: list of (items, color_rgb) where items = list of (box_norm, label).
    Draws sharp rectangles with the label at each box's top-left corner."""
    h, w = img_rgb.shape[:2]
    thick = thick or max(2, round(min(w, h) / 320))
    ts = max(0.42, min(w, h) / 1500)
    scene = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for items, color in groups:
        if not items:
            continue
        boxes = [b for b, _ in items]
        labels = [str(l) for _, l in items]
        xyxy = _scale_boxes(boxes, w, h)
        dets = sv.Detections(xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int))
        col = sv.Color(*color)
        box = sv.BoxAnnotator(color=col, thickness=thick)
        lab = sv.LabelAnnotator(color=col, text_color=sv.Color(13, 17, 23),
                                text_scale=ts, text_thickness=max(1, round(ts*2)),
                                text_padding=int(4*ts)+2, text_position=sv.Position.TOP_LEFT)
        scene = box.annotate(scene=scene, detections=dets)
        scene = lab.annotate(scene=scene, detections=dets, labels=labels)
    return cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)


def _labeled(sample, key, cats):
    """Return (tp, fp, fn) lists of (box, category) for `key` predictions."""
    tp, fp, fn = [], [], []
    for c in cats:
        t, f, n = classify(sample[key].get(c, []), sample["gt"].get(c, []))
        tp += [(b, c) for b in t]
        fp += [(b, c) for b in f]
        fn += [(b, c) for b in n]
    return tp, fp, fn


def _tag(name):
    for t, col in TAGS:
        if t in name:
            return t, col
    return "", SUB


def _card(img_rgb, title, subtitle, cell_w, img_h, pad=14):
    """Compose one image into a titled dark card (PIL)."""
    disp = Image.fromarray(img_rgb)
    disp = disp.resize((cell_w - 2*pad, img_h))
    bar = 38
    card = Image.new("RGB", (cell_w, img_h + bar + pad), PANEL)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([2, 2, cell_w-3, img_h+bar+pad-3], radius=10, outline=(48, 54, 61), width=1)
    card.paste(disp, (pad, bar))
    d.text((pad, 9), title, font=_font(16), fill=FG)
    if subtitle:
        tg, col = subtitle
        tw = d.textlength(tg, font=_font(13))
        d.rounded_rectangle([cell_w-pad-tw-14, 9, cell_w-pad, 28], radius=8, fill=col)
        d.text((cell_w-pad-tw-7, 10), tg, font=_font(13), fill=(13, 17, 23))
    return card


def legend_strip(width):
    h = 46
    strip = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(strip)
    items = [("Doğru tespit (TP)", C_TP), ("Yanlış tespit (FP)", C_FP),
             ("Kaçırılan GT (FN)", C_FN), ("Ground Truth", C_GT)]
    x = 24
    for label, col in items:
        d.rounded_rectangle([x, 16, x+26, 32], radius=5, fill=col)
        d.text((x+34, 15), label, font=_font(15, bold=False), fill=FG)
        x += 40 + int(d.textlength(label, font=_font(15, bold=False))) + 28
    return strip


def grid(cards, cols, title, subtitle):
    cw, ch = cards[0].size
    rows = (len(cards) + cols - 1) // cols
    gap = 14
    header = 78
    leg = legend_strip(cols*cw + (cols+1)*gap)
    W = cols*cw + (cols+1)*gap
    H = header + rows*ch + (rows+1)*gap + leg.height
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((24, 18), title, font=_font(30), fill=FG)
    d.text((24, 52), subtitle, font=_font(16, bold=False), fill=SUB)
    for i, card in enumerate(cards):
        r, c = divmod(i, cols)
        canvas.paste(card, (gap + c*(cw+gap), header + gap + r*(ch+gap)))
    canvas.paste(leg, (0, H - leg.height))
    return canvas


def f1_at(samples, cats, key, thr):
    out = {}
    for cat in cats + ["OVERALL"]:
        tp = fp = fn = 0
        for s in samples:
            clist = cats if cat == "OVERALL" else [cat]
            for c in clist:
                t, f, n = classify(s[key].get(c, []), s["gt"].get(c, []), thr)
                tp += len(t); fp += len(f); fn += len(n)
        P = tp/(tp+fp) if tp+fp else 0.0
        R = tp/(tp+fn) if tp+fn else 0.0
        out[cat] = 2*P*R/(P+R) if P+R else 0.0
    return out


def metrics_chart(samples, cats, out):
    import plotly.graph_objects as go
    labels = cats + ["OVERALL"]
    series = [
        ("Base · F1@0.5", f1_at(samples, cats, "base", 0.5), "#6e7681"),
        ("Fine-tuned · F1@0.5", f1_at(samples, cats, "tuned", 0.5), "#3fb950"),
        ("Base · F1@0.95", f1_at(samples, cats, "base", 0.95), "#30363d"),
        ("Fine-tuned · F1@0.95", f1_at(samples, cats, "tuned", 0.95), "#1f6feb"),
    ]
    fig = go.Figure()
    for name, vals, col in series:
        fig.add_bar(name=name, x=labels, y=[round(vals[l], 3) for l in labels],
                    marker_color=col, text=[f"{vals[l]:.2f}" for l in labels],
                    textposition="outside", textfont=dict(color="#e6edf3", size=12))
    fig.update_layout(
        title=dict(text="<b>LocateAnything-3B · Base vs Fine-tuned (val)</b>",
                   font=dict(size=24, color="#e6edf3")),
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        barmode="group", bargap=0.25, bargroupgap=0.08,
        yaxis=dict(title="F1", range=[0, 1.05], gridcolor="#21262d"),
        xaxis=dict(tickfont=dict(size=14)),
        legend=dict(orientation="h", y=-0.16, font=dict(color="#e6edf3")),
        font=dict(color="#e6edf3"), width=1100, height=620, margin=dict(t=80, b=90))
    fig.write_image(os.path.join(out, "metrics.png"), scale=2)
    fig.write_html(os.path.join(out, "metrics.html"))


def save_per_image(sample, cats, out_dir, max_side=1400):
    """Save one standalone annotated image: fine-tuned TP/FP/FN + caption bar."""
    img = np.array(Image.open(sample["image"]).convert("RGB"))
    tp, fp, fn = _labeled(sample, "tuned", cats)
    vis = annotate(img, [(fn, C_FN), (fp, C_FP), (tp, C_TP)])

    disp = Image.fromarray(vis)
    w, h = disp.size
    s = max_side / max(w, h)
    if s < 1:
        disp = disp.resize((int(w*s), int(h*s)))
    w, h = disp.size

    bar = 54
    canvas = Image.new("RGB", (w + 24, h + bar + 16), BG)
    d = ImageDraw.Draw(canvas)
    canvas.paste(disp, (12, bar))
    name = os.path.basename(sample["image"]).replace(".jpg", "")
    d.text((14, 10), name, font=_font(20), fill=FG)
    tg, col = _tag(name)
    if tg:
        d.rounded_rectangle([14, 34, 14+int(d.textlength(tg, font=_font(13)))+16, 50], radius=7, fill=col)
        d.text((22, 34), tg, font=_font(13), fill=BG)
    # counts, right-aligned
    counts = [(f"TP {len(tp)}", C_TP), (f"FP {len(fp)}", C_FP), (f"FN {len(fn)}", C_FN)]
    x = w + 24 - 14
    for txt, c in reversed(counts):
        tw = int(d.textlength(txt, font=_font(16)))
        x -= tw + 18
        d.text((x, 16), txt, font=_font(16), fill=c)
    canvas.save(os.path.join(out_dir, name + ".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="eval_preds.json")
    ap.add_argument("--out", default="eval_vis")
    ap.add_argument("--n-grid", type=int, default=9)
    ap.add_argument("--n-compare", type=int, default=4)
    ap.add_argument("--cell", type=int, default=430)
    ap.add_argument("--no-per-image", action="store_true", help="skip saving each image separately")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = json.load(open(args.preds))
    cats, samples = data["categories"], data["samples"]
    # richest images first (most GT objects) — most informative to look at
    samples.sort(key=lambda s: -sum(len(v) for v in s["gt"].values()))

    img_h = int(args.cell * 0.62)

    # ---- Panel 1: fine-tuned TP/FP/FN ----
    cards = []
    for s in samples[:args.n_grid]:
        img = np.array(Image.open(s["image"]).convert("RGB"))
        tp, fp, fn = _labeled(s, "tuned", cats)
        vis = annotate(img, [(fn, C_FN), (fp, C_FP), (tp, C_TP)])
        name = os.path.basename(s["image"]).replace(".jpg", "")
        cards.append(_card(vis, name.split("_")[0] + " · " + name.split("_")[-1],
                           _tag(name), args.cell, img_h))
    g1 = grid(cards, 3, "Fine-tuned · Tespit Sonuçları",
              "Her kutu IoU≥0.5 ile eşleştirildi · val seti en yoğun görüntüler")
    g1.save(os.path.join(args.out, "panel_finetuned.png"))
    print("saved panel_finetuned.png")

    # ---- Panel 2: base vs fine-tuned ----
    rows = []
    for s in samples[:args.n_compare]:
        img = np.array(Image.open(s["image"]).convert("RGB"))
        gt = [(b, c) for c in cats for b in s["gt"].get(c, [])]
        base = [(b, c) for c in cats for b in s["base"].get(c, [])]
        tuned = [(b, c) for c in cats for b in s["tuned"].get(c, [])]
        vb = annotate(img, [(gt, C_GT), (base, C_FP)])
        vt = annotate(img, [(gt, C_GT), (tuned, C_TP)])
        name = os.path.basename(s["image"]).replace(".jpg", "")
        rows.append(_card(vb, "BASE · " + name.split("_")[0], _tag(name), args.cell, img_h))
        rows.append(_card(vt, "FINE-TUNED · " + name.split("_")[0], _tag(name), args.cell, img_h))
    g2 = grid(rows, 2, "Base vs Fine-tuned",
              "Mavi = Ground Truth · Kırmızı = base tahmini · Yeşil = fine-tuned tahmini")
    g2.save(os.path.join(args.out, "base_vs_finetuned.png"))
    print("saved base_vs_finetuned.png")

    # ---- Per-image standalone results (each image saved separately) ----
    if not args.no_per_image:
        per_dir = os.path.join(args.out, "per_image")
        os.makedirs(per_dir, exist_ok=True)
        for i, s in enumerate(samples):
            save_per_image(s, cats, per_dir)
        print(f"saved {len(samples)} per-image results -> {per_dir}/")

    # ---- Panel 3: metrics ----
    metrics_chart(samples, cats, args.out)
    print("saved metrics.png / metrics.html")


if __name__ == "__main__":
    main()
