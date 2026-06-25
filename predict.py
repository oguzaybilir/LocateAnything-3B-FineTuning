#!/usr/bin/env python3
"""Run LocateAnything-3B detection on a single image or a folder.

Each predicted box is drawn as a rectangle in its CLASS color, with the class
name (and optionally a confidence score) at the box's top-left corner.

Examples:
    # one image, fine-tuned model
    python predict.py --input photo.jpg --categories cat,dog

    # a whole folder, base model, with confidence scores in labels
    python predict.py --input ./images --base --scores

Confidence note: the model decodes boxes as tokens and exposes no calibrated
objectness score. `--scores` reports a proxy: the model's own mean probability
over each box's coordinate tokens (teacher-forced), in [0, 1].
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import PeftModel

# Optional fixed colors for known class names. Any class not listed here cycles
# through EXTRA. Set your own class list at runtime with --categories / --prompt.
CLASS_COLORS = {
    "class_0": (63, 185, 80),       # green
    "class_1": (88, 166, 255),      # blue
    "class_2": (240, 160, 40),      # orange
}
EXTRA = [(188, 140, 255), (255, 110, 180), (45, 212, 191),
         (250, 100, 90), (255, 209, 102), (120, 200, 255)]

BOX_RE = re.compile(r"<ref>(.*?)</ref>((?:<box>(?:None|<\d+><\d+><\d+><\d+>)</box>)*)")
ONE_BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def install_vision_patch():
    import torch.nn.functional as F
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_vit") and hasattr(mod, "sdpa_attention"):
            def sdpa_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
                seq_length = q.shape[0]
                single = q_cu_seqlens is None or len(q_cu_seqlens) == 2
                q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
                if single:
                    a = F.scaled_dot_product_attention(q, k, v, None, dropout_p=0.0)
                else:
                    m = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
                    for i in range(1, len(q_cu_seqlens)):
                        m[..., q_cu_seqlens[i-1]:q_cu_seqlens[i], q_cu_seqlens[i-1]:q_cu_seqlens[i]] = True
                    a = F.scaled_dot_product_attention(q, k, v, m, dropout_p=0.0)
                return a.transpose(0, 1).reshape(seq_length, -1)
            mod.sdpa_attention = sdpa_attention
            return


def parse_pred(answer):
    pairs = []
    for m in BOX_RE.finditer(answer):
        cat = m.group(1)
        for b in ONE_BOX.findall(m.group(2)):
            pairs.append((cat, [int(x) for x in b]))
    return pairs


def color_map(classes):
    cmap, ei = {}, 0
    for c in classes:
        if c in CLASS_COLORS:
            cmap[c] = CLASS_COLORS[c]
        else:
            cmap[c] = EXTRA[ei % len(EXTRA)]
            ei += 1
    return cmap


def _resize(img, max_side):
    w, h = img.size
    s = max_side / max(w, h)
    return img.resize((int(w*s), int(h*s))) if s < 1 else img


@torch.no_grad()
def generate(model, tok, proc, img_inp, prompt, gen_mode, device):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img_inp}, {"type": "text", "text": prompt}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(msgs)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    resp = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
        tokenizer=tok, max_new_tokens=1024, use_cache=True, generation_mode=gen_mode,
        do_sample=False, verbose=False)
    return resp[0] if isinstance(resp, tuple) else resp


@torch.no_grad()
def box_confidences(model, tok, proc, img_inp, prompt, answer, device):
    """Teacher-forced mean probability over each box's coordinate tokens."""
    cfg = model.config
    box_start, box_end = cfg.box_start_token_id, cfg.box_end_token_id
    cs, ce = cfg.coord_start_token_id, cfg.coord_end_token_id

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img_inp}, {"type": "text", "text": prompt}]}]
    ptext = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_text = ptext + answer + "<|im_end|>\n"
    images, videos = proc.process_vision_info(msgs)
    full = proc(text=[full_text], images=images, videos=videos, return_tensors="pt").to(device)
    plen = proc(text=[ptext], images=images, videos=videos, return_tensors="pt")["input_ids"].shape[1]

    grid = full.get("image_grid_hws")
    if grid is not None and not torch.is_tensor(grid):
        grid = torch.as_tensor(grid)
    grid = grid.to(device).to(torch.int32) if grid is not None else None

    # Call the LLM directly with input_ids + projected vision features (the outer
    # forward feeds inputs_embeds, which the eval mask path can't handle).
    vit = model.extract_feature(full["pixel_values"].to(torch.bfloat16), grid)
    vit = model.mlp1(torch.cat(vit, dim=0))
    out = model.language_model(
        input_ids=full["input_ids"], visual_features=vit,
        image_token_index=model.image_token_index,
        attention_mask=full["attention_mask"], use_cache=False)
    out = out[0] if isinstance(out, tuple) else out
    logits = out.logits[0].float()
    ids = full["input_ids"][0]

    confs, cur, in_box = [], [], False
    for t in range(plen, ids.shape[0]):
        tid = ids[t].item()
        if tid == box_start:
            in_box, cur = True, []
        elif tid == box_end:
            if cur:
                confs.append(float(np.mean(cur)))
            in_box = False
        elif in_box and cs <= tid <= ce:
            p = torch.softmax(logits[t-1], dim=-1)[tid].item()
            cur.append(p)
    return confs


def annotate(pil, pairs, cmap, scores=None):
    w, h = pil.size
    scene = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if not pairs:
        return Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    classes = list(cmap.keys())
    xyxy = np.array([[b[0]/1000*w, b[1]/1000*h, b[2]/1000*w, b[3]/1000*h] for _, b in pairs])
    cls = np.array([classes.index(c) for c, _ in pairs])
    dets = sv.Detections(xyxy=xyxy, class_id=cls)
    palette = sv.ColorPalette([sv.Color(*cmap[c]) for c in classes])
    thick = max(2, round(min(w, h) / 300))
    ts = max(0.5, min(w, h) / 1400)
    box = sv.BoxAnnotator(color=palette, thickness=thick, color_lookup=sv.ColorLookup.CLASS)
    lab = sv.LabelAnnotator(color=palette, text_color=sv.Color(13, 17, 23), text_scale=ts,
                            text_thickness=max(1, round(ts*2)), text_padding=int(6*ts)+2,
                            text_position=sv.Position.TOP_LEFT, color_lookup=sv.ColorLookup.CLASS)
    labels = []
    for i, (c, _) in enumerate(pairs):
        labels.append(f"{c} {scores[i]:.2f}" if scores is not None and i < len(scores) else c)
    scene = box.annotate(scene, dets)
    scene = lab.annotate(scene, dets, labels=labels)
    return Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))


def to_yolo_lines(pairs, classes, scores=None):
    """YOLO labels: `class_id xc yc w h` (normalized 0..1), optional 6th conf col."""
    lines = []
    for i, (cat, b) in enumerate(pairs):
        if cat not in classes:
            continue
        cid = classes.index(cat)
        xc = (b[0] + b[2]) / 2 / 1000
        yc = (b[1] + b[3]) / 2 / 1000
        w = (b[2] - b[0]) / 1000
        h = (b[3] - b[1]) / 1000
        row = f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        if scores is not None and i < len(scores):
            row += f" {scores[i]:.4f}"
        lines.append(row)
    return lines


def write_dataset_meta(out_dir, classes):
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")


def list_images(path):
    if os.path.isdir(path):
        return sorted(f for f in glob.glob(os.path.join(path, "*"))
                      if f.lower().endswith(IMG_EXT))
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Image file or folder")
    ap.add_argument("--out", default="predictions", help="Output folder")
    ap.add_argument("--model", default=".")
    ap.add_argument("--adapter", default="lora_out/final")
    ap.add_argument("--base", action="store_true", help="Use base model (no LoRA adapter)")
    ap.add_argument("--prompt", default="Locate all the instances that matches the following "
                    "description: class_0</c>class_1</c>class_2.")
    ap.add_argument("--categories", default=None,
                    help="Comma-separated class list for the color map (default: from prompt)")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--gen-mode", default="slow", choices=["slow", "hybrid", "fast"])
    ap.add_argument("--scores", action="store_true", default=False,
                    help="Write confidence proxy in labels (default: off)")
    ap.add_argument("--save-json", action="store_true", help="Also dump predictions.json")
    ap.add_argument("--no-yolo", action="store_true",
                    help="Do not write YOLO .txt labels (default: write them)")
    ap.add_argument("--yolo-conf", action="store_true",
                    help="Append confidence as a 6th column in YOLO labels (implies --scores)")
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    write_yolo = not args.no_yolo
    if args.yolo_conf:
        args.scores = True
    label_dir = os.path.join(args.out, "labels")
    if write_yolo:
        os.makedirs(label_dir, exist_ok=True)

    if args.categories:
        classes = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        m = re.search(r"description:\s*(.*?)\.\s*$", args.prompt)
        classes = [c.strip() for c in m.group(1).split("</c>")] if m else list(CLASS_COLORS)
    cmap = color_map(classes)
    print(f"classes/colors: {cmap}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
    model.config.text_config._attn_implementation = "sdpa"
    install_vision_patch()
    if not args.base:
        model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter).eval()
        print(f"model: FINE-TUNED ({args.adapter})")
    else:
        print("model: BASE")

    images = list_images(args.input)
    print(f"{len(images)} image(s) -> {args.out}/\n")
    dump = []
    for i, path in enumerate(images):
        img = Image.open(path).convert("RGB")
        inp = _resize(img, args.max_side)
        ans = generate(model, tok, proc, inp, args.prompt, args.gen_mode, device)
        pairs = parse_pred(ans)
        scores = None
        if args.scores:
            scores = box_confidences(model, tok, proc, inp, args.prompt, ans, device)
        vis = annotate(img, pairs, cmap, scores)
        name = os.path.splitext(os.path.basename(path))[0]
        vis.save(os.path.join(args.out, name + "_pred.png"))
        if write_yolo:
            yolo_scores = scores if args.yolo_conf else None
            lines = to_yolo_lines(pairs, classes, yolo_scores)
            with open(os.path.join(label_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        counts = {}
        for c, _ in pairs:
            counts[c] = counts.get(c, 0) + 1
        cstr = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no detection"
        print(f"[{i+1}/{len(images)}] {os.path.basename(path)} -> {cstr}")
        dump.append({"image": path, "detections": [
            {"category": c, "box_xyxy_0_1000": b,
             "score": round(scores[j], 4) if scores and j < len(scores) else None}
            for j, (c, b) in enumerate(pairs)]})

    if write_yolo:
        write_dataset_meta(args.out, classes)
        print(f"\nYOLO labels -> {label_dir}/   (classes.txt, data.yaml in {args.out}/)")
    if args.save_json:
        with open(os.path.join(args.out, "predictions.json"), "w") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)
        print(f"saved {args.out}/predictions.json")
    print(f"\ndone. annotated images in {args.out}/")


if __name__ == "__main__":
    main()
