#!/usr/bin/env python3
"""Convert LabelMe polygon annotations into LocateAnything-3B detection SFT data.

Output schema follows the official NVlabs/Eagle Embodied data format
(document/DATA_PREPARATION.md): ShareGPT-style `conversations` + `image`:

    {"conversations": [{"from": "human", "value": "<prompt>"},
                       {"from": "gpt",   "value": "<target>"}],
     "image": "<path>"}

Conventions (verified against the released checkpoint, which is ground truth
for fine-tuning):
  - Coordinates are normalized integers in [0, 1000], order x1,y1,x2,y2 (xyxy).
  - Detection target groups all boxes of a category under one <ref>:
        <ref>CAT</ref><box><x1><y1><x2><y2></box><box>...</box>
  - A queried category with no instances emits <box>None</box>
    ("None" == none_token_id 4064; lowercase "none" is a different token).
  - Categories in the prompt are joined with </c>.

We also keep `objects` and `split` keys for evaluation/bookkeeping (extra keys
are ignored by trainers).
"""
import argparse
import glob
import json
import os
import random

# Default detection vocabulary, in the order they appear in the prompt/target.
# Override per-dataset with --categories or discover with --auto-categories.
CANON = ["class_0", "class_1", "class_2"]

DETECT_PROMPT = "Locate all the instances that matches the following description: {cats}."


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def norm_coord(v, size):
    q = round(v / size * 1000)
    return max(0, min(1000, q))


def build_response(objects, canon):
    parts = []
    for cat in canon:
        parts.append(f"<ref>{cat}</ref>")
        boxes = objects.get(cat, [])
        if not boxes:
            parts.append("<box>None</box>")
        else:
            for x1, y1, x2, y2 in boxes:
                parts.append(f"<box><{x1}><{y1}><{x2}><{y2}></box>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./dataset",
                    help="Directory with paired *.jpg / *.json files")
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-box", type=float, default=1.0,
                    help="Drop boxes smaller than this many pixels per side")
    ap.add_argument("--categories", default=None,
                    help="Comma-separated category list (prompt/target order). "
                         "Default: class_0,class_1,class_2")
    ap.add_argument("--auto-categories", action="store_true",
                    help="Discover categories automatically from all JSON labels")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    json_files = sorted(glob.glob(os.path.join(src, "*.json")))

    if args.auto_categories:
        found = {}
        for jf in json_files:
            try:
                d = json.load(open(jf))
            except Exception:
                continue
            for s in d.get("shapes", []):
                found[s.get("label")] = found.get(s.get("label"), 0) + 1
        canon = sorted(found, key=lambda k: -found[k])
        print(f"auto categories: {canon} (counts {found})")
    elif args.categories:
        canon = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        canon = CANON

    rows = []
    stat_objs = {c: 0 for c in canon}
    dropped_imgs = 0
    dropped_boxes = 0
    skipped_labels = {}

    for jf in json_files:
        with open(jf, "r") as f:
            d = json.load(f)
        img_name = d.get("imagePath") or os.path.basename(jf)[:-5] + ".jpg"
        img_path = os.path.join(src, os.path.basename(img_name))
        if not os.path.exists(img_path):
            dropped_imgs += 1
            continue
        W = d.get("imageWidth")
        H = d.get("imageHeight")
        if not W or not H:
            dropped_imgs += 1
            continue

        objects = {c: [] for c in canon}
        for s in d.get("shapes", []):
            label = s.get("label")
            if label not in canon:
                skipped_labels[label] = skipped_labels.get(label, 0) + 1
                continue
            if s.get("shape_type") != "polygon":
                continue
            pts = s.get("points", [])
            if len(pts) < 3:
                continue
            x1, y1, x2, y2 = polygon_to_bbox(pts)
            if (x2 - x1) < args.min_box or (y2 - y1) < args.min_box:
                dropped_boxes += 1
                continue
            box = [norm_coord(x1, W), norm_coord(y1, H),
                   norm_coord(x2, W), norm_coord(y2, H)]
            objects[label].append(box)

        n = sum(len(v) for v in objects.values())
        if n == 0:
            dropped_imgs += 1
            continue
        for c in canon:
            stat_objs[c] += len(objects[c])

        query = DETECT_PROMPT.format(cats="</c>".join(canon))
        response = build_response(objects, canon)
        rows.append({
            "conversations": [
                {"from": "human", "value": query},
                {"from": "gpt", "value": response},
            ],
            "image": img_path,
            "image_width": W,
            "image_height": H,
            "categories": canon,
            "objects": objects,
        })

    # Deterministic train/val split.
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    for i, r in enumerate(rows):
        r["split"] = "val" if i < n_val else "train"

    out_path = os.path.abspath(args.out)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_train = sum(1 for r in rows if r["split"] == "train")
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"  train={n_train}  val={n_val}")
    print(f"  objects per category: {stat_objs}")
    print(f"  dropped images (no canonical/missing): {dropped_imgs}")
    print(f"  dropped tiny boxes: {dropped_boxes}")
    print(f"  skipped non-canonical labels: {skipped_labels}")
    print("  sample conversations:", json.dumps(rows[0]["conversations"], ensure_ascii=False)[:220])


if __name__ == "__main__":
    main()
