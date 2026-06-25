#!/usr/bin/env python3
"""Compare base vs LoRA-fine-tuned LocateAnything-3B on the val split.

Loads the base model once, attaches the LoRA adapter, and evaluates both the
base (adapter disabled) and the fine-tuned (adapter enabled) model on the same
val images. Reports precision / recall / F1 at IoU 0.5 / 0.95 per category.

The fine-tuned model can be evaluated under several decoding modes at once via
`--gen-modes slow,hybrid,fast` to compare LocateAnything's Parallel Box Decoding
(`fast` = MTP only, `hybrid` = MTP + AR fallback) against the pure autoregressive
path (`slow`, which is the one the LoRA SFT actually trains). A per-mode speed /
throughput table is printed so you can see the quality-vs-latency tradeoff.
"""
import argparse
import json
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import PeftModel

BOX_RE = re.compile(r"<ref>(.*?)</ref>((?:<box>(?:None|<\d+><\d+><\d+><\d+>)</box>)*)")
ONE_BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_pred(answer):
    """ans -> {category: [[x1,y1,x2,y2] in 0..1000]}"""
    out = {}
    for m in BOX_RE.finditer(answer):
        cat = m.group(1)
        boxes = [[int(x) for x in b] for b in ONE_BOX.findall(m.group(2))]
        out.setdefault(cat, []).extend(boxes)
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    """Greedy IoU match. Returns (tp, fp, fn, matched_ious)."""
    used = [False] * len(gts)
    tp = 0
    matched_ious = []
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
            tp += 1
            matched_ious.append(iou(p, gts[bi]))
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, matched_ious


@torch.no_grad()
def predict(model, tok, proc, img, query, max_side, device, gen_mode="slow"):
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((int(w*s), int(h*s)))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": query}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(msgs)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to(device)
    resp = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16), input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
        tokenizer=tok, max_new_tokens=512, use_cache=True, generation_mode=gen_mode,
        do_sample=False, verbose=False)
    return resp[0] if isinstance(resp, tuple) else resp


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


def report(samples, cats, key, thr):
    """Print P/R/F1 at IoU `thr` plus mean-IoU for `key` in each sample."""
    print(f"\n{'category':<14} {'P':>7} {'R':>7} {'F1':>7}   (tp/fp/fn)")
    print("-" * 52)
    all_ious = []
    for cat in cats + ["__ALL__"]:
        tp = fp = fn = 0
        for s in samples:
            clist = cats if cat == "__ALL__" else [cat]
            for c in clist:
                t, f, n, ious = match(s[key].get(c, []), s["gt"].get(c, []), thr)
                tp += t; fp += f; fn += n
                if cat == "__ALL__":
                    all_ious += ious
        P = tp/(tp+fp) if tp+fp else 0.0
        R = tp/(tp+fn) if tp+fn else 0.0
        F1 = 2*P*R/(P+R) if P+R else 0.0
        name = "OVERALL" if cat == "__ALL__" else cat
        print(f"{name:<14} {P:>7.3f} {R:>7.3f} {F1:>7.3f}   ({tp}/{fp}/{fn})")
    miou = sum(all_ious)/len(all_ious) if all_ious else 0.0
    print(f"  mean IoU (matched @0.5): {miou:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".")
    ap.add_argument("--adapter", default="lora_out/final")
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="limit number of val images (0=all)")
    ap.add_argument("--dump", default="eval_preds.json", help="save predictions+GT for visualization")
    ap.add_argument("--gen-modes", default="slow",
                    help="comma-separated decoding modes to evaluate the fine-tuned model in: "
                         "any of slow,hybrid,fast (e.g. --gen-modes slow,hybrid,fast)")
    ap.add_argument("--base-mode", default="slow", choices=["slow", "hybrid", "fast"],
                    help="decoding mode for the base reference (default: slow)")
    args = ap.parse_args()
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
    model.config.text_config._attn_implementation = "sdpa"
    install_vision_patch()

    model.language_model = PeftModel.from_pretrained(model.language_model, args.adapter).eval()
    print(f"adapter loaded: {args.adapter}")

    rows = [json.loads(l) for l in open(args.data)]
    val = [r for r in rows if r.get("split") == "val"]
    if args.limit:
        val = val[:args.limit]
    cats = val[0]["categories"]
    query = val[0]["conversations"][0]["value"]

    modes = [m.strip() for m in args.gen_modes.split(",") if m.strip()]
    for m in modes:
        assert m in ("slow", "hybrid", "fast"), f"bad --gen-modes entry: {m!r}"
    print(f"fine-tuned decoding modes: {modes}   base reference: {args.base_mode}")

    # wall-clock + predicted-box counts per fine-tuned mode (PBD speed payoff)
    timing = {m: 0.0 for m in modes}
    boxcount = {m: 0 for m in modes}

    samples = []
    for i, r in enumerate(val):
        img = Image.open(r["image"]).convert("RGB")
        with model.language_model.disable_adapter():
            base_ans = predict(model, tok, proc, img, query, args.max_side, device, args.base_mode)
        rec = {
            "image": r["image"],
            "image_width": r["image_width"],
            "image_height": r["image_height"],
            "gt": r["objects"],
            "base": parse_pred(base_ans),
        }
        for m in modes:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ans = predict(model, tok, proc, img, query, args.max_side, device, m)
            torch.cuda.synchronize()
            timing[m] += time.perf_counter() - t0
            pred = parse_pred(ans)
            rec[f"tuned__{m}"] = pred
            boxcount[m] += sum(len(v) for v in pred.values())
        # primary mode kept under "tuned" for visualize_eval.py compatibility
        rec["tuned"] = rec[f"tuned__{modes[0]}"]
        samples.append(rec)
        print(f"[{i+1}/{len(val)}] {r['image'].split('/')[-1]}", flush=True)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({"categories": cats, "query": query, "modes": modes,
                       "samples": samples}, f)
        print(f"\nsaved predictions -> {args.dump}")

    for thr in (0.5, 0.95):
        print(f"\n############### IoU @ {thr} ###############")
        print(f"================ BASE · {args.base_mode} (no fine-tune) ================")
        report(samples, cats, "base", thr)
        for m in modes:
            print(f"\n========== FINE-TUNED (LoRA) · gen-mode={m} ==========")
            report(samples, cats, f"tuned__{m}", thr)

    # speed / throughput summary — the Parallel Box Decoding payoff
    n = len(val)
    print("\n############### SPEED (fine-tuned) ###############")
    print(f"{'mode':<10} {'img/s':>8} {'sec/img':>9} {'boxes/s':>9}")
    print("-" * 40)
    for m in modes:
        ips = n / timing[m] if timing[m] else 0.0
        spi = timing[m] / n if n else 0.0
        bps = boxcount[m] / timing[m] if timing[m] else 0.0
        print(f"{m:<10} {ips:>8.2f} {spi:>9.3f} {bps:>9.2f}")


if __name__ == "__main__":
    main()
