#!/usr/bin/env python3
"""Gradio demo for LocateAnything-3B — base vs LoRA fine-tuned detection.

Upload an image, pick the base or fine-tuned model, and get back the detected
boxes drawn with supervision's rounded-box annotator + category labels.

    python app.py                # launches at http://127.0.0.1:7860
"""
import re
import sys
import threading

import cv2
import numpy as np
import supervision as sv
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import PeftModel

MODEL_PATH = "."
ADAPTER_PATH = "lora_out/final"
DEFAULT_PROMPT = ("Locate all the instances that matches the following description: "
                  "class_0</c>class_1</c>class_2.")

# vibrant category palette (RGB) — edit to match your own class names
CAT_COLORS = {
    "class_0": (63, 185, 80),       # green
    "class_1": (88, 166, 255),      # blue
    "class_2": (210, 153, 34),      # amber
}
DEFAULT_COLOR = (188, 140, 255)    # purple for unknown categories

BOX_RE = re.compile(r"<ref>(.*?)</ref>((?:<box>(?:None|<\d+><\d+><\d+><\d+>)</box>)*)")
ONE_BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

_STATE = {"model": None, "tok": None, "proc": None, "adapter": False}
_LOCK = threading.Lock()


def _install_vision_patch():
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


def _ensure_loaded(progress=None):
    if _STATE["model"] is not None:
        return
    if progress:
        progress(0.1, desc="Tokenizer / processor yükleniyor…")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if progress:
        progress(0.4, desc="Base model yükleniyor (~7GB)…")
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").to("cuda").eval()
    model.config.text_config._attn_implementation = "sdpa"
    _install_vision_patch()
    try:
        if progress:
            progress(0.8, desc="LoRA adaptörü ekleniyor…")
        model.language_model = PeftModel.from_pretrained(model.language_model, ADAPTER_PATH).eval()
        _STATE["adapter"] = True
    except Exception as e:
        print("adapter load failed:", e)
        _STATE["adapter"] = False
    _STATE.update(model=model, tok=tok, proc=proc)


def parse_pred(answer):
    pairs = []
    for m in BOX_RE.finditer(answer):
        cat = m.group(1)
        for b in ONE_BOX.findall(m.group(2)):
            pairs.append((cat, [int(x) for x in b]))
    return pairs


def annotate(pil, pairs):
    w, h = pil.size
    scene = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if pairs:
        cat_list = sorted({c for c, _ in pairs})
        xyxy = np.array([[b[0]/1000*w, b[1]/1000*h, b[2]/1000*w, b[3]/1000*h] for _, b in pairs])
        cls = np.array([cat_list.index(c) for c, _ in pairs])
        dets = sv.Detections(xyxy=xyxy, class_id=cls)
        palette = sv.ColorPalette([sv.Color(*CAT_COLORS.get(c, DEFAULT_COLOR)) for c in cat_list])
        thick = max(2, round(min(w, h) / 300))
        ts = max(0.5, min(w, h) / 1400)
        box = sv.BoxAnnotator(color=palette, thickness=thick,
                              color_lookup=sv.ColorLookup.CLASS)
        lab = sv.LabelAnnotator(color=palette, text_color=sv.Color(13, 17, 23),
                                text_scale=ts, text_thickness=max(1, round(ts*2)),
                                text_padding=int(6*ts)+2, text_position=sv.Position.TOP_LEFT,
                                color_lookup=sv.ColorLookup.CLASS)
        scene = box.annotate(scene, dets)
        scene = lab.annotate(scene, dets, labels=[c for c, _ in pairs])
    return Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def run(image, model_choice, prompt, max_side, gen_mode, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Lütfen bir görsel yükleyin.")
    with _LOCK:
        _ensure_loaded(progress)
        model, tok, proc = _STATE["model"], _STATE["tok"], _STATE["proc"]
        use_tuned = model_choice.startswith("Fine")
        if use_tuned and not _STATE["adapter"]:
            raise gr.Error(f"LoRA adaptörü bulunamadı: {ADAPTER_PATH}")

        progress(0.5, desc="Model çıkarım yapıyor…")
        img = image.convert("RGB")
        w, h = img.size
        s = max_side / max(w, h)
        inp = img.resize((int(w*s), int(h*s))) if s < 1 else img

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": inp}, {"type": "text", "text": prompt}]}]
        text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(msgs)
        inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")

        def _gen():
            return model.generate(
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"), tokenizer=tok,
                max_new_tokens=1024, use_cache=True, generation_mode=gen_mode,
                do_sample=False, verbose=False)

        if _STATE["adapter"] and not use_tuned:
            with model.language_model.disable_adapter():
                resp = _gen()
        else:
            resp = _gen()
        ans = resp[0] if isinstance(resp, tuple) else resp

    pairs = parse_pred(ans)
    annotated = annotate(img, pairs)
    # detection table + per-category summary
    rows = [[c, f"{b[0]},{b[1]},{b[2]},{b[3]}"] for c, b in pairs]
    counts = {}
    for c, _ in pairs:
        counts[c] = counts.get(c, 0) + 1
    summary = "  ·  ".join(f"**{c}**: {n}" for c, n in sorted(counts.items())) or "_(tespit yok)_"
    tag = "🟢 Fine-tuned (LoRA)" if use_tuned else "⚪ Base"
    summary = f"### {tag}\n{summary}\n\n`{len(pairs)} kutu`"
    return annotated, rows, summary, ans


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(body_background_fill="#0d1117", block_background_fill="#161b22",
      block_border_color="#30363d", body_text_color="#e6edf3")

with gr.Blocks(theme=THEME, title="LocateAnything-3B Demo") as demo:
    gr.Markdown("# 🛰️ LocateAnything-3B · Tespit Demosu\n"
                "Görsel yükle, modeli seç, kutuları gör. "
                "Base ile fine-tuned (LoRA) çıktılarını karşılaştır.")
    with gr.Row():
        with gr.Column(scale=4):
            inp_image = gr.Image(type="pil", label="Görsel", height=360)
            model_choice = gr.Radio(
                ["Fine-tuned (LoRA)", "Base"], value="Fine-tuned (LoRA)", label="Model")
            prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt (kategoriler </c> ile ayrılır)",
                                lines=2)
            with gr.Accordion("Gelişmiş ayarlar", open=False):
                max_side = gr.Slider(640, 2048, value=1280, step=64, label="Maks. kenar (px)")
                gen_mode = gr.Radio(["slow", "hybrid", "fast"], value="slow",
                                    label="Üretim modu")
            btn = gr.Button("🔍 Tespit Et", variant="primary")
        with gr.Column(scale=6):
            out_image = gr.Image(type="pil", label="Sonuç", height=460)
            out_summary = gr.Markdown()
            with gr.Accordion("Kutu koordinatları (0–1000, xyxy)", open=False):
                out_table = gr.Dataframe(headers=["kategori", "x1,y1,x2,y2"],
                                         datatype=["str", "str"], wrap=True)
            with gr.Accordion("Ham model çıktısı", open=False):
                out_raw = gr.Textbox(lines=4, show_copy_button=True)

    btn.click(run, [inp_image, model_choice, prompt, max_side, gen_mode],
              [out_image, out_table, out_summary, out_raw])

    import glob, os
    examples = sorted(glob.glob("examples/*.jpg"))[:6]
    if examples:
        gr.Examples(examples=[[e] for e in examples], inputs=[inp_image], label="Örnek görseller")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
