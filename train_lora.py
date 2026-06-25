#!/usr/bin/env python3
"""LoRA SFT for LocateAnything-3B on a custom detection dataset (single 24GB GPU).

Design notes (derived from the released model code):

* The model's `forward(..., labels=...)` path is a plain causal-LM cross-entropy
  loss. The MTP / Parallel-Box-Decoding machinery lives only in `generate()`.
  So we fine-tune the autoregressive ("slow") path with standard SFT.

* In `training=True` + sdpa mode, `Qwen2Model.forward` builds a *block-diffusion*
  attention mask (for MTP training), which is wrong for plain AR SFT. We
  monkeypatch `create_block_diff_mask_by_pe_4d` to return a standard causal mask
  so training is ordinary next-token prediction while keeping gradient
  checkpointing (which requires `training=True`) available.

* Only LoRA adapters on the Qwen2 LLM are trained; the MoonViT vision tower and
  the MLP connector are frozen.

* Coordinates use the [0,1000] xyxy format the pretrained model emits. Targets
  come straight from dataset.jsonl `response` strings.
"""
import argparse
import json
import math
import os
import random
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import get_cosine_schedule_with_warmup

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    _RICH = True
except Exception:
    _RICH = False


# --------------------------------------------------------------------------- #
# Causal-mask monkeypatch
# --------------------------------------------------------------------------- #
def _install_causal_mask_patch():
    """Force the training-time attention mask to be plain causal."""
    target = None
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_qwen2") and hasattr(mod, "create_block_diff_mask_by_pe_4d"):
            target = mod
            break
    if target is None:
        raise RuntimeError("Could not find modeling_qwen2 module to patch")

    def causal_block_mask(block_size, x0_len_list, position_ids, causal_attn=False):
        # position_ids: [B, S] -> additive causal mask [B, 1, S, S] in bf16
        bsz, seq_len = position_ids.shape
        device = position_ids.device
        i = torch.arange(seq_len, device=device)
        allowed = i.view(seq_len, 1) >= i.view(1, seq_len)  # lower triangular
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16, device=device)
        mask.masked_fill_(~allowed, float("-inf"))
        mask = mask.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len)
        return mask, allowed.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len)

    target.create_block_diff_mask_by_pe_4d = causal_block_mask
    print(f"[patch] causal mask installed in {target.__name__}")


def _install_vision_attn_patch():
    """Make the MoonViT SDPA use the memory-efficient kernel.

    The released `sdpa_attention` always materializes a dense [1, N, N] boolean
    mask, which forces SDPA onto the math kernel and blows up memory at high
    resolution. For a single image (one attention segment) the mask is all-True,
    i.e. full attention, so we can pass `attn_mask=None` and let PyTorch pick the
    flash / mem-efficient kernel. Multi-segment packs keep the dense mask.
    """
    import torch.nn.functional as F
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_vit") and hasattr(mod, "sdpa_attention"):
            def sdpa_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
                seq_length = q.shape[0]
                single = q_cu_seqlens is None or len(q_cu_seqlens) == 2
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
                if single:
                    attn = F.scaled_dot_product_attention(q, k, v, None, dropout_p=0.0)
                else:
                    m = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
                    for i in range(1, len(q_cu_seqlens)):
                        m[..., q_cu_seqlens[i - 1]:q_cu_seqlens[i],
                          q_cu_seqlens[i - 1]:q_cu_seqlens[i]] = True
                    attn = F.scaled_dot_product_attention(q, k, v, m, dropout_p=0.0)
                return attn.transpose(0, 1).reshape(seq_length, -1)
            mod.sdpa_attention = sdpa_attention
            print(f"[patch] efficient vision attention installed in {mod.__name__}")
            return
    print("[patch] WARNING: modeling_vit.sdpa_attention not found")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _conv_to_qr(row):
    """Return (human_prompt, gpt_target) from a ShareGPT `conversations` row.

    Falls back to legacy query/response keys if present. The `<image>` token, if
    embedded in the human value, is stripped (the processor injects the image).
    """
    if "conversations" in row:
        human = next(c["value"] for c in row["conversations"] if c["from"] == "human")
        gpt = next(c["value"] for c in row["conversations"] if c["from"] == "gpt")
        human = human.replace("<image>\n", "").replace("<image>", "").strip()
        return human, gpt
    return row["query"], row["response"]


class DetDataset(Dataset):
    def __init__(self, rows, processor, tokenizer, max_side=1280):
        self.rows = rows
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_side = max_side

    def __len__(self):
        return len(self.rows)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = self.max_side / max(w, h)
        if s < 1.0:
            img = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                             Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = self._load_image(row["image"])

        # ShareGPT-style conversations: first human turn = prompt, gpt = target.
        query, response = _conv_to_qr(row)
        user_msg = {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": query},
        ]}
        prompt_text = self.processor.py_apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + response + "<|im_end|>\n"

        images, videos = self.processor.process_vision_info([user_msg])
        full = self.processor(text=[full_text], images=images, videos=videos,
                              return_tensors="pt")
        prompt = self.processor(text=[prompt_text], images=images, videos=videos,
                                return_tensors="pt")

        input_ids = full["input_ids"][0]
        prompt_len = prompt["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "pixel_values": full["pixel_values"],
            "image_grid_hws": full.get("image_grid_hws"),
        }


def collate(batch):
    # micro-batch size 1 keeps us free of padding-mask complications.
    assert len(batch) == 1
    b = batch[0]
    grid = b["image_grid_hws"]
    if grid is not None and not torch.is_tensor(grid):
        grid = torch.as_tensor(grid, dtype=torch.int32)
    elif torch.is_tensor(grid):
        grid = grid.to(torch.int32)
    return {
        "input_ids": b["input_ids"].unsqueeze(0),
        "attention_mask": b["attention_mask"].unsqueeze(0),
        "labels": b["labels"].unsqueeze(0),
        "pixel_values": b["pixel_values"],
        "image_grid_hws": grid,
        "image_flags": torch.tensor([1], dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Rich terminal UI
# --------------------------------------------------------------------------- #
def _make_progress(console):
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[green]loss {task.fields[loss]}"),
        TextColumn("[yellow]lr {task.fields[lr]}"),
        TextColumn("[magenta]{task.fields[mem]}GB"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )


def _config_panel(console, args, n_train, n_params, total_steps):
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="cyan", justify="right")
    t.add_column(style="white")
    t.add_row("model", str(args.model))
    t.add_row("train samples", str(n_train))
    t.add_row("epochs", str(args.epochs))
    t.add_row("optimizer steps", str(total_steps))
    t.add_row("batch (micro × accum)", f"1 × {args.grad_accum}")
    t.add_row("lr / warmup", f"{args.lr:.1e} / {args.warmup_frac:.0%}")
    t.add_row("LoRA r / alpha", f"{args.lora_r} / {args.lora_alpha}")
    t.add_row("image max side", f"{args.max_side}px")
    t.add_row("trainable params", f"{n_params/1e6:.1f}M")
    t.add_row("output dir", str(args.out))
    console.print(Panel(t, title="[bold]LocateAnything-3B · LoRA SFT", border_style="blue"))


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".")
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke test)")
    ap.add_argument("--save-every", type=int, default=0, help="save every N optimizer steps (0=only end)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rich", action="store_true", help="Plain text logs instead of rich UI")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("loading model...", flush=True)
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.config.text_config._attn_implementation = "sdpa"

    _install_causal_mask_patch()
    _install_vision_attn_patch()

    # LoRA on the LLM; freeze vision tower + connector.
    model.wrap_llm_lora(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05)
    for p in model.vision_model.parameters():
        p.requires_grad_(False)
    for p in model.mlp1.parameters():
        p.requires_grad_(False)

    # wrap_llm_lora() installs an input-require-grad hook on the embeddings, which
    # collides with the model's in-place image-embedding injection. We use
    # non-reentrant gradient checkpointing (which does not need that hook), so
    # remove it.
    try:
        model.language_model.disable_input_require_grads()
        print("[patch] disabled input_require_grads hook")
    except Exception as e:
        print("[patch] could not disable input_require_grads:", e)

    # Gradient checkpointing on the LLM to fit 24GB.
    try:
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[mem] gradient checkpointing enabled")
    except Exception as e:
        print("[mem] grad checkpointing not enabled:", e)

    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = [json.loads(l) for l in open(args.data)]
    train_rows = [r for r in rows if r.get("split", "train") == "train"]

    ds = DetDataset(train_rows, processor, tokenizer, max_side=args.max_side)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate, num_workers=4)

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_frac), total_steps)

    use_rich = _RICH and not args.no_rich
    console = Console() if use_rich else None
    if use_rich:
        _config_panel(console, args, len(train_rows), n_params, total_steps)
        progress = _make_progress(console)
        task = progress.add_task("epoch 0", total=total_steps, loss="-", lr="-", mem="-")
        progress.start()
    else:
        print(f"trainable params: {n_params/1e6:.2f}M", flush=True)
        print(f"train samples: {len(train_rows)}  total steps: {total_steps}", flush=True)

    def _log(msg):
        if use_rich:
            progress.console.log(msg)
        else:
            print(msg, flush=True)

    os.makedirs(args.out, exist_ok=True)
    opt_step = 0
    micro = 0
    running = 0.0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            grid = batch["image_grid_hws"]
            grid = grid.to(device) if grid is not None else None
            pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)

            # Vision features -> project to LLM space (mlp1), then let the LLM's
            # image_processing splice them onto the <image> token positions.
            # We call the LLM directly (not the outer forward) because the model's
            # training-mode loss path lives in Qwen2ForCausalLM and needs input_ids
            # + labels together. The vision tower + connector are frozen, so run
            # them under no_grad to avoid retaining their (large) activations.
            with torch.no_grad():
                vit_embeds = model.extract_feature(pixel_values, grid)
                vit_embeds = torch.cat(vit_embeds, dim=0)
                vit_embeds = model.mlp1(vit_embeds)

            out = model.language_model(
                input_ids=input_ids,
                visual_features=vit_embeds,
                image_token_index=model.image_token_index,
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                use_cache=False,
            )
            causal_out = out[0] if isinstance(out, tuple) else out
            loss = causal_out.loss / args.grad_accum
            loss.backward()
            running += causal_out.loss.item()
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1
                avg = running / args.grad_accum
                running = 0.0
                mem = torch.cuda.max_memory_allocated() / 1e9
                lr_now = sched.get_last_lr()[0]
                if use_rich:
                    progress.update(
                        task, advance=1, description=f"epoch {epoch}",
                        loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", mem=f"{mem:.1f}")
                else:
                    print(f"epoch {epoch} step {opt_step}/{total_steps} "
                          f"loss {avg:.4f} lr {lr_now:.2e} peakmem {mem:.1f}GB", flush=True)
                if args.save_every and opt_step % args.save_every == 0:
                    _save(model, args.out, f"step{opt_step}", _log)
                if total_steps and opt_step >= total_steps:
                    done = True
                    break

    if use_rich:
        progress.stop()
        console.print("[bold green]✓ training complete")
    _save(model, args.out, "final", _log)


def _save(model, out, tag, log=print):
    path = os.path.join(out, tag)
    os.makedirs(path, exist_ok=True)
    # Save only the LoRA adapter (PEFT-wrapped language model).
    model.language_model.save_pretrained(path)
    log(f"[save] adapter -> {path}")


if __name__ == "__main__":
    main()
