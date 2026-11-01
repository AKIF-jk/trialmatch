#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct on the training split and select by validation macro-F1.

    python scripts/train.py --name r16 [--model target|draft] [--mode lora|last2] [--r 16] [--order evidence_first] [--oversample 1] [-- --name r8 --r 8]

Arguments after a bare `--` start another run in the same process. Each run writes
results/train/<name>/{config.json,train_log.jsonl,report.json,best/}. The best epoch by strict
validation macro-F1 is kept. The test split cannot be loaded here.

Precision: the base is fp16, trainable weights are float32, forward runs under fp16 autocast with a GradScaler.
Loss is the mean over assistant tokens of each micro-batch, averaged over the micro-batches of a step.
"""
import argparse
import contextlib
import gc
import json
import os
import sys
import time

# Must be set before CUDA initialises. Float32 logits for a validation batch are several GB, so avoid fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trialmatch.data.load import load_rows
from trialmatch.data.prompt import row_prompt
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.eval.generate import generate
from trialmatch.eval.metrics import evaluate
from trialmatch.runtime import DRAFT_MODEL, DRAFT_REVISION, TARGET_MODEL, TARGET_REVISION, check_versions, environment, results_dir, write_json
from trialmatch.train.data import batches, collate, cosine_factor, encode_rows, epoch_order, oversample

STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MODELS = {"target": (TARGET_MODEL, TARGET_REVISION), "draft": (DRAFT_MODEL, DRAFT_REVISION)}


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="target", choices=tuple(MODELS))
    p.add_argument("--mode", default="lora", choices=("lora", "last2"))
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--order", default="evidence_first", choices=("evidence_first", "label_first"))
    p.add_argument("--oversample", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--val-loss-batch", type=int, default=4)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--max-steps", type=int)
    return p.parse_args(argv)


def build_model(args, device):
    name, revision = MODELS[args.model]
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float16).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.mode == "lora":
        # Kaggle ships torchao 0.10.0 and peft 0.20.0 refuses it at adapter injection. We use no torchao
        # quantization, so make it look uninstalled to peft's find_spec check.
        sys.modules.setdefault("torchao", None)
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    else:
        for p in model.parameters():
            p.requires_grad = False
        for layer in model.model.layers[-2:]:
            layer.float()
            for p in layer.parameters():
                p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    wrong = {str(p.dtype) for p in trainable} - {"torch.float32"}
    if wrong:
        raise RuntimeError(f"trainable weights must be float32 for GradScaler, found {wrong}")
    return model, trainable


def step_loss(model, tokenizer_pad, examples, device):
    batch = collate(examples, tokenizer_pad)
    batch = {k: torch.tensor(v, device=device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        return model(**batch).loss


def validation_loss(model, pad_id, examples, size, device):
    model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for chunk in batches(list(range(len(examples))), size):
            part = [examples[i] for i in chunk]
            n = sum(sum(label != -100 for label in e["labels"][1:]) for e in part)
            total += float(step_loss(model, pad_id, part, device)) * n
            tokens += n
    return total / tokens


def validate(model, tokenizer, rows, prompts, stop_ids, args):
    model.eval()
    model.config.use_cache = True
    # last2 keeps two layers in float32 next to a float16 lm_head, which only works under autocast.
    mixed = torch.autocast("cuda", dtype=torch.float16) if args.mode == "last2" else contextlib.nullcontext()
    with mixed:
        outputs = generate(model, tokenizer, prompts, args.eval_batch_size, args.max_new_tokens, stop_ids)
    model.config.use_cache = False
    texts = [o["text"] for o in outputs]
    return outputs, evaluate(rows, texts), evaluate(rows, texts, repair=True)


def save_best(model, args, out):
    path = out / "best"
    path.mkdir(exist_ok=True)
    if args.mode == "lora":
        model.save_pretrained(path)
    else:
        state = {k: v for k, v in model.state_dict().items() if any(f"layers.{i}." in k for i in (26, 27))}
        torch.save(state, path / "last2_layers.pt")


def run(args):
    if args.model == "draft" and args.mode != "lora":
        raise ValueError("the draft model is trained with LoRA only (save_best assumes the target's 28 layers for last2)")
    out = results_dir(f"train/{args.name}")
    write_json(out / "config.json", vars(args))
    device = "cuda"
    torch.manual_seed(args.seed)

    manifest, rows = load_manifest(), load_rows()
    train_rows = select_rows(rows, manifest, "train")[:args.limit_train]
    val_rows = select_rows(rows, manifest, "val")[:args.limit_val]
    train_rows = oversample(train_rows, args.oversample)

    model_name, model_revision = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, padding_side="left")
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]

    def encode(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    train = encode_rows(train_rows, encode, args.order)
    val = encode_rows(val_rows, encode, args.order)
    if any(e["input_ids"][-1] != stop_ids[0] for e in train + val):
        raise AssertionError("every example must end with <|im_end|>")
    prompts = [row_prompt(r, args.order) for r in val_rows]

    model, trainable = build_model(args, device)
    n_trainable, n_total = sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters())
    print(f"{args.name}: trainable {n_trainable:,} of {n_total:,}, train rows {len(train)}, val rows {len(val)}", flush=True)

    accumulation = args.batch_size // args.micro_batch
    if accumulation * args.micro_batch != args.batch_size:
        raise ValueError("batch-size must be a multiple of micro-batch")
    steps_per_epoch = -(-len(train) // args.batch_size)
    total_steps = min(steps_per_epoch * args.epochs, args.max_steps or 10**9)
    warmup = max(1, round(args.warmup_ratio * total_steps))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_factor(s, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda")

    log = open(out / "train_log.jsonl", "w")
    best = {"macro_f1": -1.0}
    epochs_log, step, started = [], 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        order = epoch_order(len(train), args.seed, epoch)
        for chunk in batches(order, args.batch_size):
            if step >= total_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro in batches(chunk, args.micro_batch):
                loss = step_loss(model, tokenizer.pad_token_id, [train[i] for i in micro], device) / (len(chunk) / len(micro))
                scaler.scale(loss).backward()
                loss_value += float(loss.detach())
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_value, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0], "scale": scaler.get_scale()}) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(f"  step {step}/{total_steps} loss {loss_value:.4f} grad {grad_norm:.2f}", flush=True)
        val_loss = validation_loss(model, tokenizer.pad_token_id, val, args.val_loss_batch, device)
        outputs, strict, repaired = validate(model, tokenizer, val_rows, prompts, stop_ids, args)
        entry = {"epoch": epoch, "step": step, "val_loss": val_loss, "strict_macro_f1": strict["labels"]["overall"]["macro_f1"],
                 "strict_accuracy": strict["labels"]["overall"]["accuracy"], "malformed": strict["malformed"],
                 "repaired_macro_f1": repaired["labels"]["overall"]["macro_f1"], "evidence_macro_f1": strict["evidence"]["overall"]["macro"]["f1"]}
        epochs_log.append(entry)
        log.write(json.dumps({"validation": entry}) + "\n")
        log.flush()
        print("  ", json.dumps(entry), flush=True)
        if entry["strict_macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": entry["strict_macro_f1"], "epoch": epoch, "step": step, "strict": strict, "repair": repaired}
            save_best(model, args, out)
            with open(out / "val_outputs.jsonl", "w") as f:
                for row, o in zip(val_rows, outputs):
                    f.write(json.dumps({"annotation_id": row.annotation_id, **o}) + "\n")
        if step >= total_steps:
            break
    log.close()
    write_json(out / "report.json", {
        "config": vars(args), "trainable_parameters": n_trainable, "total_parameters": n_total,
        "train_rows": len(train), "val_rows": len(val), "steps": step, "total_steps": total_steps,
        "train_seconds": round(time.perf_counter() - started, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "epochs": epochs_log, "best": best,
    })


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runs, current = [], []
    for token in argv:
        if token == "--":
            runs.append(current)
            current = []
        else:
            current.append(token)
    runs.append(current)
    check_versions()
    write_json(results_dir("train") / "environment.json", environment())
    base = parse_args(runs[0])
    run(base)
    for extra in runs[1:]:
        # Free the previous model first. Later runs inherit the first run's settings unless they override them.
        gc.collect()
        torch.cuda.empty_cache()
        run(parse_args([*runs[0], *extra]))


if __name__ == "__main__":
    main()
