"""Greedy batch generation shared by the baseline and training evaluations."""
import torch

GREEDY = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)


def generate(model, tokenizer, prompts, batch_size, max_new_tokens, stop_ids):
    """Greedy outputs in input order. The tokenizer must pad on the left. Batches are sorted by length to limit padding."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        batch = tokenizer([prompts[i] for i in ids], return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **batch, max_new_tokens=max_new_tokens, eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id, **GREEDY
            )
        new = generated[:, batch.input_ids.shape[1]:]
        for i, tokens in zip(ids, new):
            kept = tokens[tokens != tokenizer.pad_token_id]
            outputs[i] = {"text": tokenizer.decode(kept, skip_special_tokens=True), "new_tokens": int(len(kept))}
    return outputs
