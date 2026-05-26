"""Sanity-check generation from a BLT checkpoint."""

import argparse
import torch
from transformers import GPT2Tokenizer

from model import build_blt_model

PROMPTS = [
    "The history of artificial intelligence began",
    "In the field of mathematics, one of the most important",
    "The experiment was designed to test whether",
    "Scientists have long debated the origins of",
]


def generate(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids)
            logits = out.logits[:, -1, :] / temperature
            # top-k filtering
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_blt_model().to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f'  Loaded at step {ckpt["step"]}, val_ppl={ckpt.get("val_ppl", "n/a")}\n')

    for prompt in PROMPTS:
        print(f'--- Prompt: "{prompt}"')
        text = generate(model, tokenizer, device, prompt,
                        args.max_new_tokens, args.temperature, args.top_k)
        print(text)
        print()
