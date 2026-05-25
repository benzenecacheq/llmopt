"""
BLT training — GD-only control condition.
All parameters (M, Wv, Wo, FFN, embeddings, layernorms) trained with Adam.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

from model import build_blt_model
from evaluate import compute_perplexity


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
    """WikiText-103 training tokens chunked into fixed-length blocks."""

    def __init__(self, tokenizer, block_size=1024, chunk_size=10000):
        from datasets import load_dataset
        print('Loading WikiText-103 train split...')
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')

        # Tokenize in chunks to avoid building a single giant string in RAM
        all_ids = []
        texts = [t for t in ds['text'] if t.strip()]
        for i in range(0, len(texts), chunk_size):
            chunk_text = '\n\n'.join(texts[i:i + chunk_size])
            ids = tokenizer(chunk_text, return_tensors='pt',
                            truncation=False).input_ids[0]
            all_ids.append(ids)

        tokens = torch.cat(all_ids)
        n_blocks = len(tokens) // block_size
        tokens = tokens[:n_blocks * block_size]
        self.blocks = tokens.view(n_blocks, block_size).clone()
        print(f'  {n_blocks:,} blocks of {block_size} tokens ({len(tokens):,} tokens total)')

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Seed: {args.seed}')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print('Building BLT model...')
    model = build_blt_model().to(device)
    n_params = sum(p.numel() for p in set(model.parameters()))
    print(f'  {n_params:,} unique parameters')

    dataset = TokenDataset(tokenizer, block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        pin_memory=(device.type == 'cuda'), num_workers=2)

    optimizer = torch.optim.Adam(set(model.parameters()), lr=args.lr)

    # Linear warmup then cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = args.log_file or f'run_seed{args.seed}.log'
    log_f = open(log_path, 'w', buffering=1)

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')

    log(f'seed={args.seed} lr={args.lr} batch={args.batch_size} block={args.block_size}')
    log(f'step\telapsed\tlr\ttrain_loss\tval_ppl')

    step = 0
    t0 = time.time()

    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = batch.to(device)
            out = model(input_ids=batch, labels=batch)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                val_ppl = ''
                if step % args.eval_every == 0:
                    model.eval()
                    ppl = compute_perplexity(model, tokenizer, device,
                                            max_tokens=args.eval_tokens)
                    model.train()
                    val_ppl = f'{ppl:.2f}'
                log(f'{step}\t{elapsed:.0f}s\t{lr_now:.2e}\t{loss.item():.4f}\t{val_ppl}')

            step += 1

    # Final evaluation
    model.eval()
    final_ppl = compute_perplexity(model, tokenizer, device)
    log(f'\nFinal val perplexity: {final_ppl:.2f}')

    if args.save_path:
        torch.save({
            'step': step,
            'seed': args.seed,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_ppl': final_ppl,
        }, args.save_path)
        log(f'Saved to {args.save_path}')

    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-tokens', type=int, default=50000)
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    args = parser.parse_args()
    train(args)
