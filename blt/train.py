"""
BLT training — GD-only control condition.
All parameters (M, Wv, Wo, FFN, embeddings, layernorms) trained with Adam.
"""

import argparse
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

from model import build_blt_model
from evaluate import compute_perplexity
from transformers import GPT2LMHeadModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
    """Text dataset chunked into fixed-length blocks. Supports wikitext103 and lambada."""

    def __init__(self, tokenizer, block_size=1024, dataset='wikitext103', chunk_size=10000):
        from datasets import load_dataset

        if dataset == 'lambada':
            ds = load_dataset('cimec/lambada', split='train')
            texts = [t for t in ds['text'] if t.strip()]
            # LAMBADA docs are ~90K tokens each; tokenize one at a time to avoid OOM
            chunk_size = 1
        else:
            ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
            texts = [t for t in ds['text'] if t.strip()]

        all_ids = []
        for i in range(0, len(texts), chunk_size):
            chunk_text = '\n\n'.join(texts[i:i + chunk_size])
            ids = tokenizer(chunk_text, return_tensors='pt',
                            truncation=False).input_ids[0]
            all_ids.append(ids)

        tokens = torch.cat(all_ids)
        n_blocks = len(tokens) // block_size
        tokens = tokens[:n_blocks * block_size]
        self.blocks = tokens.view(n_blocks, block_size).clone()

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def save_checkpoint(path, model, optimizer, scheduler, step, seed, val_ppl=None):
    torch.save({
        'step': step,
        'seed': seed,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'val_ppl': val_ppl,
    }, path)


def train(args):
    log_path = args.log_file or f'run_seed{args.seed}.log'
    log_f = open(log_path, 'a' if args.resume else 'w', buffering=1)

    def log(msg):
        log_f.write(msg + '\n')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}  |  Seed: {args.seed}')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.baseline:
        log('Building baseline GPT-2 model...')
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    else:
        log('Building BLT model...')
        model = build_blt_model().to(device)
    # Deduplicate shared parameters (e.g. BLT's shared M matrix) preserving
    # insertion order so optimizer state is stable across resume.
    seen_ids = set()
    unique_params = []
    for p in model.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            unique_params.append(p)
    n_params = sum(p.numel() for p in unique_params)
    log(f'  {n_params:,} unique parameters')

    dataset = TokenDataset(tokenizer, block_size=args.block_size, dataset=args.dataset)
    log(f'  {len(dataset):,} blocks of {args.block_size} tokens ({args.dataset})')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        pin_memory=(device.type == 'cuda'), num_workers=2)

    optimizer = torch.optim.Adam(unique_params, lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    last_val_ppl = None

    if args.resume:
        log(f'Resuming from {args.resume} ...')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        last_val_ppl = ckpt.get('val_ppl')
        log(f'  Resumed at step {start_step}')

    if args.finetune:
        log(f'Fine-tuning from {args.finetune} (fresh optimizer/scheduler) ...')
        ckpt = torch.load(args.finetune, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        log(f'  Loaded weights from step {ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')

    if not args.resume:
        log(f'seed={args.seed} lr={args.lr} batch={args.batch_size} block={args.block_size} max_steps={args.max_steps}')
        log(f'step\telapsed\tlr\ttrain_loss\tval_ppl')

    step = start_step
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
                                            max_tokens=args.eval_tokens,
                                            dataset=args.dataset)
                    model.train()
                    val_ppl = f'{ppl:.2f}'
                    last_val_ppl = ppl
                log(f'{step}\t{elapsed:.0f}s\t{lr_now:.2e}\t{loss.item():.4f}\t{val_ppl}')

            if args.save_path and step > start_step and step % args.checkpoint_every == 0:
                save_checkpoint(args.save_path, model, optimizer, scheduler,
                                step, args.seed, last_val_ppl)

            step += 1

    model.eval()
    final_ppl = compute_perplexity(model, tokenizer, device)
    log(f'\nFinal val perplexity: {final_ppl:.2f}')

    if args.save_path:
        save_checkpoint(args.save_path, model, optimizer, scheduler,
                        step, args.seed, final_ppl)
        log(f'Saved to {args.save_path}')

    log_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=50300)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--eval-tokens', type=int, default=50000)
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None,
                        help='Load model weights only (fresh optimizer/scheduler)')
    parser.add_argument('--baseline', action='store_true',
                        help='Fine-tune vanilla GPT-2 instead of BLT')
    parser.add_argument('--dataset', type=str, default='wikitext103',
                        choices=['wikitext103', 'lambada'])
    args = parser.parse_args()
    train(args)
