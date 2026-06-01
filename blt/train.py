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
from evaluate import compute_perplexity, compute_cloze_accuracy
from transformers import GPT2LMHeadModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
    """Text dataset chunked into fixed-length blocks. Supports wikitext103, lambada, pg19, openwebtext."""

    def __init__(self, tokenizer, block_size=1024, dataset='wikitext103', chunk_size=10000):
        from datasets import load_dataset

        if dataset == 'lambada':
            ds = load_dataset('cimec/lambada', split='train')
            texts = [t for t in ds['text'] if t.strip()]
            # LAMBADA docs are ~90K tokens each; tokenize one at a time to avoid OOM
            chunk_size = 1
        elif dataset == 'pg19':
            ds = load_dataset('emozilla/pg19', split='train')
            texts = [t for t in ds['text'] if t.strip()]
            # pg19 books are 4M+ chars each; tokenize one at a time to avoid OOM
            chunk_size = 1
        elif dataset == 'openwebtext':
            cache_path = os.path.expanduser('~/.cache/blt_owt_2m_blocks.pt')
            if os.path.exists(cache_path):
                self.blocks = torch.load(cache_path)
                return
            # Load first 21 parquet files directly from local cache (~2.1M docs).
            # Bulk Arrow load is far faster than streaming row-by-row.
            import glob
            parquet_dir = os.path.expanduser(
                '~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/'
                'snapshots/b4325f019c648b1641a1784748667e8b74e5e064/plain_text/')
            files = sorted(glob.glob(parquet_dir + 'train-*.parquet'))[:21]
            ds = load_dataset('parquet', data_files={'train': files}, split='train')
            texts = [t for t in ds['text'] if t.strip()][:2000000]
            del ds
        else:
            ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
            texts = [t for t in ds['text'] if t.strip()]

        if 'ds' in dir():
            del ds
        all_ids = []
        for i in range(0, len(texts), chunk_size):
            encs = tokenizer._tokenizer.encode_batch(texts[i:i + chunk_size])
            all_ids.append(torch.tensor([t for enc in encs for t in enc.ids],
                                        dtype=torch.long))

        tokens = torch.cat(all_ids)
        del all_ids
        n_blocks = len(tokens) // block_size
        self.blocks = tokens[:n_blocks * block_size].view(n_blocks, block_size).clone()
        if dataset == 'openwebtext':
            torch.save(self.blocks, cache_path)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        x = self.blocks[idx]
        return x, x


class ClozeDataset(Dataset):
    """LAMBADA passages with loss computed only on the final token.

    Each item is the last max_length tokens of a training document, with labels
    masked to -100 everywhere except the last position. This trains the model
    directly on the last-word prediction task that LAMBADA tests.
    """

    def __init__(self, tokenizer, max_length=1024, split='train'):
        from datasets import load_dataset
        ds = load_dataset('cimec/lambada', split=split)

        self.items = []
        for text in ds['text']:
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False,
                            truncation=False)['input_ids']
            if len(ids) < 2:
                continue
            ids = ids[-max_length:]       # keep story ending, including last word
            labels = [-100] * len(ids)
            labels[-1] = ids[-1]          # only the final token contributes to loss
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def cloze_collate(batch):
    """Left-pad variable-length (input_ids, labels) pairs to batch max length."""
    max_len = max(len(ids) for ids, _ in batch)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        L = len(ids)
        padded_ids[i, max_len - L:] = torch.tensor(ids, dtype=torch.long)
        padded_labels[i, max_len - L:] = torch.tensor(labels, dtype=torch.long)
    return padded_ids, padded_labels


def save_checkpoint(path, model, optimizer, scheduler, step, seed, val_ppl=None):
    tmp = path + '.tmp'
    torch.save({
        'step': step,
        'seed': seed,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'val_ppl': val_ppl,
    }, tmp)
    # Keep previous checkpoint as .bak so a power cut during write leaves a fallback.
    if os.path.exists(path):
        os.replace(path, path + '.bak')
    os.replace(tmp, path)


def train(args):
    log_path = args.log_file or f'run_seed{args.seed}.log'
    log_f = open(log_path, 'a' if args.resume else 'w', buffering=1)

    def log(msg):
        log_f.write(msg + '\n')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}  |  Seed: {args.seed}')

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.baseline:
        if args.from_scratch:
            log('Building GPT-2 model from scratch (random init)...')
            from transformers import GPT2Config
            model = GPT2LMHeadModel(GPT2Config()).to(device)
        else:
            log('Building baseline GPT-2 model (pretrained)...')
            model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    else:
        log(f'Building BLT model (num_m_groups={args.num_m_groups}, random_m={args.random_m}, from_scratch={args.from_scratch})...')
        model = build_blt_model(num_m_groups=args.num_m_groups, random_m=args.random_m,
                                from_scratch=args.from_scratch).to(device)
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

    if args.dataset == 'lambada_cloze':
        dataset = ClozeDataset(tokenizer, max_length=args.block_size)
        log(f'  {len(dataset):,} passages for cloze training (lambada_cloze)')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=cloze_collate,
                            pin_memory=(device.type == 'cuda'), num_workers=2)
    else:
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
        log(f'step\telapsed\tlr\ttrain_loss\tval_ppl\tlambada_acc')

    step = start_step
    t0 = time.time()
    last_lambada_acc = None

    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids, labels = batch
            # Skip batches with out-of-range token IDs (rare OWT corruption)
            if input_ids.max() >= 50257 or input_ids.min() < 0:
                step += 1
                continue
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
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
                lambada_acc = ''
                if step % args.eval_every == 0:
                    model.eval()
                    if args.dataset == 'lambada_cloze':
                        acc = compute_cloze_accuracy(model, tokenizer, device)
                        val_ppl = f'{acc:.4f}'
                        last_val_ppl = acc
                    else:
                        ppl = compute_perplexity(model, tokenizer, device,
                                                max_tokens=args.eval_tokens,
                                                dataset=args.dataset)
                        val_ppl = f'{ppl:.2f}'
                        last_val_ppl = ppl
                    model.train()
                if args.lambada_eval_every and step % args.lambada_eval_every == 0:
                    model.eval()
                    last_lambada_acc = compute_cloze_accuracy(model, tokenizer, device)
                    lambada_acc = f'{last_lambada_acc:.4f}'
                    model.train()
                log(f'{step}\t{elapsed:.0f}s\t{lr_now:.2e}\t{loss.item():.4f}\t{val_ppl}\t{lambada_acc}')

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
    parser.add_argument('--num-m-groups', type=int, default=1, choices=[1, 2],
                        help='Number of shared M matrices (1=original BLT, 2=GQA-like)')
    parser.add_argument('--random-m', action='store_true',
                        help='Initialize M randomly N(0, 1/sqrt(D)) instead of from Wq@Wk^T')
    parser.add_argument('--lambada-eval-every', type=int, default=2000,
                        help='Evaluate LAMBADA cloze accuracy every N steps (0 to disable)')
    parser.add_argument('--dataset', type=str, default='wikitext103',
                        choices=['wikitext103', 'lambada', 'lambada_cloze', 'pg19', 'openwebtext'])
    parser.add_argument('--from-scratch', action='store_true',
                        help='Randomly initialize all weights (no pretrained GPT-2 load)')
    args = parser.parse_args()
    train(args)
