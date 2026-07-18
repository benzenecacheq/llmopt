"""
BLT training — GD-only control condition.
All parameters (M, Wv, Wo, FFN, embeddings, layernorms) trained with Adam.
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import GPT2Tokenizer

from model import build_blt_model, build_gqa_model, build_hybrid_model
from evaluate import compute_perplexity, compute_cloze_accuracy
from transformers import GPT2LMHeadModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResumableSampler(Sampler):
    """Deterministic shuffling that can resume mid-epoch without replaying already-consumed
    data. Each epoch's permutation is derived from seed+epoch_index (not the shared global
    RNG), so it is fully reproducible from (seed, cumulative samples consumed) alone --
    unlike DataLoader(shuffle=True), whose order depends on global RNG state that isn't
    saved in checkpoints, causing resumed runs to replay the start of the shuffle.

    `start_sample` is the cumulative sample count already consumed before this sampler was
    constructed (i.e. micro_step * batch_size from the checkpoint, or 0 for a fresh run).
    """

    def __init__(self, dataset_len, seed, start_sample=0):
        self.dataset_len = dataset_len
        self.seed = seed
        self.epoch = start_sample // dataset_len
        self.offset = start_sample % dataset_len

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.dataset_len, generator=g).tolist()
        yield from perm[self.offset:]
        self.epoch += 1
        self.offset = 0

    def __len__(self):
        return self.dataset_len - self.offset


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
        elif dataset in ('openwebtext', 'openwebtext_large'):
            # openwebtext: original 21-file / ~2M-doc setup, used by every prior run in this
            #   project -- cache path, file selection, and loading strategy unchanged for
            #   reproducibility (~2.2M docs comfortably fits in memory as a plain Python list).
            # openwebtext_large: files 0-20 + 26-79 (75 files, ~7.9B tokens, single-epoch
            #   coverage for the 1.5M-step GPT-2 medium runs), explicitly skipping files
            #   21-25 since those are the held-out set eval_owt.py has always used -- keeping
            #   that range untouched means held-out numbers stay comparable across every run,
            #   small and medium alike. At ~7.5M docs, materializing the whole corpus as a
            #   Python list of strings (as the small path does) OOM-killed the process with no
            #   traceback -- loaded and tokenized in chunks directly from the Arrow dataset
            #   instead, so peak memory stays bounded to one chunk of raw text at a time.
            large = (dataset == 'openwebtext_large')
            cache_path = os.path.expanduser(
                '~/.cache/blt_owt_75files_blocks.pt' if large else '~/.cache/blt_owt_2m_blocks.pt')
            if os.path.exists(cache_path):
                self.blocks = torch.load(cache_path)
                return
            # Bulk Arrow load is far faster than streaming row-by-row.
            import glob
            parquet_dir = os.path.expanduser(
                '~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/'
                'snapshots/b4325f019c648b1641a1784748667e8b74e5e064/plain_text/')
            all_files = sorted(glob.glob(parquet_dir + 'train-*.parquet'))
            if large:
                # At ~7.9B tokens, accumulating a growing Python list of chunk tensors and
                # then torch.cat()-ing it briefly holds BOTH the full pre-concat list AND
                # the new concatenated tensor at once (~63GB each at int64, ~126GB combined)
                # -- this OOM-killed the process with no traceback on a 94GB machine, and
                # would be even tighter on a 62GB one. Two-pass fix: pass 1 tokenizes each
                # chunk just to measure its length and discards the ids immediately (peak
                # memory O(chunk_size), not O(corpus_size)); pass 2 pre-allocates ONE int32
                # tensor sized to the known total and fills it in place chunk by chunk, so
                # there is never more than one full-corpus-sized allocation plus one small
                # transient chunk in memory at once. Costs re-tokenizing once more (CPU-bound,
                # not the resource under pressure) in exchange for a much safer memory profile.
                files = all_files[0:21] + all_files[26:80]
                ds = load_dataset('parquet', data_files={'train': files}, split='train')

                def chunk_ranges():
                    for i in range(0, len(ds), chunk_size):
                        batch_texts = [t for t in ds[i:i + chunk_size]['text'] if t.strip()]
                        if not batch_texts:
                            continue
                        yield batch_texts

                total = 0
                for batch_texts in chunk_ranges():
                    encs = tokenizer._tokenizer.encode_batch(batch_texts)
                    total += sum(len(enc.ids) for enc in encs)

                tokens = torch.empty(total, dtype=torch.int32)
                pos = 0
                for batch_texts in chunk_ranges():
                    encs = tokenizer._tokenizer.encode_batch(batch_texts)
                    ids = torch.tensor([t for enc in encs for t in enc.ids], dtype=torch.int32)
                    tokens[pos:pos + len(ids)] = ids
                    pos += len(ids)
                del ds

                # No .clone() here (unlike the shared path below): `tokens` was allocated
                # fresh just for this dataset, not sliced from some larger shared object, so
                # holding onto its storage via a view costs at most block_size-1 wasted
                # elements -- cloning would momentarily double memory again for no benefit.
                n_blocks = len(tokens) // block_size
                self.blocks = tokens[:n_blocks * block_size].view(n_blocks, block_size)
                torch.save(self.blocks, cache_path)
                return
            else:
                files = all_files[:21]
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
        x = self.blocks[idx].long()
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


def model_is_finite(model):
    return all(torch.isfinite(p).all() for p in model.parameters())


def save_checkpoint(path, model, optimizer, scheduler, step, seed, val_ppl=None, ema_loss=None,
                    micro_step=None):
    if not model_is_finite(model):
        raise RuntimeError(
            f'Refusing to save checkpoint at step {step}: model parameters contain NaN/Inf. '
            f'Existing checkpoint at {path} (and {path}.bak) left untouched -- fix the '
            f'instability and resume from there rather than overwriting with a corrupted state.'
        )
    tmp = path + '.tmp'
    torch.save({
        'step': step,
        'seed': seed,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'val_ppl': val_ppl,
        'ema_loss': ema_loss,
        'micro_step': micro_step,
    }, tmp)
    # Keep previous checkpoint as .bak so a power cut during write leaves a fallback.
    if os.path.exists(path):
        os.replace(path, path + '.bak')
    os.replace(tmp, path)


def train(args):
    if args.log_file:
        log_path = args.log_file
    elif args.save_path:
        log_path = os.path.splitext(args.save_path)[0] + '.log'
    else:
        log_path = f'run_seed{args.seed}.log'
    log_f = open(log_path, 'a' if args.resume else 'w', buffering=1)

    def log(msg):
        log_f.write(msg + '\n')

    set_seed(args.seed)
    if not torch.cuda.is_available():
        msg = ('ERROR: no CUDA GPU detected (torch.cuda.is_available() is False). '
               'Refusing to silently fall back to CPU -- a full training run would proceed '
               'at ~20x slower with no warning. Fix the GPU/driver and retry.')
        log(msg)
        print(msg, file=sys.stderr)
        sys.exit(1)
    device = torch.device('cuda')
    log(f'Device: {device}  |  Seed: {args.seed}')

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.hybrid:
        log(f'Building hybrid model ({args.n_mha_layers} MHA layers, shape={args.pretrained}, from scratch)...')
        model = build_hybrid_model(n_mha=args.n_mha_layers, pretrained=args.pretrained).to(device)
    elif args.gqa:
        log(f'Building GQA model ({args.num_kv_groups} KV groups, shape={args.pretrained}, from scratch)...')
        model = build_gqa_model(n_kv_groups=args.num_kv_groups, pretrained=args.pretrained).to(device)
    elif args.baseline:
        if args.from_scratch:
            log(f'Building GPT-2 model from scratch (random init, shape={args.pretrained})...')
            from transformers import GPT2Config
            model = GPT2LMHeadModel(GPT2Config.from_pretrained(args.pretrained)).to(device)
        else:
            log(f'Building baseline GPT-2 model (pretrained={args.pretrained})...')
            model = GPT2LMHeadModel.from_pretrained(args.pretrained).to(device)
    else:
        log(f'Building BLT model (pretrained={args.pretrained}, num_m_groups={args.num_m_groups}, random_m={args.random_m}, from_scratch={args.from_scratch}, warmstart_scale={args.warmstart_scale}, per_layer_m={args.per_layer_m})...')
        model = build_blt_model(pretrained=args.pretrained, num_m_groups=args.num_m_groups,
                                random_m=args.random_m, from_scratch=args.from_scratch,
                                warmstart_scale=args.warmstart_scale,
                                per_layer_m=args.per_layer_m).to(device)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        log('  gradient checkpointing enabled')

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

    # Peek at the checkpoint before building the DataLoader so a resumed run's sampler
    # can pick up exactly where the data stream left off, instead of replaying the start
    # of the shuffle (DataLoader(shuffle=True)'s order depends on global RNG state, which
    # isn't part of the checkpoint).
    resume_ckpt = None
    resume_micro_step = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        resume_micro_step = resume_ckpt.get('micro_step')
        if resume_micro_step is None:
            resume_micro_step = resume_ckpt['step'] * args.grad_accum_steps

    if args.dataset == 'lambada_cloze':
        dataset = ClozeDataset(tokenizer, max_length=args.block_size)
        log(f'  {len(dataset):,} passages for cloze training (lambada_cloze)')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=cloze_collate,
                            pin_memory=(device.type == 'cuda'), num_workers=2)
    else:
        dataset = TokenDataset(tokenizer, block_size=args.block_size, dataset=args.dataset)
        log(f'  {len(dataset):,} blocks of {args.block_size} tokens ({args.dataset})')
        sampler = ResumableSampler(len(dataset), args.seed,
                                   start_sample=resume_micro_step * args.batch_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
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
    vocab_size = model.config.vocab_size
    ema_loss = torch.full((vocab_size,), math.log(vocab_size), device=device)

    if args.resume:
        log(f'Resuming from {args.resume} ...')
        ckpt = resume_ckpt
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        last_val_ppl = ckpt.get('val_ppl')
        if ckpt.get('ema_loss') is not None:
            ema_loss = ckpt['ema_loss'].to(device)
        log(f'  Resumed at step {start_step}')

    if args.finetune:
        log(f'Fine-tuning from {args.finetune} (fresh optimizer/scheduler) ...')
        ckpt = torch.load(args.finetune, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        log(f'  Loaded weights from step {ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')
        if ckpt.get('ema_loss') is not None:
            ema_loss = ckpt['ema_loss'].to(device)
            log('  Loaded EMA per-token-loss buffer from checkpoint')

    if not args.resume:
        log(f'seed={args.seed} lr={args.lr} batch={args.batch_size} block={args.block_size} max_steps={args.max_steps}')
        log(f'step\telapsed\tlr\ttrain_loss\tval_ppl\tlambada_acc')

    step = start_step
    t0 = time.time()
    last_lambada_acc = None
    micro_step = resume_micro_step
    accum_loss = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.train()
    optimizer.zero_grad()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids, labels = batch
            # Skip batches with out-of-range token IDs (rare OWT corruption)
            if input_ids.max() >= 50257 or input_ids.min() < 0:
                continue
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                if args.ema_loss_weighting:
                    logits = model(input_ids=input_ids).logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    valid = shift_labels != -100
                    flat_logits = shift_logits.view(-1, vocab_size)
                    flat_labels = shift_labels.view(-1)
                    flat_valid = valid.view(-1)
                    ids = flat_labels[flat_valid]
                    per_token_loss = F.cross_entropy(flat_logits[flat_valid], ids, reduction='none')

                    weight = ema_loss[ids]
                    weight = weight / weight.mean()
                    weight = args.ema_blend * weight + (1 - args.ema_blend) * 1.0
                    loss = (weight.detach() * per_token_loss).mean()

                    with torch.no_grad():
                        sum_loss = torch.zeros(vocab_size, device=device)
                        counts = torch.zeros(vocab_size, device=device)
                        sum_loss.scatter_add_(0, ids, per_token_loss.detach())
                        counts.scatter_add_(0, ids, torch.ones_like(per_token_loss))
                        seen = counts > 0
                        batch_mean = sum_loss[seen] / counts[seen]
                        ema_loss[seen] = args.ema_decay * ema_loss[seen] + (1 - args.ema_decay) * batch_mean
                else:
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out.loss
                loss = loss / args.grad_accum_steps

            if not torch.isfinite(loss):
                # Forward-pass numerical blowup (e.g. fp16 overflow in activations) --
                # GradScaler only guards the gradient path, not this. Skip the batch
                # entirely rather than let a NaN/Inf loss reach backward() and
                # corrupt every parameter on the next optimizer step.
                log(f'  step {step}: non-finite loss ({loss.item()}), skipping batch')
                continue

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro_step += 1
            if micro_step % args.grad_accum_steps != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            avg_loss = accum_loss
            accum_loss = 0.0

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
                log(f'{step}\t{elapsed:.0f}s\t{lr_now:.2e}\t{avg_loss:.4f}\t{val_ppl}\t{lambada_acc}')

            if args.save_path and step > start_step and step % args.checkpoint_every == 0:
                save_checkpoint(args.save_path, model, optimizer, scheduler,
                                step, args.seed, last_val_ppl, ema_loss,
                                micro_step=micro_step)

            step += 1

    model.eval()
    final_ppl = compute_perplexity(model, tokenizer, device)
    log(f'\nFinal val perplexity: {final_ppl:.2f}')

    if args.save_path:
        save_checkpoint(args.save_path, model, optimizer, scheduler,
                        step, args.seed, final_ppl, ema_loss,
                        micro_step=micro_step)
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
    parser.add_argument('--hybrid', action='store_true',
                        help='Hybrid model: first --n-mha-layers use MHA, rest use BLT (always from scratch)')
    parser.add_argument('--n-mha-layers', type=int, default=6,
                        help='Number of early MHA layers in hybrid model (default: 6)')
    parser.add_argument('--gqa', action='store_true',
                        help='Use GQA baseline (always from scratch)')
    parser.add_argument('--num-kv-groups', type=int, default=2,
                        help='Number of KV groups for --gqa (must evenly divide n_head=12; default 2)')
    parser.add_argument('--baseline', action='store_true',
                        help='Fine-tune vanilla GPT-2 instead of BLT')
    parser.add_argument('--num-m-groups', type=int, default=1, choices=[1, 2],
                        help='Number of shared M matrices (1=original BLT, 2=GQA-like)')
    parser.add_argument('--random-m', action='store_true',
                        help='Initialize M randomly N(0, 1/sqrt(D)) instead of from Wq@Wk^T')
    parser.add_argument('--per-layer-m', action='store_true',
                        help='Give each layer its own M instead of sharing one M across all '
                             'layers (still shared across heads within a layer). '
                             'Only valid with --num-m-groups 1.')
    parser.add_argument('--lambada-eval-every', type=int, default=2000,
                        help='Evaluate LAMBADA cloze accuracy every N steps (0 to disable)')
    parser.add_argument('--dataset', type=str, default='wikitext103',
                        choices=['wikitext103', 'lambada', 'lambada_cloze', 'pg19', 'openwebtext',
                                'openwebtext_large'])
    parser.add_argument('--from-scratch', action='store_true',
                        help='Randomly initialize all weights (no pretrained GPT-2 load)')
    parser.add_argument('--ema-loss-weighting', action='store_true',
                        help='Weight each token\'s loss by an EMA of its historical per-token-id loss '
                             '(upweights persistently-hard tokens, downweights mastered ones)')
    parser.add_argument('--ema-decay', type=float, default=0.99,
                        help='Decay rate for the per-token-id EMA loss buffer')
    parser.add_argument('--ema-blend', type=float, default=1.0,
                        help='Blend factor between EMA-weighted loss (1.0) and standard '
                             'unweighted loss (0.0); e.g. 0.5 averages the two per-token weights')
    parser.add_argument('--pretrained', type=str, default='gpt2',
                        help='HuggingFace model id to load pretrained weights from '
                             '(e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    parser.add_argument('--warmstart-scale', type=float, default=1.0,
                        help='Blend factor in [0,1] for the Wq@Wk^T-average M init; '
                             'lower values soften the warm-start perturbation for models '
                             'with many layers/heads (no effect with --random-m)')
    parser.add_argument('--fp16', action='store_true',
                        help='Mixed-precision training (autocast + GradScaler); fp32 master '
                             'weights and optimizer state are kept, only compute is fp16')
    parser.add_argument('--grad-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to trade compute for activation memory')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Accumulate gradients over N micro-batches before each optimizer step '
                             '(effective batch size = batch-size * grad-accum-steps)')
    args = parser.parse_args()
    train(args)
