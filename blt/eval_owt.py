"""Evaluate checkpoints on OpenWebText (held-out files 21-25 by default, or training files 0-20)."""

import argparse
import glob
import os
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def owt_heldout_tokens(tokenizer, start_file=21, n_files=5, max_tokens=None):
    parquet_dir = os.path.expanduser(
        '~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/'
        'snapshots/b4325f019c648b1641a1784748667e8b74e5e064/plain_text/')
    files = sorted(glob.glob(parquet_dir + 'train-*.parquet'))[start_file:start_file + n_files]
    if not files:
        raise FileNotFoundError(f'No parquet files found in {parquet_dir}')
    print(f'  Loading {len(files)} files: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}')
    ds = load_dataset('parquet', data_files={'train': files}, split='train')
    texts = [t for t in ds['text'] if t.strip()]
    del ds
    all_ids = []
    chunk_size = 10000
    for i in range(0, len(texts), chunk_size):
        encs = tokenizer._tokenizer.encode_batch(texts[i:i + chunk_size])
        all_ids.append(torch.tensor([t for enc in encs for t in enc.ids], dtype=torch.long))
    tokens = torch.cat(all_ids)
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def sliding_window_loss(model, input_ids, device, stride=512, max_length=1024):
    N = input_ids.size(0)
    nlls = []
    prev_end = 0
    for begin in tqdm(range(0, N, stride), desc='  eval'):
        end = min(begin + max_length, N)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue
        labels = chunk.clone()
        labels[:, :-target_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=labels)
            nlls.append(out.loss.item() * target_len)
        prev_end = end
        if end == N:
            break
    avg_loss = sum(nlls) / prev_end
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blt-checkpoint', type=str, default=None,
                        help='Path to BLT from-scratch checkpoint (.pt)')
    parser.add_argument('--gqa-checkpoint', type=str, default=None,
                        help='Path to GQA from-scratch checkpoint (.pt)')
    parser.add_argument('--baseline-checkpoint', type=str, default=None,
                        help='Path to GPT-2 from-scratch checkpoint (.pt)')
    parser.add_argument('--hybrid-checkpoint', type=str, default=None,
                        help='Path to hybrid (MHA+BLT) from-scratch checkpoint (.pt)')
    parser.add_argument('--n-mha-layers', type=int, default=6,
                        help='Number of MHA layers in the hybrid model (default 6)')
    parser.add_argument('--max-tokens', type=int, default=500000,
                        help='Tokens to evaluate over (default 500K)')
    parser.add_argument('--n-files', type=int, default=5,
                        help='Number of OWT files to use (default 5)')
    parser.add_argument('--train-set', action='store_true',
                        help='Evaluate on training files (0-20) instead of held-out files (21-25)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    if args.train_set:
        start_file = 0
        n_files = min(args.n_files, 21)
        split_label = 'train-set'
    else:
        start_file = 21
        n_files = args.n_files
        split_label = 'held-out'

    print(f'Loading OWT tokens ({split_label}, files {start_file}-{start_file + n_files - 1})...')
    tokens = owt_heldout_tokens(tokenizer, start_file=start_file, n_files=n_files, max_tokens=args.max_tokens)
    print(f'  {len(tokens):,} tokens\n')

    if args.gqa_checkpoint:
        from model import build_gqa_model
        print(f'GQA: {args.gqa_checkpoint}')
        model = build_gqa_model().to(device)
        ckpt = torch.load(args.gqa_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
        torch.cuda.empty_cache()

    if args.blt_checkpoint:
        from model import build_blt_model
        print(f'BLT: {args.blt_checkpoint}')
        model = build_blt_model(from_scratch=True).to(device)
        ckpt = torch.load(args.blt_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
        torch.cuda.empty_cache()

    if args.hybrid_checkpoint:
        from model import build_hybrid_model
        print(f'Hybrid: {args.hybrid_checkpoint}')
        model = build_hybrid_model(n_mha=args.n_mha_layers).to(device)
        ckpt = torch.load(args.hybrid_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
        torch.cuda.empty_cache()

    if args.baseline_checkpoint:
        from transformers import GPT2Config, GPT2LMHeadModel
        print(f'Baseline GPT-2: {args.baseline_checkpoint}')
        model = GPT2LMHeadModel(GPT2Config()).to(device)
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
