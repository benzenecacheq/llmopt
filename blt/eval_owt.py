"""Evaluate checkpoints on OpenWebText (held-out files 21-25 by default, or training files 0-20)."""

import argparse
import glob
import os
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def owt_heldout_tokens(tokenizer, start_file=21, n_files=5, max_tokens=None, chunk_size=10000):
    parquet_dir = os.path.expanduser(
        '~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/'
        'snapshots/b4325f019c648b1641a1784748667e8b74e5e064/plain_text/')
    files = sorted(glob.glob(parquet_dir + 'train-*.parquet'))[start_file:start_file + n_files]
    if not files:
        raise FileNotFoundError(f'No parquet files found in {parquet_dir}')
    print(f'  Loading {len(files)} files: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}')
    ds = load_dataset('parquet', data_files={'train': files}, split='train')
    # Read and tokenize directly from the Arrow dataset in chunks, never materializing the
    # full text corpus as one Python list -- that pattern OOM-killed a much larger (75-file)
    # corpus elsewhere in this project (see TokenDataset's openwebtext_large path), and
    # separately OOM-killed here too the first time this function got called *during*
    # training (train.py's periodic OWT eval) rather than standalone: the training corpus
    # is already resident in RAM at that point, so there's much less headroom than this
    # function has ever had to work within before. Every real caller passes an explicit
    # max_tokens, so stop as soon as enough tokens are collected -- keeps this fast for a
    # small in-training check too, not just memory-safe, since it never has to touch most
    # of the 5-file corpus for a 50K-token budget.
    id_chunks = []
    total = 0
    for i in range(0, len(ds), chunk_size):
        batch_texts = [t for t in ds[i:i + chunk_size]['text'] if t.strip()]
        if not batch_texts:
            continue
        encs = tokenizer._tokenizer.encode_batch(batch_texts)
        ids = torch.tensor([t for enc in encs for t in enc.ids], dtype=torch.long)
        id_chunks.append(ids)
        total += len(ids)
        if max_tokens and total >= max_tokens:
            break
    del ds
    tokens = torch.cat(id_chunks)
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
    parser.add_argument('--gqa-groups', type=int, default=2,
                        help='Number of KV groups for GQA checkpoint (default: 2)')
    parser.add_argument('--baseline-checkpoint', type=str, default=None,
                        help='Path to GPT-2 from-scratch checkpoint (.pt)')
    parser.add_argument('--hybrid-checkpoint', type=str, default=None,
                        help='Path to hybrid (MHA+BLT) from-scratch checkpoint (.pt)')
    parser.add_argument('--n-mha-layers', type=int, default=6,
                        help='Number of MHA layers in the hybrid model (default 6)')
    parser.add_argument('--blt-layers-per-m', type=int, default=0,
                        help='Strided layer-M grouping for BLT checkpoint (0=disabled)')
    parser.add_argument('--blt-num-m-groups', type=int, default=1,
                        help='Number of head-based M groups for BLT checkpoint (default: 1)')
    parser.add_argument('--blt-lowrank-checkpoint', type=str, default=None,
                        help='Path to low-rank BLT (M ≈ U@V^T) from-scratch checkpoint (.pt)')
    parser.add_argument('--uv-rank', type=int, default=0,
                        help='Rank of the low-rank BLT checkpoint given via --blt-lowrank-checkpoint')
    parser.add_argument('--num-uv-groups', type=int, default=1,
                        help='Number of (U,V) groups for the low-rank BLT checkpoint (default: 1)')
    parser.add_argument('--pretrained', type=str, default='gpt2',
                        help='HuggingFace model id controlling shape for --baseline-checkpoint '
                             '(e.g. gpt2-medium). Only affects the baseline path.')
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
        model = build_gqa_model(n_kv=args.gqa_groups, from_scratch=True).to(device)
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
        model = build_blt_model(from_scratch=True,
                                layers_per_m=args.blt_layers_per_m,
                                num_m_groups=args.blt_num_m_groups).to(device)
        ckpt = torch.load(args.blt_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
        torch.cuda.empty_cache()

    if args.blt_lowrank_checkpoint:
        from model import build_blt_lowrank_model
        print(f'Low-rank BLT (rank={args.uv_rank}, num_uv_groups={args.num_uv_groups}): '
              f'{args.blt_lowrank_checkpoint}')
        # from_scratch=True only to get the right shape cheaply -- load_state_dict
        # below fully overwrites with the checkpoint's own trained U/V/Wv/Wo.
        model = build_blt_lowrank_model(rank=args.uv_rank, from_scratch=True,
                                        num_uv_groups=args.num_uv_groups,
                                        pretrained=args.pretrained).to(device)
        ckpt = torch.load(args.blt_lowrank_checkpoint, map_location=device)
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
        print(f'Baseline GPT-2 (pretrained={args.pretrained}): {args.baseline_checkpoint}')
        model = GPT2LMHeadModel(GPT2Config.from_pretrained(args.pretrained)).to(device)
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        loss, ppl = sliding_window_loss(model, tokens, device)
        print(f'  OWT {split_label} loss: {loss:.4f} nats  |  ppl: {ppl:.2f}\n')
        del model
