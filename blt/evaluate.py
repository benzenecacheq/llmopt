"""Perplexity evaluation on WikiText-103 or LAMBADA validation set."""

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm


def compute_perplexity(model, tokenizer, device, stride=512, max_length=1024,
                       dataset_split='validation', max_tokens=None, dataset='wikitext103'):
    """
    Sliding-window perplexity on WikiText-103 (default) or LAMBADA validation set.

    stride=512 with max_length=1024 means each window overlaps by 512 tokens;
    only the last `stride` tokens of each window contribute to the NLL estimate.
    """
    if dataset == 'lambada':
        ds = load_dataset('cimec/lambada', split='validation')
        text = '\n\n'.join(t for t in ds['text'] if t.strip())
    else:
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split=dataset_split)
        text = '\n\n'.join(ds['text'])

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]   # (N_tokens,)

    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]

    N = input_ids.size(0)
    model.eval()

    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, N, stride), desc='Evaluating'):
        end = min(begin + max_length, N)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)  # (1, L)

        # Only score tokens we haven't scored yet
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue

        labels = chunk.clone()
        labels[:, :-target_len] = -100   # mask the context tokens

        with torch.no_grad():
            out = model(chunk, labels=labels)
            nlls.append(out.loss.item() * target_len)

        prev_end = end
        if end == N:
            break

    ppl = torch.exp(torch.tensor(sum(nlls) / prev_end))
    return ppl.item()


def compute_cloze_accuracy(model, tokenizer, device, max_length=1024, max_samples=200):
    """Greedy last-word accuracy on LAMBADA validation passages.

    Uses max_samples for speed during training monitoring; run lm_eval for the
    full benchmark score.
    """
    from datasets import load_dataset
    ds = load_dataset('cimec/lambada', split='validation')
    correct = 0
    total = 0
    for item in ds:
        text = item['text']
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if len(ids) < 2:
            continue
        ids = ids[-max_length:]
        target = ids[-1]
        context = torch.tensor([ids[:-1]], device=device)
        with torch.no_grad():
            logits = model(input_ids=context).logits[0, -1, :]
        if logits.argmax().item() == target:
            correct += 1
        total += 1
        if max_samples and total >= max_samples:
            break
    return correct / total if total > 0 else 0.0


if __name__ == '__main__':
    import argparse
    from model import build_blt_model

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'blt', 'both'], default='both')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Limit tokens evaluated (for quick tests)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device(args.device)

    if args.model in ('baseline', 'both'):
        from transformers import GPT2LMHeadModel
        baseline = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        ppl = compute_perplexity(baseline, tokenizer, device, max_tokens=args.max_tokens)
        print(f'Baseline GPT-2 perplexity: {ppl:.2f}')
        del baseline

    if args.model in ('blt', 'both'):
        blt = build_blt_model('gpt2').to(device)
        ppl = compute_perplexity(blt, tokenizer, device, max_tokens=args.max_tokens)
        print(f'BLT (zero-shot) perplexity: {ppl:.2f}')
