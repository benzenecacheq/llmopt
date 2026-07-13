"""lm-evaluation-harness wrapper for BLT and baseline GPT-2 checkpoints."""

import argparse
import torch
from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from model import build_blt_model, build_gqa_model, build_hybrid_model


def load_model(checkpoint_path, baseline=False, gqa=False, hybrid=False,
               device='cuda', num_m_groups=1, n_mha_layers=6, pretrained='gpt2',
               per_layer_m=False, num_kv_groups=2):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if hybrid:
        model = build_hybrid_model(n_mha=n_mha_layers).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded hybrid checkpoint: step={ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')
    elif gqa:
        model = build_gqa_model(n_kv_groups=num_kv_groups).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded GQA checkpoint: step={ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')
    elif baseline:
        model = GPT2LMHeadModel.from_pretrained(pretrained).to(device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            print(f'Loaded baseline checkpoint: step={ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')
    else:
        model = build_blt_model(pretrained=pretrained, num_m_groups=num_m_groups,
                                per_layer_m=per_layer_m).to(device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            print(f'Loaded BLT checkpoint: step={ckpt["step"]}, val_ppl={ckpt.get("val_ppl")}')

    model.eval()
    return model, tokenizer


@register_model('blt')
class BLTModel(LM):
    def __init__(self, checkpoint, baseline=False, gqa=False, hybrid=False,
                 device='cuda', batch_size=8, num_m_groups=1, n_mha_layers=6,
                 pretrained='gpt2', per_layer_m=False, num_kv_groups=2):
        super().__init__()
        self._device = torch.device(device)
        self.model, self.tokenizer = load_model(
            checkpoint,
            baseline=(baseline == 'true' or baseline is True),
            gqa=(gqa == 'true' or gqa is True),
            hybrid=(hybrid == 'true' or hybrid is True),
            device=device, num_m_groups=int(num_m_groups),
            n_mha_layers=int(n_mha_layers), pretrained=pretrained,
            per_layer_m=(per_layer_m == 'true' or per_layer_m is True),
            num_kv_groups=int(num_kv_groups),
        )
        self._batch_size = int(batch_size)
        self._max_length = 1024

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        # Truncate context if needed to fit within max_length
        max_ctx = self._max_length - len(cont_ids) - 1
        if max_ctx < 1:
            max_ctx = 1
        ctx_ids = ctx_ids[-max_ctx:]
        return ctx_ids, cont_ids

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []

        for i in range(0, len(requests), self._batch_size):
            batch = requests[i:i + self._batch_size]
            pairs = [self._encode_pair(*req.args) for req in batch]

            # Left-pad with EOS so all sequences are the same length
            max_len = min(max(len(c) + len(k) for c, k in pairs), self._max_length)
            input_tensor = torch.full((len(pairs), max_len), self.eot_token_id,
                                      dtype=torch.long, device=self._device)
            for j, (ctx_ids, cont_ids) in enumerate(pairs):
                full = (ctx_ids + cont_ids)[-max_len:]
                input_tensor[j, max_len - len(full):] = torch.tensor(full)

            with torch.no_grad():
                log_probs = torch.nn.functional.log_softmax(
                    self.model(input_ids=input_tensor).logits, dim=-1
                )  # (B, L, V)

            for j, (ctx_ids, cont_ids) in enumerate(pairs):
                total = len(ctx_ids) + len(cont_ids)
                truncated = max(0, total - max_len)
                ctx_kept = max(0, len(ctx_ids) - truncated)
                pad = max_len - min(total, max_len)
                cont_start = pad + ctx_kept  # index in padded input where cont begins

                nll = 0.0
                is_greedy = True
                for k, tok in enumerate(cont_ids):
                    pos = cont_start - 1 + k  # logit at pos predicts token at pos+1
                    nll -= log_probs[j, pos, tok].item()
                    if log_probs[j, pos].argmax().item() != tok:
                        is_greedy = False

                results.append((-nll, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for req in requests:
            (string,) = req.args
            ids = self.tok_encode(string)
            total_nll = 0.0
            total_tokens = 0
            # Slide over the sequence in max_length chunks
            for i in range(0, len(ids), self._max_length - 1):
                chunk = ids[i:i + self._max_length]
                if len(chunk) < 2:
                    break
                input_ids = torch.tensor([chunk], device=self._device)
                labels = input_ids.clone()
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, labels=labels)
                n = len(chunk) - 1
                total_nll += out.loss.item() * n
                total_tokens += n
            results.append(-total_nll)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get('until', [self.tokenizer.eos_token])
            max_new = gen_kwargs.get('max_new_tokens', self.max_gen_toks)
            ids = torch.tensor([self.tok_encode(context)], device=self._device)
            ids = ids[:, -self._max_length:]

            with torch.no_grad():
                for _ in range(max_new):
                    out = self.model(input_ids=ids)
                    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = torch.cat([ids, next_id], dim=1)
                    token = self.tokenizer.decode(next_id[0])
                    if any(token.endswith(u) for u in until):
                        break

            generated = self.tokenizer.decode(ids[0][len(self.tok_encode(context)):],
                                              skip_special_tokens=True)
            results.append(generated)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='gpt2',
                        help='HuggingFace model id for pretrained weights (e.g. gpt2-xl)')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--gqa', action='store_true')
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--n-mha-layers', type=int, default=6)
    parser.add_argument('--num-m-groups', type=int, default=1, choices=[1, 2])
    parser.add_argument('--per-layer-m', action='store_true',
                        help='Checkpoint uses one M per layer instead of one shared M')
    parser.add_argument('--num-kv-groups', type=int, default=2,
                        help='Number of KV groups for --gqa checkpoints (default 2)')
    parser.add_argument('--tasks', type=str, default='lambada_openai,hellaswag,piqa,winogrande')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BLTModel(
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        gqa=args.gqa,
        hybrid=args.hybrid,
        device=device,
        batch_size=args.batch_size,
        num_m_groups=args.num_m_groups,
        n_mha_layers=args.n_mha_layers,
        pretrained=args.pretrained,
        per_layer_m=args.per_layer_m,
        num_kv_groups=args.num_kv_groups,
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size,
    )

    label = ('hybrid' if args.hybrid else 'gqa' if args.gqa else
             'baseline' if args.baseline else 'blt')
    out_path = args.output or f'lm_eval_{label}.json'

    import json
    with open(out_path, 'w') as f:
        json.dump(results['results'], f, indent=2)

    print(f'\n=== {label.upper()} lm-eval results ===')
    for task, res in results['results'].items():
        for metric, val in res.items():
            if isinstance(val, float):
                print(f'  {task}/{metric}: {val:.4f}')
    print(f'\nFull results saved to {out_path}')
