"""
Local playground: pick a trained checkpoint from the registry below and chat with it.

These are all BASE language models (next-token prediction trained on OpenWebText,
not instruction-tuned) -- there's no "assistant" behavior built in. By default the UI
primes the model with a persona/system prompt and formats turns as "Human: .../AI: ..."
(a prompting trick, not real instruction-tuning -- GPT-2 saw plenty of dialogue-shaped
text during pretraining, so this nudges it toward turn-taking Q&A, but expect it to be
unreliable, especially on the small 124M models). Flip "chat mode" off in the UI to get
the model's raw, unprompted free-association behavior instead -- it just appends your
message to a running transcript and the model continues it, with no persona framing.
That raw mode is what these checkpoints were actually benchmarked on.

Runs on CPU by default so it never contends with any training job that might be using
the GPU (see --device to override, but don't do that while a training job is running on
the same machine -- this project has repeatedly found concurrent GPU jobs on this
hardware class can crash one of them with "unspecified launch failure").

Usage:
    conda run -n blt python chatbot.py
    # then open http://127.0.0.1:5050 in a browser
"""

import argparse
import threading

import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

from model import build_blt_model, build_gqa_model, build_hybrid_model, build_blt_lowrank_model

REPO = '/home/benzene/llmopt/blt'


def _baseline(pretrained='gpt2'):
    return GPT2LMHeadModel(GPT2Config.from_pretrained(pretrained))


# key -> {path, build (skeleton factory, weights get overwritten by the checkpoint),
#         desc (one line, sourced from CLAUDE.md's own documented results)}
MODELS = {
    'baseline-small': {
        'path': 'run_gpt2_baseline_seed42.pt',
        'build': lambda: _baseline('gpt2'),
        'desc': 'Standard GPT-2 small (124M), from-scratch on OpenWebText. OWT ppl 27.78.',
    },
    'blt-small': {
        'path': 'run_blt_scratch_seed42.pt',
        'build': lambda: build_blt_model(pretrained='gpt2', from_scratch=True),
        'desc': 'Full-rank BLT (one shared 768x768 M replacing Wq/Wk), from-scratch OWT. OWT ppl 31.05.',
    },
    'gqa2-small': {
        'path': 'run_gqa_scratch_seed42.pt',
        'build': lambda: build_gqa_model(n_kv=2, pretrained='gpt2', from_scratch=True),
        'desc': 'GQA 2-group, from-scratch OWT. OWT ppl 27.64 -- edges out even the baseline.',
    },
    'gqa3-small': {
        'path': 'run_gqa3_scratch_seed42.pt',
        'build': lambda: build_gqa_model(n_kv=3, pretrained='gpt2', from_scratch=True),
        'desc': 'GQA 3-group, from-scratch OWT. OWT ppl 27.46, best of the whole GQA family.',
    },
    'hybrid-small': {
        'path': 'run_hybrid_mha6_scratch_seed42.pt',
        'build': lambda: build_hybrid_model(n_mha=6, pretrained='gpt2'),
        'desc': '6 standard MHA layers + 6 BLT layers, from-scratch OWT. Best LAMBADA ppl in the whole BLT family.',
    },
    'ema-small': {
        'path': 'run_gpt2_ema_seed42.pt',
        'build': lambda: _baseline('gpt2'),
        'desc': 'Fixed-decay EMA loss weighting (blend=1.0): upweights historically-hard tokens. LAMBADA acc 0.253 vs 0.225 baseline.',
    },
    'ema-blend75-small': {
        'path': 'run_gpt2_ema_blend75_scratch_seed7_v2.pt',
        'build': lambda: _baseline('gpt2'),
        'desc': 'Fixed-decay EMA, 75/25 blend toward uniform CE, jointly trained from scratch. LAMBADA acc 0.257.',
    },
    'cumulative-small': {
        'path': 'run_gpt2_cumulative_scratch_seed42.pt',
        'build': lambda: _baseline('gpt2'),
        'desc': 'Cumulative (exact running-mean) loss weighting, blend=1.0. LAMBADA acc 0.268, best small-model LAMBADA in this project.',
    },
    'cumulative-blend75-small': {
        'path': 'run_gpt2_cumulative_blend75_scratch_seed42.pt',
        'build': lambda: _baseline('gpt2'),
        'desc': 'Cumulative loss weighting, blend=0.75. Best OWT held-out ppl of the whole reweighted family.',
    },
    'baseline-medium': {
        'path': 'run_gpt2_medium_baseline_seed42.pt',
        'build': lambda: _baseline('gpt2-medium'),
        'desc': 'Standard GPT-2 medium (355M), from-scratch OWT, 1.5M steps. OWT ppl 18.09. Noticeably more fluent than the small models.',
    },
    'ema-sine-blend75-medium': {
        'path': 'run_gpt2_medium_ema_blend75_sine_seed42.pt',
        'build': lambda: _baseline('gpt2-medium'),
        'desc': 'Medium-scale, sine-annealed EMA loss weighting, 75% blend. Best medium-scale LAMBADA acc on record, 0.373.',
    },
    'cumulative-medium': {
        'path': 'run_gpt2_medium_cumulative_seed42.pt',
        'build': lambda: _baseline('gpt2-medium'),
        'desc': 'Medium-scale cumulative loss weighting, blend=1.0. LAMBADA acc 0.381, best in the whole medium-scale family.',
    },
    'uv-groups8-medium': {
        'path': 'run_gpt2_medium_uvgroups8_r256_scratch_seed42.pt',
        'build': lambda: build_blt_lowrank_model(rank=256, pretrained='gpt2-medium', from_scratch=True, num_uv_groups=8),
        'desc': 'Low-rank BLT, 8 (U,V) groups at rank 256, medium scale, from-scratch. Near-baseline quality from a TP-friendly, bandwidth-efficient design.',
    },
}

_lock = threading.Lock()
_cache = {}  # key -> (model, device)
_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_model(key, device):
    if key not in MODELS:
        raise ValueError(f'unknown model key: {key}')
    cached = _cache.get(key)
    if cached is not None:
        return cached
    spec = MODELS[key]
    model = spec['build']()
    ckpt = torch.load(f"{REPO}/{spec['path']}", map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    _cache[key] = model
    return model


def _tokenize_or_empty(text, device):
    if not text:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return _tokenizer(text, return_tensors='pt').input_ids.to(device)


@torch.no_grad()
def generate(model, device, system_text, convo_text, max_new_tokens, temperature, top_k,
             stop_strings=None):
    # Full recompute each step (no KV cache) -- simplest thing that's correct across
    # every architecture variant here, several of which have their own bespoke
    # attention classes with no shared caching convention.

    # GPT-2's absolute position embeddings cap the whole sequence (system prompt +
    # conversation + generated tokens) at n_positions (1024 for every checkpoint in
    # this registry, small or medium) -- feeding it a longer sequence raises an
    # index-out-of-range error deep inside the embedding lookup. Reserve room for
    # max_new_tokens and the (fixed, never-trimmed) system prompt, and silently drop
    # the oldest conversation turns (a sliding window) rather than let that happen.
    block_size = model.config.n_positions
    max_new_tokens = min(max_new_tokens, block_size - 1)

    sys_ids = _tokenize_or_empty(system_text, device)
    context_trimmed = False
    budget = max(block_size - max_new_tokens - sys_ids.shape[1], 0)
    convo_ids = _tokenize_or_empty(convo_text, device)
    if convo_ids.shape[1] > budget:
        convo_ids = convo_ids[:, -budget:]
        context_trimmed = True

    ids = torch.cat([sys_ids, convo_ids], dim=1)
    start_len = ids.shape[1]
    hit_stop = False
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :] / max(temperature, 1e-5)
        topk_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits[logits < topk_vals[:, -1:]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == _tokenizer.eos_token_id:
            hit_stop = True
            break
        if stop_strings:
            tail_text = _tokenizer.decode(ids[0, start_len:], skip_special_tokens=True)
            if any(s in tail_text for s in stop_strings):
                hit_stop = True
                break
    text = _tokenizer.decode(ids[0, start_len:], skip_special_tokens=True)
    if stop_strings:
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
    # truncated = the token budget ran out before the model chose to stop on its own --
    # distinct from hitting EOS/a stop string, which means the model (or our own
    # turn-boundary heuristic in chat mode) considers the thought finished.
    truncated = not hit_stop
    return text, truncated, context_trimmed


app = Flask(__name__)


@app.route('/api/models')
def api_models():
    return jsonify({k: v['desc'] for k, v in MODELS.items()})


@app.route('/api/info')
def api_info():
    return jsonify({'device': str(app.config['DEVICE'])})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    key = data['model']
    transcript = data.get('transcript', '')
    message = data.get('message', '')
    max_new_tokens = int(data.get('max_new_tokens', 150))
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 50))
    chat_mode = bool(data.get('chat_mode', False))

    if chat_mode:
        system_prompt = data.get('system_prompt', '')
        convo_text = transcript + (f"Human: {message}\nAI:" if message else '')
        # Not '\nHuman:' -- a degenerating model sometimes glues the next turn
        # straight onto the prior punctuation with no newline at all, which that
        # stricter match would silently miss and run to the token budget instead.
        stop_strings = ['Human:']
    else:
        system_prompt = ''
        convo_text = transcript + message
        stop_strings = None

    with _lock:
        model = get_model(key, app.config['DEVICE'])
        reply, truncated, context_trimmed = generate(
            model, app.config['DEVICE'], system_prompt, convo_text,
            max_new_tokens, temperature, top_k, stop_strings=stop_strings)

    new_transcript = convo_text + reply
    if chat_mode and not truncated:
        new_transcript += '\n'  # separates this turn from the next "Human:" line
    return jsonify({'reply': reply, 'transcript': new_transcript,
                    'truncated': truncated, 'context_trimmed': context_trimmed})


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>BLT model playground</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 760px; margin: 2em auto; padding: 0 1em; }
  select, textarea, input, button { font-size: 1em; }
  #modelDesc { color: #555; font-size: 0.9em; margin: 0.5em 0 1.5em; min-height: 2.4em; }
  #log { border: 1px solid #ccc; border-radius: 6px; padding: 1em; min-height: 300px;
         white-space: pre-wrap; font-family: Menlo, monospace; font-size: 0.92em; margin-bottom: 1em; }
  .user { color: #0a5; font-weight: bold; }
  .model { color: #06c; }
  .truncNote { color: #a60; font-style: italic; }
  #row { display: flex; gap: 0.5em; }
  #msg { flex: 1; padding: 0.5em; }
  .controls { display: flex; gap: 1em; align-items: center; margin-bottom: 0.3em; font-size: 0.85em; color: #555; flex-wrap: wrap; }
  .controls input { width: 4.5em; }
  .hint { font-size: 0.8em; color: #888; margin-bottom: 1em; }
  #status { color: #a60; font-size: 0.85em; height: 1.2em; }
  #continueBtn { display: none; }
  #sysPromptRow { display: none; margin-bottom: 0.8em; }
  #sysPrompt { width: 100%; box-sizing: border-box; font-family: Menlo, monospace; font-size: 0.82em;
               padding: 0.5em; border-radius: 6px; border: 1px solid #ccc; resize: vertical; }
</style>
</head>
<body>
<h2>BLT model playground <span id="deviceTag" style="font-size:0.5em; color:#888;"></span></h2>
<label>Model: <select id="model"></select></label>
<div id="modelDesc"></div>

<div class="controls">
  <label><input id="chatMode" type="checkbox" checked> Chat mode (persona prompt, stops at next turn)</label>
  <label>max new tokens <input id="maxTok" type="number" value="150" min="1" max="2000"></label>
  <label>temperature <input id="temp" type="number" value="0.8" step="0.1"></label>
  <label>top-k <input id="topk" type="number" value="50"></label>
  <button id="resetBtn">New conversation</button>
</div>
<div id="sysPromptRow">
  <label for="sysPrompt" style="font-size:0.85em; color:#555;">System / persona prompt (fixed context, always kept even if the conversation gets trimmed):</label>
  <textarea id="sysPrompt" rows="7"></textarea>
</div>
<div class="hint">Chat mode wraps each turn as "Human: ... / AI: ..." and primes the model with the system prompt above, and stops generation as soon as it tries to start the next "Human:" turn itself -- a prompting trick, not real instruction-tuning, so expect it to be unreliable and occasionally ignore the persona (more so on the small 124M models than the 355M medium ones). Turn chat mode off to get the model's raw, unprompted free-association behavior instead (what these checkpoints were actually benchmarked on).
<br>Generation stops early if the model reaches a natural endpoint (EOS, or the next-turn boundary in chat mode); otherwise it's cut off once "max new tokens" is reached -- raise that number for longer replies. A cut-off reply gets a "(truncated)" note and a Continue button to keep going from where it left off.
<br>Separately, every model here has a hard 1024-token context window (system prompt + conversation so far + generated tokens combined) -- it's a fixed property of the GPT-2 architecture, not something you can raise. If a long conversation runs past it, the oldest turns are silently dropped to make room (the system prompt itself is never dropped), and you'll see a "(dropped older context)" note when that happens.</div>

<div id="log"></div>
<div id="status"></div>
<div id="row">
  <input id="msg" placeholder="Type a sentence for the model to continue..." autofocus>
  <button id="sendBtn">Send</button>
  <button id="continueBtn">Continue</button>
</div>

<script>
let models = {};
let transcript = '';

const DEFAULT_SYS_PROMPT = `The following is a transcript of a conversation between a helpful, knowledgeable AI assistant and a human. The assistant answers questions directly, explains things clearly, and admits when it does not know something.

Human: What is the capital of France?
AI: The capital of France is Paris.

Human: What causes rain?
AI: Rain forms when water vapor in clouds condenses into droplets that become heavy enough to fall to the ground.

`;

function updateSysPromptVisibility() {
  document.getElementById('sysPromptRow').style.display =
    document.getElementById('chatMode').checked ? 'block' : 'none';
}

async function loadModels() {
  const r = await fetch('/api/models');
  models = await r.json();
  const sel = document.getElementById('model');
  for (const k in models) {
    const opt = document.createElement('option');
    opt.value = k; opt.textContent = k;
    sel.appendChild(opt);
  }
  updateDesc();
  sel.addEventListener('change', () => { updateDesc(); resetConversation(); });
}

function updateDesc() {
  const k = document.getElementById('model').value;
  document.getElementById('modelDesc').textContent = models[k] || '';
}

function resetConversation() {
  transcript = '';
  document.getElementById('log').textContent = '';
  document.getElementById('continueBtn').style.display = 'none';
}

function appendLog(cls, text) {
  const log = document.getElementById('log');
  const span = document.createElement('span');
  span.className = cls;
  span.textContent = text;
  log.appendChild(span);
  log.scrollTop = log.scrollHeight;
}

async function loadInfo() {
  const r = await fetch('/api/info');
  const info = await r.json();
  document.getElementById('deviceTag').textContent = '(' + info.device + ')';
}

async function generateFrom(message, isContinue) {
  document.getElementById('status').textContent = 'generating...';
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('continueBtn').disabled = true;

  const body = {
    model: document.getElementById('model').value,
    transcript: transcript,
    message: message,
    max_new_tokens: document.getElementById('maxTok').value,
    temperature: document.getElementById('temp').value,
    top_k: document.getElementById('topk').value,
    chat_mode: document.getElementById('chatMode').checked,
    system_prompt: document.getElementById('sysPrompt').value,
  };
  try {
    const r = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const data = await r.json();
    appendLog('model', data.reply);
    transcript = data.transcript;
    const continueBtn = document.getElementById('continueBtn');
    if (data.context_trimmed) {
      appendLog('truncNote', '  (dropped older context -- conversation exceeded the 1024-token window)\\n');
    }
    if (data.truncated) {
      appendLog('truncNote', '  (truncated at ' + document.getElementById('maxTok').value + ' tokens -- click Continue to keep going)\\n');
      transcript += '\\n';
      continueBtn.style.display = 'inline';
    } else {
      appendLog('model', '\\n');
      transcript += '\\n';
      continueBtn.style.display = 'none';
    }
  } catch (e) {
    document.getElementById('status').textContent = 'error: ' + e;
  } finally {
    document.getElementById('status').textContent = '';
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('continueBtn').disabled = false;
  }
}

async function send() {
  const msgBox = document.getElementById('msg');
  const message = msgBox.value;
  if (!message.trim()) return;
  msgBox.value = '';
  appendLog('user', message);
  await generateFrom(message, false);
}

async function continueGenerating() {
  document.getElementById('continueBtn').style.display = 'none';
  await generateFrom('', true);
}

document.getElementById('sendBtn').addEventListener('click', send);
document.getElementById('continueBtn').addEventListener('click', continueGenerating);
document.getElementById('msg').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
document.getElementById('resetBtn').addEventListener('click', resetConversation);
document.getElementById('chatMode').addEventListener('change', () => { updateSysPromptVisibility(); resetConversation(); });
document.getElementById('sysPrompt').value = DEFAULT_SYS_PROMPT;
updateSysPromptVisibility();
loadModels();
loadInfo();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(PAGE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help="Bind address. Default is localhost-only. Pass 0.0.0.0 to make it "
                             "reachable from other machines on the network -- this dev server has "
                             "no authentication, so anything that can reach this machine's IP on "
                             "this port can use it.")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Defaults to cpu so this never contends with a GPU training job "
                             "on the same machine. Pass 'cuda' only if you know the GPU is free.")
    args = parser.parse_args()
    app.config['DEVICE'] = torch.device(args.device)
    print(f"Serving on http://{args.host}:{args.port} (device={args.device})")
    app.run(host=args.host, port=args.port, threaded=False)
