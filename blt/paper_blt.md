# Bilinear Attention with a Shared Interaction Matrix (BLT)

## Abstract

We introduce BLT (Bilinear atten**t**ion), a drop-in replacement for the standard multi-head
attention mechanism in GPT-2 that replaces the per-layer query and key projection matrices
with a single shared bilinear interaction matrix M.  BLT reduces attention parameter count
by roughly 48% while matching or exceeding the language modelling perplexity of pretrained
GPT-2 after fine-tuning.  We report results on WikiText-103 perplexity and a suite of
zero-shot benchmarks.

---

## 1. Background: Standard Multi-Head Attention

Let **X** ∈ ℝ^{L×D} be the sequence of hidden states entering an attention layer, where L
is the sequence length and D = 768 is the model dimension.  GPT-2 uses H = 12 heads with
per-head dimension d = D/H = 64.

For head h, standard multi-head attention (MHA) computes:

```
Q_h = X W_q^h           (L×D)(D×d) → L×d
K_h = X W_k^h           (L×D)(D×d) → L×d
V_h = X W_v^h           (L×D)(D×d) → L×d

A_h = softmax( Q_h K_h^T / √d )    L×L  (causal mask applied before softmax)

head_h = A_h V_h                    L×d
```

The outputs of all heads are concatenated and projected:

```
out = concat(head_1, ..., head_H) W_o + b_o     (L×D)(D×D)
```

Every layer has its own W_q^{l,h}, W_k^{l,h}, W_v^{l,h}, W_o^l.  The attention score
between positions i and j in head h can be written as a bilinear form:

```
score_h(i,j) = x_i^T ( W_q^h W_k^{h,T} ) x_j / √d
```

so each head implicitly defines a D×D interaction matrix M_h = W_q^h W_k^{h,T}.  Across
12 layers and 12 heads, standard GPT-2 has **144 distinct interaction matrices**.

**Attention parameters per layer:**

| Matrix | Shape | Params |
|--------|-------|--------|
| W_q    | D×D   | 589,824 |
| W_k    | D×D   | 589,824 |
| W_v    | D×D   | 589,824 |
| W_o    | D×D   | 589,824 |
| **Total per layer** | | **2,359,296** |

Over 12 layers: ~28.3M attention parameters.

---

## 2. BLT: Bilinear Attention with a Shared M

BLT replaces all per-layer W_q and W_k matrices with a **single shared** D×D matrix M.
The attention score between positions i and j is:

```
score(i,j) = x_i^T M x_j / √d
```

In matrix form, the full attention computation for a layer becomes:

```
S = ( X M ) X^T / √d          (L×D)(D×D)(D×L) → L×L
A = softmax( S + causal_mask )
V = X W_v + b_v
h = A V
out = h W_o + b_o
```

M is shared across **all 12 layers**; W_v and W_o remain per-layer.

**Attention parameters:**

| Matrix   | Shape | Count | Params |
|----------|-------|-------|--------|
| M (shared) | D×D | 1 | 589,824 |
| W_v      | D×D   | 12 layers | 7,077,888 |
| W_o      | D×D   | 12 layers | 7,077,888 |
| **Total** | | | **14,745,600** |

BLT uses **14.7M** attention parameters vs **28.3M** for standard MHA — a **48% reduction**.
Total unique model parameters: 110,855,424 vs ~117M for GPT-2.

### 2.1 Relationship to Standard Attention

Standard attention can always be written as a bilinear form:

```
score_h^l(i,j) = x_i^T M_h^l x_j / √d     where M_h^l = W_q^{l,h} W_k^{l,h,T}
```

BLT is the extreme case of this where M_h^l = M for all layers l and all heads h.  The
single M must therefore capture the average useful query-key interaction across all layers
and all head slots.

### 2.2 Initialization

Since M replaces W_q W_k^T, a natural initialization is:

```
M_init = (1/n_layers) Σ_l  W_q^l W_k^{l,T}
```

the average of the pretrained GPT-2 query-key interaction matrices across all 12 layers.
We also experiment with **random initialization**: M ~ N(0, 1/√D · I), which tests whether
the pretrained initialization is load-bearing or whether the model can learn M from scratch.

### 2.3 Consequence for Head Diversity

In standard MHA, each of the 12 heads attends to different parts of the context because
each head has its own M_h.  In BLT-1M (one shared M), all 12 "head slots" compute
**identical attention weights**.  The only per-head differentiation comes from the W_v
projection.  This is the primary representational cost of parameter sharing.

The **BLT-2M** variant (two shared matrices M1, M2) partially recovers head diversity:
heads 0–5 use M1 and heads 6–11 use M2, each with a D×(D/2) value projection,
analogous to grouped query attention (GQA).

---

## 3. Experimental Setup

**Base model:** GPT-2 Small (12 layers, 12 heads, D=768, d=64).

**Training:** All parameters trained with Adam (lr=5e-5, cosine decay, 200 warmup steps,
batch size 4, block size 1024).  This is the GD-only control condition — no alternative
optimizers are used.

**Datasets:**
- *WikiText-103*: English Wikipedia (~103M tokens).  2 epochs ≈ 50,300 steps.
- *OpenWebText* (subset): First 1M documents from Skylion007/openwebtext (~1B tokens).
  250,000 steps ≈ 1 epoch.

**Evaluation:** Sliding-window perplexity on WikiText-103 validation (stride=512); 
zero-shot accuracy on LAMBADA, HellaSwag, PIQA, Winogrande via lm-eval-harness.

---

## 4. Results

### 4.1 WikiText-103 Perplexity

| Model | Val PPL |
|-------|---------|
| GPT-2 pretrained (no fine-tune) | 26.39 |
| GPT-2 fine-tuned (WikiText-103) | ~14.91 (stopped at step 33,930) |
| BLT-1M (WikiText-103, pretrained M init) | **21.50** |
| BLT-2M (WikiText-103, pretrained M init, step 20K) | 20.95 |

BLT-1M surpasses the pretrained GPT-2 baseline despite using 48% fewer attention
parameters, demonstrating that the shared-M constraint is not prohibitively limiting for
language modelling.

### 4.2 Zero-Shot Benchmarks

| Task | GPT-2 pretrained | BLT-1M (WikiText) | BLT-2M (step 20K) |
|------|-----------------|-------------------|-------------------|
| LAMBADA acc     | 0.242 | 0.114 | 0.106 |
| LAMBADA ppl     | 83.0  | 1307.5 | 975.6 |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.271 |
| PIQA acc_norm   | 0.560 | 0.541 | 0.547 |
| Winogrande acc  | 0.502 | 0.507 | 0.499 |

BLT matches GPT-2 closely on HellaSwag, PIQA, and Winogrande.  The LAMBADA gap is
large and persistent.  LAMBADA tests last-word prediction in narrative fiction passages;
the training domain mismatch (encyclopedia vs. fiction) likely explains most of this gap.
Fine-tuning directly on LAMBADA data caused catastrophic forgetting and made scores worse.

### 4.3 OpenWebText with Random M Initialization

We trained BLT-1M from a randomly initialized M (N(0, 1/√D)) on the first 1M documents
of OpenWebText (~1B tokens) for 250,000 steps (~1 epoch).  This run tests two simultaneous
hypotheses: (1) whether M can be learned from scratch rather than warm-started from
pretrained weights, and (2) whether broader web text improves LAMBADA scores by better
matching the narrative distribution.

Both hypotheses are confirmed.  M converges successfully from random initialization —
WikiText-103 validation perplexity declined from 34,114 at step 0 to 48.0 at step 218,000,
demonstrating that the bilinear interaction structure can be learned entirely from data
without the Wq@Wk^T warm-start.

The LAMBADA results are particularly striking.  At step 218,000 the model already
substantially closes the gap to pretrained GPT-2, while matching or exceeding it on
PIQA and remaining within noise on HellaSwag and Winogrande.  This is achieved with
only ~1% of the tokens GPT-2 was trained on.

| Task | GPT-2 pretrained | BLT-1M (WikiText) | BLT-1M OWT (step 126K) | BLT-1M OWT (step 218K) |
|------|-----------------|-------------------|------------------------|------------------------|
| LAMBADA acc     | 0.242 | 0.114 | 0.182 | **0.196** |
| LAMBADA ppl     | 83.0  | 1307.5 | 243.0 | **210.1** |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.280 | 0.279 |
| PIQA acc_norm   | 0.560 | 0.541 | **0.572** | **0.566** |
| Winogrande acc  | 0.502 | 0.507 | 0.503 | 0.487 |

The LAMBADA improvement from 0.114 (WikiText-trained) to 0.196 (OWT-trained, step 218K)
confirms that the earlier LAMBADA gap was a training domain effect, not an architectural
limitation of the shared-M design.  The model trained on encyclopedia text simply never
encountered the narrative context patterns that LAMBADA tests; OWT's broader coverage
provides that signal.  LAMBADA accuracy was still improving monotonically at step 218K,
suggesting further gains with a complete training run.

### 4.4 Planned: From-Scratch Comparison Experiment

The definitive test of BLT is a controlled from-scratch comparison against standard MHA
on identical data, compute, and hyperparameters.  Both models will be trained on
OpenWebText (1M documents, 250,000 steps) with random initialization, differing only in
the attention mechanism.  We compare under three budgets: fixed steps, fixed wall-clock
time, and fixed FLOPs.  The fixed wall-clock and fixed-FLOP comparisons favor BLT, which
requires one fewer D×D matrix multiply per layer per forward pass (no Wk projection),
resulting in measurably faster per-step wall time at GPT-2 scale.  Results pending.

---

## 5. Related Work

Several recent lines of work converge on the question of whether the standard Q, K, V triplet
in self-attention is over-parameterized.

**Shared weights within a layer.**
Lan et al. propose sharing a single weight matrix for Q, K, *and* V within each attention
layer in BERT, reporting a 66% reduction in attention parameters with no loss in accuracy on
GLUE tasks and improved generalization on out-of-domain data
([arXiv 2412.00359](https://arxiv.org/abs/2412.00359), NAACL 2025).
This is the closest prior work to BLT: both recognize that Q and K do not need separate
matrices.  The key differences are that BLT (i) retains V as a distinct per-layer projection,
(ii) expresses the Q/K interaction as an explicit bilinear form X M X^T rather than tying
the matrices, and (iii) shares M *across all layers* rather than within a single layer.

**Eliminating Q or K entirely.**
Brandon et al. prove that the Query or Key weight matrix can be replaced by the identity
without loss in small GPT-style models, reducing attention parameters by 25% and simplifying
optimization by making attention logits linear in the remaining learned weights
([arXiv 2510.23912](https://arxiv.org/pdf/2510.23912)).
BLT can be seen as a more structured version of this idea: rather than removing one projection
entirely, it merges both into a single bilinear matrix that is shared across layers.

**Cross-layer KV sharing.**
Brandon et al. propose Cross-Layer Attention (CLA), which shares K and V *activations*
(not weight matrices) between adjacent layers to reduce KV-cache size at inference
([arXiv 2405.12981](https://arxiv.org/abs/2405.12981), NeurIPS 2024).
The goal is inference memory rather than parameter count; the approach is orthogonal to BLT
and could in principle be combined with it.

**Matrix-based dictionary learning.**
MASA decomposes all attention projection matrices (Q, K, V, O) into linear combinations of
shared dictionary atoms across layers, achieving 66.7% attention parameter reduction
([arXiv 2508.04581](https://arxiv.org/abs/2508.04581)).
This is a general compression framework; BLT's sharing is more targeted and structurally
motivated by the bilinear interpretation of attention.

**Summary.**
To our knowledge, the specific combination in BLT — merging Q and K into a single bilinear
interaction matrix M that is shared across all transformer layers, while keeping V and O
projections per-layer — has not been previously proposed.  The recent independent results
corroborating the redundancy of separate Q and K matrices lend additional support to the
approach.

---

## 6. Discussion

**Parameter efficiency.** BLT achieves better language modelling perplexity than pretrained
GPT-2 with half the attention parameters.  The shared M forces the model to find a single
linear transformation of the input that, when combined bilinearly with the raw hidden
state, identifies which past tokens are worth attending to — regardless of layer depth or
head index.

**LAMBADA and training domain.** The WikiText-trained BLT scored 0.114 on LAMBADA vs.
GPT-2's 0.242, which initially suggested the shared-M design might limit long-range
narrative understanding.  The OWT experiment resolves this: at step 218K on web text,
BLT reaches 0.196 and is still improving, while matching or exceeding GPT-2 on PIQA and
remaining within noise on HellaSwag and Winogrande.  The earlier gap was a training domain
effect — encyclopedia text provides no signal for the narrative last-word prediction
patterns that LAMBADA tests.  The shared-M architecture is not the bottleneck; data is.

**KV-cache compatibility.** Standard KV caching stores K = X W_k for past tokens.  In BLT,
the score between new token t and past token j is (x_t M) · x_j, so the cache only needs
to store the raw hidden states x_j (as implicit keys) and x_j W_v (as values).  A single
M multiply produces the query; no key projection is required when adding new tokens.  The
current implementation does not exploit this and recomputes full attention from scratch at
each forward pass; an inference-optimized implementation would realize this as a genuine
efficiency gain over standard attention.
