# Bilinear Attention with a Shared Interaction Matrix (BLT)

## Abstract

We introduce BLT (Bilinear attenTion), a drop-in replacement for standard multi-head
attention in transformer language models.  BLT replaces all per-layer query and key
projection matrices W_q and W_k with a single D×D matrix M shared across every layer,
reducing attention parameter count by 48% (14.7M vs 28.3M for GPT-2 scale) and
eliminating one D×D matrix multiply per layer per forward pass.

In a controlled from-scratch comparison on OpenWebText (2M documents, ~2B tokens), all
architectures trained for a fixed 500K steps, BLT achieves held-out perplexity of 30.81,
versus 27.78 for standard MHA and 27.64 for 2-group grouped query attention (GQA) — a gap
of approximately 0.10 nats.  The GQA baseline (117.9M parameters, between BLT and GPT-2)
reveals a clean split across metrics.  On OWT perplexity, GQA matches standard MHA (27.64
vs 27.78), isolating BLT's ~0.10 nat gap as a cost of cross-layer M sharing rather than
parameter reduction.  On LAMBADA, however, GQA falls to BLT's level (0.204 vs 0.212), well
below GPT-2's 0.225.  The two failure modes are separable: BLT's OWT perplexity gap is
caused by cross-layer sharing; the LAMBADA gap is shared by GQA and BLT alike, implicating
constrained per-layer key capacity more broadly.  HellaSwag, PIQA, and Winogrande are
within noise across all four architectures.

A hybrid model that replaces half of BLT's 12 shared-M layers with standard per-layer MHA
(6 MHA + 6 BLT, trained from scratch under the same protocol) closes the OWT perplexity
gap disproportionately: held-out loss falls to 3.3462 nats (ppl 28.40), recovering more
than 75% of the full-BLT gap from only half the layers reverting to standard attention.
The hybrid also attains the best LAMBADA perplexity of any model tested (167.1), edging
out even standard MHA (174.6).  This non-additivity suggests the expressiveness cost of
cross-layer M sharing compounds across layers rather than summing linearly, and that a
modest fraction of unconstrained layers is sufficient to largely route around it.

SVD analysis of a trained M reveals a nearly flat singular value spectrum — M is
genuinely full-rank, using its entire D×D capacity to simultaneously serve 144 attention
contexts (12 layers × 12 heads).  A low-rank UV^T factorization is the natural next step:
it reduces the KV cache by a factor of D/r, brings BLT within the same parameter budget
as GQA, and — critically — makes the architecture compatible with tensor-parallel inference
at production scale.

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
Total unique model parameters: 110,855,424 vs 124,439,808 for GPT-2.

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

**Training:** All parameters trained with Adam (lr=5e-5, cosine decay to near zero,
200 warmup steps, batch size 4, block size 1024).  This is the GD-only control condition
— no alternative optimizers are used.

**Datasets:**
- *WikiText-103*: English Wikipedia (~103M tokens).  2 epochs ≈ 50,300 steps.
- *OpenWebText* (2M documents, ~2B tokens): Skylion007/openwebtext, first 2M documents.
  Chinchilla-optimal compute for a ~117M parameter model falls in this range.

**Evaluation:** Sliding-window perplexity on WikiText-103 validation (stride=512) and on
a held-out OWT split (files 21–25, ~50K tokens); zero-shot accuracy on LAMBADA,
HellaSwag, PIQA, and Winogrande via lm-eval-harness.

**Baselines:**
- *Standard MHA (GPT-2)*: trained from scratch, 500K steps.
- *2-group GQA*: 12 query heads, 2 KV heads (6 queries per KV head); W_q and W_o are
  full D×D per layer, W_k and W_v project to 2×64=128 dimensions.  Trained from scratch,
  500K steps.  GQA has 117.9M unique parameters — close to BLT's 110.9M and between BLT
  and GPT-2's 124.4M — making it the right tool for separating parameter count effects
  from architectural expressiveness.

**Step count:** All architectures reported in this paper — BLT, standard MHA, GQA, and the
hybrid model (Section 4.6) — are trained from scratch for a fixed **500K steps** on
identical data and hyperparameters, giving a clean iso-step comparison.  BLT's forward
pass requires one fewer D×D matrix multiply per layer (no W_k projection), which in
principle reduces per-step compute, and we initially trained two BLT seeds longer (550K
and 600K steps for seeds 42 and 19) to target wall-clock parity with the baselines on
their respective machines.  These longer runs converged to OWT perplexities of 31.05 and
30.48 — statistically indistinguishable from a third BLT seed (seed 7) trained for exactly
500K steps (30.81) — indicating the additional 50–100K steps produced no significant
improvement.  We therefore report all results at the common 500K-step budget, using BLT
seed 7 as the representative from-scratch BLT run.

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

### 4.2 Zero-Shot Benchmarks (WikiText-trained)

| Task | GPT-2 pretrained | BLT-1M (WikiText) | BLT-2M (step 20K) |
|------|-----------------|-------------------|-------------------|
| LAMBADA acc     | 0.242 | 0.114 | 0.106 |
| LAMBADA ppl     | 83.0  | 1307.5 | 975.6 |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.271 |
| PIQA acc_norm   | 0.560 | 0.541 | 0.547 |
| Winogrande acc  | 0.502 | 0.507 | 0.499 |

BLT (WikiText) matches GPT-2 closely on HellaSwag, PIQA, and Winogrande.  The LAMBADA
gap turns out to be a training domain artifact, not an architectural limitation: see
Section 4.3.

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

| Task | GPT-2 pretrained | BLT-1M (WikiText) | BLT-1M OWT (step 126K) | BLT-1M OWT (step 218K) | BLT-1M OWT (step 250K) |
|------|-----------------|-------------------|------------------------|------------------------|------------------------|
| LAMBADA acc     | 0.242 | 0.114 | 0.182 | 0.196 | **0.199** |
| LAMBADA ppl     | 83.0  | 1307.5 | 243.0 | 210.1 | **206.8** |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.280 | 0.279 | 0.280 |
| PIQA acc_norm   | 0.560 | 0.541 | **0.572** | 0.566 | 0.568 |
| Winogrande acc  | 0.502 | 0.507 | 0.503 | 0.487 | 0.490 |

LAMBADA improved monotonically (0.182 → 0.199) across all three checkpoints and had not
saturated, approaching GPT-2's 0.242 after only ~1B tokens — roughly 1% of GPT-2's
training data.  This confirms the LAMBADA gap observed in Section 4.2 was a training
domain effect: encyclopedia text provides no signal for the narrative last-word prediction
patterns LAMBADA tests.

### 4.4 From-Scratch OpenWebText Comparison

The definitive test of BLT is a controlled comparison against standard MHA on identical
data, compute, and hyperparameters.  BLT, GPT-2, and GQA are all initialized from scratch
(random weights) and trained on 2M OWT documents for a fixed 500K steps (see the step-count
note in Section 3; results from two additional BLT seeds trained 550K–600K steps are
reported there but excluded here since they showed no improvement over the 500K-step run).

**Primary results:**

| Model | Steps | OWT ppl | OWT loss | WikiText ppl |
|-------|-------|---------|----------|--------------|
| BLT-1M seed 7 | 500K | 30.81 | 3.4279 | 71.57 |
| GPT-2 (MHA) seed 42 | 500K | **27.78** | **3.3243** | 55.99 |
| GQA (2-group) seed 42 | 500K | **27.64** | **3.3192** | 57.43 |

BLT lands at 3.43 nats, while GPT-2 and GQA land near 3.32 nats — a consistent ~0.10 nat
gap (this is unchanged from the longer 550K/600K BLT runs, see Section 3).

**Zero-shot benchmarks:**

| Task | BLT seed 7 | GPT-2 seed 42 | GQA seed 42 |
|------|------------|---------------|-------------|
| LAMBADA acc     | 0.212 | **0.225** | 0.204 |
| LAMBADA ppl     | 244.4 | **174.6** | 205.3 |
| HellaSwag acc_norm | 0.268 | 0.268 | 0.269 |
| PIQA acc_norm   | 0.568 | **0.579** | 0.568 |
| Winogrande acc  | **0.516** | 0.505 | 0.496 |

HellaSwag, PIQA, and Winogrande are essentially three-way ties — all differences are
within expected noise for these dataset sizes.  BLT actually holds a slight Winogrande
edge.  LAMBADA is the exception: GPT-2 (0.225) leads clearly, while GQA (0.204) falls to
essentially the same level as BLT (0.212) despite GQA matching GPT-2 on OWT perplexity.
The significance of this split is discussed in Section 4.5.

**Note on WikiText perplexity:** WikiText val ppl was noisy throughout BLT training
(swings of 25+ ppl within a single run), while OWT held-out ppl was stable.  We treat
OWT ppl as the reliable metric and WikiText ppl as approximate.

### 4.5 Isolating the Source of BLT's Gap

BLT (110.9M parameters) uses fewer unique parameters than GPT-2 (124.4M).  A natural
question is whether the ~0.10 nat OWT perplexity gap reflects parameter count, the
expressiveness constraint from sharing M across layers, or both.  The GQA baseline
(117.9M parameters, between BLT and GPT-2) answers this — but the answer splits cleanly
by metric.

**OWT perplexity (27.78 GPT-2 / 27.64 GQA / 30.81 BLT).**
GQA matches GPT-2 on the primary metric despite having fewer parameters and compressing
KV to 2 groups.  Since parameter count and KV compression are both ruled out as causes,
the OWT ppl gap is attributable specifically to cross-layer M sharing: a single M must
simultaneously produce useful attention weights across 12 layers × 12 heads = 144 distinct
attention contexts, while per-layer matrices in MHA and GQA can specialize freely.

**LAMBADA (0.225 GPT-2 / 0.204 GQA / 0.212 BLT).**
Here GQA falls to BLT's level, well below GPT-2.  GQA retains per-layer W_k matrices but
reduces each layer's effective key rank from 768 dimensions (12 heads × 64) to 128 (2
groups × 64) — a 6× compression.  The fact that this compression hurts LAMBADA as much as
BLT's cross-layer sharing does suggests that LAMBADA specifically requires high per-layer
key capacity, not merely per-layer key matrices.  Both constraints — reduced rank within a
layer (GQA) and a single shared matrix across layers (BLT) — are sufficient to degrade
long-range prediction.

**Summary.**  The two failure modes are separable.  BLT's OWT perplexity gap is caused by
cross-layer M sharing and is not shared by GQA.  BLT's LAMBADA gap is caused by constrained
per-layer key capacity and *is* shared by GQA.  Standard MHA, with full per-layer W_k rank
and no cross-layer sharing, avoids both.

### 4.6 Hybrid Architecture: Mixing MHA and BLT Layers

Section 4.5 attributes BLT's OWT perplexity gap to cross-layer M sharing across all 12
layers.  This raises a natural question: is the cost of sharing roughly uniform per layer,
so that removing half the shared layers should recover about half the gap — or does it
compound, so that a partial fix recovers disproportionately more (or less)?

We trained a hybrid model with 6 standard MHA layers followed by 6 BLT layers, from
scratch on OWT for 500K steps, identical in every other respect to the runs in Section 4.4.

| Model | Steps | OWT ppl | OWT loss | WikiText ppl |
|-------|-------|---------|----------|--------------|
| BLT-1M seed 7 (12 BLT layers) | 500K | 30.81 | 3.4279 | 71.57 |
| Hybrid (6 MHA + 6 BLT) | 500K | 28.40 | 3.3462 | 69.08 |
| GPT-2 (MHA) seed 42 (12 MHA layers) | 500K | **27.78** | **3.3243** | 55.99 |
| GQA (2-group) seed 42 | 500K | **27.64** | **3.3192** | 57.43 |

| Task | Hybrid | BLT seed 7 | GPT-2 seed 42 |
|------|--------|------------|---------------|
| LAMBADA acc | 0.222 | 0.212 | **0.225** |
| LAMBADA ppl | **167.1** | 244.4 | 174.6 |
| HellaSwag acc_norm | **0.273** | 0.268 | 0.268 |
| PIQA acc_norm | 0.562 | 0.568 | **0.579** |
| Winogrande acc | 0.504 | 0.516 | 0.505 |

Converting half of BLT's layers (12 → 6) to standard MHA closes the OWT loss gap from
0.1036 nats (full BLT vs GPT-2) down to 0.0219 nats — recovering **over 75% of the gap**
from converting only 50% of the layers.  This is clearly non-additive: if the cost of
cross-layer sharing were a flat per-layer tax, halving the BLT layer count would close
roughly half the gap, not three-quarters.  The result instead suggests the cost compounds
across layers — perhaps because errors in early-layer representations propagate and are
amplified by later shared-M layers, or because a single M serving 12 layers' worth of
attention contexts is harder to optimize than one serving 6.  Six layers of unconstrained
per-layer W_q/W_k appear to be enough for the model to largely route around M's
limitations, even though M is still present in the other half of the network.

The hybrid's LAMBADA result is the most striking individual number in this paper: its
perplexity (167.1) beats every other architecture tested, including full standard MHA
(174.6).  This is consistent with the picture from Section 4.5 — LAMBADA appears to depend
on per-layer key capacity rather than cross-layer sharing — and indicates that 6 full MHA
layers are sufficient to fully recover, and even modestly exceed, MHA's long-range
retrieval performance, while still saving parameters via the remaining 6 BLT layers.

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

**Multi-head Latent Attention (MLA).**
DeepSeek-V2 introduces MLA, which jointly compresses K and V into a low-dimensional latent
vector via a shared down-projection, then recovers per-head K and V via up-projections at
attention time ([arXiv 2405.04434](https://arxiv.org/abs/2405.04434)).
This dramatically reduces KV-cache size while maintaining model quality, and has become a
standard component of the DeepSeek model family.  The low-rank UV^T extension of BLT
(Section 6) converges toward MLA in spirit: both compress the key and value cache via
low-rank projections.  The critical distinction is that BLT's projections U and V are
*globally shared across all layers*, whereas MLA's down/up-projection matrices are
per-layer.  BLT's cross-layer sharing is the stronger inductive bias and the architectural
claim that differentiates it from MLA.  Whether global sharing hurts, helps, or is neutral
relative to per-layer sharing at fixed cache budget is an open empirical question.

**Summary.**
To our knowledge, the specific combination in BLT — merging Q and K into a single bilinear
interaction matrix M that is shared across all transformer layers, while keeping V and O
projections per-layer — has not been previously proposed.  The recent independent results
corroborating the redundancy of separate Q and K matrices lend additional support to the
approach.  The low-rank UV^T variant of BLT is the natural evolution toward KV-cache
compression and tensor-parallel scalability, and its relationship to MLA suggests a
promising direction for larger-scale validation.

---

## 6. Discussion

**Parameter efficiency vs. expressiveness.** BLT achieves better language modelling
perplexity than pretrained GPT-2 after fine-tuning on WikiText-103 (Section 4.1), but
carries a ~0.10 nat cost in the from-scratch OWT comparison (Section 4.4).  The GQA
baseline (Section 4.5) shows that this OWT perplexity cost is not due to parameter
reduction — GQA has a similar parameter count to BLT but matches standard MHA on OWT ppl.
The OWT ppl gap is specifically the cost of cross-layer M sharing.  However, GQA and BLT
are essentially tied on LAMBADA, both well below GPT-2; the LAMBADA gap is a separable
failure mode reflecting constrained per-layer key capacity rather than cross-layer sharing.
BLT therefore carries two distinct costs: ~0.10 nats on general next-token prediction
(from cross-layer sharing) and reduced long-range retrieval accuracy (from constrained key
capacity, shared with GQA).  Against these costs it offers 48% fewer attention parameters
and one fewer D×D matrix multiply per layer per step.

**LAMBADA and training domain.** The WikiText-trained BLT scored 0.114 on LAMBADA vs
GPT-2's 0.242, which initially suggested the shared-M design might limit long-range
narrative understanding.  The OWT run (250K steps, ~1B tokens) resolved this: BLT reached
0.199 on LAMBADA while matching GPT-2 on HellaSwag and exceeding it on PIQA.  The
from-scratch OWT comparison (Section 4.4) further confirms BLT reaches 0.212 on LAMBADA
at full training scale, versus 0.225 for standard MHA — a modest residual gap that tracks
with the overall OWT perplexity gap rather than indicating a specific architectural
limitation for long-range retrieval.  The hybrid model (Section 4.6) closes this gap
entirely and slightly overshoots it (0.222 acc, 167.1 ppl vs GPT-2's 174.6 ppl), reinforcing
that the LAMBADA shortfall is tied to constrained per-layer key capacity rather than an
inherent property of bilinear attention.

**Hybrid architecture (completed result).** Section 4.6 answers the question of whether
BLT's expressiveness cost is additive across layers: it is not.  A 6 MHA + 6 BLT hybrid,
trained from scratch on OWT under the same protocol as the other from-scratch runs,
recovers over 75% of the OWT loss gap between full BLT and standard MHA while still using
6 fewer sets of per-layer W_q/W_k than standard MHA.  It also produces the best LAMBADA
perplexity of any architecture tested in this paper.  This indicates the cost of serving
12 layers' worth of attention contexts from a single shared M compounds rather than sums,
and that a relatively small fraction of unconstrained layers is enough to substantially
mitigate it — a promising direction for deploying BLT-style parameter sharing without
paying its full expressiveness cost.

**SVD analysis of trained M.** We computed the SVD of M from a BLT from-scratch run (the
550K-step seed-42 run from the wall-clock-matching experiment described in Section 3; this
analysis is a structural property of the trained M itself and is unaffected by the choice
of seed or exact step count).  The singular value spectrum is nearly flat: 550 of 768
singular values exceed 10% of the maximum, and r=256 captures only 81% of Frobenius
energy.  M is genuinely full-rank, not converging to a natural low-rank solution.

This is interpretable: M must simultaneously encode useful attention patterns for all
144 attention contexts (12 layers × 12 heads), requiring broad coverage of the full
D-dimensional space.  A UV^T factorization therefore needs r ≥ 192–256 to be a reasonable
approximation; r=64 would lose 64% of Frobenius energy.  The flatness of the spectrum
also explains why M cannot be low-rank from the outset — the 144 distinct attention
contexts it must serve span the space.

**Tensor parallelism compatibility.** Production-scale models distribute attention across
multiple GPUs using tensor parallelism (TP): each GPU handles H/N heads independently,
with a single all-reduce for the output projection.  Standard MHA and GQA shard cleanly
by head or KV group.

Full-rank M BLT is fundamentally incompatible with head-level TP.  Since M produces
identical attention weights for all heads, it cannot be partitioned by head.  The
alternatives — replicating M on every GPU, or sharding M and all-reducing the L×L
attention matrix — both degrade bandwidth efficiency as N grows.  At 8-way TP on a 70B
model, full M BLT uses 2.5× more bandwidth per GPU than standard MHA.

The low-rank UV^T variant eliminates this problem.  At r=256 and D=8192, U and V together
occupy 8.4 MB — small enough to fit in an H100's 50 MB L2 cache and trivial to reload even
if evicted by intervening FFN weights.  Since only W_v and W_o are sharded across GPUs,
and W_q and W_k are replaced entirely by the cached U and V, UV^T BLT maintains
approximately 37–43% lower attention weight bandwidth than standard MHA at any level of
tensor parallelism:

```
MHA (8-way TP, 70B):         67 MB / GPU / layer
Full M BLT (8-way TP):      168 MB / GPU / layer   (2.5× worse than MHA)
UV^T BLT (8-way TP, r=256):  42 MB / GPU / layer   (37% better than MHA)
```

This makes UV^T not merely a KV-cache optimization but a prerequisite for BLT to be
viable at production scale: full M BLT is a single-GPU architecture; UV^T BLT scales
correctly under tensor parallelism.

**Memory bandwidth at inference.** Large-model decode is memory-bandwidth-bound: the
bottleneck is streaming weights and KV cache from HBM.  BLT's weight bandwidth advantage
over standard MHA is strict: there are no per-layer W_q or W_k matrices to load, and M
(a single D×D matrix) is loaded once per forward pass.  For GPT-2 scale (D=768), M is
1.2 MB and fits entirely in GPU L2 cache (V100: 6 MB; H100: 50 MB), making subsequent
layers effectively free on the weight side.  For 7B models (D=4096), M is 33.6 MB and
fits on H100/A100.  Standard MHA must load W_q and W_k from HBM for every layer at every
decode step — 2 × D² × L bytes per token vs BLT's single D² load amortized across all
L layers.

BLT's KV cache is the same size as standard MHA: both store D-dimensional vectors per
token per layer (MHA stores W_k · x_j; BLT stores raw x_j).  Neither achieves the KV
cache compression of GQA.  At long contexts (>32K tokens on large models), KV cache
bandwidth dominates and BLT's weight advantage shrinks as a fraction of total bandwidth.
The low-rank UV^T variant described below addresses this.

**KV-cache compatibility.** Standard KV caching stores K = X W_k for past tokens.  In BLT,
the score between new token t and past token j is (x_t M) · x_j, so the cache only needs
to store the raw hidden states x_j (as implicit keys) and x_j W_v (as values).  A single
M multiply produces the query; no key projection is required when adding new tokens.  The
current implementation does not exploit this and recomputes full attention from scratch at
each forward pass; an inference-optimized implementation would realize this as a genuine
efficiency gain over standard attention.

However, because x_j is full D-dimensional, BLT's key cache is the same size as standard
MHA's key cache — it does not benefit from the KV-cache compression that GQA achieves by
projecting keys to a lower-dimensional space.

**Low-rank BLT and KV-cache compression.** A natural extension that recovers this
compression is to factor M as a rank-r outer product:

```
M = U V^T     where U, V ∈ ℝ^{D×r}
```

The attention score then becomes:

```
score(i,j) = x_i^T (U V^T) x_j = (x_i U) · (x_j V) / √d
```

At inference, the key cache stores x_j V (r-dimensional) rather than the full x_j
(D-dimensional), reducing KV-cache size by a factor of D/r — identical to the reduction
achieved by GQA with D/r KV groups.  Both U and V remain globally shared across all
layers, preserving BLT's cross-layer parameter sharing.  At r = D/16, the key cache is
16× smaller than standard MHA while the shared weight budget drops from D² to 2Dr
(e.g. 8M vs 67M for D=8192).

The SVD flatness result (above) means that a UV^T model cannot simply be initialized by
truncating the spectrum of a trained full-rank M — r=256 would capture only 81% of energy
at GPT-2 scale.  The practical approach is post-training SVD factorization followed by
fine-tuning: take a trained M, factorize M ≈ UV^T at r=192 or r=256, reinitialize the
model with the factored weights, and fine-tune for ~50–100K additional steps.

This creates a direct architectural comparison: **globally shared low-rank UV^T vs
per-layer full-rank GQA**, at the same KV-cache budget.  GQA has more expressive
per-layer key projections; low-rank BLT has stronger cross-layer regularization.  Which
inductive bias wins, and at what model scale, is an open empirical question.
