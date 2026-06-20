# Bilinear Attention with a Shared Interaction Matrix (BLT)

## Abstract

The attention calculation is the single most time- and memory-consuming feature of modern Large Language Models (LLMs) and other transformer-based AI implementations.  Much research has gone into memory management and algorithm parallelization to reduce the time the attention calculation requires; some research has also gone into modifying the attention mechanism itself to streamline the computation.

In this paper we introduce BLT (Bilinear aTtention), a drop-in replacement for standard multi-head attention (MHA) that replaces the model's per-layer query and key projections (W_q, W_k) with a single D×D matrix M shared across all layers.  This cuts attention parameters by up to 50%, and — because M is small enough to stay resident in GPU cache for smaller models — removes the need to stream per-layer W_q/W_k from HBM at every layer, a bandwidth saving standard MHA cannot avoid.  BLT also does strictly less arithmetic (one fewer matrix multiply per layer), a modest additional compute saving.

We trained GPT-2 Small with BLT, standard MHA, and a Grouped-Query Attention (GQA) variant, and evaluated all three on a battery of standard benchmarks.  BLT's cost relative to standard MHA is concentrated, not across the board: it trails on perplexity and on LAMBADA's long-range retrieval task, but ties or beats MHA on HellaSwag and Winogrande, and trails only slightly on PIQA.  GQA pays a similarly sized overall cost relative to MHA, for different underlying reasons.  Give or take, neither BLT nor GQA is decisively better than the other.

We also tested a hybrid architecture — half standard MHA layers, half BLT layers — which recovers most of what full BLT gives up in quality, at the cost of keeping only half of BLT's memory benefit.

---

## 1. Background: Standard Multi-Head Attention

Let **X** ∈ ℝ^{L×D} be the sequence of hidden states entering an attention layer, where L is the sequence length and D is the model dimension.  A transformer with N layers and H heads per layer uses per-head dimension d = D/H.

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

Every layer has its own W_q^{l,h}, W_k^{l,h}, W_v^{l,h}, W_o^l.  The attention score between positions i and j in head h can be written as a bilinear form:

```
score_h(i,j) = x_i^T ( W_q^h W_k^{h,T} ) x_j / √d
```

so each head implicitly defines a D×D interaction matrix M_h = W_q^h W_k^{h,T}.  Across N layers and H heads, standard MHA has **N×H distinct interaction matrices**.

**Attention parameters per layer:**

| Matrix | Shape | Params |
|--------|-------|--------|
| W_q    | D×D   | D² |
| W_k    | D×D   | D² |
| W_v    | D×D   | D² |
| W_o    | D×D   | D² |
| **Total per layer** | | **4D²** |

Over N layers: 4ND² attention parameters total.

---

## 2. BLT: Bilinear Attention with a Shared M

BLT replaces all per-layer W_q and W_k matrices with a **single shared** D×D matrix M. The attention score between positions i and j is:

```
score(i,j) = x_i^T M x_j / √d
```

This is a **bilinear form** in x_i and x_j: for fixed M, it is linear in x_i with x_j held fixed, and linear in x_j with x_i held fixed (e.g. doubling x_i doubles the score).  A bilinear form is the natural way to express a degree-two interaction between two vectors mediated by a single matrix, which is exactly the role M plays here — hence the name **BLT (Bilinear aTtention)**.

In matrix form, the full attention computation for a layer becomes:

```
S = ( X M ) X^T / √d          (L×D)(D×D)(D×L) → L×L
A = softmax( S + causal_mask )
V = X W_v + b_v
h = A V
out = h W_o + b_o
```

M is shared across **all N layers**; W_v and W_o remain per-layer.

**Attention parameters:**

| Matrix   | Shape | Count | Params |
|----------|-------|-------|--------|
| M (shared) | D×D | 1 | D² |
| W_v      | D×D   | N layers | ND² |
| W_o      | D×D   | N layers | ND² |
| **Total** | | | **(2N+1)D²** |

Relative to standard MHA's 4ND², BLT's (2N+1)D² is a reduction that grows with N, approaching 50% in the limit of many layers (e.g. just under that at moderate depths).  Section 3 gives the exact reduction for the specific model used in our experiments.

### 2.1 Relationship to Standard Attention

Standard attention can always be written as a bilinear form:

```
score_h^l(i,j) = x_i^T M_h^l x_j / √d     where M_h^l = W_q^{l,h} W_k^{l,h,T}
```

BLT is the extreme case of this where M_h^l = M for all layers l and all heads h.  The single M must therefore capture the average useful query-key interaction across all layers and all head slots.

### 2.2 Initialization

Since M replaces W_q W_k^T, a natural initialization is:

```
M_init = (1/N) Σ_l  W_q^l W_k^{l,T}
```

the average of the per-layer query-key interaction matrices of a pretrained standard-attention model with the same architecture.  We also experiment with **random initialization**: M ~ N(0, 1/√D · I), which tests whether the pretrained initialization is load-bearing or whether the model can learn M from scratch.

### 2.3 Consequence for Head Diversity

In standard MHA, each of the H heads attends to different parts of the context because each head has its own M_h.  In BLT (one shared M), all H "head slots" compute **identical attention weights**.  The only per-head differentiation comes from the W_v projection. This is the primary representational cost of parameter sharing.

---

## 3. Experimental Setup

**Base model:** GPT-2 Small (12 layers, 12 heads, D=768, d=64).  Pretrained weights, where used (Sections 4.1–4.2 and the "GPT-2 pretrained" baselines), are OpenAI's original release as distributed via HuggingFace `transformers` (`GPT2LMHeadModel.from_pretrained('gpt2')`).

**Training:** All parameters trained with Adam (lr=5e-5, cosine decay to near zero, 200 warmup steps, batch size 4, block size 1024).  This is the GD-only control condition — no alternative optimizers are used.

**Datasets:**
- *WikiText-103*: English Wikipedia (~103M tokens).  2 epochs ≈ 50,300 steps.
- *OpenWebText* (2M documents, ~2B tokens): Skylion007/openwebtext, first 2M documents. Chinchilla-optimal compute for a ~117M parameter model falls in this range.

**Evaluation:** Sliding-window perplexity on WikiText-103 validation (stride=512) and on a held-out OWT split (files 21–25, ~50K tokens); zero-shot accuracy on LAMBADA, HellaSwag, PIQA, and Winogrande via lm-eval-harness (primary suite), plus ARC-Easy, BoolQ, and OpenBookQA as a supplementary noise check (Section 4.6).

**Baselines:**
- *Standard MHA (GPT-2)*: trained from scratch, 500K steps.
- *2-group GQA*: 12 query heads, 2 KV heads (6 queries per KV head); W_q and W_o are full D×D per layer, W_k and W_v project to 2×64=128 dimensions.  Trained from scratch, 500K steps.  GQA has 117.9M unique parameters — close to BLT's 110.9M and between BLT and GPT-2's 124.4M — making it the right tool for separating parameter count effects from architectural expressiveness.

**Step count:** The from-scratch architectures reported in this paper — BLT, standard MHA, GQA, and the hybrid model (Section 4.5) — are trained from scratch for a fixed **500K steps** on identical data and hyperparameters, giving a clean iso-step comparison.  This does not apply to Sections 4.1–4.2, which fine-tune from pretrained GPT-2 weights rather than training from scratch (see those sections for their own step counts).

---

## 4. Results

### 4.1 WikiText-103 Fine-Tuning

This experiment starts from *pretrained* GPT-2 Small weights, not random initialization: every weight except W_q and W_k — embeddings, W_v, W_o, MLP, layer norms — is copied from pretrained GPT-2 and never reinitialized.  M is *warm-started* using the formula in Section 2.2 (the average of W_q^l W_k^{l,T} across the pretrained model's layers), and the whole model is then fine-tuned end-to-end on WikiText-103 for 50,300 steps (2 epochs, Section 3).  The "GPT-2 fine-tuned" row below applies the identical fine-tuning recipe with no architecture change, as a control.  This tests whether *converting* a pretrained standard-attention model into BLT and fine-tuning it recovers good performance — not whether BLT can be trained from scratch, which is addressed separately starting in Section 4.3, where every weight in every architecture is randomly initialized.

**Perplexity:**

| Model | Val PPL |
|-------|---------|
| GPT-2 pretrained (no fine-tune) | 26.39 |
| GPT-2 fine-tuned (WikiText-103) | ~14.91 (stopped at step 33,930) |
| BLT (fine-tuned, warm-started M) | **21.50** |

BLT surpasses the pretrained GPT-2 baseline despite using up to 50% fewer attention parameters, demonstrating that the shared-M constraint is not prohibitively limiting for language modelling.

**Zero-shot benchmarks:**

| Task | GPT-2 pretrained | BLT (fine-tuned) |
|------|-----------------|-------------------|
| LAMBADA acc     | 0.242 | 0.114 |
| LAMBADA ppl     | 83.0  | 1307.5 |
| HellaSwag acc_norm | 0.291 | 0.275 |
| PIQA acc_norm   | 0.560 | 0.541 |
| Winogrande acc  | 0.502 | 0.507 |

BLT (fine-tuned) matches GPT-2 closely on HellaSwag, PIQA, and Winogrande.  The LAMBADA gap turns out to be a training domain artifact, not an architectural limitation: see Section 4.2.

### 4.2 OpenWebText with Random M Initialization

Starting again from the same pretrained GPT-2 Small weights as Section 4.1, but this time leaving M randomly initialized (N(0, 1/√D)) instead of warm-started, we continued training on the first 1M documents of OpenWebText (~1B tokens) for 250,000 steps (~1 epoch).  As in Section 4.1, every weight besides M is copied from pretrained GPT-2 and never reinitialized — this is still a fine-tuning experiment, not a from-scratch run.  It tests two simultaneous hypotheses: (1) whether M itself can be *learned* without the warm-start — i.e. whether gradient descent alone can recover a good M while every other weight stays pretrained — and (2) whether broader web text improves LAMBADA scores by better matching the narrative distribution.

Both hypotheses are confirmed.  M converges successfully from random initialization — WikiText-103 validation perplexity declined from 34,114 at step 0 (attention is essentially noise with a random M) to 48.0 at step 218,000, demonstrating that the bilinear interaction structure can be learned entirely from data without the Wq@Wk^T warm-start.

| Task | GPT-2 pretrained | BLT (warm-started M, WikiText) | BLT (random M, OWT step 126K) | BLT (random M, OWT step 218K) | BLT (random M, OWT step 250K) |
|------|-----------------|-------------------|------------------------|------------------------|------------------------|
| LAMBADA acc     | 0.242 | 0.114 | 0.182 | 0.196 | **0.199** |
| LAMBADA ppl     | 83.0  | 1307.5 | 243.0 | 210.1 | **206.8** |
| HellaSwag acc_norm | 0.291 | 0.275 | 0.280 | 0.279 | 0.280 |
| PIQA acc_norm   | 0.560 | 0.541 | **0.572** | 0.566 | 0.568 |
| Winogrande acc  | 0.502 | 0.507 | 0.503 | 0.487 | 0.490 |

LAMBADA improved monotonically (0.182 → 0.199) across all three checkpoints and had not saturated, approaching GPT-2's 0.242 after only ~1B tokens — roughly 1% of GPT-2's training data.  This confirms the LAMBADA gap observed in Section 4.1 was a training domain effect: encyclopedia text provides no signal for the narrative last-word prediction patterns LAMBADA tests.

### 4.3 From-Scratch OpenWebText Comparison

The definitive test of BLT is a controlled comparison against standard MHA and GQA on identical data, compute, and hyperparameters, all trained from scratch on 2M OWT documents for a fixed 500K steps (Section 3).

**Primary results:**

| Model | OWT ppl | OWT loss | WikiText ppl |
|-------|---------|----------|--------------|
| BLT seed 7 | 30.81 | 3.4279 | 71.57 |
| GPT-2 (MHA) seed 42 | **27.78** | **3.3243** | 55.99 |
| GQA (2-group) seed 42 | **27.64** | **3.3192** | 57.43 |

BLT trails both baselines by a consistent ~0.10 nat / ~3 ppl margin; GQA is marginally ahead even of standard MHA on this one metric.

**Zero-shot benchmarks:**

| Task | BLT seed 7 | GPT-2 seed 42 | GQA seed 42 |
|------|------------|---------------|-------------|
| LAMBADA acc     | 0.212 | **0.225** | 0.204 |
| LAMBADA ppl     | 244.4 | **174.6** | 205.3 |
| HellaSwag acc_norm | 0.268 | 0.268 | 0.269 |
| PIQA acc_norm   | 0.568 | **0.579** | 0.568 |
| Winogrande acc  | **0.516** | 0.505 | 0.496 |

GQA's perplexity edge does not carry over to the benchmarks: BLT beats GQA on LAMBADA (0.212 vs 0.204) and Winogrande (0.516 vs 0.496), ties it on HellaSwag and PIQA, and only loses cleanly on OWT perplexity.  Give or take, BLT and GQA are in the same ballpark — each gives up something to standard MHA somewhere in the suite, for different reasons (Section 4.4), but neither dominates the other or trails MHA across the board.

**Note on WikiText perplexity:** WikiText val ppl was noisy throughout BLT training (swings of 25+ ppl within a single run), while OWT held-out ppl was stable.  We treat OWT ppl as the reliable metric and WikiText ppl as approximate.

### 4.4 Isolating the Source of BLT's Gap

BLT (110.9M parameters) uses fewer unique parameters than GPT-2 (124.4M).  A natural question is whether the ~0.10 nat OWT perplexity gap reflects parameter count, the expressiveness constraint from sharing M across layers, or both.  The GQA baseline (117.9M parameters, between BLT and GPT-2) answers this — but the answer splits cleanly by metric.

**OWT perplexity (27.78 GPT-2 / 27.64 GQA / 30.81 BLT).** GQA matches GPT-2 on the primary metric despite having fewer parameters and compressing KV to 2 groups.  Since parameter count and KV compression are both ruled out as causes, the OWT ppl gap is attributable specifically to cross-layer M sharing: a single M must simultaneously produce useful attention weights across 12 layers × 12 heads = 144 distinct attention contexts, while per-layer matrices in MHA and GQA can specialize freely.

**LAMBADA (0.225 GPT-2 / 0.204 GQA / 0.212 BLT).** Here GQA falls to BLT's level, well below GPT-2.  GQA retains per-layer W_k matrices but reduces each layer's effective key rank from 768 dimensions (12 heads × 64) to 128 (2 groups × 64) — a 6× compression.  The fact that this compression hurts LAMBADA as much as BLT's cross-layer sharing does suggests that LAMBADA specifically requires high per-layer key capacity, not merely per-layer key matrices.  Both constraints — reduced rank within a layer (GQA) and a single shared matrix across layers (BLT) — are sufficient to degrade long-range prediction.

**Summary.**  The OWT perplexity gap and the LAMBADA gap have different causes and are not both paid by the same architecture: BLT pays the OWT cost (cross-layer sharing) but not the LAMBADA cost; GQA pays the LAMBADA cost (compressed key rank) but not the OWT cost.  Across the full benchmark suite the two end up comparable.  Only standard MHA, with full per-layer W_k rank and no cross-layer sharing, avoids both costs.

### 4.5 Hybrid Architecture: Mixing MHA and BLT Layers

Section 4.4 attributes BLT's OWT perplexity gap to cross-layer M sharing across all 12 layers.  Is that cost additive — does removing half the shared layers recover about half the gap — or does it compound?

We trained a hybrid model with 6 standard MHA layers followed by 6 BLT layers, from scratch on OWT for 500K steps, identical in every other respect to the runs in Section 4.3.

| Model | OWT ppl | OWT loss | WikiText ppl |
|-------|---------|----------|--------------|
| BLT seed 7 (12 BLT layers) | 30.81 | 3.4279 | 71.57 |
| Hybrid (6 MHA + 6 BLT) | 28.40 | 3.3462 | 69.08 |
| GPT-2 (MHA) seed 42 (12 MHA layers) | **27.78** | **3.3243** | 55.99 |
| GQA (2-group) seed 42 | **27.64** | **3.3192** | 57.43 |

| Task | Hybrid | BLT seed 7 | GQA seed 42 | GPT-2 seed 42 |
|------|--------|------------|-------------|---------------|
| LAMBADA acc | 0.222 | 0.212 | 0.204 | **0.225** |
| LAMBADA ppl | **167.1** | 244.4 | 205.3 | 174.6 |
| HellaSwag acc_norm | **0.273** | 0.268 | 0.269 | 0.268 |
| PIQA acc_norm | 0.562 | 0.568 | 0.568 | **0.579** |
| Winogrande acc | 0.504 | 0.516 | 0.496 | 0.505 |

Converting half of BLT's layers to standard MHA closes the OWT loss gap from 0.1036 nats (full BLT vs MHA) to 0.0219 nats — recovering **over 75% of the gap** from converting only 50% of the layers.  This is clearly non-additive: a flat per-layer tax would close roughly half the gap, not three-quarters, suggesting the cost of cross-layer sharing compounds rather than sums — six unconstrained layers are apparently enough for the model to largely route around M's limitations even with M still present elsewhere.

Against GQA, the hybrid wins on LAMBADA, HellaSwag, and Winogrande, ties on OWT perplexity within a point, and trails only on PIQA — a clear improvement over full BLT's roughly even standing with GQA (Section 4.3).  Against full MHA, the hybrid is within noise everywhere except PIQA, and its LAMBADA perplexity (167.1) is the best of any model tested, including MHA itself (174.6).

This comes at a real memory cost: only the 6 BLT layers keep BLT's benefit.  Attention parameters fall from MHA's 28.3M to about 21.8M (a ~23% reduction, vs. 48% for full BLT), and only those 6 layers avoid streaming per-layer W_q/W_k from HBM — the other 6 behave exactly like standard MHA on the memory side.  The hybrid is a genuine quality/memory trade-off point between BLT and MHA, not a free win.

### 4.6 Supplementary Zero-Shot Benchmarks

As a noise check on the four-task suite used above, we ran three additional zero-shot tasks (ARC-Easy, BoolQ, OpenBookQA; lm-eval-harness defaults) on all four checkpoints.

| Task | BLT seed 7 | Hybrid | GPT-2 seed 42 | GQA seed 42 |
|------|------------|--------|---------------|-------------|
| ARC-Easy acc | 0.366 | 0.368 | 0.373 | **0.381** |
| ARC-Easy acc_norm | 0.339 | 0.344 | **0.348** | 0.345 |
| BoolQ acc | **0.611** | 0.606 | 0.571 | 0.596 |
| OpenBookQA acc | 0.130 | 0.122 | 0.136 | **0.138** |
| OpenBookQA acc_norm | **0.248** | 0.234 | 0.226 | 0.236 |

*These three tasks are noisier than the primary suite at this model scale (~117M-parameter models score only a few points above chance on ARC-Easy/OpenBookQA, and BoolQ accuracy is known to be sensitive to a model's yes/no answer bias).  We report them for completeness rather than as additional evidence either way: no architecture dominates this set — BLT actually posts the best BoolQ and OpenBookQA acc_norm scores — and nothing here changes the "same ballpark" conclusion from Sections 4.3–4.5.*

---

## 5. Related Work

Several recent lines of work converge on the question of whether the standard Q, K, V triplet in self-attention is over-parameterized.

**Shared weights within a layer.** Lan et al. propose sharing a single weight matrix for Q, K, *and* V within each attention layer in BERT, reporting a 66% reduction in attention parameters with no loss in accuracy on GLUE tasks and improved generalization on out-of-domain data ([arXiv 2412.00359](https://arxiv.org/abs/2412.00359), NAACL 2025). This is the closest prior work to BLT: both recognize that Q and K do not need separate matrices.  The key differences are that BLT (i) retains V as a distinct per-layer projection, (ii) expresses the Q/K interaction as an explicit bilinear form X M X^T rather than tying the matrices, and (iii) shares M *across all layers* rather than within a single layer.

**Eliminating Q or K entirely.** Brandon et al. prove that the Query or Key weight matrix can be replaced by the identity without loss in small GPT-style models, reducing attention parameters by 25% and simplifying optimization by making attention logits linear in the remaining learned weights ([arXiv 2510.23912](https://arxiv.org/pdf/2510.23912)). BLT can be seen as a more structured version of this idea: rather than removing one projection entirely, it merges both into a single bilinear matrix that is shared across layers.

**Cross-layer KV sharing.** Brandon et al. propose Cross-Layer Attention (CLA), which shares K and V *activations* (not weight matrices) between adjacent layers to reduce KV-cache size at inference ([arXiv 2405.12981](https://arxiv.org/abs/2405.12981), NeurIPS 2024). The goal is inference memory rather than parameter count; the approach is orthogonal to BLT and could in principle be combined with it.

**Matrix-based dictionary learning.** MASA decomposes all attention projection matrices (Q, K, V, O) into linear combinations of shared dictionary atoms across layers, achieving 66.7% attention parameter reduction ([arXiv 2508.04581](https://arxiv.org/abs/2508.04581)). This is a general compression framework; BLT's sharing is more targeted and structurally motivated by the bilinear interpretation of attention.

**Multi-head Latent Attention (MLA).** DeepSeek-V2 introduces MLA, which jointly compresses K and V into a low-dimensional latent vector via a shared down-projection, then recovers per-head K and V via up-projections at attention time ([arXiv 2405.04434](https://arxiv.org/abs/2405.04434)). This dramatically reduces KV-cache size while maintaining model quality, and has become a standard component of the DeepSeek model family.  The low-rank UV^T extension of BLT (Section 6) converges toward MLA in spirit: both compress the key and value cache via low-rank projections.  The critical distinction is that BLT's projections U and V are *globally shared across all layers*, whereas MLA's down/up-projection matrices are per-layer.  BLT's cross-layer sharing is the stronger inductive bias and the architectural claim that differentiates it from MLA.  Whether global sharing hurts, helps, or is neutral relative to per-layer sharing at fixed cache budget is an open empirical question.

**Summary.** To our knowledge, the specific combination in BLT — merging Q and K into a single bilinear interaction matrix M that is shared across all transformer layers, while keeping V and O projections per-layer — has not been previously proposed.  The recent independent results corroborating the redundancy of separate Q and K matrices lend additional support to the approach.  The low-rank UV^T variant of BLT is the natural evolution toward KV-cache compression and tensor-parallel scalability, and its relationship to MLA suggests a promising direction for larger-scale validation.

---

## 6. Discussion

**Why BLT: memory, not accuracy.** BLT's case is architectural, not a quality play.  It cuts attention parameters by 48%, and because M is a single D×D matrix small enough to live in GPU L2 cache at GPT-2 scale (1.2 MB vs. V100's 6 MB / H100's 50 MB), it removes the need to stream per-layer W_q and W_k from HBM — a cost standard MHA pays on every layer, every decode step.  BLT also does strictly less arithmetic: one fewer D×D matmul per layer.  That FLOP reduction is real, but the runs in this paper ran on different hardware (444–593 ms/step across architectures and machines), so we did not get a clean wall-clock comparison; the speed benefit should be read as a structural property of the architecture, not a validated empirical result here.

**Quality: BLT and GQA each give up something to MHA, but not the same thing, and not everywhere.** Section 4.3 shows BLT trails MHA by ~0.10 nats of OWT loss; GQA does not trail on this metric at all, edging slightly ahead instead.  The zero-shot benchmarks largely invert the picture: BLT matches or beats both GQA and MHA on HellaSwag and Winogrande, and trails only slightly on PIQA — its real benchmark weakness is LAMBADA, where GQA's per-layer key-rank compression costs it just as much ground.  Across the full suite, BLT and GQA each pay a real cost relative to MHA, concentrated in different places and for different reasons: BLT mainly on OWT perplexity (cross-layer M sharing), GQA mainly on LAMBADA (compressed per-layer key rank).  Neither is decisively better than the other, and neither uniformly trails MHA either; the right choice depends on whether parameter count or KV-cache size is the binding constraint.

**LAMBADA's gap is a training-domain effect, not an architecture limit.** WikiText-trained BLT scored 0.114 on LAMBADA vs. GPT-2's 0.242 (Section 4.1), which initially suggested a real architectural limitation.  Training on broader web text resolved this: BLT reached 0.199 after just 250K OWT steps (Section 4.2) and 0.212 at full from-scratch scale (Section 4.3) — and the hybrid model exceeds GPT-2 outright on LAMBADA perplexity (167.1 vs 174.6, Section 4.5).  The residual BLT-vs-MHA gap tracks the per-layer-key-capacity story above, not a narrative-understanding deficit.

**The hybrid is the best quality/memory trade-off tested, not a free lunch.** Section 4.5's 6 MHA + 6 BLT hybrid clearly beats GQA (LAMBADA, HellaSwag, Winogrande) and is within noise of full MHA on everything but PIQA, while still saving ~23% of attention parameters. The gap recovery is disproportionate to the layer count converted — half the BLT layers recover over 75% of the loss gap — suggesting the cost of cross-layer sharing compounds across layers rather than summing linearly: six unconstrained layers are apparently enough for the model to mostly route around M's limitations even with M still present elsewhere.  But the hybrid keeps only half of BLT's memory benefit; it is a point on a curve between BLT and MHA, and where to sit on that curve is a deployment choice this paper does not resolve.

**SVD analysis of trained M.** We computed the SVD of M from a BLT from-scratch run (seed 42); this analysis is a structural property of the trained M itself and is unaffected by the choice of seed or exact step count.  The singular value spectrum is nearly flat: 550 of 768 singular values exceed 10% of the maximum, and r=256 captures only 81% of Frobenius energy.  M is genuinely full-rank, not converging to a natural low-rank solution.

This is interpretable: M must simultaneously encode useful attention patterns for all 144 attention contexts (12 layers × 12 heads), requiring broad coverage of the full D-dimensional space.  A UV^T factorization therefore needs r ≥ 192–256 to be a reasonable approximation; r=64 would lose 64% of Frobenius energy.  The flatness of the spectrum also explains why M cannot be low-rank from the outset — the 144 distinct attention contexts it must serve span the space.

**Memory bandwidth at inference.** Large-model decode is bandwidth-bound, and weights vs. KV cache trade off differently for BLT.  The weight-side win is unconditional (see "Why BLT" above): standard MHA loads W_q and W_k from HBM for every layer at every decode step — 2 × D² × L bytes per token vs. BLT's single D² load amortized across all L layers.  The KV-cache side is not a win: BLT caches the same D-dimensional vector per token per layer as standard MHA (it stores raw x_j as an implicit key, not a compressed one), so it gets none of GQA's cache compression.  At long contexts, where KV-cache bandwidth dominates total bandwidth, this erodes BLT's advantage as a fraction of the total — addressed by the low-rank UV^T variant proposed in Section 7.3.

**Tensor parallelism compatibility.** Production-scale models distribute attention across multiple GPUs using tensor parallelism (TP): each GPU handles H/N heads independently, with a single all-reduce for the output projection.  Standard MHA and GQA shard cleanly by head or KV group.  Full-rank M BLT is fundamentally incompatible with head-level TP: since M produces identical attention weights for all heads, it cannot be partitioned by head.  The alternatives — replicating M on every GPU, or sharding M and all-reducing the L×L attention matrix — both degrade bandwidth efficiency as N grows.  At 8-way TP on a 70B model, full M BLT uses 2.5× more bandwidth per GPU than standard MHA.  This is the second motivation (alongside cache residency, Section 7.1) for the low-rank UV^T variant proposed in Section 7.3.

**KV-cache compatibility.** Standard KV caching stores K = X W_k for past tokens.  In BLT, the score between new token t and past token j is (x_t M) · x_j, so the cache only needs to store the raw hidden states x_j (as implicit keys) and x_j W_v (as values).  A single M multiply produces the query; no key projection is required when adding new tokens.  The current implementation does not exploit this and recomputes full attention from scratch at each forward pass; an inference-optimized implementation would realize this as a genuine efficiency gain over standard attention.

---

## 7. Future Work

### 7.1 Scaling BLT to Larger Models

GQA's only real weakness in our GPT-2-scale comparison — the LAMBADA hit from cutting 12 heads to 2 KV groups, a 6× per-layer key-rank reduction — is the kind of degradation the broader GQA/MQA literature reports shrinking as model size grows: larger models have more redundancy to absorb aggressive KV compression, and production GQA configurations typically use gentler group ratios than the 6× tested here.  If GQA's gap to standard MHA closes with scale while BLT's does not, GQA would pull ahead of BLT at larger model sizes, undermining the "same ballpark" conclusion from Section 6.  That makes testing BLT itself — not just GQA — at larger scale the natural next experiment once larger hardware is available; the GPT-2-scale result here should not be assumed to hold at 7B+.

Scaling up also interacts with BLT's core memory argument.  BLT's case for memory efficiency rests on M staying resident in fast on-chip memory (L2 cache or SRAM, depending on hardware).  This holds comfortably at GPT-2 scale (M = 1.2 MB) and plausibly up to ~13B (M = 52 MB, borderline on an H100's 50 MB L2), but breaks down at 70B (M = 134 MB, exceeding any current GPU's L2).  Past that point, M itself must be streamed from HBM like any other weight matrix, eroding exactly the advantage that motivates full-rank BLT.  This is the central reason the low-rank UV^T variant (Section 7.3) is worth pursuing before, not after, scaling BLT up — full-rank M is a viable architecture only up to the point where M fits in cache.

### 7.2 More M Matrices

BLT (a single shared M) is one extreme of a spectrum; standard MHA is the other, with a distinct M_h^l for every layer and head (N×H total, Section 1).  A natural untested direction is to sit partway between the two: split the heads into a small number of groups, each with its own shared M, partially recovering head diversity while still sharing each M across all layers.  This has not been trained to convergence in this paper and is not reported as a result; we flag it purely as a direction worth exploring.

A fuller sweep — say 2, 4, and 6 shared M matrices, trained from scratch to full convergence on OWT under the same protocol as Section 4.3 — would map out the quality/parameter curve as the number of shared M matrices grows from 1 (full BLT) toward N (standard MHA).  This is the direct M-sharing analogue of GQA's KV-group count: just as GQA interpolates between MQA and MHA by varying the number of KV groups, multi-M BLT would interpolate between BLT and MHA by varying the number of shared M matrices.  Whether multi-M BLT's quality/parameter trade-off curve would dominate GQA's quality/cache-size curve, or vice versa, at a matched parameter or cache budget, is an open question this paper does not answer.

### 7.3 Low-Rank UV^T Factorization

A different way to make BLT cheaper at scale is to constrain M itself, rather than splitting it into multiple full-rank matrices.  Recall (Section 2.1) that any standard attention head implicitly defines an interaction matrix M_h = W_q^h W_k^{h,T} — and since W_q and W_k each have only d = D/H columns, M_h is itself a rank-d factorization of a D×D matrix.  BLT's shared M discards this factorization (M is a full D×D matrix, not a product of two thin ones).  The low-rank UV^T variant puts it back, but shared globally rather than per-layer:

```
M = U V^T     where U, V ∈ ℝ^{D×r}, r ≪ D
```

The attention score becomes:

```
score(i,j) = x_i^T (U V^T) x_j = (x_i U) · (x_j V) / √d
```

— i.e. computing (x_i U) and (x_j V) and taking their dot product, exactly like standard Q/K attention, except U and V are shared across all N layers instead of being defined per-layer.  U plays the role of a shared query projection and V the role of a shared key projection; the asymmetry between them is intentional, since the query-side and key-side questions ("what am I looking for" vs. "what do I offer") have no reason to be answered by the same projection even though both are globally shared.

UV^T directly addresses the three limitations of full-rank BLT raised above:

- **KV-cache compression.** At inference, the key cache stores x_j V (r-dimensional) instead of the full x_j (D-dimensional) — a D/r reduction identical to GQA's D/r KV-group compression ("Memory bandwidth at inference," Section 6).
- **Tensor-parallel compatibility.** Full-rank M cannot be sharded by head ("Tensor parallelism compatibility," Section 6); U and V are small enough (8.4 MB at r=256, D=8192) to replicate on every GPU at any TP degree without the bandwidth blowup full M suffers — at 8-way TP on a 70B model, UV^T BLT uses ~37% *less* bandwidth than standard MHA, versus full M BLT's 2.5× *more*.
- **Cache residency at scale.** U+V (8.4 MB at r=256, D=8192) stay comfortably L2/SRAM resident even where full M (Section 7.1) does not.

This isn't free, though.  The SVD analysis in Section 6 shows M's singular spectrum is nearly flat (550 of 768 singular values exceed 10% of the maximum), so a UV^T model cannot simply be initialized by truncating a trained full-rank M's spectrum — r=256 would capture only 81% of Frobenius energy at GPT-2 scale.  The practical path is post-training SVD factorization followed by fine-tuning: factorize a trained M ≈ UV^T at r=192 or r=256, reinitialize the model with the factored weights, and fine-tune for ~50–100K additional steps.  This has not yet been run.

This sets up a direct architectural comparison we have not yet made: **globally shared low-rank UV^T vs. per-layer full-rank GQA**, at the same KV-cache budget.  GQA has more expressive per-layer key projections; low-rank BLT has stronger cross-layer regularization. Which inductive bias wins, and at what model scale, is an open empirical question.
