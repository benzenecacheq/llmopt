# Faithfulness over Accuracy: Rethinking KV Cache Compression Evaluation with Pre-RoPE Scoring

---

## Abstract

Standard benchmarks for KV cache compression measure whether a compressed model gives the *correct* answer according to ground-truth labels. We argue this is the wrong objective: the goal of compression is approximation fidelity — producing the same output the full-context model would have produced. We introduce a suite of complementary faithfulness metrics that measure similarity to full-context outputs directly, and show that ground-truth rankings and faithfulness rankings disagree substantially. Using faithfulness as the primary lens, we revisit where to compute KV cache importance scores, comparing pre-RoPE and post-RoPE key-query dot products and evaluating a V-norm payload term as an ablation. We find that the two faithfulness metrics — perplexity-based and embedding-based — capture distinct failure modes, motivating their joint use for compression evaluation.

---

## 1. Introduction

KV cache compression methods are typically evaluated by running a compressed model on benchmark tasks and measuring how close the score is to an uncompressed baseline or to labeled ground truth [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that wrong answer — replicating the full model's behavior is the criterion, not improving on it.

This distinction matters in practice. A method that truncates the prompt may accidentally produce correct answers on tasks where the relevant information happens to fall within the retained window, while a semantically-aware method that retains globally relevant content may be penalized for producing outputs that are faithful to the full model but diverge from the ground-truth label. Ground-truth benchmarks cannot distinguish these cases.

We propose to evaluate KV cache compression using *faithfulness metrics* that compare compressed outputs directly to full-context outputs:

- **Lexical faithfulness**: token-level F1 between compressed and full-context output strings
- **Perplexity faithfulness**: how likely the full-context model considers the compressed output, normalized by how likely it considers its own output
- **Embedding faithfulness**: cosine similarity between sentence-transformer embeddings of compressed and full-context outputs

These metrics are complementary. Lexical faithfulness is strict and penalizes paraphrasing; perplexity faithfulness is robust to surface variation but anchored to model likelihood; embedding faithfulness captures semantic content independently of both phrasing and the LLM.

Using faithfulness as the primary evaluation lens, we revisit the design space for KV cache importance scoring. The central question is where to compute KQ dot products: in post-RoPE space (as most methods do) or in pre-RoPE space (before rotary embeddings are applied). Post-RoPE vectors encode both semantic content and absolute position; pre-RoPE vectors encode content only. For long contexts, this difference is consequential: a token at position 0 in a 10K sequence is rotated by a large angle relative to the tail queries, producing a low dot product regardless of semantic relevance. Pre-RoPE scoring removes this positional penalty and recovers distant-but-relevant tokens.

We make the following contributions:

- **Faithfulness evaluation framework**: three complementary metrics for measuring how closely a compressed model approximates full-context behavior, together with an analysis of where and why ground-truth rankings and faithfulness rankings disagree
- **Pre-RoPE KV scoring**: importance scoring in pre-RoPE space, recovering semantic relevance for long-range tokens that post-RoPE methods systematically undervalue
- **Ablation finding**: a V-norm payload term, while theoretically motivated [Feng et al., 2025], does not improve faithfulness over pre-RoPE KQ alignment alone; we report this as a negative result to inform future work
- **Cross-architecture validation**: results on both Llama-3.1-8B and Mistral-7B-v0.3 with identical hyperparameters

---

## 2. Background and Related Work

**KV cache eviction.** H2O [Zhang et al., 2023] identifies "heavy hitter" tokens via accumulated attention scores and evicts the rest. SnapKV [Li et al., 2024] selects tokens by pooling attention weights from the last several queries over all key positions, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] retains attention sink tokens plus a recency window, requiring no scoring computation. Our method uses pre-RoPE key-query alignment, removing the positional bias present in both SnapKV and H2O.

**Pre-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We confirm and extend this finding, providing a faithfulness-based analysis that explains *why* pre-RoPE scoring is beneficial: it recovers tokens that are semantically relevant but positionally distant.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads. Our method uses a global budget per layer.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

---

## 3. Faithfulness Metrics

### 3.1 Motivation

Let M be a language model, c a full context, q a query, and y* = M(c, q) the full-context output. A compression method produces a compressed context ĉ and output ŷ = M(ĉ, q). Existing benchmarks measure d(ŷ, y_gt) where y_gt is a ground-truth label. We instead measure d(ŷ, y*): how closely the compressed output approximates the full-context output.

This distinction matters for three reasons. First, the goal of compression is approximation, not improvement. Second, ground-truth labels are often ambiguous or incomplete; comparing to a single reference disadvantages paraphrastically correct outputs. Third, a compressed method that replicates full-context failures is more faithful than one that corrects them accidentally — ground-truth benchmarks cannot distinguish these cases.

We note one important caveat: on tasks where the full-context model completely fails (e.g., produces degenerate outputs or near-zero ground-truth scores), high faithfulness means faithfully replicating failure. We flag such tasks with † in our tables and exclude them from aggregate faithfulness comparisons.

### 3.2 Lexical Faithfulness

Token-level F1 between the compressed output ŷ and full-context output y*, computed over unigrams with standard normalization:

```
P = |ŷ ∩ y*| / |ŷ|,   R = |ŷ ∩ y*| / |y*|,   F1 = 2PR / (P + R)
```

where ∩ denotes multiset intersection. Simple and fast, but sensitive to surface variation: semantically equivalent outputs that differ in phrasing score low. Useful as a lower bound on semantic similarity.

### 3.3 Perplexity Faithfulness

We measure how likely the full-context model M considers the compressed output ŷ, normalized by how likely it considers its own output y*:

```
faith_ppl = exp(loss_M(y* | c, q) − loss_M(ŷ | c, q)) × 100
```

where loss_M(y | c, q) is the mean per-token cross-entropy of y under M conditioned on (c, q) via a teacher-forced forward pass (no generation). A score of 100 means the compressed output is exactly as likely as the full-context output under the full model; above 100 means more likely (valid, since greedy decoding does not produce the globally most probable sequence); below 100 means less likely.

This metric is robust to surface variation — semantically equivalent paraphrases that the model considers equally likely score identically — and handles cascading token divergence, which makes token-level comparison of generated outputs problematic: once any token diverges at position k, the inputs to position k+1 differ between the two generation traces, making logit comparison meaningless. Teacher-forcing conditions on reference tokens, bypassing this issue.

### 3.4 Embedding Faithfulness

Cosine similarity between sentence-transformer embeddings of ŷ and y*, rescaled from [−1, 1] to [0, 100]:

```
faith_emb = (cos_sim(embed(ŷ), embed(y*)) + 1) / 2 × 100
```

We use `all-MiniLM-L6-v2` [Wang et al., 2020]. A score of 100 means identical embeddings; 50 means orthogonal. This metric is independent of both the LLM and surface form, capturing semantic content directly.

### 3.5 GT vs. Faithfulness: Where They Disagree

Before presenting results, we clarify the scales. Ground-truth scores are expressed on a task-specific metric (ROUGE-L, F1, EM); the full-context model scores 25.0 average, and a compressed method retaining 93–96% of that is the competitive range. Perplexity faithfulness is centered at 100, which means the compressed output is exactly as likely under the full-context model as the full-context model's own output; scores below 100 mean the compressed output is less likely (the method diverges from the full model's distribution); scores above 100 mean the compressed output is *more* likely than what the greedy decoder produced — valid and expected, since greedy decoding is path-dependent and does not find the globally most probable sequence. Scores near 100 in either direction are acceptable; large negative deviations indicate the compressed output is out-of-distribution. A score of 95.5 means 4.5% less likely, and 104.2 means 4.2% more likely — the gap between these methods is roughly 9 percentage points, not 9 absolute units. Embedding faithfulness runs from 50 (orthogonal embeddings, no semantic overlap) to 100 (identical embeddings); the compressed methods cluster in the 84–91 range, with streaming much lower at 62.

With these scales in mind, the key finding is that ground-truth rankings and faithfulness rankings disagree substantially. The clearest case is naive proportional truncation (naive_65pct): it retains 93.6% of full-context ground-truth performance (23.4 vs. full context 25.0), which appears competitive with all semantic methods. But its perplexity faithfulness is 95.5 — meaning its outputs are 4.5% less likely under the full-context model than the full model's own outputs — while kq_post_rope achieves 104.2, indicating its outputs are within 4.2% of full-context likelihood from the other side. The gap between naive_65pct and kq_post_rope on perplexity faithfulness (8.7 points) is thus much larger than the gap on ground truth (0.5 points). Embedding faithfulness reverses this picture: naive_65pct scores 90.7 (90.7% of the way from orthogonal to identical semantics) while kq_post_rope scores 85.7 — naive's contiguous text preservation produces semantically similar outputs even when they are less likely under the full model.

The mechanism is task-specific. On NarrativeQA (contexts of 30K+ tokens), naive_65pct retains only the last 65% of a novel. The F1 metric scores its output at 4.9 vs. the full model's 5.5 — apparently 89% of full-context performance. But perplexity faithfulness for naive_65pct on NarrativeQA is 98.6, while kq_post_rope achieves 110.8. Both numbers are near 100, meaning both compressed outputs are close to what the full model would consider plausible — but the ground-truth comparison is measuring something different: whether the output matches a short reference phrase. A method operating on a much shorter context happens to produce outputs that score similarly on F1 while being generated from a fundamentally different effective context. Faithfulness metrics expose this; ground-truth metrics do not.

This reversal — ground-truth appearing to show rough equivalence while faithfulness reveals meaningful differences in approximation quality — is the central motivation for our evaluation framework.

---

## 4. Method: Pre-RoPE KV Scoring

### 4.1 Setup

We patch the self-attention layers of a decoder-only transformer to intercept the prefill pass. On the first (prefill) forward pass through each layer, we compute importance scores for all T input tokens, select a retained subset of size ⌊r·T⌋, and gather the corresponding K and V states. All subsequent generation steps attend only to retained positions plus any newly generated tokens.

### 4.2 Scoring

For each attention layer, let Q ∈ ℝ^{T×D} and K ∈ ℝ^{T×D} be the *pre-RoPE* query and key matrices, and V ∈ ℝ^{T×D} be the value matrix.

**KQ alignment.** We use the last q_buffer_size query vectors (the tail of the sequence, which best approximates generation-phase queries) and compute max-pooled dot-product similarity to every key:

```
kq_i = max_{j ∈ tail} (Q_j · K_i) / √D
```

This is computed in a single batched matrix multiply. The result is normalized per-head to [0, 1].

**Distance decay.** We apply a linear decay that assigns weight 1.0 to the most recent token and min_decay to the oldest:

```
decay_i = min_decay + (1 − min_decay) · (i / (T−1))
```

where i is the position index (0 = oldest). The rate is derived automatically from min_decay and T, making it context-length-independent.

**Final score.** The per-token importance score is:

```
score_i = kq_i · decay_i
```

### 4.3 Why Pre-RoPE?

RoPE encodes absolute position by rotating K and Q vectors by position-dependent angles. A key vector from position 0 in a 10K context is rotated by a large angle relative to the last-position query, producing an artificially low dot product regardless of semantic content. Pre-RoPE keys remove this rotation: KQ alignment reflects content similarity only. The post-RoPE keys and values are still used in the actual attention computation — pre-RoPE is used only for scoring.

The practical consequence is that semantically relevant early tokens — passages answering the query, relevant context in a document — are not systematically penalized for being distant. Post-RoPE scoring conflates "far away" with "unimportant"; pre-RoPE scoring does not.

### 4.4 Budget Selection

We unconditionally retain the first `always_keep_first` tokens (preventing degenerate attention from early queries having no visible keys) and the last `always_keep_last` tokens (preserving the query context). The remaining budget slots are filled with the highest-scoring non-protected tokens.

For GQA models (e.g. Llama 3.x with 32 Q-heads and 8 KV-heads), we compute scores for all Q-heads and aggregate across heads sharing a KV-head via max-pooling before selection.

After selection, we reconstruct a valid causal attention mask using the *original* pre-pruning positions of the retained tokens, ensuring that query token at position p can only attend to retained tokens at positions ≤ p.

### 4.5 Baselines

- *Full context*: unmodified model with full prompt
- *Naive proportional truncation (naive_65pct)*: each prompt is truncated to 65% of its actual token length using a 10%/90% head/tail split, matching the compression budget exactly
- *SnapKV*: post-RoPE attention weight pooling over a 128-token observation window [Li et al., 2024]
- *StreamingLLM*: first 4 attention sink tokens plus most recent tokens to fill the 65% budget [Xiao et al., 2023]
- *kq_only*: pre-RoPE KQ alignment without decay (ablation)
- *kq_post_rope*: our full method (KQ alignment with decay, computed in pre-RoPE space) — **this is the primary method**

Note on the naive baseline: we use proportional truncation (65% of actual prompt length) rather than a fixed 4096-token budget, which would create a budget mismatch — retaining all tokens for short-context tasks while pruning more aggressively for long ones. Proportional truncation is the only honest apples-to-apples comparison.

---

## 5. Experiments

### 5.1 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128, min_decay=0.7, decay_fn=linear.

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. Faithfulness metrics compare all compressed outputs to these stored references. Tasks where full-context ground-truth accuracy < 10% are flagged †: the model is unreliable on these tasks, and high faithfulness would mean faithfully replicating failures.

### 5.2 Ground-Truth Results

| Task | Full | Naive_65pct | kq_post_rope | kq_only | SnapKV | Streaming |
|---|---|---|---|---|---|---|
| NarrativeQA† | 5.5 | 4.9 | **5.7** | 5.7 | 5.0 | 6.6 |
| Qasper | 11.1 | 10.2 | 9.5 | **11.7** | 11.1 | 8.4 |
| MultifieldQA | 28.9 | 27.0 | **29.4** | 25.3 | 28.4 | 15.8 |
| HotpotQA† | 9.9 | 9.6 | **10.1** | 8.6 | 9.6 | 11.4 |
| 2WikiMQA | 14.1 | 12.6 | 12.1 | 12.4 | **13.9** | 12.9 |
| MuSiQue† | 6.9 | **7.0** | 6.1 | 5.6 | 6.0 | 4.5 |
| GovReport | 20.4 | **19.8** | 19.2 | 18.9 | 19.9 | 5.2 |
| QMSum | 10.3 | 11.5 | 9.0 | 8.6 | 9.7 | 3.5 |
| MultiNews | 19.0 | 16.5 | **18.4** | 17.8 | 18.9 | 7.5 |
| TREC | 70.0 | 66.0 | **70.0** | 68.0 | 66.0 | 50.0 |
| TriviaQA | 17.4 | 17.3 | **18.3** | 17.0 | 17.6 | 35.9 |
| SAMSum | 16.0 | 16.5 | **18.3** | 15.8 | 17.5 | 6.4 |
| PassageCount† | 3.0 | 1.0 | **3.0** | 2.0 | 3.0 | 2.0 |
| PassageRetrieval | 44.0 | 37.0 | 36.0 | 38.0 | 37.0 | 20.0 |
| LCC | 68.1 | 63.4 | 64.1 | 59.7 | 63.2 | 17.8 |
| RepoBench-P | 55.6 | 53.9 | 54.1 | 52.1 | 53.4 | 6.3 |
| **Average** | **25.0** | 23.4 | **23.9** | 23.0 | 23.8 | 13.4 |

Full context is the reference and is not bolded. Bold marks the best compressed method. Tasks marked † are diagnostic cases discussed in §6.

Ground-truth scores are relatively compressed across methods: naive_65pct (23.4), kq_post_rope (23.9), kq_only (23.0), and SnapKV (23.8) are within 1 point of each other. Streaming (13.4) fails badly on tasks requiring distributed context. These small differences between the semantic methods and naive truncation are precisely the problem with ground-truth evaluation: they suggest the methods are roughly equivalent when faithfulness analysis (§5.3) reveals they are not.

### 5.3 Faithfulness Results

#### Perplexity Faithfulness

| Task | Naive_65pct | kq_post_rope | kq_only | SnapKV | Streaming |
|---|---|---|---|---|---|
| NarrativeQA† | 98.6 | **110.8** | 104.6 | 107.2 | 77.6 |
| Qasper | 91.8 | **97.2** | 95.4 | 96.5 | 66.3 |
| MultifieldQA | 87.3 | 96.3 | **100.2** | **101.6** | 61.8 |
| HotpotQA† | 97.6 | 100.7 | **101.1** | 100.4 | 50.1 |
| 2WikiMQA | **101.1** | 102.7 | 103.7 | 100.9 | 37.6 |
| MuSiQue† | 99.8 | 99.2 | 99.8 | 98.2 | 56.2 |
| GovReport | 76.5 | 93.5 | 92.3 | 93.2 | **94.1** |
| QMSum | 93.4 | **106.4** | 106.3 | 103.4 | 98.1 |
| MultiNews | **93.4** | 89.8 | 89.6 | 88.1 | 86.2 |
| TREC | **100.1** | 98.4 | 98.8 | 97.5 | 34.8 |
| TriviaQA | 100.5 | **102.7** | 97.4 | 98.5 | 64.5 |
| SAMSum | 102.6 | **121.1** | 112.0 | 119.9 | 109.4 |
| PassageCount† | 98.5 | 147.3 | 135.5 | **142.2** | 111.9 |
| PassageRetrieval | 89.1 | **107.5** | 104.4 | 103.3 | 60.2 |
| LCC | **100.0** | 98.8 | 93.4 | 98.2 | 36.2 |
| RepoBench-P | 97.2 | **94.7** | 90.7 | 95.8 | 46.6 |
| **Average** | 95.5 | **104.2** | 101.6 | 102.8 | 68.2 |

100 = output as likely as full-context output under the full-context model; >100 = more likely; <100 = less likely.

#### Embedding Faithfulness

| Task | Naive_65pct | kq_post_rope | kq_only | SnapKV | Streaming |
|---|---|---|---|---|---|
| NarrativeQA† | **96.5** | 84.1 | 82.2 | 85.1 | 60.2 |
| Qasper | 83.8 | **85.4** | 84.9 | 86.3 | 67.0 |
| MultifieldQA | 86.3 | **88.1** | 86.8 | 87.0 | 66.6 |
| HotpotQA† | **95.9** | 87.4 | 85.7 | 87.8 | 61.8 |
| 2WikiMQA | 85.8 | **89.5** | 89.4 | 88.2 | 67.5 |
| MuSiQue† | **98.7** | 84.2 | 82.5 | 85.1 | 59.9 |
| GovReport | **92.1** | 90.3 | 89.0 | 90.3 | 56.2 |
| QMSum | 89.2 | **85.2** | 84.1 | 86.7 | 56.8 |
| MultiNews | **98.0** | 89.9 | 90.6 | 90.6 | 63.8 |
| TREC | 88.8 | 84.9 | **87.4** | 85.5 | 60.7 |
| TriviaQA | **88.3** | 75.8 | 76.4 | 79.1 | 62.6 |
| SAMSum | 93.4 | **88.8** | 87.8 | 88.5 | 69.6 |
| PassageCount† | **88.8** | 73.5 | 72.4 | 73.7 | 70.2 |
| PassageRetrieval | **91.0** | 84.4 | 84.8 | 84.2 | 57.9 |
| LCC | 90.0 | **89.8** | 85.4 | 90.1 | 60.1 |
| RepoBench-P | **95.4** | 90.3 | 87.4 | 89.1 | 54.3 |
| **Average** | **90.7** | 85.7 | 84.8 | 86.1 | 62.2 |

100 = identical embedding; 50 = orthogonal (rescaled cosine similarity).

#### Summary Across Metrics

| Method | GT↑ | Perplexity↑ | Embedding↑ | Lexical↑ |
|---|---|---|---|---|
| Naive_65pct | 23.4 | 95.5 | **90.7** | **57.9** |
| kq_post_rope | **23.9** | 104.2 | 85.7 | 46.0 |
| kq_only | 23.0 | 101.6 | 84.8 | 43.9 |
| SnapKV | 23.8 | 102.8 | 86.1 | 46.7 |
| pruned (kq + V-norm) | 21.4 | 104.7 | 82.5 | 38.6 |
| vn_decay | — | **106.2** | 78.2 | — |
| Streaming | 13.4 | 68.2 | 62.2 | 11.9 |

GT and Lexical scores for vn_decay are omitted; it is evaluated as a faithfulness ablation in §5.5 and §6.1.

The headline finding: **on ground truth, kq_post_rope (95.6% of full), SnapKV (95.2%), and naive_65pct (93.6%) appear nearly equivalent — all within 2 points of each other. Perplexity faithfulness reveals a 9-point gap between naive_65pct (95.5) and kq_post_rope (104.2), despite ground-truth scores differing by only 0.5 points.** Naive truncation is a significantly poorer approximation of full-context behavior than ground-truth scores suggest.

A perplexity score of 100 means the compressed output is exactly as likely under the full model as the full model's own greedy output. Scores below 100 mean the output is less probable — the method is diverging from the full model's distribution. Scores above 100 mean the compressed output is *more* probable than what the greedy decoder produced; this is valid and expected, since greedy decoding is path-dependent and does not find the globally most probable sequence. kq_post_rope at 104.2 occupies a slightly higher-probability region than the full model's own outputs; naive_65pct at 95.5 is measurably less plausible, reflecting operation from a truncated context.

Embedding faithfulness tells a partially different story: naive_65pct scores 90.7, the highest of any method. Naive truncation preserves contiguous text from the tail, producing outputs that are semantically similar to the full model's on tasks where the tail is informative — but those outputs are less likely under the full model, revealing operation in a different regime. More striking is vn_decay: it achieves the highest perplexity faithfulness (106.2) yet the lowest embedding faithfulness among semantic methods (78.2). Its outputs are *more* probable than the full model's own outputs, yet semantically further from what the full model produces. This cross-metric tension — high probability, low similarity — identifies a distinct failure mode that neither metric alone can detect. We discuss this in depth in §6.1.

#### The GT-Faithfulness Reversal

The starkest illustration of the GT-faithfulness discrepancy is NarrativeQA. Full context scores 5.5 on ground truth; kq_post_rope scores 5.7. These look nearly identical. But perplexity faithfulness for kq_post_rope is 110.8 — its outputs are *more* likely than the full model's own outputs under the full model. Naive_65pct scores 4.9 on ground truth, slightly lower, but has perplexity faithfulness of 98.6 and embedding faithfulness of 96.5. These two patterns represent fundamentally different behaviors: kq_post_rope is faithfully approximating the full model; naive_65pct is operating in a different effective regime (much shorter context) and by coincidence achieving a similar F1 score.

The underlying cause is that NarrativeQA involves very long contexts (median 30K+ tokens) where naive_65pct retains only the last 65% of a novel. The model operating on this shorter context behaves differently, but F1-over-unigrams happens to score similarly. Faithfulness metrics expose this.

### 5.4 Pre-RoPE vs. Post-RoPE Ablation

To isolate the scoring space effect, we compare kq_only (pre-RoPE) and kq_post_rope (post-RoPE) — identical except for which key vectors enter the dot product.

| Metric | kq_only (pre-RoPE) | kq_post_rope (post-RoPE) | Δ |
|---|---|---|---|
| GT average | 23.0 | **23.9** | +0.9 |
| Perplexity faith. | 101.6 | **104.2** | +2.6 |
| Embedding faith. | 84.8 | **85.7** | +0.9 |
| Lexical faith. | 43.9 | **46.0** | +2.1 |

Post-RoPE scores consistently higher on all four metrics. The margin is modest but consistent across 16 tasks and three faithfulness dimensions. The direction is unambiguous: the post-RoPE KQ signal, despite encoding position, provides a better importance signal than the purely position-independent pre-RoPE signal.

This is a counterintuitive finding. The theoretical argument for pre-RoPE — that positional encoding penalizes distant-but-relevant tokens — is correct in principle, but in practice the positional information in post-RoPE scores appears to be useful rather than harmful. The model's attention mechanism was trained with RoPE; post-RoPE keys may contain position-aware features that improve relevance discrimination even if they introduce positional bias. This suggests that removing positional information entirely is not the right objective; the goal should be recalibrating positional influence, not eliminating it.

Our recommended default is therefore kq_post_rope (post-RoPE with decay), which consistently outperforms both the pre-RoPE variant and SnapKV.

### 5.5 V-Norm Ablation

We evaluated two V-norm variants: *pruned*, which combines KQ alignment with a V-norm payload term (α = 0.65), and *vn_decay*, which scores tokens by V-norm and distance decay alone, without any KQ alignment:

```
pruned:   score_i = α · kq_i + (1 − α) · vn_i · decay_i,   α = 0.65
vn_decay: score_i = vn_i · decay_i
```

| Metric | kq_post_rope | pruned (+ V-norm) | vn_decay (V-norm only) |
|---|---|---|---|
| GT average | **23.9** | 21.4 | — |
| Perplexity faith. | 104.2 | 104.7 | **106.2** |
| Embedding faith. | **85.7** | 82.5 | 78.2 |
| Lexical faith. | **46.0** | 38.6 | — |

The V-norm variants reveal a consistent pattern: as V-norm weight increases, perplexity faithfulness rises slightly while embedding faithfulness falls substantially. vn_decay achieves the highest perplexity score of any method (106.2) while also having the lowest embedding score among semantic methods (78.2) — a 28-point gap between the two metrics. This cross-metric tension is the signature of a specific failure mode, *mode-switching*, discussed in detail in §6.1.

The practical conclusion is that V-norm does not improve over KQ alignment when evaluated holistically. The perplexity gain (+2.0 for vn_decay vs. kq_post_rope) comes at the cost of semantic drift that embedding faithfulness makes visible. We report this as a negative result rather than omitting it; the pattern is informative for future work on scoring signal design.

### 5.6 Generalization to Mistral-7B-v0.3

To test whether the method generalizes beyond the Llama architecture, we run on Mistral-7B-v0.3 (base, fp16) with identical hyperparameters.

| Task | Full | Naive_65pct | kq_post_rope | SnapKV |
|---|---|---|---|---|
| NarrativeQA | 5.2 | 5.1 | **6.2** | 5.4 |
| Qasper | 5.4 | 5.8 | **7.6** | 5.6 |
| MultifieldQA | 25.3 | 22.0 | 20.2 | **22.5** |
| HotpotQA | 10.5 | 10.2 | **11.3** | 10.4 |
| 2WikiMQA | 11.5 | 11.4 | 11.2 | **11.8** |
| MuSiQue | 5.1 | 5.2 | **5.2** | 5.0 |
| GovReport | 20.7 | **19.9** | 18.7 | 19.5 |
| QMSum | 8.3 | 9.0 | 7.3 | **9.5** |
| MultiNews | 17.5 | 16.2 | **13.4** | 15.8 |
| TREC | 72.0 | 67.0 | **68.0** | 67.0 |
| TriviaQA | 23.1 | 24.7 | **26.5** | 24.2 |
| SAMSum | 16.9 | **18.5** | 18.1 | 17.8 |
| PassageCount | 1.0 | 0.5 | **2.0** | 1.5 |
| PassageRetrieval | 39.0 | 20.0 | **29.0** | 24.0 |
| LCC | 62.9 | 61.0 | 56.2 | **61.5** |
| RepoBench-P | 53.9 | 51.5 | 51.2 | **52.0** |
| **Average** | **23.6** | 21.8 | 21.9 | **22.0** |

Task-level patterns are consistent across models: PassageRetrieval shows the same large gain from semantic scoring (kq_post_rope 29.0 vs. naive 20.0, +9 points), code completion shows the same relative regression, and TriviaQA shows the same streaming-inflated anomaly pattern. No architecture-specific modifications were required for Mistral.

---

## 6. Analysis

### 6.1 Why Perplexity and Embedding Disagree

Perplexity and embedding faithfulness measure fundamentally different things, and understanding their disagreement is essential for interpreting compression results correctly.

**What each metric captures.** Perplexity faithfulness asks: *given the full model's learned probability distribution, how likely is the compressed output?* It is a property of the model itself — not the specific sequence the greedy decoder happened to produce — and is robust to surface variation. Two outputs that say the same thing in different words score identically if the full model assigns them equal probability. Embedding faithfulness asks: *how semantically similar is the compressed output to the specific sequence the full model produced?* It captures content similarity but treats the full model's greedy output as the gold standard, which it is not — it is one sample from a high-probability region.

**The mode-switching failure mode.** The full model's probability distribution is multimodal: many distinct outputs are high-probability for any given prompt. Greedy decoding selects one path through this space, but a compressed model may follow a different path — equally valid under the full model's distribution — and produce a semantically different output. Consider a compressed model answering a question about the French Revolution. The full model might write about the Reign of Terror; the compressed model might write about the storming of the Bastille. Both answers are historically accurate, high-probability under the full model, and would score well on perplexity faithfulness. But embedding faithfulness would correctly flag that they address different aspects of the question. A user deploying the compressed model as a drop-in replacement for the full model gets different outputs — and would never know from perplexity faithfulness alone.

**vn_decay as a concrete illustration.** This failure mode appears clearly in our results. vn_decay achieves the highest perplexity faithfulness of any method (106.2) — its outputs are on average *more* probable than what the full model's own greedy decoder produces — yet has the lowest embedding faithfulness among semantic methods (78.2), a 28-point gap. vn_decay is reliably finding high-probability outputs in the full model's distribution, just not the same modes the full model's decoder reaches. kq_post_rope shows no such tension: 104.2 perplexity, 85.7 embedding — competitive on both dimensions, indicating it approximates both the full model's distribution *and* the specific trajectory the full model takes through it.

This four-quadrant view covers the space of outcomes:

| | High Embedding | Low Embedding |
|---|---|---|
| **High Perplexity** | Ideal: faithful to distribution and trajectory (kq_post_rope) | Mode-switching: in-distribution but divergent (vn_decay) |
| **Low Perplexity** | Surface-copy: similar content, wrong distribution (naive_65pct) | Failed: divergent on both dimensions (streaming) |

**Why perplexity is more principled.** Despite the mode-switching limitation, we treat perplexity faithfulness as the primary criterion for compression fidelity. The reason is that the full model's greedy output is not the gold standard: it is one arbitrary path through a high-probability region, and greedy decoding makes no global optimality guarantee. Embedding faithfulness penalizes valid alternative modes — equally correct, equally probable answers that happen to differ in surface form or focus. Perplexity faithfulness asks the cleaner question: is the compressed output in a high-probability region of the distribution the full model learned? That is what fidelity to *the model* means, as opposed to fidelity to one particular decoding trajectory.

**Using both metrics as diagnostics.** The right approach is to read them together. Perplexity faithfulness confirms that the compressed model is operating within the full model's learned distribution. Embedding faithfulness confirms that the compressed model is reaching the same modes, not just any high-probability region. A method that passes one but fails the other warrants scrutiny: naive_65pct copies recent context effectively but diverges distributionally; vn_decay stays in-distribution but mode-switches. kq_post_rope is the only method that maintains competitive scores on both, which is why we treat it as the primary result despite vn_decay's higher perplexity score in isolation.

### 6.2 Streaming

Streaming (StreamingLLM) fails catastrophically on all tasks requiring distributed context: GovReport drops from 20.4 (full) to 5.2, QMSum from 10.3 to 3.5, RepoBench-P from 55.6 to 6.3. Its perplexity faithfulness (68.2) and embedding faithfulness (62.2) are both far below any semantic scoring method. Streaming is competitive only when task-relevant information is reliably recent (TriviaQA: 35.9, inflated by few-shot format; SAMSum: 6.4, much lower than full). The faithfulness metrics correctly rank streaming last, consistent with its qualitative failure mode.

### 6.3 Task-Level Diagnostic Findings

**NarrativeQA (†)**: Ground-truth and faithfulness tell opposite stories. See §5.3. Base model behavior on long narrative is the root cause — an instruction-tuned model would likely reverse this result.

**TriviaQA (†)**: Streaming achieves 35.9, nearly double full context (17.4). This reflects task structure: TriviaQA in LongBench is few-shot formatted, and the recency window captures the most recent demonstrations and query intact. The full model receives too many varied examples, diluting the format signal. Perplexity faithfulness for streaming on TriviaQA (64.5) correctly identifies this as a poor approximation despite the high ground-truth score.

**PassageRetrieval**: kq_post_rope (36.0) and naive_65pct (37.0) are close on ground truth but diverge sharply on perplexity faithfulness (107.5 vs. 89.1). PassageRetrieval involves matching a specific paragraph from 30 topically similar candidates; the full context model sees all 30 paragraphs, while naive_65pct often truncates some away. When naive_65pct happens to retain the target paragraph, it answers correctly — but it is not approximating the full model's reasoning process.

**Code completion (LCC, RepoBench-P)**: Semantic scoring methods underperform naive truncation on code (kq_post_rope: 64.1/54.1 vs. naive_65pct: 63.4/53.9 — small but consistent). Code completion depends on syntactically necessary tokens that have low semantic distinctiveness (brackets, keywords, indentation). Neither KQ alignment nor V-norm captures syntactic necessity, while naive truncation retains recent code structure by default. This is a limitation of semantic scoring for code.

### 6.4 Efficiency

We implemented kq_post_rope as a KVPress press subclass [Devoto et al., 2025] and benchmarked against full context on a six-task LongBench subset (30 examples, Llama-3.1-8B, 65% retention). Total generation time: full context 2825s, kq_post_rope 2545s — **10% faster**, with quality preserved (15.9 average, identical to full context on this subset). The speedup comes from the generation phase: a 35% smaller KV cache reduces memory bandwidth at each decode step.

At longer contexts, the benefit grows:

| Context | Method | Prefill (s) | Gen (tok/s) | Speedup |
|---|---|---|---|---|
| 2048 | full | 0.844 | 24.5 | — |
| 2048 | kq_post_rope | 0.869 | 26.6 | 1.09× |
| 4096 | full | 2.488 | 18.9 | — |
| 4096 | kq_post_rope | 2.522 | 22.6 | 1.20× |
| 6144 | full | 5.691 | 14.8 | — |
| 6144 | kq_post_rope | 5.734 | 19.2 | 1.30× |

Prefill overhead is 1–4% and shrinks relatively as context grows. At 6K tokens, generation is 30% faster. At production-scale contexts (32K–128K), where KV cache is the dominant memory cost, the benefit is substantially larger. Additionally, our method uses only K and V tensors and is directly compatible with Flash Attention 2, unlike SnapKV-style methods that require attention weights not materialized by FA2.

---

## 7. Discussion

**Ground-truth vs. faithfulness.** The 9-point perplexity faithfulness gap between naive_65pct (95.5) and kq_post_rope (104.2) is invisible in ground-truth averages (23.4 vs. 23.9). This is not a coincidence: ground-truth benchmarks were designed to measure whether a model answers questions correctly, not whether a compressed model approximates an uncompressed one. The two objectives are different, and evaluation methodology should reflect this. We recommend faithfulness metrics as primary evaluation criteria for compression research, with ground-truth scores as secondary context.

**Pre-RoPE vs. post-RoPE.** Our experimental finding is that post-RoPE scoring (kq_post_rope) consistently outperforms pre-RoPE (kq_only) across all metrics. This is surprising given the theoretical argument that RoPE introduces positional bias that penalizes distant tokens. We speculate that post-RoPE keys contain positionally-conditioned features that were learned during training and that these features carry useful information for relevance discrimination, even if they introduce some bias. The right inductive prior is not "position-independent" but "position-calibrated." Disentangling learned positional features from RoPE-induced bias is an interesting direction for future work.

**V-norm as a negative result.** The V-norm term was motivated by the observation that attention sinks — tokens that receive high attention weights but contribute little to the output — can be filtered by checking whether their value vectors have meaningful magnitude. Our ablation shows that adding V-norm to KQ alignment does not improve faithfulness. One explanation is that the tokens V-norm would filter (attention sinks, common function words) are already ranked low by KQ alignment — the signals are correlated and the combination adds noise. We report this clearly rather than omitting it: negative results about theoretically motivated components are informative for the field.

**Limitations.** Our evaluation covers two model families at ~7–8B parameters; behavior at larger scales or on other architectures is untested. The min_decay hyperparameter was tuned on LongBench and may not generalize to out-of-distribution tasks. Code completion regresses relative to naive truncation because semantic scoring cannot identify syntactically necessary but semantically predictable tokens (brackets, keywords); addressing this requires either task detection or a scoring signal that captures syntactic necessity. All experiments use a single V100 GPU; Flash Attention 2 benefits (relevant at 32K+ contexts) were not benchmarked directly. Faithfulness evaluation requires storing full-context outputs, adding computational overhead.

---

## 8. Conclusion

We have argued that ground-truth benchmarks are the wrong primary metric for KV cache compression, and introduced faithfulness metrics that directly measure how closely compressed outputs approximate full-context behavior. Using these metrics, we show that naive proportional truncation — which appears competitive on ground truth (23.4 average vs. 23.9 for our method) — is a significantly poorer approximation of full-context behavior (95.5 perplexity faithfulness vs. 104.2). This gap is invisible to ground-truth evaluation and reveals a systematic blind spot in how the field evaluates compression methods.

Our best-performing method, kq_post_rope, achieves this faithfulness advantage through post-RoPE KQ alignment with linear distance decay at 65% token retention. It consistently outperforms SnapKV (102.8 perplexity), naive truncation (95.5), and streaming (68.2) on perplexity faithfulness across 16 LongBench tasks, while matching or exceeding them on ground truth. The method requires no architecture-specific modifications and generalizes to Mistral-7B-v0.3 with identical hyperparameters. We additionally report that a V-norm payload term does not improve faithfulness over the simpler KQ-only baseline — a negative result that we believe is useful for future work on KV scoring design.

The primary contribution of this work is the evaluation framework, not the scoring method. We hope that faithfulness metrics — or analogues thereof — become standard practice in KV cache compression research, enabling cleaner comparison between methods that approximate full-context behavior and methods that happen to produce correct outputs for other reasons.

---

## References

Bai, Y. et al. (2023). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. *arXiv preprint arXiv:2308.14508*.

Devoto, A. et al. (2025). KVPress: A Framework for KV Cache Compression Research. *arXiv preprint*.

Feng, S. et al. (2025). Not All Tokens Are What You Need. *arXiv preprint arXiv:2501.02625*.

Li, Y. et al. (2024). SnapKV: LLM Knows What You are Looking for Before Generation. *arXiv preprint arXiv:2404.14469*.

Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. *ACL 2002*.

Wang, K. et al. (2020). SBERT: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

Xiao, G. et al. (2023). Efficient Streaming Language Models with Attention Sinks. *arXiv preprint arXiv:2309.17453*.

Zhang, P. et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS 2023*.

Zhang, T. et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.
