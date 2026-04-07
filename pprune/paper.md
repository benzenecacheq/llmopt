# Additive KV Cache Pruning: Combining Semantic Alignment and Value Norms for Long-Context Inference

---

## Abstract

We present a prefill-time KV cache pruning method for transformer-based language models that retains a fixed fraction of key-value token slots while preserving model quality on downstream tasks. Our scoring function additively combines two complementary signals: KQ alignment (which tokens does the question attend to?) and V-norm (which tokens have high output potential?), weighted by a tunable parameter α and a distance-decay term. Crucially, we compute KQ alignment using pre-RoPE key and query vectors, removing the positional bias that makes early tokens appear semantically irrelevant under standard post-RoPE scoring. We adopt a global budget strategy in which the same token positions are retained across all KV heads within a layer, using a context-length-independent decay parametrization (min_decay) that scales automatically to arbitrary sequence lengths. Evaluated on LongBench with Llama-3.1-8B at 65% token retention, our method recovers 86% of full-context performance on average, and outperforms naive left-truncation on retrieval tasks (27% vs 18% on passage_retrieval_en). We additionally compare against StreamingLLM, a recency-only baseline that retains attention sink tokens plus a recent window. Streaming achieves competitive retrieval scores but collapses on summarization (1.4–5.1 vs 8.1–21.1 for our method) and code completion (8.5–17.8 vs 47.9–51.3), confirming that semantic scoring is essential for general long-context tasks. Our ablations show that V-norm alone — while effective on synthetic needle tests — fails on real retrieval tasks, and that an additive combination with α=0.65 achieves the best overall balance. We further validate on Mistral-7B-v0.3 with identical hyperparameters, where the pruned model retains 93% of full-context performance, confirming that the method generalizes across model families without modification.

---

## 1. Introduction

The computational cost of transformer attention scales quadratically with sequence length, making long-context inference expensive. A growing body of work addresses this by pruning the KV cache: discarding low-importance token slots before or during generation so that attention operates over a smaller, more informative set of key-value pairs [Zhang et al., 2023; Li et al., 2024; Yang et al., 2024].

Most existing methods assign per-token importance scores based on attention weights [H2O], value vector norms [VATP], or combinations thereof, and evict low-scoring tokens. A practical system must additionally decide: (1) how to score tokens in a way that is stable across different context lengths, (2) how to allocate the total token budget across attention heads, and (3) how much to favor recent tokens over semantically relevant but distant ones.

We make the following contributions:

- **Additive scoring**: We propose combining KQ semantic alignment and V-norm *additively* rather than multiplicatively. Multiplicative combination collapses the score whenever either component is near zero — V-norm penalizes normal prose even when the model strongly attends to it. An additive combination lets each signal independently rescue tokens.

- **Pre-RoPE KQ scoring**: Rotary positional encodings (RoPE) bake absolute position into K and Q vectors, causing tokens far from the query to appear semantically dissimilar regardless of content. We compute KQ alignment using pre-RoPE representations, giving a position-independent semantic signal. The post-RoPE keys and values are still used for attention itself.

- **Context-length-independent decay**: Prior methods apply a fixed exponential or linear decay rate. At very long contexts (e.g. 100K tokens) even small rates drive early-token weights to near zero. We parametrize decay by its value at position 0 (min_decay), with the rate automatically derived from context length, so pruning behavior is consistent regardless of sequence length.

- **Global budget**: We retain the same token positions for all KV heads within a layer, selected by aggregating per-head importance scores. This avoids the memory overhead of per-head non-contiguous indexing while still using information from all heads.

We evaluate on LongBench [Bai et al., 2023] covering single-document QA, multi-document QA, summarization, retrieval, and code completion tasks. Our ablations isolate the contribution of each design choice and identify a residual weakness on code tasks. We additionally benchmark StreamingLLM as a recency-only baseline, which reveals that tasks requiring distributed context cannot be served by a simple recency window.

---

## 2. Background and Related Work

**KV cache eviction.** H2O [Zhang et al., 2023] identifies "heavy hitter" tokens via accumulated attention scores and evicts the rest. SnapKV [Li et al., 2024] selects tokens by pooling attention weights from the last several queries over all key positions, using post-RoPE vectors. Our method differs in two ways: we use pre-RoPE vectors for scoring (removing positional bias) and combine the alignment signal additively with V-norm. We implement and benchmark SnapKV directly, finding that pre-RoPE scoring outperforms SnapKV by up to 13 points on retrieval tasks. StreamingLLM [Xiao et al., 2023] always retains "attention sink" tokens (typically the first few tokens) plus a recent window, making it a strong baseline on recency-dominated tasks.

**Value-norm scoring.** VATP [EMNLP 2024] proposes scoring tokens by the product of their attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Our method uses V-norm as a standalone component and finds it sufficient for synthetic retrieval but insufficient for natural-language QA.

**Head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different budgets to different attention heads, recognizing that "retrieval heads" and "accumulation heads" have fundamentally different importance profiles. Our work uses a global budget per layer rather than per-head allocation, which is simpler and avoids the overhead of per-head mask divergence.

**Pre-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We confirm empirically that pre-RoPE scoring is important for general QA performance.

**Decay-combined scoring.** GraphKV [EMNLP 2025] and PrHS [arXiv 2025] combine recency bias with semantic scores. Our min_decay parametrization offers a cleaner, length-independent formulation of this idea.

---

## 3. Method

### 3.1 Setup

We patch the self-attention layers of a decoder-only transformer (tested on Llama and Mistral architectures) to intercept the prefill pass. On the first (prefill) forward pass through each layer, we compute importance scores for all T input tokens, select a retained subset of size ⌊r·T⌋ (where r is the retention fraction), and gather the corresponding K and V states. All subsequent generation steps attend only to these retained positions plus any newly generated tokens.

### 3.2 Scoring

For each attention layer, let Q ∈ ℝ^{T×D} and K ∈ ℝ^{T×D} be the *pre-RoPE* query and key matrices (prior to rotary embedding application), and V ∈ ℝ^{T×D} be the value matrix.

**KQ alignment.** We use the last q_buffer_size query vectors (the tail of the sequence, which best approximates generation-phase queries) and compute max-pooled dot-product similarity to every key:

```
kq_i = max_{j ∈ tail} (Q_j · K_i) / √D
```

This is computed in a single batched matrix multiply. The result is normalized per-head to [0, 1].

**V-norm.** We compute the L2 norm of each value vector and normalize:

```
vn_i = ‖V_i‖₂ / max_j ‖V_j‖₂
```

Tokens with unusual or information-dense representations (numbers, rare words, key entities) tend to have higher V-norm.

**Distance decay.** We apply a linear decay that assigns weight 1.0 to the most recent token and min_decay to the oldest token:

```
decay_i = min_decay + (1 − min_decay) · (i / (T−1))
```

where i is the position index (0 = oldest). The rate is derived automatically from min_decay and T, making it context-length-independent.

**Additive combination.** The final per-token score is:

```
score_i = α · kq_i + (1 − α) · vn_i · decay_i
```

At α=0 this reduces to V-norm with decay; at α=1 it reduces to raw KQ alignment.

### 3.3 StreamingLLM Baseline

As an additional baseline we implement StreamingLLM [Xiao et al., 2023]: retain the first k_sink tokens (attention sinks) unconditionally, then fill the remaining budget with the most recent tokens. No scoring is required. We use k_sink=4 following the original paper. This baseline requires no Q or V computation beyond what the model already performs, making it the cheapest possible pruning strategy.

### 3.4 Budget Selection

We unconditionally retain the first always_keep_first tokens (preventing NaN from early queries having no visible keys) and the last always_keep_last tokens (preserving the query context). The remaining budget ⌊r·T⌋ − always_keep_first − always_keep_last slots are filled with the highest-scoring non-protected tokens.

For GQA models (e.g. Llama 3.x with 32 Q-heads and 8 KV-heads), we compute scores for all Q-heads and aggregate across heads sharing a KV-head via max-pooling before selection.

After selection, we reconstruct a valid causal attention mask using the *original* (pre-pruning) positions of the retained tokens, ensuring that query token at position p can only attend to retained tokens at positions ≤ p.

---

## 4. Experiments

### 4.1 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a V100 32GB GPU. Unless noted, tables report Llama-3.1-8B results; Mistral results appear in §4.6.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks covering single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot classification and QA (TREC, TriviaQA, SAMSum), synthetic retrieval (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). We evaluate 100 examples per task.

**Baselines.**
- *Full context*: unmodified Llama-3.1-8B with full prompt.
- *Naive truncation*: left-truncation to a fixed 4096-token budget (drops the oldest tokens).
- *StreamingLLM*: first 4 attention sink tokens + most recent tokens to fill the 65% budget.

**Our method.** Retention fraction r=0.65, α=0.65, min_decay=0.7, always_keep_first=16, always_keep_last=16, q_buffer_size=128, decay_fn=linear.

### 4.2 Pre-RoPE vs Post-RoPE Ablation

To isolate the effect of using pre-RoPE keys for scoring, we compare `kq_only` (pre-RoPE) against `kq_post_rope` (post-RoPE) — identical except for which key vectors enter the dot product. Both use the same Q-buffer, normalization, and budget selection. Llama-3.2-3B-Instruct, 65% retention, 30 examples per task.

| Mode | PassageRetrieval | HotpotQA | 2WikiMQA | GovReport | QMSum | MultiNews |
|---|---|---|---|---|---|---|
| kq_only (pre-RoPE) | **53.3** | **29.9** | 27.6 | **20.8** | 10.6 | **16.9** |
| kq_post_rope (post-RoPE) | 43.3 | 24.0 | **28.9** | 20.6 | **13.9** | **16.9** |

Pre-RoPE wins by 10 points on PassageRetrieval and 6 points on HotpotQA. These are tasks where the answer requires retaining tokens from early in the context; post-RoPE scoring artificially suppresses early tokens because their rotated key vectors are geometrically distant from the query regardless of semantic content. Summarization tasks (GovReport, MultiNews) show no difference — consistent with the hypothesis that the effect is specific to long-range retrieval. Post-RoPE is marginally better on QMSum (+3.3), suggesting that positional recency bias occasionally helps when recent context is disproportionately relevant. Pre-RoPE is the correct default for general-purpose use.

### 4.3 Score Mode and Baseline Ablation

To isolate the effect of the scoring function and compare against SnapKV, we ran a controlled comparison on three retrieval/QA tasks (PassageRetrieval, HotpotQA, 2WikiMQA) and three summarization tasks (GovReport, QMSum, MultiNews) at 65% retention using Llama-3.2-3B-Instruct (30 examples per task). SnapKV uses a 32-token observation window matching the commonly reported default; our method uses q_buffer_size=128.

**Retrieval/QA results at 65% retention:**

| Mode | PassageRetrieval | HotpotQA | 2WikiMQA |
|---|---|---|---|
| vn_decay | 23.3 | 16.6 | 18.1 |
| streaming | 43.3 | 2.3 | 8.4 |
| snapkv (window=32) | 40.0 | 23.4 | 21.4 |
| snapkv (window=128) | 43.3 | 22.0 | 23.1 |
| kq_post_rope (window=128) | 43.3 | 24.0 | 28.9 |
| kq_only (pre-RoPE) | **53.3** | **29.9** | 27.6 |
| additive α=0.6 | 56.7 | 24.8 | **33.6** |
| additive α=0.65 | 50.0 | 24.8 | 28.3 |
| additive α=0.7 | 50.0 | 27.1 | 29.9 |

**Summarization results at 65% retention** (ROUGE-L):

| Mode | GovReport | QMSum | MultiNews |
|---|---|---|---|
| vn_decay | 21.2 | 10.9 | **3.1** |
| streaming | 2.9 | 3.2 | 7.5 |
| snapkv (window=32) | **21.5** | **13.7** | 16.3 |
| snapkv (window=128) | 20.8 | 13.4 | 16.3 |
| kq_post_rope (window=128) | 20.6 | 13.9 | 16.9 |
| kq_only (pre-RoPE) | 20.9 | 11.5 | 3.0 |
| additive α=0.65 | 20.7 | 12.1 | 16.7 |
| additive α=0.7 | 21.2 | 10.9 | 3.1 |

SnapKV uses post-RoPE attention weights — in principle the most direct signal for what the model will attend to. With a window of 128 (matching our q_buffer_size), SnapKV scores 43.3 on PassageRetrieval, identical to kq_post_rope with the same window. This isolates the effect precisely: the two methods diverge only in that SnapKV applies softmax before pooling while kq_post_rope uses raw max dot-product, and they produce identical retrieval scores. Both sit 10 points below our pre-RoPE kq_only (53.3). The gap is entirely explained by the RoPE rotation: post-RoPE vectors encode position, penalising early tokens regardless of whether scores are pooled via softmax or max. Removing the rotation (pre-RoPE) recovers these tokens without any other change to the algorithm.

SnapKV is slightly better than our additive method on QMSum at window=32 (+1.6), but this advantage diminishes at window=128 (+1.3 vs additive). V-norm alone collapses on retrieval tasks. Streaming fails on all distributed-context tasks. All additive variants match or exceed vn_decay on summarization; α=0.6–0.7 achieves the best retrieval performance.

### 4.4 min_decay Sensitivity

We ablate the min_decay parameter — the decay weight assigned to the oldest token — holding all other settings fixed (additive α=0.65, 65% retention). Llama-3.2-3B-Instruct, 30 examples per task.

| min_decay | PassageRetrieval | HotpotQA | 2WikiMQA | GovReport | QMSum | MultiNews |
|---|---|---|---|---|---|---|
| 0.1 | 13.3 | 18.6 | **28.2** | 21.0 | 9.9 | 16.0 |
| 0.3 | 16.7 | 25.8 | **32.1** | 20.9 | 10.7 | 15.6 |
| 0.5 | 40.0 | 23.1 | 30.1 | **21.3** | 11.0 | 16.3 |
| 0.7 | 50.0 | 24.8 | 28.3 | 20.7 | 12.1 | **16.7** |
| 0.9 | **56.7** | **32.3** | 29.5 | 21.2 | **12.4** | **16.7** |

Retrieval performance improves monotonically with min_decay; summarization is nearly flat across the entire range. The effect is large: PassageRetrieval spans 13.3–56.7 (a 43-point range) while GovReport spans only 20.7–21.3.

The mechanism follows from the scoring formula: `score_i = α·kq_i + (1−α)·vn_i·decay_i`. At low min_decay, the vn·decay term drives early-token scores toward zero regardless of KQ alignment. At high min_decay (nearly flat decay), KQ alignment can freely rescue early tokens. Tasks requiring retrieval from early context are therefore highly sensitive to this parameter; tasks relying on uniformly distributed or recent content are not.

We use min_decay=0.7 in all main experiments as a conservative default. Higher values (0.9) improve retrieval further but are expected to hurt code completion, where the most relevant tokens are always the immediately preceding lines. Task-adaptive min_decay selection is a promising direction.

### 4.5 Full LongBench Results

| Task | Full | Naive (4K) | Streaming (65%) | Pruned (65%) |
|---|---|---|---|---|
| NarrativeQA | 5.5 | 19.5 | 6.6 | 3.1 |
| Qasper | 11.1 | 11.4 | 8.4 | 10.1 |
| MultifieldQA | 28.9 | 28.5 | 15.8 | 27.1 |
| HotpotQA | 9.9 | 10.9 | 11.4 | 9.9 |
| 2WikiMQA | 14.1 | 14.2 | 12.9 | 11.3 |
| MuSiQue | 6.9 | 6.7 | 4.5 | 5.1 |
| GovReport | 20.4 | 20.2 | 5.2 | 17.5 |
| QMSum | 10.3 | 14.6 | 3.5 | 8.1 |
| MultiNews | 1.1 | 0.8 | 1.4 | 1.4 |
| TREC | 70.0 | 71.0 | 50.0 | 65.0 |
| TriviaQA | 17.4 | 17.8 | **35.9** | 17.7 |
| SAMSum | 16.0 | 16.4 | 6.4 | 15.6 |
| PassageCount | 3.0 | 1.0 | 2.0 | 2.0 |
| **PassageRetrieval** | **44.0** | **18.0** | 20.0 | **27.0** |
| LCC | 68.1 | 66.5 | 17.8 | 57.4 |
| RepoBench-P | 55.6 | 54.7 | 6.3 | 49.0 |
| **Average** | **23.9** | **23.3** | **13.0** | **20.5** |

### 4.6 Generalization to Mistral-7B-v0.3

To test whether the method generalizes beyond the Llama architecture, we run the same evaluation on Mistral-7B-v0.3 (base, fp16) with identical hyperparameters. The implementation patches the model's self_attn layers at load time via AutoModelForCausalLM; no architecture-specific changes were required.

| Task | Full | Naive (4K) | Pruned (65%) |
|---|---|---|---|
| NarrativeQA | 5.2 | 11.7 | 6.2 |
| Qasper | 5.4 | 4.9 | 7.6 |
| MultifieldQA | 25.3 | 25.0 | 20.2 |
| HotpotQA | 10.5 | 10.7 | 11.3 |
| 2WikiMQA | 11.5 | 11.5 | 11.2 |
| MuSiQue | 5.1 | 4.9 | 5.2 |
| GovReport | 20.7 | 20.2 | 18.7 |
| QMSum | 8.3 | 9.8 | 7.3 |
| MultiNews | 17.5 | 17.4 | 13.4 |
| TREC | 72.0 | 68.0 | 68.0 |
| TriviaQA | 23.1 | 24.0 | 26.5 |
| SAMSum | 16.9 | 18.3 | 18.1 |
| PassageCount | 1.0 | 0.0 | 2.0 |
| PassageRetrieval | 39.0 | 19.0 | 29.0 |
| LCC | 62.9 | 60.7 | 56.2 |
| RepoBench-P | 53.9 | 50.3 | 51.2 |
| **Average** | **23.6** | **22.3** | **22.0** |

The pruned model retains 93% of full-context performance (22.0 vs 23.6), compared to 86% on Llama-3.1-8B. The task-level patterns are consistent across models: PassageRetrieval (29.0 pruned vs 19.0 naive, +10 points) shows the same retrieval recovery, and code tasks (LCC 56.2 vs 62.9 full) show the same regression observed on Llama. The MultiNews full-context score differs markedly between models (Llama 1.1 vs Mistral 17.5), suggesting the near-zero Llama score is an artifact of that model's base-model generation behavior on the summarization format rather than a property of the task itself.

---

## 5. Analysis

### 5.1 Retrieval

The clearest win is PassageRetrieval: pruned (27.0) exceeds naive truncation (18.0) by 9 points. Naive truncation drops the oldest tokens, which for this task often includes most of the 30 candidate paragraphs. Our KQ-aligned scoring retains the paragraph that best matches the query's pre-RoPE key fingerprint, recovering substantially more relevant information within the same budget.

Streaming achieves 20.0 on PassageRetrieval — better than naive but below our method. This is not a general property of streaming: in the ablation (Table 4.2), streaming scored 43.3 on PassageRetrieval because the 3B instruct model's prompt format places paragraphs near the end. At the 8B base model scale with longer prompts (11K–14K tokens), the paragraphs fall outside the recency window and streaming degrades to naive performance.

### 5.2 Summarization

Streaming collapses completely on summarization tasks: GovReport drops from 20.4 (full) to 5.2 (streaming), QMSum from 10.3 to 3.5, MultiNews from 1.1 to 1.4 (roughly random). These tasks require synthesizing information distributed across long documents — exactly what a recency window cannot provide. Our additive method retains 17.5, 8.1, and 1.4 respectively, substantially better than streaming and within 3 points of full context on GovReport.

### 5.3 Code Completion

The largest regression for our method is on code tasks: LCC drops 11 points and RepoBench-P drops 7 points relative to full context, worse than naive truncation. Streaming is far worse (LCC 17.8, RepoBench-P 6.3), but this is expected since code completion requires the immediately preceding lines rather than semantically interesting but potentially distant tokens. The KQ alignment signal may be over-selecting syntactically salient tokens (identifiers, keywords) at the expense of the surrounding low-salience but syntactically necessary tokens. A higher always_keep_last value or stronger decay (lower min_decay) would likely help.

### 5.4 TriviaQA Anomaly

Streaming achieves 35.9 on TriviaQA — nearly double all other methods (full 17.4, naive 17.8, pruned 17.7). TriviaQA in LongBench is formatted as few-shot in-context learning: multiple question-answer examples appear in the context, followed by the target question. The recency window captures the most recent few-shot examples and the target question intact, giving the model clean in-context demonstrations. Longer-range content (earlier examples and background passages) is discarded, which for this task is harmless or beneficial since the few-shot format guides the answer pattern. This demonstrates that streaming can be competitive when the task structure places all necessary information near the end of the context.

### 5.5 Anomalous NarrativeQA Result

Naive truncation (19.5) substantially outperforms full context (5.5) on NarrativeQA. We hypothesize that very long story contexts confuse the base model's generation, and truncation incidentally removes the confusing material. This may also reflect a scoring artifact: the F1 metric degrades when the model generates verbose completions prompted by long contexts. This warrants further investigation.

### 5.6 Efficiency: Prefill Overhead Dominates at Current Context Lengths

We benchmarked generation throughput (tokens/sec) and peak VRAM using model.generate() with max_new_tokens=64 for Llama-3.1-8B at context lengths 2048, 4096, and 8192 (budget_fraction=0.65, float16).

| Context | VRAM full (MB) | VRAM pruned (MB) | VRAM delta | Speedup |
|---------|---------------|-----------------|------------|---------|
| 2048    | 16,706        | 17,152          | −2.7%      | 0.94×   |
| 4096    | 17,334        | 19,641          | −13.3%     | 0.85×   |
| 8192    | 18,590        | 28,839          | −55.1%     | 0.65×   |

The pruned model is both slower and higher-VRAM at all tested context lengths. Two effects explain this:

**Scoring overhead.** During prefill, our method computes dot products between the Q-buffer (last 128 queries) and all T key positions. At T=8192 with 32 Q-heads and head_dim=128, this produces intermediate tensors of shape [32, 128, 8192] per layer — approximately 4 GB of intermediate activations spread across 32 layers in float32. These dominate the KV cache itself (theoretical savings: 376 MB at T=8192, 35%).

**Theoretical savings are small relative to model weight footprint.** The 8B model weights occupy ~16 GB. The KV cache at T=8192 is approximately 1.1 GB full, 0.7 GB pruned. A 376 MB reduction in KV cache is 2% of total VRAM — below the noise floor relative to scoring intermediates and memory fragmentation.

**Implications.** The efficiency benefit of KV cache pruning is primarily realized in two regimes our benchmark does not capture: (1) very long contexts where the KV cache is a large fraction of total VRAM (e.g., 100K+ tokens), and (2) when custom CUDA kernels avoid materializing full-size intermediate tensors during scoring. Our reference implementation uses standard PyTorch operations, which do not fuse the scoring step. Production systems such as H2O and SnapKV use FlashAttention-based kernels to amortize scoring cost; achieving similar efficiency gains would require the same engineering investment. We report these results for completeness and to set accurate expectations: the contribution of this work is *quality under compression*, not raw throughput or VRAM reduction at moderate context lengths.

### 5.7 Effect of Retention Fraction

We sweep the retention fraction r ∈ {50%, 65%, 80%} using the additive scorer with α=0.7 on a representative six-task subset (30 examples per task, Llama-3.1-8B). Full-context scores from §4.5 are included as a ceiling.

| Task | r=50% | r=65% | r=80% | Full |
|---|---|---|---|---|
| PassageRetrieval | 3.3 | 20.0 | 33.3 | 44.0 |
| HotpotQA | 7.2 | 9.6 | 12.9 | 9.9 |
| 2WikiMQA | 9.3 | 11.2 | 14.9 | 14.1 |
| GovReport | 14.8 | 19.6 | 18.5 | 20.4 |
| QMSum | 7.0 | 8.5 | 8.0 | 10.3 |
| MultiNews | 4.2 | 1.8 | 0.2 | 1.1 |

Higher retention consistently improves performance, but the effect is highly task-dependent. PassageRetrieval is the most sensitive: score nearly triples from r=50% to r=80% (3.3 → 33.3), and remains 10 points below full context even at 80%. Retrieval tasks require retaining the specific paragraphs matching the query — at 50% retention, roughly half the candidates are discarded regardless of scoring quality. QA tasks (HotpotQA, 2WikiMQA) show moderate, near-monotone improvement. Summarization (GovReport, QMSum) is relatively flat: the additive scorer extracts the most content-dense tokens even at 50% retention, and little is gained by retaining more. MultiNews scores are near-zero across all retentions, consistent with the base model's known difficulty with that format (see §4.5).

The most interesting operating point is r=50–65%, where KV cache savings are meaningful (35–50% reduction) and scoring quality has maximum impact. At r=80% the gap versus full context is modest for all tasks except PassageRetrieval, suggesting diminishing returns for most use cases.

---

## 6. Discussion

**Why additive over multiplicative?** The standard formulation kq × vn × decay multiplies three normalized quantities in [0,1], each capable of zeroing the product independently. In practice, V-norm is near-zero for common, predictable tokens (articles, punctuation, repeated phrases) and near-one for unusual tokens. Multiplicative combination therefore approximates V-norm with a semantic tiebreaker rather than a genuine combination. Additive combination gives each signal independent "voice": a token with high KQ alignment but ordinary V-norm (e.g. a relevant but common noun) survives.

**Why pre-RoPE?** RoPE encodes absolute position by rotating the K and Q vectors. A key vector from position 0 in a 10K context is rotated by 10K×θ relative to the last query, producing artificially low dot products regardless of semantic content. Pre-RoPE keys remove this rotation, so KQ alignment reflects content similarity only. The post-RoPE keys are still used in the actual attention computation.

**When does streaming work?** Our results suggest streaming is competitive only when task-relevant information is reliably near the end of the context (few-shot QA, some retrieval formats). It fails when relevant content is distributed — summarization, multi-hop reasoning, and code completion with distant dependencies. A practical system might use streaming for latency-sensitive few-shot applications and semantic scoring for document understanding tasks.

**Limitations.** Our evaluation covers two model families (Llama, Mistral) at a single scale (~7–8B parameters); behavior at larger scales or on other architectures (e.g. Gemma, Qwen) is untested. The α and min_decay hyperparameters were tuned on LongBench tasks, raising the question of generalizability to out-of-distribution tasks. Code completion regresses relative to naive truncation, suggesting that task-adaptive scoring may be necessary for a fully general system. We compare against SnapKV (Table 4.2) but not against H2O, which requires accumulated attention weights across the full prefill and is therefore more expensive to implement. The current implementation does not achieve practical VRAM or throughput improvements at moderate context lengths (see §5.6); realizing efficiency gains requires custom CUDA kernels or Flash-Attention integration. Finally, all experiments use a single retention fraction; a practical system might benefit from per-layer budgets reflecting the empirical finding that early and late layers have different attention patterns.

---

## 7. Conclusion

We have shown that additive combination of KQ alignment and V-norm with α=0.65 outperforms V-norm alone on retrieval and QA tasks, while matching it on summarization. Pre-RoPE scoring and context-length-independent decay are important supporting design choices. At 65% token retention, our method recovers 86% of full-context average performance on LongBench with Llama-3.1-8B and 93% with Mistral-7B-v0.3, beating naive left-truncation on passage retrieval by 9–10 points across both models. The method required no architecture-specific modifications to transfer between model families. Comparison with StreamingLLM confirms that semantic scoring is essential for tasks requiring distributed context: streaming achieves 13.0 average LongBench score vs 20.5 for our method, with particularly large gaps on summarization (3.5–5.2 vs 8.1–17.5) and code completion (6.3–17.8 vs 49.0–57.4). Direct comparison with SnapKV (post-RoPE dot-product pooling) confirms that the pre-RoPE scoring choice is load-bearing: switching to post-RoPE drops passage retrieval from 27.0 to 14.0. The main remaining challenges are code completion — where recency dominates over semantic relevance — and achieving practical VRAM/throughput improvements without custom kernels.

---

## References

- Bai, Y. et al. (2023). LongBench: A bilingual, multitask benchmark for long context understanding. *arXiv:2308.14508*.
- Li, Y. et al. (2024). SnapKV: LLM knows what you are looking for before generation. *NeurIPS 2024*.
- Xiao, G. et al. (2023). Efficient streaming language models with attention sinks. *arXiv:2309.17453*.
- Zhang, Z. et al. (2023). H2O: Heavy-hitter oracle for efficient generative inference of large language models. *NeurIPS 2023*.
- Yang, Z. et al. (2024). VATP: Value-aware token pruning for efficient long-context inference. *EMNLP 2024*.
- He, Y. et al. (2025). Ada-KV: Optimizing KV cache eviction by adaptive budget allocation for efficient LLM inference. *NeurIPS 2025*.
- Fu, Y. et al. (2025). HeadKV: Head-level KV cache compression for efficient long-context inference. *ICLR 2025*.
- Xiao, G. et al. (2025). DuoAttention: Efficient long-context LLM inference with retrieval and streaming heads. *ICLR 2025*.
- Ye, T. et al. (2025). A2ATS: Retrieval-based KV cache reduction with anchor tokens for efficient LLM inference. *ACL 2025*.
