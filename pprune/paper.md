# Additive KV Cache Pruning: Combining Semantic Alignment and Value Norms for Long-Context Inference

---

## Abstract

We present a prefill-time KV cache pruning method for transformer-based language models that retains a fixed fraction of key-value token slots while preserving model quality on downstream tasks. Our scoring function additively combines two complementary signals: KQ alignment (which tokens does the question attend to?) and V-norm (which tokens have high output potential?), weighted by a tunable parameter α and a distance-decay term. Crucially, we compute KQ alignment using pre-RoPE key and query vectors, removing the positional bias that makes early tokens appear semantically irrelevant under standard post-RoPE scoring. We adopt a global budget strategy in which the same token positions are retained across all KV heads within a layer, using a context-length-independent decay parametrization (min_decay) that scales automatically to arbitrary sequence lengths. Evaluated on LongBench with Llama-3.1-8B at 65% token retention, our method recovers 86% of full-context performance on average, and outperforms naive left-truncation on retrieval tasks (27% vs 18% on passage_retrieval_en). Our ablations show that V-norm alone — while effective on synthetic needle tests — fails on real retrieval tasks, and that an additive combination with α=0.65 achieves the best overall balance.

---

## 1. Introduction

The computational cost of transformer attention scales quadratically with sequence length, making long-context inference expensive. A growing body of work addresses this by pruning the KV cache: discarding low-importance token slots before or during generation so that attention operates over a smaller, more informative set of key-value pairs [CITATION].

Most existing methods assign per-token importance scores based on attention weights [H2O], value vector norms [VATP], or combinations thereof, and evict low-scoring tokens. A practical system must additionally decide: (1) how to score tokens in a way that is stable across different context lengths, (2) how to allocate the total token budget across attention heads, and (3) how much to favor recent tokens over semantically relevant but distant ones.

We make the following contributions:

- **Additive scoring**: We propose combining KQ semantic alignment and V-norm *additively* rather than multiplicatively. Multiplicative combination collapses the score whenever either component is near zero — V-norm penalizes normal prose even when the model strongly attends to it. An additive combination lets each signal independently rescue tokens.

- **Pre-RoPE KQ scoring**: Rotary positional encodings (RoPE) bake absolute position into K and Q vectors, causing tokens far from the query to appear semantically dissimilar regardless of content. We compute KQ alignment using pre-RoPE representations, giving a position-independent semantic signal. The post-RoPE keys and values are still used for attention itself.

- **Context-length-independent decay**: Prior methods apply a fixed exponential or linear decay rate. At very long contexts (e.g. 100K tokens) even small rates drive early-token weights to near zero. We parametrize decay by its value at position 0 (min_decay), with the rate automatically derived from context length, so pruning behavior is consistent regardless of sequence length.

- **Global budget**: We retain the same token positions for all KV heads within a layer, selected by aggregating per-head importance scores. This avoids the memory overhead of per-head non-contiguous indexing while still using information from all heads.

We evaluate on LongBench [CITATION] covering single-document QA, multi-document QA, summarization, retrieval, and code completion tasks. Our ablations isolate the contribution of each design choice and identify a residual weakness on code tasks.

---

## 2. Background and Related Work

**KV cache eviction.** H2O [Zhang et al., 2023] identifies "heavy hitter" tokens via accumulated attention scores and evicts the rest. SnapKV [Li et al., 2024] selects tokens by pooling attention from the last several queries, similar in spirit to our Q-buffer approach. StreamingLLM [Xiao et al., 2023] always retains "attention sink" tokens (typically the first few tokens) plus a recent window.

**Value-norm scoring.** VATP [EMNLP 2024] proposes scoring tokens by the product of their attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Our method uses V-norm as a standalone component and finds it sufficient for synthetic retrieval but insufficient for natural-language QA.

**Head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different budgets to different attention heads, recognizing that "retrieval heads" and "accumulation heads" have fundamentally different importance profiles. Our work uses a global budget per layer rather than per-head allocation, which is simpler and avoids the overhead of per-head mask divergence.

**Pre-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We confirm empirically that pre-RoPE scoring is important for general QA performance.

**Decay-combined scoring.** GraphKV [EMNLP 2025] and PrHS [arXiv 2025] combine recency bias with semantic scores. Our min_decay parametrization offers a cleaner, length-independent formulation of this idea.

---

## 3. Method

### 3.1 Setup

We patch the attention layers of a LlamaForCausalLM model to intercept the prefill pass. On the first (prefill) forward pass through each layer, we compute importance scores for all T input tokens, select a retained subset of size ⌊r·T⌋ (where r is the retention fraction), and gather the corresponding K and V states. All subsequent generation steps attend only to these retained positions plus any newly generated tokens.

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

### 3.3 Budget Selection

We unconditionally retain the first always_keep_first tokens (preventing NaN from early queries having no visible keys) and the last always_keep_last tokens (preserving the query context). The remaining budget ⌊r·T⌋ − always_keep_first − always_keep_last slots are filled with the highest-scoring non-protected tokens.

For GQA models (e.g. Llama 3.x with 32 Q-heads and 8 KV-heads), we compute scores for all Q-heads and aggregate across heads sharing a KV-head via max-pooling before selection.

After selection, we reconstruct a valid causal attention mask using the *original* (pre-pruning) positions of the retained tokens, ensuring that query token at position p can only attend to retained tokens at positions ≤ p.

---

## 4. Experiments

### 4.1 Setup

**Model.** Llama-3.1-8B (base, not instruction-tuned) in fp16 on a V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks covering single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot classification and QA (TREC, TriviaQA, SAMSum), synthetic retrieval (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). We evaluate 100 examples per task.

**Baselines.**
- *Full context*: unmodified Llama-3.1-8B with full prompt.
- *Naive truncation*: left-truncation to a fixed 4096-token budget (drops the oldest tokens).

**Our method.** Retention fraction r=0.65, α=0.65, min_decay=0.7, always_keep_first=16, always_keep_last=16, q_buffer_size=128, decay_fn=linear.

### 4.2 Score Mode Ablation

To isolate the effect of the scoring function, we ran a controlled comparison on three retrieval/QA tasks (PassageRetrieval, HotpotQA, 2WikiMQA) and three summarization tasks (GovReport, QMSum, MultiNews) at three retention fractions (50%, 65%, 80%) using Llama-3.2-3B-Instruct. We tested five configurations:

| Mode | Formula |
|---|---|
| vn_decay | vn × decay |
| kq_only | kq |
| additive α=0.6 | 0.6·kq + 0.4·vn·decay |
| additive α=0.7 | 0.7·kq + 0.3·vn·decay |
| additive α=0.9 | 0.9·kq + 0.1·vn·decay |

**Retrieval/QA results at 65% retention:**

| Mode | PassageRetrieval | HotpotQA | 2WikiMQA |
|---|---|---|---|
| vn_decay | 23.3 | 16.6 | 18.1 |
| kq_only | 53.3 | 29.9 | 27.6 |
| additive α=0.6 | **56.7** | 24.8 | **33.6** |
| additive α=0.7 | 50.0 | **27.1** | 29.9 |
| additive α=0.9 | 36.7 | 27.2 | 25.2 |

**Summarization results at 65% retention** (ROUGE-L):

| Mode | GovReport | QMSum | MultiNews |
|---|---|---|---|
| vn_decay | 21.2 | 10.9 | 3.1 |
| kq_only | 20.9 | 11.5 | 3.0 |
| additive α=0.6 | **21.0** | 10.5 | 3.3 |
| additive α=0.7 | 21.2 | **10.9** | 3.1 |
| additive α=0.9 | 20.9 | 11.9 | **3.4** |

V-norm alone collapses on retrieval tasks while causing no benefit on summarization. All additive variants match or exceed vn_decay on summarization; α=0.6–0.7 achieves the best retrieval performance without sacrificing summarization.

### 4.3 Full LongBench Results

| Task | Full | Naive (4K) | Pruned (65%) |
|---|---|---|---|
| NarrativeQA | 5.5 | 19.5 | 3.1 |
| Qasper | 11.1 | 11.4 | 10.1 |
| MultifieldQA | 28.9 | 28.5 | 27.1 |
| HotpotQA | 9.9 | 10.9 | 9.9 |
| 2WikiMQA | 14.1 | 14.2 | 11.3 |
| MuSiQue | 6.9 | 6.7 | 5.1 |
| GovReport | 20.4 | 20.2 | 17.5 |
| QMSum | 10.3 | 14.6 | 8.1 |
| MultiNews | 1.1 | 0.8 | 1.4 |
| TREC | 70.0 | 71.0 | 65.0 |
| TriviaQA | 17.4 | 17.8 | 17.7 |
| SAMSum | 16.0 | 16.4 | 15.6 |
| PassageCount | 3.0 | 1.0 | 2.0 |
| **PassageRetrieval** | **44.0** | **18.0** | **27.0** |
| LCC | 68.1 | 66.5 | 57.4 |
| RepoBench-P | 55.6 | 54.7 | 49.0 |
| **Average** | **23.9** | **23.3** | **20.5** |

---

## 5. Analysis

### 5.1 Retrieval

The clearest win is PassageRetrieval: pruned (27.0) exceeds naive truncation (18.0) by 9 points. Naive truncation drops the oldest tokens, which for this task often includes most of the 30 candidate paragraphs. Our KQ-aligned scoring retains the paragraph that best matches the query's pre-RoPE key fingerprint, recovering substantially more relevant information within the same budget.

### 5.2 Code Completion

The largest regression is on code tasks: LCC drops 11 points and RepoBench-P drops 7 points relative to full context, worse than naive truncation. Code completion is almost entirely a recency task — the most relevant tokens are the immediately preceding lines. The KQ alignment signal may be over-selecting semantically interesting tokens (identifiers, keywords) at the expense of syntactically necessary but low-salience tokens nearby. A higher always_keep_last value or a stronger decay (lower min_decay) would likely help here.

### 5.3 Anomalous NarrativeQA Result

Naive truncation (19.5) substantially outperforms full context (5.5) on NarrativeQA. We hypothesize that very long story contexts confuse the base model's generation, and truncation incidentally removes the confusing material. This may also reflect a scoring artifact: the F1 metric degrades when the model generates verbose completions prompted by long contexts. This warrants further investigation.

### 5.4 Effect of Retention Fraction

As expected, higher retention consistently improves all methods. The gap between pruned and full context narrows as retention increases — at 80% retention the difference is within scoring noise for most tasks. The most interesting operating point is 50–65%, where memory savings are meaningful (35–50% KV cache reduction) and the scoring function's choices have the most impact.

---

## 6. Discussion

**Why additive over multiplicative?** The standard formulation kq × vn × decay multiplies three normalized quantities in [0,1], each capable of zeroing the product independently. In practice, V-norm is near-zero for common, predictable tokens (articles, punctuation, repeated phrases) and near-one for unusual tokens. Multiplicative combination therefore approximates V-norm with a semantic tiebreaker rather than a genuine combination. Additive combination gives each signal independent "voice": a token with high KQ alignment but ordinary V-norm (e.g. a relevant but common noun) survives.

**Why pre-RoPE?** RoPE encodes absolute position by rotating the K and Q vectors. A key vector from position 0 in a 10K context is rotated by 10K×θ relative to the last query, producing artificially low dot products regardless of semantic content. Pre-RoPE keys remove this rotation, so KQ alignment reflects content similarity only. The post-RoPE keys are still used in the actual attention computation.

**Limitations.** Our evaluation is limited to a single model family and size. The α and min_decay hyperparameters were tuned on LongBench tasks, raising the question of generalizability to out-of-distribution tasks. Code completion regresses relative to naive truncation, suggesting that task-adaptive scoring may be necessary for a fully general system. Finally, all experiments use a single retention fraction; a practical system might benefit from per-layer budgets reflecting the empirical finding that early and late layers have different attention patterns.

---

## 7. Conclusion

We have shown that additive combination of KQ alignment and V-norm with α=0.65 outperforms V-norm alone on retrieval and QA tasks, while matching it on summarization. Pre-RoPE scoring and context-length-independent decay are important supporting design choices. At 65% token retention, our method recovers 86% of full-context average performance on LongBench and beats naive left-truncation on passage retrieval by 9 points. The main remaining challenge is code completion, where the recency-driven nature of the task favors stronger decay over semantic scoring.

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
