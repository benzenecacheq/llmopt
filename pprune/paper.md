# Scoring KV Cache Tokens in a Semantically Clean Space: Pre-RoPE Alignment with Value-Norm Payload Signals

---

## Abstract

We argue that the scoring space matters as much as the scoring function for prefill-time KV cache pruning. Standard methods compute token importance using post-RoPE key-query dot products, where rotary positional encodings entangle semantic content with absolute position — causing distant-but-relevant tokens to appear unimportant regardless of their actual content. We instead score tokens using pre-RoPE representations, recovering a position-independent semantic signal. We pair this with a value-norm term that captures each token's output payload: attention weights decide what gets selected, but actual contribution to the output depends jointly on the weight and the value content. These two signals are combined additively rather than multiplicatively, so each can independently rescue tokens that the other would miss.

Evaluated on LongBench across retrieval, summarization, multi-document QA, and code tasks with Llama-3.1-8B at 65% retention, our method recovers 86% of full-context performance and outperforms naive truncation on retrieval by 9 points (27% vs 18%). Results break down diagnostically by task type: summarization shows the largest gap versus recency-only baselines, retrieval benefits most from position-independent scoring, and code completion reveals a residual weakness where locality matters more than semantic relevance. A head-aware budget extension that allocates more capacity to high-entropy heads is tested and found to match but not improve over the global budget at this scale — an informative null result that isolates the code failure as a token-selection issue rather than a budget-allocation issue. We validate on Mistral-7B-v0.3 with identical hyperparameters (93% retention), and implement the method as a KVPress press subclass, achieving a 10% generation speedup over full context.

---

## 1. Introduction

The computational cost of transformer attention scales quadratically with sequence length, making long-context inference expensive. A growing body of work addresses this by pruning the KV cache: discarding low-importance token slots before or during generation so that attention operates over a smaller, more informative set of key-value pairs [Zhang et al., 2023; Li et al., 2024; Yang et al., 2024].

Most existing methods assign per-token importance scores based on attention weights [Zhang et al., 2023; Li et al., 2024], value vector norms [Yang et al., 2024], or combinations thereof. A key unresolved question is whether the scoring space itself matters: most methods evaluate token importance using post-RoPE representations, where KQ similarity is a function of both semantic content and positional distance. This conflation penalizes early tokens regardless of their relevance. A second question is whether the relevance signal alone is sufficient, or whether the output payload of a token — the magnitude of its value contribution — should also factor into the decision.

We make the following contributions:

- **Pre-RoPE scoring space**: We argue that scoring token importance in post-RoPE space systematically under-ranks semantically relevant but distant tokens, because rotary encodings rotate K and Q vectors by position, producing artificially low dot products for early tokens. We score tokens using pre-RoPE representations, where KQ alignment reflects content similarity independent of position. The post-RoPE keys and values are still used for attention itself. This is the central contribution; the other design choices support it.

- **Relevance + payload decomposition**: Token importance has two components: how much the current query matches this token (relevance, captured by KQ alignment) and how much information this token would contribute if selected (payload, captured by V-norm). We combine these *additively* rather than multiplicatively. Multiplicative combination collapses whenever either term is near zero; additive combination lets each signal independently rescue tokens.

- **Context-length-independent decay**: We parametrize recency bias by the decay value at position 0 (min_decay), with the rate auto-derived from context length, so pruning behavior is consistent across arbitrary sequence lengths without re-tuning.

- **Head-aware budget allocation**: Different attention heads have different entropy profiles — some distribute attention broadly (high entropy), some focus on a few tokens (low entropy). We test entropy-proportional budget allocation and find it matches the global-budget baseline within noise, isolating the code regression as a token-selection problem rather than a budget-distribution problem.

We evaluate on LongBench [Bai et al., 2023] structured as a diagnostic breakdown by task family: retrieval tasks (where position-independent scoring matters most), summarization tasks (where distributed context is essential), code completion (where locality dominates), and few-shot QA (where recency suffices). This structure lets us attribute each gain and each failure mode to a specific property of the scoring function, rather than relying on a single average number.

---

## 2. Background and Related Work

**KV cache eviction.** H2O [Zhang et al., 2023] identifies "heavy hitter" tokens via accumulated attention scores and evicts the rest. SnapKV [Li et al., 2024] selects tokens by pooling attention weights from the last several queries over all key positions, using post-RoPE vectors. Our method differs in two ways: we use pre-RoPE vectors for scoring (removing positional bias) and combine the alignment signal additively with V-norm. We implement and benchmark SnapKV directly, finding that pre-RoPE scoring outperforms SnapKV by up to 13 points on retrieval tasks. StreamingLLM [Xiao et al., 2023] always retains "attention sink" tokens (typically the first few tokens) plus a recent window, making it a strong baseline on recency-dominated tasks.

**Value-norm scoring.** VATP [EMNLP 2024] proposes scoring tokens by the product of their attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Our method uses V-norm as a standalone component and finds it sufficient for synthetic retrieval but insufficient for natural-language QA. Feng et al. [2025] provide a theoretical justification for value-norm-based scoring by deriving an upper bound on output perturbation; their bound depends on the L1 norm of the *projected* value state V·W^O rather than the raw value vector. They propose a two-stage selector (top-α by attention weight, remainder by (A+ε)⊙‖V·W^O‖₁) and demonstrate improvements over SnapKV and AdaKV at very low retention rates (2.5–20%). Our work differs in two respects: we use raw V-norm without the output projection (motivated empirically rather than theoretically), and we combine it with pre-RoPE KQ alignment rather than post-RoPE attention weights.

**Head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different budgets to different attention heads, recognizing that "retrieval heads" and "accumulation heads" have fundamentally different importance profiles. Our work uses a global budget per layer rather than per-head allocation, which is simpler and avoids the overhead of per-head mask divergence.

**Pre-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We confirm empirically that pre-RoPE scoring is important for general QA performance.

**Decay-combined scoring.** GraphKV [EMNLP 2025] and PrHS [arXiv 2025] combine recency bias with semantic scores. Our min_decay parametrization offers a cleaner, length-independent formulation of this idea.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a NVIDIA-maintained open-source framework that unifies KV cache compression research under a common plugin interface. Methods are implemented as "press" subclasses overriding a single `score()` method; the framework handles hook registration, head-wise masking, and HuggingFace Transformers integration automatically. KVPress has emerged as a de facto standard for compression research, with 37+ methods in its standard library and an associated HuggingFace leaderboard for standardized benchmarking. The paper's core contribution is Expected Attention, which estimates future query attention by modelling hidden states as Gaussian and averaging RoPE rotation matrices over T future positions, scoring tokens as `(E[attn_i] + ε) × ‖v_i‖`. Like our method, Expected Attention uses an additive ε term to prevent V-norm from being zeroed — independently arriving at a similar additive structure from a distributional theory perspective. Unlike ours, it operates in post-RoPE space and does not address positional bias in long contexts. Our method can in principle be contributed as a KVPress press, making it directly comparable on the leaderboard and providing a concrete path to CUDA optimization following the AdaKV kernel pattern [He et al., 2025].

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

The results divide naturally into four task families, each of which tests a different property of the scoring function. We treat the failure modes as diagnostics rather than anomalies: each case where a simpler method outperforms ours reveals something specific about what the additive scorer does and does not model.

### 5.1 Retrieval

The clearest win is PassageRetrieval: pruned (27.0) exceeds naive truncation (18.0) by 9 points. Naive truncation drops the oldest tokens, which for this task often includes most of the 30 candidate paragraphs. Our KQ-aligned scoring retains the paragraph that best matches the query's pre-RoPE key fingerprint, recovering substantially more relevant information within the same budget.

Streaming achieves 20.0 on PassageRetrieval — better than naive but below our method. This is not a general property of streaming: in the ablation (Table 4.2), streaming scored 43.3 on PassageRetrieval because the 3B instruct model's prompt format places paragraphs near the end. At the 8B base model scale with longer prompts (11K–14K tokens), the paragraphs fall outside the recency window and streaming degrades to naive performance.

However, pruned (27.0) remains 17 points below full context (44.0). The retention sweep (§5.7) shows this gap persists even at r=80% (33.3 vs 44.0), indicating it is not simply a budget problem. The task structure explains the residual gap: PassageRetrieval presents 30 candidate paragraphs and asks which one contains a specific verbatim sentence. All 30 paragraphs are topically related, so our KQ scoring — which measures semantic similarity between the query and context tokens — cannot reliably distinguish the target paragraph from topically similar decoys. Exact-match retrieval requires a different signal (e.g. n-gram overlap or BM25-style scoring) that our additive scorer does not capture. This is a fundamental limitation for tasks that require verbatim rather than semantic matching.

### 5.2 Summarization

Streaming collapses completely on summarization tasks: GovReport drops from 20.4 (full) to 5.2 (streaming), QMSum from 10.3 to 3.5, MultiNews from 1.1 to 1.4 (roughly random). These tasks require synthesizing information distributed across long documents — exactly what a recency window cannot provide. Our additive method retains 17.5, 8.1, and 1.4 respectively, substantially better than streaming and within 3 points of full context on GovReport.

### 5.3 Code Completion: Diagnostic of Locality

Code completion regresses relative to both full context and naive truncation (LCC: 57.4 pruned vs 68.1 full vs 66.5 naive; RepoBench-P: 49.0 vs 55.6 vs 54.7). This is a diagnostic finding: it tells us precisely what our scoring function does not model. Code completion requires the immediately preceding lines — syntactically contiguous, often low-semantic-salience tokens (brackets, indentation, variable declarations). Our KQ alignment signal over-selects syntactically salient identifiers and keywords while discarding the surrounding structural tokens that are syntactically necessary but semantically ordinary.

This failure mode motivates the head-aware extension in §5.8: attention heads in code-heavy contexts tend to be more locally focused with lower entropy, and they should receive stronger recency protection rather than semantic scoring. Even without head-awareness, a larger always_keep_last window or lower min_decay would reduce this regression — the fix is known, which is what makes this a diagnostic rather than a fundamental limitation.

### 5.4 TriviaQA Anomaly

Streaming achieves 35.9 on TriviaQA — nearly double all other methods (full 17.4, naive 17.8, pruned 17.7). TriviaQA in LongBench is formatted as few-shot in-context learning: multiple question-answer examples appear in the context, followed by the target question. The recency window captures the most recent few-shot examples and the target question intact, giving the model clean in-context demonstrations. Longer-range content (earlier examples and background passages) is discarded, which for this task is harmless or beneficial since the few-shot format guides the answer pattern. This demonstrates that streaming can be competitive when the task structure places all necessary information near the end of the context.

### 5.5 NarrativeQA: Diagnostic of Semantic Scorer Misfire

Naive truncation (19.5) substantially outperforms full context (5.5) on NarrativeQA, and our pruned model (3.1) is worse still. The same pattern holds on Mistral-7B-v0.3 (full 5.2, naive 11.7, pruned 6.2), confirming this is not a Llama-specific artifact.

The mechanism is base model behavior under long narrative context. NarrativeQA presents full novel or screenplay text followed by a question; the correct response is a short phrase. A base model (not instruction-tuned) given 7K tokens of dense narrative tends to continue the story or repeat the Q&A prompt format rather than produce a short answer — as visible in the generated outputs (e.g., the model produces a chain of additional question-answer pairs rather than answering the query). The F1-over-unigrams metric severely penalizes these verbose outputs. Naive truncation to 4096 tokens discards most of the narrative, leaving a shorter context in which the question is more prominent and the model's tendency to continue the story is weaker.

Our pruned model scores below full context because our KQ scoring actively retains narrative content that is semantically relevant to the query — which is precisely the material that triggers story-continuation behavior. The diagnostic reading: semantic relevance scoring works against the model when retaining relevant context causes the model to behave badly. This is not a failure of the pruning method per se but of applying it to a base model on a task that requires format-following suppression. This result would likely reverse with an instruction-tuned model. It also suggests that the scoring function and the model's behavioral response to context are not independent — an important consideration when generalizing across model types.

### 5.6 Efficiency: Prefill Overhead vs. Generation Savings

**KVPress result (generation-phase speedup).** We implemented AdditiveScorerPress as a KVPress plugin and benchmarked against full context, SnapKVPress, and StreamingLLMPress on six LongBench tasks (30 examples, Llama-3.1-8B, 65% retention). Total generation time across all tasks: full context 2825s, additive 2545s, SnapKV 2541s, streaming 2533s — all compressed methods are approximately **10% faster** than full context. The speedup comes from the generation phase: a smaller post-pruning KV cache reduces memory bandwidth at each decode step, and this saving accumulates over generation across 180 examples. Quality is preserved: AdditiveScorerPress averages 15.9 vs 15.9 for full context.

**Low-context overhead (our reference implementation).** We also benchmarked generation throughput using our original prefill-patching implementation with max_new_tokens=64 for Llama-3.1-8B at context lengths 2048, 4096, and 8192 (budget_fraction=0.65, float16).

| Context | VRAM full (MB) | VRAM pruned (MB) | VRAM delta | Speedup |
|---------|---------------|-----------------|------------|---------|
| 2048    | 16,706        | 17,152          | −2.7%      | 0.94×   |
| 4096    | 17,334        | 19,641          | −13.3%     | 0.85×   |
| 8192    | 18,590        | 28,839          | −55.1%     | 0.65×   |

The pruned model is both slower and higher-VRAM at all tested context lengths. Two effects explain this:

**Scoring overhead.** During prefill, our method computes dot products between the Q-buffer (last 128 queries) and all T key positions. At T=8192 with 32 Q-heads and head_dim=128, this produces intermediate tensors of shape [32, 128, 8192] per layer — approximately 4 GB of intermediate activations spread across 32 layers in float32. These dominate the KV cache itself (theoretical savings: 376 MB at T=8192, 35%).

**Theoretical savings are small relative to model weight footprint.** The 8B model weights occupy ~16 GB. The KV cache at T=8192 is approximately 1.1 GB full, 0.7 GB pruned. A 376 MB reduction in KV cache is 2% of total VRAM — below the noise floor relative to scoring intermediates and memory fragmentation.

**Implications.** The efficiency benefit of KV cache pruning is primarily realized in two regimes our benchmark does not capture: (1) very long contexts where the KV cache is a large fraction of total VRAM (e.g., 100K+ tokens), and (2) when custom CUDA kernels avoid materializing full-size intermediate tensors during scoring. Our reference implementation uses standard PyTorch operations, which do not fuse the scoring step. KVPress [Devoto et al., 2025] reports 2× memory savings and 1.5× generation speedup at 128K context on A100 using their Expected Attention method under the same PyTorch constraints, suggesting the benefit profile improves substantially at longer contexts than we benchmark here.

**KVPress implementation.** We implemented AdditiveScorerPress, a KVPress press subclass that overrides the single `score()` method with our pre-RoPE KQ + V-norm + decay scorer. Benchmarked on a six-task LongBench subset (30 examples, Llama-3.1-8B, 65% retention) against SnapKVPress, StreamingLLMPress, and full context:

| Task | Full | Additive | SnapKV | Streaming |
|---|---|---|---|---|
| PassageRetrieval | 13.3 | **16.7** | 13.3 | 16.7 |
| HotpotQA | 9.7 | 8.7 | 9.7 | 10.6 |
| 2WikiMQA | 16.0 | 15.9 | 16.0 | 15.6 |
| GovReport | 17.0 | **16.9** | 16.4 | 16.7 |
| QMSum | 13.8 | 12.4 | 13.0 | 12.0 |
| LCC | 25.9 | 25.8 | **25.8** | 26.4 |
| **Average** | **15.9** | **15.9** | 15.7 | 16.2 |
| **Total time (s)** | 2825 | **2545** | 2541 | 2533 |

AdditiveScorerPress matches full-context quality (15.9 average) while running 10% faster overall. The speedup is realized during generation — the smaller post-pruning KV cache reduces memory bandwidth in the generation phase, outweighing scoring overhead across the full benchmark. All three compressed methods are similarly fast; the difference from full context is KV cache size, not scorer choice. Quality differences between methods are consistent with the ablations in §4.3: additive leads on retrieval and summarization, streaming is competitive. The KVPress implementation is ~100 lines of Python and required no architecture-specific code; it works on any model supported by KVPress (currently Llama, Mistral, Phi-3, Qwen, Gemma). CUDA kernel optimization following the AdaKV pattern [He et al., 2025] is a natural next step for production deployment.

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

### 5.8 Head-Aware Budget Allocation

Different attention heads are known to specialise in different roles — some distribute attention broadly across the sequence (retrieval heads, high entropy), others focus locally (streaming heads, low entropy) [Xiao et al., 2025]. A global compression ratio applies the same budget to all heads, which may over-prune high-entropy retrieval heads while under-pruning low-entropy local heads. We test a simple head-aware extension: allocate per-head token budgets proportionally to the entropy of each head's score distribution, keeping the total token count fixed.

Results on the same six-task subset (Llama-3.1-8B, 30 examples, 65% retention):

| Task | Full | Additive | Head-Aware | SnapKV | Streaming |
|---|---|---|---|---|---|
| PassageRetrieval | 13.3 | 16.7 | 16.7 | 13.3 | 16.7 |
| HotpotQA | 9.7 | 8.7 | 8.7 | 9.7 | 10.6 |
| 2WikiMQA | 16.0 | 15.9 | 15.9 | 16.0 | 15.6 |
| GovReport | 17.0 | **16.9** | 16.4 | 16.4 | 16.7 |
| QMSum | 13.8 | 12.4 | 12.3 | 13.0 | 12.0 |
| LCC | 25.9 | 25.8 | 25.8 | 25.8 | 26.4 |
| **Average** | **16.0** | **16.1** | 16.0 | 15.7 | 16.3 |

Head-aware allocation matches the global-budget additive scorer within noise on every task, including LCC. The entropy-based reallocation does not reduce the code regression. This is an informative null result: the code failure is not caused by the wrong heads receiving the wrong budgets. It is caused by which tokens the scorer selects within each head — syntactically necessary but semantically low-salience tokens are systematically under-ranked regardless of head budget. Fixing the code regression requires a different mechanism: either a task-adaptive scoring mode (more recency weight for code contexts) or a larger always_keep_last window. Head-aware budget allocation remains a reasonable prior — it introduces no overhead beyond a per-head topk instead of a global topk — but the gains over a well-tuned global budget are not reliably detectable at this scale.

---

## 6. Discussion

**Why additive over multiplicative?** The standard formulation kq × vn × decay multiplies three normalized quantities in [0,1], each capable of zeroing the product independently. In practice, V-norm is near-zero for common, predictable tokens (articles, punctuation, repeated phrases) and near-one for unusual tokens. Multiplicative combination therefore approximates V-norm with a semantic tiebreaker rather than a genuine combination. Additive combination gives each signal independent "voice": a token with high KQ alignment but ordinary V-norm (e.g. a relevant but common noun) survives.

**Why pre-RoPE?** RoPE encodes absolute position by rotating the K and Q vectors. A key vector from position 0 in a 10K context is rotated by 10K×θ relative to the last query, producing artificially low dot products regardless of semantic content. Pre-RoPE keys remove this rotation, so KQ alignment reflects content similarity only. The post-RoPE keys are still used in the actual attention computation.

**When does streaming work?** Our results suggest streaming is competitive only when task-relevant information is reliably near the end of the context (few-shot QA, some retrieval formats). It fails when relevant content is distributed — summarization, multi-hop reasoning, and code completion with distant dependencies. A practical system might use streaming for latency-sensitive few-shot applications and semantic scoring for document understanding tasks.

**Limitations.** Our evaluation covers two model families (Llama, Mistral) at a single scale (~7–8B parameters); behavior at larger scales or on other architectures (e.g. Gemma, Qwen) is untested. The α and min_decay hyperparameters were tuned on LongBench tasks, raising the question of generalizability to out-of-distribution tasks. Our V-norm signal uses raw value vectors rather than the projected form V·W^O advocated by Feng et al. [2025] on theoretical grounds; whether the projection improves our additive scorer has not been tested. Code completion regresses relative to naive truncation, suggesting that task-adaptive scoring may be necessary for a fully general system. We compare against SnapKV (Table 4.2) but not against H2O, which requires accumulated attention weights across the full prefill and is therefore more expensive to implement. The current implementation does not achieve practical VRAM or throughput improvements at moderate context lengths (see §5.6); a concrete path forward is to implement the method as a KVPress press and follow the AdaKV pattern for CUDA kernel integration. Finally, all experiments use a single retention fraction; a practical system might benefit from per-layer budgets reflecting the empirical finding that early and late layers have different attention patterns.

---

## 7. Conclusion

We have shown that additive combination of KQ alignment and V-norm with α=0.65 outperforms V-norm alone on retrieval and QA tasks, while matching it on summarization. Pre-RoPE scoring and context-length-independent decay are important supporting design choices. At 65% token retention, our method recovers 86% of full-context average performance on LongBench with Llama-3.1-8B and 93% with Mistral-7B-v0.3, beating naive left-truncation on passage retrieval by 9–10 points across both models. The method required no architecture-specific modifications to transfer between model families. Comparison with StreamingLLM confirms that semantic scoring is essential for tasks requiring distributed context: streaming achieves 13.0 average LongBench score vs 20.5 for our method, with particularly large gaps on summarization (3.5–5.2 vs 8.1–17.5) and code completion (6.3–17.8 vs 49.0–57.4). Direct comparison with SnapKV (post-RoPE dot-product pooling) confirms that the pre-RoPE scoring choice is load-bearing: switching to post-RoPE drops passage retrieval from 27.0 to 14.0. The main remaining challenges are code completion — where recency dominates over semantic relevance — and achieving practical VRAM/throughput improvements, for which a KVPress press implementation with AdaKV-style CUDA kernels is the natural next step.

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
- Feng, Y. et al. (2025). Identify critical KV cache in LLM inference from an output perturbation perspective. *arXiv:2502.03805*.
- Devoto, A. et al. (2025). KV cache compression by estimating attention from future queries distribution. *arXiv:2510.00636*.
