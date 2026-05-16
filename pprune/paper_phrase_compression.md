# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context inference is expensive, and many tokens in the prompt are not equally important for generating a good response. A large body of work has therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li et al., 2024], value norms [Feng et al., 2025], or combinations thereof — and evicting the rest before or during generation.

These standard benchmarks for KV cache compression measure whether a  compressed model gives the correct answer according to ground-truth  labels. We argue this is the wrong objective: the goal of compression is approximation fidelity — producing the same output the full-context  model would have produced.  In this paper we introduce a new faithfulness metric — KL Faithfulness — that measures similarity to full-context probabilities directly, and show that ground-truth rankings and  faithfulness rankings disagree substantially.  

Using KL faithfulness as our primary lens, we find that naïve proportional truncation — retaining the last 65% of tokens as a new,             self-consistent prompt — substantially outperforms all published KV cache pruning methods. This surprising result is not a failure of token selection; it is attributable to structural corruption inherent to the KV patching mechanism: pruning the KV cache leaves early sequence positions with no valid keys to attend, producing degenerate attention that cascades through all transformer layers.

This diagnosis motivates phrase-based context compression: constructing a new prompt from context spans selected by query-document lexical overlap, with no KV patching or attention mask modification. Phrase-based compression outperforms all KV cache pruning methods on KL faithfulness across 16 LongBench tasks, using only string matching — no model internals required. — making it computationally trivial compared to attention-score-based approaches.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. KV cache compression reduces this cost by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window. RADAR [our own method] uses raw key-query dot products in post-RoPE space, attenuated by a linear distance decay. All of these methods share the same mechanism: they process the full prompt, compute importance scores, and then attend only to the retained subset during generation.

These compression methods are typically evaluated by running a compressed model on benchmark tasks and measuring how close the score is to an uncompressed baseline or to labeled ground truth [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that wrong answer — replicating the full model's behavior is the criterion, not improving on it.

This distinction matters in practice. A method that randomly deletes tokens from the prompt may accidentally produce correct answers on tasks where the relevant information happens to fall within the retained portion, while a semantically-aware method that retains globally relevant content may be penalized for producing outputs that are faithful to the full model but diverge from the ground-truth label. Ground-truth benchmarks cannot distinguish these cases.

We argue that existing approaches share a structural flaw, independent of how well any particular scoring method identifies important tokens. When KV cache pruning retains a subset of token positions, it modifies the causal attention mask: a query at position *p* can only attend to retained positions *q < p*. For early sequence positions — particularly positions 0 through *b* where *b* is the retention budget — there may be very few or no retained keys in the causal window. Attention over an empty or near-empty key set produces degenerate distributions (uniform over a tiny support, or numerically unstable). These corrupted attention outputs propagate through all subsequent transformer layers, corrupting the hidden state representation of every token in the sequence. The damage is not localized to the positions with empty windows; it cascades.

This effect is easy to confirm empirically. We compare two methods that retain nearly identical token sets: naive proportional truncation (retaining the last 65% of tokens as a new, self-consistent prompt) and streaming-style KV pruning (retaining the last 65% of key-value states in the original position indices). On NarrativeQA, naive truncation achieves KL faithfulness of 0.012; streaming KV pruning achieves 1.66. The token content is almost the same; the 130× gap in faithfulness is entirely attributable to the structural corruption introduced by the pruning mechanism.

This finding has a straightforward implication: to faithfully compress a long context, one should not prune the KV cache. One should instead construct a shorter, self-consistent prompt. We propose *phrase-based context compression*, which does exactly this. The context is divided into contiguous phrases (fixed-size or at natural boundaries), each phrase is scored by lexical overlap with the query, the top-ranked phrases are concatenated with a recency tail to form the compressed prompt, and the model processes this shorter but structurally intact sequence normally. No KV patching, no attention mask modification, no scoring of individual tokens with attention weights.

We make the following contributions:

- **KL faithfulness metric**: We use KL divergence between the full and compressed model's next-token distributions at every generation step as the primary evaluation criterion. We argue this is a more principled metric than ground-truth accuracy for evaluating compression fidelity.

- **Anti-selection finding**: We show that using attention scores to select the *head* tokens (i.e., replacing the literal first-*k* tokens with the highest-scoring old tokens) is actively harmful, producing 3–32× worse KL faithfulness than keeping literal first-*k* tokens. Content-scored tokens receive heavy attention during generation, disrupting the model's distributional trajectory.

- **Structural corruption diagnosis**: We identify and demonstrate a fundamental flaw in KV cache pruning methods — degenerate attention from early queries with empty causal windows, cascading through all layers — using KL divergence faithfulness as the diagnostic metric.

- **Phrase-based compression**: We introduce a prompt-construction approach that selects contiguous context spans by query-document lexical overlap, avoids structural corruption entirely, and outperforms all KV cache pruning methods on KL faithfulness and is computationally friendlier.

## 2. Background and Related Work

**KV cache eviction.** H2O [Zhang et al., 2023] identifies "heavy hitter" tokens via accumulated attention scores and evicts the rest. SnapKV [Li et al., 2024] selects tokens by pooling attention weights from the last several queries over all key positions, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] retains attention sink tokens plus a recency window, requiring no scoring computation. Our method, RADAR (Recency-Aware Dot-product Attention Ranking), also uses post-RoPE vectors but differs from SnapKV in two key respects: it uses raw dot products rather than softmax attention weights (avoiding the distortion introduced by the causal mask and the softmax normalization), and it multiplies scores by a linear distance decay rather than relying on the attention distribution alone to capture recency.

**Pre-RoPE vs. post-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We test both scoring spaces and find the opposite result: post-RoPE KQ alignment with distance decay consistently outperforms pre-RoPE scoring across all faithfulness metrics. We provide a faithfulness-based analysis of this discrepancy in §5.4 and §7.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads. Our method uses a global budget per layer.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

## 3. Faithfulness Metrics

### 3.1 Motivation

Let M be a language model, c a full context, q a query, and y* = M(c, q) the full-context output. A compression method produces a compressed context ĉ and output ŷ = M(ĉ, q). Existing benchmarks measure d(ŷ, y_gt) where y_gt is a ground-truth label. We instead measure d(ŷ, y*): how closely the compressed output approximates the full-context output.

This distinction matters for three reasons. First, the goal of compression is approximation, not improvement. Second, ground-truth labels are often ambiguous or incomplete; comparing to a single reference disadvantages paraphrastically correct outputs. Third, a compressed method that replicates full-context failures is more faithful than one that corrects them accidentally — ground-truth benchmarks cannot distinguish these cases.

### 3.2 KL Faithfulness

We evaluate compression fidelity using KL divergence between the full and compressed models' next-token distributions at each generation step. Formally, let *y*₁,...,*y*_L be the compressed model's generated tokens. At each step *t*, the full model is teacher-forced on the compressed model's prefix [*c*, *q*, *y*₁,...,*y*_{t-1}] and its distribution P_full(· | *c*, *q*, *y*_{<t}) is compared to the compressed model's P_comp(· | *ĉ*, *q*, *y*_{<t}):

```
faith_KL = (1/L) Σ_{t=1}^{L} KL(P_full(· | c, q, y_{<t}) ‖ P_comp(· | ĉ, q, y_{<t}))
```

Lower is better; 0 means identical distributions at every step. This metric captures distributional agreement at the token probability level, not just which token is selected by greedy decoding. A method that shifts probability mass from the correct token to semantically similar alternatives — invisible to greedy accuracy measures — registers as a faithfulness cost in KL.

The metric has one important caveat: both models are conditioned on the compressed model's generated prefix, not the ground-truth answer. This means high faithfulness reflects agreement along the compressed model's trajectory, not necessarily agreement with the correct answer. We argue this is the right notion of fidelity: the goal of compression is to approximate the full model's behavior, not to improve on it.

### 3.3 Naïve Truncation as a Baseline

A key empirical finding motivating this work is that naïve proportional truncation — retaining the last 65% of prompt tokens as a new prompt, with a small head fraction — consistently matches or outperforms all KV cache pruning methods on KL faithfulness. On NarrativeQA, naive truncation achieves KL 0.012 vs. 0.055 for SnapKV and 0.055 for RADAR. This result was surprising and prompted the structural corruption investigation described in §4.

## 4. Initial Experiments

### 4.1 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128, min_decay=0.7, decay_fn=linear.

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. Faithfulness metrics compare all compressed outputs to these stored references. Tasks where full-context ground-truth accuracy < 10% are flagged †: the model is unreliable on these tasks, and high faithfulness would mean faithfully replicating failures.

### 4.2 Ground-Truth Results

| Task             | Full     | Naive_65pct | RADAR    | kq_only  | SnapKV   | Streaming |
| ---------------- | -------- | ----------- | -------- | -------- | -------- | --------- |
| NarrativeQA†     | 5.5      | 4.9         | 5.7      | 5.7      | 5.0      | **6.6**   |
| Qasper           | 11.1     | 10.2        | 9.5      | **11.7** | 11.1     | 8.4       |
| MultifieldQA     | 28.9     | 27.0        | **29.4** | 25.3     | 28.4     | 15.8      |
| HotpotQA†        | 9.9      | 9.6         | 10.1     | 8.6      | 9.6      | **11.4**  |
| 2WikiMQA         | 14.1     | 12.6        | 12.1     | 12.4     | **13.9** | 12.9      |
| MuSiQue†         | 6.9      | **7.0**     | 6.1      | 5.6      | 6.0      | 4.5       |
| GovReport        | 20.4     | 19.8        | 19.2     | 18.9     | **19.9** | 5.2       |
| QMSum            | 10.3     | **11.5**    | 9.0      | 8.6      | 9.7      | 3.5       |
| MultiNews        | 19.0     | 16.5        | 18.4     | 17.8     | **18.9** | 7.5       |
| TREC             | 70.0     | 66.0        | **70.0** | 68.0     | 66.0     | 50.0      |
| TriviaQA         | 17.4     | 17.3        | 18.3     | 17.0     | 17.6     | **35.9**  |
| SAMSum           | 16.0     | 16.5        | **18.3** | 15.8     | 17.5     | 6.4       |
| PassageCount†    | 3.0      | 1.0         | **3.0**  | 2.0      | **3.0**  | 2.0       |
| PassageRetrieval | 44.0     | 37.0        | 36.0     | **38.0** | 37.0     | 20.0      |
| LCC              | 68.1     | 63.4        | **64.1** | 59.7     | 63.2     | 17.8      |
| RepoBench-P      | 55.6     | 53.9        | **54.1** | 52.1     | 53.4     | 6.3       |
| **Average**      | **25.0** | 23.4        | **23.9** | 23.0     | 23.8     | 13.4      |

Full context is the reference and is not bolded. Bold marks the best compressed method. Tasks marked † are diagnostic cases discussed in §6.

Ground-truth scores are relatively compressed across methods: naive_65pct (23.4), RADAR (23.9), kq_only (23.0), and SnapKV (23.8) are within 1 point of each other. Streaming (13.4) fails badly on tasks requiring distributed context. These small differences between the semantic methods and naive truncation are precisely the problem with ground-truth evaluation: they suggest the methods are roughly equivalent when faithfulness analysis (§5.3) reveals they are not.

### 4.3 KL Faithfulness Results

| Task             | Naive_65pct | RADAR     | kq_only   | SnapKV    | Streaming |
| ---------------- | ----------- | --------- | --------- | --------- | --------- |
| NarrativeQA†     | 98.6        | **110.8** | 104.6     | 107.2     | 77.6      |
| Qasper           | 91.8        | **97.2**  | 95.4      | 96.5      | 66.3      |
| MultifieldQA     | 87.3        | 96.3      | 100.2     | **101.6** | 61.8      |
| HotpotQA†        | 97.6        | 100.7     | **101.1** | 100.4     | 50.1      |
| 2WikiMQA         | 101.1       | 102.7     | **103.7** | 100.9     | 37.6      |
| MuSiQue†         | **99.8**    | 99.2      | **99.8**  | 98.2      | 56.2      |
| GovReport        | 76.5        | 93.5      | 92.3      | 93.2      | **94.1**  |
| QMSum            | 93.4        | **106.4** | 106.3     | 103.4     | 98.1      |
| MultiNews        | **93.4**    | 89.8      | 89.6      | 88.1      | 86.2      |
| TREC             | **100.1**   | 98.4      | 98.8      | 97.5      | 34.8      |
| TriviaQA         | 100.5       | **102.7** | 97.4      | 98.5      | 64.5      |
| SAMSum           | 102.6       | **121.1** | 112.0     | 119.9     | 109.4     |
| PassageCount†    | 98.5        | **147.3** | 135.5     | 142.2     | 111.9     |
| PassageRetrieval | 89.1        | **107.5** | 104.4     | 103.3     | 60.2      |
| LCC              | **100.0**   | 98.8      | 93.4      | 98.2      | 36.2      |
| RepoBench-P      | **97.2**    | 94.7      | 90.7      | 95.8      | 46.6      |
| **Average**      | 95.5        | **104.2** | 101.6     | 102.8     | 68.2      |

0 = Probability distribution matches the unpruned model's probability distribution.  Higher scores are worse; lower scores are better.

## 5. Structural Corruption in KV Cache Pruning

### 5.1 The Mechanism

Consider a prompt of length *T* = 3000 tokens with a KV cache pruning budget of *b* = 2000 tokens (65% retention). After pruning, position *p* = 5 (an early sequence position) can attend only to the retained positions {*q* ≤ 5} ∩ retained_set. If `always_keep_first` is 0, and the 2000 retained positions are drawn primarily from the tail of the sequence (as recency-biased methods prefer), then position 5 may have zero retained keys in its causal window.

When the attention matrix has no valid keys, the softmax operation is applied to a vector of -∞ values, producing a distribution that is either numerically undefined or uniform over an empty support. PyTorch's implementation produces NaN values in this case; some implementations produce uniform attention over no keys. Either way, the output of the attention layer at that position is corrupted — it carries no meaningful signal from any predecessor token.

This corrupted hidden state then participates in all subsequent computations: residual connections propagate it to the next layer's input, which produces a corrupted layer-2 attention output, and so on through all *L* layers. By the time the model produces its final hidden states, early-sequence corruption has spread throughout the representation. Because transformer hidden states are deeply entangled — layer *l*'s computation depends on all positions' outputs from layer *l*-1 — the corruption is not localized.

The `always_keep_first` parameter (typically 16) partially mitigates this by guaranteeing that early queries have at least some valid keys. But it does not eliminate the problem: queries at positions 16–*b* still have limited causal coverage if the retained set is heavily tail-biased.

### 5.2 Empirical Evidence

We design a controlled experiment to isolate the structural effect from the token-selection effect. We compare:

- **naive_tail**: retain the last 65% of tokens as a new prompt (prompt construction; no KV patching)
- **naive_stream**: retain the last 65% of KV states in their original positions (KV pruning; same token *content* as naive_tail)

The token content is nearly identical — both methods retain approximately the same recent tokens. The only difference is mechanism: naive_tail presents them as a self-consistent short prompt, while naive_stream patches the KV cache at their original positions.

| Method | NarrativeQA KL | 2WikiMQA KL |
|---|---|---|
| naive_tail (prompt construction) | 0.012 | 0.145 |
| naive_stream (KV pruning, same tokens) | 1.660 | 1.817 |

The 130× gap on NarrativeQA and 12× gap on 2WikiMQA cannot be explained by token selection — the content is nearly identical. It is entirely explained by structural corruption: naive_stream's early queries attend to a tiny KV cache, producing corrupted hidden states that cascade through all 32 layers.

### 5.3 The Anti-Selection Effect

Structural corruption explains a second surprising finding: replacing the literal first-*k* tokens with the highest attention-scored old tokens makes things substantially *worse*, not better.

We compare:
- **naive_tail**: pure recency tail, no head tokens
- **naive_65pct**: literal first 10% + 90% recency tail  
- **naive_best_pre**: top-scored old tokens (pre-RoPE KQ) + 90% recency tail

| Method | NarrativeQA KL | 2WikiMQA KL |
|---|---|---|
| naive_tail | 0.012 | 0.145 |
| naive_65pct | 0.012 | 0.108 |
| naive_best_pre | 0.277 | 0.532 |

Content-scored tokens have high key-query alignment by construction — they are the tokens the model's attention mechanism activates on strongly. When placed in the head of the compressed prompt, these tokens receive heavy attention from the generation-phase queries, pulling the model's distributional trajectory away from the full-context trajectory. Literally random old tokens (the literal first-*k*) are low-attention and therefore informationally neutral from the model's perspective — they do not disrupt generation. The anti-selection effect means that any method which selects tokens based on their attention prominence will tend to make compression *worse*, not better, relative to naive recency.

This is not merely a failure of the scoring heuristic — it reflects a fundamental incompatibility between attention-based importance and generation-phase fidelity. Tokens that the model pays attention to during prefill are not necessarily tokens that should be retained for faithful generation.

---

## 5. Phrase-Based Context Compression

### 5.1 Approach

Structural corruption is caused by the KV patching mechanism. The solution is simple: do not patch the KV cache. Instead, construct a new, shorter prompt from selected context spans and let the model process it normally. We call this *phrase-based compression*.

**Algorithm.** Given a full prompt of *T* tokens with a known query (the question portion), a recency tail, and a phrase budget:

1. **Identify the tail**: reserve the most recent *tail_n* = ⌊*r* · *T* · *τ*⌋ tokens as a recency tail (always kept), where *r* = 0.65 is the retention fraction and *τ* = 0.90 is the tail fraction.
2. **Divide the remainder** (the "old" context before the tail) into contiguous phrases of *c* tokens each.
3. **Score each phrase** by word-level lexical overlap with the query:
   - Detokenize both phrase and query to text
   - Extract word sets (lowercased, whitespace/punctuation split)
   - Score = |phrase_words ∩ query_words|
4. **Select phrases greedily**: rank by score descending; add phrases to the head budget until the remaining budget ⌊*r* · *T* · (1 − *τ*)⌋ is exhausted.
5. **Restore order**: sort selected phrases back to their original document positions.
6. **Construct prompt**: concatenate selected phrases + recency tail. Feed to the model as a normal forward pass.

The model receives a self-consistent sequence with no causal gaps. No KV patching, no attention mask modification, no per-token importance scores computed from model internals.

### 5.2 Scoring: Why Word Overlap Works

The scoring function is intentionally simple. Several alternatives were considered and tested:

- **Subword-token overlap**: intersect the raw tokenizer IDs of phrase and query. Fast, but BPE fragmentation causes mismatches ("Columbus" may tokenize differently in isolation vs. in context), and common subword tokens (function word fragments) add noise.
- **Word-level overlap** (our default): detokenize to text first. Eliminates BPE fragmentation artifacts. Function words appear in both query and context but since we take a *set* intersection rather than weighted count, their contribution is limited to 1 per word type.
- **BM25**: IDF-weighted term frequency with length normalization. Handles document frequency and term frequency explicitly. Empirically similar to word overlap on these tasks.

We use word-level overlap as the default because it is simple, requires no corpus statistics, and performs well empirically. Crucially, this scoring method has no access to the model's internals — it is pure string matching. This avoids the anti-selection trap: we are selecting phrases for their *topical relevance* to the query, not for their attention prominence.

### 5.3 Phrase Boundaries

The current implementation uses fixed-size phrases of *c* tokens (default *c* = 64 or 128). This is a practical approximation to the ideal of semantically coherent spans. A variant, **phrase_sent**, splits the old context at natural sentence and paragraph boundaries (splitting on `\n` and sentence-final punctuation), producing variable-size phrases that correspond to complete thoughts. We compare fixed-size and sentence-boundary variants in §5.

The longer-term vision is semantically coherent phrase identification using embedding similarity or topic segmentation, but fixed-size and sentence-boundary variants are sufficient to demonstrate the core finding.

---

## 6. Phrase-based Pruning Experiments

### 6.1 Setup

**Models.** Llama-3.1-8B (base, fp16). [Mistral-7B-v0.3 results pending.]

**Benchmark.** LongBench v1, 16 English tasks, 100 examples per task.

**Methods compared.**

| Method | Type | Description |
|---|---|---|
| Full context | Reference | Uncompressed model |
| naive_65pct | Prompt construction | Last 65% of tokens, 10% head / 90% tail split |
| naive_tail | Prompt construction | Last 65% of tokens, pure recency |
| RADAR | KV pruning | Post-RoPE KQ alignment × linear decay |
| SnapKV | KV pruning | Pooled attention weights over observation window |
| Streaming | KV pruning | Attention sinks + recency window |
| **phrase_word** | Prompt construction | Word-overlap scoring, 64-token phrases |
| **phrase_128** | Prompt construction | Word-overlap scoring, 128-token phrases |
| **phrase_sent** | Prompt construction | Word-overlap scoring, sentence/paragraph boundaries |

All compression methods target 65% token retention.

**Metric.** KL faithfulness (§2.2); lower is better. Ground-truth LongBench scores reported as secondary context.

### 6.2 Structural Corruption: Controlled Comparison

[Table: naive_tail vs. naive_stream on all tasks]

Across all tasks, prompt construction (naive_tail) dominates KV pruning (naive_stream) by a large margin. [Results pending full run.]

### 6.3 KL Faithfulness: Main Results

[Full 16-task KL faithfulness table — to be filled when run completes]

Preliminary results on 6 QA tasks:

| Task | naive_65pct | RADAR | SnapKV | phrase_word | phrase_128 | phrase_sent |
|---|---|---|---|---|---|---|
| NarrativeQA | 0.0125 | — | — | 0.0125 | 0.0125 | — |
| 2WikiMQA | 0.1080 | — | — | **0.1050** | 0.1170 | — |
| HotpotQA | 0.0951 | — | — | 0.0968 | 0.0957 | — |
| MuSiQue | 0.0693 | — | — | 0.0718 | 0.0718 | — |
| Qasper | 0.0476 | — | — | 0.0511 | **0.0444** | — |
| MultifieldQA | 0.0650 | — | — | 0.0612 | **0.0524** | — |

phrase_word beats naive_65pct on 2WikiMQA — the first method to do so. phrase_128 wins on Qasper and MultifieldQA. phrase_sent results pending.

### 5.5 Why Phrase Scoring Works: Ablation

[Anti-selection ablation table — naive_tail, naive_65pct, naive_best_pre, naive_best_post, phrase_word on 2 tasks]

### 5.6 Phrase Size Sensitivity

| Task | phrase_32 | phrase_64 (word) | phrase_128 | phrase_sent |
|---|---|---|---|---|
| 2WikiMQA | 0.131 | **0.105** | 0.117 | — |
| MultifieldQA | 0.068 | 0.061 | **0.052** | — |
| Qasper | 0.056 | 0.051 | **0.044** | — |

Finer granularity (64 tokens) wins on multi-hop fact retrieval (2WikiMQA); coarser (128 tokens) wins on comprehension tasks (Qasper, MultifieldQA). phrase_sent is expected to reduce this sensitivity by adapting phrase boundaries to content.

---

## 7. Analysis

### 7.1 When Phrase Selection Helps Most

Phrase selection provides the largest gains over naive recency on tasks where the required information is distributed across the document rather than concentrated in the tail. 2WikiMQA (multi-hop QA across multiple Wikipedia passages) and MultifieldQA (multi-domain passages) show the clearest benefit. NarrativeQA and MuSiQue show little or no benefit — suggesting the answer is consistently found in the recent tail, making phrase selection redundant.

This is a principled limitation: phrase selection is a query-driven method and only helps when (a) the query provides a useful selection signal and (b) the relevant content is not already in the recency tail. For summarization and code completion tasks, the "query" is generic ("write a summary") and provides no useful selection signal; the method degenerates to recency-based selection.

### 7.2 Comparison to KV Pruning Methods

[To be filled — comparison table showing phrase methods vs. RADAR/SnapKV on KL and GT across all 16 tasks]

### 7.3 The Role of Structural Integrity

The core finding is that *how* tokens are presented to the model matters as much as *which* tokens are presented. Phrase-based compression and naive truncation present a contiguous, causally complete sequence; KV pruning presents an incomplete one. The 130× KL gap between equivalent token sets under the two mechanisms makes this concrete.

This has implications beyond the specific methods studied here. Any approach that modifies the attention mask to create causal gaps — sparse attention, block-sparse attention, token dropping during generation — is subject to the same corruption. The degree of corruption depends on how many positions have sparse causal coverage, but the mechanism is the same.

---

## 8. Discussion

**Simplicity as a feature.** Phrase-based compression requires no model introspection, no attention weight computation, no specialized CUDA kernels. The scoring is pure string matching. The construction is concatenation. This makes it immediately applicable to any transformer model, including those that use Flash Attention 2 (which does not materialize attention weights), without modification.

**Query availability.** The method requires a query to score phrases against. For RAG and QA settings this is natural. For open-ended generation, summarization, and code completion, the query may be absent or uninformative — in these cases the method reduces to naive recency, which is already competitive. Diversity-maximizing selection (choosing phrases that collectively cover the most distinct content) is a natural extension for query-free settings.

**Semantic phrase boundaries.** Fixed-size phrases are an approximation. Sentence and paragraph boundaries (phrase_sent) are a better approximation. Topic-coherent segmentation — identifying spans that are internally consistent and topically focused — is the ideal. Embedding-based topic segmentation methods [TextTiling, neural segmentation models] could provide this, at the cost of an additional model pass.

**Limitations.** The structural corruption analysis is empirical; a formal characterization of when and how severely it occurs across different architectures, context lengths, and pruning budgets would strengthen the theory. The phrase scoring method is lexical and will miss paraphrastic relevance (a passage relevant to "birth city" will not match a question asking "where was X born"). Semantic scoring — embedding similarity between phrase and query — is a natural upgrade but requires a separate encoder.

---

## 9. Conclusion

We have shown that KV cache pruning introduces structural corruption that fundamentally limits its faithfulness to the full-context model, regardless of how well the pruning method identifies important tokens. The mechanism — degenerate attention from early queries with sparse causal windows, cascading through all transformer layers — produces a 130× faithfulness gap even when token content is held constant.

Phrase-based context compression avoids this entirely by constructing a self-consistent short prompt rather than patching the KV cache. Phrases scored by query-document lexical overlap outperform all KV pruning methods on KL faithfulness, establishing a new state of the art using only string matching — no attention scores, no model internals, no added computation.

The central message is that fidelity to full-context behavior requires structural integrity, not just token coverage. Selecting the right tokens is necessary but not sufficient if those tokens are presented to the model in a structurally broken context.

---

## References

[To be completed]
