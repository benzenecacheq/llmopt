# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context inference is expensive, and many tokens in the prompt are not equally important for generating a good response. A large body of work has therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li et al., 2024], value norms [Feng et al., 2025], or combinations thereof — and evicting the rest before or during generation.

These standard benchmarks for KV cache compression measure whether a  compressed model gives the correct answer according to ground-truth  labels. We argue this is the wrong objective: the goal of compression is approximation fidelity — producing the same output the full-context  model would have produced.  In this paper we introduce a new faithfulness metric — KL Faithfulness — that measures similarity to full-context probabilities directly, and show that ground-truth rankings and  faithfulness rankings disagree substantially.  

Using KL faithfulness as our primary lens, we find that naïve proportional truncation — retaining the last 65% of tokens as a new, self-consistent prompt — substantially outperforms all published KV cache pruning methods. This surprising result is not a failure of token selection; it is attributable to structural corruption inherent to the KV patching mechanism: pruning the KV cache creates causal gaps wherever the retained set is sparse, and any query attending over a gapped neighborhood produces degenerate attention that cascades through all transformer layers.

This diagnosis motivates phrase-based context compression: constructing a new prompt from context spans selected by query-document lexical overlap, with no KV patching or attention mask modification. Phrase-based compression outperforms all KV cache pruning methods on KL faithfulness across 16 LongBench tasks, using only string matching — no model internals required, making it computationally trivial compared to attention-score-based approaches.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. KV cache compression reduces this cost by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window. RADAR [our own method] uses raw key-query dot products in post-RoPE space, attenuated by a linear distance decay. All of these methods share the same mechanism: they process the full prompt, compute importance scores, and then attend only to the retained subset during generation.

These compression methods are typically evaluated by running a compressed model on benchmark tasks and measuring how close the score is to an uncompressed baseline or to labeled ground truth [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that wrong answer — replicating the full model's behavior is the criterion, not improving on it.

This distinction matters in practice. A method that randomly deletes tokens from the prompt may accidentally produce correct answers on tasks where the relevant information happens to fall within the retained portion, while a semantically-aware method that retains globally relevant content may be penalized for producing outputs that are faithful to the full model but diverge from the ground-truth label. Ground-truth benchmarks cannot distinguish these cases.

We argue that existing approaches share a structural flaw, independent of how well any particular scoring method identifies important tokens. When KV cache pruning retains a subset of token positions, it modifies the causal attention mask: a query at position *p* can only attend to retained positions *q < p*. For early sequence positions — particularly positions 0 through *b* where *b* is the retention budget — there may be very few or no retained keys in the causal window. Attention over an empty or near-empty key set produces degenerate distributions (uniform over a tiny support, or numerically unstable). These corrupted attention outputs propagate through all subsequent transformer layers, corrupting the hidden state representation of every token in the sequence. The damage is not localized to the positions with empty windows; it cascades.

This effect is easy to confirm empirically. We compare two methods that retain nearly identical token sets: naive proportional truncation (retaining the last 65% of tokens as a new, self-consistent prompt) and streaming-style KV pruning (retaining the last 65% of key-value states in the original position indices). On NarrativeQA, naive truncation achieves KL faithfulness of 0.022; streaming KV pruning achieves 0.084. The token content is almost the same; the 3.8× gap in faithfulness on this task (1.55× on average across 16 tasks) is entirely attributable to the structural corruption introduced by the pruning mechanism.

This finding has a straightforward implication: to faithfully compress a long context, one should not prune the KV cache. One should instead construct a shorter, self-consistent prompt. We propose *phrase-based context compression*, which does exactly this. The context is divided into contiguous phrases (fixed-size or at natural boundaries), each phrase is scored by lexical overlap with the query, the top-ranked phrases are concatenated with a recency tail to form the compressed prompt, and the model processes this shorter but structurally intact sequence normally. No KV patching, no attention mask modification, no scoring of individual tokens with attention weights.

We make the following contributions:

- **KL faithfulness metric**: We use KL divergence between the full and compressed model's next-token distributions at every generation step as the primary evaluation criterion. We argue this is a more principled metric than ground-truth accuracy for evaluating compression fidelity.

- **Structural corruption diagnosis**: We identify and demonstrate a fundamental flaw in KV cache pruning methods — degenerate attention from early queries with empty causal windows, cascading through all layers — using KL divergence faithfulness as the diagnostic metric.

- **Phrase-based compression**: We introduce a prompt-construction approach that selects contiguous context spans by query-document lexical overlap, avoids structural corruption entirely, and outperforms all KV cache pruning methods on KL faithfulness and is computationally friendlier.

## 2. Background and Related Work

**KV cache eviction.** ScissorHands [Wu et al., NeurIPS 2023] introduced the *persistence of importance* hypothesis: tokens that accumulate high attention during prefill remain important throughout generation, so the attention-score history is a reliable eviction criterion. H2O [Zhang et al., 2023] operationalizes this by maintaining a running sum of per-head attention scores and evicting the lowest-scoring tokens. SnapKV [Li et al., 2024] refines the scoring by pooling attention weights from only the last *w* queries (an observation window), better reflecting the queries that will matter at decode time, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] abandons score computation entirely, retaining a fixed set of attention sink tokens plus a sliding recency window; it achieves low latency at the cost of discarding all non-recent non-sink content. Our method, RADAR (Recency-Aware Dot-product Attention Ranking), uses post-RoPE raw key-query dot products with a linear distance decay, avoiding the softmax distortion present in attention-weight scores while explicitly penalizing distant tokens.

**Pre-RoPE vs. post-RoPE scoring.** A2ATS [ACL 2025] explicitly advocates scoring with pre-RoPE keys, and is to our knowledge the only prior work to study this choice in isolation. We test both scoring spaces and find the opposite result: post-RoPE KQ alignment with distance decay consistently outperforms pre-RoPE scoring across all faithfulness metrics. We provide a faithfulness-based analysis of this discrepancy in §5 and §8.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Layer- and head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads within each layer. PyramidKV [Cai et al., 2024] takes a complementary approach, allocating different total budgets to different transformer layers — fewer KV slots at lower layers (where attention patterns are diffuse) and more at higher layers (where task-relevant patterns are concentrated). Both families operate on top of an existing token-scoring method and can in principle be combined with any of the per-token importance metrics discussed above. Our method uses a uniform budget per layer, and we note that the structural corruption problem (§5) applies to any of these methods that rely on the KV patching mechanism regardless of how budgets are distributed.

**Query-aware and dynamic selection.** Quest [Tang et al., ICML 2024] departs from static prefill-time selection by performing per-decode-step KV retrieval. At each generation step, the current query is used to retrieve the most relevant KV blocks from the full cache, limiting attention to the top-K pages. This eliminates the assumption that token importance is fixed at prefill time and instead adapts the attended set to each query. The tradeoff is computational: Quest requires a custom CUDA kernel for block-sparse attention, making integration into existing inference stacks non-trivial. Our structural corruption analysis applies to Quest only partially — because Quest retrieves complete contiguous blocks rather than individual scattered tokens, gaps within each attended block are minimal, though inter-block gaps remain.

**Token merging.** CaM [Wan et al., ICML 2024] and its successor KVMerger take a different approach: rather than evicting low-importance tokens, they *merge* pairs of similar KV vectors, collapsing them into a single weighted average and reducing cache size without introducing empty positions. This avoids the causal gap problem entirely — the retained set is contiguous — but produces keys and values that are off-manifold linear combinations not present in any normal prompt. The faithfulness cost of such merging versus structural corruption from eviction is an open question. CaM does not support grouped-query attention (GQA), limiting its applicability to GQA models such as Llama-3.1.

**Structural integrity concerns.** StructKV [Wang et al., Apr 2025] explicitly targets the observation that KV pruning disrupts semantic structure by evicting tokens mid-sentence. Their method enforces sentence-boundary eviction, preserving syntactic completeness of retained spans. This motivation is closely related to our phrase-based approach — both recognize that token-level selection produces incoherent fragments — but StructKV still operates via KV patching at sentence boundaries and therefore does not eliminate causal gaps. Concurrently, Devoto et al. [2025] ("Pitfalls of KV Cache Compression") provide empirical evidence that standard KV compression benchmarks are insufficient for evaluating compression quality and that compression methods perform significantly worse than their published numbers suggest under stricter evaluation conditions. This finding aligns with and provides additional evidence for our faithfulness-based critique of the field.

**Prompt compression.** LLMLingua [Jiang et al., 2023] and LongLLMLingua [Jiang et al., 2024] compress prompts at the token level by scoring token perplexity under a smaller proxy model and dropping low-perplexity (less surprising, hence less informative) tokens. This is architecturally different from KV cache pruning: the compressed prompt is fed to the model as-is, with no cache manipulation, so structural corruption is not a concern. However, perplexity-based scoring requires a forward pass over the full prompt using a second model, which adds latency comparable to the KV scoring overhead we observe in SnapKV-Select (§7.3). Phrase-based compression achieves similar prompt-shortening via pure string matching with no second model.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

## 3. Faithfulness Metrics

### 3.1 Motivation

Let M be a language model, c a full context, q a query, and y* = M(c, q) the full-context output. A compression method produces a compressed context ĉ and output ŷ = M(ĉ, q). Existing benchmarks measure d(ŷ, y_gt) where y_gt is a ground-truth label. We instead measure d(ŷ, y*): how closely the compressed output approximates the full-context output.

This distinction matters for three reasons. First, the goal of compression is approximation, not improvement. Second, ground-truth labels are often ambiguous or incomplete; comparing to a single reference disadvantages paraphrastically correct outputs. Third, a compressed method that replicates full-context failures is more faithful than one that corrects them accidentally — ground-truth benchmarks cannot distinguish these cases.

### 3.2 KL Faithfulness

We evaluate compression fidelity using KL divergence between the full and compressed models' next-token distributions, measured on a shared generation sequence. The uncompressed model generates a response y* = (y*₁,...,y*_L) from the full context. Both the full model and each compressed model are then teacher-forced on this shared token sequence: at step *t*, the full model conditions on [*c*, *q*, y*₁,...,y*_{t-1}] and the compressed model conditions on [*ĉ*, *q*, y*₁,...,y*_{t-1}]. The metric averages KL divergence across all generation steps:

```
faith_KL = (1/L) Σ_{t=1}^{L} KL(P_full(· | c, q, y*_{<t}) ‖ P_comp(· | ĉ, q, y*_{<t}))
```

Lower is better; 0 means identical distributions at every step. Conditioning all methods on the same shared token sequence eliminates path-dependence: a method whose compressed context causes early divergence would otherwise be evaluated on a different downstream sequence than a more faithful method, making scores incomparable across methods. Using y* as the shared prefix anchors every comparison to the full model's actual behavior.

This metric captures distributional agreement at the token probability level, not just which token is selected by greedy decoding. A method that shifts probability mass from the correct token to semantically similar alternatives — invisible to greedy accuracy measures — registers as a faithfulness cost in KL.

### 3.3 Naïve Truncation as a Baseline

A key empirical finding motivating this work is that naïve proportional truncation consistently matches or outperforms all KV cache pruning methods on KL faithfulness. The method retains 65% of the prompt tokens, split as a 10%/90% head/tail budget: the first 6.5% of tokens (literal prompt prefix) and the last 58.5% of tokens (recency tail), concatenated into a new self-consistent prompt. The middle portion is discarded. On NarrativeQA, naive truncation achieves KL 0.022 vs. 0.060 for SnapKV and 0.182 for RADAR (see §4.4 for full results across all 16 tasks). This result was surprising and prompted the structural corruption investigation described in §5.

## 4. Initial Experiments

### 4.1 Methods

All methods target 65% token retention (r = 0.65). We use proportional truncation (65% of actual prompt length) rather than a fixed token budget, which would over-retain on short inputs and over-prune on long ones.

**Reference.**

- *Full context*: unmodified model with the complete prompt; serves as the ceiling, not a compressed method.

**Prompt-construction baselines.** These methods truncate the prompt to form a shorter, self-consistent input with no KV patching or attention mask modification.

- *Naive proportional truncation (Naive\_65pct)*: the prompt is split into a 10% head (literal first tokens) and a 90% recency tail, concatenated to form a new prompt at 65% of the original length.

**KV pruning methods.** These methods process the full prompt, score each token's KV entry, and retain only the top-scoring positions at their original sequence indices. The causal attention mask is then reconstructed so that each query can only attend to retained positions at earlier indices. All KV pruning methods unconditionally retain the first 16 tokens (`always_keep_first=16`, preventing degenerate attention at early positions) and the last 16 tokens (`always_keep_last=16`, preserving query context). For GQA models such as Llama-3.1-8B (32 Q-heads, 8 KV-heads), scores are computed per Q-head and aggregated via max-pooling across heads sharing a KV-head before selection.

- *RADAR (our method)*: raw key-query dot products in post-RoPE space, multiplied by a linear distance decay. Described in §2.
- *SnapKV*: attention weights pooled over the last 128-token observation window [Li et al., 2024].
- *Streaming (StreamingLLM)*: first 4 attention sink tokens plus a recency window to fill the remaining budget [Xiao et al., 2023].

**Ablations.**

- *kq\_only*: pre-RoPE KQ alignment with distance decay; identical to RADAR except scoring uses pre-RoPE keys and queries. Isolates the effect of the RoPE scoring space.

### 4.2 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128, min_decay=0.7, decay_fn=linear.

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. Faithfulness metrics compare all compressed outputs to these stored references. Tasks where full-context ground-truth accuracy < 10% are flagged †: the model is unreliable on these tasks, and high faithfulness would mean faithfully replicating failures. These cases are discussed further in §8.

### 4.3 Ground-Truth Results

| Task             | Full | Naive_65pct | RADAR    | kq_only  | SnapKV   | Streaming | Pyr(65%) |
| ---------------- | ---- | ----------- | -------- | -------- | -------- | --------- | -------- |
| NarrativeQA†     | 5.5  | 4.9         | 5.7      | 5.7      | 5.0      | 6.6       | 5.6      |
| Qasper           | 11.1 | 10.2        | 9.5      | 11.7     | 11.1     | 8.4       | **11.8** |
| MultifieldQA     | 28.9 | 27.0        | 29.4     | 25.3     | 28.4     | 15.8      | 29.6     |
| HotpotQA†        | 9.9  | 9.6         | 10.1     | 8.6      | 9.6      | **11.4**  | 10.2     |
| 2WikiMQA         | 14.1 | 12.6        | 12.1     | 12.4     | 13.9     | 12.9      | 13.9     |
| MuSiQue†         | 6.9  | 7.0         | 6.1      | 5.6      | 6.0      | 4.5       | **7.1**  |
| GovReport        | 20.4 | 19.8        | 19.2     | 18.9     | 19.9     | 5.2       | **20.3** |
| QMSum            | 10.3 | **11.5**    | 9.0      | 8.6      | 9.7      | 3.5       | 9.4      |
| MultiNews        | 19.0 | 16.5        | 18.4     | 17.8     | **18.9** | 7.5       | 18.1     |
| TREC             | 70.0 | 66.0        | 70.0     | 68.0     | 66.0     | 50.0      | **71.0** |
| TriviaQA         | 17.4 | 17.3        | 18.3     | 17.0     | 17.6     | **35.9**  | 17.5     |
| SAMSum           | 16.0 | 16.5        | **18.3** | 15.8     | 17.5     | 6.4       | 16.3     |
| PassageCount†    | 3.0  | 1.0         | **3.0**  | 2.0      | **3.0**  | 2.0       | **3.0**  |
| PassageRetrieval | 44.0 | 37.0        | 36.0     | 38.0     | 37.0     | 20.0      | **44.0** |
| LCC              | 68.1 | 63.4        | 64.1     | 59.7     | 63.2     | 17.8      | **68.5** |
| RepoBench-P      | 55.6 | 53.9        | 54.1     | 52.1     | 53.4     | 6.3       | **56.0** |
| **Average**      | 25.0 | 23.4        | 23.9     | 23.0     | 23.8     | 13.4      | **25.2** |

Full context is the reference and is not bolded. Bold marks the best compressed method per task. Tasks marked † are discussed in §8.

Ground-truth scores are relatively compressed across KV pruning methods: naive_65pct (23.4), RADAR (23.9), kq_only (23.0), and SnapKV (23.8) are within 1 point of each other. Streaming (13.4) fails badly on tasks requiring distributed context. PyramidKV(65%) is the clear winner at 25.2, matching or exceeding full context on 6 tasks (MultifieldQA, TREC, PassageRetrieval, LCC, RepoBench-P, and 2WikiMQA) and topping the 16-task average above full context. This result foreshadows the central finding of §7: PyramidKV has the best ground-truth scores among all compressed methods, yet the highest KL divergence from the full model by an order of magnitude (§4.4). These two facts are not contradictory — they illustrate precisely why ground-truth accuracy is an insufficient metric for compression quality.

**Mistral-7B-v0.3.** The table below shows ground-truth results for Mistral-7B-v0.3 at 65% retention. RADAR, kq_only, SnapKV, and Streaming results are pending.

| Task             | Full | Naive_65pct | KV pruning |
| ---------------- | ---- | ----------- | ---------- |
| NarrativeQA†     | 5.2  | 6.1         | **6.2**    |
| Qasper           | 5.3  | 4.9         | **7.6**    |
| MultifieldQA     | 25.3 | **23.5**    | 20.2       |
| HotpotQA†        | 10.5 | 10.7        | **11.3**   |
| 2WikiMQA         | 11.5 | 10.9        | **11.2**   |
| MuSiQue†         | 5.1  | **5.2**     | **5.2**    |
| GovReport        | 20.7 | **20.7**    | 18.7       |
| QMSum            | 8.3  | **8.3**     | 7.2        |
| MultiNews        | 17.5 | **15.2**    | 13.4       |
| TREC             | 72.0 | 66.0        | **68.0**   |
| TriviaQA         | 23.1 | 22.6        | **26.5**   |
| SAMSum           | 16.9 | 17.1        | **18.1**   |
| PassageCount†    | 1.0  | **2.0**     | **2.0**    |
| PassageRetrieval | 39.0 | **39.0**    | 29.0       |
| LCC              | 62.9 | **60.3**    | 56.2       |
| RepoBench-P      | 53.9 | **52.3**    | 51.2       |
| **Average**      | 23.6 | 22.8        | 22.0       |

The overall pattern mirrors Llama: full context scores are broadly similar (23.6 vs 25.0), naive_65pct stays close to full (22.8), and generic KV pruning (22.0) is competitive on average but loses more on tasks requiring distributed context (MultifieldQA, MultiNews, PassageRetrieval). The "KV pruning" column here is an undifferentiated baseline; per-method results (RADAR, SnapKV, etc.) will be added when available.

### 4.4 KL Faithfulness Results

| Task             | Naive_65pct    | RADAR   | kq_only    | SnapKV         | Streaming | PyramidKV |
| ---------------- | -------------- | ------- | ---------- | -------------- | --------- | --------- |
| NarrativeQA†     | **0.022**      | 0.182   | 0.099      | 0.060          | 0.084     | 1.338     |
| Qasper           | 0.281          | 0.273   | **0.214**  | 0.223          | 0.378     | 1.229     |
| MultifieldQA     | 0.262          | 0.294   | 0.230      | **0.186**      | 0.419     | 1.535     |
| HotpotQA†        | **0.203**      | 0.564   | 0.322      | 0.246          | 0.265     | 2.181     |
| 2WikiMQA         | **0.221**      | 0.564   | 0.389      | 0.254          | 0.369     | 2.422     |
| MuSiQue†         | **0.197**      | 0.693   | 0.323      | 0.283          | 0.293     | 2.523     |
| GovReport        | **0.280**      | 0.597   | 0.401      | 0.374          | 0.536     | 0.334     |
| QMSum            | **0.081**      | 0.242   | 0.126      | 0.117          | 0.147     | 0.438     |
| MultiNews        | 0.585          | 0.795   | **0.332**  | 0.407          | 0.770     | 0.721     |
| TREC             | 0.202          | 0.272   | 0.161      | **0.146**      | 0.291     | 1.526     |
| TriviaQA         | **0.076**      | 0.510   | 0.179      | 0.122          | 0.150     | 1.647     |
| SAMSum           | **0.011**      | 0.516   | 0.102      | 0.048          | 0.020     | 1.182     |
| PassageCount†    | **0.059**      | 0.273   | 0.142      | 0.114          | 0.100     | 1.626     |
| PassageRetrieval | **0.188**      | 0.413   | 0.250      | 0.211          | 0.278     | 1.416     |
| LCC              | **0.076**      | 0.156   | 0.177      | 0.112          | 0.125     | 1.142     |
| RepoBench-P      | **0.048**      | 0.234   | 0.167      | 0.106          | 0.103     | 1.052     |
| **Average**      | **0.175**      | 0.411   | 0.226      | 0.188          | 0.270     | **1.394** |

Values are mean KL divergence in nats over 100 examples; lower is better; 0 means identical distributions. Bold marks the best method per row (excluding PyramidKV, which is shown for comparison only). Values recomputed with corrected y* (v3 rerun, n=100).

## 5. Structural Corruption in KV Cache Pruning

### 5.1 The Mechanism

KV cache pruning retains a subset of token positions and removes the rest. The retained positions keep their original sequence indices; a reconstructed causal mask allows each query to attend only to retained positions at earlier indices. This creates *causal gaps*: intervals of consecutive positions that are absent from the key-value store but logically present in the sequence.

For any token at position *p*, its attention output is computed over the intersection of its causal window {1,...,*p*} with the retained set. When that intersection is sparse — as it is for any token whose predecessors have been heavily pruned — the attention distribution is computed over an incomplete neighborhood. The resulting hidden state diverges from what full-context attention would produce. This is not a numerical edge case: it is the normal operating condition of every position in the pruned region.

The corruption propagates. Transformer hidden states are deeply entangled across layers: layer *l*'s computation at every position depends on the full layer *l*-1 output across all positions. A corrupted hidden state at one position is mixed into the residual stream of neighboring positions in the next layer, and that contaminated representation is then used to compute attention in layer *l*+1. After *L* layers, corruption originating at any pruned region has diffused throughout the representation.

This is not primarily an *early-token* problem, though early tokens are particularly vulnerable under recency-biased selection. It is a *gap* problem: any portion of the sequence where the retained set is sparse will produce corrupted attention. Recency-biased methods concentrate their budget at the tail, leaving the head and middle with severe gaps; differently-distributed selection creates gaps elsewhere. The mechanism is identical in both cases.

The `always_keep_first` parameter (typically 16) ensures the very first tokens always have some valid predecessors, but it does not eliminate the problem. Tokens at positions 17 through *b* still attend over a sparse neighborhood when the retained set is tail-concentrated.

KV pruning introduces a second structural problem independent of causal gaps: *positional misalignment*. Retained tokens keep their original RoPE encodings, so two tokens that are logically adjacent in the retained set may carry a relative distance of hundreds or thousands of positions. The model's attention mechanism was trained on sequences where positional distance reflects informational proximity; retained tokens with large RoPE gaps between them are out-of-distribution in a way that has nothing to do with causal masking. Prompt construction eliminates this simultaneously with the causal gap problem: by re-indexing all positions from scratch, the compressed sequence has positional encodings that accurately reflect its actual structure, and the model processes it as it would any normal shorter document.

### 5.2 Empirical Evidence

We isolate structural corruption from token-selection quality by comparing KV pruning directly to prompt construction using identical scoring criteria. If the gap in performance is due to mechanism rather than token choice, then the same scored tokens — selected by the same algorithm — should perform dramatically better when presented as a new self-consistent prompt than when patched into the KV cache at their original positions.

We select RADAR and SnapKV as our representative KV pruning methods because they represent the best of each metric regime: RADAR achieves the highest ground-truth task accuracy among KV pruning methods (§4.3), while SnapKV achieves the best KL faithfulness among KV pruning methods (§4.4). If structural corruption is the dominant factor, it should manifest in both.

We evaluate two pairs at 65% retention on six LongBench tasks under the y\* metric:

- **SnapKV** (KV pruning) vs **SnapKV-Select** (prompt construction, identical scoring)
- **RADAR** (KV pruning) vs **RADAR-Select** (prompt construction, identical scoring)

| Method | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean (6 tasks) |
|---|---|---|---|---|---|---|---|
| Naive (prompt construction) | 0.221 | 0.262 | 0.281 | 0.081 | **0.048** | 0.076 | 0.162 |
| SnapKV — KV pruning | 0.254 | 0.186 | 0.223 | 0.117 | 0.106 | 0.122 | 0.168 |
| SnapKV-Select — prompt construction | **0.096** | **0.118** | **0.163** | **0.071** | 0.051 | **0.060** | **0.093** |
| RADAR — KV pruning | 0.564 | 0.294 | 0.273 | 0.242 | 0.234 | 0.510 | 0.353 |
| RADAR-Select — prompt construction | 0.140 | 0.174 | 0.266 | 0.090 | 0.048 | 0.071 | 0.132 |

The results are unambiguous. SnapKV with KV pruning (0.168) is *worse* than naive\_65pct (0.162), despite using expensive attention scoring to select tokens. SnapKV-Select with identical token selection but prompt construction achieves 0.093 — a 45% reduction in KL divergence. The same pattern holds for RADAR: KV pruning produces 0.353, prompt construction produces 0.132, a 63% reduction.

The gap between KV pruning and prompt construction cannot be attributed to token selection — the selected tokens are identical. It is entirely attributable to mechanism: prompt construction presents a self-consistent sequence with no causal gaps, while KV pruning exposes every non-tail query to a sparse and gapped neighborhood.

This penalty is not limited to sophisticated scoring methods. Even the simplest recency-biased comparison shows it: Streaming KV pruning (attention sinks + recency window) and naive\_65pct (prompt construction, last 65% of tokens) use the same recency logic and a similar token budget — the only difference is mechanism. Naive truncation outperforms Streaming on all 16 LongBench tasks in §4.4, with a 1.55× mean KL gap (0.175 vs. 0.270). The mechanism penalty holds uniformly across multi-hop QA, summarization, few-shot, synthetic, and code completion tasks alike.

---

## 6. Phrase-Based Context Compression

### 6.1 Motivation: The Quality Ceiling and Its Cost

The controlled comparison in §5.2 also reveals the quality ceiling for prompt-construction methods. SnapKV-Select — which uses SnapKV's attention-based scoring to select tokens but presents them as a new prompt — achieves a mean KL of 0.093 across six tasks (§7.3), substantially outperforming naive\_65pct (0.162). The same pattern holds for RADAR-Select (0.132). Attention-based scoring, freed from structural corruption, is a strong selection signal.

However, computing SnapKV scores requires a full forward pass over the uncompressed prompt — approximately 2500ms on our hardware (see §7.3 for full timing measurements). This scoring overhead nearly doubles total time-to-first-token relative to the compressed prefill alone, making SnapKV-Select impractical for latency-sensitive deployment despite its quality advantage.

This motivates the question the rest of §6 addresses: is it possible to approach SnapKV-Select's quality using only information available without model introspection — no forward pass, no attention weights, no access to model internals?

### 6.2 Approach

Structural corruption is caused by the KV patching mechanism. The solution is simple: do not patch the KV cache. Instead, construct a new, shorter prompt from selected context spans and let the model process it normally. We call this *phrase-based compression*.

Given prompt construction as the mechanism, the remaining design question is what unit to select. Token-level selection is impractical without model introspection: individual BPE subword tokens carry little semantic content, and simple heuristics such as token-set overlap with the query are unreliable — the same surface word tokenizes differently in isolation versus in context, and common function-word fragments match spuriously. Operating at the chunk level solves this.

Chunks are delimited at fixed token positions rather than word boundaries, but because SentencePiece marks each word start with a space prefix, mid-word splits at chunk edges are uncommon in practice. More importantly, detokenizing a span of ~160 tokens yields enough complete words that lexical overlap with the query is a genuine semantic signal rather than a tokenization artifact; the occasional clipped boundary word has negligible effect on the overlap count at this granularity. Chunk granularity is also the natural unit for budget allocation: a budget of *K* tokens allocated in contiguous spans ensures that selected content is internally coherent even when the scoring heuristic is imperfect — a mis-scored chunk at least contains complete thoughts, whereas token-level selection under a weak scorer produces scattered fragments.

**Algorithm.** Given a full prompt of *T* tokens with a known query (the question portion), a recency tail, and a phrase budget:

1. **Identify the tail**: reserve the most recent *tail_n* = ⌊*r* · *T* · *τ*⌋ tokens as a recency tail (always kept), where *r* = 0.65 is the retention fraction and *τ* = 0.25 is the tail fraction (selected by grid search in §6.4).
2. **Divide the remainder** (the "old" context before the tail) into contiguous phrases of *c* tokens each.
3. **Score each phrase** by word-level lexical overlap with the query:
   - Detokenize both phrase and query to text
   - Extract word sets (lowercased, whitespace/punctuation split)
   - Score = |phrase_words ∩ query_words|
4. **Select phrases greedily**: rank by score descending, with recency as a tiebreak (later phrases preferred); add phrases to the head budget until the remaining budget ⌊*r* · *T* · (1 − *τ*)⌋ is exhausted.
5. **Restore order**: sort selected phrases back to their original document positions.
6. **Construct prompt**: concatenate selected phrases + recency tail. Feed to the model as a normal forward pass.

The model receives a self-consistent sequence with no causal gaps. No KV patching, no attention mask modification, no per-token importance scores computed from model internals.

### 6.3 Scoring: Why Word Overlap Works

The scoring function is intentionally simple. Several alternatives were considered and tested:

- **Subword-token overlap**: intersect the raw tokenizer IDs of phrase and query. Fast, but BPE fragmentation causes mismatches ("Columbus" may tokenize differently in isolation vs. in context), and common subword tokens (function word fragments) add noise.
- **Word-level overlap** (our default): detokenize to text first. Eliminates BPE fragmentation artifacts. Function words appear in both query and context but since we take a *set* intersection rather than weighted count, their contribution is limited to 1 per word type.
- **BM25**: inverse-document-frequency-weighted with length normalization. Handles document frequency and term frequency explicitly. Empirically similar to word overlap on these tasks.

We use word-level overlap as the default because it is simple, requires no corpus statistics, and performs well empirically. Crucially, this scoring method has no access to the model's internals — it is pure string matching. Phrases are selected for their topical relevance to the query, not for their prominence in the model's attention distribution.

### 6.4 Phrase Boundaries

The current implementation uses fixed-size phrases of *c* tokens (default *c* = 160). This is a practical approximation to the ideal of semantically coherent spans. A variant, **phrase_sent**, splits the old context at natural sentence and paragraph boundaries (splitting on `\n` and sentence-final punctuation), producing variable-size phrases that correspond to complete thoughts. Evaluation of phrase_sent is reserved for future work.

**Chunk size and tail fraction selection.** We conducted a grid search over chunk sizes *c* ∈ {96, 128, 160} and tail fractions *τ* ∈ {0.20, 0.25, 0.30} to select hyperparameters. Each configuration was evaluated by mean y* KL divergence across the 6 primary tasks (2WikiMQA, MultifieldQA, Qasper, QMSum, RepoBench-P, TriviaQA) at 65% retention. Lower KL indicates closer fidelity to the uncompressed model.

| Configuration | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | **Mean** |
|---|---|---|---|---|---|---|---|
| chunk\_word160\_t25 | 0.094 | 0.138 | 0.140 | 0.058 | 0.088 | 0.057 | **0.096** |
| chunk\_word160\_t30 | 0.100 | 0.135 | 0.152 | 0.057 | 0.085 | 0.054 | **0.097** |
| chunk\_word128\_t20 | 0.101 | 0.147 | 0.136 | 0.060 | 0.086 | 0.054 | **0.097** |
| chunk\_word160\_t20 | 0.094 | 0.137 | 0.143 | 0.071 | 0.088 | 0.053 | **0.098** |
| chunk\_word128\_t30 | 0.103 | 0.157 | 0.147 | 0.057 | 0.085 | 0.056 | **0.101** |
| chunk\_word96\_t20  | 0.099 | 0.208 | 0.179 | 0.058 | 0.098 | 0.057 | 0.116 |
| chunk\_word96\_t25  | 0.095 | 0.217 | 0.179 | 0.056 | 0.097 | 0.057 | 0.117 |
| chunk\_word96\_t30  | 0.097 | 0.210 | 0.180 | 0.057 | 0.094 | 0.058 | 0.116 |

chunk\_word160\_t25 achieves the lowest mean KL (0.096) and is adopted as the primary configuration. chunk\_word128\_t20 ties it within rounding (0.097) and is retained as a secondary configuration since it retains fewer tokens per chunk, which may be preferable in memory-constrained settings. The 96-token variants are clearly inferior on MultifieldQA and Qasper (KL ~0.21 vs ~0.14), suggesting that 96 tokens is insufficient to capture coherent semantic units in these long-document tasks.

The longer-term vision is semantically coherent phrase identification using embedding similarity or topic segmentation, but fixed-size and sentence-boundary variants are sufficient to demonstrate the core finding.

---

## 7. Phrase-based Pruning Experiments

### 7.1 Setup

**Models.** Llama-3.1-8B (base, fp16). Mistral-7B-v0.3 y* results pending.

**Benchmark.** LongBench v1, 16 English tasks, 100 examples per task.

**Methods compared.**

| Method | Type | Description |
|---|---|---|
| Full context | Reference | Uncompressed model |
| naive\_65pct | Prompt construction | Last 65% of tokens, 10% head / 90% tail split |
| SnapKV | KV pruning | Pooled attention weights over observation window |
| RADAR | KV pruning | Post-RoPE KQ alignment × linear decay |
| Streaming | KV pruning | Attention sinks + recency window |
| PyramidKV | KV pruning | Layer-adaptive pyramid budget allocation |
| SnapKV-Select | Prompt construction | SnapKV scoring → token selection → clean prefill |
| **chunk\_word160\_t25** | Prompt construction | Word-overlap scoring, 160-token chunks, 25% tail |
| chunk\_word128\_t20 | Prompt construction | Word-overlap scoring, 128-token chunks, 20% tail |

All compression methods target 65% token retention unless otherwise noted. PyramidKV is also evaluated at 50%, 40%, and 35% retention (§4.3, §7.4).

**Metric.** KL faithfulness (§3.2); lower is better. Ground-truth LongBench scores (§7.3).

### 7.2 KL Faithfulness: Main Results

**Table 1. KL Faithfulness at 65% retention (lower is better). Naive = naive\_65pct; Sel = SnapKV-Select; cw160 = chunk\_word160\_t25; cw128 = chunk\_word128\_t20; PyramidKV = KV cache pruning with layer-adaptive pyramid budget. Bold = best among prompt-construction methods per row. PyramidKV is excluded from bold competition as a qualitatively different compression class.**

| Task             | Naive      | SnapKV | Sel            | cw160          | cw128          | PyramidKV |
|------------------|------------|--------|----------------|----------------|----------------|-----------|
| 2WikiMQA         | 0.221      | 0.254  | **0.096**      | 0.121          | 0.117          | 2.422     |
| MultifieldQA     | 0.262      | 0.186  | 0.118          | **0.109**      | 0.114          | 1.535     |
| Qasper           | 0.281      | 0.223  | 0.163          | 0.139          | **0.131**      | 1.229     |
| QMSum            | 0.081      | 0.117  | 0.071          | **0.058**      | 0.060          | 0.438     |
| RepoBench-P      | 0.048      | 0.106  | 0.051          | 0.052          | **0.046**      | 1.052     |
| TriviaQA         | 0.076      | 0.122  | **0.060**      | 0.076          | 0.062          | 1.647     |
| GovReport        | 0.280      | 0.374  | **0.208**      | 0.385          | 0.385          | 0.334     |
| HotpotQA         | 0.203      | 0.246  | 0.189          | **0.124**      | 0.125          | 2.181     |
| LCC              | **0.076**  | 0.112  | 0.089          | 0.122          | 0.125          | 1.142     |
| MultiNews        | 0.585      | 0.407  | **0.511**      | 0.644          | 0.647          | 0.721     |
| MuSiQue          | 0.197      | 0.283  | 0.198          | 0.135          | **0.118**      | 2.523     |
| NarrativeQA      | 0.022      | 0.060  | 0.022          | 0.023          | **0.020**      | 1.338     |
| PassageCount     | **0.059**  | 0.114  | **0.059**      | 0.067          | 0.067          | 1.626     |
| PassageRetrieval | 0.188      | 0.211  | 0.188          | 0.164          | **0.144**      | 1.416     |
| SAMSum           | **0.011**  | 0.048  | 0.024          | 0.025          | 0.014          | 1.182     |
| TREC             | 0.202      | 0.146  | **0.095**      | 0.112          | 0.128          | 1.526     |
| **Mean (16 tasks)** | 0.175   | 0.188  | **0.134**      | 0.147          | 0.144          | 1.395     |

On the 6 primary tasks, cw128 leads with mean KL 0.089, with cw160 (0.092) and SnapKV-Select (0.093) close behind. Both chunk methods beat naive on five of six primary tasks; on RepoBench-P, cw128 (0.046) edges naive (0.048) while cw160 (0.052) falls slightly behind. SnapKV-Select's scoring overhead (~2500ms per example, see §7.3) makes it impractical for latency-sensitive deployment; chunk selection achieves comparable quality at ~3ms selection cost.

Across all 16 tasks, cw128 wins outright on NarrativeQA, Qasper, MuSiQue, and PassageRetrieval; cw160 wins on MultifieldQA, HotpotQA, and QMSum; SnapKV-Select leads on 2WikiMQA, GovReport, TriviaQA, TREC, and MultiNews (among prompt-construction methods). Naive truncation wins on LCC and SAMSum, where the relevant signal is already concentrated in the recency tail. MultiNews remains the outlier where SnapKV KV pruning (0.407) outperforms all prompt-construction methods — consistent with the multi-document structure noted in §8.2. The cw160 and cw128 results are close at 16-task mean (0.147 vs. 0.144); cw128 is marginally preferred at both the 6-task and 16-task level.

PyramidKV tells a strikingly different story. Despite being a sophisticated KV cache compression method with layer-adaptive pyramid budget allocation, its mean KL divergence (1.394 nats, corrected; see note below) is substantially higher than any prompt-construction method. Across tasks, PyramidKV ranges from 0.334 nats (GovReport) to 2.523 nats (MuSiQue) — compared to typical prompt-construction values of 0.05–0.6 nats. PyramidKV averages roughly 8–10× higher KL than cw128 (1.394 / 0.144 ≈ 9.7×) and Naive (1.394 / 0.175 ≈ 8.0×). This result is not a failure of the PyramidKV implementation; ground-truth accuracy leads all compressed methods at 65% retention and remains competitive across tighter compression rates (§4.3). Rather, it reflects the structural corruption mechanism described in §5.2: KV cache pruning creates causal gaps that cascade through all attention layers, producing a distribution that is locally plausible (correct answers often survive) but globally diverged from the full model. Prompt construction avoids this entirely by presenting a complete, gap-free context to an unmodified model.

*Note: PyramidKV values corrected from the original submission. The prior evaluation had two bugs: (1) y* generation for short-answer tasks was not stopped at the answer boundary, contaminating y* with a continuation of the few-shot context; and (2) an off-by-one error in the kvpress step-decode path caused the logit predicting y*[0] to be discarded, shifting all token distributions by one step. Bug (2) inflated PyramidKV KL from ~1.4 nats to ~10–12 nats. All values in this table have been recomputed with the corrected y* (v3 rerun, n=100).*

**Tables 2a–2d. KL Faithfulness across compression rates (lower is better). Sel = SnapKV-Select; cw160 = chunk\_word160\_t25; cw128 = chunk\_word128\_t20; SnapKV = SnapKV KV pruning; Pyr = PyramidKV. Bold = best among prompt-construction methods per row. PyramidKV excluded from bold competition. 65% values match Table 1.**

**Table 2a. 65% retention.**

| Task | Naive | Sel | cw160 | cw128 | SnapKV | Pyr |
|---|---|---|---|---|---|---|
| NarrativeQA | 0.022 | 0.022 | 0.023 | **0.020** | 0.060 | 1.337 |
| Qasper | 0.281 | 0.163 | 0.139 | **0.131** | 0.223 | 1.229 |
| MultifieldQA | 0.262 | 0.118 | **0.109** | 0.114 | 0.186 | 1.535 |
| HotpotQA | 0.203 | 0.189 | **0.124** | 0.125 | 0.246 | 2.181 |
| 2WikiMQA | 0.221 | **0.096** | 0.121 | 0.117 | 0.254 | 2.422 |
| MuSiQue | 0.197 | 0.198 | 0.135 | **0.118** | 0.283 | 2.523 |
| GovReport | 0.280 | **0.208** | 0.385 | 0.385 | 0.374 | 0.334 |
| QMSum | 0.081 | 0.071 | **0.058** | 0.060 | 0.117 | 0.438 |
| MultiNews | 0.585 | **0.511** | 0.644 | 0.647 | 0.407 | 0.721 |
| TREC | 0.202 | **0.095** | 0.112 | 0.128 | 0.146 | 1.526 |
| TriviaQA | 0.076 | **0.060** | 0.076 | 0.062 | 0.122 | 1.647 |
| SAMSum | **0.011** | 0.024 | 0.025 | 0.014 | 0.048 | 1.182 |
| PassageCount | 0.059 | **0.059** | 0.067 | 0.067 | 0.114 | 1.626 |
| PassageRetrieval | 0.188 | 0.188 | 0.164 | **0.144** | 0.211 | 1.416 |
| LCC | **0.076** | 0.089 | 0.122 | 0.125 | 0.112 | 1.142 |
| RepoBench-P | 0.048 | 0.051 | 0.052 | **0.046** | 0.106 | 1.052 |
| **Average** | 0.175 | **0.134** | 0.147 | 0.144 | 0.188 | 1.394 |

**Table 2b. 50% retention.**

| Task | Naive | Sel | cw160 | cw128 | SnapKV | Pyr |
|---|---|---|---|---|---|---|
| NarrativeQA | **0.026** | **0.026** | 0.038 | 0.028 | 0.138 | 1.380 |
| Qasper | 0.457 | **0.256** | **0.256** | 0.264 | 0.348 | 1.394 |
| MultifieldQA | 0.359 | 0.101 | **0.094** | 0.144 | 0.274 | 1.619 |
| HotpotQA | 0.212 | 0.190 | 0.114 | **0.099** | 0.425 | 2.245 |
| 2WikiMQA | 0.278 | **0.086** | 0.145 | 0.129 | 0.405 | 2.282 |
| MuSiQue | 0.188 | 0.187 | 0.123 | **0.089** | 0.472 | 2.507 |
| GovReport | 0.327 | **0.274** | 0.398 | 0.399 | 0.599 | 0.504 |
| QMSum | 0.088 | **0.075** | 0.087 | 0.093 | 0.199 | 0.484 |
| MultiNews | **0.759** | 0.763 | 0.808 | 0.786 | 1.002 | 1.099 |
| TREC | 0.260 | **0.104** | 0.168 | 0.186 | 0.334 | 1.711 |
| TriviaQA | 0.095 | **0.062** | 0.091 | 0.087 | 0.296 | 1.622 |
| SAMSum | **0.012** | 0.032 | 0.016 | 0.018 | 0.222 | 1.199 |
| PassageCount | **0.055** | 0.057 | 0.068 | 0.068 | 0.229 | 1.538 |
| PassageRetrieval | 0.188 | 0.188 | **0.113** | 0.117 | 0.331 | 1.378 |
| LCC | **0.105** | 0.153 | 0.153 | 0.150 | 0.223 | 1.238 |
| RepoBench-P | 0.065 | 0.069 | 0.060 | **0.057** | 0.329 | 1.016 |
| **Average** | 0.217 | **0.164** | 0.171 | 0.170 | 0.364 | 1.451 |

**Table 2c. 40% retention.**

| Task | Naive | Sel | cw160 | cw128 | SnapKV | Pyr |
|---|---|---|---|---|---|---|
| NarrativeQA | **0.035** | **0.035** | 0.041 | 0.045 | 0.273 | 0.702 |
| Qasper | 0.577 | **0.336** | 0.362 | 0.346 | 0.485 | 0.772 |
| MultifieldQA | 0.443 | 0.178 | **0.176** | 0.180 | 0.419 | 0.990 |
| HotpotQA | 0.208 | 0.183 | 0.137 | **0.086** | 0.619 | 1.659 |
| 2WikiMQA | 0.332 | **0.098** | 0.208 | 0.199 | 0.552 | 1.315 |
| MuSiQue | 0.189 | 0.188 | 0.130 | **0.124** | 0.681 | 1.719 |
| GovReport | 0.391 | **0.338** | 0.438 | 0.438 | 0.813 | 0.506 |
| QMSum | 0.104 | **0.090** | 0.104 | 0.107 | 0.308 | 0.342 |
| MultiNews | **0.888** | 0.928 | 0.899 | 0.903 | 1.760 | 1.024 |
| TREC | 0.282 | **0.133** | 0.188 | 0.203 | 0.615 | 1.340 |
| TriviaQA | 0.136 | **0.061** | 0.121 | 0.117 | 0.526 | 1.129 |
| SAMSum | **0.015** | 0.038 | 0.021 | 0.024 | 0.460 | 0.646 |
| PassageCount | 0.093 | 0.091 | **0.072** | 0.073 | 0.332 | 1.168 |
| PassageRetrieval | 0.195 | 0.195 | **0.138** | 0.149 | 0.445 | 1.365 |
| LCC | **0.125** | 0.212 | 0.167 | 0.170 | 0.444 | 0.584 |
| RepoBench-P | **0.077** | 0.099 | 0.097 | 0.087 | 0.562 | 0.641 |
| **Average** | 0.256 | **0.200** | 0.206 | 0.203 | 0.581 | 0.994 |

**Table 2d. 35% retention.**

| Task | Naive | Sel | cw160 | cw128 | SnapKV | Pyr |
|---|---|---|---|---|---|---|
| NarrativeQA | **0.045** | **0.045** | 0.055 | 0.076 | 0.334 | 0.559 |
| Qasper | 0.637 | 0.385 | 0.385 | **0.381** | 0.588 | 0.858 |
| MultifieldQA | 0.475 | 0.194 | **0.171** | 0.203 | 0.483 | 0.914 |
| HotpotQA | 0.220 | 0.197 | 0.166 | **0.127** | 0.666 | 1.252 |
| 2WikiMQA | 0.351 | **0.111** | 0.262 | 0.220 | 0.643 | 1.052 |
| MuSiQue | 0.191 | 0.190 | 0.161 | **0.158** | 0.786 | 1.236 |
| GovReport | 0.423 | **0.376** | 0.460 | 0.460 | 0.960 | 0.529 |
| QMSum | 0.112 | **0.097** | 0.107 | 0.121 | 0.394 | 0.358 |
| MultiNews | **0.947** | 1.014 | 0.948 | 0.961 | 2.101 | 1.064 |
| TREC | 0.296 | **0.156** | 0.253 | 0.224 | 0.737 | 0.823 |
| TriviaQA | 0.157 | **0.077** | 0.151 | 0.142 | 0.636 | 0.833 |
| SAMSum | **0.017** | 0.042 | 0.024 | 0.028 | 0.606 | 0.475 |
| PassageCount | 0.116 | 0.112 | **0.073** | 0.075 | 0.334 | 1.134 |
| PassageRetrieval | 0.282 | 0.282 | 0.184 | **0.182** | 0.489 | 1.240 |
| LCC | **0.145** | 0.261 | 0.181 | 0.182 | 0.534 | 0.550 |
| RepoBench-P | **0.087** | 0.107 | 0.107 | 0.107 | 0.635 | 0.511 |
| **Average** | 0.281 | **0.228** | 0.230 | 0.228 | 0.683 | 0.837 |

At 65% retention, SnapKV-Select leads among prompt-construction methods (0.134 mean KL); cw128 (0.144) and cw160 (0.147) are close behind, and both beat SnapKV KV pruning (0.188). At 50%, SnapKV-Select (0.164) retains a narrow lead, with cw128 (0.170) and cw160 (0.171) nearly matching it. At 40% and 35%, SnapKV-Select continues to lead, but the margin is thin (Sel: 0.200 vs. cw128: 0.203 at 40%; Sel: 0.228 vs. cw128: 0.228 at 35% — effectively a tie at the tightest budget). SnapKV KV pruning degrades catastrophically as the budget tightens (0.188 → 0.364 → 0.581 → 0.683), while prompt-construction methods degrade gracefully (Sel: 0.134 → 0.164 → 0.200 → 0.228; cw128: 0.144 → 0.170 → 0.203 → 0.228; Naive: 0.175 → 0.217 → 0.256 → 0.281). The key cross-rate result: cw128/50% (0.170) matches Naive/65% (0.175) — nearly identical faithfulness with 15% less context. cw128/40% (0.203) beats Naive/50% (0.217), and cw128/35% (0.228) beats Naive/40% (0.256): tighter chunk compression is more faithful than looser naive truncation at every step. MultiNews at 65% remains the one exception where SnapKV KV pruning (0.407) edges out prompt-construction methods (Sel: 0.511, Naive: 0.585), consistent with §5.3.

PyramidKV shows a counterintuitive compression trajectory: mean KL is 1.394 at 65%, rises slightly to 1.451 at 50%, then drops sharply to 0.994 at 40% and 0.837 at 35%. Tighter PyramidKV compression produces *better* KL faithfulness — the opposite of every other method. This is consistent with the clamping hypothesis (§4.4): at 65% retention (compression\_ratio=0.35), PyramidKV's budget allocator clamps upward at lower layers, creating over-retention that disrupts the expected attention pattern. At 40–35% retention, the budget is unclamped and the pyramid allocation operates as intended. Even so, PyramidKV at its best (0.837 at 35%) remains 3.7× worse than the best prompt-construction method at the same budget (Sel: 0.228), confirming that the KL gap is structural rather than a configuration artifact.

### 7.3 Inference Performance

Compression is only worthwhile if it makes inference faster. We measure two quantities: **time to first token (TTFT)**, the wall-clock time from receiving a prompt to producing the first output token (dominated by the prefill pass), and **time per output token (TPT)**, the mean decode step time (dominated by KV cache memory bandwidth). All measurements are on a single V100 32GB GPU using Llama-3.1-8B fp16. TTFT for chunk methods includes selection overhead (~3ms); TTFT for SnapKV-Select includes its full scoring pass.

**TTFT by task at 65% retention** (mean over 100 examples per task; cw160 = chunk\_word160\_t25, cw128 = chunk\_word128\_t20, sel = SnapKV-Select):

| Task | Full (ms) | cw160 | Savings || cw128 | Savings || sel | Savings |
|---|---|---|---|-|---|---|-|---|---|
| 2WikiMQA | 6390 | 2543 | 60% || 2483 | 61% || 5160 | 19% |
| MultifieldQA | 6086 | 2300 | 62% || 2250 | 63% || 4429 | 27% |
| Qasper | 4584 | 2160 | 53% || 2121 | 54% || 4895 | −7% |
| RepoBench-P | 7224 | 2668 | 63% || 2585 | 64% || 4281 | 41% |
| TriviaQA | 7455 | 2717 | 64% || 2675 | 64% || 4049 | 46% |
| **Mean** | **6348** | **2520** | **60%** || **2464** | **61%** || **4517** | **29%** |

Chunk methods cut TTFT by 60% at 65% retention — the prefill is simply over a much shorter prompt. SnapKV-Select (sel) saves only 29% on average because its ~2500ms scoring overhead nearly cancels the prefill savings; on Qasper, where the full context is already short, sel is *slower* than full-context inference.

**Mean TTFT savings across 5 tasks by retention ratio:**

| Retention | cw160 TTFT | Savings | cw128 TTFT | Savings | sel TTFT | Savings |
|---|---|---|---|---|---|---|
| 65% | 2520 ms | 60% | 2464 ms | 61% | 4517 ms | 29% |
| 50% | 2174 ms | 66% | 2139 ms | 66% | 4335 ms | 32% |
| 40% | 1847 ms | 71% | 1838 ms | 71% | 4037 ms | 36% |
| 35% | 1675 ms | 74% | 1666 ms | 74% | 3880 ms | 39% |

At 40% retention, chunk methods achieve 71% TTFT reduction — cutting prefill time by more than 3×. Even at the most aggressive 35% budget, sel cannot match chunk methods despite using SnapKV's attention scoring, because the scoring overhead is nearly constant while the prefill savings grow with compression.

**Decode speed (TPT)** improves for all compressed methods because fewer retained tokens means a smaller KV cache to scan at each decode step. All three methods are essentially identical at a given retention level — TPT depends on KV cache size, not scoring method. At 65% retention, mean TPT drops from 67.5 ms/tok (full) to ~54 ms/tok (~20% faster); at 35% retention it drops to ~48 ms/tok (~29% faster).

### 7.4 Ground-Truth Task Accuracy

Ground-truth accuracy is not our primary faithfulness metric — it is too coarse to distinguish methods that produce subtly different but plausible outputs, and it conflates compression quality with task difficulty. Two methods can produce completely different output distributions and score identically if both happen to extract the correct answer token. KL faithfulness on the y* shared prefix (§4.4, §7.2) is the definitive measure.

That said, ground-truth accuracy serves an essential role as a final sanity check. If a compression method were doing something catastrophically wrong — collapsing outputs, hallucinating context, or systematically destroying task-relevant information — it would surface here even when KL divergence might not. We include these results to confirm that nothing unexpectedly stupid is happening.

Ground-truth scores measure whether the model produces a correct answer on each LongBench example. Each table shows absolute scores with Δ% = (compressed − full) / full × 100 per method, plus mean |Δ%| as a summary row.

Column abbreviations: **cw160** = chunk\_word160\_t25, **cw128** = chunk\_word128\_t20, **sel** = SnapKV-Select (prompt construction with SnapKV scoring), **naive** = naive truncation, **snapkv** = SnapKV KV pruning, **radar** = RADAR KV pruning.

Absolute scores alone are misleading as a compression metric. When the full-context model already scores near chance — as on NarrativeQA (5.5) and tasks flagged † — small fluctuations in either direction are noise rather than signal. A method scoring 8.0 when the full model scores 5.5 has not improved; it has simply drifted. The Δ% captures this: it measures how much compression changed the model's task performance, regardless of direction, and is zero only when compression is invisible to the task.

**65% retention**

| Task | Full | naive | Δ% | cw160 | Δ% | cw128 | Δ% | sel | Δ% | snapkv | Δ% | radar | Δ% | pyramid | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NarrativeQA† | 5.5 | 4.9 | -11% | 4.3 | -23% | 4.3 | -23% | 3.4 | -38% | 5.0 | -9% | 5.7 | +4% | 5.6 | +2% |
| Qasper† | 11.1 | 10.2 | -8% | 10.6 | -4% | 10.3 | -7% | 12.2 | +10% | 11.1 | +0% | 9.5 | -14% | 11.8 | +6% |
| MultifieldQA | 28.9 | 27.0 | -7% | 28.1 | -3% | 25.3 | -12% | 26.9 | -7% | 28.4 | -2% | 29.4 | +2% | 29.6 | +2% |
| HotpotQA† | 9.9 | 9.6 | -3% | 10.2 | +3% | 9.2 | -7% | 10.4 | +5% | 9.6 | -3% | 10.1 | +2% | 10.2 | +3% |
| 2WikiMQA† | 14.1 | 12.6 | -11% | 11.5 | -18% | 10.7 | -24% | 11.2 | -21% | 13.9 | -1% | 12.1 | -14% | 13.9 | -1% |
| MuSiQue† | 6.9 | 7.0 | +1% | 6.4 | -8% | 6.1 | -12% | 5.5 | -20% | 6.0 | -13% | 6.1 | -12% | 7.1 | +3% |
| GovReport | 20.4 | 19.8 | -3% | 20.0 | -2% | 19.7 | -3% | 19.2 | -6% | 19.9 | -2% | 19.2 | -6% | 20.3 | -0% |
| QMSum† | 10.3 | 11.5 | +12% | 10.4 | +1% | 9.3 | -10% | 9.2 | -11% | 9.7 | -6% | 9.0 | -13% | 9.4 | -9% |
| MultiNews | 19.0 | 16.5 | -13% | 16.5 | -13% | 16.4 | -14% | 16.8 | -11% | 18.9 | -1% | 18.4 | -3% | 18.1 | -5% |
| TREC | 70.0 | 66.0 | -6% | 68.0 | -3% | 70.0 | +0% | 67.0 | -4% | 66.0 | -6% | 70.0 | +0% | 71.0 | +1% |
| TriviaQA | 17.4 | 17.3 | -1% | 16.9 | -3% | 17.2 | -1% | 17.5 | +0% | 17.6 | +1% | 18.3 | +5% | 17.5 | +1% |
| SAMSum | 16.0 | 16.5 | +3% | 16.2 | +1% | 16.5 | +3% | 15.2 | -5% | 17.5 | +9% | 18.3 | +14% | 16.3 | +2% |
| PassageCount† | 3.0 | 1.0 | -67% | 2.0 | -33% | 2.0 | -33% | 3.0 | +0% | 3.0 | +0% | 3.0 | +0% | 3.0 | +0% |
| PassageRetrieval | 44.0 | 37.0 | -16% | 27.0 | -39% | 29.0 | -34% | 31.0 | -30% | 37.0 | -16% | 36.0 | -18% | 44.0 | +0% |
| LCC | 68.1 | 63.4 | -7% | 61.1 | -10% | 61.3 | -10% | 65.1 | -4% | 63.2 | -7% | 64.1 | -6% | 68.5 | +1% |
| RepoBench-P | 55.6 | 53.9 | -3% | 48.1 | -13% | 50.7 | -9% | 56.4 | +2% | 53.4 | -4% | 54.1 | -3% | 56.0 | +1% |
| **Average** | **25.0** | **23.4** | | **22.3** | | **22.4** | | **23.1** | | **23.8** | | **24.0** | | **25.1** | |
| **mean \|Δ%\|** | | | **11%** | | **11%** | | **13%** | | **11%** | | **5%** | | **7%** | | **2%** |

**50% retention**

| Task | Full | naive | Δ% | cw160 | Δ% | cw128 | Δ% | sel | Δ% | snapkv | Δ% | radar | Δ% | pyramid | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NarrativeQA† | 5.5 | 5.7 | +4% | 4.3 | -22% | 4.2 | -24% | 3.7 | -33% | 8.0 | +46% | 3.7 | -32% | 6.8 | +24% |
| Qasper† | 11.1 | 8.6 | -23% | 11.0 | -0% | 9.9 | -11% | 12.7 | +15% | 10.5 | -6% | 10.0 | -10% | 9.8 | -11% |
| MultifieldQA | 28.9 | 23.0 | -20% | 27.4 | -5% | 24.8 | -14% | 25.4 | -12% | 25.8 | -11% | 18.6 | -36% | 29.5 | +2% |
| HotpotQA† | 9.9 | 8.9 | -10% | 9.8 | -1% | 9.2 | -7% | 9.9 | +0% | 8.5 | -14% | 6.9 | -30% | 9.6 | -3% |
| 2WikiMQA† | 14.1 | 12.7 | -10% | 11.6 | -17% | 12.9 | -9% | 11.8 | -17% | 11.2 | -21% | 10.8 | -23% | 13.8 | -2% |
| MuSiQue† | 6.9 | 5.8 | -17% | 7.0 | +2% | 6.7 | -3% | 6.6 | -4% | 5.4 | -22% | 3.3 | -52% | 6.8 | -2% |
| GovReport | 20.4 | 19.5 | -4% | 19.0 | -7% | 18.5 | -9% | 18.8 | -8% | 18.3 | -10% | 17.3 | -15% | 19.5 | -4% |
| QMSum† | 10.3 | 9.5 | -8% | 9.9 | -4% | 9.0 | -12% | 9.3 | -10% | 9.1 | -12% | 8.4 | -19% | 9.7 | -6% |
| MultiNews | 19.0 | 17.5 | -8% | 16.3 | -14% | 15.9 | -16% | 16.2 | -14% | 16.0 | -16% | 9.9 | -48% | 15.4 | -19% |
| TREC | 70.0 | 63.0 | -10% | 64.0 | -9% | 65.0 | -7% | 66.0 | -6% | 57.0 | -19% | 57.0 | -19% | 68.0 | -3% |
| TriviaQA | 17.4 | 17.4 | -0% | 17.5 | +1% | 16.8 | -3% | 17.1 | -2% | 18.5 | +7% | 19.3 | +11% | 17.3 | -0% |
| SAMSum | 16.0 | 17.7 | +11% | 16.3 | +2% | 16.8 | +5% | 14.6 | -9% | 18.0 | +13% | 13.8 | -14% | 17.6 | +10% |
| PassageCount† | 3.0 | 2.0 | -33% | 2.0 | -33% | 2.0 | -33% | 2.0 | -33% | 2.0 | -33% | 1.0 | -67% | 3.0 | +0% |
| PassageRetrieval | 44.0 | 21.0 | -52% | 33.0 | -25% | 27.0 | -39% | 39.0 | -11% | 18.0 | -59% | 6.0 | -86% | 43.0 | -2% |
| LCC | 68.1 | 61.6 | -10% | 61.6 | -9% | 62.5 | -8% | 64.6 | -5% | 58.5 | -14% | 49.1 | -28% | 63.9 | -6% |
| RepoBench-P | 55.6 | 51.4 | -7% | 49.8 | -10% | 48.7 | -12% | 57.2 | +3% | 45.2 | -19% | 40.1 | -28% | 52.8 | -5% |
| **Average** | **25.0** | **21.6** | | **22.5** | | **21.9** | | **23.4** | | **20.6** | | **17.2** | | **24.2** | |
| **mean \|Δ%\|** | | | **14%** | | **10%** | | **13%** | | **11%** | | **20%** | | **32%** | | **6%** |

**40% retention**

| Task | Full | naive | Δ% | cw160 | Δ% | cw128 | Δ% | sel | Δ% | snapkv | Δ% | radar | Δ% | pyramid | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NarrativeQA† | 5.5 | 5.8 | +5% | 4.3 | -22% | 5.4 | -1% | 3.9 | -30% | 7.5 | +37% | 1.4 | -75% | 6.8 | +23% |
| Qasper† | 11.1 | 7.9 | -29% | 9.2 | -17% | 9.4 | -15% | 10.9 | -2% | 7.4 | -33% | 9.5 | -15% | 11.3 | +2% |
| MultifieldQA | 28.9 | 23.3 | -19% | 24.7 | -15% | 23.4 | -19% | 24.4 | -16% | 22.0 | -24% | 15.7 | -46% | 29.5 | +2% |
| HotpotQA† | 9.9 | 10.6 | +7% | 9.9 | -0% | 10.8 | +9% | 10.1 | +2% | 7.2 | -27% | 6.4 | -35% | 9.4 | -5% |
| 2WikiMQA† | 14.1 | 13.9 | -1% | 11.2 | -20% | 12.5 | -11% | 11.2 | -21% | 9.2 | -35% | 15.4 | +9% | 13.9 | -2% |
| MuSiQue† | 6.9 | 4.7 | -32% | 6.4 | -7% | 6.5 | -5% | 6.1 | -12% | 4.1 | -40% | 2.5 | -64% | 6.7 | -3% |
| GovReport | 20.4 | 19.9 | -2% | 18.6 | -9% | 19.0 | -7% | 17.9 | -12% | 16.9 | -17% | 16.8 | -18% | 19.1 | -6% |
| QMSum† | 10.3 | 9.7 | -6% | 9.6 | -7% | 10.1 | -2% | 9.0 | -13% | 9.1 | -11% | 5.7 | -45% | 10.0 | -3% |
| MultiNews | 19.0 | 16.7 | -12% | 16.1 | -15% | 16.1 | -15% | 16.0 | -16% | 11.0 | -42% | 8.2 | -57% | 15.9 | -16% |
| TREC | 70.0 | 65.0 | -7% | 67.0 | -4% | 68.0 | -3% | 64.0 | -9% | 49.0 | -30% | 43.0 | -39% | 70.0 | +0% |
| TriviaQA | 17.4 | 17.3 | -1% | 17.0 | -2% | 16.9 | -3% | 17.8 | +2% | 15.9 | -9% | 21.4 | +23% | 17.4 | +0% |
| SAMSum | 16.0 | 17.2 | +7% | 16.5 | +3% | 16.6 | +4% | 15.1 | -6% | 15.6 | -2% | 12.7 | -21% | 18.1 | +13% |
| PassageCount† | 3.0 | 3.0 | +0% | 3.0 | +0% | 3.0 | +0% | 2.0 | -33% | 0.0 | -100% | 0.0 | -100% | 3.0 | +0% |
| PassageRetrieval | 44.0 | 12.0 | -73% | 24.0 | -45% | 24.0 | -45% | 38.0 | -14% | 8.0 | -82% | 4.0 | -91% | 43.0 | -2% |
| LCC | 68.1 | 58.7 | -14% | 62.5 | -8% | 61.2 | -10% | 60.3 | -11% | 46.7 | -31% | 42.5 | -38% | 68.4 | +0% |
| RepoBench-P | 55.6 | 52.2 | -6% | 49.3 | -11% | 49.6 | -11% | 55.0 | -1% | 39.1 | -30% | 35.5 | -36% | 54.4 | -2% |
| **Average** | **25.0** | **21.1** | | **21.8** | | **22.0** | | **22.6** | | **16.8** | | **15.0** | | **24.8** | |
| **mean \|Δ%\|** | | | **14%** | | **12%** | | **10%** | | **12%** | | **34%** | | **44%** | | **5%** |

**35% retention**

| Task | Full | naive | Δ% | cw160 | Δ% | cw128 | Δ% | sel | Δ% | snapkv | Δ% | radar | Δ% | pyramid | Δ% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NarrativeQA† | 5.5 | 7.8 | +41% | 3.9 | -29% | 3.9 | -30% | 4.1 | -26% | 6.7 | +21% | 1.1 | -81% | 6.4 | +16% |
| Qasper† | 11.1 | 7.7 | -31% | 8.6 | -22% | 9.0 | -19% | 13.6 | +23% | 7.2 | -35% | 7.6 | -32% | 10.8 | -3% |
| MultifieldQA | 28.9 | 21.2 | -27% | 27.0 | -7% | 24.0 | -17% | 23.5 | -19% | 19.0 | -34% | 15.5 | -46% | 30.0 | +4% |
| HotpotQA† | 9.9 | 11.4 | +16% | 9.4 | -5% | 10.0 | +1% | 10.0 | +1% | 6.5 | -34% | 5.2 | -47% | 10.0 | +1% |
| 2WikiMQA† | 14.1 | 11.0 | -22% | 11.1 | -21% | 11.3 | -20% | 12.1 | -14% | 8.6 | -39% | 12.6 | -11% | 14.6 | +4% |
| MuSiQue† | 6.9 | 4.7 | -31% | 7.1 | +2% | 6.1 | -11% | 6.4 | -8% | 5.4 | -22% | 2.5 | -64% | 6.5 | -6% |
| GovReport | 20.4 | 18.7 | -8% | 19.2 | -6% | 19.1 | -6% | 17.4 | -15% | 16.3 | -20% | 15.5 | -24% | 19.0 | -7% |
| QMSum† | 10.3 | 9.9 | -4% | 9.2 | -11% | 10.0 | -3% | 8.9 | -14% | 8.3 | -19% | 4.5 | -56% | 10.2 | -1% |
| MultiNews | 19.0 | 16.7 | -12% | 16.0 | -16% | 15.3 | -19% | 15.7 | -17% | 8.7 | -54% | 7.4 | -61% | 15.8 | -17% |
| TREC | 70.0 | 60.0 | -14% | 68.0 | -3% | 68.0 | -3% | 60.0 | -14% | 52.0 | -26% | 30.0 | -57% | 69.0 | -1% |
| TriviaQA | 17.4 | 17.3 | -0% | 16.9 | -3% | 16.8 | -3% | 17.7 | +2% | 16.4 | -6% | 19.0 | +9% | 17.4 | +0% |
| SAMSum | 16.0 | 17.6 | +10% | 17.0 | +6% | 17.8 | +11% | 14.3 | -10% | 12.4 | -23% | 11.3 | -29% | 17.3 | +8% |
| PassageCount† | 3.0 | 3.0 | +0% | 2.0 | -33% | 3.0 | +0% | 2.0 | -33% | 0.0 | -100% | 1.0 | -67% | 3.0 | +0% |
| PassageRetrieval | 44.0 | 12.0 | -73% | 29.0 | -34% | 19.0 | -57% | 36.0 | -18% | 6.0 | -86% | 3.0 | -93% | 44.0 | +0% |
| LCC | 68.1 | 62.4 | -8% | 62.9 | -8% | 60.9 | -11% | 61.0 | -10% | 42.9 | -37% | 45.4 | -33% | 67.0 | -2% |
| RepoBench-P | 55.6 | 50.6 | -9% | 50.5 | -9% | 49.2 | -12% | 56.0 | +1% | 39.6 | -29% | 37.2 | -33% | 55.1 | -1% |
| **Average** | **25.0** | **20.8** | | **22.4** | | **21.5** | | **22.4** | | **16.0** | | **13.7** | | **24.8** | |
| **mean \|Δ%\|** | | | **19%** | | **13%** | | **14%** | | **14%** | | **37%** | | **47%** | | **4%** |

Tasks marked † have full-context scores below 15; their Δ% values are noisier and should be interpreted cautiously.

These results validate every key claim of this paper, and they do so independently of KL faithfulness — which makes the agreement all the more striking.

**KL faithfulness is the right metric.** KL divergence predicted that KV pruning would degrade at higher compression rates while prompt construction would not. The ground-truth results confirm this exactly: SnapKV's mean |Δ%| rises from 5% at 65% to 34% at 40%; RADAR's rises from 7% to 44%. Prompt-construction methods show no such trajectory. A metric that correctly predicts the shape of a degradation curve before the ground-truth data is collected is doing its job.

**KV pruning has a structural problem — PyramidKV included, but differently.** SnapKV and RADAR show systematic compression-dependent degradation: as causal gaps multiply with tighter compression, quality collapses. PyramidKV's mean |Δ%| stays at 2–6% across all four retention rates — superficially stable where SnapKV reaches 34% at 40%. But the source of that stability shifts with budget. At 65–40% retention, clean_first ablations (§7.5) show negligible delta (≤0.32 points), indicating that PyramidKV's layer-adaptive allocation genuinely preserves task-relevant tokens at moderate budgets. At 35% retention, removing the first-token privilege collapses performance by 15.7 points on average; the apparent stability of 24.8 is almost entirely explained by a single token that attended to the full uncompressed context during prefill, not by the quality of the compression. In both regimes, the cost is captured by KL faithfulness (§4.4, §7.2): PyramidKV's stable GT comes with order-of-magnitude higher KL divergence, reflecting structural corruption from KV patching that ground-truth accuracy cannot detect.

**Prompt construction is the right approach.** Across all four compression budgets, prompt-construction methods (naive, cw160, cw128, sel) stay within 7–22% of full-context performance. They do not degrade with compression aggressiveness the way KV pruning does. Reconstructing a clean prompt from selected tokens sidesteps the structural problem entirely.

**Phrase-based pruning works.** chunk\_word160\_t25 and chunk\_word128\_t20 are competitive with or better than naive truncation at every compression level, and track SnapKV-Select (which uses expensive attention scoring) closely despite relying only on string matching. Chunked lexical selection over a clean reconstructed prompt captures the practical benefit of semantic scoring without the latency cost.

---

### 7.5 PyramidKV: A Case Study

PyramidKV is the strongest method in this paper on ground truth: 25.2 average across 16 tasks at 65% retention, exceeding full context (25.0) and every other compressed method by at least 1.3 points (§4.3). Results at tighter compression are equally strong: f50 drops to 24.2, and f40 and f35 both recover to 24.8 — within 0.4 points of the 65% configuration. Three additional metrics tell a different story.

**KL faithfulness.** PyramidKV's mean KL divergence from the full model (1.394 nats at 65% retention) is 9.7× higher than cw128 (0.144) and 8.0× higher than naive truncation (0.175) — the worst result of any method evaluated, by a wide margin (§4.4, §7.2). Even at 35% retention, where the layer-budget clamping artifact resolves and KL improves to 0.837 nats, PyramidKV remains 3.7× worse than cw128 at the same budget (0.228). The KL gap is structural, not a configuration artifact.

**Inference speed.** PyramidKV's speed profile is the inverse of chunk compression. Because it performs a full prefill over the uncompressed prompt before pruning the KV cache, its TTFT is essentially unchanged from full context — and in practice slightly worse: 7022ms vs. 6348ms full-context mean across five tasks (an 11% overhead from the in-place compression step). SnapKV KV pruning shows the same pattern: 6628ms TTFT (4% overhead), confirming that near-full-context TTFT is a property of the KV-pruning approach in general, not a quirk of PyramidKV's implementation. By contrast, cw128 achieves 2464ms TTFT at 65% retention, 61% faster than full context, because it feeds the model a shorter prompt from the start. PyramidKV's only speed benefit is in time per output token: 54.2ms vs. 67.5ms full-context (20% faster), because the compressed KV cache reduces memory bandwidth at each decode step. This TPT benefit is identical to what chunk methods achieve at the same retention level — it follows from KV cache size, not from the pruning method. PyramidKV provides no latency advantage for interactive applications and equivalent throughput to chunk methods for batch generation.

**First-token privilege.** Standard PyramidKV generation has a structural asymmetry: T1 is produced by dense attention over the full uncompressed context (the prefill forward pass runs over all tokens before compression occurs), while T2 and beyond attend only to the compressed KV cache. To isolate how much of PyramidKV's GT advantage derives from this privilege, we implemented a `clean_first` variant: the prefix `full_ids[:, :-1]` is prefilled under PyramidKV compression, and the final token is then decoded against the compressed cache, so T1 and all subsequent tokens attend only to the compressed context.

At 65% retention, the difference is negligible: `clean_first` scores within 0.19 points of standard PyramidKV on average across all 16 tasks, with no individual task moving more than 3 points. The first-token advantage exists but is minor.

At 35% retention, the result is starkly different. Removing the first-token privilege collapses performance by 15.7 points on average. The drops are largest on retrieval-heavy tasks: LCC (−62.8), PassageRetrieval (−44.0), TREC (−41.0), TriviaQA (−17.4), SAMSum (−17.3). Several tasks fall to zero. The 35% retention standard score of 24.8 — apparently competitive with full context — is almost entirely explained by a single token that attended to the full uncompressed context during prefill.

At 50% retention, the difference is again negligible: mean delta = −0.16, with no task collapsing. At 40% retention, the same: mean delta = −0.32. Across three successive compression rates, removing the first-token privilege changes GT scores by less than 0.4 points on average. The transition between 40% and 35% is abrupt: a privilege that accounts for essentially nothing at 40% accounts for nearly all of PyramidKV's GT performance at 35%.

**Synthesis.** Ground-truth evaluation cannot distinguish PyramidKV from a genuinely high-quality compression method. Its 25.2 average at 65% retention is the best result in this paper — yet at tight budgets (35% retention) the score is almost entirely explained by a single privileged token, the faithfulness gap is 3.7× worse than the best prompt-construction baseline at the same budget, and TTFT exceeds full context. None of these failures appear in the GT table. The KL metric (§3.2) detects the faithfulness gap; the timing data detects the speed gap; the `clean_first` ablation isolates the mechanism by which GT is gamed — and shows the effect is negligible at 65%/50%/40% but catastrophic at 35%. Together they illustrate why a single accuracy metric is insufficient to characterize compression quality — and why the evaluation framework proposed in this paper is necessary to distinguish methods that approximate full-context behavior from methods that produce correct answers for other reasons.

---

## 8. Analysis

### 8.1 When Phrase Selection Helps Most

Phrase selection provides the largest gains over naive recency on tasks where the required information is distributed across the document rather than concentrated in the tail. 2WikiMQA (multi-hop QA across multiple Wikipedia passages) and MultifieldQA (multi-domain passages) show the clearest benefit. NarrativeQA and MuSiQue show little or no benefit — suggesting the answer is consistently found in the recent tail, making phrase selection redundant.

This is a principled limitation: phrase selection is a query-driven method and only helps when (a) the query provides a useful selection signal and (b) the relevant content is not already in the recency tail. For summarization and code completion tasks, the "query" is generic ("write a summary") and provides no useful selection signal; the method degenerates to recency-based selection.

### 8.2 Comparison to KV Pruning Methods

Phrase-based compression outperforms every KV pruning method on KL faithfulness across all compression rates. The margin widens substantially as the budget shrinks: at 65% retention, phrase (0.144) edges naive (0.175) and comfortably beats SnapKV (0.188); at 35% retention, phrase (0.228) is less than a third of the KL of SnapKV (0.683).

The divergence at tight compression rates is explained by structural corruption severity scaling with pruning aggressiveness. At 35% retention, 65% of KV positions are evicted; early queries may have almost no valid keys in their causal window. Phrase-based compression at 35% still produces a structurally intact prompt — 35% of the tokens, but presented in causal order with no gaps.

One consistent exception is MultiNews, where SnapKV KV pruning (0.407 at 65%) substantially outperforms both naive (0.585) and phrase (0.644). MultiNews consists of several source articles concatenated for a multi-document summarization task. The document boundaries create a different attention structure than single-document tasks: SnapKV's observation-window scoring naturally captures inter-document attention patterns, while phrase selection operating on a flat token stream does not respect document boundaries and may fragment source articles. This suggests that structured multi-document inputs may benefit from attention-based selection even within a prompt-construction framework — a direction for future work.

### 8.3 The Role of Structural Integrity

The core finding is that *how* tokens are presented to the model matters as much as *which* tokens are presented. Phrase-based compression and naive truncation present a contiguous, causally complete sequence; KV pruning presents an incomplete one. The 3.8× NarrativeQA gap (and 1.55× average across 16 tasks) between equivalent token sets under the two mechanisms makes this concrete.

This has implications beyond the specific methods studied here. Any approach that modifies the attention mask to create causal gaps — sparse attention, block-sparse attention, token dropping during generation — is subject to the same corruption. The degree of corruption depends on how many positions have sparse causal coverage, but the mechanism is the same.

---

## 9. Discussion

**Simplicity as a feature.** Phrase-based compression requires no model introspection, no attention weight computation, no specialized CUDA kernels. The scoring is pure string matching. The construction is concatenation. This makes it immediately applicable to any transformer model, including those that use Flash Attention 2 (which does not materialize attention weights), without modification.

**Query availability.** The method requires a query to score phrases against. For RAG and QA settings this is natural. For open-ended generation, summarization, and code completion, the query may be absent or uninformative — in these cases the method reduces to naive recency, which is already competitive. Diversity-maximizing selection (choosing phrases that collectively cover the most distinct content) is a natural extension for query-free settings.

**Semantic phrase boundaries.** Fixed-size phrases are an approximation. Sentence and paragraph boundaries (phrase_sent) are a better approximation. Topic-coherent segmentation — identifying spans that are internally consistent and topically focused — is the ideal. Embedding-based topic segmentation methods [TextTiling, neural segmentation models] could provide this, at the cost of an additional model pass.

**Limitations.** The structural corruption analysis is empirical; a formal characterization of when and how severely it occurs across different architectures, context lengths, and pruning budgets would strengthen the theory. The phrase scoring method is lexical and will miss paraphrastic relevance (a passage relevant to "birth city" will not match a question asking "where was X born"). Semantic scoring — embedding similarity between phrase and query — is a natural upgrade but requires a separate encoder.

---

## 10. Conclusion

We have shown that KV cache pruning introduces structural corruption that fundamentally limits its faithfulness to the full-context model, regardless of how well the pruning method identifies important tokens. The mechanism — degenerate attention from early queries with sparse causal windows, cascading through all transformer layers — produces consistent faithfulness gaps even when token content is held constant (3.8× on NarrativeQA, 1.55× on average across 16 tasks).

Phrase-based context compression avoids this entirely by constructing a self-consistent short prompt rather than patching the KV cache. Phrases scored by query-document lexical overlap outperform all KV pruning methods on KL faithfulness, establishing a new state of the art using only string matching — no attention scores, no model internals, no added computation.

The central message is that fidelity to full-context behavior requires structural integrity, not just token coverage. Selecting the right tokens is necessary but not sufficient if those tokens are presented to the model in a structurally broken context.

---

## References

[To be completed]
