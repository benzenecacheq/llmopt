# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context inference is expensive, and many tokens in the prompt are not equally important for generating a good response. A large body of work has therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li et al., 2024], value norms [Feng et al., 2025], or combinations thereof — and evicting the rest before or during generation.

Standard benchmarks measure whether a compressed model gives the correct answer according to ground-truth labels. We argue this is the wrong objective: the goal of compression is approximation fidelity — producing the same output the full-context model would have produced. In this paper we introduce two faithfulness metrics: *KL Faithfulness*, which measures divergence from full-context token distributions, and *Output Faithfulness*, which measures text-level similarity between the full model's generated output and the compressed model's generated output. We show that ground-truth rankings and faithfulness rankings disagree substantially, and that the two new metrics reveal complementary failure modes.

Using KL faithfulness as our primary lens, we find that naïve proportional truncation — retaining the last 65% of tokens as a new, self-consistent prompt — substantially outperforms all published KV cache pruning methods. This surprising result is not a failure of token selection; it is attributable to structural corruption inherent to the KV patching mechanism: pruning the KV cache creates causal gaps wherever the retained set is sparse, and any query attending over a gapped neighborhood produces degenerate attention that cascades through all transformer layers.

This diagnosis motivates phrase-based context compression: constructing a new prompt from context spans selected by query-document lexical overlap, with no KV patching or attention mask modification. Phrase-based compression outperforms all KV cache pruning methods on KL faithfulness across 16 LongBench tasks, using only string matching — no model internals required.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. KV cache compression reduces this cost by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window. All of these methods share the same mechanism: they process the full prompt, compute importance scores, and then attend only to the retained subset during generation.

These compression methods are typically evaluated by running a compressed model on benchmark tasks and measuring how close the score is to an uncompressed baseline or to labeled ground truth [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that wrong answer — replicating the full model's behavior is the criterion, not improving on it.

We argue that existing approaches share a structural flaw, independent of how well any particular scoring method identifies important tokens. When KV cache pruning retains a subset of token positions, it modifies the causal attention mask: a query at position *p* can only attend to retained positions *q < p*. For early sequence positions, there may be very few or no retained keys in the causal window. Attention over an empty or near-empty key set produces degenerate distributions. These corrupted attention outputs propagate through all subsequent transformer layers. The damage is not localized to the positions with empty windows; it cascades.

This effect is easy to confirm empirically. We compare two methods that retain nearly identical token sets: naive proportional truncation (retaining the last 65% of tokens as a new, self-consistent prompt) and streaming-style KV pruning (retaining the last 65% of key-value states in the original position indices). On NarrativeQA, naive truncation achieves KL faithfulness of 0.022; streaming KV pruning achieves 0.084. The token content is almost the same; the 3.8× gap in faithfulness on this task (1.55× on average across 16 tasks) is entirely attributable to the structural corruption introduced by the pruning mechanism.

This finding has a straightforward implication: to faithfully compress a long context, one should not prune the KV cache. One should instead construct a shorter, self-consistent prompt. We propose *phrase-based context compression*, which does exactly this. The context is divided into contiguous phrases, each phrase is scored by lexical overlap with the query, the top-ranked phrases are concatenated with a recency tail to form the compressed prompt, and the model processes this shorter but structurally intact sequence normally.

We make the following contributions:

- **KL faithfulness metric**: We use KL divergence between the full and compressed model's next-token distributions at every generation step, averaged over a shared generation prefix, as the primary evaluation criterion.

- **Output faithfulness metric**: We introduce a second metric — word-level F1 between the full model's generated output and the compressed model's generated output — that measures behavioral convergence at the text level without any external ground-truth reference.

- **Structural corruption diagnosis**: We identify and demonstrate a fundamental flaw in KV cache pruning methods — degenerate attention from early queries with empty causal windows, cascading through all layers — using KL faithfulness as the diagnostic.

- **Phrase-based compression**: We introduce a prompt-construction approach that selects contiguous context spans by query-document lexical overlap, avoids structural corruption entirely, and outperforms all KV cache pruning methods on KL faithfulness.

## 2. Background and Related Work

**KV cache eviction.** ScissorHands [Wu et al., NeurIPS 2023] introduced the *persistence of importance* hypothesis: tokens that accumulate high attention during prefill remain important throughout generation, so the attention-score history is a reliable eviction criterion. H2O [Zhang et al., 2023] operationalizes this by maintaining a running sum of per-head attention scores and evicting the lowest-scoring tokens. SnapKV [Li et al., 2024] refines the scoring by pooling attention weights from only the last *w* queries (an observation window), better reflecting the queries that will matter at decode time, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] abandons score computation entirely, retaining a fixed set of attention sink tokens plus a sliding recency window; it achieves low latency at the cost of discarding all non-recent non-sink content.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Layer- and head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads within each layer. PyramidKV [Cai et al., 2024] takes a complementary approach, allocating different total budgets to different transformer layers — fewer KV slots at lower layers (where attention patterns are diffuse) and more at higher layers (where task-relevant patterns are concentrated). Both families operate on top of an existing token-scoring method and can in principle be combined with any of the per-token importance metrics discussed above. Our method uses a uniform budget per layer, and we note that the structural corruption problem (§6) applies to any of these methods that rely on the KV patching mechanism regardless of how budgets are distributed.

**Query-aware and dynamic selection.** Quest [Tang et al., ICML 2024] departs from static prefill-time selection by performing per-decode-step KV retrieval. At each generation step, the current query is used to retrieve the most relevant KV blocks from the full cache, limiting attention to the top-K pages. This eliminates the assumption that token importance is fixed at prefill time and instead adapts the attended set to each query. The tradeoff is computational: Quest requires a custom CUDA kernel for block-sparse attention, making integration into existing inference stacks non-trivial. Our structural corruption analysis applies to Quest only partially — because Quest retrieves complete contiguous blocks rather than individual scattered tokens, gaps within each attended block are minimal, though inter-block gaps remain.

**Token merging.** CaM [Wan et al., ICML 2024] and its successor KVMerger take a different approach: rather than evicting low-importance tokens, they *merge* pairs of similar KV vectors, collapsing them into a single weighted average and reducing cache size without introducing empty positions. This avoids the causal gap problem entirely — the retained set is contiguous — but produces keys and values that are off-manifold linear combinations not present in any normal prompt. The faithfulness cost of such merging versus structural corruption from eviction is an open question. CaM does not support grouped-query attention (GQA), limiting its applicability to GQA models such as Llama-3.1.

**Structural integrity concerns.** StructKV [Wang et al., Apr 2025] explicitly targets the observation that KV pruning disrupts semantic structure by evicting tokens mid-sentence. Their method enforces sentence-boundary eviction, preserving syntactic completeness of retained spans. This motivation is closely related to our phrase-based approach — both recognize that token-level selection produces incoherent fragments — but StructKV still operates via KV patching at sentence boundaries and therefore does not eliminate causal gaps. Concurrently, Devoto et al. [2025] ("Pitfalls of KV Cache Compression") provide empirical evidence that standard KV compression benchmarks are insufficient for evaluating compression quality and that compression methods perform significantly worse than their published numbers suggest under stricter evaluation conditions. This finding aligns with and provides additional evidence for our faithfulness-based critique of the field.

**Prompt compression.** LLMLingua [Jiang et al., 2023] and LongLLMLingua [Jiang et al., 2024] compress prompts at the token level by scoring token perplexity under a smaller proxy model and dropping low-perplexity tokens. This is architecturally different from KV cache pruning: the compressed prompt is fed to the model as-is, with no cache manipulation, so structural corruption is not a concern. However, perplexity-based scoring requires a forward pass over the full prompt using a second model, which adds latency comparable to the KV scoring overhead we observe in SnapKV-Select (§6.2). Phrase-based compression achieves similar prompt-shortening via pure string matching with no second model.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

---

## 3. Ground-Truth Evaluation

Before introducing our faithfulness metrics, we establish what the standard evaluation paradigm reveals. We run both Llama-3.1-8B and Mistral-7B-v0.3 on LongBench v1 (16 tasks, 100 examples each) at 65% retention, comparing naive proportional truncation, SnapKV, StreamingLLM (Streaming), and PyramidKV against the full uncompressed model.

**Setup.** Full details in §5.2. All methods target r = 0.65 retention. Naive truncation keeps the first 6.5% and last 58.5% of tokens as a new self-consistent prompt. SnapKV and Streaming use KV cache pruning with the full prefill. PyramidKV uses a layer-adaptive pyramid budget on top of SnapKV scoring. Ground-truth scores use the standard LongBench metrics: F1 for QA tasks, ROUGE for summarization, exact match for classification and code completion.

**Llama-3.1-8B.**

| Task             | Full | Naive | SnapKV   | Streaming | PyramidKV |
| ---------------- | ---- | ----- | -------- | --------- | --------- |
| NarrativeQA†     | 5.5  | 4.9   | 5.0      | **6.6**   | 5.6       |
| Qasper†          | 11.1 | 10.2  | 11.1     | 8.4       | **11.8**  |
| MultifieldQA     | 28.9 | 27.0  | 28.4     | 15.8      | **29.6**  |
| HotpotQA†        | 9.9  | 9.6   | 9.6      | **11.4**  | 10.2      |
| 2WikiMQA†        | 14.1 | 12.6  | **13.9** | 12.9      | **13.9**  |
| MuSiQue†         | 6.9  | 7.0   | 6.0      | 4.5       | **7.1**   |
| GovReport        | 20.4 | 19.8  | 19.9     | 5.2       | **20.3**  |
| QMSum†           | 10.3 | **11.5** | 9.7   | 3.5       | 9.4       |
| MultiNews        | 19.0 | 16.5  | **18.9** | 7.5       | 18.1      |
| TREC             | 70.0 | 66.0  | 66.0     | 50.0      | **71.0**  |
| TriviaQA         | 17.4 | 17.3  | 17.6     | **35.9**  | 17.5      |
| SAMSum           | 16.0 | 16.5  | **17.5** | 6.4       | 16.3      |
| PassageCount†    | 3.0  | 1.0   | **3.0**  | 2.0       | **3.0**   |
| PassageRetrieval | 44.0 | 37.0  | 37.0     | 20.0      | **44.0**  |
| LCC              | 68.1 | 63.4  | 63.2     | 17.8      | **68.5**  |
| RepoBench-P      | 55.6 | 53.9  | 53.4     | 6.3       | **56.0**  |
| **Average**      | 25.0 | 23.4  | 23.8     | 13.4      | **25.1**  |

Full context is the reference and is not bolded. Bold marks the best compressed method per task. Tasks marked † have full-context scores below 15; results on these tasks are noisier.

**Mistral-7B-v0.3.**

| Task             | Full | Naive    | SnapKV    | Streaming | PyramidKV |
| ---------------- | ---- | -------- | --------- | --------- | --------- |
| NarrativeQA†     | 5.1  | 2.7      | 4.4       | **8.5**   | 4.1       |
| Qasper†          | 5.3  | 4.3      | **8.3**   | 6.7       | 4.8       |
| MultifieldQA     | 25.3 | 22.1     | 23.9      | 19.9      | **24.8**  |
| HotpotQA†        | 10.5 | 10.6     | 10.5      | **12.4**  | 10.6      |
| 2WikiMQA†        | 11.5 | 9.8      | 11.6      | **14.9**  | 11.7      |
| MuSiQue†         | 5.1  | 3.9      | **5.4**   | 3.8       | 5.1       |
| GovReport        | 20.0 | 19.7     | 19.6      | 10.5      | **20.0**  |
| QMSum†           | 8.0  | **8.4**  | 7.3       | 7.5       | 7.9       |
| MultiNews        | 18.1 | 17.2     | **18.5**  | 7.9       | 17.3      |
| TREC             | 72.0 | 67.0     | 70.0      | 40.0      | **71.0**  |
| TriviaQA         | 23.1 | 24.2     | 25.4      | **59.9**  | 23.3      |
| SAMSum           | 16.9 | 17.8     | 17.3      | **20.4**  | 18.1      |
| PassageCount†    | 1.0  | **3.0**  | 2.0       | **3.0**   | 1.0       |
| PassageRetrieval | 39.0 | 26.0     | 34.0      | 17.0      | **39.0**  |
| LCC              | 62.9 | 60.2     | 57.1      | 24.7      | **63.2**  |
| RepoBench-P      | 53.9 | **54.4** | 49.4      | 22.8      | 54.0      |
| **Average**      | 23.6 | 22.0     | 22.8      | 17.5      | **23.5**  |

The overall pattern is the same across both architectures: ground-truth scores are compressed across methods, with naive (23.4 / 22.0) and SnapKV (23.8 / 22.8) within 2 points of each other and the full model. Streaming fails badly on tasks requiring distributed context retrieval. PyramidKV leads all compressed methods, nearly matching or exceeding full context on both models.

Taken at face value, PyramidKV is the clear winner. But §4 argues these scores measure the wrong thing — and §9 shows that PyramidKV's ground-truth advantage conceals a faithfulness cost an order of magnitude larger than any other method's.

---

## 4. Faithfulness Metrics

### 4.1 Why Ground-Truth Evaluation Falls Short

Let M be a language model, c a full context, q a query, and y* = M(c, q) the full-context output. A compression method produces a compressed context ĉ and output ŷ = M(ĉ, q). Standard benchmarks measure d(ŷ, y_gt) where y_gt is a ground-truth label. We instead measure d(ŷ, y*): how closely the compressed output approximates the full-context output.

Ground-truth evaluation has three distinct failure modes as a compression metric.

**Reference mismatch.** For the four ROUGE-based summarization tasks in LongBench (GovReport, QMSum, MultiNews, SAMSum), ground-truth is computed against a human-written reference. Two methods can generate completely different summaries with identical ROUGE scores, as long as both cover the same facts. Whether one summary resembles what the full model would have written is invisible to the metric. For the eight F1-based QA tasks, ground-truth measures string overlap with a gold answer extracted from the dataset — again with no reference to the full model's actual behavior. Only for classification and retrieval tasks does ground truth directly constrain the model's output to match specific tokens.

**Faithful errors are penalized.** If the full-context model produces a wrong answer, a compressed model that faithfully reproduces that same wrong answer scores 0 on ground truth. The metric penalizes faithfulness when it produces the wrong answer for the "right reasons." A compression method should not be penalized for being a good approximation.

**Accidental correctness is rewarded.** Conversely, a compressed model that accidentally produces the correct answer — because truncation happened to leave in the relevant span, or because a different reasoning path led to the same answer — scores the same as a method that genuinely approximates the full model. Ground truth cannot distinguish these cases.

**Low full-model accuracy amplifies all three problems.** On six of our 16 tasks the full model's ground-truth accuracy is below 15 (NarrativeQA: 5.5, MuSiQue: 6.9, PassageCount: 3.0, HotpotQA: 9.9, Qasper: 11.1, QMSum: 10.3). On NarrativeQA, for example, the full model gets 94.5% of examples wrong. A perfectly faithful compressed method should also get 94.5% wrong; ground truth scores all of those as 0. The compressed method's GT score on that task then reflects only how close its wrong answers happen to come to the gold reference — a function of failure-mode similarity to the reference, not of approximation quality. As a consequence, GT differences between compression methods on low-accuracy tasks are dominated by noise: small random variations in which wrong answers partially overlap with the gold string. The signal about actual compression quality is vanishingly small precisely on the tasks where the method has the most work to do.

These failure modes matter in practice. PyramidKV's ground-truth advantage (§3) is exactly the kind of result that is hard to interpret: it may reflect genuine preservation of task-relevant information, or it may reflect a combination of structural properties that happen to produce correct first tokens without approximating the full model's behavior. §9 shows it is the latter.

### 4.2 KL Faithfulness

We evaluate compression fidelity using KL divergence between the full and compressed models' next-token distributions, measured on a shared generation sequence. The uncompressed model generates a response y* = (y*₁,...,y*_L) from the full context. Both the full model and each compressed model are then teacher-forced on this shared token sequence: at step *t*, the full model conditions on [*c*, *q*, y*₁,...,y*_{t-1}] and the compressed model conditions on [*ĉ*, *q*, y*₁,...,y*_{t-1}]. The metric averages KL divergence across all generation steps:

```
KL_faith = (1/L) Σ_{t=1}^{L} KL(P_full(· | c, q, y*_{<t}) ‖ P_comp(· | ĉ, q, y*_{<t}))
```

Lower is better; 0 means identical distributions at every step. Conditioning all methods on the same shared token sequence eliminates path-dependence: a method whose compressed context causes early divergence would otherwise be evaluated on a different downstream sequence than a more faithful method, making scores incomparable across methods. Using y* as the shared prefix anchors every comparison to the full model's actual behavior.

This metric captures distributional agreement at the token probability level, not just which token is selected by greedy decoding. A method that shifts probability mass from the correct token to semantically similar alternatives — invisible to greedy accuracy measures — registers as a faithfulness cost in KL.

### 4.3 Output Faithfulness

KL faithfulness measures how well the compressed model's internal distributions match the full model's. A complementary question is: how similar is the text the compressed model actually generates?

We define **Output Faithfulness** (F_out) as:

```
F_out = (1/N) Σ_{i=1}^{N} f1(pred_full_i, pred_comp_i)
```

where f1 is word-level F1 between the full model's prediction and the compressed model's prediction on example i. Both predictions are generated autoregressively with greedy decoding; no external ground-truth reference is used.

Word-level F1 is computed after lowercasing, removing articles (a, an, the), and stripping punctuation — the same normalization used by LongBench for extractive QA evaluation. The metric ranges from 0 (no word overlap) to 1 (identical output text).

F_out addresses the failure modes of ground-truth evaluation directly: it does not compare to any external reference, so faithful errors are not penalized and accidental correctness is not rewarded. It also provides partial credit in two cases that binary accuracy metrics miss: when both models produce the same wrong answer (they are being equally and faithfully wrong), and when one model's output partially overlaps the other's (approximation with some drift). For a compression method that is a good approximation, F_out should be high regardless of whether the underlying outputs are "correct."

KL faithfulness and F_out are measuring different things:
- KL measures **distributional faithfulness**: how similar are the two models' probability distributions over the vocabulary at every generation step?
- F_out measures **behavioral faithfulness**: how similar are the two models' final generated texts?

These can disagree. A method could have low KL (distributions match well at each step) but low F_out (due to early divergence in sampled text). Conversely, a method could have high F_out (same final answers) but high KL (reached those answers via corrupted intermediate distributions). §9 shows that PyramidKV falls into this second category, making both metrics jointly necessary to characterize it.

### 4.4 Naïve Truncation as a Baseline

A key empirical finding motivating this work is that naïve proportional truncation consistently matches or outperforms all KV cache pruning methods on KL faithfulness. The method retains 65% of the prompt tokens, split as a 10%/90% head/tail budget: the first 6.5% of tokens (literal prompt prefix) and the last 58.5% of tokens (recency tail), concatenated into a new self-consistent prompt. The middle portion is discarded. On NarrativeQA, naive truncation achieves KL 0.022 vs. 0.060 for SnapKV. This result was surprising and prompted the structural corruption investigation described in §6.

### 4.5 Results at 65% Retention

**KL Faithfulness at 65%.** The KL results below compare all methods on Llama-3.1-8B; the Mistral results follow the same pattern at roughly half the absolute magnitude.

| Task             | Naive          | SnapKV         | Streaming | PyramidKV |
| ---------------- | -------------- | -------------- | --------- | --------- |
| NarrativeQA†     | **0.022**      | 0.060          | 0.084     | 1.338     |
| Qasper†          | 0.281          | **0.223**      | 0.378     | 1.229     |
| MultifieldQA     | 0.262          | **0.186**      | 0.419     | 1.535     |
| HotpotQA†        | **0.203**      | 0.246          | 0.265     | 2.181     |
| 2WikiMQA†        | **0.221**      | 0.254          | 0.369     | 2.422     |
| MuSiQue†         | **0.197**      | 0.283          | 0.293     | 2.523     |
| GovReport        | **0.280**      | 0.374          | 0.536     | 0.334     |
| QMSum†           | **0.081**      | 0.117          | 0.147     | 0.438     |
| MultiNews        | 0.585          | **0.407**      | 0.770     | 0.721     |
| TREC             | 0.202          | **0.146**      | 0.291     | 1.526     |
| TriviaQA         | **0.076**      | 0.122          | 0.150     | 1.647     |
| SAMSum           | **0.011**      | 0.048          | 0.020     | 1.182     |
| PassageCount†    | **0.059**      | 0.114          | 0.100     | 1.626     |
| PassageRetrieval | **0.188**      | 0.211          | 0.278     | 1.416     |
| LCC              | **0.076**      | 0.112          | 0.125     | 1.142     |
| RepoBench-P      | **0.048**      | 0.106          | 0.103     | 1.052     |
| **Average**      | **0.175**      | 0.188          | 0.270     | 1.394     |

Values in nats; lower is better. Bold marks the best method per row (excluding PyramidKV, which is shown for comparison). Naive leads on 11 of 16 tasks; SnapKV leads on 4. PyramidKV is worst by 7.5× (1.394 vs. 0.188).

**Output Faithfulness (F_out) at 65%.** The same methods evaluated on the new metric, comparing each model's output text against the full model's output text.

| Task             | Naive      | SnapKV     | Streaming | PyramidKV  |
| ---------------- | ---------- | ---------- | --------- | ---------- |
| NarrativeQA†     | **88.1**   | 51.0       | 2.6       | 62.1       |
| Qasper†          | 46.6       | 50.4       | 9.6       | **72.8**   |
| MultifieldQA     | 58.1       | 54.5       | 15.7      | **86.6**   |
| HotpotQA†        | 88.2       | 68.5       | 5.7       | **92.3**   |
| 2WikiMQA†        | 64.5       | 69.1       | 15.1      | **95.6**   |
| MuSiQue†         | **97.1**   | 65.0       | 4.1       | 92.3       |
| GovReport        | **55.9**   | 47.4       | 2.0       | 52.3       |
| QMSum†           | **54.4**   | 44.0       | 1.7       | 40.5       |
| MultiNews        | 44.5       | **54.4**   | 8.4       | 49.5       |
| TREC             | 75.0       | 67.6       | 10.2      | **80.9**   |
| TriviaQA         | 71.1       | 48.3       | 8.1       | **97.0**   |
| SAMSum           | 66.2       | 46.7       | 5.7       | **74.2**   |
| PassageCount†    | 70.1       | 25.4       | 15.6      | **96.4**   |
| PassageRetrieval | 80.1       | 66.1       | 4.7       | **93.8**   |
| LCC              | 66.8       | 66.6       | 7.3       | **92.7**   |
| RepoBench-P      | 81.9       | 61.0       | 1.3       | **91.9**   |
| **Average**      | **69.3**   | 55.4       | 7.4       | **79.4**   |

Values in %; higher is better. Bold marks the highest value per row. Streaming is uniformly near the floor (7.4% average, vs. 55–79% for the other methods) — its sink-plus-window mechanism discards the entire middle of the document, so it almost never reproduces the full model's output. This confirms F_out is not saturated and gives a concrete lower bound for the metric.

**The two metrics reverse the PyramidKV ranking.** On KL faithfulness, PyramidKV is the worst method (1.394 nats, 8× worse than naive). On F_out, PyramidKV is the best method (79.4%, 10 points above naive). Naive, the simplest method, ranks second on F_out (69.3%) and best on KL (0.175). These two facts are not contradictory — they reveal that KL and F_out measure orthogonal properties. §9 explains the mechanism that produces this divergence.

The Mistral results at 65% show the same pattern.

**KL Faithfulness.**

| Task             | Naive          | SnapKV         | Streaming | PyramidKV |
| ---------------- | -------------- | -------------- | --------- | --------- |
| NarrativeQA†     | **0.003**      | 0.016          | 0.024     | 0.780     |
| Qasper†          | **0.071**      | 0.079          | 0.102     | 0.472     |
| MultifieldQA     | 0.148          | **0.140**      | 0.217     | 0.718     |
| HotpotQA†        | **0.195**      | 0.227          | 0.284     | 0.725     |
| 2WikiMQA†        | 0.262          | **0.250**      | 0.356     | 0.683     |
| MuSiQue†         | **0.183**      | 0.204          | 0.268     | 0.679     |
| GovReport        | **0.180**      | 0.244          | 0.368     | 0.294     |
| QMSum†           | **0.067**      | 0.083          | 0.098     | 0.183     |
| MultiNews        | 0.586          | **0.340**      | 0.795     | 0.626     |
| TREC             | **0.012**      | 0.038          | 0.025     | 0.703     |
| TriviaQA         | **0.039**      | 0.094          | 0.097     | 1.379     |
| SAMSum           | **0.021**      | 0.046          | 0.029     | 1.050     |
| PassageCount†    | **0.037**      | 0.101          | 0.071     | 0.897     |
| PassageRetrieval | **0.195**      | 0.228          | 0.237     | 0.541     |
| LCC              | 0.055          | **0.054**      | 0.069     | 1.070     |
| RepoBench-P      | **0.076**      | 0.117          | 0.110     | 0.940     |
| **Average**      | **0.133**      | 0.141          | 0.197     | 0.734     |

Values in nats; lower is better. Bold marks the best method per row (excluding PyramidKV, which is shown for comparison). Naive leads on 12 of 16 tasks; SnapKV on 4. PyramidKV is worst by 5.5× (0.734 vs. 0.133).

**Output Faithfulness (F_out).**

| Task             | Naive    | SnapKV   | Streaming | PyramidKV |
| ---------------- | -------- | -------- | --------- | --------- |
| NarrativeQA†     | 41.0     | 32.5     | 8.6       | **55.5**  |
| Qasper†          | 68.9     | 54.8     | 7.8       | **86.0**  |
| MultifieldQA     | 64.5     | 66.9     | 20.4      | **89.2**  |
| HotpotQA†        | 66.9     | 75.9     | 20.7      | **94.7**  |
| 2WikiMQA†        | 67.3     | 74.7     | 21.9      | **95.0**  |
| MuSiQue†         | 64.4     | 69.3     | 19.3      | **93.3**  |
| GovReport        | 47.1     | 44.8     | 13.8      | **64.5**  |
| QMSum†           | 48.1     | 40.2     | 9.2       | **74.4**  |
| MultiNews        | 46.3     | **53.0** | 9.2       | 52.3      |
| TREC             | 84.7     | 80.1     | 15.8      | **92.5**  |
| TriviaQA         | 57.5     | 43.3     | 21.3      | **93.8**  |
| SAMSum           | 56.8     | 50.0     | 12.5      | **70.6**  |
| PassageCount†    | 56.8     | 39.1     | 11.0      | **97.5**  |
| PassageRetrieval | 68.8     | 72.0     | 14.5      | **97.8**  |
| LCC              | 66.9     | 67.1     | 10.4      | **91.8**  |
| RepoBench-P      | 70.8     | 63.6     | 12.0      | **92.9**  |
| **Average**      | 61.1     | 58.0     | 14.3      | **83.9**  |

Values in %; higher is better. Bold marks the highest value per row.

The ranking inversion is not model-specific: the KL ordering matches Llama (naive best, PyramidKV worst by a similar multiple), and the F_out ordering also matches — PyramidKV leads on 15 of 16 tasks, with MultiNews the one exception (SnapKV 53.0 vs. PyramidKV 52.3). Streaming is again near the floor (14.3% average vs. 7.4% on Llama), consistent with its mechanism rather than the model.

---

## 5. Initial Experiments

### 5.1 Methods

All methods in this section target 65% token retention (r = 0.65). We use proportional truncation (65% of actual prompt length) rather than a fixed token budget, which would over-retain on short inputs and over-prune on long ones.

**Reference.**
- *Full context*: unmodified model with the complete prompt; serves as the ceiling, not a compressed method.

**Prompt-construction baselines.** These methods truncate the prompt to form a shorter, self-consistent input with no KV patching or attention mask modification.
- *Naive proportional truncation (Naive)*: the prompt is split into a 10% head (literal first tokens) and a 90% recency tail, concatenated to form a new prompt at 65% of the original length.

**KV pruning methods.** These methods process the full prompt, score each token's KV entry, and retain only the top-scoring positions at their original sequence indices. The causal attention mask is then reconstructed so that each query can only attend to retained positions at earlier indices. All KV pruning methods unconditionally retain the first 16 tokens (`always_keep_first=16`) and the last 16 tokens (`always_keep_last=16`). For GQA models such as Llama-3.1-8B (32 Q-heads, 8 KV-heads), scores are computed per Q-head and aggregated via max-pooling across heads sharing a KV-head before selection.

- *SnapKV*: attention weights pooled over the last 128-token observation window [Li et al., 2024].
- *Streaming (StreamingLLM)*: first 4 attention sink tokens plus a recency window to fill the remaining budget [Xiao et al., 2023].

### 5.2 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128.

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. KL faithfulness compares compressed model distributions to full model distributions, teacher-forced on y*. F_out compares the compressed model's independently generated outputs to y* using word-level F1.

---

## 6. Structural Corruption in KV Cache Pruning

### 6.1 The Mechanism

KV cache pruning retains a subset of token positions and removes the rest. The retained positions keep their original sequence indices; a reconstructed causal mask allows each query to attend only to retained positions at earlier indices. This creates *causal gaps*: intervals of consecutive positions that are absent from the key-value store but logically present in the sequence.

For any token at position *p*, its attention output is computed over the intersection of its causal window {1,...,*p*} with the retained set. When that intersection is sparse — as it is for any token whose predecessors have been heavily pruned — the attention distribution is computed over an incomplete neighborhood. The resulting hidden state diverges from what full-context attention would produce. This is not a numerical edge case: it is the normal operating condition of every position in the pruned region.

The corruption propagates. Transformer hidden states are deeply entangled across layers: layer *l*'s computation at every position depends on the full layer *l*-1 output across all positions. A corrupted hidden state at one position is mixed into the residual stream of neighboring positions in the next layer, and that contaminated representation is then used to compute attention in layer *l*+1. After *L* layers, corruption originating at any pruned region has diffused throughout the representation.

This is not primarily an *early-token* problem, though early tokens are particularly vulnerable under recency-biased selection. It is a *gap* problem: any portion of the sequence where the retained set is sparse will produce corrupted attention. Recency-biased methods concentrate their budget at the tail, leaving the head and middle with severe gaps; differently-distributed selection creates gaps elsewhere. The mechanism is identical in both cases.

KV pruning introduces a second structural problem independent of causal gaps: *positional misalignment*. Retained tokens keep their original RoPE encodings, so two tokens that are logically adjacent in the retained set may carry a relative distance of hundreds or thousands of positions. The model's attention mechanism was trained on sequences where positional distance reflects informational proximity; retained tokens with large RoPE gaps between them are out-of-distribution. Prompt construction eliminates this simultaneously with the causal gap problem: by re-indexing all positions from scratch, the compressed sequence has positional encodings that accurately reflect its actual structure.

### 6.2 Empirical Evidence

We isolate structural corruption from token-selection quality by comparing KV pruning directly to prompt construction using identical scoring criteria. If the gap in performance is due to mechanism rather than token choice, then the same scored tokens — selected by the same algorithm — should perform dramatically better when presented as a new self-consistent prompt than when patched into the KV cache at their original positions.

We select SnapKV as our representative KV pruning method because it achieves the best KL faithfulness among KV pruning methods (§4.5). If structural corruption is the dominant factor, it should manifest even under the best-available KV scoring.

We evaluate one pair at 65% retention on six LongBench tasks under the y\* metric:

- **SnapKV** (KV pruning) vs **SnapKV-Select** (prompt construction, identical scoring)

| Method | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean (6 tasks) |
|---|---|---|---|---|---|---|---|
| Naive (prompt construction) | 0.221 | 0.262 | 0.281 | 0.081 | **0.048** | 0.076 | 0.162 |
| SnapKV — KV pruning | 0.254 | 0.186 | 0.223 | 0.117 | 0.106 | 0.122 | 0.168 |
| SnapKV-Select — prompt construction | **0.096** | **0.118** | **0.163** | **0.071** | 0.051 | **0.060** | **0.093** |

The results are unambiguous. SnapKV with KV pruning (0.168) is *worse* than naive\_65pct (0.162), despite using expensive attention scoring to select tokens. SnapKV-Select with identical token selection but prompt construction achieves 0.093 — a 45% reduction in KL divergence.

The gap between KV pruning and prompt construction cannot be attributed to token selection — the selected tokens are identical. It is entirely attributable to mechanism: prompt construction presents a self-consistent sequence with no causal gaps, while KV pruning exposes every non-tail query to a sparse and gapped neighborhood.

Switching to prompt construction also improves output faithfulness: SnapKV-Select averages 57.2% F_out vs. SnapKV's 54.6% on the same six-task subset — a 2.6 percentage-point gain. The contrast with the 45% KL reduction reflects that KL is a more sensitive measure of structural corruption than text-level output similarity; the two metrics are complementary (§4.4).

This penalty is not limited to sophisticated scoring methods. Even the simplest recency-biased comparison shows it: Streaming KV pruning (attention sinks + recency window) and naive\_65pct (prompt construction, last 65% of tokens) use the same recency logic and a similar token budget — the only difference is mechanism. Naive truncation outperforms Streaming on all 16 LongBench tasks in §4.5, with a 1.55× mean KL gap (0.175 vs. 0.270). The mechanism penalty holds uniformly across multi-hop QA, summarization, few-shot, synthetic, and code completion tasks alike.

Despite its quality advantage over stock SnapKV, SnapKV-Select is not a practical recommendation: computing SnapKV attention scores requires a full forward pass over the uncompressed prompt (~2500ms on our hardware), nearly cancelling the prefill savings from the shorter compressed prompt. This overhead motivates the question §7 addresses: can phrase-based selection — using only string matching, with no model access — approach SnapKV-Select's faithfulness at negligible cost?

---

## 7. Phrase-Based Context Compression

### 7.1 Motivation: The Quality Ceiling and Its Cost

The controlled comparison in §6.2 also reveals the quality ceiling for prompt-construction methods. SnapKV-Select — which uses SnapKV's attention-based scoring to select tokens but presents them as a new prompt — achieves a mean KL of 0.093 across six tasks, substantially outperforming naive\_65pct (0.162). Attention-based scoring, freed from structural corruption, is a strong selection signal.

However, computing SnapKV scores requires a full forward pass over the uncompressed prompt — approximately 2500ms on our hardware, nearly cancelling the prefill savings from the shorter compressed prompt (§6.2). This scoring overhead makes SnapKV-Select impractical for latency-sensitive deployment despite its quality advantage.

This motivates the question the rest of §7 addresses: is it possible to approach SnapKV-Select's quality using only information available without model introspection — no forward pass, no attention weights, no access to model internals?

### 7.2 Approach

Structural corruption is caused by the KV patching mechanism. The solution is simple: do not patch the KV cache. Instead, construct a new, shorter prompt from selected context spans and let the model process it normally. We call this *phrase-based compression*.

Given prompt construction as the mechanism, the remaining design question is what unit to select. Token-level selection is impractical without model introspection: individual BPE subword tokens carry little semantic content, and simple heuristics such as token-set overlap with the query are unreliable — the same surface word tokenizes differently in isolation versus in context, and common function-word fragments match spuriously. Operating at the phrase level solves this.

Phrases are delimited at fixed token positions rather than word boundaries, but because SentencePiece marks each word start with a space prefix, mid-word splits at phrase edges are uncommon in practice. More importantly, detokenizing a span of ~160 tokens yields enough complete words that lexical overlap with the query is a genuine semantic signal rather than a tokenization artifact; the occasional clipped boundary word has negligible effect on the overlap count at this granularity.

**Algorithm.** Given a full prompt of *T* tokens with a known query, a recency tail, and a phrase budget:

1. **Identify the tail**: reserve the most recent *tail_n* = ⌊*r* · *T* · *τ*⌋ tokens as a recency tail (always kept), where *r* = 0.65 is the retention fraction and *τ* = 0.25 is the tail fraction (selected by grid search in §7.4).
2. **Divide the remainder** (the "old" context before the tail) into contiguous phrases of *c* tokens each.
3. **Score each phrase** by word-level lexical overlap with the query:
   - Detokenize both phrase and query to text
   - Extract word sets (lowercased, whitespace/punctuation split)
   - Score = |phrase_words ∩ query_words|
4. **Select phrases greedily**: rank by score descending, with recency as a tiebreak; add phrases to the head budget until the remaining budget ⌊*r* · *T* · (1 − *τ*)⌋ is exhausted.
5. **Restore order**: sort selected phrases back to their original document positions.
6. **Construct prompt**: concatenate selected phrases + recency tail. Feed to the model as a normal forward pass.

The model receives a self-consistent sequence with no causal gaps. No KV patching, no attention mask modification, no per-token importance scores computed from model internals.

### 7.3 Scoring: Why Word Overlap Works

The scoring function is intentionally simple. Several alternatives were considered and tested:

- **Subword-token overlap**: intersect the raw tokenizer IDs of phrase and query. Fast, but BPE fragmentation causes mismatches, and common subword tokens (function word fragments) add noise.
- **Word-level overlap** (our default): detokenize to text first. Eliminates BPE fragmentation artifacts. Function words appear in both query and context but since we take a *set* intersection rather than weighted count, their contribution is limited to 1 per word type.
- **BM25**: inverse-document-frequency-weighted with length normalization. Empirically similar to word overlap on these tasks.

We use word-level overlap as the default because it is simple, requires no corpus statistics, and performs well empirically. Crucially, this scoring method has no access to the model's internals — it is pure string matching.

### 7.4 Phrase Boundaries

The current implementation uses fixed-size phrases of *c* tokens. **Phrase size and tail fraction selection.** We conducted a grid search over phrase sizes *c* ∈ {96, 128, 160} and tail fractions *τ* ∈ {0.20, 0.25, 0.30} to select hyperparameters. Each configuration was evaluated by mean y* KL divergence across the 6 primary tasks (2WikiMQA, MultifieldQA, Qasper, QMSum, RepoBench-P, TriviaQA) at 65% retention.

| Configuration | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | **Mean** |
|---|---|---|---|---|---|---|---|
| phrase\_word160\_t25 | 0.094 | 0.138 | 0.140 | 0.058 | 0.088 | 0.057 | **0.096** |
| phrase\_word160\_t30 | 0.100 | 0.135 | 0.152 | 0.057 | 0.085 | 0.054 | **0.097** |
| phrase\_word128\_t20 | 0.101 | 0.147 | 0.136 | 0.060 | 0.086 | 0.054 | **0.097** |
| phrase\_word160\_t20 | 0.094 | 0.137 | 0.143 | 0.071 | 0.088 | 0.053 | **0.098** |
| phrase\_word128\_t30 | 0.103 | 0.157 | 0.147 | 0.057 | 0.085 | 0.056 | **0.101** |
| phrase\_word96\_t20  | 0.099 | 0.208 | 0.179 | 0.058 | 0.098 | 0.057 | 0.116 |
| phrase\_word96\_t25  | 0.095 | 0.217 | 0.179 | 0.056 | 0.097 | 0.057 | 0.117 |
| phrase\_word96\_t30  | 0.097 | 0.210 | 0.180 | 0.057 | 0.094 | 0.058 | 0.116 |

The top configurations (phrase\_word160\_t25 at 0.096 and phrase\_word128\_t20 at 0.097) are statistically tied. phrase\_word128\_t20 is adopted as the single benchmark configuration (phr128) — its slightly shorter phrases produce more granular selections, and it performs marginally better at tighter compression rates. The 96-token variants are clearly inferior on MultifieldQA and Qasper, suggesting that 96 tokens is insufficient to capture coherent semantic units in these long-document tasks.

---

## 8. Main Experiments

### 8.1 Setup

**Models.** Llama-3.1-8B (base, fp16) and Mistral-7B-v0.3 (base, fp16).

**Benchmark.** LongBench v1, 16 English tasks, 100 examples per task.

**Methods compared.**

| Method | Type | Description |
|---|---|---|
| Full context | Reference | Uncompressed model |
| Naive | Prompt construction | Last 65% of tokens, 10% head / 90% tail split |
| SnapKV | KV pruning | Pooled attention weights over observation window |
| Streaming | KV pruning | Attention sinks + recency window |
| PyramidKV | KV pruning | Layer-adaptive pyramid budget allocation |
| phrase\_word128\_t20 (phr128) | Prompt construction | Word-overlap scoring, 128-token phrases, 20% tail |

All methods target 65% token retention unless noted. PyramidKV, SnapKV, Naive, and phr128 are also evaluated at 50%, 40%, and 35%.

### 8.2 KL Faithfulness: Main Results

At 65% retention, phr128 leads at 0.144 mean KL, beating naive truncation (0.175) and SnapKV KV pruning (0.188). phr128 wins outright on NarrativeQA, Qasper, MuSiQue, 2WikiMQA, PassageRetrieval, MultifieldQA, HotpotQA, and QMSum; naive truncation wins on LCC and SAMSum, where the relevant signal is already concentrated in the recency tail. MultiNews is the one task where SnapKV KV pruning (0.407) outperforms all prompt-construction methods — consistent with the multi-document structure noted in §10.2.

**Tables 2a–2d. KL Faithfulness across compression rates (lower is better). phr128 = phrase\_word128\_t20; SnapKV = SnapKV KV pruning; Pyr = PyramidKV. Bold = best among prompt-construction methods per row. PyramidKV excluded from bold competition.**

**Table 2a. 65% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 0.022 | 0.023 | **0.020** | 0.060 | 1.337 |
| Qasper† | 0.281 | 0.139 | **0.131** | 0.223 | 1.229 |
| MultifieldQA | 0.262 | **0.109** | 0.114 | 0.186 | 1.535 |
| HotpotQA† | 0.203 | **0.124** | 0.125 | 0.246 | 2.181 |
| 2WikiMQA† | 0.221 | 0.121 | **0.117** | 0.254 | 2.422 |
| MuSiQue† | 0.197 | 0.135 | **0.118** | 0.283 | 2.523 |
| GovReport | **0.280** | 0.385 | 0.385 | 0.374 | 0.334 |
| QMSum† | 0.081 | **0.058** | 0.060 | 0.117 | 0.438 |
| MultiNews | **0.585** | 0.644 | 0.647 | 0.407 | 0.721 |
| TREC | 0.202 | **0.112** | 0.128 | 0.146 | 1.526 |
| TriviaQA | 0.076 | 0.076 | **0.062** | 0.122 | 1.647 |
| SAMSum | **0.011** | 0.025 | 0.014 | 0.048 | 1.182 |
| PassageCount† | **0.059** | 0.067 | 0.067 | 0.114 | 1.626 |
| PassageRetrieval | 0.188 | 0.164 | **0.144** | 0.211 | 1.416 |
| LCC | **0.076** | 0.122 | 0.125 | 0.112 | 1.142 |
| RepoBench-P | 0.048 | 0.052 | **0.046** | 0.106 | 1.052 |
| **Average** | 0.175 | 0.147 | **0.144** | 0.188 | 1.394 |

**Table 2b. 50% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.026** | 0.038 | 0.028 | 0.138 | 1.380 |
| Qasper† | 0.457 | **0.256** | 0.264 | 0.348 | 1.394 |
| MultifieldQA | 0.359 | **0.094** | 0.144 | 0.274 | 1.619 |
| HotpotQA† | 0.212 | 0.114 | **0.099** | 0.425 | 2.245 |
| 2WikiMQA† | 0.278 | 0.145 | **0.129** | 0.405 | 2.282 |
| MuSiQue† | 0.188 | 0.123 | **0.089** | 0.472 | 2.507 |
| GovReport | **0.327** | 0.398 | 0.399 | 0.599 | 0.504 |
| QMSum† | 0.088 | **0.087** | 0.093 | 0.199 | 0.484 |
| MultiNews | **0.759** | 0.808 | 0.786 | 1.002 | 1.099 |
| TREC | 0.260 | **0.168** | 0.186 | 0.334 | 1.711 |
| TriviaQA | 0.095 | 0.091 | **0.087** | 0.296 | 1.622 |
| SAMSum | **0.012** | 0.016 | 0.018 | 0.222 | 1.199 |
| PassageCount† | **0.055** | 0.068 | 0.068 | 0.229 | 1.538 |
| PassageRetrieval | 0.188 | **0.113** | 0.117 | 0.331 | 1.378 |
| LCC | **0.105** | 0.153 | 0.150 | 0.223 | 1.238 |
| RepoBench-P | 0.065 | 0.060 | **0.057** | 0.329 | 1.016 |
| **Average** | 0.217 | 0.171 | **0.170** | 0.364 | 1.451 |

**Table 2c. 40% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.035** | 0.041 | 0.045 | 0.273 | 0.702 |
| Qasper† | 0.577 | 0.362 | **0.346** | 0.485 | 0.772 |
| MultifieldQA | 0.443 | **0.176** | 0.180 | 0.419 | 0.990 |
| HotpotQA† | 0.208 | 0.137 | **0.086** | 0.619 | 1.659 |
| 2WikiMQA† | 0.332 | 0.208 | **0.199** | 0.552 | 1.315 |
| MuSiQue† | 0.189 | 0.130 | **0.124** | 0.681 | 1.719 |
| GovReport | **0.391** | 0.438 | 0.438 | 0.813 | 0.506 |
| QMSum† | **0.104** | 0.104 | 0.107 | 0.308 | 0.342 |
| MultiNews | **0.888** | 0.899 | 0.903 | 1.760 | 1.024 |
| TREC | 0.282 | **0.188** | 0.203 | 0.615 | 1.340 |
| TriviaQA | 0.136 | 0.121 | **0.117** | 0.526 | 1.129 |
| SAMSum | **0.015** | 0.021 | 0.024 | 0.460 | 0.646 |
| PassageCount† | 0.093 | **0.072** | 0.073 | 0.332 | 1.168 |
| PassageRetrieval | 0.195 | **0.138** | 0.149 | 0.445 | 1.365 |
| LCC | **0.125** | 0.167 | 0.170 | 0.444 | 0.584 |
| RepoBench-P | **0.077** | 0.097 | 0.087 | 0.562 | 0.641 |
| **Average** | 0.256 | 0.206 | **0.203** | 0.581 | 0.994 |

**Table 2d. 35% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.045** | 0.055 | 0.076 | 0.334 | 0.559 |
| Qasper† | 0.637 | 0.385 | **0.381** | 0.588 | 0.858 |
| MultifieldQA | 0.475 | **0.171** | 0.203 | 0.483 | 0.914 |
| HotpotQA† | 0.220 | 0.166 | **0.127** | 0.666 | 1.252 |
| 2WikiMQA† | 0.351 | 0.262 | **0.220** | 0.643 | 1.052 |
| MuSiQue† | 0.191 | 0.161 | **0.158** | 0.786 | 1.236 |
| GovReport | **0.423** | 0.460 | 0.460 | 0.960 | 0.529 |
| QMSum† | 0.112 | **0.107** | 0.121 | 0.394 | 0.358 |
| MultiNews | **0.947** | 0.948 | 0.961 | 2.101 | 1.064 |
| TREC | 0.296 | 0.253 | **0.224** | 0.737 | 0.823 |
| TriviaQA | 0.157 | 0.151 | **0.142** | 0.636 | 0.833 |
| SAMSum | **0.017** | 0.024 | 0.028 | 0.606 | 0.475 |
| PassageCount† | 0.116 | **0.073** | 0.075 | 0.334 | 1.134 |
| PassageRetrieval | 0.282 | 0.184 | **0.182** | 0.489 | 1.240 |
| LCC | **0.145** | 0.181 | 0.182 | 0.534 | 0.550 |
| RepoBench-P | **0.087** | 0.107 | 0.107 | 0.635 | 0.511 |
| **Average** | 0.281 | 0.230 | **0.228** | 0.683 | 0.837 |

At 65% retention, phr128 (0.144) beats naive truncation (0.175) and SnapKV KV pruning (0.188). At 50%, phr128 (0.170) remains ahead of naive (0.217). At 40% and 35%, phr128 continues to lead (0.203 and 0.228 respectively). SnapKV KV pruning degrades catastrophically as the budget tightens (0.188 → 0.364 → 0.581 → 0.683), while prompt-construction methods degrade gracefully. The key cross-rate result: phr128/50% (0.170) matches Naive/65% (0.175) — nearly identical faithfulness with 15% less context. phr128/40% (0.203) beats Naive/50% (0.217), and phr128/35% (0.228) beats Naive/40% (0.256): tighter phrase compression is more faithful than looser naive truncation at every step. MultiNews at 65% remains the one exception where SnapKV KV pruning (0.407) edges out prompt-construction methods (Naive: 0.585), consistent with §10.2.

PyramidKV shows a counterintuitive compression trajectory: mean KL is 1.394 at 65%, rises slightly to 1.451 at 50%, then drops sharply to 0.994 at 40% and 0.837 at 35%. Tighter PyramidKV compression produces *better* KL faithfulness — the opposite of every other method. This is consistent with the clamping hypothesis: at 65% retention, PyramidKV's budget allocator clamps upward at lower layers, creating over-retention that disrupts the expected attention pattern. At 40–35% retention, the budget is unclamped and the pyramid allocation operates as intended. Even so, PyramidKV at its best (0.837 at 35%) remains 3.7× worse than the best prompt-construction method at the same budget (phr128: 0.228), confirming that the KL gap is structural rather than a configuration artifact.

**Mistral-7B-v0.3.** KL faithfulness at 65% and 35% retention. Column abbreviations and bold convention as above.

**Mistral 65% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.003** | 0.004 | 0.010 | 0.016 | 0.780 |
| Qasper† | 0.071 | 0.060 | **0.053** | 0.079 | 0.472 |
| MultifieldQA | 0.148 | 0.091 | **0.089** | 0.140 | 0.718 |
| HotpotQA† | 0.195 | 0.143 | **0.134** | 0.227 | 0.725 |
| 2WikiMQA† | 0.262 | 0.115 | **0.113** | 0.250 | 0.683 |
| MuSiQue† | 0.183 | 0.156 | **0.109** | 0.204 | 0.679 |
| GovReport | **0.180** | 0.270 | 0.271 | 0.244 | 0.294 |
| QMSum† | 0.067 | 0.065 | **0.053** | 0.083 | 0.183 |
| MultiNews | **0.586** | 0.678 | 0.677 | 0.340 | 0.626 |
| TREC | 0.012 | 0.010 | **0.009** | 0.038 | 0.703 |
| TriviaQA | 0.039 | **0.038** | 0.042 | 0.094 | 1.379 |
| SAMSum | **0.021** | 0.023 | 0.025 | 0.046 | 1.050 |
| PassageCount† | **0.037** | 0.038 | 0.038 | 0.101 | 0.897 |
| PassageRetrieval | 0.195 | **0.173** | 0.175 | 0.228 | 0.541 |
| LCC | **0.055** | 0.066 | 0.066 | 0.054 | 1.070 |
| RepoBench-P | 0.076 | **0.063** | 0.066 | 0.117 | 0.940 |
| **Average** | 0.133 | 0.125 | **0.121** | 0.141 | 0.734 |

**Mistral 35% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.011** | 0.019 | 0.019 | 0.055 | 0.203 |
| Qasper† | 0.165 | **0.110** | 0.124 | 0.155 | 0.262 |
| MultifieldQA | 0.213 | 0.129 | **0.119** | 0.243 | 0.513 |
| HotpotQA† | 0.218 | 0.122 | **0.120** | 0.284 | 0.633 |
| 2WikiMQA† | 0.362 | 0.201 | **0.176** | 0.311 | 0.684 |
| MuSiQue† | 0.186 | 0.145 | **0.136** | 0.261 | 0.687 |
| GovReport | **0.263** | 0.296 | 0.294 | 0.468 | 0.457 |
| QMSum† | 0.084 | **0.062** | 0.065 | 0.144 | 0.175 |
| MultiNews | **0.900** | 0.923 | 0.919 | 1.224 | 1.039 |
| TREC | 0.022 | **0.019** | 0.024 | 0.096 | 0.192 |
| TriviaQA | **0.062** | 0.090 | 0.085 | 0.301 | 0.722 |
| SAMSum | **0.027** | 0.033 | 0.043 | 0.109 | 0.737 |
| PassageCount† | 0.069 | **0.044** | 0.044 | 0.172 | 0.930 |
| PassageRetrieval | 0.197 | 0.186 | **0.167** | 0.274 | 0.920 |
| LCC | **0.109** | 0.114 | 0.111 | 0.135 | 0.564 |
| RepoBench-P | 0.110 | **0.090** | 0.104 | 0.270 | 0.661 |
| **Average** | 0.188 | 0.161 | **0.159** | 0.281 | 0.586 |

The Mistral rankings are consistent with Llama at both rates. At 65%, phr128 leads (0.121), beating Naive (0.133) and SnapKV KV pruning (0.141). At 35%, phr128 retains the lead (0.159). SnapKV degrades from 0.141 to 0.281 — a factor of 2× — versus Llama's 3.6× (0.188 → 0.683): the direction is the same, the severity smaller. PyramidKV again improves at tighter compression (0.734 → 0.586), consistent with the layer-budget clamping hypothesis, and again remains far worse than any prompt-construction method at the same budget (0.586 vs. 0.159 for phr128). MultiNews is the one task where SnapKV leads at 65% (0.340 vs. Naive 0.586); at 35% that advantage disappears.

### 8.3 Inference Performance

Compression is only worthwhile if it makes inference faster. We measure two quantities: **time to first token (TTFT)**, the wall-clock time from receiving a prompt to producing the first output token (dominated by the prefill pass), and **time per output token (TPT)**, the mean decode step time (dominated by KV cache memory bandwidth). All measurements are on a single V100 32GB GPU using Llama-3.1-8B fp16. TTFT for phrase methods includes selection overhead (~3ms). Means are over 100 examples × 5 tasks.

**Mean TTFT by retention rate** (Full context baseline: 6348 ms):

| Retention | phr128 | Savings | Pyr | Overhead |
|---|---|---|---|---|
| 65% | 2464 ms | 61% | 7022 ms | +11% |
| 50% | 2139 ms | 66% | 7109 ms | +12% |
| 40% | 1838 ms | 71% | 7135 ms | +12% |
| 35% | 1666 ms | 74% | 7143 ms | +13% |

Phrase methods cut TTFT by 60–74% simply by feeding the model a shorter prompt. PyramidKV goes the other direction entirely: it performs a full prefill before pruning the KV cache in-place, so its TTFT exceeds full context by 11–13% regardless of retention rate.

**Decode speed (TPT)** improves for all compressed methods because fewer retained tokens means a smaller KV cache to scan at each decode step. At 65% retention, mean TPT drops from 67.5 ms/tok (full) to ~54 ms/tok (~20% faster) for all methods including PyramidKV; at 35% retention, phrase methods reach ~48 ms/tok and PyramidKV reaches ~44 ms/tok (~29–35% faster).

### 8.4 Output Faithfulness Results

Output Faithfulness (F_out) measures how similar the compressed model's generated text is to the full model's generated text (§4.3). Higher is better. All comparisons are within-model: Llama compressed methods are compared to Llama full context; Mistral compressed methods are compared to Mistral full context.

**Tables 3a–3d. F_out across compression rates (higher is better). Bold = highest value per row.**

**Table 3a. 65% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **88.1** | 54.9 | 52.4 | 51.0 | 62.1 |
| Qasper† | 46.6 | 50.5 | 52.1 | 50.4 | **72.8** |
| MultifieldQA | 58.1 | 60.3 | 55.4 | 54.5 | **86.6** |
| HotpotQA† | 88.2 | 70.4 | 70.1 | 68.5 | **92.3** |
| 2WikiMQA† | 64.5 | 67.2 | 67.4 | 69.1 | **95.6** |
| MuSiQue† | **97.1** | 68.1 | 66.7 | 65.0 | 92.3 |
| GovReport | **55.9** | 40.4 | 40.8 | 47.4 | 52.3 |
| QMSum† | **54.4** | 41.3 | 43.6 | 44.0 | 40.5 |
| MultiNews | 44.5 | 37.3 | 36.2 | **54.4** | 49.5 |
| TREC | 75.0 | 72.8 | 73.3 | 67.6 | **80.9** |
| TriviaQA | 71.1 | 51.5 | 48.2 | 48.3 | **97.0** |
| SAMSum | 66.2 | 49.2 | 51.4 | 46.7 | **74.2** |
| PassageCount† | 70.1 | 32.5 | 32.5 | 25.4 | **96.4** |
| PassageRetrieval | 80.1 | 69.1 | 66.1 | 66.1 | **93.8** |
| LCC | 66.8 | 66.1 | 66.1 | 66.6 | **92.7** |
| RepoBench-P | 81.9 | 63.7 | 64.9 | 61.0 | **91.9** |
| **Average** | 69.3 | 56.0 | 55.5 | 55.4 | **79.4** |

**Table 3b. 50% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 48.6 | 51.2 | 51.0 | 36.4 | **53.4** |
| Qasper† | 38.3 | 44.4 | 42.7 | 44.5 | **52.4** |
| MultifieldQA | 56.1 | 59.4 | 59.2 | 45.2 | **71.3** |
| HotpotQA† | 65.2 | 69.6 | 70.8 | 54.0 | **78.2** |
| 2WikiMQA† | 62.9 | 65.8 | 63.4 | 56.5 | **84.8** |
| MuSiQue† | 64.4 | 67.9 | 67.3 | 53.8 | **80.9** |
| GovReport | 37.8 | 36.1 | 35.9 | 31.2 | **38.4** |
| QMSum† | 38.7 | 38.1 | **39.9** | 35.3 | 38.1 |
| MultiNews | **45.7** | 36.4 | 34.5 | 36.1 | 33.7 |
| TREC | 71.5 | 71.4 | **71.7** | 55.5 | 70.1 |
| TriviaQA | 45.8 | 49.9 | 51.1 | 26.7 | **76.0** |
| SAMSum | 49.7 | 48.6 | 49.1 | 35.5 | **49.8** |
| PassageCount† | 29.5 | 35.1 | 36.1 | 13.2 | **85.6** |
| PassageRetrieval | 67.0 | 66.8 | 61.8 | 45.0 | **82.6** |
| LCC | 62.6 | 63.7 | 64.1 | 55.4 | **74.0** |
| RepoBench-P | 62.3 | 61.7 | 60.6 | 39.5 | **72.7** |
| **Average** | 52.9 | 54.1 | 53.7 | 41.5 | **65.1** |

**Table 3c. 40% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 49.0 | 47.2 | 47.2 | 24.7 | **59.9** |
| Qasper† | 37.5 | 45.4 | 42.2 | 33.7 | **55.1** |
| MultifieldQA | 54.0 | 53.7 | 52.9 | 35.5 | **69.6** |
| HotpotQA† | 62.9 | 68.8 | 67.5 | 47.1 | **77.5** |
| 2WikiMQA† | 61.6 | 64.7 | 60.7 | 51.1 | **88.9** |
| MuSiQue† | 61.2 | 64.3 | 65.2 | 51.5 | **79.7** |
| GovReport | 37.9 | 33.4 | 35.6 | 24.9 | **40.4** |
| QMSum† | 38.1 | 39.0 | **39.2** | 29.8 | 38.4 |
| MultiNews | **40.8** | 33.1 | 33.4 | 17.6 | 33.9 |
| TREC | 67.4 | 70.1 | **70.5** | 50.3 | 70.3 |
| TriviaQA | 41.4 | 40.6 | 42.7 | 24.5 | **76.1** |
| SAMSum | 48.6 | 45.1 | 43.7 | 28.7 | **49.4** |
| PassageCount† | 26.1 | 29.8 | 26.3 | 16.0 | **82.7** |
| PassageRetrieval | 60.8 | 63.6 | 65.0 | 38.1 | **81.2** |
| LCC | 58.3 | 64.4 | 63.2 | 42.6 | **81.8** |
| RepoBench-P | 63.9 | 54.5 | 54.6 | 33.9 | **72.4** |
| **Average** | 50.6 | 51.1 | 50.6 | 34.4 | **66.1** |

**Table 3d. 35% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 44.9 | 44.5 | 47.7 | 18.7 | **57.6** |
| Qasper† | 34.5 | 42.3 | 42.8 | 30.3 | **51.9** |
| MultifieldQA | 48.0 | 54.3 | 57.0 | 35.8 | **70.5** |
| HotpotQA† | 58.9 | 68.7 | 65.1 | 45.2 | **85.6** |
| 2WikiMQA† | 56.9 | 60.4 | 59.8 | 50.4 | **87.7** |
| MuSiQue† | 58.5 | 62.5 | 63.5 | 47.3 | **84.7** |
| GovReport | 35.6 | 34.1 | 34.7 | 22.4 | **38.2** |
| QMSum† | 37.3 | 36.6 | **38.5** | 25.2 | 36.4 |
| MultiNews | **37.4** | 34.5 | 31.3 | 11.9 | 33.2 |
| TREC | 66.3 | 68.6 | **70.1** | 47.9 | 67.2 |
| TriviaQA | 39.0 | 38.2 | 43.6 | 21.4 | **84.7** |
| SAMSum | 43.2 | 40.1 | 40.7 | 24.0 | **57.0** |
| PassageCount† | 25.5 | 22.7 | 22.9 | 14.7 | **84.4** |
| PassageRetrieval | 57.6 | 60.9 | 61.2 | 38.9 | **85.4** |
| LCC | 59.8 | 63.6 | 61.6 | 41.0 | **80.9** |
| RepoBench-P | 60.3 | 54.2 | 50.7 | 32.8 | **76.4** |
| **Average** | 47.7 | 49.1 | 49.5 | 31.7 | **67.6** |

**Cross-rate summary.** Three patterns are visible across compression budgets:

PyramidKV leads F_out at every budget (79.4% → 65.1% → 66.1% → 67.6%), with a notably stable profile — the F_out actually improves slightly from 50% to 35% retention, the same direction as its KL improvement and for the same reason (layer-budget clamping resolves at tight compression). This stability contrasts starkly with its KL ranking, where it remains worst at every budget.

SnapKV degrades catastrophically on F_out as the budget tightens: 55.4% → 41.5% → 34.4% → 31.7%. At 35% retention, SnapKV reproduces less than a third of the full model's output text on average. The KL tables show the same direction but the F_out tables make the behavioral consequences concrete: at tight budgets, SnapKV is generating almost entirely different text from the full model.

Prompt-construction methods (naive, phr128) cluster tightly and degrade gracefully. At 65%, naive leads (69.3%); at 35%, phr128 has nearly caught up (~49%). The ordering varies by task, with no single method consistently leading — matching the KL pattern for these methods. At every budget, both prompt-construction methods are substantially more output-faithful than SnapKV.

**Mistral F_out.** The same qualitative pattern holds.

**Mistral 65% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 41.0 | 46.9 | 42.3 | 32.5 | **55.5** |
| Qasper† | 68.9 | 64.9 | 64.4 | 54.8 | **86.0** |
| MultifieldQA | 64.5 | 70.1 | 69.5 | 66.9 | **89.2** |
| HotpotQA† | 66.9 | 73.5 | 72.0 | 75.9 | **94.7** |
| 2WikiMQA† | 67.3 | 73.4 | 74.9 | 74.7 | **95.0** |
| MuSiQue† | 64.4 | 69.7 | 69.0 | 69.3 | **93.3** |
| GovReport | 47.1 | 42.9 | 45.2 | 44.8 | **64.5** |
| QMSum† | 48.1 | 48.7 | 50.9 | 40.2 | **74.4** |
| MultiNews | 46.3 | 33.9 | 34.4 | **53.0** | 52.3 |
| TREC | 84.7 | 86.8 | 84.5 | 80.1 | **92.5** |
| TriviaQA | 57.5 | 54.3 | 54.6 | 43.3 | **93.8** |
| SAMSum | 56.8 | 57.3 | 53.0 | 50.0 | **70.6** |
| PassageCount† | 56.8 | 54.8 | 54.8 | 39.1 | **97.5** |
| PassageRetrieval | 68.8 | 66.1 | 64.9 | 72.0 | **97.8** |
| LCC | 66.9 | 70.5 | 68.7 | 67.1 | **91.8** |
| RepoBench-P | 70.8 | 67.4 | 66.0 | 63.6 | **92.9** |
| **Average** | 61.1 | 61.3 | 60.6 | 58.0 | **83.9** |

**Mistral 35% retention.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 39.5 | 40.5 | 31.9 | 27.5 | **54.6** |
| Qasper† | 60.6 | 56.3 | 56.2 | 48.0 | **77.0** |
| MultifieldQA | 56.0 | 61.0 | 62.6 | 47.1 | **82.3** |
| HotpotQA† | 64.0 | 71.8 | 73.8 | 59.6 | **88.0** |
| 2WikiMQA† | 65.4 | 70.0 | 72.8 | 64.3 | **88.5** |
| MuSiQue† | 58.7 | 65.8 | 67.7 | 61.5 | **88.0** |
| GovReport | 36.8 | 35.4 | 40.1 | 25.1 | **46.1** |
| QMSum† | 39.9 | 41.6 | 43.9 | 41.1 | **61.5** |
| MultiNews | 36.3 | 29.7 | 28.9 | 32.5 | **36.3** |
| TREC | 78.9 | 80.4 | 78.2 | 68.3 | **86.8** |
| TriviaQA | 42.9 | 43.8 | 42.9 | 32.5 | **80.5** |
| SAMSum | 52.9 | 45.5 | 45.8 | 44.9 | **64.3** |
| PassageCount† | 39.2 | 41.1 | 41.4 | 22.2 | **87.1** |
| PassageRetrieval | 61.9 | 67.7 | 59.5 | 68.4 | **88.4** |
| LCC | 59.5 | 62.4 | 62.8 | 51.4 | **81.8** |
| RepoBench-P | 62.6 | 57.0 | 54.2 | 48.4 | **85.9** |
| **Average** | 53.5 | 54.4 | 53.9 | 46.4 | **74.8** |

Mistral shows the same qualitative patterns as Llama: PyramidKV leads F_out at both rates (83.9%, 74.8%); SnapKV degrades more steeply than prompt-construction methods (58.0% → 46.4% vs. ~53–61% → ~52–54%); prompt-construction methods cluster tightly. The absolute F_out values are generally higher on Mistral than Llama, consistent with Mistral having more concentrated output distributions.

**What F_out and KL together reveal.** The two metrics agree that SnapKV's F_out degradation is severe and structural — closely mirroring its KL degradation. They agree that prompt-construction methods degrade gracefully on both dimensions. They sharply disagree on PyramidKV: worst on KL, best on F_out. This divergence is the subject of §9.

---

## 9. PyramidKV: A Case Study

PyramidKV maximizes every ground-truth metric in this paper: best GT score (25.1 average at 65% retention, §3), best output faithfulness (F_out 79.4% at 65%, §8.4), and worst KL faithfulness (1.394 nats at 65%, §8.2). Its inference speed profile is the inverse of all other methods: slower than full context at prefill (+11% TTFT), identical throughput at decode (same TPT as phrase methods at equal retention). Understanding why these metrics diverge is necessary for understanding what compression quality actually means.

PyramidKV allocates KV budget unevenly across layers: near-full context at layer 0, decreasing linearly to roughly 600 of 4096 tokens by layer 31 at 65% retention. Because the lower layers see nearly the full context, they pass a largely uncorrupted decision-relevant signal into the residual stream; the sparse upper layers refine this signal but rarely overturn it — which is why F_out is high (the predicted token is fixed early) while KL is high (the full distribution, computed through all 32 layers, is corrupted by upper-layer gaps). A PyramidKV-Select ablation confirms the mechanism directly: keeping the same pyramid-weighted token scores but presenting them as a gapless prompt instead of patching the KV cache drops F_out from 79.4% to 56.3% — down to the level of other prompt-construction methods — showing that KV patching, not token selection, is what produces PyramidKV's F_out advantage.

### 9.1 What F_out Reveals

PyramidKV's output text is more similar to the full model's output text than any other compressed method, at every compression budget tested. The F_out advantage is particularly large on tasks with short, structured outputs: TriviaQA (97.0%), PassageCount (96.4%), 2WikiMQA (95.6%), PassageRetrieval (93.8%), LCC (92.7%), HotpotQA (92.3%). On these tasks, PyramidKV consistently produces the same text as the full model — the same classification label, the same retrieved passage, the same code completion.

On tasks with longer, free-form outputs, the advantage shrinks and is sometimes reversed. Table 4 compares PyramidKV, naive truncation, and SnapKV by output-length category (Llama, 65%), showing that PyramidKV's advantage is also the most length-sensitive of the three methods.

**Table 4. F_out by length category (Llama, 65%). Short = full-model reference output ≤ 90 words (n=1155); Long = > 90 words (n=445).**

| Method | Short-answer F_out | Long-form F_out | Gap |
|---|---|---|---|
| PyramidKV | **88.1** | 55.5 | 32.6 |
| Naive | 72.8 | **57.6** | 15.2 |
| SnapKV | 56.1 | 50.2 | 6.0 |

PyramidKV's short-to-long drop (32.6 points) is more than twice naive's (15.2) and five times SnapKV's (6.0). SnapKV has little length-dependent advantage to lose in the first place — it is mediocre on both categories — so its flatness is not a virtue, but PyramidKV's steepness is a real liability: the headline F_out number is propped up almost entirely by short, structured outputs, and erodes fastest of all three methods as soon as the task demands sustained generation.

This asymmetry has a structural explanation: PyramidKV performs a full prefill over the uncompressed prompt before pruning the KV cache in-place, so its output distribution at the first generated token is identical to the full model's by construction. On short-answer tasks where the answer is a single token or short phrase — a class label, a retrieved passage, a trivia answer — the first token IS the answer, so F_out is essentially 1.0 whenever the full model is also correct. The average F_out of 79.4% is substantially driven by these tasks dominating the 16-task mix.

For tasks requiring sustained coherent generation, the prefill advantage disappears. Tokens after the first attend only to the pruned KV cache, and distributional damage accumulates with each generation step. F_out is insensitive to this on short-answer tasks because a single correct token produces a high score regardless of what follows; on long-form tasks, where the comparison spans many tokens, the damage becomes visible. PyramidKV's long-form advantage over naive truncation is large on Mistral but marginal-to-negative on Llama at 65–40% retention — the same structural corruption, with architecture-dependent severity.

### 9.2 Inference Speed

PyramidKV's speed profile makes its GT advantage difficult to justify in deployment. Because it performs a full prefill over the uncompressed prompt before pruning the KV cache in-place, its TTFT is 7022ms — 11% *slower* than full context's 6348ms and 2.8× slower than phr128 at the same retention (2464ms). This penalty holds across all four retention rates (+11–13% overhead). The only speed benefit is at decode time: 54.2ms/tok vs. 67.5ms/tok full context, the same 20% improvement that any method at 65% retention achieves by reducing KV cache size.

A method that pays full prefill cost, provides no TTFT improvement, has worst-in-class KL, and best-in-class F_out presents an unusual profile: it is useful only if behavioral output similarity (F_out) is the sole criterion and latency is irrelevant.

### 9.3 Synthesis

PyramidKV illustrates why no single metric suffices to characterize compression quality. GT and F_out both rank it first. KL ranks it last. Each is measuring something real:

- GT and F_out measure *behavioral* fidelity: does the compressed model produce the same outputs as the full model? PyramidKV's near-intact lower layers establish the same representational foundation as the full model, driving the same decisions. The answer texts match.

- KL measures *distributional* fidelity: does the compressed model assign the same probability mass as the full model at each generation step? PyramidKV's upper-layer KV gaps corrupt the distributions even when the final output is correct. The distributions do not match.

These are different properties of compression quality. A method optimized purely for GT or F_out could preserve the argmax token at every step while completely misrepresenting the full model's uncertainty and alternative predictions. A method optimized purely for KL might shift probability mass to semantically equivalent alternatives — invisible to GT but costly in KL — while producing identical final text. Neither metric alone characterizes the compression.

The two metrics together reveal that PyramidKV's "faithfulness" is a particular kind: it replicates the full model's decisions without replicating the full model's reasoning. This has practical consequences: a downstream system that uses the compressed model's logits for calibration, uncertainty estimation, or beam search will see corrupted information even when the greedy output is correct.

Ground-truth evaluation cannot detect any of this. KL and F_out together can.

Putting the length-dependence (Table 4) together with the speed profile (§9.2) gives a practical verdict, not just a metrics curiosity. On short-answer tasks, PyramidKV's accuracy and F_out advantage is real, but it is bought at a cost that defeats the purpose of compression: full prefill over the uncompressed prompt means TTFT is *slower* than full context (7022ms vs. 6348ms), so on exactly the tasks where it wins, it is also the slowest method on the page. On long-form tasks, where generation cost is large enough that PyramidKV's decode-time speedup could matter, its faithfulness has converged to — or fallen behind — naive truncation and SnapKV, which deliver the same decode speedup without PyramidKV's prefill penalty. There is no regime in which PyramidKV is both faster and more faithful than a cheaper alternative: on short outputs it is more faithful than the other compressed methods but slower than using no compression at all, and on long outputs it is no more faithful than methods several times faster. Since the entire point of KV cache compression is to improve performance, this is a failure on its own terms, independent of where any individual metric ranks it.

---

## 10. Analysis

### 10.1 When Phrase Selection Helps Most

Phrase selection provides the largest gains over naive recency on tasks where the required information is distributed across the document rather than concentrated in the tail. 2WikiMQA (multi-hop QA across multiple Wikipedia passages) and MultifieldQA (multi-domain passages) show the clearest benefit. NarrativeQA and MuSiQue show little or no benefit — suggesting the answer is consistently found in the recent tail, making phrase selection redundant.

This is a principled limitation: phrase selection is a query-driven method and only helps when (a) the query provides a useful selection signal and (b) the relevant content is not already in the recency tail. For summarization and code completion tasks, the "query" is generic ("write a summary") and provides no useful selection signal; the method degenerates to recency-based selection.

### 10.2 Comparison to KV Pruning Methods

Phrase-based compression outperforms every KV pruning method on KL faithfulness across all compression rates. The margin widens substantially as the budget shrinks: at 65% retention, phrase (0.144) edges naive (0.175) and comfortably beats SnapKV (0.188); at 35% retention, phrase (0.228) is less than a third of the KL of SnapKV (0.683). The same direction holds on F_out: at 35%, phr128 (49.5%) is 18 points above SnapKV (31.7%).

One consistent exception is MultiNews, where SnapKV KV pruning (0.407 at 65%) substantially outperforms both naive (0.585) and phrase (0.644) on KL. MultiNews consists of several source articles concatenated for a multi-document summarization task. The document boundaries create a different attention structure than single-document tasks: SnapKV's observation-window scoring naturally captures inter-document attention patterns, while phrase selection operating on a flat token stream does not respect document boundaries and may fragment source articles. This suggests that structured multi-document inputs may benefit from attention-based selection even within a prompt-construction framework.

### 10.3 The Role of Structural Integrity

The core finding is that *how* tokens are presented to the model matters as much as *which* tokens are presented. Phrase-based compression and naive truncation present a contiguous, causally complete sequence; KV pruning presents an incomplete one. The 3.8× NarrativeQA gap (and 1.55× average across 16 tasks) between equivalent token sets under the two mechanisms makes this concrete.

This has implications beyond the specific methods studied here. Any approach that modifies the attention mask to create causal gaps — sparse attention, block-sparse attention, token dropping during generation — is subject to the same corruption. The degree of corruption depends on how many positions have sparse causal coverage, but the mechanism is the same.

---

## 11. Discussion

**Simplicity as a feature.** Phrase-based compression requires no model introspection, no attention weight computation, no specialized CUDA kernels. The scoring is pure string matching. The construction is concatenation. This makes it immediately applicable to any transformer model, including those that use Flash Attention 2 (which does not materialize attention weights), without modification.

**Query availability.** The method requires a query to score phrases against. For RAG and QA settings this is natural. For open-ended generation, summarization, and code completion, the query may be absent or uninformative — in these cases the method reduces to naive recency, which is already competitive. Diversity-maximizing selection (choosing phrases that collectively cover the most distinct content) is a natural extension for query-free settings.

**Semantic phrase boundaries.** Fixed-size phrases are an approximation. Sentence and paragraph boundaries (phrase_sent) are a better approximation. Topic-coherent segmentation — identifying spans that are internally consistent and topically focused — is the ideal. Embedding-based topic segmentation methods could provide this, at the cost of an additional model pass.

**Limitations.** The structural corruption analysis is empirical; a formal characterization of when and how severely it occurs across different architectures, context lengths, and pruning budgets would strengthen the theory. The phrase scoring method is lexical and will miss paraphrastic relevance (a passage relevant to "birth city" will not match a question asking "where was X born"). Semantic scoring — embedding similarity between phrase and query — is a natural upgrade but requires a separate encoder.

---

## 12. Conclusion

We have shown that KV cache pruning introduces structural corruption that fundamentally limits its faithfulness to the full-context model, regardless of how well the pruning method identifies important tokens. The mechanism — degenerate attention from early queries with sparse causal windows, cascading through all transformer layers — produces consistent faithfulness gaps even when token content is held constant (3.8× on NarrativeQA, 1.55× on average across 16 tasks).

We introduced two faithfulness metrics. KL faithfulness measures distributional agreement between the full and compressed model at every generation step, capturing whether the compressed model's internal representations match the full model's. Output faithfulness measures text-level similarity between the two models' generated outputs, capturing whether the compressed model makes the same decisions as the full model. The PyramidKV case study (§9) shows these metrics reveal orthogonal failure modes: PyramidKV achieves the best output faithfulness of any method (79.4%) while simultaneously achieving the worst KL faithfulness (1.394 nats) — because its layer-adaptive budget leaves lower layers nearly intact (enabling correct decisions) while corrupting upper-layer distributions throughout (measured by KL). Neither metric alone characterizes this behavior; both are necessary.

Phrase-based context compression avoids structural corruption entirely by constructing a self-consistent short prompt rather than patching the KV cache. Phrases scored by query-document lexical overlap outperform all KV pruning methods on KL faithfulness, establishing a new state of the art using only string matching — no attention scores, no model internals, no added computation. The method also cuts TTFT by 60–74%, providing simultaneous quality and latency advantages over KV pruning approaches.

The central message is that fidelity to full-context behavior requires structural integrity, not just token coverage. Selecting the right tokens is necessary but not sufficient if those tokens are presented to the model in a structurally broken context.

---

## References

[To be completed]
