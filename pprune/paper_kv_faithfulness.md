# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context inference is expensive, and many tokens in the prompt are not equally  important for generating a good response. A large body of work has  therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li  et al., 2024], value norms [Feng et al., 2025b], or combinations thereof — and evicting the rest before or during generation.

Standard benchmarks measure whether a compressed model gives the correct answer according to ground-truth labels. We argue this is the wrong  objective: the goal of compression should be approximation fidelity — producing the same output the full-context model would have produced. In this  paper we introduce two faithfulness metrics: *KL Faithfulness*, which measures divergence from full-context token distributions, and *Output Faithfulness*, which measures text-level similarity between the full model's generated output and the compressed model's generated output. Ground-truth rankings and faithfulness rankings may disagree substantially, and the two new metrics reveal complementary failure modes which we analyze in detail and which suggest improvements to the current popular pruning methods.

Using KL faithfulness as our primary lens, we observe that SnapKV and PyramidKV, which among the algorithms we tested yielded the best ground-truth results, are dramatically less faithful than a simple naive proportional truncation — retaining 65% of tokens (head-and-tail split) as a new, self-consistent prompt.  Deeper analysis shows that this is not because of poor token selection, but because retaining tokens at their original RoPE positions after eviction causes positional displacement that corrupts every subsequent decode step. StreamingLLM with key re-rotation is the best existing method, posting the lowest KL faithfulness among currently deployed KV cache compression approaches.

This diagnosis suggests a fix: apply key re-rotation to SnapKV after eviction, remapping retained keys to their new compact positions. The improvement is dramatic — 27× reduction in mean KL faithfulness at 65% retention, making SnapKV+rot the most faithful method overall, outperforming all other methods tested. The central practical message is that SnapKV and PyramidKV as currently deployed are substantially less faithful than they could be: key re-rotation is a single post-eviction tensor operation that recovers most of the faithfulness loss at negligible cost.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. The attention calculation itself, which uses the cache, consumes the lion's share of the token generation time.  KV cache compression reduces these costs by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window, with no importance scoring. H2O and SnapKV share the same broad mechanism: they process the full prompt, compute importance scores, then evict a subset of the tokens from the KV cache, attending only to the retained subset during generation.

We make the following contributions:

- **Ground-truth evaluation as the wrong criterion**: Standard benchmarks measure whether a compressed model gives the correct answer, not whether it behaves the same way as the full model. We argue this is the wrong objective for compression and analyze several distinct failure modes of GT evaluation as a compression metric (§5.1).

- **KL faithfulness metric (F_KL)**: KL divergence between the full and compressed model's next-token distributions at every generation step, averaged over a shared generation prefix. Measures whether the compressed model's computation matches the full model's at every step, not just whether the final output is correct (§5.2).

- **Output faithfulness metric (F_out)**: Word-level F1 between the full model's generated output and the compressed model's generated output, with no external ground-truth reference. Reveals a behavioral inversion: methods that score best on ground-truth benchmarks score worst on KL faithfulness (§5.3, §8).

- **Structural corruption diagnosis**: Positional displacement — retained tokens keeping their original RoPE encodings after eviction — is the dominant failure mode for scattered-selection KV methods, demonstrated via a controlled gap-geometry experiment and a same-selection comparison (§6.2–§6.3).

- **Key re-rotation as the fix**: Remapping retained keys to compact positions after eviction dramatically improves faithfulness, making SnapKV+rot the most faithful method overall and confirming that the failure mode is structural rather than selection-based (§6.4).

## 2. Background and Related Work

**KV cache eviction.** Scissorhands [Liu et al., 2023] introduced the *persistence of importance* hypothesis: tokens that accumulate high attention during prefill remain important throughout generation, so the attention-score history is a reliable eviction criterion. H2O [Zhang et al., 2023] operationalizes this by maintaining a running sum of per-head attention scores and evicting the lowest-scoring tokens. SnapKV [Li et al., 2024] refines the scoring by pooling attention weights from only the last *w* queries (an observation window), better reflecting the queries that will matter at decode time, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] abandons score computation entirely, retaining a fixed set of attention sink tokens plus a sliding recency window. In its original online design the cache is bounded continuously throughout processing — the model never builds a full-length KV cache — which bounds TTFT at the cost of discarding all non-recent non-sink content.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025b] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Layer- and head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads within each layer. PyramidKV [Cai et al., 2024] takes a complementary approach, allocating different total budgets to different transformer layers — more KV slots at lower layers (which the original authors find need broader context) and fewer at higher layers (where task-relevant patterns have already concentrated). Both families operate on top of an existing token-scoring method and can in principle be combined with any of the per-token importance metrics discussed above. Our method uses a uniform budget per layer, and we note that the structural corruption problem (§6) applies to any of these methods that rely on the KV patching mechanism regardless of how budgets are distributed.

**Query-aware and dynamic selection.** Quest [Tang et al., ICML 2024] departs from static prefill-time selection by performing per-decode-step KV retrieval. At each generation step, the current query is used to retrieve the most relevant KV blocks from the full cache, limiting attention to the top-K pages. This eliminates the assumption that token importance is fixed at prefill time and instead adapts the attended set to each query. The tradeoff is computational: Quest requires a custom CUDA kernel for block-sparse attention, making integration into existing inference stacks non-trivial. Our structural corruption analysis applies to Quest only partially — because Quest retrieves complete contiguous blocks rather than individual scattered tokens, gaps within each attended block are minimal, though inter-block gaps remain.

**Token merging.** CaM [Y. Zhang et al., 2024] and its successor KVMerger [Wang et al., 2024] take a different approach: rather than evicting low-importance tokens, they *merge* pairs of similar KV vectors, collapsing them into a single weighted average and reducing cache size without introducing empty positions. This avoids the causal gap problem entirely — the retained set is contiguous — but produces keys and values that are off-manifold linear combinations not present in any normal prompt. The faithfulness cost of such merging versus structural corruption from eviction is an open question. CaM does not support grouped-query attention (GQA), limiting its applicability to GQA models such as Llama-3.1.

**Evaluation critique.** Chen et al. [2025] ("Pitfalls of KV Cache Compression") provide empirical evidence that standard KV compression benchmarks are insufficient for evaluating compression quality and that compression methods perform significantly worse than their published numbers suggest under stricter evaluation conditions. This finding aligns with and provides additional evidence for our faithfulness-based critique of the field.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2025] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

---

## 3. Initial Experiments

### 3.1 Methods

All methods in this section target 65% token retention (r = 0.65). We use proportional truncation (65% of actual prompt length) rather than a fixed token budget, which would over-retain on short inputs and over-prune on long ones.

**Reference.**
- *Full context*: unmodified model with the complete prompt; serves as the ceiling, not a compressed method.

**Prompt-construction baseline.** This method truncates the prompt to form a shorter, self-consistent input with no KV patching or attention mask modification.
- *Naive proportional truncation (Naive)*: the prompt is split into a 10% head (literal first tokens) and a 90% recency tail, concatenated to form a new prompt at 65% of the original length.

**KV pruning methods.** The benchmarked KV pruning methods (SnapKV, PyramidKV, Streaming) are implemented via `kvpress` v0.5.4. They process the full prompt, score each token's KV entry using post-prefill attention weights, and evict lower-scoring entries after the prefill pass. The last `window_size` tokens (the observation window) are always retained by being assigned high scores; we use `window_size=128` throughout (kvpress stock default is 64). For GQA models such as Llama-3.1-8B (32 Q-heads, 8 KV-heads), kvpress aggregates scores via mean-pooling across query heads sharing a KV head. The §6.2 diagnostic experiments (Table 1) use a separate custom implementation with `q_buffer_size=128`, `always_keep_first=16`, `always_keep_last=16`, and max-pooling for GQA aggregation.

- *SnapKV*: attention weights pooled over the last 128-token observation window [Li et al., 2024].
- *PyramidKV*: same SnapKV-style attention scoring, but with a layer-adaptive budget that allocates more KV slots to lower layers and fewer to higher layers [Cai et al., 2024]. The per-layer budget follows a linear decay controlled by `beta=20`; budgets are clamped to `[window_size, sequence_length]`.
- *Streaming (sink + recency, with key re-rotation)*: first 4 attention sink tokens plus a recency window to fill the remaining budget, following StreamingLLM's selection rule [Xiao et al., 2023]. Implemented via `kvpress`'s `StreamingLLMPress` wrapped in `KeyRerotationPress`. The kvpress documentation for `StreamingLLMPress` explicitly recommends this wrapper "to fully match the implementation described in the paper": in the original online design, HuggingFace's `SinkCache` re-rotates keys as the recency window slides forward during generation to keep positions consistent; in kvpress's post-prefill adaptation, `KeyRerotationPress` serves the same function after eviction. Unlike the original StreamingLLM paper's online design — where the cache is bounded continuously and a full-length KV cache is never built — our kvpress implementation performs a full prefill before trimming (post-prefill cache eviction, §6.1). This departs from the original design's TTFT advantage but puts Streaming on equal footing with SnapKV and PyramidKV, which are inherently post-prefill methods, so retained tokens' hidden states are computed with access to the complete context.

### 3.2 Setup

**Models.** Llama-3.1-8B [Dubey et al., 2024] and Mistral-7B-v0.3 [Jiang et al., 2023b] (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task. LongBench is the primary evaluation benchmark used by SnapKV [Li et al., 2024], PyramidKV [Cai et al., 2024], and most directly comparable KV compression work, making it the natural choice for ground-truth comparability; we extend it with our faithfulness metrics on the same task set.

**Hyperparameters.** Retention fraction r=0.65. For kvpress methods: `window_size=128` (observation window; kvpress stock default 64), PyramidKV `beta=20`. For the §6.2 custom implementation: `q_buffer_size=128`, `always_keep_first=16`, `always_keep_last=16`.

---

## 4. Ground-Truth Evaluation

Before introducing our faithfulness metrics, we establish what the standard evaluation paradigm reveals. We run both Llama-3.1-8B and Mistral-7B-v0.3 on LongBench v1 (16 tasks, 100 examples each) at 65% retention, comparing naive proportional truncation, SnapKV, StreamingLLM (Streaming), and PyramidKV against the full uncompressed model.

**Setup.** Full details in §3.2. All methods target r = 0.65 retention. Naive truncation keeps the first 6.5% and last 58.5% of tokens as a new self-consistent prompt. PyramidKV uses a layer-adaptive pyramid budget on top of SnapKV scoring. Ground-truth scores use the standard LongBench metrics: F1 for QA tasks, ROUGE for summarization, classification-specific scoring for classification tasks, and edit-distance-based similarity (`code_sim_score`) for code completion. Per-task ground-truth results across compression rates (65/50/35%) are given in Appendix C.

**Llama-3.1-8B.**

| Task             | Full | Naive    | SnapKV    | Streaming | PyramidKV |
| ---------------- | ---- | -------- | --------- | --------- | --------- |
| NarrativeQA†     | 5.5  | 4.9      | 5.6       | **5.8**   | 5.5       |
| Qasper†          | 11.1 | 10.2     | 11.3      | 9.5       | **11.8**  |
| MultifieldQA     | 28.9 | 27.0     | **30.2**  | 26.3      | 29.4      |
| HotpotQA†        | 9.9  | 9.6      | 9.9       | 9.0       | **10.2**  |
| 2WikiMQA†        | 14.1 | 12.6     | **13.9**  | **13.9**  | 13.7      |
| MuSiQue†         | 6.9  | 7.0      | **7.2**   | 5.6       | 7.1       |
| GovReport        | 20.4 | 19.8     | 19.6      | 18.9      | **20.1**  |
| QMSum†           | 10.3 | **11.5** | 9.7       | 9.9       | 9.4       |
| MultiNews        | 19.0 | 16.5     | **18.1**  | 17.7      | 17.9      |
| TREC             | 70.0 | 66.0     | **71.0**  | 68.0      | **71.0**  |
| TriviaQA         | 17.4 | 17.3     | **17.5**  | 17.4      | **17.5**  |
| SAMSum           | 16.0 | 16.5     | 16.2      | **17.1**  | 16.3      |
| PassageCount†    | 3.0  | 1.0      | **3.0**   | **3.0**   | **3.0**   |
| PassageRetrieval | 44.0 | 37.0     | **44.0**  | 37.0      | **44.0**  |
| LCC              | 68.1 | 63.4     | 68.1      | 67.2      | **68.5**  |
| RepoBench-P      | 55.6 | 53.9     | 55.4      | 53.2      | **56.0**  |
| **Average**      | 25.0 | 23.4     | 25.0      | 23.7      | **25.1**  |

Full context is the reference and is not bolded. Bold marks the best compressed method per task. Tasks marked † have full-context scores below 15; results on these tasks are noisier.

**Mistral-7B-v0.3.**

| Task             | Full | Naive    | SnapKV    | Streaming | PyramidKV |
| ---------------- | ---- | -------- | --------- | --------- | --------- |
| NarrativeQA†     | 5.1  | 2.7      | 5.1       | **5.4**   | 4.1       |
| Qasper†          | 5.3  | 4.3      | **5.1**   | 5.0       | 4.8       |
| MultifieldQA     | 25.3 | 22.1     | **24.9**  | 23.2      | 24.8      |
| HotpotQA†        | 10.5 | 10.6     | **10.7**  | 9.8       | 10.6      |
| 2WikiMQA†        | 11.5 | 9.8      | **11.7**  | 11.3      | **11.7**  |
| MuSiQue†         | 5.1  | 3.9      | **5.1**   | 4.4       | **5.1**   |
| GovReport        | 20.0 | 19.7     | 19.8      | 19.4      | **20.0**  |
| QMSum†           | 8.0  | **8.4**  | 7.9       | 8.3       | 7.9       |
| MultiNews        | 18.1 | 17.2     | 17.2      | 17.1      | **17.3**  |
| TREC             | 72.0 | 67.0     | 70.0      | 70.0      | **71.0**  |
| TriviaQA         | 23.1 | 24.2     | 23.4      | **26.5**  | 23.3      |
| SAMSum           | 16.9 | 17.8     | 17.9      | **18.3**  | 18.1      |
| PassageCount†    | 1.0  | **3.0**  | 1.0       | 1.0       | 1.0       |
| PassageRetrieval | 39.0 | 26.0     | **39.0**  | 30.0      | **39.0**  |
| LCC              | 62.9 | 60.2     | **63.2**  | 62.1      | **63.2**  |
| RepoBench-P      | 53.9 | **54.4** | 54.1      | 54.1      | 54.0      |
| **Average**      | 23.6 | 22.0     | **23.5**  | 22.9      | **23.5**  |

The overall pattern is the same across both architectures: ground-truth scores are compressed across methods. On both Llama and Mistral, SnapKV ties PyramidKV (Llama: 25.0 vs 25.1; Mistral: 23.5 vs 23.5), both exceeding naive. Streaming (23.7 Llama, 22.9 Mistral) exceeds naive on both models and is competitive across most tasks, but trails SnapKV and PyramidKV in average. PyramidKV leads or ties all compressed methods on both models.

Taken at face value, PyramidKV is the clear winner. But §5 argues these scores measure the wrong thing — and §8 shows that PyramidKV's ground-truth advantage conceals a faithfulness cost an order of magnitude larger than any other method's.

---

## 5. Faithfulness Metrics

### 5.1 Why Ground-Truth Evaluation Falls Short

Standard benchmarks evaluate compression by running the compressed model on benchmark tasks and scoring its outputs against human-labeled reference answers. The implicit assumption is that a method which scores well on the benchmark is a good approximation of the full model. This assumption fails for a simple reason: the metric never actually compares the compressed model's outputs to the full model's outputs. Two methods can score identically while producing completely different text — one that closely replicates what the full model would have said, and one that does not. Whether the compressed model behaves like the full model is simply not measured. We instead ask that question directly: does the compressed model produce the same output the full, uncompressed model would have produced on the same input?

Ground-truth evaluation has five distinct failure modes as a compression metric.

- **Reference mismatch.** For the four ROUGE-based summarization tasks in LongBench (GovReport, QMSum, MultiNews, SAMSum), ground-truth is computed against a human-written reference. Two methods can generate completely different summaries with identical ROUGE scores, as long as both cover the same facts. Whether one summary resembles what the full model would have written is invisible to the metric. For the seven F1-based QA tasks, ground-truth measures string overlap with a gold answer extracted from the dataset — again with no reference to the full model's actual behavior. Only for classification and retrieval tasks does ground truth directly constrain the model's output to match specific tokens.

- **Faithful errors are penalized.** If the full-context model produces a wrong answer, a compressed model that faithfully reproduces that same wrong answer scores 0 on ground truth. The metric penalizes faithfulness when it produces the wrong answer for the "right reasons." A compression method should not be penalized for being a good approximation.

- **Accidental correctness is rewarded.** Conversely, a compressed model that accidentally produces the correct answer — because truncation happened to leave in the relevant span, or because a different reasoning path led to the same answer — scores the same as a method that genuinely approximates the full model. Ground truth cannot distinguish these cases.

- **Most of these metrics are not testing correctness at all.** Of the 16 tasks, only three (TREC, PassageCount, PassageRetrieval) score a prediction as binary right or wrong. The other thirteen — seven QA tasks scored by word-overlap F1, four summarization tasks scored by ROUGE-L, two code-completion tasks scored by string similarity — give continuous partial credit for overlap with one human-written reference, with no notion of "correct" or "incorrect" built into the metric itself. A summary that captures the source faithfully but phrases it differently than the dataset's one reference summary, or a QA answer that conveys the right information in different words, can score below an answer that shares more surface vocabulary with the reference while being less accurate. This does not mean the metric carries no signal — overlap with a reasonable reference is correlated with quality — but for most of this benchmark, "ground-truth accuracy" is closer to similarity to one human's phrasing than to a judgment of whether the model got the answer right.

- **Low full-model accuracy amplifies the first three problems.** The base models used here (Llama-3.1-8B and Mistral-7B-v0.3) achieve macro-average GT scores of approximately 25 out of 100 on LongBench. The instruction-tuned variants used by the most directly comparable compression papers score higher — PyramidKV reports full-context baselines of 41.46 for LLaMA-3-8B-Instruct and 39.76 for Mistral-7B-Instruct [Cai et al., 2024] — but even at a baseline of 40, roughly 60% of what the model generates does not overlap with the ground-truth reference. For that majority of output, the metric is entirely blind to whether compression changed the model's behavior: the compressed and uncompressed models could produce identical text or completely different text, and the GT score would be the same either way.

  On six of our 16 tasks the full model's ground-truth accuracy is below 15 (NarrativeQA: 5.5, MuSiQue: 6.9, PassageCount: 3.0, HotpotQA: 9.9, Qasper: 11.1, QMSum: 10.3). On NarrativeQA, for example, the full model gets 94.5% of examples wrong. Moreover, even the 5.5% that the full model answers correctly need not be the same 5.5% a compressed method gets right — two methods with identical aggregate scores may agree on no individual examples. The score carries no information about which questions were answered correctly, only that the same number were.

  A perfectly faithful compressed method should also get 94.5% wrong; ground truth scores all of those as 0. The compressed method's GT score on that task then reflects only how close its wrong answers happen to come to the gold reference — a function of failure-mode similarity to the reference, not of approximation quality. As a consequence, GT differences between compression methods on low-accuracy tasks are dominated by noise: small random variations in which wrong answers partially overlap with the gold string. The signal about actual compression quality is vanishingly small precisely on the tasks where the method has the most work to do. The problem compounds in the other direction too: a compressed method that outscores the full model would be reported as an improvement, but from a faithfulness perspective it is a failure — the compressed model is producing different outputs than the full model on examples where the full model was wrong, and those differences happen to align with the reference. Ground truth rewards divergence from the full model when that divergence is accidentally correct.


These failure modes matter in practice. PyramidKV's ground-truth advantage (§4) is exactly the kind of result that is hard to interpret: it may reflect genuine preservation of task-relevant information, or it may reflect a combination of structural properties that happen to produce correct first tokens without approximating the full model's behavior. §8 shows it is the latter.

This points to a more general limitation, beyond the failure modes above. Ground truth — or any outcome metric computed on a fixed test distribution, whether benchmark accuracy or a downstream business metric — measures *that* a method produced acceptable outputs on the examples tested, not *why*. That distinction matters because it determines whether a result generalizes. An outcome metric tells us nothing about behavior on inputs outside the test distribution; it is silent on mechanism by construction. 

PyramidKV and SnapKV are the clearest illustrations in this paper: by ground truth and output faithfulness both lead all tested methods, but §8 shows this is substantially an artifact of how post-prefill eviction methods compute their answer — a full, uncompressed prefill makes the first generated token identical to the full model's by construction, which dominates the aggregate score because the benchmark is full of short-answer tasks where the first token is the answer. A practitioner who only had the outcome metric would not know this, and would have no way to predict whether the advantage holds for a task with longer outputs, a different query distribution, or any input not resembling the test set — and indeed it mostly does not (Table 5, §8.1).

### 5.2 KL Faithfulness

We evaluate compression fidelity using KL divergence between the full and compressed models' next-token distributions, measured on a shared generation sequence. The uncompressed model generates a response y* = (y*₁,...,y*_L) from the full context. Both the full model and each compressed model are then teacher-forced on this shared token sequence: at step *t*, the full model conditions on [*c*, *q*, y*₁,...,y*_{t-1}] and the compressed model conditions on [*ĉ*, *q*, y*₁,...,y*_{t-1}]. The metric averages KL divergence across all generation steps:

```
F_KL = (1/L) Σ_{t=1}^{L} KL(P_full(· | c, q, y*_{<t}) ‖ P_comp(· | ĉ, q, y*_{<t}))
```

Lower is better; 0 means identical distributions at every step. Conditioning all methods on the same shared token sequence eliminates path-dependence: a method whose compressed context causes early divergence would otherwise be evaluated on a different downstream sequence than a more faithful method, making scores incomparable across methods. Using y* as the shared prefix anchors every comparison to the full model's actual behavior.

This metric captures distributional agreement at the token probability level, not just which token is selected by greedy decoding: a method that shifts probability mass from the correct token to semantically similar alternatives — invisible to both greedy accuracy measures and ground-truth benchmarks — registers as a faithfulness cost in F_KL. KL faithfulness does not share the limitations documented in §5.1: because it measures whether the compressed model's actual computation matches the full model's at every step, it is informative about mechanism, not just outcome. That distinction is what lets a result generalize to inputs that were never tested, and it is what makes KL faithfulness sensitive to structural corruption that GT evaluation cannot detect — as §8 demonstrates for SnapKV and PyramidKV.

### 5.3 Output Faithfulness

KL faithfulness measures how well the compressed model's internal distributions match the full model's. A complementary question is: how similar is the text the compressed model actually generates?

We define **Output Faithfulness** (F_out) as:

```
F_out = (1/N) Σ_{i=1}^{N} f1(pred_full_i, pred_comp_i)
```

where f1 is word-level F1 between the full model's prediction and the compressed model's prediction on example i. Both predictions are generated autoregressively with greedy decoding; no external ground-truth reference is used.

Word-level F1 is computed after lowercasing, removing articles (a, an, the), and stripping punctuation — the same normalization used by LongBench for extractive QA evaluation. The metric ranges from 0 (no word overlap) to 1 (identical output text).

F_out addresses the failure modes of ground-truth evaluation directly: it does not compare to any external reference, so faithful errors are not penalized and accidental correctness is not rewarded. It also provides partial credit in two cases that binary accuracy metrics miss: when both models produce the same wrong answer (they are being equally and faithfully wrong), and when one model's output partially overlaps the other's (approximation with some drift). For a compression method that is a good approximation, F_out should be high regardless of whether the underlying outputs are "correct."

F_KL and F_out are measuring different things:
- F_KL measures **distributional faithfulness**: how similar are the two models' probability distributions over the vocabulary at every generation step?
- F_out measures **behavioral faithfulness**: how similar are the two models' final generated texts?

These can disagree. A method could have low F_KL (distributions match well at each step) but low F_out (due to early divergence in sampled text). Conversely, a method could have high F_out (same final answers) but high F_KL (reached those answers via corrupted intermediate distributions). §8 shows that PyramidKV falls into this second category, making both metrics jointly necessary to characterize it.

**Why these two metrics.** Other distributional or behavioral metrics are possible — Jensen-Shannon divergence (bounded and symmetric, but without F_KL's direct expected-surprise interpretation), rank correlation between full and compressed top-k token rankings (sensitive to ordering, blind to probability mass), logit cosine similarity (no information-theoretic meaning, sensitive to logit scale), or calibration error against ground truth (reintroduces the external reference that both metrics are designed to avoid).

We use KL faithfulness because it has a direct information-theoretic interpretation — the expected excess surprise from using the compressed model's distribution in place of the full model's — and because it decomposes cleanly per generation step, which is what makes the per-step analysis in §8 possible.

We use word-level F1 because it requires no reference distribution at all, only the two models' realized text, and reuses the scoring function LongBench already applies to extractive QA, keeping faithfulness evaluation on the same footing as the ground-truth evaluation it is meant to improve on. Neither choice is the only defensible one; we expect findings that hold under KL faithfulness and F_out to hold in substance under JS divergence or a comparably motivated behavioral metric, since each is a different lens on one of the same two questions — does the distribution match, does the output match — not a different question.

**Caveat emptor on output-text metrics.** F_out is more principled than ground-truth evaluation, but word-level F1 introduces its own distortions that any output-text measurement inherits. Readers should treat F_out scores, and any generated output-based measurement, with appropriate skepticism:

- *Length bias.* Longer generated outputs have a higher statistical probability of token overlap with the reference by chance alone. A method that produces more verbose answers will score higher on recall independently of faithfulness, and F1's harmonic mean does not fully correct for this.
- *Paraphrase blindness.* Word-level F1 treats synonyms and legitimate paraphrases as mismatches. Two outputs that express the same content in different words score lower than two outputs that repeat the same words with different meaning.
- *Ceiling and floor effects.* On short-answer tasks, any method that produces the correct key term scores near 1.0; on long-form generation, divergence compounds over tokens and scores approach 0.0 for all methods. Neither extreme is discriminative.
- *Right answer, wrong reason.* A compression method that corrupts most of the context can still reproduce the full model's short output on factoid tasks where the answer is recoverable from a fragment.
- *Sample variance.* At n=100, confidence intervals on F_out are wide enough that small differences in aggregate scores should not be over-interpreted.

These limitations are why KL faithfulness is the primary metric used in this paper. KL operates at the distribution level on every generation step, is insensitive to output length, and requires no string matching. F_out is included as a behavioral complement — a check that distributional faithfulness corresponds to observable output similarity — rather than as independent evidence.

### 5.4 Naive Truncation as a Baseline

A key empirical finding motivating this work is that naïve proportional truncation substantially outperforms attention-score-based KV cache pruning (SnapKV, PyramidKV) on KL faithfulness. The method retains 65% of the prompt tokens, split as a 10%/90% head/tail budget: the first 6.5% of tokens (literal prompt prefix) and the last 58.5% of tokens (recency tail), concatenated into a new self-consistent prompt. The middle portion is discarded. On NarrativeQA, naive truncation achieves KL 0.022 vs. 0.787 for SnapKV. This result was surprising and prompted the structural corruption investigation described in §6.

### 5.5 Results at 65% Retention

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. KL faithfulness compares compressed model distributions to full model distributions, teacher-forced on y*. F_out compares the compressed model's independently generated outputs to y* using word-level F1.

**KL Faithfulness at 65%.** The KL results below compare all methods on Llama-3.1-8B; the Mistral results follow the same pattern at roughly half the absolute magnitude.

| Task             | Naive          | Streaming      | SnapKV         | PyramidKV |
| ---------------- | -------------- | -------------- | -------------- | --------- |
| NarrativeQA†     | **0.022**      | 0.045          | 0.787          | 1.338     |
| Qasper†          | 0.281          | **0.237**      | 0.881          | 1.229     |
| MultifieldQA     | **0.262**      | 0.302          | 1.316          | 1.535     |
| HotpotQA†        | 0.203          | **0.069**      | 1.935          | 2.181     |
| 2WikiMQA†        | 0.221          | **0.103**      | 1.714          | 2.422     |
| MuSiQue†         | 0.197          | **0.098**      | 1.747          | 2.523     |
| GovReport        | **0.280**      | 0.385          | 0.532          | 0.334     |
| QMSum†           | **0.081**      | 0.093          | 0.393          | 0.438     |
| MultiNews        | **0.585**      | 0.643          | 1.050          | 0.721     |
| TREC             | 0.202          | **0.084**      | 1.454          | 1.526     |
| TriviaQA         | 0.076          | **0.008**      | 1.151          | 1.647     |
| SAMSum           | **0.011**      | 0.015          | 0.791          | 1.182     |
| PassageCount†    | 0.059          | **0.002**      | 1.560          | 1.626     |
| PassageRetrieval | 0.188          | **0.032**      | 1.624          | 1.416     |
| LCC              | **0.076**      | 0.085          | 0.881          | 1.142     |
| RepoBench-P      | **0.048**      | 0.050          | 0.702          | 1.052     |
| **Average**      | 0.175          | **0.141**      | 1.157          | 1.394     |

Values in nats; lower is better. Bold marks the best F_KL per row (SnapKV and PyramidKV excluded from bold competition — both are substantially worse). Streaming leads on 8 of 16 tasks and the overall average; Naive leads on 8. SnapKV and PyramidKV are an order of magnitude worse than either Naive or Streaming.

**Output Faithfulness (F_out) at 65%.** The same methods evaluated on the new metric, comparing each model's output text against the full model's output text.

| Task             | Naive    | Streaming | SnapKV   | PyramidKV |
| ---------------- | -------- | --------- | -------- | --------- |
| NarrativeQA†     | **88.1** | 59.9      | 63.1     | 61.7      |
| Qasper†          | 46.4     | 56.9      | 71.4     | **72.6**  |
| MultifieldQA     | 57.3     | 63.6      | 81.0     | **86.0**  |
| HotpotQA†        | 88.2     | 72.6      | **95.3** | 92.2      |
| 2WikiMQA†        | 64.5     | 77.9      | **95.6** | **95.6**  |
| MuSiQue†         | **97.1** | 72.3      | 92.7     | 92.3      |
| GovReport        | **55.8** | 40.1      | 47.8     | 51.2      |
| QMSum†           | **54.4** | 37.0      | 41.7     | 40.4      |
| MultiNews        | 43.8     | 41.0      | 44.6     | **49.0**  |
| TREC             | 75.3     | 74.0      | **82.6** | 81.3      |
| TriviaQA         | 71.1     | 60.2      | 92.0     | **97.0**  |
| SAMSum           | 64.9     | 52.6      | **78.4** | 73.4      |
| PassageCount†    | 70.7     | 61.8      | 92.8     | **96.4**  |
| PassageRetrieval | 80.1     | 70.0      | **94.0** | 93.8      |
| LCC              | 62.0     | 73.8      | **92.8** | 91.3      |
| RepoBench-P      | 77.3     | 65.6      | 90.5     | **90.6**  |
| **Average**      | 68.6     | 61.2      | 78.5     | **79.1**  |

Values in %; higher is better. Bold marks the highest value per row.

**The two metrics sharply disagree on SnapKV and PyramidKV.** On KL faithfulness, Streaming (0.141) edges Naive (0.175) on average, while SnapKV (1.157) and PyramidKV (1.394) are an order of magnitude worse. On F_out, PyramidKV (79.1%) and SnapKV (78.5%) are the two leading methods — effectively tied and 10+ points above naive (68.6%) — while Streaming (61.2%) sits below naive. These reversals are not contradictory: §8 explains the mechanism that decouples F_out from KL for post-prefill eviction methods.

The Mistral results at 65% show the same pattern.

**KL Faithfulness.**

| Task             | Naive          | Streaming      | SnapKV         | PyramidKV |
| ---------------- | -------------- | -------------- | -------------- | --------- |
| NarrativeQA†     | **0.003**      | 0.125          | 0.445          | 0.780     |
| Qasper†          | 0.071          | **0.067**      | 0.274          | 0.472     |
| MultifieldQA     | **0.148**      | 0.201          | 0.639          | 0.718     |
| HotpotQA†        | **0.195**      | 0.198          | 0.714          | 0.725     |
| 2WikiMQA†        | 0.262          | **0.208**      | 0.710          | 0.683     |
| MuSiQue†         | 0.183          | **0.179**      | 0.708          | 0.679     |
| GovReport        | **0.180**      | 0.281          | 0.578          | 0.294     |
| QMSum†           | **0.067**      | 0.082          | 0.201          | 0.183     |
| MultiNews        | **0.586**      | 0.678          | 1.076          | 0.626     |
| TREC             | **0.012**      | 0.046          | 0.359          | 0.703     |
| TriviaQA         | **0.039**      | 0.070          | 1.104          | 1.379     |
| SAMSum           | **0.021**      | 0.040          | 1.030          | 1.050     |
| PassageCount†    | **0.037**      | 0.069          | 1.141          | 0.897     |
| PassageRetrieval | **0.195**      | 0.204          | 0.971          | 0.541     |
| LCC              | 0.055          | **0.041**      | 0.861          | 1.070     |
| RepoBench-P      | **0.076**      | 0.087          | 0.914          | 0.940     |
| **Average**      | **0.133**      | 0.161          | 0.733          | 0.734     |

Values in nats; lower is better. Bold marks the best value per row. Naive leads on 12 of 16 tasks; Streaming leads on 4. SnapKV (0.733) and PyramidKV (0.734) are similarly poor, both 5.5× worse than Naive (0.133).

**Output Faithfulness (F_out).**

| Task             | Naive    | Streaming | SnapKV   | PyramidKV |
| ---------------- | -------- | --------- | -------- | --------- |
| NarrativeQA†     | 40.5     | 38.2      | **57.4** | 55.4      |
| Qasper†          | 69.0     | 69.4      | **87.1** | 86.1      |
| MultifieldQA     | 63.9     | 66.8      | **90.3** | 89.3      |
| HotpotQA†        | 66.9     | 63.9      | 94.7     | **94.8**  |
| 2WikiMQA†        | 66.8     | 77.0      | **95.2** | 94.8      |
| MuSiQue†         | 64.6     | 68.0      | **94.3** | 93.3      |
| GovReport        | 46.7     | 41.8      | 62.6     | **64.2**  |
| QMSum†           | 48.3     | 47.0      | 72.2     | **73.8**  |
| MultiNews        | 45.5     | 35.8      | **51.7** | 51.6      |
| TREC             | 85.0     | 82.2      | **94.5** | 92.6      |
| TriviaQA         | 57.4     | 60.6      | 91.9     | **93.8**  |
| SAMSum           | 55.1     | 57.2      | **72.4** | 69.6      |
| PassageCount†    | 57.3     | 57.7      | 97.0     | **97.5**  |
| PassageRetrieval | 68.8     | 68.5      | 97.2     | **97.8**  |
| LCC              | 62.1     | 76.0      | 88.3     | **90.8**  |
| RepoBench-P      | 62.8     | 66.1      | **93.0** | 90.1      |
| **Average**      | 60.0     | 61.0      | **83.7** | 83.5      |

Values in %; higher is better. Bold marks the highest value per row.

The pattern is consistent across architectures. On Mistral, SnapKV (83.7%) and PyramidKV (83.5%) are effectively tied on F_out — the same reversal as Llama relative to KL faithfulness — while Streaming (61.0%) and Naive (60.0%) cluster together, confirming the KV-pruning F_out advantage holds across both models.

---

## 6. Structural Corruption in KV Cache Pruning

### 6.1 The Mechanism

All three KV-pruning methods benchmarked in this paper — SnapKV, PyramidKV, and Streaming-rerotated — use *post-prefill cache eviction*: the model performs one normal, unrestricted prefill pass over the complete prompt, attending to the full context at every layer, and only after that pass completes is the KV cache trimmed before the first decode step. Retained keys and values are computed cleanly under full attention; nothing during prefill is modified. The earliest a gap can affect anything is at T=1 — the first decode step, when the newly generated query attends back over the now-sparse cache. §8's case study shows the empirical signature: KL(T=0) = 0, a clean prefill, followed by a sharp spike at T=1.

**Prefill enrichment.** Post-prefill eviction has a second consequence beyond gap timing: the KV pairs that survive carry representations computed under full-context attention over the entire original sequence, including attention to positions that are immediately evicted once the forward pass completes. A token retained in a post-prefill-evicted cache has attended to every other token in the document at every layer; a token retained in a compressed prompt has never attended to the evicted content at any layer. Prompt construction, such as the Naïve method benchmarked here, retains the algorithmically selected tokens in a structurally clean input; post-prefill eviction retains the tokens in a structurally gapped input but with richer representations of each. The enrichment benefit accrues equally to all three post-prefill methods regardless of their selection policies. Whether it outweighs the decode-time gap cost depends on how many gaps remain and how they are distributed — examined in §6.4.

**Gap corruption.** Once a decode-time query attends over the sparse cache, the degenerate attention distribution does not stay local. Transformer hidden states are deeply entangled across layers: layer *l*'s computation at every position depends on the full layer *l*-1 output across all positions. A corrupted hidden state is carried forward into subsequent decode steps and, after *T* further steps, has diffused throughout the representation. This is not primarily an *early-token* problem. It is a *gap* problem: any portion of the retained set where positions are sparse will produce corrupted attention whenever a decode query crosses it. Recency-biased selection concentrates the budget at the tail, leaving the head and middle with severe gaps; attention-score-based selection distributes retained positions more evenly across the full sequence, leaving gaps throughout. The mechanism is identical in both cases; §6.4 examines how gap density and distribution determine whether enrichment or gap damage dominates.

**Positional misalignment.** KV pruning introduces a second structural problem independent of causal gaps: retained tokens keep their original RoPE encodings, so two tokens logically adjacent in the retained set may carry a relative distance of hundreds or thousands of positions in the original prompt. The attention mechanism was trained on sequences where positional distance reflects informational proximity; retained tokens with large RoPE gaps between them are out-of-distribution at every decode step. Key re-rotation (`KeyRerotationPress`) addresses positional misalignment by re-encoding retained keys at compact positions 0…M−1 after eviction, leaving the gap structure and enriched value representations unchanged. Prompt construction eliminates positional misalignment together with causal gaps by creating a structurally self-consistent new sequence. §6.4 applies key re-rotation to all three benchmarked methods — SnapKV, Streaming, and PyramidKV — to determine whether positional displacement is the dominant failure mode in each case.

### 6.2 Empirical Evidence

To confirm that mechanism rather than token selection drives the faithfulness gap, we compare SnapKV with KV pruning to SnapKV-Select — identical attention-score-based token selection, presented as a new prompt rather than patched into the KV cache (Table 1). The same tokens, selected by the same algorithm, behave very differently depending on how they are presented to the model. Table 1 also includes a third SnapKV variant labeled *recompute-over-gaps*: rather than running a full unrestricted prefill and then pruning the cache afterward, the attention mask is restricted during prefill so the model only ever attends to the selected positions — evicted tokens are invisible from the start. There is no post-prefill eviction and no enrichment from evicted tokens — the selected tokens never attended to the evicted context during prefill. Critically, however, the selected tokens retain their original pre-eviction RoPE positions: RoPE is applied to all T tokens at their original positions before the attention mask filters to the selected set, so the keys carry scattered positions rather than compact ones. This makes recompute-over-gaps distinct from prompt construction: both lack enrichment, but prompt construction re-indexes the selected tokens to compact positions 0..M-1, whereas recompute-over-gaps preserves the original scattered positions.

**Table 1. Mean KL faithfulness for four method configurations. Llama-3.1-8B, 65% retention, 6 tasks, n = 100 per task. All three SnapKV rows use identical attention-score-based token selection; only the mechanism varies.**

| Method | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean |
|---|---|---|---|---|---|---|---|
| Naive — prompt construction | 0.221 | 0.262 | 0.281 | 0.081 | **0.048** | 0.076 | 0.162 |
| SnapKV (pcfg) — KV pruning (recompute-over-gaps) | 0.254 | 0.186 | 0.223 | 0.117 | 0.106 | 0.122 | 0.168 |
| SnapKV — KV pruning (post-prefill eviction) | 1.714 | 1.316 | 0.881 | 0.393 | 0.702 | 1.151 | 1.026 |
| SnapKV-Select — prompt construction | **0.096** | **0.118** | **0.163** | **0.071** | 0.051 | **0.060** | **0.093** |

SnapKV-Select (0.093) is 11× better than SnapKV (1.026) using the same selected tokens — the damage is caused by the mechanism, not the token selection. The recompute-over-gaps row separates the two factors that post-prefill eviction combines: it eliminates enrichment (evicted tokens are never seen during prefill) but preserves scattered positions (RoPE is applied before the attention mask filters to the selected set). Its score (0.168) nearly matches naive prompt construction (0.162) — positional scatter alone causes limited damage. Adding enrichment back (SnapKV post-prefill, 1.026) makes things dramatically worse, not better: the full-prefill hidden state appears to be detracting rather than helping here. §6.4 examines this more fully and introduces a correction.

### 6.3 Synthetic Gap-Structure Analysis

To isolate positional displacement from enrichment effects without contamination from token selection, we evaluate four pure position geometries generated purely as a function of sequence length and retention fraction (r = 0.65), with no query, no content, and no model-based selection. At a typical context length of ~4000 tokens, the geometries are:

- **block_end** — one contiguous retained block at the tail (1 causal gap, immediately before it)
- **block_mid** — one contiguous retained block, centered (2 gaps, before and after)
- **clustered4** — four evenly-spaced contiguous blocks (~5 gaps)
- **scattered** — evenly-spaced single-token positions (~1400 gaps — the maximally scattered case)

Each geometry evaluates the same retained position set under four presentations:

- **Gapless**: positions gathered into a new, shorter, contiguous prompt — compact RoPE positions, no enrichment from evicted context, no KV patching.
- **Gapped**: attention mask restricted to the selected positions during prefill — evicted tokens are invisible, so no enrichment, but retained keys carry their original RoPE positions. This is the approach used by recompute-over-gaps in §6.2.
- **Evicted**: full prefill over the complete prompt (enabling full-context enrichment), then post-prefill eviction; retained keys remain at their original RoPE positions. This is the mechanism used by SnapKV.
- **Rerotated**: identical to evicted but with key re-rotation to compact RoPE positions after eviction (`KeyRerotationPress`). This is the mechanism used by StreamingLLM.

Comparing gapped to gapless isolates positional scatter alone, with no enrichment in either condition. Comparing evicted to gapped isolates the effect of enrichment at scattered positions. Comparing rerotated to evicted isolates position correction alone, since enrichment is identical in both conditions.  Finally, comparing gapless to rerotated isolates the effects of enrichment.

**Table 2. Mean KL faithfulness by gap geometry (Llama-3.1-8B, r = 0.65, 6 tasks, n = 20/task).**

| Geometry | Gaps | Gapless | Gapped | Evicted | Rerotated | gp/gl | ev/gl | rot/gl |
|---|---|---|---|---|---|---|---|---|
| block_end | 1 | 0.177 | 2.373 | 3.493 | 2.814 | 13.4× | 19.8× | 15.9× |
| block_mid | 2 | 3.127 | 4.592 | 4.225 | 3.973 | 1.5× | 1.4× | 1.3× |
| clustered4 | ~5 | 2.745 | 4.088 | 3.765 | 3.498 | 1.5× | 1.4× | 1.3× |
| scattered | ~1400 | 1.205 | 2.886 | 1.289 | 0.221 | 2.4× | 1.1× | 0.17× |

Table 2 reveals several non-obvious patterns across the four geometries.

- **Scattered geometry: position correction is the dominant factor.** Positional scatter without enrichment causes moderate damage: gapped (2.886) is 2.4× worse than gapless (1.205). Enrichment from the full prefill substantially offsets this — evicted (1.289) recovers most of the scatter cost and lands nearly even with gapless (1.1×). Re-rotation then eliminates the residual positional displacement: rerotated (0.221) is 0.17× of evicted and 0.18× of gapless. The full-prefill hidden state is clearly beneficial here — with positions corrected, it reaches the best result in the table, well below even the gapless baseline.
- **Block_end geometry: positional shift and structural corruption both contribute.** Even without enrichment, the gapped presentation (2.373) is 13.4× worse than gapless (0.177) — a substantial gap that is perhaps surprising given that the retained block has only one causal gap and is otherwise contiguous. Enrichment increases the damage further: evicted (3.493, 19.8×) exceeds gapped (2.373, 13.4×). Most notably, re-rotation — which fully recovers the scattered geometry — provides only a modest improvement here: rerotated (2.814, 15.9×) does not recover to the no-enrichment gapped baseline, let alone to gapless. What drives this irreversibility is not established here; several factors plausibly contribute, and we do not attempt to single one out.
- **Block_mid and clustered4** show modest damage in both gapped (1.5×), evicted (1.4×), and rerotated (1.3×) relative to gapless. The geometries are not directly comparable in absolute KL faithfulness: block_mid and clustered4 discard the recency tail, so their elevated gapless baselines reflect lower content informativeness, not mechanism damage. Within-geometry ratios are the interpretable signal.
- **Enrichment direction depends on the nature of the scatter.** The gapped column enables a direct comparison with Table 1's recompute-over-gaps result, but the two tell opposite stories about enrichment. In the synthetic scattered geometry here, enrichment appears beneficial at scattered positions: evicted (1.289) is better than gapped (2.886). In Table 1, SnapKV's attention-score-selected tokens show the reverse: recompute-over-gaps (no enrichment, 0.168) far outperforms SnapKV post-prefill (enrichment, 1.026). The most obvious difference between the two experiments is the nature of the position sets: Table 2's scattered positions are evenly spaced across the sequence, while SnapKV's are selected by attention score and therefore biased toward tokens the model heavily attended to during the full-context prefill. Whether and how this distinction drives the divergent enrichment outcomes is not fully clear, but the reversal suggests that the effect of enrichment under scatter is not a simple property of scatter alone.

These results make a concrete prediction for SnapKV and StreamingLLM, whose retained-set geometries are approximately scattered and block_end respectively: SnapKV should benefit enormously from re-rotation. StreamingLLM, with sink tokens plus a recency tail separated by a single large gap, should see a meaningfully smaller gain — a single gap creates less positional displacement than ~1400 scattered gaps, and structural corruption from attending to the evicted middle during prefill may further limit what position correction alone can recover. §6.4 tests both predictions.

### 6.4 Re-rotation Confirms Positional Displacement as the Dominant Failure Mode

**SnapKV.** SnapKV's retained positions are globally scattered — an approximate realization of the scattered geometry in Table 2 — so the synthetic analysis estimates roughly a 6× improvement from re-rotation. The actual result substantially exceeds this estimate: applying `KeyRerotationPress` after post-prefill eviction reduces mean KL from 1.157 to 0.043 at 65% retention — a 27× improvement across all 16 LongBench tasks on Llama-3.1-8B. The scattered retained set that was the worst-performing KV configuration in §6.3 becomes the best-performing method overall once positions are corrected. Positional displacement was not a secondary cost for SnapKV; it was the dominant failure mode. Multi-rate results and improvement ratios are in §7.2.

**Streaming.** The prediction from §6.3 is confirmed. Re-rotation produces a substantial 11.7× improvement at 65% (unrotated kvpress streaming 1.639 → 0.141), well below SnapKV's 27× but consistent with streaming's single-large-gap geometry producing less positional displacement than SnapKV's ~1400 scattered gaps. The remaining gap between streaming_rerotated (0.141) and SnapKV+rot (0.043) reflects structural corruption that re-rotation cannot address: streaming's recency tail attended to the large evicted middle section during prefill, and those value representations persist regardless of key position reassignment. Multi-rate comparison is in §7.2.

**PyramidKV.** PyramidKV's per-layer budget allocates from ≈99% at layer 0 down to ≈31% at layer 31 for 65% overall retention. The naive adaptation of `KeyRerotationPress` to PyramidKV maps each layer's retained keys to `[0, n_kept)`. Because `n_kept` differs per layer under the pyramid budget, no single decode position is correct for all layers simultaneously: a query at layer-0's M (≈98% of T) is out-of-distribution for layer 31 (M ≈ 31% of T), producing catastrophic degenerate output on long-form tasks. The fix is right-aligned re-rotation: target `[T − n_kept, T)` so all layers share endpoint T−1, and the decode query arrives at position T for every layer. With this fix in place, Pyr+rot (0.047) tracks SnapKV+rot (0.043) almost exactly at 65%, confirming that positional displacement is the dominant failure mode for PyramidKV as well. Multi-rate analysis is in §7.2.



---

## 7. Main Experiments

### 7.1 Setup

**Models.** Llama-3.1-8B (base, fp16) and Mistral-7B-v0.3 (base, fp16).

**Benchmark.** LongBench v1, 16 English tasks, 100 examples per task.

**Methods compared.**

| Method | Type | Description |
|---|---|---|
| Full context | Reference | Uncompressed model |
| Naive | Prompt construction | Last 65% of tokens, 10% head / 90% tail split |
| SnapKV | KV pruning | Pooled attention weights over observation window |
| SnapKV+rot | KV pruning | SnapKV + KeyRerotationPress (compact RoPE after eviction) |
| Streaming | KV pruning | Attention sinks + recency window with key re-rotation |
| PyramidKV | KV pruning | Layer-adaptive pyramid budget allocation |
| Pyr+rot | KV pruning | PyramidKV + key re-rotation to compact positions after eviction |


### 7.2 KL Faithfulness: Main Results

**Table 3. Mean KL Faithfulness by compression rate (lower is better). Streaming = StreamingLLM with key re-rotation; SnapKV = SnapKV (kvpress post-prefill eviction); SnapKV+rot = SnapKV with key re-rotation to compact positions; Pyr+rot = PyramidKV with key re-rotation; Pyr = PyramidKV. Bold = best per row. Per-task breakdown in Appendix A.**

**Llama-3.1-8B.**

| Retention | Naive | Streaming | SnapKV | SnapKV+rot | Pyr | Pyr+rot |
|---|---|---|---|---|---|---|
| 65% | 0.175 | 0.141 | 1.157 | **0.043** | 1.394 | 0.047 |
| 50% | 0.217 | 0.199 | 0.981 | **0.077** | 1.451 | 0.096 |
| 35% | 0.281 | 0.261 | 0.836 | **0.124** | 0.837 | **0.124** |

**Mistral-7B-v0.3.**

| Retention | Naive | Streaming | SnapKV | SnapKV+rot | Pyr   | Pyr+rot |
|-----------|-------|-----------|--------|------------|-------|---------|
| 65%       | 0.133 | 0.161     | 0.733  | **0.030**  | 0.734 | 0.034   |
| 50%       | 0.158 | 0.201     | 0.619  | **0.058**  | 0.796 | 0.079   |
| 35%       | 0.188 | 0.244     | 0.588  | **0.096**  | 0.586 | **0.096** |

SnapKV+rot leads at every rate on both models. The improvement ratios from re-rotation (27×/13×/6.7× at 65/50/35% on Llama; similar on Mistral) decrease at tighter budgets because two effects push in opposite directions: unrotated SnapKV KL falls as compression increases — fewer retained positions means fewer misaligned RoPE encodings — while SnapKV+rot KL rises as information loss from tighter eviction accumulates. Even at its best (35%), unrotated SnapKV remains 6.7× worse than SnapKV+rot at the same budget.

PyramidKV (unrotated) shows a similar counterintuitive cross-rate trajectory: KL rises at 50% before dropping sharply at 35%, when the `min_num` clamp trips the fallback and forces a uniform per-layer budget — PyramidKV reverting to plain SnapKV-style allocation. Pyr+rot ties SnapKV+rot exactly at 35% on all 16 tasks for both models — once the budget is uniform per layer, re-rotation produces identical positional structure regardless of which specific tokens were selected. Unrotated Pyr tracks SnapKV closely but does not tie exactly: without re-rotation, the specific positions of retained tokens matter, and the `min_num` clamp produces near-uniform but not perfectly uniform allocation across the full range of sequence lengths in the benchmark.

Pyr+rot tracks SnapKV+rot closely at 65% and ties at 35%. A gap opens at 50% — Pyr+rot is 25% above SnapKV+rot on both models — because the pyramid formula reaches its `window_size=64` floor at the top layer, leaving genuine information loss that re-rotation cannot recover. The 50% budget also produces the largest cross-layer positional disparity under right-aligned re-rotation: n_kept spans from ≈0.66T at layer 0 to 64 at layer 31, so the target range [T − n_kept, T) varies widely across layers.

Unlike Llama, Streaming trails Naive at every rate on Mistral — a difference also visible in F_out (§7.4).

**Re-rotation improvement ratios for Streaming.** Table 3 shows streaming_rerotated as the recommended method. For comparison, the unrotated kvpress StreamingLLMPress baseline and the improvement from re-rotation across rates (Llama):

| Rate | Streaming (unrotated) | Streaming+rot | Improvement |
|---|---|---|---|
| 65% | 1.639 | 0.141 | 11.7× |
| 50% | 1.122 | 0.199 | 5.6× |
| 35% | 0.953 | 0.261 | 3.7× |

The improvement ratio decreases at tighter budgets for the same reason as SnapKV — the unrotated baseline improves (fewer misaligned positions) while the rerotated method worsens (information loss). The improvement at every rate is substantially below SnapKV's (27×/13×/6.7×), consistent with streaming's single-large-gap geometry producing less positional displacement than SnapKV's scattered positions.

### 7.3 Output Faithfulness Results

Output Faithfulness (F_out) measures how similar the compressed model's generated text is to the full model's generated text (§5.3). Higher is better. All comparisons are within-model: Llama compressed methods are compared to Llama full context; Mistral compressed methods are compared to Mistral full context.

**Table 4. Mean F_out by compression rate (higher is better). Bold = highest value per row. SnapKV+rot = SnapKV with key re-rotation; Pyr+rot = PyramidKV with key re-rotation. Per-task breakdown in Appendix B.**

**Llama-3.1-8B.**

| Retention | Naive | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|-----------|-------|-----------|----------|------------|----------|---------|
| 65%       | 68.6  | 61.2      | 78.5     | 71.1       | **79.1** | 71.6    |
| 50%       | 52.1  | 56.4      | **73.2** | 67.8       | 64.2     | 61.3    |
| 35%       | 46.9  | 51.8      | 66.8     | 63.0       | **66.9** | 63.0    |

**Mistral-7B-v0.3.**

| Retention | Naive | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|-----------|-------|-----------|----------|------------|----------|---------|
| 65%       | 60.0  | 61.0      | **83.7** | 78.2       | 83.5     | 77.0    |
| 50%       | 56.9  | 57.4      | **79.0** | 74.3       | 68.8     | 66.5    |
| 35%       | 53.5  | 53.9      | 74.0     | 70.1       | **74.8** | 70.0    |

**Cross-rate summary.**

SnapKV and PyramidKV lead F_out at all rates without re-rotation on both models, for the same mechanistic reason: a full, uncompressed prefill means the first generated token's distribution is identical to the full model's by construction (§8.1). On Llama, PyramidKV leads at 65% and 35% while SnapKV leads at 50%; on Mistral, SnapKV leads at both 65% and 50% while PyramidKV leads at 35%. Both models show the two methods converging closely but not tying at 35%, consistent with PyramidKV nearly collapsing to a uniform budget at tight compression (§7.2). The `min_num` clamp produces uniform allocation only when it binds, which depends on sequence length — shorter sequences in the benchmark may still carry a shallow pyramid at 35%, retaining slightly different tokens and preventing an exact tie.

SnapKV+rot and Pyr+rot fall below the post-prefill eviction leaders but substantially above Naive on both models. All four post-prefill methods share the first-token advantage, but why key re-rotation reduces F_out relative to their unrotated counterparts is covered in §8.1. At 50%, SnapKV+rot exceeds PyramidKV on Llama — the budget floor disadvantages Pyr at that rate. Pyr+rot closely tracks SnapKV+rot at 65% and 35% but drops several points behind at 50% — the same upper-layer information loss that widens KL faithfulness at 50% degrades F_out by the same mechanism.

Streaming is above Naive on Llama but trails Naive on Mistral at every rate, consistent with the KL rankings (§7.2).

**What F_out and KL together reveal.** SnapKV and PyramidKV lead F_out while scoring worst on KL faithfulness — the first-token-advantage mechanism explains this divergence (§8). SnapKV+rot breaks the pattern: it has the best KL faithfulness of any method (Table 3) *and* F_out substantially above Naive and Streaming, at the cost of some F_out relative to plain SnapKV/Pyr, for reasons that are not yet fully understood (§8.1).

### 7.4 Inference Performance

Compression is only worthwhile if it makes inference faster. We measure two quantities: **time to first token (TTFT)**, the wall-clock time from receiving a prompt to producing the first output token (dominated by the prefill pass), and **time per output token (TPT)**, the mean decode step time (dominated by KV cache memory bandwidth). All measurements are on a single V100 32GB GPU using Llama-3.1-8B fp16. Naive and PyramidKV: n=100/task (500 total); SnapKV and Streaming: n=20/task (100 total). All timing runs use the same five tasks: 2WikiMQA, MultifieldQA, Qasper, RepoBench-P, and TriviaQA. 

**Mean TTFT by retention rate** (Full context baseline: 6348 ms):

| Retention | Naive          | SnapKV        | Streaming     | Pyr            |
|-----------|----------------|---------------|---------------|----------------|
| 65%       | 2464 ms (−61%) | 6621 ms (+4%) | 6796 ms (+7%) | 7022 ms (+11%) |
| 50%       | 2139 ms (−66%) | 6872 ms (+8%) | 6750 ms (+6%) | 7109 ms (+12%) |
| 35%       | 1666 ms (−74%) | 6834 ms (+8%) | 6861 ms (+8%) | 7143 ms (+13%) |

Naive cuts TTFT by 61–74% by simply feeding the model a shorter prompt. All post-prefill eviction methods incur overhead above the full-context baseline: performing a full prefill over the uncompressed prompt before pruning, each is consistently slower than the uncompressed model at every retention rate. PyramidKV's layer-wise budget computation adds 11–13% overhead. SnapKV and Streaming-rerotated use simpler post-prefill eviction steps and add less overhead (+4–8%), but both are consistently above full-context TTFT. No KV-pruning method using post-prefill eviction can provide prefill savings: the full prompt must be processed regardless. SnapKV+rot inherits the same timing profile as unrotated SnapKV: key re-rotation is a single O(M·H·D) tensor operation applied once to the M retained keys after eviction, adding negligible overhead (<10ms) that does not affect the TTFT or TPT figures above. The original online StreamingLLM design — which maintains a bounded cache continuously rather than performing a full prefill — would achieve TTFT closer to Naive than to the post-prefill methods shown here, since prefill attention cost scales with the bounded window size rather than the full context length.

**Decode speed (TPT)** improves for all compressed methods because fewer retained tokens means a smaller KV cache to scan at each decode step. At 65% retention, mean TPT drops from 67.5 ms/tok (full) to ~54 ms/tok (~20% faster) for all methods including PyramidKV, snapkv_press, and streaming_rerotated; at 35% retention, all compressed methods reach ~44–48 ms/tok (~29–35% faster).

---

## 8. Why Post-Prefill KV Eviction Without Re-rotation Leads F_out but Trails KL Faithfulness

The most striking pattern in Tables 3 and 4 is a metric inversion: PyramidKV and SnapKV have the best F_out of any method (79.1% and 78.5% at Llama 65%) while simultaneously having the worst KL faithfulness (1.394 and 1.157 nats) — worse than every other method including naive truncation (0.175). This section explains the mechanism that produces this inversion, and where SnapKV+rot and Streaming fit within it.

**The first-token advantage.** All post-prefill eviction methods run a complete, uncompressed prefill before evicting the KV cache. The first output token's distribution is therefore always exactly identical to the full model's: eviction has not yet occurred at that point, so the hidden state at the last prefill position is computed from full, unrestricted attention over the entire context. Divergence begins at the second decode step, when the compressed cache takes over. For SnapKV and PyramidKV, where the retained cache consists of the tokens that received the highest attention from the query window, this divergence is small on short-answer tasks: evicted tokens are precisely those with low attention weight, so the compressed cache closely approximates the full model's for the few additional steps a short answer requires. On tasks where the answer is determined by the first one or two tokens, post-prefill eviction methods therefore replicate the full model's output nearly exactly.

KL faithfulness is measured at every generation step. Post-prefill eviction creates gaps in the retained cache that corrupt distributions at all subsequent steps, even when the first output is correct. On short-answer tasks, F_out is determined almost entirely by the first-token decision and registers no corruption at subsequent steps. This is the source of the inversion: both methods produce the same output text as the full model while assigning completely different probability mass to alternatives at every subsequent step.

**Mechanistic confirmation.** That the short-answer advantage comes from the full-prefill hidden state — not from query-relevant token selection — is confirmed directly by §6.2 (Table 1). Comparing `snapkv_press` (post-prefill eviction) to the recompute-over-gaps variant of SnapKV from that table isolates this precisely: both use identical attention-score selection at the same retention budget; only the eviction point differs. The recompute-over-gaps variant restricts attention *during* prefill rather than pruning the cache after it, so the full-prefill hidden state never exists. recompute-over-gaps SnapKV: Short=56.1%, Long=50.2%, Gap=6.0 — nearly flat, no short-answer advantage. snapkv_press: Short=87.7%, Long=54.7%, Gap=33.0 — the same steep gradient shared by PyramidKV. The same token selection rule applied at a different point in the forward pass produces a completely different behavioral signature.

**Key re-rotation and the first-token advantage.** SnapKV+rot applies re-rotation after eviction, compacting retained keys to positions 0, 1, ..., M−1. The first output token y*[0] remains exactly identical to the full model's because it is generated at the end of the prefill from full, unrestricted attention, and before re-rotation takes effect. Short-answer F_out nevertheless falls relative to unrotated SnapKV: Table 5 shows SnapKV+rot Short=79.3%, below unrotated SnapKV (87.7%) but above Naive (72.8%). The cause of this reduction is not fully resolved. As §8.1 shows, Pyr+rot — which decodes from T rather than compact position M — shows an identical short-form reduction, ruling out the decode starting position as the explanation; the cost appears to be a byproduct of key re-rotation itself, for reasons that are not yet well understood. KL faithfulness improves from 1.157 to 0.043 — a 27× improvement — because positional displacement at subsequent decode steps is eliminated.

**Streaming and the limits of the first-token advantage.** Streaming-rerotated uses the same post-prefill eviction timing — and therefore also produces a first output token that is exactly identical to the full model's. The cause of its poor performance lies elsewhere: streaming's retained set is structurally a block_end configuration (§6.3, Table 2) — a few sink tokens plus a recency tail, with evicted content massed at the head of the sequence. As Table 2 shows, block_end geometry produces severe KL faithfulness degradation that re-rotation cannot fix: rerotated block_end remains 15.9× worse than gapless in the synthetic experiment, compared to near-full recovery for scattered geometry (0.17×). The structural damage from the single large gap — and from value representations shaped by attending to the evicted head during prefill — persists regardless of key position reassignment, limiting re-rotation to an 11.7× KL improvement (1.639 → 0.141) versus 27× for SnapKV. Table 5 shows the result: Streaming short-answer F_out at 66.7%, below Naive (72.8%) and well below SnapKV/Pyr (~88%).

### 8.1 What F_out Reveals

Table 5 compares all methods by output-length category (Llama, 65%), with Naive as a reference. The 90-word threshold separates the two natural output regimes in LongBench: QA, classification, and code-completion tasks have full-model mean outputs of 14–48 words (≥96% of examples fall below 90 words), while summarization tasks have means of 339–384 words (≥86% fall above). NarrativeQA, Qasper, and SAMSum straddle the boundary with means near 80–90 words.

**Table 5. F_out by length category (Llama, 65%). Short = full-model reference output ≤ 90 words (n=1155); Long = > 90 words (n=445). Streaming = streaming_rerotated.**

| Method      | Short-answer F_out | Long-form F_out | Gap  |
|-------------|---------------------|-----------------|------|
| PyramidKV   | **88.1**            | 55.5            | 32.6 |
| SnapKV      | 87.7                | 54.7            | 33.0 |
| Pyr+rot     | 79.9                | 50.1            | 29.8 |
| SnapKV+rot  | 79.3                | 50.0            | 29.3 |
| Naive       | 72.8                | **57.6**        | 15.2 |
| Streaming   | 66.7                | 47.0            | 19.7 |

PyramidKV and SnapKV show nearly identical short-to-long drops (~33 points), more than twice Naive's (15.2). Naive slightly leads both on long-form F_out (57.6 vs 55.5 and 54.7), and their headline aggregate F_out numbers are propped up almost entirely by short, structured outputs where the first-token advantage is decisive. Both methods achieve near-full-model performance on single-token or short-phrase tasks: TriviaQA (Pyr: 97.0%, Snap: 92.0%), PassageCount (96.4%, 92.8%), 2WikiMQA (95.6%, 95.6%), PassageRetrieval (93.8%, 94.0%), LCC (91.3%, 92.8%), HotpotQA (92.2%, 95.3%).

The two re-rotated methods show nearly identical profiles: SnapKV+rot Short=79.3%, Long=50.0%, Gap=29.3; Pyr+rot Short=79.9%, Long=50.1%, Gap=29.8. Since SnapKV+rot decodes from compact position M and Pyr+rot decodes from T (matching the full model's positional frame), and both show the same short- and long-form reduction relative to their unrotated counterparts, the decode starting position is not the explanation — the reduction is attributable to key re-rotation itself.

Table 6 gives the same short/long breakdown by KL faithfulness, showing what F_out cannot.

**Table 6. Mean KL faithfulness by output-length category (Llama-3.1-8B, 65% retention, all 16 tasks). Short = full-model reference output ≤ 90 words (n = 1155); Long = > 90 words (n = 445). Bold = best per column.**

| Method | Short F_KL | Long F_KL |
|---|---|---|
| SnapKV+rot | **0.012** | **0.124** |
| Pyr+rot | **0.012** | 0.135 |
| Streaming | 0.080 | 0.297 |
| Naive | 0.143 | 0.256 |
| SnapKV | 1.335 | 0.697 |
| PyramidKV | 1.658 | 0.711 |

On short outputs, unrotated SnapKV and PyramidKV produce KL faithfulness of 1.3–1.7 — far worse than Naive (0.143) — despite leading on F_out. The first-token advantage explains the inversion: F_out is dominated by the first output word, which is identical to the full model's by construction, masking severe distributional corruption at all subsequent positions. Re-rotation reduces short-output KL faithfulness to 0.012 — near-perfect fidelity. On long outputs, where F_out converges (re-rotated ~50% vs unrotated ~55%), KL faithfulness still shows a 6× improvement from re-rotation (0.124 vs 0.697). The apparent trade-off visible in F_out — re-rotated methods losing a few points at both lengths — is precisely the metric inversion this paper documents: F_out is dominated by short tasks where the first-token advantage gives unrotated methods a structural edge; in both output regimes, KL faithfulness correctly identifies re-rotation as the more faithful compression.

Table 6 shows an interesting pattern for unrotated methods: Short F_KL (1.335/1.658 for SnapKV/PyramidKV) is nearly twice Long F_KL (0.697/0.711) while re-rotated methods and Naive show the reverse — Long KL exceeds Short.  This apparent KL recovery is specific to methods with high Short KL from positional damage. Per-step trajectories on gov_report (n=32, Llama 65%) confirm it for unrotated SnapKV: KL falls from ~1.00 over the first 50 generated tokens to ~0.09 over the last 150 of a ~500-token response. This is not the corruption healing; it reflects a structural property of teacher-forced evaluation. At each generation step t, the model attends not only to the M compressed prompt tokens but also to the preceding t−1 unpruned y* tokens. As t grows, the unpruned portion grows with it — by step 400 of a 500-token response, most attended context is uncompressed ground-truth output, naturally reducing measured F_KL for the corrupted methods.

For re-rotated methods and Naive, where Short KL is already low, this dilution effect does not dominate; Long KL rises instead because longer responses draw more heavily on the compressed context and genuine information loss becomes more apparent. In free-running generation, nothing keeps the model on the true continuation after an early divergence, so the damage concentrated at the first few post-eviction steps is precisely the damage that compounds in practice — which is why F_out degrades with output length even as teacher-forced F_KL appears to recover for unrotated methods.

A cross-rate comparison identifies the mechanism for long-form outputs (Table 7). The gap between unrotated and re-rotated long-form F_out narrows sharply as retention tightens, collapsing to near-zero at 35%. Re-rotation compacts all retained keys into a narrow absolute-position window (0..M−1 for SnapKV+rot; [T−n, T−1] for Pyr+rot), so from any decode query all retained prefill tokens appear within a bounded relative-distance range. The full model's prefill spans 0..T−1 — a much wider range. At 65% retention this position-range compression is largest and the long-form gap is largest (4.7 and 5.4 points). At 35%, unrotated methods also retain fewer keys with a smaller effective spread, the difference in position distributions shrinks, and the long-form gap collapses to 0.1–0.5 points.

**Table 7. Long-form F_out by retention rate (Llama). Gap = unrotated Long − re-rotated Long.**

| Rate | SnapKV | SnapKV+rot | Gap | PyramidKV | Pyr+rot | Gap |
|------|--------|------------|-----|-----------|---------|-----|
| 65%  | 54.7   | 50.0       | 4.7 | 55.5      | 50.1    | 5.4 |
| 50%  | 47.7   | 46.2       | 1.5 | 42.7      | 40.3    | 2.4 |
| 35%  | 42.1   | 42.0       | 0.1 | 42.3      | 41.8    | 0.5 |

The headline aggregate for SnapKV+rot (71.1%) is above Naive (68.6%) on short tasks but slightly below Naive (57.6%) on long tasks.

Streaming-rerotated's profile is distinct. It is the worst-performing method on both Short F_out (66.7%) and Long F_out (47.0%), below every other method in Table 5. Both follow from the block_end geometry established in §6.3. Streaming's single large evicted region causes structural damage that re-rotation cannot repair: as Table 2 shows, re-rotated block_end remains severely degraded relative to gapless (15.9×). This corruption takes effect from the second decode step onward, suppressing Short F_out below even Naive and compounding over long responses.

The practical implication is direct. Post-prefill eviction cannot reduce TTFT, so SnapKV+rot's quality advantage is concentrated precisely where its latency savings is least: short exact-lookup tasks where generation time is negligible and prefill dominates total cost. On long-form tasks, SnapKV and Pyr fall to or below Naive's long-form F_out (Naive: 57.6%, Pyr: 55.5%, SnapKV: 54.7%) while Naive delivers a 20% decode speedup with no prefill overhead or KL faithfulness cost. Post-prefill eviction without re-rotation occupies no clear deployment niche: unrotated SnapKV/Pyr are far less faithful than Naive on KL faithfulness despite leading on F_out. Re-rotation changes the KL faithfulness picture substantially, but not the TTFT constraint or the long-form F_out shortfall — choosing between SnapKV+rot and naive truncation is a choice between better distributional fidelity and better long-form output similarity, with no prefill latency benefit either way.


---

## 9. Conclusion

We introduced two faithfulness metrics that reveal complementary failure modes. KL faithfulness measures distributional agreement at every generation step; output faithfulness measures text-level similarity to the full model's generated output. §8 exposes a striking inversion: PyramidKV (79.1% F_out) and SnapKV (78.5%) are the top two methods on output faithfulness while simultaneously being the worst on KL (1.394 and 1.157 nats). A full, uncompressed prefill makes the first generated token identical to the full model's by construction; post-prefill eviction then degrades every subsequent step. Neither metric alone captures this behavior; both are necessary to characterize what a compression method actually does.

The primary finding of this paper is a diagnosis and a fix. SnapKV and PyramidKV — the most widely deployed attention-score-based KV compression methods — achieve mean KL faithfulness 6.6× worse than naive truncation on Llama-3.1-8B, despite selecting tokens specifically chosen to be important to the query. The failure is not selection but positional displacement: retained tokens keep their original RoPE encodings after eviction, placing them at out-of-distribution positional distances from every decode query. Applying key re-rotation after eviction — remapping retained keys to compact positions 0…M−1 — reduces mean KL by 27× at 65% retention (from 1.157 to 0.043), making SnapKV+rot the most faithful method in the study, outperforming all prompt-construction baselines. The fix is a single tensor operation adding negligible overhead.

Pyr+rot is within measurement noise of SnapKV+rot at 65% (0.047 vs 0.043) and ties exactly at 35% when the pyramid budget collapses to uniform. At 50% a gap opens, driven by a structural complication that SnapKV does not face: right-aligned re-rotation assigns each layer's retained keys to `[T − n_kept, T)`, but because n_kept differs across layers under the pyramid budget, queries and keys at different layers operate on different absolute-position ranges simultaneously. This cross-layer positional disparity is most severe when the pyramid is steepest — at 50%, n_kept spans from ≈0.66T at the bottom layer to just the `window_size` floor at the top — so the 50% gap in Pyr+rot likely reflects a combination of information loss at aggressively pruned upper layers and residual positional disparity that right-aligned re-rotation cannot eliminate. In deeper models, where the pyramid budget is distributed across more layers, this disparity may be more pronounced.

Streaming faces the analogous constraint from geometry: recency-only selection produces block_end gap structure that re-rotation cannot repair regardless of per-layer allocation.

All empirical results are on two base (non-instruction-tuned) decoder-only models in the 7–8B range, Llama-3.1-8B and Mistral-7B-v0.3, evaluated on LongBench v1. The KL faithfulness findings should generalize to any decoder-only transformer: the causal-gap and positional-displacement mechanisms are properties of the attention architecture, and KL is a logit-level comparison that does not depend on model-specific generation behavior. F_out results are more model-dependent — instruction tuning, scale, and benchmark distribution could all shift the specific numbers — and we have not tested larger scales, mixture-of-experts architectures, or instruction-tuned models.

The central message is operational. SnapKV and PyramidKV, as deployed without re-rotation, sacrifice most of their achievable KL faithfulness for no benefit. The fix is known, cheap, and demonstrated. Practitioners using these methods should add key re-rotation.

---

## References

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. (2023). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. arXiv:2308.14508.

Cai, Z., Zhang, Y., Gao, B., Liu, Y., Li, Y., Liu, T., Lu, K., Xiong, W., Dong, Y., Hu, J., and Xiao, W. (2024). PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. arXiv:2406.02069.

Chen, A., Geh, R., Grover, A., et al. (2025). The Pitfalls of KV Cache Compression. arXiv:2510.00231.

Devoto, A., Jeblick, M., and Jégou, S. (2025). Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution. arXiv:2510.00636. [KVPress framework (github.com/NVIDIA/kvpress) primary citation.]

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

Feng, Y., Lv, J., Cao, Y., Xie, X., and Zhou, S. K. (2025a). Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2407.11550. [Cited in text as "Ada-KV [NeurIPS 2025]".]

Feng, Y., et al. (2025b). Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective. arXiv:2502.03805. [Cited in text as "Feng et al. [2025b]" for the ‖V·W^O‖₁ output-perturbation bound. Same first author as Ada-KV above (2025a).]

Fu, Y., Cai, Z., Asi, A., Xiong, W., Dong, Y., and Xiao, W. (2025). Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning. In *Proceedings of ICLR 2025*. arXiv:2410.19258. [Cited in text as "HeadKV [ICLR 2025]".]

Guo, Z., Kamigaito, H., and Watanabe, T. (2024). Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters. In *Proceedings of EMNLP 2024*. arXiv:2406.12335. [Cited in text as "VATP [EMNLP 2024]".]

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Renard Lavaud, L., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., and El Sayed, W. (2023b). Mistral 7B. arXiv:2310.06825.

Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. (2024). SnapKV: LLM Knows What You are Looking for Before Generation. arXiv:2404.14469.

Liu, Z., Desai, A., Liao, F., Wang, W., Xie, V., Xu, Z., Kyrillidis, A., and Shrivastava, A. (2023). Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. In *Proceedings of ACL 2002*.

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568, 127063. (arXiv:2104.09864.)

Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., and Han, S. (2024). Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. In *Proceedings of ICML 2024*.

Wang, Z., Jin, B., Yu, Z., and Zhang, M. (2024). Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks. arXiv:2407.08454. [Cited in text as "KVMerger", the successor to CaM.]

Zhang, Y., Du, Y., Luo, G., Zhong, Y., Zhang, Z., Liu, S., and Ji, R. (2024). CaM: Cache Merging for Memory-efficient LLMs Inference. In *Proceedings of ICML 2024*. PMLR 235:58840–58850. [Cited in text as "Y. Zhang et al. [2024]" to distinguish from H2O (Z. Zhang et al. [2023]) and BERTScore (T. Zhang et al. [2020]).]

Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Fu, Y., and Han, S. (2025). DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. In *Proceedings of ICLR 2025*.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453. (Published at ICLR 2024.)

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. In *Proceedings of ICLR 2020*.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., and Chen, B. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

---

## Appendix A: KL Faithfulness — Per-Task Results

**Tables A1a–A1c. KL Faithfulness per task, Llama-3.1-8B (lower is better). Bold = best per row.**

**Table A1a. 65% retention.**

| Task | Naive | Streaming | SnapKV | SnapKV+rot | Pyr | Pyr+rot |
|---|---|---|---|---|---|---|
| NarrativeQA† | 0.022 | 0.045 | 0.787 | **0.010** | 1.337 | 0.012 |
| Qasper† | 0.281 | 0.237 | 0.881 | **0.092** | 1.229 | 0.095 |
| MultifieldQA | 0.262 | 0.302 | 1.316 | **0.037** | 1.535 | 0.045 |
| HotpotQA† | 0.203 | 0.069 | 1.935 | **0.006** | 2.181 | 0.008 |
| 2WikiMQA† | 0.221 | 0.103 | 1.714 | 0.008 | 2.422 | **0.004** |
| MuSiQue† | 0.197 | 0.098 | 1.747 | **0.007** | 2.523 | 0.010 |
| GovReport | 0.280 | 0.385 | 0.532 | 0.128 | 0.334 | **0.125** |
| QMSum† | 0.081 | 0.093 | 0.393 | 0.055 | 0.438 | **0.047** |
| MultiNews | 0.585 | 0.643 | 1.050 | **0.322** | 0.721 | 0.375 |
| TREC | 0.202 | 0.084 | 1.454 | 0.011 | 1.526 | **0.006** |
| TriviaQA | 0.076 | 0.008 | 1.151 | **0.001** | 1.647 | 0.001 |
| SAMSum | 0.011 | 0.015 | 0.791 | **0.004** | 1.182 | 0.004 |
| PassageCount† | 0.059 | 0.002 | 1.560 | **0.000** | 1.626 | 0.000 |
| PassageRetrieval | 0.188 | 0.032 | 1.624 | **0.002** | 1.416 | 0.003 |
| LCC | 0.076 | 0.085 | 0.881 | **0.007** | 1.142 | 0.007 |
| RepoBench-P | 0.048 | 0.050 | 0.702 | 0.004 | 1.052 | **0.004** |
| **Average** | 0.175 | 0.141 | 1.157 | **0.043** | 1.394 | 0.047 |

**Table A1b. 50% retention.**

| Task | Naive | Streaming | SnapKV | SnapKV+rot | Pyr | Pyr+rot |
|---|---|---|---|---|---|---|
| NarrativeQA† | 0.026 | 0.068 | 0.806 | **0.019** | 1.380 | 0.067 |
| Qasper† | 0.457 | 0.447 | 0.810 | **0.165** | 1.394 | 0.220 |
| MultifieldQA | 0.359 | 0.417 | 1.024 | **0.065** | 1.619 | 0.098 |
| HotpotQA† | 0.212 | 0.097 | 1.519 | **0.012** | 2.245 | 0.027 |
| 2WikiMQA† | 0.278 | 0.198 | 1.286 | 0.021 | 2.282 | **0.020** |
| MuSiQue† | 0.188 | 0.123 | 1.457 | **0.013** | 2.507 | 0.021 |
| GovReport | 0.327 | 0.466 | 0.488 | 0.223 | 0.504 | **0.223** |
| QMSum† | 0.088 | 0.125 | 0.369 | 0.086 | 0.484 | **0.082** |
| MultiNews | 0.759 | 0.784 | 1.002 | **0.551** | 1.099 | 0.654 |
| TREC | 0.260 | 0.183 | 1.114 | **0.016** | 1.711 | 0.021 |
| TriviaQA | 0.095 | 0.015 | 0.963 | **0.001** | 1.622 | 0.005 |
| SAMSum | 0.012 | 0.020 | 0.619 | **0.008** | 1.199 | 0.017 |
| PassageCount† | 0.055 | 0.003 | 1.468 | **0.000** | 1.538 | 0.001 |
| PassageRetrieval | 0.188 | 0.043 | 1.504 | **0.004** | 1.378 | 0.007 |
| LCC | 0.105 | 0.114 | 0.677 | **0.029** | 1.238 | 0.045 |
| RepoBench-P | 0.065 | 0.088 | 0.580 | **0.011** | 1.016 | 0.023 |
| **Average** | 0.217 | 0.199 | 0.981 | **0.077** | 1.451 | 0.096 |

**Table A1c. 35% retention.**

| Task | Naive | Streaming | SnapKV | SnapKV+rot | Pyr | Pyr+rot |
|---|---|---|---|---|---|---|
| NarrativeQA† | 0.045 | 0.091 | 0.559 | **0.030** | 0.559 | 0.030 |
| Qasper† | 0.637 | 0.620 | 0.867 | **0.281** | 0.858 | 0.281 |
| MultifieldQA | 0.475 | 0.499 | 0.925 | **0.122** | 0.914 | 0.122 |
| HotpotQA† | 0.220 | 0.114 | 1.246 | **0.018** | 1.252 | 0.018 |
| 2WikiMQA† | 0.351 | 0.238 | 1.040 | **0.049** | 1.052 | 0.049 |
| MuSiQue† | 0.191 | 0.182 | 1.243 | **0.022** | 1.236 | 0.022 |
| GovReport | 0.423 | 0.583 | 0.529 | **0.351** | 0.529 | 0.351 |
| QMSum† | **0.112** | 0.160 | 0.358 | 0.122 | 0.358 | 0.122 |
| MultiNews | 0.947 | 0.961 | 1.063 | 0.822 | 1.064 | **0.821** |
| TREC | 0.296 | 0.325 | 0.812 | **0.040** | 0.823 | 0.040 |
| TriviaQA | 0.157 | 0.044 | 0.832 | **0.001** | 0.833 | 0.001 |
| SAMSum | 0.017 | 0.031 | 0.475 | **0.016** | 0.475 | 0.016 |
| PassageCount† | 0.116 | 0.004 | 1.122 | **0.001** | 1.134 | 0.001 |
| PassageRetrieval | 0.282 | 0.050 | 1.247 | **0.009** | 1.240 | 0.009 |
| LCC | 0.145 | 0.149 | 0.547 | **0.070** | 0.550 | 0.070 |
| RepoBench-P | 0.087 | 0.123 | 0.513 | **0.027** | 0.511 | 0.027 |
| **Average** | 0.281 | 0.261 | 0.836 | **0.124** | 0.837 | **0.124** |

**Tables A2a–A2c. KL Faithfulness per task, Mistral-7B-v0.3 (lower is better). Bold = best per row.**

**Table A2a. 65% retention.**

| Task             | Naive  | Streaming | SnapKV | SnapKV+rot | Pyr   | Pyr+rot |
|------------------|--------|-----------|--------|------------|-------|---------|
| NarrativeQA†     | **0.003** | 0.125  | 0.445  | 0.006      | 0.780 | 0.005 |
| Qasper†          | 0.071  | 0.067     | 0.274  | **0.023**  | 0.472 | 0.026 |
| MultifieldQA     | 0.148  | 0.201     | 0.639  | **0.018**  | 0.718 | 0.024 |
| HotpotQA†        | 0.195  | 0.198     | 0.714  | **0.010**  | 0.725 | 0.014 |
| 2WikiMQA†        | 0.262  | 0.208     | 0.710  | **0.010**  | 0.683 | 0.018 |
| MuSiQue†         | 0.183  | 0.179     | 0.708  | **0.008**  | 0.679 | 0.013 |
| GovReport        | 0.180  | 0.281     | 0.578  | **0.081**  | 0.294 | 0.087 |
| QMSum†           | 0.067  | 0.082     | 0.201  | 0.022      | 0.183 | **0.021** |
| MultiNews        | 0.586  | 0.678     | 1.076  | **0.264**  | 0.626 | 0.289 |
| TREC             | 0.012  | 0.046     | 0.359  | **0.004**  | 0.703 | 0.005 |
| TriviaQA         | 0.039  | 0.070     | 1.104  | **0.002**  | 1.379 | **0.002** |
| SAMSum           | 0.021  | 0.040     | 1.030  | **0.006**  | 1.050 | 0.007 |
| PassageCount†    | 0.037  | 0.069     | 1.141  | **0.002**  | 0.897 | **0.002** |
| PassageRetrieval | 0.195  | 0.204     | 0.971  | **0.009**  | 0.541 | 0.015 |
| LCC              | 0.055  | 0.041     | 0.861  | **0.007**  | 1.070 | 0.008 |
| RepoBench-P      | 0.076  | 0.087     | 0.914  | **0.004**  | 0.940 | **0.004** |
| **Average**      | 0.133  | 0.161     | 0.733  | **0.030**  | 0.734 | 0.034 |

**Table A2b. 50% retention.**

| Task             | Naive  | Streaming | SnapKV | SnapKV+rot | Pyr   | Pyr+rot |
|------------------|--------|-----------|--------|------------|-------|---------|
| NarrativeQA†     | **0.003** | 0.149  | 0.349  | 0.014      | 0.782 | 0.017 |
| Qasper†          | 0.130  | 0.129     | 0.248  | **0.047**  | 0.506 | 0.072 |
| MultifieldQA     | 0.181  | 0.239     | 0.526  | **0.040**  | 0.789 | 0.058 |
| HotpotQA†        | 0.206  | 0.262     | 0.678  | **0.016**  | 0.722 | 0.041 |
| 2WikiMQA†        | 0.314  | 0.296     | 0.657  | **0.022**  | 0.739 | 0.059 |
| MuSiQue†         | 0.186  | 0.242     | 0.690  | **0.018**  | 0.711 | 0.048 |
| GovReport        | 0.217  | 0.335     | 0.486  | **0.151**  | 0.455 | 0.185 |
| QMSum†           | 0.071  | 0.100     | 0.196  | **0.039**  | 0.210 | 0.042 |
| MultiNews        | 0.747  | 0.805     | 0.995  | **0.497**  | 0.982 | 0.584 |
| TREC             | 0.017  | 0.054     | 0.193  | **0.007**  | 0.676 | 0.016 |
| TriviaQA         | 0.044  | 0.094     | 0.768  | **0.003**  | 1.376 | 0.012 |
| SAMSum           | 0.023  | 0.045     | 0.814  | **0.012**  | 1.102 | 0.019 |
| PassageCount†    | 0.040  | 0.076     | 1.107  | **0.005**  | 0.908 | **0.005** |
| PassageRetrieval | 0.195  | 0.225     | 0.871  | **0.019**  | 0.654 | 0.044 |
| LCC              | 0.070  | 0.053     | 0.652  | **0.020**  | 1.151 | 0.038 |
| RepoBench-P      | 0.088  | 0.115     | 0.678  | **0.011**  | 0.979 | 0.026 |
| **Average**      | 0.158  | 0.201     | 0.619  | **0.058**  | 0.796 | 0.079 |

**Table A2c. 35% retention.**

| Task             | Naive  | Streaming | SnapKV | SnapKV+rot | Pyr   | Pyr+rot |
|------------------|--------|-----------|--------|------------|-------|---------|
| NarrativeQA†     | **0.011** | 0.170  | 0.203  | 0.022      | 0.203 | 0.022 |
| Qasper†          | 0.165  | 0.165     | 0.262  | **0.086**  | 0.262 | **0.086** |
| MultifieldQA     | 0.213  | 0.286     | 0.519  | **0.077**  | 0.513 | **0.077** |
| HotpotQA†        | 0.218  | 0.310     | 0.632  | **0.028**  | 0.633 | **0.028** |
| 2WikiMQA†        | 0.362  | 0.365     | 0.690  | **0.051**  | 0.684 | **0.051** |
| MuSiQue†         | 0.186  | 0.290     | 0.679  | **0.041**  | 0.687 | **0.041** |
| GovReport        | 0.263  | 0.399     | 0.457  | **0.248**  | 0.457 | **0.248** |
| QMSum†           | 0.084  | 0.114     | 0.175  | **0.058**  | 0.175 | **0.058** |
| MultiNews        | 0.900  | 0.937     | 1.040  | **0.750**  | 1.039 | **0.750** |
| TREC             | 0.022  | 0.070     | 0.192  | **0.017**  | 0.192 | **0.017** |
| TriviaQA         | 0.062  | 0.146     | 0.725  | **0.006**  | 0.722 | **0.006** |
| SAMSum           | 0.027  | 0.054     | 0.741  | **0.022**  | 0.737 | **0.022** |
| PassageCount†    | 0.069  | 0.084     | 0.957  | **0.010**  | 0.930 | **0.010** |
| PassageRetrieval | 0.197  | 0.272     | 0.911  | **0.037**  | 0.920 | **0.037** |
| LCC              | 0.109  | 0.090     | 0.564  | **0.050**  | 0.564 | **0.050** |
| RepoBench-P      | 0.110  | 0.146     | 0.654  | **0.025**  | 0.661 | **0.025** |
| **Average**      | 0.188  | 0.244     | 0.588  | **0.096**  | 0.586 | **0.096** |

---

## Appendix B: Output Faithfulness — Per-Task Results

**Tables B1a–B1c. F_out per task, Llama-3.1-8B (higher is better). Bold = highest value per row.**

**Table B1a. 65% retention.**

| Task             | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|------------------|----------|-----------|----------|------------|----------|---------|
| NarrativeQA†     | **88.1** | 59.9      | 63.1     | 59.6       | 61.7     | 60.4 |
| Qasper†          | 46.4     | 56.9      | 71.4     | 58.4       | **72.6** | 61.0 |
| MultifieldQA     | 57.3     | 63.6      | 81.0     | 73.9       | **86.0** | 76.3 |
| HotpotQA†        | 88.2     | 72.6      | **95.3** | 85.3       | 92.2     | 85.9 |
| 2WikiMQA†        | 64.5     | 77.9      | **95.6** | 85.2       | **95.6** | 86.4 |
| MuSiQue†         | **97.1** | 72.3      | 92.7     | 86.9       | 92.3     | 83.5 |
| GovReport        | **55.8** | 40.1      | 47.8     | 45.5       | 51.2     | 43.9 |
| QMSum†           | **54.4** | 37.0      | 41.7     | 40.3       | 40.4     | 42.1 |
| MultiNews        | 43.8     | 41.0      | 44.6     | 42.5       | **49.0** | 40.5 |
| TREC             | 75.3     | 74.0      | **82.6** | 78.2       | 81.3     | 78.6 |
| TriviaQA         | 71.1     | 60.2      | 92.0     | 83.3       | **97.0** | 86.6 |
| SAMSum           | 64.9     | 52.6      | **78.4** | 67.2       | 73.4     | 66.5 |
| PassageCount†    | 70.7     | 61.8      | 92.8     | 83.9       | **96.4** | 82.3 |
| PassageRetrieval | 80.1     | 70.0      | **94.0** | 82.2       | 93.8     | 82.8 |
| LCC              | 62.0     | 73.8      | **92.8** | 87.7       | 91.3     | 88.3 |
| RepoBench-P      | 77.3     | 65.6      | 90.5     | 77.9       | **90.6** | 80.6 |
| **Average**      | 68.6     | 61.2      | 78.5     | 71.1       | **79.1** | 71.6 |

**Table B1b. 50% retention.**

| Task             | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
|------------------|----------|-----------|----------|------------|----------|----------|
| NarrativeQA†     | 48.4     | 53.0      | **62.2** | 58.5       | 53.0     | 50.6 |
| Qasper†          | 38.3     | 46.6      | **61.1** | 54.0       | 52.2     | 46.6 |
| MultifieldQA     | 55.7     | 59.6      | **76.3** | 73.8       | 70.6     | 67.1 |
| HotpotQA†        | 65.2     | 68.4      | **90.2** | 81.6       | 77.8     | 73.9 |
| 2WikiMQA†        | 62.8     | 74.2      | **92.0** | 84.4       | 84.5     | 82.2 |
| MuSiQue†         | 64.3     | 68.1      | **90.1** | 83.3       | 80.8     | 77.3 |
| GovReport        | 37.9     | 36.6      | **41.5** | 40.2       | 37.3     | 34.7 |
| QMSum†           | 38.7     | 36.0      | 39.8     | **40.8**   | 38.0     | 38.4 |
| MultiNews        | **44.9** | 37.7      | 35.4     | 35.9       | 33.4     | 33.1 |
| TREC             | 71.8     | 71.1      | **76.3** | 73.8       | 70.5     | 68.8 |
| TriviaQA         | 45.4     | 48.8      | **90.8** | 78.5       | 76.1     | 71.8 |
| SAMSum           | 48.2     | 48.6      | **67.2** | 61.1       | 49.1     | 49.1 |
| PassageCount†    | 31.8     | 57.9      | **89.3** | 77.5       | 85.6     | 75.1 |
| PassageRetrieval | 67.1     | 69.8      | **92.6** | 81.7       | 82.7     | 76.8 |
| LCC              | 57.8     | 67.7      | **84.6** | 81.1       | 70.0     | 68.4 |
| RepoBench-P      | 55.3     | 57.7      | **82.5** | 78.1       | 65.8     | 66.5 |
| **Average**      | 52.1     | 56.4      | **73.2** | 67.8       | 64.2     | 61.3 |

**Table B1c. 35% retention.**

| Task             | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
|------------------|----------|-----------|----------|------------|----------|----------|
| NarrativeQA†     | 44.7     | 48.4      | 57.3     | **59.6**   | 57.3     | **59.6** |
| Qasper†          | 34.3     | 40.6      | **51.8** | 50.3       | **51.8** | 50.7 |
| MultifieldQA     | 47.6     | 54.5      | 69.9     | 69.0       | **70.0** | 68.6 |
| HotpotQA†        | 58.8     | 66.9      | 85.2     | 80.1       | **85.4** | 80.5 |
| 2WikiMQA†        | 56.1     | 69.8      | **87.3** | 81.1       | **87.3** | 81.2 |
| MuSiQue†         | 58.4     | 68.2      | 84.2     | 80.5       | **84.6** | 80.7 |
| GovReport        | 35.1     | 31.9      | 36.6     | 36.0       | **36.7** | 35.1 |
| QMSum†           | 37.2     | 34.0      | 36.4     | **37.7**   | 36.4     | **37.7** |
| MultiNews        | **36.8** | 32.2      | 31.9     | 29.5       | 32.6     | 29.2 |
| TREC             | 66.9     | 65.3      | 68.1     | 66.6       | **67.6** | 66.4 |
| TriviaQA         | 38.6     | 39.8      | 84.2     | 74.8       | **84.8** | 75.3 |
| SAMSum           | 41.4     | 43.7      | **56.3** | 51.0       | **56.3** | 50.5 |
| PassageCount†    | 32.1     | 47.1      | 84.7     | 71.9       | **85.1** | 72.9 |
| PassageRetrieval | 57.9     | 65.4      | **86.0** | 78.5       | 85.4     | 78.0 |
| LCC              | 54.2     | 63.0      | 77.3     | 75.2       | **77.4** | 74.9 |
| RepoBench-P      | 50.0     | 57.2      | 71.6     | 66.9       | **71.7** | 66.5 |
| **Average**      | 46.9     | 51.8      | 66.8     | 63.0       | **66.9** | 63.0 |

**Tables B2a–B2c. F_out per task, Mistral-7B-v0.3 (higher is better). Bold = highest value per row.**

**Table B2a. 65% retention.**

| Task             | Naive | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|------------------|-------|-----------|----------|------------|----------|---------|
| NarrativeQA†     | 40.5  | 38.2      | **57.4** | 53.1       | 55.4     | 55.7 |
| Qasper†          | 69.0  | 69.4      | **87.1** | 79.6       | 86.1     | 80.6 |
| MultifieldQA     | 63.9  | 66.8      | **90.3** | 84.3       | 89.3     | 81.8 |
| HotpotQA†        | 66.9  | 63.9      | 94.7     | 88.5       | **94.8** | 87.1 |
| 2WikiMQA†        | 66.8  | 77.0      | **95.2** | 89.0       | 94.8     | 88.3 |
| MuSiQue†         | 64.6  | 68.0      | **94.3** | 90.2       | 93.3     | 82.2 |
| GovReport        | 46.7  | 41.8      | 62.6     | 58.6       | **64.2** | 58.1 |
| QMSum†           | 48.3  | 47.0      | 72.2     | 68.4       | **73.8** | 66.2 |
| MultiNews        | 45.5  | 35.8      | **51.7** | 48.7       | 51.6     | 46.2 |
| TREC             | 85.0  | 82.2      | **94.5** | 91.8       | 92.6     | 90.7 |
| TriviaQA         | 57.4  | 60.6      | 91.9     | 84.0       | **93.8** | 88.5 |
| SAMSum           | 55.1  | 57.2      | **72.4** | 69.8       | 69.6     | 67.9 |
| PassageCount†    | 57.3  | 57.7      | 97.0     | 90.8       | **97.5** | 84.7 |
| PassageRetrieval | 68.8  | 68.5      | 97.2     | 85.5       | **97.8** | 83.8 |
| LCC              | 62.1  | 76.0      | 88.3     | 86.1       | **90.8** | 87.2 |
| RepoBench-P      | 62.8  | 66.1      | **93.0** | 83.1       | 90.1     | 82.2 |
| **Average**      | 60.0  | 61.0      | **83.7** | 78.2       | 83.5     | 77.0 |

**Table B2b. 35% retention.**

| Task             | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|------------------|----------|-----------|----------|------------|----------|---------|
| NarrativeQA†     | 39.5     | 37.1      | 54.4     | 53.8       | **54.6** | 53.7 |
| Qasper†          | 60.6     | 60.4      | **77.1** | 73.5       | 77.0     | 73.3 |
| MultifieldQA     | 56.0     | 52.5      | 81.8     | 78.9       | **82.3** | 79.6 |
| HotpotQA†        | 64.0     | 63.6      | **88.0** | 81.3       | **88.0** | 81.8 |
| 2WikiMQA†        | 65.4     | 68.1      | **88.5** | 83.7       | **88.5** | 83.6 |
| MuSiQue†         | 58.7     | 63.7      | 87.9     | 80.9       | **88.0** | 81.0 |
| GovReport        | 36.8     | 33.6      | 45.5     | 44.4       | **46.1** | 44.5 |
| QMSum†           | 39.9     | 40.1      | **61.6** | 59.6       | 61.5     | 58.9 |
| MultiNews        | **36.3** | 28.6      | 35.1     | 32.7       | **36.3** | 32.2 |
| TREC             | 78.9     | 73.3      | 86.8     | **87.0**   | 86.8     | 86.9 |
| TriviaQA         | 42.9     | 42.0      | 79.8     | 76.8       | **80.5** | 77.0 |
| SAMSum           | 52.9     | 52.6      | 62.8     | 59.3       | **64.3** | 59.3 |
| PassageCount†    | 39.2     | 54.6      | **87.3** | 79.3       | 87.1     | 79.2 |
| PassageRetrieval | 61.9     | 66.2      | **88.4** | 78.3       | **88.4** | 77.3 |
| LCC              | 59.5     | 68.4      | 77.8     | 74.9       | **81.8** | 74.8 |
| RepoBench-P      | 62.6     | 57.6      | 82.0     | 77.6       | **85.9** | 77.6 |
| **Average**      | 53.5     | 53.9      | 74.0     | 70.1       | **74.8** | 70.0 |

**Table B2c. 50% retention.**

| Task             | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|------------------|----------|-----------|----------|------------|----------|---------|
| NarrativeQA†     | 38.9     | 36.5      | **56.6** | 53.3       | 47.5     | 49.3 |
| Qasper†          | 61.4     | 64.0      | **81.1** | 77.5       | 71.4     | 70.4 |
| MultifieldQA     | 57.4     | 62.9      | **87.1** | 81.0       | 77.8     | 73.2 |
| HotpotQA†        | 68.1     | 61.9      | **90.9** | 82.2       | 83.6     | 79.3 |
| 2WikiMQA†        | 64.2     | 71.3      | **93.7** | 85.9       | 77.0     | 78.4 |
| MuSiQue†         | 66.3     | 66.2      | **91.6** | 84.8       | 78.6     | 76.0 |
| GovReport        | 41.1     | 38.6      | **53.8** | 50.8       | 46.5     | 43.8 |
| QMSum†           | 45.3     | 42.2      | **65.2** | 63.9       | 58.5     | 58.8 |
| MultiNews        | 40.6     | 33.8      | **41.2** | 38.2       | 35.2     | 32.6 |
| TREC             | 82.7     | 78.2      | **91.1** | 89.6       | 80.8     | 81.5 |
| TriviaQA         | 53.0     | 49.3      | **86.8** | 81.1       | 74.1     | 67.4 |
| SAMSum           | 52.6     | 56.3      | 66.3     | **66.4**   | 51.0     | 50.8 |
| PassageCount†    | 49.3     | 55.2      | **90.7** | 87.7       | 90.0     | 82.1 |
| PassageRetrieval | 69.0     | 67.2      | **94.9** | 79.6       | 83.4     | 76.3 |
| LCC              | 57.0     | 74.0      | **85.6** | 83.2       | 74.2     | 74.3 |
| RepoBench-P      | 62.9     | 60.3      | **88.0** | 83.1       | 70.6     | 70.6 |
| **Average**      | 56.9     | 57.4      | **79.0** | 74.3       | 68.8     | 66.5 |

---

## Appendix C: Ground-Truth Accuracy — Per-Task Results

**Tables C1a–C1c. Ground-truth accuracy per task, Llama-3.1-8B (higher is better). Bold = best compressed method per row; "Full" is the reference. Streaming = streaming_rerotated.**

**Table C1a. 65% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.5  | 4.9      | **5.8**   | 5.6      | 5.2        | 5.5      | 5.3 |
| Qasper†          | 11.1 | 10.2     | 9.5       | 11.3     | **12.0**   | 11.8     | 11.3 |
| MultifieldQA     | 28.9 | 27.0     | 26.3      | **30.2** | 29.1       | 29.4     | 28.6 |
| HotpotQA†        | 9.9  | 9.6      | 9.0       | 9.9      | 10.1       | 10.2     | **10.5** |
| 2WikiMQA†        | 14.1 | 12.6     | **13.9**  | **13.9** | 13.5       | 13.7     | 13.4 |
| MuSiQue†         | 6.9  | 7.0      | 5.6       | 7.2      | 7.0        | 7.1      | **7.5** |
| GovReport        | 20.4 | 19.8     | 18.9      | 19.6     | 19.8       | 20.1     | **20.3** |
| QMSum†           | 10.3 | **11.5** | 9.9       | 9.7      | 9.4        | 9.4      | 9.9 |
| MultiNews        | 19.0 | 16.5     | 17.7      | **18.1** | 17.7       | 17.9     | 17.4 |
| TREC             | 70.0 | 66.0     | 68.0      | **71.0** | 70.0       | **71.0** | **71.0** |
| TriviaQA         | 17.4 | 17.3     | 17.4      | 17.5     | **17.7**   | 17.5     | 17.2 |
| SAMSum           | 16.0 | 16.5     | **17.1**  | 16.2     | 16.2       | 16.3     | 16.4 |
| PassageCount†    | 3.0  | 1.0      | **3.0**   | **3.0**  | **3.0**    | **3.0**  | **3.0** |
| PassageRetrieval | 44.0 | 37.0     | 37.0      | 44.0     | 43.0       | 44.0     | **45.0** |
| LCC              | 68.1 | 63.4     | 67.2      | 68.1     | 67.8       | **68.5** | 67.2 |
| RepoBench-P      | 55.6 | 53.9     | 53.2      | 55.4     | 55.5       | **56.0** | 55.8 |
| **Average**      | 25.0 | 23.4     | 23.7      | 25.0     | 24.8       | **25.1** | 25.0 |

**Table C1b. 50% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.5  | 5.7      | 5.5       | 6.7      | 5.7        | **6.8**  | 6.7 |
| Qasper†          | 11.1 | 8.6      | 9.4       | 11.1     | **11.7**   | 9.8      | 9.3 |
| MultifieldQA     | 28.9 | 23.0     | 25.4      | 29.2     | 29.1       | **29.5** | 29.2 |
| HotpotQA†        | 9.9  | 8.9      | **9.9**   | 9.7      | 9.7        | 9.6      | 9.2 |
| 2WikiMQA†        | 14.1 | 12.7     | 13.8      | 14.0     | **14.6**   | 13.8     | 14.0 |
| MuSiQue†         | 6.9  | 5.8      | 5.3       | 7.0      | **7.4**    | 6.8      | 6.3 |
| GovReport        | 20.4 | 19.5     | 19.0      | **20.0** | 19.6       | 19.5     | 19.4 |
| QMSum†           | 10.3 | 9.5      | 9.1       | 9.7      | **10.3**   | 9.7      | 9.7 |
| MultiNews        | 19.0 | 17.5     | **17.6**  | 16.1     | 16.6       | 15.4     | 15.7 |
| TREC             | 70.0 | 63.0     | 67.0      | 71.0     | **72.0**   | 68.0     | 69.0 |
| TriviaQA         | 17.4 | 17.4     | **17.6**  | 17.4     | 17.4       | 17.3     | 17.3 |
| SAMSum           | 16.0 | **17.7** | 17.6      | 16.1     | 16.0       | 17.6     | 17.1 |
| PassageCount†    | 3.0  | 2.0      | **3.0**   | **3.0**  | **3.0**    | **3.0**  | **3.0** |
| PassageRetrieval | 44.0 | 21.0     | 36.0      | **44.0** | **44.0**   | 43.0     | 41.0 |
| LCC              | 68.1 | 61.6     | 66.8      | **68.1** | 67.0       | 63.9     | 63.1 |
| RepoBench-P      | 55.6 | 51.4     | 53.8      | **55.7** | 55.3       | 52.8     | 53.5 |
| **Average**      | 25.0 | 21.6     | 23.6      | 24.9     | **25.0**   | 24.2     | 24.0 |

**Table C1c. 35% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.5  | **7.8**  | 6.5       | 6.4      | 6.6        | 6.4      | 6.6 |
| Qasper†          | 11.1 | 7.7      | 7.8       | **10.8** | 10.4       | **10.8** | 10.6 |
| MultifieldQA     | 28.9 | 21.2     | 24.2      | **30.2** | 29.6       | 30.0     | 29.8 |
| HotpotQA†        | 9.9  | **11.4** | 9.4       | 10.0     | 9.7        | 10.0     | 9.7 |
| 2WikiMQA†        | 14.1 | 11.0     | 12.5      | **14.6** | 13.6       | **14.6** | 13.6 |
| MuSiQue†         | 6.9  | 4.7      | 5.2       | **6.5**  | **6.5**    | **6.5**  | **6.5** |
| GovReport        | 20.4 | 18.7     | 18.6      | 18.9     | **19.1**   | 19.0     | 18.7 |
| QMSum†           | 10.3 | 9.9      | 9.9       | **10.2** | **10.2**   | **10.2** | 10.1 |
| MultiNews        | 19.0 | **16.7** | 16.0      | 15.7     | 15.4       | 15.8     | 15.4 |
| TREC             | 70.0 | 60.0     | 63.0      | **69.0** | 68.0       | **69.0** | 68.0 |
| TriviaQA         | 17.4 | 17.3     | **18.3**  | 17.4     | 17.6       | 17.4     | **17.7** |
| SAMSum           | 16.0 | 17.6     | **18.2**  | 17.3     | 17.3       | 17.3     | 17.3 |
| PassageCount†    | 3.0  | **3.0**  | **3.0**   | **3.0**  | **3.0**    | **3.0**  | **3.0** |
| PassageRetrieval | 44.0 | 12.0     | 37.0      | **44.0** | 42.0       | **44.0** | 42.0 |
| LCC              | 68.1 | 62.4     | 66.2      | 67.0     | **67.1**   | 67.0     | **67.1** |
| RepoBench-P      | 55.6 | 50.6     | **54.0**  | **55.1** | 54.7       | **55.1** | 54.7 |
| **Average**      | 25.0 | 20.8     | 23.1      | **24.8** | 24.4       | **24.8** | 24.4 |

**Tables C2a–C2c. Ground-truth accuracy per task, Mistral-7B-v0.3 (higher is better). Bold = best compressed method per row; Full is the reference. Streaming = streaming_rerotated.**

**Table C2a. 65% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.1  | 2.7      | 5.4       | 5.1      | **5.7**    | 4.1      | **5.7**  |
| Qasper†          | 5.3  | 4.3      | 5.0       | 5.1      | **5.2**    | 4.8      | 5.1      |
| MultifieldQA     | 25.3 | 22.1     | 23.2      | 24.9     | 24.9       | 24.8     | **25.8** |
| HotpotQA†        | 10.5 | 10.6     | 9.8       | 10.7     | **10.9**   | 10.6     | **10.9** |
| 2WikiMQA†        | 11.5 | 9.8      | 11.3      | **11.7** | 11.4       | **11.7** | 11.5     |
| MuSiQue†         | 5.1  | 3.9      | 4.4       | **5.1**  | 4.7        | **5.1**  | 4.6      |
| GovReport        | 20.0 | 19.7     | 19.4      | 19.8     | 19.7       | **20.0** | 19.8     |
| QMSum†           | 8.0  | **8.4**  | 8.3       | 7.9      | 7.9        | 7.9      | 8.1      |
| MultiNews        | 18.1 | 17.2     | 17.1      | 17.2     | 17.1       | **17.3** | **17.3** |
| TREC             | 72.0 | 67.0     | 70.0      | 70.0     | 70.0       | **71.0** | 70.0     |
| TriviaQA         | 23.1 | 24.2     | **26.5**  | 23.4     | 23.8       | 23.3     | 23.2     |
| SAMSum           | 16.9 | 17.8     | **18.3**  | 17.9     | 17.7       | 18.1     | 17.7     |
| PassageCount†    | 1.0  | **3.0**  | 1.0       | 1.0      | 1.0        | 1.0      | 1.0      |
| PassageRetrieval | 39.0 | 26.0     | 30.0      | **39.0** | 29.0       | **39.0** | 29.0     |
| LCC              | 62.9 | 60.2     | 62.1      | **63.2** | 62.8       | **63.2** | 63.1     |
| RepoBench-P      | 53.9 | **54.4** | 54.1      | 54.1     | 53.5       | 54.0     | 53.3     |
| **Average**      | 23.6 | 22.0     | 22.9      | **23.5** | 22.8       | **23.5** | 22.9     |

**Table C2b. 35% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.1  | 3.9      | 4.7       | **5.1**  | **5.1**    | **5.1**  | **5.1**  |
| Qasper†          | 5.3  | 4.0      | 4.2       | **4.9**  | 4.8        | 4.8      | 4.6      |
| MultifieldQA     | 25.3 | 18.2     | 18.6      | **25.1** | 24.9       | **25.1** | **25.1** |
| HotpotQA†        | 10.5 | 11.0     | **11.3**  | 10.4     | 10.9       | 10.4     | 10.9     |
| 2WikiMQA†        | 11.5 | 11.6     | **12.4**  | 11.2     | 11.7       | 11.2     | 11.7     |
| MuSiQue†         | 5.1  | 4.6      | 3.5       | **5.2**  | 4.6        | **5.2**  | 4.6      |
| GovReport        | 20.0 | 18.6     | 18.0      | 18.6     | **19.0**   | 18.6     | 18.9     |
| QMSum†           | 8.0  | 7.8      | **8.2**   | 7.8      | 7.8        | 7.8      | 7.8      |
| MultiNews        | 18.1 | **16.5** | 15.2      | 15.7     | 15.6       | 15.7     | 15.6     |
| TREC             | 72.0 | 60.0     | 61.0      | **66.0** | **66.0**   | **66.0** | **66.0** |
| TriviaQA         | 23.1 | 25.3     | **26.9**  | 23.7     | 24.6       | 23.7     | 24.6     |
| SAMSum           | 16.9 | 17.2     | **18.3**  | 17.8     | 17.9       | 17.8     | 17.9     |
| PassageCount†    | 1.0  | **3.0**  | 1.0       | 1.0      | 1.0        | 1.0      | 1.0      |
| PassageRetrieval | 39.0 | 15.0     | 22.0      | **39.0** | 27.0       | **39.0** | 27.0     |
| LCC              | 62.9 | 57.4     | **62.6**  | 62.3     | 61.3       | 62.3     | 61.7     |
| RepoBench-P      | 53.9 | 49.5     | 52.1      | 53.7     | **54.1**   | 53.7     | **54.1** |
| **Average**      | 23.6 | 20.2     | 21.2      | **23.0** | 22.3       | **23.0** | 22.3     |

**Table C2c. 50% retention.**

| Task             | Full | Naive    | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot  |
| ---------------- | ---- | -------- | --------- | -------- | ---------- | -------- | -------- |
| NarrativeQA†     | 5.1  | 4.7      | 4.5       | 5.1      | **5.2**    | 5.0      | 5.0      |
| Qasper†          | 5.3  | 4.5      | 4.5       | 4.7      | **4.8**    | 4.4      | 4.3      |
| MultifieldQA     | 25.3 | 18.0     | 23.5      | 25.0     | 25.3       | 25.6     | **25.7** |
| HotpotQA†        | 10.5 | 10.3     | **10.9**  | 10.5     | 10.7       | 10.7     | **10.9** |
| 2WikiMQA†        | 11.5 | 9.5      | 11.3      | 11.4     | 11.6       | **11.8** | **11.8** |
| MuSiQue†         | 5.1  | 4.7      | 3.9       | **5.3**  | 4.6        | 5.1      | 4.5      |
| GovReport        | 20.0 | 19.1     | 18.4      | **19.5** | **19.5**   | 18.4     | 18.6     |
| QMSum†           | 8.0  | 8.0      | **8.1**   | 7.6      | 7.8        | 7.5      | 7.6      |
| MultiNews        | 18.1 | **16.7** | 16.6      | 16.4     | 16.6       | 15.6     | 15.7     |
| TREC             | 72.0 | 62.0     | 65.0      | **70.0** | 68.0       | 67.0     | 66.0     |
| TriviaQA         | 23.1 | 26.1     | **26.6**  | 23.9     | 23.9       | 23.8     | 24.0     |
| SAMSum           | 16.9 | 17.3     | 17.7      | 17.1     | 17.8       | 19.1     | **19.2** |
| PassageCount†    | 1.0  | **3.0**  | 1.0       | 1.0      | 1.0        | 1.0      | 1.0      |
| PassageRetrieval | 39.0 | 23.0     | 26.0      | **39.0** | 29.0       | 38.0     | 25.0     |
| LCC              | 62.9 | 58.9     | 62.9      | 62.9     | **63.1**   | 60.1     | 60.7     |
| RepoBench-P      | 53.9 | 52.2     | 52.9      | **54.1** | **54.1**   | 53.8     | 53.5     |
| **Average**      | 23.6 | 21.1     | 22.1      | **23.3** | 22.7       | 22.9     | 22.1     |

---

## Appendix D: Synthetic Gap-Structure Analysis

The synthetic gap-structure ablation is presented in §6.3 (Table 2).
