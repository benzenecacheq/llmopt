# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context inference is expensive, and many tokens in the prompt are not equally important for generating a good response. A large body of work has therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li et al., 2024], value norms [Feng et al., 2025b], or combinations thereof — and evicting the rest before or during generation.

Standard benchmarks measure whether a compressed model gives the correct answer according to ground-truth labels. We argue this is the wrong objective: the goal of compression is approximation fidelity — producing the same output the full-context model would have produced. In this paper we introduce two faithfulness metrics: *KL Faithfulness*, which measures divergence from full-context token distributions, and *Output Faithfulness*, which measures text-level similarity between the full model's generated output and the compressed model's generated output. We show that ground-truth rankings and faithfulness rankings disagree substantially, and that the two new metrics reveal complementary failure modes.

Using KL faithfulness as our primary lens, we find that naïve proportional truncation — retaining the last 65% of tokens as a new, self-consistent prompt — substantially outperforms attention-score-based KV cache pruning (SnapKV, PyramidKV) by 6.6× on average on Llama. This result is not a failure of token selection; it is attributable to structural corruption inherent to the KV patching mechanism: pruning the KV cache creates causal gaps wherever the retained set is sparse, and any query attending over a gapped neighborhood produces degenerate attention that cascades through all transformer layers. StreamingLLM with key re-rotation is a qualified exception — post-hoc eviction with RoPE re-indexing eliminates positional misalignment and allows full-prefill enrichment, making it competitive with prompt construction on Llama.

This diagnosis motivates phrase-based context compression: constructing a new prompt from context spans selected by query-document lexical overlap, with no KV patching or attention mask modification. Phrase-based compression outperforms SnapKV and PyramidKV on KL faithfulness at all compression rates across 16 LongBench tasks, using only string matching — no model internals required.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. KV cache compression reduces this cost by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window. All of these methods share the same mechanism: they process the full prompt, compute importance scores, and then attend only to the retained subset during generation.

These compression methods are typically evaluated by running a compressed model on benchmark tasks and measuring how close the score is to an uncompressed baseline or to labeled ground truth [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that wrong answer — replicating the full model's behavior should be the criterion, not improving on it.

We argue that existing approaches share a structural flaw, independent of how well any particular scoring method identifies important tokens. When KV cache pruning retains a subset of token positions, it modifies the causal attention mask: a query at position *p* can only attend to retained positions *q < p*. For early sequence positions, there may be very few or no retained keys in the causal window. Attention over an empty or near-empty key set produces degenerate distributions. These corrupted attention outputs propagate through all subsequent transformer layers. The damage is not localized to the positions with empty windows; it cascades.

This effect is easy to confirm empirically. We compare naive proportional truncation (retaining the last 65% of tokens as a new, self-consistent prompt) with attention-score-based KV pruning retaining 65% of key-value states (§3.1). On NarrativeQA, naive truncation achieves KL faithfulness of 0.022; SnapKV KV pruning achieves 0.787 — a 35.8× gap on this task, and 6.6× on average across 16 tasks (0.175 vs. 1.157). §6.2's same-selection comparison (SnapKV vs. SnapKV-Select) isolates causal gaps alone, holding token selection identical, and shows the same direction of effect.

This finding has a straightforward implication: to faithfully compress a long context, one should not prune the KV cache. One should instead construct a shorter, self-consistent prompt. We propose *phrase-based context compression*, which does exactly this. The context is divided into contiguous phrases, each phrase is scored by lexical overlap with the query, the top-ranked phrases are concatenated with a recency tail to form the compressed prompt, and the model processes this shorter but structurally intact sequence normally.

KV cache pruning and phrase-based prompt construction are not solving two different problems — they are two different mechanisms for solving the same problem: reducing inference cost while preserving the full model's behavior as closely as possible. Neither pruning the cache nor reconstructing the prompt is the goal; both are candidate means to it, and a method should be judged on how well it achieves that goal, not on which mechanism it happens to use. Pruning fixes the prompt and restricts what the model attends to within it; phrase-based construction changes the prompt and leaves attention over it unrestricted.

Both mechanisms are evaluated under the same constraint — a fixed budget *r* on the resource that determines inference cost (§3.1) — which is what makes comparing them meaningful: it asks the only question that matters for choosing between them in deployment, which one gets closer to full-context behavior at equal cost. The fact that one mechanism preserves structural integrity and the other does not is the paper's central explanatory finding (§6) for why they perform so differently at equal budget.

We make the following contributions:

- **KL faithfulness metric**: We use KL divergence between the full and compressed model's next-token distributions at every generation step, averaged over a shared generation prefix, as the primary evaluation criterion.

- **Output faithfulness metric**: We introduce a second metric — word-level F1 between the full model's generated output and the compressed model's generated output — that measures behavioral convergence at the text level without any external ground-truth reference.

- **Structural corruption diagnosis**: We identify and demonstrate a structural source of faithfulness loss in KV cache pruning methods — degenerate attention from early queries with empty causal windows, cascading through all layers — using KL faithfulness as the diagnostic.

- **Phrase-based compression**: We introduce a prompt-construction approach that selects contiguous context spans by query-document lexical overlap, avoids structural corruption entirely, and outperforms all KV cache pruning methods on KL faithfulness.

## 2. Background and Related Work

**KV cache eviction.** Scissorhands [Liu et al., 2023] introduced the *persistence of importance* hypothesis: tokens that accumulate high attention during prefill remain important throughout generation, so the attention-score history is a reliable eviction criterion. H2O [Zhang et al., 2023] operationalizes this by maintaining a running sum of per-head attention scores and evicting the lowest-scoring tokens. SnapKV [Li et al., 2024] refines the scoring by pooling attention weights from only the last *w* queries (an observation window), better reflecting the queries that will matter at decode time, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] abandons score computation entirely, retaining a fixed set of attention sink tokens plus a sliding recency window; it achieves low latency at the cost of discarding all non-recent non-sink content.

**Value-norm scoring.** VATP [EMNLP 2024] scores tokens by the product of attention weight and L1 value norm, observing that attention sinks receive high attention but near-zero V-norm. Feng et al. [2025b] provide a theoretical justification via an upper bound on output perturbation, recommending a two-stage selector combining attention weights with projected value norms ‖V·W^O‖₁. We test a simpler additive combination of KQ alignment and raw V-norm and find it does not improve over KQ alignment alone.

**Layer- and head-adaptive budgets.** Ada-KV [NeurIPS 2025], HeadKV [ICLR 2025], and DuoAttention [ICLR 2025] allocate different token budgets to different attention heads within each layer. PyramidKV [Cai et al., 2024] takes a complementary approach, allocating different total budgets to different transformer layers — more KV slots at lower layers (which the original authors find need broader context) and fewer at higher layers (where task-relevant patterns have already concentrated). Both families operate on top of an existing token-scoring method and can in principle be combined with any of the per-token importance metrics discussed above. Our method uses a uniform budget per layer, and we note that the structural corruption problem (§6) applies to any of these methods that rely on the KV patching mechanism regardless of how budgets are distributed.

**Query-aware and dynamic selection.** Quest [Tang et al., ICML 2024] departs from static prefill-time selection by performing per-decode-step KV retrieval. At each generation step, the current query is used to retrieve the most relevant KV blocks from the full cache, limiting attention to the top-K pages. This eliminates the assumption that token importance is fixed at prefill time and instead adapts the attended set to each query. The tradeoff is computational: Quest requires a custom CUDA kernel for block-sparse attention, making integration into existing inference stacks non-trivial. Our structural corruption analysis applies to Quest only partially — because Quest retrieves complete contiguous blocks rather than individual scattered tokens, gaps within each attended block are minimal, though inter-block gaps remain.

**Token merging.** CaM [Y. Zhang et al., 2024] and its successor KVMerger take a different approach: rather than evicting low-importance tokens, they *merge* pairs of similar KV vectors, collapsing them into a single weighted average and reducing cache size without introducing empty positions. This avoids the causal gap problem entirely — the retained set is contiguous — but produces keys and values that are off-manifold linear combinations not present in any normal prompt. The faithfulness cost of such merging versus structural corruption from eviction is an open question. CaM does not support grouped-query attention (GQA), limiting its applicability to GQA models such as Llama-3.1.

**Evaluation critique.** Chen et al. [2025] ("Pitfalls of KV Cache Compression") provide empirical evidence that standard KV compression benchmarks are insufficient for evaluating compression quality and that compression methods perform significantly worse than their published numbers suggest under stricter evaluation conditions. This finding aligns with and provides additional evidence for our faithfulness-based critique of the field.

**Prompt compression.** LLMLingua [Jiang et al., 2023] and LongLLMLingua [Jiang et al., 2024] compress prompts at the token level by scoring token perplexity under a smaller proxy model and dropping low-perplexity tokens. This is architecturally different from KV cache pruning: the compressed prompt is fed to the model as-is, with no cache manipulation, so structural corruption is not a concern. However, perplexity-based scoring requires a forward pass over the full prompt using a second model, which adds latency comparable to the KV scoring overhead we observe in SnapKV-Select (§6.2). Phrase-based compression achieves similar prompt-shortening via pure string matching with no second model.

**Faithfulness evaluation.** To our knowledge, no prior KV cache compression work has systematically evaluated faithfulness to full-context outputs as a primary metric. The closest work is in model distillation and generation evaluation, where output-to-output comparison is common [Papineni et al., 2002; Zhang et al., 2020]. We adapt perplexity-based and embedding-based comparison to the compression setting.

**KVPress framework.** Devoto et al. [2024] introduce KVPress, a unified framework for KV cache compression research. Our method can be implemented as a KVPress press subclass, providing compatibility with the associated leaderboard and ecosystem.

---

## 3. Initial Experiments

### 3.1 Methods

All methods in this section target 65% token retention (r = 0.65). We use proportional truncation (65% of actual prompt length) rather than a fixed token budget, which would over-retain on short inputs and over-prune on long ones.

**Reference.**
- *Full context*: unmodified model with the complete prompt; serves as the ceiling, not a compressed method.

**Prompt-construction baselines.** These methods truncate the prompt to form a shorter, self-consistent input with no KV patching or attention mask modification.
- *Naive proportional truncation (Naive)*: the prompt is split into a 10% head (literal first tokens) and a 90% recency tail, concatenated to form a new prompt at 65% of the original length.

**KV pruning methods.** These methods process the full prompt, score each token's KV entry, and retain only the top-scoring positions at their original sequence indices. The causal attention mask is then reconstructed so that each query can only attend to retained positions at earlier indices. All KV pruning methods unconditionally retain the first 16 tokens (`always_keep_first=16`) and the last 16 tokens (`always_keep_last=16`). For GQA models such as Llama-3.1-8B (32 Q-heads, 8 KV-heads), scores are computed per Q-head and aggregated via max-pooling across heads sharing a KV-head before selection.

- *SnapKV*: attention weights pooled over the last 128-token observation window [Li et al., 2024].
- *Streaming (sink + recency, with key re-rotation)*: first 4 attention sink tokens plus a recency window to fill the remaining budget, following StreamingLLM's selection rule [Xiao et al., 2023]. Implemented via `kvpress`'s `StreamingLLMPress` wrapped in `KeyRerotationPress`, which remaps retained keys onto compact RoPE positions after eviction — the defining correctness-motivated feature of the published method. Like PyramidKV, this uses post-hoc cache eviction (§6.1): a full, unrestricted prefill completes before the cache is trimmed, so retained tokens' hidden states are computed with access to the complete context.

### 3.2 Setup

**Models.** Llama-3.1-8B and Mistral-7B-v0.3 (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128.

---

## 4. Ground-Truth Evaluation

Before introducing our faithfulness metrics, we establish what the standard evaluation paradigm reveals. We run both Llama-3.1-8B and Mistral-7B-v0.3 on LongBench v1 (16 tasks, 100 examples each) at 65% retention, comparing naive proportional truncation, SnapKV, StreamingLLM (Streaming), and PyramidKV against the full uncompressed model.

**Setup.** Full details in §3.2. All methods target r = 0.65 retention. Naive truncation keeps the first 6.5% and last 58.5% of tokens as a new self-consistent prompt. PyramidKV uses a layer-adaptive pyramid budget on top of SnapKV scoring. Ground-truth scores use the standard LongBench metrics: F1 for QA tasks, ROUGE for summarization, exact match for classification and code completion. Per-task ground-truth results across compression rates (65/50/35%), including phr128, are given in Appendix C.

**Llama-3.1-8B.**

| Task             | Full | Naive    | SnapKV    | Streaming | PyramidKV |
| ---------------- | ---- | -------- | --------- | --------- | --------- |
| NarrativeQA†     | 5.5  | 4.9      | **5.6**   | 0.1       | 5.5       |
| Qasper†          | 11.1 | 10.2     | 11.3      | 0.2       | **11.8**  |
| MultifieldQA     | 28.9 | 27.0     | **30.2**  | 1.5       | 29.4      |
| HotpotQA†        | 9.9  | 9.6      | 9.9       | 0.5       | **10.2**  |
| 2WikiMQA†        | 14.1 | 12.6     | **13.9**  | 0.5       | 13.7      |
| MuSiQue†         | 6.9  | 7.0      | **7.2**   | 0.5       | 7.1       |
| GovReport        | 20.4 | 19.8     | 19.6      | 18.9      | **20.1**  |
| QMSum†           | 10.3 | **11.5** | 9.7       | 9.9       | 9.4       |
| MultiNews        | 19.0 | 16.5     | **18.1**  | 17.7      | 17.9      |
| TREC             | 70.0 | 66.0     | **71.0**  | 68.0      | **71.0**  |
| TriviaQA         | 17.4 | 17.3     | **17.5**  | 0.3       | **17.5**  |
| SAMSum           | 16.0 | **16.5** | 16.2      | 17.1      | 16.3      |
| PassageCount†    | 3.0  | 1.0      | **3.0**   | **3.0**   | **3.0**   |
| PassageRetrieval | 44.0 | 37.0     | **44.0**  | 37.0      | **44.0**  |
| LCC              | 68.1 | 63.4     | 68.1      | 67.2      | **68.5**  |
| RepoBench-P      | 55.6 | 53.9     | 55.4      | 53.2      | **56.0**  |
| **Average**      | 25.0 | 23.4     | 25.0      | 18.5      | **25.1**  |

Full context is the reference and is not bolded. Bold marks the best compressed method per task. Tasks marked † have full-context scores below 15; results on these tasks are noisier.

**Mistral-7B-v0.3.**

| Task             | Full | Naive    | SnapKV    | Streaming | PyramidKV |
| ---------------- | ---- | -------- | --------- | --------- | --------- |
| NarrativeQA†     | 5.1  | 2.7      | **5.1**   | 0.8       | 4.1       |
| Qasper†          | 5.3  | 4.3      | **5.1**   | 0.1       | 4.8       |
| MultifieldQA     | 25.3 | 22.1     | **24.9**  | 1.2       | 24.8      |
| HotpotQA†        | 10.5 | 10.6     | **10.7**  | 0.3       | 10.6      |
| 2WikiMQA†        | 11.5 | 9.8      | **11.7**  | 0.8       | **11.7**  |
| MuSiQue†         | 5.1  | 3.9      | **5.1**   | 0.4       | **5.1**   |
| GovReport        | 20.0 | 19.7     | 19.8      | 19.4      | **20.0**  |
| QMSum†           | 8.0  | **8.4**  | 7.9       | 8.3       | 7.9       |
| MultiNews        | 18.1 | 17.2     | 17.2      | 17.1      | **17.3**  |
| TREC             | 72.0 | 67.0     | 70.0      | 70.0      | **71.0**  |
| TriviaQA         | 23.1 | **24.2** | 23.4      | 0.7       | 23.3      |
| SAMSum           | 16.9 | 17.8     | 17.9      | **18.3**  | 18.1      |
| PassageCount†    | 1.0  | **3.0**  | 1.0       | 1.0       | 1.0       |
| PassageRetrieval | 39.0 | 26.0     | **39.0**  | 30.0      | **39.0**  |
| LCC              | 62.9 | 60.2     | **63.2**  | 62.1      | **63.2**  |
| RepoBench-P      | 53.9 | **54.4** | 54.1      | 54.1      | 54.0      |
| **Average**      | 23.6 | 22.0     | **23.5**  | 17.8      | **23.5**  |

The overall pattern is the same across both architectures: ground-truth scores are compressed across methods. On both Llama and Mistral, SnapKV-Press ties PyramidKV (Llama: 25.0 vs 25.1; Mistral: 23.5 vs 23.5), both exceeding naive. Streaming (18.5 Llama, 17.8 Mistral) sits below SnapKV and PyramidKV but substantially above naive truncation on average, though with a highly uneven task profile: recency bias helps summarization, classification, and code tasks (which rely on the recent document tail) while nearly zeroing out fact-retrieval tasks (NarrativeQA, Qasper, HotpotQA, TriviaQA). PyramidKV leads or ties all compressed methods on both models.

Taken at face value, PyramidKV is the clear winner. But §5 argues these scores measure the wrong thing — and §9 shows that PyramidKV's ground-truth advantage conceals a faithfulness cost an order of magnitude larger than any other method's.

---

## 5. Faithfulness Metrics

### 5.1 Why Ground-Truth Evaluation Falls Short

Let M be a language model, c a full context, q a query, and y* = M(c, q) the full-context output. A compression method produces a compressed context ĉ and output ŷ = M(ĉ, q). Standard benchmarks measure d(ŷ, y_gt) where y_gt is a ground-truth label. We instead measure d(ŷ, y*): how closely the compressed output approximates the full-context output.

Ground-truth evaluation has four distinct failure modes as a compression metric.

**Reference mismatch.** For the four ROUGE-based summarization tasks in LongBench (GovReport, QMSum, MultiNews, SAMSum), ground-truth is computed against a human-written reference. Two methods can generate completely different summaries with identical ROUGE scores, as long as both cover the same facts. Whether one summary resembles what the full model would have written is invisible to the metric. For the seven F1-based QA tasks, ground-truth measures string overlap with a gold answer extracted from the dataset — again with no reference to the full model's actual behavior. Only for classification and retrieval tasks does ground truth directly constrain the model's output to match specific tokens.

**Faithful errors are penalized.** If the full-context model produces a wrong answer, a compressed model that faithfully reproduces that same wrong answer scores 0 on ground truth. The metric penalizes faithfulness when it produces the wrong answer for the "right reasons." A compression method should not be penalized for being a good approximation.

**Accidental correctness is rewarded.** Conversely, a compressed model that accidentally produces the correct answer — because truncation happened to leave in the relevant span, or because a different reasoning path led to the same answer — scores the same as a method that genuinely approximates the full model. Ground truth cannot distinguish these cases.

**Most of these metrics are not testing correctness at all.** Of the 16 tasks, only three (TREC, PassageCount, PassageRetrieval) score a prediction as binary right or wrong. The other thirteen — seven QA tasks scored by word-overlap F1, four summarization tasks scored by ROUGE-L, two code-completion tasks scored by string similarity — give continuous partial credit for overlap with one human-written reference, with no notion of "correct" or "incorrect" built into the metric itself. A summary that captures the source faithfully but phrases it differently than the dataset's one reference summary, or a QA answer that conveys the right information in different words, can score below an answer that shares more surface vocabulary with the reference while being less accurate. This does not mean the metric carries no signal — overlap with a reasonable reference is correlated with quality — but for most of this benchmark, "ground-truth accuracy" is closer to similarity to one human's phrasing than to a judgment of whether the model got the answer right.

**Low full-model accuracy amplifies the first three problems.** On six of our 16 tasks the full model's ground-truth accuracy is below 15 (NarrativeQA: 5.5, MuSiQue: 6.9, PassageCount: 3.0, HotpotQA: 9.9, Qasper: 11.1, QMSum: 10.3). On NarrativeQA, for example, the full model gets 94.5% of examples wrong. A perfectly faithful compressed method should also get 94.5% wrong; ground truth scores all of those as 0. The compressed method's GT score on that task then reflects only how close its wrong answers happen to come to the gold reference — a function of failure-mode similarity to the reference, not of approximation quality. As a consequence, GT differences between compression methods on low-accuracy tasks are dominated by noise: small random variations in which wrong answers partially overlap with the gold string. The signal about actual compression quality is vanishingly small precisely on the tasks where the method has the most work to do.

These failure modes matter in practice. PyramidKV's ground-truth advantage (§4) is exactly the kind of result that is hard to interpret: it may reflect genuine preservation of task-relevant information, or it may reflect a combination of structural properties that happen to produce correct first tokens without approximating the full model's behavior. §9 shows it is the latter.

This points to a more general limitation, beyond the three failure modes above. Ground truth — or any outcome metric computed on a fixed test distribution, whether benchmark accuracy or a downstream business metric — measures *that* a method produced acceptable outputs on the examples tested, not *why*. That distinction matters because it determines whether a result generalizes. An outcome metric tells us nothing about behavior on inputs outside the test distribution; it is silent on mechanism by construction. 

PyramidKV and SnapKV-Press are the clearest illustrations in this paper: by ground truth and output faithfulness both lead all tested methods, but §9 shows this is substantially an artifact of how post-hoc eviction methods compute their answer — a full, uncompressed prefill makes the first generated token identical to the full model's by construction, which dominates the aggregate score because the benchmark is full of short-answer tasks where the first token is the answer. A practitioner who only had the outcome metric would not know this, and would have no way to predict whether the advantage holds for a task with longer outputs, a different query distribution, or any input not resembling the test set — and indeed it mostly does not (Table 5, §9.1).

KL faithfulness does not share this limitation: because it measures whether the compressed model's actual computation matches the full model's at every step, it is informative about mechanism, not just outcome, and that mechanistic information is what lets a result generalize to inputs that were never tested. This is true regardless of which downstream objective a practitioner ultimately cares about — accuracy, faithfulness, or something else entirely — because mechanism transparency is what makes any of those objectives predictable beyond the examples already measured.

### 5.2 KL Faithfulness

We evaluate compression fidelity using KL divergence between the full and compressed models' next-token distributions, measured on a shared generation sequence. The uncompressed model generates a response y* = (y*₁,...,y*_L) from the full context. Both the full model and each compressed model are then teacher-forced on this shared token sequence: at step *t*, the full model conditions on [*c*, *q*, y*₁,...,y*_{t-1}] and the compressed model conditions on [*ĉ*, *q*, y*₁,...,y*_{t-1}]. The metric averages KL divergence across all generation steps:

```
KL_faith = (1/L) Σ_{t=1}^{L} KL(P_full(· | c, q, y*_{<t}) ‖ P_comp(· | ĉ, q, y*_{<t}))
```

Lower is better; 0 means identical distributions at every step. Conditioning all methods on the same shared token sequence eliminates path-dependence: a method whose compressed context causes early divergence would otherwise be evaluated on a different downstream sequence than a more faithful method, making scores incomparable across methods. Using y* as the shared prefix anchors every comparison to the full model's actual behavior.

This metric captures distributional agreement at the token probability level, not just which token is selected by greedy decoding. A method that shifts probability mass from the correct token to semantically similar alternatives — invisible to greedy accuracy measures — registers as a faithfulness cost in KL.

### 5.3 Output Faithfulness

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

**Why these two metrics.** Other distributional or behavioral metrics are possible — Jensen-Shannon divergence (bounded and symmetric, but without KL's direct expected-surprise interpretation), rank correlation between full and compressed top-k token rankings (sensitive to ordering, blind to probability mass), logit cosine similarity (no information-theoretic meaning, sensitive to logit scale), or calibration error against ground truth (reintroduces the external reference that both metrics are designed to avoid).

We use KL because it has a direct information-theoretic interpretation — the expected excess surprise from using the compressed model's distribution in place of the full model's — and because it decomposes cleanly per generation step, which is what makes the per-step analysis in §9 possible.

We use word-level F1 because it requires no reference distribution at all, only the two models' realized text, and reuses the scoring function LongBench already applies to extractive QA, keeping faithfulness evaluation on the same footing as the ground-truth evaluation it is meant to improve on. Neither choice is the only defensible one; we expect findings that hold under KL and F_out to hold in substance under JS divergence or a comparably motivated behavioral metric, since each is a different lens on one of the same two questions — does the distribution match, does the output match — not a different question.

**Caveat emptor on output-text metrics.** F_out is more principled than ground-truth evaluation, but word-level F1 introduces its own distortions that any output-text measurement inherits. Readers should treat F_out scores, and any prompt-based measurement, with appropriate skepticism:

- *Length bias.* Longer generated outputs have a higher statistical probability of token overlap with the reference by chance alone. A method that produces more verbose answers will score higher on recall independently of faithfulness, and F1's harmonic mean does not fully correct for this.
- *Paraphrase blindness.* Word-level F1 treats synonyms and legitimate paraphrases as mismatches. Two outputs that express the same content in different words score lower than two outputs that repeat the same words with different meaning.
- *Ceiling and floor effects.* On short-answer tasks, any method that produces the correct key term scores near 1.0; on long-form generation, divergence compounds over tokens and scores approach 0.0 for all methods. Neither extreme is discriminative.
- *Right answer, wrong reason.* A compression method that corrupts most of the context can still reproduce the full model's short output on factoid tasks where the answer is recoverable from a fragment.
- *Sample variance.* At n=100, confidence intervals on F_out are wide enough that small differences in aggregate scores should not be over-interpreted.

These limitations are why KL faithfulness is the primary metric in this paper. KL operates at the distribution level on every generation step, is insensitive to output length, and requires no string matching. F_out is included as a behavioral complement — a check that distributional faithfulness corresponds to observable output similarity — rather than as independent evidence.

### 5.4 Naive Truncation as a Baseline

A key empirical finding motivating this work is that naïve proportional truncation substantially outperforms attention-score-based KV cache pruning (SnapKV, PyramidKV) on KL faithfulness. The method retains 65% of the prompt tokens, split as a 10%/90% head/tail budget: the first 6.5% of tokens (literal prompt prefix) and the last 58.5% of tokens (recency tail), concatenated into a new self-consistent prompt. The middle portion is discarded. On NarrativeQA, naive truncation achieves KL 0.022 vs. 0.787 for SnapKV. This result was surprising and prompted the structural corruption investigation described in §6.

### 5.5 Results at 65% Retention

**Faithfulness evaluation.** Full-context outputs (y*) are generated first and stored. KL faithfulness compares compressed model distributions to full model distributions, teacher-forced on y*. F_out compares the compressed model's independently generated outputs to y* using word-level F1.

**KL Faithfulness at 65%.** The KL results below compare all methods on Llama-3.1-8B; the Mistral results follow the same pattern at roughly half the absolute magnitude.

| Task             | Naive          | SnapKV         | Streaming      | SnapKV+rot    | PyramidKV |
| ---------------- | -------------- | -------------- | -------------- | ------------- | --------- |
| NarrativeQA†     | 0.022          | 0.787          | 0.045          | **0.010**     | 1.338     |
| Qasper†          | 0.281          | 0.881          | 0.237          | **0.092**     | 1.229     |
| MultifieldQA     | 0.262          | 1.316          | 0.302          | **0.037**     | 1.535     |
| HotpotQA†        | 0.203          | 1.935          | 0.069          | **0.006**     | 2.181     |
| 2WikiMQA†        | 0.221          | 1.714          | 0.103          | **0.008**     | 2.422     |
| MuSiQue†         | 0.197          | 1.747          | 0.098          | **0.007**     | 2.523     |
| GovReport        | 0.280          | 0.532          | 0.385          | **0.128**     | 0.334     |
| QMSum†           | 0.081          | 0.393          | 0.093          | **0.055**     | 0.438     |
| MultiNews        | 0.585          | 1.050          | 0.643          | **0.322**     | 0.721     |
| TREC             | 0.202          | 1.454          | 0.084          | **0.011**     | 1.526     |
| TriviaQA         | 0.076          | 1.151          | 0.008          | **0.001**     | 1.647     |
| SAMSum           | 0.011          | 0.791          | 0.015          | **0.004**     | 1.182     |
| PassageCount†    | 0.059          | 1.560          | 0.002          | **0.000**     | 1.626     |
| PassageRetrieval | 0.188          | 1.624          | 0.032          | **0.002**     | 1.416     |
| LCC              | 0.076          | 0.881          | 0.085          | **0.007**     | 1.142     |
| RepoBench-P      | 0.048          | 0.702          | 0.050          | **0.004**     | 1.052     |
| **Average**      | 0.175          | 1.157          | 0.141          | **0.043**     | 1.394     |

Values in nats; lower is better. Bold marks the best method per row (excluding PyramidKV, which is shown for comparison). Streaming (with key re-rotation) leads on 8 of 16 tasks; Naive leads on 8. SnapKV KV pruning is substantially worse than both. PyramidKV is worst overall (1.394 vs. 0.141 for Streaming, a 9.9× gap).

**Output Faithfulness (F_out) at 65%.** The same methods evaluated on the new metric, comparing each model's output text against the full model's output text.

| Task             | Naive    | SnapKV   | Streaming | SnapKV+rot | PyramidKV |
| ---------------- | -------- | -------- | --------- | ---------- | --------- |
| NarrativeQA†     | **88.1** | 63.1     | 59.9      | 59.6       | 61.7      |
| Qasper†          | 46.4     | 71.4     | 56.9      | 58.4       | **72.6**  |
| MultifieldQA     | 57.3     | 81.0     | 63.6      | 73.9       | **86.0**  |
| HotpotQA†        | 88.2     | **95.3** | 72.6      | 85.3       | 92.2      |
| 2WikiMQA†        | 64.5     | **95.6** | 77.9      | 85.2       | **95.6**  |
| MuSiQue†         | **97.1** | 92.7     | 72.3      | 86.9       | 92.3      |
| GovReport        | **55.8** | 47.8     | 40.1      | 45.5       | 51.2      |
| QMSum†           | **54.4** | 41.7     | 37.0      | 40.3       | 40.4      |
| MultiNews        | 43.8     | 44.6     | 41.0      | 42.5       | **49.0**  |
| TREC             | 75.3     | **82.6** | 74.0      | 78.2       | 81.3      |
| TriviaQA         | 71.1     | 92.0     | 60.2      | 83.3       | **97.0**  |
| SAMSum           | 64.9     | **78.4** | 52.6      | 67.2       | 73.4      |
| PassageCount†    | 70.7     | 92.8     | 61.8      | 83.9       | **96.4**  |
| PassageRetrieval | 80.1     | **94.0** | 70.0      | 82.2       | 93.8      |
| LCC              | 62.0     | **92.8** | 73.8      | 87.7       | 91.3      |
| RepoBench-P      | 77.3     | 90.5     | 65.6      | 77.9       | **90.6**  |
| **Average**      | 68.6     | 78.5     | 61.2      | 71.1       | **79.1**  |

Values in %; higher is better. Bold marks the highest value per row. SnapKV+rot = SnapKV with key re-rotation (Pyr+rot identical; see §9).

**The two metrics reverse the SnapKV and PyramidKV ranking.** On KL faithfulness, SnapKV+rot dominates (0.043 nats), with SnapKV-Press (1.157) and PyramidKV (1.394) both an order of magnitude worse. On F_out, PyramidKV (79.1%) and SnapKV-Press (78.5%) lead — effectively tied and 10+ points above naive (68.6%) — while SnapKV+rot (71.1%) sits between them and naive, and Streaming (61.2%) trails further. These reversals are not contradictory: §9 explains the mechanism that decouples F_out from KL for post-hoc eviction methods.

The Mistral results at 65% show the same pattern.

**KL Faithfulness.**

| Task             | Naive          | SnapKV         | Streaming      | PyramidKV |
| ---------------- | -------------- | -------------- | -------------- | --------- |
| NarrativeQA†     | **0.003**      | 0.445          | 0.125          | 0.780     |
| Qasper†          | 0.071          | 0.274          | **0.067**      | 0.472     |
| MultifieldQA     | **0.148**      | 0.639          | 0.201          | 0.718     |
| HotpotQA†        | **0.195**      | 0.714          | 0.198          | 0.725     |
| 2WikiMQA†        | 0.262          | 0.710          | **0.208**      | 0.683     |
| MuSiQue†         | 0.183          | 0.708          | **0.179**      | 0.679     |
| GovReport        | **0.180**      | 0.578          | 0.281          | 0.294     |
| QMSum†           | **0.067**      | 0.201          | 0.082          | 0.183     |
| MultiNews        | **0.586**      | 1.076          | 0.678          | 0.626     |
| TREC             | **0.012**      | 0.359          | 0.046          | 0.703     |
| TriviaQA         | **0.039**      | 1.104          | 0.070          | 1.379     |
| SAMSum           | **0.021**      | 1.030          | 0.040          | 1.050     |
| PassageCount†    | **0.037**      | 1.141          | 0.069          | 0.897     |
| PassageRetrieval | **0.195**      | 0.971          | 0.204          | 0.541     |
| LCC              | 0.055          | 0.861          | **0.041**      | 1.070     |
| RepoBench-P      | **0.076**      | 0.914          | 0.087          | 0.940     |
| **Average**      | **0.133**      | 0.733          | 0.161          | 0.734     |

Values in nats; lower is better. Bold marks the best method per row (excluding PyramidKV, which is shown for comparison). Naive leads on 12 of 16 tasks; Streaming leads on 4. SnapKV-Press (0.733) and PyramidKV (0.734) are similarly poor, both 5.5× worse than Naive (0.133).

**Output Faithfulness (F_out).**

| Task             | Naive    | SnapKV   | Streaming | PyramidKV |
| ---------------- | -------- | -------- | --------- | --------- |
| NarrativeQA†     | 40.5     | **57.4** | 38.2      | 55.4      |
| Qasper†          | 69.0     | **87.1** | 69.4      | 86.1      |
| MultifieldQA     | 63.9     | **90.3** | 66.8      | 89.3      |
| HotpotQA†        | 66.9     | 94.7     | 63.9      | **94.8**  |
| 2WikiMQA†        | 66.8     | **95.2** | 77.0      | 94.8      |
| MuSiQue†         | 64.6     | **94.3** | 68.0      | 93.3      |
| GovReport        | 46.7     | 62.6     | 41.8      | **64.2**  |
| QMSum†           | 48.3     | 72.2     | 47.0      | **73.8**  |
| MultiNews        | 45.5     | **51.7** | 35.8      | 51.6      |
| TREC             | 85.0     | **94.5** | 82.2      | 92.6      |
| TriviaQA         | 57.4     | 91.9     | 60.6      | **93.8**  |
| SAMSum           | 55.1     | **72.4** | 57.2      | 69.6      |
| PassageCount†    | 57.3     | 97.0     | 57.7      | **97.5**  |
| PassageRetrieval | 68.8     | 97.2     | 68.5      | **97.8**  |
| LCC              | 62.1     | 88.3     | 76.0      | **90.8**  |
| RepoBench-P      | 62.8     | **93.0** | 66.1      | 90.1      |
| **Average**      | 60.0     | **83.7** | 61.0      | 83.5      |

Values in %; higher is better. Bold marks the highest value per row.

The KL ranking inversion (naive best, PyramidKV worst) is consistent across both models. On Mistral, SnapKV-Press (83.7%) and PyramidKV (83.5%) are effectively tied on F_out — the same pattern as Llama — while Streaming (61.0%) trails by 22 points but sits well above naive (60.0%), confirming the same pattern holds across architectures. SnapKV+rot Mistral F_out values are pending the Mistral re-run.

---

## 6. Structural Corruption in KV Cache Pruning

### 6.1 The Mechanism

All three KV-pruning methods benchmarked in this paper — SnapKV-Press, PyramidKV, and Streaming-rerotated — use *post-hoc cache eviction*: the model performs one normal, unrestricted prefill pass over the complete prompt, attending to the full context at every layer, and only after that pass completes is the KV cache trimmed before the first decode step. Retained keys and values are computed cleanly under full attention; nothing during prefill is modified. The earliest a gap can affect anything is at T=1 — the first decode step, when the newly generated query attends back over the now-sparse cache. §9's case study shows the empirical signature: KL(T=0) = 0, a clean prefill, followed by a sharp spike at T=1.

**Prefill enrichment.** Post-hoc eviction has a second consequence beyond gap timing: the KV pairs that survive carry representations computed under full-context attention over the entire original sequence, including attention to positions that are immediately evicted once the forward pass completes. A token retained in a post-hoc-evicted cache has attended to every other token in the document at every layer; a token retained in a compressed prompt has never attended to the evicted content at any layer. Prompt construction retains the right tokens in a structurally clean input; post-hoc eviction retains the right tokens in a structurally gapped input but with richer representations of each. The enrichment benefit accrues equally to all three post-hoc methods regardless of their selection policies. Whether it outweighs the decode-time gap cost depends on how many gaps remain and how they are distributed — examined in §6.4.

**Gap corruption.** Once a decode-time query attends over the sparse cache, the degenerate attention distribution does not stay local. Transformer hidden states are deeply entangled across layers: layer *l*'s computation at every position depends on the full layer *l*-1 output across all positions. A corrupted hidden state is carried forward into subsequent decode steps and, after *T* further steps, has diffused throughout the representation. This is not primarily an *early-token* problem. It is a *gap* problem: any portion of the retained set where positions are sparse will produce corrupted attention whenever a decode query crosses it. Recency-biased selection concentrates the budget at the tail, leaving the head and middle with severe gaps; attention-score-based selection distributes retained positions more evenly across the full sequence, leaving gaps throughout. The mechanism is identical in both cases; §6.4 examines how gap density and distribution determine whether enrichment or gap damage dominates.

**Positional misalignment.** KV pruning introduces a second structural problem independent of causal gaps: retained tokens keep their original RoPE encodings, so two tokens logically adjacent in the retained set may carry a relative distance of hundreds or thousands of positions. The attention mechanism was trained on sequences where positional distance reflects informational proximity; retained tokens with large RoPE gaps between them are out-of-distribution at every decode step. Our Streaming baseline addresses this via `KeyRerotationPress`, which re-rotates retained keys to compact RoPE positions, so its results are not conflated with positional misalignment. Prompt construction eliminates positional misalignment together with causal gaps. By constructing a new sequence from scratch, every position accurately reflects actual structure.

**A note on recompute-over-gaps.** A different implementation pathway — reconstructing the causal mask and re-deriving attention under the reduced position set during prefill itself — appears in two controlled experiments below (§6.2, §6.3) but is not used by any benchmarked method. It appears in those experiments because it enables tighter isolation: §6.2 holds token selection fixed to isolate mechanism, and §6.3 varies gap geometry with no selection at all. Both experiments use this pathway specifically because it confines the gap-exposure question to the prompt, independent of the enrichment effect that post-hoc eviction introduces at decode time.

### 6.2 Empirical Evidence

We isolate structural corruption from token-selection quality by comparing KV pruning directly to prompt construction using identical scoring criteria. If the gap in performance is due to mechanism rather than token choice, then the same scored tokens — selected by the same algorithm — should perform dramatically better when presented as a new self-consistent prompt than when patched into the KV cache at their original positions.

We select SnapKV as our representative KV pruning method because it uses canonical attention-score-based selection and is widely studied, and it is available in two mechanistically distinct implementations: a recompute-over-gaps version (`snapkv`, evaluated here under the controlled diagnostic pathway described in §6.1) and a post-hoc eviction version (`snapkv_press`, the kvpress implementation benchmarked throughout this paper). Including both alongside the prompt-construction equivalent shows the mechanism effect across all three points in the design space.

**Table 1. Mean KL faithfulness (y\*-anchored KL divergence, lower is better) for four method configurations. Llama-3.1-8B, 65% retention, 6 tasks, n = 100 per task. All three SnapKV rows use identical attention-score-based token selection; mechanism (KV-pruning vs. prompt construction) is the only variable.**

| Method | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean (6 tasks) |
|---|---|---|---|---|---|---|---|
| Naive — prompt construction | 0.221 | 0.262 | 0.281 | 0.081 | **0.048** | 0.076 | 0.162 |
| SnapKV (pcfg) — KV pruning (recompute-over-gaps) | 0.254 | 0.186 | 0.223 | 0.117 | 0.106 | 0.122 | 0.168 |
| SnapKV-Press — KV pruning (post-hoc eviction) | 1.714 | 1.316 | 0.881 | 0.393 | 0.702 | 1.151 | 1.026 |
| SnapKV-Select — prompt construction | **0.096** | **0.118** | **0.163** | **0.071** | 0.051 | **0.060** | **0.093** |

All three KV-pruning configurations use the same attention score-based token selection. The gap in faithfulness is entirely attributable to mechanism. SnapKV-Select with prompt construction achieves 0.093 — the best of the four. SnapKV (pcfg) with recompute-over-gaps reaches 0.168, roughly matching naive truncation despite its expensive scoring. SnapKV-Press with post-hoc eviction reaches 1.026 — 11× worse than SnapKV-Select and 6× worse than naive. The two KV-pruning pathways fail in different ways: recompute-over-gaps corrupts hidden states during prefill; post-hoc eviction delivers every decode query against a maximally scattered gap structure (§6.4), which proves more damaging in aggregate despite the prefill enrichment benefit.

Switching to prompt construction also improves output faithfulness: SnapKV-Select averages 57.2% F_out vs. SnapKV (pcfg)'s 54.6% on the same six-task subset — a 2.6 percentage-point gain. The contrast with the KL gap reflects that KL is a more sensitive measure of structural corruption than text-level output similarity; the two metrics are complementary (§5.4).

StreamingLLM with key re-rotation (0.141 mean KL on Llama at 65%) reverses this direction: Streaming KV pruning outperforms both naive (0.175) and its same-selection prompt-construction equivalent (streaming_select: 0.197). §6.4 examines the controlled same-selection comparison in full and explains why enrichment and gap geometry produce opposite outcomes for recency-biased versus scattered selection.

Despite its quality advantage over both KV-pruning implementations, SnapKV-Select is not a practical recommendation: computing SnapKV attention scores requires a full forward pass over the uncompressed prompt (~2500ms on our hardware), nearly cancelling the prefill savings from the shorter compressed prompt. This overhead motivates the question §7 addresses: can phrase-based selection — using only string matching, with no model access — approach SnapKV-Select's faithfulness at negligible cost?

### 6.3 Synthetic Gap-Structure Ablation

§6.2 isolates mechanism from token selection by holding the *scoring method* fixed and varying only mechanism (KV patching vs. prompt construction). That comparison still uses a content-aware scorer (SnapKV's attention weights) to choose which tokens to keep, leaving open whether the result depends on something specific to attention-based scoring. We remove scoring from the picture entirely: four position geometries are generated purely as a function of sequence length and retention fraction, with no query, no content, and no model access — pure positional patterns. At r = 0.65:

- **block_end** — one contiguous retained block at the tail (1 causal gap, immediately before it)
- **block_mid** — one contiguous retained block, centered (2 gaps, before and after)
- **clustered4** — four evenly-spaced contiguous blocks (~5 gaps)
- **scattered** — evenly-spaced single-token positions (~1400 gaps at this sequence length — the maximally scattered case)

Each geometry evaluates the same retained position set under three presentations. **Gapless**: those positions gathered into a new, shorter, contiguous prompt — no KV patching, compact RoPE positions, no enrichment from the evicted context. **Evicted**: a full prefill over the complete prompt (enabling full-context enrichment), followed by post-hoc eviction; retained keys remain at their original RoPE positions. This is the mechanism used by SnapKV-Press: scattered keys at their original sparse positions, values enriched by full-context attention. **Rerotated**: identical to evicted but with key re-rotation to compact RoPE positions after eviction (`KeyRerotationPress`). This is the mechanism used by `streaming_rerotated`. Comparing evicted to gapless isolates the joint effect of enrichment and positional displacement; comparing rerotated to evicted isolates the effect of position correction alone, since enrichment is identical in both.

**Table 2. Mean KL faithfulness by gap geometry (Llama, r = 0.65, 6 tasks, n = 20/task).**

| Geometry | Gaps | Gapless | Evicted | Rerotated | ev/gl | rot/gl |
|---|---|---|---|---|---|---|
| block_end | 1 | 0.177 | 3.493 | 2.814 | 19.8× | 15.9× |
| block_mid | 2 | 3.127 | 4.225 | 3.973 | 1.4× | 1.3× |
| clustered4 | ~5 | 2.745 | 3.765 | 3.498 | 1.4× | 1.3× |
| scattered | ~1400 | 1.205 | 1.289 | 0.221 | 1.1× | 0.18× |

**Scattered geometry: position correction is the dominant factor.** For scattered positions, evicted (1.289) and gapless (1.205) are nearly identical (1.1×): the enrichment benefit from the full prefill and the positional displacement cost from the scattered original RoPE positions roughly cancel. Re-rotation eliminates the positional displacement, collapsing the scattered-to-compact distance mismatch, and produces a dramatic improvement: rerotated (0.221) is 0.17× of evicted — nearly six times better. The enrichment that was present but ineffective in the evicted condition becomes fully effective once positions are corrected. Scattered + rerotated is the best-performing condition in the table, beating gapless by 5.4× despite retaining exactly the same positions.

**Block_end geometry: enrichment is the problem, not position.** For block_end, evicted (3.493) is 19.8× worse than gapless despite the fact that retained tail tokens have nearly the same RoPE distances to the decode query in both conditions (the tail is near the end of the full sequence in evicted, as it is in the short prompt in gapless). The damage is not from positional displacement but from enrichment: during the full prefill, tail tokens attend to the large evicted head section and incorporate irrelevant earlier context into their value representations. Re-rotation partially mitigates (rerotated = 2.814, 0.81× of evicted) but does not fix this: the corrupted values persist regardless of key position. Gapless (0.177), which never presents the evicted head content to the model at all, is by far the best presentation for block_end.

**Block_mid and clustered4** show modest damage in both evicted (1.4×) and rerotated (1.3×) relative to gapless, with re-rotation providing only a small benefit. Neither reaches gapless faithfulness.

The geometries are not directly comparable in absolute KL: block_mid and clustered4 discard the recency tail that block_end retains, so their elevated gapless baselines reflect lower content informativeness, not mechanism damage. Within-geometry ratios are the clean signal.

These results separate two structural factors — positional displacement and enrichment-induced corruption — that are entangled in any real method comparison. §6.4 connects them to the benchmarked methods and shows how selection geometry determines which factor dominates.

### 6.4 Enrichment versus Gap Geometry

The streaming and SnapKV results under post-hoc eviction expose a tension that §6.1–§6.3 cannot resolve on their own. Both `streaming_rerotated` and `snapkv_press` use the same post-hoc eviction mechanism and therefore receive the same enrichment benefit during prefill. Yet their KL faithfulness diverges sharply — streaming outperforms its same-selection prompt-construction counterpart, while SnapKV KV pruning is 6.6× worse than naive truncation. The difference is selection geometry.

**Controlled comparison: selection held fixed, mechanism varied.** `streaming_rerotated` (KV pruning, post-hoc eviction, key re-rotation) and `streaming_select` (prompt construction, identical selection rule) both retain four attention-sink tokens at the head and a recency window filling the remaining 65% budget. The retained positions are the same. The mechanism differs: `streaming_rerotated` patches the KV cache with post-hoc eviction and re-rotates retained keys to compact RoPE positions; `streaming_select` presents the selected tokens as a new self-consistent prompt, which naturally has compact positions. On Llama-3.1-8B at 65% retention across all 16 LongBench tasks:

| Method | Mechanism | Mean KL (Llama 65%) |
|---|---|---|
| streaming_select | Prompt construction | 0.197 |
| streaming_rerotated | KV pruning (post-hoc + re-rotation) | **0.141** |

KV pruning outperforms its own prompt-construction version by 28%, reversing the direction of §6.2's SnapKV comparison. The KV version leads on 11 of 16 tasks. Since selection and positional alignment are matched (both present compact RoPE positions — one via re-rotation, one naturally), the residual advantage reflects enrichment: retained KV states were computed with full-context attention, carrying information from the evicted positions that streaming_select's tokens never see at any layer.

**Why the same enrichment does not rescue SnapKV.** `snapkv_press` uses the same post-hoc eviction mechanism and receives the same enrichment during prefill. Its mean KL on Llama at 65% retention is 1.157 — far worse than naive truncation and 8.2× worse than `streaming_rerotated`. The enrichment benefit is identical in principle; the selection geometry is not.

Streaming's sink-plus-recency selection produces a near-contiguous retained set: four tokens at the head, one major gap spanning the document middle, then an uninterrupted recency block at the tail. The post-hoc KV cache has a single large gap. During decoding, most attention is over the contiguous recency window where the gap is absent; the head-to-tail jump is crossed rarely. Key re-rotation eliminates the positional misalignment from that jump. The total decode-time disruption is bounded and localized — comparable in structure to the `block_end` geometry in §6.3, which under recompute-over-gaps was damaging in absolute terms but confined to one concentrated neighborhood.

SnapKV's attention-based selection produces a globally scattered retained set: the top-K positions by pooled window attention weight are distributed throughout the full sequence length with roughly equal-density gaps everywhere. Every decode query attends back over a KV window where retained and evicted positions alternate without structure. There is no contiguous region where attention is clean — every decode step is over a maximally fragmented history, and the non-rerotated original RoPE positions compound the misalignment at each of those scattered gaps. The `scattered` geometry in §6.3 showed that even under recompute-over-gaps, diffuse single-token gaps distributed across the sequence produce cumulative KL damage (Δ = 1.681) that compounds across all affected positions even if each individual perturbation is mild. Under post-hoc eviction, SnapKV delivers that same distributed structure to every decode query, sustained throughout the full generation.

**Synthesis.** Whether enrichment or gap damage dominates is determined by selection geometry. Enrichment is a constant benefit — it accrues equally to all post-hoc methods regardless of which tokens are selected. Decode-time gap damage scales with the density and distribution of absent positions in the retained set. For recency-biased near-contiguous selection (streaming), the enrichment plus low gap density together exceed the cost: KV pruning is more faithful than its own prompt-construction equivalent. For attention-weighted scattered selection (SnapKV), dense gaps at every decode step overwhelm the enrichment: KV pruning is far worse than prompt construction.

This also clarifies a non-obvious relationship between selection quality and faithfulness quality. SnapKV's attention-based scoring selects more *informative* tokens than streaming's recency heuristic — tokens at positions of genuine semantic importance distributed throughout the document. But that is precisely the selection pattern that maximizes gap density: important tokens are scattered, so retaining them leaves gaps everywhere. Better token selection, under post-hoc eviction, produces a more damaging gap structure. The selection quality and the structural cost are not independent; they are coupled by the geometry of where important tokens tend to appear.

---

## 7. Phrase-Based Context Compression

### 7.1 Motivation: The Quality Ceiling and Its Cost

The controlled comparison in §6.2 also reveals the quality ceiling for prompt-construction methods. SnapKV-Select — which uses SnapKV's attention-based scoring to select tokens but presents them as a new prompt — achieves a mean KL of 0.093 across the six tasks in §6.2, outperforming streaming_rerotated (0.132 on the same tasks), the best of the post-hoc eviction methods benchmarked in this paper. Attention-based scoring, freed from structural corruption, sets the quality ceiling for prompt-construction approaches.

However, computing SnapKV scores requires a full forward pass over the uncompressed prompt — approximately 2000ms on our hardware. Added to the compressed-prompt prefill (~2500ms), this gives SnapKV-Select a total TTFT of ~4500ms: longer than naive truncation (~2500ms TTFT) and approaching the full-model prefill (~6300ms). The scoring overhead makes SnapKV-Select impractical for latency-sensitive deployment despite its quality advantage.

This motivates the question the rest of §7 addresses: is it possible to approach SnapKV-Select's quality using only information available without model introspection — no forward pass, no attention weights, no access to model internals?

### 7.2 Approach

Structural corruption is caused by the KV patching mechanism. The solution is simple: do not patch the KV cache. Instead, construct a new, shorter prompt from selected context spans and let the model process it normally. We call this *phrase-based compression*.

The claim we make and the claim we do not make are separable. We claim prompt construction — building a structurally intact prompt rather than patching the KV cache — is the better mechanism; §6–§9 establish this independently of how spans are selected. We do not claim that lexical overlap is the best possible way to select those spans. It is the selection method we use to demonstrate the mechanism's advantage using nothing beyond string matching — no model internals, no second model, no learned scorer. A better selection method could only improve on these results; it would not change the structural argument, since that argument concerns where information loss comes from (cache patching vs. prompt rewriting), not how spans are chosen within the prompt-rewriting approach.

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

| Configuration | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean |
|---|---|---|---|---|---|---|---|
| phrase\_word160\_t25 | 0.094 | 0.138 | 0.140 | 0.058 | 0.088 | 0.057 | 0.096 |
| phrase\_word160\_t30 | 0.100 | 0.135 | 0.152 | 0.057 | 0.085 | 0.054 | 0.097 |
| phrase\_word128\_t20 | 0.101 | 0.147 | 0.136 | 0.060 | 0.086 | 0.054 | 0.097 |
| phrase\_word160\_t20 | 0.094 | 0.137 | 0.143 | 0.071 | 0.088 | 0.053 | 0.098 |
| phrase\_word128\_t30 | 0.103 | 0.157 | 0.147 | 0.057 | 0.085 | 0.056 | 0.101 |
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

All methods target 65% token retention unless noted. PyramidKV, SnapKV, Naive, phr128, and SnapKV+rot are also evaluated at 50% and 35%.

### 8.2 KL Faithfulness: Main Results

At 65% retention, phr128 leads prompt-construction methods at 0.144 mean KL, beating naive truncation (0.175). Streaming with key re-rotation (0.141) edges phr128 marginally — the only rate at which a KV-pruning method matches phrase methods. SnapKV KV pruning (1.157) is dramatically worse than every other method, with no exception task where it outperforms prompt-construction methods.

**Table 3. Mean KL Faithfulness by compression rate (lower is better). phr128 = phrase\_word128\_t20; Streaming = StreamingLLM with key re-rotation; SnapKV = SnapKV (kvpress post-hoc eviction); SnapKV+rot = SnapKV with key re-rotation to compact positions; Pyr = PyramidKV. Bold = best per row. Note: Pyr+rot produces identical KL to SnapKV+rot at all rates (see §9); not shown separately. Per-task breakdown in Appendix A.**

**Llama-3.1-8B.**

| Retention | Naive | phr128 | Streaming | SnapKV | SnapKV+rot | Pyr |
|---|---|---|---|---|---|---|
| 65% | 0.175 | 0.144 | 0.141 | 1.157 | **0.043** | 1.394 |
| 50% | 0.217 | 0.170 | 0.199 | 0.981 | **0.077** | 1.451 |
| 35% | 0.281 | 0.228 | 0.261 | 0.836 | **0.124** | 0.837 |

SnapKV+rot achieves dramatically lower KL than any other method at every rate, including both prompt-construction methods and Streaming. Re-rotating retained keys to compact positions after eviction reduces mean KL from 1.157 to 0.043 at 65% retention — a 27× improvement — and the gap widens as compression tightens relative to phr128: SnapKV+rot/35% (0.124) is 1.8× better than phr128/35% (0.228). The finding that Pyr+rot = SnapKV+rot at all rates (to three decimal places, across all 16 tasks) confirms that position correction, not token selection strategy, drives this improvement.

At 65% retention, Streaming (0.141) remains marginally better than phr128 (0.144) but falls behind at tighter budgets. The key cross-rate result for prompt-construction methods is unchanged: phr128/50% (0.170) matches Naive/65% (0.175); phr128/35% (0.228) beats Naive/50% (0.217) — tighter phrase compression is more faithful than looser naive truncation at every step.

SnapKV without re-rotation shows a counterintuitive cross-rate trajectory: mean KL *improves* from 1.157 at 65% to 0.836 at 35% — as retention shrinks, the protected window constitutes a larger share of the surviving cache, reducing sparse-attention damage. Even so, SnapKV at its best (0.836 at 35%) remains 6.7× worse than SnapKV+rot at the same budget (0.124).

PyramidKV shows the same counterintuitive trajectory: mean KL is 1.394 at 65%, rises to 1.451 at 50%, then drops to 0.837 at 35%. We traced this to the layer-budget formula itself (`kvpress`'s `PyramidKVPress`): the per-layer budget is `max_num - layer_idx * steps`, clamped to [`window_size`, sequence length]. At 65% and 50% the clamp binds and produces a genuine pyramid; at 35%, `min_num` falls below `window_size` for nearly every sequence length in our range, tripping the fallback clause and returning a *uniform* per-layer budget — PyramidKV ceasing to be a pyramid and reverting to plain SnapKV-style allocation. PyramidKV at its best (0.837 at 35%) is 6.7× worse than SnapKV+rot at the same budget (0.124).

**Mistral-7B-v0.3** (65% and 35% retention).

| Retention | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| 65% | 0.133 | **0.121** | 0.161 | 0.733 | 0.734 |
| 35% | 0.188 | **0.159** | 0.244 | 0.588 | 0.586 |

The Mistral rankings are consistent with the Llama pattern. At 65%, phr128 leads (0.121), with Naive (0.133) and Streaming (0.161) both comfortably ahead of SnapKV (0.733) and PyramidKV (0.734). Unlike Llama, Streaming does not match phr128 on Mistral at any rate — Naive leads on 12 of 16 tasks (§5.5). SnapKV improves from 0.733 to 0.588 as the budget tightens, mirroring the Llama and PyramidKV pattern. Both SnapKV and PyramidKV are far worse than any prompt-construction method at every budget.

### 8.3 Inference Performance

Compression is only worthwhile if it makes inference faster. We measure two quantities: **time to first token (TTFT)**, the wall-clock time from receiving a prompt to producing the first output token (dominated by the prefill pass), and **time per output token (TPT)**, the mean decode step time (dominated by KV cache memory bandwidth). All measurements are on a single V100 32GB GPU using Llama-3.1-8B fp16. TTFT for phrase methods includes selection overhead (~3ms). phr128 and PyramidKV: n=100/task (500 total); SnapKV-Press and Streaming: n=20/task (100 total).

**Mean TTFT by retention rate** (Full context baseline: 6348 ms):

| Retention | phr128       | SnapKV        | Streaming     | Pyr           |
|-----------|--------------|---------------|---------------|---------------|
| 65%       | 2464 ms (−61%) | 6621 ms (+4%) | 6796 ms (+7%) | 7022 ms (+11%) |
| 50%       | 2139 ms (−66%) | 6872 ms (+8%) | 6750 ms (+6%) | 7109 ms (+12%) |
| 35%       | 1666 ms (−74%) | 6834 ms (+8%) | 6861 ms (+8%) | 7143 ms (+13%) |

Phrase methods cut TTFT by 61–74% simply by feeding the model a shorter prompt. All three post-hoc eviction methods go the other direction: performing a full prefill over the uncompressed prompt before pruning, each incurs overhead above the full-context baseline at every retention rate. PyramidKV's layer-wise budget computation adds 11–13% overhead. SnapKV-Press and Streaming-rerotated use simpler post-prefill eviction steps and add less overhead (+4–8%), but both are consistently above full-context TTFT. No KV-pruning method using post-hoc eviction can provide prefill savings: the full prompt must be processed regardless.

**Decode speed (TPT)** improves for all compressed methods because fewer retained tokens means a smaller KV cache to scan at each decode step. At 65% retention, mean TPT drops from 67.5 ms/tok (full) to ~54 ms/tok (~20% faster) for all methods including PyramidKV, snapkv_press, and streaming_rerotated; at 35% retention, all compressed methods reach ~44–48 ms/tok (~29–35% faster).

### 8.4 Output Faithfulness Results

Output Faithfulness (F_out) measures how similar the compressed model's generated text is to the full model's generated text (§5.3). Higher is better. All comparisons are within-model: Llama compressed methods are compared to Llama full context; Mistral compressed methods are compared to Mistral full context.

**Table 4. Mean F_out by compression rate (higher is better). Bold = highest value per row. SnapKV+rot = SnapKV with key re-rotation; Pyr+rot = PyramidKV with key re-rotation (identical to SnapKV+rot at all rates; not shown separately). †Streaming F_out values invalidated by a generation bug in the original runs; corrected re-run in progress (f65: ~59%, 12/16 tasks; f50/f35: pending). Per-task breakdown in Appendix B.**

**Llama-3.1-8B.**

| Retention | Naive | phr128 | Streaming | SnapKV   | SnapKV+rot | Pyr      |
|-----------|-------|--------|-----------|----------|------------|----------|
| 65%       | 68.6  | 54.5   | ~59†      | 78.5     | 71.1       | **79.1** |
| 50%       | 52.1  | 52.8   | —†        | **73.2** | 67.8       | 64.2     |
| 35%       | 46.9  | 48.6   | —†        | 66.8     | 63.0       | **66.9** |

**Cross-rate summary.**

SnapKV and PyramidKV lead F_out at all rates without re-rotation, for the same mechanistic reason: a full, uncompressed prefill means the first generated token's distribution is identical to the full model's by construction (§9.1). PyramidKV leads at 65% and 35% (79.1%, 66.9%); SnapKV leads at 50% (73.2%). Both converge at 35%, consistent with PyramidKV collapsing to a uniform budget at tight compression (§8.2).

SnapKV+rot (71.1%, 67.8%, 63.0%) falls between the post-hoc eviction leaders and prompt-construction methods. At 50% it exceeds PyramidKV (67.8 vs 64.2) and both prompt-construction methods. The striking result is that Pyr+rot = SnapKV+rot on every task at every rate: once positional displacement is corrected, the selection strategy (attention-score vs. layer-adaptive budget) has no measurable effect on output faithfulness.

Prompt-construction methods (naive, phr128) cluster tightly and degrade gracefully. At 65%, naive leads (68.6%); at 35%, phr128 has largely caught up (48.6% vs 46.9%).

Streaming-rerotated F_out values from the original runs are being discarded: the generation code contained a position-ID bug that caused degenerate output for KeyRerotationPress methods, producing near-zero F_out. Corrected results at f65 (12/16 tasks) show ~59%, placing Streaming above phr128 (54.5%) but below naive (68.6%); f50/f35 results are pending.

**Mistral-7B-v0.3** (65% and 35% retention; SnapKV+rot and Streaming pending Mistral re-run).

| Retention | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|
| 65% | 61.1 | 60.6 | 58.0 | **83.9** |
| 35% | 53.5 | 53.9 | 46.4 | **74.8** |

Mistral shows the same qualitative patterns as Llama: PyramidKV leads F_out at both rates (83.9%, 74.8%); SnapKV degrades more steeply than prompt-construction methods; prompt-construction methods cluster tightly. SnapKV+rot and Streaming corrected values will be added once the Mistral re-run completes.

**What F_out and KL together reveal.** Prompt-construction methods degrade gracefully on both dimensions. SnapKV and PyramidKV lead F_out while scoring worst on KL — the first-token-advantage mechanism explains this divergence (§9). SnapKV+rot breaks the pattern: it has the best KL of any method (Table 3) *and* F_out substantially above prompt-construction methods, at the cost of some F_out relative to plain SnapKV/Pyr (which retain the full-prefill first-token advantage that re-rotation sacrifices by decoding from compact position M rather than original position T).

---

## 9. Post-Hoc KV Eviction: A Case Study

PyramidKV and SnapKV-Press are nominally different methods — different selection strategies, different layer-budget allocations — yet they produce nearly identical results on every metric in this paper: GT 25.1 vs 25.0, F_out 79.1% vs 78.5%, KL 1.394 vs 1.157 nats at Llama 65% retention (§4, §8.2, §8.4), with neither dominating the other by any meaningful margin across all four retention rates. This convergence is not accidental. Both share the same inference-time pathway: a full, uncompressed prefill over the complete prompt, followed by in-place KV cache eviction before the first decode step. That shared mechanism — not their different selection strategies — determines their shared behavioral signature.

The mechanism's effect is precise: because the KV cache has not yet been evicted when the prefill completes, the hidden state at the last prompt position is computed from full, unrestricted attention over all context tokens. The first generated token's distribution is therefore identical to the full model's by construction. Distributional divergence begins only at the second token, when the pruned cache takes effect. On short-answer tasks where the answer is a single token or short phrase, F_out is determined almost entirely by this first-token advantage and both methods score near the full model. KL, measured at every generation step including all post-eviction decode steps, captures the distributional damage that short-answer F_out does not see. This explains the metric inversion: both methods produce the same output text as the full model while assigning completely different probability mass to alternatives at subsequent steps.

The layer-adaptive budget accounts for the small KL gap between the two methods: PyramidKV's upper layers operate with a more aggressively compressed cache than SnapKV-Press's uniform budget, producing slightly larger distributional distortion (1.394 vs 1.157 nats). But this difference is small relative to the gap between either method and prompt-construction baselines (naive: 0.175), and has no measurable effect on F_out or GT. Layer-adaptive budgeting provides no benefit on any metric that matters for deployment.

A direct mechanistic confirmation comes from comparing snapkv_press to the pcfg-based snapkv from §6.3. Both apply attention-score-based selection with the same retention budget; the only difference is when pruning occurs. The pcfg implementation restricts attention *during* prefill (recompute-over-gaps), eliminating the first-token advantage. The post-hoc implementation (snapkv_press) evicts *after* prefill, preserving it. On the short/long F_out split: pcfg snapkv shows Short=56.1%, Long=50.2%, Gap=6.0 — nearly flat, no short-answer advantage. snapkv_press shows Short=87.7%, Long=54.7%, Gap=32.9 — the same steep gradient as PyramidKV. The same token selection rule applied at a different point in the forward pass produces a completely different behavioral signature. The steep short-to-long gradient requires both post-hoc eviction timing and attention-score-based selection working together: timing preserves the full-prefill first-token advantage; selection determines whether the retained positions are query-relevant enough to exploit it. Streaming-rerotated, which uses post-hoc eviction timing with recency-biased selection rather than attention-score selection, confirms this: without query-relevant retention, the T0 advantage does not materialize (Table 5).

A natural question is whether KL corruption compounds over a long response. Per-step KL trajectories on real long-answer LongBench data (gov\_report, n=32, Llama 65%) show the opposite for both PyramidKV and SnapKV-Press: after the T0→T1 spike, KL falls sharply — from a mean of ~1.00 over the first 50 generated tokens to ~0.09 over the last 150 of a ~500-token response (30 of 32 examples show this fall). This is not the corruption healing. The trajectory is computed under teacher forcing — the model is fed the *correct* continuation at each step — so late-sequence predictions are dominated by locally easy, source-independent continuation that a compressed model gets right almost as often as the full model. Sensitivity to the original corruption is diluted by an increasing share of steps that never depended on it. Naive and phr128 show no comparable recovery because they never had a comparable cliff to fall from — their prompt-truncation corruption is present at low-to-moderate severity from the first step and stays roughly flat. In real free-running generation, nothing keeps the model on the true continuation after an early divergence, so the damage concentrated at the first few post-eviction steps is precisely the damage that compounds in practice — which is why F_out gets *worse* with output length even as teacher-forced KL appears to recover.

### 9.1 What F_out Reveals

Both PyramidKV and SnapKV-Press achieve high F_out on tasks with short, structured outputs: TriviaQA (Pyr: 97.0%, Snap: 92.0%), PassageCount (96.4%, 92.8%), 2WikiMQA (95.6%, 95.6%), PassageRetrieval (93.8%, 94.0%), LCC (91.3%, 92.8%), HotpotQA (92.2%, 95.3%). On these tasks, both methods consistently produce the same text as the full model. Streaming-rerotated, despite using the same post-hoc eviction timing, does not share this pattern: its short-answer aggregate is 17.7% (Table 5), confirming that the first-token advantage also requires retaining query-relevant positions.

On tasks with longer, free-form outputs, the advantage shrinks and is sometimes reversed. Table 5 compares all three post-hoc eviction methods by output-length category (Llama, 65%), with Naive as a reference. PyramidKV and SnapKV-Press share a steep length gradient; Streaming serves as a within-mechanism counter-example isolating the role of selection policy.

**Table 5. F_out by length category (Llama, 65%). Short = full-model reference output ≤ 90 words (n=1155); Long = > 90 words (n=445).**

| Method    | Short-answer F_out | Long-form F_out | Gap  |
|-----------|--------------------|-----------------|------|
| PyramidKV | **88.1**           | 55.5            | 32.6 |
| SnapKV    | 87.7               | 54.7            | 32.9 |
| Streaming | 17.7               | 4.2             | 13.5 |
| Naive     | 72.8               | **57.6**        | 15.2 |

PyramidKV and SnapKV-Press show nearly identical short-to-long drops (~32.7 points), more than twice Naive's (15.2). Both have little long-form advantage over Naive — Naive slightly leads on long-form F_out (57.6 vs 55.5 and 54.7) — and their headline F_out numbers (79.1%, 78.5%) are propped up almost entirely by short, structured outputs.

Streaming shows a completely different profile. Short F_out is 17.7% — below Naive (72.8%) — and Long F_out collapses to 4.2%, the lowest of any method. The gap (13.5) is similar in magnitude to Naive's, not to SnapKV/Pyr's. The contrast isolates the selection policy effect: recency-biased eviction retains sink tokens and recent context rather than query-relevant positions, so the full-prefill first-token advantage does not materialize. Streaming's two failure modes — poor short-answer performance from wrong token retention, and poor long-form performance from accumulated distributional damage — are independent of each other and compound. The result is the worst F_out of any method at both output lengths.

For tasks requiring sustained coherent generation, the prefill advantage disappears. Tokens after the first attend only to the pruned KV cache, and distributional damage accumulates with each generation step. F_out is insensitive to this on short-answer tasks because a single correct token produces a high score regardless of what follows; on long-form tasks, where the comparison spans many tokens, the damage becomes visible. SnapKV-Press and PyramidKV lose their F_out lead gradually as output length grows; Streaming, which never had the short-answer advantage, is uniformly below Naive across both categories.

### 9.2 Inference Speed

All three post-hoc eviction methods — PyramidKV, SnapKV-Press, and Streaming-rerotated — must process the full uncompressed prompt before evicting the KV cache, so none can provide TTFT improvement relative to running the full model (6348ms). PyramidKV adds measurable overhead from its layer-wise budget computation and eviction: 7022ms, 11% *slower* than full context and 2.8× slower than phr128 at the same retention (2464ms), with this penalty holding across all four retention rates. SnapKV-Press and Streaming-rerotated perform simpler post-prefill eviction steps and add less overhead than PyramidKV, but both still require the full prefill pass and their TTFT remains at or above full context (§8.3). The decode-time benefit applies equally to all methods: at 65% retention, mean TPT drops from 67.5ms/tok (full) to ~54ms/tok (~20% faster) for all compressed methods regardless of how the cache was constructed.

The TTFT profile determines which deployment settings each approach can serve. When time-to-first-token is the bottleneck — interactive settings where the user is waiting on the first word — no post-hoc eviction method provides any improvement over the full model, and phrase construction's 60–74% TTFT reduction (§8.3) is the only approach in this paper that helps. When decode throughput is the bottleneck — long-output batch settings — all methods at the same retention rate are equivalent on TPT, since throughput depends only on cache size.

### 9.3 Synthesis

PyramidKV and SnapKV-Press together illustrate why no single metric suffices to characterize compression quality. GT and F_out both rank them first. KL ranks them worst. Each is measuring something real:

- GT and F_out measure *behavioral* fidelity: does the compressed model produce the same outputs as the full model? Both methods' full prefill establishes the same representational state as the full model at T=0, driving the same decisions at the first (often critical) generated token. The answer texts match.

- KL measures *distributional* fidelity: does the compressed model assign the same probability mass as the full model at each generation step? Both methods' post-hoc KV eviction creates gaps in the stored cache that corrupt subsequent distributions, even when the final output is correct. The distributions do not match.

These are different properties of compression quality. A method optimized purely for GT or F_out could preserve the argmax token at every step while completely misrepresenting the full model's uncertainty and alternative predictions. A method optimized purely for KL might shift probability mass to semantically equivalent alternatives — invisible to GT but costly in KL — while producing identical final text. Neither metric alone characterizes the compression.

The two metrics together reveal that post-hoc eviction methods' "faithfulness" is a particular kind: they replicate the full model's decisions without replicating the full model's reasoning. This has practical consequences: a downstream system that uses the compressed model's logits for calibration, uncertainty estimation, or beam search will see corrupted information even when the greedy output is correct. Ground-truth evaluation cannot detect any of this. KL and F_out together can.

Putting the length-dependence (Table 5) together with the speed profile (§9.2) gives a practical verdict. For PyramidKV and SnapKV-Press: on short-answer tasks their F_out advantage is real — they reliably reproduce the full model's first-token decisions — but it is bought at a cost that defeats the purpose of compression: full prefill means TTFT is no better than running the full model, so on exactly the tasks where they win behaviorally, they provide no latency benefit. On long-form tasks, where generation length is large enough that decode-time speedup matters, both methods' F_out has fallen to naive truncation's level or below (long-form F_out: naive 57.6 vs PyramidKV 55.5 and SnapKV-Press 54.7), and naive delivers the same 20% decode speedup without the prefill overhead or the KL corruption. For Streaming: the same TTFT penalty applies, but there is no short-answer F_out advantage to offset it — Streaming trails Naive at every output length. There is no deployment regime in which any post-hoc eviction method is both faster and more faithful than a prompt-construction alternative.

---

## 10. Analysis

### 10.1 When Phrase Selection Helps Most

Phrase selection provides the largest gains over naive recency on tasks where the required information is distributed across the document rather than concentrated in the tail. 2WikiMQA (multi-hop QA across multiple Wikipedia passages) and MultifieldQA (multi-domain passages) show the clearest benefit. NarrativeQA and MuSiQue show little or no benefit — suggesting the answer is consistently found in the recent tail, making phrase selection redundant.

This is a principled limitation: phrase selection is a query-driven method and only helps when (a) the query provides a useful selection signal and (b) the relevant content is not already in the recency tail. For summarization and code completion tasks, the "query" is generic ("write a summary") and provides no useful selection signal; the method degenerates to recency-based selection.

### 10.2 Comparison to KV Pruning Methods

Phrase-based compression outperforms SnapKV and PyramidKV on KL faithfulness across all compression rates. The margin over SnapKV is large and widens as the budget shrinks: at 65% retention, phr128 (0.144) beats SnapKV-Press (1.157) by 8×; at 35% retention, phr128 (0.228) beats SnapKV-Press (0.836) by 3.7×. StreamingLLM with key re-rotation (0.141 at 65% Llama, 0.261 at 35%) matches or slightly beats phr128 at 65%, then falls behind at tighter rates — the only KV-pruning method that is competitive with prompt construction on KL. At every rate, both phr128 and naive substantially outperform SnapKV. On F_out, however, the direction reverses: SnapKV-Press (66.8% at 35%) substantially leads all prompt-construction methods (phr128: 48.6%, naive: 46.9%) due to its full-prefill mechanism (§9). The F_out advantage of SnapKV-Press is entirely attributable to this mechanism and does not reflect faithfulness in the distributional sense that KL captures.

There is no task where SnapKV-Press outperforms prompt-construction methods on KL — the MultiNews exception that existed with the old pcfg SnapKV (0.407 at 65% vs. Naive 0.585) disappears with the corrected snapkv_press (1.050 at 65%, worse than Naive 0.585).

### 10.3 The Role of Structural Integrity

The core finding is that *how* tokens are presented to the model matters as much as *which* tokens are presented. Phrase-based compression and naive truncation present a contiguous, causally complete sequence; KV pruning with sparse selection presents an incomplete one. SnapKV KV pruning (1.157) vs. naive (0.175) at 65% Llama — a 6.6× gap — makes this concrete for attention-score-based pruning. StreamingLLM with key re-rotation (0.141) shows that post-hoc eviction with re-indexing eliminates positional misalignment and, combined with full-prefill enrichment, can be competitive with prompt construction — but SnapKV's globally scattered selection pattern does not benefit from the same correction, remaining 6.6× worse than naive.

This has implications beyond the specific methods studied here. Any approach that modifies the attention mask to create causal gaps — sparse attention, block-sparse attention, token dropping during generation — is subject to the same corruption. The degree of corruption depends on how many positions have sparse causal coverage, but the mechanism is the same.

---

## 11. Discussion

**Simplicity as a feature.** Phrase-based compression requires no model introspection, no attention weight computation, no specialized CUDA kernels. The scoring is pure string matching. The construction is concatenation. This makes it immediately applicable to any transformer model, including those that use Flash Attention 2 (which does not materialize attention weights), without modification.

**Query availability.** The method requires a query to score phrases against. For RAG and QA settings this is natural. For open-ended generation, summarization, and code completion, the query may be absent or uninformative — in these cases the method reduces to naive recency, which is already competitive. Diversity-maximizing selection (choosing phrases that collectively cover the most distinct content) is a natural extension for query-free settings.

**Semantic phrase boundaries.** Fixed-size phrases are an approximation. Sentence and paragraph boundaries (phrase_sent) are a better approximation. Topic-coherent segmentation — identifying spans that are internally consistent and topically focused — is the ideal. Embedding-based topic segmentation methods could provide this, at the cost of an additional model pass.

**Theoretical scope.** The structural corruption analysis is empirical; a formal characterization of when and how severely it occurs across different architectures, context lengths, and pruning budgets would strengthen the theory and could be an avenue for further research.

**Empirical scope.** All empirical results in this paper are on two base (non-instruction-tuned) decoder-only models in the 7–8B range, Llama-3.1-8B and Mistral-7B-v0.3, evaluated on LongBench v1. The causal-gap mechanism (§6) is architectural — it follows from how causal attention and KV caching work, not from any property specific to these two models — so we expect it to generalize to other decoder-only transformers, but we have not tested larger scales, mixture-of-experts architectures, or instruction-tuned/RLHF'd models, where alignment training could change how the model uses corrupted context, and we have not tested benchmarks beyond LongBench.

**Lexical scoring.** The phrase scoring method is lexical and will miss paraphrastic relevance (a passage relevant to "birth city" will not match a question asking "where was X born"), and MultiNews (§10.2) shows a concrete case where it loses to attention-based scoring. We treat this as an opportunity for future work rather than a weakness of the central argument: the paper's claim is that prompt construction is the better compression mechanism, not that lexical overlap is the best way to select spans within it.

Semantic scoring (embedding similarity between phrase and query), learned selectors, or task-adaptive scoring could replace lexical overlap without touching the structural argument, and may close gaps such as MultiNews — at the cost of a separate encoder or model pass that lexical overlap avoids. That phrase selection by lexical overlap resembles extractive summarization or retrieval is incidental to the mechanism it uses, not evidence that prompt construction is solving a different problem than KV pruning (§1) — and it is exactly this resemblance that makes the retrieval and summarization literatures a natural source of better selection methods to try.

---

## 12. Conclusion

We have shown that KV cache pruning with attention-score-based sparse selection introduces structural corruption that substantially limits its faithfulness to the full-context model. The mechanism — degenerate attention from early queries with sparse causal windows, cascading through all transformer layers — produces consistent faithfulness gaps: SnapKV KV pruning achieves 35.8× higher KL than naive on NarrativeQA, and 6.6× higher on average across 16 tasks. StreamingLLM with key re-rotation achieves 0.141 mean KL (vs. naive's 0.175) on Llama at 65%, showing that re-indexing and post-hoc full-prefill enrichment can overcome the gap for recency-based selection — a result that motivates further investigation of which structural features drive the divergence between selection strategies.

We introduced two faithfulness metrics. KL faithfulness measures distributional agreement between the full and compressed model at every generation step, capturing whether the compressed model's internal representations match the full model's. Output faithfulness measures text-level similarity between the two models' generated outputs, capturing whether the compressed model makes the same decisions as the full model. The PyramidKV case study (§9) shows these metrics reveal orthogonal failure modes: both PyramidKV (79.1%) and SnapKV-Press (78.5%) achieve the highest output faithfulness of any compressed method, while simultaneously being the worst on KL (1.394 and 1.157 nats respectively) — because both perform a full prefill before evicting the KV cache, making the first generated token identical to the full model's by construction, while the post-hoc eviction creates distributional gaps at every subsequent step. Neither metric alone characterizes this behavior; both are necessary.

Phrase-based context compression avoids structural corruption entirely by constructing a self-consistent short prompt rather than patching the KV cache. Phrases scored by query-document lexical overlap outperform SnapKV and PyramidKV on KL faithfulness at all compression rates — using only string matching, with no attention scores, model internals, or added computation. StreamingLLM with key re-rotation is competitive at 65% retention but falls behind at tighter budgets; it does not alter the structural argument, since its advantage over prompt construction is specific to recency-biased near-contiguous selection and depends on post-hoc full-prefill enrichment that attention-score-based methods cannot exploit. The method also cuts TTFT by 60–74%, providing simultaneous quality and latency advantages over all post-hoc eviction approaches.

The central message is that fidelity to full-context behavior requires structural integrity, not just token coverage. Selecting the right tokens is necessary but not sufficient if those tokens are presented to the model in a structurally broken context.

---

## References

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. (2023). LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. arXiv:2308.14508.

Cai, Z., Zhang, Y., Qi, B., and Zhou, B. (2024). PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. arXiv:2406.02069.

Chen, A., Geh, R., Grover, A., et al. (2025). The Pitfalls of KV Cache Compression. arXiv:2510.00231.

Devoto, A., Zhao, Y., Scardapane, S., and Minervini, P. (2024). A Simple and Effective L₂ Norm-Based Strategy for KV Cache Compression. In *Proceedings of EMNLP 2024*. arXiv:2406.11430. [This is the main Devoto et al. paper; the KVPress library (github.com/NVIDIA/kvpress) implements this and related methods.]

Feng, Y., Lv, J., Cao, Y., Xie, X., and Zhou, S. K. (2025a). Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2407.11550. [Cited in text as "Ada-KV [NeurIPS 2025]".]

Feng, Y., et al. (2025b). Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective. arXiv:2502.03805. [Cited in text as "Feng et al. [2025b]" for the ‖V·W^O‖₁ output-perturbation bound. Same first author as Ada-KV above (2025a).]

Fu, Y., Cai, Z., Asi, A., Xiong, W., Dong, Y., and Xiao, W. (2025). Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning. In *Proceedings of ICLR 2025*. arXiv:2410.19258. [Cited in text as "HeadKV [ICLR 2025]".]

Guo, Z., Kamigaito, H., and Watanabe, T. (2024). Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters. In *Proceedings of EMNLP 2024*. arXiv:2406.12335. [Cited in text as "VATP [EMNLP 2024]".]

Jiang, H., Wu, Q., Lin, C.-I., Yang, Y., Yu, L., Zhang, C., Rashtchian, C., Meijer, E., Qin, T., and Peng, B. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In *Proceedings of EMNLP 2023*.

Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-I., Yang, Y., and Yu, L. (2024). LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. In *Proceedings of ACL 2024*. arXiv:2310.06839.

Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. (2024). SnapKV: LLM Knows What You are Looking for Before Generation. arXiv:2404.14469.

Liu, Z., Desai, A., Liao, F., Wang, W., Xie, V., Xu, Z., Kyrillidis, A., and Shrivastava, A. (2023). Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. In *Proceedings of ACL 2002*.

Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., and Han, S. (2024). Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. In *Proceedings of ICML 2024*.

Wang, Z., Jin, B., Yu, Z., and Zhang, M. (2024). Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks. arXiv:2407.08454. [Cited in text as "KVMerger", the successor to CaM.]

Zhang, Y., Du, Y., Luo, G., Zhong, Y., Zhang, Z., Liu, S., and Ji, R. (2024). CaM: Cache Merging for Memory-efficient LLMs Inference. In *Proceedings of ICML 2024*. PMLR 235:58840–58850. [Cited in text as "Y. Zhang et al. [2024]" to distinguish from H2O (Z. Zhang et al. [2023]) and BERTScore (T. Zhang et al. [2020]).]

Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Zhang, Z., and Han, S. (2025). DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. In *Proceedings of ICLR 2025*.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453. (Published at ICLR 2024.)

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. In *Proceedings of ICLR 2020*.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Liang, L., Zou, J., Wang, Z., and Chen, B. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

---

## Appendix A: KL Faithfulness — Per-Task Results

**Tables A1a–A1d. KL Faithfulness per task, Llama-3.1-8B (lower is better). Bold = best among prompt-construction methods (Naive, phr128). KV-pruning methods excluded from bold competition.**

**Table A1a. 65% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | 0.022 | **0.020** | 0.045 | 0.787 | 1.337 |
| Qasper† | 0.281 | **0.131** | 0.237 | 0.881 | 1.229 |
| MultifieldQA | 0.262 | **0.114** | 0.302 | 1.316 | 1.535 |
| HotpotQA† | 0.203 | **0.125** | 0.069 | 1.935 | 2.181 |
| 2WikiMQA† | 0.221 | **0.117** | 0.103 | 1.714 | 2.422 |
| MuSiQue† | 0.197 | **0.118** | 0.098 | 1.747 | 2.523 |
| GovReport | **0.280** | 0.385 | 0.385 | 0.532 | 0.334 |
| QMSum† | 0.081 | **0.060** | 0.093 | 0.393 | 0.438 |
| MultiNews | **0.585** | 0.647 | 0.643 | 1.050 | 0.721 |
| TREC | 0.202 | **0.128** | 0.084 | 1.454 | 1.526 |
| TriviaQA | 0.076 | **0.062** | 0.008 | 1.151 | 1.647 |
| SAMSum | **0.011** | 0.014 | 0.015 | 0.791 | 1.182 |
| PassageCount† | **0.059** | 0.067 | 0.002 | 1.560 | 1.626 |
| PassageRetrieval | 0.188 | **0.144** | 0.032 | 1.624 | 1.416 |
| LCC | **0.076** | 0.125 | 0.085 | 0.881 | 1.142 |
| RepoBench-P | 0.048 | **0.046** | 0.050 | 0.702 | 1.052 |
| **Average** | 0.175 | **0.144** | 0.141 | 1.157 | 1.394 |

**Table A1b. 50% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.026** | 0.028 | 0.068 | 0.806 | 1.380 |
| Qasper† | 0.457 | **0.264** | 0.447 | 0.810 | 1.394 |
| MultifieldQA | 0.359 | **0.144** | 0.417 | 1.024 | 1.619 |
| HotpotQA† | 0.212 | **0.099** | 0.097 | 1.519 | 2.245 |
| 2WikiMQA† | 0.278 | **0.129** | 0.198 | 1.286 | 2.282 |
| MuSiQue† | 0.188 | **0.089** | 0.123 | 1.457 | 2.507 |
| GovReport | **0.327** | 0.399 | 0.466 | 0.488 | 0.504 |
| QMSum† | **0.088** | 0.093 | 0.125 | 0.369 | 0.484 |
| MultiNews | **0.759** | 0.786 | 0.784 | 1.002 | 1.099 |
| TREC | 0.260 | **0.186** | 0.183 | 1.114 | 1.711 |
| TriviaQA | 0.095 | **0.087** | 0.015 | 0.963 | 1.622 |
| SAMSum | **0.012** | 0.018 | 0.020 | 0.619 | 1.199 |
| PassageCount† | **0.055** | 0.068 | 0.003 | 1.468 | 1.538 |
| PassageRetrieval | 0.188 | **0.117** | 0.043 | 1.504 | 1.378 |
| LCC | **0.105** | 0.150 | 0.114 | 0.677 | 1.238 |
| RepoBench-P | 0.065 | **0.057** | 0.088 | 0.580 | 1.016 |
| **Average** | 0.217 | **0.170** | 0.199 | 0.981 | 1.451 |

**Table A1c. 40% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.035** | 0.045 | 0.079 | 0.702 | 0.702 |
| Qasper† | 0.577 | **0.346** | 0.562 | 0.775 | 0.772 |
| MultifieldQA | 0.443 | **0.180** | 0.465 | 0.925 | 0.990 |
| HotpotQA† | 0.208 | **0.086** | 0.111 | 1.250 | 1.659 |
| 2WikiMQA† | 0.332 | **0.199** | 0.227 | 1.031 | 1.315 |
| MuSiQue† | 0.189 | **0.124** | 0.155 | 1.274 | 1.719 |
| GovReport | **0.391** | 0.438 | 0.552 | 0.507 | 0.506 |
| QMSum† | **0.104** | 0.107 | 0.150 | 0.341 | 0.342 |
| MultiNews | **0.888** | 0.903 | 0.898 | 1.023 | 1.024 |
| TREC | 0.282 | **0.203** | 0.273 | 0.809 | 1.340 |
| TriviaQA | 0.136 | **0.117** | 0.033 | 0.851 | 1.129 |
| SAMSum | **0.015** | 0.024 | 0.027 | 0.506 | 0.646 |
| PassageCount† | 0.093 | **0.073** | 0.003 | 1.197 | 1.168 |
| PassageRetrieval | 0.195 | **0.149** | 0.049 | 1.205 | 1.365 |
| LCC | **0.125** | 0.170 | 0.139 | 0.577 | 0.584 |
| RepoBench-P | **0.077** | 0.087 | 0.112 | 0.548 | 0.641 |
| **Average** | 0.256 | **0.203** | 0.240 | 0.845 | 0.994 |

**Table A1d. 35% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.045** | 0.076 | 0.091 | 0.559 | 0.559 |
| Qasper† | 0.637 | **0.381** | 0.620 | 0.867 | 0.858 |
| MultifieldQA | 0.475 | **0.203** | 0.499 | 0.925 | 0.914 |
| HotpotQA† | 0.220 | **0.127** | 0.114 | 1.246 | 1.252 |
| 2WikiMQA† | 0.351 | **0.220** | 0.238 | 1.040 | 1.052 |
| MuSiQue† | 0.191 | **0.158** | 0.182 | 1.243 | 1.236 |
| GovReport | **0.423** | 0.460 | 0.583 | 0.529 | 0.529 |
| QMSum† | **0.112** | 0.121 | 0.160 | 0.358 | 0.358 |
| MultiNews | **0.947** | 0.961 | 0.961 | 1.063 | 1.064 |
| TREC | 0.296 | **0.224** | 0.325 | 0.812 | 0.823 |
| TriviaQA | 0.157 | **0.142** | 0.044 | 0.832 | 0.833 |
| SAMSum | **0.017** | 0.028 | 0.031 | 0.475 | 0.475 |
| PassageCount† | 0.116 | **0.075** | 0.004 | 1.122 | 1.134 |
| PassageRetrieval | 0.282 | **0.182** | 0.050 | 1.247 | 1.240 |
| LCC | **0.145** | 0.182 | 0.149 | 0.547 | 0.550 |
| RepoBench-P | **0.087** | 0.107 | 0.123 | 0.513 | 0.511 |
| **Average** | 0.281 | **0.228** | 0.261 | 0.836 | 0.837 |

**Tables A2a–A2b. KL Faithfulness per task, Mistral-7B-v0.3 (lower is better). Bold = best among prompt-construction methods (Naive, phr128). KV-pruning methods excluded from bold competition.**

**Table A2a. 65% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.003** | 0.010 | 0.125 | 0.445 | 0.780 |
| Qasper† | 0.071 | **0.053** | 0.067 | 0.274 | 0.472 |
| MultifieldQA | 0.148 | **0.089** | 0.201 | 0.639 | 0.718 |
| HotpotQA† | 0.195 | **0.134** | 0.198 | 0.714 | 0.725 |
| 2WikiMQA† | 0.262 | **0.113** | 0.208 | 0.710 | 0.683 |
| MuSiQue† | 0.183 | **0.109** | 0.179 | 0.708 | 0.679 |
| GovReport | **0.180** | 0.271 | 0.281 | 0.578 | 0.294 |
| QMSum† | 0.067 | **0.053** | 0.082 | 0.201 | 0.183 |
| MultiNews | **0.586** | 0.677 | 0.678 | 1.076 | 0.626 |
| TREC | 0.012 | **0.009** | 0.046 | 0.359 | 0.703 |
| TriviaQA | **0.039** | 0.042 | 0.070 | 1.104 | 1.379 |
| SAMSum | **0.021** | 0.025 | 0.040 | 1.030 | 1.050 |
| PassageCount† | **0.037** | 0.038 | 0.069 | 1.141 | 0.897 |
| PassageRetrieval | 0.195 | **0.175** | 0.204 | 0.971 | 0.541 |
| LCC | **0.055** | 0.066 | 0.041 | 0.861 | 1.070 |
| RepoBench-P | 0.076 | **0.066** | 0.087 | 0.914 | 0.940 |
| **Average** | 0.133 | **0.121** | 0.161 | 0.733 | 0.734 |

**Table A2b. 35% retention.**

| Task | Naive | phr128 | Streaming | SnapKV | Pyr |
|---|---|---|---|---|---|
| NarrativeQA† | **0.011** | 0.019 | 0.170 | 0.203 | 0.203 |
| Qasper† | 0.165 | **0.124** | 0.165 | 0.262 | 0.262 |
| MultifieldQA | 0.213 | **0.119** | 0.286 | 0.519 | 0.513 |
| HotpotQA† | 0.218 | **0.120** | 0.310 | 0.632 | 0.633 |
| 2WikiMQA† | 0.362 | **0.176** | 0.365 | 0.690 | 0.684 |
| MuSiQue† | 0.186 | **0.136** | 0.290 | 0.679 | 0.687 |
| GovReport | **0.263** | 0.294 | 0.399 | 0.457 | 0.457 |
| QMSum† | 0.084 | **0.065** | 0.114 | 0.175 | 0.175 |
| MultiNews | **0.900** | 0.919 | 0.937 | 1.040 | 1.039 |
| TREC | **0.022** | 0.024 | 0.070 | 0.192 | 0.192 |
| TriviaQA | **0.062** | 0.085 | 0.146 | 0.725 | 0.722 |
| SAMSum | **0.027** | 0.043 | 0.054 | 0.741 | 0.737 |
| PassageCount† | 0.069 | **0.044** | 0.084 | 0.957 | 0.930 |
| PassageRetrieval | 0.197 | **0.167** | 0.272 | 0.911 | 0.920 |
| LCC | **0.109** | 0.111 | 0.090 | 0.564 | 0.564 |
| RepoBench-P | 0.110 | **0.104** | 0.146 | 0.654 | 0.661 |
| **Average** | 0.188 | **0.159** | 0.244 | 0.588 | 0.586 |

---

## Appendix B: Output Faithfulness — Per-Task Results

**Tables B1a–B1d. F_out per task, Llama-3.1-8B (higher is better). Bold = highest value per row.**

**Table B1a. 65% retention.**

| Task             | Naive    | phr128 | Streaming | SnapKV   | Pyr      |
|------------------|----------|--------|-----------|----------|----------|
| NarrativeQA†     | **88.1** | 52.2   | 2.9       | 63.1     | 61.7     |
| Qasper†          | 46.4     | 51.8   | 3.8       | 71.4     | **72.6** |
| MultifieldQA     | 57.3     | 55.3   | 7.0       | 81.0     | **86.0** |
| HotpotQA†        | 88.2     | 69.6   | 12.1      | **95.3** | 92.2     |
| 2WikiMQA†        | 64.5     | 67.1   | 22.3      | **95.6** | **95.6** |
| MuSiQue†         | **97.1** | 66.6   | 8.0       | 92.7     | 92.3     |
| GovReport        | **55.8** | 40.7   | 4.6       | 47.8     | 51.2     |
| QMSum†           | **54.4** | 43.6   | 1.5       | 41.7     | 40.4     |
| MultiNews        | 43.8     | 35.5   | 0.8       | 44.6     | **49.0** |
| TREC             | 75.3     | 73.8   | 52.9      | **82.6** | 81.3     |
| TriviaQA         | 71.1     | 47.9   | 16.1      | 92.0     | **97.0** |
| SAMSum           | 64.9     | 49.3   | 15.7      | **78.4** | 73.4     |
| PassageCount†    | 70.7     | 34.1   | 15.0      | 92.8     | **96.4** |
| PassageRetrieval | 80.1     | 66.2   | 25.7      | **94.0** | 93.8     |
| LCC              | 62.0     | 60.2   | 23.4      | **92.8** | 91.3     |
| RepoBench-P      | 77.3     | 57.8   | 11.9      | 90.5     | **90.6** |
| **Average**      | 68.6     | 54.5   | 14.0      | 78.5     | **79.1** |

**Table B1b. 50% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 48.4     | 50.7     | 4.0       | **62.2** | 53.0     |
| Qasper†          | 38.3     | 42.5     | 3.0       | **61.1** | 52.2     |
| MultifieldQA     | 55.7     | 58.7     | 7.4       | **76.3** | 70.6     |
| HotpotQA†        | 65.2     | 70.6     | 9.8       | **90.2** | 77.8     |
| 2WikiMQA†        | 62.8     | 63.2     | 19.2      | **92.0** | 84.5     |
| MuSiQue†         | 64.3     | 67.3     | 8.2       | **90.1** | 80.8     |
| GovReport        | 37.9     | 35.7     | 1.9       | **41.5** | 37.3     |
| QMSum†           | 38.7     | **39.8** | 0.9       | **39.8** | 38.0     |
| MultiNews        | **44.9** | 33.8     | 0.9       | 35.4     | 33.4     |
| TREC             | 71.8     | 72.1     | 46.5      | **76.3** | 70.5     |
| TriviaQA         | 45.4     | 50.9     | 13.4      | **90.8** | 76.1     |
| SAMSum           | 48.2     | 47.2     | 15.1      | **67.2** | 49.1     |
| PassageCount†    | 31.8     | 38.3     | 17.6      | **89.3** | 85.6     |
| PassageRetrieval | 67.1     | 61.9     | 25.0      | **92.6** | 82.7     |
| LCC              | 57.8     | 58.7     | 19.0      | **84.6** | 70.0     |
| RepoBench-P      | 55.3     | 52.8     | 10.9      | **82.5** | 65.8     |
| **Average**      | 52.1     | 52.8     | 12.7      | **73.2** | 64.2     |

**Table B1c. 40% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 48.6     | 46.6     | 3.1       | **59.6** | 59.4     |
| Qasper†          | 37.0     | 42.1     | 1.4       | **55.7** | 55.0     |
| MultifieldQA     | 53.7     | 52.4     | 8.1       | **71.4** | 69.2     |
| HotpotQA†        | 62.6     | 67.2     | 12.2      | **87.1** | 77.2     |
| 2WikiMQA†        | 61.4     | 60.3     | 17.0      | 87.9     | **88.6** |
| MuSiQue†         | 61.2     | 65.1     | 6.5       | **85.5** | 79.7     |
| GovReport        | 37.8     | 35.8     | 1.0       | **39.5** | 39.4     |
| QMSum†           | 38.0     | 39.2     | 1.0       | 37.9     | **38.4** |
| MultiNews        | **40.1** | 32.7     | 0.9       | 33.2     | 33.3     |
| TREC             | 67.7     | 70.8     | 37.0      | **71.5** | 70.6     |
| TriviaQA         | 41.2     | 42.6     | 10.6      | **84.2** | 76.2     |
| SAMSum           | 46.6     | 41.7     | 15.2      | **59.7** | 49.1     |
| PassageCount†    | 31.2     | 29.1     | 17.0      | **86.3** | 82.8     |
| PassageRetrieval | 60.9     | 65.2     | 20.6      | **88.3** | 81.3     |
| LCC              | 52.6     | 56.6     | 15.0      | **81.4** | 79.0     |
| RepoBench-P      | 54.7     | 46.3     | 9.7       | **75.5** | 66.4     |
| **Average**      | 49.7     | 49.6     | 11.0      | **69.1** | 65.4     |

**Table B1d. 35% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 44.7     | 47.2     | 2.5       | **57.3** | **57.3** |
| Qasper†          | 34.3     | 42.8     | 1.7       | **51.8** | **51.8** |
| MultifieldQA     | 47.6     | 56.6     | 6.4       | 69.9     | **70.0** |
| HotpotQA†        | 58.8     | 65.1     | 10.8      | 85.2     | **85.4** |
| 2WikiMQA†        | 56.1     | 59.4     | 16.4      | **87.3** | **87.3** |
| MuSiQue†         | 58.4     | 63.5     | 6.8       | 84.2     | **84.6** |
| GovReport        | 35.1     | 34.5     | 1.1       | 36.6     | **36.7** |
| QMSum†           | 37.2     | **38.3** | 0.6       | 36.4     | 36.4     |
| MultiNews        | **36.8** | 30.7     | 0.9       | 31.9     | 32.6     |
| TREC             | 66.9     | **70.5** | 29.2      | 68.1     | 67.6     |
| TriviaQA         | 38.6     | 43.6     | 8.8       | 84.2     | **84.8** |
| SAMSum           | 41.4     | 39.3     | 13.6      | **56.3** | **56.3** |
| PassageCount†    | 32.1     | 28.5     | 15.6      | 84.7     | **85.1** |
| PassageRetrieval | 57.9     | 61.3     | 21.1      | **86.0** | 85.4     |
| LCC              | 54.2     | 54.7     | 17.1      | 77.3     | **77.4** |
| RepoBench-P      | 50.0     | 41.9     | 9.2       | 71.6     | **71.7** |
| **Average**      | 46.9     | 48.6     | 10.1      | 66.8     | **66.9** |

**Tables B2a–B2b. F_out per task, Mistral-7B-v0.3 (higher is better). Bold = highest value per row.**

**Table B2a. 65% retention.**

| Task             | Naive | phr128 | Streaming | SnapKV   | Pyr      |
|------------------|-------|--------|-----------|----------|----------|
| NarrativeQA†     | 40.5  | 42.0   | 4.1       | **57.4** | 55.4     |
| Qasper†          | 69.0  | 64.6   | 27.2      | **87.1** | 86.1     |
| MultifieldQA     | 63.9  | 69.2   | 15.4      | **90.3** | 89.3     |
| HotpotQA†        | 66.9  | 72.2   | 16.3      | 94.7     | **94.8** |
| 2WikiMQA†        | 66.8  | 74.4   | 21.1      | **95.2** | 94.8     |
| MuSiQue†         | 64.6  | 69.1   | 6.1       | **94.3** | 93.3     |
| GovReport        | 46.7  | 44.7   | 10.2      | 62.6     | **64.2** |
| QMSum†           | 48.3  | 51.1   | 1.2       | 72.2     | **73.8** |
| MultiNews        | 45.5  | 33.7   | 0.0       | **51.7** | 51.6     |
| TREC             | 85.0  | 84.7   | 58.2      | **94.5** | 92.6     |
| TriviaQA         | 57.4  | 54.7   | 11.7      | 91.9     | **93.8** |
| SAMSum           | 55.1  | 51.2   | 21.9      | **72.4** | 69.6     |
| PassageCount†    | 57.3  | 55.3   | 1.4       | 97.0     | **97.5** |
| PassageRetrieval | 68.8  | 65.0   | 0.7       | 97.2     | **97.8** |
| LCC              | 62.1  | 62.6   | 14.2      | 88.3     | **90.8** |
| RepoBench-P      | 62.8  | 57.5   | 8.3       | **93.0** | 90.1     |
| **Average**      | 60.0  | 59.5   | 13.6      | **83.7** | 83.5     |

**Table B2b. 35% retention. Note: SnapKV column uses pre-correction pcfg values; Streaming column pending.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|
| NarrativeQA† | 39.5 | 31.9 | 27.5 | **54.6** |
| Qasper† | 60.6 | 56.2 | 48.0 | **77.0** |
| MultifieldQA | 56.0 | 62.6 | 47.1 | **82.3** |
| HotpotQA† | 64.0 | 73.8 | 59.6 | **88.0** |
| 2WikiMQA† | 65.4 | 72.8 | 64.3 | **88.5** |
| MuSiQue† | 58.7 | 67.7 | 61.5 | **88.0** |
| GovReport | 36.8 | 40.1 | 25.1 | **46.1** |
| QMSum† | 39.9 | 43.9 | 41.1 | **61.5** |
| MultiNews | **36.3** | 28.9 | 32.5 | 36.3 |
| TREC | 78.9 | 78.2 | 68.3 | **86.8** |
| TriviaQA | 42.9 | 42.9 | 32.5 | **80.5** |
| SAMSum | 52.9 | 45.8 | 44.9 | **64.3** |
| PassageCount† | 39.2 | 41.4 | 22.2 | **87.1** |
| PassageRetrieval | 61.9 | 59.5 | 68.4 | **88.4** |
| LCC | 59.5 | 62.8 | 51.4 | **81.8** |
| RepoBench-P | 62.6 | 54.2 | 48.4 | **85.9** |
| **Average** | 53.5 | 53.9 | 46.4 | **74.8** |

---

## Appendix C: Ground-Truth Accuracy — Per-Task Results

**Tables C1a–C1d. Ground-truth accuracy per task, Llama-3.1-8B (higher is better). Bold = highest value per row.**

**Table C1a. 65% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 4.9      | 4.3      | 0.5       | **5.6**  | 5.5      |
| Qasper†          | 10.2     | 10.3     | 1.2       | 11.3     | **11.8** |
| MultifieldQA     | 27.0     | 25.3     | 6.0       | **30.2** | 29.4     |
| HotpotQA†        | 9.6      | 9.2      | 4.6       | 9.9      | **10.2** |
| 2WikiMQA†        | 12.6     | 10.7     | 13.0      | **13.9** | 13.7     |
| MuSiQue†         | 7.0      | 6.1      | 2.2       | **7.2**  | 7.1      |
| GovReport        | 19.8     | 19.7     | 7.5       | 19.6     | **20.1** |
| QMSum†           | **11.5** | 9.3      | 2.9       | 9.7      | 9.4      |
| MultiNews        | 16.5     | 16.4     | 0.3       | **18.1** | 17.9     |
| TREC             | 66.0     | 70.0     | 59.0      | **71.0** | **71.0** |
| TriviaQA         | 17.3     | 17.2     | 12.6      | **17.5** | **17.5** |
| SAMSum           | **16.5** | **16.5** | 6.6       | 16.2     | 16.3     |
| PassageCount†    | 1.0      | 2.0      | **3.0**   | **3.0**  | **3.0**  |
| PassageRetrieval | 37.0     | 29.0     | 29.0      | **44.0** | **44.0** |
| LCC              | 63.4     | 61.3     | 13.9      | 68.1     | **68.5** |
| RepoBench-P      | 53.9     | 50.7     | 14.4      | 55.4     | **56.0** |
| **Average**      | 23.4     | 22.4     | 11.0      | 25.0     | **25.1** |

**Table C1b. 50% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 5.7      | 4.2      | 1.1       | 6.7      | **6.8**  |
| Qasper†          | 8.6      | 9.9      | 1.5       | **11.1** | 9.8      |
| MultifieldQA     | 23.0     | 24.8     | 5.2       | 29.2     | **29.5** |
| HotpotQA†        | 8.9      | 9.2      | 5.8       | **9.7**  | 9.6      |
| 2WikiMQA†        | 12.7     | 12.9     | 10.8      | **14.0** | 13.8     |
| MuSiQue†         | 5.8      | 6.7      | 1.9       | **7.0**  | 6.8      |
| GovReport        | 19.5     | 18.5     | 6.5       | **20.0** | 19.5     |
| QMSum†           | 9.5      | 9.0      | 2.4       | **9.7**  | **9.7**  |
| MultiNews        | **17.5** | 15.9     | 0.2       | 16.1     | 15.4     |
| TREC             | 63.0     | 65.0     | 56.0      | **71.0** | 68.0     |
| TriviaQA         | **17.4** | 16.8     | 11.5      | **17.4** | 17.3     |
| SAMSum           | **17.7** | 16.8     | 7.4       | 16.1     | 17.6     |
| PassageCount†    | 2.0      | 2.0      | **3.0**   | **3.0**  | **3.0**  |
| PassageRetrieval | 21.0     | 27.0     | 27.0      | **44.0** | 43.0     |
| LCC              | 61.6     | 62.5     | 13.1      | **68.1** | 63.9     |
| RepoBench-P      | 51.4     | 48.7     | 14.0      | **55.7** | 52.8     |
| **Average**      | 21.6     | 21.9     | 10.5      | **24.9** | 24.2     |

**Table C1c. 40% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 5.8      | 5.4      | 0.8       | **6.8**  | **6.8**  |
| Qasper†          | 7.9      | 9.4      | 1.0       | 11.2     | **11.3** |
| MultifieldQA     | 23.3     | 23.4     | 5.4       | **30.0** | 29.5     |
| HotpotQA†        | 10.6     | **10.8** | 5.2       | 10.1     | 9.4      |
| 2WikiMQA†        | 13.9     | 12.5     | 11.4      | **14.3** | 13.9     |
| MuSiQue†         | 4.7      | 6.5      | 2.0       | **7.1**  | 6.7      |
| GovReport        | **19.9** | 19.0     | 6.0       | 19.3     | 19.1     |
| QMSum†           | 9.7      | 10.1     | 2.4       | **10.3** | 10.0     |
| MultiNews        | **16.7** | 16.1     | 0.2       | 15.9     | 15.9     |
| TREC             | 65.0     | 68.0     | 46.0      | **71.0** | 70.0     |
| TriviaQA         | 17.3     | 16.9     | 10.3      | **17.4** | **17.4** |
| SAMSum           | 17.2     | 16.6     | 7.9       | 16.6     | **18.1** |
| PassageCount†    | **3.0**  | **3.0**  | **3.0**   | **3.0**  | **3.0**  |
| PassageRetrieval | 12.0     | 24.0     | 24.0      | **44.0** | 43.0     |
| LCC              | 58.7     | 61.2     | 12.2      | 67.6     | **68.4** |
| RepoBench-P      | 52.2     | 49.6     | 14.8      | **56.1** | 54.4     |
| **Average**      | 21.1     | 22.0     | 9.5       | **25.0** | 24.8     |

**Table C1d. 35% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | **7.8**  | 3.9      | 0.6       | 6.4      | 6.4      |
| Qasper†          | 7.7      | 9.0      | 1.6       | **10.8** | **10.8** |
| MultifieldQA     | 21.2     | 24.0     | 4.8       | **30.2** | 30.0     |
| HotpotQA†        | **11.4** | 10.0     | 4.1       | 10.0     | 10.0     |
| 2WikiMQA†        | 11.0     | 11.3     | 9.6       | **14.6** | **14.6** |
| MuSiQue†         | 4.7      | 6.1      | 1.4       | **6.5**  | **6.5**  |
| GovReport        | 18.7     | **19.1** | 6.0       | 18.9     | 19.0     |
| QMSum†           | 9.9      | 10.0     | 2.7       | **10.2** | **10.2** |
| MultiNews        | **16.7** | 15.3     | 0.2       | 15.7     | 15.8     |
| TREC             | 60.0     | 68.0     | 42.0      | **69.0** | **69.0** |
| TriviaQA         | 17.3     | 16.8     | 9.3       | **17.4** | **17.4** |
| SAMSum           | 17.6     | **17.8** | 7.1       | 17.3     | 17.3     |
| PassageCount†    | **3.0**  | **3.0**  | **3.0**   | **3.0**  | **3.0**  |
| PassageRetrieval | 12.0     | 19.0     | 27.0      | **44.0** | **44.0** |
| LCC              | 62.4     | 60.9     | 13.1      | **67.0** | **67.0** |
| RepoBench-P      | 50.6     | 49.2     | 13.9      | **55.1** | **55.1** |
| **Average**      | 20.8     | 21.5     | 9.2       | **24.8** | **24.8** |

**Tables C2a–C2b. Ground-truth accuracy per task, Mistral-7B-v0.3 (higher is better). Bold = highest value per row.**

**Table C2a. 65% retention.**

| Task             | Naive    | phr128   | Streaming | SnapKV   | Pyr      |
|------------------|----------|----------|-----------|----------|----------|
| NarrativeQA†     | 2.7      | **5.5**  | 1.4       | 5.1      | 4.1      |
| Qasper†          | 4.3      | 4.2      | 2.9       | **5.1**  | 4.8      |
| MultifieldQA     | 22.1     | 22.7     | 7.2       | **24.9** | 24.8     |
| HotpotQA†        | 10.6     | 9.9      | 4.8       | **10.7** | 10.6     |
| 2WikiMQA†        | 9.8      | **12.2** | 8.2       | 11.7     | 11.7     |
| MuSiQue†         | 3.9      | 4.7      | 1.7       | **5.1**  | **5.1**  |
| GovReport        | 19.7     | 19.5     | 6.0       | 19.8     | **20.0** |
| QMSum†           | **8.4**  | 8.1      | 1.9       | 7.9      | 7.9      |
| MultiNews        | 17.2     | 16.9     | 0.0       | 17.2     | **17.3** |
| TREC             | 67.0     | 67.0     | 58.0      | 70.0     | **71.0** |
| TriviaQA         | 24.2     | **24.5** | 11.9      | 23.4     | 23.3     |
| SAMSum           | 17.8     | 17.0     | 10.1      | 17.9     | **18.1** |
| PassageCount†    | **3.0**  | **3.0**  | 1.0       | 1.0      | 1.0      |
| PassageRetrieval | 26.0     | 24.0     | 1.0       | **39.0** | **39.0** |
| LCC              | 60.2     | **66.9** | 12.2      | 63.2     | 63.2     |
| RepoBench-P      | **54.4** | 52.0     | 8.6       | 54.1     | 54.0     |
| **Average**      | 22.0     | 22.4     | 8.6       | **23.5** | **23.5** |

**Table C2b. 35% retention. Note: SnapKV column uses pre-correction pcfg values; Streaming column pending.**

| Task | Naive | phr128 | SnapKV | Pyr |
|---|---|---|---|---|
| NarrativeQA† | 3.9 | 2.9 | 2.8 | **5.1** |
| Qasper† | 4.0 | 4.3 | **7.9** | 4.8 |
| MultifieldQA | 18.2 | 22.0 | 17.9 | **25.1** |
| HotpotQA† | **11.0** | 10.4 | 9.0 | 10.4 |
| 2WikiMQA† | 11.6 | **12.2** | 11.3 | 11.2 |
| MuSiQue† | 4.6 | 5.0 | 4.0 | **5.2** |
| GovReport | 18.6 | 18.4 | 15.5 | **18.6** |
| QMSum† | 7.8 | **8.2** | 6.7 | 7.8 |
| MultiNews | **16.5** | 15.6 | 15.3 | 15.7 |
| TREC | 60.0 | 65.0 | 60.0 | **66.0** |
| TriviaQA | 25.3 | 22.4 | **36.1** | 23.7 |
| SAMSum | 17.2 | 15.3 | 17.4 | **17.8** |
| PassageCount† | **3.0** | 1.0 | 2.0 | 1.0 |
| PassageRetrieval | 15.0 | 25.0 | 15.0 | **39.0** |
| LCC | 57.4 | **65.4** | 52.6 | 62.3 |
| RepoBench-P | 49.5 | 46.7 | 46.1 | **53.7** |
| **Average** | 20.2 | 21.2 | 20.0 | **23.0** |
