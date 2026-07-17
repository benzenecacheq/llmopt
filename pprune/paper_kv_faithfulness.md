# Faithfulness over Accuracy: Rethinking KV Cache Compression

---

## Abstract

KV cache compression is motivated by a simple observation: long-context  inference is expensive, and many tokens in the prompt are not equally  important for generating a good response. A large body of work has  therefore focused on identifying which tokens matter — using accumulated attention weights [Zhang et al., 2023], pooled key-query alignment [Li  et al., 2024], value norms [Feng et al., 2025b], or combinations thereof — and evicting the rest before or during generation.

Standard benchmarks measure whether a compressed model gives the correct answer according to ground-truth labels. We argue this is the wrong  objective: the goal of compression should be approximation fidelity — producing the same output the full-context model would have produced. In this  paper we introduce two faithfulness metrics: *KL Faithfulness*, which measures divergence from full-context token distributions, and *Output Faithfulness*, which measures text-level similarity between the full model's generated output and the compressed model's generated output. Ground-truth rankings and faithfulness rankings may disagree substantially, and the two new metrics reveal complementary failure modes which we analyze in detail and which suggest improvements to the current popular pruning methods.

Using KL faithfulness as our primary lens, we observe that SnapKV and PyramidKV, which among the algorithms we tested yielded the best ground-truth results, are dramatically less faithful than a simple naïve proportional truncation — retaining the last 65% of tokens as a new, self-consistent prompt.  Deeper analysis shows that this is not because of poor token selection, but because retaining tokens at their original RoPE positions after eviction causes positional displacement that corrupts every subsequent decode step. StreamingLLM with key re-rotation is the best existing method, posting the lowest KL faithfulness among currently deployed KV cache compression approaches.

This diagnosis suggests a fix: apply key re-rotation to SnapKV after eviction, remapping retained keys to their new compact positions. The improvement is dramatic — 27× reduction in mean KL at 65% retention, making SnapKV+rot the most faithful method overall, outperforming all other methods tested. The central practical message is that SnapKV and PyramidKV as currently deployed are substantially less faithful than they could be: key re-rotation is a single post-eviction tensor operation that recovers most of the faithfulness loss at negligible cost.

---

## 1. Introduction

At inference time, a transformer decoder maintains a key-value cache that grows linearly with sequence length. For a model with *L* layers, *H* heads, head dimension *d*, and sequence length *T*, the KV cache occupies *2LHdT* values. At long contexts (32K–128K tokens), this cache dominates GPU memory and limits batch size. KV cache compression reduces this cost by retaining only a subset of the cache entries.

Existing methods differ primarily in how they score tokens. H2O [Zhang et al., 2023] accumulates attention scores during prefill to identify "heavy hitter" tokens. SnapKV [Li et al., 2024] pools attention weights from the last observation window of queries over all key positions. StreamingLLM [Xiao et al., 2023] retains a fixed set of attention sink tokens plus a recency window. All of these methods share the same mechanism: they process the full prompt, compute importance scores, then evict a subset of the tokens from the KV cache, attending only to the retained subset during generation.

These compression methods are typically evaluated by running both the compressed and uncompressed models on benchmark tasks, scoring each against labeled ground truth, and measuring how much accuracy the compressed model retains [Bai et al., 2023; Zhang et al., 2023; Li et al., 2024]. This evaluation paradigm has a fundamental mismatch with what should be the actual goal of compression: an approximation method succeeds when it produces the same output the original model would have produced. If the original model gives a wrong answer, the compressed model should also give that same wrong answer — replicating the full model's behavior should be the criterion, not improving on it.

A further limitation applies to the GT evaluation paradigm regardless of model choice. The base models evaluated here (Llama-3.1-8B and Mistral-7B-v0.3) achieve macro-average GT scores of approximately 25 out of 100 on LongBench. The instruction-tuned models used by the most directly comparable compression papers score higher — PyramidKV reports full-context baselines of 41.46 for LLaMa-3-8B-Instruct and 39.76 for Mistral-7B-Instruct [Cai et al., 2024] — but the fundamental problem is the same. GT evaluates each model's output against the reference independently and says nothing about whether the compressed and uncompressed models agree with each other; that comparison is simply absent from the metric in every case. Even at a baseline of 40, roughly 60% of what both models generate does not overlap with the ground-truth reference, and for that majority of output, the metric is entirely blind to whether compression changed the model's behavior. The compressed and uncompressed models could produce identical text or completely different text in those cases, and the metric would give the same score either way.

We argue that post-prefill KV eviction has a structural flaw independent of how well any particular scoring method identifies important tokens. After a full, unmodified prefill, evicted KV pairs are removed — but the retained keys keep their original RoPE position encodings. The retained keys are now scattered across the original position space: rather than presenting to the model as a compact M-token context at positions 0…M−1, they carry the large, irregular position IDs they held in the original T-token sequence. Because RoPE encodes position by rotating key and query vectors according to their position ID [Su et al., 2024], those wrong IDs produce wrong rotations — and therefore wrong attention weights — for every retained key at every decode step. The resulting errors propagate through all subsequent transformer layers.

This effect is easy to confirm empirically. We compare naive proportional truncation (retaining the last 65% of tokens as a new, self-consistent prompt) with attention-score-based KV pruning retaining 65% of key-value states (§3.1). On NarrativeQA, naive truncation achieves KL faithfulness of 0.022; SnapKV KV pruning achieves 0.787 — a 35.8× gap on this task, and 6.6× on average across 16 tasks (0.175 vs. 1.157). §6.2's same-selection comparison (SnapKV vs. SnapKV-Select) isolates causal gaps alone, holding token selection identical, and shows the same direction of effect.

The fix is direct: remap the retained keys from their original scattered positions to compact positions 0…M−1 immediately after eviction. The retained context now presents to the model as a normal M-token sequence; the enriched value representations from the full prefill are preserved. SnapKV with this correction (SnapKV+rot) reduces mean KL by 27× at 65% retention and outperforms every prompt-construction method.

All post-prefill eviction methods must process the full uncompressed prompt before pruning, so none can reduce TTFT — and in fact all add overhead above the full-context baseline. Key re-rotation eliminates structural corruption and is the appropriate correction for any deployment where decode throughput is the bottleneck and TTFT is acceptable.

We make the following contributions:

- **KL faithfulness metric (F_KL)**: KL divergence between the full and compressed model's next-token distributions at every generation step, averaged over a shared generation prefix (§5.2). This is the primary metric throughout the paper — it measures whether the compressed model's computation matches the full model's at every step, not just whether the final output is correct.

- **Output faithfulness metric (F_out)**: Word-level F1 between the full model's generated output and the compressed model's generated output, with no external ground-truth reference (§5.3). F_out reveals the behavioral inversion in §8: SnapKV and PyramidKV achieve the highest F_out of any method while scoring worst on F_KL — a divergence that F_KL correctly diagnoses as structural corruption, not faithfulness.

- **Structural corruption diagnosis**: Positional displacement — retained tokens keeping their original RoPE encodings after eviction — is the dominant failure mode for scattered-selection KV methods. §6.3 demonstrates this with a controlled gap-geometry ablation (Table 2); §6.4 confirms it with a 27× KL improvement when re-rotation is applied to SnapKV.

- **Key re-rotation as the fix**: Applying `KeyRerotationPress` to SnapKV after eviction reduces mean KL from 1.157 to 0.043 at 65% retention — the largest improvement of any intervention in this study and better than all prompt-construction methods (§6.4). Applying the same correction to PyramidKV (`Pyr+rot`) confirms positional displacement is the dominant failure mode there too: at 65% retention Pyr+rot (0.047) matches SnapKV+rot (0.043) almost exactly. The budget structure introduces additional cost only at 50% retention, where the pyramid reaches its minimum allocation floor at the top layer, and disappears again at 35% when the budget collapses to uniform (§6.4).

## 2. Background and Related Work

**KV cache eviction.** Scissorhands [Liu et al., 2023] introduced the *persistence of importance* hypothesis: tokens that accumulate high attention during prefill remain important throughout generation, so the attention-score history is a reliable eviction criterion. H2O [Zhang et al., 2023] operationalizes this by maintaining a running sum of per-head attention scores and evicting the lowest-scoring tokens. SnapKV [Li et al., 2024] refines the scoring by pooling attention weights from only the last *w* queries (an observation window), better reflecting the queries that will matter at decode time, using post-RoPE vectors. StreamingLLM [Xiao et al., 2023] abandons score computation entirely, retaining a fixed set of attention sink tokens plus a sliding recency window; it achieves low latency at the cost of discarding all non-recent non-sink content.

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

**Prompt-construction baselines.** These methods truncate the prompt to form a shorter, self-consistent input with no KV patching or attention mask modification.
- *Naive proportional truncation (Naive)*: the prompt is split into a 10% head (literal first tokens) and a 90% recency tail, concatenated to form a new prompt at 65% of the original length.

**KV pruning methods.** These methods process the full prompt, score each token's KV entry, and retain only the top-scoring positions at their original sequence indices. The causal attention mask is then reconstructed so that each query can only attend to retained positions at earlier indices. All KV pruning methods unconditionally retain the first 16 tokens (`always_keep_first=16`) and the last 16 tokens (`always_keep_last=16`). For GQA models such as Llama-3.1-8B (32 Q-heads, 8 KV-heads), scores are computed per Q-head and aggregated via max-pooling across heads sharing a KV-head before selection.

- *SnapKV*: attention weights pooled over the last 128-token observation window [Li et al., 2024].
- *Streaming (sink + recency, with key re-rotation)*: first 4 attention sink tokens plus a recency window to fill the remaining budget, following StreamingLLM's selection rule [Xiao et al., 2023]. Implemented via `kvpress`'s `StreamingLLMPress` wrapped in `KeyRerotationPress`, which remaps retained keys onto compact RoPE positions after eviction — the defining correctness-motivated feature of the published method. Like PyramidKV, this uses post-prefill cache eviction (§6.1): a full, unrestricted prefill completes before the cache is trimmed, so retained tokens' hidden states are computed with access to the complete context.

### 3.2 Setup

**Models.** Llama-3.1-8B [Dubey et al., 2024] and Mistral-7B-v0.3 [Jiang et al., 2023b] (both base, not instruction-tuned) in fp16 on a single V100 32GB GPU.

**Benchmark.** LongBench v1 [Bai et al., 2023], 16 English tasks: single-document QA (NarrativeQA, Qasper, MultifieldQA), multi-document QA (HotpotQA, 2WikiMQA, MuSiQue), summarization (GovReport, QMSum, MultiNews), few-shot tasks (TREC, TriviaQA, SAMSum), synthetic tasks (PassageCount, PassageRetrieval), and code completion (LCC, RepoBench-P). 100 examples per task.

**Hyperparameters.** Retention fraction r=0.65, always_keep_first=16, always_keep_last=16, q_buffer_size=128.

---

## 4. Ground-Truth Evaluation

Before introducing our faithfulness metrics, we establish what the standard evaluation paradigm reveals. We run both Llama-3.1-8B and Mistral-7B-v0.3 on LongBench v1 (16 tasks, 100 examples each) at 65% retention, comparing naive proportional truncation, SnapKV, StreamingLLM (Streaming), and PyramidKV against the full uncompressed model.

**Setup.** Full details in §3.2. All methods target r = 0.65 retention. Naive truncation keeps the first 6.5% and last 58.5% of tokens as a new self-consistent prompt. PyramidKV uses a layer-adaptive pyramid budget on top of SnapKV scoring. Ground-truth scores use the standard LongBench metrics: F1 for QA tasks, ROUGE for summarization, exact match for classification and code completion. Per-task ground-truth results across compression rates (65/50/35%) are given in Appendix C.

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

The overall pattern is the same across both architectures: ground-truth scores are compressed across methods. On both Llama and Mistral, SnapKV ties PyramidKV (Llama: 25.0 vs 25.1; Mistral: 23.5 vs 23.5), both exceeding naive. Streaming (18.5 Llama, 17.8 Mistral) sits below SnapKV and PyramidKV but substantially above naive truncation on average, though with a highly uneven task profile: recency bias helps summarization, classification, and code tasks (which rely on the recent document tail) while nearly zeroing out fact-retrieval tasks (NarrativeQA, Qasper, HotpotQA, TriviaQA). PyramidKV leads or ties all compressed methods on both models.

Taken at face value, PyramidKV is the clear winner. But §5 argues these scores measure the wrong thing — and §8 shows that PyramidKV's ground-truth advantage conceals a faithfulness cost an order of magnitude larger than any other method's.

---

## 5. Faithfulness Metrics

### 5.1 Why Ground-Truth Evaluation Falls Short

Standard benchmarks evaluate compression by comparing what the compressed model generates against human-labeled reference answers — fixed labels that were written without reference to what any particular model would say. We instead compare the compressed model's outputs against what the full, uncompressed model would have generated on the same inputs. The question we ask is not whether the compressed model gets the right answer, but whether it behaves the same way the full model would have behaved.

Ground-truth evaluation has four distinct failure modes as a compression metric.

**Reference mismatch.** For the four ROUGE-based summarization tasks in LongBench (GovReport, QMSum, MultiNews, SAMSum), ground-truth is computed against a human-written reference. Two methods can generate completely different summaries with identical ROUGE scores, as long as both cover the same facts. Whether one summary resembles what the full model would have written is invisible to the metric. For the seven F1-based QA tasks, ground-truth measures string overlap with a gold answer extracted from the dataset — again with no reference to the full model's actual behavior. Only for classification and retrieval tasks does ground truth directly constrain the model's output to match specific tokens.

**Faithful errors are penalized.** If the full-context model produces a wrong answer, a compressed model that faithfully reproduces that same wrong answer scores 0 on ground truth. The metric penalizes faithfulness when it produces the wrong answer for the "right reasons." A compression method should not be penalized for being a good approximation.

**Accidental correctness is rewarded.** Conversely, a compressed model that accidentally produces the correct answer — because truncation happened to leave in the relevant span, or because a different reasoning path led to the same answer — scores the same as a method that genuinely approximates the full model. Ground truth cannot distinguish these cases.

**Most of these metrics are not testing correctness at all.** Of the 16 tasks, only three (TREC, PassageCount, PassageRetrieval) score a prediction as binary right or wrong. The other thirteen — seven QA tasks scored by word-overlap F1, four summarization tasks scored by ROUGE-L, two code-completion tasks scored by string similarity — give continuous partial credit for overlap with one human-written reference, with no notion of "correct" or "incorrect" built into the metric itself. A summary that captures the source faithfully but phrases it differently than the dataset's one reference summary, or a QA answer that conveys the right information in different words, can score below an answer that shares more surface vocabulary with the reference while being less accurate. This does not mean the metric carries no signal — overlap with a reasonable reference is correlated with quality — but for most of this benchmark, "ground-truth accuracy" is closer to similarity to one human's phrasing than to a judgment of whether the model got the answer right.

**Low full-model accuracy amplifies the first three problems.** On six of our 16 tasks the full model's ground-truth accuracy is below 15 (NarrativeQA: 5.5, MuSiQue: 6.9, PassageCount: 3.0, HotpotQA: 9.9, Qasper: 11.1, QMSum: 10.3). On NarrativeQA, for example, the full model gets 94.5% of examples wrong. A perfectly faithful compressed method should also get 94.5% wrong; ground truth scores all of those as 0. The compressed method's GT score on that task then reflects only how close its wrong answers happen to come to the gold reference — a function of failure-mode similarity to the reference, not of approximation quality. As a consequence, GT differences between compression methods on low-accuracy tasks are dominated by noise: small random variations in which wrong answers partially overlap with the gold string. The signal about actual compression quality is vanishingly small precisely on the tasks where the method has the most work to do.

These failure modes matter in practice. PyramidKV's ground-truth advantage (§4) is exactly the kind of result that is hard to interpret: it may reflect genuine preservation of task-relevant information, or it may reflect a combination of structural properties that happen to produce correct first tokens without approximating the full model's behavior. §8 shows it is the latter.

This points to a more general limitation, beyond the three failure modes above. Ground truth — or any outcome metric computed on a fixed test distribution, whether benchmark accuracy or a downstream business metric — measures *that* a method produced acceptable outputs on the examples tested, not *why*. That distinction matters because it determines whether a result generalizes. An outcome metric tells us nothing about behavior on inputs outside the test distribution; it is silent on mechanism by construction. 

PyramidKV and SnapKV are the clearest illustrations in this paper: by ground truth and output faithfulness both lead all tested methods, but §8 shows this is substantially an artifact of how post-prefill eviction methods compute their answer — a full, uncompressed prefill makes the first generated token identical to the full model's by construction, which dominates the aggregate score because the benchmark is full of short-answer tasks where the first token is the answer. A practitioner who only had the outcome metric would not know this, and would have no way to predict whether the advantage holds for a task with longer outputs, a different query distribution, or any input not resembling the test set — and indeed it mostly does not (Table 5, §8.1).

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

These can disagree. A method could have low KL (distributions match well at each step) but low F_out (due to early divergence in sampled text). Conversely, a method could have high F_out (same final answers) but high KL (reached those answers via corrupted intermediate distributions). §8 shows that PyramidKV falls into this second category, making both metrics jointly necessary to characterize it.

**Why these two metrics.** Other distributional or behavioral metrics are possible — Jensen-Shannon divergence (bounded and symmetric, but without KL's direct expected-surprise interpretation), rank correlation between full and compressed top-k token rankings (sensitive to ordering, blind to probability mass), logit cosine similarity (no information-theoretic meaning, sensitive to logit scale), or calibration error against ground truth (reintroduces the external reference that both metrics are designed to avoid).

We use KL because it has a direct information-theoretic interpretation — the expected excess surprise from using the compressed model's distribution in place of the full model's — and because it decomposes cleanly per generation step, which is what makes the per-step analysis in §8 possible.

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

Values in nats; lower is better. Bold marks the best among Naive and Streaming (SnapKV and PyramidKV excluded from bold competition — both are substantially worse). Streaming leads on 8 of 16 tasks and the overall average; Naive leads on 8. SnapKV and PyramidKV are an order of magnitude worse than either Naive or Streaming.

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

Values in nats; lower is better. Bold marks the best among Naive and Streaming. Naive leads on 12 of 16 tasks; Streaming leads on 4. SnapKV (0.733) and PyramidKV (0.734) are similarly poor, both 5.5× worse than Naive (0.133).

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

The pattern is consistent across architectures. On Mistral, SnapKV (83.7%) and PyramidKV (83.5%) are effectively tied on F_out — the same reversal as Llama relative to KL — while Streaming (61.0%) and Naive (60.0%) cluster together, confirming the KV-pruning F_out advantage holds across both models.

---

## 6. Structural Corruption in KV Cache Pruning

### 6.1 The Mechanism

All three KV-pruning methods benchmarked in this paper — SnapKV, PyramidKV, and Streaming-rerotated — use *post-prefill cache eviction*: the model performs one normal, unrestricted prefill pass over the complete prompt, attending to the full context at every layer, and only after that pass completes is the KV cache trimmed before the first decode step. Retained keys and values are computed cleanly under full attention; nothing during prefill is modified. The earliest a gap can affect anything is at T=1 — the first decode step, when the newly generated query attends back over the now-sparse cache. §8's case study shows the empirical signature: KL(T=0) = 0, a clean prefill, followed by a sharp spike at T=1.

**Prefill enrichment.** Post-prefill eviction has a second consequence beyond gap timing: the KV pairs that survive carry representations computed under full-context attention over the entire original sequence, including attention to positions that are immediately evicted once the forward pass completes. A token retained in a post-prefill-evicted cache has attended to every other token in the document at every layer; a token retained in a compressed prompt has never attended to the evicted content at any layer. Prompt construction retains the right tokens in a structurally clean input; post-prefill eviction retains the right tokens in a structurally gapped input but with richer representations of each. The enrichment benefit accrues equally to all three post-prefill methods regardless of their selection policies. Whether it outweighs the decode-time gap cost depends on how many gaps remain and how they are distributed — examined in §6.4.

**Gap corruption.** Once a decode-time query attends over the sparse cache, the degenerate attention distribution does not stay local. Transformer hidden states are deeply entangled across layers: layer *l*'s computation at every position depends on the full layer *l*-1 output across all positions. A corrupted hidden state is carried forward into subsequent decode steps and, after *T* further steps, has diffused throughout the representation. This is not primarily an *early-token* problem. It is a *gap* problem: any portion of the retained set where positions are sparse will produce corrupted attention whenever a decode query crosses it. Recency-biased selection concentrates the budget at the tail, leaving the head and middle with severe gaps; attention-score-based selection distributes retained positions more evenly across the full sequence, leaving gaps throughout. The mechanism is identical in both cases; §6.4 examines how gap density and distribution determine whether enrichment or gap damage dominates.

**Positional misalignment.** KV pruning introduces a second structural problem independent of causal gaps: retained tokens keep their original RoPE encodings, so two tokens logically adjacent in the retained set may carry a relative distance of hundreds or thousands of positions. The attention mechanism was trained on sequences where positional distance reflects informational proximity; retained tokens with large RoPE gaps between them are out-of-distribution at every decode step. Key re-rotation (`KeyRerotationPress`) addresses positional misalignment by re-encoding retained keys at compact positions 0…M−1 after eviction, leaving the gap structure and enriched value representations unchanged. Prompt construction eliminates positional misalignment together with causal gaps by constructing a structurally self-consistent new sequence. §6.4 applies key re-rotation to all three benchmarked methods — SnapKV, Streaming, and PyramidKV — to determine whether positional displacement is the dominant failure mode in each case.

**A note on recompute-over-gaps.** A different implementation pathway — reconstructing the causal mask and re-deriving attention under the reduced position set during prefill itself — appears in two controlled experiments below (§6.2, §6.3) but is not used by any benchmarked method. It appears in those experiments because it enables tighter isolation: §6.2 holds token selection fixed to isolate mechanism, and §6.3 varies gap geometry with no selection at all. Both experiments use this pathway specifically because it confines the gap-exposure question to the prompt, independent of the enrichment effect that post-prefill eviction introduces at decode time.

### 6.2 Empirical Evidence

To confirm that mechanism rather than token selection drives the faithfulness gap, we compare SnapKV KV pruning to SnapKV-Select — identical attention-score-based token selection, presented as a new prompt rather than patched into the KV cache (Table 1). The same tokens, selected by the same algorithm, behave very differently depending on how they are presented to the model.

**Table 1. Mean KL faithfulness for four method configurations. Llama-3.1-8B, 65% retention, 6 tasks, n = 100 per task. All three SnapKV rows use identical attention-score-based token selection; only the mechanism varies.**

| Method | 2WikiMQA | MultifieldQA | Qasper | QMSum | RepoBench-P | TriviaQA | Mean |
|---|---|---|---|---|---|---|---|
| Naive — prompt construction | 0.221 | 0.262 | 0.281 | 0.081 | **0.048** | 0.076 | 0.162 |
| SnapKV (pcfg) — KV pruning (recompute-over-gaps) | 0.254 | 0.186 | 0.223 | 0.117 | 0.106 | 0.122 | 0.168 |
| SnapKV — KV pruning (post-prefill eviction) | 1.714 | 1.316 | 0.881 | 0.393 | 0.702 | 1.151 | 1.026 |
| SnapKV-Select — prompt construction | **0.096** | **0.118** | **0.163** | **0.071** | 0.051 | **0.060** | **0.093** |

SnapKV-Select (0.093) is 11× better than SnapKV (1.026) using the same selected tokens — the gap is caused by the mechanism, not token selection. SnapKV with recompute-over-gaps (0.168) roughly matches naive truncation despite its expensive scoring, confirming that the post-prefill eviction pathway is the specific source of damage. 

StreamingLLM with key re-rotation goes the other direction. We ran the same test, comparing Streaming with KV cache pruning with Streaming-Select, which created a new gapless prompt using the same token selection rules.  In this case, Streaming KV pruning (0.141) outperforms both naive (0.175) and Streaming-select (0.197), presumably because enrichment from the full prefill exceeds the small positional displacement cost of a single recency gap. §6.4 explains why gap geometry determines which effect dominates.

### 6.3 Synthetic Gap-Structure Analysis

To isolate positional displacement from enrichment effects without contamination from token selection, we evaluate four pure position geometries generated purely as a function of sequence length and retention fraction (r = 0.65), with no query, no content, and no model access. At a typical context length of ~4000 tokens, the geometries are:

- **block_end** — one contiguous retained block at the tail (1 causal gap, immediately before it)
- **block_mid** — one contiguous retained block, centered (2 gaps, before and after)
- **clustered4** — four evenly-spaced contiguous blocks (~5 gaps)
- **scattered** — evenly-spaced single-token positions (~1400 gaps — the maximally scattered case)

Each geometry evaluates the same retained position set under three presentations:

- **Gapless**: positions gathered into a new, shorter, contiguous prompt — compact RoPE positions, no enrichment from evicted context, no KV patching.
- **Evicted**: full prefill over the complete prompt (enabling full-context enrichment), then post-prefill eviction; retained keys remain at their original RoPE positions. This is the mechanism used by SnapKV.
- **Rerotated**: identical to evicted but with key re-rotation to compact RoPE positions after eviction (`KeyRerotationPress`). This is the mechanism used by StreamingLLM.

Comparing evicted to gapless isolates the joint effect of enrichment and positional displacement. Comparing rerotated to evicted isolates position correction alone, since enrichment is identical in both conditions.

**Table 2. Mean KL faithfulness by gap geometry (Llama-3.1-8B, r = 0.65, 6 tasks, n = 20/task).**

| Geometry | Gaps | Gapless | Evicted | Rerotated | ev/gl | rot/gl |
|---|---|---|---|---|---|---|
| block_end | 1 | 0.177 | 3.493 | 2.814 | 19.8× | 15.9× |
| block_mid | 2 | 3.127 | 4.225 | 3.973 | 1.4× | 1.3× |
| clustered4 | ~5 | 2.745 | 3.765 | 3.498 | 1.4× | 1.3× |
| scattered | ~1400 | 1.205 | 1.289 | 0.221 | 1.1× | 0.18× |

**Scattered geometry: position correction is the dominant factor.** For scattered positions, evicted (1.289) and gapless (1.205) are nearly identical (1.1×) — the enrichment benefit from the full prefill and the positional displacement cost from the scattered original RoPE positions roughly cancel. Re-rotation eliminates the positional displacement and produces a 5.8× improvement: rerotated (0.221) is 0.18× of evicted. The enrichment that was present but ineffective in the evicted condition becomes fully effective once positions are corrected. Scattered + rerotated is the best-performing condition in the table, beating gapless by 5.4× despite retaining exactly the same positions.

**Block_end geometry: structural corruption, not positional displacement.** For block_end, evicted (3.493) is 19.8× worse than gapless (0.177). The retained block is nearly contiguous — only 1 gap — so there is almost nothing for re-rotation to fix on the position side, and accordingly rerotated (2.814) barely improves on evicted. The large evicted-vs-gapless gap reflects something re-rotation cannot address: during the full prefill, the retained tail tokens attended to the entire evicted head and formed key/value representations shaped by content that no longer exists in the KV cache. This structural corruption is baked into the value vectors and persists regardless of what key positions are assigned afterward. Gapless (0.177) avoids it entirely by never exposing the model to the evicted content in the first place.

**Block_mid and clustered4** show modest damage in both evicted (1.4×) and rerotated (1.3×) relative to gapless, with re-rotation providing only a small benefit. The geometries are not directly comparable in absolute KL: block_mid and clustered4 discard the recency tail, so their elevated gapless baselines reflect lower content informativeness, not mechanism damage. Within-geometry ratios are the interpretable signal.

These results make a concrete prediction for SnapKV and StreamingLLM, whose retained-set geometries are approximately scattered and block_end respectively: SnapKV should benefit enormously from re-rotation, while StreamingLLM should gain little from it — its recency tail is nearly contiguous, leaving minimal positional displacement to correct, and the dominant cost is the structural corruption its retained tokens incur by attending to the large evicted head during prefill. §6.4 tests both predictions.

### 6.4 Re-rotation Confirms Positional Displacement as the Dominant Failure Mode

**SnapKV.** SnapKV's retained positions are globally scattered — an approximate realization of the scattered geometry in Table 2 — so the synthetic analysis estimates roughly a 6× improvement from re-rotation. `snapkv_rerotated` tests this directly by applying `KeyRerotationPress` after post-prefill eviction, re-encoding retained keys at compact positions 0…M−1 while leaving the enriched value representations unchanged. Across all 16 LongBench tasks on Llama-3.1-8B:

| Rate | SnapKV | SnapKV+rot | Improvement |
|---|---|---|---|
| 65% | 1.157 | 0.043 | 27× |
| 50% | 0.981 | 0.077 | 13× |
| 35% | 0.836 | 0.124 | 6.7× |

The prediction is confirmed, and the scale of the improvement exceeds the synthetic estimate. At 65% retention, re-rotation reduces mean KL by 27× — the largest gain of any intervention tested in this study. SnapKV+rot (0.043) not only surpasses unrotated SnapKV (1.157) but outperforms every other method: Naive (0.175), Streaming (0.141), and PyramidKV (1.394). The scattered retained set that was the worst-performing KV configuration in §6.3 becomes the best-performing method overall once positions are corrected. Positional displacement was not a secondary cost for SnapKV; it was the dominant failure mode.

The improvement narrows as retention decreases (27× → 13× → 6.7×). A residual cost grows as the budget shrinks: at 35% retention M ≈ 1400 tokens while the full context T ≈ 4000. Decoding from compact position M shifts the model's positional frame of reference to a shorter-than-actual context. This effect is small at 65% (M ≈ 2600, close to T) and grows at 35%, placing a floor on how much re-rotation alone can recover at tight budgets.

**Streaming.** The block_end prediction from §6.3 is also confirmed. Unrotated streaming (mean KL 0.270) improves to 0.141 with re-rotation — a 1.9× gain, versus SnapKV's 27×. The smaller gain reflects streaming's near-contiguous retained geometry: with a single large gap between the sink tokens and the recency tail, there is far less positional displacement to correct than in the scattered case. The remaining gap between streaming_rerotated (0.141) and SnapKV+rot (0.043) reflects structural corruption that re-rotation cannot address — streaming's recency tail attended to the large evicted middle section during prefill, and those corrupted value representations persist regardless of key position reassignment.

**PyramidKV.** PyramidKV's per-layer budget allocates from ≈99% at layer 0 down to ≈31% at layer 31 for 65% overall retention. Applying right-aligned re-rotation (see below) tests whether this budget asymmetry imposes a faithfulness cost beyond positional displacement. The results across all three rates:

| Rate | Pyr | Pyr+rot | SnapKV+rot |
|---|---|---|---|
| 65% | 1.394 | 0.047 | 0.043 |
| 50% | 1.451 | 0.096 | 0.077 |
| 35% | 0.837 | 0.124 | 0.124 |

At **65%**, Pyr+rot (0.047) tracks SnapKV+rot (0.043) almost exactly; on several individual tasks — `gov_report`, `2wikimqa`, `qmsum`, and `trec` — Pyr+rot marginally beats SnapKV+rot. The per-layer budget gradient (99%→31%) imposes negligible additional faithfulness cost once positions are corrected. Positional displacement was the dominant failure mode for PyramidKV as well.

At **50%**, a real gap opens: Pyr+rot (0.096) is 25% above SnapKV+rot (0.077). The pyramid budget is at its steepest at this rate: the `PyramidKVPress` formula reaches its `window_size=64` floor at layer 31, leaving the top layer with ≈64 retained tokens — approximately 1.3% of context. Re-rotation correctly reassigns positions for whatever is retained, but cannot recover information from tokens that were evicted. This is genuine information loss, not positional displacement.

At **35%**, the gap disappears entirely: both methods report 0.124. The `min_num` clamp forces PyramidKV to revert to uniform per-layer allocation identical to SnapKV at this rate, so Pyr+rot and SnapKV+rot run the same effective budget and produce identical KL.

The three-rate pattern pinpoints what re-rotation can and cannot address. It eliminates positional displacement regardless of budget geometry. When the budget is mild enough not to cause information loss (65%) or has collapsed to uniform (35%), Pyr+rot matches SnapKV+rot. When the pyramid is at its steepest (50%), the remaining gap is the irreducible cost of missing information in the most-aggressively-pruned upper layers — not positions.

**F_out tradeoff.** Re-rotation changes the decode position for each method differently. SnapKV+rot remaps retained keys to compact positions 0…M−1 and decodes from M < T, shifting the model's positional frame. Pyr+rot uses right-aligned re-rotation — all layers share endpoint T−1 and decoding begins at T — matching the full model's positional frame (see Table 4 for full results).

At 65%, Pyr+rot F_out (71.6%) and SnapKV+rot F_out (71.1%) are nearly identical, both substantially below unrotated SnapKV (78.5%) and PyramidKV (79.1%). The gap relative to unrotated methods reflects the content cost of subset retention: unrotated methods decode from T with the complete enriched KV cache, while the re-rotated variants decode with a sparser set regardless of positional frame.

At 50%, Pyr+rot F_out (61.3%) drops sharply below SnapKV+rot (67.8%), tracking the KL gap: the same upper-layer information loss that widens KL at 50% degrades generation quality by the same mechanism. SnapKV+rot (67.8%) exceeds unrotated PyramidKV (64.2%) at this rate — the only rate where this holds — as PyramidKV's mid-transition produces mixed per-layer budgets where neither the pyramid nor the uniform budget character is fully realized.

At 35%, both converge: Pyr+rot (63.0%) = SnapKV+rot (63.0%), as expected from the budget collapse. Streaming_rerotated trails substantially at all rates (61.2% / 56.4% / 51.8%), consistent with structural corruption rather than positional displacement as the dominant failure mode.

**Right-aligned re-rotation.** The naive adaptation of `KeyRerotationPress` to PyramidKV maps each layer's retained keys to `[0, n_kept)`. Because `n_kept` differs per layer under the pyramid budget, no single decode position is correct for all layers simultaneously: a query at layer-0's M (≈98% of T) is out-of-distribution for layer 31 (M ≈ 31% of T), producing catastrophic degenerate output on long-form tasks. The fix is right-aligned re-rotation: target `[T − n_kept, T)` so all layers share endpoint T−1, and the decode query arrives at position T for every layer.

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

SnapKV+rot achieves dramatically lower KL than any other method at every rate. Re-rotating retained keys to compact positions after eviction reduces mean KL from 1.157 to 0.043 at 65% retention — a 27× improvement.

SnapKV without re-rotation shows a counterintuitive cross-rate trajectory: mean KL *improves* from 1.157 at 65% to 0.836 at 35% — as retention shrinks, the protected window constitutes a larger share of the surviving cache, reducing sparse-attention damage. Even so, SnapKV at its best (0.836 at 35%) remains 6.7× worse than SnapKV+rot at the same budget (0.124).

PyramidKV shows a similar counterintuitive trajectory: mean KL is 1.394 at 65%, rises to 1.451 at 50%, then drops to 0.837 at 35%. We traced this to the layer-budget formula itself (`kvpress`'s `PyramidKVPress`): the per-layer budget is `max_num - layer_idx * steps`, clamped to [`window_size`, sequence length]. At 65% and 50% the clamp binds and produces a genuine pyramid; at 35%, `min_num` falls below `window_size` for nearly every sequence length in our range, tripping the fallback clause and returning a *uniform* per-layer budget — PyramidKV ceasing to be a pyramid and reverting to plain SnapKV-style allocation. PyramidKV at its best (0.837 at 35%) is 6.7× worse than SnapKV+rot at the same budget (0.124).

**Mistral-7B-v0.3** (65% and 35% retention).

| Retention | Naive | Streaming | SnapKV | SnapKV+rot | Pyr | Pyr+rot |
|---|---|---|---|---|---|---|
| 65% | 0.133 | 0.161 | 0.733 | **0.030** | 0.734 | 0.034 |
| 35% | 0.188 | 0.244 | 0.588 | **0.096** | 0.586 | **0.096** |

The Mistral rankings replicate the Llama pattern. SnapKV+rot leads at both rates (0.030 at 65%, 0.096 at 35%), outperforming all other methods by the same wide margin as on Llama. Pyr+rot at 65% (0.034) tracks SnapKV+rot closely, matching the Llama finding (0.047 vs 0.043). At 35%, Pyr+rot ties SnapKV+rot exactly (0.096) on all 16 tasks — the budget collapse to uniform allocation is total on Mistral as well. Unlike Llama, Streaming (0.161) trails Naive (0.133) on Mistral at every rate — Naive leads on 12 of 16 tasks (§5.5). SnapKV and PyramidKV show the same budget-tightening improvement (SnapKV 0.733→0.588, Pyr 0.734→0.586), and both remain far behind SnapKV+rot at every rate.

Mistral SnapKV+rot F_out (78.2%/74.3%/70.1% at 65/50/35%) mirrors the Llama pattern: strong improvement over prompt-construction methods, below unrotated SnapKV/PyramidKV. Pyr+rot (77.0%/66.5%/70.0%) tracks SnapKV+rot closely at 65% and 35%, and diverges at 50% by the same mechanism as Llama (Table 4).

### 7.3 Inference Performance

Compression is only worthwhile if it makes inference faster. We measure two quantities: **time to first token (TTFT)**, the wall-clock time from receiving a prompt to producing the first output token (dominated by the prefill pass), and **time per output token (TPT)**, the mean decode step time (dominated by KV cache memory bandwidth). All measurements are on a single V100 32GB GPU using Llama-3.1-8B fp16. PyramidKV: n=100/task (500 total); SnapKV and Streaming: n=20/task (100 total).

**Mean TTFT by retention rate** (Full context baseline: 6348 ms):

| Retention | SnapKV        | Streaming     | Pyr           |
|-----------|---------------|---------------|---------------|
| 65%       | 6621 ms (+4%) | 6796 ms (+7%) | 7022 ms (+11%) |
| 50%       | 6872 ms (+8%) | 6750 ms (+6%) | 7109 ms (+12%) |
| 35%       | 6834 ms (+8%) | 6861 ms (+8%) | 7143 ms (+13%) |

All post-prefill eviction methods incur overhead above the full-context baseline: performing a full prefill over the uncompressed prompt before pruning, each is consistently slower than the uncompressed model at every retention rate. PyramidKV's layer-wise budget computation adds 11–13% overhead. SnapKV and Streaming-rerotated use simpler post-prefill eviction steps and add less overhead (+4–8%), but both are consistently above full-context TTFT. No KV-pruning method using post-prefill eviction can provide prefill savings: the full prompt must be processed regardless. SnapKV+rot inherits the same timing profile as unrotated SnapKV: key re-rotation is a single O(M·H·D) tensor operation applied once to the M retained keys after eviction, adding negligible overhead (<10ms) that does not affect the TTFT or TPT figures above.

**Decode speed (TPT)** improves for all compressed methods because fewer retained tokens means a smaller KV cache to scan at each decode step. At 65% retention, mean TPT drops from 67.5 ms/tok (full) to ~54 ms/tok (~20% faster) for all methods including PyramidKV, snapkv_press, and streaming_rerotated; at 35% retention, all compressed methods reach ~44–48 ms/tok (~29–35% faster).

### 7.4 Output Faithfulness Results

Output Faithfulness (F_out) measures how similar the compressed model's generated text is to the full model's generated text (§5.3). Higher is better. All comparisons are within-model: Llama compressed methods are compared to Llama full context; Mistral compressed methods are compared to Mistral full context.

**Table 4. Mean F_out by compression rate (higher is better). Bold = highest value per row. SnapKV+rot = SnapKV with key re-rotation; Pyr+rot = PyramidKV with key re-rotation. Per-task breakdown in Appendix B.**

**Llama-3.1-8B.**

| Retention | Naive | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|-----------|-------|-----------|----------|------------|----------|---------|
| 65%       | 68.6  | 61.2      | 78.5     | 71.1       | **79.1** | 71.6    |
| 50%       | 52.1  | 56.4      | **73.2** | 67.8       | 64.2     | 61.3    |
| 35%       | 46.9  | 51.8      | 66.8     | 63.0       | **66.9** | 63.0    |

**Cross-rate summary.**

SnapKV and PyramidKV lead F_out at all rates without re-rotation, for the same mechanistic reason: a full, uncompressed prefill means the first generated token's distribution is identical to the full model's by construction (§8.1). PyramidKV leads at 65% and 35% (79.1%, 66.9%); SnapKV leads at 50% (73.2%). Both converge at 35%, consistent with PyramidKV collapsing to a uniform budget at tight compression (§7.2).

SnapKV+rot (71.1%, 67.8%, 63.0%) falls below the post-prefill eviction leaders but substantially above Naive. At 50% it exceeds PyramidKV (67.8 vs 64.2). Pyr+rot (71.6%, 61.3%, 63.0%) closely tracks SnapKV+rot at 65% and 35%, but drops 6.5 points behind at 50% — the same upper-layer information loss that widens KL at 50% degrades F_out by the same mechanism.

Streaming achieves 61.2% / 56.4% / 51.8% (65/50/35%), consistently above Naive but well below the post-prefill eviction leaders.

**Mistral-7B-v0.3.**

| Retention | Naive | Streaming | SnapKV   | SnapKV+rot | Pyr      | Pyr+rot |
|-----------|-------|-----------|----------|------------|----------|---------|
| 65%       | 61.1  | 61.0      | 83.7     | 78.2       | **83.9** | 77.0    |
| 50%       | 56.9  | 57.4      | **79.0** | 74.3       | 68.8     | 66.5    |
| 35%       | 53.5  | 53.9      | 74.0     | 70.1       | **74.8** | 70.0    |

Mistral shows the same qualitative pattern as Llama. PyramidKV leads F_out at 65% and 35% (83.9%, 74.8%); SnapKV leads at 50% (79.0%), consistent with PyramidKV's budget collapse to uniform allocation at tight compression. Pyr+rot (77.0%, 66.5%, 70.0%) closely tracks SnapKV+rot (78.2%, 74.3%, 70.1%) at 65% and 35%, but falls behind at 50% — the same upper-layer information loss as on Llama. SnapKV+rot is below unrotated SnapKV/Pyr but above Naive and Streaming at all rates.

**What F_out and KL together reveal.** SnapKV and PyramidKV lead F_out while scoring worst on KL — the first-token-advantage mechanism explains this divergence (§8). SnapKV+rot breaks the pattern: it has the best KL of any method (Table 3) *and* F_out substantially above Naive and Streaming, at the cost of some F_out relative to plain SnapKV/Pyr, which retain the full-prefill first-token advantage that re-rotation sacrifices by decoding from compact position M rather than original position T.

---

## 8. Why Post-Prefill KV Eviction Leads F_out but Trails KL

The most striking pattern in Tables 3 and 4 is a metric inversion: PyramidKV and SnapKV have the highest F_out of any method (79.1% and 78.5% at Llama 65%) while simultaneously having the worst KL (1.394 and 1.157 nats) — worse than every other method including naive truncation (0.175). This section explains the mechanism that produces this inversion, and where SnapKV+rot and Streaming fit within it.

**The first-token advantage.** All post-prefill eviction methods run a complete, uncompressed prefill before evicting the KV cache. The first output token's distribution is therefore always exactly identical to the full model's: eviction has not yet occurred at that point, so the hidden state at the last prefill position is computed from full, unrestricted attention over the entire context. Divergence begins at the second decode step, when the compressed cache takes over. For SnapKV and PyramidKV, where the retained cache consists of the tokens that received the highest attention from the query window, this divergence is small on short-answer tasks: evicted tokens are precisely those with low attention weight, so the compressed cache closely approximates the full model's for the few additional steps a short answer requires. On tasks where the answer is determined by the first one or two tokens, post-prefill eviction methods therefore replicate the full model's output nearly exactly.

KL is measured at every generation step. Post-prefill eviction creates gaps in the retained cache that corrupt distributions at all subsequent steps, even when the first output is correct. On short-answer tasks, F_out is determined almost entirely by the first-token decision and registers no corruption at subsequent steps. This is the source of the inversion: both methods produce the same output text as the full model while assigning completely different probability mass to alternatives at every subsequent step.

**Mechanistic confirmation.** That the short-answer advantage comes from the full-prefill hidden state — not from query-relevant token selection — is confirmed directly by §6.2 (Table 1). Comparing `snapkv_press` (post-prefill eviction) to the recompute-over-gaps variant of SnapKV from that table isolates this precisely: both use identical attention-score selection at the same retention budget; only the eviction point differs. The recompute-over-gaps variant restricts attention *during* prefill rather than pruning the cache after it, so the full-prefill hidden state never exists. recompute-over-gaps SnapKV: Short=56.1%, Long=50.2%, Gap=6.0 — nearly flat, no short-answer advantage. snapkv_press: Short=87.7%, Long=54.7%, Gap=33.0 — the same steep gradient as PyramidKV. The same token selection rule applied at a different point in the forward pass produces a completely different behavioral signature.

**PyramidKV–SnapKV convergence (unrotated).** Without re-rotation, both methods produce nearly identical results across every metric: GT 25.1 vs 25.0, F_out 79.1% vs 78.5%, KL 1.394 vs 1.157 nats at 65%. PyramidKV's more aggressively compressed upper layers produce slightly larger KL distortion, but the difference is negligible relative to the gap from prompt-construction baselines (naive: 0.175). The shared post-prefill eviction mechanism dominates both methods' behavioral signature at the unrotated baseline.

**Key re-rotation modulates the first-token advantage.** SnapKV+rot applies re-rotation after eviction, compacting retained keys to positions 0, 1, ..., M−1. The first output token y*[0] remains exactly identical to the full model's — it is generated at the end of the prefill from full, unrestricted attention, before re-rotation takes effect. The disruption begins at the second output token: generating y*[1] requires feeding y*[0] back into the model at compact position M rather than at the original prefill endpoint T, introducing a positional discontinuity at the prefill-to-decode boundary. Table 5 shows the result: SnapKV+rot Short=79.3%, below unrotated SnapKV (87.7%) but above Naive (72.8%). In exchange, KL drops from 1.157 to 0.043 — a 27× improvement — because positional displacement at subsequent decode steps is eliminated. Re-rotation trades a small portion of short-answer F_out accuracy for greatly improved distributional fidelity at all post-eviction steps.

**Streaming and the limits of the first-token advantage.** Streaming-rerotated uses the same post-prefill eviction timing — and therefore also produces a first output token that is exactly identical to the full model's. The cause of its poor performance lies elsewhere: streaming's retained set is structurally a block_end configuration (§6.3, Table 2) — a few sink tokens plus a recency tail, with evicted content massed at the head of the sequence. As Table 2 shows, block_end geometry produces severe KL degradation that re-rotation cannot fix: the retained tokens attended to the large evicted head during prefill, corrupting their value representations in ways that persist regardless of key position reassignment. With a nearly-contiguous retained block there is almost no positional displacement to correct, so re-rotation provides only a 1.9× KL improvement for streaming (0.270 → 0.141) versus 27× for SnapKV. Table 5 shows the result: Streaming short-answer F_out at 66.7%, below Naive (72.8%) and well below SnapKV/Pyr (~88%).

**KL recovery under teacher forcing.** A natural question is whether KL corruption compounds over a long response. Per-step KL trajectories on gov_report (n=32, Llama 65%) show the opposite: after the T0→T1 spike, KL falls sharply — from ~1.00 over the first 50 generated tokens to ~0.09 over the last 150 of a ~500-token response. This is not the corruption healing; it reflects a structural property of the teacher-forced evaluation. The compressed prompt is fixed at M tokens, but at each generation step t the model also attends to the preceding t−1 unpruned y* tokens. As t grows, the unpruned portion of the context grows with it: by step 400 of a 500-token response, most of the model's attended context is uncompressed ground-truth output. Predictions late in the sequence are increasingly made from an uncompressed window, naturally reducing KL regardless of how corrupted the original prompt representation is. In free-running generation, nothing keeps the model on the true continuation after an early divergence, so the damage concentrated at the first few post-eviction steps is precisely the damage that compounds in practice — which is why F_out degrades with output length even as teacher-forced KL appears to recover.

### 8.1 What F_out Reveals

Table 5 compares all methods by output-length category (Llama, 65%), with Naive as a reference.

**Table 5. F_out by length category (Llama, 65%). Short = full-model reference output ≤ 90 words (n=1155); Long = > 90 words (n=445). Streaming = streaming_rerotated.**

| Method      | Short-answer F_out | Long-form F_out | Gap  |
|-------------|---------------------|-----------------|------|
| PyramidKV   | **88.1**            | 55.5            | 32.6 |
| SnapKV      | 87.7                | 54.7            | 33.0 |
| SnapKV+rot  | 79.3                | 50.0            | 29.3 |
| Naive       | 72.8                | **57.6**        | 15.2 |
| Streaming   | 66.7                | 47.0            | 19.7 |

PyramidKV and SnapKV show nearly identical short-to-long drops (~33 points), more than twice Naive's (15.2). Naive slightly leads both on long-form F_out (57.6 vs 55.5 and 54.7), and their headline aggregate F_out numbers are propped up almost entirely by short, structured outputs where the first-token advantage is decisive. Both methods achieve near-full-model performance on single-token or short-phrase tasks: TriviaQA (Pyr: 97.0%, Snap: 92.0%), PassageCount (96.4%, 92.8%), 2WikiMQA (95.6%, 95.6%), PassageRetrieval (93.8%, 94.0%), LCC (91.3%, 92.8%), HotpotQA (92.2%, 95.3%).

SnapKV+rot's profile: Short=79.3%, Long=50.0%, Gap=29.3. The short-answer reduction relative to unrotated SnapKV (87.7%) is consistent with the positional discontinuity introduced at y*[1] when decoding starts at compact position M rather than T (§8). The long-form reduction relative to unrotated SnapKV (54.7%) and Pyr (55.5%) is harder to account for: re-rotation produces a 27× KL improvement, yet long-form F_out is lower than either unrotated method. We have no satisfying explanation for this. Its headline aggregate (71.1%) is above Naive (68.6%) on short tasks but slightly below Naive (57.6%) on long tasks.

Streaming-rerotated's profile is distinct. Short F_out (66.7%) is below Naive (72.8%), showing that recency selection does not capture query-relevant positions reliably enough to match even naive truncation on short answers. Long F_out (47.0%) is the lowest of any method — below Naive (57.6%), SnapKV (54.7%), and Pyr (55.5%). The gap (19.7) is larger than Naive's (15.2) but much smaller than SnapKV/Pyr's (~33), consistent with partial recency coverage providing some protection without systematic query-focused retention. Streaming has neither the short-answer advantage of attention-score selection nor the long-form robustness of naive truncation, and trails Naive at both output lengths.

The practical implication is direct. Post-prefill eviction cannot reduce TTFT, so SnapKV+rot's quality advantage is concentrated precisely where latency savings matter least: short exact-lookup tasks where generation time is negligible and prefill dominates total cost. On long-form tasks, SnapKV and Pyr fall to or below Naive's long-form F_out (Naive: 57.6%, Pyr: 55.5%, SnapKV: 54.7%) while Naive delivers a 20% decode speedup with no prefill overhead or KL corruption. There is no deployment regime in which any post-prefill eviction method is simultaneously much faster than uncompressed and more faithful than naive truncation.


---

## 9. Discussion

**Re-rotation as a practical fix.** The most direct implication of this paper is operational: SnapKV and PyramidKV are deployed without key re-rotation, and our results show re-rotation reduces mean KL by 27× at 65% retention at negligible cost (a single O(M·H·D) tensor operation after eviction). Any deployment that already uses SnapKV or PyramidKV can add `KeyRerotationPress` as a drop-in and substantially improve distributional faithfulness with no change to inference infrastructure, retention rate, or TTFT. The tradeoff is a moderate reduction in output faithfulness (F_out) due to the positional frame shift (§6.4), but the gain in KL faithfulness is larger than any other intervention in this study.

**Budget allocation and re-rotation interact.** Re-rotation corrects the positions of whatever tokens survive eviction, but cannot recover information from tokens that were never retained. PyramidKV's per-layer budget allocates tokens in a pyramid shape: lower layers retain nearly all context while upper layers are cut most aggressively. At 65% retention Pyr+rot (0.047) matches SnapKV+rot (0.043) almost exactly — the budget gradient imposes negligible additional faithfulness cost at moderate retention. At 50%, where the pyramid reaches its `window_size` floor at the top layer (≈1.3% of context retained there), Pyr+rot (0.096) diverges from SnapKV+rot (0.077): re-rotation correctly assigns positions, but the evicted tokens are simply gone. At 35%, the `min_num` clamp forces uniform allocation and the gap closes exactly (both 0.124). Streaming shows the analogous constraint from geometry: recency-only selection concentrates evictions at the head, producing block_end gap structure that re-rotation cannot repair regardless of budget.

**Empirical scope.** All empirical results are on two base (non-instruction-tuned) decoder-only models in the 7–8B range, Llama-3.1-8B and Mistral-7B-v0.3, evaluated on LongBench v1. The KL faithfulness findings should generalize to any decoder-only transformer: the causal-gap and positional-displacement mechanisms are properties of the attention architecture, and KL is a logit-level comparison that does not depend on model-specific generation behavior. F_out results are more model-dependent — instruction tuning, scale, and benchmark distribution could all shift the specific numbers — and we have not tested larger scales, mixture-of-experts architectures, or instruction-tuned models.

---

## 10. Conclusion

We introduced two faithfulness metrics that reveal complementary failure modes. KL faithfulness measures distributional agreement at every generation step; output faithfulness measures text-level similarity to the full model's generated output. The PyramidKV case study (§8) exposes a striking inversion: PyramidKV (79.1% F_out) and SnapKV (78.5%) are the top two methods on output faithfulness while simultaneously being the worst on KL (1.394 and 1.157 nats). Full-prefill enrichment makes the first generated token nearly identical to the full model's by construction; post-prefill eviction then degrades every subsequent step. Neither metric alone captures this behavior; both are necessary to characterize what a compression method actually does.

The primary finding of this paper is a diagnosis and a fix. SnapKV and PyramidKV — the most widely deployed attention-score-based KV compression methods — achieve mean KL faithfulness 6.6× worse than naïve truncation on Llama-3.1-8B, despite selecting tokens specifically chosen to be important to the query. The failure is not selection but positional displacement: retained tokens keep their original RoPE encodings after eviction, placing them at out-of-distribution positional distances from every decode query. Applying key re-rotation after eviction — remapping retained keys to compact positions 0…M−1 — reduces mean KL by 27× at 65% retention (from 1.157 to 0.043), making SnapKV+rot the most faithful method in the study, outperforming all prompt-construction baselines. The fix is a single tensor operation adding negligible overhead.

Re-rotation addresses positional displacement regardless of budget geometry, but cannot recover evicted information. Pyr+rot is within measurement noise of SnapKV+rot at 65% (0.047 vs 0.043) and ties exactly at 35% when the pyramid budget collapses to uniform. A gap appears only at 50%, where the pyramid reaches its minimum allocation floor at the top layer, causing genuine information loss rather than positional error. Streaming shows the analogous constraint from geometry: recency-only selection produces block_end gap structure that re-rotation cannot repair regardless of per-layer allocation.

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

**Tables A2a–A2b. KL Faithfulness per task, Mistral-7B-v0.3 (lower is better). Bold = best per row.**

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

**Table A2b. 35% retention.**

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
