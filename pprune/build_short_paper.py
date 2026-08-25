#!/usr/bin/env python3
"""Assemble paper_kv_short.tex from paper_kv_faithfulness.tex."""

with open("paper_kv_faithfulness.tex", "r") as f:
    orig = f.readlines()

# 1-indexed access helper
def L(start, end):
    """Lines start..end inclusive, 1-indexed."""
    return orig[start - 1 : end]


import re as _re

def text_subs(lines):
    """Apply cross-reference renumbering across the whole output."""
    text = "".join(lines)

    # ── Fix dangling colon after Caveat emptor bullet list was removed ──
    text = text.replace(
        "generated output-based measurement, with appropriate skepticism:\n",
        "generated output-based measurement, with appropriate skepticism.\n"
    )

    # ── Content fixes after removing "Accidental correctness" bullet ──
    text = text.replace(
        "Ground-truth evaluation has five distinct failure modes as a compression\nmetric.",
        "Ground-truth evaluation has four distinct failure modes as a compression metric."
    )
    text = text.replace(
        r"\textbf{Low full-model accuracy amplifies the first three problems.}",
        r"\textbf{Low full-model accuracy amplifies these problems.}"
    )

    # ── Section heading text replacements (must come before §-reference substitution) ──
    heading_subs = [
        ("7. Main Experiments", "4. Experiments"),
        ("7.1 Setup", "4.1 Setup"),
        ("7.2 KL Faithfulness: Main", "4.3 KL Faithfulness: Main"),
        ("7.3 Output Faithfulness\nResults", "4.4 Output Faithfulness\nResults"),
        ("7.3 Output Faithfulness", "4.4 Output Faithfulness"),
        ("7.4 Inference Performance", "4.5 Inference Performance"),
        ("6. Structural Corruption", "5. Structural Corruption"),
        ("6.1 The Mechanism", "5.1 The Mechanism"),
        ("6.2 Empirical Evidence", "5.2 Empirical Evidence"),
        ("6.3 Synthetic Gap-Structure", "5.3 Synthetic Gap-Structure"),
        ("5. Faithfulness Metrics", "3. Faithfulness Metrics"),
        ("5.1 Why Ground-Truth Evaluation Falls", "3.1 Why Ground-Truth Evaluation Falls"),
        ("5.2 KL Faithfulness", "3.2 KL Faithfulness"),
        ("5.3 Output Faithfulness", "3.3 Output Faithfulness"),
        ("8. Why Post-Prefill KV Eviction Without", "6. Why Post-Prefill KV Eviction Without"),
        ("8.1 What", "6.1 What"),
        ("10. Conclusion", "7. Conclusion"),
        # Remove the forward-ref to §3 since §3 content is now merged into §4
        ("Full implementation details in §3.\n", ""),
    ]
    for old, new in heading_subs:
        text = text.replace(old, new)

    # ── §-reference substitution in a single regex pass (avoids cascading) ──
    # Map OLD number → NEW number
    sec_map = {
        "5.1": "3.1", "5.2": "3.2", "5.3": "3.3", "5.4": "3.4", "5.5": "3.5",
        "6.1": "5.1", "6.2": "5.2", "6.3": "5.3",
        "7.1": "4.1", "7.2": "4.3", "7.3": "4.4", "7.4": "4.5",
        "8.1": "6.1",
        "3": "4",    # old §3 Initial Experiments → now merged into §4
        "4": "4.2",  # old §4 GT Evaluation → now §4.2 Ground-Truth Results
        "5": "3",
        "6": "5",
        "7": "4",
        "8": "6",
        "10": "7",
    }

    def replace_ref(m):
        num = m.group(1)
        return "§" + sec_map.get(num, num)

    # Match §N.M first (longer), then bare §N
    text = _re.sub(r"§(\d+\.\d+|\d+)", replace_ref, text)

    return text


# ── Tables and prose written inline ─────────────────────────────────────────

GT_ALLRATES_TABLE = r"""
\begin{table}[H]
\centering
\small
\captionsetup{justification=centering}
\caption{Mean ground-truth accuracy by method and retention rate (per-task breakdown in Appendix~C)}
\label{tab:gt-allrates}
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Llama-3.1-8B}\\[0.4em]
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & 65\% & 50\% & 35\%\\
\midrule
Naive      & 23.4 & ---  & ---\\
Streaming  & 23.7 & 23.6 & 23.1\\
SnapKV     & 25.0 & 24.9 & 24.8\\
SnapKV+rot & 24.8 & 25.0 & 24.4\\
Pyr        & 25.1 & 24.2 & 24.8\\
Pyr+rot    & 25.0 & 24.0 & 24.4\\
\bottomrule
\end{tabular}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Mistral-7B-v0.3}\\[0.4em]
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & 65\% & 50\% & 35\%\\
\midrule
Naive      & 22.0 & 21.1 & 20.2\\
Streaming  & 22.9 & 22.1 & 21.2\\
SnapKV     & 23.5 & 23.3 & 23.0\\
SnapKV+rot & 22.8 & 22.7 & 22.3\\
Pyr        & 23.5 & 22.9 & 23.0\\
Pyr+rot    & 22.9 & 22.1 & 22.3\\
\bottomrule
\end{tabular}
\end{minipage}
\par\smallskip
Full model (rate-independent): Llama~25.0, Mistral~23.6.
Naive at 50\%/35\% not evaluated for Llama (---).
\end{table}
"""

INSTRUCT_GT_TABLE = r"""
\begin{table}[H]
\centering
\small
\captionsetup{justification=centering}
\caption{Mean ground-truth accuracy on instruction-tuned models at absolute token budgets}
\label{tab:instruct-gt}
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Llama-3.1-8B-Instruct}\\[0.4em]
\begin{tabular}{@{}lrr@{}}
\toprule
Method & $b$=256 & $b$=1024\\
\midrule
Naive      & 16.9 & 20.1\\
Streaming  & 26.4 & 26.9\\
Strm+rot   & 26.4 & 26.9\\
SnapKV     & 30.7 & 31.5\\
SnapKV+rot & 30.8 & 31.4\\
Pyr        & \textbf{31.2} & \textbf{31.6}\\
Pyr+rot    & 30.9 & 31.4\\
\midrule
Full       & 31.7 & 31.7\\
\bottomrule
\end{tabular}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Mistral-7B-Instruct-v0.3}\\[0.4em]
\begin{tabular}{@{}lrr@{}}
\toprule
Method & $b$=256 & $b$=1024\\
\midrule
Naive      & 16.7 & 20.9\\
Streaming  & 23.8 & 26.0\\
Strm+rot   & 23.5 & 25.8\\
SnapKV     & 31.3 & \textbf{32.2}\\
SnapKV+rot & 30.1 & 31.5\\
Pyr        & \textbf{31.6} & 32.1\\
Pyr+rot    & 30.2 & 31.5\\
\midrule
Full       & 32.8 & 28.9\\
\bottomrule
\end{tabular}
\end{minipage}
\par\smallskip
\textbf{Bold} = best compressed method per column. Full context is the reference;
at Mistral $b$=1024, compressed selection methods exceed the full-model average
(focused-context effect on classification tasks). Per-task results in Appendix~D.
\end{table}
"""

INSTRUCT_KL_TABLE = r"""
\begin{table}[H]
\centering
\small
\captionsetup{justification=centering}
\caption{Mean KL faithfulness on instruction-tuned models at absolute token budgets
         (per-task breakdown in Appendix~D)}
\label{tab:instruct-kl}
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Llama-3.1-8B-Instruct}\\[0.4em]
\begin{tabular}{@{}lrr@{}}
\toprule
Method & $b$=256 & $b$=1024\\
\midrule
Naive      & 2.003 & 1.158\\
Streaming  & 1.133 & 1.043\\
Strm+rot   & 0.735 & 0.542\\
SnapKV     & 0.769 & 0.648\\
SnapKV+rot & \textbf{0.352} & \textbf{0.146}\\
Pyr        & 0.882 & 0.652\\
Pyr+rot    & 0.385 & 0.174\\
\bottomrule
\end{tabular}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
\textbf{Mistral-7B-Instruct-v0.3}\\[0.4em]
\begin{tabular}{@{}lrr@{}}
\toprule
Method & $b$=256 & $b$=1024\\
\midrule
Naive      & 2.561 & 1.532\\
Streaming  & 2.076 & 1.912\\
Strm+rot   & 1.207 & 0.894\\
SnapKV     & 1.912 & 1.766\\
SnapKV+rot & \textbf{0.614} & \textbf{0.294}\\
Pyr        & 1.872 & 1.596\\
Pyr+rot    & 0.689 & 0.349\\
\bottomrule
\end{tabular}
\end{minipage}
\par\smallskip
Lower is better. \textbf{Bold} = best per budget. Both models evaluated on 16 LongBench tasks,
$n$=100 examples per task.
\end{table}
"""

INSTRUCT_KL_INTRO = (
    r"\textbf{Instruct models at extreme budgets.} "
    r"Table~\ref{tab:instruct-kl} extends the comparison to instruction-tuned models "
    r"at absolute token budgets of 256 and 1024 tokens. "
    r"SnapKV+rot remains the best method at every budget on both models. "
    r"Re-rotation reduces KL by 2.2$\times$ on Llama at budget~=~256 "
    r"(0.769 $\to$ 0.352) and 6.0$\times$ on Mistral (1.912 $\to$ 0.614). "
    r"At budget~=~1024 the improvement grows further: 4.4$\times$ on Llama "
    r"(0.648 $\to$ 0.146) and 6.0$\times$ on Mistral (1.766 $\to$ 0.294)."
    "\n\n"
)

GT_SECTION_INTRO = (
    "Ground-truth accuracy measures whether a compressed model produces outputs\n"
    "that match human-written reference answers. "
    "Table~\\ref{tab:gt-allrates} shows macro-averaged scores\n"
    "across all rates on both models. "
    "Scores are compressed across methods at every retention rate: all KV\n"
    "eviction methods cluster within 2 points of each other and within 2 points of the\n"
    "full model. "
    "This clustering is the key limitation of GT as a compression metric---it\n"
    "cannot separate faithful methods from unfaithful ones (§5). "
    "Per-task results at 65\\% retention appear in Appendix~C.\n\n"
)

INSTRUCT_GT_INTRO = (
    "\\textbf{Instruct models at extreme budgets.} "
    "At absolute token budgets of 256 and 1024, all four selection methods approach\n"
    "full-model GT scores, making the metric nearly non-discriminating\n"
    "(Table~\\ref{tab:instruct-gt}). "
    "On Llama at budget~=~1024, the best compressed method (Pyr, 31.6\\%) is within\n"
    "0.1 points of the full model (31.7\\%). "
    "Naive is far below (16.9\\% at budget~=~256), confirming that prompt truncation\n"
    "destroys retrieval accuracy, but GT cannot distinguish among the four selection\n"
    "methods that all score above 30\\%.\n\n"
)

INSTRUCT_FOUT_INTRO = (
    "\\textbf{Instruct models at extreme budgets.} "
    "The instruct model's terse output style inflates $F_{\\text{out}}$ for all\n"
    "selection methods at extreme budgets. "
    "With a 4-word answer, any method that retrieves the right phrase reproduces\n"
    "the full model's output nearly verbatim, regardless of positional fidelity.\n\n"
)

INSTRUCT_SETUP_PARA = (
    "\n"
    "\\textbf{Instruct models.} "
    "We also evaluate on Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 "
    "at absolute token budgets of 256 and 1024 tokens (§4.2, §4.3, §4.4). "
    "For SnapKV and PyramidKV at budget~=~256, we use \\texttt{window\\_size=8} "
    "to avoid budget collapse (the 8-token per-layer average would otherwise force "
    "every layer to uniform SnapKV behavior with \\texttt{window\\_size=32}). "
    "At budget~=~1024, \\texttt{window\\_size=32} is sufficient. "
    "Per-task breakdowns appear in Appendix~D.\n"
)


out_lines = []

# ── Preamble + abstract + §1 + §2 ──────────────────────────────────────────
out_lines += L(1, 270)

# ── New §3: Faithfulness Metrics (= §5, tightened) ─────────────────────────
# §5 section header (429-431)
out_lines += L(429, 431)

# §5.1 header + intro text (432-473)
out_lines += L(432, 473)

# Skip "Accidental correctness is rewarded" bullet (474-482)
# Resume at line 483 (the \item for "Low full-model accuracy")

# Rest of §5.1 bullets + Figure 1 + \end{itemize} (483-631)
out_lines += L(483, 631)

# Skip the two paragraphs after figure (632-647)

# §5.2 KL Faithfulness + §5.3 Output Faithfulness header + text (648-823)
out_lines += L(648, 823)

# §5.3 "Caveat emptor" intro line embedded; skip bullets (824-850)
# Resume at 851 "These limitations are why..."
out_lines += L(851, 858)

# Skip §5.4 (859-1010) and §5.5 (1011-1025)

# ── New §4: Experiments (= §7 + §3 methods + §4.2 GT + §4.3 KL + §4.4 Fout + §4.5 Timing)
out_lines.append("\n")
out_lines.append("\\hypertarget{main-experiments}{%\n")
out_lines.append("\\section{4. Experiments}\\label{main-experiments}}\n")
out_lines.append("\n")

# ── §4.1 Setup ──────────────────────────────────────────────────────────────
out_lines += L(1417, 1419)  # \subsection{4.1 Setup}

# §7.1 Models + Benchmark + Hyperparameters (text before the methods table)
out_lines += L(1420, 1429)

# Instruct model setup paragraph (inline)
out_lines.append(INSTRUCT_SETUP_PARA)

# Methods description from §3 (skip section headers 271-276; content starts at 277)
out_lines.append("\n")
out_lines += L(277, 334)
out_lines.append("\n")

# §7.1 Methods table (1430-1449)
out_lines += L(1430, 1449)

# ── §4.2 Ground-Truth Results ──────────────────────────────────────────────
out_lines.append("\n")
out_lines.append("\\hypertarget{gt-evaluation}{%\n")
out_lines.append("\\subsection{4.2 Ground-Truth Results}\\label{gt-evaluation}}\n")
out_lines.append("\n")
out_lines.append(GT_SECTION_INTRO)
out_lines.append(GT_ALLRATES_TABLE)
out_lines.append("\n")
out_lines.append(INSTRUCT_GT_INTRO)
out_lines.append(INSTRUCT_GT_TABLE)
out_lines.append("\n")

# ── §4.3 KL Faithfulness (= §7.2) ──────────────────────────────────────────
out_lines += L(1451, 1563)

# Instruct KL intro + table (inline)
out_lines.append("\n")
out_lines.append(INSTRUCT_KL_INTRO)
out_lines.append(INSTRUCT_KL_TABLE)
out_lines.append("\n")

# "SnapKV+rot is the best method at every budget" prose from §9 (2174-2194)
out_lines += L(2174, 2194)
out_lines.append("\n")

# Tab:instruct-kl-shortlong from §9 (2248-2290)
out_lines += L(2248, 2290)
out_lines.append("\n")

# ── §4.4 Output Faithfulness (= §7.3) ──────────────────────────────────────
out_lines += L(1565, 1643)

# Instruct F_out intro (inline)
out_lines.append("\n")
out_lines.append(INSTRUCT_FOUT_INTRO)

# "Instruct model output style amplifies first-token advantage" from §9 (2130-2173)
out_lines += L(2130, 2173)
out_lines.append("\n")

# Tab:instruct-fout-shortlong from §9 (2203-2246)
out_lines += L(2203, 2246)
out_lines.append("\n")

# Deployment perspective prose from §9 (2292-2313)
out_lines += L(2292, 2313)
out_lines.append("\n")

# ── §4.5 Inference Performance (= §7.4) ─────────────────────────────────────
out_lines += L(1644, 1701)

# ── New §5: Structural Corruption (= §6 unchanged) ──────────────────────────
out_lines.append("\n")
out_lines += L(1026, 1413)

# ── New §6: Why Post-Prefill KV Eviction Without Re-rotation (= §8) ─────────
out_lines.append("\n")
out_lines += L(1702, 2073)

# ── New §7: Conclusion (= §10) ───────────────────────────────────────────────
out_lines.append("\n")
out_lines += L(2315, 2383)

# ── References ────────────────────────────────────────────────────────────────
out_lines += L(2384, len(orig))

# Apply all cross-reference text substitutions
final = text_subs(out_lines)

with open("paper_kv_short.tex", "w") as f:
    f.write(final)

print("Done. Written to paper_kv_short.tex")
print(f"Lines: {final.count(chr(10))}")
