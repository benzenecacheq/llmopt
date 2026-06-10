"""
test_core.py
------------
Unit tests for pure-Python / pure-tensor functions.

No GPU or model loading required — all tests run on CPU.

Run:
    conda run -n llmopt python -m pytest test_core.py -v

Coverage:
  gt_eval_compression  : qa_f1_score, rouge_l_score, classification_score,
                         retrieval_score, count_score, code_sim_score,
                         ckpt_key, score_results
  kl_faith_eval        : naive_truncate, chunk_truncate,
                         _chunk_scores_token, _chunk_scores_bm25
  kl_faith_eval_ystar  : kl_divergence
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from gt_eval_compression import (
    qa_f1_score,
    rouge_l_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
    ckpt_key,
    score_results,
)
from kl_faith_eval import (
    naive_truncate,
    chunk_truncate,
    _chunk_scores_token,
    _chunk_scores_bm25,
)
from kl_faith_eval_ystar import kl_divergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ids(*tokens):
    """Flat list of ints → (1, N) int64 tensor."""
    return torch.tensor([list(tokens)], dtype=torch.long)


# ---------------------------------------------------------------------------
# qa_f1_score
# ---------------------------------------------------------------------------

class TestQaF1Score:
    def test_exact_match(self):
        assert qa_f1_score("Paris", ["Paris"]) == pytest.approx(1.0)

    def test_exact_match_multiple_words(self):
        assert qa_f1_score("Albert Einstein", ["Albert Einstein"]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert qa_f1_score("London", ["Paris"]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # pred="Albert London" gt="Albert Einstein": 1 common token out of 2/2 → F1=0.5
        score = qa_f1_score("Albert London", ["Albert Einstein"])
        assert score == pytest.approx(0.5)

    def test_multiple_ground_truths_takes_max(self):
        # Second ground truth is exact match
        score = qa_f1_score("Paris", ["London", "Paris"])
        assert score == pytest.approx(1.0)

    def test_article_normalization(self):
        # "the" is stripped; "the Eiffel Tower" → "eiffel tower"
        assert qa_f1_score("the Eiffel Tower", ["Eiffel Tower"]) == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert qa_f1_score("paris", ["Paris"]) == pytest.approx(1.0)

    def test_punctuation_stripped(self):
        assert qa_f1_score("Paris.", ["Paris"]) == pytest.approx(1.0)

    def test_empty_prediction_returns_zero(self):
        assert qa_f1_score("", ["Paris"]) == pytest.approx(0.0)

    def test_repeated_tokens_counted_correctly(self):
        # pred="yes yes yes" gt="yes": common=1 (Counter intersection), p=1/3, r=1/1, F1=0.5
        score = qa_f1_score("yes yes yes", ["yes"])
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rouge_l_score
# ---------------------------------------------------------------------------

class TestRougeLScore:
    def test_identical_strings(self):
        assert rouge_l_score("hello world", ["hello world"]) == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert rouge_l_score("", ["hello world"]) == pytest.approx(0.0)

    def test_no_overlap(self):
        score = rouge_l_score("foo bar", ["hello world"])
        assert score == pytest.approx(0.0)

    def test_partial_overlap_nonzero(self):
        score = rouge_l_score("the cat sat on the mat", ["the cat sat"])
        assert 0.0 < score <= 1.0

    def test_multiple_ground_truths_takes_max(self):
        score_one = rouge_l_score("hello world test", ["hello world test"])
        score_two = rouge_l_score("hello world test", ["hello world test", "xyz"])
        assert score_one == pytest.approx(score_two)


# ---------------------------------------------------------------------------
# classification_score
# ---------------------------------------------------------------------------

class TestClassificationScore:
    def test_exact_match(self):
        assert classification_score("Science", ["Science"]) == pytest.approx(1.0)

    def test_ground_truth_substring_of_prediction(self):
        # GT "science" normalizes to "science"; pred "category science here" contains it
        assert classification_score("category science here", ["Science"]) == pytest.approx(1.0)

    def test_no_match(self):
        assert classification_score("History", ["Science"]) == pytest.approx(0.0)

    def test_multiple_ground_truths(self):
        assert classification_score("Science", ["History", "Science"]) == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert classification_score("science", ["Science"]) == pytest.approx(1.0)

    def test_article_stripped(self):
        # "the" stripped from both sides
        assert classification_score("the science", ["science"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# retrieval_score
# ---------------------------------------------------------------------------

class TestRetrievalScore:
    def test_paragraph_keyword(self):
        assert retrieval_score("Paragraph 3", ["Paragraph 3"]) == pytest.approx(1.0)

    def test_bare_number(self):
        assert retrieval_score("7", ["7"]) == pytest.approx(1.0)

    def test_paragraph_keyword_case_insensitive(self):
        assert retrieval_score("paragraph 3", ["Paragraph 3"]) == pytest.approx(1.0)

    def test_number_mismatch(self):
        assert retrieval_score("Paragraph 3", ["Paragraph 5"]) == pytest.approx(0.0)

    def test_no_number_in_prediction(self):
        assert retrieval_score("I don't know", ["Paragraph 3"]) == pytest.approx(0.0)

    def test_multiple_ground_truths(self):
        assert retrieval_score("Paragraph 3", ["Paragraph 5", "Paragraph 3"]) == pytest.approx(1.0)

    def test_number_embedded_in_text(self):
        assert retrieval_score("The answer is 3 paragraphs in", ["Paragraph 3"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# count_score
# ---------------------------------------------------------------------------

class TestCountScore:
    def test_exact_match(self):
        assert count_score("5", ["5"]) == pytest.approx(1.0)

    def test_number_embedded_in_text(self):
        assert count_score("There are 5 passages", ["5"]) == pytest.approx(1.0)

    def test_wrong_number(self):
        assert count_score("3", ["5"]) == pytest.approx(0.0)

    def test_no_number_in_prediction(self):
        assert count_score("I don't know", ["5"]) == pytest.approx(0.0)

    def test_multiple_ground_truths(self):
        assert count_score("3", ["5", "3"]) == pytest.approx(1.0)

    def test_uses_first_number_only(self):
        # "3 and then 5": first number is 3, ground truth is 5 → miss
        assert count_score("3 and then 5", ["5"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# code_sim_score
# ---------------------------------------------------------------------------

class TestCodeSimScore:
    def test_exact_match(self):
        assert code_sim_score("return x + 1", ["return x + 1"]) == pytest.approx(1.0)

    def test_uses_first_line_only(self):
        # Multi-line prediction: only first line is compared
        score_first = code_sim_score("return x + 1\nreturn y", ["return x + 1"])
        assert score_first == pytest.approx(1.0)

    def test_completely_different(self):
        score = code_sim_score("aaa bbb", ["xyz xyz"])
        assert score < 0.5

    def test_multiple_ground_truths_takes_max(self):
        score = code_sim_score("return x + 1", ["return y - 1", "return x + 1"])
        assert score == pytest.approx(1.0)

    def test_partial_similarity_nonzero(self):
        score = code_sim_score("return x + 1", ["return x + 2"])
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# ckpt_key
# ---------------------------------------------------------------------------

class TestCkptKey:
    def test_format(self):
        assert ckpt_key("hotpotqa", 0, "naive_65pct") == "hotpotqa|0|naive_65pct"

    def test_roundtrip_parse(self):
        key = ckpt_key("trec", 42, "snapkv")
        parts = key.split("|")
        assert parts == ["trec", "42", "snapkv"]


# ---------------------------------------------------------------------------
# score_results
# ---------------------------------------------------------------------------

class TestScoreResults:
    def _make_ckpt(self, entries):
        """entries: list of (task, idx, method, prediction, answers)."""
        ckpt = {}
        for task, idx, method, pred, answers in entries:
            ckpt[ckpt_key(task, idx, method)] = {"prediction": pred, "answers": answers}
        return ckpt

    def test_exact_match_averages_to_100(self):
        ckpt = self._make_ckpt([
            ("hotpotqa", 0, "naive", "Paris", ["Paris"]),
            ("hotpotqa", 1, "naive", "London", ["London"]),
        ])
        r = score_results(["hotpotqa"], ckpt, n=None)
        assert r["hotpotqa"]["naive"] == pytest.approx(100.0)

    def test_n_truncates_examples(self):
        ckpt = self._make_ckpt([
            ("hotpotqa", 0, "naive", "Paris", ["Paris"]),   # correct
            ("hotpotqa", 1, "naive", "wrong", ["London"]),  # wrong
        ])
        r_n1 = score_results(["hotpotqa"], ckpt, n=1)
        assert r_n1["hotpotqa"]["naive"] == pytest.approx(100.0)

        r_n2 = score_results(["hotpotqa"], ckpt, n=2)
        # One correct out of two: F1 is 0 for the miss, so average < 100
        assert r_n2["hotpotqa"]["naive"] < 100.0

    def test_multiple_methods(self):
        ckpt = self._make_ckpt([
            ("hotpotqa", 0, "good", "Paris", ["Paris"]),
            ("hotpotqa", 0, "bad",  "wrong", ["Paris"]),
        ])
        r = score_results(["hotpotqa"], ckpt, n=None)
        assert r["hotpotqa"]["good"] == pytest.approx(100.0)
        assert r["hotpotqa"]["bad"] == pytest.approx(0.0)

    def test_unknown_task_skipped(self):
        ckpt = self._make_ckpt([
            ("unknown_task", 0, "naive", "x", ["x"]),
        ])
        r = score_results(["unknown_task"], ckpt, n=None)
        assert "unknown_task" not in r

    def test_n_count_stored(self):
        ckpt = self._make_ckpt([
            ("hotpotqa", 0, "naive", "Paris", ["Paris"]),
            ("hotpotqa", 1, "naive", "London", ["London"]),
            ("hotpotqa", 2, "naive", "Berlin", ["Berlin"]),
        ])
        r = score_results(["hotpotqa"], ckpt, n=2)
        assert r["hotpotqa"]["n"] == 2

    def test_correct_scorer_used_per_task(self):
        # passage_count uses count_score (extracts first number)
        ckpt = self._make_ckpt([
            ("passage_count", 0, "naive", "there are 3 passages", ["3"]),
        ])
        r = score_results(["passage_count"], ckpt, n=None)
        assert r["passage_count"]["naive"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# naive_truncate
# ---------------------------------------------------------------------------

class TestNaiveTruncate:
    def test_output_length_equals_budget(self):
        x = ids(*range(100))
        out = naive_truncate(x, fraction=0.65)
        assert out.shape[1] == int(100 * 0.65)

    def test_head_tail_split(self):
        # T=100, fraction=0.5 → budget=50, head_frac=0.1 → head_n=5, tail_n=45
        x = ids(*range(100))
        out = naive_truncate(x, fraction=0.5, head_frac=0.10)
        assert out.shape[1] == 50
        # First 5 tokens should be 0..4
        assert out[0, :5].tolist() == list(range(5))
        # Last 45 tokens should be 55..99
        assert out[0, 5:].tolist() == list(range(55, 100))

    def test_head_frac_zero_is_pure_tail(self):
        x = ids(*range(100))
        out = naive_truncate(x, fraction=0.5, head_frac=0.0)
        assert out.shape[1] == 50
        assert out[0].tolist() == list(range(50, 100))

    def test_fraction_one_returns_all(self):
        x = ids(*range(100))
        out = naive_truncate(x, fraction=1.0)
        assert out.shape[1] == 100

    def test_very_short_input_returns_at_least_one(self):
        x = ids(42)
        out = naive_truncate(x, fraction=0.1)
        assert out.shape[1] >= 1

    def test_tokens_come_from_input(self):
        x = ids(*range(100))
        out = naive_truncate(x, fraction=0.5, head_frac=0.10)
        for tok in out[0].tolist():
            assert tok in range(100)


# ---------------------------------------------------------------------------
# chunk_truncate
# ---------------------------------------------------------------------------

class TestChunkTruncate:
    def _seq(self, n):
        return ids(*range(n))

    def test_output_length_does_not_exceed_budget(self):
        x = self._seq(100)
        q = ids(5, 10, 15)
        out = chunk_truncate(x, q, fraction=0.65, chunk_size=16, scoring="token")
        budget = int(100 * 0.65)
        assert out.shape[1] <= budget

    def test_tokens_come_from_input(self):
        x = self._seq(100)
        q = ids(5, 10, 15)
        out = chunk_truncate(x, q, fraction=0.65, chunk_size=16, scoring="token")
        valid = set(range(100))
        for tok in out[0].tolist():
            assert tok in valid

    def test_original_order_preserved(self):
        x = self._seq(100)
        q = ids(5)
        out = chunk_truncate(x, q, fraction=0.65, chunk_size=16, scoring="token")
        toks = out[0].tolist()
        # Verify tokens appear in strictly increasing order (original position order)
        assert toks == sorted(toks)

    def test_tail_frac_reserves_recency_tokens(self):
        # tail_frac=0.5: half the budget comes from the literal tail of the sequence
        x = self._seq(100)
        q = ids(999)  # no overlap with any token → all scores zero
        out = chunk_truncate(x, q, fraction=0.5, chunk_size=10, tail_frac=0.5, scoring="token")
        budget = int(100 * 0.5)           # 50
        tail_n = int(budget * 0.5)        # 25
        # Last tail_n tokens must all be present
        tail_tokens = set(range(100 - tail_n, 100))
        output_tokens = set(out[0].tolist())
        assert tail_tokens.issubset(output_tokens)

    def test_high_score_chunk_selected_over_low(self):
        # Build input where only one chunk shares tokens with the question.
        # question token=99; only tokens 80-89 contain 99; everything else is 0..79.
        # chunk_size=10, fraction=0.5 → budget=50 (5 chunks of 10)
        data = list(range(80)) + [99] * 10  # tokens 80-89 = 99
        x = ids(*data)
        q = ids(99)
        out = chunk_truncate(x, q, fraction=0.5, chunk_size=10, scoring="token")
        # The high-scoring chunk (positions 80-89, all token=99) must be included
        assert 99 in out[0].tolist()

    def test_no_question_falls_back_to_recency(self):
        # No question → all scores zero → recency wins → tail tokens selected
        x = self._seq(100)
        q = ids()  # empty
        out = chunk_truncate(x, q, fraction=0.5, chunk_size=10, scoring="token")
        budget = int(100 * 0.5)
        # Output should be the most recent tokens (last `budget` tokens from the sequence)
        assert out[0].tolist() == list(range(100 - budget, 100))

    def test_recency_tiebreaking(self):
        # All chunks have equal score (zero overlap). Recency tie-break means
        # we should prefer more recent (higher-index) chunks.
        x = self._seq(40)   # 4 chunks of 10
        q = ids(999)        # no overlap
        # budget = int(40 * 0.5) = 20 → select 2 chunks → should be chunks 2 and 3 (indices 2,3)
        out = chunk_truncate(x, q, fraction=0.5, chunk_size=10, scoring="token")
        expected = list(range(20, 40))
        assert out[0].tolist() == expected

    def test_budget_gte_input_returns_all_tokens(self):
        x = self._seq(20)
        q = ids(5)
        out = chunk_truncate(x, q, fraction=1.0, chunk_size=10, scoring="token")
        assert out.shape[1] == 20

    def test_partial_last_chunk(self):
        # T=25 with chunk_size=10: chunks are [0-9], [10-19], [20-24] (partial)
        x = self._seq(25)
        q = ids(999)
        out = chunk_truncate(x, q, fraction=0.5, chunk_size=10, scoring="token")
        # budget = 12; all zero scores → last 12 tokens
        assert out.shape[1] <= 12


# ---------------------------------------------------------------------------
# _chunk_scores_token
# ---------------------------------------------------------------------------

class TestChunkScoresToken:
    def _chunk(self, *tokens):
        return ids(*tokens)  # chunk is a (1, N) tensor, matching chunk_truncate's list

    def test_exact_overlap(self):
        chunks = [self._chunk(1, 2, 3), self._chunk(4, 5, 6)]
        scores = _chunk_scores_token(chunks, {1, 2})
        assert scores[0] == 2   # tokens 1 and 2 match
        assert scores[1] == 0

    def test_no_overlap(self):
        chunks = [self._chunk(7, 8, 9)]
        assert _chunk_scores_token(chunks, {1, 2, 3}) == [0]

    def test_deduplication_within_chunk(self):
        # Token 1 appears twice but set intersection counts unique tokens
        chunks = [self._chunk(1, 1, 1)]
        assert _chunk_scores_token(chunks, {1}) == [1]

    def test_empty_question(self):
        chunks = [self._chunk(1, 2, 3)]
        assert _chunk_scores_token(chunks, set()) == [0]


# ---------------------------------------------------------------------------
# _chunk_scores_bm25
# ---------------------------------------------------------------------------

class TestChunkScoresBm25:
    def _make_chunks(self, *token_lists):
        return [ids(*tl) for tl in token_lists]

    def test_matching_chunk_scores_higher(self):
        # chunk0 has the question token; chunk1 does not
        chunks = self._make_chunks([1, 2, 3, 99], [4, 5, 6, 7])
        q = ids(99)
        scores = _chunk_scores_bm25(chunks, q)
        assert scores[0] > scores[1]

    def test_no_match_scores_zero(self):
        chunks = self._make_chunks([1, 2, 3], [4, 5, 6])
        q = ids(99)
        scores = _chunk_scores_bm25(chunks, q)
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(0.0)

    def test_more_matches_scores_higher(self):
        # chunk0 has question token twice, chunk1 once
        chunks = self._make_chunks([99, 99, 1, 2], [99, 1, 2, 3])
        q = ids(99)
        scores = _chunk_scores_bm25(chunks, q)
        assert scores[0] > scores[1]

    def test_scores_length_equals_chunk_count(self):
        chunks = self._make_chunks([1, 2], [3, 4], [5, 6])
        scores = _chunk_scores_bm25(chunks, ids(1))
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def test_identical_distributions_is_zero(self):
        # KL(P || P) = 0 for any P
        logits = torch.randn(10, 50)
        log_p = F.log_softmax(logits, dim=-1)
        kl = kl_divergence(log_p, log_p)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_known_two_token_case(self):
        # P = [0.8, 0.2], Q = [0.6, 0.4]
        # KL(P||Q) = 0.8*ln(0.8/0.6) + 0.2*ln(0.2/0.4) ≈ 0.08889
        import math
        p = torch.tensor([[0.8, 0.2]])
        q = torch.tensor([[0.6, 0.4]])
        log_p = p.log()
        log_q = q.log()
        expected = 0.8 * math.log(0.8 / 0.6) + 0.2 * math.log(0.2 / 0.4)
        kl = kl_divergence(log_p, log_q)
        assert kl == pytest.approx(expected, rel=1e-4)

    def test_nonnegative(self):
        log_p = F.log_softmax(torch.randn(5, 100), dim=-1)
        log_q = F.log_softmax(torch.randn(5, 100), dim=-1)
        assert kl_divergence(log_p, log_q) >= 0.0

    def test_averages_over_steps(self):
        # Use 2-token distributions for both steps so no padding needed.
        # step 0: P=Q → KL=0
        # step 1: P=[0.8,0.2], Q=[0.6,0.4] → KL=X
        kl_step1 = 0.8 * math.log(0.8 / 0.6) + 0.2 * math.log(0.2 / 0.4)
        p_same = torch.tensor([[0.5, 0.5]])
        log_p = torch.cat([p_same.log(), torch.tensor([[0.8, 0.2]]).log()], dim=0)
        log_q = torch.cat([p_same.log(), torch.tensor([[0.6, 0.4]]).log()], dim=0)
        kl = kl_divergence(log_p, log_q)
        assert kl == pytest.approx(kl_step1 / 2, rel=1e-4)
