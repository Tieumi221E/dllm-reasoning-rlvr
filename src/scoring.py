"""
Process / answer scorers (m_P, m_A) and the unbiased pass@k estimator.

Strict, process-verified `pass@k` with `m_P ∧ m_A`, following the evaluation design of
the AR-baseline study (arXiv:2605.26934). m_A is exact (normalized) answer matching against
gold or any provided equivalent answer; m_P checks the task-specific reasoning trace.
Pure text functions — no model / torch dependency.
"""
import math
import re
import string


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_answer(text: str) -> str:
    """Extract the text following an explicit answer marker.
    Returns '' if no marker (no fallback, to avoid false positives from gold names in degenerate text).
    """
    for pat in [
        r"answer\s*:\s*(.+)",
        r"the answer is\s*(.+)",
        r"the missing information is[:\s]+(.+)",
        r"therefore[,:]?\s*(.+)",
    ]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            ans = matches[-1].group(1).strip()
            return ans.split("\n")[0].split(".")[0].strip()
    return ""


def _check_answer(gen_text: str, gold_answer: str, equivalent_answers=None) -> bool:
    """m_A: exact match of the extracted answer against gold (or any equivalent_answer) after normalization.
    An abductive missing event has many equivalent phrasings (return/lend/borrow, voice, word order),
    so gold ships equivalent_answers; match semantically, not against a single answer.
    Substring matching would accept 'broken' inside 'intact (not broken)', so use exact ==.
    """
    extracted = _normalize(_extract_answer(gen_text))
    if not extracted:
        return False
    golds = [gold_answer] + list(equivalent_answers or [])
    return any(extracted == _normalize(g) for g in golds if g)


def _missing_condition(text: str):
    """Extract abductive gap-boundary states (before_missing, after_missing), normalized."""
    gb = re.search(r"before the missing step:\s*(.+?)\.\s*after the missing step:",
                   text, re.IGNORECASE)
    ga = re.search(r"after the missing step:\s*(.+?)\.", text, re.IGNORECASE)
    return (_normalize(gb.group(1)) if gb else None,
            _normalize(ga.group(1)) if ga else None)


def _check_process(gen_text: str, task: dict) -> bool:
    """m_P: task-specific process matching."""
    task_type     = task.get("task_type", "")
    gold_solution = task.get("solution", "")

    if task_type in ("deductive", "deduction_full_info", "deduction_hard"):
        # Full owner/possessor/integrity state sequence, no missing or extra step
        gold_states = re.findall(r"State:\s*([^.]+)\.", gold_solution, re.IGNORECASE)
        if not gold_states:
            return True
        gen_states = re.findall(r"State:\s*([^.]+)\.", gen_text, re.IGNORECASE)
        if len(gen_states) != len(gold_states):
            return False
        return all(_normalize(g) == _normalize(p) for g, p in zip(gold_states, gen_states))

    elif task_type == "abductive":
        # Core process score = missing-condition: the model must mark the gap-boundary states correctly
        gold_b, gold_a = _missing_condition(gold_solution)
        if gold_b is None or gold_a is None:
            return True  # gold has no explicit missing-condition; do not penalize
        gen_b, gen_a = _missing_condition(gen_text)
        return gen_b == gold_b and gen_a == gold_a

    elif task_type == "inductive":
        m = re.search(r"the pattern is\s+([^.]+)\.", gold_solution, re.IGNORECASE)
        if not m:
            return "the pattern is" in _normalize(gen_text)
        return _normalize(m.group(1)) in _normalize(gen_text)

    elif task_type == "analogy":
        m = re.search(r"the pattern is\s+([^.]+)\.", gold_solution, re.IGNORECASE)
        gen_norm = _normalize(gen_text)
        if not m:
            return "by the same pattern" in gen_norm
        return _normalize(m.group(1)) in gen_norm and "by the same pattern" in gen_norm

    return True


def is_correct(gen_text: str, task: dict, metric: str) -> bool:
    m_a = _check_answer(gen_text, task.get("answer", ""), task.get("equivalent_answers"))
    if metric == "answer_only":
        return m_a
    return m_a and _check_process(gen_text, task)  # strict: m_P ∧ m_A


def _passk(n_correct: int, n_total: int, k: int) -> float:
    """Chen et al. (2021) unbiased estimator: 1 - C(n-c,k)/C(n,k)."""
    if n_total == 0 or k > n_total:
        return float("nan")
    if n_total - n_correct < k:
        return 1.0
    log_num = sum(math.log(n_total - n_correct - i) for i in range(k))  # log-space, avoid overflow
    log_den = sum(math.log(n_total - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)
