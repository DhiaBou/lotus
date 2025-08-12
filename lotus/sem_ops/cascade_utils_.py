import numpy as np
from numpy.typing import NDArray
from typing import List
import lotus
from lotus.types import CascadeArgs
from typing import Iterable, Tuple


def importance_sampling(
        proxy_scores: List[float],
        cascade_args: CascadeArgs,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Unbiased IS over ALL rows with a distribution that emphasizes
    (1) ambiguous mid region and (2) confident tails, mixed with uniform.
    Returns (sample_indices, correction_factors) where correction_factors
    = (1/N) / q_i are Horvitz–Thompson weights.
    """
    N = len(proxy_scores)
    rng = np.random.default_rng(cascade_args.cascade_IS_random_seed)

    ps = np.asarray(proxy_scores, dtype=np.float64)
    eps = 1e-12
    ps = np.clip(ps, eps, 1.0 - eps)

    # Emphasize both: ambiguity (mid) and decisiveness (tails)
    w_mid  = np.sqrt(ps * (1.0 - ps))      # peak near 0.5
    w_tail = np.maximum(ps, 1.0 - ps)      # peaks near 0 and 1
    w      = 0.5 * w_mid + 0.5 * w_tail

    # Normalize and mix with uniform for full support
    w_sum = float(w.sum())
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        q = np.full(N, 1.0 / N, dtype=np.float64)
    else:
        q = w / w_sum

    alpha = float(cascade_args.cascade_IS_weight)  # 0..1
    q = alpha * q + (1.0 - alpha) * (np.ones(N, dtype=np.float64) / N)
    q = q / q.sum()  # exact normalization

    # Sample WITH replacement (simple, stable HT correction)
    sample_size = max(1, int(cascade_args.sampling_percentage * N))
    idx = rng.choice(np.arange(N), size=sample_size, replace=True, p=q)

    # Horvitz–Thompson correction factors for unbiased totals
    correction_factors = (1.0 / N) / q

    return idx.astype(np.int64), correction_factors.astype(np.float64)

def n_eff(weights: np.ndarray) -> float:
    """Effective sample size under unequal weights."""
    w = np.asarray(weights, dtype=np.float64)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    return float((s1 * s1) / s2) if s2 > 0 else 0.0

def proportion_bounds_normal(p_hat: float, n_eff_val: float, delta: float) -> tuple[float, float]:
    """
    Simple symmetric normal approx bounds for a proportion with effective n.
    Clipped to [0,1]. For tight control use Wilson; normal is fine in practice here.
    """
    if n_eff_val <= 1.0:
        return (0.0, 1.0)
    # z ≈ sqrt(2 ln(1/delta)) (same spirit as your code)
    z = np.sqrt(2.0 * np.log(max(1.0 / max(delta, 1e-12), 1.0)))
    var = p_hat * (1.0 - p_hat) / max(n_eff_val, 1.0)
    se  = np.sqrt(max(var, 0.0))
    lo, hi = p_hat - z * se, p_hat + z * se
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
from typing import Iterable, Tuple

Pair = Tuple[float, bool, float]  # (proxy_score, oracle_label, HT_weight)

def weighted_recall(tau_pos: float, tau_neg: float, pairs: Iterable[Pair]) -> float:
    pairs = list(pairs)
    # Set partitions
    helper_above = [(p, y, w) for (p, y, w) in pairs if p >= tau_pos]
    routed      = [(p, y, w) for (p, y, w) in pairs if tau_neg < p < tau_pos]
    # Below neg threshold are predicted negative; they don't contribute to TP

    # Total true positives (population total, estimated)
    denom = sum((1.0 if y else 0.0) * w for _, y, w in pairs)
    if denom <= 0.0:
        return 0.0

    # True positives predicted positive:
    # - helper_above: must be truly positive
    # - routed: oracle decides => count all routed true positives
    tp_helper = sum(w for _, y, w in helper_above if y)
    tp_routed = sum(w for _, y, w in routed if y)
    num = tp_helper + tp_routed
    return float(num / denom)

def precision_lb_helper_above(tau_pos: float, pairs: Iterable[Pair], delta: float) -> float:
    """
    Conservative lower bound on precision contributed by the helper-accepted positives
    (scores >= tau_pos). Routed positives only improve overall precision, so ensuring
    this block meets the target is sufficient.
    """
    helper_above = [(p, y, w) for (p, y, w) in pairs if p >= tau_pos]
    w_sum = sum(w for _, _, w in helper_above)
    if w_sum <= 0.0:
        return 0.0  # no helper-accepted positives at this threshold

    w_tp = sum(w for _, y, w in helper_above if y)
    p_hat = w_tp / w_sum
    neff  = n_eff(np.array([w for _, _, w in helper_above], dtype=np.float64))

    lb, _ = proportion_bounds_normal(p_hat, neff, max(delta, 1e-6))
    return lb

def calibrate_llm_logprobs(true_probs: list[float], cascade_args: CascadeArgs) -> list[float]:
    """Transforms true probabilities to calibrate LLM proxies."""
    num_quantiles = cascade_args.cascade_num_calibration_quantiles
    quantile_values = np.percentile(true_probs, np.linspace(0, 100, num_quantiles + 1))
    true_probs = list((np.digitize(true_probs, quantile_values) - 1) / num_quantiles)
    true_probs = list(np.clip(true_probs, 0, 1))
    return true_probs


def learn_cascade_thresholds(
        proxy_scores: list[float],
        oracle_outputs: list[bool],
        sample_correction_factors: NDArray[np.float64],
        cascade_args: CascadeArgs,
) -> tuple[tuple[float, float], int]:
    """
    Learn (tau_pos, tau_neg) with weighted recall/precision.
    - tau_neg chosen to satisfy recall >= recall_target (weighted).
    - tau_pos chosen so helper-above precision lower bound >= precision_target.
    Returns ((tau_pos, tau_neg), estimated_oracle_calls_over_population).
    """
    assert len(proxy_scores) == len(oracle_outputs) == len(sample_correction_factors)
    pairs = list(zip(proxy_scores, oracle_outputs, sample_correction_factors))
    # Sort by score descending so threshold scans are stable
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

    # ---- Choose tau_neg (maximize rejection while keeping recall) ----
    # Candidates: include unique scores and safe endpoints
    unique_scores = sorted({p for (p, _, _) in pairs_sorted})
    # Add endpoints to be safe
    candidates_neg = sorted([0.0] + unique_scores + [1.0])

    best_tau_neg = 0.0
    for tau_neg in candidates_neg:
        r = weighted_recall(1.0, tau_neg, pairs_sorted)  # tau_pos=1 means no helper-positives shortcut
        if r >= cascade_args.recall_target:
            best_tau_neg = tau_neg
        else:
            break  # candidates_neg is ascending; recall only gets worse as tau_neg increases

    # ---- Choose tau_pos (smallest that guarantees precision) ----
    # Bonferroni-split delta across scans to be conservative
    delta = max(float(cascade_args.failure_probability), 1e-6)
    K = max(1, len(unique_scores))
    per_thresh_delta = delta / K

    candidates_pos = sorted(unique_scores + [1.0])
    tau_pos = 1.0
    for cand in candidates_pos:
        if cand < best_tau_neg:
            continue  # must keep tau_pos >= tau_neg
        lb = precision_lb_helper_above(cand, pairs_sorted, per_thresh_delta)
        if lb >= cascade_args.precision_target:
            tau_pos = cand
            break

    # Ensure ordering
    tau_pos = max(tau_pos, best_tau_neg)

    # Estimate #oracle calls across the full population using the *proxy scores distribution*
    # (This is the same definition you had; it’s a simple deterministic count.)
    tau_neg = best_tau_neg
    oracle_calls = sum(1 for s in proxy_scores if tau_neg < s < tau_pos)

    return (tau_pos, tau_neg), oracle_calls


def calibrate_sem_sim_join(true_score: list[float]) -> list[float]:
    true_score = list(np.clip(true_score, 0, 1))
    return true_score
