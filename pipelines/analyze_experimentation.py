"""Multi-granularity proximity analyzer with Empirical Anchor Error, Scale-Normalized IQR Distance, 14x4 Per-Level Tables, and Paired Statistical Tests."""
from __future__ import annotations

import argparse
import json
import math
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sacrebleu
from bert_score import score as bert_score_fn


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hebrew_readability.config import (
    CALIBRATION_DEAD_ZONE_BRIDGES,
    GRADE_NUMERIC_MAP,
    K12_GRADE_TRACK_LEVELS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    RAW_LEVEL_TO_POPULATION,
    SPECIAL_L2_TRACK_LEVELS,
    build_empirical_level_anchor_map,
    compute_scale_normalized_iqr_distance,
    setup_pipeline_logger,
)

PROMPT_ORDER = ["agentic", "feature_informed", "few_shot_persona", "cot_persona"]

NIKKUD_STRIP_RE = re.compile(r"[\u0591-\u05C7]")


def compute_preservation_batch(
    sources: List[str],
    simplified: List[str],
    model_type: str = "dicta-il/dictabert",
    num_layers: int = 12,
    batch_size: int = 32,
) -> Tuple[List[float], List[float]]:
    """Computes Nikkud-stripped DictaBERT BERTScore-F1 and chrF++ across all rows in batch."""
    import sacrebleu
    from bert_score import score as bert_score_fn

    clean_srcs = [NIKKUD_STRIP_RE.sub("", str(s or "")).strip() for s in sources]
    clean_hyps = [NIKKUD_STRIP_RE.sub("", str(h or "")).strip() for h in simplified]

    _, _, f1_tensor = bert_score_fn(
        clean_hyps,
        clean_srcs,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        verbose=False,
    )
    bert_f1_scores = [round(float(v), 4) for v in f1_tensor.tolist()]

    chrf_scores = [
        round(float(sacrebleu.sentence_chrf(h, [s], word_order=2).score), 2)
        if (h and s)
        else 0.0
        for h, s in zip(clean_hyps, clean_srcs)
    ]
    return bert_f1_scores, chrf_scores

def parse_simplified_level_string(simp_lvl_str: str) -> Tuple[List[str], Dict[str, float]]:
    """Parses '5 (48.2%), 6 (51.8%)' into matched level keys and probabilities."""
    if not isinstance(simp_lvl_str, str) or not simp_lvl_str.strip():
        return [], {}
    s = simp_lvl_str.strip()
    if s.startswith("NONE_IN_IQR"):
        m = re.search(r"Closest:\s*([^\]\s,]+)", s)
        if m:
            closest = m.group(1).strip()
            return [], {closest: 100.0}
        return [], {}

    matches: List[str] = []
    probs: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.+?)\s*\(([0-9.]+)%\)$", part)
        if m:
            lvl = m.group(1).strip()
            p = float(m.group(2))
            matches.append(lvl)
            probs[lvl] = p
        else:
            matches.append(part)
    if matches and not probs:
        unif = round(100.0 / len(matches), 2)
        probs = {m: unif for m in matches}
    return matches, probs


def get_cluster_equivalents(target_level: str) -> Set[str]:
    clean = str(target_level).strip()
    aliases = set(RAW_LEVEL_ALIASES.get(clean, [clean, f"'{clean}", clean.lstrip("'")]))
    if clean in ("1", "2", "3", "'2-4", "2-4"):
        aliases |= {"1", "2", "3", "'2-4", "2-4"}
    elif clean in ("4", "5", "6", "'4-9", "4-9"):
        aliases |= {"4", "5", "6", "'4-9", "4-9"}
    elif clean in ("7", "8", "9", "'4-9", "4-9"):
        aliases |= {"7", "8", "9", "'4-9", "4-9"}
    elif clean in ("10", "11", "12", "Bagrut"):
        aliases |= {"10", "11", "12", "Bagrut"}
    elif clean in ("Hadshon", "Olim_news", "Olim_ragil", "מבחני אולפן - רגיל"):
        aliases |= {"Hadshon", "Olim_news", "Olim_ragil", "מבחני אולפן - רגיל"}
    return aliases


def compute_track_separated_weighted_grade_err(
    target_level: str,
    matched_levels: List[str],
    parsed_probs: Dict[str, float],
    k12_probs_json: Optional[str] = None,
    special_probs_json: Optional[str] = None,
    anchor_map: Optional[Dict[str, float]] = None,
    target_mu: Optional[float] = None,
) -> float:
    """
    Computes track-separated Calibrated Anchor Distance (Err_mu):
        Err_mu = | (sum_{k in M} P_k * mu_k) - mu_target |
    When target_level is a K-12 grade, only K-12 levels contribute to the
    probability-weighted anchor expectation; when Special/L2, only Special/L2 levels contribute.
    """
    active_map = anchor_map or build_empirical_level_anchor_map()
    clean_t = str(target_level).strip()
    if target_mu is not None and pd.notna(target_mu):
        target_anchor = float(target_mu)
    else:
        target_anchor = active_map.get(
            clean_t, active_map.get(clean_t.lstrip("'"), 5.31)
        )

    is_k12_target = (
        clean_t in K12_GRADE_TRACK_LEVELS or clean_t.lstrip("'") in K12_GRADE_TRACK_LEVELS
    )

    candidate_probs: Dict[str, float] = {}
    if is_k12_target and isinstance(k12_probs_json, str) and k12_probs_json.strip():
        try:
            candidate_probs = json.loads(k12_probs_json)
        except Exception:
            candidate_probs = {}
    elif not is_k12_target and isinstance(special_probs_json, str) and special_probs_json.strip():
        try:
            candidate_probs = json.loads(special_probs_json)
        except Exception:
            candidate_probs = {}

    if not candidate_probs:
        track_set = K12_GRADE_TRACK_LEVELS if is_k12_target else SPECIAL_L2_TRACK_LEVELS
        specific_track = {
            k
            for k in parsed_probs
            if (k in track_set or k.lstrip("'") in track_set) and k not in ("'4-9", "4-9")
        }
        if specific_track:
            candidate_probs = {k: parsed_probs[k] for k in specific_track}
        else:
            candidate_probs = {
                k: v
                for k, v in parsed_probs.items()
                if k in track_set or k.lstrip("'") in track_set
            } or dict(parsed_probs)

    if not candidate_probs and matched_levels:
        candidate_probs = {m: 1.0 for m in matched_levels}

    total_w = 0.0
    weighted_mu = 0.0
    for lvl, w in candidate_probs.items():
        mu_val = active_map.get(lvl, active_map.get(lvl.lstrip("'")))
        if mu_val is not None and float(w) > 0:
            weighted_mu += float(w) * float(mu_val)
            total_w += float(w)

    if total_w <= 0:
        return 0.0
    return round(abs((weighted_mu / total_w) - target_anchor), 4)


def cochrans_q_test(binary_matrix: np.ndarray) -> Tuple[float, float]:
    """Computes Cochran's Q omnibus statistic across k matched binary treatments (N blocks x k methods)."""
    N, k = binary_matrix.shape
    row_totals = np.sum(binary_matrix, axis=1)
    col_totals = np.sum(binary_matrix, axis=0)
    grand_total = np.sum(row_totals)
    denom = k * grand_total - np.sum(row_totals ** 2)
    if denom <= 0:
        return 0.0, 1.0
    num = (k - 1) * (k * np.sum(col_totals ** 2) - (grand_total ** 2))
    q_stat = float(num / denom)
    p_val = float(stats.chi2.sf(q_stat, df=k - 1))
    return round(q_stat, 4), round(p_val, 6)


def holm_bonferroni_adjust(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni family-wise error correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = min(1.0, p_val * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(running_max, 6)
    return adjusted


def exact_mcnemar_test(vec_a: np.ndarray, vec_b: np.ndarray) -> Dict[str, Any]:
    """Computes Exact Binomial McNemar's test on paired binary vectors (vec_a vs vec_b)."""
    a = int(np.sum((vec_a == 1) & (vec_b == 1)))
    b = int(np.sum((vec_a == 1) & (vec_b == 0)))
    c = int(np.sum((vec_a == 0) & (vec_b == 1)))
    d = int(np.sum((vec_a == 0) & (vec_b == 0)))
    n_discordant = b + c
    if n_discordant == 0:
        p_exact = 1.0
    else:
        p_exact = float(stats.binomtest(b, n=n_discordant, p=0.5, alternative="two-sided").pvalue)
    odds_ratio = round((b + 0.5) / (c + 0.5), 3) if c == 0 else round(b / c, 3)
    return {
        "both_hit_a": a,
        "a_only_hit_b": b,
        "b_only_hit_c": c,
        "both_miss_d": d,
        "discordant_pairs_b_plus_c": n_discordant,
        "odds_ratio_b_over_c": odds_ratio,
        "p_value_raw": round(p_exact, 6),
    }


def paired_wilcoxon_with_rank_biserial(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    """Runs paired Wilcoxon signed-rank test on continuous errors (x vs y) and computes Rank-Biserial r_rb."""
    diffs = x - y
    nonzero = diffs[np.abs(diffs) > 1e-9]
    n_nz = len(nonzero)
    if n_nz == 0:
        return {"wilcoxon_stat": 0.0, "p_value_raw": 1.0, "rank_biserial_r_rb": 0.0, "n_nonzero": 0}

    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    total_rank_sum = n_nz * (n_nz + 1) / 2.0
    r_rb = (w_plus - w_minus) / total_rank_sum

    try:
        res = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", mode="auto")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat_val = min(w_plus, w_minus)
        p_val = 1.0

    return {
        "wilcoxon_stat": round(stat_val, 4),
        "p_value_raw": round(p_val, 6),
        "rank_biserial_r_rb": round(r_rb, 4),
        "n_nonzero": int(n_nz),
    }

def safe_friedman_test(
    piv_df: pd.DataFrame, columns: List[str]
) -> Tuple[float, float]:
    """Runs the non-parametric Friedman omnibus test across matched columns."""
    try:
        stat_v, p_v = stats.friedmanchisquare(
            *[piv_df[col].to_numpy() for col in columns]
        )
        return round(float(stat_v), 4), round(float(p_v), 6)
    except Exception:
        return 0.0, 1.0
        
def analyze_experimentation_results(
    results_csv: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generates:
      1. row_level_proximity_analysis.csv (with Calibrated Anchor Error, d_IQR & Signed IQR Residuals)
      2. method_quality_summary.csv (4 rows)
      3. level_by_method_summary.csv & level_by_method_pivot_14x4.csv (14 levels x 4 prompts)
      4. paired_statistical_tests.csv (Cochran's Q, McNemar, Friedman, Wilcoxon + r_rb)
    """
    results_csv = Path(results_csv)
    out_dir = Path(output_dir) if output_dir is not None else results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_pipeline_logger(
        name="AnalyzeExperimentation",
        artifact_dir=str(out_dir),
        log_filename="experimentation_analysis.log",
    )

    enriched_candidate = results_csv if results_csv.stem.endswith("_enriched") else results_csv.with_name(results_csv.stem + "_enriched.csv")
    df = pd.read_csv(enriched_candidate if enriched_candidate.exists() else results_csv)
    for _col in ("paragraph_id", "target_level", "prompt"):
        if _col in df.columns:
            df[_col] = df[_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=["paragraph_id", "target_level", "prompt"], keep="last").reset_index(drop=True)
    
    empirical_anchor_map = build_empirical_level_anchor_map()
    if {"target_level", "target_mu"}.issubset(df.columns):
        for _, r_mu in df[["target_level", "target_mu"]].dropna().drop_duplicates().iterrows():
            lvl_k = str(r_mu["target_level"]).strip()
            mu_v = round(float(r_mu["target_mu"]), 4)
            empirical_anchor_map[lvl_k] = mu_v
            empirical_anchor_map[lvl_k.lstrip("'")] = mu_v
            for al in RAW_LEVEL_ALIASES.get(lvl_k, []):
                empirical_anchor_map[al] = mu_v

    row_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        pid = str(row["paragraph_id"])
        prompt = str(row["prompt"]).strip()
        target_lvl = str(row["target_level"]).strip()
        y_pred = float(row["simplified_score"])
        simp_text = str(row.get("simplified_text") or "")
        src_text = str(row.get("text") or "")

        matched_lvls, parsed_probs = parse_simplified_level_string(
            str(row.get("simplified_level") or "")
        )

        # Apply Dead-Zone Bridge [3.2720, 3.3898] during evaluation if needed
        for bridge in CALIBRATION_DEAD_ZONE_BRIDGES:
            if float(bridge["lower_bound"]) <= y_pred <= float(bridge["upper_bound"]):
                nearest = (
                    str(bridge["lower_level"])
                    if y_pred <= float(bridge["midpoint"])
                    else str(bridge["upper_level"])
                )
                if nearest not in matched_lvls:
                    matched_lvls.append(nearest)
                for extra_l in bridge.get("always_include_levels", []):
                    if extra_l not in matched_lvls:
                        matched_lvls.append(extra_l)

        target_aliases = set(
            RAW_LEVEL_ALIASES.get(target_lvl, [target_lvl, f"'{target_lvl}", target_lvl.lstrip("'")])
        )
        matched_set = set(matched_lvls) | {m.lstrip("'") for m in matched_lvls}

        q1_t = float(row["target_q1"]) if "target_q1" in row and pd.notna(row["target_q1"]) else np.nan
        q3_t = float(row["target_q3"]) if "target_q3" in row and pd.notna(row["target_q3"]) else np.nan
        mu_t = float(row["target_mu"]) if "target_mu" in row and pd.notna(row["target_mu"]) else np.nan

        exact_hit = bool(
            len(target_aliases & matched_set) > 0
            or (pd.notna(q1_t) and pd.notna(q3_t) and (q1_t <= y_pred <= q3_t))
        )
        cluster_set = get_cluster_equivalents(target_lvl)
        cluster_hit = bool(exact_hit or len(cluster_set & matched_set) > 0)

        pop_name = RAW_LEVEL_TO_POPULATION.get(
            target_lvl,
            RAW_LEVEL_TO_POPULATION.get(target_lvl.lstrip("'"), "Grade 4-6 - Elementary Intermediate"),
        )
        target_cats = set(POPULATION_SPECS.get(pop_name, {}).get("engine_categories", []))
        matched_cats: Set[str] = set()
        for m in matched_set:
            for c in LEVEL_TO_CATEGORIES.get(m, []) or LEVEL_TO_CATEGORIES.get(f"'{m}", []):
                matched_cats.add(c)
        category_hit = bool(cluster_hit or len(target_cats & matched_cats) > 0)

        # Signed IQR Residual Regime (RQ2)
        if exact_hit:
            signed_iqr_residual = 0.0
            failure_regime = "Converged"
        elif pd.notna(q3_t) and y_pred > q3_t:
            signed_iqr_residual = round(y_pred - q3_t, 4)
            failure_regime = "Under-Simplified (Too Complex)"
        elif pd.notna(q1_t) and y_pred < q1_t:
            signed_iqr_residual = round(y_pred - q1_t, 4)
            failure_regime = "Over-Simplified (Too Simple)"
        else:
            signed_iqr_residual = 0.0
            failure_regime = "Out-of-Bounds"

        dist_to_iqr = round(abs(signed_iqr_residual), 4)
        abs_mu_dist = round(abs(y_pred - mu_t), 4) if pd.notna(mu_t) else np.nan
        scale_norm_iqr_dist = compute_scale_normalized_iqr_distance(y_pred, q1_t, q3_t)

        # Target probability & category probability sum
        target_prob = max(
            [parsed_probs.get(a, 0.0) for a in target_aliases] + [0.0]
        )
        cat_prob_sum = sum(
            p
            for lvl, p in parsed_probs.items()
            if set(LEVEL_TO_CATEGORIES.get(lvl, []) or LEVEL_TO_CATEGORIES.get(lvl.lstrip("'"), []))
            & target_cats
        )

        # Track-Separated Weighted Empirical Anchor Error (Delta mu_weighted)
        weighted_grade_err = compute_track_separated_weighted_grade_err(
            target_level=target_lvl,
            matched_levels=matched_lvls,
            parsed_probs=parsed_probs,
            k12_probs_json=row.get("k12_track_probs_json") if "k12_track_probs_json" in row else None,
            special_probs_json=row.get("special_track_probs_json") if "special_track_probs_json" in row else None,
            anchor_map=empirical_anchor_map,
            target_mu=mu_t if pd.notna(mu_t) else None,
        )

        target_anchor = (
            float(mu_t)
            if pd.notna(mu_t)
            else empirical_anchor_map.get(
                target_lvl, empirical_anchor_map.get(target_lvl.lstrip("'"), 5.31)
            )
        )
        if exact_hit:
            min_iqr_grade_err = 0.0
        elif matched_lvls:
            min_iqr_grade_err = round(
                min(
                    abs(
                        empirical_anchor_map.get(
                            m, empirical_anchor_map.get(m.lstrip("'"), target_anchor)
                        )
                        - target_anchor
                    )
                    for m in matched_lvls
                ),
                4,
            )
        else:
            min_iqr_grade_err = weighted_grade_err

        words_simp = len(simp_text.split())
        words_src = max(1, len(src_text.split()))

        row_records.append(
            {
                **row.to_dict(),
                "exact_iqr_hit": int(exact_hit),
                "cluster_iqr_hit": int(cluster_hit),
                "category_iqr_hit": int(category_hit),
                "signed_iqr_residual": signed_iqr_residual,
                "dist_to_target_iqr": dist_to_iqr,
                "scale_norm_iqr_dist": scale_norm_iqr_dist,
                "abs_mu_distance": abs_mu_dist,
                "failure_regime": failure_regime,
                "target_prob_pct": round(target_prob, 2),
                "cat_prob_sum_pct": round(min(100.0, cat_prob_sum), 2),
                "weighted_grade_err": weighted_grade_err,
                "min_iqr_grade_err": min_iqr_grade_err,
                "simplified_words": words_simp,
                "source_words": words_src,
                "compression_ratio": round(words_simp / words_src, 4),
            }
        )

    row_df = pd.DataFrame(row_records)
    has_bert = (
        "dictabert_bertscore_f1" in row_df.columns
        and row_df["dictabert_bertscore_f1"].notna().all()
    )
    has_chrf = (
        "chrf_plus_plus" in row_df.columns
        and row_df["chrf_plus_plus"].notna().all()
    )
    if not (has_bert and has_chrf):
        bert_scores, chrf_scores = compute_preservation_batch(
            row_df["text"].tolist(),
            row_df["simplified_text"].tolist(),
        )
        if not has_bert:
            row_df["dictabert_bertscore_f1"] = bert_scores
        if not has_chrf:
            row_df["chrf_plus_plus"] = chrf_scores
    else:
        row_df["dictabert_bertscore_f1"] = row_df["dictabert_bertscore_f1"].astype(float).round(4)
        row_df["chrf_plus_plus"] = row_df["chrf_plus_plus"].astype(float).round(2)
    
    if not (has_bert and has_chrf):
        logger.info("Computing Nikkud-stripped DictaBERT BERTScore-F1 and chrF++ for %d rows...", len(row_df))
        bert_scores, chrf_scores = compute_preservation_batch(
            row_df["text"].tolist(),
            row_df["simplified_text"].tolist(),
        )
        if not has_bert:
            row_df["dictabert_bertscore_f1"] = bert_scores
        if not has_chrf:
            row_df["chrf_plus_plus"] = chrf_scores
    else:
        logger.info("Reusing precomputed DictaBERT BERTScore-F1 and chrF++ columns from input CSV.")
        row_df["dictabert_bertscore_f1"] = row_df["dictabert_bertscore_f1"].astype(float).round(4)
        row_df["chrf_plus_plus"] = row_df["chrf_plus_plus"].astype(float).round(2)
    
    row_df.to_csv(out_dir / "row_level_proximity_analysis.csv", index=False)

    # 1. Method Quality Summary (4 rows)
    method_summary = (
        row_df.groupby("prompt", as_index=False)
        .agg(
            n_rows=("paragraph_id", "count"),
            exact_iqr_hit_pct=("exact_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            cluster_iqr_hit_pct=("cluster_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            category_iqr_hit_pct=("category_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            under_simplified_pct=(
                "failure_regime",
                lambda s: round((s == "Under-Simplified (Too Complex)").mean() * 100.0, 2),
            ),
            over_simplified_pct=(
                "failure_regime",
                lambda s: round((s == "Over-Simplified (Too Simple)").mean() * 100.0, 2),
            ),
            mean_signed_iqr_residual=("signed_iqr_residual", lambda s: round(s.mean(), 4)),
            mean_dist_to_iqr=("dist_to_target_iqr", lambda s: round(s.mean(), 4)),
            mean_scale_norm_iqr_dist=("scale_norm_iqr_dist", lambda s: round(s.mean(), 4)),
            mean_target_prob_pct=("target_prob_pct", lambda s: round(s.mean(), 2)),
            mean_cat_prob_sum_pct=("cat_prob_sum_pct", lambda s: round(s.mean(), 2)),
            mean_weighted_grade_err=("weighted_grade_err", lambda s: round(s.mean(), 4)),
            mean_min_iqr_grade_err=("min_iqr_grade_err", lambda s: round(s.mean(), 4)),
            mean_simplified_score=("simplified_score", lambda s: round(s.mean(), 4)),
            std_simplified_score=("simplified_score", lambda s: round(s.std(), 4)),
            mean_words=("simplified_words", lambda s: round(s.mean(), 1)),
            mean_compression_ratio=("compression_ratio", lambda s: round(s.mean(), 2)),
            mean_bertscore_f1=("dictabert_bertscore_f1", lambda s: round(s.mean(), 4)),
            std_bertscore_f1=("dictabert_bertscore_f1", lambda s: round(s.std(), 4)),
            mean_chrf_plus_plus=("chrf_plus_plus", lambda s: round(s.mean(), 2)),
            std_chrf_plus_plus=("chrf_plus_plus", lambda s: round(s.std(), 2)),
        )
        .sort_values("exact_iqr_hit_pct", ascending=False)
    )
    method_summary.to_csv(out_dir / "method_quality_summary.csv", index=False)

    # 2. Exact 14 x 4 Per-Level Breakdown Table
    level_by_method = (
        row_df.groupby(["target_level", "prompt"], as_index=False)
        .agg(
            n_items=("paragraph_id", "count"),
            exact_hits=("exact_iqr_hit", "sum"),
            exact_iqr_hit_pct=("exact_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            cluster_iqr_hit_pct=("cluster_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            category_iqr_hit_pct=("category_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            under_simplified_pct=(
                "failure_regime",
                lambda s: round((s == "Under-Simplified (Too Complex)").mean() * 100.0, 2),
            ),
            over_simplified_pct=(
                "failure_regime",
                lambda s: round((s == "Over-Simplified (Too Simple)").mean() * 100.0, 2),
            ),
            mean_signed_residual=("signed_iqr_residual", lambda s: round(s.mean(), 4)),
            mean_scale_norm_iqr_dist=("scale_norm_iqr_dist", lambda s: round(s.mean(), 4)),
            mean_weighted_grade_err=("weighted_grade_err", lambda s: round(s.mean(), 4)),
            mean_min_iqr_grade_err=("min_iqr_grade_err", lambda s: round(s.mean(), 4)),
            mean_score=("simplified_score", lambda s: round(s.mean(), 4)),
            mean_bertscore_f1=("dictabert_bertscore_f1", lambda s: round(s.mean(), 4)),
            mean_chrf_plus_plus=("chrf_plus_plus", lambda s: round(s.mean(), 2)),
        )
    )
    level_by_method.to_csv(out_dir / "level_by_method_summary.csv", index=False)

    # Wide 14 x 4 Pivot Table (Hit Count / N and Hit % per Target Level x Prompt)
    level_by_method["formatted_hit"] = level_by_method.apply(
        lambda r: f"{int(r['exact_hits'])}/{int(r['n_items'])} ({r['exact_iqr_hit_pct']:.1f}%)",
        axis=1,
    )
    available_prompts = [p for p in PROMPT_ORDER if p in level_by_method["prompt"].unique()]
    pivot_14x4 = level_by_method.pivot_table(
        index="target_level", columns="prompt", values="formatted_hit", aggfunc="first"
    )[available_prompts].reset_index()
    pivot_14x4.to_csv(out_dir / "level_by_method_pivot_14x4.csv", index=False)

    # 3. Paired Statistical Tests (Cochran's Q, Exact McNemar, Friedman, Wilcoxon + r_rb)
    stat_rows: List[Dict[str, Any]] = []
    piv_exact = row_df.pivot_table(index=["paragraph_id", "target_level"], columns="prompt", values="exact_iqr_hit", aggfunc="last").dropna()
    piv_min_err = row_df.pivot_table(index=["paragraph_id", "target_level"], columns="prompt", values="min_iqr_grade_err", aggfunc="last").dropna()
    piv_w_err = row_df.pivot_table(index=["paragraph_id", "target_level"], columns="prompt", values="weighted_grade_err", aggfunc="last").dropna()
    piv_bert = row_df.pivot_table(index=["paragraph_id", "target_level"], columns="prompt", values="dictabert_bertscore_f1", aggfunc="last").dropna()
    piv_chrf = row_df.pivot_table(index=["paragraph_id", "target_level"], columns="prompt", values="chrf_plus_plus", aggfunc="last").dropna()

    if len(available_prompts) >= 2 and len(piv_exact) > 0:
        q_stat, q_pval = cochrans_q_test(piv_exact[available_prompts].to_numpy())
        f_stat, f_pval = safe_friedman_test(piv_min_err, available_prompts)
        f_bert_stat, f_bert_pval = safe_friedman_test(piv_bert, available_prompts)
        f_chrf_stat, f_chrf_pval = safe_friedman_test(piv_chrf, available_prompts)

        baselines = [p for p in available_prompts if p != "agentic"]
        mcnemar_raw_p: List[float] = []
        wilcox_min_raw_p: List[float] = []
        wilcox_w_raw_p: List[float] = []
        wilcox_bert_raw_p: List[float] = []
        wilcox_chrf_raw_p: List[float] = []
        temp_records: List[Dict[str, Any]] = []

        for base_p in baselines:
            if "agentic" not in piv_exact.columns:
                continue
            mc = exact_mcnemar_test(
                piv_exact["agentic"].to_numpy(), piv_exact[base_p].to_numpy()
            )
            wx_min = paired_wilcoxon_with_rank_biserial(
                piv_min_err[base_p].to_numpy(), piv_min_err["agentic"].to_numpy()
            )
            wx_w = paired_wilcoxon_with_rank_biserial(
                piv_w_err[base_p].to_numpy(), piv_w_err["agentic"].to_numpy()
            )
            wx_bert = paired_wilcoxon_with_rank_biserial(
                piv_bert["agentic"].to_numpy(), piv_bert[base_p].to_numpy()
            )
            wx_chrf = paired_wilcoxon_with_rank_biserial(
                piv_chrf["agentic"].to_numpy(), piv_chrf[base_p].to_numpy()
            )

            mcnemar_raw_p.append(mc["p_value_raw"])
            wilcox_min_raw_p.append(wx_min["p_value_raw"])
            wilcox_w_raw_p.append(wx_w["p_value_raw"])
            wilcox_bert_raw_p.append(wx_bert["p_value_raw"])
            wilcox_chrf_raw_p.append(wx_chrf["p_value_raw"])

            temp_records.append(
                {
                    "comparison": f"agentic vs. {base_p}",
                    "n_paired_blocks": len(piv_exact),
                    "cochran_q_omnibus": q_stat,
                    "cochran_q_p_value": q_pval,
                    "friedman_chi2_omnibus": f_stat,
                    "friedman_p_value": f_pval,
                    "friedman_bertscore_chi2": f_bert_stat,
                    "friedman_bertscore_p_value": f_bert_pval,
                    "friedman_chrf_chi2": f_chrf_stat,
                    "friedman_chrf_p_value": f_chrf_pval,
                    **mc,
                    "wilcoxon_min_iqr_err_stat": wx_min["wilcoxon_stat"],
                    "wilcoxon_min_iqr_err_p_raw": wx_min["p_value_raw"],
                    "wilcoxon_min_iqr_rank_biserial_r_rb": wx_min["rank_biserial_r_rb"],
                    "wilcoxon_weighted_err_stat": wx_w["wilcoxon_stat"],
                    "wilcoxon_weighted_err_p_raw": wx_w["p_value_raw"],
                    "wilcoxon_weighted_rank_biserial_r_rb": wx_w["rank_biserial_r_rb"],
                    "wilcoxon_bertscore_stat": wx_bert["wilcoxon_stat"],
                    "wilcoxon_bertscore_p_raw": wx_bert["p_value_raw"],
                    "wilcoxon_bertscore_rank_biserial_r_rb": wx_bert["rank_biserial_r_rb"],
                    "wilcoxon_chrf_stat": wx_chrf["wilcoxon_stat"],
                    "wilcoxon_chrf_p_raw": wx_chrf["p_value_raw"],
                    "wilcoxon_chrf_rank_biserial_r_rb": wx_chrf["rank_biserial_r_rb"],
                }
            )

        mc_adj = holm_bonferroni_adjust(mcnemar_raw_p)
        wx_min_adj = holm_bonferroni_adjust(wilcox_min_raw_p)
        wx_w_adj = holm_bonferroni_adjust(wilcox_w_raw_p)
        wx_bert_adj = holm_bonferroni_adjust(wilcox_bert_raw_p)
        wx_chrf_adj = holm_bonferroni_adjust(wilcox_chrf_raw_p)

        for idx, rec in enumerate(temp_records):
            rec["mcnemar_p_holm_bonferroni"] = mc_adj[idx]
            rec["wilcoxon_min_iqr_err_p_holm"] = wx_min_adj[idx]
            rec["wilcoxon_weighted_err_p_holm"] = wx_w_adj[idx]
            rec["wilcoxon_bertscore_p_holm"] = wx_bert_adj[idx]
            rec["wilcoxon_chrf_p_holm"] = wx_chrf_adj[idx]
            stat_rows.append(rec)

    paired_stats_df = pd.DataFrame(stat_rows)
    paired_stats_df.to_csv(out_dir / "paired_statistical_tests.csv", index=False)
    logger.info("Exported proximity and paired statistical analysis tables to %s", out_dir)
    
    return {
        "row_level": row_df,
        "method_summary": method_summary,
        "level_by_method": level_by_method,
        "pivot_14x4": pivot_14x4,
        "paired_tests": paired_stats_df,
    }


def run_full_analysis(
    input_csv: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Notebook helper returning (row_level_df, method_summary_df)."""
    outputs = analyze_experimentation_results(results_csv=Path(input_csv), output_dir=output_dir)
    return outputs["row_level"], outputs["method_summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Hebrew Simplification Experimentation Results")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experimentation_results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()

    cli_logger = setup_pipeline_logger(
        name="AnalyzeExperimentation",
        artifact_dir=str(args.output_dir),
        log_filename="experimentation_analysis.log",
    )
    outputs = analyze_experimentation_results(args.results_csv, args.output_dir)
    cli_logger.info("\n=== 1. METHOD QUALITY SUMMARY ===\n%s", outputs["method_summary"].to_string(index=False))
    cli_logger.info("\n=== 2. EXACT 14 x 4 PER-LEVEL HIT RATE TABLE ===\n%s", outputs["pivot_14x4"].to_string(index=False))
    cli_logger.info("\n=== 3. PAIRED STATISTICAL TESTS (McNEMAR & WILCOXON) ===\n%s", outputs["paired_tests"].to_string(index=False))
