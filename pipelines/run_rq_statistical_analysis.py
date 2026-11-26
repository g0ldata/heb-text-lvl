"""Complete RQ1-RQ7 Statistical & Qualitative Thesis Evaluation Suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.analyze_experimentation import (
    analyze_experimentation_results,
    compute_preservation_batch,
)

from hebrew_readability.config import (
    FINAL_FEATURE_ORDER,
    build_empirical_level_anchor_map,
    POPULATION_SPECS,
    RAW_LEVEL_TO_POPULATION,
    setup_pipeline_logger,
)

NIKKUD_STRIP_RE = re.compile(r"[\u0591-\u05C7]")
EMPIRICAL_ANCHORS = build_empirical_level_anchor_map()
# Sorted empirical complexity thresholds across the 13 distinct anchor tiers
SORTED_ANCHOR_TIERS = sorted(set(EMPIRICAL_ANCHORS.values()))

# ---------------------------------------------------------------------------
# RQ3: 3-Turn Markov Transition Matrix, Hazard Convergence, Ping-Pong & Stagnation
# ---------------------------------------------------------------------------
def _classify_turn_state(turn: Dict[str, Any]) -> str:
    """Maps a single trajectory turn to {S_Complex, S_Converged, S_Simple}."""
    if turn.get("in_core_iqr"):
        return "S_Converged"
    y = float(turn.get("predicted_score", 5.0))
    q3 = float(turn.get("target_q3", 5.5))
    q1 = float(turn.get("target_q1", 4.5))
    if y > q3:
        return "S_Complex"
    if y < q1:
        return "S_Simple"
    return "S_Converged"
    
def run_rq3_markov_trajectory_analysis(
    trajectories_jsonl: Path, output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the 3-Turn Markov State Transition Matrix over {S_Complex, S_Converged, S_Simple},
    computes Cumulative Convergence at Turn t=1,2,3, Ping-Pong Oscillation Rate, and Stagnation Rate.
    """
    if not trajectories_jsonl.exists():
        return pd.DataFrame(), pd.DataFrame()

    records: List[Dict[str, Any]] = []
    with open(trajectories_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    states = ["S_Complex", "S_Converged", "S_Simple"]
    trans_counts = {s_from: {s_to: 0 for s_to in states} for s_from in states}

    n_total = len(records)
    conv_by_turn = {1: 0, 2: 0, 3: 0}
    ping_pong_transitions = 0
    total_non_conv_transitions = 0
    stagnant_transitions = 0

    for rec in records:
        traj = rec.get("trajectory") or []
        if not traj:
            continue
        # Cumulative convergence by turn t
        hit_so_far = False
        for t_idx in (1, 2, 3):
            if t_idx <= len(traj) and traj[t_idx - 1].get("in_core_iqr"):
                hit_so_far = True
            if hit_so_far:
                conv_by_turn[t_idx] += 1

        for idx in range(len(traj) - 1):
            t_curr = traj[idx]
            t_next = traj[idx + 1]
            s_curr = _classify_turn_state(t_curr)
            s_next = _classify_turn_state(t_next)
            trans_counts[s_curr][s_next] += 1

            if s_curr != "S_Converged":
                total_non_conv_transitions += 1
                if (s_curr == "S_Complex" and s_next == "S_Simple") or (
                    s_curr == "S_Simple" and s_next == "S_Complex"
                ):
                    ping_pong_transitions += 1
                y_curr = float(t_curr.get("predicted_score", 0.0))
                y_next = float(t_next.get("predicted_score", 0.0))
                if abs(y_next - y_curr) < 0.015:
                    stagnant_transitions += 1

    # Absorb converged state for trajectories that stopped early
    trans_matrix_rows: List[Dict[str, Any]] = []
    for s_from in states:
        row_sum = sum(trans_counts[s_from].values())
        if s_from == "S_Converged" and row_sum == 0:
            trans_matrix_rows.append(
                {"from_state": s_from, "S_Complex": 0.0, "S_Converged": 100.0, "S_Simple": 0.0, "n_transitions": 0}
            )
        else:
            denom = max(1, row_sum)
            trans_matrix_rows.append(
                {
                    "from_state": s_from,
                    "S_Complex": round(trans_counts[s_from]["S_Complex"] / denom * 100.0, 2),
                    "S_Converged": round(trans_counts[s_from]["S_Converged"] / denom * 100.0, 2),
                    "S_Simple": round(trans_counts[s_from]["S_Simple"] / denom * 100.0, 2),
                    "n_transitions": row_sum,
                }
            )

    markov_df = pd.DataFrame(trans_matrix_rows)
    markov_df.to_csv(output_dir / "rq3_markov_transition_matrix.csv", index=False)

    dynamics_summary = pd.DataFrame(
        [
            {
                "n_trajectories": n_total,
                "cum_convergence_turn_1_pct": round(conv_by_turn[1] / n_total * 100.0, 2),
                "cum_convergence_turn_2_pct": round(conv_by_turn[2] / n_total * 100.0, 2),
                "cum_convergence_turn_3_pct": round(conv_by_turn[3] / n_total * 100.0, 2),
                "marginal_rescue_turns_2_3_count": conv_by_turn[3] - conv_by_turn[1],
                "ping_pong_oscillation_rate_pct": round(
                    ping_pong_transitions / max(1, total_non_conv_transitions) * 100.0, 2
                ),
                "stagnation_rate_pct": round(
                    stagnant_transitions / max(1, total_non_conv_transitions) * 100.0, 2
                ),
            }
        ]
    )
    dynamics_summary.to_csv(output_dir / "rq3_trajectory_dynamics_summary.csv", index=False)
    return markov_df, dynamics_summary


# ---------------------------------------------------------------------------
# RQ4: Linguistic Controllability / Conditional Standardized Feature Elasticity
# ---------------------------------------------------------------------------
def run_rq4_feature_elasticity_analysis(
    trajectories_jsonl: Path, output_dir: Path
) -> pd.DataFrame:
    """
    Computes Conditional Standardized Feature Shift:
    Elasticity_k = E[(x_{k, t+1} - x_{k, t}) / IQR_k | k in Flagged_t]
    across all 11 engineered Hebrew features with paired t-test and Wilcoxon test against 0.
    """
    if not trajectories_jsonl.exists():
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    with open(trajectories_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Empirical feature IQRs across transformed [0,1] scale
    all_feat_vals: Dict[str, List[float]] = {k: [] for k in FINAL_FEATURE_ORDER}
    for rec in records:
        for t in rec.get("trajectory") or []:
            tf = t.get("transformed_features") or {}
            for k in FINAL_FEATURE_ORDER:
                if k in tf:
                    all_feat_vals[k].append(float(tf[k]))

    feat_iqr: Dict[str, float] = {}
    for k, vals in all_feat_vals.items():
        if len(vals) >= 4:
            q75, q25 = np.percentile(vals, [75, 25])
            feat_iqr[k] = max(float(q75 - q25), 0.05)
        else:
            feat_iqr[k] = 0.15

    shifts_when_flagged: Dict[str, List[float]] = {k: [] for k in FINAL_FEATURE_ORDER}
    directional_shifts: Dict[str, List[float]] = {k: [] for k in FINAL_FEATURE_ORDER}

    for rec in records:
        traj = rec.get("trajectory") or []
        for idx in range(len(traj) - 1):
            t_curr = traj[idx]
            t_next = traj[idx + 1]
            tf_curr = t_curr.get("transformed_features") or {}
            tf_next = t_next.get("transformed_features") or {}
            flagged_keys = set(t_curr.get("top_offending_keys") or [])
            flagged_names = set(t_curr.get("top_offending_features") or [])

            y_curr = float(t_curr.get("predicted_score", 5.0))
            q3 = float(t_curr.get("target_q3", 5.5))
            desired_sign = -1.0 if y_curr > q3 else 1.0

            for k in FINAL_FEATURE_ORDER:
                if k in tf_curr and k in tf_next:
                    is_flagged = (k in flagged_keys) or any(k[:6] in fn for fn in flagged_names)
                    if is_flagged:
                        raw_shift = (float(tf_next[k]) - float(tf_curr[k])) / feat_iqr[k]
                        shifts_when_flagged[k].append(raw_shift)
                        directional_shifts[k].append(raw_shift * desired_sign)

    rows: List[Dict[str, Any]] = []
    for k in FINAL_FEATURE_ORDER:
        arr = np.array(shifts_when_flagged[k], dtype=float)
        dir_arr = np.array(directional_shifts[k], dtype=float)
        n_obs = len(arr)
        if n_obs >= 2:
            t_res = stats.ttest_1samp(dir_arr, popmean=0.0)
            t_stat, t_p = round(float(t_res.statistic), 4), round(float(t_res.pvalue), 6)
            try:
                w_res = stats.wilcoxon(dir_arr)
                w_p = round(float(w_res.pvalue), 6)
            except Exception:
                w_p = 1.0
            mean_shift = round(float(np.mean(arr)), 4)
            mean_obedience = round(float(np.mean(dir_arr)), 4)
        elif n_obs == 1:
            t_stat, t_p, w_p = 0.0, 1.0, 1.0
            mean_shift = round(float(arr[0]), 4)
            mean_obedience = round(float(dir_arr[0]), 4)
        else:
            t_stat, t_p, w_p, mean_shift, mean_obedience = 0.0, 1.0, 1.0, 0.0, 0.0

        rows.append(
            {
                "feature_key": k,
                "n_times_flagged": n_obs,
                "feature_iqr_scale": round(feat_iqr[k], 4),
                "mean_standardized_shift": mean_shift,
                "mean_directional_elasticity": mean_obedience,
                "t_stat_vs_zero": t_stat,
                "t_test_p_value": t_p,
                "wilcoxon_p_value": w_p,
            }
        )

    elasticity_df = pd.DataFrame(rows).sort_values("n_times_flagged", ascending=False)
    elasticity_df.to_csv(output_dir / "rq4_feature_elasticity.csv", index=False)
    return elasticity_df


# ---------------------------------------------------------------------------
# RQ5: Source Passage Complexity & Length Sensitivity (GEE / GLMM Logistic Regression)
# ---------------------------------------------------------------------------
def run_rq5_mixed_effects_length_sensitivity(
    row_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """
    Fits a Generalized Estimating Equations (GEE) Logistic Regression (Binomial family,
    Exchangeable correlation clustered on paragraph_id) predicting exact_iqr_hit from
    prompt condition, simplification distance |original_score - target_mu|, and source_words.
    """
    df_model = row_df.copy()
    if "original_score" in df_model.columns and "target_mu" in df_model.columns:
        df_model["simplification_distance"] = (
            df_model["original_score"].astype(float) - df_model["target_mu"].astype(float)
        ).abs()
    else:
        df_model["simplification_distance"] = df_model["dist_to_target_iqr"].astype(float)

    df_model["long_passage_gt65"] = (df_model["source_words"].astype(float) > 65.0).astype(int)

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = (
            "exact_iqr_hit ~ C(prompt, Treatment(reference='cot_persona')) "
            "+ simplification_distance + source_words"
        )
        gee_model = smf.gee(
            formula=formula,
            groups="paragraph_id",
            data=df_model,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        gee_res = gee_model.fit()
        summary_df = pd.DataFrame(
            {
                "term": gee_res.params.index,
                "coef_log_odds": np.round(gee_res.params.values, 4),
                "odds_ratio": np.round(np.exp(gee_res.params.values), 4),
                "std_err": np.round(gee_res.bse.values, 4),
                "z_stat": np.round(gee_res.tvalues.values, 4),
                "p_value": np.round(gee_res.pvalues.values, 6),
            }
        )
    except Exception as exc:
        # Fallback logistic correlation summary if statsmodels is unavailable
        r_len, p_len = stats.pointbiserialr(df_model["exact_iqr_hit"], df_model["source_words"])
        r_dist, p_dist = stats.pointbiserialr(
            df_model["exact_iqr_hit"], df_model["simplification_distance"]
        )
        summary_df = pd.DataFrame(
            [
                {"term": "source_words_pointbiserial", "coef_log_odds": round(float(r_len), 4), "p_value": round(float(p_len), 6), "note": str(exc)},
                {"term": "simplification_distance_pointbiserial", "coef_log_odds": round(float(r_dist), 4), "p_value": round(float(p_dist), 6), "note": str(exc)},
            ]
        )

    summary_df.to_csv(output_dir / "rq5_gee_length_complexity_regression.csv", index=False)
    return summary_df


# ---------------------------------------------------------------------------
# RQ6: LLM Self-Awareness vs. XGBoost Ground Truth (Containment & Quadratic Weighted Kappa)
# ---------------------------------------------------------------------------
def compute_quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], min_rating: int = 1, max_rating: int = 13) -> float:
    """Computes Cohen's Quadratic Weighted Kappa on ordinal grade bins."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return 0.0
    num_ratings = max_rating - min_rating + 1
    O = np.zeros((num_ratings, num_ratings), dtype=float)
    for t, p in zip(y_true, y_pred):
        ti = int(np.clip(t, min_rating, max_rating)) - min_rating
        pi = int(np.clip(p, min_rating, max_rating)) - min_rating
        O[ti, pi] += 1.0

    N = float(len(y_true))
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    E = np.outer(hist_true, hist_pred) / max(1.0, N)

    W = np.zeros((num_ratings, num_ratings), dtype=float)
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    denom = np.sum(W * E)
    if denom <= 0:
        return 0.0
    return round(float(1.0 - (np.sum(W * O) / denom)), 4)


def _map_level_string_to_ordinal_grade(lvl_str: str) -> int:
    """
    Maps a raw level or population string to its empirical anchor mu_L,
    then discretizes into 1..13 complexity bins for Quadratic Weighted Kappa.
    """
    clean = str(lvl_str).strip()
    if clean in EMPIRICAL_ANCHORS:
        mu_val = EMPIRICAL_ANCHORS[clean]
    elif clean.lstrip("'") in EMPIRICAL_ANCHORS:
        mu_val = EMPIRICAL_ANCHORS[clean.lstrip("'")]
    elif clean in POPULATION_SPECS:
        mu_val = float(POPULATION_SPECS[clean]["anchor_score"])
    else:
        mu_val = 5.31

    # Find closest empirical tier index in 1..13
    closest_idx = min(
        range(len(SORTED_ANCHOR_TIERS)),
        key=lambda i: abs(SORTED_ANCHOR_TIERS[i] - mu_val),
    )
    return int(np.clip(closest_idx + 1, 1, 13))


def run_rq6_llm_self_awareness_analysis(
    row_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """
    Evaluates Set-Intersection Containment Accuracy (%) and Quadratic Weighted Kappa (kappa_w)
    between the LLM's original_level estimate and the engine's ground-truth original_tool_iqr_levels.
    """
    rows: List[Dict[str, Any]] = []
    for prompt_name, grp in row_df.groupby("prompt"):
        if prompt_name == "agentic":
            continue
        hits = 0
        y_true_ord: List[int] = []
        y_pred_ord: List[int] = []

        for _, r in grp.iterrows():
            llm_est = str(r.get("original_level") or "").strip()
            tool_gt_str = str(r.get("original_tool_iqr_levels") or r.get("original_level") or "")
            tool_lvls = [t.strip() for t in tool_gt_str.split(",") if t.strip()]

            # Check set intersection containment at raw or population level
            llm_pop = RAW_LEVEL_TO_POPULATION.get(llm_est, llm_est)
            tool_pops = {RAW_LEVEL_TO_POPULATION.get(t, t) for t in tool_lvls}
            if llm_est in tool_lvls or llm_pop in tool_pops or any(t in llm_est for t in tool_lvls):
                hits += 1

            gt_grades = [_map_level_string_to_ordinal_grade(t) for t in tool_lvls] if tool_lvls else [8]
            y_true_ord.append(int(round(float(np.mean(gt_grades)))))
            y_pred_ord.append(_map_level_string_to_ordinal_grade(llm_est))

        n_items = max(1, len(grp))
        kw = compute_quadratic_weighted_kappa(y_true_ord, y_pred_ord)
        rows.append(
            {
                "prompt": prompt_name,
                "n_evaluated": len(grp),
                "containment_accuracy_pct": round(hits / n_items * 100.0, 2),
                "quadratic_weighted_kappa_w": kw,
            }
        )

    rq6_df = pd.DataFrame(rows)
    rq6_df.to_csv(output_dir / "rq6_llm_self_awareness.csv", index=False)
    return rq6_df


# ---------------------------------------------------------------------------
# RQ7: Simplicity-Fidelity Pareto Frontier, Error Taxonomy & Krippendorff's Alpha
# ---------------------------------------------------------------------------
def _extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts whitespace-normalized character n-grams."""
    normalized = re.sub(r"\s+", " ", text)
    return {normalized[i : i + n] for i in range(max(1, len(normalized) - n + 1))}
    
def compute_nikkud_stripped_semantic_f1(source_text: str, simplified_text: str) -> float:
    """
    Strips Nikkud diacritics and computes character trigram + content token F1 similarity
    (or DictaBERT embedding cosine F1 when transformers GPU pipeline is enabled).
    """
    src_clean = NIKKUD_STRIP_RE.sub("", str(source_text or "")).strip()
    simp_clean = NIKKUD_STRIP_RE.sub("", str(simplified_text or "")).strip()
    if not src_clean or not simp_clean:
        return 0.0

    g_src = _extract_char_ngrams(src_clean, 3)
    g_simp = _char_ngrams(simp_clean, 3)
    overlap = len(g_src & g_simp)
    prec = overlap / max(1, len(g_simp))
    rec = overlap / max(1, len(g_src))
    if prec + rec <= 0:
        return 0.0
    return round(2.0 * prec * rec / (prec + rec), 4)


def krippendorffs_alpha_ordinal(ratings_matrix: np.ndarray) -> float:
    """
    Computes Krippendorff's alpha for ordinal Likert human evaluation ratings
    (shape: n_annotators x n_items, with np.nan for missing ratings).
    """
    ratings = np.asarray(ratings_matrix, dtype=float)
    n_annotators, n_items = ratings.shape
    all_vals = ratings[~np.isnan(ratings)]
    if len(all_vals) < 2:
        return 0.0

    categories = np.sort(np.unique(all_vals))
    if len(categories) <= 1:
        return 1.0

    # Coincidence matrix O_ck
    K = len(categories)
    val_to_idx = {v: i for i, v in enumerate(categories)}
    O = np.zeros((K, K), dtype=float)

    for u in range(n_items):
        unit_vals = ratings[:, u]
        valid = unit_vals[~np.isnan(unit_vals)]
        m_u = len(valid)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    O[val_to_idx[valid[i]], val_to_idx[valid[j]]] += 1.0 / (m_u - 1.0)

    n_c = np.sum(O, axis=1)
    n_total = np.sum(n_c)
    if n_total <= 1:
        return 0.0

    # Ordinal metric distance delta^2_{ck}
    delta_sq = np.zeros((K, K), dtype=float)
    for c in range(K):
        for k in range(K):
            if c != k:
                low, high = min(c, k), max(c, k)
                cum_n = np.sum(n_c[low : high + 1]) - (n_c[low] + n_c[high]) / 2.0
                delta_sq[c, k] = cum_n ** 2

    D_o = np.sum(O * delta_sq)
    E = np.outer(n_c, n_c) / (n_total - 1.0)
    np.fill_diagonal(E, 0.0)
    D_e = np.sum(E * delta_sq)
    if D_e <= 0:
        return 1.0
    return round(float(1.0 - (D_o / D_e)), 4)


def run_rq7_pareto_and_error_taxonomy(
    row_df: pd.DataFrame, output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Simplicity-Fidelity Pareto metrics per prompt, classifies any remaining
    agentic misses into the 4-category Qualitative Error Taxonomy, and exports a
    25-quadruplet Blind Human Evaluation sheet.
    """
    df_copy = row_df.copy()

    has_bert = (
        "dictabert_bertscore_f1" in df_copy.columns
        and df_copy["dictabert_bertscore_f1"].notna().all()
    )
    has_chrf = (
        "chrf_plus_plus" in df_copy.columns
        and df_copy["chrf_plus_plus"].notna().all()
    )
    if not (has_bert and has_chrf):
        bert_scores, chrf_scores = compute_preservation_batch(
            df_copy["text"].tolist(),
            df_copy["simplified_text"].tolist(),
        )
        if not has_bert:
            df_copy["dictabert_bertscore_f1"] = bert_scores
        if not has_chrf:
            df_copy["chrf_plus_plus"] = chrf_scores

    df_copy["composite_pareto_success"] = (
        (df_copy["exact_iqr_hit"] == 1) & (df_copy["dictabert_bertscore_f1"] >= 0.75)
    ).astype(int)

    pareto_summary = (
        df_copy.groupby("prompt", as_index=False)
        .agg(
            exact_iqr_hit_pct=("exact_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            mean_bertscore_f1=("dictabert_bertscore_f1", lambda s: round(s.mean(), 4)),
            std_bertscore_f1=("dictabert_bertscore_f1", lambda s: round(s.std(), 4)),
            mean_chrf_plus_plus=("chrf_plus_plus", lambda s: round(s.mean(), 2)),
            std_chrf_plus_plus=("chrf_plus_plus", lambda s: round(s.std(), 2)),
            mean_compression_ratio=("compression_ratio", lambda s: round(s.mean(), 4)),
            composite_pareto_success_pct=(
                "composite_pareto_success",
                lambda s: round(s.mean() * 100.0, 2),
            ),
        )
        .sort_values("exact_iqr_hit_pct", ascending=False)
    )
    pareto_summary.to_csv(output_dir / "rq7_simplicity_fidelity_pareto.csv", index=False)

    # Qualitative Error Taxonomy on any Agentic misses
    agent_misses = df_copy[(df_copy["prompt"] == "agentic") & (df_copy["exact_iqr_hit"] == 0)].copy()
    taxonomy_rows: List[Dict[str, Any]] = []
    for _, r in agent_misses.iterrows():
        y = float(r["simplified_score"])
        lvl = str(r["target_level"]).strip()
        dist = float(r.get("dist_to_target_iqr", 0.0))

        if 3.2720 <= y <= 3.3898:
            cat = "Dead-Zone Boundary"
        elif lvl in ("3", "'2-4", "2-4", "Hadshon", "Olim_news") and (y < 3.35 or y > 5.10):
            cat = "Nikkud Ping-Pong"
        elif lvl in ("4", "intellectual_disability", "2", "3") and y >= 5.38 and r["source_words"] >= 55:
            cat = "Volume Floor"
        elif dist <= 0.08:
            cat = "Near-Boundary Miss (<=0.08 pts)"
        else:
            cat = "Stagnation / Syntactic Resistance"

        taxonomy_rows.append(
            {
                "paragraph_id": r["paragraph_id"],
                "target_level": lvl,
                "simplified_score": y,
                "signed_iqr_residual": r["signed_iqr_residual"],
                "error_category": cat,
            }
        )

    taxonomy_df = pd.DataFrame(taxonomy_rows)
    taxonomy_df.to_csv(output_dir / "rq7_agentic_error_taxonomy.csv", index=False)

    # Export stratified n=25 quadruplets for Blind Human Evaluation
    unique_pids = df_copy["paragraph_id"].drop_duplicates().head(25)
    blind_sheet = df_copy[df_copy["paragraph_id"].isin(unique_pids)][
        ["paragraph_id", "target_level", "prompt", "text", "simplified_text"]
    ].copy()
    blind_sheet["fluency_1_to_5"] = ""
    blind_sheet["factual_adequacy_1_to_5"] = ""
    blind_sheet["pedagogical_appropriateness_1_to_5"] = ""
    blind_sheet.to_csv(output_dir / "rq7_blind_human_eval_25_quadruplets.csv", index=False)

    return pareto_summary, taxonomy_df


def run_all_research_questions(
    results_csv: Path,
    trajectories_jsonl: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Executes RQ1 through RQ7 and saves all publication-ready CSV tables."""
    core_tables = analyze_experimentation_results(results_csv, output_dir)
    row_df = core_tables["row_level"]

    rq3_markov, rq3_dyn = run_rq3_markov_trajectory_analysis(trajectories_jsonl, output_dir)
    rq4_elast = run_rq4_feature_elasticity_analysis(trajectories_jsonl, output_dir)
    rq5_gee = run_rq5_mixed_effects_length_sensitivity(row_df, output_dir)
    rq6_aware = run_rq6_llm_self_awareness_analysis(row_df, output_dir)
    rq7_pareto, rq7_tax = run_rq7_pareto_and_error_taxonomy(row_df, output_dir)

    return {
        **core_tables,
        "rq3_markov": rq3_markov,
        "rq3_dynamics": rq3_dyn,
        "rq4_elasticity": rq4_elast,
        "rq5_gee": rq5_gee,
        "rq6_self_awareness": rq6_aware,
        "rq7_pareto": rq7_pareto,
        "rq7_error_taxonomy": rq7_tax,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Complete RQ1-RQ7 Statistical & Qualitative Analysis")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experimentation_results.csv",
    )
    parser.add_argument(
        "--trajectories-jsonl",
        type=Path,
        default=PROJECT_ROOT / "artifacts/agent_trajectories.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()

    cli_logger = setup_pipeline_logger(
        name="RQStatisticalAnalysis",
        artifact_dir=str(args.output_dir),
        log_filename="rq_analysis.log",
    )
    res = run_all_research_questions(args.results_csv, args.trajectories_jsonl, args.output_dir)
    cli_logger.info("RQ1-RQ7 analysis complete. All tables exported to: %s", args.output_dir)
