"""
Phase 4 (Deliverable 6): OpenAI Experimentation Pipeline & Analysis Script.

Executes cot_persona, few_shot_persona, feature_informed, and agentic (with Hybrid
Differential TreeSHAP Attribution) with strict ablation purity, tags above-range
scores as 'academic/technical', and saves both the 8-column CSV and enriched sidecar CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path("/content/heb-text-lvl")
)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hebrew_readability.config import (
    LEVEL_DESCRIPTIONS,
    LEVEL_SHAP_WEIGHTS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    RAW_LEVEL_MEDIANS,
    RAW_LEVEL_TO_POPULATION,
    setup_pipeline_logger,
)
from inference.agent import HebrewSimplificationAgent
from inference.engine import ReadabilityInferenceEngine

OUTPUT_COLUMNS: List[str] = [
    "paragraph_id",
    "text",
    "target_level",
    "prompt",
    "original_level",
    "simplified_text",
    "simplified_score",
    "simplified_level",
]

ENRICHED_COLUMNS: List[str] = OUTPUT_COLUMNS + [
    "original_score",
    "original_tool_iqr_levels",
    "target_q1",
    "target_q3",
    "target_mu",
    "iterations_used",
    "top_shap_offenders",
    "word_retention_ratio",
]

DEFAULT_PROMPTS: List[str] = [
    "cot_persona",
    "few_shot_persona",
    "feature_informed",
    "agentic",
]

PROMPT_FILES: Dict[str, str] = {
    "cot_persona": "prompts/persona_cot_prompt.json",
    "few_shot_persona": "prompts/few_shot_prompt.json",
    "feature_informed": "prompts/feature_informed_prompt.json",
}


# ---------------------------------------------------------------------------
# 1. LEVEL RESOLUTION & FORMATTING HELPERS
# ---------------------------------------------------------------------------
def resolve_population_and_description(target_level: str) -> Tuple[str, str]:
    """Maps a raw target_level code to its 7-population profile name and level description."""
    clean_lvl = str(target_level).strip()
    if clean_lvl in POPULATION_SPECS:
        return clean_lvl, clean_lvl

    pop_name = (
        RAW_LEVEL_TO_POPULATION.get(clean_lvl)
        or RAW_LEVEL_TO_POPULATION.get(clean_lvl.lstrip("'"))
        or "Grade 4-6 - Elementary Intermediate"
    )
    lvl_desc = (
        LEVEL_DESCRIPTIONS.get(clean_lvl)
        or LEVEL_DESCRIPTIONS.get(f"'{clean_lvl}")
        or LEVEL_DESCRIPTIONS.get(clean_lvl.lstrip("'"))
        or f"Level {clean_lvl} ({pop_name})"
    )
    return pop_name, lvl_desc


def get_raw_level_iqr(engine: Any, target_level: str) -> Tuple[float, float, float]:
    """Returns calibrated (Q1, Q3, mu) for a specific raw level from engine."""
    if hasattr(engine, "get_raw_level_iqr_bounds"):
        return engine.get_raw_level_iqr_bounds(target_level)

    clean_lvl = str(target_level).strip()
    candidates = RAW_LEVEL_ALIASES.get(
        clean_lvl, [clean_lvl, f"'{clean_lvl}", clean_lvl.lstrip("'")]
    )
    cal_stats = getattr(engine, "calibration_stats", {})
    for cand in candidates:
        if cand in cal_stats:
            s = cal_stats[cand]
            return round(float(s["Q1"]), 4), round(float(s["Q3"]), 4), round(float(s["mu"]), 4)

    pop_name, _ = resolve_population_and_description(clean_lvl)
    anchor = float(POPULATION_SPECS[pop_name]["anchor_score"])
    return round(anchor - 0.45, 4), round(anchor + 0.45, 4), anchor


def format_iqr_levels_plain(eval_res: Dict[str, Any], engine: Optional[Any] = None) -> str:
    """
    Formats original_level for the agentic pipeline: comma-separated raw levels in Core IQR.
    Tags any text above the calibrated readability score ranges as 'academic/technical'.
    """
    y_pred = float(eval_res.get("raw_readability_score", 0.0))
    max_q3 = float(getattr(engine, "max_q3", 6.05)) if engine is not None else 6.05
    core_matches = [str(m).strip() for m in (eval_res.get("core_iqr_matches") or [])]

    if eval_res.get("is_above_calibrated_range") or (y_pred > max_q3 and not core_matches):
        return "academic/technical"

    if core_matches:
        return ", ".join(core_matches)

    probs = eval_res.get("probabilities") or {}
    if probs:
        top_lvl = next(iter(probs.keys()))
        return f"NONE_IN_IQR (Closest: {top_lvl})"
    return "NONE_IN_IQR"


def format_iqr_levels_with_probs(eval_res: Dict[str, Any], engine: Optional[Any] = None) -> str:
    """
    Formats simplified_level: comma-separated list of raw levels whose Core IQR (Q1-Q3)
    contains simplified_score, annotated with standardized Core IQR probabilities.
    Tags above-range scores as 'academic/technical (100.0%)'.
    """
    y_pred = float(eval_res.get("raw_readability_score", 0.0))
    max_q3 = float(getattr(engine, "max_q3", 6.05)) if engine is not None else 6.05
    core_matches = [str(m).strip() for m in (eval_res.get("core_iqr_matches") or [])]
    probs = eval_res.get("core_iqr_probabilities") or eval_res.get("probabilities") or {}

    if eval_res.get("is_above_calibrated_range") or (y_pred > max_q3 and not core_matches):
        return "academic/technical (100.0%)"

    if not core_matches:
        all_probs = eval_res.get("probabilities") or {}
        if all_probs:
            top_lvl, top_prob = next(iter(all_probs.items()))
            return f"NONE_IN_IQR (Closest: {top_lvl} [{top_prob}%])"
        return "NONE_IN_IQR"

    ordered_matches = sorted(
        core_matches,
        key=lambda lvl: float(probs.get(lvl, 0.0)),
        reverse=True,
    )
    formatted = [
        f"{lvl} ({probs[lvl]}%)" if lvl in probs else str(lvl)
        for lvl in ordered_matches
    ]
    return ", ".join(formatted)


# ---------------------------------------------------------------------------
# 2. ABLATION-PURE STATIC PROMPT BUILDER
# ---------------------------------------------------------------------------
def build_static_prompt_messages(
    prompt_name: str,
    prompt_cfg: Dict[str, Any],
    source_text: str,
    target_level: str,
) -> List[Dict[str, str]]:
    """
    Builds system and user messages for cot_persona, few_shot_persona, or feature_informed
    maintaining strict ablation purity (no leaked SHAP features in cot/few-shot, and
    no artificial word/sentence length budgets).
    """
    clean_lvl = str(target_level).strip()
    pop_name, lvl_desc = resolve_population_and_description(clean_lvl)
    prompt_json_str = json.dumps(prompt_cfg, ensure_ascii=False, indent=2).replace(
        "{{TARGET_POPULATION}}",
        f"{pop_name} (Specifically Raw Level '{clean_lvl}': {lvl_desc})",
    )

    valid_raw_levels_guide = {
        k: v
        for k, v in LEVEL_DESCRIPTIONS.items()
        if not k.startswith("'") or k in ("'2-4", "'4-9")
    }

    # Only feature_informed receives the quantitative SHAP table for the target level
    feature_supplement = ""
    if prompt_name == "feature_informed":
        lvl_shap = (
            LEVEL_SHAP_WEIGHTS.get(clean_lvl)
            or LEVEL_SHAP_WEIGHTS.get(f"'{clean_lvl}")
            or LEVEL_SHAP_WEIGHTS.get(clean_lvl.lstrip("'"))
            or {}
        )
        lvl_medians = (
            RAW_LEVEL_MEDIANS.get(clean_lvl)
            or RAW_LEVEL_MEDIANS.get(f"'{clean_lvl}")
            or RAW_LEVEL_MEDIANS.get(clean_lvl.lstrip("'"))
            or {}
        )
        nl = chr(10)
        feature_supplement = (
            f"{nl}QUANTITATIVE SHAP IMPORTANCE & FEATURE MEDIANS FOR TARGET LEVEL '{clean_lvl}':{nl}"
            f"{json.dumps({'mean_abs_shap': lvl_shap, 'target_feature_medians': lvl_medians}, ensure_ascii=False, indent=2)}{nl}"
        )

    nl = chr(10)
    system_content = (
        f"You are an expert Hebrew language educator and computational linguist.{nl}"
        f"Follow this prompt specification:{nl}{prompt_json_str}{nl}"
        f"{feature_supplement}{nl}"
        f"VALID RAW LEVEL CODES FOR SOURCE LEVEL ESTIMATION:{nl}"
        f"{json.dumps(valid_raw_levels_guide, ensure_ascii=False, indent=2)}{nl}{nl}"
        f"CRITICAL OUTPUT REQUIREMENTS:{nl}"
        f"1. Respond STRICTLY in valid JSON.{nl}"
        f"2. Include a top-level key 'estimated_original_raw_levels' containing a list of up to 3 raw level codes "
        f"(including 'academic/technical' if the source text is above school-level complexity) from VALID RAW LEVEL CODES.{nl}"
        f"3. Adapt the text specifically for Target Raw Level '{clean_lvl}' ({lvl_desc}, Population: '{pop_name}') "
        f"while preserving 100% of the core narrative and factual meaning.{nl}"
        f"4. Place the simplified Hebrew text in 'simplified_text' (as well as inside the prompt's schema)."
    )

    user_content = json.dumps(
        {
            "requested_target_raw_level": clean_lvl,
            "requested_target_level_description": lvl_desc,
            "requested_target_population": pop_name,
            "input_hebrew_text": source_text,
        },
        ensure_ascii=False,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_llm_static_output(
    llm_json: Dict[str, Any], fallback_text: str
) -> Tuple[str, str]:
    """Extracts (original_level_str, simplified_text) from static prompt JSON outputs."""
    raw_est = llm_json.get("estimated_original_raw_levels")
    levels_list: List[str] = []

    if isinstance(raw_est, list):
        levels_list = [str(x).strip() for x in raw_est if str(x).strip()]
    elif isinstance(raw_est, str) and raw_est.strip():
        levels_list = [x.strip() for x in raw_est.split(",") if x.strip()]

    if not levels_list:
        if isinstance(llm_json.get("classification"), dict):
            prim = llm_json["classification"].get("primary_level")
            if prim:
                levels_list.append(str(prim).strip())
        elif isinstance(llm_json.get("step_1_source_classification"), dict):
            s1 = llm_json["step_1_source_classification"]
            if s1.get("primary_population"):
                levels_list.append(str(s1["primary_population"]).strip())
            for sec in s1.get("secondary_populations") or []:
                if str(sec).strip():
                    levels_list.append(str(sec).strip())

    original_level_str = ", ".join(levels_list[:3]) if levels_list else "academic/technical"

    candidates = [
        llm_json.get("simplified_text"),
        llm_json.get("step_3_simplification", {}).get("simplified_text")
        if isinstance(llm_json.get("step_3_simplification"), dict)
        else None,
        llm_json.get("step_2_adaptation", {}).get("simplified_text")
        if isinstance(llm_json.get("step_2_adaptation"), dict)
        else None,
        llm_json.get("adapted_text"),
    ]
    simplified_text = fallback_text
    for cand in candidates:
        if isinstance(cand, str) and cand.strip():
            simplified_text = cand.strip()
            break

    return original_level_str, simplified_text


def call_openai_with_retry(
    client: Any,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Calls OpenAI Chat Completions with JSON mode and exponential backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(2.0 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# 3. CORE EXPERIMENTATION PIPELINE
# ---------------------------------------------------------------------------
def run_experimentation_pipeline(
    input_csv: Path,
    output_csv: Path,
    engine: Any,
    openai_client: Optional[Any],
    model_name: str = "gpt-4o-mini",
    prompts_to_run: List[str] = DEFAULT_PROMPTS,
    max_iters: int = 3,
    mock_mode: bool = False,
) -> pd.DataFrame:
    """
    Iterates over input_csv [paragraph_id, text, target_level], runs all selected prompts
    and the agentic workflow, evaluates with ReadabilityInferenceEngine, and streams
    results to both output_csv (8 columns) and output_csv_enriched.csv.
    """
    logger = setup_pipeline_logger(
        name="ExperimentationPipeline",
        artifact_dir="artifacts",
        log_filename="experimentation_run.log",
    )

    df_in = pd.read_csv(input_csv, dtype=str)
    required_in_cols = {"paragraph_id", "text", "target_level"}
    missing_cols = required_in_cols - set(df_in.columns)
    if missing_cols:
        raise ValueError(f"Input CSV {input_csv} is missing required columns: {missing_cols}")

    enriched_csv = output_csv.with_name(f"{output_csv.stem}_enriched.csv")

    loaded_prompts: Dict[str, Dict[str, Any]] = {}
    for p_name, rel_path in PROMPT_FILES.items():
        full_p = PROJECT_ROOT / rel_path
        loaded_prompts[p_name] = (
            json.loads(full_p.read_text(encoding="utf-8"))
            if full_p.exists()
            else {"prompt_id": p_name, "role": "Hebrew readability expert"}
        )

    agent = HebrewSimplificationAgent(
        engine=engine,
        feature_prompt_cfg=loaded_prompts["feature_informed"],
        openai_client=openai_client,
        model_name=model_name,
        log_dir="artifacts",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed_pairs: Set[Tuple[str, str]] = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        df_existing = pd.read_csv(output_csv, dtype=str)
        for _, r in df_existing.iterrows():
            completed_pairs.add((str(r["paragraph_id"]), str(r["prompt"])))
        logger.info("Resuming from %d completed rows in %s", len(completed_pairs), output_csv)
    else:
        with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL).writeheader()

    if not enriched_csv.exists() or enriched_csv.stat().st_size == 0:
        with open(enriched_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=ENRICHED_COLUMNS, quoting=csv.QUOTE_ALL).writeheader()

    total_rows = len(df_in)
    for idx, row in df_in.iterrows():
        pid = str(row["paragraph_id"]).strip()
        source_text = str(row["text"]).strip()
        target_level = str(row["target_level"]).strip()
        pop_name, _ = resolve_population_and_description(target_level)
        t_q1, t_q3, t_mu = get_raw_level_iqr(engine, target_level)

        if all((pid, p_name) in completed_pairs for p_name in prompts_to_run):
            continue

        # Evaluate source text ONCE per paragraph and cache across all 4 methods
        source_eval = engine.evaluate_text(source_text, print_report=False)
        orig_score = round(float(source_eval["raw_readability_score"]), 4)
        orig_tool_iqr = format_iqr_levels_plain(source_eval, engine=engine)
        src_word_count = max(len(source_text.split()), 1)

        for prompt_name in prompts_to_run:
            if (pid, prompt_name) in completed_pairs:
                continue

            t0 = time.time()
            if prompt_name == "agentic":
                original_level_val = orig_tool_iqr
                mock_turns = (
                    [
                        {"step_3_simplification": {"simplified_text": "טקסט מופשט ראשוני לבדיקה."}},
                        {
                            "step_3_simplification": {
                                "simplified_text": "טֶקְסְט פָּשׁוּט וּמְנֻקָּד לִבְדִיקָה."
                                if target_level in ("1", "2", "3", "'2-4", "2-4")
                                else "טקסט פשוט, קצר וברור לבדיקה."
                            }
                        },
                    ]
                    if mock_mode
                    else None
                )
                agent_res = agent.run_agentic_simplification(
                    source_text=source_text,
                    target_population=pop_name,
                    paragraph_id=f"{pid}_agentic",
                    max_iters=max_iters,
                    mock_responses=mock_turns,
                    target_raw_level=target_level,
                    cached_source_eval=source_eval,
                )
                simplified_text = agent_res["final_simplified_text"]
                eval_simplified = agent_res.get("final_eval_payload") or engine.evaluate_text(
                    simplified_text, print_report=False
                )
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified, engine=engine)
                iters_used = int(agent_res["total_iterations_run"])
                top_offenders = ", ".join(
                    agent_res["trajectory"][0].get("top_offending_features", [])
                )
            else:
                messages = build_static_prompt_messages(
                    prompt_name=prompt_name,
                    prompt_cfg=loaded_prompts[prompt_name],
                    source_text=source_text,
                    target_level=target_level,
                )
                llm_json = (
                    {
                        "estimated_original_raw_levels": ["11", "12", "academic/technical"],
                        "simplified_text": (
                            "טֶקְסְט פָּשׁוּט עִם נִיקּוּד."
                            if target_level in ("1", "2", "3", "'2-4", "2-4")
                            else "טקסט מופשט וברור המותאם לרמת היעד."
                        ),
                    }
                    if mock_mode
                    else call_openai_with_retry(
                        client=openai_client, model_name=model_name, messages=messages
                    )
                )
                original_level_val, simplified_text = parse_llm_static_output(
                    llm_json=llm_json, fallback_text=source_text
                )
                eval_simplified = engine.evaluate_text(simplified_text, print_report=False)
                simplified_score = round(float(eval_simplified["raw_readability_score"]), 4)
                simplified_level_val = format_iqr_levels_with_probs(eval_simplified, engine=engine)
                iters_used = 1
                diag_static = agent.diagnose_draft_in_natural_language(
                    eval_simplified, pop_name, target_raw_level=target_level
                )
                top_offenders = ", ".join(diag_static.get("top_offending_features", []))

            base_row = {
                "paragraph_id": pid,
                "text": source_text,
                "target_level": target_level,
                "prompt": prompt_name,
                "original_level": original_level_val,
                "simplified_text": simplified_text,
                "simplified_score": simplified_score,
                "simplified_level": simplified_level_val,
            }
            with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL).writerow(base_row)
                f.flush()

            enriched_row = {
                **base_row,
                "original_score": orig_score,
                "original_tool_iqr_levels": orig_tool_iqr,
                "target_q1": t_q1,
                "target_q3": t_q3,
                "target_mu": t_mu,
                "iterations_used": iters_used,
                "top_shap_offenders": top_offenders,
                "word_retention_ratio": round(len(simplified_text.split()) / src_word_count, 3),
            }
            with open(enriched_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=ENRICHED_COLUMNS, quoting=csv.QUOTE_ALL).writerow(
                    enriched_row
                )
                f.flush()

            completed_pairs.add((pid, prompt_name))
            logger.info(
                "[%d/%d] (%s, %s) -> Score=%.4f | Level=%s (%.2fs)",
                idx + 1,
                total_rows,
                pid,
                prompt_name,
                simplified_score,
                simplified_level_val,
                time.time() - t0,
            )

    return pd.read_csv(output_csv)


def analyze_experiment_results(output_csv: Path) -> pd.DataFrame:
    """Computes summary metrics across prompts on output_csv."""
    df = pd.read_csv(output_csv)
    if df.empty:
        return pd.DataFrame()

    def check_exact_raw_hit(row: pd.Series) -> bool:
        target = str(row["target_level"]).strip()
        aliases = set(RAW_LEVEL_ALIASES.get(target, [target, f"'{target}", target.lstrip("'")]))
        simp_str = str(row["simplified_level"])
        if simp_str.startswith("NONE_IN_IQR"):
            return False
        matched_raw = {
            part.split("(")[0].strip() for part in simp_str.split(",") if part.strip()
        }
        return len(aliases & matched_raw) > 0

    df["exact_raw_iqr_hit"] = df.apply(check_exact_raw_hit, axis=1)
    df["simplified_score"] = pd.to_numeric(df["simplified_score"], errors="coerce")

    summary = (
        df.groupby("prompt", as_index=False)
        .agg(
            n_samples=("paragraph_id", "count"),
            exact_raw_iqr_hit_pct=("exact_raw_iqr_hit", lambda s: round(s.mean() * 100.0, 2)),
            mean_simplified_score=("simplified_score", lambda s: round(s.mean(), 3)),
            std_simplified_score=("simplified_score", lambda s: round(s.std(), 3)),
        )
        .sort_values("exact_raw_iqr_hit_pct", ascending=False)
    )
    print("\n" + "=" * 75)
    print("DELIVERABLE 6: EXPERIMENTAL COMPARISON SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("=" * 75 + "\n")
    return summary


def _run_smoke_test() -> None:
    """Verifies 8-column CSV, 'academic/technical' above-range tagging, and Hybrid TreeSHAP."""
    import tempfile

    class DummyEngine:
        max_q3 = 6.00
        max_p99 = 6.75
        calibration_stats = {
            "2": {"mu": 3.10, "sigma": 0.45, "Q1": 2.75, "Q3": 3.45},
            "4": {"mu": 5.10, "sigma": 0.40, "Q1": 4.80, "Q3": 5.40},
        }

        def get_raw_level_iqr_bounds(self, target_raw_level: str) -> Tuple[float, float, float]:
            s = self.calibration_stats.get(str(target_raw_level), {"Q1": 4.8, "Q3": 5.4, "mu": 5.1})
            return s["Q1"], s["Q3"], s["mu"]

        def evaluate_text(self, text: str, print_report: bool = False) -> Dict[str, Any]:
            if "מקור מורכב" in text:
                return {
                    "raw_readability_score": 6.42,
                    "core_iqr_matches": ["academic/technical"],
                    "core_iqr_categories": ["Technical/Academic"],
                    "core_iqr_probabilities": {"academic/technical": 100.0},
                    "extended_p1_p99_matches": ["academic/technical"],
                    "is_above_calibrated_range": True,
                    "transformed_features": {"1_nikkud_ratio_log_inv": 1.0},
                    "local_shap_values": {"1_nikkud_ratio_log_inv": 0.35},
                    "shap_base_value": 5.20,
                }
            has_nikkud = any("\u0591" <= c <= "\u05c7" for c in text)
            return {
                "raw_readability_score": 3.20 if has_nikkud else 5.05,
                "core_iqr_matches": ["2"] if has_nikkud else ["4"],
                "core_iqr_categories": ["Elementary - Beginner"] if has_nikkud else ["Elementary - Intermediate"],
                "core_iqr_probabilities": {"2": 100.0} if has_nikkud else {"4": 100.0},
                "extended_p1_p99_matches": ["2", "4"],
                "is_above_calibrated_range": False,
                "transformed_features": {"1_nikkud_ratio_log_inv": 0.05 if has_nikkud else 0.92},
                "local_shap_values": {"1_nikkud_ratio_log_inv": -1.05 if has_nikkud else 0.12},
                "shap_base_value": 5.20,
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_csv = tmp_dir / "test_in.csv"
        out_csv = tmp_dir / "test_out.csv"

        pd.DataFrame(
            [
                {"paragraph_id": "p_01", "text": "טקסט מקור מורכב מאוד.", "target_level": "2"},
                {"paragraph_id": "p_02", "text": "טקסט מקור מורכב נוסף.", "target_level": "4"},
            ]
        ).to_csv(in_csv, index=False)

        df_res = run_experimentation_pipeline(
            input_csv=in_csv,
            output_csv=out_csv,
            engine=DummyEngine(),
            openai_client=None,
            mock_mode=True,
        )
        assert list(df_res.columns) == OUTPUT_COLUMNS
        assert len(df_res) == 8
        agent_rows = df_res[df_res["prompt"] == "agentic"]
        assert (agent_rows["original_level"] == "academic/technical").all()
        print("[✓] Phase 4 Experimentation smoke test passed ('academic/technical' + 8 columns verified)!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 OpenAI Experimentation Pipeline")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/wikipedia_simplification_test_100.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/experimentation_results.csv"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, choices=DEFAULT_PROMPTS)
    parser.add_argument("--test", action="store_true")
    args, _ = parser.parse_known_args()

    if args.test:
        _run_smoke_test()
        return

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    openai_client = OpenAI(api_key=api_key)
    engine = ReadabilityInferenceEngine()

    run_experimentation_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        engine=engine,
        openai_client=openai_client,
        model_name=args.model,
        prompts_to_run=args.prompts,
        max_iters=args.max_iters,
        mock_mode=False,
    )
    analyze_experiment_results(args.output_csv)


if __name__ == "__main__":
    main()
