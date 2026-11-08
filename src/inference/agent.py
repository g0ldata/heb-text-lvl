"""
Autonomous agentic simplification workflow with Hybrid Differential SHAP Attribution,
above-range 'academic/technical' tagging, and content-preserving linguistic feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from hebrew_readability.config import (
    FEATURE_DIAGNOSTIC_RULES,
    LEVEL_DESCRIPTIONS,
    LEVEL_SHAP_WEIGHTS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    RAW_LEVEL_MEDIANS,
    RAW_LEVEL_TO_POPULATION,
    setup_pipeline_logger,
)


class HebrewSimplificationAgent:
    """
    Autonomous multi-turn simplification agent that uses ReadabilityInferenceEngine
    and Hybrid Differential SHAP Attribution to guide the LLM into the target level's
    Core IQR (Q1-Q3) while preserving 100% of factual content.
    """

    def __init__(
        self,
        engine: Any,
        feature_prompt_cfg: Dict[str, Any],
        openai_client: Optional[Any] = None,
        model_name: str = "gpt-4o-mini",
        log_dir: str = "artifacts",
    ) -> None:
        self.engine = engine
        self.prompt_cfg = feature_prompt_cfg
        self.client = openai_client
        self.model_name = model_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_log_path = os.path.join(self.log_dir, "agent_trajectories.jsonl")
        self.logger = setup_pipeline_logger(
            name="HebrewSimplificationAgent",
            artifact_dir=self.log_dir,
            log_filename="agent_execution.log",
        )

    @staticmethod
    def _extract_categories_and_levels(
        eval_result: Dict[str, Any], cat_key: str, match_key: str
    ) -> Tuple[Set[str], Set[str]]:
        raw_matches = {str(m).strip() for m in (eval_result.get(match_key) or [])}
        raw_matches |= {m.lstrip("'") for m in raw_matches}

        explicit_cats = {str(c).strip() for c in (eval_result.get(cat_key) or [])}
        derived_cats: Set[str] = set()
        for m in raw_matches:
            for c in LEVEL_TO_CATEGORIES.get(m, []):
                derived_cats.add(c)

        return explicit_cats | derived_cats, raw_matches

    def _get_calibration_targets(
        self, target_population: str, target_raw_level: Optional[str] = None
    ) -> Tuple[float, float, float, Set[str], Dict[str, float], Dict[str, float]]:
        """
        Resolves (Q1, Q3, mu_anchor, raw_aliases, target_medians, level_shap_weights)
        for either an exact raw level or a broad target population.
        """
        spec = POPULATION_SPECS.get(
            target_population, POPULATION_SPECS["Grade 4-6 - Elementary Intermediate"]
        )
        cal_stats = getattr(self.engine, "calibration_stats", {})
        max_q3 = getattr(self.engine, "max_q3", 6.05)

        if target_raw_level:
            clean = str(target_raw_level).strip()
            if clean.lower() in ("academic/technical", "technical/academic"):
                return (
                    round(max_q3, 4),
                    round(max_q3 + 1.5, 4),
                    round(max_q3 + 0.45, 4),
                    {"academic/technical", "Technical/Academic"},
                    RAW_LEVEL_MEDIANS["academic/technical"],
                    LEVEL_SHAP_WEIGHTS["12"],
                )

            aliases = RAW_LEVEL_ALIASES.get(clean, [clean, f"'{clean}", clean.lstrip("'")])
            medians = (
                RAW_LEVEL_MEDIANS.get(clean)
                or RAW_LEVEL_MEDIANS.get(f"'{clean}")
                or RAW_LEVEL_MEDIANS.get(clean.lstrip("'"))
                or spec["medians"]
            )
            shap_weights = (
                LEVEL_SHAP_WEIGHTS.get(clean)
                or LEVEL_SHAP_WEIGHTS.get(f"'{clean}")
                or LEVEL_SHAP_WEIGHTS.get(clean.lstrip("'"))
                or {k: v["weight"] for k, v in FEATURE_DIAGNOSTIC_RULES.items()}
            )

            for cand in aliases:
                if cand in cal_stats:
                    s = cal_stats[cand]
                    return (
                        round(float(s["Q1"]), 4),
                        round(float(s["Q3"]), 4),
                        round(float(s["mu"]), 4),
                        set(aliases),
                        medians,
                        shap_weights,
                    )

        if hasattr(self.engine, "get_population_iqr_bounds"):
            q1, q3 = self.engine.get_population_iqr_bounds(target_population)
        else:
            anchor = float(spec["anchor_score"])
            q1, q3 = round(anchor - 0.45, 4), round(anchor + 0.45, 4)

        return (
            q1,
            q3,
            float(spec["anchor_score"]),
            set(),
            spec["medians"],
            {k: v["weight"] for k, v in FEATURE_DIAGNOSTIC_RULES.items()},
        )

    def diagnose_draft_in_natural_language(
        self,
        eval_result: Dict[str, Any],
        target_population: str,
        target_raw_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verifies whether Y_pred falls within the target's Core IQR (Q1-Q3) and performs
        Hybrid Differential SHAP Attribution combining per-prediction local TreeSHAP
        values phi_i(x) with per-level Mean_Abs_SHAP weights and median deviations.
        """
        if target_population not in POPULATION_SPECS:
            target_population = RAW_LEVEL_TO_POPULATION.get(
                str(target_population).strip(), "Grade 4-6 - Elementary Intermediate"
            )

        spec = POPULATION_SPECS[target_population]
        matched_iqr_cats, matched_iqr_levels = self._extract_categories_and_levels(
            eval_result, "core_iqr_categories", "core_iqr_matches"
        )
        matched_ext_cats, matched_ext_levels = self._extract_categories_and_levels(
            eval_result, "extended_p1_p99_categories", "extended_p1_p99_matches"
        )

        y_pred = float(eval_result["raw_readability_score"])
        max_q3 = float(getattr(self.engine, "max_q3", 6.05))

        # Requirement 3: Tag any text with readability score above the calibrated ranges as 'academic/technical'
        is_above_range = bool(
            eval_result.get("is_above_calibrated_range")
            or (y_pred > max_q3 and len(matched_iqr_levels) == 0)
        )
        if is_above_range:
            matched_iqr_levels.add("academic/technical")
            matched_iqr_cats.add("Technical/Academic")

        (
            q1_bound,
            q3_bound,
            anchor_score,
            raw_aliases,
            target_medians,
            level_shap_weights,
        ) = self._get_calibration_targets(target_population, target_raw_level)

        score_diff = y_pred - anchor_score
        direction_sign = 1.0 if score_diff >= 0 else -1.0

        if raw_aliases:
            in_core_iqr = bool(
                (len(raw_aliases & matched_iqr_levels) > 0)
                or (q1_bound <= y_pred <= q3_bound)
            )
            in_extended = bool(
                in_core_iqr or (len(raw_aliases & matched_ext_levels) > 0)
            )
            lvl_desc = LEVEL_DESCRIPTIONS.get(str(target_raw_level), str(target_raw_level))
            target_label = f"Raw Level '{target_raw_level}' ({lvl_desc})"
        else:
            target_cats = set(spec["engine_categories"])
            target_raw_set = set(spec.get("raw_level_keys", []))
            in_core_iqr = bool(
                (len(target_cats & matched_iqr_cats) > 0)
                or (len(target_raw_set & matched_iqr_levels) > 0)
                or (q1_bound <= y_pred <= q3_bound)
            )
            in_extended = bool(
                in_core_iqr
                or (len(target_cats & matched_ext_cats) > 0)
                or (len(target_raw_set & matched_ext_levels) > 0)
            )
            target_label = target_population

        draft_feats = dict(eval_result.get("transformed_features") or {})
        local_shap = dict(eval_result.get("local_shap_values") or {})
        shap_base = float(eval_result.get("shap_base_value", 5.20))

        # Hybrid Differential SHAP Attribution:
        # Priority_i = |delta_i| * Level_SHAP_i + 0.5 * max(0, sign(y_pred - mu_target) * local_shap_i)
        offenders: List[Tuple[float, str, str, float, float, float]] = []
        shap_attribution_breakdown: Dict[str, Dict[str, float]] = {}

        for feat_key, rule in FEATURE_DIAGNOSTIC_RULES.items():
            if feat_key in draft_feats and feat_key in target_medians:
                val = float(draft_feats[feat_key])
                med = float(target_medians[feat_key])
                delta = val - med
                w_lvl = float(level_shap_weights.get(feat_key, rule["weight"]))
                phi_i = float(local_shap.get(feat_key, 0.0))

                directional_local_push = max(0.0, direction_sign * phi_i)
                hybrid_priority = (abs(delta) * w_lvl) + (0.5 * directional_local_push)

                shap_attribution_breakdown[feat_key] = {
                    "draft_scaled": round(val, 4),
                    "target_median": round(med, 4),
                    "delta_from_median": round(delta, 4),
                    "local_shap_phi": round(phi_i, 4),
                    "level_mean_abs_shap": round(w_lvl, 4),
                    "hybrid_priority": round(hybrid_priority, 5),
                }

                # Skip Nikkud directive when targeting unvocalized levels (4+) and text is already unvocalized
                if (
                    feat_key == "1_nikkud_ratio_log_inv"
                    and med >= 0.85
                    and val >= 0.85
                ):
                    continue

                if score_diff >= 0 and delta > 0.02:
                    offenders.append(
                        (hybrid_priority, rule["name"], rule["too_high"], val, med, phi_i)
                    )
                elif score_diff < 0 and delta < -0.03:
                    offenders.append(
                        (hybrid_priority, rule["name"], rule["too_low"], val, med, phi_i)
                    )

        offenders.sort(key=lambda x: x[0], reverse=True)
        top_offending_features = [name for _, name, _, _, _, _ in offenders[:3]] or [
            "Sentence Breath Load (Syllables per Sentence)",
            "Syntactic Dependency Tree Depth",
        ]

        top_directives: List[str] = []
        for priority, name, instr, val, med, phi_i in offenders[:3]:
            top_directives.append(
                f"- {name} [Draft Scaled: {val:.2f} vs Target Median: {med:.2f} | "
                f"Local TreeSHAP Attribution: {phi_i:+.3f} pts]: {instr}"
            )

        if not top_directives:
            top_directives.append(
                "- Syntactic & Lexical Calibration: Adjust sentence clause structure and lexical familiarity toward the target level while retaining 100% of the factual content."
            )

        # Attach draft-specific linguistic evidence (without artificial word or sentence count caps)
        evidence = eval_result.get("draft_evidence") or {}
        evidence_lines: List[str] = []
        if score_diff > 0:
            if evidence.get("longest_sentence") and evidence.get("longest_sentence_words", 0) >= 14:
                evidence_lines.append(
                    f"- Most Syntactically Dense Sentence in Draft: «{evidence['longest_sentence']}» "
                    f"-> Consider unpacking the clauses in this sentence into clearer sequential statements without dropping any facts."
                )
            if evidence.get("rare_lexical_items"):
                rare_str = ", ".join(evidence["rare_lexical_items"])
                evidence_lines.append(
                    f"- Low-Frequency Lexical Items Detected: [{rare_str}] "
                    f"-> Replace or briefly contextualize these terms using familiar Hebrew equivalents."
                )

        dist_to_iqr = round(max(0.0, q1_bound - y_pred, y_pred - q3_bound), 4)
        nl = chr(10)

        if in_core_iqr:
            feedback = (
                f"TOOL DIAGNOSTIC REPORT: Draft readability (Y_pred={y_pred:.2f}) CONVERGED "
                f"inside Target Core IQR [{q1_bound:.2f}, {q3_bound:.2f}] for {target_label}."
            )
        else:
            overall_direction = (
                "TOO COMPLEX (above Q3; currently in 'academic/technical' range)"
                if is_above_range and score_diff > 0
                else ("TOO COMPLEX (above Q3)" if score_diff > 0 else "TOO SIMPLE (below Q1)")
            )
            step_guidance = (
                "NOTE: Your draft is very close to the target Core IQR boundary "
                f"(distance = {dist_to_iqr:.2f} pts). Apply subtle, targeted edits to the features below "
                "rather than rewriting the passage from scratch."
                if 0 < dist_to_iqr <= 0.18
                else "Apply the feature adjustments below while preserving 100% of the core narrative and factual meaning."
            )
            ev_block = (nl + nl.join(evidence_lines)) if evidence_lines else ""
            feedback = (
                f"TOOL DIAGNOSTIC REPORT (Hybrid Differential TreeSHAP):{nl}"
                f"- Draft Predicted Score: Y_pred={y_pred:.2f} (Model Base phi_0={shap_base:.2f}){nl}"
                f"- Target Core IQR (Q1-Q3): [{q1_bound:.2f}, {q3_bound:.2f}] (Target Anchor={anchor_score:.2f}){nl}"
                f"- Status: {overall_direction} for {target_label} (Distance to IQR={dist_to_iqr:.2f} pts){nl}"
                f"- Guidance: {step_guidance}{nl}{nl}"
                f"Top Offending Linguistic Features:{nl}"
                + nl.join(top_directives)
                + ev_block
            )

        return {
            "in_core_iqr": in_core_iqr,
            "in_extended": in_extended,
            "is_above_calibrated_range": is_above_range,
            "target_q1": q1_bound,
            "target_q3": q3_bound,
            "target_mu": anchor_score,
            "distance_to_iqr": dist_to_iqr,
            "abs_distance_to_anchor": round(abs(score_diff), 4),
            "top_offending_features": top_offending_features,
            "shap_attribution_breakdown": shap_attribution_breakdown,
            "natural_language_feedback": feedback,
        }

    def _build_initial_messages(
        self,
        source_text: str,
        target_population: str,
        target_raw_level: Optional[str] = None,
        q1_bound: float = 4.5,
        q3_bound: float = 5.5,
        anchor_score: float = 5.0,
        target_medians: Optional[Dict[str, float]] = None,
        level_shap_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, str]]:
        """
        Injects the feature-informed prompt profile, per-level SHAP weights, and
        calibrated target medians without imposing artificial word/sentence count caps.
        """
        role = self.prompt_cfg.get(
            "role",
            "You are an expert Hebrew language educator and computational linguist.",
        )
        pop_profile = self.prompt_cfg.get("population_feature_profiles", {}).get(
            target_population, {}
        )
        lvl_desc = (
            LEVEL_DESCRIPTIONS.get(str(target_raw_level), target_population)
            if target_raw_level
            else target_population
        )
        schema = self.prompt_cfg.get(
            "required_output_schema",
            {
                "step_1_source_classification": {
                    "primary_population": "Grade 7-9 - Middle School",
                    "secondary_populations": [],
                },
                "step_2_linguistic_diagnosis": {
                    "target_population": target_population,
                    "key_features_to_modify": ["Sentence Length", "Vocabulary Rarity"],
                    "adaptation_explanation": "Brief explanation.",
                },
                "step_3_simplification": {
                    "simplified_text": "הטקסט המופשט בעברית כאן."
                },
            },
        )

        nl = chr(10)
        system_prompt = (
            f"{role}{nl}{nl}"
            f"Requested Target Level: '{target_raw_level or target_population}' ({lvl_desc}){nl}"
            f"Target Pedagogical Population: '{target_population}'{nl}"
            f"Calibrated Target Readability Core IQR (Q1-Q3): [{q1_bound:.2f}, {q3_bound:.2f}] (Anchor: {anchor_score:.2f}){nl}{nl}"
            f"Target Population Linguistic Profile:{nl}"
            f"{json.dumps(pop_profile, ensure_ascii=False, indent=2)}{nl}{nl}"
            f"Target Level Feature Importance (Mean_Abs_SHAP) & Calibrated Feature Medians:{nl}"
            f"{json.dumps({'level_mean_abs_shap': level_shap_weights or {}, 'target_scaled_feature_medians': target_medians or {}}, ensure_ascii=False, indent=2)}{nl}{nl}"
            f"CONTENT RETENTION PRINCIPLE:{nl}"
            f"Preserve 100% of the core narrative, facts, and conceptual meaning of the source passage. "
            f"Achieve the target readability level by adapting syntactic structure, clause nesting, voice, morphology, "
            f"and lexical familiarity—NOT by deleting factual content.{nl}{nl}"
            f"Respond STRICTLY in valid JSON following this exact schema "
            f"(place your adapted Hebrew text inside 'step_3_simplification.simplified_text'):{nl}"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

        user_prompt = json.dumps(
            {
                "requested_target_raw_level": target_raw_level or target_population,
                "requested_target_population": target_population,
                "input_hebrew_text": source_text,
            },
            ensure_ascii=False,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_draft_text(llm_json: Dict[str, Any], fallback_text: str) -> str:
        candidates = [
            llm_json.get("step_3_simplification", {}).get("simplified_text")
            if isinstance(llm_json.get("step_3_simplification"), dict)
            else None,
            llm_json.get("step_2_adaptation", {}).get("simplified_text")
            if isinstance(llm_json.get("step_2_adaptation"), dict)
            else None,
            llm_json.get("simplified_text"),
            llm_json.get("adapted_text"),
            llm_json.get("text"),
        ]
        for cand in candidates:
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
        return fallback_text

    def _call_llm_step(
        self,
        messages: List[Dict[str, str]],
        mock_responses: Optional[List[Dict[str, Any]]] = None,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        if mock_responses is not None:
            return mock_responses[min(iteration - 1, len(mock_responses) - 1)]
        if self.client is None:
            raise RuntimeError("openai_client is None and no mock_responses were provided.")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)

    def _append_jsonl_log(self, record: Dict[str, Any]) -> None:
        with open(self.jsonl_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def format_evaluated_iqr_levels(self, eval_result: Dict[str, Any]) -> List[str]:
        """
        Returns the matched Core IQR raw levels for an evaluated text.
        Tags any text whose readability score is above the calibrated ranges as 'academic/technical'.
        """
        y_pred = float(eval_result["raw_readability_score"])
        max_q3 = float(getattr(self.engine, "max_q3", 6.05))
        core_matches = [str(m).strip() for m in (eval_result.get("core_iqr_matches") or [])]

        if eval_result.get("is_above_calibrated_range") or (y_pred > max_q3 and not core_matches):
            return ["academic/technical"]
        return core_matches

    def run_agentic_simplification(
        self,
        source_text: str,
        target_population: str,
        paragraph_id: str = "sample_01",
        max_iters: int = 3,
        mock_responses: Optional[List[Dict[str, Any]]] = None,
        target_raw_level: Optional[str] = None,
        cached_source_eval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        if target_population not in POPULATION_SPECS:
            if target_raw_level is None:
                target_raw_level = str(target_population).strip()
            target_population = RAW_LEVEL_TO_POPULATION.get(
                str(target_population).strip(), "Grade 4-6 - Elementary Intermediate"
            )

        source_eval = cached_source_eval or self.engine.evaluate_text(
            source_text, print_report=False
        )
        source_score = round(float(source_eval["raw_readability_score"]), 4)
        source_iqr_levels = self.format_evaluated_iqr_levels(source_eval)
        source_cats, _ = self._extract_categories_and_levels(
            source_eval, "core_iqr_categories", "core_iqr_matches"
        )
        if "academic/technical" in source_iqr_levels:
            source_cats.add("Technical/Academic")

        (
            q1_bound,
            q3_bound,
            anchor_score,
            _,
            target_medians,
            level_shap_weights,
        ) = self._get_calibration_targets(target_population, target_raw_level)

        self.logger.info(
            f"[{paragraph_id}] Source Score: {source_score:.4f} | "
            f"Source Levels: {source_iqr_levels} -> Target: '{target_raw_level or target_population}' "
            f"(IQR=[{q1_bound:.2f}, {q3_bound:.2f}])"
        )

        messages = self._build_initial_messages(
            source_text=source_text,
            target_population=target_population,
            target_raw_level=target_raw_level,
            q1_bound=q1_bound,
            q3_bound=q3_bound,
            anchor_score=anchor_score,
            target_medians=target_medians,
            level_shap_weights=level_shap_weights,
        )

        trajectory: List[Dict[str, Any]] = []
        best_turn: Optional[Dict[str, Any]] = None
        nl = chr(10)

        for iteration in range(1, max_iters + 1):
            iter_start = time.time()
            llm_json = self._call_llm_step(messages, mock_responses, iteration)
            draft_text = self._extract_draft_text(llm_json, fallback_text=source_text)

            eval_res = self.engine.evaluate_text(draft_text, print_report=False)
            diag = self.diagnose_draft_in_natural_language(
                eval_res, target_population, target_raw_level=target_raw_level
            )
            predicted_score = round(float(eval_res["raw_readability_score"]), 4)
            draft_iqr_levels = self.format_evaluated_iqr_levels(eval_res)

            turn: Dict[str, Any] = {
                "iteration": iteration,
                "draft_text": draft_text,
                "predicted_score": predicted_score,
                "draft_iqr_levels": draft_iqr_levels,
                "is_above_calibrated_range": diag["is_above_calibrated_range"],
                "target_q1": diag["target_q1"],
                "target_q3": diag["target_q3"],
                "target_mu": diag["target_mu"],
                "in_core_iqr": diag["in_core_iqr"],
                "in_extended": diag["in_extended"],
                "distance_to_iqr": diag["distance_to_iqr"],
                "abs_distance_to_anchor": diag["abs_distance_to_anchor"],
                "top_offending_features": diag["top_offending_features"],
                "local_shap_values": eval_res.get("local_shap_values", {}),
                "shap_base_value": eval_res.get("shap_base_value", 5.20),
                "llm_self_diagnosis": llm_json.get("step_2_linguistic_diagnosis", {}),
                "tool_natural_language_feedback": diag["natural_language_feedback"],
                "transformed_features": eval_res.get("transformed_features", {}),
                "eval_payload": eval_res,
                "latency_sec": round(time.time() - iter_start, 2),
            }
            trajectory.append(turn)

            self.logger.info(
                f"[{paragraph_id}] Iter {iteration}/{max_iters} | "
                f"Y_pred={predicted_score:.4f} | Levels={draft_iqr_levels} | "
                f"Target_IQR=[{diag['target_q1']:.2f}, {diag['target_q3']:.2f}] | "
                f"Converged={diag['in_core_iqr']} | Offenders={diag['top_offending_features']}"
            )

            if (
                best_turn is None
                or (turn["in_core_iqr"] and not best_turn["in_core_iqr"])
                or (
                    turn["in_core_iqr"] == best_turn["in_core_iqr"]
                    and (
                        turn["distance_to_iqr"],
                        turn["abs_distance_to_anchor"],
                    )
                    < (
                        best_turn["distance_to_iqr"],
                        best_turn["abs_distance_to_anchor"],
                    )
                )
            ):
                best_turn = turn

            if diag["in_core_iqr"]:
                break

            if iteration < max_iters:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(llm_json, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{diag['natural_language_feedback']}{nl}{nl}"
                            f"Revise your Hebrew adaptation in 'step_3_simplification.simplified_text' "
                            f"so that its readability score falls inside [{diag['target_q1']:.2f}, {diag['target_q3']:.2f}] "
                            f"while preserving 100% of the factual content from the original passage. "
                            f"Respond strictly in valid JSON."
                        ),
                    }
                )

        assert best_turn is not None
        final_result: Dict[str, Any] = {
            "paragraph_id": paragraph_id,
            "target_population": target_population,
            "target_raw_level": target_raw_level,
            "source_text": source_text,
            "source_score": source_score,
            "source_iqr_levels": source_iqr_levels,
            "final_simplified_text": best_turn["draft_text"],
            "final_score": best_turn["predicted_score"],
            "final_iqr_levels": best_turn["draft_iqr_levels"],
            "final_eval_payload": best_turn["eval_payload"],
            "target_q1": best_turn["target_q1"],
            "target_q3": best_turn["target_q3"],
            "target_mu": best_turn["target_mu"],
            "converged_in_core_iqr": best_turn["in_core_iqr"],
            "best_iteration_selected": best_turn["iteration"],
            "total_iterations_run": len(trajectory),
            "total_latency_sec": round(time.time() - start_time, 2),
            "trajectory": [
                {k: v for k, v in t.items() if k != "eval_payload"} for t in trajectory
            ],
        }

        self._append_jsonl_log(
            {k: v for k, v in final_result.items() if k != "final_eval_payload"}
        )
        return final_result


def _run_smoke_test() -> None:
    """Verifies Hybrid Differential SHAP, 'academic/technical' tagging, and Core IQR loop."""
    import tempfile

    class MockSHAPEngine:
        max_q3 = 6.00
        max_p99 = 6.75
        calibration_stats = {
            "2": {"mu": 3.15, "sigma": 0.45, "Q1": 2.75, "Q3": 3.55, "P1": 2.1, "P99": 4.2},
            "4": {"mu": 5.10, "sigma": 0.35, "Q1": 4.80, "Q3": 5.40, "P1": 4.2, "P99": 5.9},
        }

        def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
            return (2.75, 3.55)

        def evaluate_text(self, text: str, print_report: bool = False) -> Dict[str, Any]:
            if "מקור מורכב" in text:
                return {
                    "raw_readability_score": 6.35,
                    "core_iqr_matches": ["academic/technical"],
                    "core_iqr_categories": ["Technical/Academic"],
                    "core_iqr_probabilities": {"academic/technical": 100.0},
                    "extended_p1_p99_matches": ["academic/technical"],
                    "is_above_calibrated_range": True,
                    "transformed_features": {"1_nikkud_ratio_log_inv": 1.0, "16_total_syllables_log": 0.72},
                    "local_shap_values": {"1_nikkud_ratio_log_inv": 0.42, "16_total_syllables_log": 0.31},
                    "shap_base_value": 5.20,
                    "draft_evidence": {"longest_sentence": text, "longest_sentence_words": 18},
                }
            has_nikkud = any("\u0591" <= c <= "\u05c7" for c in text)
            score = 3.18 if has_nikkud else 5.55
            return {
                "raw_readability_score": score,
                "core_iqr_matches": ["2"] if has_nikkud else [],
                "core_iqr_categories": ["Elementary - Beginner"] if has_nikkud else [],
                "core_iqr_probabilities": {"2": 100.0} if has_nikkud else {},
                "extended_p1_p99_matches": ["2", "4"],
                "is_above_calibrated_range": False,
                "transformed_features": {
                    "1_nikkud_ratio_log_inv": 0.02 if has_nikkud else 0.98,
                    "16_total_syllables_log": 0.47 if has_nikkud else 0.63,
                    "18_avg_syllables_per_sentence_log": 0.45 if has_nikkud else 0.59,
                },
                "local_shap_values": {
                    "1_nikkud_ratio_log_inv": -1.10 if has_nikkud else 0.28,
                    "16_total_syllables_log": -0.15 if has_nikkud else 0.14,
                    "18_avg_syllables_per_sentence_log": -0.08 if has_nikkud else 0.11,
                },
                "shap_base_value": 5.20,
                "draft_evidence": {"longest_sentence": text, "longest_sentence_words": 15},
            }

    with tempfile.TemporaryDirectory() as tmp:
        agent = HebrewSimplificationAgent(
            engine=MockSHAPEngine(),
            feature_prompt_cfg={"role": "Smoke test"},
            log_dir=tmp,
        )
        mocks = [
            {"step_3_simplification": {"simplified_text": "טקסט ללא ניקוד ארוך מדי."}},
            {"step_3_simplification": {"simplified_text": "טֶקְסְט קָצָר עִם נִיקּוּד."}},
        ]
        res = agent.run_agentic_simplification(
            source_text="טקסט מקור מורכב מאוד מעל הטווח.",
            target_population="Grade 1-3 - Elementary Beginner",
            target_raw_level="2",
            paragraph_id="smoke_shap_01",
            max_iters=3,
            mock_responses=mocks,
        )
        assert res["source_iqr_levels"] == ["academic/technical"]
        assert res["converged_in_core_iqr"] is True
        assert res["best_iteration_selected"] == 2
        assert "Local TreeSHAP Attribution" in res["trajectory"][0]["tool_natural_language_feedback"]
        print("[✓] Agent smoke test passed: Hybrid TreeSHAP + 'academic/technical' verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run standalone smoke test")
    args, _ = parser.parse_known_args()
    if args.test:
        _run_smoke_test()
