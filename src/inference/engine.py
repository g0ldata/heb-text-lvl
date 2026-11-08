"""ReadabilityInferenceEngine with Local TreeSHAP attribution and 'academic/technical' above-range tagging."""

import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
import xgboost as xgb

from hebrew_readability.config import (
    FINAL_FEATURE_ORDER,
    HF_DICT_REPO_ID,
    HF_MODEL_REPO_ID,
    LEVEL_DESCRIPTIONS,
    LEVEL_TO_CATEGORIES,
    POPULATION_SPECS,
    RAW_LEVEL_ALIASES,
    read_table,
)
from hebrew_readability.hf_hub import download_model_artifacts

try:
    from features.extractor import FeatureExtractor
    from features.preprocessor import HebrewReadabilityPreprocessor
except ImportError:
    from hebrew_readability.features.extractor import FeatureExtractor
    from hebrew_readability.features.preprocessor import HebrewReadabilityPreprocessor

logger = logging.getLogger(__name__)


class ReadabilityInferenceEngine:
    """Evaluates raw Hebrew strings or pre-extracted DataFrames against calibrated levels."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        model_repo_id: str = HF_MODEL_REPO_ID,
        dict_repo_id: str = HF_DICT_REPO_ID,
        feature_extractor_fn: Optional[Callable[[str], pd.DataFrame]] = None,
        lazy_nlp: bool = True,
    ):
        paths = download_model_artifacts(repo_id=model_repo_id, local_dir=artifact_dir)

        self.preprocessor = joblib.load(paths["preprocessor"])
        self.model = joblib.load(paths["model"])

        calib_path = paths.get("calibration_table") or paths.get("calibration_csv")
        if not calib_path:
            raise KeyError(f"Calibration path missing from downloaded artifacts: {paths}")

        calib_df = read_table(
            calib_path,
            dtype={"pedagogical_level": str},
        )
        self.calibration_stats: Dict[str, Dict[str, float]] = calib_df.set_index(
            "pedagogical_level"
        ).to_dict(orient="index")

        # Upper bounds of calibrated readability ranges for 'academic/technical' tagging
        self.max_q3: float = max(
            (float(s["Q3"]) for s in self.calibration_stats.values() if "Q3" in s),
            default=6.05,
        )
        self.max_p99: float = max(
            (float(s["P99"]) for s in self.calibration_stats.values() if "P99" in s),
            default=6.80,
        )
        self.min_q1: float = min(
            (float(s["Q1"]) for s in self.calibration_stats.values() if "Q1" in s),
            default=2.00,
        )

        self.dict_repo_id = dict_repo_id
        self._extractor: Optional[FeatureExtractor] = None
        self.feature_extractor_fn = feature_extractor_fn

        if not lazy_nlp and self.feature_extractor_fn is None:
            self._extractor = FeatureExtractor()
            self.feature_extractor_fn = self._extractor.extract_single_text

    def _ensure_extractor(self) -> Callable[[str], pd.DataFrame]:
        if self.feature_extractor_fn is None:
            logger.info("Initializing NLP FeatureExtractor on first text evaluation...")
            self._extractor = FeatureExtractor()
            self.feature_extractor_fn = self._extractor.extract_single_text
        return self.feature_extractor_fn

    @staticmethod
    def _levels_to_categories(levels: List[str]) -> List[str]:
        """Maps raw calibration level keys to unique pedagogical categories."""
        cats: List[str] = []
        for lvl in levels:
            norm_lvl = str(lvl).strip()
            mapped = (
                LEVEL_TO_CATEGORIES.get(norm_lvl)
                or LEVEL_TO_CATEGORIES.get(norm_lvl.lstrip("'"))
                or []
            )
            for c in mapped:
                if c not in cats:
                    cats.append(c)
        return cats

    def _compute_local_shap(
        self, X_infer: pd.DataFrame
    ) -> Tuple[Dict[str, float], float]:
        """
        Extracts exact per-prediction TreeSHAP values phi_i(x) and base value phi_0
        directly from the XGBoost C++ booster via pred_contribs=True (<0.5ms overhead).
        """
        try:
            booster = (
                self.model.get_booster()
                if hasattr(self.model, "get_booster")
                else self.model
            )
            dmat = xgb.DMatrix(X_infer, feature_names=list(X_infer.columns))
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            feat_cols = list(X_infer.columns)
            local_shap = {
                col: round(float(contribs[idx]), 6)
                for idx, col in enumerate(feat_cols)
            }
            base_val = round(float(contribs[-1]), 6)
            return local_shap, base_val
        except Exception as exc:
            logger.debug("Fallback from TreeSHAP pred_contribs (%s)", exc)
            return {col: 0.0 for col in X_infer.columns}, 5.20

    @staticmethod
    def _extract_draft_evidence(
        df_raw: pd.DataFrame, raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts text-specific linguistic evidence (longest sentence and rare lexical
        items from 19_debug_lexical_tuples) to ground agent diagnostics without word caps.
        """
        evidence: Dict[str, Any] = {
            "sentence_count": 1,
            "longest_sentence": "",
            "longest_sentence_words": 0,
            "rare_lexical_items": [],
        }
        text_val = raw_text
        if not text_val and "raw_text" in df_raw.columns:
            text_val = str(df_raw.iloc[0]["raw_text"])

        if text_val:
            sents = [s.strip() for s in re.split(r"[.!?]+", str(text_val)) if s.strip()]
            if sents:
                evidence["sentence_count"] = len(sents)
                longest = max(sents, key=lambda s: len(s.split()))
                evidence["longest_sentence"] = longest
                evidence["longest_sentence_words"] = len(longest.split())

        if "19_debug_lexical_tuples" in df_raw.columns:
            raw_tuples = str(df_raw.iloc[0]["19_debug_lexical_tuples"] or "")
            # Extract word/lemma tokens with low Zipf frequency if formatted in debug string
            matches = re.findall(r"([א-ת\"׳']+)(?:[^0-9\-]*?)([1-3]\.\d+)", raw_tuples)
            if matches:
                matches_sorted = sorted(matches, key=lambda x: float(x[1]))
                seen = []
                for word, z_val in matches_sorted:
                    if word not in seen and len(word) > 2:
                        seen.append(f"{word} (Zipf={float(z_val):.2f})")
                    if len(seen) >= 4:
                        break
                evidence["rare_lexical_items"] = seen

        return evidence

    def get_raw_level_iqr_bounds(self, target_raw_level: str) -> Tuple[float, float, float]:
        """Returns (Q1, Q3, mu) for a specific raw level or 'academic/technical'."""
        clean = str(target_raw_level).strip()
        if clean.lower() in ("academic/technical", "technical/academic"):
            return round(self.max_q3, 4), round(max(self.max_p99, self.max_q3 + 1.5), 4), round(self.max_q3 + 0.45, 4)

        candidates = RAW_LEVEL_ALIASES.get(clean, [clean, f"'{clean}", clean.lstrip("'")])
        for cand in candidates:
            if cand in self.calibration_stats:
                s = self.calibration_stats[cand]
                return round(float(s["Q1"]), 4), round(float(s["Q3"]), 4), round(float(s["mu"]), 4)

        return 4.80, 5.50, 5.15

    def get_population_iqr_bounds(self, target_population: str) -> Tuple[float, float]:
        """Computes the calibrated Core IQR (Q1, Q3) span for a target population."""
        if target_population in ("Technical/Academic", "academic/technical"):
            return round(self.max_q3, 4), round(max(self.max_p99, self.max_q3 + 1.5), 4)

        spec = POPULATION_SPECS.get(target_population, {})
        candidate_keys = set(spec.get("raw_level_keys", []))
        target_cats = set(spec.get("engine_categories", []))

        q1_vals, q3_vals = [], []
        for lvl, stats in self.calibration_stats.items():
            norm_lvl = str(lvl).strip()
            lvl_cats = set(
                LEVEL_TO_CATEGORIES.get(norm_lvl)
                or LEVEL_TO_CATEGORIES.get(norm_lvl.lstrip("'"))
                or []
            )
            if norm_lvl in candidate_keys or norm_lvl.lstrip("'") in candidate_keys or (target_cats & lvl_cats):
                if "Q1" in stats and "Q3" in stats:
                    q1_vals.append(float(stats["Q1"]))
                    q3_vals.append(float(stats["Q3"]))

        if q1_vals and q3_vals:
            return round(min(q1_vals), 4), round(max(q3_vals), 4)

        anchor = float(spec.get("anchor_score", 5.0))
        return round(anchor - 0.45, 4), round(anchor + 0.45, 4)

    def evaluate_features_df(
        self,
        df_raw: pd.DataFrame,
        print_report: bool = True,
        raw_text_for_evidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scores a 1-row DataFrame of raw features, extracts per-prediction TreeSHAP
        values, tags above-range readability scores as 'academic/technical', and
        computes both extended and Core IQR standardized probabilities.
        """
        X_infer = self.preprocessor.transform(df_raw)
        y_pred = float(self.model.predict(X_infer)[0])
        local_shap, shap_base = self._compute_local_shap(X_infer)

        core_matches = [
            lvl for lvl, s in self.calibration_stats.items() if s["Q1"] <= y_pred <= s["Q3"]
        ]
        ext_matches = [
            lvl for lvl, s in self.calibration_stats.items() if s["P1"] <= y_pred <= s["P99"]
        ]

        # Requirement 3: Tag any text with readability score above the calibrated ranges as 'academic/technical'
        is_above_calibrated_range = bool(y_pred > self.max_q3 and len(core_matches) == 0)
        if y_pred > self.max_p99:
            is_above_calibrated_range = True

        if is_above_calibrated_range:
            if "academic/technical" not in core_matches:
                core_matches.append("academic/technical")
            if "academic/technical" not in ext_matches:
                ext_matches.append("academic/technical")

        core_categories = self._levels_to_categories(core_matches)
        ext_categories = self._levels_to_categories(ext_matches)

        is_oob = len(ext_matches) == 0
        eval_levels = [
            lvl for lvl in (ext_matches if not is_oob else list(self.calibration_stats.keys()))
            if lvl in self.calibration_stats
        ]

        # Standardized Z-score Gaussian proximity s_k(y) = exp(-0.5 * ((y - mu)/sigma)^2)
        z_proximities = {
            lvl: math.exp(
                -0.5
                * (
                    (y_pred - float(self.calibration_stats[lvl]["mu"]))
                    / max(float(self.calibration_stats[lvl]["sigma"]), 1e-4)
                )
                ** 2
            )
            for lvl in eval_levels
        }
        total_prox = sum(z_proximities.values())
        sorted_probs = (
            {
                lvl: round((score / total_prox) * 100.0, 1)
                for lvl, score in sorted(z_proximities.items(), key=lambda x: x[1], reverse=True)
            }
            if total_prox > 0
            else {}
        )

        # Core-IQR specific probability normalization (eliminates 17-level dilution)
        core_cal_levels = [lvl for lvl in core_matches if lvl in self.calibration_stats]
        if core_cal_levels:
            core_prox = {lvl: z_proximities.get(lvl, 0.01) for lvl in core_cal_levels}
            core_sum = sum(core_prox.values())
            core_iqr_probs = {
                lvl: round((val / core_sum) * 100.0, 1)
                for lvl, val in sorted(core_prox.items(), key=lambda x: x[1], reverse=True)
            }
        elif is_above_calibrated_range:
            core_iqr_probs = {"academic/technical": 100.0}
            sorted_probs = {"academic/technical": 100.0, **sorted_probs}
        else:
            core_iqr_probs = {}

        top_3_natural = [
            f"{LEVEL_DESCRIPTIONS.get(lvl, lvl)} ({prob}%)"
            for lvl, prob in list(core_iqr_probs.items() or sorted_probs.items())[:3]
        ]
        if is_above_calibrated_range:
            summary_text = (
                f"Readability Score: {y_pred:.2f}/10 (Above calibrated school ranges -> 'academic/technical')."
            )
        elif is_oob:
            summary_text = (
                f"Readability Score: {y_pred:.2f}/10 (Outside P1-P99 bounds). "
                f"Nearest pedagogical matches: {', '.join(top_3_natural)}."
            )
        else:
            summary_text = (
                f"Readability Score: {y_pred:.2f}/10. "
                f"Most suitable for: {', '.join(top_3_natural)}."
            )

        feat_dict = {
            k: round(float(v), 6) for k, v in X_infer.iloc[0].to_dict().items()
        }
        draft_evidence = self._extract_draft_evidence(df_raw, raw_text=raw_text_for_evidence)

        payload = {
            "raw_readability_score": round(y_pred, 4),
            "core_iqr_matches": core_matches,
            "core_iqr_categories": core_categories,
            "core_iqr_probabilities": core_iqr_probs,
            "extended_p1_p99_matches": ext_matches,
            "extended_p1_p99_categories": ext_categories,
            "probabilities": sorted_probs,
            "is_out_of_bounds": is_oob,
            "is_above_calibrated_range": is_above_calibrated_range,
            "natural_language_summary": summary_text,
            "transformed_features": feat_dict,
            "local_shap_values": local_shap,
            "shap_base_value": shap_base,
            "draft_evidence": draft_evidence,
        }

        if print_report:
            print(f"\n[+] Raw Continuous Readability Score: {y_pred:.4f}")
            print(
                f"[+] Core IQR (Q1-Q3) Matches:         {', '.join(core_matches) if core_matches else 'None'}"
            )
            print(
                f"[+] Core IQR Categories:              {', '.join(core_categories) if core_categories else 'None'}"
            )
            print(f"[Summary] {summary_text}")

        return payload

    def evaluate_text(self, raw_hebrew_text: str, print_report: bool = True) -> Dict[str, Any]:
        """Extracts linguistic features from a raw Hebrew string and predicts readability."""
        if not raw_hebrew_text or not str(raw_hebrew_text).strip():
            raise ValueError("Cannot evaluate empty or whitespace-only Hebrew text.")
        clean_input = str(raw_hebrew_text).strip()
        extractor_fn = self._ensure_extractor()
        df_raw = extractor_fn(clean_input)
        return self.evaluate_features_df(
            df_raw, print_report=print_report, raw_text_for_evidence=clean_input
        )

    def evaluate_txt_file(self, txt_filepath: str, print_report: bool = True) -> Dict[str, Any]:
        """Reads a UTF-8 .txt file and evaluates its readability."""
        if not os.path.exists(txt_filepath):
            raise FileNotFoundError(f"Input file not found: {txt_filepath}")
        with open(txt_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        return self.evaluate_text(raw_text, print_report=print_report)
