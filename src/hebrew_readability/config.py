"""Centralized configuration, per-level SHAP tables, raw-level medians, and I/O helpers."""

from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Hugging Face Repositories
HF_MODEL_REPO_ID = "GolDL/hebrew-readability-classifier"
HF_DICT_REPO_ID = "GolDL/hebrew-lemma-frequency-dictionary"
HF_DATASET_REPO_ID = "GolDL/heb-readability-txt"

# Default Filenames in HF Repos
RAW_DATASET_FILENAME = "heb_paragraphs_dataset.parquet"
DICT_PARQUET_FILENAME = "frequency_dictionary_clean.parquet"
DICT_CSV_FILENAME = "frequency_dictionary_clean.csv"
DICT_ONNX_URL = "https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.int8.onnx"

STRING_METADATA_COLS: List[str] = [
    "paragraph_id",
    "source_file",
    "category",
    "level",
    "passage_title",
    "raw_text",
    "2_cleaned_text",
    "15_vocalized_text",
    "19_debug_lexical_tuples",
]


def read_table(filepath: str, **kwargs) -> pd.DataFrame:
    """Reads a DataFrame from either .parquet or .csv automatically."""
    if str(filepath).endswith(".parquet"):
        df = pd.read_parquet(filepath, engine="pyarrow")
        if "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
            for col, dt in kwargs["dtype"].items():
                if col in df.columns:
                    df[col] = df[col].astype(dt)
        return df
    return pd.read_csv(filepath, **kwargs)


def save_table(df: pd.DataFrame, filepath: str) -> None:
    """Saves a DataFrame to .parquet (enforcing clean string types for PyArrow) or .csv."""
    out_df = df.copy()
    if str(filepath).endswith(".parquet"):
        for col in STRING_METADATA_COLS:
            if col in out_df.columns:
                out_df[col] = out_df[col].fillna("").astype(str)
        out_df.to_parquet(filepath, index=False, engine="pyarrow")
    else:
        out_df.to_csv(filepath, index=False)


def setup_pipeline_logger(
    name: str, artifact_dir: str = "artifacts", log_filename: str = "pipeline_execution.log"
) -> logging.Logger:
    """Configures a logger that outputs to both stdout and a persistent UTF-8 log file."""
    os.makedirs(artifact_dir, exist_ok=True)
    log_path = os.path.join(artifact_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Exact 11 Core Engineered Features expected by the purged XGBoost regressor
FINAL_FEATURE_ORDER: List[str] = [
    "1_nikkud_ratio_log_inv",
    "7_stz_avg_clauses_per_sent_log",
    "8_stz_avg_tree_depth_per_sent_log",
    "9_stz_ratio_passive_verbs_log",
    "10_dct_ratio_adj_log",
    "11_dct_ratio_adv_log",
    "12_dct_ratio_pron_log",
    "14a_zipf_rarity_tuple_inv",
    "16_total_syllables_log",
    "17_avg_syllables_per_word",
    "18_avg_syllables_per_sentence_log",
]

LEVEL_TO_CATEGORIES: Dict[str, List[str]] = {
    "1": ["Elementary - Beginner"],
    "2": ["Elementary - Beginner"],
    "3": ["Elementary - Beginner"],
    "'2-4": ["Elementary - Beginner", "Elementary - Intermediate"],
    "2-4": ["Elementary - Beginner", "Elementary - Intermediate"],
    "4": ["Elementary - Intermediate"],
    "5": ["Elementary - Intermediate"],
    "6": ["Elementary - Intermediate"],
    "'4-9": ["Elementary - Intermediate", "Middle School"],
    "4-9": ["Elementary - Intermediate", "Middle School"],
    "7": ["Middle School"],
    "8": ["Middle School"],
    "9": ["Middle School"],
    "10": ["High School"],
    "11": ["High School"],
    "12": ["High School", "Technical/Academic"],
    "Bagrut": ["High School", "Technical/Academic"],
    "Bagrut_Olim": ["High School", "Second-language - Advanced"],
    "hearing_impaired": ["High School"],
    "Bagrut_hearing_impaired": ["High School"],
    "intellectual_disability": ["Intellectual Disability"],
    "Hadshon": ["Second-language - Beginner"],
    "Olim_news": ["Second-language - Beginner"],
    "מבחני אולפן - רגיל": ["Second-language - Beginner"],
    "Olim_ragil": ["Second-language - Beginner"],
    "מבחני אולפן - מתקדמים": ["Second-language - Advanced"],
    "Olim_mitkadmim": ["Second-language - Advanced"],
    "academic/technical": ["Technical/Academic"],
    "Technical/Academic": ["Technical/Academic"],
}

LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "1": "Grade 1 (Elementary - Beginner)",
    "2": "Grade 2 (Elementary - Beginner)",
    "3": "Grade 3 (Elementary - Beginner)",
    "'2-4": "Grades 2-4 Cluster (Elementary - Beginner, Elementary - Intermediate)",
    "2-4": "Grades 2-4 Cluster (Elementary - Beginner, Elementary - Intermediate)",
    "4": "Grade 4 (Elementary - Intermediate)",
    "5": "Grade 5 (Elementary - Intermediate)",
    "6": "Grade 6 (Elementary - Intermediate)",
    "'4-9": "Grades 4-9 Broad Cluster (Elementary - Intermediate, Middle School)",
    "4-9": "Grades 4-9 Broad Cluster (Elementary - Intermediate, Middle School)",
    "7": "Grade 7 - Interpolated (Middle School)",
    "8": "Grade 8 (Middle School)",
    "9": "Grade 9 (Middle School)",
    "10": "Grade 10 - Interpolated (High School)",
    "11": "Grade 11 (High School)",
    "12": "Grade 12 (High School)",
    "Bagrut": "Bagrut (High School)",
    "Bagrut_Olim": "High School Bagrut (High School, Olim)",
    "hearing_impaired": "Hearing Impaired (High School)",
    "Bagrut_hearing_impaired": "Hearing Impaired Bagrut (High School)",
    "intellectual_disability": "Supported Easy-Read (Intellectual Disability)",
    "Hadshon": "Hadshon Easy News (Second-language - Beginner)",
    "Olim_news": "Hadshon Easy News (Second-language - Beginner)",
    "מבחני אולפן - רגיל": "מבחני אולפן - רגיל (Second-language - Beginner)",
    "Olim_ragil": "Ulpan Olim Regular (Second-language - Beginner)",
    "מבחני אולפן - מתקדמים": "מבחני אולפן - מתקדמים (Second-language - Advanced)",
    "Olim_mitkadmim": "Ulpan Olim Advanced (Second-language - Advanced)",
    "academic/technical": "Academic / Technical (Above Calibrated School Ranges)",
}

RAW_LEVEL_ALIASES: Dict[str, List[str]] = {
    "1": ["1", "2"],
    "2": ["2"],
    "3": ["3"],
    "'2-4": ["'2-4", "2-4"],
    "2-4": ["'2-4", "2-4"],
    "4": ["4"],
    "5": ["5"],
    "6": ["6"],
    "'4-9": ["'4-9", "4-9"],
    "4-9": ["'4-9", "4-9"],
    "7": ["7"],
    "8": ["8"],
    "9": ["9"],
    "10": ["10"],
    "11": ["11"],
    "12": ["12"],
    "Bagrut": ["Bagrut", "Bagrut_Olim"],
    "Bagrut_Olim": ["Bagrut_Olim", "Bagrut"],
    "hearing_impaired": ["hearing_impaired", "Bagrut_hearing_impaired"],
    "Bagrut_hearing_impaired": ["Bagrut_hearing_impaired", "hearing_impaired"],
    "intellectual_disability": ["intellectual_disability"],
    "Hadshon": ["Hadshon", "Olim_news"],
    "Olim_news": ["Olim_news", "Hadshon"],
    "Olim_ragil": ["Olim_ragil", "מבחני אולפן - רגיל"],
    "מבחני אולפן - רגיל": ["מבחני אולפן - רגיל", "Olim_ragil"],
    "Olim_mitkadmim": ["Olim_mitkadmim", "מבחני אולפן - מתקדמים"],
    "מבחני אולפן - מתקדמים": ["מבחני אולפן - מתקדמים", "Olim_mitkadmim"],
    "academic/technical": ["academic/technical", "Technical/Academic"],
    "Technical/Academic": ["academic/technical", "Technical/Academic"],
}

RAW_LEVEL_TO_POPULATION: Dict[str, str] = {
    "1": "Grade 1-3 - Elementary Beginner",
    "2": "Grade 1-3 - Elementary Beginner",
    "3": "Grade 1-3 - Elementary Beginner",
    "'2-4": "Grade 1-3 - Elementary Beginner",
    "2-4": "Grade 1-3 - Elementary Beginner",
    "4": "Grade 4-6 - Elementary Intermediate",
    "5": "Grade 4-6 - Elementary Intermediate",
    "6": "Grade 4-6 - Elementary Intermediate",
    "'4-9": "Grade 7-9 - Middle School",
    "4-9": "Grade 7-9 - Middle School",
    "7": "Grade 7-9 - Middle School",
    "8": "Grade 7-9 - Middle School",
    "9": "Grade 7-9 - Middle School",
    "10": "Grade 10-12 - High School",
    "11": "Grade 10-12 - High School",
    "12": "Grade 10-12 - High School",
    "Bagrut": "Grade 10-12 - High School",
    "hearing_impaired": "Grade 10-12 - High School",
    "Bagrut_hearing_impaired": "Grade 10-12 - High School",
    "Bagrut_Olim": "Second-language learners",
    "Hadshon": "Second-language learners",
    "Olim_news": "Second-language learners",
    "Olim_ragil": "Second-language learners",
    "מבחני אולפן - רגיל": "Second-language learners",
    "Olim_mitkadmim": "Second-language learners",
    "מבחני אולפן - מתקדמים": "Second-language learners",
    "intellectual_disability": "Intellectual Disability Accessible",
    "academic/technical": "Technical/Academic",
    "Technical/Academic": "Technical/Academic",
}

# Exact per-level Mean_Abs_SHAP values from training SHAP analysis
LEVEL_SHAP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "2": {"1_nikkud_ratio_log_inv": 1.1365, "18_avg_syllables_per_sentence_log": 0.0618, "16_total_syllables_log": 0.0365, "8_stz_avg_tree_depth_per_sent_log": 0.0238, "14a_zipf_rarity_tuple_inv": 0.0171, "7_stz_avg_clauses_per_sent_log": 0.0136, "10_dct_ratio_adj_log": 0.0077, "9_stz_ratio_passive_verbs_log": 0.0069, "11_dct_ratio_adv_log": 0.0042, "17_avg_syllables_per_word": 0.0037, "12_dct_ratio_pron_log": 0.0013},
    "3": {"1_nikkud_ratio_log_inv": 0.5540, "16_total_syllables_log": 0.1308, "18_avg_syllables_per_sentence_log": 0.0671, "9_stz_ratio_passive_verbs_log": 0.0315, "7_stz_avg_clauses_per_sent_log": 0.0276, "10_dct_ratio_adj_log": 0.0272, "8_stz_avg_tree_depth_per_sent_log": 0.0211, "11_dct_ratio_adv_log": 0.0167, "14a_zipf_rarity_tuple_inv": 0.0146, "17_avg_syllables_per_word": 0.0101, "12_dct_ratio_pron_log": 0.0100},
    "'2-4": {"1_nikkud_ratio_log_inv": 1.0722, "16_total_syllables_log": 0.0765, "18_avg_syllables_per_sentence_log": 0.0529, "10_dct_ratio_adj_log": 0.0240, "8_stz_avg_tree_depth_per_sent_log": 0.0182, "14a_zipf_rarity_tuple_inv": 0.0142, "7_stz_avg_clauses_per_sent_log": 0.0141, "9_stz_ratio_passive_verbs_log": 0.0125, "11_dct_ratio_adv_log": 0.0099, "17_avg_syllables_per_word": 0.0050, "12_dct_ratio_pron_log": 0.0049},
    "4": {"1_nikkud_ratio_log_inv": 0.2499, "16_total_syllables_log": 0.1283, "18_avg_syllables_per_sentence_log": 0.0780, "10_dct_ratio_adj_log": 0.0378, "8_stz_avg_tree_depth_per_sent_log": 0.0249, "9_stz_ratio_passive_verbs_log": 0.0224, "14a_zipf_rarity_tuple_inv": 0.0154, "7_stz_avg_clauses_per_sent_log": 0.0148, "12_dct_ratio_pron_log": 0.0144, "11_dct_ratio_adv_log": 0.0138, "17_avg_syllables_per_word": 0.0074},
    "5": {"1_nikkud_ratio_log_inv": 0.2778, "16_total_syllables_log": 0.1147, "18_avg_syllables_per_sentence_log": 0.0674, "10_dct_ratio_adj_log": 0.0381, "8_stz_avg_tree_depth_per_sent_log": 0.0243, "9_stz_ratio_passive_verbs_log": 0.0215, "14a_zipf_rarity_tuple_inv": 0.0205, "7_stz_avg_clauses_per_sent_log": 0.0152, "11_dct_ratio_adv_log": 0.0112, "17_avg_syllables_per_word": 0.0089, "12_dct_ratio_pron_log": 0.0079},
    "6": {"1_nikkud_ratio_log_inv": 0.3030, "16_total_syllables_log": 0.1232, "18_avg_syllables_per_sentence_log": 0.0504, "10_dct_ratio_adj_log": 0.0400, "8_stz_avg_tree_depth_per_sent_log": 0.0232, "14a_zipf_rarity_tuple_inv": 0.0170, "9_stz_ratio_passive_verbs_log": 0.0169, "7_stz_avg_clauses_per_sent_log": 0.0162, "12_dct_ratio_pron_log": 0.0090, "17_avg_syllables_per_word": 0.0081, "11_dct_ratio_adv_log": 0.0048},
    "'4-9": {"1_nikkud_ratio_log_inv": 0.2962, "16_total_syllables_log": 0.0887, "18_avg_syllables_per_sentence_log": 0.0571, "10_dct_ratio_adj_log": 0.0240, "8_stz_avg_tree_depth_per_sent_log": 0.0206, "14a_zipf_rarity_tuple_inv": 0.0205, "9_stz_ratio_passive_verbs_log": 0.0168, "7_stz_avg_clauses_per_sent_log": 0.0135, "17_avg_syllables_per_word": 0.0068, "11_dct_ratio_adv_log": 0.0064, "12_dct_ratio_pron_log": 0.0044},
    "8": {"1_nikkud_ratio_log_inv": 0.3552, "16_total_syllables_log": 0.0956, "18_avg_syllables_per_sentence_log": 0.0562, "14a_zipf_rarity_tuple_inv": 0.0226, "10_dct_ratio_adj_log": 0.0197, "8_stz_avg_tree_depth_per_sent_log": 0.0192, "9_stz_ratio_passive_verbs_log": 0.0127, "7_stz_avg_clauses_per_sent_log": 0.0084, "17_avg_syllables_per_word": 0.0075, "12_dct_ratio_pron_log": 0.0045, "11_dct_ratio_adv_log": 0.0041},
    "9": {"1_nikkud_ratio_log_inv": 0.3284, "16_total_syllables_log": 0.1040, "18_avg_syllables_per_sentence_log": 0.0737, "8_stz_avg_tree_depth_per_sent_log": 0.0239, "10_dct_ratio_adj_log": 0.0194, "14a_zipf_rarity_tuple_inv": 0.0189, "9_stz_ratio_passive_verbs_log": 0.0170, "7_stz_avg_clauses_per_sent_log": 0.0109, "17_avg_syllables_per_word": 0.0078, "11_dct_ratio_adv_log": 0.0061, "12_dct_ratio_pron_log": 0.0056},
    "11": {"1_nikkud_ratio_log_inv": 0.3298, "16_total_syllables_log": 0.1038, "18_avg_syllables_per_sentence_log": 0.0590, "14a_zipf_rarity_tuple_inv": 0.0200, "10_dct_ratio_adj_log": 0.0199, "8_stz_avg_tree_depth_per_sent_log": 0.0182, "9_stz_ratio_passive_verbs_log": 0.0137, "7_stz_avg_clauses_per_sent_log": 0.0125, "17_avg_syllables_per_word": 0.0092, "12_dct_ratio_pron_log": 0.0045, "11_dct_ratio_adv_log": 0.0039},
    "12": {"1_nikkud_ratio_log_inv": 0.3242, "16_total_syllables_log": 0.1050, "18_avg_syllables_per_sentence_log": 0.0641, "14a_zipf_rarity_tuple_inv": 0.0204, "10_dct_ratio_adj_log": 0.0195, "8_stz_avg_tree_depth_per_sent_log": 0.0189, "9_stz_ratio_passive_verbs_log": 0.0138, "7_stz_avg_clauses_per_sent_log": 0.0103, "17_avg_syllables_per_word": 0.0094, "12_dct_ratio_pron_log": 0.0051, "11_dct_ratio_adv_log": 0.0048},
    "Bagrut_Olim": {"1_nikkud_ratio_log_inv": 0.3314, "16_total_syllables_log": 0.1044, "18_avg_syllables_per_sentence_log": 0.0660, "10_dct_ratio_adj_log": 0.0311, "8_stz_avg_tree_depth_per_sent_log": 0.0252, "14a_zipf_rarity_tuple_inv": 0.0197, "9_stz_ratio_passive_verbs_log": 0.0152, "7_stz_avg_clauses_per_sent_log": 0.0102, "17_avg_syllables_per_word": 0.0071, "11_dct_ratio_adv_log": 0.0060, "12_dct_ratio_pron_log": 0.0035},
    "Bagrut_hearing_impaired": {"1_nikkud_ratio_log_inv": 0.3323, "16_total_syllables_log": 0.1103, "18_avg_syllables_per_sentence_log": 0.0682, "10_dct_ratio_adj_log": 0.0192, "8_stz_avg_tree_depth_per_sent_log": 0.0191, "14a_zipf_rarity_tuple_inv": 0.0175, "9_stz_ratio_passive_verbs_log": 0.0090, "17_avg_syllables_per_word": 0.0079, "7_stz_avg_clauses_per_sent_log": 0.0075, "12_dct_ratio_pron_log": 0.0037, "11_dct_ratio_adv_log": 0.0037},
    "Olim_mitkadmim": {"1_nikkud_ratio_log_inv": 0.3385, "16_total_syllables_log": 0.0864, "18_avg_syllables_per_sentence_log": 0.0298, "10_dct_ratio_adj_log": 0.0232, "8_stz_avg_tree_depth_per_sent_log": 0.0220, "14a_zipf_rarity_tuple_inv": 0.0176, "9_stz_ratio_passive_verbs_log": 0.0157, "7_stz_avg_clauses_per_sent_log": 0.0095, "17_avg_syllables_per_word": 0.0056, "11_dct_ratio_adv_log": 0.0037, "12_dct_ratio_pron_log": 0.0028},
    "Olim_ragil": {"1_nikkud_ratio_log_inv": 0.3096, "16_total_syllables_log": 0.0854, "18_avg_syllables_per_sentence_log": 0.0653, "14a_zipf_rarity_tuple_inv": 0.0351, "8_stz_avg_tree_depth_per_sent_log": 0.0243, "9_stz_ratio_passive_verbs_log": 0.0213, "10_dct_ratio_adj_log": 0.0203, "7_stz_avg_clauses_per_sent_log": 0.0153, "17_avg_syllables_per_word": 0.0074, "12_dct_ratio_pron_log": 0.0043, "11_dct_ratio_adv_log": 0.0035},
    "intellectual_disability": {"1_nikkud_ratio_log_inv": 0.2825, "16_total_syllables_log": 0.1875, "18_avg_syllables_per_sentence_log": 0.0823, "10_dct_ratio_adj_log": 0.0533, "14a_zipf_rarity_tuple_inv": 0.0399, "9_stz_ratio_passive_verbs_log": 0.0242, "8_stz_avg_tree_depth_per_sent_log": 0.0221, "7_stz_avg_clauses_per_sent_log": 0.0184, "11_dct_ratio_adv_log": 0.0122, "12_dct_ratio_pron_log": 0.0115, "17_avg_syllables_per_word": 0.0094},
}

# Granular per-raw-level feature medians across all 11 core engineered features
RAW_LEVEL_MEDIANS: Dict[str, Dict[str, float]] = {
    "1": {"1_nikkud_ratio_log_inv": 0.000, "16_total_syllables_log": 0.440, "18_avg_syllables_per_sentence_log": 0.435, "10_dct_ratio_adj_log": 0.150, "8_stz_avg_tree_depth_per_sent_log": 0.275, "14a_zipf_rarity_tuple_inv": 0.385, "7_stz_avg_clauses_per_sent_log": 0.190, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.130, "17_avg_syllables_per_word": 0.295, "12_dct_ratio_pron_log": 0.295},
    "2": {"1_nikkud_ratio_log_inv": 0.000, "16_total_syllables_log": 0.475, "18_avg_syllables_per_sentence_log": 0.460, "10_dct_ratio_adj_log": 0.175, "8_stz_avg_tree_depth_per_sent_log": 0.295, "14a_zipf_rarity_tuple_inv": 0.400, "7_stz_avg_clauses_per_sent_log": 0.215, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.145, "17_avg_syllables_per_word": 0.308, "12_dct_ratio_pron_log": 0.285},
    "3": {"1_nikkud_ratio_log_inv": 0.120, "16_total_syllables_log": 0.505, "18_avg_syllables_per_sentence_log": 0.482, "10_dct_ratio_adj_log": 0.205, "8_stz_avg_tree_depth_per_sent_log": 0.308, "14a_zipf_rarity_tuple_inv": 0.412, "7_stz_avg_clauses_per_sent_log": 0.240, "9_stz_ratio_passive_verbs_log": 0.020, "11_dct_ratio_adv_log": 0.165, "17_avg_syllables_per_word": 0.325, "12_dct_ratio_pron_log": 0.275},
    "'2-4": {"1_nikkud_ratio_log_inv": 0.050, "16_total_syllables_log": 0.490, "18_avg_syllables_per_sentence_log": 0.472, "10_dct_ratio_adj_log": 0.190, "8_stz_avg_tree_depth_per_sent_log": 0.302, "14a_zipf_rarity_tuple_inv": 0.406, "7_stz_avg_clauses_per_sent_log": 0.228, "9_stz_ratio_passive_verbs_log": 0.015, "11_dct_ratio_adv_log": 0.155, "17_avg_syllables_per_word": 0.318, "12_dct_ratio_pron_log": 0.280},
    "4": {"1_nikkud_ratio_log_inv": 0.920, "16_total_syllables_log": 0.535, "18_avg_syllables_per_sentence_log": 0.495, "10_dct_ratio_adj_log": 0.235, "8_stz_avg_tree_depth_per_sent_log": 0.310, "14a_zipf_rarity_tuple_inv": 0.418, "7_stz_avg_clauses_per_sent_log": 0.265, "9_stz_ratio_passive_verbs_log": 0.055, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.345, "12_dct_ratio_pron_log": 0.260},
    "5": {"1_nikkud_ratio_log_inv": 0.950, "16_total_syllables_log": 0.567, "18_avg_syllables_per_sentence_log": 0.512, "10_dct_ratio_adj_log": 0.260, "8_stz_avg_tree_depth_per_sent_log": 0.318, "14a_zipf_rarity_tuple_inv": 0.425, "7_stz_avg_clauses_per_sent_log": 0.290, "9_stz_ratio_passive_verbs_log": 0.080, "11_dct_ratio_adv_log": 0.200, "17_avg_syllables_per_word": 0.360, "12_dct_ratio_pron_log": 0.250},
    "6": {"1_nikkud_ratio_log_inv": 0.970, "16_total_syllables_log": 0.592, "18_avg_syllables_per_sentence_log": 0.530, "10_dct_ratio_adj_log": 0.285, "8_stz_avg_tree_depth_per_sent_log": 0.332, "14a_zipf_rarity_tuple_inv": 0.430, "7_stz_avg_clauses_per_sent_log": 0.312, "9_stz_ratio_passive_verbs_log": 0.105, "11_dct_ratio_adv_log": 0.210, "17_avg_syllables_per_word": 0.372, "12_dct_ratio_pron_log": 0.242},
    "'4-9": {"1_nikkud_ratio_log_inv": 0.975, "16_total_syllables_log": 0.605, "18_avg_syllables_per_sentence_log": 0.542, "10_dct_ratio_adj_log": 0.295, "8_stz_avg_tree_depth_per_sent_log": 0.340, "14a_zipf_rarity_tuple_inv": 0.432, "7_stz_avg_clauses_per_sent_log": 0.325, "9_stz_ratio_passive_verbs_log": 0.120, "11_dct_ratio_adv_log": 0.215, "17_avg_syllables_per_word": 0.380, "12_dct_ratio_pron_log": 0.236},
    "7": {"1_nikkud_ratio_log_inv": 0.980, "16_total_syllables_log": 0.610, "18_avg_syllables_per_sentence_log": 0.545, "10_dct_ratio_adj_log": 0.300, "8_stz_avg_tree_depth_per_sent_log": 0.344, "14a_zipf_rarity_tuple_inv": 0.433, "7_stz_avg_clauses_per_sent_log": 0.330, "9_stz_ratio_passive_verbs_log": 0.128, "11_dct_ratio_adv_log": 0.218, "17_avg_syllables_per_word": 0.384, "12_dct_ratio_pron_log": 0.234},
    "8": {"1_nikkud_ratio_log_inv": 0.985, "16_total_syllables_log": 0.621, "18_avg_syllables_per_sentence_log": 0.556, "10_dct_ratio_adj_log": 0.310, "8_stz_avg_tree_depth_per_sent_log": 0.351, "14a_zipf_rarity_tuple_inv": 0.435, "7_stz_avg_clauses_per_sent_log": 0.340, "9_stz_ratio_passive_verbs_log": 0.140, "11_dct_ratio_adv_log": 0.220, "17_avg_syllables_per_word": 0.390, "12_dct_ratio_pron_log": 0.230},
    "9": {"1_nikkud_ratio_log_inv": 0.990, "16_total_syllables_log": 0.640, "18_avg_syllables_per_sentence_log": 0.568, "10_dct_ratio_adj_log": 0.325, "8_stz_avg_tree_depth_per_sent_log": 0.358, "14a_zipf_rarity_tuple_inv": 0.444, "7_stz_avg_clauses_per_sent_log": 0.355, "9_stz_ratio_passive_verbs_log": 0.155, "11_dct_ratio_adv_log": 0.228, "17_avg_syllables_per_word": 0.402, "12_dct_ratio_pron_log": 0.222},
    "10": {"1_nikkud_ratio_log_inv": 0.994, "16_total_syllables_log": 0.655, "18_avg_syllables_per_sentence_log": 0.578, "10_dct_ratio_adj_log": 0.340, "8_stz_avg_tree_depth_per_sent_log": 0.362, "14a_zipf_rarity_tuple_inv": 0.450, "7_stz_avg_clauses_per_sent_log": 0.368, "9_stz_ratio_passive_verbs_log": 0.168, "11_dct_ratio_adv_log": 0.235, "17_avg_syllables_per_word": 0.412, "12_dct_ratio_pron_log": 0.216},
    "11": {"1_nikkud_ratio_log_inv": 0.996, "16_total_syllables_log": 0.670, "18_avg_syllables_per_sentence_log": 0.587, "10_dct_ratio_adj_log": 0.350, "8_stz_avg_tree_depth_per_sent_log": 0.367, "14a_zipf_rarity_tuple_inv": 0.457, "7_stz_avg_clauses_per_sent_log": 0.380, "9_stz_ratio_passive_verbs_log": 0.180, "11_dct_ratio_adv_log": 0.240, "17_avg_syllables_per_word": 0.420, "12_dct_ratio_pron_log": 0.210},
    "12": {"1_nikkud_ratio_log_inv": 0.998, "16_total_syllables_log": 0.688, "18_avg_syllables_per_sentence_log": 0.602, "10_dct_ratio_adj_log": 0.370, "8_stz_avg_tree_depth_per_sent_log": 0.378, "14a_zipf_rarity_tuple_inv": 0.472, "7_stz_avg_clauses_per_sent_log": 0.395, "9_stz_ratio_passive_verbs_log": 0.195, "11_dct_ratio_adv_log": 0.248, "17_avg_syllables_per_word": 0.432, "12_dct_ratio_pron_log": 0.200},
    "Bagrut": {"1_nikkud_ratio_log_inv": 0.997, "16_total_syllables_log": 0.680, "18_avg_syllables_per_sentence_log": 0.595, "10_dct_ratio_adj_log": 0.360, "8_stz_avg_tree_depth_per_sent_log": 0.372, "14a_zipf_rarity_tuple_inv": 0.465, "7_stz_avg_clauses_per_sent_log": 0.388, "9_stz_ratio_passive_verbs_log": 0.188, "11_dct_ratio_adv_log": 0.244, "17_avg_syllables_per_word": 0.426, "12_dct_ratio_pron_log": 0.205},
    "Bagrut_Olim": {"1_nikkud_ratio_log_inv": 0.965, "16_total_syllables_log": 0.635, "18_avg_syllables_per_sentence_log": 0.548, "10_dct_ratio_adj_log": 0.285, "8_stz_avg_tree_depth_per_sent_log": 0.342, "14a_zipf_rarity_tuple_inv": 0.455, "7_stz_avg_clauses_per_sent_log": 0.315, "9_stz_ratio_passive_verbs_log": 0.115, "11_dct_ratio_adv_log": 0.205, "17_avg_syllables_per_word": 0.375, "12_dct_ratio_pron_log": 0.240},
    "hearing_impaired": {"1_nikkud_ratio_log_inv": 0.995, "16_total_syllables_log": 0.660, "18_avg_syllables_per_sentence_log": 0.575, "10_dct_ratio_adj_log": 0.335, "8_stz_avg_tree_depth_per_sent_log": 0.355, "14a_zipf_rarity_tuple_inv": 0.448, "7_stz_avg_clauses_per_sent_log": 0.360, "9_stz_ratio_passive_verbs_log": 0.150, "11_dct_ratio_adv_log": 0.225, "17_avg_syllables_per_word": 0.405, "12_dct_ratio_pron_log": 0.220},
    "Bagrut_hearing_impaired": {"1_nikkud_ratio_log_inv": 0.995, "16_total_syllables_log": 0.660, "18_avg_syllables_per_sentence_log": 0.575, "10_dct_ratio_adj_log": 0.335, "8_stz_avg_tree_depth_per_sent_log": 0.355, "14a_zipf_rarity_tuple_inv": 0.448, "7_stz_avg_clauses_per_sent_log": 0.360, "9_stz_ratio_passive_verbs_log": 0.150, "11_dct_ratio_adv_log": 0.225, "17_avg_syllables_per_word": 0.405, "12_dct_ratio_pron_log": 0.220},
    "Hadshon": {"1_nikkud_ratio_log_inv": 0.850, "16_total_syllables_log": 0.520, "18_avg_syllables_per_sentence_log": 0.485, "10_dct_ratio_adj_log": 0.215, "8_stz_avg_tree_depth_per_sent_log": 0.305, "14a_zipf_rarity_tuple_inv": 0.435, "7_stz_avg_clauses_per_sent_log": 0.245, "9_stz_ratio_passive_verbs_log": 0.045, "11_dct_ratio_adv_log": 0.175, "17_avg_syllables_per_word": 0.335, "12_dct_ratio_pron_log": 0.270},
    "Olim_news": {"1_nikkud_ratio_log_inv": 0.850, "16_total_syllables_log": 0.520, "18_avg_syllables_per_sentence_log": 0.485, "10_dct_ratio_adj_log": 0.215, "8_stz_avg_tree_depth_per_sent_log": 0.305, "14a_zipf_rarity_tuple_inv": 0.435, "7_stz_avg_clauses_per_sent_log": 0.245, "9_stz_ratio_passive_verbs_log": 0.045, "11_dct_ratio_adv_log": 0.175, "17_avg_syllables_per_word": 0.335, "12_dct_ratio_pron_log": 0.270},
    "Olim_ragil": {"1_nikkud_ratio_log_inv": 0.900, "16_total_syllables_log": 0.575, "18_avg_syllables_per_sentence_log": 0.515, "10_dct_ratio_adj_log": 0.245, "8_stz_avg_tree_depth_per_sent_log": 0.320, "14a_zipf_rarity_tuple_inv": 0.460, "7_stz_avg_clauses_per_sent_log": 0.275, "9_stz_ratio_passive_verbs_log": 0.075, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.348, "12_dct_ratio_pron_log": 0.262},
    "מבחני אולפן - רגיל": {"1_nikkud_ratio_log_inv": 0.900, "16_total_syllables_log": 0.575, "18_avg_syllables_per_sentence_log": 0.515, "10_dct_ratio_adj_log": 0.245, "8_stz_avg_tree_depth_per_sent_log": 0.320, "14a_zipf_rarity_tuple_inv": 0.460, "7_stz_avg_clauses_per_sent_log": 0.275, "9_stz_ratio_passive_verbs_log": 0.075, "11_dct_ratio_adv_log": 0.185, "17_avg_syllables_per_word": 0.348, "12_dct_ratio_pron_log": 0.262},
    "Olim_mitkadmim": {"1_nikkud_ratio_log_inv": 0.955, "16_total_syllables_log": 0.615, "18_avg_syllables_per_sentence_log": 0.538, "10_dct_ratio_adj_log": 0.270, "8_stz_avg_tree_depth_per_sent_log": 0.335, "14a_zipf_rarity_tuple_inv": 0.478, "7_stz_avg_clauses_per_sent_log": 0.295, "9_stz_ratio_passive_verbs_log": 0.105, "11_dct_ratio_adv_log": 0.198, "17_avg_syllables_per_word": 0.362, "12_dct_ratio_pron_log": 0.252},
    "מבחני אולפן - מתקדמים": {"1_nikkud_ratio_log_inv": 0.955, "16_total_syllables_log": 0.615, "18_avg_syllables_per_sentence_log": 0.538, "10_dct_ratio_adj_log": 0.270, "8_stz_avg_tree_depth_per_sent_log": 0.335, "14a_zipf_rarity_tuple_inv": 0.478, "7_stz_avg_clauses_per_sent_log": 0.295, "9_stz_ratio_passive_verbs_log": 0.105, "11_dct_ratio_adv_log": 0.198, "17_avg_syllables_per_word": 0.362, "12_dct_ratio_pron_log": 0.252},
    "intellectual_disability": {"1_nikkud_ratio_log_inv": 1.000, "16_total_syllables_log": 0.427, "18_avg_syllables_per_sentence_log": 0.464, "10_dct_ratio_adj_log": 0.190, "8_stz_avg_tree_depth_per_sent_log": 0.316, "14a_zipf_rarity_tuple_inv": 0.475, "7_stz_avg_clauses_per_sent_log": 0.210, "9_stz_ratio_passive_verbs_log": 0.000, "11_dct_ratio_adv_log": 0.160, "17_avg_syllables_per_word": 0.320, "12_dct_ratio_pron_log": 0.290},
    "academic/technical": {"1_nikkud_ratio_log_inv": 1.000, "16_total_syllables_log": 0.720, "18_avg_syllables_per_sentence_log": 0.640, "10_dct_ratio_adj_log": 0.420, "8_stz_avg_tree_depth_per_sent_log": 0.400, "14a_zipf_rarity_tuple_inv": 0.520, "7_stz_avg_clauses_per_sent_log": 0.430, "9_stz_ratio_passive_verbs_log": 0.240, "11_dct_ratio_adv_log": 0.260, "17_avg_syllables_per_word": 0.460, "12_dct_ratio_pron_log": 0.180},
}

POPULATION_SPECS: Dict[str, Dict[str, Any]] = {
    "Grade 1-3 - Elementary Beginner": {
        "engine_categories": ["Elementary - Beginner"],
        "raw_level_keys": ["1", "2", "3", "'2-4", "2-4"],
        "anchor_score": 3.21,
        "medians": RAW_LEVEL_MEDIANS["2"],
    },
    "Grade 4-6 - Elementary Intermediate": {
        "engine_categories": ["Elementary - Intermediate"],
        "raw_level_keys": ["4", "5", "6", "'4-9", "4-9"],
        "anchor_score": 5.31,
        "medians": RAW_LEVEL_MEDIANS["5"],
    },
    "Grade 7-9 - Middle School": {
        "engine_categories": ["Middle School"],
        "raw_level_keys": ["7", "8", "9", "'4-9", "4-9"],
        "anchor_score": 5.69,
        "medians": RAW_LEVEL_MEDIANS["8"],
    },
    "Grade 10-12 - High School": {
        "engine_categories": ["High School"],
        "raw_level_keys": ["10", "11", "12", "Bagrut", "hearing_impaired", "Bagrut_hearing_impaired"],
        "anchor_score": 5.95,
        "medians": RAW_LEVEL_MEDIANS["11"],
    },
    "Technical/Academic": {
        "engine_categories": ["Technical/Academic"],
        "raw_level_keys": ["academic/technical", "12", "Bagrut"],
        "anchor_score": 6.80,
        "medians": RAW_LEVEL_MEDIANS["academic/technical"],
    },
    "Second-language learners": {
        "engine_categories": ["Second-language - Beginner", "Second-language - Advanced"],
        "raw_level_keys": [
            "Hadshon", "Olim_news", "מבחני אולפן - רגיל", "Olim_ragil",
            "מבחני אולפן - מתקדמים", "Olim_mitkadmim", "Bagrut_Olim",
        ],
        "anchor_score": 5.33,
        "medians": RAW_LEVEL_MEDIANS["Olim_ragil"],
    },
    "Intellectual Disability Accessible": {
        "engine_categories": ["Intellectual Disability"],
        "raw_level_keys": ["intellectual_disability"],
        "anchor_score": 4.96,
        "medians": RAW_LEVEL_MEDIANS["intellectual_disability"],
    },
}

# Linguistic adaptation rules emphasizing 100% factual content retention
FEATURE_DIAGNOSTIC_RULES: Dict[str, Dict[str, Any]] = {
    "1_nikkud_ratio_log_inv": {
        "name": "Vocalization (Nikkud)",
        "weight": 0.300,
        "too_high": "Add Hebrew vowel diacritics (ניקוד) appropriate for this reading level while preserving all content.",
        "too_low": "Remove Nikkud vowel marks (write unvocalized Hebrew) unless disambiguating rare terms.",
    },
    "16_total_syllables_log": {
        "name": "Information & Syllabic Density (Paragraph Volume)",
        "weight": 0.120,
        "too_high": "Express all facts more concisely by eliminating redundant phrasing and wordy transitions WITHOUT omitting any factual content.",
        "too_low": "Elaborate slightly on key concepts with clearer context so the passage is not overly telegraphic.",
    },
    "18_avg_syllables_per_sentence_log": {
        "name": "Sentence Breath Load (Syllables per Sentence)",
        "weight": 0.068,
        "too_high": "Break long, multi-part sentences into shorter standalone sentences while retaining every fact.",
        "too_low": "Combine choppy, fragmented sentences into smoother, naturally connected sentences.",
    },
    "10_dct_ratio_adj_log": {
        "name": "Adjective & Modifier Density",
        "weight": 0.032,
        "too_high": "Reduce stacked or ornamental adjectives; keep only modifiers essential to the factual meaning.",
        "too_low": "Incorporate precise descriptive adjectives to enrich noun phrases.",
    },
    "8_stz_avg_tree_depth_per_sent_log": {
        "name": "Syntactic Dependency Tree Depth",
        "weight": 0.023,
        "too_high": "Flatten deeply nested syntactic hierarchies into direct, linear Subject-Verb-Object order.",
        "too_low": "Use more mature syntactic framing and natural hierarchical structure.",
    },
    "14a_zipf_rarity_tuple_inv": {
        "name": "Lexical Rarity (Zipf Lemma-POS Frequency)",
        "weight": 0.022,
        "too_high": "Replace low-frequency, archaic, or specialized lemmas with familiar, high-frequency everyday Hebrew equivalents that convey the exact same meaning.",
        "too_low": "Replace overly basic words with more precise, grade-appropriate curricular terminology.",
    },
    "9_stz_ratio_passive_verbs_log": {
        "name": "Passive Voice Ratio",
        "weight": 0.018,
        "too_high": "Convert passive verb constructions (פועל, הופעל, נפעל) into active voice with a clear subject.",
        "too_low": "Use standard passive or impersonal verbal constructions where appropriate for register.",
    },
    "7_stz_avg_clauses_per_sent_log": {
        "name": "Clause Density (Subordination per Sentence)",
        "weight": 0.015,
        "too_high": "Unpack subordinate and relative clauses (אשר, ש..., כיוון ש...) into separate, sequential sentences.",
        "too_low": "Connect related clauses smoothly using coordinating or subordinating conjunctions.",
    },
    "17_avg_syllables_per_word": {
        "name": "Morphological Word Load (Syllables per Word)",
        "weight": 0.008,
        "too_high": "Decompose heavy bound suffixes and stacked prefixes into analytic forms (e.g., 'הבית שלו' instead of 'ביתו').",
        "too_low": "Use standard inflected Hebrew morphology and bound forms.",
    },
    "11_dct_ratio_adv_log": {
        "name": "Adverbial & Qualifier Load",
        "weight": 0.008,
        "too_high": "Trim non-essential rhetorical qualifiers and adverbs while keeping temporal/factual markers.",
        "too_low": "Include qualifying adverbs where helpful for precision.",
    },
    "12_dct_ratio_pron_log": {
        "name": "Pronoun Reference Clarity",
        "weight": 0.007,
        "too_high": "Replace ambiguous pronouns with explicit noun referents.",
        "too_low": "Use clear personal pronouns ('הוא', 'היא', 'הם') to maintain cohesion across split sentences.",
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for reproducible training, calibration, and evaluation."""

    raw_data_path: str = "artifacts/features_dataset_clean.parquet"
    artifact_dir: str = "artifacts"
    shap_dir: str = "artifacts/shap_analysis_output"
    shap_advanced_dir: str = "artifacts/shap_analysis_output/advanced_gt_4_4"

    group_col: str = "passage_title"
    level_col: str = "level"
    target_col: str = "Target_Readability_Score"
    advanced_score_threshold: float = 4.4

    metadata_cols: List[str] = field(
        default_factory=lambda: [
            "paragraph_id",
            "raw_text",
            "level",
            "source_file",
            "passage_title",
            "paragraph_index",
        ]
    )

    inverse_base_features: List[str] = field(
        default_factory=lambda: [
            "1_nikkud_ratio",
            "13a_oov_ratio_tuple",
            "13b_oov_ratio_pos",
            "13c_oov_ratio_lemma",
            "14a_zipf_rarity_tuple",
            "14b_zipf_rarity_pos",
            "14c_zipf_rarity_lemma",
        ]
    )

    selected_base_features: List[str] = field(
        default_factory=lambda: [
            "1_nikkud_ratio",
            "7_stz_avg_clauses_per_sent",
            "8_stz_avg_tree_depth_per_sent",
            "9_stz_ratio_passive_verbs",
            "10_dct_ratio_adj",
            "11_dct_ratio_adv",
            "12_dct_ratio_pron",
            "14a_zipf_rarity_tuple",
            "16_total_syllables",
            "17_avg_syllables_per_word",
            "18_avg_syllables_per_sentence",
        ]
    )

    skew_threshold: float = 1.0
    fixed_nikkud_weight: float = 0.20
    remaining_weight_pool: float = 0.80
    n_splits: int = 5
    random_state: int = 42
