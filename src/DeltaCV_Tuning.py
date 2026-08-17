#!/usr/bin/python3
"""
Delta CV Tuning – Two-Stage Pipeline
=====================================
Stage A  (MODE="build_cache")
    Runs repeated stratified k-fold CV once per dataset.
    Saves avg / std / se NLL per model to cv_cache.csv.
    No delta logic here -- only raw CV scores.

Stage B  (MODE="delta_sweep")
    Loads cv_cache.csv.
    Datasets where CV was NOT feasible are EXCLUDED from delta tuning entirely
    (logged to skipped with reason="CV_not_feasible"), mirroring the original:
        if SORT_BY == "CV" and not cv_used: skipped_rows.append(...); continue
    For every remaining (dataset, delta) pair:
        - Applies the same equivalence + parsimony rule as cross_validate_weibull_models
        - Fits the selected model on the full in-sample data
        - Evaluates RMSE on the holdout set
    Aggregates RMSE by delta and writes all output CSVs.

Usage:
    Set MODE = "build_cache"  -> runs Stage A, writes cv_cache.csv
    Set MODE = "delta_sweep"  -> runs Stage B, requires cv_cache.csv to exist
"""

import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from weibull import weibull_fit_best
from Synthetic_Data import load_datasets_from_csv
from Validate_Weibull_CI import _parse_proportion_from_filename, _get_true_cdf
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P, Fit_Weibull_CR, Fit_Weibull_Mixture
from utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# CONFIGURATION  --  edit here
# ------------------------------------------------------------------------------

MODE = "delta_sweep"   # "build_cache"  |  "delta_sweep"

# To be defined
CSV_DIR    = r""
OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

CV_CACHE_CSV = os.path.join(OUTPUT_DIR, "cv_cache.csv")

DELTA_GRID       = [0.462, 0.463, 0.464, 0.465, 0.466, 0.467, 0.468]
SAMPLE_SEED      = 42
SAMPLES_PER_TYPE = 187
CV_SEED          = 42
CV_N_FOLDS       = 5
CV_N_REPEATS     = 5


# ------------------------------------------------------------------------------
# SHARED CONSTANTS
# ------------------------------------------------------------------------------

MODEL_NAMES = ["Weibull_2P", "Weibull_3P", "Weibull_CR", "Weibull_Mixture"]

COMPLEXITY = {"Weibull_2P":      2,
              "Weibull_3P":      3,
              "Weibull_CR":      4,
              "Weibull_Mixture": 5
}

FOLD_MIN_FAIL = {"Weibull_2P":      2,
                 "Weibull_3P":      3,
                 "Weibull_CR":      4,
                 "Weibull_Mixture": 5
}


# ------------------------------------------------------------------------------
# HELPERS -- file sampling
# ------------------------------------------------------------------------------

def _get_csv_files_by_type(csv_dir: str) -> dict:
    """Group valid synthetic CSV files by distribution type."""
    groups: dict = defaultdict(list)
    for f in os.listdir(csv_dir):
        if (f.lower().endswith(".csv") and f.startswith("synth_") and not f.endswith("_reliasoft.csv")):
            stem = Path(f).stem
            if   "2P"  in stem: groups["2P"].append(f)
            elif "3P"  in stem: groups["3P"].append(f)
            elif "CR"  in stem: groups["CR"].append(f)
            elif "Mix" in stem: groups["Mix"].append(f)
    return groups


def _sample_csv_files(csv_dir: str, n: int = SAMPLES_PER_TYPE, seed: int = SAMPLE_SEED) -> list[str]:
    """Reproducibly sample up to n files per distribution type."""
    groups = _get_csv_files_by_type(csv_dir)
    rng    = random.Random(seed)
    sampled: list[str] = []
    for files in groups.values():
        sampled.extend(rng.sample(files, min(n, len(files))))
    return sampled


def _load_ds_arrays(ds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and coerce failures / censored / holdout arrays from a dataset dict."""
    failures = ds.get("failures",  None)
    censored = ds.get("censored",  ds.get("suspensions", None))
    holdout  = ds.get("holdout",   None)
    failures = np.array([], dtype=float) if failures is None else np.asarray(failures, dtype=float)
    censored = np.array([], dtype=float) if censored is None else np.asarray(censored, dtype=float)
    holdout  = np.array([], dtype=float) if holdout  is None else np.asarray(holdout,  dtype=float)
    return failures, censored, holdout


# ------------------------------------------------------------------------------
# HELPERS -- RMSE and fit-table extraction
# ------------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def _extract_ic_from_fit_table(fit_table, selected: str) -> tuple[float, float]:
    """Return (AICc, BIC) for the selected model from a fit_table (metadata only)."""
    if fit_table is None or fit_table.empty:
        return np.nan, np.nan
    row = fit_table.loc[fit_table["Distribution"] == selected]
    if row.empty:
        return np.nan, np.nan
    return float(row["AICc"].iloc[0]), float(row["BIC"].iloc[0])


# ------------------------------------------------------------------------------
# FEASIBILITY GUARDS  (identical to weibull_evaluation)
# ------------------------------------------------------------------------------

def _get_globally_allowed_models(failures: np.ndarray) -> list[str]:
    """
    Determine which Weibull models are globally eligible for CV given failure
    count and distinct failure count. Mirrors get_globally_allowed_models_for_cv
    in weibull_evaluation exactly.
    """
    failures = np.asarray(failures, dtype=float)
    n_fail   = len(failures)
    n_dist   = len(np.unique(failures))

    allowed = set(MODEL_NAMES)

    if n_dist < 3:
        allowed -= {"Weibull_3P", "Weibull_CR", "Weibull_Mixture"}
    elif n_dist < 4:
        allowed -= {"Weibull_CR", "Weibull_Mixture"}
    elif n_dist < 5:
        if n_fail < 16:
            allowed -= {"Weibull_CR", "Weibull_Mixture"}
        else:
            allowed -= {"Weibull_Mixture"}
    else:
        if n_fail < 16:
            allowed -= {"Weibull_CR", "Weibull_Mixture"}

    min_req = {"Weibull_2P": 2, "Weibull_3P": 3, "Weibull_CR": 4, "Weibull_Mixture": 5}
    allowed = {m for m in allowed if n_fail >= min_req[m]}
    return sorted(allowed)


# ------------------------------------------------------------------------------
# STAGE A -- CV fold execution and NLL aggregation
# ------------------------------------------------------------------------------

def _run_cv_folds(failures: np.ndarray, censored: np.ndarray, globally_allowed: list[str]) -> dict[str, list[float]]:
    """
    Repeated stratified k-fold cross-validation.
    Returns raw per-fold NLL scores per model.

    All logic is identical to cross_validate_weibull_models in weibull_evaluation:
      - stratification guard (each class >= n_folds members)
      - early-failure protection (earliest failure always in train)
      - per-fold feasibility (FOLD_MIN_FAIL, global allowed)
      - robust NLL: PDF clamped to [1e-50, 1e50], SF clamped to (0, 1-1e-12)
      - Weibull_3P guard: val failures below gamma -> inf NLL for that fold
    """
    status_labels = np.concatenate([np.zeros(len(failures), dtype=int), np.ones (len(censored), dtype=int)])
    all_data   = np.concatenate([failures, censored])
    n_failures = int(np.sum(status_labels == 0))
    n_censored = int(np.sum(status_labels == 1))

    # Stratification guard
    if n_censored > 0 and min(n_failures, n_censored) < CV_N_FOLDS:
        return {m: [] for m in MODEL_NAMES}   # empty -> not feasible

    # Earliest failure: always keep in training fold
    early_indices = np.array([], dtype=int)
    fail_mask     = (status_labels == 0)
    if np.sum(fail_mask) >= 1:
        first_in_fail = np.argsort(all_data[fail_mask])[:1]
        early_indices = np.where(fail_mask)[0][first_in_fail]

    skf = RepeatedStratifiedKFold(n_splits=CV_N_FOLDS, n_repeats=CV_N_REPEATS, random_state=CV_SEED)

    fold_scores = {m: [] for m in MODEL_NAMES}

    fold_args_base = dict(show_probability_plot=False, print_results=False,
                          method="MLE", CI_type="none", optimizer="best")

    for train_idx, val_idx in skf.split(all_data, status_labels):

        # Move early failures from val -> train if needed
        if len(early_indices) > 0:
            early_in_val = np.isin(early_indices, val_idx)
            if np.any(early_in_val):
                to_move   = early_indices[early_in_val]
                train_idx = np.unique(np.concatenate([train_idx, to_move]))
                val_idx   = val_idx[~np.isin(val_idx, to_move)]

        f_train = all_data[train_idx][status_labels[train_idx] == 0]
        c_train = all_data[train_idx][status_labels[train_idx] == 1]
        f_val   = all_data[val_idx  ][status_labels[val_idx  ] == 0]
        c_val   = all_data[val_idx  ][status_labels[val_idx  ] == 1]

        # No validation failures -> fold is not informative
        if len(f_val) < 1:
            for m in MODEL_NAMES:
                fold_scores[m].append(float("inf"))
            continue

        fold_args = {**fold_args_base,
                     "failures":       f_train,
                     "right_censored": c_train if len(c_train) > 0 else None
        }

        def _fit_or_none(fitter, label: str):
            if label not in globally_allowed:
                return None
            if len(f_train) < FOLD_MIN_FAIL[label]:
                return None
            try:
                res = fitter(**fold_args)
            except Exception:
                return None
            if getattr(res, "optimizer", None) is None:
                return None
            return res

        models = {"Weibull_2P":      _fit_or_none(Fit_Weibull_2P,      "Weibull_2P"),
                  "Weibull_3P":      _fit_or_none(Fit_Weibull_3P,      "Weibull_3P"),
                  "Weibull_CR":      _fit_or_none(Fit_Weibull_CR,       "Weibull_CR"),
                  "Weibull_Mixture": _fit_or_none(Fit_Weibull_Mixture,  "Weibull_Mixture")
        }

        for name in MODEL_NAMES:
            res = models.get(name)
            if res is None or not hasattr(res, "distribution"):
                fold_scores[name].append(float("inf"))
                continue

            try:
                dist = res.distribution

                # 3P guard: val samples below gamma are impossible under this model
                if (name == "Weibull_3P" and hasattr(dist, "gamma") and np.any(f_val < dist.gamma)):
                    fold_scores[name].append(float("inf"))
                    continue

                # Failure log-likelihood via PDF
                pdf_vals = dist.PDF(f_val, show_plot=False)
                pdf_vals = np.clip(pdf_vals, 1e-50, 1e50)
                ll_fail  = float(np.sum(np.log(pdf_vals)))

                # Censored log-likelihood via SF
                ll_cens = 0.0
                if len(c_val) > 0:
                    sf_vals = dist.SF(c_val, show_plot=False)
                    sf_vals = np.clip(sf_vals, 1e-50, 1.0 - 1e-12)
                    ll_cens = float(np.sum(np.log(sf_vals)))

                fold_scores[name].append(-(ll_fail + ll_cens))   # NLL

            except Exception:
                fold_scores[name].append(float("inf"))

    return fold_scores


def _aggregate_fold_scores(fold_scores: dict[str, list[float]], globally_allowed: list[str]) -> tuple[dict, dict, dict, dict, bool]:
    """
    Aggregate raw fold NLL scores to avg / std / se / n_finite and a
    cv_has_valid_models flag (True if at least one globally-allowed model
    has at least one finite fold score).
    """
    avg_nll:  dict[str, float] = {}
    std_nll:  dict[str, float] = {}
    se_nll:   dict[str, float] = {}
    n_finite: dict[str, int]   = {}
    valid_models: list[str]    = []

    for name, scores in fold_scores.items():
        finite = [s for s in scores if np.isfinite(s)]
        n      = len(finite)
        n_finite[name] = n

        if n == 0:
            avg_nll[name] = float("inf")
            std_nll[name] = float("nan")
            se_nll[name]  = float("nan")
        else:
            avg_nll[name] = float(np.mean(finite))
            std_nll[name] = float(np.std(finite, ddof=1)) if n > 1 else float("nan")
            se_nll[name]  = (float(np.std(finite, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))
            if name in globally_allowed:
                valid_models.append(name)

    cv_has_valid_models = len(valid_models) > 0
    return avg_nll, std_nll, se_nll, n_finite, cv_has_valid_models


def _cv_nll_for_dataset(failures: np.ndarray, censored: np.ndarray) -> dict:
    """
    Compute per-model avg / std / se CV NLL for one dataset.
    cv_has_valid_models mirrors the cv_used flag from compare_best_distribution.
    """
    failures = np.asarray(failures, dtype=float)
    censored = (np.asarray(censored, dtype=float) if censored is not None and len(censored) > 0 else np.array([], dtype=float))

    _empty = {"cv_has_valid_models": False,
              "avg_cv_nll": {m: float("inf") for m in MODEL_NAMES},
              "std_cv_nll": {m: float("nan") for m in MODEL_NAMES},
              "se_cv_nll":  {m: float("nan") for m in MODEL_NAMES},
              "n_finite":   {m: 0            for m in MODEL_NAMES},
    }

    if len(failures) < 2:
        return _empty

    globally_allowed = _get_globally_allowed_models(failures)
    if not globally_allowed:
        return _empty

    fold_scores = _run_cv_folds(failures, censored, globally_allowed)

    # All empty -> stratification guard triggered, CV not feasible
    if all(len(v) == 0 for v in fold_scores.values()):
        return _empty

    avg_nll, std_nll, se_nll, n_fin, cv_ok = _aggregate_fold_scores(fold_scores, globally_allowed)

    return {"cv_has_valid_models": cv_ok,
            "avg_cv_nll": avg_nll,
            "std_cv_nll": std_nll,
            "se_cv_nll":  se_nll,
            "n_finite":   n_fin
    }


def build_cv_cache(csv_files: list[str], csv_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Stage A entry point.
    Iterate over all sampled datasets, run CV once per dataset,
    write one row per (dataset x model) to output_csv.
    """
    rows: list[dict] = []
    skip: list[dict] = []

    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        csv_name = Path(csv_file).stem

        try:
            datasets = load_datasets_from_csv(csv_path)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_file}: {e}", UserWarning)
            continue

        if not datasets:
            continue

        for ds in datasets:
            seed      = ds.get("seed", np.nan)
            data_type = ds.get("data_type", "NA")
            failures, censored, holdout = _load_ds_arrays(ds)

            if len(holdout) == 0:
                skip.append({"csv_name": csv_name, "seed": seed, "reason": "no_holdout"})
                continue

            try:
                cv_result = _cv_nll_for_dataset(failures, censored)
            except Exception as e:
                warnings.warn(f"CV failed for {csv_name} seed={seed}: {e}", UserWarning)
                skip.append({"csv_name": csv_name, "seed": seed, "reason": str(e)})
                continue

            for model in MODEL_NAMES:
                rows.append({"csv_name":            csv_name,
                             "seed":                seed,
                             "data_type":           data_type,
                             "model":               model,
                             "avg_cv_nll":          cv_result["avg_cv_nll"][model],
                             "std_cv_nll":          cv_result["std_cv_nll"][model],
                             "se_cv_nll":           cv_result["se_cv_nll"][model],
                             "n_finite_folds":      cv_result["n_finite"][model],
                             "cv_has_valid_models": cv_result["cv_has_valid_models"],
                             "n_failures":          int(len(failures)),
                             "n_censored":          int(len(censored)),
                             "n_holdout":           int(len(holdout))
                })

            gc.collect()

    df_cache = pd.DataFrame(rows)
    df_skip  = pd.DataFrame(skip)

    df_cache.to_csv(output_csv, index=False)
    df_skip.to_csv(output_csv.replace(".csv", "_build_skipped.csv"), index=False)

    n_ds       = df_cache[["csv_name", "seed"]].drop_duplicates().shape[0] if not df_cache.empty else 0
    n_feasible = (df_cache[df_cache["cv_has_valid_models"] == True][["csv_name", "seed"]].drop_duplicates().shape[0]) if not df_cache.empty else 0

    print(f"[Stage A] Cache rows written     : {len(df_cache)}")
    print(f"[Stage A] Unique datasets        : {n_ds}")
    print(f"[Stage A] CV feasible            : {n_feasible}")
    print(f"[Stage A] CV NOT feasible        : {n_ds - n_feasible}  <- excluded in Stage B")
    print(f"[Stage A] Datasets skipped       : {len(df_skip)}")
    print(f"[Stage A] Saved -> {output_csv}")

    return df_cache


# ------------------------------------------------------------------------------
# STAGE B -- delta sweep from cached CV NLL
# ------------------------------------------------------------------------------

def _select_cv_winner(avg_cv_nll: dict[str, float], delta: float) -> tuple[str | None, str | None, list[str]]:
    """
    Apply delta equivalence + parsimony on pre-computed avg CV NLL.

    Identical to the final selection block in cross_validate_weibull_models:
        equiv_group = {m : avg_cv_nll[m] - best_nll <= delta}
        winner      = min(equiv_group, key=complexity)

    Returns (selected_model, numeric_best_model, equiv_group).
    """
    valid = {m: v for m, v in avg_cv_nll.items() if np.isfinite(v)}
    if not valid:
        return None, None, []

    numeric_best = min(valid, key=valid.get)
    best_nll     = valid[numeric_best]
    equiv_group  = [m for m, v in valid.items() if v - best_nll <= delta]
    selected     = min(equiv_group, key=lambda m: COMPLEXITY.get(m, np.inf))

    return selected, numeric_best, equiv_group


def _fit_selected_model(selected: str, failures: np.ndarray, censored: np.ndarray):
    """Fit the CV-selected model on the full in-sample data."""
    shared = dict(failures = failures, right_censored = censored if len(censored) > 0 else None,
                  show_probability_plot=False, print_results = False,
                  optimizer = "best"
    )

    try:
        if   selected == "Weibull_2P":      res = Fit_Weibull_2P(**shared, method="MLE")
        elif selected == "Weibull_3P":      res = Fit_Weibull_3P(**shared, method="MLE")
        elif selected == "Weibull_CR":      res = Fit_Weibull_CR(**shared)
        elif selected == "Weibull_Mixture": res = Fit_Weibull_Mixture(**shared)
        else: return None
    except Exception:
        return None

    if res is None or getattr(res, "optimizer", None) is None:
        return None
    return res


def run_delta_sweep(cv_cache_df: pd.DataFrame, csv_dir: str, deltas: list[float] = DELTA_GRID) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage B entry point.

    Key rule (mirrors original script exactly):
        Datasets where cv_has_valid_models == False are SKIPPED for ALL deltas.
        No IC fallback is used -- these datasets are excluded from the RMSE
        evaluation entirely, logged once per delta to the skipped CSV.

    Returns (df_all, df_skip, df_by_delta, df_by_delta_model, df_by_csv_delta).
    """

    # Build per-dataset lookup structures from the cache
    ds_feasibility: dict[tuple, bool] = {}
    cv_lookup:      dict[tuple, dict[str, float]] = {}

    for (csv_name, seed), grp in cv_cache_df.groupby(["csv_name", "seed"]):
        key = (csv_name, seed)
        ds_feasibility[key] = bool(grp["cv_has_valid_models"].iloc[0])
        cv_lookup[key]      = dict(zip(grp["model"], grp["avg_cv_nll"]))

    # Load all CSV files once and build in-memory dataset store
    needed_files = (cv_cache_df["csv_name"].drop_duplicates().apply(lambda n: n + ".csv").tolist())

    dataset_store:   dict[tuple, dict] = {}
    fit_table_store: dict[tuple, object] = {}   # AICc/BIC metadata only

    for csv_file in needed_files:
        csv_path = os.path.join(csv_dir, csv_file)
        csv_name = Path(csv_file).stem

        try:
            datasets = load_datasets_from_csv(csv_path)
        except Exception as e:
            warnings.warn(f"[Stage B] Failed to reload {csv_file}: {e}", UserWarning)
            continue

        if not datasets:
            continue

        for ds in datasets:
            seed      = ds.get("seed",      np.nan)
            data_type = ds.get("data_type", "NA")
            failures, censored, holdout = _load_ds_arrays(ds)

            if len(holdout) == 0:
                continue

            if data_type == "Mix":
                ds["proportion_1"] = _parse_proportion_from_filename(csv_name)

            key = (csv_name, seed)
            dataset_store[key] = {"failures":  failures,
                                  "censored":  censored,
                                  "holdout":   holdout,
                                  "data_type": data_type,
                                  "ds":        ds,
            }

            # fit_table used only to supply AICc / BIC metadata columns in output
            try:
                ft, _, _, _ = weibull_fit_best(part=csv_name, sort_by="BIC", data=ds)
                fit_table_store[key] = ft
            except Exception:
                fit_table_store[key] = None

    # Main delta sweep
    all_rows:  list[dict] = []
    skip_rows: list[dict] = []

    for key, rec in dataset_store.items():
        csv_name  = key[0]
        seed      = key[1]
        data_type = rec["data_type"]
        failures  = rec["failures"]
        censored  = rec["censored"]
        holdout   = rec["holdout"]
        ds        = rec["ds"]

        cv_feasible = ds_feasibility.get(key, False)

        # Gate: exclude CV-infeasible datasets from ALL deltas.
        # Mirrors: if SORT_BY == "CV" and not cv_used: skipped_rows.append(...); continue
        if not cv_feasible:
            for delta in deltas:
                skip_rows.append({"csv_name":  csv_name,
                                  "seed":      seed,
                                  "data_type": data_type,
                                  "delta":     delta,
                                  "reason":    "CV_not_feasible"
                })
            continue

        avg_cv_nll = cv_lookup.get(key, {})

        for delta in deltas:

            # Select model using cached CV NLL + delta rule
            selected, numeric_best, equiv_group = _select_cv_winner(avg_cv_nll, delta)

            if selected is None:
                skip_rows.append({"csv_name":  csv_name,
                                  "seed":      seed,
                                  "data_type": data_type,
                                  "delta":     delta,
                                  "reason":    "no_cv_winner"
                })
                continue

            # Full in-sample fit of selected model
            try:
                res = _fit_selected_model(selected, failures, censored)
            except Exception as e:
                warnings.warn(f"Fit failed {csv_name} seed={seed} delta={delta}: {e}", UserWarning)
                skip_rows.append({"csv_name":  csv_name,
                                  "seed":      seed,
                                  "data_type": data_type,
                                  "delta":     delta,
                                  "reason":    f"fit_failed_{selected}"
                })
                continue

            if res is None:
                skip_rows.append({"csv_name":  csv_name,
                                  "seed":      seed,
                                  "data_type": data_type,
                                  "delta":     delta,
                                  "reason":    f"optimizer_failed_{selected}"
                })
                continue

            if not hasattr(res, "distribution"):
                continue

            dist   = res.distribution
            t_eval = np.asarray(holdout, dtype=float)

            # Predict CDF and compute RMSE
            try:
                y_pred = 1.0 - dist.SF(t_eval, show_plot=False)
                y_true = _get_true_cdf(ds, t_eval)
            except Exception as e:
                warnings.warn(f"CDF eval failed {csv_name} seed={seed} delta={delta}: {e}", UserWarning)
                continue

            if y_true is None:
                continue

            rmse_val = _rmse(y_true, y_pred)
            if not np.isfinite(rmse_val):
                continue

            # AICc / BIC: metadata only, not used for selection
            aicc, bic = _extract_ic_from_fit_table(fit_table_store.get(key), selected)

            all_rows.append({"csv_name":           csv_name,
                             "seed":               seed,
                             "data_type":          data_type,
                             "delta":              delta,
                             "selected_model":     selected,
                             "numeric_best_model": numeric_best,
                             "equivalent_models":  ";".join(equiv_group),
                             "rmse":               rmse_val,
                             "n_holdout":          int(len(t_eval)),
                             "n_failures":         int(len(failures)),
                             "n_censored":         int(len(censored)),
                             "fit_AICc":           aicc,
                             "fit_BIC":            bic
            })

            del res
            gc.collect()

    # Aggregate RMSE by delta
    df_all  = pd.DataFrame(all_rows)
    df_skip = pd.DataFrame(skip_rows)

    _agg = dict(mean_rmse   = ("rmse", "mean"),
                std_rmse    = ("rmse", "std"),
                median_rmse = ("rmse", "median"),
                min_rmse    = ("rmse", "min"),
                max_rmse    = ("rmse", "max"),
                n_valid     = ("rmse", "size")
    )

    if df_all.empty:
        df_by_delta       = pd.DataFrame(columns=["delta", *_agg])
        df_by_delta_model = pd.DataFrame(columns=["delta", "selected_model", *_agg])
        df_by_csv_delta   = pd.DataFrame(columns=["csv_name", "delta", *_agg])
    else:
        df_by_delta = (df_all.groupby("delta", as_index=False).agg(**_agg).sort_values("delta"))
        df_by_delta_model = (df_all.groupby(["delta", "selected_model"], as_index=False).agg(**_agg).sort_values(["delta", "selected_model"]))
        df_by_csv_delta = (df_all.groupby(["csv_name", "delta"], as_index=False).agg(**_agg).sort_values(["csv_name", "delta"]))

    return df_all, df_skip, df_by_delta, df_by_delta_model, df_by_csv_delta


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

def main() -> None:
    # Reproducible file sampling -- identical RNG to original script
    sampled_files = _sample_csv_files(CSV_DIR)
    pd.DataFrame({"csv_file": sampled_files}).to_csv(os.path.join(OUTPUT_DIR, "sampled_filenames.csv"), index=False)
    print(f"Sampled {len(sampled_files)} files across distribution types.")

    # Stage A
    if MODE == "build_cache":
        print("\n=== Stage A: Building CV cache ===")
        build_cv_cache(csv_files = sampled_files, csv_dir = CSV_DIR, output_csv = CV_CACHE_CSV)

    # Stage B
    elif MODE == "delta_sweep":
        print("\n=== Stage B: Delta sweep from CV cache ===")

        if not os.path.isfile(CV_CACHE_CSV):
            raise FileNotFoundError(f"CV cache not found: {CV_CACHE_CSV}\nRun with MODE='build_cache' first.")

        cv_cache_df = pd.read_csv(CV_CACHE_CSV)
        n_total    = cv_cache_df[["csv_name", "seed"]].drop_duplicates().shape[0]
        n_feasible = (cv_cache_df[cv_cache_df["cv_has_valid_models"] == True][["csv_name", "seed"]].drop_duplicates().shape[0])

        print(f"Loaded cache : {len(cv_cache_df)} rows | {n_total} unique datasets | {n_feasible} CV-feasible | "
              f"{n_total - n_feasible} excluded (CV_not_feasible)")

        df_all, df_skip, df_by_delta, df_by_delta_model, df_by_csv_delta = run_delta_sweep(cv_cache_df = cv_cache_df, csv_dir = CSV_DIR, deltas = DELTA_GRID)

        # Print best delta summary
        if not df_by_delta.empty:
            best = df_by_delta.loc[df_by_delta["mean_rmse"].idxmin()]

            print(f"\n>>> Best delta = {best['delta']}  (mean RMSE = {best['mean_rmse']:.6f},  n_valid = {int(best['n_valid'])})")
            print("\nDelta summary (all deltas):")
            print(df_by_delta.to_string(index=False))

        # Save -- same file names as original script
        paths = {"all_results":    os.path.join(OUTPUT_DIR, "DeltaCV_Tuning_all_results.csv"),
                 "by_delta":       os.path.join(OUTPUT_DIR, "DeltaCV_Tuning_by_delta.csv"),
                 "by_delta_model": os.path.join(OUTPUT_DIR, "DeltaCV_Tuning_by_delta_and_model.csv"),
                 "by_csv_delta":   os.path.join(OUTPUT_DIR, "DeltaCV_Tuning_by_csv_and_delta.csv"),
                 "skipped":        os.path.join(OUTPUT_DIR, "DeltaCV_Tuning_skipped.csv"),
                 }

        df_all.to_csv(paths["all_results"], index=False)
        df_by_delta.to_csv(paths["by_delta"], index=False)
        df_by_delta_model.to_csv(paths["by_delta_model"], index=False)
        df_by_csv_delta.to_csv(paths["by_csv_delta"], index=False)
        df_skip.to_csv(paths["skipped"], index=False)

        for label, path in paths.items():
            print(f"Saved [{label:15s}] -> {path}")

        if df_skip.empty or "reason" not in df_skip.columns:
            n_cv_skip = 0
            n_other = 0
        else:
            n_cv_skip = len(df_skip[df_skip["reason"] == "CV_not_feasible"])
            n_other   = len(df_skip[df_skip["reason"] != "CV_not_feasible"])
        n_deltas  = len(DELTA_GRID)
        print(f"\nValid runs                  : {len(df_all)}")
        print(f"Skipped (CV_not_feasible)   : {n_cv_skip}  ({n_cv_skip // max(n_deltas, 1)} datasets x {n_deltas} deltas)")
        print(f"Skipped (other reasons)     : {n_other}")
        print(f"Delta summary rows          : {len(df_by_delta)}")
        print(f"Delta-model summary rows    : {len(df_by_delta_model)}")
        print(f"CSV-delta summary rows      : {len(df_by_csv_delta)}")

    else:
        raise ValueError(f"Unknown MODE='{MODE}'. Use 'build_cache' or 'delta_sweep'.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()

    print(f'Calculation time for {MODE} and {SAMPLES_PER_TYPE} samples per model type: {(end_time - start_time) / 60} minutes')