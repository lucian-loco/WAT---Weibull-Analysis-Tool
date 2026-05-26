#!/usr/bin/python3
import numpy as np
import pandas as pd
# from scipy import stats
from utils import DataError, ThresholdError, NoCacheError
from sklearn.model_selection import RepeatedStratifiedKFold
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P, Fit_Weibull_CR, Fit_Weibull_Mixture


# ToDo: Tune the delta factor
# ToDo: Make the feedback messages passing upwards for the webtool in the end!
def compare_best_distribution(df: pd.DataFrame, sort_by: str, part: str, data=None, ic_fallback: str = 'BIC', delta: float = 0.1):
    """
    Central model selection.

    sort_by:
      - 'AICc' or 'BIC' → use pure information-criterion selection.
      - 'CV'            → try cross-validation; if not feasible, fall back to IC using ic_fallback.

    Behavior:
      - Always computes ΔAICc / ΔBIC and strong-support sets (Δ<2).
      - If sort_by == 'CV' AND data is provided AND CV is feasible:
          * Calls cross_validate_weibull_models(failures, censored)
          * Uses cv_parsimonious_winner as the winner
          * Warns if CV winner is NOT in strong-support set of AICc/BIC (Δ>=2)
      - Otherwise:
          * Uses the original IC rule:
              - If AICc and BIC winners agree → that distribution.
              - Else → use ic_fallback column ('AICc' or 'BIC').
    """
    df = df.reset_index(drop=True).copy()

    required_cols = {'Distribution', 'AICc', 'BIC'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"compare_best_distribution: missing columns {missing} in results for part '{part}'")

    # 1) Basic IC winners (as before)
    best_aicc = df.at[df['AICc'].idxmin(), 'Distribution']
    best_bic = df.at[df['BIC'].idxmin(), 'Distribution']

    # 2) Compute ΔAICc / ΔBIC
    min_aicc = df['AICc'].min()
    min_bic = df['BIC'].min()

    df['delta_AICc'] = df['AICc'] - min_aicc
    df['delta_BIC']  = df['BIC']  - min_bic

    # Strong-support sets (Δ<2)
    aicc_strong = set(df.loc[df['delta_AICc'] < 2.0, 'Distribution'])
    bic_strong  = set(df.loc[df['delta_BIC']  < 2.0, 'Distribution'])
    strong_both = aicc_strong & bic_strong

    complexity = {"Weibull_2P": 2,
                  "Weibull_3P": 3,
                  "Weibull_CR": 4,
                  "Weibull_Mixture": 5}

    # Helper: IC selection with parsimony (Δ <= 2)
    def _select_by_ic(fallback_col: str):
        """
        IC-based selection with Δ<=2 and parsimony.

        Reports:
          - primary IC column used (AICc/BIC)
          - numeric IC best (without parsimony)
          - final winner with parsimony
          - whether parsimony changed the winner
        """
        # 1) Decide primary IC column and numeric IC winner
        if best_aicc == best_bic:
            primary_col = "BIC"
            ic_numeric_winner = best_bic
        else:
            if fallback_col not in ("AICc", "BIC"):
                raise KeyError(f"compare_best_distribution: fallback_col='{fallback_col}' not in {{'AICc','BIC'}} for part '{part}'")
            primary_col = fallback_col
            ic_numeric_winner = df.at[df[primary_col].idxmin(), 'Distribution']

        # 2) Compute deltas for primary IC
        min_ic = df[primary_col].min()
        delta_values = df[primary_col] - min_ic

        # 3) Support set: models within Δ <= 2
        support_mask = (delta_values <= 2.0)
        candidate_dists = df.loc[support_mask, 'Distribution'].tolist()

        if not candidate_dists:
            # This should not happen, but fall back to numeric best
            final_winner = ic_numeric_winner
        else:
            # 4) Parsimony: choose simplest model among candidates
            final_winner = min(candidate_dists, key=lambda d: complexity.get(d, np.inf))

        # 5) Feedback message
        parsimony_changed = (final_winner != ic_numeric_winner)

        if best_aicc == best_bic:
            # AICc and BIC agree numerically
            if parsimony_changed:
                print(f'[{part}]: AICc and BIC agree on numeric best → {ic_numeric_winner}, '
                      f'but Δ{primary_col}<=2 and parsimony selected simpler model → {final_winner}.')
            else:
                print(f'[{part}]: AICc and BIC agree on numeric best → {ic_numeric_winner}, '
                      f'and parsimony kept the same model.')
        else:
            # AICc and BIC disagree → we used primary_col as the IC basis
            if parsimony_changed:
                print(f'[{part}]: ⚠ AICc → {best_aicc} vs BIC → {best_bic}: disagreement, '
                      f'using "{primary_col}". Numeric {primary_col} best → {ic_numeric_winner}, '
                      f'but Δ{primary_col}<=2 and parsimony selected simpler model → {final_winner}.')
            else:
                print(f'[{part}]: ⚠ AICc → {best_aicc} vs BIC → {best_bic}: disagreement, '
                      f'using "{primary_col}". Numeric {primary_col} best → {ic_numeric_winner}, '
                      f'parsimony did not change the winner.')

        return final_winner

    # 3) Decide mode based on sort_by
    if sort_by == "CV":
        # Try CV; if not feasible, use ic_fallback (AICc/BIC)
        if data is not None:
            failures = np.asarray(data['failures'], dtype=float)
            censored = np.asarray(data['suspensions'], dtype=float) if data['suspensions'] is not None else None

            cv_results = cross_validate_weibull_models(part=part, failures=failures, censored=censored, seed=42, n_folds=5, n_repeats=5, delta=delta)

            if cv_results.get("cv_has_valid_models", False):
                winner_cv = cv_results.get("cv_parsimonious_winner", None)
                if winner_cv is not None:
                    if winner_cv not in strong_both:
                        print(f'[{part}]: ⚠ CV winner "{winner_cv}" is NOT in strong-support set '
                              f'of AICc and BIC (ΔAICc / ΔBIC ≥ 2). Please check the fit carefully.')
                    return winner_cv
                else:
                    # If data is None or CV not feasible → fall back
                    print(f'[{part}]: ⚠ CV fallback → {ic_fallback}: CV ran but returned no parsimonious winner.')
            else:
                print(f'[{part}]: ⚠ CV fallback → {ic_fallback}: CV ran but no model had finite avg NLL '
                      f'(all models infeasible or all folds failed).')
        else:
            print(f'[{part}]: ⚠ CV fallback → {ic_fallback}: no raw data provided (data=None).')

        return _select_by_ic(ic_fallback)

    else:
        # Pure IC mode: sort_by is 'AICc' or 'BIC'
        fallback_col = sort_by if sort_by in ("AICc", "BIC") else ic_fallback

        return _select_by_ic(fallback_col)


def get_globally_allowed_models_for_cv(failures: np.ndarray) -> list[str]:
    failures = np.asarray(failures, dtype=float)
    failuresize = len(failures)
    distinctfailurecount = len(np.unique(failures))

    candidates = ['Weibull_2P', 'Weibull_3P', 'Weibull_CR', 'Weibull_Mixture']

    # Start by assuming all four are possible
    allowed = set(candidates)

    # Mirror the logic of weibull_fit_best for 3P/CR/Mixture
    if distinctfailurecount < 3:
        allowed -= {'Weibull_3P', 'Weibull_CR', 'Weibull_Mixture'}
    elif distinctfailurecount < 4:
        allowed -= {'Weibull_CR', 'Weibull_Mixture'}
    elif distinctfailurecount < 5:
        if failuresize < 16:
            allowed -= {'Weibull_CR', 'Weibull_Mixture'}
        else:
            allowed -= {'Weibull_Mixture'}
    else:
        if failuresize < 16:
            allowed -= {'Weibull_CR', 'Weibull_Mixture'}
        # else: keep all

    # Technical minima: 2P needs >=2 failures, 3P>=3, CR>=4, Mixture>=5
    min_fail = {'Weibull_2P': 2,
                'Weibull_3P': 3,
                'Weibull_CR': 4,
                'Weibull_Mixture': 5}
    allowed = {m for m in allowed if failuresize >= min_fail[m]}

    return sorted(allowed)


def cross_validate_weibull_models(part, failures: np.ndarray, censored: np.ndarray | None, seed: int = 42, n_folds: int = 5, n_repeats: int = 5, delta: float = 0.1) -> dict:
    """
    Stratified K-fold CV on Weibull 2P, 3P, Competing Risk, Mixture.

    Per-model minimum failures are enforced globally and per fold:
      - 2P:  >= 4 failures (global) and >= 2 failures in train fold
      - 3P:  >= 4 failures (global) and >= 3 failures in train fold
      - CR:  >= 16 failures (global) and >= 4 failures in train fold
      - Mix: >= 16 failures (global) and >= 5 failures in train fold

    Returns:
      - avg_cv_nll: dict[model_name -> avg NLL or inf]
      - cv_numeric_winner: best model by avg NLL (among feasible ones) or None
      - cv_equivalent_models: list of equivalent models (Nadeau&Bengio + Bonferroni) among feasible ones
      - cv_parsimonious_winner: simplest model in equivalent group or None
      - cv_has_valid_models: bool indicating if at least one model had finite avg NLL
    """
    failures = np.asarray(failures, dtype=float)
    censored = np.asarray(censored, dtype=float) if censored is not None and len(censored) > 0 else np.array([], dtype=float)

    n_fail = len(failures)
    if n_fail < 2:
        # No model is feasible
        return {"avg_cv_nll": {},
                "std_cv_nll": {},
                "se_cv_nll": {},
                "cv_numeric_winner": None,
                "cv_equivalent_models": [],
                "cv_parsimonious_winner": None,
                "cv_has_valid_models": False}

    # ------------------------------------------------------------------
    # 0. Global feasibility per model
    # ------------------------------------------------------------------
    globally_allowed = get_globally_allowed_models_for_cv(failures=failures)

    if not globally_allowed:
        # CV not meaningful (no candidate model has enough failures)
        all_model_names = ['Weibull_2P', 'Weibull_3P', 'Weibull_CR', 'Weibull_Mixture']

        return {"avg_cv_nll": {name: float("inf") for name in all_model_names},
                "cv_numeric_winner": None,
                "cv_equivalent_models": [],
                "cv_parsimonious_winner": None,
                "cv_has_valid_models": False}

    # status: 0 = failure, 1 = censored
    status_labels = np.concatenate([np.zeros(len(failures), dtype=int), np.ones(len(censored), dtype=int)])
    all_train_data = np.concatenate([failures, censored])

    # Feasibility guard: each class (failures and censored) must have at least n_folds members
    n_failures = int(np.sum(status_labels == 0))
    n_censored = int(np.sum(status_labels == 1))
    min_class_size = min(n_failures, n_censored) if n_censored > 0 else n_failures

    if n_censored > 0 and min_class_size < n_folds:
        return {"avg_cv_nll": {},
                "std_cv_nll": {},
                "se_cv_nll": {},
                "cv_numeric_winner": None,
                "cv_equivalent_models": [],
                "cv_parsimonious_winner": None,
                "cv_has_valid_models": False}

    # ------------------------------------------------------------------
    # 1. Early-failure fix (same idea as in run_single_simulation)
    # ------------------------------------------------------------------
    early_indices: list[int] = []
    early_fail_mask = (status_labels == 0)
    if np.sum(early_fail_mask) >= 1:
        early_sorted = np.argsort(all_train_data[early_fail_mask])[:1]
        # Find the true indices in all_train_data array
        early_indices = np.where(early_fail_mask)[0][early_sorted]

    # ------------------------------------------------------------------
    # 2. Stratified K-Fold
    # ------------------------------------------------------------------
    skf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)

    model_names = ['Weibull_2P', 'Weibull_3P', 'Weibull_CR', 'Weibull_Mixture']
    cv_results = {name: [] for name in model_names}
    mle_fail_flags = {name: False for name in model_names}

    # Fold-level minimum failures per model
    fold_min_fail = {"Weibull_2P": 2,
                     "Weibull_3P": 3,
                     "Weibull_CR": 4,
                     "Weibull_Mixture": 5}

    # ------------------------------------------------------------------
    # 3. Fold loop
    # ------------------------------------------------------------------
    for train_idx, val_idx in skf.split(all_train_data, status_labels):

        # Early failures always in training
        if len(early_indices) > 0:
            # Check whether early failures are in validation set (not wanted)
            early_in_val = np.isin(early_indices, val_idx)
            if np.any(early_in_val):
                # Catch the "wrong" indices
                indices_to_move = early_indices[early_in_val]
                # Add them to training set
                train_idx = np.unique(np.concatenate([train_idx, indices_to_move]))
                # Remove them of validation set
                val_idx = val_idx[~np.isin(val_idx, indices_to_move)]

        fold_data_train = all_train_data[train_idx]
        fold_status_train = status_labels[train_idx]
        fold_data_val = all_train_data[val_idx]
        fold_status_val = status_labels[val_idx]

        f_train = fold_data_train[fold_status_train == 0]
        c_train = fold_data_train[fold_status_train == 1]
        f_val = fold_data_val[fold_status_val == 0]
        c_val = fold_data_val[fold_status_val == 1]

        # If no validation failures, the fold is not informative for model ranking
        if len(f_val) < 1:
            for name in model_names:
                cv_results[name].append(float("inf"))
            continue

        # ------------------------------------------------------------------
        # 3a. Fit models (only those allowed globally and with enough train failures)
        # ------------------------------------------------------------------
        fold_args = dict(failures=f_train,
                         right_censored=c_train if len(c_train) > 0 else None,
                         show_probability_plot=False,
                         print_results=False,
                         method="MLE",
                         CI_type="none",
                         optimizer="best")

        models = {}

        def _fit_or_none(fitter, label: str):
            # Global feasibility
            if label not in globally_allowed:
                return None
            # Fold-level feasibility
            if len(f_train) < fold_min_fail[label]:
                return None
            try:
                res = fitter(**fold_args)
            except Exception:
                mle_fail_flags[label] = True
                return None
            if getattr(res, "optimizer", None) is None:
                mle_fail_flags[label] = True
                return None
            return res

        models['Weibull_2P'] = _fit_or_none(Fit_Weibull_2P, 'Weibull_2P')
        models['Weibull_3P'] = _fit_or_none(Fit_Weibull_3P, 'Weibull_3P')
        models['Weibull_CR'] = _fit_or_none(Fit_Weibull_CR, 'Weibull_CR')
        models['Weibull_Mixture'] = _fit_or_none(Fit_Weibull_Mixture, 'Weibull_Mixture')

        # ------------------------------------------------------------------
        # 3b. Compute robust NLL for validation
        # ------------------------------------------------------------------
        for name in model_names:
            model = models.get(name)
            if model is not None and hasattr(model, "distribution"):
                try:
                    dist = model.distribution

                    # 3P guard
                    if name == 'Weibull_3P' and hasattr(dist, 'gamma') and np.any(f_val < dist.gamma):
                        cv_results[name].append(float("inf"))
                        continue

                    pdf_vals = dist.PDF(f_val, show_plot=False)
                    pdf_vals = np.where(pdf_vals <= 0, 1e-50, pdf_vals)
                    pdf_vals = np.where(pdf_vals > 1e50, 1e50, pdf_vals)
                    log_lik_failures = np.sum(np.log(pdf_vals))

                    if len(c_val) > 0:
                        sf_vals = dist.SF(c_val, show_plot=False)
                        sf_vals = np.where(sf_vals <= 0, 1e-50, sf_vals)
                        sf_vals = np.where(sf_vals >= 1, 1 - 1e-12, sf_vals)
                        log_lik_censored = np.sum(np.log(sf_vals))
                    else:
                        log_lik_censored = 0.0

                    total_log_lik = log_lik_failures + log_lik_censored
                    cv_results[name].append(-total_log_lik)
                except Exception:
                    cv_results[name].append(float("inf"))
            else:
                # Model not feasible or failed to fit in this fold
                cv_results[name].append(float("inf"))

    # ------------------------------------------------------------------
    # 4. Aggregate per-model results and check feasibility
    # ------------------------------------------------------------------
    avg_cv_nll = {}
    std_cv_nll = {}
    se_cv_nll = {}
    valid_models = []

    for name, scores in cv_results.items():
        finite_scores = [s for s in scores if np.isfinite(s)]
        if len(finite_scores) == 0:
            avg_cv_nll[name] = float("inf")
            std_cv_nll[name] = float("nan")
            se_cv_nll[name] = float("nan")
        else:
            avg_cv_nll[name] = float(np.mean(finite_scores))
            std_cv_nll[name] = float(np.std(finite_scores, ddof=1)) if len(finite_scores) > 1 else float("nan")
            se_cv_nll[name] = float(np.std(finite_scores, ddof=1) / np.sqrt(len(finite_scores))) if len(finite_scores) > 1 else float("nan")
            # Only models that are globally allowed and have at least one finite score
            if name in globally_allowed:
                valid_models.append(name)

    if len(valid_models) == 0:
        # None of the models had usable CV scores → CV not feasible
        return {"avg_cv_nll": avg_cv_nll,
                "std_cv_nll": std_cv_nll,
                "se_cv_nll": se_cv_nll,
                "cv_numeric_winner": None,
                "cv_equivalent_models": [],
                "cv_parsimonious_winner": None,
                "cv_has_valid_models": False}

    # numeric winner among valid models
    best_name = min(valid_models, key=lambda m: avg_cv_nll[m])

    # Approach with delta on NLL
    best_nll = avg_cv_nll[best_name]

    equivalent_group = [m for m in valid_models if avg_cv_nll[m] - best_nll <= delta]

    # t-test: significance check to apply parsimony --> too hard, old state ********************************************
    # best_scores = np.array([s for s in cv_results[best_name] if np.isfinite(s)])
    #
    # # Bonferroni-Correction
    # alpha = 0.1
    # n_comparisons = len(valid_models) - 1
    # # n_comparisons = 4 - 1
    # bonferroni_threshold = alpha / n_comparisons #if n_comparisons > 0 else 1.0
    #
    # equivalent_group = [best_name]
    #
    # for model_name in valid_models:
    #     if model_name == best_name:
    #         continue
    #
    #     model_scores = np.array([s for s in cv_results[model_name] if np.isfinite(s)])
    #     n_total = min(len(best_scores), len(model_scores))
    #     if n_total < 2:
    #         # Not enough paired scores to compare
    #         continue
    #
    #     diffs = model_scores[:n_total] - best_scores[:n_total]
    #
    #     # Nadeau & Bengio correction
    #     n_k = n_folds * n_repeats
    #     var_diff = np.var(diffs, ddof=1)
    #     rho = 1 / (n_k - 1)
    #     correction_factor = (1 / n_total) + rho
    #     standard_error = np.sqrt(correction_factor * var_diff)
    #
    #     if standard_error == 0:
    #         t_stat = 0.0
    #     else:
    #         t_stat = np.mean(diffs) / standard_error
    #
    #     df = n_k - 1
    #     p_val = stats.t.sf(np.abs(t_stat), df) * 2
    #
    #     if p_val >= bonferroni_threshold:
    #         equivalent_group.append(model_name)
    # ******************************************************************************************************************

    complexity = {"Weibull_2P": 2,
                  "Weibull_3P": 3,
                  "Weibull_CR": 4,
                  "Weibull_Mixture": 5}

    final_model = min(equivalent_group, key=lambda m: complexity.get(m, np.inf))

    # Feedback: did parsimony change the CV winner?
    parsimony_changed = (final_model != best_name)
    if parsimony_changed:
        # print(f'[{part}]: CV - numeric best (avg NLL) → {best_name}, '
        #       f'but Nadeau & Bengio equivalence + parsimony selected simpler model → {final_model}.')
        print(f'[{part}]: CV - numeric best (avg NLL) → {best_name} ({best_nll:.4f}), '
              f'but delta≤{delta} equivalence + parsimony selected simpler model → {final_model}.')
    else:
        # print(f'[{part}]: CV - numeric best (avg NLL) → {best_name}, '
        #       f'parsimony (within equivalence group) kept the same model.')
        print(f'[{part}]: CV - numeric best (avg NLL) → {best_name} ({best_nll:.4f}), '
              f'parsimony kept the same model.')

    return {"avg_cv_nll": avg_cv_nll,
            "std_cv_nll": std_cv_nll,
            "se_cv_nll": se_cv_nll,
            "cv_numeric_winner": best_name,
            "cv_equivalent_models": equivalent_group,
            "cv_parsimonious_winner": final_model,
            "cv_has_valid_models": True}

