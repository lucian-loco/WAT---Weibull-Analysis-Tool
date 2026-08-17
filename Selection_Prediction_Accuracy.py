import pandas as pd
import re
import os



# ==== SET YOUR PATHS HERE ====
INPUT_CSV = r"C:\...\DeltaCV_Tuning\DeltaCV_Tuning_all_results.csv"
OUTPUT_DIR = r"C:\...\DeltaCV_Tuning"

TARGET_DELTA = 0.466
VALID_N = [100, 500, 1000]

CR_LEVELS_2P3P = [0.2, 0.8, 0.9]
CR_LEVELS_MIXCR = [0.1, 0.3, 0.7, 0.9]

TYPE_MAP = {
    "2P": "Weibull_2P",
    "3P": "Weibull_3P",
    "CR": "Weibull_CR",
    "Mix": "Weibull_Mixture",
}


def extract_n_from_name(name):
    m = re.search(r"_n(\d+)_", name)
    return int(m.group(1)) if m else None


def extract_cr_from_name(name):
    m = re.search(r"_cr(\d+(?:_\d+)?)(?:_|$)", name)
    if not m:
        return None
    return float(m.group(1).replace("_", "."))


def load_and_prepare(path):
    df = pd.read_csv(path)

    # Determine sample size: prefer value parsed from csv_name, fall back to
    # n_holdout + n_failures if parsing fails.
    df["n_sum"] = df["n_holdout"] + df["n_failures"]
    df["n_from_name"] = df["csv_name"].apply(extract_n_from_name)
    df["n_final"] = df["n_from_name"].fillna(df["n_sum"]).astype(int)

    # Parse censoring rate from csv_name (e.g. "..._cr0_9" -> 0.9)
    df["cr_from_name"] = df["csv_name"].apply(extract_cr_from_name)

    # Filter: only target delta and valid sample sizes
    df = df[df["delta"].round(3) == TARGET_DELTA].copy()
    df = df[df["n_final"].isin(VALID_N)].copy()

    # Filter: only valid censoring levels depending on model family
    def cr_valid(row):
        cr = row["cr_from_name"]
        if cr is None:
            return False
        if row["data_type"] in ("2P", "3P"):
            return any(abs(cr - lvl) < 1e-6 for lvl in CR_LEVELS_2P3P)
        elif row["data_type"] in ("Mix", "CR"):
            return any(abs(cr - lvl) < 1e-6 for lvl in CR_LEVELS_MIXCR)
        return False

    df = df[df.apply(cr_valid, axis=1)].copy()

    # Map true model type ("2P","3P","CR","Mix") to selected_model naming
    # ("Weibull_2P", "Weibull_3P", "Weibull_CR", "Weibull_Mixture")
    df["expected_model"] = df["data_type"].map(TYPE_MAP)
    df["correct_selection"] = df["selected_model"] == df["expected_model"]

    return df


def build_summary_table(df):
    grouped = (
        df.groupby(["data_type", "n_final", "cr_from_name"])
        .agg(
            n_rows=("correct_selection", "size"),
            selection_accuracy_pct=("correct_selection", lambda x: 100 * x.mean()),
            mean_rmse=("rmse", "mean"),
        )
        .reset_index()
        .sort_values(["data_type", "n_final", "cr_from_name"])
    )
    return grouped


if __name__ == "__main__":
    df = load_and_prepare(INPUT_CSV)
    summary = build_summary_table(df)

    print(f"Rows after filtering (delta={TARGET_DELTA}, n in {VALID_N}): {len(df)}")
    print(summary.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save flat summary table
    summary.to_csv(os.path.join(OUTPUT_DIR, "DeltaCV_summary_table.csv"), index=False)

    # Build pivot-table "grid" views: rows = (data_type, n), columns = censoring level
    pivot_acc = summary.pivot_table(
        index=["data_type", "n_final"], columns="cr_from_name", values="selection_accuracy_pct"
    )
    pivot_rmse = summary.pivot_table(
        index=["data_type", "n_final"], columns="cr_from_name", values="mean_rmse"
    )

    pivot_acc.to_csv(os.path.join(OUTPUT_DIR, "DeltaCV_selection_accuracy_pivot.csv"))
    pivot_rmse.to_csv(os.path.join(OUTPUT_DIR, "DeltaCV_mean_rmse_pivot.csv"))