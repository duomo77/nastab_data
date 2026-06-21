import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdrobust import rdrobust

# ==============================================================================
# 0. 기본 설정 (상수 정의)
# ==============================================================================
WAVES = [11, 12, 13, 14, 15, 16, 17]
WAVE_YEAR = {
    11: 2018, 12: 2019, 13: 2020, 14: 2021,
    15: 2022, 16: 2023, 17: 2024
}

OUTCOMES = [
    "exp_tot", "exp_con", "exp_ncn", "exp_hou", "exp_fod",
    "exp_cul", "exp_clo", "exp_tou", "exp_edu", "exp_hel", "exp_ins"
]

BANDWIDTHS = [4, 5, 6]
POLY_ORDERS = [1, 2]
CHILD_IDS = range(1, 10)

DATA_DIR = "./data"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. 데이터 로드 (웨이브별 .dta)
# ==============================================================================
def load_wave_data(data_dir):
    """Load h{wave}.dta files and concatenate with wave column."""
    frames = []
    for wave in WAVES:
        fpath = os.path.join(data_dir, f"h{wave}.dta")
        if not os.path.exists(fpath):
            print(f"[WARN] File not found, skipping: {fpath}")
            continue
        df = pd.read_stata(fpath)
        df["wave"] = wave
        frames.append(df)
        print(f"[INFO] Loaded {fpath} ({len(df)} rows)")

    if not frames:
        raise FileNotFoundError(f"No .dta files found in {data_dir}")

    pooled = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Total pooled rows: {len(pooled)}")
    return pooled

# ==============================================================================
# 2. 자녀 Pooling (wide → long) + 나이(개월) 계산
# ==============================================================================
def pool_children(df):
    """
    Reshape each household row into up to 9 child rows.
    Compute age_month = (survey_year - byr) * 12 - bmn.
    """
    all_child_rows = []

    for wave in WAVES:
        wave_df = df[df["wave"] == wave].copy()
        if wave_df.empty:
            continue

        survey_year = WAVE_YEAR[wave]

        for child_id in CHILD_IDS:
            byr_col = f"h{wave}byr{child_id:02d}"
            bmn_col = f"h{wave}bmn{child_id:02d}"

            if byr_col not in wave_df.columns or bmn_col not in wave_df.columns:
                print(f"[WARN] Missing child columns: {byr_col}, {bmn_col} — skipping child {child_id} in wave {wave}")
                continue

            child_rows = wave_df[["wave"]].copy()
            child_rows["child_id"] = child_id
            child_rows["byr"] = wave_df[byr_col]
            child_rows["bmn"] = wave_df[bmn_col]

            # age_month = (survey_year - byr) * 12 - bmn
            child_rows["age_month"] = (survey_year - child_rows["byr"]) * 12 - child_rows["bmn"]

            all_child_rows.append(child_rows)

    if not all_child_rows:
        raise ValueError("No child rows extracted. Check column naming patterns.")

    child_df = pd.concat(all_child_rows, ignore_index=True)
    print(f"[INFO] Pooled {len(child_df)} child rows from {len(df)} household rows")
    return child_df

# ==============================================================================
# 3. Cutoff 할당 + Running Variable
# ==============================================================================
def compute_cutoff(df):
    """
    Assign survey_year and cutoff based on WAVE_YEAR:
      survey_year >= 2022 → 96
      survey_year >= 2019 → 84
      else                → 72
    """
    df = df.copy()
    df["survey_year"] = df["wave"].map(WAVE_YEAR)

    def _cutoff(year):
        if year >= 2022:
            return 96
        elif year >= 2019:
            return 84
        else:
            return 72

    df["cut_off"] = df["survey_year"].apply(_cutoff)
    df["running"] = df["age_month"] - df["cut_off"]
    print(f"[INFO] Cutoff distribution:\n{df['cut_off'].value_counts().sort_index()}")
    return df

# ==============================================================================
# 4. 결과변수 표준화 (h{wave}exp_tot → exp_tot)
# ==============================================================================
def create_outcomes(df):
    """Extract wave-specific outcome columns into standardized names."""
    for outcome in OUTCOMES:
        df[outcome] = np.nan
        for wave in WAVES:
            col = f"h{wave}{outcome}"
            if col in df.columns:
                mask = df["wave"] == wave
                df.loc[mask, outcome] = df.loc[mask, col]
    return df

# ==============================================================================
# 5. 공변량 생성 (inc_a, edu01, capital_area)
# ==============================================================================
def create_covariates(df):
    # 소득
    df["inc_a"] = np.nan
    for wave in WAVES:
        col = f"h{wave}inc_a"
        if col in df.columns:
            mask = df["wave"] == wave
            df.loc[mask, "inc_a"] = df.loc[mask, col]

    # 교육
    df["edu01"] = np.nan
    for wave in WAVES:
        col = f"w{wave}edu01"
        if col in df.columns:
            mask = df["wave"] == wave
            df.loc[mask, "edu01"] = df.loc[mask, col]

    # 지역
    df["region_code"] = np.nan
    for wave in WAVES:
        col = f"h{wave}b10"
        if col in df.columns:
            mask = df["wave"] == wave
            df.loc[mask, "region_code"] = df.loc[mask, col]

    # 수도권 더미 (서울 11, 인천 23, 경기 31)
    capital_area_codes = [11, 23, 31]
    df["capital_area"] = df["region_code"].isin(capital_area_codes).astype(int)

    return df

# ==============================================================================
# 6. Balance Test
# ==============================================================================
def run_balance_test(df):
    balance_vars = ["inc_a", "edu01", "capital_area"]
    balance_results = []

    temp = df.dropna(subset=["age_month", "cut_off"] + balance_vars).copy()
    if temp.empty:
        print("[WARN] No data for balance test")
        return pd.DataFrame()

    temp["running"] = temp["age_month"] - temp["cut_off"]

    for var in balance_vars:
        temp_var = temp.dropna(subset=[var])
        if temp_var.empty:
            continue

        rd = rdrobust(
            y=temp_var[var].astype(float).values,
            x=temp_var["running"].astype(float).values,
            c=0,
            kernel="triangular",
            p=1
        )

        balance_results.append({
            "variable": var,
            "coef": float(rd.coef.iloc[0, 0]),
            "se": float(rd.se.iloc[0, 0]),
            "pvalue": float(rd.pv.iloc[0, 0])
        })

    return pd.DataFrame(balance_results)

# ==============================================================================
# 7. RDD 실행 (covariates + masspoints + bandwidth + polynomial order)
# ==============================================================================
def run_pooled_rd(df, outcome, bandwidth, poly_order):
    tmp = df.dropna(subset=[outcome, "running", "inc_a", "edu01", "capital_area"]).copy()
    if tmp.empty:
        return None

    y_vals = tmp[outcome].astype(float).values
    x_vals = tmp["running"].astype(float).values
    covs = tmp[["inc_a", "edu01", "capital_area"]].astype(float).values

    rd = rdrobust(
        y=y_vals,
        x=x_vals,
        c=0,
        h=bandwidth,
        kernel="triangular",
        p=poly_order,
        covs=covs,
        masspoints="adjust"
    )

    return {
        "outcome": outcome,
        "bandwidth": bandwidth,
        "polynomial_order": poly_order,
        "coef": float(rd.coef.iloc[0, 0]),
        "se": float(rd.se.iloc[0, 0]),
        "pvalue": float(rd.pv.iloc[0, 0]),
        "ci_lower": float(rd.ci.iloc[0, 0]),
        "ci_upper": float(rd.ci.iloc[0, 1]),
        "n_obs": int(len(tmp)),
        "mean_income": float(tmp["inc_a"].mean()),
        "mean_edu": float(tmp["edu01"].mean()),
        "capital_area_pct": float(tmp["capital_area"].mean() * 100),
    }

# ==============================================================================
# 8. RDD 그래프 (matplotlib 커스텀)
# ==============================================================================
def plot_rdd(df, outcome, bandwidth, poly_order, rd_result):
    tmp = df.dropna(subset=[outcome, "running"]).copy()
    if tmp.empty:
        return

    y_vals = tmp[outcome].astype(float).values
    x_vals = tmp["running"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 6))

    # Binned scatter
    n_bins = 50
    bins = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_x_centers = []
    for i in range(len(bins) - 1):
        mask = (x_vals >= bins[i]) & (x_vals < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(y_vals[mask].mean())
            bin_x_centers.append(bin_centers[i])

    ax.scatter(bin_x_centers, bin_means, alpha=0.4, s=20, color="gray", label="Binned means")

    # Fitted polynomial lines (left and right of cutoff)
    x_left = x_vals[x_vals < 0]
    x_right = x_vals[x_vals >= 0]
    y_left = y_vals[x_vals < 0]
    y_right = y_vals[x_vals >= 0]

    x_fit_left = np.linspace(x_left.min(), 0, 100)
    x_fit_right = np.linspace(0, x_right.max(), 100)

    if len(x_left) > poly_order and len(y_left) > poly_order:
        coeffs_left = np.polyfit(x_left, y_left, poly_order)
        ax.plot(x_fit_left, np.polyval(coeffs_left, x_fit_left),
                color="blue", linewidth=2, label=f"Left (p={poly_order})")

    if len(x_right) > poly_order and len(y_right) > poly_order:
        coeffs_right = np.polyfit(x_right, y_right, poly_order)
        ax.plot(x_fit_right, np.polyval(coeffs_right, x_fit_right),
                color="red", linewidth=2, label=f"Right (p={poly_order})")

    # Cutoff line
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Cutoff")

    ax.set_xlabel("Running variable (age_month - cutoff)", fontsize=12)
    ax.set_ylabel(outcome, fontsize=12)
    ax.set_title(f"RDD: {outcome}  |  h={bandwidth}, p={poly_order}  |  N={rd_result['n_obs']}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fpath = os.path.join(RESULTS_DIR, f"{outcome}_bw{bandwidth}_p{poly_order}_plot.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved plot: {fpath}")

# ==============================================================================
# 9. 결과 테이블 이미지 (matplotlib table)
# ==============================================================================
def plot_table(rd_result):
    outcome = rd_result["outcome"]
    bandwidth = rd_result["bandwidth"]
    poly_order = rd_result["polynomial_order"]

    rows = [
        ["Coefficient", f"{rd_result['coef']:.4f}"],
        ["Std. Error", f"{rd_result['se']:.4f}"],
        ["p-value", f"{rd_result['pvalue']:.4f}"],
        ["95% CI Lower", f"{rd_result['ci_lower']:.4f}"],
        ["95% CI Upper", f"{rd_result['ci_upper']:.4f}"],
        ["N", f"{rd_result['n_obs']}"],
        ["Mean Income", f"{rd_result['mean_income']:.2f}"],
        ["Mean Edu", f"{rd_result['mean_edu']:.2f}"],
        ["Capital Area %", f"{rd_result['capital_area_pct']:.1f}"],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")
    ax.set_title(f"RDD Results: {outcome}  |  h={bandwidth}, p={poly_order}", fontsize=13, pad=20)

    table = ax.table(
        cellText=[[v] for _, v in rows],
        rowLabels=[k for k, _ in rows],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Color header row differently
    for (row, col), cell in table.get_celld().items():
        if col == -1:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")

    fpath = os.path.join(RESULTS_DIR, f"{outcome}_bw{bandwidth}_p{poly_order}_table.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved table: {fpath}")

# ==============================================================================
# 실행 (Main)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Pooled RDD Analysis — Child-level, bandwidths [4,5,6], p=[1,2]")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[Step 1] Loading wave data...")
    pooled_df = load_wave_data(DATA_DIR)

    # 2. 자녀 풀링 + 나이 계산
    print("\n[Step 2] Pooling children...")
    child_df = pool_children(pooled_df)
    del pooled_df  # free memory

    # 3. Cutoff + Running Variable
    print("\n[Step 3] Computing cutoff...")
    child_df = compute_cutoff(child_df)

    # 4. 결과변수 표준화
    print("\n[Step 4] Extracting outcomes...")
    child_df = create_outcomes(child_df)

    # 5. 공변량 생성
    print("\n[Step 5] Creating covariates...")
    child_df = create_covariates(child_df)

    # 6. Balance Test
    print("\n[Step 6] Running balance test...")
    balance_df = run_balance_test(child_df)
    if not balance_df.empty:
        print(balance_df.to_string(index=False))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.set_title("Balance Test Results", fontsize=13, pad=15)
        table = ax.table(
            cellText=[
                [f"{row['coef']:.4f}", f"{row['se']:.4f}", f"{row['pvalue']:.4f}"]
                for _, row in balance_df.iterrows()
            ],
            colLabels=["Coefficient", "Std. Error", "p-value"],
            rowLabels=balance_df["variable"].tolist(),
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        balance_path = os.path.join(RESULTS_DIR, "balance_table.png")
        fig.savefig(balance_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved balance table: {balance_path}")

    # 7. RDD 실행 (outcome × bandwidth × polynomial)
    print("\n[Step 7] Running RDD...")
    all_results = []
    total = len(OUTCOMES) * len(BANDWIDTHS) * len(POLY_ORDERS)
    count = 0

    for outcome in OUTCOMES:
        for bw in BANDWIDTHS:
            for p in POLY_ORDERS:
                count += 1
                print(f"  [{count}/{total}] {outcome} | bw={bw} | p={p}")

                result = run_pooled_rd(child_df, outcome, bw, p)
                if result is None:
                    print(f"    [WARN] No data — skipped")
                    continue

                all_results.append(result)

                # Plot & table
                plot_rdd(child_df, outcome, bw, p, result)
                plot_table(result)

    # 8. 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n[INFO] Saved summary: {summary_path}")
        print("\n=== Summary ===")
        print(results_df[["outcome", "bandwidth", "polynomial_order", "coef", "se", "pvalue", "n_obs"]].to_string(index=False))
    else:
        print("\n[WARN] No RDD results generated.")

    print("\nDone.")
