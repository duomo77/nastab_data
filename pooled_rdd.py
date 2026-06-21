import pandas as pd
import numpy as np
from rdrobust import rdrobust

# ==============================================================================
# 0. 기본 설정 (상수 정의)
# ==============================================================================
WAVES = [1, 2, 3, 4, 5, 6, 7, 8] # 실제 사용하시는 웨이브로 수정
OUTCOMES = [
    "exp_tot", "exp_con", "exp_ncn", "exp_hou", "exp_fod", 
    "exp_cul", "exp_clo", "exp_tou", "exp_edu", "exp_hel", "exp_ins"
]

# ==============================================================================
# 1. pooled 변수 생성 함수 (공변량 생성)
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
# 2. 데이터 Pooling 및 공변량 추가
# ==============================================================================
# all_dfs는 각 웨이브별 데이터프레임 리스트라고 가정
# pooled_df = pd.concat(all_dfs, ignore_index=True)
# pooled_df = create_covariates(pooled_df)

# ==============================================================================
# 3~9. RDD 실행 함수 (공변량 통제 및 Mass Point Adjustment 적용)
# ==============================================================================
def run_pooled_rd_for_group(df, group_col, group_val, outcome_col):
    # 3. 필요한 변수만 선택 (공변량 추가)
    temp_df = df[df[group_col] == group_val][
        [
            "age_month",
            "cut_off",
            "wave",
            "inc_a",
            "edu01",
            "capital_area",
            outcome_col
        ]
    ].copy()

    # 4. y 변수 할당 
    # (주의: 요청하신 np.nan 유지 시 5번 dropna에서 데이터가 모두 지워지므로 outcome 데이터 할당)
    temp_df["y_pooled"] = temp_df[outcome_col] 

    # 5. 결측 제거 (공변량 포함)
    tmp = temp_df.dropna(
        subset=[
            "y_pooled",
            "age_month",
            "cut_off",
            "inc_a",
            "edu01",
            "capital_area"
        ]
    )

    if tmp.empty:
        return None

    # 6. Running Variable
    tmp["running"] = tmp["age_month"] - tmp["cut_off"]

    # RDD용 데이터 추출
    y_vals = tmp["y_pooled"].astype(float).values
    x_vals_for_rd = tmp["running"].astype(float).values
    cutoff = 0.0

    # 7. Covariate Matrix 생성
    covs = tmp[
        [
            "inc_a",
            "edu01",
            "capital_area"
        ]
    ].astype(float).values

    # 8. rdrobust 실행 (공변량 통제, Mass Point Adjustment)
    rd = rdrobust(
        y=y_vals,
        x=x_vals_for_rd,
        c=cutoff,
        kernel="triangular",
        p=1,
        covs=covs,
        masspoints="adjust"
    )

    # 결과 추출
    coef = float(rd.coef.iloc[0, 0])
    se = float(rd.se.iloc[0, 0])
    pvalue = float(rd.pv.iloc[0, 0])
    ci_left = float(rd.ci.iloc[0, 0])
    ci_right = float(rd.ci.iloc[0, 1])
    n_obs = len(tmp)

    # 9. 결과 저장 (기초통계 포함)
    result = {
        "group": group_val,
        "outcome": outcome_col,
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "ci_left": ci_left,
        "ci_right": ci_right,
        "n_obs": n_obs,
        "mean_income": float(tmp["inc_a"].mean()),
        "mean_edu": float(tmp["edu01"].mean()),
        "capital_area_pct": float(tmp["capital_area"].mean() * 100),
    }
    
    return result

# ==============================================================================
# 10. Balance Test (조작 가능성 및 공변량 균형 검증)
# ==============================================================================
def run_balance_test(pooled_df):
    balance_vars = ["inc_a", "edu01", "capital_area"]
    balance_results = []

    # Running variable 계산을 위해 필수 변수만 남기고 결측 제거
    temp = pooled_df.dropna(
        subset=["age_month", "cut_off"] + balance_vars
    ).copy()

    temp["running"] = temp["age_month"] - temp["cut_off"]

    for var in balance_vars:
        # 각 변수별 결측 제거
        temp_var = temp.dropna(subset=[var])
        
        # 공변량 없이 단순 RDD 실행 (계수 0 유의미한지 확인)
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
# 실행 예시 (Main)
# ==============================================================================
if __name__ == "__main__":
    # 1. 데이터 합치기 및 공변량 생성 (2번 단계)
    # pooled_df = pd.concat(all_dfs, ignore_index=True)
    # pooled_df = create_covariates(pooled_df)
    
    # 2. Balance Test 실행 (10번 단계)
    # balance_df = run_balance_test(pooled_df)
    # print("=== Balance Test Results ===")
    # print(balance_df)
    
    # 3. 그룹별/결과변수별 RDD 실행 (3~9번 단계)
    # rd_results = []
    # for outcome in OUTCOMES:
    #     res = run_pooled_rd_for_group(pooled_df, "group_col", "target_group", outcome)
    #     if res:
    #         rd_results.append(res)
    #         
    # rd_results_df = pd.DataFrame(rd_results)
    # print("=== RDD Results ===")
    # print(rd_results_df)
