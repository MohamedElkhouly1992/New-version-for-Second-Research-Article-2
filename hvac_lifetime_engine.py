"""
HVAC Lifetime Optimizer Engine
Hybrid ML/surrogate forecasting, retrofit assessment, and S3 optimization.
Author-ready research utility for severity-strategy HVAC datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

RANDOM_STATE = 42
DEFAULT_TARGETS = [
    "annual_energy_MWh",
    "annual_cost_usd",
    "annual_co2_tonne",
    "mean_COP",
    "mean_delta",
    "mean_comfort_dev",
    "occupied_discomfort_days",
]

ALIASES = {
    "energy": ["annual_energy_MWh", "energy_MWh", "energy", "annual_energy", "total_energy_MWh"],
    "cost": ["annual_cost_usd", "cost_usd", "cost", "annual_cost"],
    "co2": ["annual_co2_tonne", "co2_tonne", "co2", "annual_co2"],
    "cop": ["mean_COP", "COP", "cop", "mean_cop"],
    "delta": ["mean_delta", "delta", "degradation", "degradation_index", "mean_degradation_index"],
    "comfort": ["mean_comfort_dev", "comfort", "comfort_dev", "mean_comfort"],
    "discomfort_days": ["occupied_discomfort_days", "discomfort_days", "uncomfortable_days"],
}


def _numeric_severity_from_any(value) -> float:
    """Convert numeric or categorical severity values to a normalized 0-1 score."""
    if pd.isna(value):
        return np.nan
    # Direct numeric values first.
    try:
        v = float(value)
        if np.isfinite(v):
            return max(0.0, min(1.0, v))
    except Exception:
        pass

    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "clean": 0.00, "none": 0.00, "no_degradation": 0.00, "baseline": 0.00, "s0": 0.00,
        "very_low": 0.10, "verylow": 0.10, "low": 0.25, "mild": 0.25,
        "medium": 0.50, "moderate": 0.50, "mid": 0.50,
        "high": 0.75, "severe": 0.75,
        "critical": 0.90, "very_high": 0.90, "veryhigh": 0.90, "failure": 1.00,
    }
    if text in mapping:
        return mapping[text]

    # Extract values from labels such as severity_0.45, delta=0.7, L3, level_4.
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        try:
            v = float(nums[-1])
            # Interpret 1-5 ordinal labels as normalized severity levels.
            if 1 <= v <= 5 and not (0 <= v <= 1):
                v = (v - 1) / 4
            # Interpret percentages like 45 or 75 as 0.45 or 0.75.
            elif v > 1 and v <= 100:
                v = v / 100
            return max(0.0, min(1.0, v))
        except Exception:
            return np.nan
    return np.nan


def _safe_float(value, fallback=np.nan) -> float:
    """Safe float conversion used during long-horizon projection."""
    try:
        v = float(value)
        return v if np.isfinite(v) else fallback
    except Exception:
        sev = _numeric_severity_from_any(value)
        return sev if np.isfinite(sev) else fallback


def _prepare_severity_column(out: pd.DataFrame) -> pd.DataFrame:
    """Ensure severity is numeric while preserving the original label when supplied."""
    if "severity" in out.columns:
        original = out["severity"].copy()
        numeric = pd.to_numeric(original, errors="coerce")
        mapped = original.map(_numeric_severity_from_any)
        severity_num = numeric.fillna(mapped)
        if severity_num.isna().all():
            if "mean_delta" in out.columns:
                severity_num = pd.to_numeric(out["mean_delta"], errors="coerce")
            else:
                severity_num = pd.Series(0.0, index=out.index)
        # Keep labels for traceability only when the input was not purely numeric.
        if not pd.to_numeric(original, errors="coerce").notna().all():
            out["severity_label"] = original.astype(str)
        out["severity"] = severity_num.fillna(0.0).clip(0.0, 1.0)
    else:
        if "mean_delta" in out.columns:
            out["severity"] = pd.to_numeric(out["mean_delta"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        else:
            out["severity"] = 0.0
    return out


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    lower_map = {c.lower(): c for c in out.columns}
    for canon, names in ALIASES.items():
        preferred = names[0]
        if preferred in out.columns:
            continue
        for n in names:
            if n in out.columns:
                rename[n] = preferred
                break
            if n.lower() in lower_map:
                rename[lower_map[n.lower()]] = preferred
                break
    out = out.rename(columns=rename)
    if "strategy" not in out.columns:
        out["strategy"] = "S0"
    out = _prepare_severity_column(out)
    if "mean_delta" in out.columns:
        out["mean_delta"] = pd.to_numeric(out["mean_delta"], errors="coerce")
    if "year" not in out.columns:
        out["year"] = np.arange(len(out)) // max(1, len(out)//5) + 1
    if "climate" not in out.columns:
        out["climate"] = "uploaded"
    return out


def load_dataset(files: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        p = Path(f)
        if p.suffix.lower() in [".xlsx", ".xls"]:
            frame = pd.read_excel(p)
        else:
            frame = pd.read_csv(p)
        frame["source_file"] = p.name
        frames.append(frame)
    if not frames:
        raise ValueError("No dataset files were supplied.")
    df = pd.concat(frames, ignore_index=True)
    return canonicalize_columns(df)


def detect_targets(df: pd.DataFrame) -> List[str]:
    return [c for c in DEFAULT_TARGETS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def prepare_features(df: pd.DataFrame, targets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    drop_cols = set(targets + ["source_file"])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[targets].copy()
    # remove completely empty cols
    X = X.dropna(axis=1, how="all")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def build_model(model_name: str, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    model_name = model_name.lower()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ], remainder="drop"
    )
    if model_name == "random forest":
        base = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "extra trees":
        base = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "gradient boosting":
        base = MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE))
    elif model_name == "catboost" and CATBOOST_AVAILABLE:
        base = MultiOutputRegressor(CatBoostRegressor(iterations=700, depth=6, learning_rate=0.04, loss_function="RMSE", verbose=False, random_seed=RANDOM_STATE))
    else:
        base = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", base)])


def train_models(df: pd.DataFrame, model_names: List[str], targets: Optional[List[str]] = None, test_size: float = 0.2):
    df = canonicalize_columns(df)
    targets = targets or detect_targets(df)
    if not targets:
        raise ValueError("No recognized KPI target columns found.")
    X, y, num_cols, cat_cols = prepare_features(df, targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    results = []
    models = {}
    for name in model_names:
        t0 = time.time()
        pipe = build_model(name, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        pred = pd.DataFrame(pipe.predict(X_test), columns=targets, index=y_test.index)
        elapsed = time.time() - t0
        for target in targets:
            rmse = math.sqrt(mean_squared_error(y_test[target], pred[target]))
            mae = mean_absolute_error(y_test[target], pred[target])
            r2 = r2_score(y_test[target], pred[target]) if len(y_test) > 1 else np.nan
            results.append({"model": name, "target": target, "RMSE": rmse, "MAE": mae, "R2": r2, "train_seconds": elapsed})
        models[name] = pipe
    metrics = pd.DataFrame(results).sort_values(["target", "RMSE"])
    return models, metrics, targets, X.columns.tolist()


def choose_best_model(metrics: pd.DataFrame) -> str:
    rank = metrics.groupby("model")["RMSE"].mean().sort_values()
    return rank.index[0]


def annualize_template(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    group_cols = [c for c in ["strategy", "severity", "climate", "year"] if c in df.columns]
    if not group_cols:
        return df.copy()
    num = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in group_cols]
    cat = [c for c in df.columns if c not in num and c not in group_cols]
    annual = df.groupby(group_cols, dropna=False)[num].mean().reset_index()
    return annual


def project_future_inputs(df: pd.DataFrame, horizons: List[int], degradation_rate: float = 0.018, climate_load_growth: float = 0.004) -> pd.DataFrame:
    """Create future input rows for 10/20/30-year scenario projections.

    This function is robust to uploaded datasets where severity is supplied as
    numeric values (0-1), percentages, ordinal levels, or labels such as Low,
    Medium, High, Severe, and Critical. The projection stores severity as a
    numeric 0-1 feature because long-term degradation must be mathematically
    updated over time.
    """
    base = annualize_template(df)
    base = canonicalize_columns(base)
    rows = []

    year_series = pd.to_numeric(base.get("year", pd.Series([1])), errors="coerce")
    max_year = int(year_series.max()) if year_series.notna().any() else 1

    group_keys = [c for c in ["strategy", "severity", "climate"] if c in base.columns]
    if group_keys:
        last = base.sort_values("year").groupby(group_keys, dropna=False).tail(1)
    else:
        last = base.tail(1)

    for _, r in last.iterrows():
        initial_severity = _safe_float(r.get("severity", r.get("mean_delta", 0.0)), 0.0)
        initial_delta = _safe_float(r.get("mean_delta", initial_severity), initial_severity)
        initial_load = _safe_float(r.get("annual_thermal_hvac_MWh", np.nan), np.nan)
        for horizon in horizons:
            for yr in range(max_year + 1, max_year + int(horizon) + 1):
                rr = r.copy()
                t = yr - max_year
                rr["year"] = yr
                rr["severity"] = min(1.0, max(0.0, initial_severity + degradation_rate * t))
                if "mean_delta" in rr.index:
                    rr["mean_delta"] = min(1.0, max(0.0, initial_delta + degradation_rate * t))
                if "annual_thermal_hvac_MWh" in rr.index and np.isfinite(initial_load):
                    rr["annual_thermal_hvac_MWh"] = initial_load * (1 + climate_load_growth) ** t
                rr["forecast_horizon_years"] = int(horizon)
                rows.append(rr)
    return pd.DataFrame(rows).reset_index(drop=True)

def forecast_kpis(model: Pipeline, future_inputs: pd.DataFrame, targets: List[str], feature_columns: List[str]) -> pd.DataFrame:
    Xf = canonicalize_columns(future_inputs)
    for c in feature_columns:
        if c not in Xf.columns:
            Xf[c] = np.nan
    pred = pd.DataFrame(model.predict(Xf[feature_columns]), columns=targets)
    out = future_inputs.reset_index(drop=True).copy()
    for c in targets:
        out[f"pred_{c}"] = pred[c].values
    return out


def retrofit_analysis(forecast: pd.DataFrame, discount_rate: float = 0.08) -> pd.DataFrame:
    f = forecast.copy()
    energy_col = "pred_annual_energy_MWh" if "pred_annual_energy_MWh" in f else None
    delta_col = "pred_mean_delta" if "pred_mean_delta" in f else None
    co2_col = "pred_annual_co2_tonne" if "pred_annual_co2_tonne" in f else None
    cost_col = "pred_annual_cost_usd" if "pred_annual_cost_usd" in f else None
    cases = [
        ("R0_No_Retrofit", 0.00, 0.00, 0.0, 0),
        ("R1_Filter_Coil", 0.06, 0.08, 0.03, 8000),
        ("R2_Chiller_COP", 0.12, 0.05, 0.10, 35000),
        ("R3_AHU_Control", 0.09, 0.06, 0.06, 18000),
        ("R4_Full_S3_Retrofit", 0.18, 0.15, 0.14, 55000),
    ]
    rows = []
    groups = [c for c in ["strategy", "forecast_horizon_years"] if c in f.columns]
    for keys, g in f.groupby(groups, dropna=False) if groups else [((), f)]:
        base_energy = g[energy_col].sum() if energy_col else np.nan
        base_co2 = g[co2_col].sum() if co2_col else np.nan
        base_cost = g[cost_col].sum() if cost_col else np.nan
        base_life = int((g[delta_col] < 0.85).sum()) if delta_col else np.nan
        for name, e_sav, delta_red, opex_sav, capex in cases:
            annual_saving = base_cost * opex_sav / max(1, g["year"].nunique()) if cost_col else np.nan
            npv = -capex + sum(annual_saving / ((1 + discount_rate) ** i) for i in range(1, max(2, g["year"].nunique()+1))) if cost_col else np.nan
            simple_payback = capex / annual_saving if annual_saving and annual_saving > 0 else np.nan
            row = {"retrofit_case": name, "energy_saving_pct": e_sav*100, "degradation_reduction_pct": delta_red*100,
                   "capex_usd": capex, "NPV_usd": npv, "simple_payback_years": simple_payback,
                   "baseline_life_years_before_delta_0_85": base_life,
                   "estimated_life_extension_years": round(base_life * delta_red, 2) if not pd.isna(base_life) else np.nan,
                   "total_energy_MWh_after_retrofit": base_energy*(1-e_sav) if energy_col else np.nan,
                   "total_CO2_tonne_after_retrofit": base_co2*(1-e_sav) if co2_col else np.nan}
            if groups:
                if len(groups)==1: row[groups[0]] = keys
                else: row.update(dict(zip(groups, keys)))
            rows.append(row)
    return pd.DataFrame(rows)


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx-mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def optimize_s3(forecast: pd.DataFrame, weights: Dict[str, float], comfort_limit: float = 1.5, delta_limit: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = forecast.copy()
    # candidate table using all forecast rows; strategy comparison and feasible S3 domain
    cols = {
        "energy": "pred_annual_energy_MWh",
        "comfort": "pred_mean_comfort_dev",
        "co2": "pred_annual_co2_tonne",
        "delta": "pred_mean_delta",
        "cost": "pred_annual_cost_usd",
    }
    for k, c in cols.items():
        if c not in f.columns:
            f[c] = np.nan
    score = 0
    for k, c in cols.items():
        score = score + weights.get(k, 0.0) * normalize(f[c])
    f["objective_J"] = score
    f["feasible"] = (f[cols["comfort"]].fillna(0) <= comfort_limit) & (f[cols["delta"]].fillna(0) <= delta_limit)
    group_cols = [c for c in ["forecast_horizon_years", "year", "climate"] if c in f.columns]
    best_rows = []
    for keys, g in f.groupby(group_cols, dropna=False) if group_cols else [((), f)]:
        feasible = g[g["feasible"]]
        selected = feasible.loc[feasible["objective_J"].idxmin()] if len(feasible) else g.loc[g["objective_J"].idxmin()]
        best_rows.append(selected)
    best = pd.DataFrame(best_rows)
    strategy_summary = f.groupby([c for c in ["strategy", "forecast_horizon_years"] if c in f.columns], dropna=False).agg(
        mean_objective_J=("objective_J", "mean"), feasible_ratio=("feasible", "mean"),
        mean_energy=(cols["energy"], "mean"), mean_delta=(cols["delta"], "mean"), mean_comfort=(cols["comfort"], "mean")
    ).reset_index()
    return best, strategy_summary


def limitation_map(forecast: pd.DataFrame, comfort_limit: float = 1.5, delta_limit: float = 0.85) -> pd.DataFrame:
    f = forecast.copy()
    if "pred_mean_delta" in f.columns:
        f["_delta_for_map"] = pd.to_numeric(f["pred_mean_delta"], errors="coerce")
    elif "mean_delta" in f.columns:
        f["_delta_for_map"] = pd.to_numeric(f["mean_delta"], errors="coerce")
    else:
        f["_delta_for_map"] = pd.to_numeric(f.get("severity", pd.Series(np.nan, index=f.index)), errors="coerce")

    if "pred_mean_comfort_dev" in f.columns:
        f["_comfort_for_map"] = pd.to_numeric(f["pred_mean_comfort_dev"], errors="coerce")
    else:
        f["_comfort_for_map"] = np.nan

    f["severity_bin"] = pd.cut(
        f["_delta_for_map"],
        bins=[-0.001, .25, .45, .65, .85, 1.01],
        labels=["Very low", "Low", "Moderate", "High", "Critical"],
    )
    f["limitation_region"] = np.where(
        (f["_delta_for_map"] <= .65) & (f["_comfort_for_map"] <= comfort_limit),
        "Green: S3 recommended",
        np.where(
            (f["_delta_for_map"] <= delta_limit) & (f["_comfort_for_map"] <= comfort_limit * 1.25),
            "Yellow: S3 conditional",
            "Red: S3 not recommended",
        ),
    )
    return f.groupby(["severity_bin", "limitation_region"], dropna=False, observed=False).agg(
        cases=("limitation_region", "size"),
        mean_delta=("_delta_for_map", "mean"),
        mean_comfort=("_comfort_for_map", "mean"),
    ).reset_index()

def save_outputs(output_dir: str | Path, metrics: pd.DataFrame, forecast: pd.DataFrame, retrofit: pd.DataFrame, opt_best: pd.DataFrame, opt_summary: pd.DataFrame) -> Dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    tables = {"model_metrics": metrics, "future_forecast": forecast, "retrofit_analysis": retrofit, "s3_optimum_rows": opt_best, "strategy_optimization_summary": opt_summary}
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
    xlsx = out / "hvac_lifetime_optimizer_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    paths["excel"] = str(xlsx)
    return paths
