

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from saas_domain import FUNDING_STAGES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "india_saas_ctc_dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CATEGORICAL_FEATURES = [
    "role_archetype", "role", "level", "city", "funding_stage",
    "target_market", "skill_category", "education_pedigree", "pricing_model",
]
NUMERIC_FEATURES = ["years_experience", "health_score"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "cash_ctc_lpa"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # --- Baseline: Linear Regression ---
    linear = LinearRegression()
    linear.fit(X_train_t, y_train)
    linear_pred = linear.predict(X_test_t)

    # --- Production: quantile HistGradientBoostingRegressor (P25/P50/P75) ---
    quantile_models = {}
    quantile_preds = {}
    for q in (0.25, 0.50, 0.75):
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.06,
            max_depth=6, random_state=42,
        )
        model.fit(X_train_t, y_train)
        quantile_models[q] = model
        quantile_preds[q] = model.predict(X_test_t)

    p50_pred = quantile_preds[0.50]

    metrics = {
        "Linear Regression (baseline)": {
            "r2": r2_score(y_test, linear_pred),
            "mae_lpa": mean_absolute_error(y_test, linear_pred),
            "mape_pct": mean_absolute_percentage_error(y_test, linear_pred) * 100,
        },
        "HistGradientBoostingRegressor (P50, production)": {
            "r2": r2_score(y_test, p50_pred),
            "mae_lpa": mean_absolute_error(y_test, p50_pred),
            "mape_pct": mean_absolute_percentage_error(y_test, p50_pred) * 100,
        },
    }
    for name, m in metrics.items():
        print(f"{name}: R2={m['r2']:.3f}  MAE={m['mae_lpa']:.2f} LPA  MAPE={m['mape_pct']:.1f}%")

    # Sanity check: P25 <= P50 <= P75 on the test set, on average
    coverage_ok = float(np.mean((quantile_preds[0.25] <= p50_pred) & (p50_pred <= quantile_preds[0.75])))
    print(f"P25<=P50<=P75 holds for {coverage_ok*100:.1f}% of test predictions")
    metrics["quantile_monotonicity_pct"] = coverage_ok * 100

    # --- 5-fold cross-validation: is the held-out R2/MAE above a stable estimate,
    # or a lucky split? Each fold refits its own preprocessor (fit only on that
    # fold's training rows) inside the Pipeline, so there's no leakage across folds. ---
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, estimator in [
        ("Linear Regression (baseline)", LinearRegression()),
        ("HistGradientBoostingRegressor (P50, production)", HistGradientBoostingRegressor(
            loss="quantile", quantile=0.50, max_iter=300, learning_rate=0.06, max_depth=6, random_state=42,
        )),
    ]:
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error"))
        r2_scores = scores["test_r2"]
        mae_scores = -scores["test_neg_mean_absolute_error"]
        cv_results[name] = {
            "r2_mean": float(r2_scores.mean()), "r2_std": float(r2_scores.std()),
            "mae_mean_lpa": float(mae_scores.mean()), "mae_std_lpa": float(mae_scores.std()),
        }
        print(f"5-fold CV  {name}: R2={r2_scores.mean():.3f}+/-{r2_scores.std():.3f}  "
              f"MAE={mae_scores.mean():.2f}+/-{mae_scores.std():.2f} LPA")
    metrics["cross_validation_5fold"] = cv_results

    # --- Face-validity check: do P50 predictions on canonical profiles fall inside
    # the benchmark ranges cited in README.md? Not a scrape-based external
    # validation (see README on why that wasn't attempted) - a sanity check against
    # the same public ranges the calibration itself was built from. ---
    canonical_checks = [
        dict(label="Chennai, Domestic, Mid, Series A SaaS (cited: Rs.7.9-8.7 LPA for Chennai engineers)",
             cited_range=(6.0, 12.0),
             row=dict(role_archetype="Product", role="Software Engineer", level="Mid", city="Chennai",
                       funding_stage="Series A", target_market="Domestic", skill_category="Core",
                       education_pedigree="Regional university", pricing_model="Subscription",
                       years_experience=3.0, health_score=55.0)),
        dict(label="Bengaluru, Cross-border, Senior, Series C+ (cited: FAANG/cross-border tier Rs.35-60 LPA)",
             cited_range=(28.0, 70.0),
             row=dict(role_archetype="Product", role="Software Engineer", level="Senior", city="Bengaluru",
                       funding_stage="Series C+", target_market="Cross-border", skill_category="GenAI / ML",
                       education_pedigree="Tier-1 institute (IIT/IIM/BITS)", pricing_model="Subscription",
                       years_experience=7.0, health_score=75.0)),
        dict(label="Domestic Seed-stage SaaS, Junior (cited IT-services-adjacent range: Rs.3.5-6 LPA)",
             cited_range=(4.0, 12.0),
             row=dict(role_archetype="GTM", role="Sales Development Rep (SDR)", level="Junior", city="Jaipur",
                       funding_stage="Seed", target_market="Domestic", skill_category="Core",
                       education_pedigree="Regional university", pricing_model="Freemium",
                       years_experience=1.0, health_score=42.0)),
    ]
    face_validity = []
    for check in canonical_checks:
        row_df = pd.DataFrame([check["row"]])[FEATURE_COLUMNS]
        pred = float(quantile_models[0.50].predict(preprocessor.transform(row_df))[0])
        lo, hi = check["cited_range"]
        in_range = lo <= pred <= hi
        face_validity.append({"label": check["label"], "predicted_p50_lpa": round(pred, 2),
                               "cited_range_lpa": check["cited_range"], "within_cited_range": in_range})
        print(f"Face-validity: {check['label']}\n  -> predicted P50={pred:.1f} LPA, cited range={check['cited_range']}, "
              f"{'OK' if in_range else 'OUT OF RANGE'}")
    metrics["face_validity_checks"] = face_validity

    # --- SHAP on the P50 model ---
    explainer = shap.TreeExplainer(quantile_models[0.50])
    sample = X_test_t[:200] if hasattr(X_test_t, "shape") else X_test_t
    shap_values = explainer.shap_values(sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = preprocessor.get_feature_names_out()
    top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda t: -t[1])[:10]
    print("\nTop SHAP feature contributions (P50 model, mean |impact|, LPA):")
    for name, val in top_features:
        print(f"  {name}: {val:.2f}")

    # --- Compa-ratio lookup: empirical median cash CTC per segment ---
    compa_lookup = (
        df.groupby(["role", "level", "city", "funding_stage"])[TARGET]
        .median()
        .to_dict()
    )
    # keys are tuples -> stringify for a portable, JSON-safe pickle
    compa_lookup = {"|".join(map(str, k)): v for k, v in compa_lookup.items()}
    global_median_ctc = float(df[TARGET].median())

    # --- Default health score per funding stage (data-driven, for when a user skips
    # the advanced SaaS-metrics block on the predictor form) ---
    stage_default_health = (
        df.drop_duplicates(subset="company_id")
        .groupby("funding_stage")["health_score"]
        .median()
        .reindex(FUNDING_STAGES.keys())
        .to_dict()
    )

    # --- Save artifacts ---
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    joblib.dump(linear, ARTIFACTS_DIR / "model_baseline_linear.pkl")
    for q, model in quantile_models.items():
        joblib.dump(model, ARTIFACTS_DIR / f"model_quantile_p{int(q*100)}.pkl")
    joblib.dump({"lookup": compa_lookup, "global_median": global_median_ctc}, ARTIFACTS_DIR / "compa_lookup.pkl")

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump({"categorical": CATEGORICAL_FEATURES, "numeric": NUMERIC_FEATURES}, f, indent=2)
    with open(ARTIFACTS_DIR / "top_shap_features.json", "w") as f:
        json.dump([{"feature": n, "mean_abs_shap_lpa": float(v)} for n, v in top_features], f, indent=2)
    with open(ARTIFACTS_DIR / "stage_default_health.json", "w") as f:
        json.dump(stage_default_health, f, indent=2)

    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
