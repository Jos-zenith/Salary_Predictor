

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, Response, make_response, render_template, request
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import market_signals
import saas_metrics as sm
from ctc_rules import check_ctc_compliance, compute_ctc_breakup
from saas_domain import (
    CITIES, EDUCATION_PEDIGREE, FUNDING_STAGES, PRICING_MODELS,
    ROLE_ARCHETYPES, SKILL_CATEGORIES, TARGET_MARKETS,
    experience_to_level, role_archetype_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.template_global()
def meter_pct(value, lo, hi):
    """Clamp value into [lo, hi] and return its position as 0-100, for meter/
    range-chart templates. The value can exceed the domain (e.g. an LTV:CAC of
    9 on a 0-6 scale) — the marker clamps to the track edge, the label next to
    it still shows the real number."""
    if value is None or hi == lo:
        return 0.0
    pct = (value - lo) / (hi - lo) * 100
    return max(0.0, min(100.0, pct))


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"

CATEGORICAL_FEATURES = [
    "role_archetype", "role", "level", "city", "funding_stage",
    "target_market", "skill_category", "education_pedigree", "pricing_model",
]
NUMERIC_FEATURES = ["years_experience", "health_score"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_asset(filename):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        logger.info("Loaded asset: %s", filename)
        return joblib.load(path)
    logger.warning("Asset not found: %s — predictions will fail until train_model.py is run.", filename)
    return None


preprocessor = load_asset("preprocessor.pkl")
quantile_models = {
    0.25: load_asset("model_quantile_p25.pkl"),
    0.50: load_asset("model_quantile_p50.pkl"),
    0.75: load_asset("model_quantile_p75.pkl"),
}
_compa = load_asset("compa_lookup.pkl") or {"lookup": {}, "global_median": None}
COMPA_LOOKUP = _compa["lookup"]
GLOBAL_MEDIAN_CTC = _compa["global_median"]

_stage_health_path = ARTIFACTS_DIR / "stage_default_health.json"
STAGE_DEFAULT_HEALTH = json.loads(_stage_health_path.read_text()) if _stage_health_path.exists() else {}

_metrics_path = ARTIFACTS_DIR / "metrics.json"
MODEL_METRICS = json.loads(_metrics_path.read_text()) if _metrics_path.exists() else {}

_shap_explainer = shap.TreeExplainer(quantile_models[0.50]) if quantile_models[0.50] is not None else None

market_signals.start_background_emitter(interval_seconds=7.0)


class InvalidPredictionInput(Exception):
    """Raised when submitted form data can't be turned into a prediction."""


# --------------------------------------------------------------------------
# SaaS metrics calculator
# --------------------------------------------------------------------------

def compute_saas_metrics_bundle(inputs: dict) -> dict:
    """Runs every raw company input through saas_metrics.py and returns every
    computed metric plus the health score — the course content, made interactive."""
    mrr = inputs["mrr"]
    prior_mrr = inputs["prior_mrr"]
    revenue = inputs["revenue"]
    cogs = inputs["cogs"]
    customers_lost = inputs["customers_lost"]
    customers_start = inputs["customers_start"]
    sm_spend = inputs["sm_spend"]
    new_customers = inputs["new_customers"]
    arpa = inputs["arpa"]
    cac_override = inputs.get("cac")
    ndr_input = inputs.get("ndr")
    headcount = inputs["headcount"]
    yoy_growth_pct = inputs.get("yoy_growth_pct")
    profit_margin_pct = inputs["profit_margin_pct"]
    prior_quarter_revenue = inputs["prior_quarter_revenue"]
    current_quarter_revenue = inputs["current_quarter_revenue"]
    prior_quarter_sm_spend = inputs["prior_quarter_sm_spend"]
    tam = inputs["tam"]
    saas_sam = inputs["sam"]
    som = inputs["som"]

    arr = sm.compute_arr(mrr)
    growth_rate = yoy_growth_pct if yoy_growth_pct is not None else sm.compute_mrr_growth_rate(mrr, prior_mrr) * 100
    gross_margin = sm.compute_gross_margin(revenue, cogs)
    churn_rate = sm.compute_churn_rate(customers_lost, customers_start)
    cac = cac_override if cac_override else sm.compute_cac(sm_spend, new_customers)
    ltv = sm.compute_ltv(arpa, gross_margin, churn_rate)
    ltv_cac = sm.ltv_to_cac_ratio(ltv, cac)
    ndr = ndr_input if ndr_input is not None else (1 - churn_rate) * 100
    rule40 = sm.rule_of_40(growth_rate, profit_margin_pct)
    magic_number = sm.saas_magic_number(current_quarter_revenue, prior_quarter_revenue, prior_quarter_sm_spend)
    arr_pe = sm.arr_per_employee(arr, headcount)
    market = sm.market_opportunity_tier(tam, saas_sam, som)
    health = sm.saas_health_score(rule40, ndr, ltv_cac, arr_pe)

    return {
        "arr": arr,
        "growth_rate_pct": growth_rate,
        "gross_margin_pct": gross_margin * 100,
        "churn_rate_pct": churn_rate * 100,
        "cac": cac,
        "ltv": ltv,
        "ltv_cac_ratio": ltv_cac,
        "ndr_pct": ndr,
        "rule_of_40": rule40,
        "magic_number": magic_number,
        "arr_per_employee": arr_pe,
        "market": market,
        "health": health,
    }


# --------------------------------------------------------------------------
# SHAP: aggregate one-hot dummy contributions back to their original feature
# --------------------------------------------------------------------------

def _original_feature_for(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name[len("num__"):]
    remainder = transformed_name[len("cat__"):]
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if remainder == col or remainder.startswith(col + "_"):
            return col
    return remainder


def aggregate_shap(shap_row: np.ndarray, feature_names: list[str]) -> dict:
    totals = {col: 0.0 for col in FEATURE_COLUMNS}
    for name, value in zip(feature_names, shap_row):
        totals[_original_feature_for(name)] += float(value)
    return totals


# --------------------------------------------------------------------------
# Compensation prediction
# --------------------------------------------------------------------------

def generate_prediction(form: dict) -> dict:
    if not all([preprocessor, quantile_models[0.25], quantile_models[0.50], quantile_models[0.75]]):
        raise InvalidPredictionInput("Prediction service is not configured. Run train_model.py to generate artifacts.")

    role = form.get("role", "")
    role_archetype = role_archetype_for(role)
    city = form.get("city", "")
    funding_stage = form.get("funding_stage", "")
    target_market = form.get("target_market", "")
    skill_category = form.get("skill_category", "")
    education_pedigree = form.get("education_pedigree", "")
    pricing_model = form.get("pricing_model", "")

    try:
        years_experience = float(form.get("years_experience"))
    except (TypeError, ValueError):
        raise InvalidPredictionInput("Years of experience must be a number.")
    if not (0 <= years_experience <= 45):
        raise InvalidPredictionInput("Years of experience must be between 0 and 45.")

    if role_archetype is None:
        raise InvalidPredictionInput("Unrecognized role.")
    if city not in CITIES:
        raise InvalidPredictionInput("Unrecognized city.")
    if funding_stage not in FUNDING_STAGES:
        raise InvalidPredictionInput("Unrecognized funding stage.")
    if target_market not in TARGET_MARKETS:
        raise InvalidPredictionInput("Unrecognized target market.")
    if skill_category not in SKILL_CATEGORIES:
        raise InvalidPredictionInput("Unrecognized skill category.")
    if education_pedigree not in EDUCATION_PEDIGREE:
        raise InvalidPredictionInput("Unrecognized education pedigree.")
    if pricing_model not in PRICING_MODELS:
        raise InvalidPredictionInput("Unrecognized pricing model.")

    level = experience_to_level(years_experience)

    # --- Health score: from the user's own SaaS metrics if supplied, else a
    # data-driven default for the chosen funding stage ---
    health_source = "funding-stage default"
    if form.get("health_score_override") not in (None, ""):
        try:
            health_score = float(form.get("health_score_override"))
        except (TypeError, ValueError):
            raise InvalidPredictionInput("health_score_override must be a number.")
        if not (0 <= health_score <= 100):
            raise InvalidPredictionInput("health_score_override must be between 0 and 100.")
        health_band = sm.health_band_from_score(health_score)
        health_source = "a what-if override, not a submitted or default value"
    elif form.get("use_advanced_metrics") in ("on", "true", "1", True):
        try:
            raw = {
                "mrr": float(form.get("mrr", 0)),
                "prior_mrr": float(form.get("prior_mrr", 0)),
                "revenue": float(form.get("revenue", 0)),
                "cogs": float(form.get("cogs", 0)),
                "customers_lost": float(form.get("customers_lost", 0)),
                "customers_start": float(form.get("customers_start", 1)),
                "sm_spend": float(form.get("sm_spend", 0)),
                "new_customers": float(form.get("new_customers", 1)),
                "arpa": float(form.get("arpa", 0)),
                "cac": float(form.get("cac")) if form.get("cac") else None,
                "ndr": float(form.get("ndr")) if form.get("ndr") else None,
                "headcount": float(form.get("headcount", 1)),
                "yoy_growth_pct": float(form.get("yoy_growth_pct")) if form.get("yoy_growth_pct") else None,
                "profit_margin_pct": float(form.get("profit_margin_pct", 0)),
                "prior_quarter_revenue": float(form.get("prior_quarter_revenue", 0)),
                "current_quarter_revenue": float(form.get("current_quarter_revenue", 0)),
                "prior_quarter_sm_spend": float(form.get("prior_quarter_sm_spend", 1)),
                "tam": float(form.get("tam", 1)),
                "sam": float(form.get("sam", 1)),
                "som": float(form.get("som", 1)),
            }
            bundle = compute_saas_metrics_bundle(raw)
            health_score = bundle["health"].score
            health_band = bundle["health"].band
            health_source = "your company's submitted SaaS metrics"
        except (TypeError, ValueError):
            raise InvalidPredictionInput("Advanced SaaS metrics must be numbers — check the values entered there.")
    else:
        health_score = STAGE_DEFAULT_HEALTH.get(funding_stage, 50.0)
        health_band = sm.health_band_from_score(health_score)

    row = pd.DataFrame([{
        "role_archetype": role_archetype, "role": role, "level": level, "city": city,
        "funding_stage": funding_stage, "target_market": target_market,
        "skill_category": skill_category, "education_pedigree": education_pedigree,
        "pricing_model": pricing_model, "years_experience": years_experience,
        "health_score": health_score,
    }])[FEATURE_COLUMNS]

    transformed = preprocessor.transform(row)

    preds = {q: float(quantile_models[q].predict(transformed)[0]) for q in (0.25, 0.50, 0.75)}
    p25, p50, p75 = sorted(preds.values())  # guard against quantile crossing

    # --- Simulated live-signal adjustment ---
    signal_mult = market_signals.state.multiplier_for(city, skill_category)
    p25, p50, p75 = p25 * signal_mult, p50 * signal_mult, p75 * signal_mult

    # --- ESOP overlay (rule-based, kept visibly separate from cash) ---
    stage_cfg = FUNDING_STAGES[funding_stage]
    esop_ratio = stage_cfg["esop_ratio"] * (1.10 if role_archetype == "Product" else 0.90)
    esop_annualized_lpa = p50 * esop_ratio
    esop_expected = sm.esop_expected_value(esop_annualized_lpa, health_score)

    # --- Compa-ratio ---
    key = "|".join([role, level, city, funding_stage])
    segment_median = COMPA_LOOKUP.get(key, GLOBAL_MEDIAN_CTC)
    compa_ratio = (p50 / segment_median) if segment_median else None

    # --- CTC statutory breakup (on the P50 estimate) ---
    breakup = compute_ctc_breakup(p50, city)

    # --- SHAP explanation (P50 model) ---
    shap_row = _shap_explainer.shap_values(transformed)[0]
    base_value = float(np.ravel(_shap_explainer.expected_value)[0])
    feature_names = list(preprocessor.get_feature_names_out())
    agg = aggregate_shap(shap_row, feature_names)

    display_values = {
        "role_archetype": role_archetype,
        "role": role,
        "level": level,
        "city": city,
        "funding_stage": funding_stage,
        "target_market": target_market,
        "skill_category": skill_category,
        "education_pedigree": education_pedigree,
        "pricing_model": pricing_model,
        "years_experience": f"{years_experience:g} yrs experience",
        "health_score": f"{health_score:.0f}/100",
    }
    contributions = []
    for feature in FEATURE_COLUMNS:
        contributions.append({
            "feature": feature.replace("_", " ").title(),
            "display_value": display_values[feature],
            "lpa": agg[feature],
            "positive": agg[feature] >= 0,
        })
    contributions.sort(key=lambda c: abs(c["lpa"]), reverse=True)
    max_abs = max((abs(c["lpa"]) for c in contributions), default=0) or 1
    for c in contributions:
        c["width_pct"] = round(abs(c["lpa"]) / max_abs * 100, 1)
        c["value_display"] = f"{'+' if c['lpa'] >= 0 else '-'}Rs.{abs(c['lpa']):.2f} LPA"

    return {
        "p25": p25, "p50": p50, "p75": p75,
        "range_text": f"Rs.{p25:.1f} - Rs.{p75:.1f} LPA",
        "median_text": f"Rs.{p50:.1f} LPA",
        "esop_annualized_lpa": esop_annualized_lpa,
        "esop_expected": esop_expected,
        "compa_ratio": compa_ratio,
        "segment_median": segment_median,
        "breakup": breakup,
        "health_score": health_score,
        "health_band": health_band,
        "health_source": health_source,
        "signal_adjustment_pct": (signal_mult - 1) * 100,
        "base_lpa": base_value,
        "contributions": contributions,
        "level": level,
        "inputs": {
            "role_archetype": role_archetype, "role": role, "years_experience": years_experience,
            "city": city, "funding_stage": funding_stage, "target_market": target_market,
            "skill_category": skill_category, "education_pedigree": education_pedigree,
            "pricing_model": pricing_model,
        },
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", metrics=MODEL_METRICS)


@app.route("/about-tech")
def about_tech():
    return render_template("about_tech.html", metrics=MODEL_METRICS)


@app.route("/saas-metrics", methods=["GET", "POST"])
def saas_metrics_page():
    result = None
    error_text = None
    form_data = {}
    if request.method == "POST":
        form_data = request.form
        try:
            raw = {
                "mrr": float(form_data.get("mrr", 0)),
                "prior_mrr": float(form_data.get("prior_mrr", 0)),
                "revenue": float(form_data.get("revenue", 0)),
                "cogs": float(form_data.get("cogs", 0)),
                "customers_lost": float(form_data.get("customers_lost", 0)),
                "customers_start": float(form_data.get("customers_start", 1)),
                "sm_spend": float(form_data.get("sm_spend", 0)),
                "new_customers": float(form_data.get("new_customers", 1)),
                "arpa": float(form_data.get("arpa", 0)),
                "cac": float(form_data.get("cac")) if form_data.get("cac") else None,
                "ndr": float(form_data.get("ndr")) if form_data.get("ndr") else None,
                "headcount": float(form_data.get("headcount", 1)),
                "yoy_growth_pct": float(form_data.get("yoy_growth_pct")) if form_data.get("yoy_growth_pct") else None,
                "profit_margin_pct": float(form_data.get("profit_margin_pct", 0)),
                "prior_quarter_revenue": float(form_data.get("prior_quarter_revenue", 0)),
                "current_quarter_revenue": float(form_data.get("current_quarter_revenue", 0)),
                "prior_quarter_sm_spend": float(form_data.get("prior_quarter_sm_spend", 1)),
                "tam": float(form_data.get("tam", 1)),
                "sam": float(form_data.get("sam", 1)),
                "som": float(form_data.get("som", 1)),
            }
            result = compute_saas_metrics_bundle(raw)
        except (TypeError, ValueError):
            error_text = "All fields must be numbers — please check your inputs."
        except Exception:
            logger.exception("Failed computing SaaS metrics")
            error_text = "Something went wrong computing these metrics. Please try again."

    return render_template("saas_metrics.html", result=result, error_text=error_text, form_data=form_data)


@app.route("/compliance-check", methods=["GET", "POST"])
def compliance_check_page():
    """
    Turns ctc_rules.py's rules into a visible, self-serve feature: an HR user
    checks an offer they've already structured against the same Code on
    Wages/HRA/EPF/Gratuity conventions compute_ctc_breakup() encodes.
    """
    checks = None
    error_text = None
    form_data = {}
    if request.method == "POST":
        form_data = request.form
        try:
            total_ctc = float(form_data.get("total_ctc", 0))
            basic = float(form_data.get("basic", 0))
            hra = float(form_data.get("hra", 0))
            epf = float(form_data.get("epf", 0))
            gratuity = float(form_data.get("gratuity", 0))
            city = form_data.get("city", "")
            if city not in CITIES:
                raise InvalidPredictionInput("Unrecognized city.")
            if total_ctc <= 0:
                raise InvalidPredictionInput("Total CTC must be greater than zero.")
            checks = check_ctc_compliance(total_ctc, basic, hra, epf, gratuity, city)
        except InvalidPredictionInput as exc:
            error_text = str(exc)
        except (TypeError, ValueError):
            error_text = "All amounts must be numbers — please check your inputs."
        except Exception:
            logger.exception("Failed running compliance check")
            error_text = "Something went wrong running this check. Please try again."

    return render_template("compliance_check.html", checks=checks, error_text=error_text, form_data=form_data, cities=CITIES)


@app.route("/predict_form")
def predict_form():
    return render_template(
        "predict_form.html", form_data={},
        role_archetypes=ROLE_ARCHETYPES, cities=CITIES, funding_stages=FUNDING_STAGES,
        target_markets=TARGET_MARKETS, skill_categories=SKILL_CATEGORIES,
        education_pedigree=EDUCATION_PEDIGREE, pricing_models=PRICING_MODELS,
    )


_COMPARE_DROPDOWNS = dict(
    role_archetypes=ROLE_ARCHETYPES, cities=CITIES, funding_stages=FUNDING_STAGES,
    target_markets=TARGET_MARKETS, skill_categories=SKILL_CATEGORIES,
    education_pedigree=EDUCATION_PEDIGREE, pricing_models=PRICING_MODELS,
)


@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    """
    Same candidate profile against two companies' context side by side — the
    highest-value feature for a candidate evaluating competing offers. Reuses
    generate_prediction() twice; the quick (funding-stage/target-market/pricing-model)
    path only, not each company's full advanced-metrics block, to keep the form usable.
    """
    result_a = result_b = None
    error_text = None
    form_data = {}
    if request.method == "POST":
        form_data = request.form
        shared = {
            "role": form_data.get("role", ""),
            "years_experience": form_data.get("years_experience", ""),
            "city": form_data.get("city", ""),
            "skill_category": form_data.get("skill_category", ""),
            "education_pedigree": form_data.get("education_pedigree", ""),
        }
        try:
            form_a = dict(shared,
                          funding_stage=form_data.get("company_a_funding_stage", ""),
                          target_market=form_data.get("company_a_target_market", ""),
                          pricing_model=form_data.get("company_a_pricing_model", ""))
            form_b = dict(shared,
                          funding_stage=form_data.get("company_b_funding_stage", ""),
                          target_market=form_data.get("company_b_target_market", ""),
                          pricing_model=form_data.get("company_b_pricing_model", ""))
            result_a = generate_prediction(form_a)
            result_b = generate_prediction(form_b)
        except InvalidPredictionInput as exc:
            error_text = str(exc)
        except Exception:
            logger.exception("Unexpected error during comparison")
            error_text = "Something went wrong generating this comparison. Please try again."

    return render_template("compare.html", result_a=result_a, result_b=result_b,
                            error_text=error_text, form_data=form_data, **_COMPARE_DROPDOWNS)


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    try:
        result = generate_prediction(form_data)
        return render_template(
            "predict_form.html", result=result, form_data=form_data,
            role_archetypes=ROLE_ARCHETYPES, cities=CITIES, funding_stages=FUNDING_STAGES,
            target_markets=TARGET_MARKETS, skill_categories=SKILL_CATEGORIES,
            education_pedigree=EDUCATION_PEDIGREE, pricing_models=PRICING_MODELS,
        )
    except InvalidPredictionInput as exc:
        return render_template(
            "predict_form.html", error_text=str(exc), form_data=form_data,
            role_archetypes=ROLE_ARCHETYPES, cities=CITIES, funding_stages=FUNDING_STAGES,
            target_markets=TARGET_MARKETS, skill_categories=SKILL_CATEGORIES,
            education_pedigree=EDUCATION_PEDIGREE, pricing_models=PRICING_MODELS,
        )
    except Exception:
        logger.exception("Unexpected error during prediction")
        return render_template(
            "predict_form.html", error_text="Something went wrong generating a prediction. Please try again.",
            form_data=form_data,
            role_archetypes=ROLE_ARCHETYPES, cities=CITIES, funding_stages=FUNDING_STAGES,
            target_markets=TARGET_MARKETS, skill_categories=SKILL_CATEGORIES,
            education_pedigree=EDUCATION_PEDIGREE, pricing_models=PRICING_MODELS,
        )


@app.route("/api/predict", methods=["POST"])
def predict_json():
    """
    Lightweight JSON prediction, reusing generate_prediction(). Powers the "what if
    this company's health score were different?" slider on the results page — the
    same profile is re-scored with `health_score_override` swapped in, without a full
    page reload.
    """
    try:
        result = generate_prediction(request.form)
        return {
            "p25": round(result["p25"], 2), "p50": round(result["p50"], 2), "p75": round(result["p75"], 2),
            "range_text": result["range_text"], "median_text": result["median_text"],
            "health_score": result["health_score"],
            "compa_ratio": round(result["compa_ratio"], 2) if result["compa_ratio"] else None,
        }
    except InvalidPredictionInput as exc:
        return {"error": str(exc)}, 400
    except Exception:
        logger.exception("Unexpected error during what-if prediction")
        return {"error": "Something went wrong."}, 500


@app.route("/live-signals")
def live_signals():
    """
    SSE stream of the SIMULATED market-signal feed (see market_signals.py) — not a
    live external connection. Streams newly-emitted synthetic events as they occur.
    """
    def stream():
        last_count = 0
        while True:
            _adj, events = market_signals.state.snapshot()
            if len(events) > last_count:
                for event in events[-(len(events) - last_count):]:
                    payload = json.dumps({"ts": event.ts, "message": event.message, "kind": event.kind})
                    yield f"data: {payload}\n\n"
                last_count = len(events)
            elif len(events) < last_count:
                last_count = len(events)  # deque trimmed from the front; resync
            else:
                yield ": keep-alive\n\n"
            time.sleep(2)

    return Response(stream(), mimetype="text/event-stream")


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------

_PDF_CHAR_REPLACEMENTS = {
    "—": "-", "–": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...",
    "₹": "Rs.",
}


def pdf_safe(text):
    text = str(text)
    for char, replacement in _PDF_CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SalaryReportPDF(FPDF):
    ACCENT = (255, 122, 89)
    ACCENT_SOFT = (255, 237, 231)
    POSITIVE = (0, 150, 136)
    NEGATIVE = (216, 67, 21)
    TEXT = (26, 29, 36)
    TEXT_SECONDARY = (91, 98, 112)

    def header(self):
        self.set_fill_color(*self.ACCENT)
        self.rect(0, 0, 210, 26, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.set_xy(0, 8)
        self.cell(0, 10, "SaaS Compensation Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.set_xy(0, 18)
        self.cell(0, 8, "Generated by SaaS Compensation Intelligence", align="C")
        self.set_y(32)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Educational project - not financial or HR advice. Cash CTC only; ESOP shown separately.", align="C")


@app.route("/download_report")
def download_report():
    try:
        result = generate_prediction(request.args)
    except InvalidPredictionInput as exc:
        return f"<h1>Could not generate report</h1><p>{exc}</p>", 400
    except Exception:
        logger.exception("Failed to recompute prediction for PDF report")
        return "<h1>Error generating report</h1><p>Please try again.</p>", 500

    try:
        pdf = SalaryReportPDF()
        pdf.add_page()

        y = pdf.get_y()
        pdf.set_draw_color(*SalaryReportPDF.ACCENT)
        pdf.set_fill_color(*SalaryReportPDF.ACCENT_SOFT)
        pdf.rect(20, y, 170, 30, "DF")
        pdf.set_xy(20, y + 5)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(170, 8, "PREDICTED CASH CTC RANGE (P25-P75)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*SalaryReportPDF.ACCENT)
        pdf.cell(170, 12, pdf_safe(result["range_text"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(14)

        pdf.set_text_color(*SalaryReportPDF.TEXT)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Profile Used for This Prediction", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(225, 227, 235)
        pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 12)
        for field, value in result["inputs"].items():
            pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
            pdf.cell(70, 9, pdf_safe(field.replace("_", " ").title()), border=0)
            pdf.set_text_color(*SalaryReportPDF.TEXT)
            pdf.cell(0, 9, pdf_safe(str(value)), new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=0)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*SalaryReportPDF.TEXT)
        pdf.cell(0, 8, "CTC Breakup (statutory + typical components)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 11)
        for row in result["breakup"].as_rows():
            pdf.set_text_color(*SalaryReportPDF.TEXT)
            pdf.cell(120, 8, pdf_safe(row["component"]))
            pdf.cell(0, 8, pdf_safe(f"Rs.{row['lpa']:.2f} LPA"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(120, 8, "ESOP - paper FMV (annualized, separate from cash)")
        pdf.cell(0, 8, pdf_safe(f"Rs.{result['esop_annualized_lpa']:.2f} LPA"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.cell(120, 7, pdf_safe(f"ESOP - risk-adjusted expected value ({result['esop_expected']['band']} band)"))
        pdf.cell(0, 7, pdf_safe(f"Rs.{result['esop_expected']['expected_value_lpa']:.2f} LPA"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Why This Number", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.multi_cell(0, 6, pdf_safe(
            f"Baseline for an average profile: Rs.{result['base_lpa']:.2f} LPA. Each factor below is an "
            f"exact SHAP contribution and they sum precisely to the median prediction."
        ))
        pdf.ln(4)
        for c in result["contributions"]:
            pdf.set_text_color(*SalaryReportPDF.TEXT)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(100, 8, pdf_safe(f"{c['feature']} ({c['display_value']})"))
            color = SalaryReportPDF.POSITIVE if c["positive"] else SalaryReportPDF.NEGATIVE
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, pdf_safe(c["value_display"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.multi_cell(0, 6, pdf_safe(
            "Trained on a benchmark-calibrated synthetic Indian SaaS dataset (see README for sources) - "
            "not live payroll data. Health score reflects " + result["health_source"] + ". "
            "No personal data was stored to produce this report."
        ))

        pdf_content = pdf.output()
        if isinstance(pdf_content, str):
            pdf_content = pdf_content.encode("latin-1")
        response = make_response(bytes(pdf_content))
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=saas-compensation-report.pdf"
        return response
    except Exception:
        logger.exception("Failed to render PDF report")
        return "<h1>Error generating report</h1><p>Please try again.</p>", 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
