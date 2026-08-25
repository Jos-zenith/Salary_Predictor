"""
SalaryPredict AI — Flask application.

Serves the salary prediction web app: a landing page, a prediction form
backed by a pre-trained Random Forest regression model, and a PDF report
generator for the result. The model, scaler, and label encoders are
trained offline (see the notebook) and loaded here read-only — this
module performs inference only, it never trains anything.

Every prediction is paired with an exact, per-feature explanation (via
SHAP's TreeExplainer): the amount each of the five inputs pushed the
result up or down from the model's baseline. Because the underlying
model is a tree ensemble, these contributions are exact, not
approximated, and always sum to the displayed prediction.
"""

import os
import logging

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, request, make_response
from fpdf import FPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Model artifact filenames ---
MODEL_FILENAME = "salary_model.pkl"
SCALER_FILENAME = "scaler1.pkl"
LABEL_ENCODERS_FILENAME = "label_encoders.pkl"

# The fields the model was actually trained on, in the exact order the
# feature vector expects them.
#
# IMPORTANT: `Salary Data.csv` has no country or location field at all —
# it's a generic US-compensation dataset (salaries from $350 to $250,000,
# mean ~$100k). The model's output is a USD figure for that data, nothing
# more specific. An earlier version of this app multiplied that number by
# a USD→INR exchange rate and presented it as an Indian salary estimate,
# which is wrong: currency conversion isn't labor-market conversion, and
# Indian compensation for the same role isn't "US salary × exchange rate."
# Predictions are shown in USD to avoid implying a claim the data can't
# support. Retraining on an actual India-market dataset would be the real
# fix if that's the goal.
MODEL_FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]


def load_asset(filename):
    """Load a joblib-serialized model asset, logging clearly if it's missing."""
    if os.path.exists(filename):
        logger.info("Loaded asset: %s", filename)
        return joblib.load(filename)
    logger.warning("Asset not found: %s — predictions will fail until it's added.", filename)
    return None


model = load_asset(MODEL_FILENAME)
scaler = load_asset(SCALER_FILENAME)
label_encoders = load_asset(LABEL_ENCODERS_FILENAME)

# The trained estimator, unwrapped from its pipeline, for SHAP's TreeExplainer.
# `feature_perturbation="tree_path_dependent"` needs no background dataset and
# is exact for tree ensembles like this one.
_estimator = model.steps[-1][1] if model is not None else None
explainer = shap.TreeExplainer(_estimator, feature_perturbation="tree_path_dependent") if _estimator else None


class InvalidPredictionInput(Exception):
    """Raised when the submitted form data can't be turned into a prediction."""


def generate_prediction(age, gender, education, job_title, experience):
    """
    Run the full inference pipeline for one profile and explain the result.

    Returns a dict with the final prediction (in USD/year — see the note by
    MODEL_FEATURES on why this isn't converted to any other currency) and an
    exact, per-feature breakdown of how each input moved the prediction away
    from the model's baseline. Raises InvalidPredictionInput with a
    user-facing message on bad input.
    """
    if not all([model, scaler, label_encoders, explainer]):
        raise InvalidPredictionInput("Prediction service is not configured. Please check server logs for missing model files.")

    try:
        age = float(age)
        experience = float(experience)
    except (TypeError, ValueError):
        raise InvalidPredictionInput("Please check your inputs — age and experience must be numbers.")

    try:
        gender_encoded = label_encoders["Gender"].transform([gender])[0]
        education_encoded = label_encoders["Education Level"].transform([education])[0]
        job_title_encoded = label_encoders["Job Title"].transform([job_title])[0]
    except ValueError:
        raise InvalidPredictionInput("Unrecognized gender, education level, or job title. Please choose one of the listed options.")

    input_row = pd.DataFrame(
        [[age, gender_encoded, education_encoded, job_title_encoded, experience]],
        columns=MODEL_FEATURES,
    )
    scaled_features = scaler.transform(input_row)

    predicted_salary_usd = float(model.predict(scaled_features)[0])

    # --- Explanation: exact SHAP contribution per feature, in the same units ---
    shap_values = explainer.shap_values(scaled_features)[0]
    base_value_usd = float(np.ravel(explainer.expected_value)[0])

    feature_display = {
        "Age": f"{age:g} years old",
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": f"{experience:g} yrs experience",
    }

    contributions = []
    for feature_name, shap_value_usd in zip(MODEL_FEATURES, shap_values):
        contributions.append({
            "feature": feature_name,
            "display_value": feature_display[feature_name],
            "usd": float(shap_value_usd),
            "positive": shap_value_usd >= 0,
        })
    contributions.sort(key=lambda c: abs(c["usd"]), reverse=True)

    max_abs = max((abs(c["usd"]) for c in contributions), default=0) or 1
    for c in contributions:
        c["width_pct"] = round(abs(c["usd"]) / max_abs * 100, 1)
        c["value_display"] = f"{'+' if c['usd'] >= 0 else '-'}${abs(c['usd']):,.0f}"

    return {
        "result_text": f"${predicted_salary_usd:,.0f} / year",
        "final_usd": predicted_salary_usd,
        "base_usd": base_value_usd,
        "base_text": f"${base_value_usd:,.0f} / year",
        "contributions": contributions,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    """Landing page."""
    return render_template("index.html")


@app.route("/about-tech")
def about_tech():
    """Methodology / tech-stack page."""
    return render_template("about_tech.html")


@app.route("/predict_form")
def predict_form():
    """Prediction form, empty on first load."""
    return render_template("predict_form.html", form_data={})


@app.route("/predict", methods=["POST"])
def predict():
    """Validate form input, run inference, and render the result with its explanation."""
    form_data = request.form
    try:
        result = generate_prediction(
            age=form_data.get("Age"),
            gender=form_data.get("Gender"),
            education=form_data.get("Education Level"),
            job_title=form_data.get("Job Title"),
            experience=form_data.get("Years of Experience"),
        )
        return render_template(
            "predict_form.html",
            prediction_text=result["result_text"],
            explanation=result,
            form_data=form_data,
        )
    except InvalidPredictionInput as exc:
        return render_template("predict_form.html", error_text=str(exc), form_data=form_data)
    except Exception:
        logger.exception("Unexpected error during prediction")
        return render_template(
            "predict_form.html",
            error_text="Something went wrong generating a prediction. Please try again.",
            form_data=form_data,
        )


# FPDF's built-in Helvetica font can only render latin-1. It's an easy trap:
# a perfectly normal em dash or smart quote in a Python string will crash
# PDF generation with an obscure FPDFUnicodeEncodingException at render
# time, well after the string was written. `pdf_safe` is a defensive
# backstop against that class of bug for any text that reaches the PDF -
# it should never trigger in practice, but if new copy sneaks in a
# character Helvetica can't render, this degrades gracefully instead of
# 500ing the whole report.
_PDF_CHAR_REPLACEMENTS = {
    "—": "-", "–": "-",   # em dash, en dash
    "‘": "'", "’": "'",   # smart single quotes
    "“": '"', "”": '"',   # smart double quotes
    "…": "...",                # ellipsis
}


def pdf_safe(text):
    """Make a string safe for FPDF's latin-1-only core fonts."""
    for char, replacement in _PDF_CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SalaryReportPDF(FPDF):
    """A one-page PDF summary of a single prediction, including its explanation."""

    ACCENT = (67, 56, 202)      # matches --primary in the site's design system
    ACCENT_SOFT = (238, 240, 253)
    POSITIVE = (13, 148, 136)
    NEGATIVE = (185, 28, 28)
    TEXT = (26, 29, 36)
    TEXT_SECONDARY = (91, 98, 112)

    def header(self):
        self.set_fill_color(*self.ACCENT)
        self.rect(0, 0, 210, 26, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 20)
        self.set_xy(0, 8)
        self.cell(0, 10, "Salary Prediction Report", align="C", ln=1)
        self.set_font("Helvetica", "", 10)
        self.set_xy(0, 18)
        self.cell(0, 8, "Generated by SalaryPredict AI", align="C")
        self.set_y(32)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Educational project - not financial or HR advice.", align="C")


@app.route("/download_report")
def download_report():
    """
    Re-run the prediction from the submitted profile and render it as a
    downloadable PDF. Recomputing server-side (rather than trusting a
    prediction string passed back from the browser) keeps the PDF
    consistent with the model, not with whatever a client sent.
    """
    try:
        result = generate_prediction(
            age=request.args.get("age"),
            gender=request.args.get("gender"),
            education=request.args.get("education"),
            job_title=request.args.get("job_title"),
            experience=request.args.get("experience"),
        )
    except InvalidPredictionInput as exc:
        return f"<h1>Could not generate report</h1><p>{exc}</p>", 400
    except Exception:
        logger.exception("Failed to recompute prediction for PDF report")
        return "<h1>Error generating report</h1><p>Please try again.</p>", 500

    try:
        pdf = SalaryReportPDF()
        pdf.add_page()

        # Prediction highlight card
        y = pdf.get_y()
        pdf.set_draw_color(*SalaryReportPDF.ACCENT)
        pdf.set_fill_color(*SalaryReportPDF.ACCENT_SOFT)
        pdf.rect(20, y, 170, 30, "DF")
        pdf.set_xy(20, y + 5)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(170, 8, "PREDICTED ANNUAL SALARY", ln=1, align="C")
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(*SalaryReportPDF.ACCENT)
        pdf.cell(170, 12, pdf_safe(result["result_text"]), ln=1, align="C")
        pdf.ln(14)

        # Profile section
        pdf.set_text_color(*SalaryReportPDF.TEXT)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Profile Used for This Prediction", ln=1)
        pdf.set_draw_color(225, 227, 235)
        pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)

        profile_rows = [
            ("Age", request.args.get("age", "N/A")),
            ("Gender", request.args.get("gender", "N/A")),
            ("Education Level", request.args.get("education", "N/A")),
            ("Job Title", request.args.get("job_title", "N/A")),
            ("Years of Experience", request.args.get("experience", "N/A")),
        ]
        pdf.set_font("Helvetica", "", 12)
        for field, value in profile_rows:
            pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
            pdf.cell(60, 9, pdf_safe(field), border=0)
            pdf.set_text_color(*SalaryReportPDF.TEXT)
            pdf.cell(0, 9, pdf_safe(str(value)), ln=1, border=0)
        pdf.ln(6)

        # Explanation section
        pdf.set_text_color(*SalaryReportPDF.TEXT)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Why This Number", ln=1)
        pdf.set_draw_color(225, 227, 235)
        pdf.line(pdf.get_x(), pdf.get_y(), 190, pdf.get_y())
        pdf.ln(6)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.multi_cell(0, 6, pdf_safe(f"Baseline for an average profile: {result['base_text']}. Each factor below shifts the prediction up or down from there."))
        pdf.ln(4)

        for c in result["contributions"]:
            pdf.set_text_color(*SalaryReportPDF.TEXT)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(90, 8, pdf_safe(f"{c['feature']} ({c['display_value']})"), border=0)
            color = SalaryReportPDF.POSITIVE if c["positive"] else SalaryReportPDF.NEGATIVE
            pdf.set_text_color(*color)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, pdf_safe(c["value_display"]), ln=1, align="R")

        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*SalaryReportPDF.TEXT_SECONDARY)
        pdf.multi_cell(
            0, 6,
            "This estimate is generated by a machine learning model trained on a general "
            "US compensation dataset - it is a USD figure, not converted to any other "
            "currency or labor market. Treat it as a reference point, not a guarantee. "
            "No personal data was stored to produce it."
        )

        pdf_content = pdf.output()
        if isinstance(pdf_content, str):
            pdf_content = pdf_content.encode("latin-1")
        response = make_response(bytes(pdf_content))
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=salary-prediction-report.pdf"
        return response

    except Exception:
        logger.exception("Failed to render PDF report")
        return "<h1>Error generating report</h1><p>Please try again.</p>", 500


if __name__ == "__main__":
    app.run(debug=True)
