# SalaryPredict AI — Employee Salary Prediction

A machine learning web application that predicts likely annual salary from
five inputs (age, gender, education level, job title, years of experience),
using a Random Forest regressor served through a Flask backend. Every
prediction ships with an exact, per-feature explanation via SHAP.

## Overview

- **Problem:** Compensation data is asymmetric — employers have market
  benchmarks that individual candidates and hiring managers typically don't.
- **Approach:** Train a regression model on historical salary data, then
  serve it through a small web app that runs the same preprocessing at
  inference time as was used during training.
- **Output:** An estimated annual salary in USD (the currency of the
  training data — see Limitations below on why it isn't converted to
  anything else), a breakdown of which inputs pushed that number up or down
  and by how much, and an optional PDF summary of both.

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | Python, Pandas, NumPy, Scikit-learn |
| Model tuning | GridSearchCV, K-Fold cross-validation |
| Explainability | SHAP (`TreeExplainer`) |
| Web app | Flask, Jinja2, HTML/CSS/JS |
| Reporting | fpdf2 (PDF generation) |
| Experimentation | Jupyter Notebook |

## How it works

1. **`EMPLOYEE_PREDICTION_ML_MODEL.ipynb`** — the research phase. Data
   cleaning, feature engineering, model comparison (Linear Regression,
   Random Forest, gradient-boosted trees), and evaluation (R², RMSE, MAE,
   cross-validation). The final model — a Random Forest Regressor — plus the
   scaler and label encoders are serialized to `.pkl` files.
2. **`main.py`** — the serving phase. Loads the `.pkl` artifacts, applies the
   same encoding/scaling to incoming form data, runs inference, explains the
   result with SHAP, and renders both. It does not train or modify the model.
3. **`templates/`** — the Flask/Jinja2 templates for the landing page,
   prediction form, and methodology page, sharing a single stylesheet at
   `static/css/style.css`.

## Explainability

Because the model is a tree ensemble, `main.py` uses SHAP's `TreeExplainer`
to compute an *exact* contribution for each of the five inputs on every
prediction — not a post-hoc approximation. The baseline (an average profile)
plus the five contributions always sum precisely to the displayed number,
and the same breakdown is recomputed server-side for the PDF report so it
can't drift from what the model actually did.

One thing that breakdown surfaces: **Age accounts for ~58% of the model's
overall feature importance** — well ahead of Job Title (~12%) or Education
(~3%). That's worth knowing before treating any single prediction as
authoritative.

## Model performance

- **R² score:** 0.92 on the held-out test set
- **Mean absolute error:** ~$45,000
- **Feature importance:** Age 58.3% · Experience 27.1% · Job Title 11.6% ·
  Education 2.7% · Gender 0.3%


## Limitations

- **This is a US-market model, and predictions are shown in USD.**
  `Salary Data.csv` has no country or location field at all — it's a
  general US compensation dataset (373 rows, salaries $350–$250,000, mean
  ~$100,577). An earlier version of this app multiplied the model's USD
  output by a fixed exchange rate and presented that as an Indian salary
  estimate. That's wrong: a currency conversion isn't a labor-market
  conversion, and Indian compensation for the same role isn't "US salary ×
  exchange rate." This version shows the raw USD prediction and applies no
  currency conversion anywhere. Getting an India-specific (or any other
  market-specific) estimate would require retraining on data from that
  market, not converting this one's output.
- Trained on a single historical dataset — predictions won't generalize
  to every region, role, or company.
- Reflects whatever patterns (including biases) exist in the training data.
- Most accurate for common corporate/tech roles; not calibrated for niche
  professions, freelance work, or executive compensation.
- No input data is stored or logged; every prediction request is stateless.

## Background

Originally built during a 6-week AI/ML internship with IBM SkillsBuild,
supported by the Edunet Foundation (June–July 2025). This version cleans up
the original UI, trims the prediction form to only the fields the model
actually uses, and removes dead/duplicated code from the backend.

## Roadmap

- Broader international salary datasets (US, UK, Canada)
- Confidence intervals around each prediction, not just a point estimate
- Investigate the model's heavy reliance on Age (see Explainability above) —
  possibly a proxy-for-experience effect worth correcting for

## Acknowledgements

Thanks to Edunet Foundation and IBM SkillsBuild for the internship
opportunity this project grew out of.
