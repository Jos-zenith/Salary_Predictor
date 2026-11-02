# CompIntel — India SaaS Compensation Intelligence

A machine learning web application that predicts a **cash CTC range (P25/P50/P75)**
for roles at Indian SaaS companies — grounded in the hiring company's *actual computed
unit economics* (ARR, NDR, LTV:CAC, Rule of 40), not just a guessed funding-round
label. ESOP is estimated separately, and every prediction is broken down into its
statutory CTC components and explained per-feature with SHAP.

## Overview

- **Problem:** Indian SaaS compensation depends on factors generic salary data
  doesn't capture — cross-border vs. domestic target market, funding-stage-driven
  cash/ESOP mix, and, most of all, a company's actual business health. Two companies
  at the same funding round can have very different ability to pay; a funding-stage
  label alone can't tell them apart.
- **Approach:** A B2B SaaS unit-economics module (`saas_metrics.py`) turns a
  company's raw numbers into a computed health score, which is a real input feature
  to a quantile gradient-boosting model trained on a benchmark-calibrated synthetic
  dataset. The median prediction is then split into its statutory CTC components.
- **Output:** A P25/P50/P75 cash CTC range (LPA), a separate ESOP estimate (both a
  paper FMV and a risk-adjusted expected value), a Basic/HRA/EPF/Gratuity/LTA/Bonus
  breakdown that sums exactly to the median estimate, a compa-ratio against the
  empirical segment median, and an exact SHAP explanation of what drove the number.

## Feature tour

| Page | What it does |
|---|---|
| `/predict_form` | The core predictor. Also has a **what-if slider** — drag a company's health score and watch the P25/P50/P75 range recompute live (`/api/predict`, no page reload), holding role/city/funding-stage fixed so only the health score moves. |
| `/compare` | Same candidate profile against **two companies side by side** — two CTC ranges, two health scores, and a "Company A is X% higher" bottom line. The highest-value feature for a candidate evaluating competing offers. |
| `/saas-metrics` | The course-content calculator: raw company numbers in, every formula (ARR, CAC, LTV, NDR, Rule of 40, Magic Number, TAM/SAM/SOM) out, each labeled with its formula. |
| `/compliance-check` | Turns `ctc_rules.py` into a visible feature: enter an offer's already-decided Basic/HRA/EPF/Gratuity and check it against the same Code on Wages / HRA / EPF / Gratuity conventions the predictor itself uses. |

## Where each course topic lives in this project

This grew out of a B2B SaaS Metrics course module. Rather than a form with
SaaS-flavored dropdowns, every topic became a real, working piece:

| Course topic | Where it lives |
|---|---|
| SaaS fundamentals, IaaS/PaaS/SaaS, business models | Methodology page — conceptual primer |
| Pricing models (subscription, freemium, tiered, pay-as-you-go, per-user) | Primer, and a live input feature on the predictor |
| ARR, MRR, Churn, CAC, LTV, LTV:CAC, NDR, Rule of 40, Magic Number, Gross Margin | `saas_metrics.py` + the `/saas-metrics` calculator |
| TAM / SAM / SOM | `market_opportunity_tier()` in the calculator |
| Funding stages, ESOP | Compensation engine — funding-stage input, ESOP overlay (FMV + risk-adjusted expected value) |
| "Real-time" business signals | Simulated live-signal layer (`market_signals.py`) — labeled as a simulation, not a live feed |

## Tech stack

| Layer | Tools |
|---|---|
| SaaS unit economics | Pure Python (`saas_metrics.py`) — no external formula library |
| Data | Pandas, NumPy — seeded synthetic, benchmark-calibrated (`data/generate_dataset.py`) |
| Modeling | scikit-learn `HistGradientBoostingRegressor` (quantile loss), `LinearRegression` baseline |
| Explainability | SHAP (`TreeExplainer`), aggregated from one-hot columns back to original features |
| Web app | Flask, Jinja2, HTML/CSS/JS, Server-Sent Events (simulated live-signal feed) |
| Reporting | fpdf2 (PDF generation) |
| Research | Jupyter Notebook (`INDIA_SAAS_CTC_PREDICTION_MODEL.ipynb`) |

## How it works

1. **`saas_metrics.py`** — every SaaS growth-metric formula (ARR, CAC, LTV, LTV:CAC,
   NDR, Rule of 40, Magic Number, Gross Margin, TAM/SAM/SOM), plus `saas_health_score()`
   — an authored, documented weighting scheme (Rule of 40 30% + NDR 30% + LTV:CAC 20%
   + ARR/employee 20%), explicitly *not* claimed as an industry-standard formula.
   Also `cross_border_premium_multiplier()` — the domestic-vs-cross-border pay
   premium as a named, testable function rather than a bare config constant — and
   `esop_expected_value()`, an explicitly-labeled risk-adjustment heuristic (paper
   FMV × liquidity probability × payout multiple, indexed to health band) on top of
   the ESOP estimate, deliberately *not* a Black-Scholes-style pricing model, since
   an illiquid private-company option has no observable strike price or volatility
   to price one against.
2. **`saas_domain.py`** — the shared vocabulary (funding stages, cities, roles, skill
   categories, education pedigree, pricing models, experience-to-level bands) that
   both the dataset generator and the Flask app import, so served form options can
   never drift from what the model was trained on.
3. **`data/generate_dataset.py`** — generates a seeded, synthetic, ~24,500-row Indian
   SaaS dataset across ~1,000 synthetic companies. Each company's health score is
   computed by running its (also synthetic, stage-appropriate) raw metrics through
   `saas_metrics.py` — the same functions the live calculator uses.
4. **`train_model.py`** / **`INDIA_SAAS_CTC_PREDICTION_MODEL.ipynb`** — the research
   phase: baseline `LinearRegression` vs. three `HistGradientBoostingRegressor`
   quantile models (P25/P50/P75), evaluation, SHAP, and artifact export.
5. **`ctc_rules.py`** — deterministic statutory CTC breakup (not learned).
6. **`market_signals.py`** — the simulated live-signal layer.
7. **`main.py`** — the serving phase. Loads artifacts read-only, runs inference, computes
   the CTC breakup/ESOP/compa-ratio, and renders the result. Never trains or modifies
   the model.

## Model performance

Evaluated on a held-out 20% test split (run `train_model.py` or the notebook to
reproduce; exact numbers below are from this session's training run):

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Linear Regression (baseline) | 0.884 | ±3.80 LPA | 21.3% |
| HistGradientBoostingRegressor, quantile (production) | 0.961 | ±1.95 LPA | 7.6% |

The gap is real, not assumed: the synthetic pay curve has genuine step-function jumps
at each promotion boundary, which a linear model structurally cannot represent. The
three quantile models (alpha=0.25/0.5/0.75) are trained independently and can
technically "cross"; `main.py` sorts them at serving time to guarantee a sane P25 ≤
P50 ≤ P75 range is always displayed.

**Is the held-out score a lucky split?** 5-fold cross-validation (each fold refits its
own preprocessor, no leakage) says no — both models are tight across folds:

| Model | 5-fold R² | 5-fold MAE |
|---|---|---|
| Linear Regression (baseline) | 0.878 ± 0.004 | 3.78 ± 0.04 LPA |
| HistGradientBoostingRegressor (P50) | 0.959 ± 0.003 | 1.94 ± 0.04 LPA |

**Face validity.** P50 predictions on three canonical profiles were checked against
the same cited benchmark ranges the calibration was built from (not a scrape-based
external validation — see Limitations on why that wasn't attempted). Two of three
land inside their cited range; a Chennai/Domestic/Mid-level/Series A profile predicts
₹15.7 LPA against a cited ₹6-12 LPA band. Read honestly: that cited figure is a broad
city-wide average across all experience levels and employer types (including
lower-paying IT-services roles), not a segment-specific one — a population average and
a specific well-funded-startup, mid-level, product-engineering prediction aren't the
same quantity. Reproducible in `train_model.py` / the notebook, section 8.

## Sources for the calibration constants

- **City/company pay ranges:** Chennai engineers ~₹7.9-8.7 LPA; Bengaluru
  structurally higher (concentrates product/AI/senior-IC work); Freshworks/Zoho-tier
  SaaS ₹6-12 LPA; funded startups/unicorns ₹15-30 LPA; cross-border/FAANG-tier
  ₹35-60 LPA; IT services ₹3.5-6 LPA (SDE-1) up to ₹10-16 LPA (7y senior), vs.
  product companies paying 3-5x more for the same level.
- **CTC breakup:** India's Code on Wages (enforced 2026-04-01) — Basic+DA must be
  ≥50% of CTC. HRA ~40% of Basic (non-metro) / 50% (metro). Employer EPF = 12% of
  Basic. Gratuity accrual ~4.81% of Basic.
- **NDR benchmark:** industry median ~106%, top-quartile ≥120% (KeyBanc Capital
  Markets).
- **ESOP pools:** Indian B2B SaaS/deeptech typically run 10-20% ESOP pools, larger
  than other sectors, front-loaded to early hires.

## Limitations

- **Synthetic, benchmark-calibrated data — not live payroll.**
  `data/india_saas_ctc_dataset.csv` is seeded and synthetic; real Indian SaaS payroll
  data is confidential and wasn't available to this project. What makes it calibrated
  rather than arbitrary: base pay bands and premiums are anchored to the cited ranges
  above, and every synthetic company's health score is computed by the same
  `saas_metrics.py` functions the live calculator uses — not a separate shortcut.
  This is the same standard the project held itself to before: an earlier version of
  this app applied a flat USD→INR exchange rate to a US-trained model's output and
  called it an India estimate, which was wrong for the same reason a synthetic
  dataset would be wrong if it weren't calibrated — a conversion or a guess isn't a
  labor-market fact. See `archive/` for that earlier version.
- **The live-signal feed is a labeled simulation.** A real integration would need
  paid Crunchbase/Tracxn API access, always-on infrastructure, and raises job-board
  scraping ToS concerns — none of which this project has the budget or standing to
  take on. `market_signals.py` demonstrates the event-driven architecture (state,
  decay, an SSE-streamed UI) a real feed would plug into at one function
  (`_random_event()`), without claiming a live connection that doesn't exist.
- **SaaS-specific scope.** Calibrated for Indian B2B SaaS roles specifically — not
  generic IT services, non-tech sectors, or executive compensation.
- **The SaaS Health Score is an authored rubric**, not an industry-standard formula —
  no single canonical one exists; weights are documented in `saas_metrics.py`.
- **ESOP is a modeled assumption**, not a scraped or verified figure — exact
  cash:equity mixes by funding stage aren't publicly standardized. Its "expected
  value" figure is a documented heuristic (liquidity probability × payout multiple
  by health band), not options pricing — deliberately not a Black-Scholes-style
  model, since an illiquid private-company option has no observable strike price or
  volatility to price one against.
- **The compliance checker validates against this project's own conventions**, not a
  certified legal or statutory payroll audit — it's a sanity check on an offer's
  components against the same Code on Wages / HRA / EPF / Gratuity rules
  `ctc_rules.py` uses to generate a breakup, not a substitute for legal review.
- No input data is stored or logged; every prediction request is stateless.

## Background

Grew out of a B2B SaaS Metrics course module (fundamentals, business models, pricing
models, and growth metrics) and a 6-week AI/ML internship with IBM SkillsBuild,
supported by Edunet Foundation (June-July 2025). The original internship project — a
Random Forest regressor on a 373-row generic US salary dataset — is preserved in
`archive/`, not deleted.

## Roadmap

- A real funding/hiring-signal integration behind the same event interface the
  simulation already uses.
- Broader role coverage beyond SaaS-specific archetypes.
- Confidence intervals validated against real (not synthetic) outcome data, if a
  legitimate source becomes available.

## Running locally

```
pip install -r requirements.txt
python data/generate_dataset.py   # regenerate the synthetic dataset (optional — already committed)
python train_model.py             # train and export artifacts to artifacts/ (optional — already committed)
python main.py                    # serve at http://127.0.0.1:5000
```

## Acknowledgements

Thanks to Edunet Foundation and IBM SkillsBuild for the internship opportunity this
project grew out of.
