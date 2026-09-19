

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import saas_metrics as sm  # noqa: E402
from saas_domain import (  # noqa: E402
    CITIES, EDUCATION_PEDIGREE, FUNDING_STAGES, PRICING_MODELS,
    ROLE_ARCHETYPES, SKILL_CATEGORIES, TARGET_MARKETS,
    experience_to_level, experience_to_level_mult,
)

RNG_SEED = 42
N_COMPANIES = 1000
EMPLOYEES_PER_COMPANY = (10, 40)  # uniform range

OUTPUT_PATH = Path(__file__).resolve().parent / "india_saas_ctc_dataset.csv"


def generate_companies(rng: np.random.Generator, n: int) -> pd.DataFrame:
    stages = rng.choice(list(FUNDING_STAGES.keys()), size=n, p=[0.22, 0.18, 0.20, 0.18, 0.14, 0.08])
    rows = []
    for i, stage in enumerate(stages):
        cfg = FUNDING_STAGES[stage]
        headcount = rng.integers(cfg["headcount"][0], cfg["headcount"][1])
        growth = rng.uniform(*cfg["growth"])
        margin = rng.uniform(*cfg["margin"])
        ndr = rng.uniform(*cfg["ndr"])
        ltv_cac = rng.uniform(*cfg["ltv_cac"])
        # ARR/employee in USD: later-stage, healthier companies run more revenue per head.
        arr_per_employee_usd = rng.uniform(60_000, 90_000) * (1 + 0.35 * list(FUNDING_STAGES).index(stage))

        rule40 = sm.rule_of_40(growth, margin)
        health = sm.saas_health_score(
            rule40_value=rule40,
            ndr_value=ndr,
            ltv_cac_ratio=ltv_cac,
            arr_per_employee_usd=arr_per_employee_usd,
        )

        target_market = rng.choice(list(TARGET_MARKETS.keys()), p=[0.45, 0.55])
        pricing_model = rng.choice(list(PRICING_MODELS.keys()))

        rows.append(dict(
            company_id=i,
            funding_stage=stage,
            headcount=int(headcount),
            yoy_growth_pct=round(growth, 1),
            profit_margin_pct=round(margin, 1),
            ndr_pct=round(ndr, 1),
            ltv_cac_ratio=round(ltv_cac, 2),
            arr_per_employee_usd=round(arr_per_employee_usd, 0),
            rule_of_40=round(rule40, 1),
            health_score=health.score,
            health_band=health.band,
            target_market=target_market,
            pricing_model=pricing_model,
        ))
    return pd.DataFrame(rows)


def generate_employees(rng: np.random.Generator, companies: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, co in companies.iterrows():
        n_emp = rng.integers(EMPLOYEES_PER_COMPANY[0], EMPLOYEES_PER_COMPANY[1])
        stage_cfg = FUNDING_STAGES[co["funding_stage"]]
        # pay multiplier blends the stage's baseline band with the company's own
        # *computed* health score (healthier companies pay a bit above their stage's
        # baseline, cash-constrained ones a bit below) — documented linear scaling.
        health_mult = 0.85 + (co["health_score"] / 100) * 0.30

        for _ in range(n_emp):
            archetype = rng.choice(list(ROLE_ARCHETYPES.keys()), p=[0.65, 0.35])
            role = rng.choice(list(ROLE_ARCHETYPES[archetype].keys()))
            base_lpa = ROLE_ARCHETYPES[archetype][role]

            years_experience = round(float(rng.exponential(4.0)), 1)
            years_experience = min(years_experience, 22.0)
            level = experience_to_level(years_experience)
            level_mult = experience_to_level_mult(years_experience)

            city = rng.choice(list(CITIES.keys()))
            hub_tier, city_mult = CITIES[city]

            skill = rng.choice(list(SKILL_CATEGORIES.keys()))
            pedigree = rng.choice(list(EDUCATION_PEDIGREE.keys()), p=[0.15, 0.35, 0.35, 0.15])

            target_mult = TARGET_MARKETS[co["target_market"]]
            pricing_mult = PRICING_MODELS[co["pricing_model"]]

            noise = rng.lognormal(mean=0.0, sigma=0.08)

            cash_ctc_lpa = (
                base_lpa * level_mult * city_mult * stage_cfg["pay_mult"] *
                health_mult * target_mult * SKILL_CATEGORIES[skill] *
                EDUCATION_PEDIGREE[pedigree] * pricing_mult * noise
            )

            esop_ratio = stage_cfg["esop_ratio"] * (1.10 if archetype == "Product" else 0.90)
            esop_annualized_lpa = cash_ctc_lpa * esop_ratio * rng.uniform(0.85, 1.15)

            records.append(dict(
                company_id=co["company_id"],
                funding_stage=co["funding_stage"],
                health_score=co["health_score"],
                health_band=co["health_band"],
                target_market=co["target_market"],
                pricing_model=co["pricing_model"],
                role_archetype=archetype,
                role=role,
                years_experience=years_experience,
                level=level,
                city=city,
                saas_hub_tier=hub_tier,
                skill_category=skill,
                education_pedigree=pedigree,
                cash_ctc_lpa=round(cash_ctc_lpa, 2),
                esop_annualized_lpa=round(esop_annualized_lpa, 2),
            ))
    return pd.DataFrame(records)


def main():
    rng = np.random.default_rng(RNG_SEED)
    companies = generate_companies(rng, N_COMPANIES)
    employees = generate_employees(rng, companies)
    employees.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(employees):,} rows to {OUTPUT_PATH}")
    summary = employees.describe(include="all").T[["count", "unique", "top", "freq"]].astype(object).fillna("")
    print(summary.to_string())
    print("\ncash_ctc_lpa summary:\n", employees["cash_ctc_lpa"].describe())


if __name__ == "__main__":
    main()
