

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def compute_arr(mrr: float) -> float:
    """Annual Recurring Revenue = MRR x 12."""
    return mrr * 12


def compute_mrr_growth_rate(current_mrr: float, prior_mrr: float) -> float:
    """Month-over-month MRR growth, as a fraction (0.10 = 10%)."""
    if prior_mrr <= 0:
        return 0.0
    return (current_mrr - prior_mrr) / prior_mrr


def compute_gross_margin(revenue: float, cogs: float) -> float:
    """Gross Margin = (Revenue - COGS) / Revenue, as a fraction."""
    if revenue <= 0:
        return 0.0
    return (revenue - cogs) / revenue


def compute_churn_rate(customers_lost: float, customers_at_period_start: float) -> float:
    """Customer churn rate for the period, as a fraction."""
    if customers_at_period_start <= 0:
        return 0.0
    return customers_lost / customers_at_period_start


def compute_cac(sales_and_marketing_spend: float, new_customers_acquired: float) -> float:
    """Customer Acquisition Cost = S&M spend / new customers acquired in the period."""
    if new_customers_acquired <= 0:
        return 0.0
    return sales_and_marketing_spend / new_customers_acquired


def compute_ltv(average_revenue_per_account: float, gross_margin: float, churn_rate: float) -> float:
    """
    Customer Lifetime Value (standard SaaS approximation):
        LTV = (ARPA x Gross Margin) / Churn Rate
    """
    if churn_rate <= 0:
        return 0.0
    return (average_revenue_per_account * gross_margin) / churn_rate


def ltv_to_cac_ratio(ltv: float, cac: float) -> float:
    """LTV:CAC ratio. 3:1 or higher is the standard 'healthy unit economics' bar."""
    if cac <= 0:
        return 0.0
    return ltv / cac


def compute_ndr(revenue_from_cohort_at_period_end: float, revenue_from_cohort_at_period_start: float) -> float:
    """
    Net Dollar/Revenue Retention (NDR/NRR), as a percentage, over the *same customer
    cohort*: (revenue from those customers at period end / revenue from those same
    customers at period start) x 100. Industry median for private SaaS is ~106%;
    top-quartile companies run 120%+ (KeyBanc Capital Markets benchmark).
    """
    if revenue_from_cohort_at_period_start <= 0:
        return 0.0
    return (revenue_from_cohort_at_period_end / revenue_from_cohort_at_period_start) * 100


def rule_of_40(yoy_growth_rate_pct: float, profit_margin_pct: float) -> float:
    """Rule of 40 = YoY growth % + profit margin %. >=40 is considered healthy."""
    return yoy_growth_rate_pct + profit_margin_pct


def saas_magic_number(current_quarter_revenue: float, prior_quarter_revenue: float, prior_quarter_sm_spend: float) -> float:
    """
    SaaS Magic Number (Bessemer definition) = (quarter-over-quarter revenue growth x 4)
    / prior quarter S&M spend. >0.75 is considered efficient; >1.0 suggests spending
    more on S&M would pay off.
    """
    if prior_quarter_sm_spend <= 0:
        return 0.0
    return ((current_quarter_revenue - prior_quarter_revenue) * 4) / prior_quarter_sm_spend


def arr_per_employee(arr: float, headcount: float) -> float:
    """ARR per employee - a standard SaaS efficiency/scale metric."""
    if headcount <= 0:
        return 0.0
    return arr / headcount


# --------------------------------------------------------------------------
# Cross-border pay premium: a named, testable formula rather than a bare
# multiplier buried in a config dict. Rationale: an Indian engineer building
# for a US/EU-billed product is priced against a share of US-market willingness
# to pay, not domestic-India willingness to pay - a real cost-arbitrage effect,
# not a currency conversion. Calibrated (see README) from the cited gap between
# domestic-India SaaS pay (Freshworks/Zoho-tier ~Rs.6-12 LPA, unicorns ~Rs.15-30
# LPA) and cross-border/FAANG-tier pay (~Rs.35-60 LPA) for comparable levels.
# --------------------------------------------------------------------------

CROSS_BORDER_PREMIUM_MULTIPLIER = 1.35   # domestic = 1.00 baseline


def cross_border_premium_multiplier(target_market: str) -> float:
    """Pay multiplier for a company's target market. Domestic = 1.00 baseline;
    cross-border (US/EU-billed) = CROSS_BORDER_PREMIUM_MULTIPLIER, calibrated from
    the cited domestic-vs-cross-border India SaaS pay gap."""
    if target_market == "Cross-border":
        return CROSS_BORDER_PREMIUM_MULTIPLIER
    if target_market == "Domestic":
        return 1.00
    raise ValueError(f"Unknown target_market: {target_market!r}")


@dataclass
class MarketOpportunity:
    sam_of_tam_pct: float
    som_of_sam_pct: float
    som_of_tam_pct: float
    tier: str


def market_opportunity_tier(tam: float, sam: float, som: float) -> MarketOpportunity:
    """
    TAM/SAM/SOM sizing: how much of the addressable market is realistically
    reachable/captured. Purely a ratio read on market sizing, not a revenue projection.
    """
    sam_of_tam = (sam / tam * 100) if tam > 0 else 0.0
    som_of_sam = (som / sam * 100) if sam > 0 else 0.0
    som_of_tam = (som / tam * 100) if tam > 0 else 0.0

    if som_of_tam >= 5:
        tier = "Large near-term opportunity"
    elif som_of_tam >= 1:
        tier = "Moderate near-term opportunity"
    else:
        tier = "Niche / early-stage opportunity"

    return MarketOpportunity(
        sam_of_tam_pct=round(sam_of_tam, 2),
        som_of_sam_pct=round(som_of_sam, 2),
        som_of_tam_pct=round(som_of_tam, 2),
        tier=tier,
    )


# --------------------------------------------------------------------------
# SaaS Health Score: an authored, documented weighting scheme (not a standard
# industry formula) that turns the metrics above into one comparable number.
# Weights: Rule of 40 30% + NDR 30% + LTV:CAC 20% + ARR/employee 20%.
# --------------------------------------------------------------------------

HEALTH_SCORE_WEIGHTS = {
    "rule_of_40": 0.30,
    "ndr": 0.30,
    "ltv_cac": 0.20,
    "arr_per_employee": 0.20,
}

# ARR/employee benchmark in USD: ~$200k/FTE is a commonly cited generic efficiency
# bar for SaaS companies (many India-built, globally-sold SaaS companies report ARR
# in USD even though payroll runs in INR).
ARR_PER_EMPLOYEE_BENCHMARK_USD = 200_000


def _rule_of_40_score(rule40_value: float) -> float:
    # 40 (the "healthy" threshold) maps to 50/100; 80 maps to 100/100.
    return clamp(rule40_value / 80 * 100)


def _ndr_score(ndr_value: float) -> float:
    # 80% NDR (badly churning) -> 0; 140% NDR (elite expansion) -> 100.
    return clamp((ndr_value - 80) / 60 * 100)


def _ltv_cac_score(ratio: float) -> float:
    # 3:1 (healthy bar) -> 50/100; 6:1 -> 100/100.
    return clamp(ratio / 6 * 100)


def _arr_per_employee_score(arr_per_employee_usd: float) -> float:
    return clamp(arr_per_employee_usd / ARR_PER_EMPLOYEE_BENCHMARK_USD * 100)


@dataclass
class SaaSHealthScore:
    score: float
    band: str
    components: dict


def health_band_from_score(score: float) -> str:
    """Shared banding logic, so a health score always maps to the same band
    regardless of whether it came from submitted metrics, a funding-stage default,
    or a what-if override."""
    if score >= 80:
        return "Hypergrowth / Efficient"
    if score >= 60:
        return "Healthy"
    if score >= 40:
        return "Cash-constrained"
    return "At-risk"


def saas_health_score(
    rule40_value: float,
    ndr_value: float,
    ltv_cac_ratio: float,
    arr_per_employee_usd: float,
) -> SaaSHealthScore:
    components = {
        "rule_of_40": _rule_of_40_score(rule40_value),
        "ndr": _ndr_score(ndr_value),
        "ltv_cac": _ltv_cac_score(ltv_cac_ratio),
        "arr_per_employee": _arr_per_employee_score(arr_per_employee_usd),
    }
    score = sum(components[k] * HEALTH_SCORE_WEIGHTS[k] for k in HEALTH_SCORE_WEIGHTS)
    band = health_band_from_score(score)

    return SaaSHealthScore(score=round(score, 1), band=band, components={k: round(v, 1) for k, v in components.items()})


# --------------------------------------------------------------------------
# ESOP expected value: an explicitly-labeled HEURISTIC, not options pricing.
# Real Black-Scholes needs an observable strike price and volatility; neither
# exists for an illiquid private-company option, so pricing one that way would
# be false precision, not rigor. Instead: paper FMV x probability of a liquidity
# event x typical payout multiple on realization, both indexed to health band -
# a healthier company is both more likely to have an exit AND more likely to
# realize/exceed paper FMV there; a struggling one is discounted on both counts.
# --------------------------------------------------------------------------

ESOP_EXIT_ASSUMPTIONS = {
    "At-risk":                dict(liquidity_probability=0.15, payout_multiple=0.30),
    "Cash-constrained":       dict(liquidity_probability=0.30, payout_multiple=0.50),
    "Healthy":                dict(liquidity_probability=0.55, payout_multiple=0.80),
    "Hypergrowth / Efficient":dict(liquidity_probability=0.75, payout_multiple=1.10),
}


def esop_expected_value(esop_annualized_fmv_lpa: float, health_score: float) -> dict:
    """Risk-adjusted expected value of the ESOP estimate, on top of its paper FMV."""
    band = health_band_from_score(health_score)
    assumptions = ESOP_EXIT_ASSUMPTIONS[band]
    expected_value = esop_annualized_fmv_lpa * assumptions["liquidity_probability"] * assumptions["payout_multiple"]
    return {
        "fmv_lpa": round(esop_annualized_fmv_lpa, 2),
        "band": band,
        "liquidity_probability": assumptions["liquidity_probability"],
        "payout_multiple": assumptions["payout_multiple"],
        "expected_value_lpa": round(expected_value, 2),
    }
