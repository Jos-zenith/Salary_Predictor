"""
Rule-based Indian CTC breakup: converts a predicted annual cash CTC (in Lakhs) into
its statutory components. Deterministic, not learned — the ML model predicts the
total; this module splits it the way an Indian payroll structure actually would.

Rules cited (see README.md for sources):
- Basic + DA must be >=50% of CTC under India's Code on Wages (enforced 2026-04-01).
- HRA is typically 50% of Basic in metro cities, 40% in non-metro.
- Employer EPF contribution = 12% of Basic (+DA).
- Gratuity accrual ~= 4.81% of Basic.
- LTA ~= one month of Basic per year (~8.33%) is a common convention, not a statute.
- A ~10% performance-linked variable/bonus target is typical for SaaS/product roles.
Special Allowance is the balancing figure so the components sum exactly to the CTC.
"""

from __future__ import annotations

from dataclasses import dataclass

METRO_CITIES = {"Bengaluru", "Chennai", "Hyderabad", "Gurugram", "Pune"}

BASIC_PCT = 0.50
HRA_METRO_PCT = 0.50   # of Basic
HRA_NON_METRO_PCT = 0.40  # of Basic
EPF_EMPLOYER_PCT = 0.12   # of Basic
GRATUITY_PCT = 0.0481     # of Basic
LTA_PCT = 0.0833          # of Basic
VARIABLE_BONUS_PCT = 0.10  # of CTC


@dataclass
class CTCBreakup:
    ctc_lpa: float
    basic: float
    hra: float
    employer_epf: float
    gratuity: float
    lta: float
    variable_bonus: float
    special_allowance: float
    is_metro: bool

    def as_rows(self) -> list[dict]:
        return [
            {"component": "Basic + DA", "lpa": self.basic},
            {"component": "HRA", "lpa": self.hra},
            {"component": "Special Allowance", "lpa": self.special_allowance},
            {"component": "LTA", "lpa": self.lta},
            {"component": "Variable / Performance Bonus", "lpa": self.variable_bonus},
            {"component": "Employer EPF Contribution", "lpa": self.employer_epf},
            {"component": "Gratuity (accrued)", "lpa": self.gratuity},
        ]


@dataclass
class ComplianceCheck:
    rule: str
    passed: bool
    detail: str


def check_ctc_compliance(
    total_ctc_lpa: float, basic_lpa: float, hra_lpa: float,
    epf_lpa: float, gratuity_lpa: float, city: str,
) -> list[ComplianceCheck]:
    """
    Checks a PROPOSED offer's already-decided components against the same rules
    `compute_ctc_breakup()` encodes (Code on Wages 50%-basic rule; HRA/EPF/Gratuity
    conventions) - a sanity check an HR user can run on an offer they're structuring,
    not a certified legal or statutory payroll audit.
    """
    is_metro = city in METRO_CITIES
    checks = []

    basic_pct = (basic_lpa / total_ctc_lpa * 100) if total_ctc_lpa > 0 else 0
    checks.append(ComplianceCheck(
        rule="Basic + DA >= 50% of CTC (India's Code on Wages, enforced 2026-04-01)",
        passed=basic_pct >= 49.5,
        detail=f"Basic is {basic_pct:.1f}% of CTC (Rs.{basic_lpa:.2f} of Rs.{total_ctc_lpa:.2f} LPA).",
    ))

    expected_hra_pct = HRA_METRO_PCT if is_metro else HRA_NON_METRO_PCT
    actual_hra_pct = (hra_lpa / basic_lpa) if basic_lpa > 0 else 0
    checks.append(ComplianceCheck(
        rule=f"HRA ~{expected_hra_pct*100:.0f}% of Basic ({'metro' if is_metro else 'non-metro'} convention for {city})",
        passed=abs(actual_hra_pct - expected_hra_pct) <= 0.05,
        detail=f"HRA is {actual_hra_pct*100:.1f}% of Basic (expected ~{expected_hra_pct*100:.0f}%, +/-5pp).",
    ))

    actual_epf_pct = (epf_lpa / basic_lpa) if basic_lpa > 0 else 0
    checks.append(ComplianceCheck(
        rule="Employer EPF = 12% of Basic",
        passed=abs(actual_epf_pct - EPF_EMPLOYER_PCT) <= 0.01,
        detail=f"EPF is {actual_epf_pct*100:.1f}% of Basic (expected 12%, +/-1pp).",
    ))

    actual_gratuity_pct = (gratuity_lpa / basic_lpa) if basic_lpa > 0 else 0
    checks.append(ComplianceCheck(
        rule="Gratuity accrual ~4.81% of Basic",
        passed=abs(actual_gratuity_pct - GRATUITY_PCT) <= 0.01,
        detail=f"Gratuity is {actual_gratuity_pct*100:.2f}% of Basic (expected ~4.81%, +/-1pp).",
    ))

    components_total = basic_lpa + hra_lpa + epf_lpa + gratuity_lpa
    checks.append(ComplianceCheck(
        rule="Basic + HRA + EPF + Gratuity does not exceed total CTC",
        passed=components_total <= total_ctc_lpa + 0.01,
        detail=f"These four components sum to Rs.{components_total:.2f} LPA against a Rs.{total_ctc_lpa:.2f} LPA CTC "
               f"(remainder of Rs.{total_ctc_lpa - components_total:.2f} LPA is available for LTA/Special "
               f"Allowance/Bonus).",
    ))

    return checks


def compute_ctc_breakup(ctc_lpa: float, city: str) -> CTCBreakup:
    is_metro = city in METRO_CITIES
    basic = ctc_lpa * BASIC_PCT
    hra = basic * (HRA_METRO_PCT if is_metro else HRA_NON_METRO_PCT)
    employer_epf = basic * EPF_EMPLOYER_PCT
    gratuity = basic * GRATUITY_PCT
    lta = basic * LTA_PCT
    variable_bonus = ctc_lpa * VARIABLE_BONUS_PCT

    allocated = basic + hra + employer_epf + gratuity + lta + variable_bonus
    special_allowance = max(ctc_lpa - allocated, 0.0)

    return CTCBreakup(
        ctc_lpa=round(ctc_lpa, 2),
        basic=round(basic, 2),
        hra=round(hra, 2),
        employer_epf=round(employer_epf, 2),
        gratuity=round(gratuity, 2),
        lta=round(lta, 2),
        variable_bonus=round(variable_bonus, 2),
        special_allowance=round(special_allowance, 2),
        is_metro=is_metro,
    )
