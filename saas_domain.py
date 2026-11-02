

from __future__ import annotations

from saas_metrics import cross_border_premium_multiplier

# funding_stage -> synthetic-data generation ranges + serving-time constants.
# `pay_mult` and `esop_ratio` are also used directly by main.py (ESOP estimate).
FUNDING_STAGES = {
    "Bootstrapped":     dict(headcount=(8, 60),    growth=(10, 30),  margin=(0, 20),   ndr=(95, 110),  ltv_cac=(2.5, 4.5), pay_mult=0.85, esop_ratio=0.00),
    "Seed":             dict(headcount=(15, 80),   growth=(50, 110), margin=(-60, -15),ndr=(85, 108),  ltv_cac=(1.5, 3.5), pay_mult=0.90, esop_ratio=0.35),
    "Series A":         dict(headcount=(40, 180),  growth=(60, 130), margin=(-45, -5), ndr=(95, 115),  ltv_cac=(2.0, 4.0), pay_mult=1.00, esop_ratio=0.25),
    "Series B":         dict(headcount=(150, 420), growth=(45, 95),  margin=(-25, 10), ndr=(100, 122), ltv_cac=(2.5, 4.5), pay_mult=1.15, esop_ratio=0.18),
    "Series C+":        dict(headcount=(350, 1200),growth=(30, 70),  margin=(-10, 18), ndr=(102, 125), ltv_cac=(3.0, 5.5), pay_mult=1.35, esop_ratio=0.12),
    "Pre-IPO / Public": dict(headcount=(1000, 5000),growth=(15, 40), margin=(5, 25),   ndr=(100, 118), ltv_cac=(3.0, 5.0), pay_mult=1.55, esop_ratio=0.06),
}

TARGET_MARKETS = {
    "Domestic": cross_border_premium_multiplier("Domestic"),
    "Cross-border": cross_border_premium_multiplier("Cross-border"),
}

PRICING_MODELS = {
    "Subscription": 1.00,
    "Tiered": 1.02,
    "Per-user": 1.02,
    "Freemium": 0.98,
    "Pay-as-you-go": 0.98,
}

# City -> (SaaS hub tier label, pay multiplier used only at data-generation time).
CITIES = {
    "Bengaluru":  ("Tier 1 SaaS Hub", 1.20),
    "Gurugram":   ("Tier 1 SaaS Hub", 1.12),
    "Chennai":    ("Tier 2 SaaS Hub", 1.00),
    "Pune":       ("Tier 2 SaaS Hub", 0.98),
    "Hyderabad":  ("Tier 2 SaaS Hub", 1.00),
    "Kochi":      ("Emerging Hub", 0.88),
    "Jaipur":     ("Emerging Hub", 0.85),
    "Coimbatore": ("Emerging Hub", 0.85),
    "Indore":     ("Emerging Hub", 0.84),
    "Remote (other)": ("Emerging Hub", 0.90),
}

ROLE_ARCHETYPES = {
    "Product": {
        "Software Engineer": 9.0,
        "Data Scientist": 11.0,
        "ML / AI Engineer": 12.5,
        "DevOps / Cloud Engineer": 10.0,
        "Product Manager": 12.0,
        "QA Engineer": 7.0,
        "UI/UX Designer": 8.0,
    },
    "GTM": {
        "Sales Development Rep (SDR)": 6.0,
        "Account Executive": 8.0,
        "Customer Success Manager": 7.0,
        "Marketing Manager": 8.0,
        "Enterprise Sales Manager": 11.0,
    },
}

SKILL_CATEGORIES = {
    "Core": 1.00,
    "Cloud architecture": 1.10,
    "GenAI / ML": 1.18,
    "Cybersecurity": 1.08,
    "Data engineering": 1.08,
    "Legacy / traditional ops": 0.95,
}

EDUCATION_PEDIGREE = {
    "Tier-1 institute (IIT/IIM/BITS)": 1.08,
    "Recognised university": 1.02,
    "Regional university": 1.00,
    "Skills-first / bootcamp": 0.96,
}

LEVEL_BANDS = [
    (0, 2, "Junior", 1.00),
    (2, 5, "Mid", 1.65),
    (5, 9, "Senior", 2.60),
    (9, 100, "Lead / Principal", 3.80),
]

METRO_CITIES = {"Bengaluru", "Chennai", "Hyderabad", "Gurugram", "Pune"}


def experience_to_level(years: float) -> str:
    for lo, hi, name, _mult in LEVEL_BANDS:
        if lo <= years < hi:
            return name
    return LEVEL_BANDS[-1][2]


def experience_to_level_mult(years: float) -> float:
    for lo, hi, name, mult in LEVEL_BANDS:
        if lo <= years < hi:
            band_width = hi - lo if hi < 100 else 6
            progress = min((years - lo) / band_width, 1.0)
            return mult * (1 + 0.15 * progress)
    return LEVEL_BANDS[-1][3]


def role_archetype_for(role: str) -> str | None:
    for archetype, roles in ROLE_ARCHETYPES.items():
        if role in roles:
            return archetype
    return None
