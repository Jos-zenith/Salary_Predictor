"""
Simulated live market-signal feed.

This is explicitly a SIMULATION, not a live external connection. Building a real one
would need paid Crunchbase/Tracxn API access and continuously-running scraping
infrastructure this project has neither the credentials nor the budget for — and a
"live" panel with nothing real behind it is worse than not having one. What this module
demonstrates honestly is the *event-driven architecture*: a background emitter produces
market-style events, they're held as small bounded in-memory adjustments, and a
prediction reads the current adjustment for its segment. Swapping `_random_event()`
for a real funding/hiring-signal webhook consumer is the intended extension point.

State is in-memory and resets on process restart. Nothing here calls out to the network.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

EVENT_POOL = [
    ("funding", "{company} raised a {round} round - funding-stage pay band shifts up", "funding_stage_up"),
    ("hiring", "Hiring surge detected for {skill} roles in {city} - skill premium +{pct}%", "skill_premium_up"),
    ("retention", "{company}'s NDR climbed to {ndr}% - health score +{pts}, pay band firms up", "health_score_up"),
    ("attrition", "Attrition spike reported at {company} - comp band under pressure", "health_score_down"),
    ("market", "{city} SaaS hiring index up {pct}% this week", "city_premium_up"),
]

COMPANIES = ["Bloomline SaaS", "NimbusStack", "Verdant Cloud", "Orbital CRM", "Ledgerly", "Flowcart B2B"]
ROUNDS = ["Seed", "Series A", "Series B", "Series C"]
SKILLS = ["GenAI / ML", "Cloud architecture", "Cybersecurity"]
CITIES = ["Bengaluru", "Chennai", "Pune", "Hyderabad", "Gurugram"]

MAX_ADJUSTMENT = 0.06   # cap any single simulated nudge at +/-6%
DECAY_PER_TICK = 0.85   # adjustments fade over time rather than accumulating forever


@dataclass
class SignalEvent:
    ts: str
    message: str
    kind: str
    target: str | None
    delta: float


@dataclass
class MarketSignalState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: deque = field(default_factory=lambda: deque(maxlen=25))
    # bounded multiplicative adjustments, keyed by segment (e.g. a city, skill, or "global")
    adjustments: dict = field(default_factory=dict)

    def apply(self, target: str, delta: float):
        with self.lock:
            current = self.adjustments.get(target, 0.0)
            self.adjustments[target] = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, current + delta))

    def decay(self):
        with self.lock:
            for k in list(self.adjustments.keys()):
                self.adjustments[k] *= DECAY_PER_TICK
                if abs(self.adjustments[k]) < 0.002:
                    del self.adjustments[k]

    def snapshot(self):
        with self.lock:
            return dict(self.adjustments), list(self.events)

    def multiplier_for(self, city: str | None, skill: str | None) -> float:
        with self.lock:
            mult = 1.0 + self.adjustments.get("global", 0.0)
            if city:
                mult += self.adjustments.get(f"city:{city}", 0.0)
            if skill:
                mult += self.adjustments.get(f"skill:{skill}", 0.0)
            return mult


state = MarketSignalState()
_started = False
_start_lock = threading.Lock()


def _random_event() -> SignalEvent:
    """
    Draws one synthetic market event. This is the function a real integration would
    replace with a webhook/queue consumer — everything downstream (state, decay,
    SSE streaming) stays the same either way.
    """
    kind, template, effect = random.choice(EVENT_POOL)
    company = random.choice(COMPANIES)
    city = random.choice(CITIES)
    skill = random.choice(SKILLS)
    pct = random.randint(2, 6)
    ndr = random.randint(105, 128)
    pts = random.randint(3, 8)

    message = template.format(company=company, round=random.choice(ROUNDS), skill=skill, city=city, pct=pct, ndr=ndr, pts=pts)

    target_map = {
        "funding_stage_up": "global",
        "skill_premium_up": f"skill:{skill}",
        "health_score_up": "global",
        "health_score_down": "global",
        "city_premium_up": f"city:{city}",
    }
    target = target_map[effect]
    sign = -1 if effect.endswith("down") else 1
    delta = sign * (pct / 100.0) if "premium" in effect else sign * 0.01 * (pts if "health" in effect else 3)

    return SignalEvent(ts=datetime.now(timezone.utc).strftime("%H:%M:%S"), message=message, kind=kind, target=target, delta=delta)


def _emitter_loop(interval_seconds: float):
    while True:
        event = _random_event()
        with state.lock:
            state.events.append(event)
        if event.target:
            state.apply(event.target, event.delta)
        state.decay()
        time.sleep(interval_seconds)


def start_background_emitter(interval_seconds: float = 6.0):
    """Idempotent: safe to call on every Flask worker import."""
    global _started
    with _start_lock:
        if _started:
            return
        thread = threading.Thread(target=_emitter_loop, args=(interval_seconds,), daemon=True)
        thread.start()
        _started = True
