import numpy as np
import pandas as pd
from .evidence import EVENTS, POSITIVE

STRUCTURED = ["debt_to_income", "annual_income", "principal", "delinquencies", "rate"]
TEXT_SIGNALS = ["event_score", "sentiment", "uncertainty", "entropy"]
RETRIEVAL_SIGNALS = ["retrieved_event", "retrieved_sentiment", "source_strength", "evidence_count"]


def simulate(rows=1800, seed=19):
    if rows < 600: raise ValueError("At least 600 rows required")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01", tz="UTC")
    applicants, docs = [], []
    phrases = list(EVENTS)
    for i in range(rows):
        issued = start + pd.Timedelta(days=int(i*2190/rows))
        entity = f"ENTITY{i:05d}"
        stress = rng.beta(2, 4)
        adverse = rng.random(5) < stress
        notes = [phrases[j] for j in range(5) if adverse[j]]
        text = "Applicant operating update: " + ("; ".join(notes) if notes else "stable revenue; cash reserve")
        if rng.random() < .3: text += ". Outlook may be uncertain."
        dti = float(rng.uniform(.08, .62)); income = float(rng.lognormal(11.2, .45))
        delinq = int(rng.poisson(.3)); evidence_events = int(rng.binomial(3, stress))
        for number, age in enumerate([250, 70, 12, -30]):
            evidence_text = "Issuer financial update: "
            if number == 3: evidence_text += "Future outcome announcement not available at assessment."
            elif number < evidence_events: evidence_text += phrases[(number+i)%5] + "; potential cash pressure."
            else: evidence_text += POSITIVE[(number+i)%3] + "; operations continue."
            published = issued-pd.Timedelta(days=age)
            docs.append({"doc_id": f"D{i:05d}_{number}", "entity_id": entity,
                "published_at": published, "available_at": published+pd.Timedelta(days=2),
                "quality": float(rng.uniform(.6, 1)), "text": evidence_text})
        eta = -3.2+3*(dti-.2)+.65*delinq+.35*sum(adverse)+.4*evidence_events
        label = int(rng.random() < 1/(1+np.exp(-eta)))
        applicants.append({"sample_id": f"S{i:05d}", "entity_id": entity, "assessed_at": issued,
            "label_available_at": issued+pd.Timedelta(days=90), "debt_to_income": dti,
            "annual_income": income, "principal": income*float(rng.uniform(.1, .8)),
            "delinquencies": delinq, "rate": float(rng.uniform(.04, .23)),
            "narrative": text, "label_default": label})
    return pd.DataFrame(applicants), pd.DataFrame(docs)


def chronological_partitions(applicants):
    ordered = applicants.sort_values("assessed_at").index.to_numpy()
    n = len(ordered); cuts = [int(n*.55), int(n*.70), int(n*.85)]
    groups = dict(zip(["train", "validation", "calibration", "test"], np.split(ordered,cuts)))
    for name,next_name in [("train","validation"),("validation","calibration"),("calibration","test")]:
        cutoff = applicants.loc[groups[next_name], "assessed_at"].min()
        indices = groups[name]
        groups[name] = indices[(applicants.loc[indices,"label_available_at"] <= cutoff).to_numpy()]
    return groups
