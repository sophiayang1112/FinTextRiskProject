"""As-of retrieval, event extraction, source weighting, and evidence provenance."""
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EVENTS = {"liquidity shortfall": "liquidity", "late payment": "payment_delay",
          "revenue decline": "revenue", "covenant breach": "covenant", "layoff": "employment"}
POSITIVE = ("stable revenue", "debt repaid", "cash reserve")


def clean_text(value):
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", str(value))
    text = re.sub(r"(?<!\w)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\w)", "[PHONE]", text)
    return " ".join(text.split())


def mine(text):
    lowered = clean_text(text).lower()
    events = [name for phrase, name in EVENTS.items()
              if phrase in lowered and not re.search(r"\b(?:no|without)\s+" + re.escape(phrase), lowered)]
    positive = sum(phrase in lowered for phrase in POSITIVE)
    uncertainty = sum(lowered.count(word) for word in ["may", "uncertain", "potential", "unconfirmed"])
    logits = np.array([positive, len(events), 1.], dtype=float)
    p = np.exp(logits-logits.max()); p /= p.sum()
    return {"event_score": len(events)/len(EVENTS), "sentiment": float(p[0]-p[1]),
            "uncertainty": min(uncertainty/4, 1.), "entropy": float(-np.sum(p*np.log(p))),
            "events": events, "word_count": len(lowered.split())}


class EvidenceStore:
    def __init__(self, documents, fit_before):
        required = {"doc_id", "entity_id", "published_at", "available_at", "quality", "text"}
        if not required <= set(documents): raise ValueError("Document schema incomplete")
        docs = documents.copy()
        if docs.doc_id.isna().any() or docs.doc_id.duplicated().any(): raise ValueError("Document IDs must be unique")
        docs["published_at"] = pd.to_datetime(docs.published_at, utc=True)
        docs["available_at"] = pd.to_datetime(docs.available_at, utc=True)
        if docs[["published_at", "available_at", "entity_id"]].isna().any().any(): raise ValueError("Missing document identity/time")
        if (docs.available_at < docs.published_at).any(): raise ValueError("Availability precedes publication")
        if not pd.to_numeric(docs.quality).between(0, 1).all(): raise ValueError("Source quality outside [0,1]")
        docs["text"] = docs.text.map(clean_text)
        docs["content_hash"] = docs.text.str.lower().map(lambda text: hashlib.sha256(text.encode()).hexdigest())
        docs = docs.sort_values(["available_at", "doc_id"]).drop_duplicates(["entity_id", "content_hash"], keep="first").reset_index(drop=True)
        before = pd.Timestamp(fit_before)
        if before.tzinfo is None: before = before.tz_localize("UTC")
        fit_docs = docs.loc[(docs.available_at < before) & (docs.published_at < before), "text"]
        if fit_docs.empty: raise ValueError("No documents available to fit retrieval vocabulary")
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1500)
        self.vectorizer.fit(fit_docs)
        self.matrix = self.vectorizer.transform(docs.text)
        self.documents = docs
        self.fit_before = before

    def retrieve(self, entity_id, as_of, query, top_k=3, decay=.005):
        if top_k < 1 or decay < 0: raise ValueError("Invalid retrieval parameters")
        time = pd.Timestamp(as_of)
        if time.tzinfo is None: time = time.tz_localize("UTC")
        docs = self.documents
        eligible = np.flatnonzero(((docs.entity_id == entity_id) & (docs.available_at <= time)
                                    & (docs.published_at <= time)).to_numpy())
        if not len(eligible): return []
        similarity = cosine_similarity(self.vectorizer.transform([clean_text(query)]), self.matrix[eligible]).ravel()
        selected = docs.iloc[eligible]
        age = (time-selected.published_at).dt.total_seconds().to_numpy()/86400
        weights = np.exp(-decay*age)*selected.quality.to_numpy()
        ranking = [i for i in np.argsort(-(similarity*weights), kind="stable") if similarity[i] > 0 and weights[i] > 0][:top_k]
        output = []
        for position in ranking:
            row = selected.iloc[position]
            output.append({"doc_id": str(row.doc_id), "entity_id": str(row.entity_id), "text": row.text,
                "published_at": row.published_at.isoformat(), "available_at": row.available_at.isoformat(),
                "age_days": float(age[position]), "quality": float(row.quality),
                "weight": float(weights[position]), "similarity": float(similarity[position]),
                "content_hash": row.content_hash, "events": mine(row.text)["events"]})
        return output

    def graph(self):
        nodes, edges = {}, []
        for row in self.documents.itertuples():
            entity, doc = str(row.entity_id), str(row.doc_id)
            nodes["entity:"+entity] = {"id": "entity:"+entity, "type": "entity"}
            nodes["document:"+doc] = {"id": "document:"+doc, "type": "document", "published_at": row.published_at.isoformat(), "available_at": row.available_at.isoformat()}
            edges.append({"source": "entity:"+entity, "target": "document:"+doc, "relation": "has_evidence"})
            for event in mine(row.text)["events"]:
                nodes["event:"+event] = {"id": "event:"+event, "type": "risk_event"}
                edges.append({"source": "document:"+doc, "target": "event:"+event, "relation": "mentions"})
        return {"nodes": list(nodes.values()), "edges": edges}
