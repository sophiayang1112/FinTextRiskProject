import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,OneToOneFeatureMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .data import STRUCTURED, TEXT_SIGNALS, RETRIEVAL_SIGNALS
from .evidence import clean_text, mine


def make_features(applicants, store):
    required = set(STRUCTURED+["sample_id","entity_id","assessed_at","narrative"])
    if not required <= set(applicants): raise ValueError("Application schema incomplete")
    features = applicants[STRUCTURED].copy()
    for column in STRUCTURED:
        features[column] = pd.to_numeric(features[column], errors="raise")
        if np.isinf(features[column].to_numpy(dtype=float)).any(): raise ValueError("Infinite feature")
    features["narrative"] = applicants.narrative.map(clean_text)
    evidence_records=[]
    rows=[]
    for record in applicants.itertuples():
        signals=mine(record.narrative)
        retrieved=store.retrieve(record.entity_id,record.assessed_at,record.narrative)
        weight=sum(doc["weight"] for doc in retrieved)
        retrieved_event=sum(doc["weight"]*mine(doc["text"])["event_score"] for doc in retrieved)/max(weight,1e-12)
        retrieved_sentiment=sum(doc["weight"]*mine(doc["text"])["sentiment"] for doc in retrieved)/max(weight,1e-12)
        rows.append({key: signals[key] for key in TEXT_SIGNALS} | {
            "retrieved_event": retrieved_event,"retrieved_sentiment":retrieved_sentiment,
            "source_strength":weight/max(len(retrieved),1),"evidence_count":len(retrieved)})
        evidence_records.append({"sample_id":record.sample_id,"entity_id":str(record.entity_id),
            "assessed_at":pd.Timestamp(record.assessed_at).isoformat(),"evidence":retrieved})
    for column in TEXT_SIGNALS+RETRIEVAL_SIGNALS:features[column]=[row[column] for row in rows]
    return features,evidence_records


class QuantileClipper(OneToOneFeatureMixin,TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        self.bounds_=np.quantile(np.asarray(x,dtype=float),[.01,.99],axis=0)
        self.n_features_in_=np.asarray(x).shape[1]
        return self

    def transform(self,x):
        return np.clip(np.asarray(x,dtype=float),*self.bounds_)


def model(numeric, text=True, cost_sensitive=True):
    transformers=[]
    if numeric:
        transformers.append(("numeric",Pipeline([("impute",SimpleImputer(strategy="median")),
                             ("clip",QuantileClipper()),("scale",StandardScaler())]),numeric))
    if text:
        transformers.append(("text",TfidfVectorizer(ngram_range=(1,2),max_features=250),"narrative"))
    return Pipeline([("features",ColumnTransformer(transformers)),
        ("model",LogisticRegression(max_iter=2000,C=.5,class_weight={0:1,1:3} if cost_sensitive else None,
                                    random_state=19))])


def variants(include_llm=False):
    all_signals=STRUCTURED+TEXT_SIGNALS+RETRIEVAL_SIGNALS
    result={"structured_lr":model(STRUCTURED,text=False),"text_tfidf_lr":model([],text=True),
        "fusion":model(all_signals),"no_retrieval":model(STRUCTURED+TEXT_SIGNALS),
        "no_events":model(STRUCTURED+["sentiment","uncertainty","entropy"]+RETRIEVAL_SIGNALS[1:]),
        "no_cost_weight":model(all_signals,cost_sensitive=False)}
    if include_llm:result["fusion_with_llm"]=model(all_signals+["llm_risk"])
    return result


def fingerprint(record):
    parts=[str(record[key]) for key in ["sample_id","entity_id","assessed_at","narrative"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def attach_llm(features,applicants,records,evidence_records):
    if len(records)!=len(applicants):raise ValueError("One LLM record per application required")
    mapping={r["sample_id"]:r for r in records}
    if len(mapping)!=len(records):raise ValueError("Duplicate LLM sample IDs")
    scores=[]
    for (_,row),evidence in zip(applicants.iterrows(),evidence_records):
        record=mapping.get(row.sample_id)
        if record is None or record.get("input_hash")!=fingerprint(row):raise ValueError("LLM input mismatch")
        allowed={doc["doc_id"] for doc in evidence["evidence"]}
        cited=set(record.get("evidence_ids",[]))
        if not cited or not cited<=allowed:raise ValueError("Invalid LLM evidence citations")
        score=record.get("risk_score")
        if isinstance(score,bool) or not isinstance(score,(float,int)) or not np.isfinite(score) or not 0<=score<=1:
            raise ValueError("LLM score outside [0,1]")
        scores.append(score)
    result=features.copy();result["llm_risk"]=scores
    return result


def integrated_gradients(pipeline,frame):
    """Exact straight-line probability IG for a linear-logit sigmoid, zero baseline."""
    transformed=pipeline.named_steps["features"].transform(frame)
    x=transformed.toarray() if hasattr(transformed,"toarray") else np.asarray(transformed)
    coefficients=pipeline.named_steps["model"].coef_[0]
    bias=float(pipeline.named_steps["model"].intercept_[0])
    delta=x@coefficients
    baseline=1/(1+np.exp(-bias))
    p=1/(1+np.exp(-(bias+delta)))
    factor=np.divide(p-baseline,delta,out=np.full_like(delta,baseline*(1-baseline)),where=np.abs(delta)>1e-10)
    values=x*coefficients*factor[:,None]
    names=pipeline.named_steps["features"].get_feature_names_out()
    output=[]
    for row in values:
        order=np.argsort(-np.abs(row))[:8]
        output.append([{"feature":str(names[i]),"attribution":float(row[i])} for i in order])
    return output
