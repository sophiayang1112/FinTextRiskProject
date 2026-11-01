import argparse
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import simulate,chronological_partitions,STRUCTURED
from .evidence import EvidenceStore,clean_text,mine
from .modeling import make_features,variants,integrated_gradients,attach_llm,fingerprint
from .governance import fit_temperature,conformal_cutoff,prediction_sets,metrics,cost_threshold,psi
from .llm import assess,PROMPT_VERSION


def dataset(rows,seed):
    applicants,documents=simulate(rows,seed)
    parts=chronological_partitions(applicants)
    fit_before=applicants.loc[parts["validation"],"assessed_at"].min()
    store=EvidenceStore(documents,fit_before)
    return applicants,store,parts


def train(output="reports",rows=1800,seed=19,llm_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    applicants,store,parts=dataset(rows,seed)
    started=time.perf_counter();features,evidence=make_features(applicants,store)
    extraction_seconds=time.perf_counter()-started
    if llm_path:
        records=[json.loads(line) for line in Path(llm_path).read_text().splitlines() if line.strip()]
        features=attach_llm(features,applicants,records,evidence)
    tr,va,ca,te=[parts[key] for key in ["train","validation","calibration","test"]]
    mid=len(ca)//2;temperature_ids,conformal_ids=ca[:mid],ca[mid:]
    y=applicants.label_default
    for name,ids in list(parts.items())+[("temperature",temperature_ids),("conformal",conformal_ids)]:
        if len(ids)<15 or y.iloc[ids].nunique()<2:raise ValueError(f"Insufficient outcomes in {name}; increase rows")
    models=variants(bool(llm_path));results={};settings={};probabilities={}
    # Architecture is specified in advance, not selected on test performance.
    primary="fusion_with_llm" if llm_path else "fusion"
    for name,model in models.items():
        model.fit(features.iloc[tr],y.iloc[tr])
        logits=model.decision_function(features.iloc[temperature_ids])
        temperature=fit_temperature(logits,y.iloc[temperature_ids])
        valid=expit(model.decision_function(features.iloc[va])/temperature)
        threshold=cost_threshold(y.iloc[va].to_numpy(),valid)
        start=time.perf_counter();test=expit(model.decision_function(features.iloc[te])/temperature)
        elapsed=time.perf_counter()-start
        results[name]=metrics(y.iloc[te],test,threshold)
        results[name]["model_scoring_ms_per_row"]=1000*elapsed/len(te)
        settings[name]={"temperature":temperature,"threshold":threshold}
        probabilities[name]=test
    fitted=models[primary];temperature=settings[primary]["temperature"]
    raw_test=expit(fitted.decision_function(features.iloc[te]))
    raw_threshold=cost_threshold(y.iloc[va].to_numpy(),expit(fitted.decision_function(features.iloc[va])))
    results["no_calibration"]=metrics(y.iloc[te],raw_test,raw_threshold)
    p=probabilities[primary];threshold=settings[primary]["threshold"]
    conformal_p=expit(fitted.decision_function(features.iloc[conformal_ids])/temperature)
    cutoff=conformal_cutoff(conformal_p,y.iloc[conformal_ids].to_numpy())
    sets=prediction_sets(p,cutoff)
    coverage=float(np.mean([int(label) in values for label,values in zip(y.iloc[te],sets)]))
    attributions=integrated_gradients(fitted,features.iloc[te[:8]])
    audits=[]
    for offset,index in enumerate(te[:8]):
        record=evidence[index]
        audits.append({"sample_id":record["sample_id"],"assessed_at":record["assessed_at"],
            "backend":"local_llm_fusion" if llm_path else "tfidf_event_retrieval_fusion",
            "default_probability":float(p[offset]),"prediction_set":sets[offset],
            "abstain":len(sets[offset])!=1 or not record["evidence"],
            "evidence":record["evidence"],"attribution_target":"uncalibrated fusion probability",
            "integrated_gradients":attributions[offset]})
    predicted=p>=threshold;actual=y.iloc[te].to_numpy()
    text_length=features.iloc[te].narrative.str.split().map(len).to_numpy()
    age=np.array([max([doc["age_days"] for doc in evidence[i]["evidence"]],default=0) for i in te])
    conditions={"short_text":text_length<20,"high_dti":features.iloc[te].debt_to_income.to_numpy()>.35,
                "old_evidence":age>180,"all_samples":np.ones(len(te),dtype=bool)}
    errors={}
    for name,mask in conditions.items():
        errors[name]={"samples":int(mask.sum()),"false_positive":int(np.sum(mask&predicted&(actual==0))),
                     "false_negative":int(np.sum(mask&~predicted&(actual==1)))}
    changed=features.iloc[te].copy();changed["debt_to_income"]+=.1
    shifted=expit(fitted.decision_function(changed)/temperature)
    train_dti=features.iloc[tr].debt_to_income.to_numpy()
    q25=features.iloc[tr].annual_income.quantile(.25)
    low=features.iloc[te].annual_income.to_numpy()<=q25
    report={"data_source":"synthetic multi-source financial text and structured records", "seed":seed,"rows":rows,
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__},
        "backend":"local_llm_fusion" if llm_path else "offline_text_retrieval_fusion",
        "live_llm_executed":bool(llm_path),"primary_variant":primary,
        "partitions_after_label_maturity_purge":{name:len(ids) for name,ids in parts.items()},
        "periods":{name:{"start":applicants.loc[ids,"assessed_at"].min().isoformat(),
                         "end":applicants.loc[ids,"assessed_at"].max().isoformat()} for name,ids in parts.items()},
        "temperature_rows":len(temperature_ids),"conformal_rows":len(conformal_ids),
        "threshold_costs":{"false_negative":3,"false_positive":1},"model_settings":settings,
        "holdout_metrics":results,"feature_extraction_ms_per_row":extraction_seconds/rows*1000,
        "conformal":{"nominal_coverage":.9,"observed_coverage":coverage,"score_cutoff":cutoff,
            "mean_set_size":float(np.mean([len(s) for s in sets])),
            "abstention_rate":float(np.mean([len(s)!=1 for s in sets])),
            "interpretation":"empirical label-set coverage; temporal exchangeability not assumed"},
        "error_buckets":errors,"stress_test":{"scenario":"DTI increased by 0.10 on holdout inputs",
            "mean_probability_change":float(np.mean(shifted-p)),"shifted_metrics":metrics(actual,shifted,threshold),
            "dti_psi":psi(train_dti,features.iloc[te].debt_to_income.to_numpy()+.1)},
        "income_q1_slice":{"rows":int(low.sum()),"metrics":metrics(actual[low],p[low],threshold)},
        "scope_notes":["Synthetic results are separate from paper tables.","No FinBERT/CRF/fine-tuning run is represented by the offline baseline.",
                       "LLM model pretraining chronology requires separate assessment."]}
    bundle={"pipeline":fitted,"store":store,"temperature":temperature,"threshold":threshold,"conformal_cutoff":cutoff,
        "backend":report["backend"],"llm_required":bool(llm_path),"seed":seed,"rows":rows}
    joblib.dump(bundle,output/"risk_model.joblib")
    applicants.iloc[te[:8]].drop(columns=["label_default","label_available_at"]).to_csv(output/"applications.csv",index=False)
    (output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (output/"audit_examples.json").write_text(json.dumps(audits,indent=2),encoding="utf-8")
    graph=store.graph();allowed_docs={"document:"+d["doc_id"] for record in audits for d in record["evidence"]}
    graph["edges"]=[edge for edge in graph["edges"] if edge["source"] in allowed_docs or edge["target"] in allowed_docs]
    allowed={edge[key] for edge in graph["edges"] for key in ["source","target"]}
    graph["nodes"]=[node for node in graph["nodes"] if node["id"] in allowed]
    (output/"evidence_graph.json").write_text(json.dumps(graph,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    observed,mean=calibration_curve(actual,p,n_bins=8,strategy="quantile")
    observed_raw,mean_raw=calibration_curve(actual,raw_test,n_bins=8,strategy="quantile")
    axes[0].plot([0,1],[0,1],"--",color="gray");axes[0].plot(mean_raw,observed_raw,"o-",label="raw")
    axes[0].plot(mean,observed,"o-",label="temperature scaled");axes[0].legend()
    axes[0].set(xlabel="Mean predicted default probability",ylabel="Observed default fraction",title="Holdout reliability")
    names=list(results);positions=np.arange(len(names));width=.25
    for shift,key in [(-1,"f1"),(0,"mcc"),(1,"average_precision")]:
        axes[1].bar(positions+shift*width,[results[n][key] for n in names],width,label=key)
    axes[1].set_xticks(positions,names,rotation=35,ha="right");axes[1].legend(fontsize=8);axes[1].set_title("Baselines and ablations")
    fig.suptitle("Synthetic chronological holdout | offline and LLM runs identified separately")
    fig.tight_layout();fig.savefig(output/"evaluation.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    names=list(errors);positions=np.arange(len(names))
    ax.bar(positions-.2,[errors[n]["false_positive"] for n in names],.4,label="FP")
    ax.bar(positions+.2,[errors[n]["false_negative"] for n in names],.4,label="FN")
    ax.set_xticks(positions,names);ax.set_ylabel("Count");ax.set_title("Overlapping error buckets");ax.legend()
    fig.tight_layout();fig.savefig(output/"error_buckets.png",dpi=150);plt.close(fig)
    print(json.dumps({"backend":report["backend"],"primary":results[primary],"coverage":coverage,"output":str(output)},indent=2))
    return report


def score(model_path,application_path,output_path,llm_path=None):
    bundle=joblib.load(model_path)
    applicants=pd.read_csv(application_path)
    features,evidence=make_features(applicants,bundle["store"])
    if bundle["llm_required"]:
        if not llm_path:raise ValueError("This model requires matching LLM signal records")
        records=[json.loads(line) for line in Path(llm_path).read_text().splitlines() if line.strip()]
        features=attach_llm(features,applicants,records,evidence)
    p=expit(bundle["pipeline"].decision_function(features)/bundle["temperature"])
    sets=prediction_sets(p,bundle["conformal_cutoff"])
    result=[]
    for index,record in enumerate(evidence):
        result.append({"sample_id":record["sample_id"],"default_probability":float(p[index]),
            "review_flag":bool(p[index]>=bundle["threshold"]),"prediction_set":sets[index],
            "abstain":len(sets[index])!=1 or not record["evidence"],
            "evidence_ids":[d["doc_id"] for d in record["evidence"]],"backend":bundle["backend"]})
    Path(output_path).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Saved {len(result)} risk assessments to {output_path}")


def enrich(output,rows,seed,model_name):
    applicants,store,_=dataset(rows,seed)
    with Path(output).open("w",encoding="utf-8") as stream:
        for _,row in applicants.iterrows():
            evidence=store.retrieve(row.entity_id,row.assessed_at,row.narrative)
            application={key:float(row[key]) for key in STRUCTURED}|{"narrative":clean_text(row.narrative)}
            response=assess(application,evidence,model_name)
            record={"sample_id":row.sample_id,"input_hash":fingerprint(row),**response}
            stream.write(json.dumps(record)+"\n");stream.flush()
    print("Local LLM signal extraction complete")


def main():
    parser=argparse.ArgumentParser(description="Financial text and evidence risk assessment")
    sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("train");fit.add_argument("--output",default="reports")
    fit.add_argument("--rows",type=int,default=1800);fit.add_argument("--seed",type=int,default=19)
    fit.add_argument("--llm-signals")
    predict=sub.add_parser("score");predict.add_argument("--model",default="reports/risk_model.joblib")
    predict.add_argument("--applications",default="reports/applications.csv");predict.add_argument("--output",default="assessments.json")
    predict.add_argument("--llm-signals")
    local=sub.add_parser("llm-enrich");local.add_argument("--model-name",required=True)
    local.add_argument("--rows",type=int,default=1800);local.add_argument("--seed",type=int,default=19)
    local.add_argument("--output",default="llm_signals.jsonl")
    args=parser.parse_args()
    if args.command=="train":train(args.output,args.rows,args.seed,args.llm_signals)
    elif args.command=="score":score(args.model,args.applications,args.output,args.llm_signals)
    else:enrich(args.output,args.rows,args.seed,args.model_name)


if __name__=="__main__":main()
