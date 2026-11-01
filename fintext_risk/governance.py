import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import f1_score,matthews_corrcoef,average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix


def fit_temperature(logits,labels):
    logits=np.asarray(logits);labels=np.asarray(labels)
    objective=lambda log_t:float(np.mean(np.logaddexp(0,logits/np.exp(log_t))-labels*logits/np.exp(log_t)))
    result=minimize_scalar(objective,bounds=(-3,3),method="bounded")
    if not result.success:raise RuntimeError("Temperature fitting failed")
    return float(np.exp(result.x))


def conformal_cutoff(probabilities,labels,alpha=.1):
    probabilities=np.asarray(probabilities);labels=np.asarray(labels)
    scores=np.where(labels==1,1-probabilities,probabilities)
    if not len(scores):raise ValueError("Empty conformal calibration set")
    rank=int(np.ceil((len(scores)+1)*(1-alpha)))
    return 1. if rank>len(scores) else float(np.sort(scores)[rank-1])


def prediction_sets(probabilities,cutoff):
    return [[label for label,score in [(0,float(p)),(1,float(1-p))] if score<=cutoff] for p in probabilities]


def ece(labels,p,bins=10):
    result=0.
    for index in range(bins):
        mask=(p>=index/bins)&((p<(index+1)/bins) if index<bins-1 else (p<=1))
        if mask.any():result+=mask.mean()*abs(p[mask].mean()-labels[mask].mean())
    return float(result)


def metrics(labels,p,threshold=.5):
    labels=np.asarray(labels);p=np.asarray(p);pred=p>=threshold
    return {"f1":float(f1_score(labels,pred,zero_division=0)),"mcc":float(matthews_corrcoef(labels,pred)),
        "average_precision":float(average_precision_score(labels,p)),"roc_auc":float(roc_auc_score(labels,p)),
        "brier":float(brier_score_loss(labels,p)),"ece_10_bins":ece(labels,p),
        "confusion_matrix":confusion_matrix(labels,pred,labels=[0,1]).tolist()}


def cost_threshold(labels,p):
    candidates=np.linspace(0,1,201)
    costs=[float(np.mean(3*((labels==1)&(p<t))+((labels==0)&(p>=t)))) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def psi(reference,current,bins=10):
    boundaries=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)))
    boundaries[0]=-np.inf;boundaries[-1]=np.inf
    a=np.histogram(reference,bins=boundaries)[0].astype(float)+.5
    b=np.histogram(current,bins=boundaries)[0].astype(float)+.5
    a/=a.sum();b/=b.sum()
    return float(np.sum((b-a)*np.log(b/a)))
