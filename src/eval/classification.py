import numpy as np

def l2norm(e):
    n=np.linalg.norm(e,axis=1,keepdims=True)
    return e/np.maximum(n,1e-12)
def knn_classify(pool_e,pool_y,query_e,k=5):
    sim=l2norm(query_e)@l2norm(pool_e).T
    topk=np.argsort(-sim,axis=1)[:,:k]
    preds=[]
    for row in topk:
        votes={}
        for rank,i in enumerate(row):
            lab=pool_y[i]
            votes[lab]=votes.get(lab,0.0)+1.0/(rank+1)
        preds.append(max(votes,key=votes.get))
    return np.array(preds)
def macro_f1(y_true, y_pred):
    f1s = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))