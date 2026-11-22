"""Métricas para classe rara: com ~0,6% de positivos a AUC engana; o que importa é precisão e recall no topo."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TOPS = (5, 10, 20)


def metricas(y, score) -> dict:
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    ordem = np.argsort(-s, kind="stable")
    taxa = float(y.mean())
    prec, rec = {}, {}
    for t in TOPS:
        k = max(1, int(round(len(y) * t / 100)))
        acertos = y[ordem[:k]].sum()
        prec[t] = float(acertos / k)
        rec[t] = float(acertos / y.sum()) if y.sum() else None
    return {"n": int(len(y)), "positivos": int(y.sum()), "taxa_base": taxa,
            "pr_auc": float(average_precision_score(y, s)), "auc": float(roc_auc_score(y, s)),
            "precisao_top": prec, "recall_top": rec, "lift_top10": prec[10] / taxa if taxa else None}
