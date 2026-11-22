"""Teste de vazamento: o mesmo contrato aparece em várias safras. Validação cruzada aleatória espalha as fotos do
mesmo contrato entre treino e teste e fica otimista; GroupKFold por contrato é o que vale."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from src.base_analitica import ALVO, CAT, NUM
from src.candidatos import bases_das_safras, separar_oot
from src.metricas import metricas
from src.modelos import criar_modelos


def teste_vazamento(tabelas: dict, bases: dict | None = None, n_dobras: int = 5) -> dict:
    bases = bases_das_safras(tabelas) if bases is None else bases
    treino, _ = separar_oot(sorted(bases))
    df = pd.concat(bases[s] for s in treino).reset_index(drop=True)
    X, y, g = df[NUM + CAT], df[ALVO].to_numpy(), df["id_contrato"].to_numpy()
    esquemas = {"aleatorio": list(KFold(n_dobras, shuffle=True, random_state=42).split(X)),
                "grupo": list(GroupKFold(n_dobras).split(X, y, g))}
    out = {}
    for nome, m in criar_modelos().items():
        r = {}
        for esq, dobras in esquemas.items():
            aucs, prs = [], []
            for tr, te in dobras:
                p = clone(m).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                mt = metricas(y[te], p)
                aucs.append(mt["auc"])
                prs.append(mt["pr_auc"])
            r[esq] = {"auc": float(np.mean(aucs)), "pr_auc": float(np.mean(prs))}
        r["diferenca_auc"] = r["aleatorio"]["auc"] - r["grupo"]["auc"]
        r["diferenca_pr_auc"] = r["aleatorio"]["pr_auc"] - r["grupo"]["pr_auc"]
        out[nome] = r
    out["n_linhas"], out["n_contratos"] = int(len(df)), int(df["id_contrato"].nunique())
    return out
