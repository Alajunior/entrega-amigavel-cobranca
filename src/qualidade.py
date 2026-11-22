"""O modelo em produção: a classe PUBLICADA em cada dia maduro contra a formalização real nos 60 dias seguintes."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import CLASSES


def _maduro(esc: pd.DataFrame) -> pd.DataFrame:
    return esc[esc["maduro"].astype(bool) & esc["ea_60d"].notna()]


def backtest_producao(esc: pd.DataFrame) -> pd.DataFrame:
    e = _maduro(esc)
    g = e.groupby(e["classe"].astype(str), observed=True)["ea_60d"].agg(contratos_dia="size", eas="sum")
    g = g.reindex(CLASSES).fillna(0)
    taxa_geral = g["eas"].sum() / g["contratos_dia"].sum()
    g["taxa"] = g["eas"] / g["contratos_dia"].replace(0, np.nan)
    g["lift"] = g["taxa"] / taxa_geral
    g["part_eas"] = g["eas"] / g["eas"].sum()
    g["part_estoque"] = g["contratos_dia"] / g["contratos_dia"].sum()
    return g.reset_index(names="classe")


def ordenacao_semanal(esc: pd.DataFrame, min_ea: int = 5) -> pd.DataFrame:
    """Por semana (segunda-feira): taxa por classe e status perfeita / invertida / sem dados."""
    e = _maduro(esc)
    semana = e["dia"] - pd.to_timedelta(e["dia"].dt.weekday, unit="D")
    t = e.groupby([semana.rename("semana"), e["classe"].astype(str).rename("classe")], observed=True)["ea_60d"] \
        .agg(["size", "sum"]).unstack("classe")
    linhas = []
    for s, r in t.iterrows():
        eas = {c: r.get(("sum", c), 0) for c in CLASSES}
        n = {c: r.get(("size", c), 0) for c in CLASSES}
        taxa = {c: eas[c] / n[c] if n[c] else None for c in CLASSES}
        if any(pd.isna(eas[c]) or eas[c] < min_ea for c in CLASSES):
            status = "sem dados"
        elif taxa["ALTO"] < taxa["MEDIO"] or taxa["MEDIO"] < taxa["BAIXO"]:
            status = "invertida"
        else:
            status = "perfeita"
        linhas.append({"semana": s, **{f"taxa_{c}": taxa[c] for c in CLASSES}, "status": status})
    return pd.DataFrame(linhas)


def estabilidade_mensal(esc: pd.DataFrame) -> pd.DataFrame:
    e = _maduro(esc)
    linhas = []
    for mes, g in e.groupby(e["dia"].dt.strftime("%Y-%m")):
        y = g["ea_60d"].to_numpy()
        if not 0 < y.sum() < len(y):
            continue
        alto = (g["classe"].astype(str) == "ALTO").to_numpy()
        linhas.append({"mes": mes, "pr_auc": float(average_precision_score(y, g["score"])),
                       "auc": float(roc_auc_score(y, g["score"])), "recall_alto": float(y[alto].sum() / y.sum()),
                       "pct_alto": float(alto.mean()), "taxa_base": float(y.mean())})
    return pd.DataFrame(linhas)
