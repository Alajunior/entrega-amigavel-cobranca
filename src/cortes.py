"""Cortes ALTO/MEDIO por captura, calculados FORA da amostra de treino, por fase e com data de vigência.

    ALTO  = score a partir do qual estão 80% das EAs do conjunto de calibração
    MEDIO = idem para 95%
O conjunto de calibração são safras que o modelo não viu no treino (no projeto real os cortes saíam das predições
do próprio treino e o ALTO ficava menor que o previsto). Fase com menos de 30 EAs usa o corte global.
Cada retreino acrescenta uma vigência: reescorar um dia antigo usa a regra que valia naquele dia.
"""
import math

import numpy as np
import pandas as pd

from src.config import CAPTURA_ALTO, CAPTURA_MEDIO, FASES, MIN_EA_FASE

GLOBAL = "__GLOBAL__"


def corte_captura(score, y, captura: float) -> float:
    s = np.sort(np.asarray(score, float)[np.asarray(y) == 1])[::-1]
    if len(s) == 0:
        return float("nan")
    return float(s[math.ceil(captura * len(s)) - 1])


def calcular_cortes(score, y, fase, vigente_desde, captura_alto: float = CAPTURA_ALTO,
                    captura_medio: float = CAPTURA_MEDIO, min_ea: int = MIN_EA_FASE) -> pd.DataFrame:
    score, y, fase = np.asarray(score, float), np.asarray(y, float), np.asarray(fase)
    linhas = [{"fase": GLOBAL, "corte_alto": corte_captura(score, y, captura_alto),
               "corte_medio": corte_captura(score, y, captura_medio), "n_ea": int(y.sum()), "origem": "global"}]
    for f in FASES:
        m = fase == f
        n_ea = int(y[m].sum())
        if n_ea >= min_ea:
            linhas.append({"fase": f, "corte_alto": corte_captura(score[m], y[m], captura_alto),
                           "corte_medio": corte_captura(score[m], y[m], captura_medio), "n_ea": n_ea, "origem": "fase"})
        else:
            linhas.append({**linhas[0], "fase": f, "n_ea": n_ea})
    return pd.DataFrame(linhas).assign(vigente_desde=pd.Timestamp(vigente_desde))


def cortes_vigentes(tabela: pd.DataFrame, D) -> pd.DataFrame:
    t = tabela[tabela["vigente_desde"] <= pd.Timestamp(D)]
    if t.empty:
        return t
    return t.sort_values("vigente_desde").groupby("fase", as_index=False).last()


def classificar(score, fase, cortes_dia: pd.DataFrame) -> np.ndarray:
    c = cortes_dia.set_index("fase")
    f = pd.Series(np.asarray(fase)).where(lambda x: x.isin(c.index), GLOBAL)
    alto, medio = f.map(c["corte_alto"]).to_numpy(), f.map(c["corte_medio"]).to_numpy()
    s = np.asarray(score, float)
    return np.where(s >= alto, "ALTO", np.where(s >= medio, "MEDIO", "BAIXO"))


def pct_alto_esperado(score, fase, cortes: pd.DataFrame) -> float:
    return float((classificar(score, fase, cortes) == "ALTO").mean())
