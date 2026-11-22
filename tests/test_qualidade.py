import numpy as np
import pandas as pd

from src.qualidade import backtest_producao, estabilidade_mensal, ordenacao_semanal


def _esc(taxas_por_semana, n=2000, seed=0):
    """Escoragem sintética: para cada semana, taxas (ALTO, MEDIO, BAIXO) e 5 dias úteis."""
    rng = np.random.default_rng(seed)
    partes = []
    for i, tx in enumerate(taxas_por_semana):
        for d in pd.bdate_range(pd.Timestamp("2026-01-05") + pd.Timedelta(weeks=i), periods=5):
            classe = rng.choice(["ALTO", "MEDIO", "BAIXO"], n, p=[.1, .2, .7])
            p = pd.Series(classe).map(dict(zip(["ALTO", "MEDIO", "BAIXO"], tx))).to_numpy()
            y = (rng.random(n) < p).astype(float)
            partes.append(pd.DataFrame({"dia": d, "classe": classe, "ea_60d": y, "maduro": True,
                                        "score": p + rng.random(n) * 0.01}))
    return pd.concat(partes, ignore_index=True)


def test_backtest_so_dias_maduros():
    e = _esc([(0.05, 0.01, 0.001)] * 2)
    e.loc[e["dia"] >= "2026-01-12", "maduro"] = False
    e.loc[~e["maduro"], "ea_60d"] = np.nan
    b = backtest_producao(e).set_index("classe")
    assert list(b.index) == ["ALTO", "MEDIO", "BAIXO"]
    assert b["contratos_dia"].sum() == 5 * 2000
    assert b.loc["ALTO", "lift"] > 1 > b.loc["BAIXO", "lift"]
    assert abs(b["part_eas"].sum() - 1) < 1e-9


def test_inversao_detectada():
    o = ordenacao_semanal(_esc([(0.05, 0.01, 0.003), (0.01, 0.05, 0.003)]), min_ea=5)
    assert o["status"].tolist() == ["perfeita", "invertida"]


def test_semana_sem_dados():
    o = ordenacao_semanal(_esc([(0.05, 0.01, 0.0)]), min_ea=5)
    assert o["status"].tolist() == ["sem dados"]


def test_estabilidade_mensal():
    s = estabilidade_mensal(_esc([(0.05, 0.01, 0.001)] * 3))
    assert {"mes", "pr_auc", "auc", "recall_alto", "pct_alto"} <= set(s.columns)
    assert s["auc"].between(0.5, 1).all()
