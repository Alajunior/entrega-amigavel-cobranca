import numpy as np
import pandas as pd

from src.cortes import calcular_cortes, classificar, corte_captura, cortes_vigentes, pct_alto_esperado
from src.config import FASES


def test_corte_captura_exato():
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    assert corte_captura(s, y, 0.8) == 0.6
    assert np.isnan(corte_captura(s, np.zeros(7), 0.8))


def _dados(seed=0, n=40000):
    rng = np.random.default_rng(seed)
    fase = rng.choice(FASES[1:3], n)                       # 0031-0090 e 0361-0720 ficam sem dados
    s = rng.random(n)
    y = (rng.random(n) < 0.02 * s * 2).astype(float)
    return s, y, fase


def test_fase_sem_ea_usa_global():
    s, y, fase = _dados()
    fase = fase.copy()
    fase[:50] = "0031-0090"
    y[:50] = 0                                            # 0031-0090 presente, mas sem nenhuma EA
    t = calcular_cortes(s, y, fase, pd.Timestamp("2025-12-01"))
    g = t.set_index("fase").loc["__GLOBAL__"]
    for f in ("0031-0090", "0361-0720"):
        r = t.set_index("fase").loc[f]
        assert r["origem"] == "global" and r["corte_alto"] == g["corte_alto"] and r["corte_medio"] == g["corte_medio"]
    assert (t.set_index("fase").loc[["0091-0180", "0181-0360"], "origem"] == "fase").all()
    assert set(t["fase"]) == set(FASES) | {"__GLOBAL__"}


def test_vigencia_reproduz_regra_do_dia():
    s, y, fase = _dados()
    a = calcular_cortes(s, y, fase, pd.Timestamp("2025-12-01")).assign(corte_alto=0.9)
    b = calcular_cortes(s, y, fase, pd.Timestamp("2026-02-02")).assign(corte_alto=0.8)
    tab = pd.concat([a, b], ignore_index=True)
    assert (cortes_vigentes(tab, "2026-01-15")["corte_alto"] == 0.9).all()
    assert (cortes_vigentes(tab, "2026-02-02")["corte_alto"] == 0.8).all()
    assert cortes_vigentes(tab, "2025-11-30").empty


def test_classificar_ordena():
    dia = pd.DataFrame({"fase": ["__GLOBAL__", "0091-0180"], "corte_alto": [0.8, 0.5], "corte_medio": [0.6, 0.3]})
    c = classificar(np.array([0.55, 0.4, 0.2, 0.7, 0.85]),
                    np.array(["0091-0180", "0091-0180", "0091-0180", "nova", "nova"]), dia)
    assert c.tolist() == ["ALTO", "MEDIO", "BAIXO", "MEDIO", "ALTO"]
    assert 0 < pct_alto_esperado(np.array([0.9, 0.1]), np.array(["x", "x"]), dia) == 0.5
