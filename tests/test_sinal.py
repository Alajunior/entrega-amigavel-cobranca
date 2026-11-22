import numpy as np
import pandas as pd
import pytest

from src.base_analitica import ALVO, CAT, NUM, montar_base
from src.modelos import criar_modelos
from src.sinal import FRASES, NOMES, contribuicoes, direcoes, importancia, motivos, referencia


@pytest.fixture(scope="module")
def ajustado(carteira_pequena):
    t = {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}
    b = pd.concat([montar_base(d, t) for d in ("2025-08-01", "2025-11-03", "2026-02-02", "2026-05-04")])
    m = criar_modelos()["logistica"].fit(b[NUM + CAT], b[ALVO])
    return m, b[NUM + CAT]


def test_contribuicoes_fecham_o_logodds(ajustado):
    m, X = ajustado
    ref = referencia(m, X)
    c = contribuicoes(m, X.head(50), ref)
    lr = m[-1]
    zref = ref.to_numpy() @ lr.coef_[0] + lr.intercept_[0]
    assert np.allclose(c.sum(axis=1).to_numpy() + zref, m.decision_function(X.head(50)), atol=1e-6)
    assert set(c.columns) <= set(NUM) | set(CAT) | {"dias_desde_ult_cpc_nulo"}
    assert importancia(c)["media_abs"].is_monotonic_decreasing


def test_motivo_direcional_sem_repeticao():
    c = pd.DataFrame({"qtd_cpc_30d": [0.9], "dias_desde_ult_cpc": [0.5], "ltv_atual": [0.4], "idade": [-0.2]})
    d = pd.DataFrame({"qtd_cpc_30d": [1], "dias_desde_ult_cpc": [-1], "ltv_atual": [1], "idade": [1]})
    assert motivos(c, d) == [[FRASES["qtd_cpc_30d"][1], FRASES["ltv_atual"][1]]]   # "contato certo recente" uma vez
    assert motivos(pd.DataFrame({"fase": [0.3], "idade": [-0.1]})) == [[NOMES["fase"]]]


def test_flag_de_nulo_e_variavel_propria(ajustado):
    m, X = ajustado
    ref = referencia(m, X)
    c = contribuicoes(m, X, ref)
    assert "dias_desde_ult_cpc_nulo" in c.columns
    mot = motivos(c, direcoes(m, X, ref))
    coef = dict(zip(m[:-1].get_feature_names_out(), m[-1].coef_[0]))
    lado_ruim = FRASES["dias_desde_ult_cpc"][1 if coef["num__dias_desde_ult_cpc"] < 0 else 0]
    assert not any(lado_ruim in ms for ms in mot)
    assert all(len(ms) == len(set(ms)) for ms in mot)
