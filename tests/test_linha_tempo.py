import numpy as np
import pandas as pd

from src.linha_tempo import cenarios_mensais, conclusoes


def _escs():
    dias = pd.bdate_range("2026-05-04", "2026-07-31")
    n = 100
    m = pd.DataFrame({"dia": np.repeat(dias, n), "id_contrato": np.tile(np.arange(n), len(dias))})
    m["ea_60d"] = (m["id_contrato"] < 10).astype(float)          # os 10 primeiros contratos são as EAs
    m["maduro"] = True
    m["score"] = -m["id_contrato"].astype(float)                 # mensal perfeito
    m["regua"] = m["id_contrato"].astype(float)                  # régua invertida
    c = m[["dia", "id_contrato"]].assign(score=np.where(m["id_contrato"] < 5, 1000.0, m["id_contrato"]))  # acha metade
    return m, c


def test_cenarios_capacidade():
    m, c = _escs()
    df = cenarios_mensais(m, c, capacidade=0.10).set_index(["mes", "cenario"])["pct_eas"]
    assert df[("2026-06", "mensal")] == 1.0 and df[("2026-06", "regua")] == 0.0
    assert df[("2026-06", "congelado")] == 0.5


def test_conclusoes_saem_dos_numeros():
    m, c = _escs()
    t1 = conclusoes(cenarios_mensais(m, c, capacidade=0.10))
    c2 = c.assign(score=-c["id_contrato"].astype(float))         # congelado agora perfeito: sem perda
    t2 = conclusoes(cenarios_mensais(m, c2, capacidade=0.10))
    assert t1[0]["titulo"].startswith("Com 10% do estoque, o modelo alcança 100% das entregas")
    assert t1[1]["titulo"] != t2[1]["titulo"] and "50,0 pontos" in t1[1]["titulo"]
    assert all({"titulo", "texto", "numero"} <= set(x) for x in t1)
    assert "praticamente não muda" in t2[1]["titulo"] and "-" not in t2[1]["titulo"]


def test_conclusao_com_perda_negativa_usa_sinal_de_menos():
    m, c = _escs()
    c3 = c.assign(score=-c["id_contrato"].astype(float))
    m3 = m.assign(score=np.where(m["id_contrato"] == 9, -1000.0, -m["id_contrato"]))   # mensal perde 1 de 10
    t = conclusoes(cenarios_mensais(m3, c3, capacidade=0.10))
    assert "−10,0 pontos" in t[1]["titulo"] and t[1]["numero"] == "−10,0 pp"
