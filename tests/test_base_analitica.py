import numpy as np
import pandas as pd
import pytest

from src.base_analitica import ALVO, CAT, NUM, montar_base, safras
from src.config import PRIMEIRA_SAFRA


def _sem_cache(t):
    return {k: v.copy() for k, v in t.items() if not k.startswith("_")}


def test_contato_do_dia_nao_entra_na_foto(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    D = pd.Timestamp("2026-03-10")
    b0 = montar_base(D, t)
    alvo = b0.loc[b0["qtd_cpc_30d"] == 0, "id_contrato"].iloc[0]
    novo = pd.DataFrame([{"id_contrato": alvo, "data": D, "canal": "discador", "operador": "OP-01", "alo": True,
                          "cpc": True, "desfecho": "SEM DESFECHO", "tabulacao": "SEM DESFECHO", "id_ligacao": "LX"}])
    t2 = _sem_cache(t)
    t2["contatos"] = pd.concat([t["contatos"], novo], ignore_index=True)
    a = montar_base(D, t2).set_index("id_contrato").loc[alvo]
    assert a["qtd_cpc_30d"] == 0
    depois = montar_base(D + pd.Timedelta(days=1), t2).set_index("id_contrato").loc[alvo]
    assert depois["qtd_cpc_30d"] == 1 and depois["dias_desde_ult_cpc"] == 1


def _um_formalizado(t):
    e = t["eventos_ea"]
    f = e[(e["desfecho"] == "FORMALIZADA") & e["data_desfecho"].between("2025-09-01", "2026-09-01")]
    return f.iloc[0]["id_contrato"], f.iloc[0]["data_desfecho"]


def test_alvo_janela_aberta_e_fechada(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    c, F = _um_formalizado(t)
    dia = pd.Timedelta(days=1)
    def y(D):
        b = montar_base(D, t).set_index("id_contrato")
        return b.loc[c, ALVO] if c in b.index else None
    assert y(F - 60 * dia) == 1 and y(F - dia) == 1
    assert y(F - 61 * dia) in (0, None)
    assert y(F) is None


def test_alvo_nao_maduro_e_nan(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    assert montar_base("2026-10-15", t)[ALVO].isna().all()
    assert montar_base("2026-10-01", t)[ALVO].notna().all()


def test_idade_na_data_da_foto(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    a = montar_base("2025-06-02", t).set_index("id_contrato")
    b = montar_base("2026-06-02", t).set_index("id_contrato")
    comum = a.index.intersection(b.index)
    assert len(comum) > 0
    assert ((b.loc[comum, "idade"] - a.loc[comum, "idade"]) == 1).all()


def test_colunas_e_safras(carteira_pequena):
    b = montar_base("2026-01-05", _sem_cache(carteira_pequena))
    assert set(NUM + CAT + [ALVO, "id_contrato", "id_cliente", "uf", "faixa_atraso", "data_form"]) <= set(b.columns)
    assert b["id_contrato"].is_unique
    s = safras()
    assert s[0] == PRIMEIRA_SAFRA and all((b - a).days == 28 for a, b in zip(s, s[1:]))
    assert s[-1] <= pd.Timestamp("2026-10-01")


def test_ltv_deprecia_so_desde_a_contratacao(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    D = pd.Timestamp("2026-03-02")
    b = montar_base(D, t).set_index("id_contrato")
    c = t["contratos"].set_index("id_contrato").loc[b.index]
    anos = (D - c["data_contratacao"]).dt.days / 365.25
    esperado = b["divida_total"] / (c["valor_veiculo_inicial"] * 0.85 ** anos)
    assert np.allclose(b["ltv_atual"], esperado)


def test_acordo_e_insucesso_so_contam_antes_de_D(carteira_pequena):
    t = _sem_cache(carteira_pequena)
    D = pd.Timestamp("2026-03-10")
    alvo = montar_base(D, t)["id_contrato"].iloc[0]
    def com(acordos=(), insucessos=()):
        t2 = _sem_cache(t)
        t2["acordos"] = pd.concat([t["acordos"], pd.DataFrame({"id_contrato": alvo, "data_acordo": pd.to_datetime(list(acordos)),
                                                              "quebrado": True})], ignore_index=True)
        ins = pd.DataFrame({"id_indicacao": -1, "id_contrato": alvo, "data_indicacao": pd.to_datetime(list(insucessos)) - pd.Timedelta(days=5),
                            "canal": "discador", "operador": "OP-01", "desfecho": "INSUCESSO",
                            "data_desfecho": pd.to_datetime(list(insucessos)), "motivo_insucesso": "DESISTENCIA"})
        t2["eventos_ea"] = pd.concat([t["eventos_ea"], ins], ignore_index=True)
        return montar_base(D, t2).set_index("id_contrato").loc[alvo]
    base = com()
    dia = pd.Timedelta(days=1)
    assert com(acordos=[D - 31 * dia])["acordos_quebrados_180d"] == base["acordos_quebrados_180d"] + 1
    assert com(acordos=[D - 29 * dia, D])["acordos_quebrados_180d"] == base["acordos_quebrados_180d"]   # quebra ainda invisível
    assert com(acordos=[D - 181 * dia])["acordos_quebrados_180d"] == base["acordos_quebrados_180d"]
    assert com(insucessos=[D])["insucessos_ea_antes"] == base["insucessos_ea_antes"]
    assert com(insucessos=[D - dia])["insucessos_ea_antes"] == base["insucessos_ea_antes"] + 1
