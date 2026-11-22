import json

import numpy as np
import pandas as pd
import pytest

from src.config import HOJE
from src.montar_painel import agregar_fluxo, classe_asof, dia_util_seguinte, montar_dados, montar_html
from src.treinar import main as treinar

INICIO_T = "2026-07-01"


@pytest.fixture(scope="module")
def dados(carteira_pequena, pasta_pequena, tmp_path_factory):
    rel = tmp_path_factory.mktemp("relatorios")
    treinar(pasta_pequena, rel)
    esc = json.loads((rel / "escada.json").read_text(encoding="utf-8"))
    esc["sombra"] = None                                            # força o caso "sem desafiante em sombra"
    (rel / "escada.json").write_text(json.dumps(esc), encoding="utf-8")
    t = {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}
    return montar_dados(t, inicio=INICIO_T, cache=tmp_path_factory.mktemp("cache"), relatorios=rel)


def test_evento_no_fim_de_semana_vai_para_segunda():
    dias = pd.bdate_range("2026-03-02", "2026-03-13")
    d = dia_util_seguinte(dias, pd.Series(pd.to_datetime(["2026-03-07", "2026-03-08", "2026-03-09", "2026-03-14"])))
    assert d.tolist()[:3] == [pd.Timestamp("2026-03-09")] * 3 and pd.isna(d.iloc[3])


def _esc_sint():
    dias = pd.bdate_range("2026-03-02", "2026-03-06")
    return pd.DataFrame({"dia": dias[2:], "id_contrato": 7, "classe": ["MEDIO", "ALTO", "ALTO"],
                         "faixa_atraso": "181-360", "fase": "0181-0360"})


def test_classe_asof_cascata():
    esc = _esc_sint()
    ev = pd.DataFrame({"id_contrato": [7, 7, 7], "data": pd.to_datetime(["2026-03-02", "2026-03-05", "2026-03-09"])})
    assert classe_asof(ev, esc, "data").tolist() == ["MEDIO", "ALTO", "ALTO"]   # antes da 1ª foto: a primeira depois
    sem = pd.DataFrame({"id_contrato": [99], "data": pd.to_datetime(["2026-03-05"])})
    assert classe_asof(sem, esc, "data").tolist() == ["SEM CLASSE"]


def test_reindicacao_conta_duas_e_efetividade_ate_100(dados):
    esc = _esc_sint()
    dias = pd.bdate_range("2026-03-02", "2026-03-13")
    ev = pd.DataFrame({"id_contrato": [7, 7], "data_indicacao": pd.to_datetime(["2026-03-04", "2026-03-06"]),
                       "canal": "discador", "operador": "OP-01", "desfecho": ["INSUCESSO", "FORMALIZADA"],
                       "data_desfecho": pd.to_datetime(["2026-03-05", "2026-03-11"]), "motivo_insucesso": ["DESISTENCIA", ""]})
    ct = pd.DataFrame(columns=["id_contrato", "data", "canal", "alo", "cpc"])
    f = agregar_fluxo(ct, ev, esc, dias, HOJE)
    assert f["indicacoes"].sum() == 2 and f["form_por_ind"].sum() == 1 and f["form_por_form"].sum() == 1
    assert f.loc[f["form_por_ind"] == 1, "dia"].iloc[0] == pd.Timestamp("2026-03-06")
    fl = pd.DataFrame(dados["fluxo"])
    assert (fl["form_por_ind"] <= fl["indicacoes"]).all()


def test_payload_sem_nan_e_sem_infinito(dados):
    html = montar_html(dados)
    for ruim in ("NaN", "Infinity", "undefined"):
        assert ruim not in html.split("/*__INICIO_DADOS__*/")[-1].split("/*__FIM_DADOS__*/")[0]


def test_chave_ausente_quando_sem_dado(dados):
    assert "sombra" not in dados["modelo"]
    assert {"escada", "vazamento", "metricas_oot", "curva_pr", "captura", "cortes", "backtest", "importancia",
            "tempo"} <= set(dados["modelo"])
    assert {"meta", "estoque", "fluxo", "fila", "perfil", "uf", "eventos", "modelo"} <= set(dados)


def test_nada_depois_de_hoje(dados):
    assert max(dados["meta"]["dias"]) <= f"{HOJE:%Y-%m-%d}"
    assert all(e["data"] <= f"{HOJE:%Y-%m-%d}" for e in dados["eventos"])
    assert max(dados["fluxo"]["dia"]) < len(dados["meta"]["dias"]) and max(dados["estoque"]["dia"]) < len(dados["meta"]["dias"])
    assert all(r["classe"] == "ALTO" for r in dados["fila"]) and len(dados["fila"]) <= 200
