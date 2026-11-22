import json
from pathlib import Path

import pytest

from src.audio.medir import ablacao, avaliar, cruzamento_tabulacao, matriz_confusao, metricas_alertas, ranking_operador

FIX = json.loads((Path(__file__).parent / "dados_ligacoes_fixture.json").read_text(encoding="utf-8"))


@pytest.fixture
def df():
    return avaliar(FIX["roteiros"], FIX["transcricoes"])


def test_avaliar_colunas_e_acerto(df):
    assert {"id_ligacao", "data", "operador", "tabulacao", "desfecho", "oferta_gabarito", "oferta_regra", "tipo_regra",
            "quem", "trecho", "acerto_oferta", "alertas_gabarito", "alertas_regra", "wer_operador", "wer_cliente",
            "sem_audio"} <= set(df.columns)
    r = df.set_index("id_ligacao")
    assert r.loc["LA", "oferta_regra"] == "OFERECEU" and r.loc["LA", "acerto_oferta"]
    assert r.loc["LA", "alertas_regra"] == ["desconto_percentual"]
    assert r.loc["LC", "oferta_regra"] == "SÓ O CLIENTE FALOU" and r.loc["LC", "quem"] == "cliente"
    assert r.loc["LB", "wer_operador"] == 0 and r.loc["LA", "wer_operador"] > 0


def test_sem_transcricao_vira_sem_audio(df):
    r = df.set_index("id_ligacao").loc["LD"]
    assert r["oferta_regra"] == "SEM ÁUDIO" and r["sem_audio"] and not r["acerto_oferta"]
    assert r["alertas_regra"] == [] and r["wer_operador"] is None


def test_matriz_soma_total(df):
    m = matriz_confusao(df)
    assert sum(v for linha in m["valores"] for v in linha) == len(df)
    i, j = m["linhas"].index("OFERECEU"), m["colunas"].index("SEM ÁUDIO")
    assert m["valores"][i][j] == 1
    assert "INCONCLUSIVO" in m["colunas"]


def test_metricas_alertas_vp_fp_fn(df):
    m = metricas_alertas(df)
    assert m["desconto_percentual"] == {"vp": 1, "fp": 0, "fn": 0, "precisao": 1.0, "recall": 1.0}
    q = m["quitacao_sem_ressalva"]              # a única ligação com quitação está sem áudio: não entra na conta
    assert q["vp"] == 0 and q["fn"] == 0 and q["recall"] is None and q["precisao"] is None


def test_ablacao_tolerante_ganha_na_fixture():
    a = ablacao(FIX["roteiros"], FIX["transcricoes"])
    w = a["whisper"]
    assert w["tolerante"]["recall_oferta"] > w["simples"]["recall_oferta"]
    assert w["tolerante"]["acerto"] > w["simples"]["acerto"]
    assert a["roteiro"]["tolerante"]["recall_oferta"] == a["roteiro"]["simples"]["recall_oferta"] == 1.0
    assert a["ganhos"] and a["ganhos"][0]["id_ligacao"] == "LA"


def test_ofereceu_e_tabulou_outra_coisa():
    df = avaliar(FIX["roteiros"], FIX["transcricoes"] | {"LD": {"id_ligacao": "LD", "turnos": [
        {"lado": "operador", "inicio": 0.3, "fim": 3.0, "texto": "Uma saída é o senhor devolver o veículo."}]}})
    c = cruzamento_tabulacao(df)
    linha = c[(c["oferta_regra"] == "OFERECEU") & (c["tabulacao"] == "ACORDO")]
    assert len(linha) == 1 and linha["ligacoes"].iloc[0] == 1 and linha["divergente"].iloc[0]
    assert not c[(c["oferta_regra"] == "OFERECEU") & (c["tabulacao"] == "RECUSA EA")]["divergente"].iloc[0]


def test_ranking_operador(df):
    r = ranking_operador(df).set_index("operador")
    assert r.loc["OP-01", "ligacoes"] == 2 and r.loc["OP-01", "taxa_oferta"] == 0.5
    assert r.loc["OP-01", "alertas"] == 1
    assert r.loc["OP-02", "ligacoes"] == 2                  # LD sem áudio não conta como auditada
