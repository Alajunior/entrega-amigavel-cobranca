import re

import numpy as np
import pandas as pd
import pytest

from src.audio.roteiros import ALERTAS, OFERTAS, PERSONAS, montar_roteiros, selecionar_ligacoes
from src.config import HOJE

DIAS = pd.bdate_range(end=HOJE, periods=30)
OFERTA_TXT = re.compile(r"amig[aá]vel|devol|remanescente|entreg\w* (o|seu|esse) (ve[ií]culo|carro|bem)", re.I)


def _universo(n_contratos=4000, seed=0):
    """Contatos e escoragem sintéticos: metade dos contatos fora do universo (digital, sem CPC, fora da janela)."""
    rng = np.random.default_rng(seed)
    datas = pd.bdate_range(end=HOJE, periods=45)
    m = n_contratos
    desf = rng.choice(["SEM DESFECHO", "ACORDO", "EA INDICADA", "RECUSA EA", "ALEGA PAGAMENTO"], m,
                      p=[0.5, 0.15, 0.15, 0.1, 0.1])
    contatos = pd.DataFrame({
        "id_contrato": np.arange(1, m + 1), "data": rng.choice(datas, m),
        "canal": rng.choice(["discador", "digital"], m, p=[0.8, 0.2]),
        "operador": [f"OP-{k:02d}" for k in rng.integers(1, 13, m)],
        "alo": True, "cpc": rng.random(m) < 0.8, "desfecho": desf, "tabulacao": desf,
        "id_ligacao": [f"L{k:08d}" for k in range(1, m + 1)]})
    contatos.loc[contatos["canal"] == "digital", "operador"] = "DIGITAL"
    esc = pd.DataFrame({"dia": contatos["data"], "id_contrato": contatos["id_contrato"],
                        "classe": rng.choice(["ALTO", "MEDIO", "BAIXO"], m, p=[0.5, 0.25, 0.25])})
    return {"contatos": contatos}, esc


@pytest.fixture(scope="module")
def universo():
    return _universo()


@pytest.fixture(scope="module")
def roteiros(universo):
    t, esc = universo
    return montar_roteiros(t, esc, n=60, seed=42)


def test_selecao_respeita_universo(universo):
    t, esc = universo
    sel = selecionar_ligacoes(t, esc, n=60)
    assert len(sel) == 60 and sel["id_ligacao"].is_unique
    ct = t["contatos"].set_index("id_ligacao").loc[sel["id_ligacao"]]
    assert (ct["canal"] == "discador").all() and ct["cpc"].all()
    assert pd.to_datetime(sel["data"]).isin(DIAS).all()
    classe = esc.set_index(["dia", "id_contrato"])["classe"]
    chaves = list(zip(pd.to_datetime(sel["data"]), sel["id_contrato"]))
    assert (classe.loc[chaves] == "ALTO").all()
    assert set(sel.columns) >= {"id_ligacao", "data", "id_contrato", "operador", "desfecho", "tabulacao", "oferta_alvo"}


def test_selecao_na_carteira_real_pequena(carteira_pequena):
    """Com a carteira do gerador e uma escoragem que marca todo mundo ALTO, a seleção funciona de ponta a ponta."""
    ct = carteira_pequena["contatos"]
    esc = ct[["data", "id_contrato"]].rename(columns={"data": "dia"}).drop_duplicates().assign(classe="ALTO")
    sel = selecionar_ligacoes(carteira_pequena, esc, n=20)
    assert len(sel) == 20 and (sel["operador"] != "DIGITAL").all()


def test_distribuicao_do_gabarito(roteiros):
    cont = pd.Series([r["gabarito"]["oferta"] for r in roteiros]).value_counts()
    assert cont.to_dict() == {"OFERECEU": 33, "NÃO OFERECEU": 18, "SÓ O CLIENTE FALOU": 9}
    assert set(cont.index) <= set(OFERTAS)
    com_alerta = sum(bool(r["gabarito"]["alertas"]) for r in roteiros)
    assert com_alerta >= 13
    assert {a for r in roteiros for a in r["gabarito"]["alertas"]} == set(ALERTAS)


def test_pegadinhas_presentes(roteiros):
    def n_com(rotulo):
        return sum(bool(r["gabarito"]["frases"].get(rotulo)) for r in roteiros)
    assert n_com("pegadinha_loja") >= 3
    assert n_com("pegadinha_boleto") >= 3
    assert n_com("pegadinha_alerta_cliente") >= 3
    assert sum(r["gabarito"]["tipo"] == "devolucao" for r in roteiros) >= 3
    assert sum(r["gabarito"]["oferta"] == "SÓ O CLIENTE FALOU" for r in roteiros) >= 3
    for r in roteiros:                                         # toda frase registrada está mesmo no roteiro
        texto = " ".join(t["texto"] for t in r["turnos"])
        for frases in r["gabarito"]["frases"].values():
            assert all(f in texto for f in frases)


def test_duracao_estimada(roteiros):
    for r in roteiros:
        palavras = sum(len(t["texto"].split()) for t in r["turnos"])
        assert 45 <= palavras / 130 * 60 <= 150, (r["id_ligacao"], palavras)


def _texto(r, lado=None):
    """Texto do lado pedido sem as pegadinhas (que de propósito parecem oferta)."""
    texto = " ".join(t["texto"] for t in r["turnos"] if lado in (None, t["lado"]))
    for rotulo, frases in r["gabarito"]["frases"].items():
        if rotulo.startswith("pegadinha"):
            for fr in frases:
                texto = texto.replace(fr, "")
    return texto


def test_gabarito_coerente(roteiros):
    for r in roteiros:
        g = r["gabarito"]
        if g["oferta"] == "OFERECEU":
            assert g["tipo"] in ("explicita", "devolucao")
            assert any(f in _texto(r, "operador") for f in g["frases"]["oferta"])
            if g["tipo"] == "devolucao":
                assert not re.search(r"amig[aá]vel", _texto(r), re.I)
        elif g["oferta"] == "NÃO OFERECEU":
            assert g["tipo"] is None and not OFERTA_TXT.search(_texto(r))
        else:
            assert g["tipo"] is None and not OFERTA_TXT.search(_texto(r, "operador"))
            assert any(f in _texto(r, "cliente") for f in g["frases"]["cliente_pede"])
        for a in g["alertas"]:
            assert g["frases"][a] and all(f in _texto(r, "operador") for f in g["frases"][a])
            assert not any(f in _texto(r, "cliente") for f in g["frases"][a])


def test_personas_distintas_e_conhecidas(roteiros):
    for r in roteiros:
        po, pc = PERSONAS[r["persona_operador"]], PERSONAS[r["persona_cliente"]]
        assert po["voz"] != pc["voz"] and po["papel"] == "operador" and pc["papel"] == "cliente"
    por_op = {}
    for r in roteiros:                                         # o mesmo operador fala sempre com a mesma voz
        por_op.setdefault(r["operador"], set()).add(r["persona_operador"])
    assert all(len(v) == 1 for v in por_op.values())


def test_roteiro_carrega_contrato_e_tabulacao(universo, roteiros):
    ct = universo[0]["contatos"].set_index("id_ligacao")
    for r in roteiros:
        assert r["id_contrato"] == ct.loc[r["id_ligacao"], "id_contrato"]
        assert r["tabulacao"] == ct.loc[r["id_ligacao"], "tabulacao"]
        assert r["desfecho"] == ct.loc[r["id_ligacao"], "desfecho"]


def test_mesma_seed_mesmos_roteiros(universo, roteiros):
    t, esc = universo
    assert montar_roteiros(t, esc, n=60, seed=42) == roteiros
    assert montar_roteiros(t, esc, n=60, seed=7) != roteiros
