import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.transcrever import (PROMPT, gravar_checkpoint, ja_feitas, limpar, transcrever_ligacao,
                                   turnos_de_palavras, wer)


def p(inicio, fim, texto):
    return {"inicio": inicio, "fim": fim, "texto": texto}


def test_turnos_por_pausa_e_lado():
    op = [p(0.0, 0.4, "bom"), p(0.5, 0.9, "dia"), p(4.0, 4.5, "certo")]
    cli = [p(1.5, 2.0, "oi"), p(2.1, 2.6, "pode"), p(2.7, 3.0, "falar")]
    t = turnos_de_palavras(op, cli)
    assert [(x["lado"], x["texto"]) for x in t] == [("operador", "bom dia"), ("cliente", "oi pode falar"),
                                                    ("operador", "certo")]
    assert t[0]["inicio"] == 0.0 and t[0]["fim"] == 0.9 and t[1]["inicio"] == 1.5


def test_funde_balao_do_mesmo_lado():
    """Pausa longa sem ninguém falar no meio continua sendo o mesmo turno."""
    op = [p(0.0, 0.5, "olha"), p(2.0, 2.5, "só"), p(5.0, 5.4, "isso")]
    t = turnos_de_palavras(op, [p(6.0, 6.5, "tá")])
    assert [(x["lado"], x["texto"]) for x in t] == [("operador", "olha só isso"), ("cliente", "tá")]
    assert t[0]["fim"] == 5.4


def test_turnos_sem_palavras():
    assert turnos_de_palavras([], []) == []
    assert [x["lado"] for x in turnos_de_palavras([], [p(0, 1, "alô")])] == ["cliente"]


def test_limpa_eco_do_prompt_e_loop():
    eco = "Ligação de cobrança de financiamento de veículo"
    assert eco.lower() in PROMPT.lower()
    assert limpar(f"{eco} bom dia senhor", PROMPT) == "bom dia senhor"
    assert limpar("pode falar pode falar pode falar pode falar comigo", PROMPT) == "pode falar comigo"
    assert limpar("não não não", PROMPT) == "não"
    assert limpar("acordo de parcelas", PROMPT) == "acordo de parcelas"      # 3 palavras do prompt: fica
    assert limpar("", PROMPT) == ""


def test_wer_exemplos():
    assert wer("bom dia senhor Bruno", "Bom dia, senhor Bruno!") == 0
    assert wer("bom dia senhor Bruno", "bom dia senhora Bruno") == 0.25
    assert wer("bom dia senhor Bruno", "") == 1.0
    assert wer("", "") == 0.0
    assert wer("entrega amigável", "entrega a migável") == 1.0           # uma troca + uma inserção em 2 palavras


def test_checkpoint_idempotente(tmp_path):
    arq = tmp_path / "t.jsonl"
    gravar_checkpoint(arq, {"id_ligacao": "A", "v": 1})
    gravar_checkpoint(arq, {"id_ligacao": "B", "v": 1})
    gravar_checkpoint(arq, {"id_ligacao": "A", "v": 2})
    linhas = [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines()]
    assert [(x["id_ligacao"], x["v"]) for x in linhas] == [("A", 2), ("B", 1)]
    assert ja_feitas(arq) == {"A", "B"} and ja_feitas(tmp_path / "nao_existe.jsonl") == set()
    assert not list(tmp_path.glob("*.tmp"))


class ModeloFalso:
    """Devolve palavras diferentes por canal: reconhece o canal pela energia (operador em L é mais alto)."""
    def __init__(self):
        self.chamadas = []

    def transcribe(self, audio, **kw):
        self.chamadas.append(kw)
        alto = float(np.abs(audio).mean()) > 0.05
        w = (SimpleNamespace(start=0.5, end=1.0, word=" Bom"), SimpleNamespace(start=1.1, end=1.5, word=" dia")) \
            if alto else (SimpleNamespace(start=2.0, end=2.4, word=" Oi"),)
        return iter([SimpleNamespace(words=list(w), text="")]), SimpleNamespace(duration=3.0)


def test_transcrever_com_modelo_falso(tmp_path):
    pytest.importorskip("av")
    from src.audio.sintetizar import exportar_mp3
    t = np.arange(3 * 8000) / 8000
    est = np.stack([0.5 * np.sin(2 * np.pi * 500 * t), 0.01 * np.sin(2 * np.pi * 700 * t)], 1).astype(np.float32)
    exportar_mp3(est, 8000, tmp_path / "LX.mp3")
    m = ModeloFalso()
    r = transcrever_ligacao(tmp_path / "LX.mp3", m)
    assert r["id_ligacao"] == "LX" and r["versao"] == "turnos_v1" and r["modelo"] == "large-v3-turbo"
    assert [(x["lado"], x["texto"]) for x in r["turnos"]] == [("operador", "Bom dia"), ("cliente", "Oi")]
    assert len(m.chamadas) == 2 and all(k["language"] == "pt" and k["vad_filter"] and k["word_timestamps"]
                                        and k["beam_size"] == 5 for k in m.chamadas)
    assert abs(r["duracao"] - 3.0) < 0.3
