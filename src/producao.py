"""Escoragem diária com as travas de produção: melhor não publicar do que publicar errado (porte do Propensão).

Ordem das travas (a primeira que falha encerra com o código dela):
    7 DIA_PASSADO        nunca regrava um dia anterior ao último publicado (reexecutar o MESMO dia pode: substitui)
    5 HASH_DIVERGENTE    o modelo em disco não é o do manifest (sha256), ou não há manifest
    3 FOTO_AUSENTE       a carteira do dia não chegou (foto vazia): nada é publicado
    4 VOLUME_BAIXO       a foto tem menos de 85% dos contratos do último dia publicado: carga incompleta
    6 DISTRIBUICAO_FORA  a participação de ALTO saiu de 0,5x–1,5x do que a calibração prevê
    0 OK                 grava escoragem_AAAAMMDD.csv via arquivo temporário + os.replace (publicação atômica)
"""
import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd

from src.apto import apto_ea, proximo_passo
from src.base_analitica import montar_base
from src.config import HOJE
from src.escoragem import Vigente, escorar_dia

OK, FOTO_AUSENTE, VOLUME_BAIXO, HASH_DIVERGENTE, DISTRIBUICAO_FORA, DIA_PASSADO = 0, 3, 4, 5, 6, 7
NOMES_CODIGO = {OK: "ok", FOTO_AUSENTE: "foto ausente", VOLUME_BAIXO: "volume baixo", HASH_DIVERGENTE: "hash divergente",
                DISTRIBUICAO_FORA: "distribuição fora", DIA_PASSADO: "dia passado"}
VOLUME_MIN = 0.85
FAIXA_ALTO = (0.5, 1.5)


def _sha(dados: bytes) -> str:
    return hashlib.sha256(dados).hexdigest()


def salvar_modelo(vig: Vigente, pasta) -> dict:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    dados = pickle.dumps(vig)
    (pasta / "modelo.pkl").write_bytes(dados)
    manifest = {"sha256": _sha(dados), "versao": f"mensal-{vig.treinado_em:%Y-%m}", "modelo": vig.nome,
                "pct_alto_calibracao": vig.pct_alto, "safras_calib": [f"{d:%Y-%m-%d}" for d in vig.safras_calib],
                "safras_treino": [f"{d:%Y-%m-%d}" for d in vig.safras_treino]}
    (pasta / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def carregar_modelo(pasta) -> tuple[Vigente | None, int]:
    """Os bytes são lidos UMA vez: o hash é conferido nesses bytes e só eles vão para o pickle (sem janela para troca).
    Com o manifest ao lado do modelo, o hash detecta arquivo trocado ou corrompido, não adulteração deliberada."""
    pasta = Path(pasta)
    try:
        esperado = json.loads((pasta / "manifest.json").read_text(encoding="utf-8"))["sha256"]
        dados = (pasta / "modelo.pkl").read_bytes()
    except (OSError, ValueError, KeyError):
        return None, HASH_DIVERGENTE
    if _sha(dados) != esperado:
        return None, HASH_DIVERGENTE
    return pickle.loads(dados), OK


def _publicados(saida: Path) -> dict[pd.Timestamp, Path]:
    return {pd.Timestamp(p.stem.split("_")[1]): p for p in saida.glob("escoragem_*.csv")}


def distribuicao_ok(e: pd.DataFrame, pct_alto_esperado: float) -> bool:
    real = (e["classe"] == "ALTO").mean()
    return bool(FAIXA_ALTO[0] * pct_alto_esperado <= real <= FAIXA_ALTO[1] * pct_alto_esperado)


def escorar_dia_guardado(D, tabelas: dict, pasta_modelo, pasta_saida, foto: pd.DataFrame | None = None,
                         hoje=HOJE) -> dict:
    D = pd.Timestamp(D)
    saida = Path(pasta_saida)
    saida.mkdir(parents=True, exist_ok=True)
    pub = _publicados(saida)
    r = {"dia": f"{D:%Y-%m-%d}", "codigo": OK, "motivo": "", "linhas": 0, "volume_vs_anterior": None,
         "distribuicao_ok": None, "sha_ok": None}

    def fim(codigo, motivo):
        r.update(codigo=codigo, motivo=motivo)
        return r

    if pub and D < max(pub):
        return fim(DIA_PASSADO, f"o último dia publicado é {max(pub):%d/%m/%Y}; dia passado não é regravado")
    vig, cod = carregar_modelo(pasta_modelo)
    r["sha_ok"] = cod == OK
    if cod != OK:
        return fim(HASH_DIVERGENTE, "o modelo em disco não confere com o sha256 do manifest")
    base = montar_base(D, tabelas) if foto is None else foto
    r["linhas"] = int(len(base))
    if len(base) == 0:
        return fim(FOTO_AUSENTE, "a foto do estoque não chegou: nada foi publicado")
    anteriores = [d for d in pub if d < D]
    if anteriores:
        n_ant = sum(1 for _ in open(pub[max(anteriores)], encoding="utf-8")) - 1
        r["volume_vs_anterior"] = round(len(base) / n_ant, 4) if n_ant else None
        if n_ant and len(base) / n_ant < VOLUME_MIN:
            return fim(VOLUME_BAIXO, f"a foto tem {len(base) / n_ant:.0%} dos contratos do último dia publicado")
    e = escorar_dia(D, tabelas, vig, base=base)
    r["distribuicao_ok"] = distribuicao_ok(e, vig.pct_alto)
    if not r["distribuicao_ok"]:
        return fim(DISTRIBUICAO_FORA, f"ALTO em {(e['classe'] == 'ALTO').mean():.1%} contra {vig.pct_alto:.1%} "
                                      f"previstos na calibração")
    e = e.assign(apto_ea=apto_ea(base, tabelas, D).to_numpy())
    e["proximo_passo"] = [proximo_passo(a, c) for a, c in zip(e["apto_ea"], e["classe"])]
    final = saida / f"escoragem_{D:%Y%m%d}.csv"
    tmp = saida / f".{final.name}.tmp"
    e.to_csv(tmp, index=False)
    os.replace(tmp, final)            # atômico: quem lê vê o arquivo antigo ou o novo inteiro, nunca pela metade
    return fim(OK, "publicado")
