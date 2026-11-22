"""Escoragem diária como em produção: cada dia útil D é escorado com o modelo VIGENTE em D.

Modelo vigente (retreinado no 1º dia útil de cada mês e no primeiro dia escorado):
    - elegíveis = safras cujo alvo (s, s+60] já fechou em D;
    - calibração = as 3 elegíveis mais recentes (fora do treino): de lá saem os cortes ALTO/MEDIO por fase;
    - treino = até 8 elegíveis anteriores à calibração (sem 4, não há vigente).
Modos: "mensal" (logística retreinada, o que publica), "congelado" (a logística do 1º dia, nunca retreinada: o
contrafactual) e "sombra" (LightGBM retreinado nas mesmas datas, com cortes próprios: o desafiante).
A foto de D só sabe o que aconteceu antes de D; o alvo só existe quando D + 60 <= HOJE (senão NaN, nunca zero).
"""
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.base_analitica import ALVO, CAT, NUM, montar_base
from src.candidatos import bases_das_safras
from src.config import HOJE, INICIO_PAINEL, JANELA
from src.cortes import calcular_cortes, classificar, cortes_vigentes, pct_alto_esperado
from src.modelos import criar_modelos, score_regua
from src.sinal import referencia

COLUNAS = ["dia", "id_contrato", "id_cliente", "uf", "fase", "faixa_atraso", "dias_atraso", "divida_total",
           "ltv_atual", "score", "classe", "ea_60d", "maduro", "regua"]
ENXUTAS = ["dia", "id_contrato", "score", "classe"]
MODELO_DO_MODO = {"mensal": "logistica", "congelado": "logistica", "sombra": "lightgbm"}
N_CALIB, MAX_TREINO, MIN_TREINO = 3, 8, 4


@dataclass
class Vigente:
    modelo: object
    nome: str
    cortes: pd.DataFrame
    pct_alto: float
    ref: pd.Series | None
    safras_treino: list = field(default_factory=list)
    safras_calib: list = field(default_factory=list)
    treinado_em: pd.Timestamp | None = None


def treinar_vigente(D, bases: dict, nome: str = "logistica") -> Vigente | None:
    D = pd.Timestamp(D)
    eleg = [s for s in sorted(bases) if s + pd.Timedelta(days=JANELA) <= D]
    calib, antes = eleg[-N_CALIB:], eleg[:-N_CALIB]
    treino = antes[-MAX_TREINO:]
    if len(calib) < N_CALIB or len(treino) < MIN_TREINO:
        return None
    tr = pd.concat(bases[s] for s in treino)
    ca = pd.concat(bases[s] for s in calib)
    m = clone(criar_modelos()[nome]).fit(tr[NUM + CAT], tr[ALVO])
    p = m.predict_proba(ca[NUM + CAT])[:, 1]
    cortes = calcular_cortes(p, ca[ALVO], ca["fase"], D)
    return Vigente(modelo=m, nome=nome, cortes=cortes, pct_alto=pct_alto_esperado(p, ca["fase"], cortes),
                   ref=referencia(m, tr[NUM + CAT]) if nome == "logistica" else None,
                   safras_treino=treino, safras_calib=calib, treinado_em=D)


def datas_retreino(dias: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(dias)
    return list(s.groupby(s.dt.to_period("M")).min())


def escorar_dia(D, tabelas: dict, vig: Vigente, base: pd.DataFrame | None = None,
                cortes: pd.DataFrame | None = None) -> pd.DataFrame:
    D = pd.Timestamp(D)
    b = montar_base(D, tabelas) if base is None else base
    score = vig.modelo.predict_proba(b[NUM + CAT])[:, 1]
    tab = vig.cortes if cortes is None else cortes_vigentes(cortes, D)
    e = b.assign(dia=D, score=score, classe=classificar(score, b["fase"], tab),
                 maduro=D + pd.Timedelta(days=JANELA) <= HOJE, regua=score_regua(b))
    return e[COLUNAS]


def _compactar(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("uf", "fase", "faixa_atraso", "classe"):
        if c in df:
            df[c] = df[c].astype("category")
    for c in ("score", "divida_total", "ltv_atual", "regua", "ea_60d"):
        if c in df:
            df[c] = df[c].astype("float32")
    return df


def assinatura(tabelas: dict) -> str:
    """Impressão digital do conteúdo das tabelas (não só do tamanho): mudou o gerador, muda o nome do cache."""
    h = hashlib.sha256()
    for nome in ("contratos", "contatos", "eventos_ea", "acordos"):
        h.update(pd.util.hash_pandas_object(tabelas[nome], index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def escorar_periodos(tabelas: dict, modos=("mensal", "congelado", "sombra"), inicio=INICIO_PAINEL, fim=HOJE,
                     cache: Path | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Escora os dias úteis do período em vários modos de uma vez (a foto de cada dia é montada uma só vez).
    Devolve {modo: (escoragem, tabela de cortes com todas as vigências)}. O modo mensal guarda todas as colunas;
    os demais, só dia/contrato/score/classe (o resto se junta pelo contrato)."""
    inicio, fim = pd.Timestamp(inicio), pd.Timestamp(fim)
    arq = None
    if cache is not None:
        arq = Path(cache) / f"escoragem_{'-'.join(modos)}_{inicio:%Y%m%d}_{fim:%Y%m%d}_{assinatura(tabelas)}.pkl"
        if arq.exists():
            return pickle.loads(arq.read_bytes())
    dias = pd.bdate_range(inicio, fim)
    retreinos = set(datas_retreino(dias)) | {dias[0]}
    bases = bases_das_safras(tabelas)
    vig, cortes, partes = {}, {m: [] for m in modos}, {m: [] for m in modos}
    for D in dias:
        b = montar_base(D, tabelas)
        for modo in modos:
            if D in retreinos and (modo != "congelado" or modo not in vig):
                v = treinar_vigente(D, bases, MODELO_DO_MODO[modo])
                if v is not None:
                    vig[modo] = v
                    cortes[modo].append(v.cortes)
            if modo not in vig:
                continue
            e = escorar_dia(D, tabelas, vig[modo], base=b, cortes=pd.concat(cortes[modo], ignore_index=True))
            partes[modo].append(_compactar(e if modo == "mensal" else e[ENXUTAS].copy()))
    out = {}
    for modo in modos:
        esc = pd.concat(partes[modo], ignore_index=True) if partes[modo] else pd.DataFrame(columns=COLUNAS)
        out[modo] = (_compactar(esc), pd.concat(cortes[modo], ignore_index=True) if cortes[modo] else pd.DataFrame())
    if arq is not None:
        arq.parent.mkdir(parents=True, exist_ok=True)
        arq.write_bytes(pickle.dumps(out))
    return out


def escorar_periodo(tabelas: dict, modo: str = "mensal", inicio=INICIO_PAINEL, fim=HOJE,
                    cache: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return escorar_periodos(tabelas, (modo,), inicio, fim, cache)[modo]
