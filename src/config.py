"""Constantes do projeto: datas, janela do alvo, fases, faixas de atraso e classes.

Hoje simulado = 30/11/2026. O alvo é a formalização de Entrega Amigável em (D, D+60]: a foto D só é
"madura" (alvo conhecido) quando D + 60 <= HOJE.
"""
import numpy as np
import pandas as pd

HOJE = pd.Timestamp("2026-11-30")
INICIO_DADOS = pd.Timestamp("2025-01-01")
INICIO_PAINEL = pd.Timestamp("2025-12-01")
JANELA = 60
ULTIMA_MADURA = HOJE - pd.Timedelta(days=JANELA)
PRIMEIRA_SAFRA = pd.Timestamp("2025-04-01")
PASSO_SAFRA = 28

FASES = ["0031-0090", "0091-0180", "0181-0360", "0361-0720"]
_LIM_FASES = [31, 91, 181, 361, 721]
FAIXAS_ATRASO = [(31, 60), (61, 90), (91, 120), (121, 180), (181, 360), (361, 540), (541, 720)]
FAIXAS = [f"{a:03d}-{b:03d}" for a, b in FAIXAS_ATRASO]
CLASSES = ["ALTO", "MEDIO", "BAIXO"]
CANAIS = ["discador", "digital"]
CAPTURA_ALTO = 0.80
CAPTURA_MEDIO = 0.95
MIN_EA_FASE = 30


def _rotular(dias: pd.Series, limites: list[int], rotulos: list[str]) -> pd.Series:
    i = np.searchsorted(limites, np.asarray(dias, dtype=float), side="right") - 1
    ok = (i >= 0) & (i < len(rotulos))
    return pd.Series(np.where(ok, np.array(rotulos, dtype=object)[np.clip(i, 0, len(rotulos) - 1)], None),
                     index=getattr(dias, "index", None))


def fase_de(dias: pd.Series) -> pd.Series:
    return _rotular(dias, _LIM_FASES, FASES)


def faixa_de(dias: pd.Series) -> pd.Series:
    return _rotular(dias, [a for a, _ in FAIXAS_ATRASO] + [FAIXAS_ATRASO[-1][1] + 1], FAIXAS)
