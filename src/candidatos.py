"""Escada de candidatos com gate: régua → logística → LightGBM. Vence o mais simples que passa.

Gate do desafiante contra o campeão (na pilha fora do tempo, OOT):
    - captura de EAs nos 10% de maior score de cada safra: diferença com IC 95% por bootstrap de CONTRATOS
      (1000 reamostras, seed 42), limite inferior > 0;
    - PR-AUC não pior que a do campeão (tolerância de 0,005).
Se o LightGBM também passar contra a logística, ele vira "novo campeão" em MODO SOMBRA: é escorado e comparado,
mas a publicação segue com a logística até o duelo confirmar.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.base_analitica import ALVO, CAT, NUM, montar_base, safras
from src.config import JANELA
from src.metricas import metricas
from src.modelos import criar_modelos, score_regua

NOMES = {"regua": "régua", "logistica": "logística", "lightgbm": "LightGBM"}
TOL_PR_AUC = 0.005


def _pp(x: float) -> str:
    return f"{x * 100:+.1f}".replace(".", ",")


def _dec(x: float, c: int = 3) -> str:
    return f"{x:.{c}f}".replace(".", ",")


def _top(score, safra, pct: float) -> np.ndarray:
    s, g = pd.Series(np.asarray(score, float)), pd.Series(np.asarray(safra))
    pos = s.groupby(g).rank(method="first", ascending=False)
    return (pos <= np.maximum(1, (pct * g.map(g.value_counts())).round())).to_numpy()


def captura_top(y, score, safra, pct: float = 0.10) -> float:
    y = np.asarray(y, float)
    return float(y[_top(score, safra, pct)].sum() / y.sum())


def ic_bootstrap_diff(df: pd.DataFrame, col_a: str, col_b: str, n: int = 1000, seed: int = 42,
                      pct: float = 0.10) -> tuple[float, float, float]:
    """Diferença de captura (b − a) e IC 95% reamostrando CONTRATOS (as fotos do mesmo contrato andam juntas).

    shortcut: o top 10% é fixado na amostra original (não recalculado a cada reamostragem); recalcular se a
    decisão ficar na fronteira.
    """
    y = df[ALVO].to_numpy(float)
    g = pd.factorize(df["id_contrato"])[0]
    k = g.max() + 1
    Y = np.bincount(g, y, k)
    A = np.bincount(g, y * _top(df[col_a], df["safra"], pct), k)
    B = np.bincount(g, y * _top(df[col_b], df["safra"], pct), k)
    rng = np.random.default_rng(seed)
    difs = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        difs[i] = (w @ B - w @ A) / (w @ Y)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return float((B.sum() - A.sum()) / Y.sum()), float(lo), float(hi)


def gate(df: pd.DataFrame, campeao: str, desafiante: str) -> dict:
    dif, lo, hi = ic_bootstrap_diff(df, campeao, desafiante)
    pr = {c: metricas(df[ALVO], df[c])["pr_auc"] for c in (campeao, desafiante)}
    ok_ic, ok_pr = lo > 0, pr[desafiante] >= pr[campeao] - TOL_PR_AUC
    det = (f"captura no top 10% {_pp(dif)} pp (IC 95% {_pp(lo)} a {_pp(hi)}); "
           f"PR-AUC {_dec(pr[desafiante])} × {_dec(pr[campeao])}")
    if ok_ic and ok_pr:
        motivo = f"{NOMES.get(desafiante, desafiante)} passou contra a {NOMES.get(campeao, campeao)}: {det}"
    elif not ok_ic:
        motivo = f"{NOMES.get(desafiante, desafiante)} sem ganho claro sobre a {NOMES.get(campeao, campeao)}: {det}"
    else:
        motivo = f"{NOMES.get(desafiante, desafiante)} captura mais, mas com PR-AUC pior que a {NOMES.get(campeao, campeao)}: {det}"
    return {"campeao": campeao, "desafiante": desafiante, "passou": bool(ok_ic and ok_pr), "diff": dif,
            "ic": [lo, hi], "pr_auc": pr, "motivo": motivo}


def separar_oot(lista: list, n_oot: int = 6) -> tuple[list, list]:
    oot = list(lista[-n_oot:])
    treino = [s for s in lista if s + pd.Timedelta(days=JANELA) <= oot[0]]
    return treino, oot


def bases_das_safras(tabelas: dict, lista: list | None = None) -> dict:
    return {d: montar_base(d, tabelas) for d in (safras() if lista is None else lista)}


def pilha_oot(tabelas: dict, n_oot: int = 6, bases: dict | None = None) -> pd.DataFrame:
    bases = bases_das_safras(tabelas) if bases is None else bases
    treino, oot = separar_oot(sorted(bases), n_oot)
    tr = pd.concat(bases[s] for s in treino)
    te = pd.concat(bases[s].assign(safra=s) for s in oot).reset_index(drop=True)
    p = te[["safra", "id_contrato", "fase", ALVO]].assign(regua=score_regua(te))
    for nome, m in criar_modelos().items():
        p[nome] = clone(m).fit(tr[NUM + CAT], tr[ALVO]).predict_proba(te[NUM + CAT])[:, 1]
    return p


def avaliar_escada(tabelas: dict | None, df: pd.DataFrame | None = None, bases: dict | None = None) -> dict:
    df = pilha_oot(tabelas, bases=bases) if df is None else df
    degraus = {c: {**metricas(df[ALVO], df[c]), "captura_top10": captura_top(df[ALVO], df[c], df["safra"])}
               for c in ("regua", "logistica", "lightgbm")}
    g1 = gate(df, "regua", "logistica")
    campeao = "logistica" if g1["passou"] else "regua"
    g2 = gate(df, campeao, "lightgbm")
    publicado, sombra = campeao, None
    if g2["passou"]:
        if campeao == "logistica":
            sombra = "lightgbm"
        else:
            publicado = "lightgbm"
    lo, hi = (_pp(x) for x in g2["ic"])
    if sombra:
        frase = (f"LightGBM passou contra a logística (IC {lo} a {hi} pp): é o novo campeão e entra em modo sombra; "
                 f"a logística segue publicando até o duelo confirmar.")
    else:
        frase = f"{g2['motivo']}. Publica a {NOMES[publicado]}."
    return {"degraus": degraus, "gates": [g1, g2], "publicado": publicado, "sombra": sombra, "frase": frase,
            "safras_oot": [f"{pd.Timestamp(d):%Y-%m-%d}" for d in sorted(df["safra"].unique())]}
