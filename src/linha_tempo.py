"""Linha do tempo para a diretoria: quantas entregas a operação alcançaria, mês a mês, ligando para 10% do estoque.

    regua      sem modelo: deve mais do que o carro vale primeiro, depois o maior atraso
    congelado  a logística do primeiro dia, nunca retreinada
    mensal     a logística retreinada todo mês (o que publica)
    sombra     o LightGBM retreinado todo mês (o desafiante), quando existe
Como o dado é sintético, a entrega não depende de quem foi acionado: a comparação mede a QUALIDADE DA ORDEM.
Só dias maduros. Os títulos saem dos números, nunca escritos à mão.
"""
import pandas as pd

CAPACIDADE = 0.10
CORTE_REGIME = "2026-06"


def cenarios_mensais(esc_m: pd.DataFrame, esc_c: pd.DataFrame, esc_s: pd.DataFrame | None = None,
                     capacidade: float = CAPACIDADE) -> pd.DataFrame:
    e = esc_m.loc[esc_m["maduro"].astype(bool), ["dia", "id_contrato", "score", "regua", "ea_60d"]] \
        .rename(columns={"score": "mensal"})
    e = e.merge(esc_c[["dia", "id_contrato", "score"]].rename(columns={"score": "congelado"}),
                on=["dia", "id_contrato"], how="left")
    cen = ["regua", "congelado", "mensal"]
    if esc_s is not None and len(esc_s):
        e = e.merge(esc_s[["dia", "id_contrato", "score"]].rename(columns={"score": "sombra"}),
                    on=["dia", "id_contrato"], how="left")
        cen.append("sombra")
    k = (capacidade * e.groupby("dia")["id_contrato"].transform("size")).round()
    e["mes"] = e["dia"].dt.strftime("%Y-%m")
    linhas = []
    for c in cen:
        topo = e.groupby("dia")[c].rank(method="first", ascending=False) <= k
        t = e.assign(ac=topo, ea_ac=e["ea_60d"] * topo).groupby("mes").agg(
            acionados=("ac", "sum"), eas_alcancadas=("ea_ac", "sum"), eas=("ea_60d", "sum"))
        for mes, r in t.iterrows():
            linhas.append({"mes": mes, "cenario": c, "pct_eas": r.eas_alcancadas / r.eas if r.eas else None,
                           "eas": int(r.eas), "acionados": int(r.acionados)})
    return pd.DataFrame(linhas)


def _pp(x: float, c: int = 1) -> str:
    return f"{x:.{c}f}".replace(".", ",")


def _sinal(x: float) -> str:
    return "+" if x >= 0 else "−"


def _pts(x: float) -> str:
    return "ponto" if abs(x) < 2 else "pontos"


def conclusoes(df: pd.DataFrame) -> list[dict]:
    cap = round(CAPACIDADE * 100)
    media = df.groupby("cenario")["pct_eas"].mean()
    pos = df[df["mes"] >= CORTE_REGIME]
    pos = pos if len(pos) else df
    p = pos.groupby("cenario")["pct_eas"].mean()
    perda = (p["mensal"] - p["congelado"]) * 100
    if perda >= 1:
        t_ret = f"Sem retreino, o modelo perde {_pp(perda)} {_pts(perda)} depois de jun/2026"
    elif perda <= -1:
        t_ret = f"Depois de jun/2026, o retreino mensal ficou {_sinal(perda)}{_pp(abs(perda))} {_pts(perda)} atrás do congelado"
    else:
        t_ret = (f"Sem retreino, o resultado depois de jun/2026 praticamente não muda "
                 f"({_sinal(perda)}{_pp(abs(perda))} {_pts(perda)})")
    out = [
        {"titulo": f"Com {cap}% do estoque, o modelo alcança {_pp(media['mensal'] * 100, 0)}% das entregas",
         "texto": f"A régua sem modelo (deve mais do que o carro vale, depois o maior atraso) alcança "
                  f"{_pp(media['regua'] * 100, 0)}% com a mesma capacidade. Média dos meses maduros.",
         "numero": f"{_pp(media['mensal'] * 100, 0)}%"},
        {"titulo": t_ret,
         "texto": f"De jun/2026 em diante (mudança de perfil da carteira), o retreino mensal alcançou "
                  f"{_pp(p['mensal'] * 100)}% das entregas contra {_pp(p['congelado'] * 100)}% do modelo congelado.",
         "numero": f"{_sinal(perda)}{_pp(abs(perda))} pp"},
    ]
    if "sombra" in media.index:
        d = (media["sombra"] - media["mensal"]) * 100
        out.append({"titulo": f"O desafiante em sombra alcança {_pp(media['sombra'] * 100, 0)}% das entregas",
                    "texto": f"LightGBM retreinado nas mesmas datas: {'+' if d >= 0 else '−'}{_pp(abs(d))} "
                             f"{_pts(d)} sobre a logística publicada.",
                    "numero": f"{'+' if d >= 0 else '−'}{_pp(abs(d))} pp"})
    return out
