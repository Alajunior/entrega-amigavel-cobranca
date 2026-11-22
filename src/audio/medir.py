"""Medição das regras contra o gabarito: matriz de confusão, alertas, ablação, oferta × tabulação e operador.

O gabarito é o que o roteiro mandou dizer; a regra lê o que o Whisper entendeu. A diferença entre os dois é o
custo real da transcrição, e a ablação mostra quanto a tolerância a erro de ASR recupera.
"""
import pandas as pd

from src.audio.regras import alertas_operador, classificar_oferta
from src.audio.roteiros import ALERTAS, DESFECHOS_OFERTA, OFERTAS
from src.audio.transcrever import wer

COLUNAS_REGRA = OFERTAS + ("INCONCLUSIVO", "SEM ÁUDIO")


def _lado(turnos: list[dict], lado: str) -> str:
    return " ".join(t.get("texto") or "" for t in turnos if t.get("lado") == lado)


def avaliar(roteiros: list[dict], transcricoes: dict[str, dict], tolerante: bool = True) -> pd.DataFrame:
    """Uma linha por ligação: o que o gabarito diz, o que a regra achou na transcrição e o WER de cada lado.
    Ligação sem transcrição (síntese ou transcrição falhou) → SEM ÁUDIO, fora das métricas de acerto."""
    linhas = []
    for r in roteiros:
        g, tr = r["gabarito"], transcricoes.get(r["id_ligacao"])
        base = {"id_ligacao": r["id_ligacao"], "data": r["data"], "operador": r["operador"],
                "tabulacao": r.get("tabulacao", ""), "desfecho": r.get("desfecho", ""),
                "oferta_gabarito": g["oferta"], "tipo_gabarito": g["tipo"], "alertas_gabarito": list(g["alertas"])}
        if tr is None:
            linhas.append(base | {"oferta_regra": "SEM ÁUDIO", "tipo_regra": None, "quem": None, "trecho": "",
                                  "acerto_oferta": False, "alertas_regra": [], "wer_operador": None,
                                  "wer_cliente": None, "sem_audio": True})
            continue
        c = classificar_oferta(tr["turnos"], tolerante)
        linhas.append(base | {
            "oferta_regra": c["oferta"], "tipo_regra": c["tipo"], "quem": c["quem"], "trecho": c["trecho"],
            "acerto_oferta": c["oferta"] == g["oferta"],
            "alertas_regra": [a["alerta"] for a in alertas_operador(tr["turnos"])],
            "wer_operador": round(wer(_lado(r["turnos"], "operador"), _lado(tr["turnos"], "operador")), 4),
            "wer_cliente": round(wer(_lado(r["turnos"], "cliente"), _lado(tr["turnos"], "cliente")), 4),
            "sem_audio": False})
    df = pd.DataFrame(linhas)
    for col in ("wer_operador", "wer_cliente"):          # None continua None (não vira NaN) para o JSON do painel
        df[col] = pd.Series([x[col] for x in linhas], index=df.index, dtype=object)
    return df


def matriz_confusao(df: pd.DataFrame) -> dict:
    """Linhas = gabarito, colunas = regra (inclui INCONCLUSIVO e SEM ÁUDIO)."""
    ct = pd.crosstab(df["oferta_gabarito"], df["oferta_regra"])
    ct = ct.reindex(index=list(OFERTAS), columns=list(COLUNAS_REGRA), fill_value=0)
    return {"linhas": list(OFERTAS), "colunas": list(COLUNAS_REGRA), "valores": ct.astype(int).values.tolist()}


def _razao(a: int, b: int):
    return round(a / b, 3) if b else None


def metricas_alertas(df: pd.DataFrame) -> dict[str, dict]:
    """Precisão e recall por alerta, só nas ligações com transcrição."""
    d = df[~df["sem_audio"]]
    out = {}
    for a in ALERTAS:
        g = d["alertas_gabarito"].map(lambda x: a in x)
        r = d["alertas_regra"].map(lambda x: a in x)
        vp, fp, fn = int((g & r).sum()), int((~g & r).sum()), int((g & ~r).sum())
        out[a] = {"vp": vp, "fp": fp, "fn": fn, "precisao": _razao(vp, vp + fp), "recall": _razao(vp, vp + fn)}
    return out


def _resumo_oferta(df: pd.DataFrame) -> dict:
    d = df[~df["sem_audio"]]
    of_g, of_r = d["oferta_gabarito"] == "OFERECEU", d["oferta_regra"] == "OFERECEU"
    return {"acerto": _razao(int(d["acerto_oferta"].sum()), len(d)),
            "recall_oferta": _razao(int((of_g & of_r).sum()), int(of_g.sum())),
            "precisao_oferta": _razao(int((of_g & of_r).sum()), int(of_r.sum()))}


def ablacao(roteiros: list[dict], transcricoes: dict[str, dict]) -> dict:
    """Regex simples × regras tolerantes, no texto do Whisper e no texto perfeito do roteiro. No texto perfeito as
    duas empatam; a distância no Whisper é o que a tolerância a erro de transcrição rende."""
    perfeito = {r["id_ligacao"]: {"id_ligacao": r["id_ligacao"], "turnos": r["turnos"]} for r in roteiros}
    out = {}
    for fonte, trs in (("whisper", transcricoes), ("roteiro", perfeito)):
        out[fonte] = {modo: _resumo_oferta(avaliar(roteiros, trs, tolerante=(modo == "tolerante")))
                      for modo in ("simples", "tolerante")}
    s, t = avaliar(roteiros, transcricoes, False), avaliar(roteiros, transcricoes, True)
    ganho = t["acerto_oferta"] & ~s["acerto_oferta"]
    out["ganhos"] = [{"id_ligacao": i, "trecho": tr, "simples": so}
                     for i, tr, so in zip(t.loc[ganho, "id_ligacao"], t.loc[ganho, "trecho"], s.loc[ganho, "oferta_regra"])]
    return out


def cruzamento_tabulacao(df: pd.DataFrame) -> pd.DataFrame:
    """Oferta da regra × tabulação do operador; `divergente` marca quem ofereceu e tabulou outra coisa."""
    d = df[~df["sem_audio"]]
    c = d.groupby(["oferta_regra", "tabulacao"]).size().rename("ligacoes").reset_index()
    c["divergente"] = (c["oferta_regra"] == "OFERECEU") & ~c["tabulacao"].isin(DESFECHOS_OFERTA)
    return c.sort_values(["ligacoes", "oferta_regra"], ascending=[False, True]).reset_index(drop=True)


def ranking_operador(df: pd.DataFrame) -> pd.DataFrame:
    """Por operador: ligações auditadas, taxa de oferta (regra), alertas e acerto da regra."""
    d = df[~df["sem_audio"]].assign(ofereceu=lambda x: x["oferta_regra"] == "OFERECEU",
                                    n_alertas=lambda x: x["alertas_regra"].map(len))
    r = d.groupby("operador").agg(ligacoes=("id_ligacao", "size"), taxa_oferta=("ofereceu", "mean"),
                                  alertas=("n_alertas", "sum"), acerto=("acerto_oferta", "mean")).reset_index()
    r[["taxa_oferta", "acerto"]] = r[["taxa_oferta", "acerto"]].round(3)
    return r.sort_values(["alertas", "taxa_oferta"], ascending=[False, False]).reset_index(drop=True)
