"""Calcula tudo o que o painel mostra e grava UM HTML com os dados embutidos (docs/index.html).

    python -m src.treinar          # antes: relatorios/escada.json, vazamento.json, importancia.csv
    python -m src.montar_painel    # escoragem diária (cache em dados/escoragem/), agregados e HTML

Contrato com o front: chave ausente = card escondido. O front nunca reclassifica nem recalcula modelo; só filtra,
soma dias e divide. Nada posterior a HOJE entra; alvo só de dia maduro (D + 60 <= HOJE).
Contagens em contratos-dia: somar os dias de um período dá contatos/eventos do período (não clientes únicos).
Cada evento leva a classe que o contrato tinha NA DATA do evento (última foto <= data; senão a primeira depois).
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.apto import apto_ea, proximo_passo
from src.base_analitica import ALVO, CAT, NUM, carregar, montar_base
from src.candidatos import bases_das_safras
from src.config import CANAIS, CLASSES, FASES, FAIXAS, FAIXAS_ATRASO, HOJE, INICIO_PAINEL, JANELA, fase_de
from src.cortes import classificar, cortes_vigentes
from src.escoragem import escorar_periodos, treinar_vigente
from src.linha_tempo import cenarios_mensais, conclusoes
from src.qualidade import backtest_producao, estabilidade_mensal, ordenacao_semanal
from src.sinal import contribuicoes, direcoes, motivos

RAIZ = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "painel.html"
MARCADOR = "/*__DADOS__*/"
CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
FONTES = ("https://fonts.googleapis.com/css2?family=Sora:wght@700;800&family=Inter:wght@400;600;700;800"
          "&family=JetBrains+Mono:wght@600&display=swap")
N_FILA = 200
PERFIL = ["dias_atraso", "ltv_atual", "pct_pago", "idade_veiculo", "divida_total", "qtd_cpc_30d",
          "dias_desde_ult_cpc", "idade"]
METRICAS_FLUXO = ["trabalhado", "discagem", "alo", "cpc", "indicacoes", "form_por_form", "form_por_ind", "insucessos"]


def _iso(d) -> str:
    return f"{pd.Timestamp(d):%Y-%m-%d}"


def _limpo(x):
    """Pronto para json.dumps(allow_nan=False): NaN/inf viram None, numpy vira Python, datas viram texto."""
    if isinstance(x, dict):
        return {str(k): _limpo(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpo(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return _iso(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is pd.NaT or (x is not None and not isinstance(x, (str, bool, int)) and pd.isna(x)):
        return None
    return x


def _recs(df: pd.DataFrame) -> list[dict]:
    return _limpo(df.to_dict(orient="records"))


def dia_util_seguinte(dias: pd.DatetimeIndex, datas: pd.Series) -> pd.Series:
    """Data em dia não útil vai para o próximo dia útil do período; depois do último dia útil, NaT."""
    d = np.asarray(pd.DatetimeIndex(dias).values)
    v = pd.to_datetime(datas).to_numpy()
    i = np.searchsorted(d, v, side="left")
    ok = i < len(d)
    out = np.full(len(v), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = d[i[ok]]
    return pd.Series(out, index=getattr(datas, "index", None))


def _asof(ev: pd.DataFrame, esc: pd.DataFrame, col: str) -> pd.DataFrame:
    """classe, faixa e fase do contrato na data do evento: última foto <= data; senão a primeira depois."""
    cols = ["classe", "faixa_atraso", "fase"]
    if len(ev) == 0:
        return pd.DataFrame(columns=cols, index=ev.index)
    e = pd.DataFrame({"_i": np.arange(len(ev)), "id_contrato": ev["id_contrato"].astype("int64").to_numpy(),
                      "_d": pd.to_datetime(ev[col]).astype("datetime64[ns]").to_numpy()}).sort_values("_d")
    s = esc[["dia", "id_contrato"] + cols].copy()
    s["id_contrato"] = s["id_contrato"].astype("int64")
    s["dia"] = s["dia"].astype("datetime64[ns]")
    for c in cols:
        s[c] = s[c].astype(str)
    s = s.sort_values("dia")
    tras = pd.merge_asof(e, s, left_on="_d", right_on="dia", by="id_contrato", direction="backward")
    frente = pd.merge_asof(e, s, left_on="_d", right_on="dia", by="id_contrato", direction="forward")
    r = tras.set_index("_i")[cols].combine_first(frente.set_index("_i")[cols]).sort_index()
    r["classe"] = r["classe"].fillna("SEM CLASSE")
    r.index = ev.index
    return r


def classe_asof(eventos: pd.DataFrame, esc: pd.DataFrame, col_data: str) -> pd.Series:
    return _asof(eventos, esc, col_data)["classe"]


def agregar_fluxo(contatos: pd.DataFrame, eventos: pd.DataFrame, esc: pd.DataFrame, dias, hoje=HOJE) -> pd.DataFrame:
    """Contatos e eventos por (dia útil, faixa, canal, classe as-of), em contratos-dia."""
    dias = pd.DatetimeIndex(dias)
    ini, hoje = dias[0] - pd.Timedelta(days=3), pd.Timestamp(hoje)
    partes = []

    def parte(df, col, **metricas):
        if len(df) == 0:
            return
        df = df[(df[col] >= ini) & (df[col] <= hoje)]
        a = _asof(df, esc, col)
        p = pd.DataFrame({"dia": dia_util_seguinte(dias, df[col]).to_numpy(), "faixa": a["faixa_atraso"].to_numpy(),
                          "canal": df["canal"].to_numpy(), "classe": a["classe"].to_numpy()})
        for k, v in metricas.items():
            p[k] = np.asarray(v(df) if callable(v) else v, dtype=int)
        partes.append(p)

    parte(contatos, "data", trabalhado=1, discagem=lambda d: d["canal"] == "discador", alo=lambda d: d["alo"],
          cpc=lambda d: d["cpc"])
    parte(eventos, "data_indicacao", indicacoes=1,
          form_por_ind=lambda d: (d["desfecho"] == "FORMALIZADA") & (d["data_desfecho"] <= hoje))
    parte(eventos[eventos["desfecho"] == "FORMALIZADA"], "data_desfecho", form_por_form=1)
    parte(eventos[eventos["desfecho"] == "INSUCESSO"], "data_desfecho", insucessos=1)
    if not partes:
        return pd.DataFrame(columns=["dia", "faixa", "canal", "classe"] + METRICAS_FLUXO)
    f = pd.concat(partes, ignore_index=True)
    f = f[f["dia"].notna() & f["classe"].isin(CLASSES) & f["faixa"].isin(FAIXAS)]
    for m in METRICAS_FLUXO:
        f[m] = f[m].fillna(0).astype(int) if m in f else 0
    return f.groupby(["dia", "faixa", "canal", "classe"], as_index=False)[METRICAS_FLUXO].sum()


def _colunar(df: pd.DataFrame, dias: list, chaves: dict) -> dict:
    """Colunas de dimensão viram índices (dia → posição em meta.dias etc.): o payload fica pequeno."""
    pos_dia = {d: i for i, d in enumerate(dias)}
    out = {"dia": [pos_dia[_iso(d)] for d in df["dia"]]}
    for col, lista in chaves.items():
        idx = {v: i for i, v in enumerate(lista)}
        out[col] = [idx[str(v)] for v in df[col]]
    for c in df.columns:
        if c not in out and c != "dia":
            out[c] = _limpo(df[c].tolist())
    return out


def _fila(t: dict, bases: dict, hoje) -> list[dict]:
    vig = treinar_vigente(hoje, bases, "logistica")
    if vig is None:
        return []
    b = montar_base(hoje, t)
    score = vig.modelo.predict_proba(b[NUM + CAT])[:, 1]
    b = b.assign(score=score, classe=classificar(score, b["fase"], vig.cortes))
    top = b[b["classe"] == "ALTO"].sort_values("score", ascending=False).head(N_FILA)
    if top.empty:
        return []
    top = top.assign(apto=apto_ea(top, t, hoje).to_numpy())
    mot = motivos(contribuicoes(vig.modelo, top[NUM + CAT], vig.ref), direcoes(vig.modelo, top[NUM + CAT], vig.ref))
    return [{"id_contrato": int(r.id_contrato), "score": float(r.score), "classe": "ALTO", "fase": r.fase,
             "dias_atraso": int(r.dias_atraso), "ltv_atual": float(r.ltv_atual), "apto": r.apto,
             "proximo_passo": proximo_passo(r.apto, "ALTO"), "motivos": m}
            for r, m in zip(top.itertuples(), mot)]


def _perfil(bases: dict, hoje) -> dict | None:
    maduras = [s for s in sorted(bases) if s + pd.Timedelta(days=JANELA) <= hoje][-6:]
    if not maduras:
        return None
    b = pd.concat(bases[s] for s in maduras)
    f = b[b[ALVO] == 1]
    if f.empty:
        return None
    return _limpo({"formaliza": f[PERFIL].median().to_dict(), "base": b[PERFIL].median().to_dict(),
                   "n_formaliza": len(f), "n_base": len(b), "safras": [_iso(s) for s in maduras]})


def _uf(t: dict, esc: pd.DataFrame, ev: pd.DataFrame) -> list[dict]:
    n_dias = esc["dia"].nunique()
    est = esc.groupby(esc["uf"].astype(str), observed=True).size() / n_dias
    uf_c = t["contratos"].set_index("id_contrato")["uf"]
    ind = ev["id_contrato"].map(uf_c).value_counts()
    form = ev.loc[ev["desfecho"] == "FORMALIZADA", "id_contrato"].map(uf_c).value_counts()
    return [{"uf": u, "contratos": float(est.get(u, 0)), "indicacoes": int(ind.get(u, 0)),
             "formalizadas": int(form.get(u, 0))} for u in sorted(est.index)]


def _eventos(ev: pd.DataFrame, esc: pd.DataFrame, ini, hoje) -> list[dict]:
    linhas = []
    for tipo, sub, col in (("indicacao", ev, "data_indicacao"),
                           ("formalizada", ev[ev["desfecho"] == "FORMALIZADA"], "data_desfecho"),
                           ("insucesso", ev[ev["desfecho"] == "INSUCESSO"], "data_desfecho")):
        sub = sub[(sub[col] >= ini) & (sub[col] <= hoje)]
        a = _asof(sub, esc, col)
        for r, (_, x) in zip(sub.itertuples(), a.iterrows()):
            linhas.append({"data": _iso(getattr(r, col)), "tipo": tipo, "id_contrato": int(r.id_contrato),
                           "classe": x["classe"], "fase": x["fase"] if isinstance(x["fase"], str) else None,
                           "canal": r.canal, "operador": r.operador,
                           "motivo": r.motivo_insucesso if tipo == "insucesso" else ""})
    return sorted(linhas, key=lambda z: (z["data"], z["tipo"], z["id_contrato"]))


def _modelo(relatorios: Path, esc, cortes, esc_c, esc_s) -> dict:
    escada = json.loads((relatorios / "escada.json").read_text(encoding="utf-8"))
    curvas = escada.pop("curvas", {})
    m = {"escada": escada,
         "vazamento": json.loads((relatorios / "vazamento.json").read_text(encoding="utf-8")),
         "metricas_oot": escada["degraus"][escada["publicado"]]}
    if curvas:
        m["curva_pr"] = {**curvas["pr"], "taxa_base": curvas["taxa_base"]}
        m["captura"] = {"modelo": curvas["captura"], "regua": curvas["captura_regua"]}
    if len(cortes):
        m["cortes"] = _recs(cortes.assign(vigente_desde=cortes["vigente_desde"].map(_iso)))
    bt = backtest_producao(esc)
    if bt["contratos_dia"].sum() > 0:
        m["backtest"] = _recs(bt)
        o = ordenacao_semanal(esc)
        if len(o):
            m["ordenacao"] = _recs(o.assign(semana=o["semana"].map(_iso)))
        s = estabilidade_mensal(esc)
        if len(s):
            m["estabilidade"] = _recs(s)
        cen = cenarios_mensais(esc, esc_c, esc_s if escada.get("sombra") else None)
        if len(cen):
            m["tempo"] = {"cenarios": _recs(cen), "conclusoes": conclusoes(cen)}
            if escada.get("sombra"):
                d = cen[cen["cenario"].isin(["mensal", "sombra"])].pivot(index="mes", columns="cenario",
                                                                         values="pct_eas").reset_index()
                m["sombra"] = {"modelo": escada["sombra"], "meses": _recs(d)}
    imp = relatorios / "importancia.csv"
    if imp.exists():
        m["importancia"] = _recs(pd.read_csv(imp).head(12))
    return m


def montar_dados(tabelas: dict, hoje=HOJE, inicio=INICIO_PAINEL, cache=None, relatorios=RAIZ / "relatorios") -> dict:
    hoje, inicio, relatorios = pd.Timestamp(hoje), pd.Timestamp(inicio), Path(relatorios)
    t = {k: v for k, v in tabelas.items() if not k.startswith("_")}
    escada = json.loads((relatorios / "escada.json").read_text(encoding="utf-8"))
    modos = ("mensal", "congelado") + (("sombra",) if escada.get("sombra") else ())
    r = escorar_periodos(t, modos, inicio, hoje, cache)
    esc, cortes = r["mensal"]
    esc_c = r["congelado"][0]
    esc_s = r["sombra"][0] if "sombra" in r else None
    dias = [_iso(d) for d in sorted(esc["dia"].unique())]
    ev = t["eventos_ea"]

    est = esc.groupby([esc["dia"], esc["faixa_atraso"].astype(str).rename("faixa"),
                       esc["classe"].astype(str).rename("classe")], observed=True) \
        .agg(contratos=("id_contrato", "size"), divida=("divida_total", "sum")).reset_index()
    fl = agregar_fluxo(t["contatos"], ev, esc, pd.DatetimeIndex(dias), hoje)
    bases = bases_das_safras(t)
    meta = {"hoje": _iso(hoje), "dias": dias, "faixas": FAIXAS, "fase_da_faixa": fase_de(pd.Series([a for a, _ in FAIXAS_ATRASO])).tolist(),
            "fases": FASES, "classes": CLASSES, "canais": CANAIS,
            "maduro_ind": [pd.Timestamp(d) + pd.Timedelta(days=JANELA) <= hoje for d in dias],
            "janela": JANELA, "n_contratos": int(len(t["contratos"])),
            "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M")}
    dados = {"meta": meta,
             "estoque": _colunar(est.assign(divida=est["divida"].round(0)), dias, {"faixa": FAIXAS, "classe": CLASSES}),
             "fluxo": _colunar(fl, dias, {"faixa": FAIXAS, "canal": CANAIS, "classe": CLASSES}),
             "fila": _fila(t, bases, hoje), "uf": _uf(t, esc, ev[(ev["data_indicacao"] >= inicio)
                                                                 & (ev["data_indicacao"] <= hoje)]),
             "eventos": _eventos(ev, esc, inicio, hoje),
             "modelo": _modelo(relatorios, esc, cortes, esc_c, esc_s)}
    perfil = _perfil(bases, hoje)
    if perfil:
        dados["perfil"] = perfil
    return _limpo(dados)


def montar_html(dados: dict) -> str:
    js = json.dumps(dados, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count(MARCADOR) == 1, "o template precisa ter o marcador de dados exatamente uma vez"
    return (html.replace(MARCADOR, "/*__INICIO_DADOS__*/" + js + "/*__FIM_DADOS__*/")
            .replace("{{CHARTJS}}", CHARTJS).replace("{{FONTES}}", FONTES))


def main(saida=RAIZ / "docs" / "index.html"):
    t = carregar()
    dados = montar_dados(t, cache=RAIZ / "dados" / "escoragem")
    pasta = RAIZ / "dados" / "escoragem"
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "payload.json").write_text(json.dumps(dados, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(montar_html(dados), encoding="utf-8")
    print(f"{saida} ({saida.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
