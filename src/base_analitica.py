"""Base analítica as-of: a foto do dia D com as variáveis que se sabia ANTES de D e o alvo dos 60 dias seguintes.

Uma só função para treino e escoragem (paridade). Regras anti-futuro:
    - contatos, indicações e insucessos entram só com data < D (a escoragem roda de manhã, antes do dia);
    - acordo só conta como "quebrado" 30 dias depois de feito (a quebra só é observável quando a parcela vence);
    - idade e idade do veículo são calculadas NA DATA D;
    - alvo ea_60d = 1 se a 1ª formalização de EA cai em (D, D+60]; NaN se D+60 > HOJE (alvo ainda não fechou).
O grão é o contrato (um contrato ativo em D = uma linha).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HOJE, JANELA, PASSO_SAFRA, PRIMEIRA_SAFRA, ULTIMA_MADURA, faixa_de, fase_de

PASTA = Path(__file__).resolve().parent.parent / "dados" / "carteira_exemplo"
TABELAS = ("contratos", "contatos", "eventos_ea", "acordos")
ALVO = "ea_60d"
NUM = ["idade", "dias_atraso", "valor_vencido", "divida_total", "pct_pago", "plano", "valor_parcela", "ltv_atual",
       "idade_veiculo", "qtd_tentativas_30d", "qtd_cpc_30d", "dias_desde_ult_cpc", "acordos_quebrados_180d",
       "insucessos_ea_antes"]
CAT = ["fase", "faixa_bureau", "uf_grupo", "canal_pref"]
REGIAO = {**dict.fromkeys(["SP", "RJ", "MG", "ES"], "SE"), **dict.fromkeys(["PR", "SC", "RS"], "S"),
          **dict.fromkeys(["BA", "PE", "CE", "MA", "PB", "RN", "AL", "PI", "SE"], "NE"),
          **dict.fromkeys(["PA", "AM", "RO", "TO", "AC", "AP", "RR"], "N"),
          **dict.fromkeys(["GO", "MT", "MS", "DF"], "CO")}


def safras(fim=ULTIMA_MADURA) -> list[pd.Timestamp]:
    return list(pd.date_range(PRIMEIRA_SAFRA, pd.Timestamp(fim), freq=f"{PASSO_SAFRA}D"))


def carregar(pasta: Path = PASTA) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(Path(pasta) / f"{n}.parquet") for n in TABELAS}


def _ano(d: pd.Timestamp) -> float:
    return d.year + d.dayofyear / 365.25


def _prep(t: dict) -> dict:
    """Índices por data (calculados uma vez por dict de tabelas): cada foto filtra uma janela, não milhões de linhas."""
    if "_prep" not in t:
        ct = t["contatos"].sort_values("data", kind="stable")
        cpc = ct[ct["cpc"]]
        ac = t["acordos"].sort_values("data_acordo", kind="stable")
        e = t["eventos_ea"]
        ins = e[e["desfecho"] == "INSUCESSO"].sort_values("data_desfecho", kind="stable")
        form = e[e["desfecho"] == "FORMALIZADA"].groupby("id_contrato")["data_desfecho"].min()
        t["_prep"] = {"ct": ct, "ct_d": ct["data"].to_numpy(), "cpc": cpc, "cpc_d": cpc["data"].to_numpy(),
                      "ac": ac, "ac_d": ac["data_acordo"].to_numpy(), "ins": ins,
                      "ins_d": ins["data_desfecho"].to_numpy(), "form": form}
    return t["_prep"]


def _janela(df, datas, ini, fim) -> pd.DataFrame:
    """Linhas com ini <= data < fim (datas ordenadas)."""
    a, b = np.searchsorted(datas, np.datetime64(ini), "left"), np.searchsorted(datas, np.datetime64(fim), "left")
    return df.iloc[a:b]


def montar_base(data_ref, tabelas: dict) -> pd.DataFrame:
    D = pd.Timestamp(data_ref)
    p = _prep(tabelas)
    c = tabelas["contratos"]
    ativo = (c["data_entrada"] <= D) & (c["data_saida"].isna() | (c["data_saida"] > D))
    b = c[ativo].copy()
    dia = pd.Timedelta(days=1)

    b["dias_atraso"] = 31 + (D - b["data_entrada"]).dt.days
    b["fase"] = fase_de(b["dias_atraso"]).to_numpy()
    b["faixa_atraso"] = faixa_de(b["dias_atraso"]).to_numpy()
    b["idade"] = _idade(b["data_nascimento"], D)
    meses = np.ceil(b["dias_atraso"] / 30)
    b["valor_vencido"] = np.round(b["valor_parcela"] * meses * 1.02 ** meses, 2)
    b["divida_total"] = np.round(b["valor_parcela"] * (b["plano"] - b["parcelas_pagas"]), 2)
    b["pct_pago"] = b["parcelas_pagas"] / b["plano"]
    b["idade_veiculo"] = np.clip(_ano(D) - b["ano_veiculo"], 0, None)
    b["ltv_atual"] = b["divida_total"] / (b["valor_veiculo_inicial"] * 0.85 ** b["idade_veiculo"])
    b["uf_grupo"] = b["uf"].map(REGIAO)

    ids = b["id_contrato"]
    c30 = _janela(p["ct"], p["ct_d"], D - 30 * dia, D)
    b["qtd_tentativas_30d"] = ids.map(c30["id_contrato"].value_counts()).fillna(0).astype(int).to_numpy()
    b["qtd_cpc_30d"] = ids.map(c30.loc[c30["cpc"], "id_contrato"].value_counts()).fillna(0).astype(int).to_numpy()
    ult = _janela(p["cpc"], p["cpc_d"], pd.Timestamp("1900-01-01"), D).groupby("id_contrato")["data"].max()
    b["dias_desde_ult_cpc"] = (D - ids.map(ult)).dt.days.to_numpy()
    c90 = _janela(p["ct"], p["ct_d"], D - 90 * dia, D)
    pref = c90.groupby(["id_contrato", "canal"]).size().unstack(fill_value=0)
    canal = pref.idxmax(axis=1) if len(pref) else pd.Series(dtype=object)
    b["canal_pref"] = ids.map(canal).fillna("nenhum").to_numpy()
    ac = _janela(p["ac"], p["ac_d"], D - 180 * dia, D - 30 * dia)
    b["acordos_quebrados_180d"] = ids.map(ac.loc[ac["quebrado"], "id_contrato"].value_counts()).fillna(0).astype(int).to_numpy()
    ins = _janela(p["ins"], p["ins_d"], pd.Timestamp("1900-01-01"), D)
    b["insucessos_ea_antes"] = ids.map(ins["id_contrato"].value_counts()).fillna(0).astype(int).to_numpy()

    b["data_form"] = ids.map(p["form"]).to_numpy()
    if D + pd.Timedelta(days=JANELA) <= HOJE:
        b[ALVO] = ((b["data_form"] > D) & (b["data_form"] <= D + pd.Timedelta(days=JANELA))).astype(float)
    else:
        b[ALVO] = np.nan
    cols = ["id_contrato", "id_cliente", "uf", "faixa_atraso"] + NUM + [x for x in CAT if x != "fase"] + ["fase", ALVO,
                                                                                                     "data_form"]
    return b[cols].reset_index(drop=True)


def _idade(nasc: pd.Series, D: pd.Timestamp) -> np.ndarray:
    antes = (nasc.dt.month > D.month) | ((nasc.dt.month == D.month) & (nasc.dt.day > D.day))
    return (D.year - nasc.dt.year - antes.astype(int)).to_numpy()
