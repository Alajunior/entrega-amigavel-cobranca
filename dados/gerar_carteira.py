"""Gera a "Carteira Exemplo" de financiamento de veículos em cobrança (01/01/2025 a 30/11/2026). Tudo FICTÍCIO.

Uso:
    python dados/gerar_carteira.py                 # 120 mil contratos, seed 42
    python dados/gerar_carteira.py 3000            # base menor (usada na CI)
    python dados/gerar_carteira.py 3000 --saida /tmp/carteira

Saída (padrão em dados/carteira_exemplo/), em parquet:
    contratos   1 linha por contrato (cadastro, veículo, financiamento, entrada e saída da carteira)
    contatos    1 linha por contrato-dia com tentativa (discador ou digital), ALÔ, CPC, desfecho e tabulação
    eventos_ea  1 linha por indicação de Entrega Amigável (EA) e o desfecho: FORMALIZADA, INSUCESSO ou ABERTA
    acordos     acordos de pagamento (quebrado ou não)

Como a carteira funciona:
    - o contrato ENTRA na carteira no dia em que completa 31 dias de atraso (data_entrada). Enquanto está
      ativo, dias de atraso em D = 31 + (D - data_entrada) (simplificação: acordo não reduz o atraso).
    - sai por pagamento (PAGO), busca e apreensão (APREENDIDO), EA formalizada (EA) ou ao passar de 720 dias
      (LIMITE_720). Quem não saiu até HOJE está ATIVO.
    - a simulação é semanal, com riscos concorrentes. O risco de decidir pela EA é latente (não é salvo) e sobe
      com atraso, ltv (deve mais do que o carro vale), pct pago baixo, bureau ruim, CPC recente e acordo quebrado,
      com interações (ltv alto x carro velho) que uma logística não captura sozinha.
    - a decisão vira INDICAÇÃO no primeiro CPC dos 14 dias seguintes (se não houver, o cliente retorna e gera um
      CPC). A indicação é formalizada (5 a 50 dias depois) ou termina em insucesso com motivo; depois de um
      insucesso o contrato volta ao risco normal, com risco menor de nova indicação.
    - mudança de regime em 01/06/2026: entram contratos com bureau pior, a cura fica mais lenta (o estoque
      envelhece) e o efeito do ltv sobre a EA fica mais forte (o modelo antigo perde ordenação).
    - tabulação do operador: em ~15% das ligações com desfecho de EA o operador lança outra coisa.

Valores medidos com a base padrão (120 mil contratos, ~99 mil com passagem pelo período, geração em ~10 s):
    estoque ativo ~28 mil por dia útil; ~3,2 mil tentativas e ~580 CPC por dia útil; ~7 indicações e ~4,4
    formalizações por dia útil; taxa ea_60d nas safras maduras de 0,45% (abr/2025) a 1,1% (set/2026), média 0,55%.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOJE = pd.Timestamp("2026-11-30")
INICIO = pd.Timestamp("2025-01-01")
REGIME = pd.Timestamp("2026-06-01")
D0 = INICIO - pd.Timedelta(days=400)          # dia 0 da simulação (estoque que já estava em atraso)
DIA_HOJE, DIA_INICIO, DIA_REGIME = ((x - D0).days for x in (HOJE, INICIO, REGIME))
DUR_MAX = 690                                  # 31 + 690 = 721 dias de atraso: sai da carteira
SEMANAS = DUR_MAX // 7 + 1
SAIDA_PADRAO = Path(__file__).parent / "carteira_exemplo"

UFS = ["SP", "MG", "RJ", "BA", "PR", "RS", "PE", "CE", "PA", "SC", "GO", "MA", "AM", "ES", "PB", "RN", "MT", "AL",
       "PI", "DF", "MS", "SE", "RO", "TO", "AC", "AP", "RR"]
PESO_UF = np.array([22, 10, 8, 7, 5.5, 5.3, 4.5, 4.3, 4.1, 3.6, 3.4, 3.3, 2, 1.9, 1.9, 1.7, 1.8, 1.6, 1.6, 1.4, 1.4,
                    1.1, 0.9, 0.8, 0.4, 0.4, 0.3])
BUREAU = np.array(list("ABCDE"))
EFEITO_BUREAU_EA = np.array([-0.5, -0.25, 0.0, 0.3, 0.55])
EFEITO_BUREAU_CURA = np.array([0.6, 0.35, 0.0, -0.3, -0.6])
DESFECHOS = np.array(["SEM DESFECHO", "ACORDO", "RENEGOCIACAO", "ALEGA PAGAMENTO", "ACAO CONTRA", "RECUSA EA",
                      "EA INDICADA"])
P_DESFECHO = np.array([0.62, 0.12, 0.06, 0.08, 0.02, 0.10])          # sem "EA INDICADA": essa vem da decisão
MOTIVOS = np.array(["AVALIACAO DO VEICULO", "DOCUMENTACAO", "DESISTENCIA", "DEBITOS DO VEICULO", "NAO LOCALIZADO"])
P_ATTEMPT = 0.12


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _datas(dias) -> pd.Series:
    d = pd.Series(dias, dtype="float64")
    return (D0 + pd.to_timedelta(d, unit="D")).astype("datetime64[ns]")


def _contratos(rng, n):
    entrada = rng.integers(0, DIA_HOJE + 1, n)
    novo = entrada >= DIA_REGIME
    bureau = np.where(novo, rng.choice(5, n, p=[.07, .16, .29, .27, .21]), rng.choice(5, n, p=[.12, .22, .30, .22, .14]))
    plano = rng.choice([24, 36, 48, 60], n, p=[.15, .3, .3, .25])
    pagas = np.floor(rng.beta(1.3, 2.6, n) * plano).astype(int)
    contratacao = entrada - 31 - pagas * 30 - rng.integers(0, 30, n)
    ano_contr = (D0 + pd.to_timedelta(contratacao, unit="D")).year.to_numpy()
    idade_carro = rng.choice(np.arange(0, 12), n, p=np.r_[[.2, .12, .1, .09, .08, .08, .07, .07, .06, .05], [.04, .04]])
    valor_veic = np.round(rng.lognormal(np.log(62000) - 0.035 * idade_carro, 0.38), -2)
    financiado = np.round(valor_veic * rng.uniform(0.5, 0.92, n), 2)
    r = 0.021
    parcela = np.round(financiado * r / (1 - (1 + r) ** -plano), 2)
    n_cli = max(1, int(round(n / 1.15)))
    id_cli = np.concatenate([np.arange(1, n_cli + 1), rng.integers(1, n_cli + 1, n - n_cli)])[:n]
    nasc = D0 + pd.to_timedelta(-rng.integers(20 * 365, 72 * 365, n), unit="D")
    return pd.DataFrame({
        "id_contrato": np.arange(1, n + 1), "id_cliente": rng.permutation(id_cli),
        "data_nascimento": nasc.astype("datetime64[ns]"), "uf": rng.choice(UFS, n, p=PESO_UF / PESO_UF.sum()),
        "faixa_bureau": BUREAU[bureau], "ano_veiculo": ano_contr - idade_carro, "valor_veiculo_inicial": valor_veic,
        "data_contratacao": _datas(contratacao), "valor_financiado": financiado, "plano": plano,
        "valor_parcela": parcela, "parcelas_pagas": pagas, "data_entrada": _datas(entrada),
    }), entrada, bureau, contratacao


def _contatos_potenciais(rng, entrada, contatavel):
    """Tentativas em todo o período em que o contrato PODERIA estar ativo (corta-se depois da saída)."""
    ini = np.maximum(entrada, DIA_INICIO)
    fim = np.minimum(entrada + DUR_MAX, DIA_HOJE + 1)               # exclusivo
    span = np.clip(fim - ini, 0, None)
    k = rng.binomial(span, P_ATTEMPT)
    idx = np.repeat(np.arange(len(entrada)), k)
    dia = ini[idx] + np.floor(rng.random(len(idx)) * span[idx]).astype(np.int64)
    ordem = np.lexsort((dia, idx))
    idx, dia = idx[ordem], dia[ordem]
    unico = np.r_[True, (idx[1:] != idx[:-1]) | (dia[1:] != dia[:-1])]
    idx, dia = idx[unico], dia[unico]
    m = len(idx)
    digital = rng.random(m) < 0.2
    c = contatavel[idx]
    alo = np.where(digital, rng.random(m) < 0.10 + 0.25 * c, rng.random(m) < 0.22 + 0.35 * c)
    cpc = alo & np.where(digital, True, rng.random(m) < 0.30 + 0.45 * c)
    desf = np.where(cpc, rng.choice(len(P_DESFECHO), m, p=P_DESFECHO), -1)
    oper = np.where(digital, 0, rng.integers(1, 25, m))
    return pd.DataFrame({"i": idx, "dia": dia, "digital": digital, "alo": alo, "cpc": cpc, "desf": desf, "oper": oper})


def _bins_semanais(df, entrada, n, mask):
    """Contagem por (contrato, semana desde a entrada) dos eventos marcados em mask."""
    sem = (df["dia"].to_numpy() - entrada[df["i"].to_numpy()]) // 7
    m = mask & (sem >= 0) & (sem < SEMANAS)
    b = np.zeros((n, SEMANAS), dtype=np.int16)
    np.add.at(b, (df["i"].to_numpy()[m], sem[m]), 1)
    return b


def _simular(rng, n, entrada, bureau, contratos, cont, quebra):
    """Laço semanal com riscos concorrentes. Devolve saída (dia, motivo) e as decisões de EA."""
    pct_pago = contratos["parcelas_pagas"].to_numpy() / contratos["plano"].to_numpy()
    divida = contratos["valor_parcela"].to_numpy() * (contratos["plano"].to_numpy() - contratos["parcelas_pagas"].to_numpy())
    valor0 = contratos["valor_veiculo_inicial"].to_numpy()
    ano_v = contratos["ano_veiculo"].to_numpy()
    fragil = rng.normal(0, 0.6, n)                                    # heterogeneidade não observada
    cpc_sem = _bins_semanais(cont, entrada, n, cont["cpc"].to_numpy())
    desf = cont["desf"].to_numpy()
    acordo = desf == 1
    quebrado = acordo & quebra
    ok_sem = _bins_semanais(cont, entrada, n, acordo & ~quebrado)
    queb_sem = _bins_semanais(cont, entrada, n, quebrado)

    saida = np.full(n, -1, dtype=np.int64)
    motivo = np.full(n, "ATIVO", dtype=object)
    livre_em = np.zeros(n, dtype=np.int64)                             # em processo de EA até esse dia
    insucessos = np.zeros(n, dtype=np.int64)
    decisoes = []                                                      # (i, dia da indicação, dia do desfecho, formalizou)

    for w in range(SEMANAS):
        dia0 = entrada + 7 * w
        ativo = (saida < 0) & (dia0 <= DIA_HOJE)
        if not ativo.any():
            break
        dias_atraso = 31 + 7 * w
        if dias_atraso + 7 > 721:                                     # completa 721 dias nesta semana
            lim = ativo & (entrada + DUR_MAX <= DIA_HOJE)
            saida[lim], motivo[lim] = entrada[lim] + DUR_MAX, "LIMITE_720"
            break
        ano_cal = D0.year + (D0.dayofyear + dia0) / 365.25
        idade_v = np.clip(ano_cal - ano_v, 0, None)
        ltv = divida / (valor0 * 0.85 ** idade_v)
        cpc30 = cpc_sem[:, max(0, w - 4):w].sum(1)
        ok_rec = ok_sem[:, max(0, w - 2):w + 1].sum(1) > 0
        queb26 = queb_sem[:, max(0, w - 26):w].sum(1) > 0
        regime = dia0 >= DIA_REGIME
        f_dias = -0.9 if dias_atraso < 90 else (0.0 if dias_atraso <= 180 else (0.45 if dias_atraso <= 400 else 0.15))
        l_ea = (-10.6 + f_dias + (1.15 + 0.35 * regime) * np.clip(ltv, 0, 2.5)
                + 0.9 * ((ltv > 1.1) & (idade_v > 7)) + EFEITO_BUREAU_EA[bureau] - 1.3 * pct_pago
                + 0.35 * np.minimum(cpc30, 3) + 0.25 * (cpc30 > 0) * (bureau >= 3) + 0.6 * queb26
                - 0.8 * (insucessos > 0) + fragil)
        h_ea = _sig(l_ea) * (dia0 >= DIA_INICIO) * (dia0 >= livre_em)
        h_cura = _sig(-3.3 + EFEITO_BUREAU_CURA[bureau] + 2.2 * ok_rec - 0.0035 * dias_atraso - 0.35 * regime)
        h_bap = _sig(-6.6 + 0.004 * min(dias_atraso, 500))
        livre = ativo & (dia0 >= livre_em)
        u = rng.random(n)
        dia_ev = np.maximum(dia0 + rng.integers(0, 7, n), entrada + 1)      # nunca no próprio dia da entrada
        cura = livre & (u < h_cura) & (dia_ev <= DIA_HOJE)
        bap = livre & ~cura & (u < h_cura + h_bap) & (dia_ev <= DIA_HOJE)
        ea = livre & ~cura & ~bap & (u < h_cura + h_bap + h_ea) & (dia_ev <= DIA_HOJE)
        saida[cura], motivo[cura] = dia_ev[cura], "PAGO"
        saida[bap], motivo[bap] = dia_ev[bap], "APREENDIDO"
        if ea.any():
            # o desfecho da indicação é resolvido aqui para o laço saber se o contrato sai ou volta ao risco
            ii = np.flatnonzero(ea)
            p_form = _sig(0.15 + 0.9 * (ltv[ii] > 1) - 0.08 * (idade_v[ii] - 5))
            form = rng.random(len(ii)) < p_form
            atraso_ind = rng.integers(0, 15, len(ii))
            prazo = rng.integers(5, 51, len(ii))
            dia_ind = dia_ev[ii] + atraso_ind
            dia_desf = dia_ind + prazo
            decisoes.extend(zip(ii, dia_ind, dia_desf, form))
            sai = form & (dia_desf <= DIA_HOJE) & (dia_desf < entrada[ii] + DUR_MAX)
            saida[ii[sai]], motivo[ii[sai]] = dia_desf[sai], "EA"
            livre_em[ii] = np.where(form, np.iinfo(np.int64).max, dia_desf + 1)
            insucessos[ii[~form]] += 1
    return saida, motivo, decisoes


def gerar(n_contratos: int, saida: Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_contratos
    contratos, entrada, bureau, _ = _contratos(rng, n)
    contatavel = rng.beta(2, 3, n)
    cont = _contatos_potenciais(rng, entrada, contatavel)
    quebra = rng.random(len(cont)) < 0.6
    dia_saida, motivo, decisoes = _simular(rng, n, entrada, bureau, contratos, cont, quebra)

    # ---------- indicações: no 1º CPC entre a decisão e a data sorteada; sem CPC, o cliente retorna (CPC novo) ----------
    dec = pd.DataFrame(decisoes, columns=["i", "dia_ind", "dia_desf", "form"])
    dec = dec[dec["dia_ind"] <= DIA_HOJE].reset_index(drop=True)
    novos = pd.DataFrame({"i": dec["i"], "dia": dec["dia_ind"], "digital": rng.random(len(dec)) < 0.15, "alo": True,
                          "cpc": True, "desf": 6, "oper": 0})
    novos["oper"] = np.where(novos["digital"], 0, rng.integers(1, 25, len(dec)))
    cont = pd.concat([cont.assign(q=quebra), novos.assign(q=False)], ignore_index=True)
    cont = cont.sort_values(["i", "dia", "desf"], ascending=[True, True, False]).drop_duplicates(["i", "dia"])
    # ---------- corta tudo o que cai depois da saída (ou antes do início dos dados) ----------
    fim = np.where(dia_saida >= 0, dia_saida, DIA_HOJE + 1)
    cont = cont[(cont["dia"].to_numpy() < fim[cont["i"].to_numpy()]) & (cont["dia"] >= DIA_INICIO)].reset_index(drop=True)

    # ---------- eventos de EA ----------
    dec = dec[dec["dia_ind"] < fim[dec["i"].to_numpy()]].copy()
    dec["desfecho"] = np.where(dec["dia_desf"] > DIA_HOJE, "ABERTA", np.where(dec["form"], "FORMALIZADA", "INSUCESSO"))
    # formalização que cairia depois do limite de 720 dias vira insucesso (o contrato saiu por limite antes)
    dec.loc[(dec["desfecho"] == "FORMALIZADA") & (motivo[dec["i"].to_numpy()] != "EA"), "desfecho"] = "INSUCESSO"
    ct_ind = cont[cont["desf"] == 6][["i", "dia", "digital", "oper"]]
    dec = dec.merge(ct_ind, left_on=["i", "dia_ind"], right_on=["i", "dia"], how="left")
    dec = dec.sort_values(["i", "dia_ind"]).reset_index(drop=True)
    ltv_alto = rng.random(len(dec))
    motivo_ins = np.where(dec["desfecho"] == "INSUCESSO", MOTIVOS[np.minimum((ltv_alto * 5).astype(int), 4)], "")
    eventos = pd.DataFrame({
        "id_indicacao": np.arange(1, len(dec) + 1), "id_contrato": dec["i"].to_numpy() + 1,
        "data_indicacao": _datas(dec["dia_ind"].to_numpy()), "canal": np.where(dec["digital"], "digital", "discador"),
        "operador": np.where(dec["digital"], "DIGITAL", [f"OP-{int(o):02d}" for o in dec["oper"]]),
        "desfecho": dec["desfecho"].to_numpy(),
        "data_desfecho": _datas(np.where(dec["desfecho"] == "ABERTA", np.nan, dec["dia_desf"].to_numpy())),
        "motivo_insucesso": motivo_ins,
    })

    # ---------- contatos e acordos finais ----------
    m = len(cont)
    desf = np.where(cont["cpc"], DESFECHOS[cont["desf"].clip(lower=0).to_numpy()], "")
    tab = np.where(~cont["alo"], "NAO ATENDEU", np.where(~cont["cpc"], "TERCEIRO", desf))
    diverge = np.isin(desf, ["EA INDICADA", "RECUSA EA"]) & (rng.random(m) < 0.15)
    tab = np.where(diverge, rng.choice(["SEM DESFECHO", "ACORDO", "RENEGOCIACAO"], m), tab)
    contatos = pd.DataFrame({
        "id_contrato": cont["i"].to_numpy() + 1, "data": _datas(cont["dia"].to_numpy()),
        "canal": np.where(cont["digital"], "digital", "discador"),
        "operador": np.where(cont["digital"], "DIGITAL", [f"OP-{int(o):02d}" for o in cont["oper"]]),
        "alo": cont["alo"].to_numpy(), "cpc": cont["cpc"].to_numpy(), "desfecho": desf, "tabulacao": tab,
        "id_ligacao": [f"L{k:08d}" for k in range(1, m + 1)],
    })
    ac = cont[cont["desf"] == 1]
    acordos = pd.DataFrame({"id_contrato": ac["i"].to_numpy() + 1, "data_acordo": _datas(ac["dia"].to_numpy()),
                            "quebrado": ac["q"].to_numpy()})

    contratos["data_saida"] = _datas(np.where(dia_saida >= 0, dia_saida, np.nan))
    contratos["motivo_saida"] = motivo
    # contratos que saíram antes do início dos dados nunca aparecem na carteira observada
    vivos = contratos["data_saida"].isna() | (contratos["data_saida"] > INICIO)
    contratos = contratos[vivos].reset_index(drop=True)
    tabelas = {"contratos": contratos, "contatos": contatos, "eventos_ea": eventos, "acordos": acordos}
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    for nome, df in tabelas.items():
        df.to_parquet(saida / f"{nome}.parquet", index=False)
    return tabelas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n", nargs="?", type=int, default=120_000)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    a = ap.parse_args()
    t = gerar(a.n, a.saida)
    print({k: len(v) for k, v in t.items()}, "→", a.saida)
