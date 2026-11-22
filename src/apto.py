"""Regra explícita APTO_EA por contrato (o score diz QUEM; a regra diz SE pode ofertar agora) e o próximo passo.

A primeira regra que vale, nesta ordem, olhando só o que se sabia antes de D:
    EM ANDAMENTO      indicação aberta em D (indicada antes de D e sem desfecho até D)
    VERIFICAR         insucesso de EA nos últimos 60 dias
    NÃO               acordo feito nos últimos 30 dias (está pagando: não ofertar a entrega do carro)
    A PARTIR DE dd/mm ainda não tem 90 dias de atraso
    SIM
Apreendido, pago e formalizado já saíram da carteira do dia, por isso não aparecem aqui.
"""
import numpy as np
import pandas as pd

PASSOS = {"SIM": {"ALTO": "Ofertar EA no próximo contato", "MEDIO": "Ofertar se houver CPC",
                  "BAIXO": "Seguir régua padrão"},
          "A PARTIR DE": "Aguardar 90 dias de atraso", "EM ANDAMENTO": "Acompanhar negociação",
          "VERIFICAR": "Rever motivo do insucesso antes de ofertar", "NÃO": "Não ofertar: acordo em dia"}


def apto_ea(base: pd.DataFrame, tabelas: dict, D) -> pd.Series:
    D = pd.Timestamp(D)
    e, ac = tabelas["eventos_ea"], tabelas["acordos"]
    ids = base["id_contrato"]
    aberta = e[(e["data_indicacao"] < D) & (e["data_desfecho"].isna() | (e["data_desfecho"] >= D))]["id_contrato"]
    ins = e[(e["desfecho"] == "INSUCESSO") & (e["data_desfecho"] >= D - pd.Timedelta(days=60))
            & (e["data_desfecho"] < D)]["id_contrato"]
    acordo = ac[(ac["data_acordo"] >= D - pd.Timedelta(days=30)) & (ac["data_acordo"] < D)]["id_contrato"]
    falta = (90 - base["dias_atraso"]).clip(lower=0)
    a_partir = "A PARTIR DE " + (D + pd.to_timedelta(falta, unit="D")).dt.strftime("%d/%m")
    out = np.select([ids.isin(aberta), ids.isin(ins), ids.isin(acordo), base["dias_atraso"] < 90],
                    ["EM ANDAMENTO", "VERIFICAR", "NÃO", a_partir], default="SIM")
    return pd.Series(out, index=base.index)


def proximo_passo(apto: str, classe: str) -> str:
    if apto == "SIM":
        return PASSOS["SIM"][classe]
    return PASSOS["A PARTIR DE" if apto.startswith("A PARTIR DE") else apto]
