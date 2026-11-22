import pandas as pd

from src.apto import apto_ea, proximo_passo

D = pd.Timestamp("2026-03-10")


def _tabelas():
    contratos = pd.DataFrame({"id_contrato": [1, 2, 3, 4, 5]})
    eventos = pd.DataFrame({
        "id_contrato": [1, 3, 5], "data_indicacao": pd.to_datetime(["2026-03-01", "2026-01-20", "2026-02-01"]),
        "desfecho": ["ABERTA", "INSUCESSO", "INSUCESSO"],
        "data_desfecho": pd.to_datetime([None, "2026-02-28", "2026-03-12"])})   # 5: aberto em D (desfecho depois)
    acordos = pd.DataFrame({"id_contrato": [4], "data_acordo": pd.to_datetime(["2026-03-01"]), "quebrado": [True]})
    return {"contratos": contratos, "eventos_ea": eventos, "acordos": acordos}


def test_apto_ordem_das_regras():
    base = pd.DataFrame({"id_contrato": [1, 2, 3, 4, 5, 6], "dias_atraso": [50, 70, 200, 200, 200, 200]})
    a = apto_ea(base, _tabelas(), D)
    assert a.tolist() == ["EM ANDAMENTO", "A PARTIR DE 30/03", "VERIFICAR", "NÃO", "EM ANDAMENTO", "SIM"]


def test_proximo_passo():
    assert proximo_passo("SIM", "ALTO") == "Ofertar EA no próximo contato"
    assert proximo_passo("A PARTIR DE 30/03", "ALTO") == "Aguardar 90 dias de atraso"
    assert proximo_passo("NÃO", "BAIXO") == "Não ofertar: acordo em dia"
