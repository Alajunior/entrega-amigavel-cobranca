import pandas as pd
import pytest

from src.candidatos import bases_das_safras
from src.config import JANELA
from src.escoragem import COLUNAS, datas_retreino, escorar_dia, escorar_periodo, treinar_vigente


@pytest.fixture(scope="module")
def t(carteira_pequena):
    return {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def bases(t):
    return bases_das_safras(t)


def test_vigente_so_usa_safra_madura_em_D(bases):
    D = pd.Timestamp("2026-02-02")
    v = treinar_vigente(D, bases)
    assert v is not None and len(v.safras_calib) == 3 and 4 <= len(v.safras_treino) <= 8
    assert all(s + pd.Timedelta(days=JANELA) <= D for s in v.safras_treino + v.safras_calib)
    assert not set(v.safras_treino) & set(v.safras_calib) and max(v.safras_treino) < min(v.safras_calib)
    assert treinar_vigente(pd.Timestamp("2025-08-01"), bases) is None


def test_retreino_no_primeiro_dia_util():
    dias = pd.bdate_range("2025-12-01", "2026-02-27")
    assert datas_retreino(dias) == [pd.Timestamp(x) for x in ("2025-12-01", "2026-01-01", "2026-02-02")]


def test_escorar_dia_alvo_nan_quando_imaturo(t, bases):
    v = treinar_vigente(pd.Timestamp("2026-10-15"), bases)
    e = escorar_dia(pd.Timestamp("2026-10-15"), t, v)
    assert list(e.columns) == COLUNAS and e["ea_60d"].isna().all() and not e["maduro"].any()
    assert set(e["classe"]) <= {"ALTO", "MEDIO", "BAIXO"}


def test_cortes_acumulam_vigencias(t):
    r = escorar_periodo(t, "mensal", "2026-01-26", "2026-02-06")
    esc, cortes = r
    assert sorted(cortes["vigente_desde"].unique()) == [pd.Timestamp("2026-01-26"), pd.Timestamp("2026-02-02")]
    assert esc["dia"].nunique() == 10
