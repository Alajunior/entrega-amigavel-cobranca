import pandas as pd

from src.config import ULTIMA_MADURA, faixa_de, fase_de


def test_fase_e_faixa_nos_limites():
    d = pd.Series([31, 90, 91, 180, 181, 360, 361, 720])
    assert fase_de(d).tolist() == ["0031-0090", "0031-0090", "0091-0180", "0091-0180",
                                   "0181-0360", "0181-0360", "0361-0720", "0361-0720"]
    assert faixa_de(pd.Series([60, 61, 541])).tolist() == ["031-060", "061-090", "541-720"]


def test_ultima_safra_madura():
    assert ULTIMA_MADURA == pd.Timestamp("2026-10-01")
