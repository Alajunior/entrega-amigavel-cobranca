from src.base_analitica import ALVO, CAT, NUM, montar_base
from src.modelos import criar_modelos, score_regua


def test_modelos_treinam_e_nomeiam_colunas(carteira_pequena):
    t = {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}
    b = __import__("pandas").concat([montar_base(d, t) for d in ("2025-06-02", "2025-09-01", "2026-01-05")])
    for nome, m in criar_modelos().items():
        m.fit(b[NUM + CAT], b[ALVO])
        p = m.predict_proba(b[NUM + CAT])[:, 1]
        assert len(p) == len(b) and ((p >= 0) & (p <= 1)).all(), nome
    nomes = list(criar_modelos()["logistica"].fit(b[NUM + CAT], b[ALVO])[:-1].get_feature_names_out())
    assert "num__ltv_atual" in nomes and "num__missingindicator_dias_desde_ult_cpc" in nomes
    assert len(score_regua(b)) == len(b)
