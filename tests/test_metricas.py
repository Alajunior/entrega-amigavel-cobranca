import numpy as np

from src.metricas import metricas


def test_metricas_classe_rara():
    y = np.zeros(1000)
    y[:10] = 1
    m = metricas(y, -np.arange(1000.0))            # score perfeito: os 10 positivos no topo
    assert m["taxa_base"] == 0.01 and m["pr_auc"] == 1.0
    assert m["recall_top"][10] == 1.0 and m["lift_top10"] == 10.0
    r = metricas(y, np.random.default_rng(42).random(1000))
    assert abs(r["pr_auc"] - r["taxa_base"]) < 0.02
