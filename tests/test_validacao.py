from src.validacao import teste_vazamento


def test_vazamento_grupo_menor_ou_igual(carteira_pequena):
    t = {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}
    r = teste_vazamento(t)
    for m in ("logistica", "lightgbm"):
        assert r[m]["grupo"]["auc"] <= r[m]["aleatorio"]["auc"] + 0.01
        assert set(r[m]) >= {"aleatorio", "grupo", "diferenca_auc", "diferenca_pr_auc"}
