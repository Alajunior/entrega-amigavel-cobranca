import numpy as np
import pandas as pd

from src.candidatos import avaliar_escada, captura_top, gate, ic_bootstrap_diff, separar_oot


def test_captura_top_por_safra():
    df = pd.DataFrame({"safra": ["a"] * 10 + ["b"] * 20, "y": [1] + [0] * 9 + [0, 1] + [0] * 18,
                       "s": list(range(10, 0, -1)) + list(range(20, 0, -1))})
    # top 10%: 1 linha da safra a (o positivo) e 2 da safra b (inclui o positivo na 2ª posição)
    assert captura_top(df["y"], df["s"], df["safra"], 0.10) == 1.0


def test_bootstrap_reamostra_contratos():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"id_contrato": np.repeat(np.arange(200), 3), "safra": np.tile([1, 2, 3], 200),
                       "ea_60d": (rng.random(600) < 0.05).astype(float), "a": rng.random(600)})
    df["b"] = df["a"]
    assert ic_bootstrap_diff(df, "a", "b", n=200) == (0.0, 0.0, 0.0)


def _df_escada(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    sinal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-5 + 1.8 * sinal)))).astype(float)
    return pd.DataFrame({"id_contrato": np.arange(n), "safra": rng.integers(0, 4, n), "fase": "x", "ea_60d": y,
                         "regua": sinal + rng.normal(0, 3, n), "logistica": sinal + rng.normal(0, 1.0, n),
                         "lightgbm": sinal + rng.normal(0, 0.3, n)})


def test_gate_exige_ic_e_pr_auc():
    df = _df_escada()
    assert gate(df, "regua", "logistica")["passou"]
    df["pior"] = df["regua"]
    g = gate(df, "logistica", "pior")
    assert not g["passou"] and g["ic"][0] <= 0


def test_escada_publica_mais_simples_e_sombra():
    e = avaliar_escada(None, df=_df_escada())
    assert e["publicado"] == "logistica" and e["sombra"] == "lightgbm"
    assert [g["desafiante"] for g in e["gates"]] == ["logistica", "lightgbm"]


def test_pilha_oot_sem_sobreposicao():
    s = list(pd.date_range("2025-04-01", "2026-10-01", freq="28D"))
    treino, oot = separar_oot(s, 6)
    assert oot == s[-6:] and not set(treino) & set(oot)
    assert all(t + pd.Timedelta(days=60) <= oot[0] for t in treino)
