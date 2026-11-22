"""Treino e avaliação fora do tempo: grava em relatorios/ o que o README e o painel citam.

    python -m src.treinar                    # base em dados/carteira_exemplo/
    escada.json       escada régua → logística → LightGBM, gates, publicado, sombra e curvas do OOT
    vazamento.json    validação cruzada aleatória × GroupKFold por contrato
    importancia.csv   média |contribuição| de cada variável na logística, nas safras OOT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

from src.base_analitica import ALVO, CAT, NUM, PASTA, carregar
from src.candidatos import avaliar_escada, bases_das_safras, pilha_oot, separar_oot
from src.modelos import criar_modelos
from src.sinal import contribuicoes, importancia, referencia
from src.validacao import teste_vazamento

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVOS = ("escada.json", "vazamento.json", "importancia.csv")


def _curva_pr(y, s, pontos: int = 100) -> dict:
    p, r, _ = precision_recall_curve(y, s)
    i = np.unique(np.linspace(0, len(p) - 1, min(pontos, len(p))).astype(int))
    return {"precisao": [round(float(x), 5) for x in p[i]], "recall": [round(float(x), 5) for x in r[i]]}


def _curva_captura(y, s, pontos: int = 100) -> dict:
    """% do estoque acionado (x) × % das EAs alcançadas (y), da maior para a menor chance."""
    y = np.asarray(y, float)[np.argsort(-np.asarray(s, float), kind="stable")]
    acum = np.r_[0, np.cumsum(y)] / y.sum()
    x = np.linspace(0, 1, pontos + 1)
    return {"x": [round(float(v), 4) for v in x],
            "y": [round(float(acum[int(round(v * len(y)))]), 5) for v in x]}


def main(pasta: Path = PASTA, relatorios: Path = RAIZ / "relatorios") -> dict:
    relatorios = Path(relatorios)
    relatorios.mkdir(parents=True, exist_ok=True)
    t = carregar(pasta)
    bases = bases_das_safras(t)
    oot = pilha_oot(t, bases=bases)
    esc = avaliar_escada(t, df=oot)
    pub = esc["publicado"]
    esc["curvas"] = {"pr": _curva_pr(oot[ALVO], oot[pub]), "captura": _curva_captura(oot[ALVO], oot[pub]),
                     "captura_regua": _curva_captura(oot[ALVO], oot["regua"]), "taxa_base": float(oot[ALVO].mean())}
    (relatorios / "escada.json").write_text(json.dumps(esc, indent=2, ensure_ascii=False, allow_nan=False),
                                            encoding="utf-8")
    vaz = teste_vazamento(t, bases=bases)
    (relatorios / "vazamento.json").write_text(json.dumps(vaz, indent=2, allow_nan=False), encoding="utf-8")
    treino, safras_oot = separar_oot(sorted(bases))
    tr = pd.concat(bases[s] for s in treino)
    te = pd.concat(bases[s] for s in safras_oot)
    m = clone(criar_modelos()["logistica"]).fit(tr[NUM + CAT], tr[ALVO])
    importancia(contribuicoes(m, te[NUM + CAT], referencia(m, tr[NUM + CAT]))).to_csv(
        relatorios / "importancia.csv", index=False)
    d = esc["degraus"]
    print(f"OOT ({len(oot)} linhas, taxa {oot[ALVO].mean():.2%}):")
    for c in ("regua", "logistica", "lightgbm"):
        print(f"  {c:10s} PR-AUC {d[c]['pr_auc']:.3f}  AUC {d[c]['auc']:.3f}  captura top10 {d[c]['captura_top10']:.1%}")
    print(" ", esc["frase"])
    print(f"  vazamento: AUC aleatório − grupo = logística {vaz['logistica']['diferenca_auc']:+.4f}, "
          f"LightGBM {vaz['lightgbm']['diferenca_auc']:+.4f}")
    return esc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")         # console do Windows (cp1252) não tem o sinal de menos
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA)
