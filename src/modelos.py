"""Candidatos: régua de negócio, logística e LightGBM, com o pré-processamento dentro do Pipeline (sem vazamento)."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.base_analitica import CAT, NUM


def _preparo(escala: bool) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median", add_indicator=True)]   # nulo vira flag própria (ex.: nunca teve CPC)
    if escala:
        num.append(StandardScaler())
    ct = ColumnTransformer([("num", make_pipeline(*num), NUM),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return ct.set_output(transform="pandas")


def criar_modelos() -> dict[str, Pipeline]:
    # class_weight="balanced" no LightGBM = scale_pos_weight negativos/positivos do treino, sem precisar do dado aqui
    return {
        "logistica": make_pipeline(_preparo(True), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "lightgbm": make_pipeline(_preparo(False), LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, class_weight="balanced",
            random_state=42, verbose=-1)),
    }


def score_regua(base: pd.DataFrame) -> np.ndarray:
    """Sem modelo: deve mais do que o carro vale primeiro; dentro disso, quem está há mais tempo em atraso."""
    return (base["ltv_atual"].clip(0, 3) * 1000 + base["dias_atraso"]).to_numpy(dtype=float)
