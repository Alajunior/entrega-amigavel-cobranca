"""Explicação do score para a operação (porte do Propensão): por que este contrato está no ALTO.

Na logística a explicação é exata, sem SHAP: log-odds = intercepto + Σ coef × z. Medido contra o contrato "médio"
do treino, cada variável contribui coef × (z − média de z). Somando as colunas transformadas de uma mesma variável
original (o valor, as categorias) sai a contribuição da variável; a flag de nulo é variável própria (x_nulo).
"""
import pandas as pd

from src.base_analitica import CAT

NOMES = {"idade": "idade", "dias_atraso": "dias de atraso", "valor_vencido": "valor vencido",
         "divida_total": "dívida total", "pct_pago": "parte do plano já paga", "plano": "prazo do plano",
         "valor_parcela": "valor da parcela", "ltv_atual": "dívida ÷ valor do carro", "idade_veiculo": "idade do carro",
         "qtd_tentativas_30d": "tentativas em 30 dias", "qtd_cpc_30d": "contato certo em 30 dias",
         "dias_desde_ult_cpc": "dias sem contato certo", "acordos_quebrados_180d": "acordo quebrado",
         "insucessos_ea_antes": "EA anterior sem sucesso", "fase": "fase de cobrança", "faixa_bureau": "bureau",
         "uf_grupo": "região", "canal_pref": "canal de contato", "dias_desde_ult_cpc_nulo": "nunca teve contato certo"}
# (abaixo da média, acima da média): o motivo diz o lado em que o contrato está, não só a variável
FRASES = {"idade": ("mais jovem", "mais velho"), "dias_atraso": ("atraso menor", "atraso maior"),
          "valor_vencido": ("vencido menor", "vencido maior"), "divida_total": ("dívida menor", "dívida maior"),
          "pct_pago": ("pagou pouco do plano", "pagou boa parte do plano"),
          "plano": ("plano mais curto", "plano mais longo"), "valor_parcela": ("parcela menor", "parcela maior"),
          "ltv_atual": ("deve menos do que o carro vale", "deve mais do que o carro vale"),
          "idade_veiculo": ("carro mais novo", "carro mais velho"),
          "qtd_tentativas_30d": ("poucas tentativas recentes", "muitas tentativas recentes"),
          "qtd_cpc_30d": ("sem contato certo recente", "contato certo recente"),
          "dias_desde_ult_cpc": ("contato certo recente", "contato certo antigo"),
          "acordos_quebrados_180d": ("sem acordo quebrado", "quebrou acordo recente"),
          "insucessos_ea_antes": ("sem EA anterior", "já tentou EA antes"),
          "dias_desde_ult_cpc_nulo": ("já teve contato certo", "nunca teve contato certo")}


def _original(col: str) -> str:
    nome = col.split("__", 1)[-1]
    if nome.startswith("missingindicator_"):
        return nome.replace("missingindicator_", "") + "_nulo"
    if col.startswith("cat__"):
        for cat in CAT:
            if nome.startswith(cat + "_"):
                return cat
    return nome


def _z(modelo, X_df: pd.DataFrame) -> pd.DataFrame:
    return modelo[:-1].transform(X_df)


def referencia(modelo, X_treino: pd.DataFrame) -> pd.Series:
    """O contrato "médio" do treino, no espaço transformado que a logística enxerga."""
    return _z(modelo, X_treino).mean()


def contribuicoes(modelo, X_df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    z = _z(modelo, X_df)
    coef = pd.Series(modelo[-1].coef_[0], index=z.columns)
    c = (z - ref[z.columns]) * coef
    return c.T.groupby(c.columns.map(_original)).sum().T


def direcoes(modelo, X_df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    """+1 se o valor do contrato está acima do contrato médio do treino, −1 se abaixo; só numéricas e flags."""
    z = _z(modelo, X_df)
    cols = [c for c in z.columns if c.startswith("num__")]
    d = (z[cols] - ref[cols]).ge(0).astype(int) * 2 - 1
    d.columns = [_original(c) for c in cols]
    return d


def motivos(contrib: pd.DataFrame, direc: pd.DataFrame | None = None, k: int = 2) -> list[list[str]]:
    """As k variáveis que MAIS empurraram a chance para cima, com a frase do lado em que o contrato está."""
    out = []
    for i, (_, linha) in enumerate(contrib.iterrows()):
        frases = []
        for v in linha[linha > 0].sort_values(ascending=False).index:
            if direc is not None and v in FRASES and v in direc.columns:
                frase = FRASES[v][0 if direc.iloc[i][v] < 0 else 1]
            else:
                frase = NOMES.get(v, v)
            if frase not in frases:
                frases.append(frase)
            if len(frases) == k:
                break
        out.append(frases)
    return out


def importancia(contrib: pd.DataFrame) -> pd.DataFrame:
    m = contrib.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"variavel": m.index, "nome": [NOMES.get(v, v) for v in m.index], "media_abs": m.to_numpy()})
