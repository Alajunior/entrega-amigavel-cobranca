import pytest

from dados.gerar_carteira import gerar


@pytest.fixture(scope="session")
def pasta_pequena(tmp_path_factory):
    """Carteira reduzida (3 mil contratos), gerada uma vez para toda a suíte."""
    pasta = tmp_path_factory.mktemp("carteira")
    gerar(3000, pasta)
    return pasta


@pytest.fixture(scope="session")
def carteira_pequena(pasta_pequena):
    import pandas as pd
    return {n: pd.read_parquet(pasta_pequena / f"{n}.parquet") for n in ("contratos", "contatos", "eventos_ea", "acordos")}
