import pandas as pd

from dados.gerar_carteira import gerar
from src.config import HOJE, INICIO_DADOS


def test_mesma_seed_mesma_carteira(tmp_path):
    a = gerar(300, tmp_path / "a")
    b = gerar(300, tmp_path / "b")
    pd.testing.assert_frame_equal(a["contratos"], b["contratos"])
    pd.testing.assert_frame_equal(a["contatos"], b["contatos"])


def _janela(c, df, col):
    m = df.merge(c[["id_contrato", "data_entrada", "data_saida"]], on="id_contrato")
    fim = m["data_saida"].fillna(HOJE + pd.Timedelta(days=1))
    return m, (m[col] >= m["data_entrada"]) & (m[col] < fim) & (m[col] >= INICIO_DADOS) & (m[col] <= HOJE)


def test_invariantes_de_tempo(carteira_pequena):
    c = carteira_pequena["contratos"]
    assert (c["data_entrada"] >= INICIO_DADOS - pd.Timedelta(days=400)).all()
    assert (c["data_entrada"] <= HOJE).all()
    s = c.dropna(subset=["data_saida"])
    assert (s["data_saida"] > s["data_entrada"]).all()
    assert ((s["data_saida"] - s["data_entrada"]).dt.days <= 690).all()
    assert (s["data_saida"] <= HOJE).all()
    assert set(c["motivo_saida"]) <= {"PAGO", "APREENDIDO", "EA", "LIMITE_720", "ATIVO"}
    assert (c["data_saida"].isna() == (c["motivo_saida"] == "ATIVO")).all()
    for nome, col in [("contatos", "data"), ("eventos_ea", "data_indicacao"), ("acordos", "data_acordo")]:
        _, ok = _janela(c, carteira_pequena[nome], col)
        assert ok.all(), nome


def test_ea_formalizada_bate_com_saida(carteira_pequena):
    c, e = carteira_pequena["contratos"], carteira_pequena["eventos_ea"]
    f = e[e["desfecho"] == "FORMALIZADA"]
    assert f["id_contrato"].is_unique
    ea = c[c["motivo_saida"] == "EA"].merge(f, on="id_contrato", how="left")
    assert (ea["data_desfecho"] == ea["data_saida"]).all()
    assert set(f["id_contrato"]) == set(c.loc[c["motivo_saida"] == "EA", "id_contrato"])


def test_indicacao_nasce_de_cpc(carteira_pequena):
    e, ct = carteira_pequena["eventos_ea"], carteira_pequena["contatos"]
    cpc = ct[ct["cpc"] & (ct["desfecho"] == "EA INDICADA")][["id_contrato", "data"]]
    m = e.merge(cpc, left_on=["id_contrato", "data_indicacao"], right_on=["id_contrato", "data"], how="left")
    assert m["data"].notna().all()


def test_reindicacao_so_depois_de_insucesso(carteira_pequena):
    e = carteira_pequena["eventos_ea"].sort_values(["id_contrato", "data_indicacao"])
    ant = e.groupby("id_contrato").shift(1)
    seg = ant["desfecho"].notna()
    assert (ant.loc[seg, "desfecho"] == "INSUCESSO").all()
    assert (ant.loc[seg, "data_desfecho"] < e.loc[seg, "data_indicacao"]).all()
    assert seg.any(), "a carteira pequena deveria ter ao menos uma reindicação"


def test_sem_coluna_sexo(carteira_pequena):
    for df in carteira_pequena.values():
        assert not any("sexo" in col.lower() for col in df.columns)
