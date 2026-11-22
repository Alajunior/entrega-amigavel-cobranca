import pickle
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from src.base_analitica import montar_base
from src.candidatos import bases_das_safras
from src.escoragem import treinar_vigente
from src.producao import (DIA_PASSADO, DISTRIBUICAO_FORA, FOTO_AUSENTE, HASH_DIVERGENTE, OK, VOLUME_BAIXO,
                          carregar_modelo, escorar_dia_guardado, salvar_modelo)


@pytest.fixture(scope="module")
def base_modelo(carteira_pequena):
    t = {k: v for k, v in carteira_pequena.items() if not k.startswith("_")}
    return t, treinar_vigente("2026-11-16", bases_das_safras(t))


@pytest.fixture
def ctx(base_modelo, tmp_path):
    t, vig = base_modelo
    mod = tmp_path / "modelo"
    salvar_modelo(vig, mod)
    return SimpleNamespace(t=t, vig=vig, mod=mod, out=tmp_path / "saida",
                           vazia=montar_base("2026-11-17", t).iloc[0:0])


def test_manifest_confere_o_hash(ctx):
    vig, cod = carregar_modelo(ctx.mod)
    assert cod == OK and vig.treinado_em == pd.Timestamp("2026-11-16")


def test_dia_normal_grava_e_e_idempotente(ctx):
    r1 = escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    r2 = escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    assert r1["codigo"] == r2["codigo"] == OK and len(list(ctx.out.glob("escoragem_*.csv"))) == 1
    assert r1["distribuicao_ok"] and r1["sha_ok"] and r1["linhas"] > 0
    assert not list(ctx.out.glob("*.tmp"))
    pub = pd.read_csv(ctx.out / "escoragem_20261116.csv")
    assert {"apto_ea", "proximo_passo", "classe", "score"} <= set(pub.columns)


def test_foto_ausente_e_volume_baixo(ctx):
    escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    r = escorar_dia_guardado("2026-11-17", ctx.t, ctx.mod, ctx.out, foto=ctx.vazia)
    assert r["codigo"] == FOTO_AUSENTE and not (ctx.out / "escoragem_20261117.csv").exists()
    meia = montar_base("2026-11-18", ctx.t).sample(frac=0.5, random_state=1)
    r = escorar_dia_guardado("2026-11-18", ctx.t, ctx.mod, ctx.out, foto=meia)
    assert r["codigo"] == VOLUME_BAIXO and r["volume_vs_anterior"] < 0.85


def test_hash_divergente_e_dia_passado(ctx):
    escorar_dia_guardado("2026-11-18", ctx.t, ctx.mod, ctx.out)
    assert escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)["codigo"] == DIA_PASSADO
    (ctx.mod / "modelo.pkl").write_bytes(b"adulterado")
    r = escorar_dia_guardado("2026-11-19", ctx.t, ctx.mod, ctx.out)
    assert r["codigo"] == HASH_DIVERGENTE and r["sha_ok"] is False


def test_sem_manifest_e_codigo_5(ctx):
    (ctx.mod / "manifest.json").unlink()
    r = escorar_dia_guardado("2026-11-16", ctx.t, ctx.mod, ctx.out)
    assert r["codigo"] == HASH_DIVERGENTE and r["sha_ok"] is False


def test_distribuicao_fora_e_codigo_6(ctx, tmp_path):
    adulterado = tmp_path / "adulterado"
    salvar_modelo(replace(ctx.vig, pct_alto=min(1.0, ctx.vig.pct_alto * 10)), adulterado)
    r = escorar_dia_guardado("2026-11-16", ctx.t, adulterado, ctx.out)
    assert r["codigo"] == DISTRIBUICAO_FORA and r["distribuicao_ok"] is False
    assert not (ctx.out / "escoragem_20261116.csv").exists()


def test_calibracao_sem_ea_e_codigo_6(ctx, tmp_path):
    cortes = ctx.vig.cortes.assign(corte_alto=float("nan"), corte_medio=float("nan"))
    sem_ea = tmp_path / "sem_ea"
    salvar_modelo(replace(ctx.vig, cortes=cortes, pct_alto=0.0), sem_ea)
    r = escorar_dia_guardado("2026-11-16", ctx.t, sem_ea, ctx.out)
    assert r["codigo"] == DISTRIBUICAO_FORA and not (ctx.out / "escoragem_20261116.csv").exists()
