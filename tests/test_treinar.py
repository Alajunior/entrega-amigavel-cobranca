import json

from src.treinar import ARQUIVOS, main


def test_treinar_grava_relatorios(pasta_pequena, tmp_path):
    r = main(pasta_pequena, tmp_path)
    assert all((tmp_path / a).exists() for a in ARQUIVOS)
    esc = json.loads((tmp_path / "escada.json").read_text(encoding="utf-8"))
    assert esc["publicado"] in ("regua", "logistica", "lightgbm") and "curvas" in esc
    assert len(esc["curvas"]["pr"]["precisao"]) <= 100 and len(esc["curvas"]["captura"]["x"]) <= 101
    json.loads((tmp_path / "vazamento.json").read_text(encoding="utf-8"))
    assert r["publicado"] == esc["publicado"]
