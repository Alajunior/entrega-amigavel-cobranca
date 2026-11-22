import numpy as np
import pytest

from src.audio.roteiros import PERSONAS
from src.audio.sintetizar import SR_TEL, exportar_mp3, misturar, montar_estereo, telefone


def _tom(freq, seg, sr):
    t = np.arange(int(seg * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _potencia_em(x, sr, freq):
    espec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return espec[np.abs(f - freq) < 5].sum()


def test_telefone_8khz_e_faixa():
    sr = 24000
    graves, medios = _tom(100, 2, sr), _tom(1000, 2, sr)
    rng = np.random.default_rng(0)
    y_g = telefone(graves, sr, snr_db=60, rng=rng)
    y_m = telefone(medios, sr, snr_db=60, rng=rng)
    assert SR_TEL == 8000 and y_m.dtype == np.float32 and abs(len(y_m) - 2 * 8000) <= 1
    ref_g = _potencia_em(graves[::3], 8000, 100)
    ref_m = _potencia_em(medios[::3], 8000, 1000)
    assert 10 * np.log10(_potencia_em(y_g, 8000, 100) / ref_g) <= -20     # 100 Hz cai pelo menos 20 dB
    assert 10 * np.log10(_potencia_em(y_m, 8000, 1000) / ref_m) > -3       # 1 kHz passa


def test_snr_aproximada():
    sinal = _tom(700, 3, 8000)
    rng = np.random.default_rng(1)
    y = misturar(sinal, 15, rng)
    ruido = y - sinal
    snr = 10 * np.log10(np.mean(sinal ** 2) / np.mean(ruido ** 2))
    assert 13 <= snr <= 17


def test_snr_mede_so_a_fala():
    """Silêncio entre falas não pode baixar o ruído: a SNR vale para o trecho com voz."""
    sinal = np.concatenate([_tom(700, 1, 8000), np.zeros(8000 * 3, np.float32)])
    y = misturar(sinal, 15, np.random.default_rng(2))
    ruido = y - sinal
    snr = 10 * np.log10(np.mean(sinal[:8000] ** 2) / np.mean(ruido ** 2))
    assert 13 <= snr <= 17


def test_estereo_separa_lados():
    sr = 8000
    falas = [("operador", _tom(500, 1.0, sr)), ("cliente", _tom(900, 0.7, sr)),
             ("operador", _tom(500, 1.2, sr)), ("cliente", _tom(900, 0.5, sr))]
    est, tempos = montar_estereo(falas, sr, np.random.default_rng(3))
    assert est.shape[1] == 2 and est.dtype == np.float32
    L, R = est[:, 0], est[:, 1]
    assert _potencia_em(L, sr, 500) > 1000 * _potencia_em(L, sr, 900) + 1e-9
    assert _potencia_em(R, sr, 900) > 1000 * _potencia_em(R, sr, 500) + 1e-9
    assert [t["lado"] for t in tempos] == ["operador", "cliente", "operador", "cliente"]
    inicios = [t["inicio"] for t in tempos]
    assert inicios == sorted(inicios)
    for a, b in zip(tempos, tempos[1:]):
        assert b["inicio"] - a["fim"] <= 0.8 + 1e-6 and b["inicio"] - a["fim"] >= -0.2 - 1e-6
    assert abs((tempos[0]["fim"] - tempos[0]["inicio"]) - 1.0) < 0.01


def test_personas_tem_voz_rate_pitch():
    for p in PERSONAS.values():
        assert p["voz"].startswith("pt-BR-") and p["rate"].endswith("%") and p["pitch"].endswith("Hz")


def test_exportar_mp3_estereo_32k(tmp_path):
    av = pytest.importorskip("av")
    est = np.stack([_tom(500, 4, 8000), _tom(900, 4, 8000)], axis=1)
    arq = tmp_path / "x.mp3"
    exportar_mp3(est, 8000, arq)
    with av.open(str(arq)) as c:
        s = c.streams.audio[0]
        assert s.channels == 2
        dur = float(c.duration / 1e6)
    assert abs(dur - 4) < 0.3
    kbps = arq.stat().st_size * 8 / dur / 1000
    assert 26 <= kbps <= 38


def test_cache_da_fala_muda_com_texto_ou_persona():
    """Regerar o roteiro não pode reaproveitar o áudio de uma fala que mudou."""
    from src.audio.sintetizar import nome_fala
    a = nome_fala(3, "Bom dia.", "op_antonio")
    assert a == nome_fala(3, "Bom dia.", "op_antonio") and a.startswith("03_") and a.endswith(".mp3")
    assert a != nome_fala(3, "Boa tarde.", "op_antonio") and a != nome_fala(3, "Bom dia.", "op_thalita")


def test_perda_de_pacotes_zera_quadros_de_20ms():
    from src.audio.sintetizar import perda_pacotes
    x = np.ones(8000 * 20, np.float32)
    y = perda_pacotes(x, 8000, 0.05, np.random.default_rng(4))
    quadros = y[: len(y) // 160 * 160].reshape(-1, 160)
    zerados = (np.abs(quadros).max(1) == 0).mean()
    assert 0.03 <= zerados <= 0.07 and set(np.unique(quadros.max(1))) <= {0.0, 1.0}
    assert perda_pacotes(x, 8000, 0.0, np.random.default_rng(4)) is not None


def test_conversa_de_fundo_na_relacao_pedida():
    """A conversa de fundo (outras posições da central) entra na relação sinal/fundo pedida, medida na fala."""
    from src.audio.sintetizar import conversa_de_fundo
    sr = 8000
    fala = _tom(600, 4, sr)
    fundo = [_tom(f, 1.3, sr) for f in (400, 900, 1500)]
    y = conversa_de_fundo(fala, sr, fundo, sbr_db=10, rng=np.random.default_rng(5))
    b = y - fala
    sbr = 10 * np.log10(np.mean(fala ** 2) / np.mean(b ** 2))
    assert len(y) == len(fala) and 8 <= sbr <= 12


def test_fundo_so_usa_falas_neutras():
    """A conversa de fundo nunca pode carregar uma oferta ou um alerta para dentro de outra ligação."""
    from src.audio.sintetizar import fala_neutra
    assert fala_neutra("Qual é a data de vencimento?")
    for t in ("Temos a entrega amigável.", "Dá para devolver o carro.", "Vão bloquear a sua conta.",
              "Consigo 40% de desconto.", "Isso zera a dívida.", "Fica um saldo remanescente.", "Seu nome fica limpo."):
        assert not fala_neutra(t), t
