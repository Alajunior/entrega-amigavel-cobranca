"""Ligações sintéticas: vozes neurais (edge-tts) → telefone → MP3 estéreo (operador à esquerda, cliente à direita).

Cada turno do roteiro é sintetizado com a voz da persona, as falas são encaixadas com pausas curtas (e às vezes
uma sobreposição, como em ligação de verdade) e cada canal passa pela linha telefônica: 8 kHz, faixa 300–3400 Hz,
lei µ (G.711), ruído rosa e 3% de pacotes perdidos. O canal do operador leva ainda a conversa das posições vizinhas
da central (falas neutras de OUTRAS ligações). O MP3 sai a 32 kbps.

O nível de ruído foi calibrado com o próprio Whisper: com SNR de 10–25 dB e sem central, ele errava só ~5% das
palavras e acertava toda oferta; a "central ruidosa" abaixo leva o erro do operador a ~30%, como em ligação real.

    py -m src.audio.sintetizar [--so ID] [--proxy http://127.0.0.1:3128]

Idempotente: pula o MP3 que já existe e guarda cada fala sintetizada em dados/ligacoes/_trabalho/ (fora do git),
para que uma interrupção não custe a rede de novo. A rede externa só é usada aqui.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.audio.roteiros import PERSONAS, RAIZ

SR_TEL = 8000
SR_MP3 = 16000                 # MPEG-2 a 16 kHz: toca em qualquer navegador (8 kHz é MPEG-2.5, nem todos tocam)
SNR_OPERADOR, SNR_CLIENTE = (10.0, 18.0), (8.0, 16.0)
SBR_CENTRAL = (6.0, 12.0)      # fala do operador ÷ conversa de fundo da central, em dB
PERDA = 0.03
N_FUNDO = 240                  # falas neutras sorteadas para a conversa de fundo
PAUSA = (0.25, 0.8)
P_SOBREPOE, SOBREPOE_MAX = 0.15, 0.2
PASTA_LIG = RAIZ / "dados" / "ligacoes"
PASTA_AUDIO = RAIZ / "docs" / "audio"
TRABALHO = PASTA_LIG / "_trabalho"
_SOS = butter(4, [300, 3400], btype="bandpass", fs=SR_TEL, output="sos")


def _ruido_rosa(n: int, rng: np.random.Generator) -> np.ndarray:
    espec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(espec))
    espec[1:] /= np.sqrt(f[1:])
    espec[0] = 0
    r = np.fft.irfft(espec, n)
    return r / (np.std(r) + 1e-12)


def misturar(sinal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Soma ruído rosa na SNR pedida, medida só nos trechos com voz (o silêncio entre falas não conta)."""
    ativo = np.abs(sinal) > 0.02 * (np.abs(sinal).max() + 1e-12)
    pot = np.mean(sinal[ativo] ** 2) if ativo.any() else 1e-6
    ruido = _ruido_rosa(len(sinal), rng) * np.sqrt(pot / 10 ** (snr_db / 10))
    return (sinal + ruido).astype(np.float32)


def conversa_de_fundo(fala: np.ndarray, sr: int, fundo: list[np.ndarray], sbr_db: float,
                      rng: np.random.Generator, vozes: int = 3) -> np.ndarray:
    """Soma `vozes` falas de outras posições da central (trechos de `fundo` em sequência, com pausas) na relação
    sinal/fundo pedida, medida nos trechos com voz da fala principal."""
    n = len(fala)
    b = np.zeros(n, np.float32)
    for _ in range(vozes):
        partes, tam = [np.zeros(int(sr * rng.uniform(0, 1.5)), np.float32)], 0
        while tam < n:
            x = fundo[int(rng.integers(len(fundo)))]
            partes += [x, np.zeros(int(sr * rng.uniform(0.2, 1.0)), np.float32)]
            tam = sum(map(len, partes))
        b += np.concatenate(partes)[:n] * rng.uniform(0.5, 1.0)
    ativo = np.abs(fala) > 0.02 * (np.abs(fala).max() + 1e-12)
    pot = np.mean(fala[ativo] ** 2) if ativo.any() else 1e-6
    b *= np.sqrt(pot / 10 ** (sbr_db / 10) / (np.mean(b ** 2) + 1e-12))
    return (fala + b).astype(np.float32)


def perda_pacotes(x: np.ndarray, sr: int, p: float, rng: np.random.Generator, quadro: float = 0.02) -> np.ndarray:
    """Perda de pacotes da voz sobre IP: quadros de 20 ms zerados com probabilidade `p`."""
    q = int(quadro * sr)
    y = x.copy()
    n = len(y) // q
    perdidos = np.flatnonzero(rng.random(n) < p)
    for i in perdidos:
        y[i * q:(i + 1) * q] = 0
    return y


def _mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Quantização de 8 bits com lei µ (G.711), a compressão da telefonia."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
    return np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu


def telefone(x: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mono float32 a 8 kHz: reamostra, passa-faixa 300–3400 Hz, ruído rosa na SNR pedida e lei µ."""
    fr = Fraction(SR_TEL, sr)
    y = resample_poly(np.asarray(x, np.float64), fr.numerator, fr.denominator)
    y = sosfiltfilt(_SOS, y)
    y = misturar(y, snr_db, rng)
    pico = np.abs(y).max()
    y = y / pico * 0.9 if pico > 0.9 else y
    return _mu_law(y).astype(np.float32)


def _aparar(x: np.ndarray, sr: int) -> np.ndarray:
    """Tira o silêncio do começo e do fim de uma fala sintetizada (deixa 50 ms de margem)."""
    ativo = np.flatnonzero(np.abs(x) > 0.02 * (np.abs(x).max() + 1e-12))
    if not len(ativo):
        return x
    m = int(0.05 * sr)
    return x[max(0, ativo[0] - m):ativo[-1] + m]


def montar_estereo(audios: list[tuple[str, np.ndarray]], sr: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Encaixa as falas em dois canais (operador em L, cliente em R) com pausa de 0,25–0,8 s e, em ~15% das trocas,
    sobreposição de até 0,2 s. Devolve o áudio (n, 2) e o início/fim real de cada turno, em segundos."""
    inicio, tempos, pos = int(0.3 * sr), [], []
    t = inicio
    for k, (lado, x) in enumerate(audios):
        if k:
            gap = -rng.uniform(0, SOBREPOE_MAX) if rng.random() < P_SOBREPOE else rng.uniform(*PAUSA)
            t = max(pos[-1][1], pos[-1][2] + int(gap * sr))
        pos.append((lado, t, t + len(x)))
        tempos.append({"lado": lado, "inicio": round(t / sr, 3), "fim": round((t + len(x)) / sr, 3)})
    est = np.zeros((pos[-1][2] + int(0.3 * sr) if pos else 0, 2), np.float32)
    for (lado, a, b), (_, x) in zip(pos, audios):
        est[a:b, 0 if lado == "operador" else 1] += x
    return est, tempos


def exportar_mp3(estereo: np.ndarray, sr: int, caminho: Path) -> None:
    """MP3 estéreo 32 kbps (PyAV/libmp3lame), reamostrado a 16 kHz."""
    import av
    fr = Fraction(SR_MP3, sr)
    x = resample_poly(estereo, fr.numerator, fr.denominator, axis=0) if sr != SR_MP3 else estereo
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".tmp.mp3")
    with av.open(str(tmp), "w") as c:
        s = c.add_stream("libmp3lame", rate=SR_MP3, layout="stereo")
        s.bit_rate = 32000
        for i in range(0, len(pcm), 1152):
            bloco = np.ascontiguousarray(pcm[i:i + 1152].reshape(1, -1))      # s16 intercalado: L R L R …
            quadro = av.AudioFrame.from_ndarray(bloco, format="s16", layout="stereo")
            quadro.sample_rate = SR_MP3
            for p in s.encode(quadro):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    os.replace(tmp, caminho)


_NAO_NEUTRA = re.compile(r"entreg|devol|amig|remanesc|bloque|penhor|desconto|zer[ao]|encerr|acab|limp|quit", re.I)


def fala_neutra(texto: str) -> bool:
    """Fala que pode virar conversa de fundo: sem nada que a regra de oferta ou de alerta leria."""
    return not _NAO_NEUTRA.search(texto)


def nome_fala(k: int, texto: str, persona: str) -> str:
    """Arquivo de cache de uma fala: muda se o texto ou a persona mudarem (regerar o roteiro não reaproveita errado)."""
    return f"{k:02d}_{hashlib.sha1(f'{persona}|{texto}'.encode()).hexdigest()[:10]}.mp3"


def _decodificar(dados: bytes) -> tuple[np.ndarray, int]:
    import av
    with av.open(io.BytesIO(dados)) as c:
        s = c.streams.audio[0]
        sr = s.rate
        partes = [f.to_ndarray().astype(np.float32).reshape(f.layout.nb_channels, -1).mean(0) for f in c.decode(s)]
    return np.concatenate(partes), sr


async def _falar(texto: str, persona: str, destino: Path, proxy: str | None, sem: asyncio.Semaphore) -> bytes:
    if destino.exists() and destino.stat().st_size > 0:
        return destino.read_bytes()
    import edge_tts
    p = PERSONAS[persona]
    async with sem:
        for tentativa in range(4):
            try:
                com = edge_tts.Communicate(texto, p["voz"], rate=p["rate"], pitch=p["pitch"], proxy=proxy)
                dados = b"".join([c["data"] async for c in com.stream() if c["type"] == "audio"])
                break
            except Exception:
                if tentativa == 3:
                    raise
                await asyncio.sleep(2 * (tentativa + 1))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(dados)
    return dados


def _personas(r: dict) -> dict:
    return {"operador": r["persona_operador"], "cliente": r["persona_cliente"]}


async def _garantir_falas(r: dict, proxy: str | None, sem: asyncio.Semaphore) -> None:
    """Sintetiza (ou acha no cache) cada fala do roteiro."""
    pers = _personas(r)
    await asyncio.gather(*[_falar(t["texto"], pers[t["lado"]], TRABALHO / r["id_ligacao"] / nome_fala(k, t["texto"], pers[t["lado"]]),
                                  proxy, sem) for k, t in enumerate(r["turnos"])])


def _falas(r: dict) -> tuple[list[tuple[str, np.ndarray]], int]:
    pers, falas, sr = _personas(r), [], None
    for k, t in enumerate(r["turnos"]):
        x, sr = _decodificar((TRABALHO / r["id_ligacao"] / nome_fala(k, t["texto"], pers[t["lado"]])).read_bytes())
        falas.append((t["lado"], _aparar(x, sr)))
    return falas, sr


def _banco_de_fundo(roteiros: list[dict], seed: int = 42) -> list[tuple[str, np.ndarray]]:
    """Falas neutras sorteadas de todas as ligações, com o id de origem (uma ligação não ouve a si mesma)."""
    cand = [(r, k) for r in roteiros for k, t in enumerate(r["turnos"]) if fala_neutra(t["texto"])]
    rng = np.random.default_rng(seed)
    banco = []
    for i in sorted(rng.choice(len(cand), min(N_FUNDO, len(cand)), replace=False)):
        r, k = cand[i]
        t = r["turnos"][k]
        x, sr = _decodificar((TRABALHO / r["id_ligacao"] / nome_fala(k, t["texto"], _personas(r)[t["lado"]])).read_bytes())
        banco.append((r["id_ligacao"], _aparar(x, sr)))
    return banco


def montar_ligacao(r: dict, idx: int, banco: list[tuple[str, np.ndarray]]) -> dict:
    """Encaixa as falas, aplica central + telefone + perda de pacotes em cada canal e grava o MP3."""
    falas, sr = _falas(r)
    rng = np.random.default_rng([42, idx])
    est, tempos = montar_estereo(falas, sr, rng)
    nivel = {"sbr_central": float(rng.uniform(*SBR_CENTRAL)), "snr_operador": float(rng.uniform(*SNR_OPERADOR)),
             "snr_cliente": float(rng.uniform(*SNR_CLIENTE))}
    fundo = [x for i, x in banco if i != r["id_ligacao"]]
    op = conversa_de_fundo(est[:, 0], sr, fundo, nivel["sbr_central"], rng)
    op = perda_pacotes(telefone(op, sr, nivel["snr_operador"], rng), SR_TEL, PERDA, rng)
    cli = perda_pacotes(telefone(est[:, 1], sr, nivel["snr_cliente"], rng), SR_TEL, PERDA, rng)
    tel = np.stack([op, cli], 1)
    exportar_mp3(tel, SR_TEL, PASTA_AUDIO / f"{r['id_ligacao']}.mp3")
    return {"duracao": round(len(tel) / SR_TEL, 2), "ruido": {k: round(v, 1) for k, v in nivel.items()}, "turnos": tempos}


def _gravar_json(caminho: Path, dados) -> None:
    tmp = caminho.with_suffix(".tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, caminho)


async def _principal(so: str | None, proxy: str | None) -> None:
    roteiros = json.loads((PASTA_LIG / "roteiros.json").read_text(encoding="utf-8"))
    arq_tempos = PASTA_LIG / "tempos.json"
    tempos = json.loads(arq_tempos.read_text(encoding="utf-8")) if arq_tempos.exists() else {}
    sem = asyncio.Semaphore(4)
    for r in roteiros:                                       # 1) todas as falas no cache (só aqui usa a rede)
        await _garantir_falas(r, proxy, sem)
    banco = _banco_de_fundo(roteiros)                        # 2) conversa de fundo e montagem, sem rede
    for idx, r in enumerate(roteiros):
        lid = r["id_ligacao"]
        if (so and lid != so) or ((PASTA_AUDIO / f"{lid}.mp3").exists() and lid in tempos):
            continue
        tempos[lid] = montar_ligacao(r, idx, banco)
        _gravar_json(arq_tempos, dict(sorted(tempos.items())))
        print(f"{lid}: {tempos[lid]['duracao']:.0f} s, ruído {tempos[lid]['ruido']}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", help="sintetiza só esta ligação")
    ap.add_argument("--proxy", default=os.environ.get("EA_PROXY"), help="proxy HTTP local (sem credenciais)")
    a = ap.parse_args()
    asyncio.run(_principal(a.so, a.proxy))


if __name__ == "__main__":
    main()
