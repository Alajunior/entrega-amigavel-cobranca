"""Transcrição por canal com faster-whisper (roda local, uma vez) e montagem dos turnos.

Cada canal do MP3 (operador em L, cliente em R) é transcrito separado, com timestamp por palavra. Os turnos saem
das palavras: pausa ≥ 1 s abre balão novo e balões seguidos do mesmo lado são fundidos. Antes, a limpeza tira dois
vícios conhecidos do Whisper: o eco do `initial_prompt` e o loop de repetição.

    py -m src.audio.transcrever      → dados/ligacoes/transcricoes.jsonl (checkpoint por ligação)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.audio.regras import normalizar
from src.audio.sintetizar import PASTA_AUDIO, PASTA_LIG

# prompt genérico de cobrança, o mesmo que serviria a qualquer carteira: sem o nome do produto. Com "entrega amigável"
# no prompt o Whisper acerta o termo até no ruído (a calibração mostrou); sem ele, aparecem "entrega migada" e
# "entrega ligada", que é o que a regra tolerante precisa provar que resolve.
PROMPT = "Ligação de cobrança de financiamento de veículo: acordo, parcelas em atraso, boleto, desconto."
MODELO = "large-v3-turbo"
VERSAO = "turnos_v1"
SAIDA = PASTA_LIG / "transcricoes.jsonl"
PARAMS = {"language": "pt", "beam_size": 5, "vad_filter": True, "word_timestamps": True, "initial_prompt": PROMPT}


def turnos_de_palavras(pal_op: list[dict], pal_cli: list[dict], pausa: float = 1.0) -> list[dict]:
    """Palavras {inicio, fim, texto} de cada lado → turnos {lado, inicio, fim, texto} em ordem de início."""
    baloes = []
    for lado, pals in (("operador", pal_op), ("cliente", pal_cli)):
        atual = None
        for w in sorted(pals, key=lambda w: w["inicio"]):
            if atual is None or w["inicio"] - atual["fim"] >= pausa:
                atual = {"lado": lado, "inicio": w["inicio"], "fim": w["fim"], "pal": []}
                baloes.append(atual)
            atual["pal"].append(w["texto"].strip())
            atual["fim"] = max(atual["fim"], w["fim"])
    turnos = []
    for b in sorted(baloes, key=lambda b: (b["inicio"], b["lado"])):
        if turnos and turnos[-1]["lado"] == b["lado"]:             # ninguém falou no meio: mesmo turno
            turnos[-1]["fim"] = max(turnos[-1]["fim"], b["fim"])
            turnos[-1]["texto"] += " " + " ".join(b["pal"])
        else:
            turnos.append({"lado": b["lado"], "inicio": b["inicio"], "fim": b["fim"], "texto": " ".join(b["pal"])})
    for t in turnos:
        t["texto"] = " ".join(t["texto"].split())
    return turnos


def limpar(texto: str, prompt: str = PROMPT) -> str:
    """Tira o eco do prompt (≥ 4 palavras seguidas do prompt) e colapsa loops (1–6 palavras repetidas ≥ 3 vezes)."""
    pal = texto.split()
    norm = [normalizar(p) for p in pal]
    pnorm = normalizar(prompt).split()
    seq_prompt = " " + " ".join(pnorm) + " "
    manter = [True] * len(pal)
    i = 0
    while i < len(pal):
        k = 0
        while i + k < len(pal) and f" {' '.join(norm[i:i + k + 1])} " in seq_prompt:
            k += 1
        if k >= 4:
            manter[i:i + k] = [False] * k
            i += k
        else:
            i += 1
    pal = [p for p, m in zip(pal, manter) if m]
    for n in range(1, 7):
        i, saida = 0, []
        while i < len(pal):
            bloco = [normalizar(p) for p in pal[i:i + n]]
            rep = 1
            while len(bloco) == n and [normalizar(p) for p in pal[i + rep * n:i + (rep + 1) * n]] == bloco:
                rep += 1
            if rep >= 3:
                saida += pal[i:i + n]
                i += rep * n
            else:
                saida.append(pal[i])
                i += 1
        pal = saida
    return " ".join(pal)


def wer(ref: str, hip: str) -> float:
    """Taxa de erro por palavra: distância de edição (sobre o texto normalizado) ÷ palavras da referência."""
    r, h = normalizar(ref).split(), normalizar(hip).split()
    if not r:
        return 0.0 if not h else 1.0
    ant = list(range(len(h) + 1))
    for i, wr in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, wh in enumerate(h, 1):
            cur[j] = min(ant[j] + 1, cur[j - 1] + 1, ant[j - 1] + (wr != wh))
        ant = cur
    return ant[-1] / len(r)


def _ler(caminho: Path) -> list[dict]:
    if not Path(caminho).exists():
        return []
    return [json.loads(x) for x in Path(caminho).read_text(encoding="utf-8").splitlines() if x.strip()]


def ja_feitas(caminho: Path) -> set[str]:
    return {r["id_ligacao"] for r in _ler(caminho)}


def gravar_checkpoint(caminho: Path, registro: dict) -> None:
    """Reescreve o .jsonl inteiro trocando o registro de mesmo id (rodar de novo não duplica), via .tmp + replace."""
    caminho = Path(caminho)
    regs = {r["id_ligacao"]: r for r in _ler(caminho)}
    regs[registro["id_ligacao"]] = registro
    tmp = caminho.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs.values()), encoding="utf-8")
    os.replace(tmp, caminho)


def _canais_16k(mp3: Path) -> tuple[np.ndarray, float]:
    import av
    with av.open(str(mp3)) as c:
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=16000)
        partes = [g.to_ndarray() for f in c.decode(s) for g in rs.resample(f)]
        partes += [g.to_ndarray() for g in rs.resample(None)]
    x = np.concatenate(partes, axis=1).astype(np.float32)
    return x, x.shape[1] / 16000


def transcrever_ligacao(mp3: Path, modelo) -> dict:
    """Transcreve os dois canais separados e monta os turnos (palavras limpas do eco e do loop, por balão)."""
    x, dur = _canais_16k(mp3)
    palavras = []
    for canal in (0, 1):
        segs, _ = modelo.transcribe(x[canal], **PARAMS)
        palavras.append([{"inicio": round(w.start, 2), "fim": round(w.end, 2), "texto": w.word.strip()}
                         for s in segs for w in (s.words or []) if w.word.strip()])
    turnos = turnos_de_palavras(*palavras)
    for t in turnos:
        t["texto"] = limpar(t["texto"])
    turnos = [t for t in turnos if t["texto"]]
    return {"id_ligacao": Path(mp3).stem, "turnos": turnos, "duracao": round(dur, 2), "modelo": MODELO,
            "versao": VERSAO}


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # o modelo já está em cache: não toca a rede
    sys.stdout.reconfigure(encoding="utf-8")
    from faster_whisper import WhisperModel
    feitas = ja_feitas(SAIDA)
    faltam = [p for p in sorted(PASTA_AUDIO.glob("*.mp3")) if p.stem not in feitas]
    print(f"{len(feitas)} já transcritas, {len(faltam)} a fazer", flush=True)
    if not faltam:
        return
    modelo = WhisperModel(MODELO, device="cpu", compute_type="int8")
    for mp3 in faltam:
        t0 = time.time()
        gravar_checkpoint(SAIDA, transcrever_ligacao(mp3, modelo))
        print(f"{mp3.stem}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
