"""Regras de oferta de Entrega Amigável e de alertas de conduta, aplicadas ao texto da transcrição.

O texto vem do Whisper, que erra: "entrega a migável", "entrga amigável", "entrega ligável". O modo tolerante
aceita essas grafias, com cuidado para não abrir a porta a falso positivo:
    - palavra rara (migavel, amigaval, anigavel, ligavel) vale sozinha até 3 palavras depois da âncora;
    - palavra comum (amiga, amigo, ligada, brigado) só IMEDIATAMENTE depois da âncora;
    - a âncora (entrega, entregar, devolver, devolução) aceita 1 edição (entrga, entegar).
Exclusões: "entrega" que a loja/concessionária/revenda faz ao cliente, e entrega de boleto/documento/segunda via.
Alertas valem só na fala do operador e no máximo um por tipo.
"""
import re
import unicodedata

ANCORAS = ("entrega", "entregar", "devolucao", "devolver")
VERBOS_DEVOLUCAO = ANCORAS + ("devolve", "devolvendo", "entregando")
OBJETOS = ("veiculo", "carro", "bem", "automovel")
DETERMINANTES = ("o", "a", "seu", "sua", "esse", "este")   # até dois seguidos: "o seu carro"
RARAS = ("migavel", "amigaval", "anigavel", "ligavel", "amigavo", "amigaveu",
         "migada", "amigada")                # estas duas apareceram na rodada de calibração do ruído
COMUNS = ("amiga", "amigo", "ligada", "brigado", "amigas")
CONTEXTO_LOJA = ("loja", "lojas", "concessionaria", "concessionarias", "revenda", "revendas", "vendedor")
PARA_CLIENTE = ("para o senhor", "para a senhora", "para voce", "pra voce")
NAO_E_VEICULO = ("boleto", "boletos", "documento", "documentos", "segunda", "comprovante", "carne")
VOCAB_PRODUTO = ("saldo remanescente", "quitativa")
VOCAB_TOLERANTE = (r"saldo remanesc\w*", r"quitativ\w*")    # "saldo remanescendo", "quitativo"

_NUM = r"(\d{1,2}|dez|quinze|vinte|trinta|quarenta|cinquenta|sessenta|setenta|oitenta|noventa)( e (cinco|\w+))?"
_PCT = _NUM + r" ?(%|por cento)"
_RE_ALERTA = {
    "ameaca_bloqueio": re.compile(r"\b(bloque\w*|penhor\w*)( \w+){0,3} (cpf|conta|contas|salario|bens|nome)\b"),
    "desconto_percentual": re.compile(rf"\b{_PCT} de desconto\b|\bdesconto de (ate )?{_PCT}"),
    "quitacao_sem_ressalva": re.compile(r"\b(zer(a|ar|ou|ando|ada|ado)|encerr\w*|acab\w* com|liquid\w*)( \w+){0,3} divida\b"
                                        r"|\bdivida( \w+){0,2} (zer(a|ar|ou|ando|ada|ado|o)|encerr\w*|acab\w*)\b"),
    "nome_limpo": re.compile(r"\bnome( \w+){0,2} limpo\b|\blimp\w* (o )?(seu |sua )?nome\b"),
}
_RESSALVA = re.compile(r"saldo remanescente|pode ficar saldo|fica um saldo|pode sobrar|sobrar saldo|nao zera|nao quita")
_ANCORA_EA = re.compile(r"^(entreg|devol|amigav)")
_NEGACAO = {"nao", "ninguem", "nunca", "jamais", "nem"}   # "ninguém vai bloquear a sua conta" é tranquilizar
JANELA_NOME_LIMPO = 45


def normalizar(txt: str) -> str:
    """Minúsculas, sem acento, só letras/dígitos/%, espaços colapsados."""
    txt = unicodedata.normalize("NFKD", str(txt or "").lower())
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9%]+", " ", txt).split())


def ate_1_edicao(a: str, b: str) -> bool:
    """Distância de Levenshtein ≤ 1 (troca, inserção ou remoção de uma letra)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] if len(a) == len(b) else a[i:] == b[i + 1:]


def _e_ancora(tok: str, tolerante: bool, lista=ANCORAS) -> bool:
    return tok in lista or (tolerante and len(tok) >= 6 and any(ate_1_edicao(tok, a) for a in lista))


def _trecho(toks: list[str], i: int) -> str:
    return " ".join(toks[max(0, i - 4):i + 8])


def _excluida(toks: list[str], i: int, devolucao: bool = False) -> bool:
    """Loja/concessionária antes, "me entregou", ou entrega de boleto/documento. "Entregar o carro para o senhor"
    só exclui a devolução (é a loja entregando); "a entrega amigável para a senhora" é oferta."""
    antes, depois = toks[max(0, i - 6):i], " ".join(toks[i + 1:i + 9])
    if any(t in CONTEXTO_LOJA for t in antes) or "me" in toks[max(0, i - 1):i]:
        return True
    if not devolucao:                     # "entrega amigável segunda-feira" é oferta; o boleto só exclui a devolução
        return False
    if any(t in NAO_E_VEICULO for t in toks[i + 1:i + 4]):
        return True
    return any(p in depois for p in PARA_CLIENTE)


def achar_mencoes(texto: str, tolerante: bool = True) -> list[dict]:
    """Menções de oferta no texto: `explicita` ("entrega amigável" e grafias erradas) ou `devolucao`
    ("devolver o veículo", "saldo remanescente"). No máximo uma menção por âncora; a explícita tem prioridade."""
    toks = normalizar(texto).split()
    achadas = []
    for i, tok in enumerate(toks):
        if _e_ancora(tok, tolerante) and not _excluida(toks, i):
            seg = toks[i + 1:i + 4]
            alvo = [j for j, t in enumerate(seg) if t == "amigavel"
                    or (tolerante and (t in RARAS or ate_1_edicao(t, "amigavel")))]
            if not alvo and tolerante and seg and seg[0] in COMUNS:
                alvo = [0]
            if alvo:
                achadas.append({"tipo": "explicita", "trecho": _trecho(toks, i), "pos": i, "n": alvo[0] + 2})
                continue
        # devolução: verbo exato (a tolerância de 1 letra transformaria "devolveu", "devolvi" em oferta)
        if tok in VERBOS_DEVOLUCAO and not _excluida(toks, i, devolucao=True):
            j = i + 1
            while j < min(i + 3, len(toks)) and toks[j] in DETERMINANTES:
                j += 1
            if j < len(toks) and toks[j] in OBJETOS:
                achadas.append({"tipo": "devolucao", "trecho": _trecho(toks, i), "pos": i, "n": j - i + 1})
    norm = " ".join(toks)
    for termo in VOCAB_TOLERANTE if tolerante else VOCAB_PRODUTO:
        for m in re.finditer(rf"\b{termo}\b", norm):
            i = norm[:m.start()].count(" ")
            achadas.append({"tipo": "devolucao", "trecho": _trecho(toks, i), "pos": i, "n": len(m.group(0).split())})
    return sorted(achadas, key=lambda m: m["pos"])


def classificar_oferta(turnos: list[dict], tolerante: bool = True) -> dict:
    """OFERECEU se o operador mencionou; senão SÓ O CLIENTE FALOU se o cliente mencionou; senão NÃO OFERECEU.
    Canal do operador sem texto (VAD cortou tudo, ruído) → INCONCLUSIVO, mesmo que o cliente tenha falado:
    ninguém ouviu o operador, então não dá para afirmar que ele não ofertou."""
    por_lado = {"operador": [], "cliente": []}
    tem_texto = False
    for t in turnos:
        texto = t.get("texto") or ""
        tem_texto = tem_texto or (t.get("lado") == "operador" and bool(texto.strip()))
        por_lado.setdefault(t.get("lado"), []).extend(achar_mencoes(texto, tolerante))
    r = {"oferta": "NÃO OFERECEU", "tipo": None, "quem": None, "trecho": "",
         "mencoes_operador": len(por_lado["operador"]), "mencoes_cliente": len(por_lado["cliente"])}
    if not tem_texto:
        return r | {"oferta": "INCONCLUSIVO"}
    if por_lado["operador"]:
        exp = [m for m in por_lado["operador"] if m["tipo"] == "explicita"]
        m = (exp or por_lado["operador"])[0]
        return r | {"oferta": "OFERECEU", "tipo": m["tipo"], "quem": "operador", "trecho": m["trecho"]}
    if por_lado["cliente"]:
        return r | {"oferta": "SÓ O CLIENTE FALOU", "quem": "cliente", "trecho": por_lado["cliente"][0]["trecho"]}
    return r


def _negado(norm: str, m: re.Match) -> bool:
    """Negação nas 4 palavras antes do trecho ou dentro dele ("não vamos penhorar", "a dívida não vai zerar")."""
    antes = norm[:m.start()].split()[-4:]
    return bool(_NEGACAO & set(antes + m.group(0).split()))


def alertas_operador(turnos: list[dict]) -> list[dict]:
    """Alertas de conduta na fala do operador (a do cliente não conta), no máximo um por tipo.
    Cada alerta diz onde está: `turno` (índice na lista), `pos` e `n` (palavras do texto normalizado do turno)."""
    achados = {}
    falas = [(k, normalizar(t.get("texto"))) for k, t in enumerate(turnos) if t.get("lado") == "operador"]
    for k, norm in falas:
        for alerta in ("ameaca_bloqueio", "desconto_percentual", "quitacao_sem_ressalva"):
            m = next((m for m in _RE_ALERTA[alerta].finditer(norm) if not _negado(norm, m)), None)
            if alerta in achados or not m:
                continue
            if alerta == "quitacao_sem_ressalva" and _RESSALVA.search(norm):
                continue
            achados[alerta] = {"trecho": norm[max(0, m.start() - 40):m.end() + 40].strip(), "turno": k,
                               "pos": norm[:m.start()].count(" "), "n": len(m.group(0).split())}
    onde = [(k, j) for k, norm in falas for j in range(len(norm.split()))]   # palavra global → (turno, posição)
    toks = " ".join(norm for _, norm in falas).split()  # nome limpo: só perto de uma menção de entrega/devolução
    texto = " ".join(toks)
    ancoras = [i for i, t in enumerate(toks) if _ANCORA_EA.match(t)]
    for m in _RE_ALERTA["nome_limpo"].finditer(texto):
        i = texto[:m.start()].count(" ")
        if not _negado(texto, m) and any(abs(i - a) <= JANELA_NOME_LIMPO for a in ancoras):
            achados["nome_limpo"] = {"trecho": _trecho(toks, i), "turno": onde[i][0], "pos": onde[i][1],
                                     "n": len(m.group(0).split())}
            break
    ordem = ("ameaca_bloqueio", "desconto_percentual", "quitacao_sem_ressalva", "nome_limpo")
    return [{"alerta": a} | achados[a] for a in ordem if a in achados]
