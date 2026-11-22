"""Roteiros de ligação com gabarito, ligados a contatos reais da carteira sintética (fase 2).

Cada ligação vem de um contato de verdade do gerador (discador, CPC, cliente ALTO no dia, últimos 30 dias úteis),
com o operador e a tabulação daquele contato. O roteiro sai de uma gramática (seed 42) e o gabarito registra o que
foi dito: se houve oferta de Entrega Amigável, de que tipo, os alertas de conduta e as frases de cada rótulo.
As pegadinhas existem para a regra provar que não confunde "entrega" em outro contexto com oferta.

    py -m src.audio.roteiros      → dados/ligacoes/roteiros.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import HOJE

OFERTAS = ("OFERECEU", "SÓ O CLIENTE FALOU", "NÃO OFERECEU")
ALERTAS = ("ameaca_bloqueio", "desconto_percentual", "quitacao_sem_ressalva", "nome_limpo")
DESFECHOS_OFERTA = ("EA INDICADA", "RECUSA EA")
RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "dados" / "ligacoes" / "roteiros.json"

# vozes neurais pt-BR do edge-tts; a mesma voz muda de persona com velocidade e tom
PERSONAS = {
    "op_antonio":   {"papel": "operador", "voz": "pt-BR-AntonioNeural", "rate": "+0%", "pitch": "+0Hz", "genero": "M"},
    "op_francisca": {"papel": "operador", "voz": "pt-BR-FranciscaNeural", "rate": "+4%", "pitch": "+0Hz", "genero": "F"},
    "op_thalita":   {"papel": "operador", "voz": "pt-BR-ThalitaMultilingualNeural", "rate": "+2%", "pitch": "-2Hz",
                     "genero": "F"},
    "op_antonio_grave": {"papel": "operador", "voz": "pt-BR-AntonioNeural", "rate": "-4%", "pitch": "-6Hz", "genero": "M"},
    "cli_ana":   {"papel": "cliente", "voz": "pt-BR-FranciscaNeural", "rate": "-6%", "pitch": "+3Hz", "genero": "F"},
    "cli_bruno": {"papel": "cliente", "voz": "pt-BR-AntonioNeural", "rate": "+6%", "pitch": "-3Hz", "genero": "M"},
    "cli_carla": {"papel": "cliente", "voz": "pt-BR-ThalitaMultilingualNeural", "rate": "-8%", "pitch": "+2Hz",
                  "genero": "F"},
    "cli_davi":  {"papel": "cliente", "voz": "pt-BR-AntonioNeural", "rate": "-10%", "pitch": "-8Hz", "genero": "M"},
    "cli_elisa": {"papel": "cliente", "voz": "pt-BR-FranciscaNeural", "rate": "+8%", "pitch": "-2Hz", "genero": "F"},
}
_OP_PERSONAS = [p for p, v in PERSONAS.items() if v["papel"] == "operador"]
_NOMES_OP = {"M": ["Rafael", "Lucas", "Gustavo", "Felipe", "Diego", "Leandro"],
             "F": ["Camila", "Beatriz", "Larissa", "Fernanda", "Aline", "Vanessa"]}
_NOMES_CLI = {"M": ["Bruno", "Davi", "Marcos", "Paulo", "Ricardo", "Sérgio", "Tiago", "Wagner"],
              "F": ["Ana", "Carla", "Elisa", "Marta", "Juliana", "Patrícia", "Renata", "Sônia"]}
_SOBRENOMES = ["Silva", "Souza", "Pereira", "Costa", "Rocha", "Lima", "Martins", "Barbosa"]

# ---------------------------------------------------------------- gramática
# {sr}/{Sr}: o senhor / a senhora; {obg_op}/{obg_cli}: obrigado(a) de quem fala; {ele}: é ele / é ela
ABERTURA = ["{saud}, aqui é {op}, da Carteira Exemplo. Eu falo com {sr} {nome}?",
            "{saud}! Meu nome é {op}, falo da Carteira Exemplo. Por favor, {sr} {nome}?",
            "{saud}, tudo bem? Aqui é {op}, da Carteira Exemplo. É {sr} {nome} que está falando?"]
ABERTURA_CLI = ["Sim, sou eu.", "{ele}, pode falar.", "Sou eu, quem fala?", "Isso, sou eu mesmo. Pode falar."]
IDENTIDADE = ["Para sua segurança, pode me confirmar a sua data de nascimento?",
              "Antes de continuar, {sr} pode me confirmar os três primeiros números do seu documento?",
              "Só para confirmar o cadastro, qual é a sua data de nascimento?"]
IDENTIDADE_CLI = {"nasc": ["Doze de março de oitenta.", "Vinte e três de julho de oitenta e cinco.",
                           "Primeiro de outubro de setenta e nove.", "Quinze de maio de noventa e um."],
                  "doc": ["Quatro, sete, um.", "Dois, nove, cinco.", "Oito, três, seis."]}
SITUACAO = ["{Obg_op}, confirmado. Estou ligando sobre o financiamento do seu veículo, que está com {atraso} dias de "
            "atraso, somando {valor} reais.",
            "Perfeito, {obg_op}. O motivo da ligação é o contrato do seu carro: são {atraso} dias de atraso e o total "
            "está em {valor} reais.",
            "Certo, cadastro confirmado. O financiamento do seu veículo está com {atraso} dias em aberto, "
            "num total de {valor} reais."]
OBJECAO = ["Eu sei, perdi o emprego e está bem difícil.", "Esse mês está apertado, tive uns gastos com saúde.",
           "Eu paguei uma parcela semana passada, não caiu?", "Estou tentando organizar, mas não está fácil.",
           "Pois é, eu sei que está atrasado, mas a renda caiu muito."]
EMPATIA = ["Entendo, {sr}, por isso estou ligando, para achar uma saída.",
           "Eu compreendo, vamos ver o que dá para fazer.",
           "Certo, deixa eu verificar aqui as opções para {sr}.",
           "Entendi. Vamos olhar juntos o que cabe no seu bolso."]
OFERTA_EXPLICITA = ["Uma alternativa para {sr} é a entrega amigável do veículo, sem custo de remoção.",
                    "{Sr} já ouviu falar da entrega amigável? É quando {sr} devolve o carro de forma voluntária.",
                    "Hoje eu consigo abrir uma proposta de entrega amigável para o seu contrato.",
                    "Temos a opção da entrega amigável: {sr} leva o carro até o pátio e a gente cuida do resto."]
OFERTA_DEVOLUCAO = ["Uma saída é {sr} devolver o veículo; o valor da venda abate a dívida e, se sobrar, fica um saldo "
                    "remanescente.",
                    "Se não está dando para pagar, existe a possibilidade de entregar o carro para a gente e parar as "
                    "parcelas.",
                    "Tem também a opção de devolver o carro de forma voluntária, e o saldo remanescente a gente "
                    "negocia depois."]
REACAO_OFERTA = ["Como funciona isso?", "E o que acontece com o que eu já paguei?", "Mas e se sobrar dívida?",
                 "Nunca tinha ouvido falar. Explica melhor?"]
EXPLICACAO = ["O carro passa por uma avaliação, o valor da venda abate o que {sr} deve e o processo termina sem ação "
              "judicial.",
              "A gente agenda a vistoria, o valor apurado abate o saldo do contrato e o restante a gente parcela.",
              "{Sr} leva o carro no dia combinado, ele vai para avaliação e o valor entra direto no contrato."]
REACAO_FINAL = ["Vou pensar e converso com a minha família.", "Pode me mandar isso por escrito?",
                "Preciso do carro para trabalhar, não sei se consigo.", "Faz sentido, vou ver isso essa semana."]
CLIENTE_PEDE = ["Eu vi que dá para fazer a entrega amigável, isso vale para o meu caso?",
                "E se eu devolver o carro? Não dá para fazer isso?",
                "Um amigo meu fez a entrega amigável, eu posso fazer também?",
                "Eu queria entregar o veículo, como faço?"]
DESVIO = ["Isso eu não consigo ver por aqui agora; hoje a proposta que eu tenho para {sr} é o acordo.",
          "Esse assunto é com outro setor; o que eu posso fazer hoje é parcelar o atraso.",
          "Vou anotar aqui, mas o que está disponível agora é o pagamento das parcelas em atraso."]
ACORDO = ["Consigo parcelar o atraso em {vezes} vezes, com a primeira para a semana que vem.",
          "Para regularizar, a gente pode fazer um acordo com entrada pequena e o restante parcelado.",
          "Se {sr} pagar as duas parcelas mais antigas, o contrato volta a ficar em dia."]
REACAO_ACORDO = ["Isso eu até consigo.", "Não tenho esse dinheiro agora.", "Vou ver se consigo até sexta.",
                 "Me manda o valor certinho que eu vejo."]
EXTRA = [("E esse valor tem juros?", "Tem sim, mas no acordo os juros caem bastante."),
         ("Vocês aceitam pagamento por pix?", "Aceitamos, o código vai junto com a proposta."),
         ("Qual é a data de vencimento?", "A data a gente escolhe junto, do jeito que ficar melhor para {sr}."),
         ("Esse número é de vocês mesmo?", "É sim, pode salvar este contato para falar com a gente.")]
FECHO = ["Então fica combinado assim. {Obg_op} pela atenção e tenha {boa}.",
         "Combinado, {sr}. Qualquer dúvida é só retornar neste número. Tenha {boa}.",
         "Certo, fico no aguardo do seu retorno. {Obg_op} e tenha {boa}."]
FECHO_CLI = ["{Obg_cli}, tchau.", "Tá bom, até logo.", "{Obg_cli}, até mais."]

ALERTA_FRASES = {
    "ameaca_bloqueio": ["Se não regularizar, o próximo passo é bloquear a sua conta.",
                        "Do jeito que está, a gente pode penhorar os seus bens.",
                        "Se não pagar até sexta, vão bloquear o seu CPF.",
                        "Sem acordo, o banco pode penhorar o seu salário."],
    "desconto_percentual": ["Fechando hoje eu consigo {p}% de desconto.", "Tem um desconto de {p}% para quitar à vista.",
                            "Para hoje eu libero {p}% de desconto no total."],
    "quitacao_sem_ressalva": ["Fazendo isso, a sua dívida zera e não fica mais nada.",
                              "Com isso acaba com a dívida de vez.",
                              "Isso encerra a dívida inteira, {sr} não deve mais nada."],
    "nome_limpo": ["E com a entrega o seu nome fica limpo na hora.", "Devolvendo o carro, {sr} já sai com o nome limpo."],
}
ALERTA_PREFIXO = ["Pensa com carinho.", "Olha, vou ser sincero com {sr}.", "Escuta só."]
ALERTA_REACAO = ["Vou ver.", "Entendi.", "Hum, tá certo.", "Tá, vou pensar."]
# quase-alertas: parecem conduta irregular, mas não são (testam a precisão da regra)
QUASE = ["O desconto depende do valor que {sr} consegue pagar.", "Ninguém vai bloquear nada agora.",
         "Para tirar a restrição do nome, primeiro precisa regularizar."]
QUASE_OFERTA = "A entrega não zera a dívida automaticamente, pode ficar saldo."
PEGADINHA_LOJA = ["Quando a loja me entregou o carro, ele já veio com defeito.",
                  "A concessionária fez a entrega do carro com dois meses de atraso, sabia?"]
PEGADINHA_LOJA_RESP = ["Entendo, isso a gente pode registrar no seu cadastro.", "Certo, vou deixar anotado aqui."]
PEGADINHA_BOLETO = ["Vou te entregar o boleto por mensagem agora mesmo.", "A entrega do boleto é pelo aplicativo, tudo bem?",
                    "Posso fazer a entrega da segunda via por e-mail."]
PEGADINHA_ALERTA_CLI = ["Vocês vão bloquear minha conta?", "Eu ouvi dizer que vocês podem penhorar meus bens, é verdade?",
                        "Vão bloquear o meu salário por causa disso?"]
PEGADINHA_ALERTA_RESP = ["Não, não é assim que funciona; o que a gente quer é encontrar uma saída.",
                         "Fique tranquilo que não, a ideia aqui é resolver da melhor forma."]
VALORES = ["oito mil", "doze mil", "quinze mil", "dezoito mil", "vinte e dois mil", "trinta mil"]
ATRASOS = ["quarenta", "setenta", "noventa", "cento e vinte", "duzentos"]
PALAVRAS_MIN, PALAVRAS_MAX = 110, 300           # ~51 a ~138 s a 130 palavras por minuto


def _persona_operador(operador: str) -> str:
    return _OP_PERSONAS[int(operador.split("-")[1]) % len(_OP_PERSONAS)]


def _nome_operador(operador: str) -> str:
    k = int(operador.split("-")[1])
    return _NOMES_OP[PERSONAS[_persona_operador(operador)]["genero"]][k % 6]


def gerar_roteiro(lig: dict, rng: np.random.Generator) -> dict:
    """Roteiro e gabarito de UMA ligação. `lig` traz `oferta_alvo` e, opcionalmente, o plano montado por
    `montar_roteiros`: `tipo_alvo`, `alertas_alvo`, `pegadinhas` (loja, boleto, alerta_cliente) e `quase`."""
    oferta = lig["oferta_alvo"]
    tipo = lig.get("tipo_alvo") or (rng.choice(["explicita", "devolucao"]) if oferta == "OFERECEU" else None)
    tipo = str(tipo) if oferta == "OFERECEU" else None
    alertas = list(lig.get("alertas_alvo", []))
    peg = set(lig.get("pegadinhas", []))
    p_op = _persona_operador(lig["operador"])
    p_cli = str(rng.choice([p for p, v in PERSONAS.items()
                            if v["papel"] == "cliente" and v["voz"] != PERSONAS[p_op]["voz"]]))
    g_op, g_cli = PERSONAS[p_op]["genero"], PERSONAS[p_cli]["genero"]
    manha = rng.random() < 0.5
    sub = {"op": _nome_operador(lig["operador"]),
           "nome": f"{rng.choice(_NOMES_CLI[g_cli])} {rng.choice(_SOBRENOMES)}",
           "sr": "o senhor" if g_cli == "M" else "a senhora", "Sr": "O senhor" if g_cli == "M" else "A senhora",
           "ele": "É ele" if g_cli == "M" else "É ela",
           "obg_op": "obrigado" if g_op == "M" else "obrigada", "Obg_op": "Obrigado" if g_op == "M" else "Obrigada",
           "Obg_cli": "Obrigado" if g_cli == "M" else "Obrigada",
           "saud": "Bom dia" if manha else "Boa tarde", "boa": "um bom dia" if manha else "uma boa tarde",
           "valor": str(rng.choice(VALORES)), "atraso": str(rng.choice(ATRASOS)),
           "vezes": str(rng.choice(["três", "quatro", "seis"])), "p": str(int(rng.choice([30, 40, 50, 60, 70])))}

    def f(lista):                                              # sorteia e preenche
        return str(rng.choice(lista)).format(**sub)

    turnos, frases = [], {}

    def fala(lado, texto, rotulo=None):
        if rotulo:
            frases.setdefault(rotulo, []).append(texto)
        if turnos and turnos[-1]["lado"] == lado:              # falas seguidas do mesmo lado viram um turno só
            turnos[-1]["texto"] += " " + texto
        else:
            turnos.append({"lado": lado, "texto": texto})

    ident = int(rng.integers(len(IDENTIDADE)))
    fala("operador", f(ABERTURA))
    fala("cliente", f(ABERTURA_CLI))
    fala("operador", IDENTIDADE[ident].format(**sub))
    fala("cliente", f(IDENTIDADE_CLI["doc" if ident == 1 else "nasc"]))
    fala("operador", f(SITUACAO))
    fala("cliente", f(OBJECAO))
    fala("operador", f(EMPATIA))
    if "alerta_cliente" in peg:
        fala("cliente", f(PEGADINHA_ALERTA_CLI), "pegadinha_alerta_cliente")
        fala("operador", f(PEGADINHA_ALERTA_RESP))
    if "loja" in peg:
        fala("cliente", f(PEGADINHA_LOJA), "pegadinha_loja")
        fala("operador", f(PEGADINHA_LOJA_RESP))

    if oferta == "OFERECEU":
        fala("operador", f(OFERTA_EXPLICITA if tipo == "explicita" else OFERTA_DEVOLUCAO), "oferta")
        if "nome_limpo" in alertas:
            fala("operador", f(ALERTA_FRASES["nome_limpo"]), "nome_limpo")
        fala("cliente", f(REACAO_OFERTA))
        fala("operador", f(EXPLICACAO))
        fala("cliente", f(REACAO_FINAL))
        if lig.get("quase") and "quitacao_sem_ressalva" not in alertas:
            fala("operador", QUASE_OFERTA)
            fala("cliente", f(ALERTA_REACAO))
    else:
        if oferta == "SÓ O CLIENTE FALOU":
            fala("cliente", f(CLIENTE_PEDE), "cliente_pede")
            fala("operador", f(DESVIO))
        fala("operador", f(ACORDO))
        fala("cliente", f(REACAO_ACORDO))
        if lig.get("quase"):
            fala("operador", f(QUASE))
            fala("cliente", f(ALERTA_REACAO))

    outros = [a for a in alertas if a != "nome_limpo"]
    if outros:                                                 # alerta num turno próprio, longe de qualquer ressalva
        fala("operador", f(ALERTA_PREFIXO))
        for a in outros:
            fala("operador", f(ALERTA_FRASES[a]), a)
        fala("cliente", f(ALERTA_REACAO))

    extras = list(rng.permutation(len(EXTRA)))
    n_extra = int(rng.integers(0, 3))

    def palavras():
        return sum(len(t["texto"].split()) for t in turnos) + sum(len(x.split()) for x in fecho)

    fecho = [f(PEGADINHA_BOLETO) if "boleto" in peg else "", f(FECHO), f(FECHO_CLI)]
    for k in range(len(extras)):
        if k >= n_extra and palavras() >= PALAVRAS_MIN:
            break
        if palavras() + len(" ".join(EXTRA[extras[k]]).split()) > PALAVRAS_MAX:
            break
        cli, op = EXTRA[extras[k]]
        fala("cliente", cli)
        fala("operador", op.format(**sub))
    if fecho[0]:
        fala("operador", fecho[0], "pegadinha_boleto")
    fala("operador", fecho[1])
    fala("cliente", fecho[2])

    return {"id_ligacao": lig["id_ligacao"], "data": lig["data"], "operador": lig["operador"],
            "persona_operador": p_op, "persona_cliente": p_cli, "turnos": turnos,
            "gabarito": {"oferta": oferta, "tipo": tipo, "alertas": [a for a in ALERTAS if a in alertas],
                         "frases": frases}}


def selecionar_ligacoes(tabelas: dict, esc: pd.DataFrame, n: int = 60, dias_uteis: int = 30,
                        seed: int = 42) -> pd.DataFrame:
    """Contatos de discador com CPC nos últimos `dias_uteis` dias úteis até HOJE, de contratos ALTO naquele dia.
    OFERECEU sai das ligações com desfecho de EA; SÓ O CLIENTE FALOU e NÃO OFERECEU, dos demais desfechos.
    Um contrato entra uma vez só. Se faltar universo num estrato, completa com o vizinho e avisa."""
    rng = np.random.default_rng(seed)
    dias = pd.bdate_range(end=HOJE, periods=dias_uteis)
    ct = tabelas["contatos"]
    ct = ct[(ct["canal"] == "discador") & ct["cpc"] & pd.to_datetime(ct["data"]).isin(dias)]
    alto = esc.loc[esc["classe"].astype(str) == "ALTO", ["dia", "id_contrato"]]
    u = ct.merge(alto, left_on=[pd.to_datetime(ct["data"]), "id_contrato"], right_on=["dia", "id_contrato"])
    u = u.iloc[rng.permutation(len(u))].drop_duplicates("id_contrato")
    n_of, n_so = round(0.55 * n), round(0.15 * n)
    alvo = [("OFERECEU", n_of), ("SÓ O CLIENTE FALOU", n_so), ("NÃO OFERECEU", n - n_of - n_so)]
    ofe, resto = u[u["desfecho"].isin(DESFECHOS_OFERTA)], u[~u["desfecho"].isin(DESFECHOS_OFERTA)]
    if len(ofe) < n_of:
        print(f"aviso: só {len(ofe)} ligações com desfecho de EA; OFERECEU completado com outros desfechos")
        ofe, resto = pd.concat([ofe, resto.iloc[:n_of - len(ofe)]]), resto.iloc[n_of - len(ofe):]
    partes, pos = [ofe.iloc[:n_of].assign(oferta_alvo="OFERECEU")], 0
    for rotulo, k in alvo[1:]:
        partes.append(resto.iloc[pos:pos + k].assign(oferta_alvo=rotulo))
        pos += k
    sel = pd.concat(partes)
    if len(sel) < n:
        print(f"aviso: universo com só {len(sel)} ligações (pedidas {n})")
    sel = sel.assign(data=pd.to_datetime(sel["data"]).dt.strftime("%Y-%m-%d"))
    cols = ["id_ligacao", "data", "id_contrato", "operador", "desfecho", "tabulacao", "oferta_alvo"]
    return sel[cols].sort_values(["data", "id_ligacao"]).reset_index(drop=True)


def montar_roteiros(tabelas: dict, esc: pd.DataFrame, n: int = 60, seed: int = 42) -> list[dict]:
    """Seleciona as ligações e distribui o plano com cotas fixas (assim cada pegadinha aparece de verdade):
    ~30% das ofertas por devolução, ~25% das ligações com alerta (todos os tipos presentes), pegadinhas em 5–6
    ligações cada e quase-alertas em ~30%."""
    sel = selecionar_ligacoes(tabelas, esc, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    ligs = sel.to_dict("records")
    for lig in ligs:
        lig.update(tipo_alvo=None, alertas_alvo=[], pegadinhas=[], quase=bool(rng.random() < 0.3))
    ofe = [i for i, lig in enumerate(ligs) if lig["oferta_alvo"] == "OFERECEU"]
    for j, i in enumerate(rng.permutation(ofe)):
        ligs[i]["tipo_alvo"] = "devolucao" if j < round(0.3 * len(ofe)) else "explicita"
    n_alerta = math.ceil(0.25 * len(ligs))
    com_of = list(rng.permutation(ofe))[:max(1, n_alerta // 3)]
    demais = [i for i in rng.permutation(len(ligs)) if i not in com_of][:n_alerta - len(com_of)]
    for j, i in enumerate(com_of + demais):
        eleg = list(ALERTAS) if ligs[i]["oferta_alvo"] == "OFERECEU" else list(ALERTAS[:3])
        eleg = eleg[::-1] if ligs[i]["oferta_alvo"] == "OFERECEU" else eleg     # nome_limpo primeiro nas ofertas
        a = [eleg[j % len(eleg)]]
        if rng.random() < 0.2:
            a.append(str(rng.choice([x for x in ALERTAS[:3] if x not in a])))
        ligs[i]["alertas_alvo"] = a
    for peg, k in (("loja", 5), ("boleto", 6), ("alerta_cliente", 5)):
        for i in rng.choice(len(ligs), min(k, len(ligs)), replace=False):
            ligs[i]["pegadinhas"].append(peg)
    return [gerar_roteiro(lig, np.random.default_rng([seed, i])) for i, lig in enumerate(ligs)]


def main():
    from src.base_analitica import carregar
    from src.escoragem import escorar_periodos
    sys.stdout.reconfigure(encoding="utf-8")
    t = carregar()
    esc = escorar_periodos(t, ("mensal", "congelado"), cache=RAIZ / "dados" / "escoragem")["mensal"][0]
    rot = montar_roteiros(t, esc)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(rot, ensure_ascii=False, indent=1), encoding="utf-8")
    g = pd.Series([r["gabarito"]["oferta"] for r in rot]).value_counts().to_dict()
    print(f"{len(rot)} roteiros → {SAIDA.relative_to(RAIZ)}; gabarito {g}; "
          f"com alerta {sum(bool(r['gabarito']['alertas']) for r in rot)}")


if __name__ == "__main__":
    main()
