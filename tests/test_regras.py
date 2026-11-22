import pytest

from src.audio.regras import achar_mencoes, alertas_operador, ate_1_edicao, classificar_oferta, normalizar


def op(*textos):
    return [{"lado": "operador", "texto": t} for t in textos]


def cli(*textos):
    return [{"lado": "cliente", "texto": t} for t in textos]


def test_normalizar_e_levenshtein():
    assert normalizar("  Entrega   AMIGÁVEL, 30%! ") == "entrega amigavel 30%"
    assert ate_1_edicao("entrga", "entrega") and ate_1_edicao("entegar", "entregar") and ate_1_edicao("a", "a")
    assert not ate_1_edicao("entrar", "entregar") and not ate_1_edicao("ab", "ba c")


@pytest.mark.parametrize("frase,tipo", [
    ("Uma alternativa é a entrega amigável do veículo, sem custo.", "explicita"),
    ("O senhor já ouviu falar da entrega amigável?", "explicita"),
    ("A gente faz a devolução amigável do bem.", "explicita"),
    ("Uma saída é o senhor devolver o veículo e o valor abate a dívida.", "devolucao"),
    ("Existe a possibilidade de entregar o carro para a gente.", "devolucao"),
    ("Se sobrar, fica um saldo remanescente que a gente negocia.", "devolucao"),
])
def test_oferta_explicita_e_devolucao(frase, tipo):
    r = classificar_oferta(op(frase))
    assert r["oferta"] == "OFERECEU" and r["tipo"] == tipo and r["quem"] == "operador" and r["trecho"]


def test_explicita_tem_prioridade_sobre_devolucao():
    r = classificar_oferta(op("Dá para devolver o carro.", "É a entrega amigável."))
    assert r["tipo"] == "explicita" and r["mencoes_operador"] == 2


@pytest.mark.parametrize("frase", ["Temos a entrega a migável para o seu contrato.", "Posso abrir a entrega amigaval.",
                                   "É a entrga amigável do veículo.", "A opção é a entrega ligável.",
                                   "Temos a entrega anigavel hoje."])
def test_tolerancia_a_erro_de_transcricao(frase):
    assert classificar_oferta(op(frase), tolerante=True)["oferta"] == "OFERECEU"
    assert classificar_oferta(op(frase), tolerante=False)["oferta"] == "NÃO OFERECEU"


def test_palavra_comum_so_depois_de_entrega():
    assert classificar_oferta(op("Uma amiga minha disse que a entrega é rápida."))["oferta"] == "NÃO OFERECEU"
    assert classificar_oferta(op("A senhora conhece a entrega amiga?"))["oferta"] == "OFERECEU"
    assert classificar_oferta(op("Vou fazer a entrega da ligação amiga."))["oferta"] == "NÃO OFERECEU"


@pytest.mark.parametrize("frase", ["Quando a loja me entregou o carro, ele já veio com defeito.",
                                   "A concessionária fez a entrega do carro com atraso.",
                                   "Vou entregar o boleto por mensagem.", "A entrega do boleto é pelo aplicativo.",
                                   "Posso fazer a entrega da segunda via por e-mail.",
                                   "Vou entregar os documentos na agência.",
                                   "A revenda fez a entrega amigável do carro para o senhor."])
def test_pegadinhas_nao_sao_oferta(frase):
    assert classificar_oferta(op(frase) + cli(frase))["oferta"] == "NÃO OFERECEU"


def test_so_o_cliente_falou():
    r = classificar_oferta(op("Bom dia, aqui é da Carteira Exemplo.") + cli("Posso fazer a entrega amigável?")
                           + op("Hoje a proposta é o acordo."))
    assert r["oferta"] == "SÓ O CLIENTE FALOU" and r["mencoes_cliente"] == 1 and r["quem"] == "cliente"
    assert r["tipo"] is None


@pytest.mark.parametrize("turnos", [[], op(""), op("   ") + cli(""), [{"lado": "operador", "texto": None}]])
def test_texto_vazio_e_inconclusivo(turnos):
    r = classificar_oferta(turnos)
    assert r["oferta"] == "INCONCLUSIVO" and r["tipo"] is None


def test_mencao_traz_trecho_curto():
    texto = " ".join(["palavra"] * 30) + " temos a entrega amigável " + " ".join(["fim"] * 30)
    m = achar_mencoes(texto)
    assert len(m) == 1 and "entrega amigavel" in m[0]["trecho"] and len(m[0]["trecho"].split()) <= 12


@pytest.mark.parametrize("alerta,positiva,negativa", [
    ("ameaca_bloqueio", "Se não regularizar, o próximo passo é bloquear a sua conta.", "Ninguém vai bloquear nada agora."),
    ("ameaca_bloqueio", "Sem acordo, o banco pode penhorar o seu salário.", "A conta do acordo eu te passo depois."),
    ("desconto_percentual", "Fechando hoje eu consigo 40% de desconto.", "O desconto depende do valor que o senhor paga."),
    ("desconto_percentual", "Tem um desconto de quarenta por cento à vista.", "Quarenta por cento dos clientes pagam."),
    ("desconto_percentual", "Consigo um desconto de até 30% se fechar hoje.", "Paga 30% de entrada e o resto parcelado."),
    ("quitacao_sem_ressalva", "Fazendo isso, a sua dívida zera e não fica mais nada.",
     "A entrega não zera a dívida automaticamente, pode ficar saldo."),
    ("quitacao_sem_ressalva", "Com isso acaba com a dívida de vez.",
     "Isso encerra a dívida, mas se sobrar fica um saldo remanescente."),
    ("nome_limpo", "Com a entrega amigável o seu nome fica limpo na hora.",
     "Para tirar a restrição do nome, primeiro precisa regularizar."),
])
def test_alertas_por_tipo(alerta, positiva, negativa):
    assert [a["alerta"] for a in alertas_operador(op(positiva))] == [alerta]
    assert alerta not in [a["alerta"] for a in alertas_operador(op(negativa))]


def test_nome_limpo_so_perto_de_entrega():
    longe = "Pague o acordo e o seu nome fica limpo. " + "blá " * 60 + "e tem a entrega amigável."
    assert alertas_operador(op(longe)) == []
    assert [a["alerta"] for a in alertas_operador(op("Devolvendo o carro, o senhor já sai com o nome limpo."))] \
        == ["nome_limpo"]


def test_alerta_do_cliente_nao_conta():
    assert alertas_operador(cli("Vão bloquear minha conta?", "Vocês podem penhorar meus bens?")) == []


def test_um_alerta_por_tipo():
    r = alertas_operador(op("Consigo 30% de desconto.", "Repito: 30% de desconto só hoje."))
    assert [a["alerta"] for a in r] == ["desconto_percentual"] and r[0]["trecho"]


def test_mencao_e_alerta_dizem_onde_estao():
    """O painel destaca as palavras que dispararam a regra: posição e tamanho em palavras do turno original."""
    m = achar_mencoes("Olha, temos a entrega a migável hoje.")
    assert (m[0]["pos"], m[0]["n"]) == (3, 3)
    turnos = op("Bom dia.") + cli("Oi.") + op("Fechando hoje eu consigo 40% de desconto.",
                                              "A entrega amigável deixa o seu nome limpo.")
    al = {a["alerta"]: a for a in alertas_operador(turnos)}
    assert (al["desconto_percentual"]["turno"], al["desconto_percentual"]["pos"]) == (2, 4)
    assert al["desconto_percentual"]["n"] == 3
    assert (al["nome_limpo"]["turno"], al["nome_limpo"]["pos"], al["nome_limpo"]["n"]) == (3, 6, 2)


def test_grafias_descobertas_na_calibracao():
    """Grafias erradas que o Whisper produziu na rodada de calibração (outro ruído, outros sorteios)."""
    assert classificar_oferta(op("Temos a opção da entrega migada."))["oferta"] == "OFERECEU"
    assert classificar_oferta(op("Temos a entrega amigada para a senhora."))["oferta"] == "OFERECEU"
    r = classificar_oferta(op("Se sobrar, fica um saldo remanescendo."))
    assert r["oferta"] == "OFERECEU" and r["tipo"] == "devolucao"
    assert classificar_oferta(op("Se sobrar, fica um saldo remanescendo."), tolerante=False)["oferta"] == "NÃO OFERECEU"
    assert classificar_oferta(op("Uma amiga migada veio aqui."))["oferta"] == "NÃO OFERECEU"


def test_entrega_amigavel_para_a_senhora_e_oferta():
    assert classificar_oferta(op("Posso abrir a entrega amigável para a senhora hoje."))["oferta"] == "OFERECEU"
    assert classificar_oferta(op("A loja vai entregar o carro para o senhor amanhã."))["oferta"] == "NÃO OFERECEU"


# ---------------- correções da revisão final ----------------

@pytest.mark.parametrize("turnos", [cli("alô, alô"), op("") + cli("Posso fazer a entrega amigável?")])
def test_canal_do_operador_vazio_e_inconclusivo(turnos):
    """Se ninguém ouviu o operador (VAD cortou o canal), não dá para dizer que ele não ofertou."""
    assert classificar_oferta(turnos)["oferta"] == "INCONCLUSIVO"


@pytest.mark.parametrize("frase", ["Fique tranquilo, ninguém vai bloquear a sua conta.", "Não vamos penhorar os seus bens.",
                                   "A dívida não vai zerar com a entrega.", "A dívida não acaba com isso.",
                                   "O senhor não sai com o nome limpo com a entrega.", "Juros zero na dívida do acordo."])
def test_negacao_e_ressalva_nao_sao_alerta(frase):
    assert alertas_operador(op(frase)) == []


@pytest.mark.parametrize("frase", ["O senhor devolveu o carro para a loja?", "As lojas entregam o carro com nota.",
                                   "As concessionárias entregam o carro revisado.", "Eu já devolvi o carro uma vez."])
def test_conjugacao_e_loja_no_plural_nao_sao_oferta(frase):
    assert classificar_oferta(op(frase) + cli(frase))["oferta"] == "NÃO OFERECEU"


@pytest.mark.parametrize("frase,tipo", [("Podemos fazer a entrega amigável segunda-feira.", "explicita"),
                                        ("A entrega amigável, o documento sai depois.", "explicita"),
                                        ("Dá para entregar o seu carro e parar as parcelas.", "devolucao"),
                                        ("O senhor pode devolver o seu veículo.", "devolucao")])
def test_oferta_com_segunda_documento_ou_dois_determinantes(frase, tipo):
    r = classificar_oferta(op(frase))
    assert r["oferta"] == "OFERECEU" and r["tipo"] == tipo


def test_divida_fica_zero_e_quitacao():
    """"Zero" só conta depois de "dívida" ("a sua dívida fica zero"); "juros zero na dívida" não."""
    assert [a["alerta"] for a in alertas_operador(op("Fazendo isso, a sua dívida fica zero."))] == ["quitacao_sem_ressalva"]
