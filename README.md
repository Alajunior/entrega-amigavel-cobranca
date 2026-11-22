# Entrega Amigável: modelo de classe rara, painel e auditoria de ligações

**Painel ao vivo:** https://alajunior.github.io/entrega-amigavel-cobranca/ · dados 100% sintéticos · projeto de portfólio

No financiamento de veículo em atraso, a **Entrega Amigável (EA)** é quando o cliente devolve o carro de forma voluntária e a dívida é tratada sem busca e apreensão. É bom para os dois lados, e é raro: **menos de 1%** dos contratos em cobrança formaliza uma EA nos 60 dias seguintes. Este projeto mostra o que muda quando a classe positiva é tão rara:

- o modelo diz **quem** a operação deve ligar para ofertar a EA (classes ALTO, MEDIO e BAIXO);
- uma regra explícita diz **se** dá para ofertar agora, e o painel mostra o **porquê** de cada contrato;
- a qualidade é medida pelo que importa nesse caso: **precisão e recall no topo, PR-AUC e captura**. A AUC aparece, mas sozinha engana.

A fase 2 (em construção) audita as ligações desses clientes: ligações sintéticas com vozes neurais, transcrição com faster-whisper e regras que detectam se o operador ofertou a EA, medidas contra o gabarito.

![Painel, aba Operação](docs/painel.png)

## Como usar (2 minutos)

1. **Aba Operação.** Abre no mês vigente contra o mês anterior.
   - **KPIs:** base ativa, valor em risco, CPC (contato com a pessoa certa), indicações, formalizadas, **conversão** (indicações ÷ CPC) e **efetividade** (formalizadas ÷ indicações).
   - **Filtros globais:** fase de atraso, canal (discador ou digital) e "sucesso atribuído por". Pela data da formalização, a contagem é de caixa. Pela data da indicação, a formalização volta para a coorte da indicação e aparece "em maturação" enquanto a janela não fecha.
2. **Comparação dinâmica.** Três visões: por classe, com recorte por faixa de atraso; funil operacional (Contratos → Trabalhado → Discagem → ALÔ → CPC → Indicação → Formalizada); e por fase.
3. **Fila de ação.** Os contratos ALTO de hoje, cada um com:
   - **APTO:** SIM, A PARTIR DE dd/mm, EM ANDAMENTO, VERIFICAR ou NÃO;
   - o próximo passo;
   - os **2 motivos** que mais empurraram a chance para cima, por exemplo "deve mais do que o carro vale" ou "contato certo recente".
4. **Aba Modelo.** Curva precisão × recall, captura acumulada com os cortes, escada de candidatos com gate, teste de vazamento, backtest de produção, ordenação semanal, cortes por fase com vigência e linha do tempo.

Cada card tem um botão **ⓘ** com o "como ler". Há três temas (Midnight, Dark e Light). O painel funciona no celular.

## O problema e o alvo

- **Foto do dia D:** os contratos ativos em D, com as variáveis calculadas só com o que se sabia **antes** de D. Contato do próprio dia não entra, e acordo só conta como "quebrado" 30 dias depois de feito.
- **Alvo `ea_60d`:** 1 se a primeira formalização de EA cai em **(D, D+60]**. A janela fixa corrige um problema do projeto real, que usava janela aberta e deixava as safras recentes com rótulo cortado.
- **Foto madura:** D + 60 ≤ hoje (30/11/2026). Antes disso o alvo é desconhecido (NaN), nunca zero.
- **Taxa base:** 0,55% nas safras maduras, subindo de 0,49% para 0,73% com a mudança de perfil da carteira em jun/2026.

## A carteira sintética

`dados/gerar_carteira.py` gera 120 mil contratos de financiamento de veículos entre jan/2025 e nov/2026 (seed 42, ~10 s). Isso dá ~28 mil contratos em estoque por dia útil, ~580 CPC, ~6,5 indicações e ~3,8 formalizações por dia útil.

- O contrato entra na carteira com 31 dias de atraso. Ele sai por pagamento, busca e apreensão, EA formalizada ou ao passar de 720 dias.
- **Simulação semanal com riscos concorrentes.** O risco latente de EA (não salvo) sobe com:
  - atraso;
  - **dívida ÷ valor atual do carro** (mediana 0,98, p90 1,42);
  - pouco do plano pago e bureau ruim;
  - CPC recente e acordo quebrado.

  Ele também tem interações (por exemplo, deve mais do que o carro vale **e** o carro é velho) e ruído individual.
- A decisão vira **indicação** no primeiro CPC seguinte. A indicação é formalizada (5 a 50 dias depois) ou termina em insucesso com motivo; nesse caso pode haver nova indicação depois.
- **Mudança de regime em 01/06/2026:** entram contratos com bureau pior, o estoque envelhece e o efeito de "dever mais do que o carro vale" fica mais forte.

## Resultados

Os números abaixo saem de `relatorios/` e do painel, gerados com a base completa.

### 1. Escada de candidatos: a logística vence, e o LightGBM não ganha o suficiente para publicar

A pilha fora do tempo (OOT) são as 6 últimas safras maduras: 168 mil fotos e 1.063 entregas (taxa de 0,63%). O treino usa só as safras cujo alvo fechou antes da 1ª safra OOT.

| | PR-AUC | AUC | Precisão no top 10% | Recall no top 10% | Lift |
|---|---|---|---|---|---|
| Régua (dívida ÷ carro, depois maior atraso) | 0,018 | 0,772 | 2,0% | 31% | 3,1× |
| **Logística (publicada)** | **0,040** | **0,837** | **3,2%** | **51%** | **5,1×** |
| LightGBM | 0,041 | 0,839 | 3,3% | 52% | 5,2× |

O **gate** compara cada desafiante com o campeão na captura de entregas nos 10% de maior score. O intervalo de confiança de 95% vem de um bootstrap por **contrato** (1.000 reamostras) e precisa ter limite inferior acima de zero, sem piorar a PR-AUC.

- **Logística contra régua:** **+19,9 pp** (IC +15,9 a +24,0). Passou.
- **LightGBM contra logística:** +0,8 pp (IC −1,8 a +3,5). Não passou, então a logística publica e não há modelo em sombra.

O painel mostraria o duelo publicado × sombra se o LightGBM tivesse passado.

![Painel, aba Modelo](docs/painel_modelo.png)

### 2. Teste de vazamento: a validação aleatória mente para o modelo flexível

O mesmo contrato aparece em várias safras. Numa validação cruzada **aleatória**, fotos do mesmo contrato caem no treino e no teste, e o modelo "reconhece" o contrato. A validação **agrupada por contrato** (GroupKFold) mantém todas as fotos de um contrato do mesmo lado.

| Modelo | AUC aleatória | AUC por contrato | Diferença | PR-AUC aleatória | PR-AUC por contrato |
|---|---|---|---|---|---|
| Logística | 0,844 | 0,842 | +0,001 | 0,037 | 0,038 |
| LightGBM | **0,874** | **0,839** | **+0,035** | **0,052** | **0,037** |

Na validação aleatória, o LightGBM parecia bem melhor que a logística. Agrupado por contrato, ele empata. O projeto real fazia a validação cruzada sem agrupar; aqui isso vira um card do painel.

### 3. Cortes por captura, fora da amostra, por fase e com vigência

- **ALTO** = score a partir do qual estão **80%** das entregas; **MEDIO** = 95%. O critério é a captura, não um limiar de 0,5.
- **Os cortes saem de 3 safras que o modelo não viu no treino.** No projeto real, eles saíam das predições do próprio treino, e o ALTO acabava menor do que o previsto.
- **Cada fase tem o próprio corte.** Fase com menos de 30 entregas na calibração usa o global. Na vigência de 02/11/2026, a fase 0031–0090 tinha 11 e usou o corte global.
- **Cada retreino mensal acrescenta uma vigência** (12 no período). Reescorar um dia antigo usa a regra daquele dia.

### 4. Backtest de produção: a classe publicada se comportou

Compara a classe publicada em cada dia útil maduro com a formalização real nos 60 dias seguintes:

| Classe | % do estoque | Taxa de entrega em 60 dias | Lift | % das entregas |
|---|---|---|---|---|
| ALTO | 27,7% | 1,63% | 2,80× | **77,5%** |
| MEDIO | 28,3% | 0,32% | 0,55× | 15,7% |
| BAIXO | 44,0% | 0,09% | 0,16× | 6,9% |

Nas 44 semanas maduras, **nenhuma inversão**: em todas, a ordem foi ALTO > MEDIO > BAIXO. O ALTO capturou 77,5% das entregas, perto dos 80% previstos na calibração.

### 5. Linha do tempo para a diretoria

![Linha do tempo](docs/painel_tempo.png)

Ligando para **10% do estoque** por dia, o modelo alcança **51%** das entregas, contra **31%** da régua sem modelo (média dos meses maduros). Depois da mudança de perfil de jun/2026, o retreino mensal ganhou **+1,1 pp** do modelo congelado. A logística é estável, e o retreino é barato. Os títulos dessas conclusões são calculados a partir dos números, não escritos à mão.

Como o dado é sintético, a entrega não depende de quem foi acionado. Os simuladores medem a **qualidade da ordem**, não o efeito causal da ligação, e o painel avisa isso.

### 6. Por que o modelo decide e quem formaliza

Na logística, a contribuição de cada variável é exata, sem SHAP: coeficiente × (valor − média do treino), em log-odds. Fora do tempo, as variáveis que mais movem a chance são:

1. fase de cobrança;
2. **dívida ÷ valor do carro**;
3. parte do plano já paga;
4. contato certo em 30 dias;
5. bureau.

Na mediana, quem formaliza deve 1,30× o valor do carro (a base deve 0,97×), pagou 17% do plano (contra 29%), está com 326 dias de atraso (contra 216) e teve contato certo há 17 dias (contra 29).

### 7. Regra APTO_EA: o score diz quem, a regra diz se pode ofertar agora

As regras são aplicadas nesta ordem, olhando só o que se sabia antes do dia:

1. **EM ANDAMENTO:** indicação aberta.
2. **VERIFICAR:** insucesso de EA nos últimos 60 dias.
3. **NÃO:** acordo feito nos últimos 30 dias, ou seja, o cliente está pagando.
4. **A PARTIR DE dd/mm:** ainda não tem 90 dias de atraso.
5. **SIM:** nenhuma das anteriores.

Nos 200 ALTO de hoje: 148 SIM, 33 NÃO, 14 EM ANDAMENTO e 5 VERIFICAR.

### 8. Travas de produção: melhor não publicar do que publicar errado

`src/producao.py` escora o dia e verifica as travas na ordem abaixo. A primeira que falha encerra a escoragem com o código dela:

| Código | Trava |
|---|---|
| 7 | Dia anterior ao último publicado (nunca regrava o passado). |
| 5 | Modelo em disco diferente do sha256 do manifest, ou manifest ausente. |
| 3 | A foto do estoque não chegou. |
| 4 | Volume abaixo de 85% do último dia publicado. |
| 6 | Participação de ALTO fora de 0,5× a 1,5× do previsto na calibração. |

Se todas passam, o código é **0** e a publicação é atômica: arquivo temporário seguido de `os.replace`. O CSV publicado leva a classe, o APTO e o próximo passo.

## Como rodar

```bash
pip install -r requirements.txt
python dados/gerar_carteira.py      # carteira sintética de 120 mil contratos (~10 s)
python -m src.treinar               # escada com gate, teste de vazamento e importância em relatorios/ (~1 min)
python -m src.montar_painel         # escoragem diária em 3 modos, agregados e docs/index.html (~3 min; cache em dados/escoragem/)
python -m pytest                    # 63 testes
```

Para ver o painel, sirva a pasta `docs/` (por exemplo `python -m http.server --directory docs`). A cada push na `main`, a CI gera uma base pequena, roda o treino e os testes, e publica `docs/` no GitHub Pages.

## Estrutura

```
dados/gerar_carteira.py      gerador sintético (financiamento de veículos, funil de EA, mudança de regime)
src/config.py                datas, janela do alvo, fases, faixas de atraso, classes
src/base_analitica.py        foto as-of do dia D e alvo de 60 dias sem futuro
src/modelos.py               régua, logística e LightGBM no mesmo Pipeline
src/metricas.py              PR-AUC, precisão e recall no topo, lift
src/candidatos.py            escada com gate e IC bootstrap por contrato
src/validacao.py             teste de vazamento: aleatório × GroupKFold por contrato
src/cortes.py                cortes ALTO/MEDIO por captura, por fase, com vigência
src/escoragem.py             modelo vigente (retreino mensal), congelado e sombra
src/sinal.py                 contribuição exata da logística e motivos direcionais
src/apto.py                  regra APTO_EA e próximo passo
src/producao.py              escoragem diária com travas e publicação atômica
src/qualidade.py             backtest de produção, ordenação semanal, estabilidade
src/linha_tempo.py           régua × congelado × retreino, conclusões calculadas
src/treinar.py               orquestra a escada, o vazamento e a importância
src/montar_painel.py         monta o payload (atribuição as-of) e grava o HTML
src/painel.html              o painel (HTML, CSS e JS puro; Chart.js; fundo em canvas)
docs/                        o painel publicado e as imagens deste README
relatorios/                  escada.json, vazamento.json, importancia.csv
tests/                       63 testes: futuro, alvo, gate, cortes, vigência, travas, APTO, as-of, payload, guarda de publicação
```

## Fase 2 (em construção): auditoria de ligações

- 60 ligações sintéticas com clientes ALTO, geradas a partir de roteiros com **gabarito**: ofertou ou não, tipo de oferta e frases de alerta do operador.
- Vozes neurais com o operador num canal e o cliente no outro, e simulação de telefone (8 kHz e ruído).
- Transcrição com faster-whisper large-v3-turbo.
- Regras tolerantes a erro de transcrição, medidas contra o gabarito: matriz de confusão, precisão e recall por alerta, e ablação.
- Uma aba nova no painel, com o player do áudio e a transcrição sincronizada.

Nenhum áudio ou texto real é usado.
