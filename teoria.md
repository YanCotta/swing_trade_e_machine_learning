

# **Fundamentação Teórica: Ondas de Elliott e Machine Learning**

## **Introdução**

A busca pela previsão dos movimentos do mercado financeiro é um desafio central e perene que atrai tanto acadêmicos quanto operadores de mercado. Esta empreitada não é um problema resolvido, mas sim um campo de pesquisa ativa e intensa, onde teorias clássicas e tecnologias de ponta se encontram. Dentro deste cenário, duas estruturas de análise se destacam: a Teoria das Ondas de Elliott (TOE), um modelo clássico, e o Machine Learning (ML), um conjunto de ferramentas computacionais modernas.

A Teoria das Ondas de Elliott, desenvolvida na década de 1930 por Ralph Nelson Elliott, propõe uma ideia cativante: os preços de mercado não se movem de forma caótica, mas sim em padrões reconhecíveis e repetitivos. Estes padrões, segundo Elliott, são uma manifestação direta do ritmo natural da psicologia das massas, refletindo os ciclos de otimismo e pessimismo dos investidores.1 Por outro lado, o Machine Learning emergiu como um poderoso conjunto de técnicas computacionais para reconhecimento de padrões e previsão. O argumento central deste relatório é que o ML pode introduzir objetividade, testabilidade e escalabilidade à arte inerentemente subjetiva da análise de Ondas de Elliott, transformando uma teoria qualitativa em uma estratégia quantitativa.4

Este relatório fornecerá uma exploração abrangente de como as técnicas modernas de machine learning, especificamente a classificação supervisionada e o clustering temporal não supervisionado, podem ser aplicadas à detecção algorítmica dos padrões de Ondas de Elliott. O objetivo é demonstrar como essa fusão pode transformar uma teoria psicológica em uma estratégia de negociação sistemática e baseada em dados, equipando o leitor com o conhecimento teórico e prático necessário para abordar este fascinante domínio.

## **Seção 1: A Teoria das Ondas de Elliott: Decodificando o Ritmo do Mercado**

Para aplicar qualquer método computacional à Teoria das Ondas de Elliott, é imperativo primeiro compreender profundamente seus fundamentos teóricos. A teoria não é primariamente um conjunto de regras de negociação, mas sim um modelo descritivo do comportamento social. Esta seção estabelece a base teórica completa da TOE, enquadrando-a como uma teoria da psicologia coletiva antes de mergulhar em suas regras técnicas.

### **1.1 O Coração da Teoria: A Psicologia das Massas em Ondas**

A descoberta central de R.N. Elliott foi que os mercados não são aleatórios, mas sim impulsionados pela natureza repetitiva e cíclica das emoções humanas, que oscilam entre o pessimismo e o otimismo em sequências previsíveis.2 Ele postulou que os gráficos de preços são, na essência, um registro visual dessa psicologia de massa. Esta premissa fundamental é o que distingue a TOE de métodos puramente estatísticos ou técnicos que não consideram o fator humano como a principal força motriz.1

A TOE compartilha raízes com a Teoria de Dow, pois ambas reconhecem que os preços são uma função tanto da razão (fundamentos econômicos) quanto da emoção (medo e ganância). No entanto, a contribuição única de Elliott foi dar uma ênfase especial ao componente emocional, argumentando que ele se manifesta no gráfico através de uma geometria fractal específica e repetível.1 Em outras palavras, Elliott propôs que os ciclos de mercado eram uma resposta às reações dos investidores a fatores externos, levando o mercado da euforia ao pânico em padrões estruturados.1

### **1.2 O Padrão Fundamental: A Estrutura 5-3**

A base da teoria de Elliott é um padrão fundamental composto por oito ondas, que formam um ciclo completo. Este ciclo é dividido em duas fases distintas: a fase de impulso e a fase de correção.

* **Ondas de Impulso (ou Ondas Motrizes):** Representam a tendência principal do mercado. Este padrão é composto por cinco ondas, rotuladas de 1 a 5\. As ondas 1, 3 e 5 movem-se na direção da tendência principal, enquanto as ondas 2 e 4 são correções menores contra essa tendência.2  
* **Ondas Corretivas:** Representam uma correção contra a tendência principal. Este padrão é composto por três ondas, rotuladas A, B e C. Elas seguem a conclusão do padrão de cinco ondas de impulso e movem-se na direção oposta à tendência anterior.10

Cada uma dessas ondas possui uma personalidade distinta, que reflete a psicologia coletiva do mercado em cada etapa do ciclo:

* **Onda 1:** É o movimento inicial, muitas vezes surgindo de uma base de pessimismo generalizado, quando as notícias ainda são negativas. Apenas os investidores mais astutos e bem informados tendem a participar, reconhecendo uma mudança fundamental. Esta onda é frequentemente difícil de identificar em tempo real.1  
* **Onda 2:** É uma onda corretiva que apaga uma parte dos ganhos da Onda 1\. O sentimento de baixa retorna, e muitos participantes do mercado concluem que o movimento inicial foi apenas um repique passageiro. Crucialmente, a Onda 2 nunca retrocede abaixo do início da Onda 1\.11  
* **Onda 3:** Esta é frequentemente a onda mais longa, mais forte e mais poderosa da sequência. O mercado finalmente reconhece a nova tendência, e o otimismo começa a se espalhar. A participação do público aumenta, e as notícias começam a se tornar positivas. A Onda 3 é o coração da tendência.11  
* **Onda 4:** Uma onda corretiva que se segue à forte alta da Onda 3\. Seu caráter é frequentemente complexo e frustrante, com movimentos laterais que testam a paciência dos investidores. O otimismo da Onda 3 diminui, mas a tendência subjacente ainda é forte.11  
* **Onda 5:** É o último impulso na direção da tendência principal. A euforia é generalizada, e a participação do público atinge seu pico. Este movimento é frequentemente impulsionado mais pela emoção do que por fundamentos sólidos, e é comum observar divergências em indicadores de momentum, sinalizando que a força da tendência está diminuindo.11  
* **Ondas A, B e C:** Após o pico da Onda 5, a fase corretiva começa. A Onda A é a queda inicial. A Onda B é um repique de alta, muitas vezes chamado de "armadilha para touros" (bull trap), que engana muitos a acreditarem que a tendência de alta está sendo retomada. Finalmente, a Onda C é a perna final e mais forte da correção, quebrando as esperanças dos otimistas e confirmando a mudança de tendência.11

### **1.3 As Regras de Engajamento: Princípios Fundamentais e Natureza Fractal**

Para que uma contagem de ondas seja considerada válida, ela deve aderir a três regras invioláveis. Estas regras são a espinha dorsal da aplicação da teoria e são essenciais para qualquer tentativa de implementação algorítmica.9

1. **Regra 1:** A Onda 2 nunca pode retroceder mais de 100% da Onda 1\. Se o preço cair abaixo do ponto de partida da Onda 1, a contagem está incorreta.  
2. **Regra 2:** A Onda 3 nunca pode ser a mais curta entre as três ondas de impulso (Ondas 1, 3 e 5). Frequentemente, ela é a mais longa, mas nunca pode ser a mais curta.  
3. **Regra 3:** A Onda 4 nunca deve entrar no território de preço da Onda 1\. O fundo da Onda 4 não pode se sobrepor ao topo da Onda 1\. (Uma exceção rara ocorre em um padrão chamado triângulo diagonal).

Além dessas regras, um conceito central da teoria é sua **natureza fractal** ou de "auto-similaridade".1 Isso significa que a estrutura 5-3 é universal em todas as escalas de tempo. Um padrão de impulso de cinco ondas em um gráfico anual é, ele próprio, composto por padrões 5-3 menores em um gráfico mensal. A Onda 1 desse padrão anual é, por si só, um padrão de cinco ondas em uma escala de tempo menor. Essa propriedade fractal é o que permite que a teoria seja aplicada tanto para investimentos de longo prazo quanto para negociações de curto prazo, como o swing trade.

### **1.4 Adicionando Rigor Matemático: A Relação com Fibonacci**

Embora os padrões de Elliott sejam qualitativos em sua descrição da psicologia do mercado, suas proporções são frequentemente governadas por relações matemáticas encontradas na sequência de Fibonacci e suas razões derivadas (

0.382  
,

0.618  
,

1.0  
,

1.618  
,

2.618  
, etc.).1 Essa conexão fornece uma base quantitativa para projetar alvos de preço para as ondas e identificar zonas de alta probabilidade para reversões (suporte e resistência), tornando a teoria testável e aplicável algoritmicamente.15

Por exemplo, é comum que a Onda 2 corrija

$$61.8\\%$$do comprimento da Onda 1\. Um alvo de preço comum para a Onda 3 é uma extensão de$$161.8\\%$$  
do comprimento da Onda 1\.11 O uso dessas razões é fundamental para transformar a teoria em uma estratégia de negociação com regras definidas. A tabela a seguir codifica essas relações, servindo como um guia direto para a engenharia de características e a criação de sistemas baseados em regras.

| Onda/Correção | Tipo de Relação | Razões de Fibonacci Comuns | Contexto e Interpretação |  |  |  |  |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Onda 2** | Retração da Onda 1 | 50% | , | 61.8% | , | 78.6% | Uma retração profunda é comum, pois o otimismo inicial desaparece. É considerado um ponto de entrada chave para a tendência principal.11 |  |  |
| **Onda 3** | Extensão da Onda 1 | 161.8% | , | 261.8% | , | 423.6% | A onda mais forte. Uma Onda 2 superficial (ex: | 38.2% | ) frequentemente leva a uma Onda 3 forte e estendida.11 |
| **Onda 4** | Retração da Onda 3 | 23.6% | , | 38.2% | Geralmente uma retração superficial, refletindo a força da tendência subjacente. Não deve se sobrepor à Onda 1\.15 |  |  |  |  |
| **Onda 5** | Extensão/Igualdade | 61.8% | , | 100% | , | 161.8% | da Onda 1 | Pode ter o mesmo tamanho da Onda 1 ou ser uma extensão. Divergências em indicadores são comuns aqui. |  |
| **Onda A** | Correção | Comprimento semelhante à Onda 5 | O primeiro sinal de uma mudança de tendência. |  |  |  |  |  |  |
| **Onda B** | Retração da Onda A | 38.2% | , | 50% | , | 61.8% | A "armadilha de touros" em uma tendência de baixa. Engana os participantes do mercado. |  |  |
| **Onda C** | Extensão da Onda A | 100% | , | 161.8% | O movimento corretivo final e decisivo. |  |  |  |  |

### **1.5 Uma Visão Equilibrada: Subjetividade Inerente e Críticas**

Apesar de sua estrutura elegante, a principal crítica à Teoria das Ondas de Elliott é sua subjetividade inerente. Diferentes analistas, olhando para o mesmo gráfico, podem chegar a contagens de ondas completamente diferentes, resultando em previsões conflitantes e tornando a teoria difícil de ser refutada cientificamente.7 Essa flexibilidade na interpretação é tanto uma força (permitindo adaptação) quanto sua maior fraqueza (falta de rigor).

Estudos acadêmicos também lançaram dúvidas sobre seu poder preditivo. Uma tese de mestrado da UFSM, por exemplo, concluiu que a analogia entre o comportamento das massas e a sequência de Fibonacci se mostrou incapaz de prever as oscilações do mercado durante uma crise econômica.17 Esta crítica é crucial, pois destaca o desafio que um projeto de ML nesta área busca resolver.

É exatamente nesta contradição central que reside a oportunidade para a aplicação de machine learning. A TOE é um modelo psicológico poderoso, mas sua aplicação prática é, em grande parte, uma forma de arte. O objetivo central da aplicação de ML é tentar transformar essa arte em uma ciência. Ao criar regras objetivas baseadas nos princípios e nas relações de Fibonacci, e ao testá-las sistematicamente em grandes volumes de dados, é possível validar ou refutar a utilidade prática da teoria de uma maneira que a análise manual nunca poderia. Este projeto, portanto, não é apenas sobre construir um robô de negociação, mas sobre conduzir uma investigação científica sobre a validade de uma teoria financeira fundamental.

## **Seção 2: A Abordagem Supervisionada: Ensinando Máquinas a Reconhecer Ondas**

A abordagem supervisionada de machine learning oferece um caminho direto para automatizar o reconhecimento de padrões de Elliott. A ideia é "ensinar" um algoritmo a identificar ondas da mesma forma que um analista humano faria, mas com a velocidade, escala e objetividade de uma máquina. Esta seção detalha como o problema de reconhecimento de padrões da TOE pode ser enquadrado como uma tarefa de aprendizado supervisionado e revisa os métodos e descobertas da literatura acadêmica.

### **2.1 Formulação do Problema: Do Gráfico à Classificação**

O primeiro passo para aplicar o aprendizado supervisionado é transformar o problema de análise gráfica em um problema de classificação matemática. O processo funciona da seguinte forma: um segmento de dados históricos de preços (uma "janela" de tempo, por exemplo, 60 dias de dados de fechamento) torna-se a entrada do modelo, representada por um vetor de características X. O rótulo "correto" da Onda de Elliott para essa janela (por exemplo, "Onda 3", "Onda C", "Não é uma onda de Elliott") torna-se a saída alvo, ou a classe, Y.12

O objetivo, então, é treinar um modelo de classificação, uma função f, que aprenda a mapear as entradas para as saídas, de modo que f(X)≈Y. Essencialmente, estamos tratando o reconhecimento de ondas como um problema de classificação multicasse, onde cada tipo de onda é uma classe distinta.

### **2.2 O Pipeline Metodológico**

A construção de um modelo supervisionado eficaz segue um pipeline bem definido, mas cada etapa apresenta desafios únicos no contexto da TOE.

#### **2.2.1 Rotulagem de Dados: O Calcanhar de Aquiles**

A etapa mais crítica e, sem dúvida, a mais difícil na abordagem supervisionada é a criação do conjunto de dados rotulado. Isso exige que um ou mais analistas especialistas em Ondas de Elliott analisem manualmente vastas quantidades de dados históricos de preços e atribuam um rótulo de onda a cada segmento de tempo relevante.18

Aqui reside uma implicação profunda da subjetividade da TOE. A qualidade e a consistência do modelo de machine learning são fundamentalmente limitadas pela qualidade e consistência do especialista humano que rotula os dados. Se o especialista for inconsistente, tendencioso ou simplesmente cometer erros, o modelo aprenderá essas falhas perfeitamente. Isso cria um potencial problema de lógica circular: o modelo é treinado para replicar a subjetividade de um humano, em vez de descobrir uma verdade objetiva no mercado. Este é o clássico problema de "lixo entra, lixo sai" (Garbage In, Garbage Out), e representa a maior barreira para a criação de um modelo supervisionado verdadeiramente robusto e confiável para a TOE.9

#### **2.2.2 Engenharia de Características: Tornando as Ondas Legíveis para Máquinas**

Alimentar o modelo com dados brutos de preços raramente é a abordagem mais eficaz. Em vez disso, é necessário realizar a engenharia de características (feature engineering), que consiste em criar variáveis de entrada que capturem as *características* distintivas das ondas.

* **Características Básicas:** Métricas estatísticas simples (média, desvio padrão, etc.) e indicadores técnicos populares como o MACD (Moving Average Convergence Divergence), o Estocástico e o RSI (Relative Strength Index) podem ser usados como características. Esses indicadores já são projetados para capturar momentum, sobrecompra/sobrevenda e tendências, que são conceitos alinhados com a psicologia das ondas.18  
* **Características Avançadas (Transformada Rápida de Fourier \- FFT):** Uma técnica mais sofisticada é o uso da Transformada Rápida de Fourier (FFT). A FFT converte uma série temporal do domínio do tempo para o domínio da frequência. Os coeficientes da FFT resultantes atuam como uma "impressão digital" da forma geral do padrão de onda. Padrões de impulso rápidos podem ter componentes de alta frequência diferentes de padrões de correção lentos e laterais. Esses coeficientes podem então ser usados como um vetor de características de entrada para um modelo, como uma Rede Neural, que pode aprender a associar essas "impressões digitais" de frequência aos rótulos de onda correspondentes.21

### **2.3 Algoritmos na Prática: SVMs e Redes Neurais**

Dois tipos de algoritmos de aprendizado supervisionado são comumente citados na literatura para esta tarefa: Máquinas de Vetores de Suporte (SVM) e Redes Neurais (NN).

* **Máquinas de Vetores de Suporte (SVMs):** O conceito central de uma SVM é encontrar um hiperplano ótimo que melhor separe as diferentes classes (os padrões de onda) em um espaço de características de alta dimensão.22 As SVMs são particularmente poderosas porque se baseiam em uma sólida teoria estatística (Minimização do Risco Estrutural), o que as torna menos propensas a overfitting em comparação com outros métodos, especialmente quando o número de características é grande e o conjunto de dados de treinamento não é massivo.23 Vários estudos relatam sucesso na aplicação de SVMs para a previsão de tendências baseada em padrões de Elliott, com alguns artigos reivindicando taxas de precisão superiores a 90%.4  
* **Redes Neurais (NNs) e Deep Learning:** As Redes Neurais Artificiais (RNAs) são modelos computacionais inspirados no cérebro humano, capazes de aprender relações complexas e não-lineares diretamente dos dados.12 Usando um algoritmo de treinamento como o backpropagation, elas ajustam seus pesos internos para minimizar o erro de previsão.12 Modelos mais avançados de Deep Learning, como Redes Neurais Convolucionais (CNNs) e Redes Neurais Recorrentes (como LSTMs), podem extrair características hierárquicas automaticamente dos dados, potencialmente reduzindo a necessidade de engenharia manual de características.5 A literatura mostra o uso de NNs, frequentemente em conjunto com a FFT, para reconhecer padrões de Elliott e prever a direção subsequente do mercado.12

### **2.4 Avaliando o Sucesso: Desempenho e Limitações Chave**

Uma síntese da literatura acadêmica revela um quadro misto. Muitos artigos de pesquisa publicados relatam taxas de sucesso impressionantemente altas, frequentemente acima de 70% e, em alguns casos, ultrapassando 90% na previsão da direção da tendência após o reconhecimento de um padrão de Elliott.4

No entanto, é crucial justapor esses resultados otimistas com as conclusões mais sóbrias de trabalhos como a tese da UFSM.17 Este estudo não apenas considerou a TOE ineficaz para prever movimentos durante uma crise, mas também destacou um problema comum em modelagem: redes neurais que apresentaram um desempenho superior no treinamento falharam na etapa mais crucial de validação em dados não vistos.

Essa contradição sugere que muitos dos modelos publicados podem ser *frágeis*. Eles podem estar superajustados (overfitted) a um conjunto de dados específico ou a um regime de mercado particular (por exemplo, um mercado de alta estável e com baixa volatilidade). Quando a estrutura do mercado muda — um fenômeno conhecido como *concept drift* 28 — os padrões aprendidos pelo modelo perdem sua validade, e o desempenho do sistema degrada-se catastroficamente. Isso ressalta a necessidade absoluta de técnicas de validação robustas que simulem o desempenho no mundo real. Em vez de uma simples divisão treino-teste, métodos como a

**Otimização Walk-Forward** 30 são essenciais. Nesta abordagem, o modelo é treinado em um período de dados passados, testado no período seguinte e, em seguida, o processo é repetido, deslizando a janela de treinamento para a frente. Isso garante que o modelo seja continuamente reavaliado e retreinado em dados novos, forçando-o a se adaptar às dinâmicas de mercado em constante mudança.

## **Seção 3: O Paradigma Não Supervisionado: Descobrindo Padrões a Partir dos Dados**

Enquanto a abordagem supervisionada tenta ensinar a máquina a replicar o conhecimento de um especialista, a abordagem não supervisionada adota uma filosofia fundamentalmente diferente. Ela busca descobrir padrões e estruturas intrínsecas nos dados sem qualquer conhecimento prévio ou rótulos. Esta seção apresenta o aprendizado não supervisionado, especificamente o clustering temporal, como uma alternativa mais exploratória e potencialmente mais robusta, que contorna as armadilhas da rotulagem manual.

### **3.1 Uma Nova Filosofia: Encontrando Estrutura sem Pré-Rotulagem**

O aprendizado não supervisionado é uma classe de algoritmos de ML que analisa dados de entrada não rotulados para encontrar padrões ocultos ou estruturas intrínsecas.31 A filosofia aqui é de descoberta, não de replicação. Em vez de dizer à máquina: "Estes são os padrões de Elliott, encontre-os" (abordagem supervisionada), a pergunta torna-se: "Quais padrões recorrentes existem nestes dados de preço?" (abordagem não supervisionada).

Este método oferece uma vantagem científica significativa: ele evita o viés do especialista humano introduzido na etapa de rotulagem. Ao permitir que os dados "falem por si mesmos", podemos descobrir os arquétipos de comportamento de mercado que realmente existem na série temporal, em vez de forçar os dados a se encaixarem em um modelo teórico pré-concebido. Isso torna a abordagem não supervisionada um ponto de partida mais objetivo e cientificamente sólido para a análise de padrões.

### **3.2 Técnica Fundamental: Clustering K-Means para Agrupamento de Padrões**

Uma das técnicas de aprendizado não supervisionado mais populares e fundamentais é o clustering, e o algoritmo K-Means é um dos seus principais representantes. O processo de clustering com K-Means pode ser aplicado a séries temporais financeiras da seguinte maneira 32:

1. **Preparação dos Dados:** A série temporal de preços é dividida em janelas de tempo sobrepostas de um comprimento fixo (por exemplo, janelas de 30 dias de dados OHLC). Cada uma dessas janelas é um "ponto de dados" a ser clusterizado.  
2. **Escolha de K:** O analista define o número de clusters, K, que deseja encontrar. Este é um hiperparâmetro importante.  
3. **Inicialização:** O algoritmo inicializa aleatoriamente K pontos no espaço de dados. Estes pontos são chamados de "centroides".  
4. **Atribuição:** Cada janela de tempo (ponto de dados) é atribuída ao centroide mais próximo, com base em uma métrica de distância.  
5. **Atualização:** Os centroides são recalculados como a média de todos os pontos de dados (janelas) atribuídos a eles. O novo centroide representa o "padrão médio" daquele cluster.  
6. **Iteração:** Os passos 4 e 5 são repetidos até que as atribuições dos clusters não mudem mais, significando que o algoritmo convergiu.

Ao final do processo, temos K grupos de janelas de tempo, onde cada grupo contém padrões de preço com formas semelhantes. Esta técnica tem sido usada para agrupar barras diárias de OHLC ou subsequências mais longas de preços para encontrar "padrões gráficos" recorrentes.36

### **3.3 O Desafio da Série Temporal: Por que a Distância Euclidiana Falha e o DTW Triunfa**

A aplicação direta do K-Means a séries temporais revela uma falha crítica. A métrica de distância padrão usada pela maioria das implementações de clustering é a **Distância Euclidiana**. Esta métrica mede a distância ponto a ponto entre duas séries nos mesmos instantes de tempo. O problema é que ela é extremamente sensível a pequenos desalinhamentos, deslocamentos ou distorções no eixo do tempo.

Imagine dois padrões de impulso de alta. Um se desenvolve ao longo de 20 dias, e o outro, uma versão ligeiramente mais rápida, se desenvolve em 18 dias. Embora visualmente idênticos em forma, a Distância Euclidiana os consideraria muito "distantes" um do outro, pois os picos e vales não se alinham perfeitamente no tempo. Isso levaria o algoritmo de clustering a colocá-los em grupos separados, falhando em reconhecer sua semelhança estrutural.

A solução para este problema é usar uma métrica de distância projetada especificamente para séries temporais: o **Dynamic Time Warping (DTW)**. O DTW é um algoritmo que encontra o alinhamento não-linear ótimo entre duas séries temporais. Ele "estica" ou "comprime" o eixo do tempo de uma série para que ela se alinhe da melhor forma possível com a outra, minimizando a distância entre elas.38 O DTW é, portanto, invariante a deslocamentos e distorções no tempo, permitindo que ele reconheça padrões com formas semelhantes, mesmo que ocorram em velocidades diferentes. Esta capacidade é fundamental para o "clustering temporal" e é a chave para uma implementação bem-sucedida.

| Característica | Distância Euclidiana | Dynamic Time Warping (DTW) |
| :---- | :---- | :---- |
| **Princípio Central** | Mede a distância ponto a ponto em instantes de tempo idênticos. | Encontra o alinhamento não-linear ótimo entre duas séries para minimizar a distância. |
| **Manuseio de Deslocamentos Temporais** | Ruim. Considera padrões deslocados no tempo como altamente dissimilares. | Excelente. Invariante a deslocamentos e escalonamento no eixo do tempo. |
| **Custo Computacional** | Baixo (tempo linear, O(n)). | Alto (tempo quadrático, O(n2)). Existem aproximações mais rápidas. |
| **Caso de Uso** | Adequado para vetores de características estáticos. | Essencial para comparar formas de séries temporais (ex: fala, ECG, padrões financeiros). |
| **Exemplo Financeiro** | Falharia em agrupar um padrão "Ombro-Cabeça-Ombro" que se forma em 20 dias com um que se forma em 25 dias. | Agruparia com sucesso os dois padrões "Ombro-Cabeça-Ombro" ao "deformar" o eixo do tempo para um alinhamento ideal. |

### **3.4 Dos Clusters à Análise: Interpretação e Insight**

Uma vez que o processo de clustering (usando K-Means com DTW) está completo, a etapa final é a análise e interpretação humana. O resultado do algoritmo são K clusters, cada um representado por seu centroide. O centroide de um cluster é a "média" de todas as janelas de tempo naquele grupo e, portanto, representa o padrão arquetípico ou a forma média daquele cluster.36

O trabalho do analista é examinar visualmente esses centroides e interpretá-los através da lente das teorias financeiras.

* Algum dos centroides se parece com uma Onda de Impulso clássica da TOE?  
* Outro se assemelha a uma correção lateral (flat)?  
* Existem clusters que correspondem a padrões gráficos clássicos, como "topos duplos" ou "fundos duplos"?40

Este processo permite a descoberta de regimes de mercado significativos, impulsionada pelos próprios dados. Uma vez que esses clusters de padrões são identificados e interpretados, eles podem ser usados para construir modelos preditivos ou regras de negociação. Por exemplo, pode-se analisar o comportamento médio do preço nos dias seguintes à ocorrência de um padrão de um determinado cluster para estimar probabilidades de movimentos futuros.

## **Seção 4: Síntese e Roteiro de Apresentação**

Após explorar as abordagens supervisionada e não supervisionada, esta seção sintetiza os resultados, compara os dois métodos e fornece um roteiro detalhado e prático para a apresentação do grupo, juntamente com uma recomendação estratégica para a implementação do projeto.

### **4.1 Uma Análise Comparativa dos Métodos**

As duas abordagens de machine learning oferecem caminhos distintos para a análise de padrões de Elliott, cada uma com seus próprios pontos fortes e fracos.

* **Abordagem Supervisionada:**  
  * **Prós:** Pode ser altamente precisa se existir um conjunto de dados rotulado de alta qualidade. O modelo resultante prevê diretamente uma categoria de onda específica (ex: "Onda 3"), o que é fácil de interpretar.  
  * **Contras:** Totalmente dependente de rótulos de especialistas, que são subjetivos, demorados para criar e potencialmente falhos. Os modelos são frágeis e propensos a falhas quando os regimes de mercado mudam (baixa robustez a *concept drift*).  
* **Abordagem Não Supervisionada:**  
  * **Prós:** É objetiva e orientada pelos dados, descobrindo padrões que realmente existem na série temporal e evitando o viés humano. Com o uso do DTW, é mais robusta a variações na forma e na duração dos padrões.  
  * **Contras:** Não produz diretamente rótulos preditivos; requer uma segunda etapa de interpretação humana para dar sentido aos clusters. É computacionalmente mais cara, especialmente com o DTW.

### **4.2 Roteiro para uma Apresentação de Impacto**

Este roteiro detalhado, slide a slide, foi projetado para ajudar o grupo a estruturar uma apresentação clara, lógica e convincente.

* **Slide 1: Título**  
  * Título do Projeto: "Estratégias Algorítmicas para Swing Trade: Detecção de Padrões de Elliott com Machine Learning"  
  * Nomes dos Integrantes do Grupo  
  * Disciplina / Professor  
* **Slide 2: Introdução e Definição do Problema**  
  * O Desafio: Prever os mercados financeiros é notoriamente difícil.  
  * A Teoria: A Teoria das Ondas de Elliott (TOE) oferece um modelo da psicologia do mercado.1  
  * O Problema com a Teoria: É altamente subjetiva e difícil de aplicar consistentemente.9  
  * Nosso Objetivo: Aplicar Machine Learning para trazer objetividade e testabilidade à TOE.  
* **Slide 3: O que são as Ondas de Elliott? O Padrão 5-3**  
  * Incluir um diagrama visual claro do padrão de 5 ondas de impulso e 3 ondas de correção.10  
  * Explicar brevemente a psicologia: Impulso \= a favor da tendência; Correção \= contra a tendência.  
* **Slide 4: As Regras do Jogo: Fractais e Fibonacci**  
  * Listar as 3 regras invioláveis (Onda 2 \> 100% da 1; Onda 3 nunca é a mais curta; Onda 4 não sobrepõe a 1).9  
  * Explicar o papel crucial das razões de Fibonacci para quantificar as relações entre as ondas e projetar alvos.15  
* **Slide 5: O Desafio Central: A Subjetividade**  
  * Mostrar um único gráfico de preços com duas contagens de ondas diferentes, mas plausíveis, lado a lado.  
  * Citar: "Essa é uma das maiores críticas à Teoria das Ondas de Elliott".9  
  * Este slide prepara o terreno para justificar a necessidade do Machine Learning.  
* **Slide 6: Nossa Abordagem: Uma Análise de Dois Métodos**  
  * Introduzir os dois paradigmas de ML explorados:  
    1. **Aprendizado Supervisionado:** Ensinar a máquina com exemplos de especialistas.  
    2. **Clustering Não Supervisionado:** Pedir à máquina para descobrir padrões por conta própria.  
* **Slide 7: Método 1: Aprendizado Supervisionado (O "Professor Especialista")**  
  * Explicar o processo: Rotular Dados → Engenharia de Características → Treinar Modelo (SVM/NN) → Prever.12  
  * Destacar o principal desafio: O problema do "Garbage In, Garbage Out" devido à subjetividade dos rótulos.  
* **Slide 8: Método 2: Clustering Não Supervisionado (O "Explorador de Dados")**  
  * Explicar o processo: Pegar janelas de preço não rotuladas → Agrupá-las por similaridade de forma → Analisar os padrões resultantes.36  
  * Enfatizar que esta abordagem evita o problema da rotulagem manual.  
* **Slide 9: Aprofundamento Técnico: Clustering com DTW**  
  * Mostrar uma animação ou diagrama simples comparando a Distância Euclidiana com o Dynamic Time Warping (DTW) (usar a Tabela 2 como referência).  
  * Explicar por que o DTW é essencial para encontrar padrões que são "esticados" ou "comprimidos" no tempo.38  
* **Slide 10: Nossa Implementação e Resultados (Seção para o grupo preencher)**  
  * Apresentar os clusters que o grupo descobriu usando K-Means com DTW.  
  * Mostrar o gráfico do centroide (padrão médio) para alguns dos clusters mais interessantes.  
  * Interpretar os resultados: "O Cluster 1 parece ser uma onda de impulso clássica", "O Cluster 4 se assemelha a um padrão de correção lateral (flat)".  
* **Slide 11: Discussão e Limitações**  
  * Recapitular os prós e contras de cada método.  
  * Discutir a fragilidade dos modelos e a necessidade de validação robusta (ex: teste walk-forward) para evitar overfitting.17  
  * Reconhecer que estas são ferramentas de análise, não "bolas de cristal".  
* **Slide 12: Conclusão e Trabalhos Futuros**  
  * O ML fornece uma estrutura poderosa para testar sistematicamente a TOE.  
  * O clustering não supervisionado é um ponto de partida robusto e objetivo para a descoberta de padrões.  
  * Trabalhos Futuros: Construir modelos preditivos baseados nos clusters descobertos; testar em mais ativos e condições de mercado; explorar modelos híbridos.  
* **Slide 13: Perguntas e Respostas**

### **4.3 Um Caminho Prático a Seguir: Estratégia de Implementação**

Para um projeto acadêmico, a recomendação estratégica é focar na **abordagem de clustering não supervisionado (K-Means com DTW)**. A justificativa para esta recomendação é pragmática e cientificamente sólida. A abordagem supervisionada, embora conceitualmente direta, depende de um passo metodologicamente falho e extremamente trabalhoso: a criação de um grande conjunto de dados rotulado manualmente. Este passo não é apenas um gargalo de tempo, mas também introduz um viés fundamental que compromete a validade dos resultados.

Em contraste, a abordagem não supervisionada é mais viável. Ela não requer rotulagem prévia e produz resultados tangíveis e orientados pelos dados (os clusters e seus centroides) que podem ser analisados e apresentados de forma eficaz. Permite que o grupo realize uma investigação científica genuína sobre os padrões que existem nos dados, em vez de apenas tentar replicar a análise subjetiva de um especialista.

## **Conclusão**

A fusão da Teoria das Ondas de Elliott, um modelo clássico da psicologia de mercado, com as técnicas modernas de ciência de dados representa uma fronteira fascinante na análise financeira. Este relatório demonstrou como as abordagens de machine learning, tanto supervisionadas quanto não supervisionadas, podem ser aplicadas para trazer rigor quantitativo e escalabilidade a uma teoria inerentemente qualitativa.

A abordagem supervisionada oferece a promessa de classificação direta de ondas, mas é limitada pela subjetividade da rotulagem humana e pela fragilidade dos modelos resultantes. A abordagem não supervisionada, particularmente o clustering temporal com Dynamic Time Warping, surge como um método mais robusto e cientificamente sólido, permitindo a descoberta de padrões arquetípicos diretamente dos dados, livre de vieses pré-concebidos.

É crucial entender que o objetivo final dessas ferramentas algorítmicas não é criar uma "máquina de dinheiro" totalmente autônoma, mas sim aumentar a análise humana. Elas fornecem os meios para escanear milhares de ativos em busca de padrões de alta probabilidade, filtrar vieses cognitivos e testar hipóteses de negociação em escala. Em última análise, elas capacitam o trader e o analista com insights orientados por dados, transformando a arte da análise de gráficos em uma disciplina mais próxima da ciência.

#### **Works cited**

1. Ondas de Elliott: conheça a teoria \- Fast Trade, accessed on June 23, 2025, [https://plataformafasttrade.com.br/blog/como-identificar-as-ondas-de-elliott-e-entender-a-teoria/](https://plataformafasttrade.com.br/blog/como-identificar-as-ondas-de-elliott-e-entender-a-teoria/)  
2. Ondas de Elliott: de forma fácil e prática \- Nelogica Análise Técnica, accessed on June 23, 2025, [https://blog.nelogica.com.br/ondas-de-elliott-de-forma-facil-e-pratica/](https://blog.nelogica.com.br/ondas-de-elliott-de-forma-facil-e-pratica/)  
3. Compreendendo a Teoria das Ondas de Elliott: guia dos ciclos de mercado \- Earn2Trade, accessed on June 23, 2025, [https://www.earn2trade.com/blog/pt/principio-da-onda-de-elliott/](https://www.earn2trade.com/blog/pt/principio-da-onda-de-elliott/)  
4. An algorithm for Elliott Waves pattern detection | Request PDF \- ResearchGate, accessed on June 23, 2025, [https://www.researchgate.net/publication/322460091\_An\_algorithm\_for\_Elliott\_Waves\_pattern\_detection](https://www.researchgate.net/publication/322460091_An_algorithm_for_Elliott_Waves_pattern_detection)  
5. An Improved Deep-Learning-Based Financial Market Forecasting Model in the Digital Economy \- MDPI, accessed on June 23, 2025, [https://www.mdpi.com/2227-7390/11/6/1466](https://www.mdpi.com/2227-7390/11/6/1466)  
6. Recognition of Patterns With Fractal Structure in Time Series \- IGI Global, accessed on June 23, 2025, [https://www.igi-global.com/chapter/recognition-of-patterns-with-fractal-structure-in-time-series/196961](https://www.igi-global.com/chapter/recognition-of-patterns-with-fractal-structure-in-time-series/196961)  
7. Ondas de Elliot: entendendo os padrões de mercado \- Nord Investimentos, accessed on June 23, 2025, [https://www.nordinvestimentos.com.br/blog/ondas-de-elliot-entendendo-os-padroes-de-mercado-atraves-da-teoria-de-ondas/](https://www.nordinvestimentos.com.br/blog/ondas-de-elliot-entendendo-os-padroes-de-mercado-atraves-da-teoria-de-ondas/)  
8. Como a Teoria das Ondas de Elliott Pode Ajudar na Analise dos Mercados \- Hantec Markets, accessed on June 23, 2025, [https://hmarkets.com/pt/blog/teoria-das-ondas-de-elliott/](https://hmarkets.com/pt/blog/teoria-das-ondas-de-elliott/)  
9. Teoria das Ondas de Elliott: Superestimada ou Subestimada? \- EBC Financial Group, accessed on June 23, 2025, [https://www.ebc.com/pt/forex/197047.html](https://www.ebc.com/pt/forex/197047.html)  
10. Ondas de Elliott: aprenda o que são e como usar na análise técnica \- \- SmarttBot, accessed on June 23, 2025, [https://smarttbot.com/trader/voce-sabe-o-que-sao-ondas-de-elliot/](https://smarttbot.com/trader/voce-sabe-o-que-sao-ondas-de-elliot/)  
11. Dominando a onda de Elliott: prevendo as ondas 3 e 5 com base nas retrações da onda 2 | sailortrades no Binance Square, accessed on June 23, 2025, [https://www.binance.com/pt-BR/square/post/12860693433586](https://www.binance.com/pt-BR/square/post/12860693433586)  
12. Multi-classifier based on Elliott wave's recognition \- ResearchGate, accessed on June 23, 2025, [https://www.researchgate.net/publication/257312966\_Multi-classifier\_based\_on\_Elliott\_wave's\_recognition](https://www.researchgate.net/publication/257312966_Multi-classifier_based_on_Elliott_wave's_recognition)  
13. Fibonacci e Ondas de Elliot \- Livreto | PDF \- Scribd, accessed on June 23, 2025, [https://pt.scribd.com/document/472344967/Fibonacci-e-ondas-de-Elliot-livreto](https://pt.scribd.com/document/472344967/Fibonacci-e-ondas-de-Elliot-livreto)  
14. Market Prices Trend Forecasting Supported By Elliott Wave's Theory \- EUDL, accessed on June 23, 2025, [https://eudl.eu/pdf/10.4108/eai.27-2-2017.152341](https://eudl.eu/pdf/10.4108/eai.27-2-2017.152341)  
15. Ondas de Elliott: saiba o que são, como utilizar e contagem, accessed on June 23, 2025, [https://cmcapital.com.br/blog/ondas-de-elliott/](https://cmcapital.com.br/blog/ondas-de-elliott/)  
16. Aplicação do princípio das ondas de Elliott à bolsa portuguesa \- repositorio@ipl.pt, accessed on June 23, 2025, [https://repositorio.ipl.pt/entities/publication/3edf10f4-f9db-40f2-a37c-369cf6f5f44b](https://repositorio.ipl.pt/entities/publication/3edf10f4-f9db-40f2-a37c-369cf6f5f44b)  
17. Metadados do item: Previsão do mercado acionário por meio de ..., accessed on June 23, 2025, [https://bdtd.ibict.br/vufind/Record/UFSM\_cc63e67a333faad63afd5a0ea7f22b83](https://bdtd.ibict.br/vufind/Record/UFSM_cc63e67a333faad63afd5a0ea7f22b83)  
18. Predicting Stock Market Trends of Iran Using Elliott Wave Oscillation and Relative Strength Index, accessed on June 23, 2025, [https://jfr.ut.ac.ir/m/article\_82176.html?lang=en](https://jfr.ut.ac.ir/m/article_82176.html?lang=en)  
19. Vladyslav Kabachii Roman Maslii Serhii Kozlovskyi Oleksandr Dronchack \- Neuro-Fuzzy Modeling Techniques in Economics, accessed on June 23, 2025, [https://nfmte.kneu.ua/download/2023/12.07/Identifying%20moments%20of%20decision%20making%20on%20trade%20in%20financial%20time%20series%20using%20fuzzy%20cluster%20analysis.pdf](https://nfmte.kneu.ua/download/2023/12.07/Identifying%20moments%20of%20decision%20making%20on%20trade%20in%20financial%20time%20series%20using%20fuzzy%20cluster%20analysis.pdf)  
20. 2019 Mustafa Ozorhan | PDF | Time Series | Foreign Exchange Market \- Scribd, accessed on June 23, 2025, [https://www.scribd.com/document/468043109/2019-Mustafa-Ozorhan](https://www.scribd.com/document/468043109/2019-Mustafa-Ozorhan)  
21. (PDF) Elliott Wave Prediction Using a Neural Network and Its ..., accessed on June 23, 2025, [https://www.researchgate.net/publication/382635050\_Elliott\_Wave\_Prediction\_Using\_a\_Neural\_Network\_and\_Its\_Application\_to\_The\_Formation\_of\_Investment\_Portfolios\_on\_The\_Indonesian\_Stock\_Exchange](https://www.researchgate.net/publication/382635050_Elliott_Wave_Prediction_Using_a_Neural_Network_and_Its_Application_to_The_Formation_of_Investment_Portfolios_on_The_Indonesian_Stock_Exchange)  
22. Using SVM with Financial Statement Analysis for Prediction of Stocks \- CSUSB ScholarWorks, accessed on June 23, 2025, [https://scholarworks.lib.csusb.edu/cgi/viewcontent.cgi?article=1059\&context=ciima](https://scholarworks.lib.csusb.edu/cgi/viewcontent.cgi?article=1059&context=ciima)  
23. Financial time series forecasting using support vector machines, accessed on June 23, 2025, [https://c.mql5.com/forextsd/forum/35/kim2003.pdf](https://c.mql5.com/forextsd/forum/35/kim2003.pdf)  
24. Financial Forecasting Using Support Vector Machines \- CiteSeerX, accessed on June 23, 2025, [https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=0ebab3f4c2c949a5e1ee1ea3ae0cf3eb72353fef](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=0ebab3f4c2c949a5e1ee1ea3ae0cf3eb72353fef)  
25. Machine learning approaches for financial time series forecasting, accessed on June 23, 2025, [https://lib.iitta.gov.ua/id/eprint/727244/1/paper47.pdf](https://lib.iitta.gov.ua/id/eprint/727244/1/paper47.pdf)  
26. News Impact on Stock Trend \- SciSpace, accessed on June 23, 2025, [https://scispace.com/pdf/news-impact-on-stock-trend-1ommp02e1h.pdf](https://scispace.com/pdf/news-impact-on-stock-trend-1ommp02e1h.pdf)  
27. Large Language Models and the Elliott Wave Principle: A Multi-Agent Deep Learning Approach to Big Data Analysis in Financial Markets \- MDPI, accessed on June 23, 2025, [https://www.mdpi.com/2076-3417/14/24/11897](https://www.mdpi.com/2076-3417/14/24/11897)  
28. Detecção de deriva em redes neurais convolucionais aplicada ao reconhecimento de intenções em frases curtas \- Adelpha Repositório Digital, accessed on June 23, 2025, [https://dspace.mackenzie.br/handle/10899/33747](https://dspace.mackenzie.br/handle/10899/33747)  
29. Stock trading rule discovery with an evolutionary trend following model | Request PDF, accessed on June 23, 2025, [https://www.researchgate.net/publication/265052969\_Stock\_trading\_rule\_discovery\_with\_an\_evolutionary\_trend\_following\_model](https://www.researchgate.net/publication/265052969_Stock_trading_rule_discovery_with_an_evolutionary_trend_following_model)  
30. philippe-ostiguy/PyBacktesting: Optimizing the Elliott Wave Theory using genetic algorithms to forecast the financial markets. \- GitHub, accessed on June 23, 2025, [https://github.com/philippe-ostiguy/PyBacktesting](https://github.com/philippe-ostiguy/PyBacktesting)  
31. Unsupervised Learning Methods for Molecular Simulation Data | Chemical Reviews, accessed on June 23, 2025, [https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01195](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01195)  
32. K-Means Clustering Algorithm \- Analytics Vidhya, accessed on June 23, 2025, [https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)  
33. Unveiling patterns in unlabeled data with k-means clustering \- Hex, accessed on June 23, 2025, [https://hex.tech/blog/Unveiling-patterns-in-unlabeled-data-with-k-means-clustering/](https://hex.tech/blog/Unveiling-patterns-in-unlabeled-data-with-k-means-clustering/)  
34. K means Clustering – Introduction \- GeeksforGeeks, accessed on June 23, 2025, [https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)  
35. K Means Clustering Algorithm in Machine Learning \- Simplilearn.com, accessed on June 23, 2025, [https://www.simplilearn.com/tutorials/machine-learning-tutorial/k-means-clustering-algorithm](https://www.simplilearn.com/tutorials/machine-learning-tutorial/k-means-clustering-algorithm)  
36. K-Means Clustering of Daily OHLC Bar Data \- QuantStart, accessed on June 23, 2025, [https://www.quantstart.com/articles/k-means-clustering-of-daily-ohlc-bar-data/](https://www.quantstart.com/articles/k-means-clustering-of-daily-ohlc-bar-data/)  
37. Stock Market Analysis: A Review and Taxonomy of Prediction Techniques \- MDPI, accessed on June 23, 2025, [https://www.mdpi.com/2227-7072/7/2/26](https://www.mdpi.com/2227-7072/7/2/26)  
38. Time series motif discovery: Dimensions and applications | Request PDF \- ResearchGate, accessed on June 23, 2025, [https://www.researchgate.net/publication/264716280\_Time\_series\_motif\_discovery\_Dimensions\_and\_applications](https://www.researchgate.net/publication/264716280_Time_series_motif_discovery_Dimensions_and_applications)  
39. Using a Genetic Algorithm to Build a Volume Weighted Average Price Model in a Stock Market \- MDPI, accessed on June 23, 2025, [https://www.mdpi.com/2071-1050/13/3/1011](https://www.mdpi.com/2071-1050/13/3/1011)  
40. Trabajo de Fin de Grado Herramienta de inteligencia artificial para detectar patrones en bolsa \- riull@ull, accessed on June 23, 2025, [https://riull.ull.es/xmlui/bitstream/handle/915/29420/Herramienta%20de%20inteligencia%20artificial%20para%20detectar%20patrones%20en%20bolsa.pdf?sequence=1](https://riull.ull.es/xmlui/bitstream/handle/915/29420/Herramienta%20de%20inteligencia%20artificial%20para%20detectar%20patrones%20en%20bolsa.pdf?sequence=1)  
41. K-Means Clustering: Stop \#3 on Your DIY Data Science Roadmap \- YouTube, accessed on June 23, 2025, [https://www.youtube.com/watch?v=9Mmj8NMCqEQ](https://www.youtube.com/watch?v=9Mmj8NMCqEQ)