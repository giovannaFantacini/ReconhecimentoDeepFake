# ReconhecimentoDeepFake
Projeto final desenvolvido para a disciplina de Processamento Digital de Imagem

## Resumo 
Este artigo aborda a detecção de deepfakes, destacando métodos teóricos e práticos, incluindo o uso de Convolutional Neural Networks, Inception, Resnet, e a técnica Grad-CAM. O projeto resulta em uma aplicação interativa para verificar a autenticidade de imagens, contribuindo para a segurança da informação e a mitigação dos riscos associados aos deepfakes.

## Introdução

Com o acelerado avanço tecnológico na área da inteligência artificial, diversas aplicações foram desenvolvidas, dentre essas, os deepfakes, algoritmos de manipulação de mídia baseados em aprendizado profundo, os quais, surgindo em 2017, levantam preocupações significativas em relação à segurança e privacidade da sociedade [8]. Os deepfakes permitem a substituição de uma pessoa em uma imagem ou vídeo existente por outra, criando o potencial para enganos e sabotagens.

Nos últimos anos, diversos deepfakes com rostos de políticos e figuras proeminentes surgiram, como imagens do Papa vestido de maneira "estilosa", gravações musicais com a voz de artistas sem sua autorização, entre outros casos. Devido a esta situação é notável que, mesmo que esta tecnologia possa ser utilizada de forma ética para que contribua para o desenvolvimento da sociedade, como melhorar a educação ou ressuscitar artistas falecidos, os impactos negativos, como sabotagem, ameaças e danos à reputação, superam os positivos.

Nesse contexto, a detecção de deepfakes é essencial para atestar a veracidade das informações, evitando assim a propagação de vídeos e imagens falsos. Portanto, este artigo busca explorar não apenas a criação de deepfakes, mas principalmente métodos para identificá-los de maneira eficaz.

## Objetivos

Este trabalho visa aprofundar e compreender os fundamentos teóricos por trás da criação de deepfakes utilizando algoritmos CNN, assim como desenvolver um sistema de reconhecimento de deepfakes utilizando uma rede neural ResNetInception treinada, avaliar a eficácia do modelo ResNetInception na identificação de deepfakes e criar uma interface amigável utilizando Gradio para facilitar o uso do aplicativo de reconhecimento de deepfakes.

## Referencial Teórico

### DeepFakes
Deepfake, uma junção de "deep learning" e "fake", refere-se a vídeos, imagens e áudios hiper-realistas manipulados digitalmente para retratar pessoas dizendo e fazendo coisas que nunca ocorreram [6]. Essa tecnologia utiliza redes neurais para analisar grandes conjuntos de dados e aprender a imitar expressões faciais, maneirismos, voz e inflexões de uma pessoa específica. O processo envolve alimentar um algoritmo de deep learning com imagens de duas pessoas para treiná-lo a trocar os rostos. Em termos simples, deepfakes utilizam tecnologia de mapeamento facial e IA para substituir o rosto de uma pessoa em um vídeo pelo rosto de outra.

O fenômeno dos deepfakes ganhou notoriedade em 2017, quando um usuário do Reddit publicou vídeos mostrando celebridades em situações sexuais comprometedoras [5]. A detecção de deepfakes é desafiadora, uma vez que eles usam imagens reais, podem ter áudio autêntico e são otimizados para se espalhar rapidamente nas redes sociais. Assim, muitos espectadores assumem que o vídeo que estão assistindo é autêntico. Essa manipulação visa especialmente plataformas de mídia social, onde teorias conspiratórias, boatos e desinformação se propagam facilmente, já que os usuários tendem a seguir a multidão.

A crescente disponibilidade de hardware acessível, como unidades de processamento gráfico eficientes, facilita a disseminação de deepfakes de baixa qualidade, conhecidos como "cheap fakes" [6]. Além disso, o software para criar deepfakes realistas está se tornando cada vez mais disponível como código aberto, permitindo que usuários com poucas habilidades técnicas editem vídeos, troquem rostos, alterem expressões e sintetizem fala de forma quase perfeita.

A tecnologia por trás dos deepfakes é fundamentada nas Redes Generativas Adversariais (GANs), duas redes neurais artificiais trabalhando juntas para criar mídia realista. O "gerador" e o "discriminador" dessas redes são treinados com o mesmo conjunto de dados de imagens, vídeos ou sons. O gerador tenta criar novas amostras que enganem o discriminador, que por sua vez determina se a nova mídia que vê é real [6]. Essa interação impulsiona a melhoria contínua, permitindo que as GANs criem retratos, inclusive trocar cabeças, corpos inteiros e vozes no futuro.

Sucintamente, deepfake é uma produção artificial que parece autêntica aos olhos humanos, gerada por Inteligência Artificial [6]. A manipulação de imagens humanas é comum, substituindo rostos em imagens ou vídeos existentes para criar uma reencenação fictícia que parece real.

Além disso, aplicações de Deepfake vão além do entretenimento, incluindo dublagem artificial, reanimação de personagens históricos e modelos digitais personalizados [5]. No entanto, ele alerta para os impactos sociais nefastos, que vão desde constrangimentos pessoais até implicações legais, como violação de direitos de imagem e propriedade intelectual, resultando em prejuízos econômicos e ameaças à reputação. Em casos extremos, vídeos falsificados de figuras políticas podem desencadear crises na mídia, ameaçando a estabilidade social e a segurança nacional.

### Deep Learning
A técnica de deep learning, traduzida para o português como aprendizado profundo, é uma subcategoria do aprendizado de máquina que surgiu como um paradigma poderoso para resolver problemas complexos, permitindo que modelos aprendam automaticamente representações hierárquicas a partir de dados. Essa abordagem utiliza múltiplos níveis de abstração para transformar dados brutos em representações mais abstratas e informativas. Ao contrário dos modelos tradicionais de aprendizado de máquina, o aprendizado profundo não depende de engenharia manual de características; em vez disso, utiliza redes neurais compostas por camadas de módulos não lineares para aprender e extrair automaticamente características [2].

Um dos aspectos notáveis do aprendizado profundo é sua capacidade de descobrir estruturas complexas em dados de alta dimensão, tornando-o aplicável a uma ampla gama de domínios, incluindo ciência, negócios e governo. Ao longo dos anos, o aprendizado profundo obteve resultados inovadores em diversas áreas, como reconhecimento de imagens e fala, previsão da atividade de moléculas de drogas, análise de dados de aceleradores de partículas, reconstrução de circuitos cerebrais e tarefas de compreensão de linguagem natural, como classificação de tópicos, análise de sentimentos, resposta a perguntas e tradução de idiomas.

O aprendizado de máquina mais comum, seja ele profundo ou não, é o aprendizado supervisionado. Nesse cenário, o objetivo é treinar um sistema para classificar imagens em categorias, como casas, carros, pessoas ou animais de estimação. Isso é feito por meio de um grande conjunto de dados de imagens, onde cada imagem é rotulada com sua categoria. Durante o treinamento, a máquina produz uma saída em forma de vetor de pontuações para cada categoria, e a meta é que a categoria desejada tenha a maior pontuação. Para alcançar isso, é usado um procedimento de aprendizado que ajusta os pesos internos da máquina para minimizar o erro entre as pontuações previstas e as desejadas. A grande novidade do aprendizado profundo é que esses ajustes são automáticos e feitos por meio de gradientes, permitindo que modelos aprendam representações complexas e sensíveis a detalhes sutis em dados de alta dimensão, tornando-os ideais para tarefas como reconhecimento de imagens [2].

### Convolutional Neural Network
Uma Rede Neural Convolucional (ConvNet / Convolutional Neural Network / CNN) é um algoritmo de Aprendizado Profundo. As ConvNets são projetadas para processar dados dispostos em múltiplas matrizes, como uma imagem colorida composta por três matrizes 2D que representam as intensidades de pixel nos três canais de cor. Essa arquitetura é aplicável a diversas modalidades de dados que possuem formato de múltiplas matrizes, como sinais unidimensionais, imagens bidimensionais ou espectrogramas de áudio, e dados tridimensionais, como vídeos ou imagens volumétricas [2].

As CNNs se destacam por quatro ideias-chave: conexões locais, compartilhamento de pesos, pooling e múltiplas camadas. A estrutura típica de uma ConvNet é composta por estágios, onde camadas convolucionais e de pooling desempenham papéis fundamentais. As camadas convolucionais são usadas para extrair características locais, como detalhes faciais em uma imagem, enquanto as camadas de pooling combinam características semanticamente semelhantes, como expressões faciais. Essa abordagem hierárquica é inspirada na organização das representações naturais de sinais [2].

A técnica de deepfake utiliza pares de codificador-decodificador baseados em ConvNets para extrair características latentes de imagens faciais e reconstruir faces-alvo a partir dessas características, permitindo a troca de rostos entre imagens de origem e alvo, como mostra a Figura 1, onde duas redes usam o mesmo codificador, mas decodificadores diferentes para o processo de treinamento. Uma imagem da face A é codificada com o codificador comum e decodificada com o decodificador B para criar um deepfake.

![DeepFake](https://github.com/giovannaFantacini/ReconhecimentoDeepFake/assets/74154716/8266c071-6bf3-4b1e-a4ba-05635391aea1)

Além disso, perdas adversárias e perdas perceptuais são aplicadas na arquitetura para melhorar a qualidade e realismo das imagens geradas. Essa abordagem hierárquica de ConvNets é essencial para a criação de deepfakes realistas [7].

### Inception e Resnet

Dentre as arquiteturas de CNNs (Redes Neurais Convolucionais) estão enquadradas Inception e Resnet, as quais são amplamente utilizadas no processamento e análise de imagens digitais. Inception, desenvolvida pela Google, tem como característica principal a utilização de módulos de convolução em paralelo, empregando diferentes tamanhos de filtros com o intuito de capturar características em diferentes escalas, permitindo que a rede aprenda representações consideravelmente mais ricas e robustas das imagens de entrada [10]. Resnet, por sua vez, é definida como uma arquitetura neural profunda, desenvolvida em 2015 com o intuito de resolver o problema do gradiente desvanecente em redes neurais profundas. Para tal, utiliza conexões residuais, as quais permitem à rede aprender o objeto de estudo em camadas intermediárias, facilitando o treinamento. É composta por blocos residuais que abarcam uma conexão direta entre a entrada e saída do bloco, possibilitando à rede aprender a diferença existente entre entrada e saída do bloco [1].

###  Grad-CAM

A técnica Grad-CAM (Gradient-weighted Class Activation Mapping) surge como uma abordagem para tornar os modelos baseados em Convolutional Neural Networks (CNNs) mais transparentes, proporcionando visualizações das regiões de entrada que são "importantes" para as previsões desses modelos, ou seja, explicações visuais [3]. Essas visualizações são tanto de alta resolução, identificando características específicas da classe de interesse, quanto discriminativas em relação às classes, destacando a classe de interesse sem incluir outras classes irrelevantes.

Para compreender a evolução do Grad-CAM, é necessário revisitar a Class Activation Mapping (CAM). CAM produz um mapa de localização a partir de CNNs de classificação de imagem, utilizando feature maps globais médios-poolados que alimentam diretamente um softmax. Esse método é limitado em sua aplicabilidade, exigindo a substituição de camadas totalmente conectadas por convolucionais e a necessidade de re-treinamento. Grad-CAM, por outro lado, generaliza a abordagem CAM para qualquer arquitetura diferenciável baseada em CNN, sem a necessidade de re-treinamento [9].

A obtenção do mapa de localização discriminativo de classe Grad-CAM envolve o cálculo do gradiente de uma classe alvo em relação aos feature maps de uma camada convolucional. Esses gradientes são então agregados globalmente para obter pesos que representam a importância de cada feature map para a classe alvo. O mapa Grad-CAM resultante é uma combinação ponderada de feature maps seguida por uma ReLU, resultando em um mapa de calor grosseiro, normalizado para fins de visualização [9].

No entanto, apesar da capacidade do Grad-CAM de ser discriminativo em relação às classes e localizar regiões relevantes na imagem, ele carece da capacidade de mostrar importâncias detalhadas, como métodos de visualização de gradiente de espaço de pixel. Para superar essa limitação, a fusão do Grad-CAM com visualizações de alta resolução, como o Guided Backpropagation, é proposta para criar o Guided Grad-CAM. Essa fusão, realizada por meio de uma multiplicação pontual, combina a natureza discriminativa do Grad-CAM com a alta resolução do Guided Backpropagation, oferecendo uma visualização mais detalhada e informativa [3].

Assim, a combinação de Grad-CAM com visualizações de alta resolução, como o Guided Backpropagation, representa uma evolução significativa na interpretabilidade de modelos baseados em CNN, permitindo uma compreensão mais profunda das decisões tomadas por esses modelos em tarefas de classificação de imagem.

## Materiais e Metódos

O projeto foi desenvolvido utilizando a plataforma Google Colab e diversas bibliotecas essenciais, incluindo Gradio, torch, os, numpy, PIL, zipfile, cv2 e pytorch. Adicionalmente, foi incorporado um modelo pré-treinado, o InceptionResnetV1, disponível na plataforma HuggingFace [4], contribuindo para a eficácia e robustez do sistema.

## Desenvolvimento
O desenvolvimento do projeto de reconhecimento de deepfake foi conduzido com base em uma abordagem metodológica estruturada, elaborada para enfrentar os desafios específicos relacionados à detecção de conteúdo falso gerado por inteligência artificial. As etapas descritas a seguir delineiam a implementação direta desse método, com o objetivo de construir um sistema eficiente e robusto.

Inicialmente, foi utilizado um modelo previamente treinado [4], o qual emprega a arquitetura InceptionResnetV1, escolhida devido à sua habilidade em discernir padrões complexos em imagens faciais. A escolha de um modelo pré-treinado, proveniente de uma fonte confiável, assegurou que o sistema iniciasse com um conhecimento sólido, fundamentado em uma ampla gama de dados.

Métricas do Modelo InceptionResnet Retreinado:

  Validation Accuracy: 96.6%
  Validation Loss: 0.16
  Train Loss: 0.017
  Train Accuracy: 99.89%
  Epoch: 80
  
Antes de serem inseridas no modelo de classificação, as imagens foram submetidas a processos essenciais de pré-processamento, incluindo redimensionamento para o tamanho uniforme de 256x256 pixels e normalização. Essas etapas são cruciais para garantir que todas as imagens estejam em um formato compatível com a entrada do modelo, permitindo uma análise consistente e precisa.

A interpretabilidade do modelo foi aprimorada significativamente por meio da técnica Grad-CAM (Gradient-weighted Class Activation Mapping). Esta abordagem visual proporciona uma explicação detalhada das decisões do modelo, destacando as regiões específicas da imagem que mais influenciaram na classificação final. A função "predict" foi desenvolvida com o objetivo de conduzir essa análise e gerar visualizações explicativas, permitindo uma compreensão mais profunda das características relevantes consideradas pelo modelo durante a predição. Essa transparência é crucial, especialmente em cenários onde a detecção de manipulações, como deepfakes, demanda uma compreensão clara das bases da decisão do modelo.

Além disso, para tornar o projeto mais interativo, foi incorporada a biblioteca Gradio. Essa adição possibilita uma experiência mais amigável, permitindo o upload fácil de imagens e a exibição dos resultados da classificação, juntamente com explicações visuais geradas pela técnica Grad-CAM. A interface simplificada oferece aos usuários a capacidade de avaliar o modelo de maneira intuitiva, contribuindo para uma compreensão mais efetiva das decisões tomadas pelo sistema.

## Resultados

O desfecho deste projeto resultou em uma aplicação prática e interativa, conforme apresentado na Figura 2. Nessa aplicação, os usuários têm a capacidade de efetuar o upload de uma imagem ou optar por utilizar uma das imagens de teste, possibilitando uma análise minuciosa da resposta gerada pelo modelo.
![GradioResult](https://github.com/giovannaFantacini/ReconhecimentoDeepFake/assets/74154716/209bf61e-a739-4f25-9921-5dcba9be72e6)

Esta resposta não apenas fornece a probabilidade de autenticidade da imagem, determinada pelo modelo, mas também inclui uma representação visual elucidativa. Através da técnica Grad-CAM, destacam-se de maneira proeminente as regiões específicas da imagem que mais influenciaram na classificação final, conforme ilustrado na Figura 3. Esse processo contribui para uma compreensão mais profunda e transparente das decisões do modelo, permitindo aos usuários uma avaliação criteriosa da autenticidade das imagens analisadas.

![Resultadofake](https://github.com/giovannaFantacini/ReconhecimentoDeepFake/assets/74154716/d0d99a0c-d6c4-4352-b39c-8984f3396f13)


## Conclusão
O projeto teve como objetivo enfrentar os desafios decorrentes do avanço rápido da inteligência artificial, especialmente no contexto dos deepfakes, que têm gerado preocupações significativas relacionadas à segurança e privacidade desde sua introdução em 2017. Os deepfakes, baseados em algoritmos de aprendizado profundo, possibilitam a manipulação de mídias, como fotos e vídeos, tendo o potencial de gerar constrangimento ao alvo, instigar crimes, principalmente de ódio, ao substituir inocentes em imagens e vídeos existentes.

O foco principal deste trabalho foi a detecção eficaz de deepfakes, reconhecendo a importância de salvaguardar a veracidade das informações em meio à propagação de conteúdos falsos. Foram explorados métodos para a criação de deepfakes, destacando a necessidade crítica de desenvolver abordagens eficientes para identificá-los.

Os objetivos delineados incluíram a compreensão teórica da criação de deepfakes usando algoritmos CNN, o desenvolvimento de um sistema de reconhecimento de deepfakes baseado na arquitetura ResNetInception, a avaliação da eficácia desse modelo e a criação de uma interface amigável utilizando Gradio para facilitar o uso do aplicativo de reconhecimento de deepfakes.

Além disso, foram explorados os fundamentos dos deepfakes, sua evolução desde 2017, e os impactos sociais e legais decorrentes do uso inadequado dessa tecnologia. Também houve aprofundamento em conceitos fundamentais, como deep learning, Convolutional Neural Networks (CNNs), Inception, Resnet e a técnica Grad-CAM, essenciais para compreender o desenvolvimento do projeto.

Os materiais e métodos empregados abrangeram o uso de uma variedade de bibliotecas e ferramentas, incluindo Google Colab, Gradio, torch, MTCNN, InceptionResnetV1, entre outras. A escolha criteriosa de um modelo pré-treinado, aliada a técnicas de pré-processamento, contribuiu para a robustez e eficácia do sistema.

O desenvolvimento do projeto seguiu uma metodologia estruturada, incluindo etapas como o pré-processamento de imagens, o uso de técnicas de interpretabilidade como Grad-CAM, e a integração de uma interface amigável para facilitar a interação do usuário com o sistema.

Os resultados apresentaram uma aplicação funcional que permite aos usuários realizar uploads de imagens para verificação de autenticidade, recebendo respostas detalhadas do modelo, incluindo a probabilidade de ser real ou falso e visualizações destacando as regiões decisivas da imagem. Em última análise, este projeto não apenas explorou os aspectos técnicos e teóricos relacionados aos deepfakes, mas também proporcionou uma solução prática e interativa para a detecção dessas manipulações de mídia. O sistema desenvolvido representa um passo significativo em direção à mitigação dos riscos associados aos deepfakes, contribuindo para a segurança da informação e a preservação da integridade nas comunicações visuais.

## Referências

[1] ALZUBAIDI, L., AL-FAHDAWI, S., AL-JUMEILY, D., HUSSAIN, A., MALLUCCI, C. (2021). Deep learning: a review from the perspective of practice. Journal of Big Data, 8(1), 53. DOI: https://doi.org/10.1186/s40537-021-00444-8. Disponível em: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-
021-00444-8. Acesso em: 26 out. 2023.

[2] BENGIO, Y.; LECUN, Y.; HINTON, G. Deep learning. Nature, [s. l.], p. 436-444, 27 maio 2015. DOI: https://doi.org/10.1038/nature14539. Disponível em: https://www.nature.com/articles/nature14539citeas. Acesso em: 25 out. 2023.

[3] CHETOUI, Mohamed. Gradient-weighted Class Activation Mapping Grad-CAM. In: Medium. [S. l.], 14 mar. 2019. Disponível em:
https://medium.com/@mohamedchetoui/grad-cam-gradient-weightedclass-activation-mapping-ffd72742243a. Acesso em: 18 nov. 2023.

[4] ESPASAND´IN, Aaron. Deepfake-detection. In: Huggingface. [S. l.], Junho 2022. Disponível em: https://huggingface.co/spaces/aaronespasa/deepfake-detection/tree/main. Acesso em: 16 nov. 2023.

[5] LEANDRO, Jorge de Jesus Gomes. Deepfake: explorando tecnicas de detecção de manipulaçãoo digital de imagens de faces.
2022. Monografia (MBA em Inteligencia Artificial e Big Data) - Universidade de Sao Paulo - ICMC/USP, [S. l.], 2022. Disponível em:
https://repositorio.usp.br/directbitstream/03772a14-e798-44da-ba33-18c3ff3161fe/Jorge

[6] MIKA, Mika. The Emergence of Deepfake Technology: A Review. Technology innovation management review, [s. l.], v. 9, Novembro 2019. Disponível em: https://timreview.ca/sites/default/files/articleP DF/T IMReviewN ovember2019

[7] NGUYEN, Cuong M. et al. Deep learning for deepfakes creation and detection: A survey. Computer Vision and Image Understanding, [s. l.], v. 223, 2022. DOI: https://doi.org/10.1016/j.cviu.2022.103525. Disponível em: https://encurtador.com.br/tAFO6. Acesso em: 25 out.
2023.

[8] SEOW, Jia Wen; LIM, Mei Kuan; PHAN, Raphael C.W.; LIU, Joseph  K. A comprehensive overview of Deepfake: Generation, detection,
datasets, and opportunities. Neurocomputing, [S. l.], v. 513, p. 351-371, 19 out. 2022. DOI: https://doi.org/10.1016/j.neucom.2022.09.135. Disponível em: https://encurtador.com.br/fhDER Acesso em: 19 out. 2023.

[9] SELVARAJU, Ramprasaath R. et al. Grad-CAM: Why did you say that?. Cornell University, [s. l.], 25 jan. 2017. Disponível em: https://arxiv.org/abs/1611.07450. Acesso em: 18 nov. 2023.

[10] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision.
Proceedings of the IEEE conference on computer vision and pattern recognition. DOI: DOI: 10.1109/CVPR.2016.308. Disponível em:
https://ieeexplore.ieee.org/document/7780677Acesso em: 26 out. 2023.

