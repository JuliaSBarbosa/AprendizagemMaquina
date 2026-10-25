# Projeto: classificação de objetos pela webcam

**Equipe:** Amabile Silverio, Danilo Alves, Gustavo Marques e Julia Barbosa
**Disciplina/professor:** Rafael de Souza Teixeira 
**Data de Entrega:** 27/10/2026 

## Objetivo e interpretação do enunciado

Identificar caneta ou garrafa em uma região central da webcam, incluindo uma classe “outro” para exemplos negativos. A implementação usa um modelo de visão e linguagem (VLM), CLIP, e treina um classificador com imagens próprias.

Há uma distinção conceitual relevante: um classificador visual não é, por si só, um modelo de linguagem. CLIP possui codificadores de imagem e texto e aprende associações entre modalidades; não é um LM generativo que conversa ou produz descrições livres. Nesta implementação utiliza-se apenas seu ramo visual durante o treinamento e a inferência. Portanto, a proposta atende a uma interpretação de “LM com visão” como modelo visão-linguagem, mas **não demonstra geração de linguagem**. Se o requisito for um LM multimodal generativo, será necessário adequar o projeto, por exemplo com um VLM gerador e ajuste em pares imagem/instrução/resposta. Esse requisito deve ser esclarecido com o professor antes da entrega.

## 1. Abordagens viáveis

| Abordagem | Vantagens para este projeto | Limitações |
|---|---|---|
| Hugging Face Transformers + PyTorch + CLIP congelado + regressão logística | Reutiliza representações de um modelo visão-linguagem; treinamento de poucos parâmetros; adequado para experimento com poucos dados; pode executar em CPU | Extrator relativamente pesado; não gera linguagem; congelar CLIP limita a adaptação ao domínio; objetos pequenos podem perder detalhes |
| TensorFlow/Keras + MobileNetV2 pré-treinada + camada classificadora | Rede visual compacta; fluxo de transferência de aprendizado bem documentado; pode ajustar camadas e converter para execução em dispositivos | Não é modelo de linguagem; requer treinamento supervisionado e cuidado com sobreajuste; a qualidade depende do conjunto local |
| PyTorch/torchvision + ResNet18 pré-treinada + camada final | Treinamento flexível e tutorial oficial de transferência de aprendizado; permite comparar extrator congelado e fine-tuning | Não é modelo de linguagem; ajustar toda a rede aumenta custo e risco de sobreajuste em poucos exemplos |

PyTorch e TensorFlow são frameworks; Transformers é uma biblioteca que disponibiliza arquiteturas e pesos e pode usar PyTorch. Portanto, as alternativas não são categorias mutuamente exclusivas. A escolha prática aqui é CLIP + Transformers + PyTorch para aproximar o trabalho do componente visão-linguagem do enunciado.

## 2. Dados, pré-processamento e treinamento

**Dados:** seleção do dataset acadêmico Office-Home, disponibilizado pela Voxel51 na Hugging Face. Usam-se somente fotografias dos domínios Real World e Product. Pen corresponde a caneta; Bottle, a garrafa; Mug, Keyboard, Notebook e Mouse compõem outro. O script seleciona até 125 imagens por classe, separadas em 75/25/25 para treino/validação/teste com semente 42, removendo duplicatas exatas. Quantidades finais e fontes estão em `dados/fontes.json`. A divisão não garante objetos físicos ou sessões independentes, pois esses grupos não estão disponíveis na seleção. O desempenho na webcam deve ser avaliado separadamente. Consulte `FONTES_IMAGENS.md` para atribuição e condições de uso acadêmico.

**Pré-processamento:** na coleta e no uso, recorta-se um quadrado central de 80% da menor dimensão do frame. O OpenCV produz BGR; na inferência convertemos para RGB. No treinamento, Pillow abre as imagens em RGB. O `CLIPProcessor` aplica o redimensionamento, recorte central e normalização esperados pelo checkpoint. Os vetores de características são normalizados pela norma L2. Não se aplicam aumentos de dados nesta versão; a diversidade vem da coleta.

**Modelo:** `openai/clip-vit-base-patch32`. O pré-treinamento original é contrastivo com pares imagem/texto. Neste projeto seus pesos ficam congelados: não se afirma ter treinado CLIP do zero nem feito fine-tuning de seus codificadores. Treina-se uma regressão logística multiclasse sobre as características visuais, abordagem conhecida como linear probe. A perda é a log loss com regularização L2. O código compara C = 0,1; 1; 10 e escolhe pelo F1 macro da validação. O máximo é de 2.000 iterações de otimização, não épocas de fine-tuning. Usa-se ponderação balanceada das classes.

**Avaliação:** após a seleção, mede-se uma vez no teste separado. Relatam-se acurácia, precisão, recall e F1 por classe, além da matriz de confusão. Não ajustar parâmetros com base no teste; se forem necessárias novas decisões, preservar um novo teste independente para a avaliação final. Duplicatas e capturas quase idênticas entre divisões invalidam a independência, mesmo quando os nomes dos arquivos são diferentes.

## 3. Exemplo funcional em Python

O código comentado está em `projeto.py`. Os três comandos são `capturar`, `treinar` e `webcam`, detalhados no README. Fluxo: webcam → recorte → RGB → processador CLIP → vetor visual → regressão logística → classe exibida na janela. A captura grava exemplos rotulados; o treinamento gera o artefato; a webcam carrega o mesmo extrator e classificador.

## 4. Implantação

1. Preparar computador com Python 64 bits, ambiente virtual, webcam e interface gráfica.
2. Instalar `requirements.txt` e baixar os pesos na primeira execução conectada.
3. Coletar os nove grupos de imagens e treinar localmente.
4. Inspecionar `avaliacao.json` e testar condições reais da apresentação.
5. Distribuir código, versões efetivas das dependências e `artefatos/modelo.joblib`; disponibilizar também o checkpoint CLIP no cache do computador de destino.
6. Executar `python projeto.py webcam` no ambiente instalado. A inferência pode ocorrer sem rede depois que os arquivos do modelo estão em cache.

As imagens ficam no disco local. O processamento não utiliza uma API remota de inferência. CPU é suficiente para tentar o projeto, mas a taxa de atualização depende do hardware; não se promete uma quantidade de FPS. O intervalo entre inferências é configurável.

## Resultados a preencher após executar

**Revisão do uso na webcam:** o primeiro experimento registrou acurácia 1,0 em 73 imagens de teste da internet, mas o usuário relatou reconhecimento ruim pela webcam. Esse resultado evidencia uma limitação de transferência entre cenários; não é uma medição de desempenho em vídeo. Foi acrescentado o comando `calibrar`, que treina uma regressão logística local (C=10, valor fixo heurístico) com 20 exemplos por classe capturados automaticamente na webcam. Esse modelo é salvo separadamente e não possui avaliação independente automática. Também foram adicionados ajuste da região de interesse e média exponencial das probabilidades (65% da previsão atual, 35% do histórico). A eficácia dessas mudanças no ambiente real ainda precisa ser medida com novas amostras.

| Item | Resultado medido |
|---|---|
| Quantidade real por classe e divisão | Preencher |
| Exemplares, fundos e sessões usados | Preencher |
| C selecionado / F1 macro de validação | Copiar do JSON |
| Acurácia e F1 por classe no teste | Copiar do JSON |
| Matriz de confusão | Copiar do JSON |
| Hardware e versões instaladas | Preencher |
| Erros observados e condições em que ocorreram | Preencher |

## Limitações e conclusão esperada

O sistema classifica uma região inteira e não separa objetos simultâneos. Canetas pequenas, reflexos em garrafas, fundos novos e objetos semelhantes podem provocar erros. A classe “outro” cobre somente parte das situações desconhecidas. O limiar inicial de 0,65 é heurístico; se necessário, deve ser escolhido com a validação. Uma probabilidade alta não comprova que a previsão está correta. O sucesso do experimento será determinado pelos resultados medidos, não pela existência do código.

## Referências oficiais consultadas

- Hugging Face. [Documentação CLIP](https://huggingface.co/docs/transformers/model_doc/clip): arquitetura visão-linguagem e extração de características.
- Hugging Face/OpenAI. [Model card de CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32): checkpoint, treinamento original e limitações.
- PyTorch. [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html): extrator congelado versus ajuste da rede.
- TensorFlow. [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning): transferência de aprendizado com MobileNetV2.

Consulta: 25/10/2026.
