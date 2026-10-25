# Reconhecimento de caneta e garrafa pela webcam

Para executar o projeto do começo ao fim, consulte [PASSO_A_PASSO.md](PASSO_A_PASSO.md), com todos os comandos para PowerShell.

Projeto acadêmico em Python: CLIP (Hugging Face Transformers/PyTorch) como extrator visual congelado e regressão logística supervisionada para classificar **caneta**, **garrafa** e **outro**. O programa classifica a região central da webcam; apresente um objeto por vez dentro do quadrado verde. Ele não localiza vários objetos nem desenha caixas individuais de detecção.

## Instalação no Windows

Use Python 3.11 ou 3.12 de 64 bits como ambiente recomendado. No PowerShell, dentro desta pasta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente virtual. Se `py -3.12` não existir, instale essa versão ou use outra compatível com as dependências. Uma GPU é opcional; o programa usa CUDA quando disponível e CPU nos demais casos. Reserve espaço para PyTorch e os pesos do CLIP (centenas de MB, além das bibliotecas). O primeiro treinamento baixa os pesos públicos da Hugging Face e precisa de internet. As imagens capturadas são processadas localmente.

## 1. Coletar imagens

**Atualização:** foram selecionadas imagens da internet pelo script `baixar_imagens.py`, que preenche estas mesmas pastas. Veja `FONTES_IMAGENS.md` para origem, quantidades e limitações. Com o download concluído, você pode seguir diretamente para o treinamento; a coleta abaixo é opcional para complementar o conjunto.

Execute um comando de cada vez. **Espaço** salva a imagem central e **Q** fecha a janela.

```powershell
.\.venv\Scripts\python.exe projeto.py capturar --classe caneta --divisao treino
.\.venv\Scripts\python.exe projeto.py capturar --classe garrafa --divisao treino
.\.venv\Scripts\python.exe projeto.py capturar --classe outro --divisao treino
```

Repita para cada classe com `--divisao validacao` e depois com `--divisao teste`: são nove combinações no total. Sugestão inicial por classe: 100 imagens de treino, 30 de validação e 30 de teste (480 imagens no total). O mínimo de cinco por pasta serve apenas para verificar o fluxo, não para sustentar uma conclusão de qualidade.

Faça as divisões em **sessões separadas**, variando iluminação, fundo, pessoa, posição e exemplares dos objetos. Evite fotos quase idênticas nos três conjuntos: isso produziria uma avaliação artificialmente alta. Para teste, use também canetas e garrafas físicas que não apareceram no treino. Não reutilize imagens entre divisões. Para “outro”, capture mão vazia, fundo vazio, celular, copo, caderno e outros objetos. Mostre a caneta suficientemente perto para que ela ocupe uma parte significativa do quadrado. Inclua as mesmas condições de fundo em todas as classes para evitar que o modelo aprenda apenas o cenário.

Estrutura criada automaticamente pela coleta:

```text
dados/
  treino/caneta/       treino/garrafa/       treino/outro/
  validacao/caneta/    validacao/garrafa/    validacao/outro/
  teste/caneta/        teste/garrafa/        teste/outro/
```

Também é possível colocar JPG/PNG próprios nessas pastas. Nesse caso, use imagens enquadradas de forma semelhante ao recorte da webcam.

## 2. Treinar e avaliar

```powershell
.\.venv\Scripts\python.exe projeto.py treinar
```

São gerados `artefatos/modelo.joblib` e `artefatos/avaliacao.json`. O primeiro contém o classificador treinado e o identificador do extrator; o segundo contém acurácia, precisão, recall, F1 por classe, F1 macro de validação e matriz de confusão do teste. Na matriz, linhas são classes reais e colunas são classes previstas, na ordem indicada no JSON. Preencha o relatório com esses resultados reais. Nenhuma acurácia foi presumida.

## 3. Reconhecer pela webcam

### Melhoria: calibrar para sua webcam

O teste com imagens da internet não representa necessariamente sua mão, fundo e iluminação. Se o reconhecimento estiver ruim, rode:

```powershell
.\.venv\Scripts\python.exe projeto.py calibrar
```

Para cada classe, a janela pede que você posicione o objeto e pressione Espaço. Há dois segundos de preparação e depois a captura automática de 20 exemplos, com intervalo de pelo menos 0,7 segundo. Varie ângulo, distância e posição durante a coleta. Na etapa `outro`, mostre o fundo vazio, a mão vazia e outros objetos; não mostre caneta ou garrafa. Q cancela sem substituir o modelo anterior.

O comando treina um classificador local separado em `artefatos/modelo_local.joblib`, usando o mesmo CLIP. As imagens ficam em `dados/calibracao/`. Não é necessário treinar novamente com `treinar` após calibrar. O comando `webcam` escolhe o modelo local automaticamente; `webcam --original` permite comparar com o anterior. A calibração substitui apenas o modelo local de uma calibração anterior, preservando o modelo e o relatório do experimento da internet.

Na webcam, use `-` para diminuir a região central e aproximar o objeto no recorte, e `+` para aumentá-la. Mantenha o objeto inteiro dentro do quadrado. A previsão usa média temporal e mostra as probabilidades das três classes. Essa suavização reduz oscilações, mas pode adicionar um pequeno atraso à mudança de objeto.

**Avalie com novas poses e exemplares depois de calibrar.** As imagens da calibração são treinamento; não possuem teste independente e não herdam os 100% obtidos no conjunto da internet. Para o relatório, registre separadamente o teste da internet e uma avaliação nova pela webcam.

```powershell
.\.venv\Scripts\python.exe projeto.py webcam
```

Apresente caneta ou garrafa dentro do quadrado, aguarde a classificação e pressione **Q** para sair. “Outro” é uma classe aprendida; “Incerto” aparece quando a maior probabilidade é inferior ao limiar. As probabilidades da regressão logística não são garantias de acerto, e o limiar não garante rejeitar todo objeto desconhecido.

```powershell
# Outra webcam e intervalo maior para computadores mais lentos:
.\.venv\Scripts\python.exe projeto.py webcam --camera 1 --intervalo 1.0
# Ajuste opcional do limiar, usando apenas a validação:
.\.venv\Scripts\python.exe projeto.py webcam --limiar 0.75
```

Se a webcam não abrir, feche Teams/Zoom e habilite acesso à câmera para aplicativos de área de trabalho nas configurações de privacidade do Windows. Execute localmente em uma sessão com interface gráfica; o OpenCV precisa abrir uma janela. Para parar um comando pelo terminal, use Ctrl+C.

## Entrega e reprodução

Entregue `projeto.py`, `requirements.txt`, este README, `RELATORIO.md`, o JSON com os resultados e o classificador treinado. As imagens podem acompanhar a entrega se houver autorização dos participantes. Para registrar as versões efetivamente usadas depois de validar a execução:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

No computador de apresentação, instale as dependências, copie `artefatos/` e execute `webcam`. O CLIP também precisa estar disponível: faça uma primeira execução conectada para preencher o cache. O arquivo joblib não inclui os pesos do CLIP. Valide tudo no computador de apresentação com antecedência.

## Situação da validação

O projeto exige coleta real de imagens e teste com webcam. Verificações de sintaxe/CLI não substituem a instalação das dependências, o download dos pesos, o treinamento e a avaliação prática. Consulte a resposta de entrega para o que foi efetivamente verificado neste ambiente.
