# Reconhecimento de caneta e garrafa pela webcam

Desenvolvi este projeto para a disciplina de Aprendizagem de Máquina com o objetivo de reconhecer **canetas e garrafas pela webcam**. Utilizo Python, CLIP e regressão logística para classificar o objeto apresentado na região central da câmera.

Incluí também a classe `outro`, com exemplos como teclado, caderno, caneca e mouse. O reconhecimento acontece localmente, em uma janela do OpenCV.

## Como funciona

```text
Webcam → recorte central → CLIP → características visuais → classificador → classe na tela
```

Utilizo o CLIP pré-treinado como extrator de características, mantendo seus pesos congelados. O treinamento ajusta uma regressão logística para distinguir as três classes. Embora o CLIP seja um modelo de visão e linguagem, esta implementação utiliza seu ramo visual e não gera respostas textuais livres.

Para adaptar o reconhecimento ao fundo, à iluminação e aos objetos usados na apresentação, implementei uma calibração com captura automática pela webcam.

## Tecnologias

| Tecnologia | Uso |
|---|---|
| Python | Implementação do projeto |
| PyTorch e Hugging Face Transformers | Carregamento e inferência do CLIP |
| scikit-learn | Treinamento e avaliação do classificador |
| OpenCV | Captura da webcam e interface |
| Pillow e NumPy | Processamento das imagens e vetores |
| joblib | Persistência dos classificadores |

## Executar no Windows

Pré-requisitos: **Python 3.12 de 64 bits**, webcam, PowerShell e internet para os downloads iniciais. GPU é opcional: o código usa CPU quando CUDA não está disponível.

### 1. Baixar o projeto

```powershell
git clone https://github.com/JuliaSBarbosa/AprendizagemMaquina.git
cd AprendizagemMaquina
```

Também é possível baixar o ZIP pelo GitHub, extrair e abrir o PowerShell dentro da pasta extraída.

### 2. Instalar as dependências

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os comandos usam diretamente o Python do ambiente virtual; não é necessário ativá-lo. Reserve espaço para as bibliotecas e os pesos do CLIP.

### 3. Baixar as imagens e treinar

```powershell
.\.venv\Scripts\python.exe baixar_imagens.py
.\.venv\Scripts\python.exe projeto.py treinar
```

O download organiza as imagens em treino, validação e teste. O treinamento salva o modelo em `artefatos/modelo.joblib` e as métricas em `artefatos/avaliacao.json`. No primeiro carregamento, os pesos do CLIP são baixados automaticamente.

### 4. Calibrar para a webcam

```powershell
.\.venv\Scripts\python.exe projeto.py calibrar
```

A janela solicita **caneta**, **garrafa** e **outro**, nessa ordem:

1. Posicione o objeto dentro do quadrado e pressione **Espaço** com a janela selecionada.
2. Aguarde dois segundos de preparação e a captura automática de 20 imagens.
3. Durante a captura, varie levemente posição, ângulo e distância.

Na etapa `outro`, apresente o fundo vazio, a mão vazia e objetos diferentes de caneta e garrafa. **Q** cancela a calibração.

Esse comando já treina e salva o modelo local em `artefatos/modelo_local.joblib`. Não é necessário executar `treinar` novamente após calibrar. As imagens ficam em `dados/calibracao/`.

### 5. Iniciar o reconhecimento

```powershell
.\.venv\Scripts\python.exe projeto.py webcam
```

O programa carrega o modelo local quando ele existe; caso contrário, utiliza o modelo treinado com as imagens da internet. O terminal informa qual arquivo foi carregado.

| Tecla | Ação na janela de reconhecimento |
|---|---|
| `-` | Diminuir a região analisada |
| `+` | Aumentar a região analisada |
| `Q` | Encerrar |

Apresente um objeto por vez, inteiro dentro do quadrado. Para canetas pequenas, aproxime o objeto e diminua a região para reduzir o fundo analisado. A tela mostra a previsão e as probabilidades das três classes. A resposta `Incerto` aparece quando a maior probabilidade fica abaixo do limiar.

Nos próximos usos, basta abrir o terminal na pasta e executar `webcam`. Mudanças significativas de fundo ou iluminação podem exigir uma nova calibração.

## Opções úteis

```powershell
# Outra câmera: usar o mesmo índice na calibração e no reconhecimento
.\.venv\Scripts\python.exe projeto.py calibrar --camera 1
.\.venv\Scripts\python.exe projeto.py webcam --camera 1

# Mais exemplos por classe na calibração
.\.venv\Scripts\python.exe projeto.py calibrar --fotos 40

# Comparar com o modelo treinado nas imagens da internet
.\.venv\Scripts\python.exe projeto.py webcam --original

# Menor frequência de análise para computadores mais lentos
.\.venv\Scripts\python.exe projeto.py webcam --intervalo 1.0

# Alterar o limiar de aceitação
.\.venv\Scripts\python.exe projeto.py webcam --limiar 0.75
```

Se a câmera não abrir, feche outros aplicativos que estejam usando-a e confira a permissão de câmera para aplicativos da área de trabalho no Windows. O código tenta DirectShow e MSMF e verifica se recebe imagens antes de iniciar.

## Dados e avaliação

Utilizei uma seleção do dataset acadêmico [Office-Home](https://huggingface.co/datasets/Voxel51/Office-Home), com as classes originais `Pen` e `Bottle`, além de outras categorias para compor `outro`. O script seleciona os domínios Real World e Product e remove duplicatas exatas. Fontes e hashes ficam registrados em `dados/fontes.json`.

Na execução inicial, o conjunto ficou com **368 imagens**:

| Classe | Treino | Validação | Teste |
|---|---:|---:|---:|
| Caneta | 75 | 25 | 25 |
| Garrafa | 72 | 23 | 23 |
| Outro | 75 | 25 | 25 |

A regularização do classificador é selecionada pelo F1 macro da validação. A avaliação final gera acurácia, precisão, recall, F1 por classe e matriz de confusão no teste.

O primeiro modelo obteve 100% de acurácia nas 73 imagens de teste da internet, mas apresentou dificuldades na webcam. Por isso, acrescentei a calibração local. Esse resultado não representa a acurácia da webcam: o modelo calibrado precisa ser avaliado com novas poses e exemplares que não participaram da coleta.

A divisão do dataset é aleatória e não garante independência por objeto físico ou sessão. A inspeção por amostragem também encontrou uma ilustração entre os domínios selecionados. As condições de uso acadêmico estão em [FONTES_IMAGENS.md](FONTES_IMAGENS.md).

## Estrutura do projeto

```text
AprendizagemMaquina/
├── projeto.py           # Captura, treinamento, calibração e reconhecimento
├── baixar_imagens.py    # Download e organização do dataset
├── requirements.txt     # Dependências
├── PASSO_A_PASSO.md      # Guia completo de comandos
├── RELATORIO.md          # Fundamentação e avaliação
├── FONTES_IMAGENS.md     # Origem e condições de uso das imagens
├── dados/               # Gerado localmente; ignorado pelo Git
└── artefatos/           # Modelos e métricas locais; ignorado pelo Git
```

O ambiente `.venv`, as imagens e os modelos treinados não acompanham o repositório. Em outro computador, é necessário preparar o ambiente e gerar os modelos seguindo as etapas acima. Os pesos do CLIP ficam no cache da Hugging Face e não estão incluídos no arquivo joblib. Após os downloads iniciais, o reconhecimento pode funcionar sem internet usando os arquivos em cache.

## Limitações

O sistema classifica a região central inteira, sem localizar ou separar vários objetos ao mesmo tempo. Objetos pequenos, reflexos, fundos diferentes e iluminação inadequada podem causar erros. A classe `outro` representa os exemplos aprendidos e não cobre todos os objetos desconhecidos. As probabilidades exibidas e o limiar de aceitação não garantem acerto.

## Documentação

- [Passo a passo completo](PASSO_A_PASSO.md): comandos, coleta manual e solução de problemas.
- [Relatório acadêmico](RELATORIO.md): comparação de abordagens, treinamento e avaliação.
- [Fontes das imagens](FONTES_IMAGENS.md): dataset, atribuição e condições de uso.
