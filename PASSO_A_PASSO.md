# Passo a passo — caneta ou garrafa pela webcam

Execute os comandos no PowerShell, um por vez. Espere cada comando terminar antes de executar o próximo. Não copie o texto `PS C:\...>` do terminal. Não é necessário ativar o ambiente virtual.

## 1. Entrar na pasta

```powershell
cd "C:\Users\jb385\Documents\ChatGPT\trabalho ml"
```

Faça isso sempre que abrir um novo terminal.

## 2. Preparar o ambiente — primeira vez

O ambiente `.venv` já foi criado neste computador. Se estiver usando outro computador ou a pasta não existir, crie com Python 3.12 instalado:

```powershell
py -3.12 -m venv .venv
```

Instale as dependências:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

As instalações precisam de internet e podem demorar. Não é preciso repeti-las a cada teste. O ambiente atual foi criado com Python 3.13; não é necessário recriá-lo se já estiver funcionando.

## 3. Baixar as imagens — se ainda não estiverem em dados/

As imagens já foram baixadas neste projeto. Em uma cópia sem os dados, execute:

```powershell
.\.venv\Scripts\python.exe baixar_imagens.py
```

O script organiza caneta, garrafa e outros objetos nas divisões de treino, validação e teste. Registra a origem em `dados/fontes.json`. Consulte `FONTES_IMAGENS.md` para atribuição e condições de uso acadêmico.

## 4. Treinar com as imagens da internet

```powershell
.\.venv\Scripts\python.exe projeto.py treinar
```

O primeiro uso do CLIP precisa de internet para baixar os pesos. Aguarde a exibição dos resultados no terminal. São gerados:

- `artefatos/modelo.joblib`: classificador treinado nas imagens da internet.
- `artefatos/avaliacao.json`: métricas do teste com imagens da internet.

Se esses arquivos já existem e você não alterou os dados, não é necessário repetir esta etapa. Rodar novamente substitui esses dois arquivos. Esse treinamento não substitui um modelo local de calibração que já exista.

## 5. Calibrar para sua webcam — recomendado

Feche outras janelas do projeto que estejam usando a câmera e aplicativos como Câmera, Teams ou Zoom. Execute:

```powershell
.\.venv\Scripts\python.exe projeto.py calibrar
```

Na janela da câmera:

1. Quando pedir `caneta`, coloque uma caneta inteira dentro do quadrado e pressione **Espaço** com a janela selecionada.
2. Aguarde dois segundos de preparação. Depois, o programa captura 20 imagens automaticamente. Varie levemente posição, ângulo e distância; mantenha a caneta visível.
3. Quando pedir `garrafa`, faça o mesmo com uma garrafa.
4. Quando pedir `outro`, mostre mão vazia, fundo vazio e outros objetos. Não mostre caneta nem garrafa nessa etapa.
5. Aguarde a mensagem de modelo local salvo. **Q** cancela a calibração.

O resultado fica em `artefatos/modelo_local.joblib`; as imagens ficam em `dados/calibracao/`. A calibração treina o classificador local automaticamente: **não rode `treinar` para concluir essa etapa**. Uma nova calibração substitui o modelo local anterior, preservando o modelo da internet.

Opcionalmente, capture mais exemplos por classe:

```powershell
.\.venv\Scripts\python.exe projeto.py calibrar --fotos 40
```

## 6. Testar o reconhecimento

```powershell
.\.venv\Scripts\python.exe projeto.py webcam
```

Se existir um modelo local, ele será usado automaticamente. Caso contrário, será usado o modelo da internet. O terminal informa qual foi carregado.

- Mostre um objeto por vez, inteiro dentro do quadrado verde.
- Para canetas, aproxime o objeto e use **-** para diminuir o quadrado, reduzindo o fundo analisado.
- Use **+** para aumentar o quadrado.
- Aguarde a atualização da previsão após trocar o objeto.
- Pressione **Q** com a janela selecionada para sair.

Teste caneta, garrafa, mão vazia, fundo vazio e outros objetos. Use novas posições e, se possível, exemplares diferentes dos usados na calibração. Anote acertos e erros. A avaliação da internet não mede a qualidade da calibração nem o desempenho real da webcam.

## 7. Usar novamente nos próximos dias

Com tudo instalado e calibrado, normalmente bastam:

```powershell
cd "C:\Users\jb385\Documents\ChatGPT\trabalho ml"
.\.venv\Scripts\python.exe projeto.py webcam
```

Se mudar bastante o fundo, a iluminação ou os objetos, faça uma nova calibração e teste novamente.

## 8. Se a câmera não abrir

O código tenta DirectShow e depois MSMF no Windows, verificando se recebe imagens. Se falhar, feche outros aplicativos e tente outra câmera:

```powershell
.\.venv\Scripts\python.exe projeto.py calibrar --camera 1
```

Se essa câmera funcionar, use o mesmo índice ao testar:

```powershell
.\.venv\Scripts\python.exe projeto.py webcam --camera 1
```

Se necessário, tente `--camera 2`. Confira também, nas configurações de privacidade do Windows, o acesso à câmera para aplicativos da área de trabalho. Caso continue falhando, teste a câmera no aplicativo Câmera do Windows, feche-o antes de tentar o projeto novamente e registre o erro do terminal.

## 9. Opções de teste e comparação

Comparar com o modelo original da internet:

```powershell
.\.venv\Scripts\python.exe projeto.py webcam --original
```

Começar com uma região central menor (proporção entre 0,25 e 1):

```powershell
.\.venv\Scripts\python.exe projeto.py webcam --area 0.5
```

Diminuir a frequência de análise em um computador lento:

```powershell
.\.venv\Scripts\python.exe projeto.py webcam --intervalo 1.0
```

Exigir probabilidade maior antes de exibir a classe:

```powershell
.\.venv\Scripts\python.exe projeto.py webcam --limiar 0.75
```

Um limiar maior pode produzir mais respostas `Incerto`; ele não melhora o treinamento. As probabilidades exibidas não são garantias de acerto.

## 10. Capturar fotos manualmente — opcional

Para complementar o conjunto da internet, execute o comando desejado. **Espaço** salva uma imagem e **Q** encerra. Use fotos diferentes entre treino, validação e teste, de preferência de sessões e exemplares distintos.

```powershell
.\.venv\Scripts\python.exe projeto.py capturar --classe caneta --divisao treino
.\.venv\Scripts\python.exe projeto.py capturar --classe garrafa --divisao treino
.\.venv\Scripts\python.exe projeto.py capturar --classe outro --divisao treino
.\.venv\Scripts\python.exe projeto.py capturar --classe caneta --divisao validacao
.\.venv\Scripts\python.exe projeto.py capturar --classe garrafa --divisao validacao
.\.venv\Scripts\python.exe projeto.py capturar --classe outro --divisao validacao
.\.venv\Scripts\python.exe projeto.py capturar --classe caneta --divisao teste
.\.venv\Scripts\python.exe projeto.py capturar --classe garrafa --divisao teste
.\.venv\Scripts\python.exe projeto.py capturar --classe outro --divisao teste
```

Adicione `--camera 1` a cada comando se necessário. Após alterar esse conjunto, execute `treinar` novamente e use `webcam --original` para testar o modelo resultante. As fotos de `dados/calibracao/` não são incorporadas por esse comando; pertencem ao fluxo separado de calibração local.

## 11. Consultar resultados e registrar versões para a entrega

Ler as métricas do teste da internet:

```powershell
Get-Content .\artefatos\avaliacao.json
```

Registrar as versões instaladas após validar o funcionamento:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

Em outro ambiente virtual, essas versões podem ser instaladas com:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Preencha `RELATORIO.md` com os resultados reais. Registre separadamente as métricas da internet e os testes novos pela webcam. Não apresente imagens da calibração como um teste independente.

## 12. Ver todos os comandos disponíveis

```powershell
.\.venv\Scripts\python.exe projeto.py --help
.\.venv\Scripts\python.exe projeto.py capturar --help
.\.venv\Scripts\python.exe projeto.py treinar --help
.\.venv\Scripts\python.exe projeto.py calibrar --help
.\.venv\Scripts\python.exe projeto.py webcam --help
```
