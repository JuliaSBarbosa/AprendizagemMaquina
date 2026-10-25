# Imagens obtidas da internet

Fonte: [Office-Home, versão disponibilizada por Voxel51](https://huggingface.co/datasets/Voxel51/Office-Home).
Página dos autores: https://www.hemanthdv.org/officeHomeDataset.html

O dataset é disponibilizado para pesquisa acadêmica e educação não comercial sob o aviso de fair use dos autores; isso não significa que todas as imagens tenham licença aberta individual. Preserve esta atribuição na entrega acadêmica.

Referência: Venkateswara, H.; Eusebio, J.; Chakraborty, S.; Panchanathan, S. Deep Hashing Network for Unsupervised Domain Adaptation. CVPR, 2017.

O script `baixar_imagens.py` seleciona fotografias dos domínios Real World e Product, excluindo Art e Clipart. Mapeia Pen para caneta, Bottle para garrafa, e Mug/Keyboard/Notebook/Mouse para outro. Seleciona 125 imagens por classe e divide em 75 de treino, 25 de validação e 25 de teste. Remove duplicatas de conteúdo exato; por isso a quantidade final pode ser menor. Os URLs, rótulos originais e hashes SHA-256 ficam em `dados/fontes.json`.

A separação é aleatória com semente 42, não por identidade do objeto ou sessão: o dataset não fornece esses grupos nesta seleção. Imagens visualmente parecidas ainda podem ocorrer entre conjuntos. A seleção usa os rótulos originais; uma inspeção por amostragem encontrou uma ilustração de canetas mesmo fora de Art/Clipart, portanto os domínios não garantem somente fotografias. A validação automática verifica integridade dos arquivos e duplicatas exatas, não a correção visual de todos os rótulos. O teste mede desempenho em imagens da internet e não comprova desempenho na webcam. Faça também uma avaliação local com exemplares reais que não participaram do treinamento.

Para reproduzir o download, após instalar Pillow:

```powershell
.\.venv\Scripts\python.exe baixar_imagens.py
```

## Atualização do plano de dados

Esta seleção substitui o plano inicial de coleta própria para permitir começar sem fotografar manualmente. Ao entregar `RELATORIO.md`, descreva os dados como imagens do Office-Home e use as quantidades reais registradas em `dados/fontes.json` e na avaliação. Não afirme que a separação foi por sessões ou objetos físicos distintos. A coleta pela webcam continua disponível para complementar o treino, caso necessário.
