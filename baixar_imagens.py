"""Baixa fotografias rotuladas do Office-Home para uso acadêmico."""
import concurrent.futures
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from time import sleep

RAIZ = Path(__file__).resolve().parent
BASE = "https://huggingface.co/datasets/Voxel51/Office-Home/resolve/main/"


def obter(url):
    for tentativa in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resposta:
                return resposta.read()
        except OSError:
            if tentativa == 3:
                raise
            sleep(tentativa + 1)


def main():
    from PIL import Image
    from io import BytesIO
    amostras = json.loads(obter(BASE + "samples.json"))["samples"]
    grupos = {"caneta": ["Pen"], "garrafa": ["Bottle"],
              "outro": ["Mug", "Keyboard", "Notebook", "Mouse"]}
    rng = random.Random(42)
    tarefas = []
    for classe, rotulos in grupos.items():
        candidatos = [s for s in amostras if s["object"]["label"] in rotulos
                      and s["domain"]["label"] in ("Real World", "Product")]
        rng.shuffle(candidatos)
        # Mesma quantidade em todas as classes, limitada pela classe caneta.
        for i, s in enumerate(candidatos[:125]):
            divisao = "treino" if i < 75 else "validacao" if i < 100 else "teste"
            nome = "officehome_" + s["filepath"].replace("/", "_")
            destino = RAIZ / "dados" / divisao / classe / nome
            tarefas.append((s, destino, classe, divisao))

    def baixar(tarefa):
        s, destino, classe, divisao = tarefa
        conteudo = destino.read_bytes() if destino.exists() else obter(BASE + s["filepath"])
        # Verifica que a resposta é uma imagem íntegra antes de salvar.
        with Image.open(BytesIO(conteudo)) as imagem:
            imagem.verify()
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists():
            destino.write_bytes(conteudo)
        return {"arquivo": str(destino.relative_to(RAIZ)), "classe": classe,
                "divisao": divisao, "rotulo_original": s["object"]["label"],
                "dominio": s["domain"]["label"], "url": BASE + s["filepath"],
                "sha256": hashlib.sha256(conteudo).hexdigest()}

    registros = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for registro in pool.map(baixar, tarefas):
            registros.append(registro)
            if len(registros) % 25 == 0:
                print(f"{len(registros)}/{len(tarefas)} imagens verificadas", flush=True)
    # Remove apenas cópias exatas selecionadas por este script, evitando vazamento.
    vistos, unicos = set(), []
    for r in registros:
        if r["sha256"] in vistos:
            (RAIZ / r["arquivo"]).unlink()
        else:
            vistos.add(r["sha256"])
            unicos.append(r)
    (RAIZ / "dados" / "fontes.json").write_text(
        json.dumps(unicos, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print(dict(Counter((r["divisao"], r["classe"]) for r in unicos)))
    print("Fontes e hashes registrados em dados/fontes.json")


if __name__ == "__main__":
    main()
