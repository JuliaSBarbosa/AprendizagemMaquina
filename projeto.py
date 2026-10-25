"""Captura, treinamento e reconhecimento de caneta/garrafa com CLIP."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic, time_ns

CLASSES = ("caneta", "garrafa", "outro")
MODELO = "openai/clip-vit-base-patch32"
RAIZ = Path(__file__).resolve().parent


def camera(indice):
    import cv2
    # No Windows, algumas webcams abrem com MSMF, mas não entregam frames.
    # DirectShow é tentado primeiro; validamos leitura, não apenas isOpened().
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(indice, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(15):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return cap
        cap.release()
    raise RuntimeError(
        f"A camera {indice} nao entregou imagens. Feche outras janelas deste projeto, "
        "Camera/Teams/Zoom e tente --camera 1. Confira tambem a permissao de camera "
        "para aplicativos da area de trabalho nas configuracoes do Windows.")


def recorte(frame, proporcao=0.8):
    # A mesma região central é usada na coleta e na inferência.
    h, w = frame.shape[:2]
    lado = max(1, int(min(h, w) * proporcao))
    x, y = (w - lado) // 2, (h - lado) // 2
    return frame[y:y + lado, x:x + lado], (x, y, lado)


def capturar(args):
    import cv2
    pasta = RAIZ / "dados" / args.divisao / args.classe
    pasta.mkdir(parents=True, exist_ok=True)
    cap = camera(args.camera)
    total, ultima = 0, -float("inf")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler imagem da webcam.")
            roi, (x, y, lado) = recorte(frame)
            # Copiar antes de desenhar evita gravar o texto nas imagens.
            imagem = roi.copy()
            cv2.rectangle(frame, (x, y), (x + lado, y + lado), (0, 255, 0), 2)
            cv2.putText(frame, f"{args.classe}: {total} | ESPACO salva | Q sai",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Coleta", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == 32 and monotonic() - ultima >= 0.5:
                if not cv2.imwrite(str(pasta / f"{time_ns()}.jpg"), imagem):
                    raise RuntimeError("Nao foi possivel salvar a imagem.")
                total += 1
                ultima = monotonic()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"{total} imagens salvas em {pasta}")


class Extrator:
    def __init__(self, nome=MODELO):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(nome, use_fast=False)
        self.model = CLIPModel.from_pretrained(nome).to(self.device).eval()

    def imagens(self, imagens):
        entradas = self.processor(images=imagens, return_tensors="pt")
        entradas = {k: v.to(self.device) for k, v in entradas.items()}
        # Congelamos CLIP: só o classificador posterior será treinado.
        with self.torch.inference_mode():
            vetores = self.model.get_image_features(**entradas)
            vetores = vetores / vetores.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return vetores.cpu().numpy()


def inventario(divisao):
    itens = []
    for classe in CLASSES:
        pasta = RAIZ / "dados" / divisao / classe
        arquivos = sorted(p for p in pasta.glob("*")
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(arquivos) < 5:
            raise ValueError(f"Colete ao menos 5 imagens em {pasta}; veja README para a meta.")
        itens.extend((p, classe) for p in arquivos)
    return itens


def extrair(extrator, itens):
    import numpy as np
    from PIL import Image
    lotes = []
    for inicio in range(0, len(itens), 16):
        imagens = []
        for caminho, _ in itens[inicio:inicio + 16]:
            with Image.open(caminho) as img:
                imagens.append(img.convert("RGB"))
        lotes.append(extrator.imagens(imagens))
        print(f"Extraidas {min(inicio + 16, len(itens))}/{len(itens)} imagens")
    return np.concatenate(lotes), [classe for _, classe in itens]


def treinar(_args):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    # Validamos todos os diretórios antes de baixar/carregar os pesos.
    listas = {d: inventario(d) for d in ("treino", "validacao", "teste")}
    extrator = Extrator()
    dados = {d: extrair(extrator, itens) for d, itens in listas.items()}
    x, y = dados["treino"]
    xv, yv = dados["validacao"]
    melhor, melhor_f1, melhor_c = None, -1, None
    # A validação escolhe a regularização; o teste não participa da seleção.
    for c in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(x, y)
        f1 = f1_score(yv, clf.predict(xv), average="macro", zero_division=0)
        if f1 > melhor_f1:
            melhor, melhor_f1, melhor_c = clf, f1, c
    xt, yt = dados["teste"]
    pred = melhor.predict(xt)
    relatorio = {"modelo_base": MODELO, "C": melhor_c,
                 "f1_macro_validacao": melhor_f1,
                 "acuracia_teste": accuracy_score(yt, pred),
                 "metricas_teste": classification_report(yt, pred, labels=list(CLASSES),
                                                         output_dict=True, zero_division=0),
                 "ordem_matriz": list(CLASSES),
                 "matriz_confusao": confusion_matrix(yt, pred, labels=list(CLASSES)).tolist(),
                 "quantidade_imagens": {d: len(v) for d, v in listas.items()}}
    pasta = RAIZ / "artefatos"
    pasta.mkdir(exist_ok=True)
    joblib.dump({"classificador": melhor, "modelo_base": MODELO}, pasta / "modelo.joblib")
    (pasta / "avaliacao.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


def calibrar(args):
    """Aprende o cenário real sem alterar o modelo e a avaliação da internet."""
    import cv2
    import joblib
    import numpy as np
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    extrator = Extrator()
    cap = camera(args.camera)
    vetores, rotulos = [], []
    pasta = RAIZ / "dados" / "calibracao" / str(time_ns())
    print("Para cada classe, posicione o objeto e pressione ESPACO. Q cancela.", flush=True)
    try:
        for classe in CLASSES:
            contando, total, proxima = False, 0, 0
            while total < args.fotos:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Falha ao ler a webcam.")
                roi, (x, y, lado) = recorte(frame, args.area)
                # Captura limpa antes de desenhar as instruções.
                imagem = roi.copy()
                agora = monotonic()
                if contando and agora >= proxima:
                    rgb = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
                    vetores.append(extrator.imagens([rgb])[0])
                    rotulos.append(classe)
                    destino = pasta / classe
                    destino.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destino / f"{total:03}.jpg"), imagem):
                        raise RuntimeError("Falha ao salvar imagem de calibracao.")
                    total += 1
                    proxima = monotonic() + 0.7
                texto = (f"{classe}: {total}/{args.fotos} - varie angulo e distancia"
                         if contando else f"Mostre {classe} e pressione ESPACO")
                cv2.rectangle(frame, (x, y), (x + lado, y + lado), (0, 255, 0), 2)
                cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Calibracao - Q cancela", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    print("Calibracao cancelada; modelo anterior preservado.")
                    return
                if tecla == 32 and not contando:
                    contando, proxima = True, monotonic() + 2
    finally:
        cap.release()
        cv2.destroyAllWindows()
    clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(np.asarray(vetores), rotulos)
    destino = RAIZ / "artefatos" / "modelo_local.joblib"
    destino.parent.mkdir(exist_ok=True)
    joblib.dump({"classificador": clf, "modelo_base": MODELO, "area": args.area,
                 "origem": "calibracao_webcam", "imagens": str(pasta)}, destino)
    print(f"Modelo local salvo em {destino}. Teste com novas poses e objetos.")
    print("Esta calibracao nao possui avaliacao independente; nao implica uma acuracia medida.")


def webcam(args):
    import cv2
    import joblib
    from PIL import Image
    import numpy as np
    local = RAIZ / "artefatos" / "modelo_local.joblib"
    caminho = local if local.exists() and not args.original else RAIZ / "artefatos" / "modelo.joblib"
    if not caminho.exists():
        raise ValueError("Treine primeiro: python projeto.py treinar")
    # Carregue apenas artefatos locais confiáveis: joblib usa pickle.
    pacote = joblib.load(caminho)
    print(f"Usando {caminho.name}. +/- ajustam area; Q sai.", flush=True)
    clf = pacote["classificador"]
    extrator = Extrator(pacote["modelo_base"])
    cap = camera(args.camera)
    texto, ultima = "Analisando...", -float("inf")
    area = args.area if args.area is not None else pacote.get("area", 0.8)
    media, detalhes = None, ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Falha ao ler a webcam.")
            roi, (x, y, lado) = recorte(frame, area)
            if monotonic() - ultima >= args.intervalo:
                rgb = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                probs = clf.predict_proba(extrator.imagens([rgb]))[0]
                # Média exponencial evita trocar de classe a cada frame.
                media = probs if media is None else 0.65 * probs + 0.35 * media
                indice = int(np.argmax(media))
                classe, score = clf.classes_[indice], float(media[indice])
                detalhes = " | ".join(f"{c}: {p:.0%}" for c, p in zip(clf.classes_, media))
                texto = (f"{classe}: {score:.0%}" if score >= args.limiar
                         else "Incerto - reposicione o objeto")
                ultima = monotonic()
            cv2.rectangle(frame, (x, y), (x + lado, y + lado), (0, 255, 0), 2)
            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, detalhes, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, "+/- area | Q sai", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("Caneta ou garrafa? Q para sair", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla in (ord("+"), ord("="), ord("-")):
                area = max(0.25, min(1.0, area + (0.05 if tecla != ord("-") else -0.05)))
                media, ultima = None, -float("inf")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    coleta = comandos.add_parser("capturar", help="Salvar imagens pela webcam")
    coleta.add_argument("--classe", choices=CLASSES, required=True)
    coleta.add_argument("--divisao", choices=("treino", "validacao", "teste"), required=True)
    coleta.add_argument("--camera", type=int, default=0)
    coleta.set_defaults(func=capturar)
    treino = comandos.add_parser("treinar", help="Treinar e avaliar o classificador")
    treino.set_defaults(func=treinar)
    calibracao = comandos.add_parser("calibrar", help="Coleta automatica e treino no seu ambiente")
    calibracao.add_argument("--camera", type=int, default=0)
    calibracao.add_argument("--fotos", type=int, default=20)
    calibracao.add_argument("--area", type=float, default=0.6)
    calibracao.set_defaults(func=calibrar)
    uso = comandos.add_parser("webcam", help="Reconhecer objetos")
    uso.add_argument("--camera", type=int, default=0)
    uso.add_argument("--limiar", type=float, default=0.65)
    uso.add_argument("--intervalo", type=float, default=0.4)
    uso.add_argument("--area", type=float, default=None)
    uso.add_argument("--original", action="store_true", help="Usar modelo da internet mesmo apos calibrar")
    uso.set_defaults(func=webcam)
    args = parser.parse_args()
    if getattr(args, "area", None) is not None and not 0.25 <= args.area <= 1:
        parser.error("Use area entre 0.25 e 1.")
    if args.comando == "calibrar" and args.fotos < 5:
        parser.error("Use pelo menos 5 fotos por classe.")
    if args.comando == "webcam" and (not 0 <= args.limiar <= 1 or args.intervalo <= 0):
        parser.error("Use limiar entre 0 e 1 e intervalo positivo.")
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError) as erro:
        parser.exit(1, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
