from pathlib import Path
import csv
import shutil
import zipfile
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

PASTA_DADOS = Path("dados")
PASTA_CACHE = PASTA_DADOS / "cache_npz"
PASTA_CHECKPOINTS = Path("checkpoints")
PASTA_RESULTADOS = Path("resultados")

ARQUIVOS = {
    "treino": PASTA_DADOS / "Dataset5_raw_train.npz",
    "validacao": PASTA_DADOS / "Dataset5_raw_val.npz",
    "teste": PASTA_DADOS / "Dataset5_raw_test.npz",
}

NOMES_CLASSES = [
    "COVID-19",
    "Lung Opacity",
    "Normal",
    "Viral Pneumonia",
    "Tuberculosis",
]

NUM_CLASSES = len(NOMES_CLASSES)
TAMANHO_LOTE = 16
NUM_EPOCAS = 20
PACIENCIA = 5
TAXA_APRENDIZADO = 0.001
PESO_DECAY = 0.0001
SEMENTE = 42


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def preparar_arquivo_npz(caminho_npz, nome_conjunto):
    if not caminho_npz.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_npz}. Coloque os três arquivos NPZ dentro da pasta 'dados'."
        )

    pasta_cache = PASTA_CACHE / nome_conjunto
    pasta_cache.mkdir(parents=True, exist_ok=True)
    caminho_imagens = pasta_cache / "image.npy"

    with zipfile.ZipFile(caminho_npz, "r") as arquivo_zip:
        membros = arquivo_zip.namelist()
        membro_imagens = next(
            (nome for nome in membros if Path(nome).name == "image.npy"),
            None,
        )

        if membro_imagens is None:
            raise KeyError(
                f"O arquivo {caminho_npz.name} não contém a matriz 'image.npy'. Conteúdo encontrado: {membros}"
            )

        informacao = arquivo_zip.getinfo(membro_imagens)

        if not caminho_imagens.exists() or caminho_imagens.stat().st_size != informacao.file_size:
            with arquivo_zip.open(membro_imagens, "r") as origem:
                with caminho_imagens.open("wb") as destino:
                    shutil.copyfileobj(origem, destino, length=1024 * 1024)

    with np.load(caminho_npz, allow_pickle=False) as dados:
        chaves = set(dados.files)
        if "image_label" not in chaves:
            raise KeyError(
                f"O arquivo {caminho_npz.name} não contém a chave 'image_label'. Chaves encontradas: {sorted(chaves)}"
            )
        rotulos = np.asarray(dados["image_label"]).reshape(-1)

    imagens = np.load(caminho_imagens, mmap_mode="r", allow_pickle=False)
    rotulos = rotulos.astype(np.int64)

    if imagens.shape[0] != rotulos.shape[0]:
        raise ValueError(
            f"O arquivo {caminho_npz.name} possui {imagens.shape[0]} imagens e {rotulos.shape[0]} rótulos."
        )

    if imagens.ndim != 4:
        raise ValueError(
            f"Formato de imagens inesperado em {caminho_npz.name}: {imagens.shape}. Era esperado (N, H, W, C)."
        )

    if imagens.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Quantidade de canais inesperada em {caminho_npz.name}: {imagens.shape[-1]}."
        )

    if not np.isin(rotulos, np.arange(NUM_CLASSES)).all():
        raise ValueError(
            f"Foram encontrados rótulos fora do intervalo 0 a {NUM_CLASSES - 1}: {np.unique(rotulos)}"
        )

    return imagens, rotulos


class RadiografiasDataset(Dataset):
    def __init__(self, imagens, rotulos, transformacao=None):
        self.imagens = imagens
        self.rotulos = rotulos
        self.transformacao = transformacao

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, indice):
        imagem = np.asarray(self.imagens[indice])

        if imagem.ndim == 3:
            if imagem.shape[-1] in (1, 3, 4):
                imagem = imagem[:, :, 0]
            elif imagem.shape[0] in (1, 3, 4):
                imagem = imagem[0, :, :]
            else:
                raise ValueError(f"Formato de imagem não reconhecido: {imagem.shape}")
        elif imagem.ndim != 2:
            raise ValueError(f"Formato de imagem não reconhecido: {imagem.shape}")

        imagem = imagem.astype(np.float32, copy=False)

        if not np.isfinite(imagem).all():
            raise ValueError(f"A imagem de índice {indice} contém valores inválidos.")

        minimo = float(imagem.min())
        maximo = float(imagem.max())

        if minimo < 0 or maximo > 255:
            raise ValueError(
                f"A imagem de índice {indice} possui valores fora da faixa esperada de 0 a 255: {minimo} a {maximo}."
            )

        if maximo <= 1.0:
            imagem = imagem * 255.0

        imagem = np.rint(imagem).clip(0, 255).astype(np.uint8)

        if self.transformacao is not None:
            imagem = self.transformacao(imagem)

        rotulo = torch.tensor(int(self.rotulos[indice]), dtype=torch.long)
        return imagem, rotulo


def calcular_estatisticas(loader):
    soma = torch.zeros(1, dtype=torch.float64)
    soma_quadrados = torch.zeros(1, dtype=torch.float64)
    total_pixels = 0

    for imagens, _ in loader:
        imagens = imagens.to(torch.float64)
        soma += imagens.sum(dim=(0, 2, 3))
        soma_quadrados += imagens.square().sum(dim=(0, 2, 3))
        total_pixels += imagens.shape[0] * imagens.shape[2] * imagens.shape[3]

    media = soma / total_pixels
    variancia = soma_quadrados / total_pixels - media.square()
    desvio = torch.sqrt(variancia.clamp(min=1e-12))

    if not torch.isfinite(media).all() or not torch.isfinite(desvio).all():
        raise ValueError("Não foi possível calcular estatísticas válidas para o conjunto de treinamento.")

    if (desvio <= 1e-8).any():
        raise ValueError("O desvio-padrão calculado é zero ou muito pequeno.")

    return media.float(), desvio.float()


def executar_epoca(modelo, loader, criterio, dispositivo, otimizador=None):
    treinamento = otimizador is not None
    modelo.train(treinamento)
    perda_total = 0.0
    acertos = 0
    total = 0

    contexto = torch.enable_grad() if treinamento else torch.inference_mode()

    with contexto:
        for imagens, rotulos in loader:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            if treinamento:
                otimizador.zero_grad(set_to_none=True)

            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

            if treinamento:
                perda.backward()
                otimizador.step()

            tamanho_lote_atual = rotulos.size(0)
            perda_total += perda.item() * tamanho_lote_atual
            previsoes = saidas.argmax(dim=1)
            acertos += (previsoes == rotulos).sum().item()
            total += tamanho_lote_atual

    return perda_total / total, acertos / total


def avaliar_modelo(modelo, loader, dispositivo):
    modelo.eval()
    rotulos_reais = []
    rotulos_previstos = []

    with torch.inference_mode():
        for imagens, rotulos in loader:
            imagens = imagens.to(dispositivo, non_blocking=True)
            saidas = modelo(imagens)
            previsoes = saidas.argmax(dim=1)
            rotulos_reais.extend(rotulos.numpy().tolist())
            rotulos_previstos.extend(previsoes.cpu().numpy().tolist())

    return rotulos_reais, rotulos_previstos


def salvar_historico(historico, caminho):
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(historico[0].keys()))
        escritor.writeheader()
        escritor.writerows(historico)


def main():
    fixar_semente(SEMENTE)
    PASTA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = dispositivo.type == "cuda"

    print(f"Dispositivo: {dispositivo}")
    print("Preparando os arquivos do dataset...")

    imagens_treino, rotulos_treino = preparar_arquivo_npz(ARQUIVOS["treino"], "treino")
    imagens_validacao, rotulos_validacao = preparar_arquivo_npz(ARQUIVOS["validacao"], "validacao")
    imagens_teste, rotulos_teste = preparar_arquivo_npz(ARQUIVOS["teste"], "teste")

    print(f"Treino: {len(rotulos_treino)} imagens")
    print(f"Validação: {len(rotulos_validacao)} imagens")
    print(f"Teste: {len(rotulos_teste)} imagens")

    contagens = np.bincount(rotulos_treino, minlength=NUM_CLASSES)
    print("\nDistribuição das classes no treinamento:")
    for indice, nome in enumerate(NOMES_CLASSES):
        print(f"{nome}: {contagens[indice]}")

    transformacao_base = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset_estatisticas = RadiografiasDataset(
        imagens_treino,
        rotulos_treino,
        transformacao_base,
    )

    loader_estatisticas = DataLoader(
        dataset_estatisticas,
        batch_size=TAMANHO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print("\nCalculando média e desvio-padrão somente com o conjunto de treinamento...")
    media, desvio = calcular_estatisticas(loader_estatisticas)
    print(f"Média do treinamento: {media.item():.8f}")
    print(f"Desvio-padrão do treinamento: {desvio.item():.8f}")

    transformacao_treino = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=media.tolist(), std=desvio.tolist()),
    ])

    transformacao_avaliacao = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=media.tolist(), std=desvio.tolist()),
    ])

    dataset_treino = RadiografiasDataset(imagens_treino, rotulos_treino, transformacao_treino)
    dataset_validacao = RadiografiasDataset(imagens_validacao, rotulos_validacao, transformacao_avaliacao)
    dataset_teste = RadiografiasDataset(imagens_teste, rotulos_teste, transformacao_avaliacao)

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=TAMANHO_LOTE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    loader_validacao = DataLoader(
        dataset_validacao,
        batch_size=TAMANHO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=TAMANHO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    modelo = models.resnet18(weights=None)
    modelo.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    modelo.fc = nn.Linear(modelo.fc.in_features, NUM_CLASSES)
    modelo = modelo.to(dispositivo)

    pesos = len(rotulos_treino) / (NUM_CLASSES * np.maximum(contagens, 1))
    pesos = torch.tensor(pesos, dtype=torch.float32, device=dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)

    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO,
        weight_decay=PESO_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador,
        mode="min",
        factor=0.5,
        patience=2,
    )

    caminho_modelo = PASTA_CHECKPOINTS / "resnet18_melhor.pth"
    historico = []
    melhor_perda_validacao = float("inf")
    epocas_sem_melhora = 0

    print("\nIniciando treinamento...")

    for epoca in range(NUM_EPOCAS):
        perda_treino, acuracia_treino = executar_epoca(
            modelo,
            loader_treino,
            criterio,
            dispositivo,
            otimizador,
        )

        perda_validacao, acuracia_validacao = executar_epoca(
            modelo,
            loader_validacao,
            criterio,
            dispositivo,
        )

        scheduler.step(perda_validacao)
        taxa_atual = otimizador.param_groups[0]["lr"]

        linha = {
            "epoca": epoca + 1,
            "perda_treino": perda_treino,
            "acuracia_treino": acuracia_treino,
            "perda_validacao": perda_validacao,
            "acuracia_validacao": acuracia_validacao,
            "taxa_aprendizado": taxa_atual,
        }
        historico.append(linha)

        print(
            f"Época {epoca + 1:02d}/{NUM_EPOCAS} | "
            f"Loss treino: {perda_treino:.4f} | "
            f"Acurácia treino: {acuracia_treino:.4f} | "
            f"Loss validação: {perda_validacao:.4f} | "
            f"Acurácia validação: {acuracia_validacao:.4f} | "
            f"LR: {taxa_atual:.6f}"
        )

        if perda_validacao < melhor_perda_validacao:
            melhor_perda_validacao = perda_validacao
            epocas_sem_melhora = 0
            torch.save(
                {
                    "model_state_dict": modelo.state_dict(),
                    "media": media,
                    "desvio": desvio,
                    "nomes_classes": NOMES_CLASSES,
                    "epoca": epoca + 1,
                    "perda_validacao": perda_validacao,
                },
                caminho_modelo,
            )
        else:
            epocas_sem_melhora += 1

        if epocas_sem_melhora >= PACIENCIA:
            print("Early stopping ativado.")
            break

    salvar_historico(historico, PASTA_RESULTADOS / "historico_treinamento.csv")

    checkpoint = torch.load(caminho_modelo, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nMelhor modelo: época {checkpoint['epoca']}")
    print("Avaliando o conjunto de teste...")

    y_true, y_pred = avaliar_modelo(modelo, loader_teste, dispositivo)

    relatorio_texto = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=NOMES_CLASSES,
        digits=4,
        zero_division=0,
    )

    relatorio_dados = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=NOMES_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\nRelatório de classificação no conjunto de teste:")
    print(relatorio_texto)

    caminho_relatorio = PASTA_RESULTADOS / "metricas_teste.txt"
    caminho_relatorio.write_text(relatorio_texto, encoding="utf-8")

    caminho_metricas_csv = PASTA_RESULTADOS / "metricas_por_classe.csv"
    with caminho_metricas_csv.open("w", newline="", encoding="utf-8") as arquivo:
        campos = ["classe", "precision", "recall", "f1-score", "support"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for nome in NOMES_CLASSES:
            valores = relatorio_dados[nome]
            escritor.writerow(
                {
                    "classe": nome,
                    "precision": valores["precision"],
                    "recall": valores["recall"],
                    "f1-score": valores["f1-score"],
                    "support": valores["support"],
                }
            )

    matriz = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.savetxt(
        PASTA_RESULTADOS / "matriz_confusao.csv",
        matriz,
        delimiter=",",
        fmt="%d",
        header=",".join(NOMES_CLASSES),
        comments="",
    )

    print(f"\nModelo salvo em: {caminho_modelo}")
    print(f"Resultados salvos em: {PASTA_RESULTADOS.resolve()}")


if __name__ == "__main__":
    main()