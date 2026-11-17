from pathlib import Path
import csv
import shutil
import zipfile
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PASTA_DADOS = Path("dados")
PASTA_CACHE = PASTA_DADOS / "cache_npz" / "teste"
PASTA_CHECKPOINTS = Path("checkpoints")
PASTA_RESULTADOS = Path("resultados_embeddings")

ARQUIVO_TESTE = PASTA_DADOS / "Dataset5_raw_test.npz"
ARQUIVO_MODELO = PASTA_CHECKPOINTS / "resnet18_melhor.pth"

NOMES_CLASSES = [
    "COVID-19",
    "Lung Opacity",
    "Normal",
    "Viral Pneumonia",
    "Tuberculosis",
]

NUM_CLASSES = len(NOMES_CLASSES)
NUM_CLUSTERS = 5
TAMANHO_LOTE = 16
SEMENTE = 42


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def carregar_imagens_teste(caminho_npz):
    if not caminho_npz.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_npz}. Verifique a pasta 'dados'."
        )

    PASTA_CACHE.mkdir(parents=True, exist_ok=True)
    caminho_imagens = PASTA_CACHE / "image.npy"

    with zipfile.ZipFile(caminho_npz, "r") as arquivo_zip:
        membro_imagens = next(
            (
                nome
                for nome in arquivo_zip.namelist()
                if Path(nome).name == "image.npy"
            ),
            None,
        )

        if membro_imagens is None:
            raise KeyError(
                "O arquivo NPZ não contém 'image.npy'."
            )

        tamanho_interno = arquivo_zip.getinfo(membro_imagens).file_size

        if (
            not caminho_imagens.exists()
            or caminho_imagens.stat().st_size != tamanho_interno
        ):
            with arquivo_zip.open(membro_imagens, "r") as origem:
                with caminho_imagens.open("wb") as destino:
                    shutil.copyfileobj(
                        origem,
                        destino,
                        length=1024 * 1024,
                    )

    with np.load(caminho_npz, allow_pickle=False) as dados:
        chaves = set(dados.files)

        if "image_label" not in chaves:
            raise KeyError(
                f"Chave 'image_label' não encontrada. Chaves disponíveis: {sorted(chaves)}"
            )

        rotulos = np.asarray(dados["image_label"]).reshape(-1)

    imagens = np.load(
        caminho_imagens,
        mmap_mode="r",
        allow_pickle=False,
    )

    rotulos = rotulos.astype(np.int64)

    if imagens.shape[0] != rotulos.shape[0]:
        raise ValueError(
            "A quantidade de imagens não corresponde à quantidade de rótulos."
        )

    if imagens.ndim != 4:
        raise ValueError(
            f"Formato de imagens inesperado: {imagens.shape}."
        )

    if not np.isin(rotulos, np.arange(NUM_CLASSES)).all():
        raise ValueError(
            f"Foram encontrados rótulos fora do intervalo esperado: {np.unique(rotulos)}"
        )

    return imagens, rotulos


class RadiografiasTeste(Dataset):
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
                raise ValueError(
                    f"Formato de imagem não reconhecido: {imagem.shape}"
                )
        elif imagem.ndim != 2:
            raise ValueError(
                f"Formato de imagem não reconhecido: {imagem.shape}"
            )

        imagem = imagem.astype(np.float32, copy=False)

        if not np.isfinite(imagem).all():
            raise ValueError(
                f"A imagem de índice {indice} contém valores inválidos."
            )

        minimo = float(imagem.min())
        maximo = float(imagem.max())

        if minimo < 0 or maximo > 255:
            raise ValueError(
                f"Imagem de índice {indice} fora da faixa esperada de 0 a 255."
            )

        if maximo <= 1.0:
            imagem = imagem * 255.0

        imagem = np.rint(imagem).clip(0, 255).astype(np.uint8)
        imagem = np.array(imagem, copy=True)

        if self.transformacao is not None:
            imagem = self.transformacao(imagem)

        rotulo = torch.tensor(
            int(self.rotulos[indice]),
            dtype=torch.long,
        )

        return imagem, rotulo, indice


def carregar_modelo_e_remover_classificador(caminho_modelo, dispositivo):
    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo treinado não encontrado: {caminho_modelo}. "
            "Execute primeiro a atividade anterior para gerar checkpoints/resnet18_melhor.pth."
        )

    try:
        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
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

    modelo.fc = nn.Linear(
        modelo.fc.in_features,
        NUM_CLASSES,
    )

    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.fc = nn.Identity()
    modelo = modelo.to(dispositivo)
    modelo.eval()

    media = torch.as_tensor(
        checkpoint["media"],
        dtype=torch.float32,
    ).reshape(-1).tolist()

    desvio = torch.as_tensor(
        checkpoint["desvio"],
        dtype=torch.float32,
    ).reshape(-1).tolist()

    return modelo, media, desvio


def extrair_embeddings(modelo, loader, dispositivo):
    embeddings = []
    rotulos_reais = []
    indices = []

    modelo.eval()

    with torch.inference_mode():
        for imagens, rotulos, indices_lote in loader:
            imagens = imagens.to(
                dispositivo,
                non_blocking=True,
            )

            vetores = modelo(imagens)

            embeddings.append(
                vetores.cpu().numpy()
            )

            rotulos_reais.extend(
                rotulos.numpy().tolist()
            )

            indices.extend(
                indices_lote.numpy().tolist()
            )

    embeddings = np.concatenate(embeddings, axis=0)
    rotulos_reais = np.asarray(rotulos_reais, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)

    ordem = np.argsort(indices)

    return embeddings[ordem], rotulos_reais[ordem], indices[ordem]


def salvar_resultados(embeddings, rotulos_reais, clusters, indices, pca_dados, inercia, caminho):
    caminho.mkdir(parents=True, exist_ok=True)

    np.save(
        caminho / "embeddings_teste.npy",
        embeddings,
    )

    np.save(
        caminho / "clusters_kmeans.npy",
        clusters,
    )

    with (caminho / "resultado_clusters.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        campos = [
            "indice_imagem",
            "classe_real_id",
            "classe_real",
            "cluster_kmeans",
        ]

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos,
        )

        escritor.writeheader()

        for indice, classe_id, cluster in zip(
            indices,
            rotulos_reais,
            clusters,
        ):
            escritor.writerow(
                {
                    "indice_imagem": indice,
                    "classe_real_id": int(classe_id),
                    "classe_real": NOMES_CLASSES[int(classe_id)],
                    "cluster_kmeans": int(cluster),
                }
            )

    matriz = np.zeros((NUM_CLASSES, NUM_CLUSTERS), dtype=np.int64)

    for classe_id, cluster in zip(rotulos_reais, clusters):
        matriz[int(classe_id), int(cluster)] += 1

    caminho_matriz = caminho / "contingencia_classes_clusters.csv"

    with caminho_matriz.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            ["classe_real"] + [
                f"cluster_{i}" for i in range(NUM_CLUSTERS)
            ]
        )

        for nome, linha in zip(NOMES_CLASSES, matriz):
            escritor.writerow(
                [nome] + linha.astype(int).tolist()
            )

    caminho_pca = caminho / "pca_clusters.png"

    plt.figure(figsize=(10, 7))

    for cluster in range(NUM_CLUSTERS):
        mascara = clusters == cluster

        plt.scatter(
            pca_dados[mascara, 0],
            pca_dados[mascara, 1],
            s=12,
            alpha=0.65,
            label=f"Cluster {cluster}",
        )

    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("K-Means aplicado aos embeddings das radiografias de teste")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_pca, dpi=200)
    plt.close()

    ari = adjusted_rand_score(
        rotulos_reais,
        clusters,
    )

    nmi = normalized_mutual_info_score(
        rotulos_reais,
        clusters,
    )

    caminho_metricas = caminho / "metricas_clusterizacao.txt"

    with caminho_metricas.open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        arquivo.write(f"Quantidade de imagens: {len(clusters)}\n")
        arquivo.write(f"Quantidade de dimensões do embedding: {embeddings.shape[1]}\n")
        arquivo.write(f"Quantidade de clusters: {NUM_CLUSTERS}\n")
        arquivo.write(f"Inércia do K-Means: {inercia:.6f}\n")
        arquivo.write(f"Adjusted Rand Index (ARI): {ari:.6f}\n")
        arquivo.write(f"Normalized Mutual Information (NMI): {nmi:.6f}\n\n")

        for cluster in range(NUM_CLUSTERS):
            quantidade = int(np.sum(clusters == cluster))
            arquivo.write(
                f"Cluster {cluster}: {quantidade} imagens\n"
            )

    return ari, nmi, matriz


def main():
    fixar_semente(SEMENTE)

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {dispositivo}")
    print("Carregando o conjunto de teste...")

    imagens_teste, rotulos_teste = carregar_imagens_teste(
        ARQUIVO_TESTE
    )

    print(f"Imagens de teste: {len(rotulos_teste)}")

    modelo, media, desvio = carregar_modelo_e_remover_classificador(
        ARQUIVO_MODELO,
        dispositivo,
    )

    transformacao = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=media,
            std=desvio,
        ),
    ])

    dataset_teste = RadiografiasTeste(
        imagens_teste,
        rotulos_teste,
        transformacao,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=TAMANHO_LOTE,
        shuffle=False,
        num_workers=0,
        pin_memory=dispositivo.type == "cuda",
    )

    print("Extraindo embeddings da ResNet-18...")

    embeddings, rotulos_reais, indices = extrair_embeddings(
        modelo,
        loader_teste,
        dispositivo,
    )

    print(f"Formato dos embeddings: {embeddings.shape}")

    if embeddings.shape[0] < NUM_CLUSTERS:
        raise ValueError(
            "O conjunto de teste possui menos imagens do que o número de clusters."
        )

    print("Executando K-Means com K=5...")

    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=10,
        random_state=SEMENTE,
    )

    clusters = kmeans.fit_predict(embeddings)
    print("Calculando projeção PCA para visualização...")

    pca = PCA(
        n_components=2,
        random_state=SEMENTE,
    )

    pca_dados = pca.fit_transform(embeddings)

    ari, nmi, matriz = salvar_resultados(
        embeddings,
        rotulos_reais,
        clusters,
        indices,
        pca_dados,
        float(kmeans.inertia_),
        PASTA_RESULTADOS,
    )

    print("\nQuantidade de imagens em cada cluster:")

    for cluster in range(NUM_CLUSTERS):
        quantidade = int(np.sum(clusters == cluster))
        print(f"Cluster {cluster}: {quantidade}")

    print("\nDistribuição das classes reais dentro dos clusters:")
    print("Linhas: classes reais | Colunas: clusters")
    print(matriz)

    print(f"\nInércia do K-Means: {kmeans.inertia_:.6f}")
    print(f"Adjusted Rand Index (ARI): {ari:.6f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.6f}")

    print(
        f"\nResultados salvos em: {PASTA_RESULTADOS.resolve()}"
    )


if __name__ == "__main__":
    main()