from pathlib import Path
from copy import deepcopy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
MAX_EPOCHS = 300
PATIENCE = 45

COLUNAS_REMOVER = [
    "Utilities", "Street", "Condition2", "RoofMatl", "Heating",
    "PoolQC", "Exterior1st", "SaleType", "Id"
]

MAPEAMENTOS_ORDINAIS = {
    "ExterQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Gtl": 0, "Mod": 1, "Sev": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def encontrar_csv():
    pasta = Path(__file__).resolve().parent
    opcoes = [pasta / "train.csv", pasta / "train (2).csv"]

    for caminho in opcoes:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        "Nao encontrei train.csv nem train (2).csv na pasta do script. "
        "Coloque o CSV na mesma pasta do arquivo .py."
    )


def preparar_features(X_ajuste, X_aplicar):
    ajuste = X_ajuste.copy()
    aplicar = X_aplicar.copy()

    for coluna in ["MSSubClass", "MoSold"]:
        if coluna in ajuste.columns:
            ajuste[coluna] = ajuste[coluna].astype("string")
            aplicar[coluna] = aplicar[coluna].astype("string")

    colunas_cat_originais = ajuste.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    for coluna in colunas_cat_originais:
        ajuste[coluna] = ajuste[coluna].fillna("None").astype(str)
        aplicar[coluna] = aplicar[coluna].fillna("None").astype(str)

    ordinais = [c for c in MAPEAMENTOS_ORDINAIS if c in colunas_cat_originais]
    nominais = [c for c in colunas_cat_originais if c not in ordinais]
    numericas = [c for c in ajuste.columns if c not in colunas_cat_originais]

    ajuste_num = ajuste[numericas].apply(pd.to_numeric, errors="coerce")
    aplicar_num = aplicar[numericas].apply(pd.to_numeric, errors="coerce")

    medias_imputacao = ajuste_num.mean().fillna(0.0)
    ajuste_num = ajuste_num.fillna(medias_imputacao)
    aplicar_num = aplicar_num.fillna(medias_imputacao)

    medias_z = ajuste_num.mean()
    desvios_z = ajuste_num.std(ddof=0).replace(0, 1.0).fillna(1.0)
    ajuste_num = (ajuste_num - medias_z) / desvios_z
    aplicar_num = (aplicar_num - medias_z) / desvios_z

    ord_ajuste = pd.DataFrame(index=ajuste.index)
    ord_aplicar = pd.DataFrame(index=aplicar.index)

    for coluna in ordinais:
        mapa = dict(MAPEAMENTOS_ORDINAIS[coluna])
        if "None" not in mapa:
            mapa["None"] = min(mapa.values()) - 1

        desconhecidos = (
            set(ajuste[coluna].unique()) | set(aplicar[coluna].unique())
        ) - set(mapa)
        if desconhecidos:
            raise ValueError(
                f"Categorias nao mapeadas em {coluna}: {sorted(desconhecidos)}. "
                "Revise MAPEAMENTOS_ORDINAIS."
            )

        ord_ajuste[coluna] = ajuste[coluna].map(mapa).astype(np.float64)
        ord_aplicar[coluna] = aplicar[coluna].map(mapa).astype(np.float64)

    cat_ajuste = pd.get_dummies(ajuste[nominais], dtype=np.float32)
    cat_aplicar = pd.get_dummies(aplicar[nominais], dtype=np.float32)
    cat_aplicar = cat_aplicar.reindex(columns=cat_ajuste.columns, fill_value=0)

    X_ajuste_final = pd.concat([ajuste_num, ord_ajuste, cat_ajuste], axis=1)
    X_aplicar_final = pd.concat([aplicar_num, ord_aplicar, cat_aplicar], axis=1)
    X_aplicar_final = X_aplicar_final.reindex(
        columns=X_ajuste_final.columns, fill_value=0
    )

    X_ajuste_final = X_ajuste_final.astype(np.float32)
    X_aplicar_final = X_aplicar_final.astype(np.float32)

    if not np.isfinite(X_ajuste_final.to_numpy()).all():
        raise ValueError("Ainda existem valores invalidos no conjunto de ajuste.")
    if not np.isfinite(X_aplicar_final.to_numpy()).all():
        raise ValueError("Ainda existem valores invalidos no conjunto aplicado.")

    return X_ajuste_final, X_aplicar_final


def transformar_alvo_treino(y_ajuste):
    y_log = np.log(np.asarray(y_ajuste, dtype=np.float64))
    media_log = float(y_log.mean())
    desvio_log = float(y_log.std(ddof=0))

    if not np.isfinite(desvio_log) or desvio_log == 0:
        desvio_log = 1.0

    y_z = ((y_log - media_log) / desvio_log).astype(np.float32).reshape(-1, 1)
    return y_z, media_log, desvio_log


def voltar_preco(previsoes_z, media_log, desvio_log):
    previsoes_log = (
        np.asarray(previsoes_z).reshape(-1) * desvio_log + media_log
    )
    return np.exp(previsoes_log)

def criar_normalizacao(tipo, numero_neuronios):
    tipo = tipo.lower()

    if tipo == "layer":
        return nn.LayerNorm(numero_neuronios)
    if tipo == "batch":
        return nn.BatchNorm1d(numero_neuronios)
    if tipo == "none":
        return nn.Identity()

    raise ValueError(
        "normalizacao deve ser 'none', 'layer' ou 'batch'."
    )


class BlocoDenso(nn.Module):

    def __init__(
        self,
        entrada,
        saida,
        normalizacao="none",
        dropout=0.0,
        residual=False,
    ):
        super().__init__()

        self.linear = nn.Linear(entrada, saida)
        self.normalizacao = criar_normalizacao(normalizacao, saida)
        self.ativacao = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual

        if residual:
            if entrada == saida:
                self.atalho = nn.Identity()
            else:
                self.atalho = nn.Linear(entrada, saida, bias=False)
        else:
            self.atalho = None

    def forward(self, x):
        ramo_principal = self.linear(x)
        ramo_principal = self.normalizacao(ramo_principal)

        if self.residual:
            ramo_principal = ramo_principal + self.atalho(x)

        saida = self.ativacao(ramo_principal)
        saida = self.dropout(saida)
        return saida


class RedePrecos(nn.Module):
    
    def __init__(
        self,
        numero_entradas,
        ocultas=(128, 64),
        normalizacao="none",
        dropout=0.0,
        residual=False,
    ):
        super().__init__()

        blocos = []
        anterior = numero_entradas

        for tamanho in ocultas:
            blocos.append(
                BlocoDenso(
                    entrada=anterior,
                    saida=tamanho,
                    normalizacao=normalizacao,
                    dropout=dropout,
                    residual=residual,
                )
            )
            anterior = tamanho

        self.blocos = nn.Sequential(*blocos)
        self.saida = nn.Linear(anterior, 1)

    def forward(self, x):
        x = self.blocos(x)
        return self.saida(x)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)

def treinar_com_validacao(
    X_fit,
    y_fit_z,
    X_val,
    y_val_raw,
    media_log,
    desvio_log,
    config,
    max_epocas=MAX_EPOCHS,
):
    fix_seed(SEED)

    modelo = RedePrecos(
        numero_entradas=X_fit.shape[1],
        ocultas=config["ocultas"],
        normalizacao=config["normalizacao"],
        dropout=config["dropout"],
        residual=config["residual"],
    )

    perda_fn = nn.MSELoss()
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    Xt = torch.tensor(X_fit.to_numpy(), dtype=torch.float32)
    yt = torch.tensor(y_fit_z, dtype=torch.float32)
    Xv = torch.tensor(X_val.to_numpy(), dtype=torch.float32)

    melhor_rmse = float("inf")
    melhor_epoca = 1
    melhor_estado = deepcopy(modelo.state_dict())
    sem_melhora = 0
    epoca_final = 0

    for epoca in range(1, max_epocas + 1):
        modelo.train()
        otimizador.zero_grad()

        saida = modelo(Xt)
        perda = perda_fn(saida, yt)
        perda.backward()
        otimizador.step()

        modelo.eval()
        with torch.no_grad():
            pred_val_z = modelo(Xv).cpu().numpy().reshape(-1)

        pred_val = voltar_preco(pred_val_z, media_log, desvio_log)
        rmse_val = float(np.sqrt(mean_squared_error(y_val_raw, pred_val)))

        if rmse_val < melhor_rmse - 1e-6:
            melhor_rmse = rmse_val
            melhor_epoca = epoca
            melhor_estado = deepcopy(modelo.state_dict())
            sem_melhora = 0
        else:
            sem_melhora += 1

        epoca_final = epoca
        if sem_melhora >= PATIENCE:
            break

    modelo.load_state_dict(melhor_estado)
    return modelo, melhor_rmse, melhor_epoca, epoca_final


def treinar_final(X_fit, y_fit_z, config, epocas):
    fix_seed(SEED)

    modelo = RedePrecos(
        numero_entradas=X_fit.shape[1],
        ocultas=config["ocultas"],
        normalizacao=config["normalizacao"],
        dropout=config["dropout"],
        residual=config["residual"],
    )

    perda_fn = nn.MSELoss()
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    Xt = torch.tensor(X_fit.to_numpy(), dtype=torch.float32)
    yt = torch.tensor(y_fit_z, dtype=torch.float32)

    for _ in range(epocas):
        modelo.train()
        otimizador.zero_grad()
        perda = perda_fn(modelo(Xt), yt)
        perda.backward()
        otimizador.step()

    return modelo

CONFIGURACOES = [
    {
        "nome": "baseline_sem_novos_recursos",
        "ocultas": (128, 64),
        "normalizacao": "none",
        "residual": False,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "somente_layernorm",
        "ocultas": (128, 64),
        "normalizacao": "layer",
        "residual": False,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "somente_batchnorm",
        "ocultas": (128, 64),
        "normalizacao": "batch",
        "residual": False,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "somente_residual",
        "ocultas": (128, 64),
        "normalizacao": "none",
        "residual": True,
        "dropout": 0.0,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "dropout_rede_rasa",
        "ocultas": (128, 64),
        "normalizacao": "none",
        "residual": False,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "dropout_rede_mais_profunda",
        "ocultas": (256, 128, 64),
        "normalizacao": "none",
        "residual": False,
        "dropout": 0.15,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
       
        "nome": "todos_layernorm_residual_dropout_mesma_rede",
        "ocultas": (128, 64),
        "normalizacao": "layer",
        "residual": True,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "todos_batchnorm_residual_dropout_mesma_rede",
        "ocultas": (128, 64),
        "normalizacao": "batch",
        "residual": True,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {

        "nome": "todos_layernorm_residual_dropout_rede_profunda",
        "ocultas": (128, 128, 64),
        "normalizacao": "layer",
        "residual": True,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
    {
        "nome": "todos_batchnorm_residual_dropout_rede_profunda",
        "ocultas": (128, 128, 64),
        "normalizacao": "batch",
        "residual": True,
        "dropout": 0.10,
        "lr": 0.001,
        "weight_decay": 1e-5,
    },
]


def main():
    fix_seed()
    caminho_csv = encontrar_csv()
    dados = pd.read_csv(caminho_csv)

    if "SalePrice" not in dados.columns:
        raise ValueError("A coluna SalePrice nao foi encontrada no CSV.")

    dados = dados.dropna(subset=["SalePrice"]).copy()
    y = dados["SalePrice"].astype(float)

    if (y <= 0).any():
        raise ValueError("SalePrice precisa ser positivo para aplicar ln.")

    X = dados.drop(
        columns=["SalePrice"] + COLUNAS_REMOVER,
        errors="ignore",
    ).copy()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    X_fit_raw, X_val_raw, y_fit_raw, y_val_raw = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=SEED + 1,
    )

    X_fit, X_val = preparar_features(X_fit_raw, X_val_raw)
    y_fit_z, media_log_fit, desvio_log_fit = transformar_alvo_treino(y_fit_raw)

    print("=" * 72)
    print(f"Arquivo: {caminho_csv.name}")
    print(
        f"Registros: {len(dados)} | treino externo: {len(X_train_raw)} | "
        f"teste final: {len(X_test_raw)}"
    )
    print(
        f"Treino interno: {len(X_fit_raw)} | validacao: {len(X_val_raw)} | "
        f"entradas: {X_fit.shape[1]}"
    )
    print("=" * 72)
    print("Comparacao das arquiteturas usando somente a validacao:\n")

    resultados = []

    for config in CONFIGURACOES:
        modelo, rmse_val, epoca_melhor, epocas_rodadas = treinar_com_validacao(
            X_fit=X_fit,
            y_fit_z=y_fit_z,
            X_val=X_val,
            y_val_raw=y_val_raw.to_numpy(dtype=np.float64),
            media_log=media_log_fit,
            desvio_log=desvio_log_fit,
            config=config,
        )

        resultado = {
            "experimento": config["nome"],
            "camadas_ocultas": str(config["ocultas"]),
            "normalizacao": config["normalizacao"],
            "residual": config["residual"],
            "dropout": config["dropout"],
            "learning_rate": config["lr"],
            "weight_decay": config["weight_decay"],
            "parametros_treinaveis": contar_parametros(modelo),
            "rmse_validacao": rmse_val,
            "melhor_epoca": epoca_melhor,
            "epocas_executadas": epocas_rodadas,
        }
        resultados.append(resultado)

        print(
            f"{config['nome']}: RMSE validacao = {rmse_val:,.2f} | "
            f"melhor epoca = {epoca_melhor} | "
            f"parametros = {resultado['parametros_treinaveis']:,}"
        )

    resultados_df = pd.DataFrame(resultados).sort_values(
        by="rmse_validacao",
        ascending=True,
    )

    melhor_nome = resultados_df.iloc[0]["experimento"]
    melhor_config = next(c for c in CONFIGURACOES if c["nome"] == melhor_nome)
    epocas_finais = int(resultados_df.iloc[0]["melhor_epoca"])

    pasta_saida = Path(__file__).resolve().parent
    arquivo_experimentos = pasta_saida / "comparacao_camadas_validacao.csv"
    resultados_df.to_csv(arquivo_experimentos, index=False)

    print("\nMelhor configuracao escolhida pela validacao:")
    print(melhor_config)
    print(f"Epocas para o treinamento final: {epocas_finais}")
    print(f"Tabela de experimentos salva em: {arquivo_experimentos}")

    X_train_final, X_test_final = preparar_features(X_train_raw, X_test_raw)
    y_train_z, media_log_train, desvio_log_train = transformar_alvo_treino(
        y_train_raw
    )

    modelo_final = treinar_final(
        X_fit=X_train_final,
        y_fit_z=y_train_z,
        config=melhor_config,
        epocas=epocas_finais,
    )

    modelo_final.eval()
    Xt = torch.tensor(X_test_final.to_numpy(), dtype=torch.float32)

    with torch.no_grad():
        pred_test_z = modelo_final(Xt).cpu().numpy().reshape(-1)

    precos_previstos = voltar_preco(
        pred_test_z,
        media_log_train,
        desvio_log_train,
    )
    precos_reais = y_test_raw.to_numpy(dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(precos_reais, precos_previstos)))
    mae = float(mean_absolute_error(precos_reais, precos_previstos))
    r2 = float(r2_score(precos_reais, precos_previstos))

    baseline = np.full_like(
        precos_reais,
        fill_value=float(y_train_raw.mean()),
    )
    rmse_baseline = float(np.sqrt(mean_squared_error(precos_reais, baseline)))
    mae_baseline = float(mean_absolute_error(precos_reais, baseline))
    r2_baseline = float(r2_score(precos_reais, baseline))

    print("\n" + "=" * 72)
    print("AVALIACAO FINAL NO TESTE - PRECO NA ESCALA ORIGINAL")
    print("=" * 72)
    print(f"Configuracao escolhida: {melhor_config['nome']}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"R2:   {r2:.4f}")

    print("\nBASELINE - PREVER A MEDIA DOS PRECOS DO TREINO")
    print(f"RMSE baseline: {rmse_baseline:,.2f}")
    print(f"MAE baseline:  {mae_baseline:,.2f}")
    print(f"R2 baseline:   {r2_baseline:.4f}")

    comparacao = pd.DataFrame({
        "SalePrice_real": precos_reais,
        "SalePrice_previsto": precos_previstos,
        "erro_absoluto": np.abs(precos_reais - precos_previstos),
    })
    arquivo_previsoes = pasta_saida / "previsoes_teste_camadas.csv"
    comparacao.to_csv(arquivo_previsoes, index=False)
    print(f"\nPrevisoes do teste salvas em: {arquivo_previsoes}")


if __name__ == "__main__":
    main()
