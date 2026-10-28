from pathlib import Path
from copy import deepcopy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

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