import pandas as pd
import numpy as np
from pathlib import Path

pasta = Path(__file__).resolve().parent

dados = pd.read_csv(pasta / "train.csv")

colunas_remover = [
    "Utilities", "Street", "Condition2", "RoofMatl", "Heating", "PoolQC", "Exterior1st", "SaleType"
]
dados = dados.drop(columns=colunas_remover)

for coluna in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
    dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

numericas = dados.select_dtypes(include="number").columns
categoricas = dados.select_dtypes(include="object").columns

for coluna in numericas:
    dados[coluna] = dados[coluna].fillna(dados[coluna].mean())

for coluna in categoricas:
    dados[coluna] = dados[coluna].fillna("None")

for coluna in numericas:
    if coluna == "SalePrice":
        continue

    desvio = dados[coluna].std(ddof=0)

    if desvio != 0:
        dados[coluna] = (
            dados[coluna] - dados[coluna].mean()
        ) / desvio
    else:
        dados[coluna] = 0.0

dados["SalePrice"] = np.log(dados["SalePrice"])

media_log = dados["SalePrice"].mean()
desvio_log = dados["SalePrice"].std(ddof=0)

dados["SalePrice"] = (
    dados["SalePrice"] - media_log
) / desvio_log

print("Primeiros valores de SalePrice após a transformação:")
print(dados["SalePrice"].head())

print("\nMédia de SalePrice:", dados["SalePrice"].mean())
print("Desvio-padrão de SalePrice:",
      dados["SalePrice"].std(ddof=0))

mapeamentos = {
    "ExterQual": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "ExterCond": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "BsmtQual": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "BsmtCond": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "HeatingQC": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "KitchenQual": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "FireplaceQu": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "GarageQual": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "GarageCond": {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5
    },
    "BsmtExposure": {
        "None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4
    },
    "BsmtFinType1": {
        "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6
    },
    "BsmtFinType2": {
        "None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6
    },
    "GarageFinish": {
        "None": 0, "Unf": 1, "RFn": 2, "Fin": 3
    },
    "LotShape": {
        "IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3
    },
    "LandSlope": {
        "Gtl": 0, "Mod": 1, "Sev": 2
    },
    "PavedDrive": {
        "N": 0, "P": 1, "Y": 2
    },
    "Functional": {
        "Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7
    },
    "Fence": {
        "None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4
    }
}

for coluna, mapa in mapeamentos.items():
    if coluna in dados.columns:
        dados[coluna] = dados[coluna].map(mapa)


colunas_categoricas = dados.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Quantidade de colunas categóricas antes da modificação:",
      len(colunas_categoricas))

dados = pd.get_dummies(
    dados,
    columns=colunas_categoricas,
    dtype=int
)

print("\nDimensões após o One-Hot Encoding:", dados.shape)

print("\nPrimeiras linhas do dataset transformado:")
print(dados.head())

print("\nColunas categóricas após modificação:")
print(
    len(dados.select_dtypes(
        include=["object", "category"]
    ).columns)
)
