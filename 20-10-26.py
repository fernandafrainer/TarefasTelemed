
import torch

def mse_batch(predicoes, alvos):
    erros = predicoes - alvos
    erros_quadrados = erros ** 2
    perda = torch.mean(erros_quadrados)
    return perda


predicoes = torch.tensor([
    [2.0, 4.0],
    [3.0, 5.0],
    [6.0, 7.0]
])

alvos = torch.tensor([
    [1.0, 5.0],
    [2.0, 5.0],
    [4.0, 9.0]
])

resultado = mse_batch(predicoes, alvos)

print("MSE do batch:", resultado)
