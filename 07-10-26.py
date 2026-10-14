# Implementar uma loss function MSE

import torch

def mse_loss(dado_base, valor_real):
    erro = dado_base - valor_real
    erro_quadrado = erro ** 2
    return torch.mean(erro_quadrado)


dado_base = torch.tensor([2.0, 4.0, 6.0])
valor_real = torch.tensor([1.0, 5.0, 4.0])

resultado = mse_loss(dado_base, valor_real)

print("MSE:", resultado)

#loss function MAE

def mae_loss(teorico, real):
    erros = teorico - real
    erros_total = torch.abs(erros)
    return torch.mean(erros_total)

teorico = torch.tensor([2.0,4.0, 6.0])
real = torch.tensor([1.0,5.0,4.0])

resposta = mae_loss(teorico, real)

print("MAE:", resposta)