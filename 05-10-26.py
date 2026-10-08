# Criar dois tensores aleatorios, fazer a multiplicação matricial deles

import torch

tensor1 = torch.rand(3, 3)
tensor2 = torch.rand(3, 2)

    #print(tensor1)
    #print(tensor2)

multiplica = torch.matmul(tensor1, tensor2)

print(multiplica)

# Criar uma função que, dado duas tuplas, 
    # cria 2 tensores com aquele shape e retorna a multiplicação matricial entre eles

def multiplicar(shape1, shape2):
    tensor1 = torch.rand(shape1)
    tensor2 = torch.rand(shape2)

    return torch.matmul(tensor1, tensor2)

resultado = multiplicar((2, 3), (3, 2))

print(resultado)