import numpy as np
import matplotlib.pyplot as plt

# a mesma nuvem da seção anterior
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y_real = 4 + 3 * x + np.random.randn(100, 1)

ERRO_ALVO = 1.1    # o laço para quando o erro fica abaixo deste valor
LIMITE = 20000     # trava de segurança: nenhum laço roda para sempre
TAXAS = [0.001, 0.01, 0.1]

# previsão, erro quadrático médio e uma iteração de backpropagation
# um treino por taxa: iterações até o alvo e a lista de erros do caminho
