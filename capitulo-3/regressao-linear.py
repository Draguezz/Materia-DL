import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Gerando dados de exemplo (distribuição de pontos)
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)  # Reta com alguma variação aleatória

# Uma reta qualquer, sorteada sem olhar para os dados. O LinearRegression NAO
# parte dela: ele resolve de uma vez. Ela serve de termo de comparacao, e e' a
# mesma reta inicial da secao de retropropagacao, mais adiante.
w_inicial = np.random.randn(1)
b_inicial = np.random.randn(1)

# Ajustando o modelo de regressão linear
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)

# Plotando os pontos, a reta sorteada e a reta de regressão. A reta sorteada e'
# desenhada entre os EXTREMOS DE x, com y calculado neles, que vale para
# qualquer sinal do peso.
x_extremos = np.array([np.min(x), np.max(x)])
plt.scatter(x, y, color='blue', label='Pontos de Dados')
plt.plot(x_extremos, w_inicial * x_extremos + b_inicial, color='orange',
         linestyle='--', linewidth=2, label='Reta Inicial (sorteada)')
plt.plot(x, y_pred, color='red', linewidth=2, label='Reta de Regressão')
plt.title('Distribuição de Pontos e Regressão Linear')
plt.xlabel('Variável Independente (x)')
plt.ylabel('Variável Dependente (y)')
plt.legend()
plt.show()
