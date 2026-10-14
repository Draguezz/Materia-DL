import numpy as np
import matplotlib.pyplot as plt

# Definindo os valores de inclinação (a) e intercepto (b) para cada reta
params = [
    {"a": 0.5, "b": 1},  # Inclinação 0.5, Intercepto 1
    {"a": 2, "b": -3},   # Inclinação 2, Intercepto -3
    {"a": -1, "b": 4}    # Inclinação -1, Intercepto 4
]

# Gerando valores de x (variável independente)
x = np.linspace(-10, 10, 100)

# Criando o gráfico para as três retas
plt.figure(figsize=(8, 6))

for param in params:
    a = param["a"]
    b = param["b"]
    y = a * x + b  # Calculando y com base na equação da reta: y = a*x + b
    plt.plot(x, y, label=f'y = {a}x + {b}', linewidth=2)

    # Destacando o intercepto no gráfico
    plt.scatter(0, b, color='blue', zorder=5)
    plt.text(0, b + 0.5, f'Intercepto = {b}', color='blue')

# Adicionando grades, título e legendas explicativas
plt.title('Três Retas com Diferentes Inclinações e Interceptos')
plt.xlabel('Variável Independente (x)')
plt.ylabel('Variável Dependente (y)')
plt.axhline(0, color='black', linewidth=0.5)  # Linha horizontal no eixo Y
plt.axvline(0, color='black', linewidth=0.5)  # Linha vertical no eixo X
plt.grid(True)  # Adicionando grid para facilitar a leitura
plt.legend()

# Exibindo o gráfico
plt.show()
