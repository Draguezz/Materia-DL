import numpy as np
import matplotlib.pyplot as plt

# Gerando dados de exemplo (nuvem de pontos)
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y_real = 4 + 3 * x + np.random.randn(100, 1)  # Reta com algum ruído

# Inicializa pesos e bias
w = np.random.randn(1)  # Peso inicial aleatório
b = np.random.randn(1)  # Bias inicial aleatório

# Definição da taxa de aprendizado
eta = 0.01  # Taxa de aprendizado

# Função para calcular a previsão
def predict(x, w, b):
    return w * x + b   #y_pred

# Função para calcular o erro quadrático
def loss(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)

# Função para realizar uma iteração de backpropagation
def backpropagation(x, y_real, w, b, eta):
    # Forward pass: cálculo da previsão
    y_pred = predict(x, w, b)
    
    # Cálculo do erro quadrático
    erro = loss(y_real, y_pred)
    
    # Gradientes em relação a w e b
    grad_w = -2 * np.mean((y_real - y_pred) * x)
    grad_b = -2 * np.mean(y_real - y_pred)
    
    # Atualizando os parâmetros
    w -= eta * grad_w
    b -= eta * grad_b
    
    return w, b, erro

# Calculando a reta antes de ajustar os parâmetros
w_inicial, b_inicial = w.copy(), b.copy()

# Executando múltiplas iterações para ajustar os parâmetros
num_iteracoes = 100
erros = []

for i in range(num_iteracoes):
    w, b, erro = backpropagation(x, y_real, w, b, eta)
    erros.append(erro)

# Plotando as retas antes e depois do ajuste
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.scatter(x, y_real, color='blue', label='Pontos de Dados')
# A reta e' desenhada entre os EXTREMOS DE x, com y calculado neles. Usar
# min(y) e max(y) inverte a reta quando o peso e' negativo, que e' o caso da
# reta inicial com esta semente: w comeca em -0,354.
x_extremos = np.array([np.min(x), np.max(x)])
plt.plot(x_extremos, w_inicial * x_extremos + b_inicial, color='orange', label='Reta Inicial')
plt.plot(x_extremos, w * x_extremos + b, color='red', label='Reta Ajustada')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Reta Inicial vs Reta Ajustada')
plt.legend()

# Plotando o erro ao longo das iterações
plt.subplot(1, 2, 2)
plt.plot(range(num_iteracoes), erros, color='purple')
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.title('Redução do Erro com Backpropagation')

plt.tight_layout()
plt.show()
