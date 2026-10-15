import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Semente única para reprodutibilidade. `keras.utils.set_random_seed` semeia o
# Python, o NumPy e o Keras de uma vez; `tf.random.set_seed` sozinho NÃO basta,
# porque o Keras 3 tem gerador próprio e os pesos iniciais sairiam diferentes a
# cada execução.
keras.utils.set_random_seed(42)

# Gerando dados de área e preço de aluguel
X = 30 + 120 * np.random.rand(100, 1)  # Áreas entre 30m² e 150m²
y = 1500 + 50 * X + np.random.randn(100, 1) * 250  # Preço base + preço por m² + variação

# Construindo o modelo de regressão linear
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Compilação com Adam para melhor convergência
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

# Treinamento do modelo
history = model.fit(X, y, epochs=2500, verbose=0)

# Previsões para visualização da reta de regressão
X_range = np.linspace(30, 150, 100).reshape(-1, 1)
y_pred = model.predict(X_range)

# Coeficientes do modelo
weights, bias = model.layers[0].get_weights()
print(f"weights: {weights[0][0]:.2f}")
print(f"bias: {bias[0]:.2f}")

# Visualização do modelo e da convergência, lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.scatter(X, y, color='blue', alpha=0.5, label='Dados reais')
ax1.plot(X_range, y_pred, color='red', label='Modelo de precificação')
ax1.set_xlabel("Área do imóvel (m²)")
ax1.set_ylabel("Valor do Aluguel (R$)")
ax1.set_title("Relação entre Área e Valor do Aluguel")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], color='purple')
ax2.set_xlabel("Épocas")
ax2.set_ylabel("Erro Quadrado Médio (MSE)")
ax2.set_title("Convergência do Treinamento")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
