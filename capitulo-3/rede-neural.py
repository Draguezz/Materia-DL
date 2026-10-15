#Rede Neural com Aproximadora de funções Não Lineares

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dados de treinamento
x_train = np.linspace(0, 10, 1000)
y_train = np.sin(x_train)

# Construindo o modelo
model = Sequential([
    tf.keras.layers.Input(shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinamento
history = model.fit(x_train, y_train, epochs=1500, batch_size=32, verbose=0)

# Dados de teste
x_test = np.linspace(0, 10, 100)
y_test = np.sin(x_test)
y_pred = model.predict(x_test)

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='Função Senoidal')
plt.plot(x_test, y_pred, "--", label='Aproximação do Modelo')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Aproximação de Função Senoidal')
plt.legend()
plt.grid(True)
plt.show()

# Visualizando a convergência do treinamento
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio')
plt.title('Convergência do Treinamento')
plt.grid(True)
plt.show()
