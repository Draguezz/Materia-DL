#Testar três otimizadores do Keras (Adam, SGD e RMSprop), alterando o código conforme indicado.
#Visualizar a curva de perda para cada otimizador ao longo do treinamento.
#Comparar o desempenho na aproximação da função seno e discutir qual otimizador teve melhor desempenho.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definindo os dados de treino
x_train = np.linspace(0, 10, 1000)
y_train = np.sin(x_train)

# Função para criar e treinar o modelo com um otimizador específico
def train_model(optimizer_name):
    # Construir o modelo
    model = Sequential([
        tf.keras.layers.Input(shape=(1,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    # Compilar o modelo com o otimizador especificado
    model.compile(optimizer=optimizer_name, loss='mean_squared_error')

    # Treinar o modelo
    history = model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=0)
    
    return model, history

# Lista de otimizadores para o teste
optimizers = ['adam', 'sgd', 'rmsprop']
histories = {}
models = {}

# Treinar o modelo e salvar os históricos de perda
for opt in optimizers:
    ...

# Gerando dados de teste
...

# Visualização das aproximações da função seno
...

# Visualização da convergência do treinamento (perda) para cada otimizador
...
