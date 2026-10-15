import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Carregando e pré-processando os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Visualizando uma amostra de imagens
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Rótulo: {y_train[i]}')
plt.show()

# Definindo e construindo o modelo
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Salvando modelo
model.save('mlp_mnist_model.keras')

# Avaliando o modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

# Fazendo previsões
predictions = model.predict(x_test[:10])
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Previsto: {np.argmax(predictions[i])}\nReal: {y_test[i]}')
plt.show()

# Visualizando a convergência do treinamento
plt.figure(figsize=(18, 5))

# Gráfico de perda
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.title("Perda durante o Treinamento")

# Gráfico de acurácia
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.title("Acurácia durante o Treinamento")

# Matriz de Confusão
plt.subplot(1, 3, 3)
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues", ax=plt.gca())  # Ajuste para plotar no subplot atual
plt.title("Matriz de Confusão")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")

plt.tight_layout()  # Ajuste para evitar sobreposição
plt.show()
