import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron  # Importa o Perceptron criado anteriormente

# 1. Carregar o Dataset de Teste
data = pd.read_csv("data/dataset.csv", header=None)
X_test = data.iloc[:, :-1].values  # Características
y_test = data.iloc[:, -1].values  # Rótulos

# 2. Converter rótulos de 0 para -1 (Necessário para o Perceptron)
y_test = np.where(y_test == 0, -1, 1)

# 3. Criar e treinar o Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X_test, y_test)

# 4. Testar o modelo com novos exemplos
novos_exemplos = np.array([[2.0, 3.0], [1.2, 1.5], [4.0, 4.0], [0.5, 0.8]])
predicoes = perceptron.predict(novos_exemplos)

print("Novas previsões:", predicoes)

# 5. Avaliar a Acurácia
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Acurácia do modelo: {accuracy:.2f}%")

# 6. Visualizar Resultados
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.title("Distribuição dos Dados de Teste")
plt.colorbar(label="Classe")
plt.show()