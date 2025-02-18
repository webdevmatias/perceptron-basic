import pandas as pd
import numpy as np
from perceptron import Perceptron

# Carregar dados do CSV
data = pd.read_csv("data/dataset.csv")

# Suponha que temos duas colunas de características e uma coluna de rótulos
X = data.iloc[:, :-1].values  # Características (features)
y = data.iloc[:, -1].values   # Rótulo (classe)

# Convertendo rótulos para valores -1 e 1 (exigido pelo Perceptron)
y = np.where(y == 0, -1, 1)

# Criar e treinar o Perceptron
perceptron = Perceptron(learning_rate=0.01, epochs=1000)
perceptron.fit(X, y)

# Testar o modelo
predictions = perceptron.predict(X)

# Mostrar precisão
accuracy = np.mean(predictions == y) * 100
print(f"Acurácia do Perceptron: {accuracy:.2f}%")
