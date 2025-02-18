import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron

# Carregar dados
data = pd.read_csv("data/dataset.csv")
X = data.iloc[:, :-1].values
y = np.where(data.iloc[:, -1].values == 0, -1, 1)

# Treinar modelo
perceptron = Perceptron(learning_rate=0.01, epochs=1000)
perceptron.fit(X, y)

# Criar gráfico de dispersão das classes
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

# Criar linha de decisão
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx = np.linspace(x_min, x_max, 100)
yy = -(perceptron.weights[0] * xx + perceptron.bias) / perceptron.weights[1]

plt.plot(xx, yy, 'k-', label="Fronteira de decisão")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.title("Classificação com Perceptron")
plt.show()
