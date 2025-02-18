import numpy as np

class Perceptron:
    """
    Implementação de um Perceptron simples para classificação binária.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Inicializa o perceptron com taxa de aprendizado e número de épocas.

        :param learning_rate: Taxa de aprendizado para ajuste dos pesos.
        :param epochs: Número de iterações sobre os dados de treinamento.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def activation(self, x):
        """
        Função de ativação degrau.

        :param x: Entrada ponderada
        :return: 1 se x >= 0, caso contrário -1
        """
        return 1 if x >= 0 else -1

    def fit(self, X, y):
        """
        Treina o perceptron ajustando os pesos com base no erro de classificação.

        :param X: Matriz de características (shape: amostras x atributos)
        :param y: Vetor de rótulos (valores esperados)
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for epoch in range(self.epochs):
            errors = 0
            for i in range(num_samples):
                # Produto escalar entre pesos e entrada + bias
                weighted_sum = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(weighted_sum)

                # Atualização dos pesos se houver erro
                error = y[i] - y_pred
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1
            
            # Para visualizar a convergência
            if errors == 0:
                break

    def predict(self, X):
        """
        Realiza a predição de novos exemplos.

        :param X: Matriz de características a ser classificada
        :return: Vetor de previsões (-1 ou 1)
        """
        return np.array([self.activation(np.dot(x, self.weights) + self.bias) for x in X])
