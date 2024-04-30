from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Simulando um conjunto de dados de exemplo
X = np.random.rand(100, 2)  # Características (features) simuladas
y = np.random.randint(0, 2, 100)  # Rótulos (labels) simulados

# Pré-processamento de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escolher um algoritmo de Machine Learning
model = LogisticRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')
