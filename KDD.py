import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Cargar los datos
print("Cargando datos...")
data = pd.read_excel('prueba.xlsx', engine='openpyxl')
print("Forma de los datos:", data.shape)
print("Primeras 5 filas:")
print(data.head())

# 2. Preprocesamiento
print("\nPreprocesando datos...")
# Renombrar las columnas para mayor claridad
data.columns = ['cliente_id', 'edad', 'genero', 'tipo_contrato', 'churn']

# Eliminar la columna 'cliente_id'
data = data.drop('cliente_id', axis=1)

# Codificar variables categóricas
le = LabelEncoder()
data['genero'] = le.fit_transform(data['genero'])
data['tipo_contrato'] = le.fit_transform(data['tipo_contrato'])

# Asegurarse de que 'churn' sea numérico (0 o 1)
data['churn'] = le.fit_transform(data['churn'])

print("Datos después del preprocesamiento:")
print(data.head())

# Separar características y variable objetivo
X = data.drop('churn', axis=1)
y = data['churn']

# Normalizar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir los datos
print("\nDividiendo los datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

# 4. Entrenar el árbol de decisión
print("\nEntrenando el árbol de decisión...")
dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# 5. Evaluar el modelo
print("\nEvaluando el modelo...")
y_pred = dt_classifier.predict(X_test)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# 6. Visualizar el árbol de decisión
print("\nGenerando visualización del árbol de decisión...")
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['No Churn', 'Churn'], filled=True, rounded=True)
plt.savefig('img/decision_tree.png')
plt.close()

# 7. Analizar la importancia de las características
print("\nAnalizando la importancia de las características...")
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt_classifier.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("Importancia de las Características:")
print(feature_importance)

# 8. Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Importancia de las Características')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.close()

print("\nEl árbol de decisión se ha guardado como 'decision_tree.png'")


print("\nAnálisis completado.")