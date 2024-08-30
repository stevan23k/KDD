import pandas as pd
import numpy as np

# Configurar la semilla aleatoria para reproducibilidad
np.random.seed(42)

# Número de clientes
n_clientes = 60

# Generar datos
data = pd.DataFrame({
    'cliente_id': range(1, n_clientes + 1),
    'edad': np.random.randint(19, 80, n_clientes),
    'genero': np.random.choice(['Male', 'Female'], n_clientes), 
    'tipoDeContrato': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_clientes),
    
    'churn': np.random.choice([0, 1], n_clientes, p=[0.7, 0.3])  # 30% tasa de churn
})

# Guardar en Excel
nombre_archivo = "prueba.xlsx"
data.to_excel(nombre_archivo, index=False)

print(f"Datos generados y guardados en {nombre_archivo}")