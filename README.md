# Predicción de Fuga de Clientes (Customer Churn)

## Descripción General
Proyecto de Machine Learning e ingeniería de datos enfocado en predecir la fuga de clientes en una empresa de telecomunicaciones.
El objetivo principal es optimizar la sensibilidad (Recall) para minimizar las pérdidas financieras derivadas de cancelaciones de
servicio imprevistas, priorizando la detección temprana de desertores por encima de una precisión general inflada.

## Dataset
Se utiliza el dataset público "Telco Customer Churn". Contiene información de 7,043 registros con variables demográficas, servicios contratados, información de facturación y la variable objetivo `Churn` (Sí/No).

## Fases del Proyecto
1. **Limpieza y validación:** Tratamiento de valores nulos, duplicados y corrección de tipos de datos.
2. **Preprocesamiento:** Codificación de variables categóricas (`Label Encoding` y `One-Hot Encoding`) y escalado numérico con `MinMaxScaler`.
3. **Partición de datos:** División estratificada con semilla fija (`random_state=42`) para garantizar la reproducibilidad del pipeline.
4. **Modelado y balanceo:** Entrenamiento con mitigación de desbalance de clases (`class_weight='balanced'` / `scale_pos_weight`).
5. **Evaluación y explicabilidad:** Comparativa mediante matrices de confusión, métricas de negocio y extracción de importancia de variables (Feature Importance).

## Comparativa de Modelos
Se evaluaron cinco algoritmos bajo idénticas condiciones de validación:

| Modelo | Tiempo de Cómputo | Exactitud (Accuracy) | Precisión | Sensibilidad (Recall) | Veredicto / Uso en Producción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística** | ~0.05s | 74.04% | 46.17% | 80.69% | Máxima protección. El mejor detectando fugas, aunque genera más falsas alarmas. |
| **LightGBM** | ~0.06s | 76.60% | 49.29% | 76.01% | Ganador Absoluto. Excelente equilibrio entre velocidad, falsos positivos reducidos y alto Recall. |
| **XGBoost** | ~0.39s | 75.75% | 47.92% | 68.22% | Superado en sensibilidad por LightGBM y Regresión Logística. |
| **Random Forest** | ~0.45s | 77.39% | 50.47% | 67.29% | Alta precisión general pero deja escapar a un volumen elevado de clientes en fuga. |
| **SVM** | ~0.81s | 74.32% | 46.38% | 77.88% | Descartado. Alto costo computacional y carencia de interpretabilidad directa. |

## Hallazgos e Insights de Negocio
* **Tipo de contrato:** Los contratos a largo plazo reducen significativamente la tasa de fuga gracias a incentivos y tarifas preferenciales frente al modelo mes a mes.
* **Soporte y calidad de servicio:** La tenencia de servicios de valor añadido como soporte técnico y seguridad online actúa como un amortiguador clave contra la deserción.
* **Dinámica de hogar:** Los hogares con múltiples integrantes aprovechan mejor los planes de alta velocidad y streaming, mostrando menor propensión a cancelar en comparación con usuarios unipersonales.

## Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Manipulación de Datos:** Pandas, NumPy
* **Modelado y Métricas:** Scikit-Learn, LightGBM, XGBoost
* **Visualización:** Matplotlib, Seaborn
* **Reproducibilidad:** Semilla determinista configurada (`random_state=42`)

## Cómo Ejecutar Localmente
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/mauriciobustillosmeneses17-art/customer-churn-prediction.git
cd customer-churn-prediction

Instalar las dependencias necesarias:

 Bash
 pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
 Ejecutar los scripts o abrir el Jupyter Notebook en orden secuencial desde la limpieza de datos hasta el entrenamiento de modelos.
