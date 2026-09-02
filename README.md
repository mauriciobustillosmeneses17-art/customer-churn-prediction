# Predicción de Fuga de Clientes (Customer Churn)

## Descripción General
Proyecto de Machine Learning e ingeniería de datos enfocado en predecir la fuga de clientes en una empresa de telecomunicaciones.
El objetivo principal es optimizar la sensibilidad (Recall) para minimizar las pérdidas financieras derivadas de cancelaciones de
servicio imprevistas, priorizando la detección temprana de desertores por encima de una precisión general inflada.

## Dataset
Se utiliza el dataset público ["Telco Customer Churn"](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle). Contiene información de 7,043 registros con variables demográficas, servicios contratados, información de facturación y la variable objetivo `Churn` (Sí/No).

> Descarga el CSV `WA_Fn-UseC_-Telco-Customer-Churn.csv` desde el link anterior y colócalo en la raíz del proyecto antes de ejecutar el notebook.

## Fases del Proyecto
1. **Limpieza y validación:** Tratamiento de valores nulos, duplicados y corrección de tipos de datos.
2. **Preprocesamiento:** Codificación de variables categóricas (`Label Encoding` y `One-Hot Encoding`) y escalado numérico con `MinMaxScaler`.
3. **Partición de datos:** División estratificada con semilla fija (`random_state=42`) para garantizar la reproducibilidad del pipeline.
4. **Modelado y balanceo:** Entrenamiento con mitigación de desbalance de clases (`class_weight='balanced'` / `scale_pos_weight`).
5. **Evaluación y explicabilidad:** Comparativa mediante matrices de confusión, validación cruzada (5-Fold), curvas ROC/PR-AUC y análisis SHAP.
6. **Optimización:** Ajuste de hiperparámetros con `RandomizedSearchCV`.

## Comparativa de Modelos
Se evaluaron cinco algoritmos bajo idénticas condiciones de validación:

| Modelo | Tiempo de Cómputo | Exactitud (Accuracy) | Precisión | Sensibilidad (Recall) | Veredicto / Uso en Producción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regresión Logística** | ~0.05s | 74.04% | 46.17% | 80.69% | Máxima protección. El mejor detectando fugas, aunque genera más falsas alarmas. |
| **LightGBM (Base)** | ~0.06s | 76.60% | 49.29% | 76.01% | Ganador Absoluto entre modelos base. Excelente equilibrio entre velocidad, falsos positivos reducidos y alto Recall. |
| **XGBoost** | ~0.39s | 75.75% | 47.92% | 68.22% | Superado en sensibilidad por LightGBM y Regresión Logística. |
| **Random Forest** | ~0.45s | 77.39% | 50.47% | 67.29% | Alta precisión general pero deja escapar a un volumen elevado de clientes en fuga. |
| **SVM** | ~0.81s | 74.32% | 46.38% | 77.88% | Descartado. Alto costo computacional y carencia de interpretabilidad directa. |
| **LightGBM (Tuneado)** | ~0.06s | **79.24%** | **53.75%** | 66.98% | Mejor Accuracy/Precisión y ROC-AUC (0.849), pero con menor Recall que la versión base. Ver nota abajo. |

**Validación cruzada (5-Fold estratificado) sobre LightGBM base:** Recall promedio de 74.96% (± 2.33%), ROC-AUC promedio de 0.8379 — confirma que el resultado del split original no fue casualidad.

> **Nota sobre el tuning:** la búsqueda de hiperparámetros (`RandomizedSearchCV`, optimizando por Recall) mejoró Accuracy, Precisión y ROC-AUC, pero el Recall del modelo tuneado (66.98%) quedó por debajo del LightGBM base (76.01%). Dado que el criterio de negocio de este proyecto prioriza el Recall, **el modelo recomendado para producción sigue siendo LightGBM (Base)**, salvo que se amplíe el espacio de búsqueda de hiperparámetros en una futura iteración.

## Impacto de Negocio y Visión Estratégica
A partir del análisis y el modelado predictivo, el proyecto trasciende la métrica técnica para proponer soluciones accionables de retención:

**1. Insights Clave del Comportamiento del Cliente**
* **Fricción Administrativa:** El modelo de suscripción "mes a mes" y el uso de cheques electrónicos son los principales motores de fuga.
* **Anclas de Retención:** La tenencia de Soporte Técnico y Seguridad en línea actúan como barreras críticas contra el *churn*. Asimismo, los hogares familiares muestran mayor lealtad que los usuarios individuales al aprovechar al máximo la capacidad de la red.

**2. Acciones Comerciales Propuestas (Data-Driven)**
* **Segmentación de Oferta:** Creación de *Planes Familiares* dinámicos según el tamaño del hogar, y *Planes Unipersonales* de menor costo para evitar que usuarios solteros cancelen al percibir que pagan por capacidad ociosa.
* **Migración Digital:** Implementar incentivos para trasladar a los usuarios hacia pagos automáticos (tarjetas/débito directo), reduciendo la tasa de cancelación por fricción de cobro.

**3. Roadmap Técnico y Próximos Pasos (Visión de Producción)**
* **Despliegue (Deployment):** Encapsular el modelo ganador (LightGBM) mediante una API REST utilizando **FastAPI** para integrarlo a los CRMs de la empresa y emitir alertas de fuga en tiempo real.
* **Optimización Avanzada:** Ampliar la búsqueda de hiperparámetros (mayor `n_iter` y rango de `scale_pos_weight`) e implementar **Optuna** para exprimir el rendimiento matemático del modelo sin sacrificar Recall.
* **Ciclo de Vida (MLOps):** Establecer un monitoreo continuo del desempeño del algoritmo en producción para detectar degradación de datos e inyectar nuevas variables de calidad (ej. latencia de conexión, métricas de satisfacción).

## Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Manipulación de Datos:** Pandas, NumPy
* **Modelado y Métricas:** Scikit-Learn, LightGBM, XGBoost
* **Interpretabilidad:** SHAP
* **Visualización:** Matplotlib, Seaborn
* **Reproducibilidad:** Semilla determinista configurada (`random_state=42`)

