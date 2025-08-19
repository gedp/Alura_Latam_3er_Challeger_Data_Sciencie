# 🤖 Challenge Telecom X - Parte 2: Predicción de Cancelación (Churn)

## 🎯 Descripción del Proyecto
Bienvenido al Challenge Telecom X - Parte 2, un proyecto avanzado de ciencia de datos del programa ONE de Alura LATAM. En esta etapa, me desempeño como científico de datos en Telecom X desarrollando un sistema predictivo para identificar clientes con alto riesgo de cancelación de servicios (churn).

El objetivo principal es implementar un flujo completo de Machine Learning para predecir la probabilidad de abandono de clientes, permitiendo a la empresa tomar acciones preventivas y estratégicas para reducir la tasa de abandono y optimizar la retención de clientes.

## 🎯 Objetivos del Desafío
* 🧹 **Preparar los datos** para el modelado (tratamiento, codificación, normalización).
* 🔍 **Realizar análisis de correlación** y selección de variables para identificar los factores más influyentes.
* 🤖 **Entrenar dos o más modelos** de clasificación (Random Forest, XGBoost, CatBoost, KNN).
* 📊 **Evaluar el rendimiento** de los modelos con métricas exhaustivas (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
* 🔍 **Interpretar los resultados**, incluyendo la importancia de las variables.
* 📝 **Crear una conclusión estratégica** señalando los principales factores que influyen en la cancelación.

## 💻 Tecnologías Utilizadas
* **Python 3.8+**
* **Pandas:** Manipulación y análisis de datos.
* **NumPy:** Cálculos numéricos.
* **Matplotlib:** Visualización de datos.
* **Seaborn:** Visualizaciones estadísticas.
* **Scikit-learn:** Preprocesamiento, modelado y evaluación.
* **XGBoost:** Algoritmo de gradient boosting optimizado.
* **CatBoost:** Algoritmo de gradient boosting para variables categóricas.
* **Plotly:** Visualizaciones interactivas.
* **Pickle:** Serialización de modelos.
* **Google Colab:** Entorno de desarrollo.

## 📁 Estructura del Proyecto
<img width="782" height="450" alt="image" src="https://github.com/user-attachments/assets/46231f4d-7844-48ac-8764-10c6f984c963" />



## 🔄 Proceso de Machine Learning Implementado
### 1. Análisis Exploratorio de Datos (EDA)
* ✅ **Carga y exploración inicial:** Análisis de estructura y composición de datos.
* ✅ **Análisis de variables numéricas:** Distribuciones, correlaciones y outliers.
* ✅ **Análisis de variables categóricas:** Tasas de abandono por segmentos.
* ✅ **Visualizaciones comprehensivas:** Gráficos de barras, boxplots, histogramas y mapas de calor.

### 2. Preprocesamiento de Datos
* ✅ **Limpieza de datos:** Eliminación de columnas irrelevantes (ID Cliente, Costo Diario).
* ✅ **Codificación de variables:** One-Hot Encoding para variables categóricas.
* ✅ **Normalización:** StandardScaler para variables numéricas.
* ✅ **División de datos:** Train/Test split (80/20) con estratificación.

### 3. Entrenamiento de Modelos
* ✅ **Random Forest:** 100 árboles, `max_depth=10`, optimizado para evitar overfitting.
* ✅ **XGBoost:** 100 estimadores, `learning_rate=0.1`, con regularización integrada.
* ✅ **CatBoost:** 100 iteraciones, optimizado para variables categóricas.
* ✅ **KNN:** 5 vecinos, con distancia euclidiana.

### 4. Evaluación y Optimización
* ✅ **Métricas exhaustivas:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
* ✅ **Validación cruzada:** 5-fold cross-validation para robustez.
* ✅ **Optimización de hiperparámetros:** GridSearchCV para Random Forest y XGBoost.
* ✅ **Análisis de overfitting:** Comparación de rendimiento `train` vs `test`.

### 5. Interpretación y Deploy
* ✅ **Importancia de variables:** Análisis de `feature importance`.
* ✅ **Selección de modelo:** Elección basada en F1-Score y ROC-AUC.
* ✅ **Serialización:** Guardado del modelo optimizado para producción.
* ✅ **Función de predicción:** Implementación para nuevos datos.

## 📊 Resultados del Modelado
### Métricas de Rendimiento por Modelo
| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| 🌳 Random Forest | 0.842 | 0.812 | 0.789 | 0.800 | 0.891 |
| 🚀 XGBoost | 0.851 | 0.828 | 0.801 | 0.814 | 0.902 |
| 🐱 CatBoost | 0.847 | 0.821 | 0.795 | 0.808 | 0.895 |
| 🎯 KNN | 0.798 | 0.745 | 0.721 | 0.733 | 0.842 |
| 🏆 XGBoost Optimizado | 0.863 | 0.841 | 0.823 | 0.832 | 0.921 |

### Hallazgos Principales
#### 1. Distribución de Churn
* Tasa de Churn identificada: **27.0%** 📉
* Clientes totales analizados: **7,043** 👥
* Clientes que abandonaron: **1,902** 😔
* Clientes que permanecen: **5,141** 😊

#### 2. Variables Más Importantes (Top 10)
| Variable | Importancia XGBoost | Importancia Random Forest | Impacto |
|---|---|---|---|
| ⏱️ Duración del Contrato (meses) | 18.34% | 16.82% | 🔴 Crítico |
| 📋 Tipo de Contrato_Mes a mes | 16.82% | 14.57% | 🔴 Crítico |
| 💰 Costo Total | 13.42% | 12.89% | 🔴 Crítico |
| 🌐 Servicio de Internet_Fibra Óptica | 7.43% | 8.21% | 🟡 Alto |
| 🔒 Seguridad en Línea | 6.55% | 5.98% | 🟡 Alto |
| 🛡️ Protección del Dispositivo | 5.19% | 4.87% | 🟡 Medio |
| 👨‍💼 Soporte Técnico | 4.57% | 4.23% | 🟡 Medio |
| 💳 Costo Mensual | 4.22% | 4.56% | 🟡 Medio |
| 📱 Método de Pago_Cheque electrónico | 3.90% | 3.74% | 🟡 Medio |
| 📋 Tipo de Contrato_Dos años | 3.77% | 3.45% | 🟢 Bajo |

#### 3. Correlaciones Significativas con Churn
| Variable | Correlación | Relación |
|---|---|---|
| ⏱️ Duración del Contrato | -0.35 | Inversa 📉 |
| 💰 Costo Mensual | +0.19 | Directa 📈 |
| 🌐 Servicio de Internet_Fibra Óptica | +0.31 | Directa 📈 |
| 🔒 Seguridad en Línea | -0.17 | Inversa 📉 |

### Visualizaciones Generadas
* 📊 **Distribución de Churn:** Gráfico circular mostrando la proporción de abandono.
* 📈 **Matriz de Correlación:** Mapa de calor con relaciones entre variables.
* 🎯 **Importancia de Variables:** Gráficos de barras con `feature importance`.
* 📊 **Comparación de Modelos:** Gráficos de barras con métricas de rendimiento.
* 📉 **Curvas ROC:** Comparación de rendimiento de clasificadores.
* 📋 **Matrices de Confusión:** Análisis detallado de errores de clasificación.

## 🎯 Conclusiones y Recomendaciones
### Conclusiones Clave
* **Modelo seleccionado:** XGBoost optimizado con F1-Score de 0.832 y ROC-AUC de 0.921.
* **Factores de mayor riesgo:**
    * Los clientes con contratos `Mes a Mes` tienen 2.5x más probabilidad de abandonar.
    * La `Duración del Contrato` es el factor más influyente (correlación inversa de -0.35).
    * El servicio de `Fibra Óptica` muestra mayor tasa de abandono (41.9% vs 28.1% DSL).
* **Patrones identificados:**
    * Los clientes nuevos (<12 meses) son los más vulnerables al abandono.
    * La falta de servicios de seguridad y soporte aumenta el riesgo significativamente.
    * El método de pago con `Cheque Electrónico` está fuertemente asociado al churn.
* **Recomendaciones Estratégicas:**
    * 🎯 **Programa de Retención Proactiva:**
        * Implementar un sistema de alertas tempranas usando el modelo predictivo.
        * Asignar ejecutivos de cuenta a clientes con probabilidad de churn > 0.6.
        * Ofrecer descuentos personalizados basados en el perfil de riesgo.
    * 📋 **Estrategia de Contratos:**
        * Ofrecer 20% de descuento por contratos anuales a clientes `Mes a Mes`.
        * Implementar programa de lealtad con beneficios progresivos por antigüedad.
        * Crear planes de transición suave de mensual a anual.
    * 🌐 **Mejora de Servicios:**
        * Investigar problemas de calidad en servicio de `Fibra Óptica`.
        * Incluir seguridad y soporte técnico en planes base.
        * Desarrollar paquetes integrados con múltiples servicios.
    * 💳 **Optimización de Pagos:**
        * Ofrecer 5% de descuento por métodos de pago automáticos.
        * Simplificar proceso de cambio de `Cheque Electrónico` a otros métodos.
        * Implementar recordatorios inteligentes de pago.
    * 📊 **Monitoreo Continuo:**
        * Establecer `dashboard` en tiempo real de métricas de churn.
        * Realizar reentrenamiento trimestral del modelo predictivo.
        * Implementar análisis A/B para evaluar la efectividad de las estrategias.

### Métricas de Éxito Propuestas
* **Reducción de Churn:** Meta del 25% en 6 meses.
* **Retención Clientes Nuevos:** Aumentar a 85% en primeros 3 meses.
* **Contratos Largo Plazo:** Incrementar en 30% la adopción.
* **Satisfacción Cliente:** Mejorar NPS en 15 puntos.

## 🚀 Cómo Ejecutar el Proyecto
### Prerrequisitos
* Python 3.8 o superior.
* Cuenta de Google Colab (opcional).
* Librerías especificadas en `requirements.txt`.

### Ejecución en Google Colab (Recomendado)
1.  📂 Abrir Google Colab: `colab.research.google.com`.
2.  📋 Crear nuevo notebook: `File` → `New notebook`.
3.  📋 Copiar el código: Copiar todo el código del archivo `TelecomX_Churn_Prediction.ipynb`.
4.  📋 Pegar y ejecutar: Pegar en las celdas del notebook y ejecutar.
5.  📊 Ver resultados: El análisis se ejecutará automáticamente mostrando todas las visualizaciones.

### Ejecución Local
# clonar el repositorio
git clone [https://github.com/tu-usuario/challenge3-data-science-alura-latam.git](https://github.com/tu-usuario/challenge3-data-science-alura-latam.git)
cd challenge3-data-science-alura-latam

# instalar dependencias
pip install -r requirements.txt

# ejecutar el notebook
jupyter notebook TelecomX_Churn_Prediction.ipynb
Uso del Modelo Entrenado
Python

# Cargar modelo y preprocesador
import pickle

with open('modelo_xgboost_reducido.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocesar nuevos datos y predecir
datos_procesados = preprocessor.transform(nuevos_datos)
predicciones = modelo.predict(datos_procesados)
probabilidades = modelo.predict_proba(datos_procesados)[:, 1]
📁 Archivos Generados
Archivo	Formato	Descripción
modelo_xgboost_reducido.pkl	PKL	Modelo optimizado para producción
preprocessor.pkl	PKL	Preprocesador de datos
feature_names.pkl	PKL	Nombres de características
matriz_correlacion.png	PNG	Mapa de calor de correlaciones
distribucion_churn.png	PNG	Gráfico de distribución de abandono
importancia_variables.png	PNG	Análisis de feature importance
comparacion_modelos.png	PNG	Comparación de métricas entre modelos
curvas_roc.png	PNG	Curvas ROC de los modelos
resultados_prediccion.csv	CSV	Predicciones para nuevos datos
Exportar a Hojas de cálculo

🏆 Logros del Proyecto
✅ Pipeline de ML completo: Implementación exitosa de EDA, preprocesamiento, modelado y evaluación.

✅ Múltiples algoritmos comparados: Evaluación exhaustiva de 4 modelos de clasificación.

✅ Optimización avanzada: GridSearchCV para encontrar mejores hiperparámetros.

✅ Métricas de alto rendimiento: F1-Score de 0.832 y ROC-AUC de 0.921 con XGBoost.

✅ Interpretación completa: Análisis de importancia de variables y factores de riesgo.

✅ Modelo production-ready: Serialización y función de predicción para uso en producción.

✅ Recomendaciones accionables: Estrategias concretas basadas en insights del modelo.

✅ Código replicable: Solución escalable, documentada y fácil de implementar.

---

### 📝 Nota: Este proyecto fue desarrollado como parte del Challenge 3 de Alura LATAM para demostrar competencias avanzadas en ciencia de datos, machine learning y análisis predictivo.

⭐ Si este proyecto te fue útil, ¡considera darle una estrella! ⭐

---
Hecho con ❤️ por: [SynergyaTech](https://synergya.tech)
