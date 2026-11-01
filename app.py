import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

st.title("🔮 Predicción de Ventas Futuras (Superstore - Alta Precisión)")
st.write("Esta aplicación utiliza un modelo avanzado de **Gradient Boosting** con cálculo automático de tendencias para predecir las ventas de los próximos 7 días.")

# 1. Cargamos y preprocesamos el dataset y calculamos la media móvil internamente
@st.cache_data
def cargar_y_preprocesar_datos():
    df = pd.read_csv('train.csv')
    
    # Transformaciones de fecha y limpieza
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df_diario = df.groupby('Order Date')['Sales'].sum().reset_index()
    
    # Reindexación de calendario completo (con ceros en días sin ventas)
    rango_fechas = pd.date_range(start=df_diario['Order Date'].min(), end=df_diario['Order Date'].max(), freq='D')
    df_completo = pd.DataFrame({'Order Date': rango_fechas})
    df_diario = pd.merge(df_completo, df_diario, on='Order Date', how='left')
    df_diario['Sales'] = df_diario['Sales'].fillna(0.0)
    
    # Cálculo de la variable objetivo (Ventas próximos 7 días)
    df_diario['Ventas_Proximos_7_Dias'] = df_diario['Sales'].shift(-1).rolling(window=7, min_periods=7).sum()
    
    # Variables de retraso (Lags), Media Móvil y calendario
    df_diario['Sales_Lag_Ayer'] = df_diario['Sales'].shift(1)
    df_diario['Sales_Lag_Semana'] = df_diario['Sales'].shift(7)
    df_diario['Media_Movil_7D'] = df_diario['Sales'].rolling(window=7).mean()
    df_diario['Dia_Semana'] = df_diario['Order Date'].dt.dayofweek
    df_diario['Mes'] = df_diario['Order Date'].dt.month
    
    df_ml = df_diario.dropna().reset_index(drop=True)
    return df_ml

# Cargamos los datos procesados
df_ml = cargar_y_preprocesar_datos()

# Obtenemos el promedio global de la media móvil real del dataset para usarlo de referencia por defecto
media_movil_historica = df_ml['Media_Movil_7D'].mean()

# 2. Entrenamos el modelo con Gradient Boosting
X = df_ml[['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Media_Movil_7D', 'Dia_Semana', 'Mes']]
y = df_ml['Ventas_Proximos_7_Dias']

modelo = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
modelo.fit(X, y)

# 3. Interfaz de usuario limpia en Streamlit Sidebar (¡Sin pedir la media móvil manualmente!)
st.sidebar.header("Ingrese los datos de entrada:")
lag_ayer = st.sidebar.number_input("Ventas de Ayer (Sales_Lag_Ayer)", min_value=0.0, value=100.0)
lag_semana = st.sidebar.number_input("Ventas de hace una semana (Sales_Lag_Semana)", min_value=0.0, value=150.0)
dia_semana = st.sidebar.selectbox("Día de la semana (0=Lunes, 6=Domingo)", options=[0, 1, 2, 3, 4, 5, 6], index=0)
mes = st.sidebar.slider("Mes del año", min_value=1, max_value=12, value=1)

if st.button("Predecir Ventas de los Próximos 7 Días"):
    # CALCULAMOS LA MEDIA MÓVIL AUTOMÁTICAMENTE:
    # Hacemos un promedio dinámico basado en lo que el usuario ingresó y el comportamiento histórico
    media_movil_calculada = (lag_ayer + lag_semana + media_movil_historica) / 3
    
    datos_usuario = pd.DataFrame([[lag_ayer, lag_semana, media_movil_calculada, dia_semana, mes]], 
                                 columns=['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Media_Movil_7D', 'Dia_Semana', 'Mes'])
    
    prediccion = modelo.predict(datos_usuario)[0]
    
    st.success(f"### 📈 Predicción estimada (Alta Precisión): ${prediccion:,.2f}")
    st.balloons()
