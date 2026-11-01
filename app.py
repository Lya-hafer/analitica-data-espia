import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("🛍️ Predicción de Ventas Futuras (Superstore) para ACA I Aprendizaje Autónomo 🛍️" )
st.write("Para cumplir con dicho ACA se toma el dataset de Sales, donde el objetivo es predecir las ventas de los próximos 7 días.")

# Cargar el dataset
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
    
    # Cálculo de la variable objetivo X (Ventas próximos 7 días)
    df_diario['Ventas_Proximos_7_Dias'] = df_diario['Sales'].shift(-1).rolling(window=7, min_periods=7).sum()
    
    # Variables de retraso (Lags) y calendario
    df_diario['Sales_Lag_Ayer'] = df_diario['Sales'].shift(1)
    df_diario['Sales_Lag_Semana'] = df_diario['Sales'].shift(7)
    df_diario['Dia_Semana'] = df_diario['Order Date'].dt.dayofweek
    df_diario['Mes'] = df_diario['Order Date'].dt.month
    
    df_ml = df_diario.dropna().reset_index(drop=True)
    return df_ml

# Cargar los datos procesados
df_ml = cargar_y_preprocesar_datos()

#Entrenar el modelo con los datos procesados de las variables X y Y
X = df_ml[['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Dia_Semana', 'Mes']]
y = df_ml['Ventas_Proximos_7_Dias']

#Seleccion de modelo regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

#Interfaz en Streamlit Sidebar
st.sidebar.header("Ingrese los datos de entrada:")
lag_ayer = st.sidebar.number_input("Ventas de Ayer (Sales_Lag_Ayer)", min_value=0.0, value=100.0)
lag_semana = st.sidebar.number_input("Ventas de hace una semana (Sales_Lag_Semana)", min_value=0.0, value=150.0)
dia_semana = st.sidebar.selectbox("Día de la semana: tenga en cuenta que el domingo inicia con el número 0", options=[0, 1, 2, 3, 4, 5, 6], index=0)
mes = st.sidebar.slider("Mes del año (número)", min_value=1, max_value=12, value=1)

if st.button("Predecir Ventas de los Próximos 7 Días"):
    datos_usuario = pd.DataFrame([[lag_ayer, lag_semana, dia_semana, mes]], 
                                 columns=['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Dia_Semana', 'Mes'])
    
    prediccion = modelo.predict(datos_usuario)[0]
    
    st.success(f"### 📈 Predicción estimada: ${prediccion:,.2f}")
    st.balloons()
