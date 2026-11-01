import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("🔮 Predicción de Ventas Futuras (Superstore)")
st.write("Esta aplicación utiliza un modelo de Machine Learning para predecir las ventas de los próximos 7 días basándose en el historial.")

# Entrenamos el modelo directamente con tu DataFrame actual (df_ml)
# (En un entorno profesional se carga un archivo .pkl guardado previamente)
from sklearn.model_selection import train_test_split
X = df_ml[['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Dia_Semana', 'Mes']]
y = df_ml['Ventas_Proximos_7_Dias']

modelo = LinearRegression()
modelo.fit(X, y)

st.sidebar.header("Ingrese los datos de entrada:")
lag_ayer = st.sidebar.number_input("Ventas de Ayer (Sales_Lag_Ayer)", min_value=0.0, value=100.0)
lag_semana = st.sidebar.number_input("Ventas de hace una semana (Sales_Lag_Semana)", min_value=0.0, value=150.0)
dia_semana = st.sidebar.selectbox("Día de la semana (0=Lunes, 6=Domingo)", options=[0, 1, 2, 3, 4, 5, 6], index=0)
mes = st.sidebar.slider("Mes del año", min_value=1, max_value=12, value=1)

if st.button("Predecir Ventas de los Próximos 7 Días"):
    # Preparamos los datos ingresados
    datos_usuario = pd.DataFrame([[lag_ayer, lag_semana, dia_semana, mes]], 
                                 columns=['Sales_Lag_Ayer', 'Sales_Lag_Semana', 'Dia_Semana', 'Mes'])
    
    prediccion = modelo.predict(datos_usuario)[0]
    
    st.success(f"### 📈 Predicción estimada: ${prediccion:,.2f}")
    st.balloons()
