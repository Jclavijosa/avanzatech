# app.py

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Credenciales de usuarios permitidos
USER_CREDENTIALS = {
    'admin': 'Avanza2024*',
    'avanza': 'Tech2024*'
}

def autenticar(usuario, contraseña):
    """Función para autenticar al usuario."""
    if usuario in USER_CREDENTIALS and USER_CREDENTIALS[usuario] == contraseña:
        return True
    else:
        return False

class AdvancedBeverageModel:
    """Clase que encapsula los modelos y métodos para el análisis avanzado del sector de bebidas."""
    def __init__(self):
        self.prophet_models = {}
        self.rf_models = {}
        self.lstm_models = {}
        self.scalers = {}
        self.data_scalers = {}
        self.data = None  # Almacenar datos preparados

    def normalize_data(self, data, sector, variable):
        """Normaliza los datos a una escala manejable."""
        key = f"{sector}_{variable}"
        if key not in self.data_scalers:
            self.data_scalers[key] = StandardScaler()
            data_reshaped = data.reshape(-1, 1)
            return self.data_scalers[key].fit_transform(data_reshaped).flatten()
        return self.data_scalers[key].transform(data.reshape(-1, 1)).flatten()

    def denormalize_predictions(self, predictions, sector, variable):
        """Desnormaliza las predicciones."""
        key = f"{sector}_{variable}"
        if key in self.data_scalers:
            return self.data_scalers[key].inverse_transform(
                predictions.reshape(-1, 1)).flatten()
        return predictions

    def prepare_data(self, df):
        """Preparación de datos con normalización."""
        sectors = ['1101', '1102', '1103', '1104']
        prepared_data = {}
        
        for sector in sectors:
            sector_data = pd.DataFrame()
            try:
                sector_data['ds'] = pd.to_datetime(df['año'].astype(str), format='%Y')
            except KeyError:
                st.error("La columna 'año' no se encuentra en el DataFrame.")
                st.stop()
            
            missing_columns = []
            for var in ['empleo', 'produccion', 'venta']:
                column_name = f'{var}_{sector}'
                if column_name in df.columns:
                    original_series = df[column_name].values
                    normalized_series = self.normalize_data(original_series, sector, var)
                    if var in ['produccion', 'venta']:
                        normalized_series = np.log1p(np.abs(normalized_series))
                    sector_data[var] = normalized_series
                else:
                    missing_columns.append(column_name)
            
            if missing_columns:
                st.warning(f"Advertencia: Las siguientes columnas están faltando para el sector {sector}: {missing_columns}")
                continue  # Saltamos este sector si faltan columnas esenciales
            
            if 'inflacion' in df.columns:
                sector_data['inflacion'] = self.normalize_data(df['inflacion'].values, sector, 'inflacion')
            else:
                st.warning("La columna 'inflacion' no se encuentra en el DataFrame.")
                sector_data['inflacion'] = 0  # Puedes asignar un valor por defecto o manejarlo según tus necesidades
            
            prepared_data[sector] = sector_data
        
        self.data = prepared_data  # Guardamos los datos preparados
        return prepared_data

    def create_features(self, data, target_col, lookback=1):
        """Creación de características con manejo de escala."""
        df = data.copy()
        for col in ['empleo', 'produccion', 'venta', 'inflacion']:
            if col in df.columns:
                for i in range(1, lookback + 1):
                    df[f'{col}_lag_{i}'] = df[col].shift(i)
                df[f'{col}_diff'] = df[col].diff()
            else:
                st.warning(f"La columna '{col}' no está presente en los datos.")
        df['year'] = df['ds'].dt.year
        df['quarter'] = df['ds'].dt.quarter
        # Rellenar valores nulos
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df

    def train_prophet(self, data, target_col):
        """Entrenamiento de Prophet con manejo de escala."""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        df_prophet = pd.DataFrame({'ds': data['ds'], 'y': data[target_col]})
        model.fit(df_prophet)
        return model

    def create_lstm_model(self, input_shape):
        """Crea y compila el modelo LSTM."""
        model = Sequential([
            LSTM(32, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber')
        return model

    def train_models(self, data, validation_split=0.2):
        """Entrenamiento de los modelos Prophet, Random Forest y LSTM."""
        sectors = list(data.keys())
        targets = ['produccion', 'venta', 'empleo']

        for sector in sectors:
            sector_data = data[sector]
            for target in targets:
                if target not in sector_data.columns:
                    st.warning(f"La variable '{target}' no está disponible para el sector {sector}.")
                    continue
                # Prophet
                prophet_model = self.train_prophet(sector_data, target)
                self.prophet_models[f'{sector}_{target}'] = prophet_model

                # Random Forest
                features_df = self.create_features(sector_data, target, lookback=1)
                feature_cols = [col for col in features_df.columns if col not in ['ds', target]]
                X_rf = features_df[feature_cols]
                y_rf = features_df[target]

                # Verificar si hay suficientes datos
                if len(X_rf) < 2:
                    st.warning(f"No hay suficientes datos para entrenar Random Forest para {sector} - {target}")
                    continue

                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                rf_model.fit(X_rf, y_rf)
                self.rf_models[f'{sector}_{target}'] = rf_model

                # LSTM
                if len(sector_data[target]) < 2:
                    st.warning(f"No hay suficientes datos para entrenar LSTM para {sector} - {target}")
                    continue

                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(sector_data[target].values.reshape(-1, 1))
                self.scalers[f'{sector}_{target}'] = scaler

                X_lstm, y_lstm = [], []
                lookback = 1
                for i in range(len(scaled_data) - lookback):
                    X_lstm.append(scaled_data[i:i + lookback])
                    y_lstm.append(scaled_data[i + lookback])

                X_lstm = np.array(X_lstm)
                y_lstm = np.array(y_lstm)

                if X_lstm.size == 0 or y_lstm.size == 0:
                    st.warning(f"No hay suficientes datos para entrenar LSTM para {sector} - {target}")
                    continue

                lstm_model = self.create_lstm_model((lookback, 1))
                lstm_model.fit(
                    X_lstm, y_lstm,
                    epochs=50,
                    batch_size=32,
                    validation_split=validation_split,
                    verbose=0
                )
                self.lstm_models[f'{sector}_{target}'] = lstm_model

    def predict(self, sector, target, future_periods=12, scenario='base'):
        """Realiza predicciones utilizando los modelos entrenados."""
        scenario_factors = {
            'optimista': 1.1,
            'base': 1.0,
            'pesimista': 0.9
        }
        factor = scenario_factors.get(scenario, 1.0)
        future_dates = pd.DataFrame({
            'ds': pd.date_range(
                start='2023-01-01',
                periods=future_periods,
                freq='Y'
            )
        })

        # Predicción con Prophet
        prophet_key = f'{sector}_{target}'
        if prophet_key not in self.prophet_models:
            st.error(f"Modelo Prophet no entrenado para {sector} - {target}")
            st.stop()
        prophet_pred = self.prophet_models[prophet_key].predict(future_dates)
        prophet_predictions = prophet_pred['yhat'].values * factor
        prophet_predictions = self.denormalize_predictions(prophet_predictions, sector, target)

        # Predicción con Random Forest
        if prophet_key in self.rf_models:
            features_df = self.create_features(self.data[sector], target, lookback=1)
            feature_cols = [col for col in features_df.columns if col not in ['ds', target]]
            last_features = features_df[feature_cols].iloc[-1].values.reshape(1, -1)
            rf_predictions = []
            for _ in range(future_periods):
                rf_pred = self.rf_models[prophet_key].predict(last_features)[0] * factor
                rf_predictions.append(rf_pred)
                # Actualizar las características para la siguiente predicción
                new_row = np.append(last_features[0][1:], rf_pred)
                last_features = new_row.reshape(1, -1)
        else:
            rf_predictions = prophet_predictions

        rf_predictions = np.array(rf_predictions)
        rf_predictions = self.denormalize_predictions(rf_predictions, sector, target)

        # Predicción con LSTM
        if prophet_key in self.lstm_models:
            scaler = self.scalers[prophet_key]
            lstm_data = scaler.transform(self.data[sector][target].values.reshape(-1, 1))
            X_lstm = lstm_data[-1].reshape(1, 1, 1)
            lstm_predictions = []
            for _ in range(future_periods):
                lstm_pred = self.lstm_models[prophet_key].predict(X_lstm)[0][0] * factor
                lstm_predictions.append(lstm_pred)
                X_lstm = np.array([[[lstm_pred]]])
            lstm_predictions = np.array(lstm_predictions)
            lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
            lstm_predictions = self.denormalize_predictions(lstm_predictions, sector, target)
        else:
            lstm_predictions = prophet_predictions

        # Combinación de predicciones (promedio simple)
        combined_predictions = (prophet_predictions + rf_predictions + lstm_predictions) / 3

        # Intervalos de confianza (usando Prophet)
        confidence_intervals = {
            'lower': self.denormalize_predictions(prophet_pred['yhat_lower'].values * factor, sector, target),
            'upper': self.denormalize_predictions(prophet_pred['yhat_upper'].values * factor, sector, target)
        }

        return {
            'predictions': combined_predictions,
            'confidence_intervals': confidence_intervals
        }

    def evaluate_models(self, data, sector, target):
        """Evaluación del modelo con métricas robustas."""
        tscv = TimeSeriesSplit(n_splits=3)
        metrics = {'mae': [], 'rmse': [], 'mape': []}

        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Entrenar modelos con datos de entrenamiento
            self.train_models({sector: train_data}, validation_split=0.0)

            pred_result = self.predict(sector, target, future_periods=len(test_data))
            predictions = pred_result['predictions']
            actual = self.denormalize_predictions(test_data[target].values, sector, target)

            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mape = np.mean(np.abs((actual - predictions) / (actual + 1e-6))) * 100

            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['mape'].append(mape)

        # Promedio de las métricas
        return {k: np.mean(v) for k, v in metrics.items()}

# Configuración de la aplicación Streamlit
SECTOR_NAMES = {
    '1101': 'Destilación y bebidas alcohólicas',
    '1102': 'Bebidas fermentadas no destiladas',
    '1103': 'Cervezas y bebidas malteadas',
    '1104': 'Bebidas no alcohólicas y aguas'
}

def load_data():
    """Carga los datos desde un archivo CSV."""
    try:
        df = pd.read_csv('Data.csv', encoding='utf-8')
        if df.empty:
            st.error("El archivo 'Data.csv' está vacío o no se cargó correctamente.")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("El archivo 'Data.csv' no se encontró en el directorio.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar 'Data.csv': {e}")
        st.stop()

def initialize_model():
    """Inicializa el modelo y prepara los datos."""
    try:
        model = AdvancedBeverageModel()
        df = load_data()
        # Imprimir columnas para depuración
        st.write("Columnas del DataFrame:", df.columns.tolist())
        prepared_data = model.prepare_data(df)
        # Imprimir sectores disponibles
        st.write("Sectores en prepared_data:", list(prepared_data.keys()))
        model.train_models(prepared_data)
        return model, prepared_data
    except Exception as e:
        st.error(f"Error al inicializar el modelo: {e}")
        st.stop()

def run_dashboard(usuario):
    """Función que ejecuta el dashboard principal."""
    st.sidebar.write(f"Usuario: {usuario}")
    if st.sidebar.button("Cerrar sesión"):
        st.session_state['authenticated'] = False
        st.session_state['usuario'] = ''
        st.rerun()

    # Inicio del dashboard
    st.title("🍺 Análisis Avanzado del Sector de Bebidas")
    
    try:
        with st.spinner('Inicializando modelo y cargando datos...'):
            df = load_data()
            model, prepared_data = initialize_model()
        
        st.sidebar.header("Configuración")
        
        available_sectors = list(prepared_data.keys())
        selected_sector = st.sidebar.selectbox(
            "Sector",
            options=available_sectors,
            format_func=lambda x: SECTOR_NAMES.get(x, x)
        )
        
        selected_variable = st.sidebar.selectbox(
            "Variable",
            options=['produccion', 'venta', 'empleo'],
            format_func=lambda x: x.title()
        )
        
        tab1, tab2, tab3 = st.tabs([
            "Predicciones y Simulaciones",
            "Análisis Estadístico",
            "Evaluación del Modelo"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                scenarios = ['optimista', 'base', 'pesimista']
                predictions = {}
                
                for scenario in scenarios:
                    pred_result = model.predict(
                        selected_sector,
                        selected_variable,
                        scenario=scenario
                    )
                    predictions[scenario] = pred_result
                
                fig = go.Figure()
                
                years = list(range(2023, 2023 + len(predictions['base']['predictions'])))
                
                for scenario in scenarios:
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=predictions[scenario]['predictions'],
                        name=scenario.title(),
                        mode='lines+markers'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=list(predictions[scenario]['confidence_intervals']['upper']) + 
                           list(predictions[scenario]['confidence_intervals']['lower'])[::-1],
                        fill='toself',
                        fillcolor=f'rgba(0,100,80,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'IC {scenario}'
                    ))
                
                fig.update_layout(
                    title=f'Predicciones para {SECTOR_NAMES.get(selected_sector, selected_sector)}',
                    xaxis_title='Año',
                    yaxis_title=selected_variable.title()
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Ajustes de Simulación")
                
                inflacion_adj = st.slider(
                    "Ajuste de Inflación (%)",
                    -5.0, 5.0, 0.0, 0.1
                )
                
                crecimiento_sector = st.slider(
                    "Crecimiento del Sector (%)",
                    -10.0, 10.0, 0.0, 0.1
                )
        
        with tab2:
            st.subheader("Análisis Estadístico")
            
            # Obtener datos originales para el sector seleccionado
            column_name = f'{selected_variable}_{selected_sector}'
            if column_name in df.columns:
                original_data = df[[column_name, 'año']]
                
                # Estadísticas descriptivas
                st.write("Estadísticas Descriptivas:")
                st.dataframe(original_data[column_name].describe())
                
                # Gráfico de tendencia histórica
                fig_hist = px.line(
                    original_data,
                    x='año',
                    y=column_name,
                    title=f'Tendencia Histórica - {SECTOR_NAMES.get(selected_sector, selected_sector)}'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Agregar análisis de estacionalidad
                if st.checkbox("Mostrar Análisis de Estacionalidad"):
                    seasonal_data = df.pivot(index='año', 
                                           columns=None, 
                                           values=column_name)
                    fig_seasonal = px.box(seasonal_data)
                    st.plotly_chart(fig_seasonal, use_container_width=True)
            else:
                st.warning(f"No hay datos disponibles para {column_name}")
        
        with tab3:
            st.subheader("Evaluación del Modelo")
            
            # Verificar si hay suficientes datos para evaluar el modelo
            if selected_sector in prepared_data:
                metrics = model.evaluate_models(
                    prepared_data[selected_sector],
                    selected_sector,
                    selected_variable
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                
                # Agregar explicación de métricas
                with st.expander("Explicación de Métricas"):
                    st.write("""
                    - **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos en las predicciones.
                    - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del promedio de los errores al cuadrado. Penaliza más los errores grandes.
                    - **MAPE (Error Porcentual Absoluto Medio)**: Promedio de los errores porcentuales. Indica el error en términos relativos.
                    """)
    
                # Agregar visualización de errores de predicción
                if st.checkbox("Mostrar Análisis de Errores"):
                    st.subheader("Distribución de Errores de Predicción")
                    # Calcular errores de predicción históricos
                    historical_predictions = model.predict(
                        selected_sector,
                        selected_variable,
                        future_periods=len(prepared_data[selected_sector])
                    )['predictions']
                    
                    actual_values = model.denormalize_predictions(
                        prepared_data[selected_sector][selected_variable].values,
                        selected_sector,
                        selected_variable
                    )
                    prediction_errors = actual_values - historical_predictions[:len(actual_values)]
                    
                    fig_errors = px.histogram(
                        prediction_errors,
                        title="Distribución de Errores de Predicción",
                        labels={'value': 'Error', 'count': 'Frecuencia'}
                    )
                    st.plotly_chart(fig_errors, use_container_width=True)
            else:
                st.warning(f"No hay suficientes datos para evaluar el modelo para el sector {selected_sector}")
    
    except Exception as e:
        st.error(f"""
        Error: {str(e)}
        
        Verifica que:
        1. El archivo 'Data.csv' está en el directorio correcto
        2. El formato de los datos es correcto
        3. Todas las dependencias están instaladas
        
        Detalles técnicos del error para depuración:
        {str(e.__class__.__name__)}: {str(e)}
        """)
        st.stop()

def main():
    """Función principal de la aplicación Streamlit."""
    st.set_page_config(page_title="Análisis Avanzado del Sector de Bebidas", 
                       layout="wide")
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        # Logo en la página de login
        st.image("logo.jpg", width=200)
        
        st.title("🍺 Análisis Avanzado del Sector de Bebidas")
        st.subheader("Por favor, inicia sesión para continuar.")
        
        usuario = st.text_input("Usuario")
        contraseña = st.text_input("Contraseña", type="password")
        boton_login = st.button("Iniciar Sesión")
        
        if boton_login:
            if autenticar(usuario, contraseña):
                st.session_state['authenticated'] = True
                st.session_state['usuario'] = usuario
                st.success(f"Bienvenido, {usuario}!")
                st.rerun()  # Reemplazado por st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos. Inténtalo de nuevo.")
    else:
        usuario = st.session_state['usuario']
        run_dashboard(usuario)

if __name__ == "__main__":
    main()
