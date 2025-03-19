import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
import os
from dash.exceptions import PreventUpdate
import gc  # Para limpieza de memoria

# Crea la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # IMPORTANTE: Esto expone el servidor para Render

# Configuración para reducir el uso de memoria
MUESTREO_REDUCIDO = True  # Reducir el número de puntos de datos
CACHE_GRAFICOS = True     # Almacenar en caché los gráficos

# Función optimizada para generar datos (menos puntos, más eficiente)
def generar_datos_economicos_rd(muestreo_reducido=MUESTREO_REDUCIDO):
    # Ajustar el rango de fechas si se solicita muestreo reducido
    if muestreo_reducido:
        fechas = pd.date_range(start='2015-01-01', end='2024-10-01', freq='Q')  # Menos datos históricos
    else:
        fechas = pd.date_range(start='2010-01-01', end='2024-10-01', freq='Q')
    
    # Creamos DataFrame con las fechas
    df = pd.DataFrame({'fecha': fechas})
    df['año'] = df['fecha'].dt.year
    df['trimestre'] = df['fecha'].dt.quarter
    
    # Resto del código de generación de datos
    # [CÓDIGO ORIGINAL AQUÍ]
    
    # PIB en millones de pesos dominicanos (con tendencia creciente y estacionalidad)
    base_pib = 500000  # 500 mil millones como base
    tendencia = np.linspace(0, 1500000, len(fechas))  # Tendencia creciente
    estacionalidad = 50000 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Efecto estacional
    ruido = np.random.normal(0, 30000, len(fechas))  # Componente aleatorio
    
    # COVID impact (caída en 2020-Q2)
    covid_impact = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] == 2020) & (df['trimestre'] >= 2))[0]
    if len(covid_idx) > 0:
        covid_impact[covid_idx[0]] = -200000  # Fuerte caída
    if len(covid_idx) > 1:
        covid_impact[covid_idx[1]] = -150000  # Recuperación lenta
    if len(covid_idx) > 2:
        covid_impact[covid_idx[2]] = -100000  # Recuperación gradual
    
    df['pib'] = base_pib + tendencia + estacionalidad + ruido + covid_impact
    df['pib'] = df['pib'].clip(lower=0)  # Asegurarse que no hay valores negativos
    
    # Crecimiento del PIB (tasa interanual)
    df['pib_crecimiento'] = df['pib'].pct_change(4) * 100
    
    # Inflación (%)
    inflacion_base = 4.0  # Promedio histórico aproximado
    inflacion_tendencia = np.linspace(0, 1, len(fechas))  # Ligera tendencia al alza
    inflacion_estacional = 1.5 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Estacionalidad
    inflacion_ruido = np.random.normal(0, 1, len(fechas))  # Componente aleatorio
    
    # Pico de inflación post-COVID y por factores globales
    inflacion_picos = np.zeros(len(fechas))
    picos_idx = np.where((df['año'] >= 2021) & (df['año'] <= 2023))[0]
    inflacion_picos[picos_idx] = 3 * np.exp(-np.linspace(0, 2, len(picos_idx)))
    
    df['inflacion'] = inflacion_base + inflacion_tendencia + inflacion_estacional + inflacion_ruido + inflacion_picos
    
    # Tasa de desempleo (%)
    desempleo_base = 14.0  # Base alta
    desempleo_tendencia = -3 * np.linspace(0, 1, len(fechas))  # Tendencia a la baja con el tiempo
    desempleo_estacional = 1.0 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Estacionalidad
    desempleo_ruido = np.random.normal(0, 0.5, len(fechas))  # Componente aleatorio
    
    # Efecto COVID
    desempleo_covid = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] == 2020) & (df['trimestre'] >= 2))[0]
    if len(covid_idx) > 0:
        desempleo_covid[covid_idx[0]] = 5.0  # Aumento significativo
    if len(covid_idx) > 1:
        desempleo_covid[covid_idx[1]] = 4.5  # Disminución gradual
    if len(covid_idx) > 2:
        desempleo_covid[covid_idx[2]] = 3.0
    
    # Buscar el primer trimestre de 2021 para continuar el efecto
    idx_2021q1 = np.where((df['año'] == 2021) & (df['trimestre'] == 1))[0]
    if len(idx_2021q1) > 0:
        desempleo_covid[idx_2021q1[0]] = 2.0
    
    df['desempleo'] = desempleo_base + desempleo_tendencia + desempleo_estacional + desempleo_ruido + desempleo_covid
    
    # Tipo de cambio USD/DOP
    tc_base = 36.0  # Valor para 2010
    tc_tendencia = 18 * np.linspace(0, 1, len(fechas))  # Depreciación gradual
    tc_estacional = 0.5 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Estacionalidad menor
    tc_ruido = np.random.normal(0, 0.3, len(fechas))  # Ruido limitado
    
    # Aceleración después de COVID
    tc_aceleracion = np.zeros(len(fechas))
    accel_idx = np.where(df['año'] >= 2020)[0]
    tc_aceleracion[accel_idx] = np.linspace(0, 3, len(accel_idx))
    
    df['tipo_cambio'] = tc_base + tc_tendencia + tc_estacional + tc_ruido + tc_aceleracion
    
    # Exportaciones (millones USD)
    exp_base = 1500
    exp_tendencia = 2000 * np.linspace(0, 1, len(fechas))
    exp_estacional = 300 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))
    exp_ruido = np.random.normal(0, 100, len(fechas))
    
    df['exportaciones'] = exp_base + exp_tendencia + exp_estacional + exp_ruido
    
    # Importaciones (millones USD)
    imp_base = 2500
    imp_tendencia = 3000 * np.linspace(0, 1, len(fechas))
    imp_estacional = 400 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))
    imp_ruido = np.random.normal(0, 150, len(fechas))
    
    df['importaciones'] = imp_base + imp_tendencia + imp_estacional + imp_ruido
    
    # Deuda pública (% del PIB)
    deuda_base = 35.0
    deuda_tendencia = 10 * np.linspace(0, 1, len(fechas))
    deuda_ruido = np.random.normal(0, 1, len(fechas))
    
    # Aumento por COVID
    deuda_covid = np.zeros(len(fechas))
    covid_idx = np.where(df['año'] >= 2020)[0]
    deuda_covid[covid_idx] = 8.0
    
    df['deuda_publica'] = deuda_base + deuda_tendencia + deuda_ruido + deuda_covid
    
    # Reservas internacionales (millones USD)
    reservas_base = 3000
    reservas_tendencia = 9000 * np.linspace(0, 1, len(fechas))
    reservas_ruido = np.random.normal(0, 200, len(fechas))
    
    df['reservas_internacionales'] = reservas_base + reservas_tendencia + reservas_ruido
    
    # Inversión extranjera directa (millones USD)
    ied_base = 2000
    ied_tendencia = 1500 * np.linspace(0, 1, len(fechas))
    ied_estacional = 500 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))
    ied_ruido = np.random.normal(0, 300, len(fechas))
    
    # Caída y recuperación por COVID
    ied_covid = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] == 2020))[0]
    ied_covid[covid_idx] = -800
    
    df['inversion_extranjera'] = ied_base + ied_tendencia + ied_estacional + ied_ruido + ied_covid
    df['inversion_extranjera'] = df['inversion_extranjera'].clip(lower=0)
    
    # Ingresos por turismo (millones USD)
    turismo_base = 5000
    turismo_tendencia = 3000 * np.linspace(0, 1, len(fechas))
    turismo_estacional = 1500 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Fuerte estacionalidad
    turismo_ruido = np.random.normal(0, 200, len(fechas))
    
    # Colapso por COVID
    turismo_covid = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] == 2020) & (df['trimestre'] >= 2))[0]
    if len(covid_idx) > 0:
        turismo_covid[covid_idx[0]] = -6000  # Caída extrema
    if len(covid_idx) > 1:
        turismo_covid[covid_idx[1]] = -5000  # Recuperación muy lenta
    if len(covid_idx) > 2:
        turismo_covid[covid_idx[2]] = -4000
    
    recovery_idx = np.where((df['año'] == 2021))[0]
    turismo_covid[recovery_idx] = -3000 + np.linspace(0, 2500, len(recovery_idx))
    
    df['ingresos_turismo'] = turismo_base + turismo_tendencia + turismo_estacional + turismo_ruido + turismo_covid
    df['ingresos_turismo'] = df['ingresos_turismo'].clip(lower=0)
    
    # Remesas (millones USD)
    remesas_base = 3000
    remesas_tendencia = 4000 * np.linspace(0, 1, len(fechas))
    remesas_estacional = 500 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))
    remesas_ruido = np.random.normal(0, 150, len(fechas))
    
    # Aumento por COVID (las remesas aumentaron durante la pandemia)
    remesas_covid = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] >= 2020) & (df['año'] <= 2021))[0]
    remesas_covid[covid_idx] = 700
    
    df['remesas'] = remesas_base + remesas_tendencia + remesas_estacional + remesas_ruido + remesas_covid
    
    # Limpieza final y redondeo para más realismo
    numeric_cols = df.columns.drop(['fecha', 'año', 'trimestre'])
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Agregar etiquetas de fecha en formato string para facilizar visualizaciones
    df['fecha_str'] = df.apply(lambda x: f"{x['año']}-Q{x['trimestre']}", axis=1)
    
    # Liberar memoria
    gc.collect()
    
    return df

# Variable global para almacenar los datos (evitar regenerarlos)
df_economia = generar_datos_economicos_rd()

# Configuración de estilos y colores
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#00a65a',
    'secondary': '#0038a8',
    'accent': '#ce1126',
    'light': '#f5f5f5'
}

# Estilo general
app_style = {
    'backgroundColor': colors['background'],
    'color': colors['text'],
    'fontFamily': 'Arial, sans-serif',
    'margin': '0px',
    'padding': '0px'
}

# Estilo para títulos
title_style = {
    'color': colors['secondary'],
    'padding': '20px 10px',
    'textAlign': 'center',
    'fontSize': '28px',
    'fontWeight': 'bold',
    'borderBottom': f'3px solid {colors["accent"]}',
    'marginBottom': '20px'
}

# Estilo para subtítulos
subtitle_style = {
    'color': colors['primary'],
    'padding': '10px',
    'fontSize': '18px',
    'fontWeight': 'bold',
    'borderBottom': f'1px solid {colors["primary"]}',
    'marginBottom': '15px'
}

# Estilo para tarjetas/paneles
card_style = {
    'backgroundColor': 'white',
    'borderRadius': '5px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'padding': '15px',
    'marginBottom': '20px'
}

# Función para crear indicadores (optimizada)
def create_indicator(title, value, change):
    # Determinar color según si el cambio es positivo o negativo
    if change.startswith('+'):
        change_color = colors['primary']  # verde para positivo
    else:
        change_color = colors['accent']  # rojo para negativo
        
    return html.Div([
        html.P(title, style={'fontSize': '14px', 'color': '#666', 'margin': '0px'}),
        html.H4(value, style={'fontSize': '24px', 'margin': '5px 0'}),
        html.P(f"Variación: {change}", 
               style={'fontSize': '12px', 'color': change_color, 'margin': '0px'})
    ], style={'width': '30%', 'padding': '15px', 'backgroundColor': colors['light'], 'borderRadius': '5px'})

# Gráficos precalculados para reducir procesamiento
_cached_graphs = {}

# Gráfico de evolución del PIB (con caché)
def create_pib_evolution_graph():
    # Si está activado el caché y el gráfico ya existe, devolverlo
    if CACHE_GRAFICOS and 'pib_evolution' in _cached_graphs:
        return _cached_graphs['pib_evolution']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Línea para el PIB
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['pib'],
                  mode='lines', name='PIB (MM RD$)',
                  line=dict(color=colors['primary'], width=3)),
        secondary_y=False
    )
    
    # Barras para el crecimiento
    fig.add_trace(
        go.Bar(x=df_economia['fecha'], y=df_economia['pib_crecimiento'],
              name='Crecimiento (%)',
              marker=dict(color=df_economia['pib_crecimiento'].apply(
                  lambda x: colors['primary'] if x >= 0 else colors['accent']))),
        secondary_y=True
    )
    
    # Configuración de ejes y diseño
    fig.update_layout(
        title='Evolución del PIB y Tasa de Crecimiento',
        xaxis_title='Período',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="PIB (Millones RD$)", secondary_y=False)
    fig.update_yaxes(title_text="Crecimiento del PIB (%)", secondary_y=True)
    
    # Almacenar en caché
    if CACHE_GRAFICOS:
        _cached_graphs['pib_evolution'] = fig
    
    return fig

# Resto de funciones para crear gráficos
# [RESTO DE TU CÓDIGO AQUÍ]

# Gráfico de inflación y desempleo (con caché)
def create_inflation_unemployment_graph():
    # Si está activado el caché y el gráfico ya existe, devolverlo
    if CACHE_GRAFICOS and 'inflation_unemployment' in _cached_graphs:
        return _cached_graphs['inflation_unemployment']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Línea para inflación
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['inflacion'],
                  mode='lines', name='Inflación (%)',
                  line=dict(color=colors['accent'], width=3)),
        secondary_y=False
    )
    
    # Línea para desempleo
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['desempleo'],
                  mode='lines', name='Desempleo (%)',
                  line=dict(color=colors['secondary'], width=3)),
        secondary_y=True
    )
    
    # Configuración de ejes y diseño
    fig.update_layout(
        title='Evolución de Inflación y Desempleo',
        xaxis_title='Período',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Inflación (%)", secondary_y=False)
    fig.update_yaxes(title_text="Desempleo (%)", secondary_y=True)
    
    # Almacenar en caché
    if CACHE_GRAFICOS:
        _cached_graphs['inflation_unemployment'] = fig
    
    return fig

# Layout principal de la aplicación (versión simplificada)
app.layout = html.Div(style=app_style, children=[
    html.Div([
        html.H1("Análisis Económico de República Dominicana", style=title_style),
        html.P("Dashboard interactivo con machine learning para análisis y predicción de indicadores económicos", 
               style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '30px'})
    ]),
    
    # Menú de navegación (solo dos pestañas para reducir carga)
    html.Div([
        dcc.Tabs(id='tabs', value='tab-overview', children=[
            dcc.Tab(label='Vista General', value='tab-overview',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}),
            dcc.Tab(label='Análisis Detallado', value='tab-detailed',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'})
        ], style={'marginBottom': '20px'})
    ]),
    
    # Contenido de los tabs
    html.Div(id='tabs-content')
])

# Callback para actualizar el contenido de los tabs
@callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    # Limpiar cache innecesario
    gc.collect()
    
    if tab == 'tab-overview':
        return html.Div([
            html.Div([
                # Indicadores principales
                html.Div(style=card_style, children=[
                    html.H3("Indicadores Económicos Actuales", style=subtitle_style),
                    html.Div([
                        html.Div([
                            create_indicator("PIB (Último Trimestre)", 
                                            f"{df_economia['pib'].iloc[-1]:,.2f} MM RD$",
                                            f"{df_economia['pib_crecimiento'].iloc[-1]:+.2f}%"),
                            create_indicator("Inflación", 
                                            f"{df_economia['inflacion'].iloc[-1]:.2f}%",
                                            f"{df_economia['inflacion'].iloc[-1] - df_economia['inflacion'].iloc[-5]:+.2f} pts"),
                            create_indicator("Desempleo", 
                                            f"{df_economia['desempleo'].iloc[-1]:.2f}%",
                                            f"{df_economia['desempleo'].iloc[-1] - df_economia['desempleo'].iloc[-5]:+.2f} pts"),
                        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                    ])
                ]),
                
                # Gráfico de evolución histórica del PIB
                html.Div(style=card_style, children=[
                    html.H3("Evolución del PIB", style=subtitle_style),
                    dcc.Graph(
                        id='graph-pib-evolution',
                        figure=create_pib_evolution_graph()
                    )
                ]),
                
                # Gráfico de evolución de inflación y desempleo
                html.Div(style=card_style, children=[
                    html.H3("Inflación y Desempleo", style=subtitle_style),
                    dcc.Graph(
                        id='graph-inflation-unemployment',
                        figure=create_inflation_unemployment_graph()
                    )
                ])
            ])
        ])
    
    elif tab == 'tab-detailed':
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("Análisis Detallado - Versión Optimizada", style=subtitle_style),
                html.P("Esta es una versión optimizada para despliegue en Render. Para ver todas las funcionalidades, ejecute la aplicación localmente."),
                html.P("La aplicación completa incluye:"),
                html.Ul([
                    html.Li("Análisis Detallado de múltiples indicadores"),
                    html.Li("Modelos de Machine Learning para predicciones"),
                    html.Li("Simulador de escenarios económicos"),
                    html.Li("Análisis de correlación entre variables"),
                ])
            ])
        ])

# Punto de entrada para el servidor
if __name__ == '__main__':
    # Para desarrollo local
    app.run_server(debug=True)
    # Para producción en Render, se usa el objeto server exportado