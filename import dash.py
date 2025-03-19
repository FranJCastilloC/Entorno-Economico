import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
import os
from dash.exceptions import PreventUpdate
import base64
import io

# Creación de datos de ejemplo para República Dominicana
# En un proyecto real, estos datos se obtendrían de fuentes oficiales como el Banco Central

def generar_datos_economicos_rd():
    # Generamos datos históricos desde 2010 hasta 2024
    fechas = pd.date_range(start='2010-01-01', end='2024-10-01', freq='Q')
    
    # Creamos DataFrame con las fechas
    df = pd.DataFrame({'fecha': fechas})
    df['año'] = df['fecha'].dt.year
    df['trimestre'] = df['fecha'].dt.quarter
    
    # PIB en millones de pesos dominicanos (con tendencia creciente y estacionalidad)
    base_pib = 500000  # 500 mil millones como base para 2010
    tendencia = np.linspace(0, 1500000, len(fechas))  # Tendencia creciente
    estacionalidad = 50000 * np.sin(np.linspace(0, 8*np.pi, len(fechas)))  # Efecto estacional
    ruido = np.random.normal(0, 30000, len(fechas))  # Componente aleatorio
    
    # COVID impact (caída en 2020-Q2)
    covid_impact = np.zeros(len(fechas))
    covid_idx = np.where((df['año'] == 2020) & (df['trimestre'] >= 2))[0]
    covid_impact[covid_idx[0]] = -200000  # Fuerte caída
    covid_impact[covid_idx[1]] = -150000  # Recuperación lenta
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
    turismo_covid[covid_idx[0]] = -6000  # Caída extrema
    turismo_covid[covid_idx[1]] = -5000  # Recuperación muy lenta
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
    
    # Agregar etiquetas de fecha en formato string para facilitar visualizaciones
    df['fecha_str'] = df.apply(lambda x: f"{x['año']}-Q{x['trimestre']}", axis=1)
    
    return df

# Configuración inicial
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Generar los datos
df_economia = generar_datos_economicos_rd()

# Aseguramos que exista el directorio para modelos
if not os.path.exists('models'):
    os.makedirs('models')

# Configuración de estilos y colores
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#00a65a',  # Verde esmeralda (color bandera RD)
    'secondary': '#0038a8', # Azul (color bandera RD)
    'accent': '#ce1126',   # Rojo (color bandera RD)
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

# Layout principal de la aplicación
app.layout = html.Div(style=app_style, children=[
    html.Div([
        html.H1("Análisis Económico de República Dominicana", style=title_style),
        html.P("Dashboard interactivo con machine learning para análisis y predicción de indicadores económicos", 
               style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '30px'})
    ]),
    
    # Menú de navegación
    html.Div([
        dcc.Tabs(id='tabs', value='tab-overview', children=[
            dcc.Tab(label='Vista General', value='tab-overview',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}),
            dcc.Tab(label='Análisis Detallado', value='tab-detailed',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}),
            dcc.Tab(label='Predicciones', value='tab-predictions',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}),
            dcc.Tab(label='Simulaciones', value='tab-simulations',
                    style={'backgroundColor': colors['light'], 'color': colors['text']},
                    selected_style={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'}),
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
                        html.Div([
                            create_indicator("Tipo de Cambio", 
                                            f"{df_economia['tipo_cambio'].iloc[-1]:.2f} DOP/USD",
                                            f"{df_economia['tipo_cambio'].iloc[-1] - df_economia['tipo_cambio'].iloc[-5]:+.2f}"),
                            create_indicator("Deuda Pública", 
                                            f"{df_economia['deuda_publica'].iloc[-1]:.2f}% del PIB",
                                            f"{df_economia['deuda_publica'].iloc[-1] - df_economia['deuda_publica'].iloc[-5]:+.2f} pts"),
                            create_indicator("Reservas Internacionales", 
                                            f"{df_economia['reservas_internacionales'].iloc[-1]:,.2f} MM USD",
                                            f"{df_economia['reservas_internacionales'].iloc[-1] - df_economia['reservas_internacionales'].iloc[-5]:+.2f} MM"),
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'})
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
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Balanza Comercial y Sectores Clave", style=subtitle_style),
                dcc.Graph(
                    id='graph-trade-tourism',
                    figure=create_trade_tourism_graph()
                )
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Remesas e Inversión Extranjera", style=subtitle_style),
                dcc.Graph(
                    id='graph-remittances-fdi',
                    figure=create_remittances_fdi_graph()
                )
            ])
        ])
    
    elif tab == 'tab-detailed':
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("Análisis Detallado de Indicadores", style=subtitle_style),
                html.P("Seleccione los indicadores que desea analizar:"),
                dcc.Dropdown(
                    id='indicator-selection',
                    options=[
                        {'label': 'PIB', 'value': 'pib'},
                        {'label': 'Inflación', 'value': 'inflacion'},
                        {'label': 'Desempleo', 'value': 'desempleo'},
                        {'label': 'Tipo de Cambio', 'value': 'tipo_cambio'},
                        {'label': 'Exportaciones', 'value': 'exportaciones'},
                        {'label': 'Importaciones', 'value': 'importaciones'},
                        {'label': 'Deuda Pública', 'value': 'deuda_publica'},
                        {'label': 'Reservas Internacionales', 'value': 'reservas_internacionales'},
                        {'label': 'Inversión Extranjera', 'value': 'inversion_extranjera'},
                        {'label': 'Ingresos por Turismo', 'value': 'ingresos_turismo'},
                        {'label': 'Remesas', 'value': 'remesas'}
                    ],
                    value=['pib', 'inflacion', 'desempleo'],
                    multi=True
                ),
                
                html.Div([
                    html.Div([
                        html.P("Período de análisis:"),
                        dcc.RangeSlider(
                            id='year-range-slider',
                            min=2010,
                            max=2024,
                            step=1,
                            marks={i: f'{i}' for i in range(2010, 2025, 2)},
                            value=[2018, 2024]
                        )
                    ], style={'marginTop': '20px', 'marginBottom': '20px'})
                ]),
                
                dcc.Graph(id='detailed-indicators-graph')
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Análisis de Correlación", style=subtitle_style),
                dcc.Graph(id='correlation-heatmap')
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Estadísticas Descriptivas", style=subtitle_style),
                html.Div(id='descriptive-stats')
            ])
        ])
    
    elif tab == 'tab-predictions':
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("Predicción de Indicadores Económicos", style=subtitle_style),
                html.P("Seleccione el indicador que desea predecir:"),
                dcc.Dropdown(
                    id='prediction-indicator',
                    options=[
                        {'label': 'PIB', 'value': 'pib'},
                        {'label': 'Inflación', 'value': 'inflacion'},
                        {'label': 'Desempleo', 'value': 'desempleo'},
                        {'label': 'Tipo de Cambio', 'value': 'tipo_cambio'},
                        {'label': 'Deuda Pública', 'value': 'deuda_publica'}
                    ],
                    value='pib'
                ),
                
                html.Div([
                    html.P("Seleccione el modelo de predicción:"),
                    dcc.RadioItems(
                        id='model-selection',
                        options=[
                            {'label': 'Regresión Lineal', 'value': 'linear'},
                            {'label': 'Random Forest', 'value': 'rf'},
                            {'label': 'Ridge (Regularización)', 'value': 'ridge'}
                        ],
                        value='linear',
                        labelStyle={'display': 'block', 'margin': '10px 0'}
                    )
                ], style={'marginTop': '20px'}),
                
                html.Div([
                    html.Button('Entrenar Modelo', id='train-model-button', 
                                style={'backgroundColor': colors['primary'], 'color': 'white', 
                                       'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                                       'marginTop': '20px', 'cursor': 'pointer'})
                ]),
                
                html.Div(id='training-results', style={'marginTop': '20px'})
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Visualización de Predicciones", style=subtitle_style),
                html.P("Horizonte de predicción (trimestres):"),
                dcc.Slider(
                    id='forecast-horizon',
                    min=1,
                    max=8,
                    step=1,
                    marks={i: f'{i}' for i in range(1, 9)},
                    value=4
                ),
                
                dcc.Graph(id='prediction-graph'),
                
                html.Div(id='prediction-metrics', style={'marginTop': '20px'})
            ])
        ])
    
    elif tab == 'tab-simulations':
        return html.Div([
            html.Div(style=card_style, children=[
                html.H3("Simulador de Escenarios Económicos", style=subtitle_style),
                html.P("Configure los parámetros para la simulación:"),
                
                html.Div([
                    html.Div([
                        html.P("Escenario de crecimiento del PIB:"),
                        dcc.RadioItems(
                            id='gdp-growth-scenario',
                            options=[
                                {'label': 'Optimista (+1-2% sobre tendencia)', 'value': 'optimistic'},
                                {'label': 'Base (mantiene tendencia)', 'value': 'base'},
                                {'label': 'Pesimista (-1-2% bajo tendencia)', 'value': 'pessimistic'}
                            ],
                            value='base',
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        )
                    ], style={'width': '48%'}),
                    
                    html.Div([
                        html.P("Escenario de inflación:"),
                        dcc.RadioItems(
                            id='inflation-scenario',
                            options=[
                                {'label': 'Baja (2-3%)', 'value': 'low'},
                                {'label': 'Moderada (4-5%)', 'value': 'moderate'},
                                {'label': 'Alta (6-8%)', 'value': 'high'}
                            ],
                            value='moderate',
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        )
                    ], style={'width': '48%'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.P("Escenario de tipo de cambio:"),
                        dcc.RadioItems(
                            id='exchange-rate-scenario',
                            options=[
                                {'label': 'Apreciación', 'value': 'appreciation'},
                                {'label': 'Estable', 'value': 'stable'},
                                {'label': 'Depreciación', 'value': 'depreciation'}
                            ],
                            value='stable',
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        )
                    ], style={'width': '48%'}),
                    
                    html.Div([
                        html.P("Horizonte de simulación:"),
                        dcc.RadioItems(
                            id='simulation-horizon',
                            options=[
                                {'label': '1 año', 'value': 4},
                                {'label': '2 años', 'value': 8},
                                {'label': '3 años', 'value': 12}
                            ],
                            value=4,
                            labelStyle={'display': 'block', 'margin': '5px 0'}
                        )
                    ], style={'width': '48%'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
                
                html.Button('Ejecutar Simulación', id='run-simulation-button', 
                            style={'backgroundColor': colors['primary'], 'color': 'white', 
                                   'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                                   'marginTop': '10px', 'cursor': 'pointer'})
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Resultados de la Simulación", style=subtitle_style),
                dcc.Graph(id='simulation-results-graph'),
                html.Div(id='simulation-summary', style={'marginTop': '20px'})
            ]),
            
            html.Div(style=card_style, children=[
                html.H3("Análisis de Sensibilidad", style=subtitle_style),
                html.P("Impacto de variables en el crecimiento del PIB:"),
                dcc.Graph(id='sensitivity-analysis-graph')
            ])
        ])

# Función para crear indicadores
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

# Gráfico de evolución del PIB
def create_pib_evolution_graph():
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
    
    return fig

# Gráfico de inflación y desempleo
def create_inflation_unemployment_graph():
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
    
    return fig

# Gráfico de comercio exterior y turismo
def create_trade_tourism_graph():
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Balanza Comercial', 'Ingresos por Turismo'),
                        vertical_spacing=0.12)
    
    # Exportaciones e importaciones
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['exportaciones'],
                   mode='lines', name='Exportaciones (MM USD)',
                   line=dict(color=colors['primary'], width=3)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['importaciones'],
                   mode='lines', name='Importaciones (MM USD)',
                   line=dict(color=colors['accent'], width=3)),
        row=1, col=1
    )
    
    # Calcular balance comercial
    df_economia['balance_comercial'] = df_economia['exportaciones'] - df_economia['importaciones']
    
    # Añadir área para el balance comercial
    fig.add_trace(
        go.Bar(x=df_economia['fecha'], y=df_economia['balance_comercial'],
               name='Balance Comercial (MM USD)',
               marker=dict(color=df_economia['balance_comercial'].apply(
                   lambda x: colors['primary'] if x >= 0 else colors['accent']))),
        row=1, col=1
    )
    
    # Turismo
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['ingresos_turismo'],
                  mode='lines+markers', name='Ingresos por Turismo (MM USD)',
                  line=dict(color=colors['secondary'], width=3)),
        row=2, col=1
    )
    
    # Configuración de ejes y diseño
    fig.update_layout(
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig

# Gráfico de remesas e inversión extranjera
def create_remittances_fdi_graph():
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Línea para remesas
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['remesas'],
                  mode='lines', name='Remesas (MM USD)',
                  line=dict(color=colors['primary'], width=3)),
    )
    
    # Línea para inversión extranjera
    fig.add_trace(
        go.Scatter(x=df_economia['fecha'], y=df_economia['inversion_extranjera'],
                  mode='lines', name='Inversión Extranjera Directa (MM USD)',
                  line=dict(color=colors['secondary'], width=3)),
    )
    
    # Configuración de ejes y diseño
    fig.update_layout(
        title='Remesas e Inversión Extranjera Directa',
        xaxis_title='Período',
        yaxis_title='Millones USD',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig

# Callback para el gráfico detallado de indicadores
@callback(
    Output('detailed-indicators-graph', 'figure'),
    [Input('indicator-selection', 'value'),
     Input('year-range-slider', 'value')]
)
def update_detailed_graph(selected_indicators, year_range):
    if not selected_indicators:
        # Si no hay indicadores seleccionados, devolver un gráfico vacío
        return go.Figure()
    
    # Filtrar por rango de años
    filtered_df = df_economia[(df_economia['año'] >= year_range[0]) & (df_economia['año'] <= year_range[1])]
    
    fig = go.Figure()
    
    # Nombres y unidades de los indicadores para las leyendas
    indicator_labels = {
        'pib': 'PIB (MM RD$)',
        'inflacion': 'Inflación (%)',
        'desempleo': 'Desempleo (%)',
        'tipo_cambio': 'Tipo de Cambio (DOP/USD)',
        'exportaciones': 'Exportaciones (MM USD)',
        'importaciones': 'Importaciones (MM USD)',
        'deuda_publica': 'Deuda Pública (% PIB)',
        'reservas_internacionales': 'Reservas Internacionales (MM USD)',
        'inversion_extranjera': 'Inversión Extranjera (MM USD)',
        'ingresos_turismo': 'Ingresos por Turismo (MM USD)',
        'remesas': 'Remesas (MM USD)'
    }
    
    # Colores para cada indicador
    indicator_colors = {
        'pib': colors['primary'],
        'inflacion': colors['accent'],
        'desempleo': '#8B4513',  # Marrón
        'tipo_cambio': '#4B0082',  # Índigo
        'exportaciones': '#006400',  # Verde oscuro
        'importaciones': '#8B0000',  # Rojo oscuro
        'deuda_publica': '#FF8C00',  # Naranja oscuro
        'reservas_internacionales': '#1E90FF',  # Azul dodger
        'inversion_extranjera': '#9932CC',  # Púrpura oscuro
        'ingresos_turismo': '#FF1493',  # Rosa profundo
        'remesas': '#20B2AA'   # Verde mar claro
    }
    
    # Agregar trazas para cada indicador seleccionado
    for i, indicator in enumerate(selected_indicators):
        # Crear figura con múltiples ejes Y si hay más de un indicador
        if len(selected_indicators) > 1:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['fecha'],
                    y=filtered_df[indicator],
                    mode='lines',
                    name=indicator_labels.get(indicator, indicator),
                    line=dict(color=indicator_colors.get(indicator, f'hsl({i*360/len(selected_indicators)}, 70%, 50%)'), width=3),
                    yaxis=f'y{i+1}' if i > 0 else 'y'
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['fecha'],
                    y=filtered_df[indicator],
                    mode='lines',
                    name=indicator_labels.get(indicator, indicator),
                    line=dict(color=indicator_colors.get(indicator, 'blue'), width=3)
                )
            )
    
    # Ajustar el diseño para múltiples ejes Y
    if len(selected_indicators) > 1:
        layout_dict = {
            'yaxis': {'title': indicator_labels.get(selected_indicators[0], selected_indicators[0]), 
                      'titlefont': {'color': indicator_colors.get(selected_indicators[0], 'black')},
                      'tickfont': {'color': indicator_colors.get(selected_indicators[0], 'black')}}
        }
        
        # Configurar ejes Y adicionales
        for i in range(1, len(selected_indicators)):
            layout_dict[f'yaxis{i+1}'] = {
                'title': indicator_labels.get(selected_indicators[i], selected_indicators[i]),
                'titlefont': {'color': indicator_colors.get(selected_indicators[i], f'hsl({i*360/len(selected_indicators)}, 70%, 50%)')},
                'tickfont': {'color': indicator_colors.get(selected_indicators[i], f'hsl({i*360/len(selected_indicators)}, 70%, 50%)')},
                'overlaying': 'y',
                'side': 'right',
                'position': 0.95 - (i-1)*0.08  # Desplazar los ejes
            }
        
        fig.update_layout(**layout_dict)
    else:
        fig.update_layout(
            yaxis_title=indicator_labels.get(selected_indicators[0], selected_indicators[0])
        )
    
    # Configuración general del diseño
    fig.update_layout(
        title=f'Análisis Detallado de Indicadores Económicos ({year_range[0]}-{year_range[1]})',
        xaxis_title='Período',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    return fig

# Callback para el mapa de calor de correlación
@callback(
    Output('correlation-heatmap', 'figure'),
    [Input('year-range-slider', 'value')]
)
def update_correlation_heatmap(year_range):
    # Filtrar por rango de años
    filtered_df = df_economia[(df_economia['año'] >= year_range[0]) & (df_economia['año'] <= year_range[1])]
    
    # Calcular la matriz de correlación
    numeric_cols = ['pib', 'inflacion', 'desempleo', 'tipo_cambio', 'exportaciones', 
                   'importaciones', 'deuda_publica', 'reservas_internacionales', 
                   'inversion_extranjera', 'ingresos_turismo', 'remesas']
    
    corr_matrix = filtered_df[numeric_cols].corr()
    
    # Nombres más descriptivos para las etiquetas
    labels = {
        'pib': 'PIB',
        'inflacion': 'Inflación',
        'desempleo': 'Desempleo',
        'tipo_cambio': 'Tipo de Cambio',
        'exportaciones': 'Export.',
        'importaciones': 'Import.',
        'deuda_publica': 'Deuda Púb.',
        'reservas_internacionales': 'Reservas Int.',
        'inversion_extranjera': 'IED',
        'ingresos_turismo': 'Turismo',
        'remesas': 'Remesas'
    }
    
    # Crear el mapa de calor
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[labels.get(col, col) for col in corr_matrix.columns],
        y=[labels.get(col, col) for col in corr_matrix.index],
        colorscale=[[0, colors['accent']], [0.5, '#FFFFFF'], [1, colors['primary']]],
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={'size': 10},
        hoverinfo='text',
        hovertext=[[f'{labels.get(corr_matrix.index[i], corr_matrix.index[i])} vs {labels.get(corr_matrix.columns[j], corr_matrix.columns[j])}: {corr_matrix.iloc[i, j]:.3f}' 
                    for j in range(len(corr_matrix.columns))] 
                   for i in range(len(corr_matrix.index))]
    ))
    
    # Actualizar el diseño
    fig.update_layout(
        title=f'Matriz de Correlación de Indicadores Económicos ({year_range[0]}-{year_range[1]})',
        height=550,
        plot_bgcolor='white'
    )
    
    return fig

# Callback para las estadísticas descriptivas
@callback(
    Output('descriptive-stats', 'children'),
    [Input('year-range-slider', 'value')]
)
def update_descriptive_stats(year_range):
    # Filtrar por rango de años
    filtered_df = df_economia[(df_economia['año'] >= year_range[0]) & (df_economia['año'] <= year_range[1])]
    
    # Calcular estadísticas descriptivas
    numeric_cols = ['pib', 'inflacion', 'desempleo', 'tipo_cambio', 'exportaciones', 
                   'importaciones', 'deuda_publica', 'reservas_internacionales', 
                   'inversion_extranjera', 'ingresos_turismo', 'remesas']
    
    desc_stats = filtered_df[numeric_cols].describe().T
    
    # Nombres más descriptivos para las etiquetas
    labels = {
        'pib': 'PIB (MM RD$)',
        'inflacion': 'Inflación (%)',
        'desempleo': 'Desempleo (%)',
        'tipo_cambio': 'Tipo de Cambio (DOP/USD)',
        'exportaciones': 'Exportaciones (MM USD)',
        'importaciones': 'Importaciones (MM USD)',
        'deuda_publica': 'Deuda Pública (% PIB)',
        'reservas_internacionales': 'Reservas Int. (MM USD)',
        'inversion_extranjera': 'IED (MM USD)',
        'ingresos_turismo': 'Turismo (MM USD)',
        'remesas': 'Remesas (MM USD)'
    }
    
    # Crear tabla de estadísticas
    table_header = [
        html.Thead(html.Tr([html.Th("Indicador"), html.Th("Media"), html.Th("Desv. Est."), 
                           html.Th("Mínimo"), html.Th("25%"), html.Th("Mediana"), 
                           html.Th("75%"), html.Th("Máximo")]))
    ]
    
    table_rows = []
    for index, row in desc_stats.iterrows():
        table_rows.append(html.Tr([
            html.Td(labels.get(index, index)),
            html.Td(f"{row['mean']:.2f}"),
            html.Td(f"{row['std']:.2f}"),
            html.Td(f"{row['min']:.2f}"),
            html.Td(f"{row['25%']:.2f}"),
            html.Td(f"{row['50%']:.2f}"),
            html.Td(f"{row['75%']:.2f}"),
            html.Td(f"{row['max']:.2f}")
        ]))
    
    table_body = [html.Tbody(table_rows)]
    
    table = html.Table(table_header + table_body, style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'textAlign': 'center',
        'fontSize': '12px'
    })
    
    return table

# Function to prepare data for model
def prepare_data_for_model(target_var, data=df_economia):
    # Convert dates to numerical features
    X = data.copy()
    X['year'] = X['año']
    X['quarter'] = X['trimestre']
    
    # Create lag features
    for lag in range(1, 5):
        X[f'{target_var}_lag{lag}'] = X[target_var].shift(lag)
    
    # Drop rows with NaN values (due to lags)
    X = X.dropna()
    
    # Select features and target
    feature_cols = ['year', 'quarter']
    lag_cols = [f'{target_var}_lag{lag}' for lag in range(1, 5)]
    feature_cols.extend(lag_cols)
    
    # Add some related features depending on the target variable
    if target_var == 'pib':
        related_features = ['inflacion', 'desempleo', 'inversion_extranjera']
        for feature in related_features:
            X[f'{feature}_lag1'] = X[feature].shift(1)
        feature_cols.extend([f'{feature}_lag1' for feature in related_features])
        X = X.dropna()
    elif target_var == 'inflacion':
        related_features = ['tipo_cambio', 'pib_crecimiento']
        for feature in related_features:
            X[f'{feature}_lag1'] = X[feature].shift(1)
        feature_cols.extend([f'{feature}_lag1' for feature in related_features])
        X = X.dropna()
    elif target_var == 'desempleo':
        related_features = ['pib_crecimiento', 'inflacion']
        for feature in related_features:
            X[f'{feature}_lag1'] = X[feature].shift(1)
        feature_cols.extend([f'{feature}_lag1' for feature in related_features])
        X = X.dropna()
    elif target_var == 'tipo_cambio':
        related_features = ['inflacion', 'reservas_internacionales']
        for feature in related_features:
            X[f'{feature}_lag1'] = X[feature].shift(1)
        feature_cols.extend([f'{feature}_lag1' for feature in related_features])
        X = X.dropna()
    
    # Split the data
    X_model = X[feature_cols]
    y_model = X[target_var]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X_model, feature_cols

# Callback para entrenar el modelo
@callback(
    [Output('training-results', 'children'),
     Output('prediction-graph', 'figure'),
     Output('prediction-metrics', 'children')],
    [Input('train-model-button', 'n_clicks')],
    [State('prediction-indicator', 'value'),
     State('model-selection', 'value'),
     State('forecast-horizon', 'value')]
)
def train_prediction_model(n_clicks, target_var, model_type, horizon):
    if n_clicks is None:
        raise PreventUpdate
    
    # Preparar los datos
    X_train, X_test, y_train, y_test, X_model, feature_cols = prepare_data_for_model(target_var)
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Seleccionar y entrenar el modelo
    if model_type == 'linear':
        model = LinearRegression()
        model_name = "Regresión Lineal"
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "Random Forest"
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
        model_name = "Ridge (Regularización)"
    
    # Entrenar el modelo
    model.fit(X_train_scaled, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Guardar el modelo
    model_filename = f"models/{target_var}_{model_type}_model.pkl"
    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, model_filename)
    
    # Generar predicciones futuras
    last_data = df_economia.iloc[-4:].copy()  # Últimos 4 trimestres como base
    future_predictions = []
    
    # Preparar datos para la predicción
    current_data = last_data.copy()
    
    for i in range(horizon):
        # Crear un nuevo registro para la predicción
        next_quarter = current_data.iloc[-1].copy()
        
        # Avanzar al siguiente trimestre
        if next_quarter['trimestre'] == 4:
            next_quarter['año'] = next_quarter['año'] + 1
            next_quarter['trimestre'] = 1
        else:
            next_quarter['trimestre'] = next_quarter['trimestre'] + 1
        
        next_quarter['fecha'] = pd.to_datetime(f"{int(next_quarter['año'])}-{int(next_quarter['trimestre'])*3-2}-01")
        next_quarter['fecha_str'] = f"{int(next_quarter['año'])}-Q{int(next_quarter['trimestre'])}"
        
        # Preparar características para la predicción
        pred_features = {}
        pred_features['year'] = next_quarter['año']
        pred_features['quarter'] = next_quarter['trimestre']
        
        # Agregar variables de rezago para el target
        for lag in range(1, 5):
            lag_idx = -lag
            pred_features[f'{target_var}_lag{lag}'] = current_data.iloc[lag_idx][target_var]
        
        # Agregar características relacionadas si es necesario
        if target_var == 'pib':
            related_vars = ['inflacion', 'desempleo', 'inversion_extranjera']
            for var in related_vars:
                pred_features[f'{var}_lag1'] = current_data.iloc[-1][var]
        elif target_var == 'inflacion':
            pred_features['tipo_cambio_lag1'] = current_data.iloc[-1]['tipo_cambio']
            pred_features['pib_crecimiento_lag1'] = current_data.iloc[-1]['pib_crecimiento']
        elif target_var == 'desempleo':
            pred_features['pib_crecimiento_lag1'] = current_data.iloc[-1]['pib_crecimiento']
            pred_features['inflacion_lag1'] = current_data.iloc[-1]['inflacion']
        elif target_var == 'tipo_cambio':
            pred_features['inflacion_lag1'] = current_data.iloc[-1]['inflacion']
            pred_features['reservas_internacionales_lag1'] = current_data.iloc[-1]['reservas_internacionales']
        
        # Crear DataFrame con las características
        pred_df = pd.DataFrame([pred_features])
        
        # Asegurarse de que todas las columnas necesarias estén presentes
        for col in feature_cols:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        # Mantener solo las columnas necesarias en el orden correcto
        pred_df = pred_df[feature_cols]
        
        # Escalar los datos
        pred_df_scaled = scaler.transform(pred_df)
        
        # Hacer la predicción
        prediction = model.predict(pred_df_scaled)[0]
        
        # Guardar la predicción
        next_quarter[target_var] = prediction
        future_predictions.append(next_quarter)
        
        # Actualizar los datos actuales para la siguiente iteración
        current_data = pd.concat([current_data, pd.DataFrame([next_quarter])]).reset_index(drop=True)
    
    # Crear DataFrame con las predicciones
    future_df = pd.DataFrame(future_predictions)
    
    # Crear gráfico con datos históricos y predicciones
    historical = df_economia[['fecha', 'fecha_str', target_var]]
    
    # Preparar datos para el gráfico
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(
        go.Scatter(
            x=historical['fecha'],
            y=historical[target_var],
            mode='lines',
            name='Datos Históricos',
            line=dict(color=colors['secondary'], width=3)
        )
    )
    
    # Predicciones
    fig.add_trace(
        go.Scatter(
            x=future_df['fecha'],
            y=future_df[target_var],
            mode='lines+markers',
            name='Predicciones',
            line=dict(color=colors['primary'], width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        )
    )
    
    # Intervalo de confianza (simplificado)
    if model_type in ['linear', 'ridge']:
        std_dev = np.std(y_test - y_pred)
        upper_bound = future_df[target_var] + 1.96 * std_dev
        lower_bound = future_df[target_var] - 1.96 * std_dev
        
        fig.add_trace(
            go.Scatter(
                x=future_df['fecha'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_df['fecha'],
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 166, 90, 0.2)',
                name='Intervalo de Confianza (95%)'
            )
        )
    
    # Nombres y unidades de los indicadores
    indicator_labels = {
        'pib': 'PIB (MM RD$)',
        'inflacion': 'Inflación (%)',
        'desempleo': 'Desempleo (%)',
        'tipo_cambio': 'Tipo de Cambio (DOP/USD)',
        'deuda_publica': 'Deuda Pública (% PIB)'
    }
    
    # Actualizar diseño
    fig.update_layout(
        title=f'Predicción de {indicator_labels.get(target_var, target_var)} - Modelo: {model_name}',
        xaxis_title='Período',
        yaxis_title=indicator_labels.get(target_var, target_var),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Resultados del entrenamiento
    training_results = html.Div([
        html.H4(f"Resultados del Entrenamiento - {model_name}", style={'color': colors['primary']}),
        html.P([
            html.Strong("RMSE: "), f"{rmse:.4f}", html.Br(),
            html.Strong("R²: "), f"{r2:.4f}", html.Br(),
            html.Strong("Modelo guardado: "), f"{model_filename}"
        ]),
        html.P(f"Se han generado predicciones para los próximos {horizon} trimestres.")
    ])
    
    # Métricas de predicción
    prediction_metrics = html.Div([
        html.H4("Resumen de Predicciones", style={'color': colors['primary']}),
        html.Div([
            html.Div([
                html.P([html.Strong("Valor Actual: "), f"{df_economia[target_var].iloc[-1]:.2f}"]),
                html.P([html.Strong("Valor Predicho (Último Trimestre): "), f"{future_df[target_var].iloc[-1]:.2f}"]),
                html.P([html.Strong("Cambio Esperado: "), f"{future_df[target_var].iloc[-1] - df_economia[target_var].iloc[-1]:+.2f} ({(future_df[target_var].iloc[-1] / df_economia[target_var].iloc[-1] - 1) * 100:+.2f}%)"]),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.P([html.Strong("Predicción Mínima: "), f"{future_df[target_var].min():.2f}"]),
                html.P([html.Strong("Predicción Máxima: "), f"{future_df[target_var].max():.2f}"]),
                html.P([html.Strong("Tendencia: "), get_trend_text(future_df[target_var])]),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ])
    
    return training_results, fig, prediction_metrics

# Función para determinar la tendencia
def get_trend_text(series):
    first_val = series.iloc[0]
    last_val = series.iloc[-1]
    
    if last_val > first_val:
        return html.Span("Alcista", style={'color': colors['primary'], 'fontWeight': 'bold'})
    elif last_val < first_val:
        return html.Span("Bajista", style={'color': colors['accent'], 'fontWeight': 'bold'})
    else:
        return html.Span("Estable", style={'color': colors['secondary'], 'fontWeight': 'bold'})

# Callback para la simulación
@callback(
    [Output('simulation-results-graph', 'figure'),
     Output('simulation-summary', 'children'),
     Output('sensitivity-analysis-graph', 'figure')],
    [Input('run-simulation-button', 'n_clicks')],
    [State('gdp-growth-scenario', 'value'),
     State('inflation-scenario', 'value'),
     State('exchange-rate-scenario', 'value'),
     State('simulation-horizon', 'value')]
)
def run_simulation(n_clicks, gdp_scenario, inflation_scenario, exchange_rate_scenario, horizon):
    if n_clicks is None:
        raise PreventUpdate
    
    # Definir factores de ajuste para cada escenario
    gdp_factors = {
        'optimistic': 0.02,  # +2% sobre tendencia
        'base': 0,          # mantiene tendencia
        'pessimistic': -0.02 # -2% bajo tendencia
    }
    
    inflation_targets = {
        'low': 2.5,         # inflación baja
        'moderate': 4.5,    # inflación moderada
        'high': 7.0         # inflación alta
    }
    
    exchange_rate_factors = {
        'appreciation': -0.5,  # apreciación gradual
        'stable': 0,          # estable
        'depreciation': 0.5    # depreciación gradual
    }
    
    # Obtener los últimos datos como base
    last_data = df_economia.iloc[-4:].copy()
    
    # Crear DataFrame para simulación
    simulation_base = df_economia.iloc[-1:].copy()
    future_quarters = []
    
    # Generar trimestres futuros
    for i in range(horizon):
        next_quarter = simulation_base.iloc[-1].copy()
        
        # Avanzar al siguiente trimestre
        if next_quarter['trimestre'] == 4:
            next_quarter['año'] += 1
            next_quarter['trimestre'] = 1
        else:
            next_quarter['trimestre'] += 1
        
        next_quarter['fecha'] = pd.to_datetime(f"{int(next_quarter['año'])}-{int(next_quarter['trimestre'])*3-2}-01")
        next_quarter['fecha_str'] = f"{int(next_quarter['año'])}-Q{int(next_quarter['trimestre'])}"
        
        future_quarters.append(next_quarter.to_dict())
    
    # Crear DataFrame de simulación
    simulation_df = pd.DataFrame(future_quarters)
    
    # Aplicar escenarios
    # 1. PIB
    base_pib_growth = df_economia['pib_crecimiento'].tail(8).mean()
    simulation_df['pib_crecimiento'] = base_pib_growth + gdp_factors[gdp_scenario] * 100  # en puntos porcentuales
    
    # Aplicar crecimiento al PIB base
    simulation_df['pib'] = simulation_df.index.map(lambda i: 
        df_economia['pib'].iloc[-1] * (1 + simulation_df['pib_crecimiento'].iloc[:i+1].sum() / 400)  # Efecto acumulativo
    )
    
    # 2. Inflación
    current_inflation = df_economia['inflacion'].iloc[-1]
    target_inflation = inflation_targets[inflation_scenario]
    
    # Convergencia gradual hacia el objetivo
    simulation_df['inflacion'] = simulation_df.index.map(lambda i: 
        current_inflation + (target_inflation - current_inflation) * ((i + 1) / horizon)
    )
    
    # 3. Tipo de cambio
    current_exchange = df_economia['tipo_cambio'].iloc[-1]
    
    # Aplicar evolución según escenario
    simulation_df['tipo_cambio'] = simulation_df.index.map(lambda i: 
        current_exchange * (1 + exchange_rate_factors[exchange_rate_scenario] / 100 * (i + 1))
    )
    
    # 4. Impacto en desempleo (relación inversa con crecimiento del PIB)
    current_unemployment = df_economia['desempleo'].iloc[-1]
    simulation_df['desempleo'] = simulation_df.index.map(lambda i: 
        max(4.0, current_unemployment - 0.2 * (simulation_df['pib_crecimiento'].iloc[i] - base_pib_growth))
    )
    
    # 5. Impacto en deuda pública
    # La deuda aumenta más rápido si el crecimiento es bajo
    current_debt = df_economia['deuda_publica'].iloc[-1]
    simulation_df['deuda_publica'] = simulation_df.index.map(lambda i: 
        current_debt + (0.5 - simulation_df['pib_crecimiento'].iloc[i] / 20) * ((i + 1) / 4)
    )
    
    # Crear gráficos para visualizar resultados
    # 1. Gráfico principal de resultados
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PIB y Crecimiento', 'Inflación', 'Tipo de Cambio', 'Desempleo'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Datos históricos para comparación (últimos 8 trimestres)
    historical = df_economia.iloc[-8:].copy()
    
    # PIB y Crecimiento
    fig.add_trace(
        go.Scatter(
            x=historical['fecha'],
            y=historical['pib'],
            mode='lines',
            name='PIB Histórico',
            line=dict(color=colors['secondary'])
        ),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=simulation_df['fecha'],
            y=simulation_df['pib'],
            mode='lines',
            name='PIB Proyectado',
            line=dict(color=colors['primary'], dash='dash')
        ),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=historical['fecha'],
            y=historical['pib_crecimiento'],
            name='Crecimiento Histórico',
            marker=dict(color=colors['accent'])
        ),
        row=1, col=1, secondary_y=True
    )
    
    fig.add_trace(
        go.Bar(
            x=simulation_df['fecha'],
            y=simulation_df['pib_crecimiento'],
            name='Crecimiento Proyectado',
            marker=dict(color=colors['primary'])
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Inflación
    fig.add_trace(
        go.Scatter(
            x=historical['fecha'],
            y=historical['inflacion'],
            mode='lines',
            name='Inflación Histórica',
            line=dict(color=colors['secondary'])
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=simulation_df['fecha'],
            y=simulation_df['inflacion'],
            mode='lines',
            name='Inflación Proyectada',
            line=dict(color=colors['accent'], dash='dash')
        ),
        row=1, col=2
    )
    
    # Tipo de Cambio
    fig.add_trace(
        go.Scatter(
            x=historical['fecha'],
            y=historical['tipo_cambio'],
            mode='lines',
            name='Tipo de Cambio Histórico',
            line=dict(color=colors['secondary'])
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=simulation_df['fecha'],
            y=simulation_df['tipo_cambio'],
            mode='lines',
            name='Tipo de Cambio Proyectado',
            line=dict(color='purple', dash='dash')
        ),
        row=2, col=1
    )
    
    # Desempleo
    fig.add_trace(
        go.Scatter(
            x=historical['fecha'],
            y=historical['desempleo'],
            mode='lines',
            name='Desempleo Histórico',
            line=dict(color=colors['secondary'])
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=simulation_df['fecha'],
            y=simulation_df['desempleo'],
            mode='lines',
            name='Desempleo Proyectado',
            line=dict(color='brown', dash='dash')
        ),
        row=2, col=2
    )
    
    # Actualizar ejes y diseño
    fig.update_layout(
        height=700,
        title=f'Resultados de Simulación Económica (Escenario: PIB {gdp_scenario}, Inflación {inflation_scenario}, Tipo de Cambio {exchange_rate_scenario})',
        showlegend=False,
        plot_bgcolor='white'
    )
    
    fig.update_yaxes(title_text="PIB (MM RD$)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Crecimiento (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Inflación (%)", row=1, col=2)
    fig.update_yaxes(title_text="DOP/USD", row=2, col=1)
    fig.update_yaxes(title_text="Desempleo (%)", row=2, col=2)
    
    # Resumen de la simulación
    last_simulated = simulation_df.iloc[-1]
    first_simulated = simulation_df.iloc[0]
    current_values = df_economia.iloc[-1]
    
    summary = html.Div([
        html.H4("Resumen de Resultados de Simulación", style={'color': colors['primary']}),
        html.Div([
            html.Div([
                html.H5("PIB y Crecimiento", style={'borderBottom': f'2px solid {colors["secondary"]}'}),
                html.P([
                    html.Strong("PIB Final: "), f"{last_simulated['pib']:,.2f} MM RD$", html.Br(),
                    html.Strong("Crecimiento Acumulado: "), f"{(last_simulated['pib'] / current_values['pib'] - 1) * 100:.2f}%", html.Br(),
                    html.Strong("Crecimiento Promedio: "), f"{simulation_df['pib_crecimiento'].mean():.2f}%"
                ])
            ], style={'width': '48%'}),
            
            html.Div([
                html.H5("Inflación y Precios", style={'borderBottom': f'2px solid {colors["accent"]}'}),
                html.P([
                    html.Strong("Inflación Final: "), f"{last_simulated['inflacion']:.2f}%", html.Br(),
                    html.Strong("Cambio en Inflación: "), f"{last_simulated['inflacion'] - current_values['inflacion']:+.2f} pts", html.Br(),
                    html.Strong("Inflación Promedio: "), f"{simulation_df['inflacion'].mean():.2f}%"
                ])
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.H5("Tipo de Cambio", style={'borderBottom': f'2px solid purple'}),
                html.P([
                    html.Strong("Tipo de Cambio Final: "), f"{last_simulated['tipo_cambio']:.2f} DOP/USD", html.Br(),
                    html.Strong("Variación: "), f"{(last_simulated['tipo_cambio'] / current_values['tipo_cambio'] - 1) * 100:+.2f}%", html.Br(),
                    html.Strong("Depreciación Anualizada: "), f"{((last_simulated['tipo_cambio'] / current_values['tipo_cambio']) ** (4/horizon) - 1) * 100:.2f}%"
                ])
            ], style={'width': '48%'}),
            
            html.Div([
                html.H5("Desempleo", style={'borderBottom': f'2px solid brown'}),
                html.P([
                    html.Strong("Desempleo Final: "), f"{last_simulated['desempleo']:.2f}%", html.Br(),
                    html.Strong("Cambio en Desempleo: "), f"{last_simulated['desempleo'] - current_values['desempleo']:+.2f} pts", html.Br(),
                    html.Strong("Desempleo Promedio: "), f"{simulation_df['desempleo'].mean():.2f}%"
                ])
            ], style={'width': '48%'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'})
    ])
    
    # Gráfico de análisis de sensibilidad
    # Simular diferentes valores de parámetros
    gdp_growth_range = np.linspace(-2, 6, 9)  # Desde -2% hasta 6% en saltos de 1%
    inflation_range = np.linspace(2, 10, 9)   # Desde 2% hasta 10% en saltos de 1%
    
    sensitivity_results = []
    
    # Simular diferentes combinaciones
    for growth in gdp_growth_range:
        for infl in inflation_range:
            # Estimar impacto en el PIB
            pib_impact = 100 * (1 + growth/100) ** 2 * (1 - (infl-4)/20)
            
            sensitivity_results.append({
                'Crecimiento': growth,
                'Inflación': infl,
                'Impacto PIB': pib_impact
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Crear mapa de calor
    sensitivity_fig = px.density_heatmap(
        sensitivity_df, 
        x='Crecimiento', 
        y='Inflación', 
        z='Impacto PIB',
        title='Análisis de Sensibilidad: Impacto de Crecimiento e Inflación',
        labels={'Impacto PIB': 'Índice de Impacto (base 100)'},
        color_continuous_scale=[[0, colors['accent']], [0.5, 'white'], [1, colors['primary']]],
    )
    
    sensitivity_fig.update_layout(
        plot_bgcolor='white',
        height=450
    )
    
    return fig, summary, sensitivity_fig

# Iniciar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)