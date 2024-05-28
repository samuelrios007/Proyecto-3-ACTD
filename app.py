# -*- coding: utf-8 -*-

# Ejecute esta aplicación 
# y luego visite el sitio
# http://127.0.0.1:8050/ 
# en su navegador.

import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
from dotenv import load_dotenv # pip install python-dotenv
import plotly.express as px
import os

def anti_rural(valor):
    if valor=='Rural':
        return(1)
    elif valor=='Urbano':
        return(0)

def anti_bilingue(valor):
    if valor=='Sí':
        return(0)
    elif valor=='No':
        return(1)
    

df  = pd.read_csv('datos_limpios.csv')

fig_default = go.Figure()

fig_default.add_trace(go.Box(
    y=df['PUNT_GLOBAL'],
    name='Datos',
    marker_color='lightseagreen'
))
fig_default.update_layout(
    title='Distribución del Puntaje Global',
    yaxis_title='Puntaje Global',
    height=600,
    width=550
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# path to model file
ruta_actual=os.path.dirname(os.path.abspath(__file__))

# This line does the deserialization
model=keras.models.load_model(os.path.join(ruta_actual, 'default_pred.keras'))

# en este primer ejemplo usamos unos datos de prueba que creamos directamente
# en un dataframe de pandas 
df  = pd.read_csv('Datos_Santander.csv')

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                # Primera imagen
                html.Img(
                    src='https://preparacionsaber11.com/pluginfile.php/1/theme_moove/logo/1624294499/logo-saber-11_logo.png',
                    style={'height': '80px', 'marginRight': '20px'}
                ),
                # Segunda imagen
                html.Img(
                    src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Flag_of_Santander_Department.svg/200px-Flag_of_Santander_Department.svg.png',
                    style={'height': '80px', 'marginRight': '20px'}
                ),
                # Título
                html.H1(
                    children='Diagnóstico Académico de las Saber 11 en Santander',
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial Black'}
                ),
            ],
            style={'display': 'flex', 'alignItems': 'center', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '2px 2px 2px lightgrey'}
        ),
        html.Div(
            children=[
                html.H3(children='Esta aplicación predice el desempeño de un estudiante en las pruebas Saber 11 y permite comparar su información con la de los demás estudiantes de Santander.',
                        style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                html.H3(children='Ingrese la información del estudiante:',
                        style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial Black'})
            ]
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H4('Estrato',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='estrato',
                            options=['Estrato 1','Estrato 2','Estrato 3','Estrato 4','Estrato 5','Estrato 6'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Nivel de Inglés',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='ingles',
                            options=['A-', 'A1', 'A2', 'B1', 'B+'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Año de nacimiento',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        html.Br(),
                        dcc.Input(
                            id='anio',
                            type='number',
                            placeholder="Ingrese un año",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                )
            ],
            style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'marginTop': '20px'
            }
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H4('Colegio bilingüe',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='bilingue',
                            options=['Sí', 'No'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Tipo de colegio',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='mixto',
                            options=['Mixto', 'Femenino', 'Masculino'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Calendario',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        html.Br(),
                        dcc.Dropdown(
                            id='calendario',
                            options=['Calendario A', 'Calendario B', 'Otro'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                )
            ],
            style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'marginTop': '20px'
            }
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H4('Jornada del colegio',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='jornada',
                            options=['Completa', 'Única', 'Mañana', 'Tarde', 'Noche', 'Sabatina'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Sexo del estudiante',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        dcc.Dropdown(
                            id='sexo',
                            options=['Femenino', 'Masculino'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                ),
                html.Div(
                    children=[
                        html.H4('Ubicación del colegio',
                                style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
                        html.Br(),
                        dcc.Dropdown(
                            id='ubicacion',
                            options=['Rural', 'Urbano'],
                            placeholder="Seleccione una opción",
                            style={'width':'400px'}
                        )
                    ],
                    style={'marginRight': '100px'}
                )
            ],
            style={
                'display': 'flex',
                'alignItems': 'flex-start',
                'marginTop': '20px'
            }
        ),
        html.Div(children=[
            html.H1('Puntaje esperado del estudiante en el Saber 11:',
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial Black', 'marginRight': '25px'}),
            html.H1(id='prediccion', 
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#FF0000', 'fontFamily': 'Arial Black'})
        ]
        ),
        html.Div([
            dcc.Graph(id='graph-comp'),
            ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
        dcc.Graph(id='graph-rural'),
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
        dcc.Graph(id='graph-biling'),
        ], style={'width': '33%', 'display': 'inline-block'})
    ]
)
'''
html.Div(children=[
            html.H3("Desempeño del estudiante vs. los demás estudiantes",
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
            dcc.Graph(id='graph-comp'),
        ]),
        html.Div(children=[
            html.H3("Comparación de desempeño frente a otros escenarios sociodemográficos",
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
            dcc.Graph(id='graph-rural'),
        ]),
        html.Div(children=[
            html.H3("Comparación de desempeño frente a otros escenarios académicos",
                    style={'display': 'inline-block', 'verticalAlign': 'middle', 'color': '#2C3E50', 'fontFamily': 'Arial'}),
            dcc.Graph(id='graph-biling'),
        ])'''

@app.callback(
    [Output('prediccion','children'),
     Output('graph-comp','figure'),
     Output('graph-rural','figure'),
     Output('graph-biling','figure')],
    [Input('estrato','value'),
     Input('ingles','value'),
     Input('anio','value'),
     Input('bilingue','value'),
     Input('mixto','value'),
     Input('calendario','value'),
     Input('jornada','value'),
     Input('sexo','value'),
     Input('ubicacion','value')]
)
def update_todo(estrato, ingles, anio, bilingue, mixto, calendario, jornada, sexo, ubicacion):
    
    if None in [estrato, ingles, anio, bilingue, mixto, calendario, jornada, sexo, ubicacion]:
        return "", fig_default, fig_default, fig_default
    
    estrato_mapping = {'Estrato 1':1,'Estrato 2':2,'Estrato 3':3,'Estrato 4':4,'Estrato 5':5,'Estrato 6':6}
    ingles_mapping = {'A-': 1, 'A1': 2, 'A2': 3, 'B1': 4, 'B+': 5}
    bilingue_mapping = {'Sí':1, 'No':0}
    mixto_mapping_masculino = {'Mixto':0, 'Femenino':0, 'Masculino':1}
    mixto_mapping_mixto = {'Mixto':1, 'Femenino':0, 'Masculino':0}
    calendario_mapping_B = {'Calendario A':0, 'Calendario B':1, 'Otro':0}
    calendario_mapping_otro = {'Calendario A':0, 'Calendario B':0, 'Otro':1}
    jornada_mapping_manana = {'Completa':0, 'Única':0,'Mañana':1, 'Tarde':0, 'Noche':0, 'Sabatina':0}
    jornada_mapping_noche = {'Completa':0, 'Única':0,'Mañana':0, 'Tarde':0, 'Noche':1, 'Sabatina':0}
    jornada_mapping_sabatina = {'Completa':0, 'Única':0,'Mañana':0, 'Tarde':0, 'Noche':0, 'Sabatina':1}
    jornada_mapping_tarde = {'Completa':0, 'Única':0,'Mañana':0, 'Tarde':1, 'Noche':0, 'Sabatina':0}
    jornada_mapping_unica = {'Completa':0, 'Única':1,'Mañana':0, 'Tarde':0, 'Noche':0, 'Sabatina':0}
    sexo_mapping = {'Masculino':1, 'Femenino':0}
    ubicacion_mapping = {'Urbano':1, 'Rural':0}
    
    estrato_value = estrato_mapping.get(estrato)
    ingles_value = ingles_mapping.get(ingles)
    bilingue_value = bilingue_mapping.get(bilingue)
    mixto_value_masculino = mixto_mapping_masculino.get(mixto)
    mixto_value_mixto = mixto_mapping_mixto.get(mixto)
    calendario_value_B = calendario_mapping_B.get(calendario)
    calendario_value_otro = calendario_mapping_otro.get(calendario)
    jornada_value_manana = jornada_mapping_manana.get(jornada)
    jornada_value_noche = jornada_mapping_noche.get(jornada)
    jornada_value_sabatina = jornada_mapping_sabatina.get(jornada)
    jornada_value_tarde = jornada_mapping_tarde.get(jornada)
    jornada_value_unica = jornada_mapping_unica.get(jornada)
    sexo_value = sexo_mapping.get(sexo)
    ubicacion_value = ubicacion_mapping.get(ubicacion)
    
    #Escalar los datos necesarios
    scaled_estrato = (estrato_value-2.0699163786901296)/1.0750916877147871
    scaled_ingles = (ingles_value-2.058369797256655)/1.0581880012461748
    scaled_anio = (anio-1999.9994810525197)/7.801538969874645
    
    #Construct the input X
    x = np.array([[scaled_estrato, scaled_ingles, scaled_anio, bilingue_value, calendario_value_B, calendario_value_otro, 
                mixto_value_masculino, mixto_value_mixto, jornada_value_manana, jornada_value_noche,
                jornada_value_sabatina, jornada_value_tarde, jornada_value_unica, sexo_value, ubicacion_value]])
    
    print(scaled_estrato, scaled_ingles, scaled_anio, bilingue_value, calendario_value_B, calendario_value_otro, 
                mixto_value_masculino, mixto_value_mixto, jornada_value_manana, jornada_value_noche,
                jornada_value_sabatina, jornada_value_tarde, jornada_value_unica, sexo_value, ubicacion_value)
    
    # Check inputs are correct 
    ypred = model.predict(x)
    
    # Convert ypred to float and format the probability as a percentage
    valor = round(float((ypred[[0]]*52.131977555078095)+266.7053679702895),2)
    
    # Crear el boxplot
    fig1 = go.Figure()

    fig1.add_trace(go.Box(
        y=df['PUNT_GLOBAL'],
        name='Datos',
        marker_color='salmon'
    ))

    # Agregar el punto específico
    fig1.add_trace(go.Scatter(
        y=[valor],
        x=['Datos'],
        mode='markers',
        marker=dict(color='green', size=12, symbol='circle'),
        name=f'Puntaje del estudiante'
    ))
    fig1.update_layout(
    title='Desempeño individual del estudiante vs. los demás estudiantes',
    yaxis_title='Puntaje Global',
    height=600,
    width=550
)
    
    #Crear el anti rural/urbano
    anti_ubi_value = anti_rural(ubicacion)
    #Construct the input X
    x_anti_ubi = np.array([[scaled_estrato, scaled_ingles, scaled_anio, bilingue_value, calendario_value_B, calendario_value_otro, 
                mixto_value_masculino, mixto_value_mixto, jornada_value_manana, jornada_value_noche,
                jornada_value_sabatina, jornada_value_tarde, jornada_value_unica, sexo_value, anti_ubi_value]])
    
    # Check inputs are correct 
    ypred_anti_ubi = model.predict(x_anti_ubi)
    
    # Convert ypred to float and format the probability as a percentage
    valor_anti_ubi = round(float((ypred_anti_ubi[[0]]*52.131977555078095)+266.7053679702895),2)
    
    fig2 = go.Figure()

    fig2.add_trace(go.Box(
        y=df['PUNT_GLOBAL'],
        name='Datos',
        marker_color='lightseagreen'))
    
    fig2.add_trace(go.Scatter(
        y=[valor],
        x=['Datos'],
        mode='markers',
        marker=dict(color='green', size=12, symbol='circle'),
        name=f'Escenario actual'))
    
    fig2.add_trace(go.Scatter(
        y=[valor_anti_ubi],
        x=['Datos'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle'),
        name=f'Escenario opuesto'))
    fig2.update_layout(
    title='Comparación de escenarios sociodemográficos (Rural vs. Urbano)',
    yaxis_title='Puntaje Global',
    height=600,
    width=550)
    
    #Crear el anti rural/urbano
    anti_bili_value = anti_bilingue(bilingue)
    #Construct the input X
    x_anti_bili = np.array([[scaled_estrato, scaled_ingles, scaled_anio, anti_bili_value, calendario_value_B, calendario_value_otro, 
                mixto_value_masculino, mixto_value_mixto, jornada_value_manana, jornada_value_noche,
                jornada_value_sabatina, jornada_value_tarde, jornada_value_unica, sexo_value, ubicacion_value]])
    
    # Check inputs are correct 
    ypred_anti_bili = model.predict(x_anti_bili)
    
    # Convert ypred to float and format the probability as a percentage
    valor_anti_bili = round(float((ypred_anti_bili[[0]]*52.131977555078095)+266.7053679702895),2)
    
    fig3 = go.Figure()

    fig3.add_trace(go.Box(
        y=df['PUNT_GLOBAL'],
        name='Datos',
        marker_color='purple'))
    
    fig3.add_trace(go.Scatter(
        y=[valor],
        x=['Datos'],
        mode='markers',
        marker=dict(color='green', size=12, symbol='circle'),
        name=f'Escenario actual'))
    
    fig3.add_trace(go.Scatter(
        y=[valor_anti_bili],
        x=['Datos'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='circle'),
        name=f'Escenario opuesto'))
    fig3.update_layout(
    title='Comparación de escenarios académicos (Bilingüe vs. No Bilingüe)',
    yaxis_title='Puntaje Global',
    height=600,
    width=550)
    
    return valor, fig1, fig2, fig3
    

if __name__ == '__main__':
    app.run_server(debug=True)
