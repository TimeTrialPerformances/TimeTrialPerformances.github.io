from dash import Dash, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import sqlite3
import plotly.graph_objects as go
from datetime import date
import numpy as np
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import statistics


def calcula_edad(nacimiento):
    cachos = nacimiento.split('-')
    hoy = date.today()
    edad = hoy.year - int(cachos[0]) - ((hoy.month, hoy.day) < (int(cachos[1]), int(cachos[2])))
    return edad

def destaca_color(ranking,num,color):
    if ranking == num:
        return color
    else:
        return 'lavender'

# conexion = sqlite3.connect('Datos.db')
# df_plantillas = pd.read_sql('SELECT * FROM Plantillas',conexion)
# df_historicos = pd.read_sql('SELECT * FROM Datos_dash',conexion)

url='https://drive.google.com/file/d/1PjD1UZ9uxqqxh_yecrw4jmOg4CiW6axx/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df_plantillas = pd.read_csv(dwn_url,sep=';')

url='https://drive.google.com/file/d/1GmOYm7uobiCecpMH7efNFaISz0bGsN7P/view?usp=sharing'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df_historicos = pd.read_csv(dwn_url,sep=';')


df_plantillas['edad'] = df_plantillas['nacimiento'].apply(calcula_edad)

header_comparacion = ['Position','Rider','Age','Rating','# ITT','Wins','Podiums','Top 10s','UCI pts']

app = Dash()
server = app.server

app.layout = dmc.Container([
    dmc.Title('Individual Time Trial Performance', color="blue", size="h3",align='center'),
    dmc.RadioGroup([
        dmc.Select(
            label='Select a WT team',
            id='Seleccion_equipo',
            value=sorted(df_plantillas['Equipo'].to_list())[0],
            data=[{'value': equipo, 'label': equipo} for equipo in sorted(df_plantillas['Equipo'].unique())]
        ),
        dmc.Select(
            label='Select a rider',
            id='Seleccion_corredor',
            data=[],
            value=None  # Añadimos esta línea para inicializar el dropdown en blanco
        ),
    ]),
    dmc.Grid(children=[
        dmc.Col([dcc.Graph(id='numero-de-cronos', style={'width': '150px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='rating-global', style={'width': '150px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='pct_victorias', style={'width': '150px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='pct_podiums', style={'width': '150px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='pct_top10', style={'width': '150px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='tabla_comparacion', style={'width': '850px', 'height': '200px'})], span='content'),
        dmc.Col([dcc.Graph(id='scatters', style={'width': '850px', 'height': '350px'})], span='content'),
        dmc.Col([dcc.Graph(id='boxplot', style={'width': '850px', 'height': '350px'})], span='content'),
        dmc.Col([dcc.Graph(id='evolucion_temp', style={'width': '850px', 'height': '650px'})], span='content'),
        dmc.Col([dcc.Graph(id='polar', style={'width': '850px', 'height': '650px'})], span='content'),
    ],gutter="xl",),
], fluid=True)

@callback( #SELECTOR DE CORREDOR ---------------------------------------------------------------------------------------
    [Output('Seleccion_corredor', 'data'), Output('Seleccion_corredor', 'value')],
    Input('Seleccion_equipo', 'value')
)

def update_corredores(equipo_seleccionado):
    corredores_0 = sorted(df_plantillas[df_plantillas['Equipo'] == equipo_seleccionado]['Corredor'].to_list())
    corredores_0.sort()
    corredores = [x.title() for x in corredores_0]
    return [{'value': corredor, 'label': corredor} for corredor in corredores], None

@callback( #INDICADOR DE NUMERO DE CRONOS ---------------------------------------------------------------------------------------
    Output('numero-de-cronos', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = go.Figure(go.Indicator(
        mode = "number+delta",
        value = None,
        delta = None,
        title = {'text': "# of analyzed ITT", 'font': {'size':12}},
        ))

        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig
        
    n_cronos = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['numero_cri'].mean()
    n_cronos_old = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['numero_cri_old'].mean()
    
    fig = go.Figure(go.Indicator(
        mode = "number+delta",
        value = n_cronos,
        delta = {'reference': n_cronos_old, 'font': {'size':16}},
        title = {'text': "# of analyzed ITT", 'font': {'size':12}},
    ))

    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig

@callback( #RATING GLOBAL ---------------------------------------------------------------------------------------
    Output('rating-global', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator1(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig1 =  go.Figure(go.Indicator(
        mode = "number+delta+gauge",
        delta = None,
        value = None,
        title = {'text': "Overall<br />rating", 'font': {'size':14}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}}
        ))

        fig1.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig1
        
    rating = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Rating_historico'].mean()
    rating_old = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Rating_historico_old'].mean()
    color = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Color_gauge'].values[0]
    
    fig1 =  go.Figure(go.Indicator(
    mode = "number+delta+gauge",
    delta = {'reference': rating_old},
    value = rating,
    title = {'text': "Overall<br />rating", 'font': {'size':14}},
    gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},'bar': {'color': color}}
    ))

    fig1.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig1

@callback( #PCT VICTORIAS ---------------------------------------------------------------------------------------
    Output('pct_victorias', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator1(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig1 =  go.Figure(go.Indicator(
        mode = "number+delta+gauge",
        delta = None,
        value = None,
        title = {'text': "Win percentage", 'font': {'size':14}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}}
        ))

        fig1.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig1
        
    rating = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['victorias_pct'].mean()
    rating_old = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['victorias_pct_old'].mean()
    
    fig1 =  go.Figure(go.Indicator(
    mode = "number+delta+gauge",
    delta = {'reference': rating_old},
    value = rating,
    title = {'text': "Win percentage", 'font': {'size':14}},
    gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},'bar': {'color': '#DAA520'}}
    ))

    fig1.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig1

@callback( #PCT PODIUM ---------------------------------------------------------------------------------------
    Output('pct_podiums', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator1(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig1 =  go.Figure(go.Indicator(
        mode = "number+delta+gauge",
        delta = None,
        value = None,
        title = {'text': "Podium percentage", 'font': {'size':14}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}}
        ))

        fig1.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )
        
        return fig1
        
    rating = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['podiums_pct'].mean()
    rating_old = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['podiums_pct_old'].mean()
    
    fig1 =  go.Figure(go.Indicator(
    mode = "number+delta+gauge",
    delta = {'reference': rating_old},
    value = rating,
    title = {'text': "Podium percentage", 'font': {'size':14}},
    gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},'bar': {'color': '#A9A9A9'}}
    ))

    fig1.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig1

@callback( #PCT TOP10 ---------------------------------------------------------------------------------------
    Output('pct_top10', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator1(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig1 =  go.Figure(go.Indicator(
        mode = "number+delta+gauge",
        delta = None,
        value = None,
        title = {'text': "Top 10 percentage", 'font': {'size':14}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}}
        ))

        fig1.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig1
        
    rating = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['top10_pct'].mean()
    rating_old = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['top10_pct_old'].mean()
    
    fig1 =  go.Figure(go.Indicator(
    mode = "number+delta+gauge",
    delta = {'reference': rating_old},
    value = rating,
    title = {'text': "Top 10 percentage", 'font': {'size':14}},
    gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},'bar': {'color': '#8b4513'}}
    ))

    fig1.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig1

@callback( #TABLA COMPARACION ---------------------------------------------------------------------------------------
    Output('tabla_comparacion', 'figure'),
    [Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator1(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = go.Figure(data=[go.Table(
        columnwidth = [80,300,80,100,80,100,100,100,100],
        header=dict(values=list(header_comparacion),
                    fill_color='paleturquoise',
                    align='left',
                    height=23),
        cells=dict(values=[['-' for _ in range(5)] for _ in range(9)],
                fill_color='lavender',
                align='left',
                height=23))
    ])
        
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        height=200
        )

        return fig
        
    rnkg_pos = df_plantillas.loc[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Ranking'].values[0]
    if rnkg_pos == 1:
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] <= rnkg_pos + 4]
    elif rnkg_pos == 2:
        df_tabla = df_plantillas.loc[(df_plantillas['Ranking'] <= rnkg_pos + 3) & (df_plantillas['Ranking'] >= rnkg_pos - 1)]
    elif rnkg_pos == len(df_plantillas.index) -1:
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] >= rnkg_pos - 3]
    elif rnkg_pos == len(df_plantillas.index):
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] >= rnkg_pos - 4]
    else:
        df_tabla = df_plantillas.loc[(df_plantillas['Ranking'] <= rnkg_pos + 2) & (df_plantillas['Ranking'] >= rnkg_pos - 2)]

    df_tabla['Corredor'] = df_tabla['Corredor'].apply(lambda x: x.title())
    df_tabla['color'] = df_tabla.apply(lambda x: destaca_color(x.Ranking, rnkg_pos, x.Color_gauge), axis=1)
    
    fig = go.Figure(data=[go.Table(
        columnwidth = [80,300,80,100,80,100,100,100,100],
        header=dict(values=list(header_comparacion),
                    fill_color='paleturquoise',
                    align='left',
                    height=23),
        cells=dict(values=[df_tabla.Ranking, df_tabla.Corredor, df_tabla.edad,df_tabla.Rating_historico,
                            df_tabla.numero_cri, df_tabla.victorias, df_tabla.podiums,df_tabla.top10, df_tabla.uci],
                fill_color = [df_tabla.color.tolist()*5] ,
                align='center',
                height=23))
    ])

    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    height=200
    )
    
    return fig


@callback( #SCATTERS ---------------------------------------------------------------------------------------
Output('scatters', 'figure'),
[Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
        go.Scatter(x=[], y=[]),
        row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=[], y=[]),
            row=1, col=2
        )

        fig.update_layout(
            margin=dict(l=0,r=0,b=0,t=0),
            )
        
        fig.update_xaxes(title_text="Time (min)", row=1, col=1)
        fig.update_xaxes(title_text="Distance (km)", row=1, col=2)

        fig.update_yaxes(title_text="Performance index", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Vertical ascent (m)", row=1, col=2)

        return fig
        
    lista_coeficientes = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['rating_carrera'].to_list()
    lista_tiempos = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['minutos_decimales'].to_list()
    lista_distancias = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['distancia'].to_list()
    lista_desniveles = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['desnivel'].to_list()
    lista_colores = df_historicos[(df_historicos['corredor'] == corredor_seleccionado.lower())]['color_carrera'].to_list()
    lista_anios = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['anio'].to_list()
    lista_etapas = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['etapa'].to_list()
    lista_carreras = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['nombre_carr'].to_list()

    reg = LinearRegression().fit(np.vstack(lista_tiempos), lista_coeficientes)
    lista_predicciones = reg.predict(np.vstack(lista_tiempos))
    

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
    go.Scatter(x=lista_tiempos, y=lista_coeficientes, mode='markers',customdata = np.stack((lista_anios, lista_carreras, lista_etapas),axis=-1),
                    hovertemplate= "Año: %{customdata[0]}<br>" + "Carrera: %{customdata[1]}<br>" + "Etapa: %{customdata[2]}<br>",name=""), 
    row=1, col=1
    )

    fig.add_trace(
        go.Scatter(name='line of best fit', x=lista_tiempos, y=lista_predicciones, mode='lines',hoverinfo='skip')
    )

    fig.add_trace(
        go.Scatter(x=lista_distancias, y=lista_desniveles, mode='markers', marker=dict(color=lista_colores),customdata = np.stack((lista_anios, lista_carreras, lista_etapas,lista_coeficientes),axis=-1),
                    hovertemplate= "Año: %{customdata[0]}<br>" + "Carrera: %{customdata[1]}<br>" + "Etapa: %{customdata[2]}<br>" + "Valoración: %{customdata[3]}<br>",name=""),
        row=1, col=2
    )

    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    showlegend=False
    )

    fig.update_xaxes(title_text="Time (min)", row=1, col=1)
    fig.update_xaxes(title_text="Distance (km)", row=1, col=2)

    fig.update_yaxes(title_text="Performance index", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Vertical ascent (m)", row=1, col=2)

    
    return fig


@callback( #BOXPLOT ---------------------------------------------------------------------------------------
Output('boxplot', 'figure'),
[Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = go.Figure(go.Box(x = [0,1,2,3,4,5], marker_color = 'lightseagreen', name= '',boxpoints='all'))

        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        xaxis=dict(title='Ranking distribution', zeroline=False),
        )

        return fig
        
    lista_coeficientes = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['puesto'].to_list()
    lista_anios = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['anio'].to_list()
    lista_etapas = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['etapa'].to_list()
    lista_carreras = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]['nombre_carr'].to_list()
    color = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Color_gauge'].values[0]

    

    fig = go.Figure(go.Box(x = lista_coeficientes, boxmean=True,marker_color = color, name= '', boxpoints='all',customdata = np.stack((lista_coeficientes, lista_anios, lista_carreras, lista_etapas),axis=-1),
                    hovertemplate= "Puesto: %{customdata[0]}<br>" + "Año: %{customdata[1]}<br>" + "Carrera: %{customdata[2]}<br>"+ "Etapa: %{customdata[3]}<br>"))

    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    xaxis=dict(title='Ranking distribution', zeroline=False),
    )
    
    return fig

@callback( #EVOLUCION TEMPORAL ---------------------------------------------------------------------------------------
Output('evolucion_temp', 'figure'),
[Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = go.Figure(data=go.Scatter(x=[]))

        fig.update_yaxes(title_text="Overall rating", range=[0, 100])

        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig
        
    rnkg_pos = df_plantillas.loc[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Ranking'].values[0]
    if rnkg_pos == 1:
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] <= rnkg_pos + 4]
    elif rnkg_pos == 2:
        df_tabla = df_plantillas.loc[(df_plantillas['Ranking'] <= rnkg_pos + 3) & (df_plantillas['Ranking'] >= rnkg_pos - 1)]
    elif rnkg_pos == len(df_plantillas.index) -1:
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] >= rnkg_pos - 3]
    elif rnkg_pos == len(df_plantillas.index):
        df_tabla = df_plantillas.loc[df_plantillas['Ranking'] >= rnkg_pos - 4]
    else:
        df_tabla = df_plantillas.loc[(df_plantillas['Ranking'] <= rnkg_pos + 2) & (df_plantillas['Ranking'] >= rnkg_pos - 2)]

    corredores = df_tabla['Corredor'].unique()

    df_tabla = df_historicos.loc[df_historicos['corredor'].isin(corredores)]

    primer_anio = df_tabla['anio'].min()
    anios = list(range(primer_anio,2025,1))
    valoraciones = []
    for corredor in corredores:
        V_corredor = []
        for anio in anios:
            coeficientes = df_tabla.loc[(df_tabla['corredor'] == corredor) & (df_tabla['anio'] <= anio)]['Coef'].tolist()
            if coeficientes == []:
                V_corredor.append(None)
            else:
                coef_medio = round((statistics.mean(coeficientes)+1) * ((99-40)/(2.1+1))+40,0)
                if coef_medio > 99:
                    V_corredor.append(99)
                else:
                    V_corredor.append(coef_medio)
        valoraciones.append(V_corredor)


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=anios, y=valoraciones[0],
                        mode='lines+markers',
                        name= corredores[0].title(),
                        line=dict(color='Crimson')))
    fig.add_trace(go.Scatter(x=anios, y=valoraciones[1],
                        mode='lines+markers',
                        name= corredores[1].title(),
                        line=dict(color='MediumSeaGreen')))
    fig.add_trace(go.Scatter(x=anios, y=valoraciones[2],
                        mode='lines+markers', 
                        name=corredores[2].title(),
                        line=dict(color='DodgerBlue')))
    fig.add_trace(go.Scatter(x=anios, y=valoraciones[3],
                        mode='lines+markers',
                        name=corredores[3].title(),
                        line=dict(color='DarkOrange')))
    fig.add_trace(go.Scatter(x=anios, y=valoraciones[4],
                        mode='lines+markers',
                        name=corredores[4].title(),
                        line=dict(color='Violet')))
    
    fig.update_yaxes(title_text="Overall rating")

    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )
    
    return fig

@callback( #POLAR ---------------------------------------------------------------------------------------
Output('polar', 'figure'),
[Input('Seleccion_equipo', 'value'), Input('Seleccion_corredor', 'value')]
)
def update_indicator(equipo_seleccionado, corredor_seleccionado):
    if corredor_seleccionado is None:
        fig = go.Figure(data=go.Scatterpolar(r=[0,0,0,0,0], theta=['Prologues','Mountain TT','One day TT','TT that stage > 9',"TT longer than 40'"], fill='toself'))
        fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        )

        return fig
        
    df_polar = df_historicos.loc[df_historicos['corredor'] == corredor_seleccionado.lower()]
    prologos = df_polar.loc[df_polar['prologo'] == 'Si']['Coef'].tolist()
    if prologos == []:
        prologos = 0
    else:
        prologos = round((statistics.mean(prologos)+1) * ((99-40)/(2.1+1))+40,0)
        if prologos > 99:
            prologos = 99

    cronoescaladas = df_polar.loc[df_polar['cronoescalada'] == 'Si']['Coef'].tolist()
    if cronoescaladas == []:
        cronoescaladas = 0
    else:
        cronoescaladas = round((statistics.mean(cronoescaladas)+1) * ((99-40)/(2.1+1))+40,0)
        if cronoescaladas > 99:
            cronoescaladas = 99

    etapa_1 = df_polar.loc[df_polar['etapa'] == -1]['Coef'].tolist()
    if etapa_1 == []:
        etapa_1 = 0
    else:
        etapa_1 = round((statistics.mean(etapa_1)+1) * ((99-40)/(2.1+1))+40,0)
        if etapa_1 > 99:
            etapa_1 = 99
    
    etapa_9 = df_polar.loc[df_polar['etapa'] >= 10]['Coef'].tolist()
    if etapa_9 == []:
        etapa_9 = 0
    else:
        etapa_9 = round((statistics.mean(etapa_9)+1) * ((99-40)/(2.1+1))+40,0)
        if etapa_9 > 99:
            etapa_9 = 99

    crono_larga = df_polar.loc[df_polar['minutos'] >= 40]['Coef'].tolist()
    if crono_larga == []:
        crono_larga = 0
    else:
        crono_larga = round((statistics.mean(crono_larga)+1) * ((99-40)/(2.1+1))+40,0)
        if crono_larga > 99:
            crono_larga = 99
    
    color = df_plantillas[(df_plantillas['Equipo'] == equipo_seleccionado) & (df_plantillas['Corredor'] == corredor_seleccionado.lower())]['Color_gauge'].values[0]

    fig = go.Figure(data=go.Scatterpolar(r=[prologos,cronoescaladas,etapa_1,etapa_9,crono_larga], 
                    theta=['Prologues','Mountain TT','One day TT','TT that stage > 9',"TT longer than 40'"], 
                    fill='toself', fillcolor=color,line_color=color,name=''))

    fig.update_polars(radialaxis=dict(range=[0, 100]))
    
    fig.update_layout(
    margin=dict(l=0,r=0,b=0,t=0),
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
