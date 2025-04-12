import math

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Static movie data for placeholders
movies_data = [
    {"title": "Movie01", "director": "Director01", "actors": ["Actor01", "Actor02", "Actor03", "Actor04", "Actor05"], "awards": [], "grossing": 25.5, "budget": 10.8},
    {"title": "Movie02", "director": "Director02", "actors": ["Actor06", "Actor07", "Actor08", "Actor09", "Actor10"], "awards": [], "grossing": 6.0, "budget": 1.0},
    {"title": "Movie03", "director": "Director03", "actors": ["Actor11", "Actor12", "Actor13", "Actor14", "Actor15"], "awards": [{"country": "UK", "category": "Best Soundtrack"}, {"country": "USA", "category": "Best Soundtrack"}, {"country": "UK", "category": "Best Sound"}, {"country": "USA", "category": "Best Special Effects"}, {"country": "USA", "category": "Best Sound"}, {"country": "USA", "category": "Best Song"}], "grossing": 60.2, "budget": 40.9},
    {"title": "Movie04", "director": "Director04", "actors": ["Actor16", "Actor17", "Actor18", "Actor19", "Actor20"], "awards": [], "grossing": 1.7, "budget": 1.2},
    {"title": "Movie05", "director": "Director05", "actors": ["Actor21", "Actor22", "Actor23", "Actor24", "Actor25"], "awards": [], "grossing": 1.5, "budget": 0.8},
    {"title": "Movie06", "director": "Director06", "actors": ["Actor26", "Actor27", "Actor28", "Actor29", "Actor30"], "awards": [{"country": "Spain", "category": "Best Actor"}, {"country": "Spain", "category": "Best Actress"}, {"country": "Spain", "category": "Best Original Screenplay"}, {"country": "Spain", "category": "Best Film"}, {"country": "Spain", "category": "Best Song"}, {"country": "Spain", "category": "Best Soundtrack"}, {"country": "USA", "category": "Best Foreign Film"}], "grossing": 0.6, "budget": 1.0},
    {"title": "Movie07", "director": "Director07", "actors": ["Actor31", "Actor32", "Actor33", "Actor34", "Actor35"], "awards": [], "grossing": 3.2, "budget": 2.9},
    {"title": "Movie08", "director": "Director08", "actors": ["Actor36", "Actor37", "Actor38", "Actor39", "Actor40"], "awards": [{"country": "UK", "category": "Best Actress"}], "grossing": 51.7, "budget": 41.2},
    {"title": "Movie09", "director": "Director09", "actors": ["Actor41", "Actor42", "Actor43", "Actor44", "Actor45"], "awards": [], "grossing": 2.5, "budget": 1.8},
    {"title": "Movie10", "director": "Director01", "actors": ["Actor46", "Actor02", "Actor47", "Actor48", "Actor05"], "awards": [], "grossing": 2.0, "budget": 1.0},
    {"title": "Movie11", "director": "Director10", "actors": ["Actor49", "Actor50", "Actor51", "Actor52", "Actor53"], "awards": [], "grossing": 1.2, "budget": 6.9},
    {"title": "Movie12", "director": "Director11", "actors": ["Actor54", "Actor55", "Actor56", "Actor57", "Actor58"], "awards": [], "grossing": 11.7, "budget": 9.2},
    {"title": "Movie13", "director": "Director12", "actors": ["Actor59", "Actor60", "Actor61", "Actor62", "Actor63"], "awards": [], "grossing": 6.5, "budget": 7.8},
    {"title": "Movie14", "director": "Director13", "actors": ["Actor64", "Actor65", "Actor66", "Actor67", "Actor68"], "awards": [], "grossing": 9.0, "budget": 11.0},
    {"title": "Movie15", "director": "Director02", "actors": ["Actor69", "Actor06", "Actor70", "Actor71", "Actor01"], "awards": [{"country": "USA", "category": "Best Actress"}, {"country": "USA", "category": "Best Soundtrack"}], "grossing": 21.2, "budget": 20.9},
    {"title": "Movie16", "director": "Director14", "actors": ["Actor72", "Actor73", "Actor74", "Actor75", "Actor76"], "awards": [], "grossing": 17.7, "budget": 11.2},
    {"title": "Movie17", "director": "Director15", "actors": ["Actor77", "Actor78", "Actor79", "Actor80", "Actor81"], "awards": [], "grossing": 2.5, "budget": 3.8},
    {"title": "Movie18", "director": "Director16", "actors": ["Actor82", "Actor83", "Actor84", "Actor85", "Actor86"], "awards": [], "grossing": 2.0, "budget": 1.0},
    {"title": "Movie19", "director": "Director02", "actors": ["Actor01", "Actor05", "Actor03", "Actor87", "Actor04"], "awards": [], "grossing": 1.2, "budget": 0.9},
    {"title": "Movie20", "director": "Director04", "actors": ["Actor88", "Actor16", "Actor89", "Actor90", "Actor20"], "awards": [{"country": "UK", "category": "Best Film"}, {"country": "UK", "category": "Best Director"}, {"country": "Spain", "category": "Best European Film"}, {"country": "UK", "category": "Best Actor"}, {"country": "UK", "category": "Best Adapted Screenplay"}, {"country": "UK", "category": "Best Supporting Actress"}], "grossing": 51.7, "budget": 31.2},
]

# Define the app layout
app.layout = html.Div(
    children=[
        html.H1("Visual Analytics Dashboard", style={'textAlign': 'center'}),
        html.Div(
            children=[
                html.Button("Overview", id="btn-overview", n_clicks=0, className="btn"),
                html.Button("Q1", id="btn-q1", n_clicks=0, className="btn"),
                html.Button("Q2", id="btn-q2", n_clicks=0, className="btn"),
                html.Button("Q3", id="btn-q3", n_clicks=0, className="btn"),
                html.Button("Q4", id="btn-q4", n_clicks=0, className="btn"),
            ],
            style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}
        ),
        html.Div(id="page-content"),
    ]
)

# Functions for creating content for Q1-Q4
def createQ1Content():
    return html.Div([
    html.H1("Director-Actor Connections"),
    dcc.Graph(
        id="force-directed-graph",
        figure={
            'data': [],
            'layout': go.Layout(
                title="Director-Actor Connections",
                hovermode="closest",
                showlegend=False
            )
        }
    ),
])

def createQ2Content():
    data = []
    for movie in movies_data:
        data.append({
            "Title": movie["title"],
            "AwardCount": len(movie["awards"]),  # Count awards
            "Grossing": movie["grossing"]
        })
    df = pd.DataFrame(data)

    # Plotly scatter plot for Awards vs Grossing
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["AwardCount"], 
        y=df["Grossing"], 
        mode="markers", 
        marker=dict(color=df["AwardCount"], colorscale='Viridis', size=12)
    ))

    fig.update_layout(
        title='Awards vs Grossing for Movies',
        xaxis_title='Number of Awards',
        yaxis_title='Worldwide Grossing ($)',
        template='plotly_dark',
        showlegend=False,
        xaxis=dict(range=[0, df["AwardCount"].max() + 1]),  # Ensure awards axis starts at 0
        yaxis=dict(range=[df["Grossing"].min() - 1, df["Grossing"].max() + 1])  # Adjust range for better viewing
    )

    return html.Div([
        html.H3("Q2: Awards vs Grossing"),
        dcc.Graph(figure=fig)  # Display the Plotly graph here
    ])

def createQ3Content():
    dirSliderMin = 1
    dirSliderMax = len(set([movie["director"] for movie in movies_data]))

    return html.Div([
    html.H1("Directors' Profitability"),
    
    # Slider to select number of top directors
    dcc.Slider(
        id='q3-top-n-slider',
        min=dirSliderMin,
        max=dirSliderMax,  # Dynamically set the maximum value based on number of unique actors
        step=1,
        value=5,
        marks={i: str(i) for i in range(dirSliderMin, dirSliderMax, math.floor((dirSliderMax-dirSliderMin)/10))},  # Adjust marks for more flexibility
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    
    # Plotly graph
    dcc.Graph(id='directors-profitability-graph')
])

def createQ4Content():
    theSliderMin = 1
    theSliderMax = len(set([actor for movie in movies_data for actor in movie["actors"]]))

    return html.Div([
    html.H1("Actors' Profitability"),
        
    # Slider to select number of top actors
    dcc.Slider(
        id='q4-top-n-slider',
        min=theSliderMin,
        max=theSliderMax,  # Dynamically set the maximum value based on number of unique actors
        step=1,
        value=5,
        marks={i: str(i) for i in range(theSliderMin, theSliderMax, math.floor((theSliderMax-theSliderMin)/10))},  # Adjust marks for more flexibility
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    
    # Plotly graph
    dcc.Graph(id='actors-profitability-graph')
])

# Function to generate a simplified 2x2 view for Q1-Q4 placeholders
def createOverview():
    return html.Div([
        dbc.Row([
            dbc.Col(createQ1Content(), width=6),
            dbc.Col(createQ2Content(), width=6),
        ]),
        dbc.Row([
            dbc.Col(createQ3Content(), width=6),
            dbc.Col(createQ4Content(), width=6),
        ]),
    ])

def generateForceDirectedGraph():
    G = nx.Graph()

    # Add nodes and edges
    for movie in movies_data:
        G.add_node(movie["director"], type="director")
        for actor in movie["actors"]:
            G.add_node(actor, type="actor")
            G.add_edge(movie["director"], actor)

    pos = nx.spring_layout(G, seed=42)  # Layout positions using spring layout

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Edge trace with hover information
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []  # Color differentiation between directors and actors
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if G.nodes[node]['type'] == 'director':
            node_text.append(f'{node} (Director)')
            node_color.append('blue')  # Color for directors
        else:
            node_text.append(f'{node} (Actor)')
            node_color.append('orange')  # Color for actors

    # Node trace with hover information
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',  # Enable hover tooltips
        text=node_text,  # Tooltip content
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=node_color,
        )
    )

    return edge_trace, node_trace

def createDirectorsProfitability(top_n=5):
    # Calculate profitability for each director
    directors_profitability = {}
    for movie in movies_data:
        director = movie["director"]
        profitability = 100 * (movie["grossing"] / movie["budget"] - 1)
        if director not in directors_profitability:
            directors_profitability[director] = []
        directors_profitability[director].append(profitability)

    # Aggregate profitability by director
    director_names = list(directors_profitability.keys())
    average_profitability = [sum(profit) / len(profit) for profit in directors_profitability.values()]

    # Sort by profitability in descending order
    sorted_directors = sorted(zip(director_names, average_profitability), key=lambda x: x[1], reverse=True)

    # Extract top N directors
    top_directors = sorted_directors[:top_n]
    top_director_names = [director for director, _ in top_directors]
    top_director_profitability = [profit for _, profit in top_directors]

    # Create bar plot for the top directors
    fig = go.Figure([go.Bar(x=top_director_names, y=top_director_profitability)])
    fig.update_layout(
        title="Top Directors by Profitability",
        xaxis_title="Director",
        yaxis_title="Profitability (%)",
        showlegend=False
    )

    return fig

def createActorsProfitability(top_n=5):
    actors_profitability = {}
    for movie in movies_data:
        for actor in movie["actors"]:
            profitability = 100 * (movie["grossing"] / movie["budget"] - 1)
            if actor not in actors_profitability:
                actors_profitability[actor] = []
            actors_profitability[actor].append(profitability)

    actor_names = list(actors_profitability.keys())
    average_profitability = [sum(profit) / len(profit) for profit in actors_profitability.values()]

    # Sort by profitability in descending order
    sorted_actors = sorted(zip(actor_names, average_profitability), key=lambda x: x[1], reverse=True)

    # Extract top N actors
    top_actors = sorted_actors[:top_n]
    top_actor_names = [actor for actor, _ in top_actors]
    top_actor_profitability = [profit for _, profit in top_actors]

    # Create bar plot for the top actors
    fig = go.Figure([go.Bar(x=top_actor_names, y=top_actor_profitability)])
    fig.update_layout(
        title="Top Actors by Profitability",
        xaxis_title="Actor",
        yaxis_title="Profitability (%)",
        showlegend=False
    )

    return fig

@app.callback(
    Output("force-directed-graph", "figure"),
    Input('force-directed-graph', 'relayoutData')  # You can add other inputs for interactivity if required
)
def updateQ1Graph(selectedData):
    edge_trace, node_trace = generateForceDirectedGraph()

    return {
        'data': [edge_trace, node_trace],
        'layout': go.Layout(
            title="Director-Actor Connections",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, b=0, t=0)  # Remove unnecessary margins
        )
    }

@app.callback(
    Output('directors-profitability-graph', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateQ3Graph(top_n):
    fig = createDirectorsProfitability(top_n)
    return fig
    
@app.callback(
    Output('actors-profitability-graph', 'figure'),
    [Input('q4-top-n-slider', 'value')]
)
def updateQ4Graph(top_n):
    fig = createActorsProfitability(top_n)
    return fig

# Callback to update the page content based on button clicks
@app.callback(
    Output("page-content", "children"),
    [Input("btn-overview", "n_clicks"),
     Input("btn-q1", "n_clicks"),
     Input("btn-q2", "n_clicks"),
     Input("btn-q3", "n_clicks"),
     Input("btn-q4", "n_clicks")]
)
def updateContent(overview_clicks, q1_clicks, q2_clicks, q3_clicks, q4_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return createOverview()

    buttonID = ctx.triggered_id

    if buttonID == "btn-overview":
        return createOverview()
    elif buttonID == "btn-q1":
        return createQ1Content()
    elif buttonID == "btn-q2":
        return createQ2Content()
    elif buttonID == "btn-q3":
        return createQ3Content()
    elif buttonID == "btn-q4":
        return createQ4Content()

# Callback to highlight the active button
@app.callback(
    [Output("btn-overview", "style"),
     Output("btn-q1", "style"),
     Output("btn-q2", "style"),
     Output("btn-q3", "style"),
     Output("btn-q4", "style")],
    [Input("btn-overview", "n_clicks"),
     Input("btn-q1", "n_clicks"),
     Input("btn-q2", "n_clicks"),
     Input("btn-q3", "n_clicks"),
     Input("btn-q4", "n_clicks")]
)
def highlightButton(overview_clicks, q1_clicks, q2_clicks, q3_clicks, q4_clicks):
    ctx = dash.callback_context
    buttonID = ctx.triggered_id if ctx.triggered else None

    baseStyle = {'width': '10vw'}
    defaultStyle = {**baseStyle, 'background-color': 'white', 'color': 'black'}
    highlightedStyle = {**baseStyle, 'background-color': 'blue', 'color': 'white'}

    if buttonID == "btn-overview":
        return highlightedStyle, defaultStyle, defaultStyle, defaultStyle, defaultStyle
    elif buttonID == "btn-q1":
        return defaultStyle, highlightedStyle, defaultStyle, defaultStyle, defaultStyle
    elif buttonID == "btn-q2":
        return defaultStyle, defaultStyle, highlightedStyle, defaultStyle, defaultStyle
    elif buttonID == "btn-q3":
        return defaultStyle, defaultStyle, defaultStyle, highlightedStyle, defaultStyle
    elif buttonID == "btn-q4":
        return defaultStyle, defaultStyle, defaultStyle, defaultStyle, highlightedStyle

    return highlightedStyle, defaultStyle, defaultStyle, defaultStyle, defaultStyle

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8051)
