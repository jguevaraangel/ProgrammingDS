import ast
import math
import re
import os
import dash
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from Q3 import get_q3_layout, register_q3_callbacks

from typing import List, Dict, Tuple

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
register_q3_callbacks(app)

# movie loader
def validateMovieData(moviesDF: pd.DataFrame):
    """
    Validates that the 'directors' and 'thespians' columns in the DataFrame have the
    expected format: a list of dictionaries, where each dictionary has 'name' and 'url' keys.

    :param df: The pandas DataFrame to validate.
    :raises ValueError: If any invalid data is found in the 'directors' or 'thespians' columns.
    """
    # Check if the columns 'directors' and 'thespians' exist in the dataframe
    assert 'directors' in moviesDF.columns, "'directors' column is missing."
    assert 'thespians' in moviesDF.columns, "'thespians' column is missing."

    for index, row in moviesDF.iterrows():
        # Ensure that the 'directors' and 'thespians' are lists of dictionaries
        if isinstance(row['directors'], str):
            # If the column is a string (likely a string representation of a list), try parsing it
            try:
                directors = ast.literal_eval(row['directors'])
                assert isinstance(directors, list), f"Expected list but got {type(directors)} for 'directors' at row {index}."
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid format in 'directors' at row {index}: {row['directors']}") from e
        else:
            directors = row['directors']
        
        if isinstance(row['thespians'], str):
            try:
                thespians = ast.literal_eval(row['thespians'])
                assert isinstance(thespians, list), f"Expected list but got {type(thespians)} for 'thespians' at row {index}."
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid format in 'thespians' at row {index}: {row['thespians']}") from e
        else:
            thespians = row['thespians']
        
        # Validate each director and actor entry
        for director in directors:
            if not isinstance(director, dict) or 'name' not in director or 'url' not in director:
                raise ValueError(f"Invalid director data at row {index}: {director}")
        
        for actor in thespians:
            if not isinstance(actor, dict) or 'name' not in actor or 'url' not in actor:
                raise ValueError(f"Invalid actor data at row {index}: {actor}")
        
def loadMoviesFromCSV(filePath: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Reads movie data from a CSV file and returns a list of movies and a dictionary
    mapping IMDb URLs to actor/director names.

    :param file_path: Path to the CSV file.
    :return: A tuple (moviesData, directorsThespiansURLtoNameDict)
    """
    
    assert isinstance(filePath, str), "The file path must be a string."
    assert os.path.exists(filePath), f"The file at {file_path} does not exist."
    
    moviesDF = pd.read_csv(filePath)
    validateMovieData(moviesDF)
    moviesData = []
    directorsThespiansURLtoNameDict = {}

    for _, row in moviesDF.iterrows():
        # Safely parse the string to a list of dictionaries
        directors = ast.literal_eval(row['directors']) if isinstance(row['directors'], str) else row['directors']
        thespians = ast.literal_eval(row['thespians']) if isinstance(row['thespians'], str) else row['thespians']
        
        movie = {
            'title': row['title'],
            'directors': [{'name': director['name'], 'url': director['url']} for director in directors],
            'actors': [{'name': actor['name'], 'url': actor['url']} for actor in thespians],
            'awards': row['awards'],
            'grossing': row['worldwideGross'],
            'budget': row['budget']
        }
        moviesData.append(movie)

        # Update dictionary mapping URLs to names
        for director in movie['directors']:
            directorsThespiansURLtoNameDict[director['url']] = director['name']
        for actor in movie['actors']:
            directorsThespiansURLtoNameDict[actor['url']] = actor['name']

    return moviesData, directorsThespiansURLtoNameDict

moviesData, directorsThespiansURLtoNameDict = loadMoviesFromCSV('outputs/movies_all_countries.csv')

def loadMoviesFromCSVDirectors():
    directorsInfo = pd.read_csv("./outputs/directors_info.csv")
    return directorsInfo.to_dict("records")

moviesDataDirectors = loadMoviesFromCSVDirectors()

def loadMoviesFromCSVQ2():
    return [
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

moviesDataQ2 = loadMoviesFromCSVQ2()

# End of CSV loading section

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
    forceGraphID = "force-directed-graph"
    forceGraphTitle = "Director-Actor Connections"

    return html.Div([
    html.H1(forceGraphTitle),
    dcc.Graph(
        id=forceGraphID,
        figure={
            'data': [],
            'layout': go.Layout(
                title=forceGraphTitle,
                hovermode="closest",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
                margin=dict(l=0, r=0, b=0, t=0)
            )
        },
        style={"width": "100%", "height": "600px"} 
    ),

    # Popover for node click
    dbc.Popover(
        [
            dbc.PopoverHeader(id="popover-header"),
            dbc.PopoverBody(id="popover-body")
        ],
        id="popover",
        is_open=False,
        target=forceGraphID,
        placement="top-start",
    ),
])

def createQ2Content():
    data = []
    for movie in moviesDataQ2:
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
    return get_q3_layout()


def createQ4Content():
    theSliderMin = 1
    theSliderMax = len(set([actor for movie in moviesDataQ2 for actor in movie["actors"]]))

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
            dbc.Col(html.Div([
                html.H3("Top Directors by Profitability (Bubble Chart)"),
                dcc.Graph(figure=createDirectorsProfitabilityBubble(topN=5))
            ]), width=6),
            dbc.Col(createQ4Content(), width=6),
        ]),
    ])

# Generate Force Directed Graph
def generateForceDirectedGraph() -> Tuple[go.Scatter, go.Scatter, nx.Graph]:
    """
    Generates a force-directed graph of directors and actors based on movie data.

    :param moviesData: List of movie data where each movie contains directors and actors information.
    :param directorsThespiansURLtoNameDict: Dictionary mapping URLs to actor/director names.
    :return: A tuple containing edge trace, node trace, and the graph object.
    """

    G = nx.Graph()

    for movie in moviesData:
        # Iterate over each director in the movie
        for director in movie['directors']:
            directorURL = director['url']
            if directorURL not in G:
                G.add_node(directorURL, type="director", moviesCount=1)
            else:
                G.nodes[directorURL]['moviesCount'] += 1  # Increment movie count for the director

            # Iterate over each actor in the movie
            for actor in movie['actors']:
                actorURL = actor['url']
                if actorURL not in G:
                    G.add_node(actorURL, type="actor", moviesCount=1)
                else:
                    G.nodes[actorURL]['moviesCount'] += 1  # Increment movie count for the actor

                # Add edge between director and actor
                G.add_edge(directorURL, actorURL)

    # Set positions for nodes using a spring layout
    pos = nx.spring_layout(G, seed=42)  # Spring layout positions
    edgeX, edgeY = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edgeX.append(x0)
        edgeX.append(x1)
        edgeY.append(y0)
        edgeY.append(y1)

    edgeTrace = go.Scatter(
        x=edgeX, y=edgeY,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    nodeX, nodeY, nodeText, nodeColor, nodeSize = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        nodeX.append(x)
        nodeY.append(y)
        
        # Use the directorsThespiansURLtoNameDict to translate URLs to names
        nodeName = directorsThespiansURLtoNameDict.get(node, node)  # Use URL if not found in the dict
        nodeText.append(f"{nodeName} ({G.nodes[node]['type']})<br><a href='{node}'>IMDb page</a>")
        nodeColor.append('blue' if G.nodes[node]['type'] == 'director' else 'orange')

        # Vary node size based on the number of movies (moviesCount)
        nodeSize.append(5 + G.nodes[node]['moviesCount'] * 5)  # Scale size (adjust multiplier if needed)

    nodeTrace = go.Scatter(
        x=nodeX, y=nodeY,
        mode='markers',
        hoverinfo='text',
        text=nodeText,
        marker=dict(
            showscale=False,
            colorscale=None,
            size=nodeSize,  # Vary the size based on movie count
            color=nodeColor
        )
    )
    return edgeTrace, nodeTrace, G

def createDirectorsProfitability(top_n=5):
    # Calculate profitability for each director
    directors_profitability = {}
    for movie in moviesData:
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

def clean_money_column(series):
    return series.replace("[\$,]", "", regex=True)

def createDirectorsProfitabilityBubble(topN: int = 5) -> go.Figure:
    df = pd.DataFrame(moviesDataDirectors)

    df["budget"] = clean_money_column(df["budget"])
    df["worldwideGross"] = clean_money_column(df["worldwideGross"])

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["worldwideGross"] = pd.to_numeric(df["worldwideGross"], errors="coerce")

    df = df.dropna(subset=["budget", "worldwideGross"])

    # Initialize tracking
    directors = {}

    for _, row in df.iterrows():
        director = row["director_name"]
        profit = 100 * (row["worldwideGross"] / row["budget"] - 1)
        if director not in directors:
            directors[director] = {
                "profitabilities": [],
                "gross": 0,
                "count": 0
            }
        directors[director]["profitabilities"].append(profit)
        directors[director]["gross"] += row["worldwideGross"]
        directors[director]["count"] += 1

    # Compute average profitability and format into list
    data = [
        (
            director,
            sum(d["profitabilities"]) / len(d["profitabilities"]),
            d["gross"],
            d["count"]
        )
        for director, d in directors.items()
    ]

    # Sort by profitability and take top N
    data = sorted(data, key=lambda x: x[1], reverse=True)[:topN]

    names = [d[0] for d in data]
    profits = [d[1] for d in data]
    gross = [d[2] for d in data]
    counts = [d[3] for d in data]

    fig = go.Figure(go.Scatter(
        x=gross,
        y=profits,
        mode="markers+text",
        text=names,
        textposition="top center",
        marker=dict(
            size=[max(8, c * 3) for c in counts],
            sizemode="area",
            sizeref=2.*max(counts)/(100**2),
            sizemin=8,
            color=gross,
            colorscale="earth",
            colorbar=dict(title="Total Gross ($)")
        ),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Profitability: %{y}<br>" +
            "Gross: %{x:$,.0f}<br>" +
            "Movies: %{marker.size:.0f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=f"Top {topN} Directors by Profitability",
        xaxis_title="Total Worldwide Gross ($)",
        yaxis_title="Profitability (%)",
        template="plotly_white",
        height=600,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig

def createActorsProfitability(top_n=5):
    actors_profitability = {}
    for movie in moviesDataQ2:
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
    [
        Output("force-directed-graph", "figure"),
        Output("popover", "is_open"),
        Output("popover-header", "children"),
        Output("popover-body", "children"),
        Output("popover-header", "style"),
    ],
    [
        Input('force-directed-graph', 'clickData')
    ]
)

def updateGraphAndPopover(clickData: Dict) -> Tuple[Dict, bool, str, str, Dict]:
    """
    Updates the graph and popover based on the click event data.

    :param clickData: The data from the click event on the graph. This should contain information 
                      such as the node clicked and its associated tooltip.
    :return: A tuple containing:
             - The graph data (edges and nodes).
             - A boolean indicating if the popover is open.
             - The popover header text.
             - The popover body content (as Markdown).
             - The header style for the popover.
    """
    
    edgeTrace, nodeTrace, G = generateForceDirectedGraph()

    # Default popover state
    is_open = False
    header = "Node info error"
    body = "Try again later"
    header_style = {'color': 'white', 'backgroundColor': 'red', 'padding': '5px'}

    if clickData:
        # Assert clickData structure is as expected
        assert isinstance(clickData, dict), "clickData should be a dictionary."
        assert 'points' in clickData, "clickData must contain 'points'."
        assert len(clickData['points']) > 0, "clickData['points'] cannot be empty."

        is_open = True
        nodeTooltip = clickData['points'][0].get('text', None)
        if nodeTooltip:
            # Regular expression to capture name, type, and IMDb URL
            pattern = r"(?P<name>.*?)\s\((?P<type>.*?)\)<br><a href='(?P<url>.*?)'>IMDb page</a>"
            match = re.search(pattern, nodeTooltip)

            if match:
                nodeName = match.group("name")
                nodeType = match.group("type")
                nodeURL = match.group("url")
                moviesCount = G.nodes[nodeURL]['moviesCount']
            else:
                nodeName = "Name not found"
                nodeType = "Profession not found"
                nodeURL = "IMDb URL not found"
                moviesCount = 0

            # Set popover content
            header = f"{nodeName} ({nodeType})"
            header_style = {'color': 'white', 'backgroundColor': 'blue', 'padding': '5px'} if nodeType == "director" else {'color': 'black', 'backgroundColor': 'orange', 'padding': '5px'}
            body = dcc.Markdown(f'''
                Number of movies: {moviesCount}                
                [IMDb page]({nodeURL})
            ''', link_target="_blank",)

    return {
        'data': [edgeTrace, nodeTrace],
        'layout': go.Layout(
            title="Director-Actor Connections",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, b=0, t=0)
        ),
    }, is_open, header, body, header_style

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
