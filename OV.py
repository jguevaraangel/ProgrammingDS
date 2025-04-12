import ast
import math
import os
import re
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


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
    """
    Reads director-level movie data from a CSV file and returns it as a pandas DataFrame.

    The CSV file is expected at the relative path './outputs/Q3/directors_info.csv'. Each row
    is initially read and converted into a list of dictionaries, then restructured into a DataFrame
    for further processing and analysis.

    :return: A pandas DataFrame containing the contents of the directors_info.csv file.
    """
    directorsInfo = pd.read_csv("./outputs/Q3/directors_info.csv")
    directorsInfo = directorsInfo.to_dict("records")
    directorsInfo = pd.DataFrame(directorsInfo)
    return directorsInfo


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
    dirSliderMin = 1
    dirSliderMax = 100

    app.layout = html.Div([
        html.H1("Directors Dashboard", style={"textAlign": "center"}),

        dcc.Slider(
            id='q3-top-n-slider',
            min=dirSliderMin,
            max=dirSliderMax,  # Dynamically set the maximum value based on number of unique actors
            step=1,
            value=5,
            marks={i: str(i) for i in range(dirSliderMin, dirSliderMax, math.floor((dirSliderMax-dirSliderMin)/10))},  # Adjust marks for more flexibility
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.H2("Profitability vs Worldwide Gross per Director"),
        dcc.Graph(id='directors-profitability-graph'),

        html.H2("Top Directors by Award Count"),
        html.Div([
            html.Label("Filter by Award Outcome:"),
            dcc.Dropdown(
                id="award-type-dropdown",
                options=[
                    {"label": "All", "value": "all"},
                    {"label": "Wins Only", "value": "wins"},
                    {"label": "Nominations Only", "value": "nominations"},
                ],
                value="all",
                clearable=False,
                style={"width": "300px"}
            )
        ]),
        dcc.Graph(id='top-awarded-directors'),

        html.H2("Roles by Award Director Category"),
        html.Div([
            html.Label("Select Award Director Category:"),
            dcc.Dropdown(
                id="award-category-dropdown",
                options=[
                    {"label": category, "value": category}
                    for category, count in (
                        moviesDataDirectors
                        .assign(award_categories=lambda df: df["award_categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []))
                        .explode("award_categories")
                        .dropna(subset=["award_categories"])
                        .query("award_categories.str.contains('director', case=False)", engine='python')
                        .groupby("award_categories")["director_name"]
                        .nunique()
                        .items()
                    ) if count > 3
                ],
                placeholder="Choose a director-related award...",
                style={"width": "400px"}
            ),
            dcc.Graph(id="roles-stacked-bar")
        ]),

        html.H2("IMDb Rating vs Metascore per Director"),
        dcc.Graph(id='ratings-vs-metascore')
    ])
    return app.layout


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


##############################################################
#################### Functions used in Q3 ####################
##############################################################
def clean_money_column(series):
    """
    Cleans a pandas Series containing monetary values by removing dollar signs and commas.

    This function uses a regular expression to strip out common currency formatting characters,
    making the values suitable for numeric conversion.

    :param series: A pandas Series containing string-formatted monetary values.
    :return: A pandas Series with the "$" and "," characters removed.
    """
    return series.replace("[\$,]", "", regex=True)


def normalize_roles(raw_roles):
    """
    Standardizes role names into a consistent format using a predefined translation map.

    This function helps to group similar or synonymous role labels (e.g., "directora",
    "direccion", and "director") under a common label ("director"). It is useful for
    aggregating roles when analyzing participation in movies.

    :param raw_roles: A list of role strings, potentially in various languages or formats.
    :return: A list of normalized role strings.
    """
    translation_map = {
        # Director
        "director": "director",
        "directora": "director",
        "direccion": "director",
        "direccion o asistencia de la segunda unidad": "assistant director",
        "second unit or assistant director": "assistant director",

        # Writer
        "writer": "writer",
        "guionista": "writer",
        "guion": "writer",
        "escritor": "writer",
        "escritora": "writer",

        # Producer
        "producer": "producer",
        "productor": "producer",
        "productora": "producer",
        "produccion": "producer",

        # Actor / Actress
        "actor": "actor",
        "actriz": "actor",
        "actress": "actor",
        "reparto": "actor",

        # Editor
        "editor": "editor",
        "edicion": "editor",
        "editorial department": "editor",

        # Cinematographer
        "cinematographer": "cinematographer",
        "direccion de fotografia y camara": "cinematographer",
        "camera and electrical department": "cinematographer",

        # Composer / Music
        "composer": "composer",
        "music department": "composer",
        "música": "composer",
        "score": "composer",

        # Visual Effects
        "visual effects": "visual effects",

        # Art / Design
        "art department": "art department",
        "set designer": "art department",
        "decorador": "art department",
        "escenógrafo": "art department",
        "production designer": "art department",
        "direccion de arte": "art department",

        # Animation
        "animation department": "animation",

        # Stunts / Others
        "stunts": "stunts",
        "especialistas": "stunts",
        "additional crew": "crew",
        "location management": "crew",
        "script and continuity department": "crew"
    }
    return [translation_map.get(role.lower().strip(), role.lower().strip()) for role in raw_roles]


def createDirectorsProfitabilityBubble(topN: int = 5) -> go.Figure:
    """
    Creates a bubble chart showing the top N directors ranked by average profitability.

    Profitability is calculated as the percentage gain from budget to worldwide gross
    for each movie, and directors are ranked by their average profitability across all their movies.
    Bubble size encodes the number of movies directed, and color encodes the total worldwide gross.

    :param topN: The number of top directors to include in the chart (default is 5).
    :return: A Plotly Figure object representing the bubble chart.
    """
    df = moviesDataDirectors.copy()

    df["budget"] = clean_money_column(df["budget"])
    df["worldwideGross"] = clean_money_column(df["worldwideGross"])

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["worldwideGross"] = pd.to_numeric(df["worldwideGross"], errors="coerce")

    df = df.dropna(subset=["budget", "worldwideGross"])

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

    data = [
        (
            director,
            sum(d["profitabilities"]) / len(d["profitabilities"]),
            d["gross"],
            d["count"]
        )
        for director, d in directors.items()
    ]

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


def createTopAwardedDirectorsTreemap(topN=5, award_filter="all") -> go.Figure:
    """
    Generates a treemap visualization of the top N most awarded directors.

    The function allows filtering by award outcome type: 'wins', 'nominations', or 'all'.
    If 'all' is selected, it shows both wins and nominations together, with separate counts.
    Each rectangle in the treemap represents a director, sized by total awards.

    :param topN: The number of top directors to display (default is 5).
    :param award_filter: The type of award outcome to include: 'wins', 'nominations', or 'all'.
    :return: A Plotly Figure object representing the treemap of top awarded directors.
    """
    df = moviesDataDirectors.copy()

    df = df[df["award_name"].notna() & (df["award_name"].str.strip() != "")]
    df = df[df["year_result"].notna() & (df["year_result"].str.strip() != "")]

    df["year_result"] = df["year_result"].str.lower()

    if award_filter == "wins":
        df = df[df["year_result"].str.contains("winner|ganador", na=False)]
    elif award_filter == "nominations":
        df = df[df["year_result"].str.contains("nominee|nominado", na=False)]

    if award_filter == "all":
        def classify_award(result):
            """
            Classifies an award result string as either a 'win', 'nomination', or None.

            This helper function checks if a given result string contains keywords related to
            winning or being nominated. It supports both English and Spanish terms.

            :param result: A string describing the award outcome (e.g., "Winner", "Ganador", "Nominado").
            :return: 'win' if the result indicates a win, 'nomination' if it's a nomination, otherwise None.
            """
            if "winner" in result or "ganador" in result:
                return "win"
            elif "nominee" in result or "nominado" in result:
                return "nomination"
            return None

        df["award_type"] = df["year_result"].apply(classify_award)
        df = df[df["award_type"].notna()]

        award_counts = df.groupby(["director_name", "award_type"])["award_name"].count().unstack(fill_value=0)
        award_counts["total"] = award_counts.sum(axis=1)
        award_counts = award_counts.sort_values(by="total", ascending=False).head(topN)

        directors = award_counts.index.tolist()
        values = award_counts["total"].tolist()
        wins = award_counts.get("win", [0]*len(directors)).tolist()
        noms = award_counts.get("nomination", [0]*len(directors)).tolist()

        customdata = list(zip(directors, values, wins, noms))

        fig = go.Figure(go.Treemap(
            labels=directors,
            parents=[""] * len(directors),
            values=values,
            customdata=customdata,
            texttemplate=(
                "<b>%{customdata[0]}</b><br>"  # Director
                "Total: %{customdata[1]}<br>"  # Total
                "Wins: %{customdata[2]}<br>"  # Wins
                "Nominations: %{customdata[3]}<br>"
                "%{percentParent:.1%}"
            ),
            hovertemplate="%{label}<br>Total: %{value}<br>%{percentParent:.1%}<extra></extra>",
            insidetextfont=dict(size=16)
        ))

    else:
        award_counts = (
            df.groupby("director_name")["award_name"]
            .count()
            .sort_values(ascending=False)
            .head(topN)
        )

        if award_counts.empty:
            print("No award data available for the selected filter.")
            return go.Figure()

        directors = award_counts.index.tolist()
        values = award_counts.values.tolist()

        customdata = list(zip(directors, values))

        fig = go.Figure(go.Treemap(
            labels=directors,
            parents=[""] * len(directors),
            values=values,
            customdata=customdata,
            texttemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Total: %{customdata[1]}<br>"
                "%{percentParent:.1%}"
            ),
            hovertemplate="%{label}<br>Total: %{value}<br>%{percentParent:.1%}<extra></extra>",
            insidetextfont=dict(size=16)
        ))

    fig.update_layout(
        title=f"Top {topN} ({award_filter.capitalize()})",
        height=600,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig


def createRatingVsMetascoreScatter() -> go.Figure:
    """
    Generates a scatter plot showing the relationship between IMDb rating and Metascore
    for each director, averaged over their directed movies.

    Each point represents a director, with:
    - X-axis: average Metascore across their movies.
    - Y-axis: average IMDb rating across their movies.
    - Point size/color: number of movies directed (indicated in colorbar and hover info).

    Data is cleaned by:
    - Converting rating values to numeric.
    - Removing rows with missing ratings or scores.

    :return: A Plotly Figure object containing the scatter plot.
    """
    df = moviesDataDirectors.copy()

    df["imdbRating"] = df["imdbRating"].astype(str).str.replace(",", ".")
    df["imdbRating"] = pd.to_numeric(df["imdbRating"], errors="coerce")

    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")

    df = df.dropna(subset=["imdbRating", "metascore"])

    if df.empty:
        print("No data left after cleaning.")
        return go.Figure()

    movie_counts = df.groupby("director_name").size().rename("movie_count")

    avg_scores = df.groupby("director_name")[["imdbRating", "metascore"]].mean()
    avg_scores = avg_scores.join(movie_counts)
    avg_scores = avg_scores.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=avg_scores["metascore"],
        y=avg_scores["imdbRating"],
        mode='markers',
        marker=dict(
            size=10,
            color=avg_scores["movie_count"],
            colorscale="Viridis",
            colorbar=dict(title="Number of Movies"),
            showscale=True,
            line=dict(width=1, color="white")
        ),
        text=avg_scores["director_name"],
        customdata=avg_scores["movie_count"],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Avg IMDb Rating: %{y:.1f}<br>" +
            "Avg Metascore: %{x:.1f}<br>" +
            "Movies Directed: %{customdata}<extra></extra>"
        )
    ))

    fig.update_layout(
        xaxis_title="Average Metascore",
        yaxis_title="Average IMDb Rating",
        template="plotly_dark",
        height=600,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    return fig


def createRolesByAwardCategory(selected_award: str = "Best Director of the Year") -> go.Figure:
    """
    Creates a stacked bar chart showing the roles played by directors in movies that were
    nominated for or won a specific award category.

    This visualization helps to understand the diversity of roles (e.g., director, writer, producer)
    undertaken by directors in award-recognized movies.

    :param selected_award: The name of the award category to filter movies by.
                           Defaults to "Best Director of the Year".
    :return: A Plotly stacked bar chart showing role counts per director for
             the given award category.
    """
    df = moviesDataDirectors.copy()

    df["participation_categories"] = df["participation_categories"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df["participation_categories"] = df["participation_categories"].apply(normalize_roles)
    df["award_categories"] = df["award_categories"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    participation_df = df.explode("participation_categories")
    award_df = df.explode("award_categories")

    selected_award = selected_award or "Best Director of the Year"

    filtered = award_df[award_df["award_categories"] == selected_award]

    matched = participation_df.merge(
        filtered[["director_name", "movie_name"]],
        on=["director_name", "movie_name"],
        how="inner"
    )

    role_count = matched.groupby(
        ["director_name", "participation_categories"]
    ).size().unstack(fill_value=0)

    fig = go.Figure()
    for role in role_count.columns:
        fig.add_trace(go.Bar(
            x=role_count.index,
            y=role_count[role],
            name=role
        ))

    fig.update_layout(
        title=f"Roles in '{selected_award}' Awards",
        xaxis_title="Director",
        yaxis_title="Participation Count",
        barmode="stack",
        template="plotly_white"
    )
    return fig
##############################################################
################ End of Functions used in Q3 #################
##############################################################


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

##############################################################
###################### Callbacks for Q3 ######################
##############################################################
@app.callback(
    Output('directors-profitability-graph', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateQ3Graph(topN: int) -> go.Figure:
    try:
        return createDirectorsProfitabilityBubble(topN)
    except Exception as e:
        print(f"Error in updateQ3Graph: {e}")
        return go.Figure()


@app.callback(
    Output('top-awarded-directors', 'figure'),
    [Input('q3-top-n-slider', 'value'),
     Input('award-type-dropdown', 'value')]
)
def updateAwardsGraph(topN, award_filter):
    return createTopAwardedDirectorsTreemap(topN, award_filter)


@app.callback(
    Output("roles-stacked-bar", "figure"),
    [Input("award-category-dropdown", "value")]
)
def updateRolesGraph(selected_award):
    return createRolesByAwardCategory(selected_award or "Best Director of the Year")


@app.callback(
    Output('ratings-vs-metascore', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateScatterGraph(_):
    return createRatingVsMetascoreScatter()
##############################################################
################## End of Callbacks for Q3 ###################
##############################################################


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
