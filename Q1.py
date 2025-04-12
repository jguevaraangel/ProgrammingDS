import networkx as nx
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math
import pandas as pd
import ast
import re
import os
from typing import List, Dict, Tuple

###############################################################
# BEGIN OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET #
###############################################################

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

moviesData, directorsThespiansURLtoNameDict = loadMoviesFromCSV('movies_all_countries.csv')

# Standalone app initialiser
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

###############################################################
#  END OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET  #
###############################################################

#################################
# BEGIN OF THE INTEGRATION ZONE #
#################################

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

# Layout
forceGraphID = "force-directed-graph"
forceGraphTitle = "Director-Actor Connections"

app.layout = html.Div([
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

# Callback for updating the popover with node details

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

#################################
#  END OF THE INTEGRATION ZONE  #
#################################

if __name__ == "__main__":
    app.run(debug=True)
