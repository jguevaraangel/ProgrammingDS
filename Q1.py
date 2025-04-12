import networkx as nx
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import math
import pandas as pd

###############################################################
# BEGIN OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET #
###############################################################

# Begin of static mock movie loader

def loadMoviesFromCSV():
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

moviesData = loadMoviesFromCSV()

# End of static mock movie loader

# Begin of layout settings

# End of layout settings

# Standalone app initialiser

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

###############################################################
#  END OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET  #
###############################################################

#################################
# BEGIN OF THE INTEGRATION ZONE #
#################################

# Begin of the AUXILIARY FUNCTIONS section

def generateForceDirectedGraph():
    G = nx.Graph()

    # Add nodes and edges
    for movie in moviesData:
        G.add_node(movie["director"], type="director")
        for actor in movie["actors"]:
            G.add_node(actor, type="actor")
            G.add_edge(movie["director"], actor)

    pos = nx.spring_layout(G, seed=42)  # Spring layout positions
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node} ({G.nodes[node]["type"]})')  # Correctly assign the text field
        node_color.append('blue' if G.nodes[node]['type'] == 'director' else 'orange')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',  # Show the text on hover
        text=node_text,  # This provides the text to be shown on hover and used in clickData
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=node_color
        )
    )
    return edge_trace, node_trace

# Layout
app.layout = html.Div([
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

    # Popover for node click
    dbc.Popover(
        [
            dbc.PopoverHeader(id="popover-header"),
            dbc.PopoverBody(id="popover-body")
        ],
        id="popover",
        is_open=False,
        target="force-directed-graph",  # Popover is connected to the graph
        placement="top",  # Popover placement relative to the graph
    ),
])

# Callback for updating the popover with node details
@app.callback(
    [Output("popover", "is_open"),
     Output("popover-header", "children"),
     Output("popover-body", "children"),
     Output("popover", "target"),
     Output("popover", "placement")],
    [Input('force-directed-graph', 'clickData')]
)
def displayPopover(clickData):
    if not clickData:
        return False, "", "", "", ""

    # Get the clicked node data
    node_name = clickData['points'][0]['text'].split(" ")[0]  # Extract node name from text
    node_type = clickData['points'][0]['text'].split(" ")[1].strip('()')  # Extract node type (actor/director)

    # Find number of movies the actor/director is involved in
    movies_count = sum(1 for movie in moviesData if node_name in movie['actors'] or node_name == movie['director'])

    # Set popover content
    header = f"{node_name} ({node_type})"
    body = f"Number of movies: {movies_count}"

    # Set the target dynamically (optional, not needed for popover but can be useful)
    target = f"graph-node-{node_name}"

    # Placement logic (show at the bottom if close to the top)
    placement = "bottom" if clickData['points'][0]['y'] < 0 else "top"

    return True, header, body, target, placement

# Callback for updating the graph
@app.callback(
    Output("force-directed-graph", "figure"),
    [Input('force-directed-graph', 'relayoutData')]
)
def updateQ1Graph(selectedData):
#    edgeTrace, nodeTrace, pos = generateForceDirectedGraph()
    edgeTrace, nodeTrace = generateForceDirectedGraph()

    return {
        'data': [edgeTrace, nodeTrace],
        'layout': go.Layout(
            title="Director-Actor Connections",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, b=0, t=0)  # Remove unnecessary margins
        )
    }

if __name__ == "__main__":
    app.run(debug=True)
