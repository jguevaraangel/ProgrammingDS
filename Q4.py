# pip install plotly dash

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import math

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

theSliderMin = 1
theSliderMax = len(set([actor for movie in moviesData for actor in movie["actors"]]))

# End of layout settings

# Standalone app initialiser

app = dash.Dash(__name__)

###############################################################
#  END OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET  #
###############################################################

#################################
# BEGIN OF THE INTEGRATION ZONE #
#################################

# Begin of the AUXILIARY FUNCTIONS section

def createActorsProfitability(topN: int = 5) -> go.Figure:
    actorsProfitability = {}
    for movie in moviesData:
        for actor in movie["actors"]:
            profitability = 100 * (movie["grossing"] / movie["budget"] - 1)
            if actor not in actorsProfitability:
                actorsProfitability[actor] = []
            actorsProfitability[actor].append(profitability)

    actorNames = list(actorsProfitability.keys())
    averageProfitability = [sum(profit) / len(profit) for profit in actorsProfitability.values()]

    # Sort by profitability in descending order
    sortedActors = sorted(zip(actorNames, averageProfitability), key=lambda x: x[1], reverse=True)

    # Extract top N actors
    topActors = sortedActors[:topN]
    topActorNames = [actor for actor, _ in topActors]
    topActorProfitability = [profit for _, profit in topActors]

    # Create bar plot for the top actors
    fig = go.Figure([go.Bar(x=topActorNames, y=topActorProfitability)])
    fig.update_layout(
        title="Top Actors by Profitability",
        xaxis_title="Actor",
        yaxis_title="Profitability (%)",
        showlegend=False
    )

    return fig

# End of the AUXILIARY FUNCTIONS section

# Begin of the LAYOUT section
app.layout = html.Div([
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

# End of the LAYOUT section

# Begin of the CALLBACK section

@app.callback(
    Output('actors-profitability-graph', 'figure'),
    [Input('q4-top-n-slider', 'value')]
)
def updateQ4Graph(topN: int) -> go.Figure:
    fig = createActorsProfitability(topN)
    return fig

# End of the CALLBACK section

#################################
#  END OF THE INTEGRATION ZONE  #
#################################

##########################################
# BEGIN OF THE STANDALONE EXECUTION ZONE #
##########################################

if __name__ == "__main__":
    app.run(debug=True)

##########################################
#  END OF THE STANDALONE EXECUTION ZONE  #
##########################################
