 # pip install plotly dash

import ast
import math

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

###############################################################
# BEGIN OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET #
###############################################################

# Begin of static mock movie loader


def loadMoviesFromCSV():
    directorsInfo = pd.read_csv("./outputs/Q3/directors_info.csv")
    return directorsInfo.to_dict("records")


moviesData = loadMoviesFromCSV()

# End of static mock movie loader

dirSliderMin = 1
dirSliderMax = 100

# Standalone app initialiser

app = dash.Dash(__name__)

###############################################################
#  END OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET  #
###############################################################

#################################
# BEGIN OF THE INTEGRATION ZONE #
#################################

# Begin of the AUXILIARY FUNCTIONS section

def clean_money_column(series):
    return series.replace("[\$,]", "", regex=True)


def createDirectorsProfitability(topN: int = 5) -> go.Figure:
    df = pd.DataFrame(moviesData)
    df["budget"] = clean_money_column(df["budget"])
    df["worldwideGross"] = clean_money_column(df["worldwideGross"])

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["worldwideGross"] = pd.to_numeric(df["worldwideGross"], errors="coerce")

    df = df.dropna(subset=["budget", "worldwideGross"])

    # Calculate profitability for each director
    directorsProfitability = {}
    for _, row in df.iterrows():
        director = row["director_name"]
        profitability = 100 * (row["worldwideGross"] / row["budget"] - 1)
        if director not in directorsProfitability:
            directorsProfitability[director] = []
        directorsProfitability[director].append(profitability)


    # Aggregate profitability by director
    directorNames = list(directorsProfitability.keys())
    averageProfitability = [sum(profit) / len(profit) for profit in directorsProfitability.values()]

    # Sort by profitability in descending order
    sortedDirectors = sorted(zip(directorNames, averageProfitability), key=lambda x: x[1], reverse=True)

    # Extract top N directors
    topDirectors = sortedDirectors[:topN]
    topDirectorNames = [director for director, _ in topDirectors]
    topDirectorProfitability = [profit for _, profit in topDirectors]

    # Create bar plot for the top directors
    fig = go.Figure([go.Bar(x=topDirectorNames, y=topDirectorProfitability)])
    fig.update_layout(
        title=f"Top {topN} Directors by Profitability",
        xaxis_title="Director",
        yaxis_title="Profitability (%)",
        showlegend=False
    )
    return fig


def createTopAwardedDirectors(topN: int = 5) -> go.Figure:
    df = pd.DataFrame(moviesData)
    # Drop rows with missing or empty award names
    df = df[df["award_name"].notna() & (df["award_name"].str.strip() != "")]
    # Count awards per director
    award_counts = (
        df.groupby("director_name")["award_name"]
        .count()
        .sort_values(ascending=False)
        .head(topN)
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=award_counts.index.tolist(),
                y=award_counts.values.tolist()
            )
        ]
    )
    fig.update_layout(
        title=f"Top {topN} Directors by Award Count",
        xaxis_title="Director",
        yaxis_title="Number of Awards"
    )

    return fig


def createRatingVsMetascoreScatter() -> go.Figure:
    df = pd.DataFrame(moviesData)

    # Clean 'imdbRating' (commas to dots)
    df["imdbRating"] = df["imdbRating"].astype(str).str.replace(",", ".")
    df["imdbRating"] = pd.to_numeric(df["imdbRating"], errors="coerce")

    # Convert metascore to numeric
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")

    df = df.dropna(subset=["imdbRating", "metascore"])

    if df.empty:
        print("No data left after cleaning.")
        return go.Figure()

    # Group by director and compute mean scores
    avg_scores = df.groupby("director_name")[["imdbRating", "metascore"]].mean().reset_index()

    # Create scatter plot using go.Figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=avg_scores["metascore"],
        y=avg_scores["imdbRating"],
        mode='markers',
        marker=dict(size=8, color='rgba(0, 123, 255, 0.7)', line=dict(width=1)),
        text=avg_scores["director_name"],
        hovertemplate='<b>%{text}</b><br>Metascore: %{x}<br>IMDb: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title="Director: IMDb Rating vs Metascore",
        xaxis_title="Average Metascore",
        yaxis_title="Average IMDb Rating",
        template="plotly_dark"
    )

    return fig


def createDirectorsByGrossing(topN=5) -> go.Figure:
    df = pd.DataFrame(moviesData)

    # Optional: Clean currency strings if needed
    if df["worldwideGross"].dtype == object:
        df["worldwideGross"] = df["worldwideGross"].astype(str).str.replace(r"[\$,]", "", regex=True)

    df["worldwideGross"] = pd.to_numeric(df["worldwideGross"], errors="coerce")
    df = df.dropna(subset=["worldwideGross"])
    if df.empty:
        print("No data available for worldwideGross.")
        return go.Figure()

    # Group by director and sum gross
    gross = (
        df.groupby("director_name")["worldwideGross"]
        .sum()
        .sort_values(ascending=False)
        .head(topN)
    )

    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=gross.index.tolist(),
                y=gross.values.tolist()
            )
        ]
    )

    fig.update_layout(
        title=f"Top {topN} Directors by Worldwide Gross",
        xaxis_title="Director",
        yaxis_title="Total Worldwide Gross"
    )

    return fig


# End of the AUXILIARY FUNCTIONS section

# Begin of the LAYOUT section


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
    html.H2("Directors Profitability"),
    dcc.Graph(id='directors-profitability-graph'),

    html.H2("Top Directors by Award Count"),
    dcc.Graph(id='top-awarded-directors'),

    html.H2("Total Worldwide Gross by Director"),
    dcc.Graph(id='grossing-directors'),

    html.H2("IMDb Rating vs Metascore per Director"),
    dcc.Graph(id='ratings-vs-metascore')
])


# End of the LAYOUT section

# Begin of the CALLBACK section


@app.callback(
    Output('directors-profitability-graph', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateQ3Graph(topN: int) -> go.Figure:
    try:
        return createDirectorsProfitability(topN)
    except Exception as e:
        print(f"Error in updateQ3Graph: {e}")
        return go.Figure()


@app.callback(
    Output('top-awarded-directors', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateAwardsGraph(topN: int) -> go.Figure:
    return createTopAwardedDirectors(topN)


@app.callback(
    Output('grossing-directors', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateGrossGraph(topN: int) -> go.Figure:
    return createDirectorsByGrossing(topN)


@app.callback(
    Output('ratings-vs-metascore', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def updateScatterGraph(_):
    return createRatingVsMetascoreScatter()


# End of the CALLBACK section

#################################
#  END OF THE INTEGRATION ZONE  #
#################################

##########################################
# BEGIN OF THE STANDALONE EXECUTION ZONE #
##########################################


if __name__ == "__main__":
    app.run(debug=True, port=8051)

##########################################
#  END OF THE STANDALONE EXECUTION ZONE  #
##########################################
