# pip install plotly dash

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import math
import ast
import re
import pandas as pd
import numpy as np

###############################################################
# BEGIN OF CODE TO MAKE IT WORK, NOT THE INTEGRATION ZONE YET #
###############################################################

# Begin of static mock movie loader


def loadMoviesFromCSV(file_path="outputs/movies_all_countries.csv"):
    df = pd.read_csv(file_path)
    return df


def processMovieData(data):
    data["title"] = data["title"].apply(lambda x: re.sub(r"^\d+\.\s*", "", x))
    data["budget"] = (
        data["budget"].str.replace(",", "").str.replace("$", "").astype(float)
    )
    data["grossing"] = (
        data["domesticGross"].str.replace(",", "").str.replace("$", "").astype(float)
    )

    parsed_thespians = []
    for thespians_list in data["thespians"]:
        thespians_list = ast.literal_eval(thespians_list)
        if len(thespians_list) == 0:
            parsed_thespians.append([])
        else:
            parsed_thespians.append([obj["name"] for obj in thespians_list])
    data["actors"] = parsed_thespians

    json_list = (
        data[
            [
                "title",
                "country",
                "imdbRating",
                "imdbVotes",
                "metascore",
                "budget",
                "awards",
                "directors",
                "actors",
                "grossing",
            ]
        ]
        .dropna()
        .to_dict(orient="records")
    )
    return json_list


rawMoviesData = loadMoviesFromCSV()
moviesData = processMovieData(rawMoviesData)

# End of static mock movie loader

# Begin of layout settings
theSliderMin = 1
theSliderMax = 90
# theSliderMax = len(set([actor for movie in moviesData for actor in movie["actors"]]))

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

from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# Transform data to a DataFrame for flexibility
def get_actor_profitability_df(country_filter="All"):
    records = []
    for movie in moviesData:
        if country_filter != "All" and movie["country"] != country_filter:
            continue
        for actor in movie["actors"]:
            profitability = 100 * (movie["grossing"] / movie["budget"] - 1)
            records.append(
                {
                    "actor": actor,
                    "profitability": profitability,
                    "title": movie["title"],
                    "country": movie["country"],
                }
            )
    return pd.DataFrame(records)


# Visual 1: Top N Actors by Average Profitability
def plot_top_actors_profitability(topN, country):
    df = get_actor_profitability_df(country)
    top_avg = (
        df.groupby("actor")["profitability"]
        .mean()
        .sort_values(ascending=False)
        .head(topN)
    )
    fig = go.Figure(go.Bar(x=top_avg.index, y=top_avg.values))
    fig.update_layout(
        title="Top Actors by Average Profitability",
        xaxis_title="Actor",
        yaxis_title="Profitability",
    )
    return fig


# Visual 2: Bottom N Actors by Profitability
def plot_bottom_actors_profitability(topN, country):
    df = get_actor_profitability_df(country)

    # Compute average profitability per actor
    avg_profit = df.groupby("actor")["profitability"].mean()

    # Filter only actors with negative profitability
    negative_profit_actors = avg_profit[avg_profit < 0].sort_values().head(topN)

    if negative_profit_actors.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Actors with Negative Profitability Found",
            xaxis_title="Actor",
            yaxis_title="Profitability",
        )
    else:
        fig = go.Figure(
            go.Bar(x=negative_profit_actors.index, y=negative_profit_actors.values)
        )
        fig.update_layout(
            title="Bottom Actors by Negative Profitability",
            xaxis_title="Actor",
            yaxis_title="Profitability",
        )
    return fig


# Visual 3: Profitability Distribution Histogram
def plot_profitability_distribution(country):
    df = get_actor_profitability_df(country)
    x = df["profitability"].dropna()

    # Histogram
    fig = px.histogram(
        df, x="profitability", nbins=50, histnorm="probability density", opacity=0.6
    )

    # KDE line
    kde = gaussian_kde(x)
    x_range = np.linspace(x.min(), x.max(), 500)
    kde_values = kde(x_range)
    fig.add_trace(
        go.Scatter(
            x=x_range, y=kde_values, mode="lines", name="KDE", line=dict(color="red")
        )
    )

    fig.update_layout(
        title="Distribution of Actor Profitabilities with KDE",
        xaxis_title="Profitability",
        yaxis_title="Density",
        legend=dict(x=0.7, y=0.95),
    )

    return fig


# App Layout
app.layout = html.Div(
    [
        html.H1("Actors Profitability Analysis"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Visualization Type:"),
                        dcc.Dropdown(
                            id="viz-type",
                            options=[
                                {
                                    "label": "Top N Actors by Profitability",
                                    "value": "top",
                                },
                                {
                                    "label": "Bottom N Actors by Profitability",
                                    "value": "bottom",
                                },
                                {
                                    "label": "Profitability Distribution",
                                    "value": "hist",
                                },
                            ],
                            value="top",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"flex": "1", "margin-right": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Filter by Country:"),
                        dcc.Dropdown(
                            id="country-dropdown",
                            options=[{"label": "All Countries", "value": "All"}]
                            + [
                                {"label": country, "value": country}
                                for country in sorted(
                                    set(movie["country"] for movie in moviesData)
                                )
                            ],
                            value="All",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    style={"flex": "1", "margin-left": "10px"},
                ),
            ],
            style={
                "display": "flex",
                "width": "100%",
                "justify-content": "space-between",
            },
        ),
        html.Br(),
        # ✅ Wrap slider in a container div
        html.Div(
            id="slider-container",
            children=[
                dcc.Slider(
                    id="top-n-slider",
                    min=1,
                    max=theSliderMax,
                    step=1,
                    value=10,
                    marks={
                        i: str(i)
                        for i in range(
                            theSliderMin,
                            theSliderMax,
                            max(1, math.floor((theSliderMax - theSliderMin) // 10)),
                        )
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                )
            ],
        ),
        html.Br(),
        dcc.Graph(id="profitability-graph"),
    ]
)


# Callback to switch plots
@app.callback(
    Output("profitability-graph", "figure"),
    Input("viz-type", "value"),
    Input("top-n-slider", "value"),
    Input("country-dropdown", "value"),
)
def update_graph(viz_type, topN, selected_country):
    if viz_type == "top":
        return plot_top_actors_profitability(topN, selected_country)
    elif viz_type == "bottom":
        return plot_bottom_actors_profitability(topN, selected_country)
    elif viz_type == "hist":
        return plot_profitability_distribution(selected_country)
    return go.Figure()


@app.callback(Output("slider-container", "style"), Input("viz-type", "value"))
def toggle_slider_visibility(viz_type):
    if viz_type == "hist":  # Hide the slider for histogram
        return {"display": "none"}
    return {"display": "block"}


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
