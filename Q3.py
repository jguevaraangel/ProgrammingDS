 # pip install plotly dash
import ast
import math

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

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

    # Calculate profitability and gross per director
    directorsProfitability = {}
    directorsGross = {}

    for _, row in df.iterrows():
        director = row["director_name"]
        profitability = 100 * (row["worldwideGross"] / row["budget"] - 1)

        if director not in directorsProfitability:
            directorsProfitability[director] = []
            directorsGross[director] = 0

        directorsProfitability[director].append(profitability)
        directorsGross[director] += row["worldwideGross"]

    # Aggregate values
    directorNames = list(directorsProfitability.keys())
    averageProfitability = [sum(p) / len(p) for p in directorsProfitability.values()]
    totalGross = [directorsGross[d] for d in directorNames]

    # Sort and extract top directors
    sortedData = sorted(zip(directorNames, averageProfitability, totalGross), key=lambda x: x[1], reverse=True)
    topDirectors = sortedData[:topN]

    topDirectorNames = [d for d, _, _ in topDirectors]
    topDirectorProfitability = [p for _, p, _ in topDirectors]
    topDirectorGross = [g for _, _, g in topDirectors]

    # Create bar plot with color based on gross
    fig = go.Figure(go.Bar(
        x=topDirectorNames,
        y=topDirectorProfitability,
        marker=dict(
            color=topDirectorGross,
            colorscale="earth",  # <-- light-friendly palette
            colorbar=dict(title="Total Gross ($)")
        ),
        hovertemplate="<b>%{x}</b><br>Profitability: %{y}<br>Total Gross: $%{marker.color:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Top {topN} Directors by Profitability",
        xaxis_title="Director",
        yaxis_title="Profitability (%)",
        template="plotly_white",  # <-- light theme
        height=500,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig


def createDirectorsProfitabilityBubble(topN: int = 5) -> go.Figure:
    df = pd.DataFrame(moviesData)

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


def createTopAwardedDirectorsTreemap(topN=5, award_filter="all") -> go.Figure:
    df = pd.DataFrame(moviesData)

    # Clean missing values
    df = df[df["award_name"].notna() & (df["award_name"].str.strip() != "")]
    df = df[df["year_result"].notna() & (df["year_result"].str.strip() != "")]

    df["year_result"] = df["year_result"].str.lower()

    if award_filter == "wins":
        df = df[df["year_result"].str.contains("winner|ganador", na=False)]
    elif award_filter == "nominations":
        df = df[df["year_result"].str.contains("nominee|nominado", na=False)]

    if award_filter == "all":
        # Classify award outcome
        def classify_award(result):
            if "winner" in result or "ganador" in result:
                return "win"
            elif "nominee" in result or "nominado" in result:
                return "nomination"
            return None

        df["award_type"] = df["year_result"].apply(classify_award)
        df = df[df["award_type"].notna()]

        # Count wins + nominations + total
        award_counts = df.groupby(["director_name", "award_type"])["award_name"].count().unstack(fill_value=0)
        award_counts["total"] = award_counts.sum(axis=1)
        award_counts = award_counts.sort_values(by="total", ascending=False).head(topN)

        directors = award_counts.index.tolist()
        values = award_counts["total"].tolist()
        wins = award_counts.get("win", [0]*len(directors)).tolist()
        noms = award_counts.get("nomination", [0]*len(directors)).tolist()

        # Use customdata to include wins, noms, total
        customdata = list(zip(directors, values, wins, noms))

        fig = go.Figure(go.Treemap(
            labels=directors,
            parents=[""] * len(directors),
            values=values,
            customdata=customdata,
            texttemplate=(
                "<b>%{customdata[0]}</b><br>"  # Director
                "Total: %{customdata[1]}<br>"  # Total
                "🏆 Wins: %{customdata[2]}<br>"  # Wins
                "🎯 Nominations: %{customdata[3]}<br>"
                "%{percentParent:.1%}"  # Show percent of root (all)
            ),
            hovertemplate="%{label}<br>Total: %{value}<br>%{percentParent:.1%}<extra></extra>",
            insidetextfont=dict(size=16)
        ))

    else:
        # Single outcome (wins or nominations)
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
            hovertemplate="%{label}<br>Total: %{value}<br>%{percentParent:.1%}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Top {topN} ({award_filter.capitalize()})",
        height=600,
        margin=dict(t=50, l=25, r=25, b=25)
    )

    return fig


def createRatingVsMetascoreScatter() -> go.Figure:
    df = pd.DataFrame(moviesData)

    # Clean IMDb rating: replace commas and convert
    df["imdbRating"] = df["imdbRating"].astype(str).str.replace(",", ".")
    df["imdbRating"] = pd.to_numeric(df["imdbRating"], errors="coerce")

    # Convert metascore to numeric
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=["imdbRating", "metascore"])

    if df.empty:
        print("No data left after cleaning.")
        return go.Figure()

    # Count number of movies per director
    movie_counts = df.groupby("director_name").size().rename("movie_count")

    # Group by director and calculate average scores
    avg_scores = df.groupby("director_name")[["imdbRating", "metascore"]].mean()
    avg_scores = avg_scores.join(movie_counts)
    avg_scores = avg_scores.reset_index()

    # Create scatter plot
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

    # Layout styling
    fig.update_layout(
        xaxis_title="Average Metascore",
        yaxis_title="Average IMDb Rating",
        template="plotly_dark",
        height=600,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    return fig


def normalize_roles(raw_roles):
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


def createRolesByAwardCategory(selected_award: str) -> go.Figure:
    df = pd.DataFrame(moviesData)

    # Clean up lists
    df["participation_categories"] = df["participation_categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df["participation_categories"] = df["participation_categories"].apply(normalize_roles)
    df["award_categories"] = df["award_categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    participation_df = df.explode("participation_categories")
    award_df = df.explode("award_categories")

    # Filter based on dropdown selection
    filtered = award_df[award_df["award_categories"] == selected_award]

    matched = participation_df.merge(
        filtered[["director_name", "movie_name"]],
        on=["director_name", "movie_name"],
        how="inner"
    )

    role_count = matched.groupby(["director_name", "participation_categories"]).size().unstack(fill_value=0)

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


# End of the AUXILIARY FUNCTIONS section

# Begin of the LAYOUT section
def get_q3_layout():
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
                        pd.DataFrame(moviesData)
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


# End of the LAYOUT section

# Begin of the CALLBACK section

def register_q3_callbacks(app):
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
        if not selected_award:
            return go.Figure()
        return createRolesByAwardCategory(selected_award)



    @app.callback(
        Output('ratings-vs-metascore', 'figure'),
        [Input('q3-top-n-slider', 'value')]
    )
    def updateScatterGraph(_):
        return createRatingVsMetascoreScatter()


# End of the CALLBACK section


if __name__ == "__main__":
    app.run(debug=True, port=8051)