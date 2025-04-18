import ast
import math
import os
import re
import json
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import gaussian_kde

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

##############################################################
######################## Content Load ########################
##############################################################

def validate_movie_data(movies_df: pd.DataFrame) -> None:
    """
    Validates that the 'directors' and 'thespians' columns in the DataFrame have the
    expected format: a list of dictionaries, where each dictionary has 'name' and 'url' keys.

    :param df: The pandas DataFrame to validate.
    :raises ValueError: If any invalid data is found in the 'directors' or 'thespians' columns.
    """
    # Check if the required columns exist in the dataframe
    assert 'directors' in movies_df.columns, "'directors' column is missing."
    assert 'thespians' in movies_df.columns, "'thespians' column is missing."
    assert 'title' in movies_df.columns, "'title' column is missing."
    assert 'awards' in movies_df.columns, "'awards' column is missing."
    assert 'worldwideGross' in movies_df.columns, "'worldwideGross' column is missing."
    assert 'budget' in movies_df.columns, "'budget' column is missing."

    for index, row in movies_df.iterrows():
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


def load_movies_df_from_csv(file_path: str = 'outputs/movies_all_countries.csv') -> pd.DataFrame:
    """
    Reads movie data from a CSV file and returns a list of movies in a Pandas DataFrame format

    :param file_path: Path to the CSV file. By default, movies_all_countries.csv is loaded
    :return: A Pandas DataFrame with the movies retrieved from the CSV file
    """
    assert isinstance(file_path, str), "The file path must be a string."
    assert os.path.exists(file_path), f"The file at {file_path} does not exist."

    df = pd.read_csv(file_path)
    return df


def load_movies_from_df_q1(movies_df: pd.DataFrame) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Reads movie data from a Pandas DataFrame and returns a list of movies and a dictionary
    mapping IMDb URLs to actor/director names.

    :param file_path: Path to the CSV file.
    :return: A tuple (movies_list, directors_thespians_url_to_name_dict)
    """

    df = movies_df.copy()
    validate_movie_data(df)
    # Clean monetary values
    for col in ['budget', 'domesticGross', 'worldwideGross']:
        df[col] = df[col].str.replace('$', '').str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # not all movies have values on grossing and the question aims at movies above a certain box office performance
    df = df.dropna(subset=['worldwideGross'])

    movies_list = []
    directors_thespians_url_to_name_dict = {}

    for _, row in df.iterrows():
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
        movies_list.append(movie)

        # Update dictionary mapping URLs to names
        for director in movie['directors']:
            directors_thespians_url_to_name_dict[director['url']] = director['name']
        for actor in movie['actors']:
            directors_thespians_url_to_name_dict[actor['url']] = actor['name']

    return movies_list, directors_thespians_url_to_name_dict

def process_movie_data_q2(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads movie data from a pandas DataFrame and returns an enriched DataFrame with extra info
    such as the award count of each movie and parsed/cast info such as numbers (budget and grossing)
    in string format into numeric format

    :param movies_df: the DataFrame retrieved from the CSV
    :return: the processed DataFrame with new data and converted content
    """

    df = movies_df.copy()
    validate_movie_data(df)
    # Clean monetary values
    for col in ['budget', 'domesticGross', 'worldwideGross']:
        df[col] = df[col].str.replace('$', '').str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert to millions
    df['worldwideGross_M'] = df['worldwideGross'] / 1000000

    # Convert imdbRating to numeric
    df['imdbRating'] = pd.to_numeric(df['imdbRating'], errors='coerce')

    # Count awards
    def count_awards(awards_str):
        try:
            awards_str = awards_str.replace("'", '"')
            if awards_str == "[]":
                return 0
            awards_list = json.loads(awards_str)
            return len(awards_list)
        except:
            if awards_str == "[]":
                return 0
            matches = re.findall(r"'[^']*'", awards_str)
            return len(matches)

    df['award_count'] = df['awards'].apply(count_awards)

    # Map countries to regions
    country_mapping = {
        'US': 'USA',
        'GB': 'UK',
        'ES': 'Spain'
    }
    df['region'] = df['country'].map(lambda x: country_mapping.get(x, 'Other'))

    return df

def generate_q2_chart_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = movies_df_q2.copy()
    analysis_df = analysis_df.dropna(subset=['award_count', 'worldwideGross_M', 'imdbRating'])

    # Group by number of awards
    awards_df = analysis_df.groupby('award_count').agg({
        'worldwideGross_M': ['mean', 'median', 'count'],
        'imdbRating': ['mean', 'count']
    }).reset_index()

    # Flatten column structure
    awards_df.columns = ['_'.join(col).strip('_') for col in awards_df.columns.values]

    # Rename columns for clarity
    awards_df = awards_df.rename(columns={
        'award_count': 'num_awards',
        'worldwideGross_M_mean': 'mean_box_office',
        'worldwideGross_M_median': 'median_box_office',
        'worldwideGross_M_count': 'movie_count',
        'imdbRating_mean': 'average_rating'
    })

    return analysis_df, awards_df

def load_movies_from_csv_directors_q3() -> None:
    """
    Reads director-level movie data from a CSV file and returns it as a pandas DataFrame.

    The CSV file is expected at the relative path './outputs/directors_info.csv'. Each row
    is initially read and converted into a list of dictionaries, then restructured into a DataFrame
    for further processing and analysis.

    :return: A pandas DataFrame containing the contents of the directors_info.csv file.
    """
    directors_info = load_movies_df_from_csv("outputs/directors_info.csv")
    directors_info = directors_info.to_dict("records")
    directors_info = pd.DataFrame(directors_info)
    return directors_info
    
def process_movie_data_q4(movies_df: pd.DataFrame) -> List[Dict]:
    """
    Reads movie data from a pandas DataFrame and returns an enriched DataFrame with extra info
    such as the award count of each movie and parsed/cast info such as numbers (budget and grossing)
    in string format into numeric format

    :param movies_df: the DataFrame retrieved from the CSV
    :return: the processed DataFrame contents as a list of dicts
    """

    data = movies_df.copy()
    validate_movie_data(data)
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

# Load and process: first, load the CSV, then adapt it to the specs of each question

movies_df = load_movies_df_from_csv()

# Prepare data for Q1
movies_list_q1, directors_thespians_url_to_name_dict = load_movies_from_df_q1(movies_df)

# Prepare data for Q2
movies_df_q2 = process_movie_data_q2(movies_df)
analysis_df, awards_grouped = generate_q2_chart_dfs()

# Prepare data for Q3
movies_df_q3 = load_movies_from_csv_directors_q3()

# Prepare data for Q4
movies_json_q4 = process_movie_data_q4(movies_df)

##############################################################
##################### End of Content Load ####################
##############################################################

##############################################################
#################### Functions used in Q1 ####################
##############################################################

# Generate Force Directed Graph
def generate_force_directed_graph(top_movies: List[Dict]) -> Tuple[go.Scatter, go.Scatter, nx.Graph]:
    """
    Generates a force-directed graph of directors and actors based on movie data.

    :param movies_list_q1: List of movie data where each movie contains directors and actors information.
    :param directors_thespians_url_to_name_dict: Dictionary mapping URLs to actor/director names.
    :return: A tuple containing edge trace, node trace, and the graph object.
    """

    G = nx.Graph()
    
    for movie in top_movies:
        # Iterate over each director in the movie
        for director in movie['directors']:
            directorURL = director['url']
            if directorURL not in G:
                G.add_node(directorURL, type="director", movies_count=1)
            else:
                G.nodes[directorURL]['movies_count'] += 1  # Increment movie count for the director

            # Iterate over each actor in the movie
            for actor in movie['actors']:
                actorURL = actor['url']
                if actorURL not in G:
                    G.add_node(actorURL, type="actor", movies_count=1)
                else:
                    G.nodes[actorURL]['movies_count'] += 1  # Increment movie count for the actor

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

    edge_trace = go.Scatter(
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

        # Use the directors_thespians_url_to_name_dict to translate URLs to names
        node_name = directors_thespians_url_to_name_dict.get(node, node)  # Use URL if not found in the dict
        nodeText.append(f"{node_name} ({G.nodes[node]['type']})<br><a href='{node}'>IMDb page</a>")
        nodeColor.append('blue' if G.nodes[node]['type'] == 'director' else 'orange')

        # Vary node size based on the number of movies (movies_count)
        nodeSize.append(5 + G.nodes[node]['movies_count'] * 5)  # Scale size (adjust multiplier if needed)

    node_trace = go.Scatter(
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

    return edge_trace, node_trace, G

##############################################################
################ End of Functions used in Q1 #################
##############################################################

##############################################################
#################### Functions used in Q2 ####################
##############################################################

##############################################################
################ End of Functions used in Q2 #################
##############################################################

##############################################################
#################### Functions used in Q3 ####################
##############################################################

def clean_money_column(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series containing monetary values by removing dollar signs and commas.

    This function uses a regular expression to strip out common currency formatting characters,
    making the values suitable for numeric conversion.

    :param series: A pandas Series containing string-formatted monetary values.
    :return: A pandas Series with the "$" and "," characters removed.
    """
    return series.replace("[\$,]", "", regex=True)


def normalize_roles(raw_roles: List[str]) -> List[str]:
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


def create_directors_profitability_bubble(top_n: int = 5) -> go.Figure:
    """
    Creates a bubble chart showing the top N directors ranked by average profitability.

    Profitability is calculated as the percentage gain from budget to worldwide gross
    for each movie, and directors are ranked by their average profitability across all their movies.
    Bubble size encodes the number of movies directed, and color encodes the total worldwide gross.

    :param top_n: The number of top directors to include in the chart (default is 5).
    :return: A Plotly Figure object representing the bubble chart.
    """
    df = movies_df_q3.copy()

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

    data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]

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
        title=f"Top {top_n} Directors by Profitability",
        xaxis_title="Total Worldwide Gross ($)",
        yaxis_title="Profitability (%)",
        template="plotly_white",
        height=600,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    return fig


def create_top_awarded_directors_treemap(top_n=5, award_filter="all") -> go.Figure:
    """
    Generates a treemap visualization of the top N most awarded directors.

    The function allows filtering by award outcome type: 'wins', 'nominations', or 'all'.
    If 'all' is selected, it shows both wins and nominations together, with separate counts.
    Each rectangle in the treemap represents a director, sized by total awards.

    :param top_n: The number of top directors to display (default is 5).
    :param award_filter: The type of award outcome to include: 'wins', 'nominations', or 'all'.
    :return: A Plotly Figure object representing the treemap of top awarded directors.
    """
    df = movies_df_q3.copy()

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
        award_counts = award_counts.sort_values(by="total", ascending=False).head(top_n)

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
            .head(top_n)
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
        title=f"Top {top_n} ({award_filter.capitalize()})",
        height=600,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig


def create_rating_vs_metascore_scatter() -> go.Figure:
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
    df = movies_df_q3.copy()

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


def create_roles_by_award_category(selected_award: str = "Best Director of the Year") -> go.Figure:
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
    df = movies_df_q3.copy()

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

##############################################################
#################### Functions used in Q4 ####################
##############################################################

# Transform data to a DataFrame for flexibility
def get_actor_profitability_df(country_filter: str = "All") -> pd.DataFrame:
    records = []
    for movie in movies_json_q4:
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
def plot_top_actors_profitability(top_n: int, country: str) -> go.Figure:
    df = get_actor_profitability_df(country)
    top_avg = (
        df.groupby("actor")["profitability"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig = go.Figure(go.Bar(x=top_avg.index, y=top_avg.values))
    fig.update_layout(
        title="Top Actors by Average Profitability",
        xaxis_title="Actor",
        yaxis_title="Profitability",
    )
    return fig


# Visual 2: Bottom N Actors by Profitability
def plot_bottom_actors_profitability(top_n: int, country: str) -> go.Figure:
    df = get_actor_profitability_df(country)

    # Compute average profitability per actor
    avg_profit = df.groupby("actor")["profitability"].mean()

    # Filter only actors with negative profitability
    negative_profit_actors = avg_profit[avg_profit < 0].sort_values().head(top_n)

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
def plot_profitability_distribution(country: str) -> go.Figure:
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

##############################################################
################ End of Functions used in Q4 #################
##############################################################

##############################################################
################# Initial Visual Representation ##############
##############################################################

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

# Visualization helper functions

def create_slider(slider_id: str, min_value: int, max_value: int, slider_step: int, default_value: int, slider_type: str) -> dcc.Slider or dcc.RangeSlider:
    slider_class = dcc.Slider if slider_type == 'slider' else dcc.RangeSlider
    marks_range_step = max(min_value, math.floor((max_value - 1) / 10)) if max(min_value, math.floor((max_value - 1) / 10)) > 0 else max(min_value, math.ceil((max_value - 1) / 10))
    return slider_class(
        id=slider_id,
        min=min_value,
        max=max_value,
        step=slider_step,
        value=default_value,
        marks={i: str(i) for i in range(min_value, max_value+1, marks_range_step)},
        tooltip={"placement": "bottom", "always_visible": True}
    )

def create_dropdown(dropdown_id: str, dropdown_options: List, default_value: str) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=dropdown_id,
        options=dropdown_options,
        value=default_value,
        clearable=False,
        style={"width": "100%", "max-width": "300px"},
    )

def create_loading(loading_id: str, *loading_children):
    return dcc.Loading(
        id=loading_id,
        type="circle",
        color="#00ff00",
        children=loading_children,
    )

# question visualization creator functions

def create_q1_control_panel() -> html.Div:
    max_value = len(movies_list_q1)
    slider_text = "Filter by Number of Movies:"
    min_value = 1
    slider_step = 1
    slider_id = "num-movies-slider-q1"
    default_value = 50
    slider_div_style = {'width': '100%', 'display': 'inline-block', 'padding': '20px'}
    q1_style = {'borderRadius': '10px', 'padding': '10px', 'margin': '10px 0', 'verticalAlign': 'top'}

    return html.Div([
        html.Div([
            html.Label(slider_text),
            create_slider(slider_id, min_value, max_value, slider_step, default_value, 'slider')
        ], style=slider_div_style
    )], style=q1_style)

def create_q1_content() -> html.Div:
    force_graph_id = "force-directed-graph"
    force_graph_title = "Director-Actor Connections"
    force_graph_style = {"width": "100%", "height": "700px", "padding-top": "0px"}
    loading_id = "loading-graph"

    return html.Div([
        html.H1(force_graph_title),
        create_q1_control_panel(),
        create_loading(loading_id,dcc.Graph(id=force_graph_id, style=force_graph_style)),
        dbc.Popover(
            [
                dbc.PopoverHeader(id="popover-header"),
                dbc.PopoverBody(id="popover-body")
            ],
            id="popover",
            is_open=False,
            target=force_graph_id,
            placement="top-start",
        ),
    ])
    
def create_q2_control_panel() -> html.Div:
    slider_max = int(awards_grouped['num_awards'].max())
    slider_min = 0
    slider_text = "Filter by Number of Awards:"
    slider_text_style = {'color': 'white', 'marginRight': '10px'}
    slider_id = 'awards-slider'
    slider_step = 1
    slider_value = [slider_min, min(5, slider_max)]
    slider_div_style = {'width': '70%', 'display': 'inline-block', 'padding': '20px'}
    metric_id = 'metric-selector'
    metric_text = "Box Office Metric:"
    metric_text_style = {'color': 'white', 'marginRight': '10px'}
    metric_selector_style = {'display': 'block', 'color': 'white'}
    metric_value = 'mean_box_office'
    metric_div_style = {'width': '20%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}
    metric_options = [
                    {'label': ' Mean', 'value': 'mean_box_office'},
                    {'label': ' Median', 'value': 'median_box_office'}
                ]
    q2_control_style = {'backgroundColor': '#1E1E1E', 'borderRadius': '10px', 'padding': '10px', 'margin': '10px 0'}

    return html.Div([
        html.Div([
            html.Label(slider_text, style=slider_text_style),
            create_slider(slider_id, slider_min, slider_max, slider_step, slider_value, 'range'),
        ], style=slider_div_style),

        html.Div([
            html.Label(metric_text, style=metric_text_style),
            dcc.RadioItems(
                id=metric_id,
                options=metric_options,
                value=metric_value,
                labelStyle=metric_selector_style
            )
        ], style=metric_div_style)
    ], style=q2_control_style)


def create_q2_content() -> html.Div:
    heading_text = "Awards vs Box Office Analysis"
    heading_style = {'textAlign': 'center', 'color': 'white', 'marginBottom': '20px'}
    loading_id = "loading-graphs"
    award_graph_id = 'award-boxoffice-graph'
    award_graph_style = {'height': '400px'}
    filt_movies_id = 'filtered-movies-graph'
    filt_movies_style = {'height': '500px'}
    charts_style = {'marginTop': '20px'}
    q2_style = {'backgroundColor': 'black', 'padding': '20px', 'fontFamily': 'Arial'}

    return html.Div([
        html.H1(heading_text, style=heading_style),
        create_q2_control_panel(),
        html.Div([
            create_loading(loading_id, dcc.Graph(id=award_graph_id, style=award_graph_style),
            html.Div([dcc.Graph(id=filt_movies_id, style=filt_movies_style)], style=charts_style))
        ])
    ], style=q2_style)

def create_q3_content() -> html.Div:
    heading_text = "Directors Dashboard"
    heading_style = style={"textAlign": "center"}
    subheading_one_text = "Profitability vs Worldwide Gross per Director"
    subheading_two_text = "Top Directors by Award Count"
    subheading_three_text = "Roles by Award Director Category"
    subheading_four_text = "IMDb Rating vs Metascore per Director"
    dir_slider_min = 1
    dir_slider_max = 100
    slider_step = 1
    slider_value = 5
    slider_id = "q3-top-n-slider"
    award_count_id = "award-type-dropdown"
    award_count_text = "Filter by Award Outcome:"
    award_count_options = [
                    {"label": "All", "value": "all"},
                    {"label": "Wins Only", "value": "wins"},
                    {"label": "Nominations Only", "value": "nominations"},
                ]
    award_count_value = "all"
    director_id = "award-category-dropdown"
    director_text = "Select Award Director Category:"
    director_value = "Best Director of the Year"
    director_options = [
                    {"label": category, "value": category}
                    for category, count in (
                        movies_df_q3
                        .assign(award_categories=lambda df: df["award_categories"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []))
                        .explode("award_categories")
                        .dropna(subset=["award_categories"])
                        .query("award_categories.str.contains('director', case=False)", engine='python')
                        .groupby("award_categories")["director_name"]
                        .nunique()
                        .items()
                    ) if count > 3
                ]
    profit_graph_id = 'directors-profitability-graph'
    director_awards_graph_id = 'top-awarded-directors'
    roles_graph_id = "roles-stacked-bar"
    ratings_graph_id = 'ratings-vs-metascore'

    app.layout = html.Div([
        html.H1(heading_text, heading_style),
        create_slider(slider_id, dir_slider_min, dir_slider_max, slider_step, slider_value, 'slider'),
        html.H2(subheading_one_text),
        dcc.Graph(id=profit_graph_id),

        html.H2(subheading_two_text),
        html.Div([
            html.Label(award_count_text),
            create_dropdown(award_count_id, award_count_options, award_count_value),
        ]),
        dcc.Graph(id=director_awards_graph_id),

        html.H2(subheading_three_text),
        html.Div([
            html.Label(director_text),
            create_dropdown(director_id, director_options, director_value),
            dcc.Graph(id=roles_graph_id)
        ]),

        html.H2(subheading_four_text),
        dcc.Graph(id=ratings_graph_id)
    ])
    return app.layout

def create_q4_control_panel() -> html.Div:
    the_slider_min = 1
    the_slider_max = 90
    slider_text = "Filter by Number of Actors:"
    slider_id = "top-n-slider"
    slider_step = 1
    slider_value = 10
    slider_div_style = {'width': '70%', 'display': 'inline-block', 'padding': '20px'}
    viz_text = "Select Visualization Type:"
    viz_id = "viz-type"
    viz_value = "top"
    viz_options = [
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
    ]
    viz_div_style = {"flex": "1", "margin-right": "10px"}
    country_text = "Filter by Country:"
    country_id = "country-dropdown"
    country_options = [{"label": "All Countries", "value": "All"}] + [
        {"label": country, "value": country}
        for country in sorted(
            set(movie["country"] for movie in movies_json_q4)
        )
    ]
    country_div_style = {"flex": "1"}
    country_value = "All"
    dropdowns_style = {'width': '20%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}
    q4_style = {'borderRadius': '10px', 'padding': '10px', 'margin': '10px 0', 'verticalAlign': 'top'}

    return html.Div([
        html.Div([
            html.Label(slider_text),
            create_slider(slider_id, the_slider_min, the_slider_max, slider_step, slider_value, 'slider'),
        ], style=slider_div_style),
        html.Div(
            [
                html.Div(
                    [
                        html.Label(viz_text),
                        create_dropdown(viz_id, viz_options, viz_value),
                    ],
                    style=viz_div_style,
                ),
                html.Div(
                    [
                        html.Label(country_text),
                        create_dropdown(country_id, country_options, country_value),
                    ],
                    style=country_div_style,
                ),
        ], style=dropdowns_style),
    ], style=q4_style)

def create_q4_content() -> html.Div:
    heading_text = "Actors Profitability Analysis"
    graph_id = "profitability-graph"
    return html.Div(
        [
            html.H1(heading_text),
            create_q4_control_panel(),
            dcc.Graph(id=graph_id),
        ]
    )


# Function to generate a simplified 2x2 view for Q1-Q4 placeholders
def create_overview() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(create_q1_content(), width=6),
            dbc.Col(create_q2_content(), width=6),
        ]),
        dbc.Row([
            dbc.Col(html.Div([
                html.H3("Top Directors by Profitability (Bubble Chart)"),
                dcc.Graph(figure=create_directors_profitability_bubble(top_n=5))
            ]), width=6),
            dbc.Col(create_q4_content(), width=6),
        ]),
    ])

##############################################################
############# End of Initial Visual Representation ###########
##############################################################

##############################################################
###################### Callbacks for Q1 ######################
##############################################################

@app.callback(
    [
        Output("force-directed-graph", "figure"),
        Output("popover", "is_open"),
        Output("popover-header", "children"),
        Output("popover-body", "children"),
        Output("popover-header", "style"),
    ],
    [
        Input('force-directed-graph', 'clickData'),
        Input('num-movies-slider-q1', 'value')
    ]
)


def update_graph_and_popover(clickData: Dict, slider_value: int) -> Tuple[Dict, bool, str, str, Dict]:
    """
    Updates the graph and popover based on the click event data.

    :param click_data: The data from the click event on the graph. This should contain information 
                      such as the node clicked and its associated tooltip.
    :return: A tuple containing:
             - The graph data (edges and nodes).
             - A boolean indicating if the popover is open.
             - The popover header text.
             - The popover body content (as Markdown).
             - The header style for the popover.
    """
    top_movies = sorted(movies_list_q1, key=lambda x: x['grossing'], reverse=True)[:slider_value]
    
    # Generate the force-directed graph
    edge_trace, node_trace, G = generate_force_directed_graph(top_movies)

    # Default popover state
    is_open = False
    header = "Node info error"
    body = "Try again later"
    header_style = {'color': 'white', 'backgroundColor': 'red', 'padding': '5px'}

    if clickData:
        try:
            node_tooltip = clickData['points'][0].get('text', None)
            if node_tooltip:
                # Regular expression to capture name, type, and IMDb URL
                pattern = r"(?P<name>.*?)\s\((?P<type>.*?)\)<br><a href='(?P<url>.*?)'>IMDb page</a>"
                match = re.search(pattern, node_tooltip)

                if match:
                    node_name = match.group("name")
                    node_type = match.group("type")
                    node_url = match.group("url")
                    movies_count = G.nodes[node_url]['movies_count']
                else:
                    node_name = "Name not found"
                    node_type = "Profession not found"
                    node_url = "IMDb URL not found"
                    movies_count = 0

                # Set popover content
                header = f"{node_name} ({node_type})"
                header_style = {'color': 'white', 'backgroundColor': 'blue', 'padding': '5px'} if node_type == "director" else {'color': 'black', 'backgroundColor': 'orange', 'padding': '5px'}
                body = dcc.Markdown(f'''
                    Number of movies: {movies_count}
                    [IMDb page]({node_url})
                ''', link_target="_blank",)
                is_open = True
        except Exception as e:
            print(f"Error processing click data: {e}")

    return {
        'data': [edge_trace, node_trace],
        'layout': go.Layout(
            title="Director-Actor Connections",
            hovermode="closest",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, b=0, t=0)
        ),
    }, is_open, header, body, header_style

##############################################################
################## End of Callbacks for Q1 ###################
##############################################################

##############################################################
###################### Callbacks for Q2 ######################
##############################################################

@app.callback(
    [Output('award-boxoffice-graph', 'figure'),
     Output('filtered-movies-graph', 'figure')],
    [Input('awards-slider', 'value'),
     Input('metric-selector', 'value')]
)
def update_graphs(awards_range, metric):
    # Filter data by selected range
    filtered_data = awards_grouped[
        (awards_grouped['num_awards'] >= awards_range[0]) & 
        (awards_grouped['num_awards'] <= awards_range[1])
    ]
    
    # Filter individual movies for the second chart
    filtered_movies = analysis_df[
        (analysis_df['award_count'] >= awards_range[0]) & 
        (analysis_df['award_count'] <= awards_range[1])
    ]
    
    # Create hover labels
    hover_text = [
        f"Awards: {row['num_awards']}<br>" +
        f"Mean Box Office: ${row['mean_box_office']:.1f}M<br>" +
        f"Median Box Office: ${row['median_box_office']:.1f}M<br>" + 
        f"IMDb Rating: {row['average_rating']:.1f}<br>" + 
        f"Number of Movies: {row['movie_count']}"
        for _, row in filtered_data.iterrows()
    ]
    
    # 1. Bar chart with trend line
    bar_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bars for box office
    bar_fig.add_trace(
        go.Bar(
            x=filtered_data['num_awards'],
            y=filtered_data[metric],
            text=filtered_data['movie_count'],
            textposition='auto',
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                color='rgba(50, 171, 96, 0.7)',
                line=dict(
                    color='rgba(50, 171, 96, 1.0)',
                    width=2
                )
            ),
            name='Box Office'
        )
    )
    
    # Line for rating
    bar_fig.add_trace(
        go.Scatter(
            x=filtered_data['num_awards'],
            y=filtered_data['average_rating'],
            mode='lines+markers',
            marker=dict(
                size=8,
                color='rgba(220, 57, 18, 0.8)',
                line=dict(
                    color='rgba(220, 57, 18, 1.0)',
                    width=1
                )
            ),
            line=dict(
                color='rgba(220, 57, 18, 0.8)',
                width=3
            ),
            name='IMDb Rating',
        ),
        secondary_y=True
    )
    
    # Update layout of first chart
    metric_title = "Mean" if metric == 'mean_box_office' else "Median"
    
    bar_fig.update_layout(
        title=f'{metric_title} Box Office and Rating by Number of Awards',
        template='plotly_dark',
        plot_bgcolor='rgba(30, 30, 30, 1)',
        paper_bgcolor='rgba(30, 30, 30, 1)',
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    # Update axes
    bar_fig.update_xaxes(
        title='Number of Awards',
        tickvals=filtered_data['num_awards'],
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    bar_fig.update_yaxes(
        title='Box Office (Millions $)',
        tickformat='$,.0f',
        gridcolor='rgba(255, 255, 255, 0.1)',
        secondary_y=False
    )
    
    bar_fig.update_yaxes(
        title='IMDb Rating',
        range=[min(filtered_data['average_rating']) - 0.5, max(filtered_data['average_rating']) + 0.5],
        tickformat='.1f',
        gridcolor='rgba(255, 255, 255, 0.1)',
        secondary_y=True
    )
    
    # 2. Filtered movies chart
    # Sort to highlight the most relevant
    if metric == 'mean_box_office':
        filtered_movies = filtered_movies.sort_values(by=['worldwideGross_M'], ascending=False)
    else:
       filtered_movies['combined_score'] = filtered_movies['worldwideGross_M'] * filtered_movies['imdbRating'] / 10
       filtered_movies = filtered_movies.sort_values(by=['combined_score'], ascending=False)

    # Limit to a manageable number of movies
    top_movies = filtered_movies.head(100)

    # Create scatter plot of movies
    movie_fig = go.Figure()

    # Group by region for color consistency
    for region in top_movies['region'].unique():
        region_movies = top_movies[top_movies['region'] == region]
    
        movie_fig.add_trace(go.Scatter(
            x=region_movies['imdbRating'],
            y=region_movies['worldwideGross_M'],
            mode='markers',
            marker=dict(
                size=region_movies['award_count'].apply(
                lambda x: np.interp(x, [0, analysis_df['award_count'].max()], [8, 30])  # Escala mejor
                ),
               color={
                    'USA': 'rgb(255, 215, 0)',        # amarillo
                    'UK': 'rgb(30, 144, 255)',        # azul
                    'Spain': 'rgb(220, 20, 60)',      # rojo
                    'Other': 'rgb(100, 100, 100)'     # gris
                }.get(region, 'rgb(100, 100, 100)'),
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name={
                'USA': 'Estados Unidos',
                'UK': 'Reino Unido',
                'Spain': 'España',
                'Other': 'Otro'
            }.get(region, region),
            text=[
                f"<b>{row['title']}</b><br>" +
                f"Awards: {int(row['award_count'])}<br>" +
                f"Rating: {row['imdbRating']}/10<br>" +
                f"Box Office: ${row['worldwideGross_M']:.1f}M<br>" +
                f"Country: {row['region']}"
                for _, row in region_movies.iterrows()
            ],
            hoverinfo='text',
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='white')),
        ))

    # Update layout of second chart
    movie_fig.update_layout(
        title=f'Movies with {awards_range[0]} to {awards_range[1]} Awards (Showing top {len(top_movies)} of {len(filtered_movies)} movies)',
        xaxis_title='IMDb Rating',
        yaxis_title='Box Office (Millions $)',
        template='plotly_dark',
        plot_bgcolor='rgba(30, 30, 30, 1)',
        paper_bgcolor='rgba(30, 30, 30, 1)',
        legend=dict(
            title='Region',
            x=0.01,
            y=0.99,
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode='closest'
    )

    # Handle log scale for y-axis if the range is large
    if top_movies['worldwideGross_M'].max() / (top_movies['worldwideGross_M'].min() + 0.1) > 100:
        movie_fig.update_yaxes(type='log', tickformat='$,.0f')
    else:
        movie_fig.update_yaxes(tickformat='$,.0f')

    movie_fig.update_xaxes(range=[min(top_movies['imdbRating']) - 0.5, 10])
    
    return bar_fig, movie_fig

##############################################################
################## End of Callbacks for Q2 ###################
##############################################################

##############################################################
###################### Callbacks for Q3 ######################
##############################################################
@app.callback(
    Output('directors-profitability-graph', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def update_q3_graph(top_n: int) -> go.Figure:
    try:
        return create_directors_profitability_bubble(top_n)
    except Exception as e:
        print(f"Error in updateQ3Graph: {e}")
        return go.Figure()


@app.callback(
    Output('top-awarded-directors', 'figure'),
    [Input('q3-top-n-slider', 'value'),
     Input('award-type-dropdown', 'value')]
)
def update_awards_graph(top_n, award_filter):
    return create_top_awarded_directors_treemap(top_n, award_filter)


@app.callback(
    Output("roles-stacked-bar", "figure"),
    [Input("award-category-dropdown", "value")]
)
def update_roles_graph(selected_award: str) -> go.Figure:
    return create_roles_by_award_category(selected_award or "Best Director of the Year")


@app.callback(
    Output('ratings-vs-metascore', 'figure'),
    [Input('q3-top-n-slider', 'value')]
)
def update_scatter_graph(_) -> go.Figure:
    return create_rating_vs_metascore_scatter()
##############################################################
################## End of Callbacks for Q3 ###################
##############################################################

##############################################################
###################### Callbacks for Q4 ######################
##############################################################

@app.callback(
    Output("profitability-graph", "figure"),
    Input("viz-type", "value"),
    Input("top-n-slider", "value"),
    Input("country-dropdown", "value"),
)
def update_q4_graph(viz_type: str, top_n: int, selected_country: str) -> go.Figure:
    if viz_type == "top":
        return plot_top_actors_profitability(top_n, selected_country)
    elif viz_type == "bottom":
        return plot_bottom_actors_profitability(top_n, selected_country)
    elif viz_type == "hist":
        return plot_profitability_distribution(selected_country)
    return go.Figure()


@app.callback(Output("slider-container", "style"), Input("viz-type", "value"))
def toggle_q4_slider_visibility(viz_type: str) -> Dict:
    if viz_type == "hist":  # Hide the slider for histogram
        return {"display": "none"}
    return {"display": "inline-block"}

##############################################################
################## End of Callbacks for Q4 ###################
##############################################################

##############################################################
####################### App Callbacks ########################
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
def update_content(overview_clicks: int, q1_clicks: int, q2_clicks: int, q3_clicks: int, q4_clicks: int) -> html.Div:
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_overview()

    button_id = ctx.triggered_id

    if button_id == "btn-overview":
        return create_overview()
    elif button_id == "btn-q1":
        return create_q1_content()
    elif button_id == "btn-q2":
        return create_q2_content()
    elif button_id == "btn-q3":
        return create_q3_content()
    elif button_id == "btn-q4":
        return create_q4_content()


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
def highlightButton(overview_clicks: int, q1_clicks: int, q2_clicks: int, q3_clicks: int, q4_clicks: int) -> List[Dict]:
    ctx = dash.callback_context
    button_id = ctx.triggered_id if ctx.triggered else None

    base_style = {'width': '10vw'}
    default_style = {**base_style, 'background-color': 'white', 'color': 'black'}
    highlighted_style = {**base_style, 'background-color': 'blue', 'color': 'white'}

    if button_id == "btn-overview":
        return highlighted_style, default_style, default_style, default_style, default_style
    elif button_id == "btn-q1":
        return default_style, highlighted_style, default_style, default_style, default_style
    elif button_id == "btn-q2":
        return default_style, default_style, highlighted_style, default_style, default_style
    elif button_id == "btn-q3":
        return default_style, default_style, default_style, highlighted_style, default_style
    elif button_id == "btn-q4":
        return default_style, default_style, default_style, default_style, highlighted_style

    return highlighted_style, default_style, default_style, default_style, default_style

##############################################################
################### End of App Callbacks #####################
##############################################################

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8051)
