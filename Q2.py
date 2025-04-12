# pip install plotly dash pandas

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import re

###############################################################
# 1. LOAD AND PROCESS DATA
###############################################################

def loadMoviesFromCSV(file_path='movies_all_countries.csv'):
    df = pd.read_csv(file_path)
    return df

def process_movie_data(df):
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


# Load and process
raw_movies_df = loadMoviesFromCSV()
movies_df = process_movie_data(raw_movies_df)

###############################################################
# 2. PREPARE DATA FOR ANALYSIS
###############################################################

# Prepare data for interactive chart
analysis_df = movies_df.copy()
analysis_df = analysis_df.dropna(subset=['award_count', 'worldwideGross_M', 'imdbRating'])

# Group by number of awards
awards_grouped = analysis_df.groupby('award_count').agg({
    'worldwideGross_M': ['mean', 'median', 'count'],
    'imdbRating': ['mean', 'count']
}).reset_index()

# Flatten column structure
awards_grouped.columns = ['_'.join(col).strip('_') for col in awards_grouped.columns.values]

# Rename columns for clarity
awards_grouped = awards_grouped.rename(columns={
    'award_count': 'num_awards',
    'worldwideGross_M_mean': 'mean_box_office',
    'worldwideGross_M_median': 'median_box_office',
    'worldwideGross_M_count': 'movie_count',
    'imdbRating_mean': 'average_rating'
})

###############################################################
# 3. DASH APP SETUP
###############################################################

app = dash.Dash(__name__, suppress_callback_exceptions=True)

###############################################################
# 4. DASH LAYOUT
###############################################################

app.layout = html.Div([
    html.H1("Awards vs Box Office Analysis", style={'textAlign': 'center', 'color': 'white', 'marginBottom': '20px'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Filter by Number of Awards:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.RangeSlider(
                id='awards-slider',
                min=0,
                max=int(awards_grouped['num_awards'].max()),
                step=1,
                marks={i: str(i) for i in range(0, int(awards_grouped['num_awards'].max()) + 1, 1)},
                value=[0, min(5, int(awards_grouped['num_awards'].max()))],  # Default to 0-5 awards or max if less
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'}),
        
        html.Div([
            html.Label("Box Office Metric:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='metric-selector',
                options=[
                    {'label': ' Mean', 'value': 'mean_box_office'},
                    {'label': ' Median', 'value': 'median_box_office'}
                ],
                value='mean_box_office',
                labelStyle={'display': 'block', 'color': 'white'}
            )
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'})
    ], style={'backgroundColor': '#1E1E1E', 'borderRadius': '10px', 'padding': '10px', 'margin': '10px 0'}),
    
    # Charts
    html.Div([
        dcc.Loading(
            id="loading-graphs",
            children=[
                dcc.Graph(id='award-boxoffice-graph', style={'height': '400px'}),
                html.Div([
                    dcc.Graph(id='filtered-movies-graph', style={'height': '500px'})
                ], style={'marginTop': '20px'})
            ],
            type="circle",
            color="#00ff00"
        )
    ])
], style={'backgroundColor': 'black', 'padding': '20px', 'fontFamily': 'Arial'})

###############################################################
# 5. CALLBACKS
###############################################################

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
        # For median, show those with a balance of rating and box office
        filtered_movies['combined_score'] = filtered_movies['worldwideGross_M'] * filtered_movies['imdbRating'] / 10
        filtered_movies = filtered_movies.sort_values(by=['combined_score'], ascending=False)
    
    # Limit to a manageable number of movies
    top_movies = filtered_movies.head(100)
    
    # FIX: Create scatter plot of movies with explicit hover template for each point
    movie_fig = go.Figure()
    
    # Group by region for color consistency
    for region in top_movies['region'].unique():
        region_movies = top_movies[top_movies['region'] == region]
        
        movie_fig.add_trace(go.Scatter(
            x=region_movies['imdbRating'],
            y=region_movies['worldwideGross_M'],
            mode='markers',
            marker=dict(
                size=region_movies['award_count'].apply(lambda x: min(30, max(10, x * 5))),  # Scale size better
                color={
                    'USA': 'rgb(65, 105, 225)',
                    'UK': 'rgb(147, 112, 219)',
                    'Spain': 'rgb(218, 165, 32)',
                    'Other': 'rgb(100, 100, 100)'
                }.get(region, 'rgb(100, 100, 100)'),
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name=region,
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
    
    # Add labels for featured movies
    # Identify movies to label (top performers in both box office and awards)
    highlight_movies = pd.concat([
        top_movies.nlargest(5, 'worldwideGross_M'),
        top_movies.nlargest(5, 'award_count')
    ]).drop_duplicates()
    
    for i, row in highlight_movies.iterrows():
        movie_fig.add_annotation(
            x=row['imdbRating'],
            y=row['worldwideGross_M'],
            text=row['title'],
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="white",
            font=dict(size=9, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            ax=20,
            ay=-30
        )
    
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

###############################################################
# 6. RUN APP
###############################################################

if __name__ == "__main__":
    app.run(debug=True)