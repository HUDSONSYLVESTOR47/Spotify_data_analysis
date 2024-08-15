import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pandas as pd
from yellowbrick.target import FeatureCorrelation
from streamlit_option_menu import option_menu
from tqdm import tqdm
import pyarrow.feather as feather
from scipy.optimize import curve_fit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def plot_corr(spotify_df):
    numeric_columns = spotify_df.select_dtypes(include=['int64', 'float64'])
    if len(numeric_columns.columns) == 0:
        st.write("No numeric columns found in the dataset.")
        return
    correlation_matrix = numeric_columns.corr()
    correlation_fig = px.imshow(correlation_matrix, color_continuous_scale='Viridis')
    correlation_fig.update_layout(title='Correlation Plot', width=900, height=700)
    st.plotly_chart(correlation_fig)

# Function to display popular tracks
def display_popular_tracks(spotify_df, num_tracks):
    if 'track_name' not in spotify_df.columns:
        st.write("The 'track_name' column is not present in the dataset.")
        return
    
    popular_tracks = spotify_df.groupby("track_name")['popularity'].mean().sort_values(ascending=False).head(num_tracks)
    tracks_fig = px.bar(popular_tracks, orientation='v', labels={'value': 'Popularity', 'track_name': 'Tracks'}, title=f'Top {num_tracks} Popular Tracks')
    tracks_fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(tracks_fig)

# Function to display popular artists
def display_popular_artists(spotify_df, num_artists):
    popular_artists = spotify_df.groupby("artists")['popularity'].sum().sort_values(ascending=False)[:num_artists]
    artists_fig = px.bar(popular_artists, orientation='v', labels={'value': 'Popularity', 'artists': 'Artists'}, title=f'Top {num_artists} Artists with Popularity')
    artists_fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(artists_fig)

# Function to visualize popularity of artists over time
def visualize_artist_popularity(spotify_df, selected_artist):
    artist_df = spotify_df[spotify_df['artists'] == selected_artist]
    popularity_fig = px.line(artist_df, x='year', y='popularity', title=f"{selected_artist} Popularity Over Time")
    st.plotly_chart(popularity_fig)

# Function to display distribution of a numerical feature
def display_feature_distribution(spotify_df, feature_col):
    dist_fig = px.histogram(spotify_df, x=feature_col, title=f"Distribution of {feature_col}")
    st.plotly_chart(dist_fig)

# Function to display box plot of a numerical feature
def display_box_plot(spotify_df, feature_col):
    box_fig = px.box(spotify_df, y=feature_col, title=f"Box Plot of {feature_col}")
    st.plotly_chart(box_fig)

# Function to display scatter plot between two numerical features
def display_scatter_plot(spotify_df, x_col, y_col):
    scatter_fig = px.scatter(spotify_df, x=x_col, y=y_col, title=f"Scatter Plot between {x_col} and {y_col}")
    st.plotly_chart(scatter_fig)

# Function to display bar plot of a categorical feature
def display_categorical_barplot(spotify_df, feature_col):
    cat_bar_fig = px.bar(spotify_df, x=feature_col, title=f"Bar Plot of {feature_col}")
    st.plotly_chart(cat_bar_fig)


with st.sidebar:
    select= option_menu("MAIN MENU",["Home", "Project", "About"])

if select == "Home":
# Streamlit UI
    st.header(':blue[Spotify Song Recommendation System]')
    st.header(':red[About project]')
    st.write("(This dataset comprises Spotify tracks spanning across 125 diverse genres. Every track includes specific audio characteristics. Presented in CSV format, the data is organized in a table-like structure, ensuring swift loading.)")
    st.header(':red[Potential applications of this dataset include:]')
    st.write("")
    st.write("Creating a Recommendation System tailored to user preferences or inputs")
    st.write("Classifying tracks using their audio features and the range of genres they cover")
    st.write("Any other innovative use you can conceive. Suggestions and discussions are welcome!")
    st.header(':green[Mentor-Harshith Panchal]')
elif select == "Project":
# Read data
    spotify_df = pd.read_csv(r'C:\Users\31syl\Desktop\fin\data_w_genres_p.csv')

    st.markdown("<hr style='border: 2px solid #00cc00;'></h1>", unsafe_allow_html=True)

    # First Sidebar
    st.sidebar.title(":red[Spotify Song Recommendation Data Analysis]")

    explore_option = st.sidebar.radio(
        ":green[What would you like to explore?]",
        ("Correlation Plot", "Top Artists", "Spotify in Graphs")
    )

    if explore_option == "Correlation Plot":
        st.header(':purple[Correlation Plot]')
        plot_corr(spotify_df)
    elif explore_option == "Top Artists":
        st.header(':purple[Top Artists]')
        num_artists = st.slider('Select number of artists to display', min_value=1, max_value=100, value=100)
        display_popular_artists(spotify_df, num_artists)
    elif explore_option == "Spotify in Graphs":
        st.header(':purple[Spotify in Graphs]')
        feature_options = ["acousticness", "danceability", "speechiness", "liveness", "valence"]
        selected_feature = st.radio('Choose a feature to visualize', feature_options)

        if selected_feature != "":
            st.write(f"You selected '{selected_feature}' feature.")

            if selected_feature in spotify_df.columns:
                if selected_feature == "acousticness":
                    display_feature_distribution(spotify_df, 'acousticness')
                elif selected_feature == "danceability":
                    display_feature_distribution(spotify_df, 'danceability')
                elif selected_feature == "speechiness":
                    display_feature_distribution(spotify_df, 'speechiness')
                elif selected_feature == "liveness":
                    display_feature_distribution(spotify_df, 'liveness')
                elif selected_feature == "valence":
                    display_feature_distribution(spotify_df, 'valence')
            else:
                st.write(f"'{selected_feature}' feature is not present in the dataset.")

    st.markdown("<hr style='border: 2px solid #00cc00;'></h1>", unsafe_allow_html=True)

    # Second Sidebar
    st.sidebar.title(":red[Spotify Song Recommendation Data Analysis]")

    explore_option_2 = st.sidebar.radio(
        ":green[What would you like to explore?]",
        ("Distribution of Numerical Features", "Box Plot of Numerical Features", "Scatter Plot between Numerical Features", "Bar Plot of Categorical Features")
    )

    if explore_option_2 == "Distribution of Numerical Features":
        st.header(':purple[Distribution of Numerical Features]')
        num_features = spotify_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_feature = st.radio('Choose a numerical feature to visualize', num_features)
        
        if selected_feature:
            display_feature_distribution(spotify_df, selected_feature)

    elif explore_option_2 == "Box Plot of Numerical Features":
        st.header(':purple[Box Plot of Numerical Features]')
        num_features = spotify_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_feature = st.radio('Choose a numerical feature to visualize', num_features)
        
        if selected_feature:
            display_box_plot(spotify_df, selected_feature)

    elif explore_option_2 == "Scatter Plot between Numerical Features":
        st.header(':purple[Scatter Plot between Numerical Features]')
        num_features = spotify_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_x_feature = st.radio('Choose the X-axis feature', num_features)
        selected_y_feature = st.radio('Choose the Y-axis feature', num_features)
        
        if selected_x_feature and selected_y_feature:
            display_scatter_plot(spotify_df, selected_x_feature, selected_y_feature)

    elif explore_option_2 == "Bar Plot of Categorical Features":
        st.header(':purple[Bar Plot of Categorical Features]')
        cat_features = spotify_df.select_dtypes(include=['object']).columns.tolist()
        selected_feature = st.radio('Choose a categorical feature to visualize', cat_features)
        
        if selected_feature:
            display_categorical_barplot(spotify_df, selected_feature)  

elif select == "About":
    st.header(':red[Created by]')
    st.write("HUDSON SYLVESTOR") 
    st.header(':red[Batch]') 
    st.write("DW73DW74")                                                  

    st.markdown("<hr style='border: 2px solid #00cc00;'></h1>", unsafe_allow_html=True)

    st.write("Thank You")