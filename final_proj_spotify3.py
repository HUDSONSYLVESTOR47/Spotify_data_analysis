
import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pandas as pd
from yellowbrick.target import FeatureCorrelation
from tqdm import tqdm
import pyarrow.feather as feather
from scipy.optimize import curve_fit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# Function to display correlation plot
def corr_plot(ds):
    numeric_cols = ds.select_dtypes(include=['int64', 'float64'])
    if len(numeric_cols.columns) == 0:
        st.write("No numeric columns found in the dataset.")
        return
    corr = numeric_cols.corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu_r')
    fig.update_layout(title='Correlation Plot', width=800, height=600)
    st.plotly_chart(fig)

# Function to display popular tracks
def popular_tracks(ds, number):
    if 'track_name' not in ds.columns:
        st.write("The 'track_name' column is not present in the dataset.")
        return
    
    popular = ds.groupby("track_name")['popularity'].mean().sort_values(ascending=False).head(number)
    fig = px.bar(popular, orientation='h', labels={'value': 'Popularity', 'track_name': 'Tracks'}, title=f'Top {number} Popular Tracks')
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig)

# Function to display popular artists
def popular_artists(ds, number):
    popular = ds.groupby("artists")['popularity'].sum().sort_values(ascending=False)[:number]
    fig = px.bar(popular, orientation='h', labels={'value': 'Popularity', 'artists': 'Artists'}, title=f'Top {number} Artists with Popularity')
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig)

# Function to visualize popularity of artists over time
def visualise_popularity(ds, artist_name):
    artist_data = ds[ds['artists'] == artist_name]
    fig = px.line(artist_data, x='year', y='popularity', title=f"{artist_name} Popularity Over Time")
    st.plotly_chart(fig)

# Function to display distribution of a numerical feature
def feature_distribution(ds, feature_name):
    fig = px.histogram(ds, x=feature_name, title=f"Distribution of {feature_name}")
    st.plotly_chart(fig)

# Function to display box plot of a numerical feature
def feature_boxplot(ds, feature_name):
    fig = px.box(ds, y=feature_name, title=f"Box Plot of {feature_name}")
    st.plotly_chart(fig)

# Function to display scatter plot between two numerical features
def scatter_plot(ds, x_feature, y_feature):
    fig = px.scatter(ds, x=x_feature, y=y_feature, title=f"Scatter Plot between {x_feature} and {y_feature}")
    st.plotly_chart(fig)

# Function to display bar plot of a categorical feature
def categorical_barplot(ds, feature_name):
    fig = px.bar(ds, x=feature_name, title=f"Bar Plot of {feature_name}")
    st.plotly_chart(fig)

# Streamlit UI
st.header(':green[Spotify Song Recommendation System]')

st.write("(Mentor-Harshith Panchal)") 

# Read data
df = pd.read_csv(r'C:\Users\31syl\Desktop\fin\data_w_genres_p.csv')

st.markdown("<hr style='border: 2px solid #ff5733;'></h1>", unsafe_allow_html=True)

# First Sidebar
st.sidebar.title(":blue[Spotify Song Recommendation Data Analysis]")

add_sidebar = st.sidebar.selectbox(
    ":red[What would you like to explore?]",
    ("Correlation Plot", "Top Tracks", "Top Artists", "Spotify in Graphs")
)

if add_sidebar == "Correlation Plot":
    st.header(':orange[Correlation Plot]')
    corr_plot(df)
elif add_sidebar == "Top Tracks":
    st.header(':orange[Top Tracks]')
    number_of_tracks = st.slider('Select number of tracks to display', min_value=1, max_value=100, value=100)
    popular_tracks(df, number_of_tracks)
elif add_sidebar == "Top Artists":
    st.header(':orange[Top Artists]')
    number_of_artists = st.slider('Select number of artists to display', min_value=1, max_value=100, value=100)
    popular_artists(df, number_of_artists)
elif add_sidebar == "Spotify in Graphs":
    st.header(':orange[Spotify in Graphs]') 
    features = ["", "acousticness", "danceability", "energy", "speechiness", "liveness", "valence"]
    selected_feature = st.selectbox('Choose a feature to visualize', features)

    if selected_feature != "":
        st.write(f"You selected '{selected_feature}' feature.")

        if selected_feature in df.columns:
            if selected_feature == "acousticness":
                feature_distribution(df, 'acousticness')
            elif selected_feature == "danceability":
                feature_distribution(df, 'danceability')
            elif selected_feature == "speechiness":
                feature_distribution(df, 'speechiness')
            elif selected_feature == "liveness":
                feature_distribution(df, 'liveness')
            elif selected_feature == "valence":
                feature_distribution(df, 'valence') 
        else:
            st.write(f"'{selected_feature}' feature is not present in the dataset.")

st.markdown("<hr style='border: 2px solid #ff5733;'></h1>", unsafe_allow_html=True)

# Second Sidebar
st.sidebar.title(":blue[Spotify Song Recommendation Data Analysis]")

add_sidebar_2 = st.sidebar.selectbox(
    ":red[What would you like to explore?]",
    ("Distribution of Numerical Features", "Box Plot of Numerical Features", "Scatter Plot between Numerical Features", "Bar Plot of Categorical Features")
)

if add_sidebar_2 == "Distribution of Numerical Features":
    st.header(':orange[Distribution of Numerical Features]')
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_feature = st.selectbox('Choose a numerical feature to visualize', numeric_features)
    
    if selected_feature:
        feature_distribution(df, selected_feature)

elif add_sidebar_2 == "Box Plot of Numerical Features":
    st.header(':orange[Box Plot of Numerical Features]')
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_feature = st.selectbox('Choose a numerical feature to visualize', numeric_features)
    
    if selected_feature:
        feature_boxplot(df, selected_feature)

elif add_sidebar_2 == "Scatter Plot between Numerical Features":
    st.header(':orange[Scatter Plot between Numerical Features]')
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_x_feature = st.selectbox('Choose the X-axis feature', numeric_features)
    selected_y_feature = st.selectbox('Choose the Y-axis feature', numeric_features)
    
    if selected_x_feature and selected_y_feature:
        scatter_plot(df, selected_x_feature, selected_y_feature)

elif add_sidebar_2 == "Bar Plot of Categorical Features":
    st.header(':orange[Bar Plot of Categorical Features]')
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    selected_feature = st.selectbox('Choose a categorical feature to visualize', categorical_features)
    
    if selected_feature:
        categorical_barplot(df, selected_feature)



        

st.write("Created By- HUDSON SYLVESTOR")  
st.write("Batch--DW73DW74")
st.write("Year--2023-2024")                                                    

st.markdown("<hr style='border: 2px solid #ff5733;'></h1>", unsafe_allow_html=True)

st.write("Thank You") 