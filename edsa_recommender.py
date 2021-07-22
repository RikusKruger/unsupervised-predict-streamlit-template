"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from re import I
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# web scraping imports
import requests
from PIL import Image
import bs4

import urllib.request
import re

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
movies_df = pd.read_csv('resources/data/movies.csv')
imdb_df = pd.read_csv('resources/data/imdb_data.csv')
dates = []

for title in movies_df['title']:
    if title[-1] == " ":
        year = title[-6: -2]
        try:
            dates.append(int(year))
        except:
            dates.append(9999)
    else:
        year = title[-5: -1]
        try:
            dates.append(int(year))
        except:
            dates.append(9999)
movies_df['release_year'] = dates


genres_df = pd.DataFrame(movies_df['genres'].
                      str.split("|").
                      tolist(),
                      index=movies_df['movieId']).stack()

genres_df = genres_df.reset_index([0, 'movieId'])
genres_df.columns = ['movieId', 'Genre']
genre_list = genres_df["Genre"].unique()

year_list = sorted(movies_df[movies_df['release_year'] > 1000]['release_year'].unique(), reverse=True)
year_list.remove(9999)


imdb_df['title_cast'] = imdb_df['title_cast'].str.split('|')

# title_cast = title_cast.reset_index([0, 'movieId'])
# title_cast.columns = ['movieId', 'title_cast']


filter_list = ['Content Based Filtering', 'Collaborative Based Filtering']

 
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "Popcorn"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    st.set_page_config(layout="wide")
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")



    if page_selection == "Popcorn":
        st.image('resources/videos/popcorn_gif_2.gif',use_column_width=True)

        st.write('### Enter Your Three Favorite Movies')
        col1, col2, col3 = st.beta_columns(3)

        col_dict = {f'col_0': col1,
                    f'col_1': col2,
                    f'col_2': col3}

        movies_titles = [None] * 3
        movies_ids = [None] * 3
        genres = [None] * 3
        years = [None] * 3
        

        for col, key in enumerate(col_dict.keys()):
            #Get user 
            genres[col] = col_dict[key].selectbox(f'Movie {col + 1} genre', genre_list)
            years[col] = col_dict[key].selectbox(f'Movie {col + 1} year', year_list)

            genre_movie_id_temp = genres_df[genres_df["Genre"] == genres[col]]["movieId"]
            genre_movies_temp = movies_df[movies_df['movieId'].isin(genre_movie_id_temp)]
            genre_year_movies_temp = genre_movies_temp[genre_movies_temp['release_year'] == years[col]]

            movies_titles[col] = col_dict[key].selectbox(f'Select movie {col + 1}:', genre_year_movies_temp['title'].to_list())
            movies_ids[col] = movies_df[movies_df['title'] == movies_titles[col]]["movieId"].values



        if col1.button("Recommend"):
            # try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = collab_model(movies_titles,
                                                        top_n=10)
                # st.write(top_recommendations)

                pred_ids = movies_df[movies_df["title"].isin(top_recommendations)]
                # st.write(f"preds = {pred_ids['movieId'].values}")
                new_col1, new_col2, new_col3, new_col4, new_col5 = st.beta_columns(5)

                new_col_dict = {f'col_0': new_col1,
                                f'col_1': new_col2,
                                f'col_2': new_col3,
                                f'col_3': new_col4,
                                f'col_4': new_col5}

                for i, id in enumerate(pred_ids[:5].values):
                    
                    # new_col_dict[f'col_{i}'].write(id)
                    title_dict = {f'movie_{id[0]}_title': id[1]}
                    # st.write(id[0])

                    # try:
                    director_dict = {f'movie_{id[0]}_director': imdb_df[imdb_df["movieId"] == id[0]]["director"]}
                    # st.markdown(f'director dict = {director_dict}')
                    cast_dict = {f'movie_{id[0]}_cast': imdb_df[imdb_df["movieId"] == id[0]]["title_cast"]}

                    search_keyword = title_dict[f'movie_{id[0]}_title'].replace(' ', '+')
                    html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword + "+trailer")
                    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

                    new_col_dict[f'col_{i}'].title(title_dict[f'movie_{id[0]}_title'])
                    new_col_dict[f'col_{i}'].video("https://www.youtube.com/watch?v=" + video_ids[0])
                    new_col_dict[f'col_{i}'].subheader("Director:")
                    new_col_dict[f'col_{i}'].write(f" - {director_dict[f'movie_{id[0]}_director'].values[0]}")

                    new_col_dict[f'col_{i}'].subheader("Cast:")
                    for member in cast_dict[f'movie_{id[0]}_cast']:
                        for name in member:
                            new_col_dict[f'col_{i}'].write(f" - {name}\n")   


if __name__ == '__main__':
    main()