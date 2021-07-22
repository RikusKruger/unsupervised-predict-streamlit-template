"""
    Content-based filtering for item recommendation.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.
    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!
    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.
    ---------------------------------------------------------------------
    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.
"""

# Script dependencies
from operator import truediv
import os
from numpy.lib.npyio import load
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.snowball import SnowballStemmer

# # Importing data
df_movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')
# df_movies.dropna(inplace=True)



def storeData(db, filename):
    dbfile = open(f'{filename}', 'ab')
    # source, destination
    pickle.dump(db, dbfile)                     
    dbfile.close()
    return print("Stored")
  
def loadData(filename):
    # for reading also binary mode is important
    dbfile = open(f'{filename}', 'rb')     
    db = pickle.load(dbfile)
    dbfile.close()
    return db

# # data_1 = data_preprocessing(0, 31211)
# data_1 = loadData('data_1')
# print("data_1 loaded\n\n")

# # data_2 = data_preprocessing(31211, 62423)

# data_2 = loadData('data_2')
# print("data_2 loaded\n\n")


# count_vec = CountVectorizer()

# count_matrix_1 = count_vec.fit_transform(data_1['combined_features'])
# cosine_sim_1 = cosine_similarity(count_matrix_1, count_matrix_1)
# cosine_sim_1 = loadData('cosine_sim_1')
# print("cosine_sim_1 loaded\n\n")


# # count_matrix_2 = count_vec.fit_transform(data_2['combined_features'])
# # cosine_sim_2 = cosine_similarity(count_matrix_2, count_matrix_2)
# cosine_sim_2 = loadData('cosine_sim_2')
# print("cosine_sim_2 loaded\n\n")


def data_preprocessing(subset_size=28000):
    """Prepare data for use within Content filtering algorithm.
    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.
    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.
    """
    imdb = df_imdb[['movieId','title_cast','director', 'plot_keywords']]
    merge = df_movies.merge(imdb[['movieId','title_cast','director', 'plot_keywords']], on='movieId', how='left')

    # Convert data types to string in order to do string manipulation
    merge['title_cast'] = merge.title_cast.astype(str)
    merge['plot_keywords'] = merge.plot_keywords.astype(str)
    # merge['genres'] = merge.genres.astype(str)
    merge['director'] = merge.director.astype(str)

    df_cbr = pd.DataFrame()

    # handle cast
    # clean directors and title_cast column
    # remove spaces and "|"
    #convert title cast back to string and remove commas
    df_cbr['cast'] = merge['title_cast'].apply(lambda x: x.split('|'))

    df_cbr['cast'] = df_cbr['cast'].apply(lambda x: ','.join(map(str, x)))
    df_cbr['cast'] = df_cbr['cast'].replace(',',' ', regex=True)

    df_cbr['director'] = merge['director'].apply(lambda x: "".join(x.lower() for x in x.split()))

    # handle keyword
    stemmer = SnowballStemmer('english')
    df_cbr['keyword'] = merge['plot_keywords'].apply(lambda x: x.split('|'))

    df_cbr['keyword'] = df_cbr['keyword'].map(lambda x: [i.split(' ') for i in x])
    df_cbr['keyword'] = df_cbr['keyword'].map(lambda x: [item for sublist in x for item in sublist])


    # df_cbr['keyword'] = df_cbr['keyword'].apply(lambda x: [eval(i) for i in x])
    df_cbr['keyword'] = df_cbr['keyword'].apply(lambda x: [stemmer.stem(i) for i in x])
    df_cbr['keyword'] = df_cbr['keyword'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    # handle genres


    df_cbr['genre'] = merge['genres'].map(lambda x: x.lower().split('|'))
    df_cbr['genre'] = df_cbr['genre'].apply(lambda x: " ".join(x))

    # handle title
    df_cbr['title'] = merge['title']

    # merge all
    df_cbr['mixed'] = df_cbr['keyword'].astype(str) + df_cbr['cast'].astype(str) + df_cbr['genre'].astype(str) + df_cbr['director'].astype(str)

    df_cbr_1 = df_cbr.iloc[:subset_size]
    df_cbr_2 = df_cbr.iloc[subset_size:]


    return df_cbr_1, df_cbr_2


df_cbr_1, df_cbr_2 = data_preprocessing()
# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.
    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """

    store = False
    recommended = []
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix_1 = count.fit_transform(df_cbr_1['mixed'])
    count_matrix_2 = count.fit_transform(df_cbr_2['mixed'])
    if store == False:
        return movie_list

    print("fit_transform done")
    # count_matrix.todense()

    if store:
        # cosine_sim_1 = cosine_similarity(count_matrix_1, count_matrix_1)
        # storeData(cosine_sim_1, 'cosine_sim_1')
        # print("cosine_sim_1 stored\n\n")
       
        cosine_sim_1 = cosine_similarity(count_matrix_1)
        storeData(cosine_sim_1, 'cosine_sim_1')
        print("cosine_sim_1 stored\n\n")


        cosine_sim_2 = cosine_similarity(count_matrix_2)
        storeData(cosine_sim_2, 'cosine_sim_2')
        print("cosine_sim_2 stored\n\n")

        indices_1 = pd.Series(df_cbr_1.index, index=df_cbr_1['title'])
        storeData(indices_1, 'indices_1')
        print("indices_1 stored\n\n")

        indices_2 = pd.Series(df_cbr_2.index, index=df_cbr_2['title'])
        storeData(indices_2, 'indices_2')
        print("indices_2 stored\n\n")

        titles_1 = df_cbr_1['title']
        storeData(titles_1, 'titles_1')
        print("titles_1 stored\n\n")

        titles_2 = df_cbr_2['title']
        storeData(titles_2, 'titles_2')
        print("titles_2 stored\n\n")

    else:
        
        cosine_sim_1 = loadData('cosine_sim_1')

        cosine_sim_2 = loadData('cosine_sim_2')

        indices_1 = loadData('indices_1')

        indices_2 = loadData('indices_2')

        titles_1 = loadData('titles_1')

        titles_2 = loadData('titles_2')



    for title in movie_list:
        try:
            idx = indices_1[title]
            similarity_scores = list(enumerate(cosine_sim_1[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[1:21]
            movie_indices = [i[0] for i in similarity_scores]
            recommended.append(titles_1.iloc[movie_indices[:4]])

        except:
            idx = indices_2[title]
            similarity_scores = list(enumerate(cosine_sim_2[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[1:21]
            movie_indices = [i[0] for i in similarity_scores]
            recommended.append(titles_2.iloc[movie_indices[:4]])
    print(recommended[:top_n])

    return recommended




    # # Initializing the empty list of recommended movies
    # recommended_movies = []


    # # Getting the index of the movie that matches the title
    # # print(f"I am printing here !!!!!!!!!!!!!!!!: \n\n{str(data[data['title'] == 'Toy Story (1995)']['movieId'][0])}")
    # # print(f"I am printing here !!!!!!!!!!!!!!!!: \n\n{movie_list[0]}")

    # try:
    #     idx_1 = data_1[data_1['title'] == movie_list[0]]['movieId'].index[0]
    #     rank_1 = cosine_sim_1[idx_1]
    # except:
    #     idx_1 = data_2[data_2['title'] == movie_list[0]]['movieId'].index[0]
    #     rank_1 = cosine_sim_2[idx_1]

    # try:
    #     idx_2 = data_1[data_1['title'] == movie_list[1]]['movieId'].index[0]
    #     rank_2 = cosine_sim_1[idx_2]
    # except:
    #     idx_2 = data_2[data_2['title'] == movie_list[1]]['movieId'].index[0]
    #     rank_2 = cosine_sim_2[idx_2]

    # try:
    #     idx_3 = data_1[data_1['title'] == movie_list[2]]['movieId'].index[0]
    #     rank_3 = cosine_sim_1[idx_3]
    # except:
    #     idx_3 = data_2[data_2['title'] == movie_list[2]]['movieId'].index[0]
    #     rank_3 = cosine_sim_2[idx_3]


    # # Creating a Series with the similarity scores in descending order
    # # Calculating the scores
    # score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    # score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    # score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
     
    # # Getting the indexes of the 10 most similar movies
    # listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)

    # # Store movie names
    # recommended_movies = []
    # # Appending the names of movies
    # top_50_indexes = list(listings.iloc[1:50].index)
    # # Removing chosen movies
    # top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # for i in top_indexes[:top_n]:
    #     recommended_movies.append(list(df_movies['title'])[i])
    # return recommended_movies