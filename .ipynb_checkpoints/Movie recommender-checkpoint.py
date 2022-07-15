# print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))

import streamlit as st
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('../ml-latest-small/movies.csv')
ratings = pd.read_csv('../ml-latest-small/ratings.csv')
links = pd.read_csv('../ml-latest-small/links.csv')
tags = pd.read_csv('../ml-latest-small/tags.csv')

st.title('Recommend me some movie!')

st.markdown('You like to know movies similar to flicks you like? Then select your favorite movie here and the number of other movies you are interested in:')

col_one_list = movies['title'].tolist()
title = st.selectbox('Select a movie you like:', col_one_list)

number = st.slider('Select the number of movies you want to be displayed:', 1, 100)
#st.write("I'm ", age, 'years old')

def top_movies(title, number):
    movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
    top_popular_movieID = int(movies['movieId'].loc[movies['title'] == title])
    top_ratings = movies_crosstab[top_popular_movieID]
    top_ratings = top_ratings[top_ratings.notna()] # not inplace? 
    similar_to_movie_id = movies_crosstab.corrwith(top_ratings)
    corr_movie_id = pd.DataFrame(similar_to_movie_id, columns=['PearsonR'])
    corr_movie_id.dropna(inplace=True)
    rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    rating['rating_count'] = ratings.groupby('movieId')['rating'].count()
    movie_id_corr_summary = corr_movie_id.join(rating['rating_count'])
    movie_id_corr_summary.drop(top_popular_movieID, inplace=True)
    top_number = movie_id_corr_summary[movie_id_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(number)
    top_number_with_titles = pd.merge(movies, top_number, how='inner', on = 'movieId')
    st.dataframe(top_number_with_titles['title'].to_list())
    
top_movies(title, number)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose your recommender",
        ("Popularity-based", "Item-based", "User-based")
    )