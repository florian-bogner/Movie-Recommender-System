# print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))

import streamlit as st
import pandas as pd
import sklearn

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
links = pd.read_csv('data/links.csv')
tags = pd.read_csv('data/tags.csv')

st.title('Recommend me some movie!')

st.markdown('You like to know movies similar to flicks you like? Then select your favorite movie here and the number of other movies you are interested in:')

col_one_list = movies['title'].tolist()
title_ = st.selectbox('Select a movie you like:', col_one_list)

number = st.slider('Select the number of movies you want to be displayed:', 1, 100)
#st.write("I'm ", age, 'years old')

# 1. Popularity-based
def pop_movies(number):
    ratings_new = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    ratings_new['rating_count'] = ratings.groupby('movieId')['rating'].count()
    # 1. initialize the transformer (optionally, set parameters)
    my_min_max = StandardScaler()
    # 2. fit the transformer to the data
    my_min_max.fit(ratings_new)
    # 3. use the transformer to transform the data
    min_max_scaled_ratings_new = my_min_max.transform(ratings_new)
    # 4. reconvert the transformed data back to a DataFrame
    df_min_max_scaled_ratings_new = pd.DataFrame(min_max_scaled_ratings_new,
                 index=ratings_new.index,
                 columns=ratings_new.columns)
    df_min_max_scaled_ratings_new['score'] = df_min_max_scaled_ratings_new['rating'] * df_min_max_scaled_ratings_new['rating_count']
    top_number = df_min_max_scaled_ratings_new.sort_values(by = 'score', ascending = False).head(number).reset_index()
    top_number_with_titles = pd.merge(movies, top_number, how='inner', on = 'movieId')
    st.dataframe(top_number_with_titles.sort_values(by = 'score', ascending = False)['title'].to_list())

# 2. Item-based
def item_movies(title_, number):
    movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
    top_popular_movieID = int(movies['movieId'].loc[movies['title'] == title_])
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
    
# 3. User-based
def user_recom(user_id, n):
    movie_titles = movies[['movieId', 'title']]
    users_items = pd.pivot_table(data=ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')
    users_items.fillna(0, inplace=True)
    user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)
    weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))
    not_visited_restaurants = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
    weighted_averages = pd.DataFrame(not_visited_restaurants.T.dot(weights), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movie_titles, left_index=True, right_on="movieId")
    st.dataframe(recommendations.sort_values("predicted_rating", ascending=False).head(n)['title'].to_list())

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose your recommender",
        ("Popularity-based", "Item-based", "User-based")
    )

if add_radio == "Popularity-based":
    pop_movies(number)
elif add_radio == "Item-based":
	item_movies(title_, number)
else add_radio == "User-based":
    user_recom(user_id, n)