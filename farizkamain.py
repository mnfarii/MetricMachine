import numpy as np
from sklearn.neighbors import NearestNeighbors

user_movie_ratings = np.array([
    [5, 4, 0, 0, 2, 0],
    [0, 0, 5, 4, 0, 3],
    [3, 0, 0, 0, 4, 5],
    [0, 2, 0, 5, 4, 0]
])

k = 2
model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
model.fit(user_movie_ratings)

user_index = 0
user_ratings = user_movie_ratings[user_index, :].reshape(1, -1)
distances, indices = model.kneighbors(user_ratings, n_neighbors=k+1) 
recommended_movies = [i for i in range(user_movie_ratings.shape[1]) if i not in indices.flatten()]

print(f"Рекомендуемые фильмы для пользователя {user_index}: {recommended_movies}")