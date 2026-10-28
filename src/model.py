
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def build_similarity_model(
    movies,
    genre_weight=2.0,
    director_weight=1.0,
    actor_weight=1.0,
):
    """Build a content-based movie similarity model."""

    weights = [genre_weight, director_weight, actor_weight]
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("Weights must be non-negative, with at least one positive")

    movies = movies.copy()

    for column in ["genres", "directors", "actors"]:
        movies[column] = movies[column].fillna("").str.lower()

    genre_vectorizer = TfidfVectorizer()
    director_vectorizer = TfidfVectorizer()
    actor_vectorizer = TfidfVectorizer()

    genre_matrix = genre_vectorizer.fit_transform(movies["genres"])
    director_matrix = director_vectorizer.fit_transform(movies["directors"])
    actor_matrix = actor_vectorizer.fit_transform(movies["actors"])

    combined_matrix = hstack([
        genre_matrix * genre_weight,
        director_matrix * director_weight,
        actor_matrix * actor_weight,
    ]).tocsr()

    similarity_matrix = cosine_similarity(combined_matrix)

    return combined_matrix, similarity_matrix
