
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_model(movies):
    """
    Build a content-based movie similarity model.

    Returns:
        tfidf_matrix: Numerical representation of movie metadata.
        similarity_matrix: Similarity scores between all movies.
    """
    movies = movies.copy()

    # Replace missing text values with empty strings
    for column in ["directors", "actors", "genres"]:
        movies[column] = movies[column].fillna("")

    # Combine movie metadata into a single text profile
    movies["combined_features"] = (
        "genres " + movies["genres"] + " "
        "director " + movies["directors"] + " "
        "actors " + movies["actors"]
    ).str.lower()

    # Convert text profiles into TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(
        movies["combined_features"]
    )

    # Calculate similarity between every pair of movies
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return tfidf_matrix, similarity_matrix
