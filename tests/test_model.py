
import numpy as np
import pandas as pd

from src.model import build_similarity_model


def test_build_similarity_model_shapes():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Toy Story", "Jumanji", "Heat"],
        "genres": [
            "Animation Comedy",
            "Adventure Fantasy",
            "Action Crime"
        ],
        "directors": [
            "John Lasseter",
            "Joe Johnston",
            "Michael Mann"
        ],
        "actors": [
            "Tom Hanks",
            "Robin Williams",
            "Al Pacino"
        ]
    })

    tfidf_matrix, similarity_matrix = build_similarity_model(movies)

    assert tfidf_matrix.shape[0] == 3
    assert similarity_matrix.shape == (3, 3)


def test_similarity_matrix_is_symmetric():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Comedy", "Action", "Drama"],
        "directors": ["Director A", "Director B", "Director C"],
        "actors": ["Actor A", "Actor B", "Actor C"]
    })

    _, similarity_matrix = build_similarity_model(movies)

    assert np.allclose(similarity_matrix, similarity_matrix.T)


def test_missing_text_values_are_supported():
    movies = pd.DataFrame({
        "movie_id": [1, 2],
        "title": ["Movie A", "Movie B"],
        "genres": ["Comedy", None],
        "directors": [None, "Director B"],
        "actors": ["Actor A", None]
    })

    tfidf_matrix, similarity_matrix = build_similarity_model(movies)

    assert tfidf_matrix.shape[0] == 2
    assert similarity_matrix.shape == (2, 2)
    assert np.isfinite(similarity_matrix).all()
