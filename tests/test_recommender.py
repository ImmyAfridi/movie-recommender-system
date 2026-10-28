
import numpy as np
import pandas as pd
import pytest

from src.recommender import recommend_movies


@pytest.fixture
def sample_data():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["toy story", "jumanji", "grumpier old men"],
        "year": [1995, 1995, 1995],
        "genres": [
            "Animation Adventure Comedy",
            "Adventure Fantasy",
            "Comedy Romance"
        ]
    })

    similarity_matrix = np.array([
        [1.00, 0.80, 0.20],
        [0.80, 1.00, 0.30],
        [0.20, 0.30, 1.00]
    ])

    return movies, similarity_matrix


def test_returns_most_similar_movie(sample_data):
    movies, similarity_matrix = sample_data

    result = recommend_movies(
        "toy story", movies, similarity_matrix, top_n=1
    )

    assert len(result) == 1
    assert result.iloc[0]["title"] == "Jumanji"
    assert result.iloc[0]["similarity"] == 0.80


def test_excludes_selected_movie(sample_data):
    movies, similarity_matrix = sample_data

    result = recommend_movies(
        "toy story", movies, similarity_matrix
    )

    assert "toy story" not in result["title"].tolist()


def test_title_matching_is_case_insensitive(sample_data):
    movies, similarity_matrix = sample_data

    result = recommend_movies(
        "TOY STORY", movies, similarity_matrix, top_n=1
    )

    assert result.iloc[0]["title"] == "Jumanji"


def test_unknown_title_raises_error(sample_data):
    movies, similarity_matrix = sample_data

    with pytest.raises(ValueError, match="Movie not found"):
        recommend_movies(
            "unknown movie", movies, similarity_matrix
        )
def test_invalid_top_n_raises_error(sample_data):
    movies, similarity_matrix = sample_data

    with pytest.raises(
        ValueError,
        match="top_n must be a positive integer"
    ):
        recommend_movies(
            "toy story",
            movies,
            similarity_matrix,
            top_n=0
        )