
import pandas as pd
import pytest

from src.data_loader import load_movies


def test_load_movies_from_csv(tmp_path):
    # Create a small temporary dataset
    sample = pd.DataFrame({
        "movie_id": [1, 2],
        "title": ["Toy Story", "Jumanji"],
        "year": [1995, 1995],
        "genres": ["Animation Comedy", "Adventure Fantasy"],
        "directors": ["John Lasseter", "Joe Johnston"],
        "actors": ["Tom Hanks", "Robin Williams"]
    })

    csv_path = tmp_path / "movies.csv"
    sample.to_csv(csv_path, index=False)

    # Load the temporary dataset
    result = load_movies(csv_path)

    assert result.shape == (2, 6)
    assert result["title"].tolist() == ["Toy Story", "Jumanji"]


def test_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Movie dataset not found"):
        load_movies(missing_path)


def test_missing_columns_raise_error(tmp_path):
    sample = pd.DataFrame({
        "movie_id": [1],
        "title": ["Toy Story"]
    })

    csv_path = tmp_path / "incomplete.csv"
    sample.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Dataset is missing columns"):
        load_movies(csv_path)
