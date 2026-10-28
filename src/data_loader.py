
from pathlib import Path

import pandas as pd


def load_movies(data_path=None):
    """
    Load movie metadata from a CSV file.
    """
    if data_path is None:
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / "data" / "raw" / "movielens_100k.csv"

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Movie dataset not found: {data_path}"
        )

    movies = pd.read_csv(data_path)

    required_columns = {
        "movie_id", "title", "year", "genres",
        "directors", "actors"
    }

    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    return movies
