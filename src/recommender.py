
import pandas as pd


def recommend_movies(
    title,
    movies,
    similarity_matrix,
    top_n=10
):
    """
    Recommend movies based on precomputed similarity scores.

    Parameters:
        title (str): Title of the movie selected by the user.
        movies (pd.DataFrame): Movie metadata.
        similarity_matrix: Matrix containing movie-to-movie
            similarity scores.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Recommended movies with similarity scores.

    Raises:
        ValueError: If the title is missing, the movie is not found,
            or top_n is not a positive integer.
    """

    # Validate the number of recommendations
    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer")

    # Validate the movie title
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    # Find the selected movie (case-insensitive)
    matches = movies[
        movies["title"].str.casefold() == title.casefold()
    ]

    if matches.empty:
        raise ValueError(f"Movie not found: {title}")

    # Use the first match if multiple movies share a title
    movie_index = matches.index[0]

    # Get similarity scores for the selected movie
    scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort from highest similarity to lowest
    scores.sort(key=lambda item: item[1], reverse=True)

    # Exclude the selected movie itself
    scores = [
        (idx, score)
        for idx, score in scores
        if idx != movie_index
    ]

    # Select the top N recommendations
    top_movies = scores[:top_n]

    # Build the recommendation results
    recommendations = movies.loc[
        [idx for idx, score in top_movies],
        ["movie_id", "title", "year", "genres"]
    ].copy()

    # Add the similarity score for each recommendation
    recommendations["similarity"] = [
        score for idx, score in top_movies
    ]

    return recommendations.reset_index(drop=True)
