# Movie Recommendation System

A content-based movie recommendation system built with Python and scikit-learn. It recommends movies similar to a selected title by comparing their genres, directors, and actors.

## Features

- Recommends similar movies based on movie metadata.
- Uses TF-IDF vectorization and cosine similarity.
- Supports configurable weights for genres, directors, and actors.
- Provides a command-line interface for interactive recommendations.
- Includes automated tests using pytest.

## How It Works

1. **Load data:** Read movie metadata, including titles, genres, directors, and actors.
2. **Preprocess features:** Handle missing metadata and prepare text features.
3. **Vectorize text:** Convert genres, directors, and actors into numerical TF-IDF representations.
4. **Calculate similarity:** Combine the feature matrices using configurable weights and calculate cosine similarity.
5. **Recommend movies:** Return the most similar movies while excluding the selected movie itself.

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- pytest

## Project Structure

```text
movie-recommender-system/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── movie_recommender.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── recommender.py
├── tests/
├── app.py
├── download_data.py
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/ImmyAfridi/movie-recommender-system.git
cd movie-recommender-system
```

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset is available at `data/raw/movielens_100k.csv`.

## Run the Application

```bash
python app.py
```

Enter a movie title when prompted. Type `quit` to exit.

## Run Tests

```bash
python -m pytest
```

## Model Configuration

The model allows independent weighting of genres, directors, and actors. For example:

```python
_, similarity = build_similarity_model(
    movies,
    genre_weight=2,
    director_weight=1,
    actor_weight=1,
)
```

This configuration gives genres greater influence while retaining director and actor information.

## Evaluation

The model was evaluated using genre overlap for four example movies: *Toy Story*, *Jumanji*, *Titanic*, and *Star Wars*. The selected configuration achieved high genre overlap on these examples.

Genre overlap is a proxy metric, not a complete measure of recommendation quality. Recommendations may share genres without matching a movie's story, themes, or audience preferences.

## Limitations

- Recommendations depend on available movie metadata.
- The model does not learn from user ratings, watch history, or preferences.
- Genre similarity alone cannot capture plot or thematic similarity.
- Evaluation on four movies is limited and does not establish general performance.

## Future Improvements

- Incorporate plot summaries or descriptions.
- Evaluate recommendations with a larger test set and additional ranking metrics.
- Add a web interface for easier interaction.
- Explore collaborative filtering using user-rating data.

## Author

**Imaad Afridi**

GitHub: [ImmyAfridi](https://github.com/ImmyAfridi)
