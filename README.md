# Movie Recommendation System

A content-based movie recommendation system built with Python and scikit-learn. It recommends movies with similar metadata, including genres, directors, and actors.

## How It Works

1. Loads movie metadata from a CSV dataset.
2. Combines genres, directors, and actors into a text profile for each movie.
3. Uses TF-IDF to convert text profiles into numerical vectors.
4. Calculates cosine similarity between movies.
5. Returns the top 10 most similar movies for a selected title.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF and cosine similarity
- Pytest
- Git and GitHub

## Project Structure

```text
movie-recommender-system/
├── data/
│   └── raw/
├── notebooks/
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

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ImmyAfridi/movie-recommender-system.git
cd movie-recommender-system
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Enter a movie title to receive recommendations. Type `quit` to exit.

## Run Tests

```bash
python -m pytest
```

## Limitations

This is a content-based recommendation system. It uses movie metadata rather than user ratings or viewing histories, so its recommendations reflect metadata similarity, not demonstrated user preferences.

## Future Improvements

- Improve feature weighting and recommendation quality.
- Add a web interface.
- Handle duplicate movie titles more clearly.
- Add recommendation evaluation methods.