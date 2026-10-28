
from src.data_loader import load_movies
from src.model import build_similarity_model
from src.recommender import recommend_movies


def main():
    # Load movie metadata
    movies = load_movies()

    # Build the similarity model
    _, similarity_matrix = build_similarity_model(movies)

    print("\n=== Movie Recommendation System ===")
    print("Enter a movie title to get recommendations.")
    print("Examples: Toy Story, Jumanji, Aladdin")
    print("Type 'quit' to exit.\n")

    while True:
        title = input("Movie title: ").strip()

        if title.casefold() == "quit":
            print("Goodbye!")
            break

        try:
            recommendations = recommend_movies(
                title=title,
                movies=movies,
                similarity_matrix=similarity_matrix,
                top_n=10
            )

            print(f"\nRecommendations for '{title}':\n")
            print(recommendations.to_string(index=False))
            print()

        except ValueError as error:
            print(f"\n{error}\n")


if __name__ == "__main__":
    main()