
from pathlib import Path
import shutil
import kagglehub

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MovieLens 100K dataset through KaggleHub...")

    try:
        path = Path(
            kagglehub.dataset_download(
                "fakhrealam0786/movielens-100k-dataset"
            )
        )

        print(f"Downloaded dataset location: {path}")

        # Copy downloaded files into our project
        for item in path.iterdir():
            destination = RAW_DATA_DIR / item.name

            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

        print(f"Dataset files copied to: {RAW_DATA_DIR}")

    except Exception as error:
        print(f"Dataset download failed: {error}")


if __name__ == "__main__":
    download_dataset()
