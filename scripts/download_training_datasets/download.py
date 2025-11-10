import kagglehub
import os
from pathlib import Path
import shutil

def check_existence_path(OUTPUT_DIR : Path) -> None:
    """
    Check if path exists
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset(OUTPUT_DIR : Path, link:str, name:str) -> None:
    """
    Download the dataset using the specified link
    """
    check_existence_path(OUTPUT_DIR)
    output_final = OUTPUT_DIR / name
    
    if output_final.exists():
        print(f"The dataset {name} has been already downloaded!\n")
    else:
        cached_path = kagglehub.dataset_download(link)
        print(f"Download dataset {name} to:", cached_path)

        shutil.copytree(cached_path, output_final, dirs_exist_ok=True)

        print("Path to dataset files:", output_final, "\n")

def main() -> None:
    # Path handling
    current_file = Path(__file__).resolve()
    ROOT_PATH = current_file.parents[2]
    OUTPUT_PATH = ROOT_PATH / "dataset_training"

    check_existence_path(OUTPUT_DIR=OUTPUT_PATH)
    
    # Download datasets
    datasets = {
        "RAVDESS" : "uwrfkaggler/ravdess-emotional-speech-audio",
        "TESS"    : "ejlok1/toronto-emotional-speech-set-tess",
        "EmoDB"   : "piyushagni5/berlin-database-of-emotional-speech-emodb",
        "IEMOCAP" : "samuelsamsudinng/iemocap-emotion-speech-database"
        }
    
    for db, link in datasets.items():
        download_dataset(
            OUTPUT_DIR=OUTPUT_PATH,
            link = link,
            name = db
        )
    
    print(f"All datasets have been downloaded to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()