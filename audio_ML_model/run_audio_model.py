from pathlib import Path
import pandas as pd
import numpy as np
#from audio_prediction import processing_data
from audio_model import train_model 

def main():

    path_model = Path(__file__).resolve().parent
    infile = path_model / "merged_with_features.csv"
    model, int2emotions, emotions2int = train_model(path_model, infile)


if __name__ == "__main__":

    main()