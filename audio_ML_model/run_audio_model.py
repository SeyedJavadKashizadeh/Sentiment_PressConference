from pathlib import Path
import pandas as pd
import numpy as np
from audio_prediction import audio_pred
from audio_model import train_model 

def main_train(model_path, infile):

    model, int2emotions, emotions2int = train_model(model_path, infile)

def main_test(model_path, infile, outfile):

    audio_pred(model_path, infile, outfile)

if __name__ == "__main__":
    path_dir = Path(__file__).resolve().parent
    infile = path_dir / "merged_with_features.csv"
    outfile = path_dir / "predictions.csv"
    model_path = path_dir / 'voice_model.h5'
    model_path = str(model_path)

    
    main_train(model_path, infile)
    main_test(model_path, infile, outfile)