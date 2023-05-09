import matplotlib.pyplot as plt
from model import Model
from data_loader import DataLoader
from sklearn import metrics
from warnings import simplefilter
import numpy as np
import json
import pandas as pd
from predict_single import get_prediction

model = Model()
data_loader = DataLoader()

results = []
model_checkpoints = ["checkpoints/ckpt70_reduced.csv/cp70_reduced.csv.ckpt", "checkpoints/ckpt70/cp70.ckpt"]
test_sets = ["test30", "test30_reduced", "test30_augmented"]
CHUNK_SIZE = 10 ** 6

def test_model(checkpoint_path, test_set):
    global results
    testing_path = "datasets/Data/FINAL_CSV/" + test_set + ".csv"
    simplefilter(action = "ignore", category = FutureWarning)

    for idx in range(500):
        df = pd.read_csv(testing_path, skiprows=idx, nrows=1)
        print(f"In chunk {idx} with Test Set: {test_set}")

        df.columns.drop("target")


        detector, _ = model.create_model(df.shape[1], checkpoint_path)
        detector = model.load_model(detector, checkpoint_path)   

        print(get_prediction(detector, df))

test_model(model_checkpoints[0], test_sets[1])