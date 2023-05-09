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

def test_model(checkpoint_path, test_set):
    global results
    testing_path = "datasets/Data/FINAL_CSV/" + test_set + ".csv"
    simplefilter(action = "ignore", category = FutureWarning)
    detector, _ = model.create_model(33, checkpoint_path)
    detector = model.load_model(detector, checkpoint_path) 
    df = pd.read_csv(testing_path, chunksize = 1)  

    ctrLeg = 0
    ctrMal = 0

    for chunk in df:
        x_test, y_test = data_loader.initialize_test_data(chunk)

        res = get_prediction(detector, np.array( x_test ))
        if res["type"] == "LEGITIMATE":
            ctrLeg += 1
        else:
            ctrMal += 1

    print (ctrLeg, ctrMal, ctrLeg / ctrMal)

test_model(model_checkpoints[1], test_sets[1])