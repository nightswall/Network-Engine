import matplotlib.pyplot as plt
from model import Model
from data_loader import DataLoader
from sklearn import metrics
from warnings import simplefilter
import numpy as np
import json
import pandas as pd

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
    setDF = pd.read_csv(testing_path, chunksize = 50)

    accuracies = []
    f1_scores = []
    chunkNumber = 0
    for chunk in setDF:
        chunkNumber += 1
        print(f"In chunk {chunkNumber} with Test Set: {test_set}")
        x_test, y_test = data_loader.initialize_test_data(chunk)
        detector, _ = model.create_model(x_test.shape[1], checkpoint_path)
        detector = model.load_model(detector, checkpoint_path)   

        y_pred = detector.predict(x_test)
        y_pred = np.argmax(y_pred, axis = 1)
        print(y_pred)
        print(y_test)

        accuracies.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred, average = "weighted"))
        break

    avgAccuracy = 0
    avgF1Score = 0

    for idx in range(len(accuracies)):
        avgAccuracy += accuracies[idx]
        avgF1Score += f1_scores[idx]

    avgAccuracy = avgAccuracy / len(accuracies)
    avgF1Score = avgF1Score / len(f1_scores)

    model_name = checkpoint_path.split("/")

    result = {"Dataset": test_set, "Model": model_name[1], "Accuracy": avgAccuracy, "F1 Score": avgF1Score}
    result_json = json.dumps(result, indent = 4)
    
    file_name = "result" + "_" + test_set + "_" + model_name[1] + ".json"

    with open(file_name, "w") as out:
        out.write(result_json)

    results.append(result)

for predictor in model_checkpoints:
    print("Testing started with model: %s " % (predictor.split("/")[1]))
    for test in test_sets:
        test_model(predictor, test)

print(results)
