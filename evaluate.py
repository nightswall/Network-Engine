import matplotlib.pyplot as plt
from model import Model
from data_loader import DataLoader
from sklearn import metrics
from warnings import simplefilter
import numpy as np
import json

model = Model()
data_loader = DataLoader()

results = []
model_checkpoints = ["checkpoints/ckpt70reduced/cp70_reduced.ckpt"]

def test30(checkpoint_path):
    global results
    testing_path = "datasets/Data/FINAL_CSV/test30.csv"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    x_test, y_test = data_loader.initialize_test_data(testing_path)
    detector, _ = model.create_model(x_test.shape[1], checkpoint_path)
    detector = model.load_model(detector, checkpoint_path)

    y_pred = detector.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    loss, acc = detector.evaluate(x_test, y_test, verbose=2)

    model_name = checkpoint_path.split("/")

    result = {"Dataset": "test30", "Model": model_name[1], "Accuracy": [metrics.accuracy_score(y_test, y_pred), 100 * acc], "F1 Score": metrics.f1_score(y_test, y_pred, average = "weighted")}
    result_json = json.dumps(result, indent = 4)
    with open("result.json", "w") as out:
        out.write(result_json)
    results.append(result)

def test30_reduced(checkpoint_path):
    global results
    testing_path = "datasets/Data/FINAL_CSV/test30_reduced.csv"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    x_test, y_test = data_loader.initialize_test_data(testing_path)
    detector, _ = model.create_model(x_test.shape[1], checkpoint_path)
    detector = model.load_model(detector, checkpoint_path)

    y_pred = detector.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    loss, acc = detector.evaluate(x_test, y_test, verbose=2)

    model_name = checkpoint_path.split("/")

    result = {"Dataset": "test30_reduced", "Model": model_name[1], "Accuracy": [metrics.accuracy_score(y_test, y_pred), 100 * acc], "F1 Score": metrics.f1_score(y_test, y_pred, average = "weighted")}
    result_json = json.dumps(result, indent = 4)
    with open("result.json", "w") as out:
        out.write(result_json)
    results.append(result)

def test30_augmented(checkpoint_path):
    global results
    testing_path = "datasets/Data/FINAL_CSV/test30_augmented.csv"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    x_test, y_test = data_loader.initialize_test_data(testing_path)
    detector, _ = model.create_model(x_test.shape[1], checkpoint_path)
    detector = model.load_model(detector, checkpoint_path)

    y_pred = detector.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    loss, acc = detector.evaluate(x_test, y_test, verbose=2)

    model_name = checkpoint_path.split("/")

    result = {"Dataset": "test30_augmented", "Model": model_name[1], "Accuracy": [100 * acc], "F1 Score": 0}
    result_json = json.dumps(result, indent = 4)
    with open("result.json", "w") as out:
        out.write(result_json)
    results.append(result)

for predictor in model_checkpoints:
    print("Testing started with model: %s " % (predictor.split("/")[1]))
    test30(predictor)
    test30_reduced(predictor)
    test30_augmented(predictor)

print(results)