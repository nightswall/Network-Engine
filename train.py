import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--trainset",  help = "Name of the dataset that should be used for training.", required=True)
parser.add_argument("--testset", help = "Name of the dataset that should be used for validating.", required=True)
parser.add_argument("--evaluate", help = "Automatically evaluate after training finishes.", action = "store_true", required=True)
args = parser.parse_args()

from warnings import simplefilter
import numpy as np
from model import Model
from data_loader import DataLoader
from threading import Thread


def train_model(training_set, testing_set):
    training_path = "datasets/Data/FINAL_CSV/" + training_set + ".csv"
    testing_path = "datasets/Data/FINAL_CSV/" + testing_set + ".csv"

    model_name = training_path.split("n")[1]

    checkpoint_path = "checkpoints/ckpt" + model_name + "/cp" + model_name + ".ckpt"
    simplefilter(action = "ignore", category = FutureWarning)

    model = Model()
    data_loader = DataLoader()

    x_train, y_train = data_loader.initalize_training_data(training_path)
    x_test, y_test = data_loader.initalize_training_data(testing_path)

    detector, callbacks = model.create_model(x_train.shape[1], checkpoint_path)
    print(f"Training with {training_set} dataset started!\n")

    _ = detector.fit(
                x_train,
                y_train,
                validation_data = (x_test, y_test),
                callbacks = callbacks,
                verbose = 2,
                epochs = 200,
                batch_size = 1000
    )

if __name__ == "__main__":
    auto_evaluate = args.evaluate
    train_set_name = args.trainset
    test_set_name = args.testset
    train_model(train_set_name, test_set_name)
    print("Training finished!")
    if auto_evaluate:
        print("Calling evaluator!")
        command = "python3 evaluate.py"
        os.system(command)
        exit(0)