from warnings import simplefilter
import numpy as np
from model import Model
from data_loader import DataLoader
from threading import Thread

def train70():
    training_path = "drive/MyDrive/train70.csv"
    testing_path = "drive/MyDrive/test30.csv"
    checkpoint_path = "checkpoints/cp70.ckpt"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    model = Model()
    data_loader = DataLoader()

    x_train, y_train = data_loader.initalize_training_data(training_path)
    x_test, y_test = data_loader.initialize_test_data(testing_path)

    detector, callbacks = model.create_model(x_train.shape[1], checkpoint_path)
    print("Training with train70 dataset started!\n")

    history = detector.fit(
                x_train,
                y_train,
                validation_data = (x_test, y_test),
                callbacks = callbacks,
                verbose = 2,
                epochs = 200,
                batch_size = 1000
    )

def train70_reduced():
    training_path = "datasets/Data/FINAL_CSV/train70_reduced.csv"
    testing_path = "datasets/Data/FINAL_CSV/test30_reduced.csv"
    checkpoint_path = "checkpoints/ckpt70reduced/cp70_reduced.ckpt"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    model = Model()
    data_loader = DataLoader()

    x_train, y_train = data_loader.initalize_training_data(training_path)
    x_test, y_test = data_loader.initialize_test_data(testing_path)

    detector, callbacks = model.create_model(x_train.shape[1], checkpoint_path)
    print("Training with train70_reduced started!\n")

    history = detector.fit(
                x_train,
                y_train,
                validation_data = (x_test, y_test),
                callbacks = callbacks,
                verbose = 2,
                epochs = 200,
                batch_size = 1000
    )

    y_pred_nn = detector.predict(x_test)

def train70_augmented():
    training_path = "drive/MyDrive/train70_augmented.csv"
    testing_path = "drive/MyDrive/test30_augmented.csv"
    checkpoint_path = "checkpoints/cp70_augmented.ckpt"
    simplefilter(action = "ignore", category = FutureWarning)
    seed = 7

    model = Model()
    data_loader = DataLoader()

    x_train, y_train = data_loader.initalize_training_data(training_path)
    x_test, y_test = data_loader.initialize_test_data(testing_path)

    detector, callbacks = model.create_model(x_train.shape[1], checkpoint_path)
    print("Training with train70_augmented started!\n")

    history = detector.fit(
                x_train,
                y_train,
                validation_data = (x_test, y_test),
                callbacks = callbacks,
                verbose = 2,
                epochs = 200,
                batch_size = 1000
    )

def main():
    train70_reduced()


if __name__ == "__main__":
    main()