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

class fl:
    def __init__(this, value=0, byte_size=4):
        this.value = value
        if this.value: # speedy check (before performing any calculations)
            Fe=((byte_size*8)-1)//(byte_size+1)+(byte_size>2)*byte_size//2+(byte_size==3)
            Fm,Fb,Fie=(((byte_size*8)-(1+Fe)), ~(~0<<Fe-1), (1<<Fe)-1)
            FS,FE,FM=((this.value>>((byte_size*8)-1))&1,(this.value>>Fm)&Fie,this.value&~(~0 << Fm))
            if FE == Fie: this.value=(float('NaN') if FM!=0 else (float('+inf') if FS else float('-inf')))
            else: this.value=((pow(-1,FS)*(2**(FE-Fb-Fm)*((1<<Fm)+FM))) if FE else pow(-1,FS)*(2**(1-Fb-Fm)*FM))
            del Fe; del Fm; del Fb; del Fie; del FS; del FE; del FM
        else: this.value = 0.0

def test_model(checkpoint_path, test_set):
    global results
    testing_path = "datasets/Data/FINAL_CSV/" + test_set + ".csv"
    simplefilter(action = "ignore", category = FutureWarning)
    detector, _ = model.create_model(33, checkpoint_path)
    detector = model.load_model(detector, checkpoint_path)   

    for idx in range(500):
        df = pd.read_csv(testing_path, skiprows=idx+1, nrows=1)
        print(f"In chunk {idx} with Test Set: {test_set}")

        del df[df.columns[-1]]
        x_testing = df[df.columns].values

        for idx in range(len(x_testing[0])):
            try:
                x_testing[0][idx] = fl(int(x_testing[0][idx], 16)).value
            except:
                pass
        print(x_testing)
        print(get_prediction(detector, x_testing))

test_model(model_checkpoints[0], test_sets[1])