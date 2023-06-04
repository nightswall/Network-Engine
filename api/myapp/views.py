from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse

from io import StringIO
import os
import time
import numpy as np
import pandas as pd
import json
import csv

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_v2_behavior()

class Model():
  def create_model(__self__, input_dim, checkpoint_path):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
              50, 
              input_dim = input_dim, 
              kernel_initializer = 'normal', 
              activation = 'relu'))
    model.add(tf.keras.layers.Dense(
              30, 
              input_dim = input_dim, 
              kernel_initializer = 'normal', 
              activation = 'relu'))
    model.add(tf.keras.layers.Dense(
              20, 
              kernel_initializer = 'normal'))
    model.add(tf.keras.layers.Dense(
              6,
              activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # Code below will help us to quit training when decrease in
    # validation loss happens more than 5 times. Model is either
    # overfitting or underfitting in this case. So, continueing the
    # training is pointless.
    monitor = tf.keras.callbacks.EarlyStopping(
              monitor = 'val_loss',
              min_delta = 1e-3,
              patience = 5,
              verbose = 1,
              mode = 'auto')
    # Code below will help us to save trained models and use them afterwards.
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath = checkpoint_path,
                save_weights_only = True,
                save_best_only = True,
                verbose = 1)
    return model, [monitor, checkpoint]

  def load_model(__self__, model, checkpoint_path, session_path):
    model.load_weights(checkpoint_path).expect_partial()
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.keras.backend.get_session()
    saver.restore(sess, session_path)
    return model

class DataLoader():
    def initalize_training_data(__self__, dataset_path):
        training_set = pd.read_csv(dataset_path)

        class_names = training_set.target.unique()
        training_set = training_set.astype("category")
        category_columns = training_set.select_dtypes(["category"]).columns

        training_set[category_columns] = training_set[category_columns].apply(lambda x : x.cat.codes)

        x_columns = training_set.columns.drop("target")
        x_training = training_set[x_columns].values
        y_training = training_set["target"]

        return x_training, y_training

    def initialize_test_data(__self__, testing_set):
        
        class_names = testing_set.target.unique()
        testing_set = testing_set.astype("category")
        category_columns = testing_set.select_dtypes(["category"]).columns
        classes = {"bruteforce": 0, "dos": 1, "legitimate": 2, "malformed": 3, "slowite": 4, "flooding": 5}
        for category in category_columns:
            if category != "target":
                testing_set[category] = testing_set[category].apply(lambda x : x.cat.codes)

        
        #testing_set["tcp.flags"] = testing_set["tcp.flags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.conflags"] = testing_set["mqtt.conflags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.hdrflags"] = testing_set["mqtt.hdrflags"].apply(lambda x : float(int(str(x), 16)))
        testing_set["target"] = testing_set["target"].apply(lambda x : classes[x])

        x_testing = testing_set[x_columns].values
        y_testing = testing_set["target"]

        return x_testing, y_testing

flow_types = {2: "legitimate", 1: "dos", 0: "bruteforce", 3: "malformed", 4: "slowite", 5: "flooding"}
model_checkpoint = "/home/gorkem/network-engine/api/myapp/cp70_reduced.ckpt"
session_checkpoint = "/home/gorkem/network-engine/api/myapp/session.ckpt"


def get_prediction(incoming_message = None):
	global flow_types, model_checkpoint, session_checkpoint
	model = Model()
	detector, _ = model.create_model(33, model_checkpoint) # Creating a new model for reference
	detector = model.load_model(detector, model_checkpoint, session_checkpoint) # Loading model checkpoint and session
	detector.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	#print(incoming_message)

	if incoming_message is not None: # Checking if a valid request is made
		prediction = detector.predict(incoming_message) # Prediction from the engine
		prediction = np.argmax(prediction, axis = 1)
		
		decisions = dict()

		if flow_types[prediction[0]] != "legitimate":
			decisions = {"type": "MALICIOUS", "predictions": flow_types[prediction[0]]}
		else:
			decisions = {"type": "LEGITIMATE", "predictions": flow_types[prediction[0]]}
		return decisions

@csrf_exempt
def network_prediction(request):
	data = request.POST.get("data")
	csv_data = StringIO("{}".format(data))
	columns = ["tcp.flags", "tcp.time_delta", "tcp.len", "mqtt.conack.flags", "mqtt.conack.flags.reversed", "mqtt.conack.flags.sp", "mqtt.conack.val", "mqtt.conflag.cleansess", "mqtt.conflag.passwd", "mqtt.conflag.qos", "mqtt.conflag.reversed", "mqtt.conflag.retain", "mqtt.conflag.uname", "mqtt.conflag.willflag", "mqtt.conflags", "mqtt.dupflag", "mqtt.hdrflags", "mqtt.kalive", "mqtt.len", "mqtt.msg", "mqtt.msgid", "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.qos", "mqtt.retain", "mqtt.sub.qos", "mqtt.suback.qos", "mqtt.ver", "mqtt.willmsg", "mqtt.willmsg_len", "mqtt.willtopic", "mqtt.willtopic_len", "target"]

	df = pd.read_csv(csv_data, header = None, names = columns)
	#print(data.split(","))
	#del csv_data[-1]
	#print(df)
	#print(csv_data)
	data_loader = DataLoader()
	x, _ = data_loader.initialize_test_data(df)
	#print(x)
	return HttpResponse(json.dumps({"prediction": get_prediction(x)}))

@csrf_exempt
def predict_network(request):
	return network_prediction(request)