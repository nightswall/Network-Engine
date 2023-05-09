from model import Model
import numpy as np
import json

model = Model()
detector, detectorReduced = model.load_model("checkpoints/ckpt70/cp70.ckpt"), model.load_model("checkpoints/ckpt70_reduced.csv/cp70_reduced.csv.ckpt")
flow_types = {0: "bruteforce",
			  1: "dos",
			  2: "legitimate",
			  3: "malformed",
			  4: "slowite"
			 }

def get_prediction(incoming_message = None):
	global detector, detectorReduced, flow_types
	if incoming_message:
		# Prediction from the first engine
		prediction = detector.predict(incoming_message, batch_size = 1)
		prediction = np.argmax(prediction, axis = 1)

		# Prediction from the second engine
		predictionReduced = detectorReduced.predict(incoming_message, batch_size = 1)
		predictionReduced = np.argmax(predictionReduced, axis = 1)

		print(prediction)
		print(predictionReduced)
