from model import Model
import numpy as np
import json

flow_types = {0: "bruteforce",
			  1: "dos",
			  2: "legitimate",
			  3: "malformed",
			  4: "slowite"
			 }

def get_prediction(model = None, incoming_message = None):
	global flow_types
	if model:
		# Prediction from the engine
		prediction = model.predict(incoming_message, batch_size = 1)
		prediction = np.argmax(prediction, axis = 1)

		print(prediction)
		print(predictionReduced)
