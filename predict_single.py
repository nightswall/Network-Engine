from model import Model
import numpy as np
import json

flow_types = {1: "dos",
			  0: "legitimate",
			 }

def get_prediction(model = None, incoming_message = None):
	global flow_types
	if model is not None and incoming_message is not None:
		# Prediction from the engine
		prediction = model.predict_classes(incoming_message, batch_size = 1)

		print(prediction)

		result = dict()
		result = {"type": "MALICIOUS", "prediction": "asd"}

		return result
