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
		prediction = model.predict(incoming_message)
		prediction = np.argmax(prediction, axis = 0)

		print(prediction)

		result = dict()

		if prediction[0] != 0:
			result = {"type": "MALICIOUS", "prediction": flow_types[prediction[0]]}
		else:
			result = {"type": "LEGITIMATE", "prediction": flow_types[prediction[0]]}

		return result
