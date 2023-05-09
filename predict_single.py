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
	if model is not None and incoming_message is not None:
		# Prediction from the engine
		prediction = model.predict(incoming_message)
		prediction = np.argmax(prediction, axis = 1)

		print(prediction)

		result = dict()

		if prediction[0] != 2:
			result = {"type": "MALICIOUS", "prediction": flow_types[prediction[0]]}
		else:
			result = {"type": "LEGITIMATE", "prediction": flow_types[prediction[0]]}

		return result
