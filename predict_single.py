from model import Model
import numpy as np
import json

flow_types = {0: "dos",
			  1: "legitimate",
			 }

def get_prediction(model = None, incoming_message = None):
	global flow_types
	if model is not None and incoming_message is not None:
		# Prediction from the engine
		prediction = model.predict(incoming_message, batch_size = 1)
		prediction = np.argmax(prediction, axis = 1)

		print(prediction)

		result = dict()

		if prediction[1] != 1:
			result = {"type": "MALICIOUS", "prediction": flow_types[prediction[1]]}
		else:
			result = {"type": "LEGITIMATE", "prediction": flow_types[prediction[1]]}

		return result
