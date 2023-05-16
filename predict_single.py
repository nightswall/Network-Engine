from model import Model
import numpy as np
import json

flow_types = {0: "BruteForce", 1: "dos", 3: "legitimate", 4: "malformed", 2: "SlowITe", 5: "Flooding"}

def get_prediction(model = None, incoming_message = None):
	global flow_types
	if model is not None and incoming_message is not None:
		# Prediction from the engine
		prediction = model.predict(incoming_message)
		prediction = np.argmax(prediction, axis = 1)
		#print(flow_types[prediction[0]])
		result = []

		for p in prediction:
			f = dict()
			if flow_types[p] == "legitimate":
				f = {"type": "LEGITIMATE", "prediction": flow_types[p]}
			else:
				f = {"type": "MALICIOUS", "prediction": flow_types[p]} 
			result.append(f)

		decisions = {"type": "RESPONSE", "predictions": result}
		#print(decisions["predictions"])
		return decisions
