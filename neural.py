import numpy as np

input_neurons = np.array([7,6])

neuron_weights = {"weight_0": np.array([3,-2]),
                  "weight_1": np.array([-5,2]),
                  "output_weight": np.array([4,2])}

first_hidden_input_1 = (input_neurons * neuron_weights["weight_0"]).sum()
first_hidden_input_2 = (input_neurons * neuron_weights["weight_1"]).sum()

hidden_layer = {"hidden_layer": np.array([first_hidden_input_1, first_hidden_input_2])}

output_input = (hidden_layer["hidden_layer"] * neuron_weights["output_weight"]).sum()    
print(output_input)