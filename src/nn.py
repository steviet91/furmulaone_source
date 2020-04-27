import numpy as np
import os

class NeuralNetwork(object):
    """
        Class to store a neural network for driving the vehicle
    """


    def __init__(self, input_len, output_len, hidden_layer_lens, dtype=np.float64):
        """
            Initialise the object
        """
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        # instantiate input layer
        self.inputs = np.zeros(input_len, dtype=np.float64)
        if len(hidden_layer_lens) > 0:
            self.inputs_w = np.zeros((input_len, hidden_layer_lens[0]), dtype=dtype)
        else:
            # no hidden layers
            self.inputs_w = np.zeros((input_len, output_len), dtype=dtype)

        # instantiate hidden layers
        if len(hidden_layer_lens) > 0:
            self.h_layers = []
            self.h_layers_w = []
            self.h_layers_b = []
            for i,h in enumerate(hidden_layer_lens):
                self.h_layers.append(np.zeros(h))
                self.h_layers_b.append(np.zeros(h, dtype=dtype))
                if i == (len(hidden_layer_lens)-1):
                    # last layer, hook up onto outputs
                    self.h_layers_w.append(np.zeros((h, output_len), dtype=dtype))
                else:
                    # hook up to the next layer
                    self.h_layers_w.append(np.zeros((h, hidden_layer_lens[i+1]), dtype=dtype))
        else:
            # No hidden layers
            self.h_layers = None
            self.h_layers_w = None
            self.h_layers_b = None

        # instantiate outputs
        self.outputs = np.zeros(output_len)
        self.outputs_b = np.zeros(output_len, dtype=dtype)

        self.init_type = 'RANDO'

    def update_network(self, inputs):
        """
            Run the update on the network
        """
        self.inputs = inputs

        if self.h_layers is not None:
            # set the hidden layers
            for i in range(0, len(self.h_layers)):
                if i == 0:
                    self.h_layers[i] = self.sigmoid_function(np.dot(self.inputs, self.inputs_w) + self.h_layers_b[i])
                else:
                    self.h_layers[i] = self.sigmoid_function(np.dot(self.h_layers[i-1], self.h_layers_w[i-1]) + self.h_layers_b[i])
            # update the outputs
            self.outputs = self.sigmoid_function(np.dot(self.h_layers[-1], self.h_layers_w[-1]) + self.outputs_b)
        else:
            # No hidden layers - just update the ouputs
            self.outputs = self.sigmoid_function(np.dot(self.inputs, self.inputs_w) + self.outputs_b)

    def sigmoid_function(self, x):
        """
            Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def pickle_nn(self, suffix=None):
        """
            Save the NN as a binary by pickling it
        """
        import pickle
        from datetime import datetime
        f_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix is not None:
            f_name += '_' + suffix
        pickle.dump(self, open(self.module_path + '/../data/nn/' + f_name + '.nn', 'wb'))

    @classmethod
    def loader(cls, nn_name):
        import pickle
        # load the nn
        module_path = os.path.dirname(os.path.abspath(__file__))
        return pickle.load(open(module_path + '/../data/nn/' + nn_name + '.nn', 'rb'))
