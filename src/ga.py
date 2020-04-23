from src.nn import NeuralNetwork
import numpy as np


class GeneticAlgorithm(object):

    _SMALLMUT = 0.0001
    _MUT = 0.01
    _w_min = -1.0
    _w_max = 1.0
    _b_min = -4
    _b_max = 4

    def __init__(self, max_gens=20, population_size=20, num_inputs=9, num_outputs=3, hidden_layer_lens=[], per_new_members=0):
        """
            Initialise the object
        """
        self.NGenMax = max_gens
        self.NPopSize = population_size
        self.NInputs = num_inputs
        self.NHiddenLayerLengths = hidden_layer_lens
        self.NOutputs = num_outputs
        self.rNewMembers = per_new_members  # percentage of brand new pop members in each generation
        self.NNewMembers = int(np.floor(self.NPopSize * per_new_members))

    def create_population(self, is_first=False, parents=[]):
        """
            Initialise the population
        """
        self.fitness = np.zeros(self.NPopSize)

        if is_first:
            # this is the first population
            self.pop = []
            for i in range(0, self.NPopSize):
                self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths))
                self.initialise_vars_random(self.pop[-1])
        else:
            # create a new generation
            in_ws = [p.inputs_w for p in parents]
            if parents[0].h_layers_w is not None:
                h_ws = [p.h_layers_w for p in parents]
                h_b = [p.h_layers_b for p in parents]
            else:
                h_ws = None
                h_b = None
            out_b = [p.outputs_b for p in parents]
            self.pop = []
            for i in range(0, self.NPopSize-self.NNewMembers):
                self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths))
                # randomly define 'the genes' of the member
                NParents = len(parents)
                NCreationChoice = np.random.randint(NParents * 2 + 1)
                if NCreationChoice == 0:
                    # Cross over - alwyas first
                    self.initialise_vars_crossover(in_ws, h_ws, h_b, out_b, self.pop[-1])
                else:
                    # we're going to mutate a single parent
                    p_idx = (NCreationChoice-1) // 2
                    is_small = ((NCreationChoice-1) % 2 == 0)
                    if is_small:
                        self.initialise_vars_small_mutation(parents[p_idx].inputs_w, parents[p_idx].h_layers_w, parents[p_idx].h_layers_b, parents[p_idx].outputs_b, self.pop[-1])
                    else:
                        self.initialise_vars_mutation(parents[p_idx].inputs_w, parents[p_idx].h_layers_w, parents[p_idx].h_layers_b, parents[p_idx].outputs_b, self.pop[-1])
            # initialise any brand new members to the population
            if self.NNewMembers > 0:
                for i in range(self.NPopSize-self.NNewMembers, self.NPopSize):
                    self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths))
                    self.initialise_vars_random(self.pop[-1])

    def initialise_vars_random(self, nn):
        """
            Initialise the weights randomly between -1 and +1
        """
        # inputs
        nn.inputs_w = np.random.rand(nn.inputs_w.shape[0], nn.inputs_w.shape[1]) * (GeneticAlgorithm._w_max * 2) - GeneticAlgorithm._w_max

        # hidden layers
        if nn.h_layers is not None:
            for i,h in enumerate(nn.h_layers_w):
                nn.h_layers_w[i] = np.random.rand(h.shape[0], h.shape[1]) * (GeneticAlgorithm._w_max * 2) - GeneticAlgorithm._w_max
                nn.h_layers_b[i] = np.random.rand(nn.h_layers_b[i].shape[0]) * (GeneticAlgorithm._b_max * 2) - GeneticAlgorithm._b_max

        # outputs
        nn.outputs_b = np.random.rand(nn.outputs_b.shape[0]) * (GeneticAlgorithm._b_max * 2) - GeneticAlgorithm._b_max


    def initialise_vars_small_mutation(self, in_w, h_w, h_b, o_b, nn):
        """
            Initialise the weights with small mutations of the provdided weights bounded by -1 and 1
        """
        nn.init_type = 'SMALL_MUT'
        # inputs
        nn.inputs_w = np.maximum(GeneticAlgorithm._w_min , np.minimum(GeneticAlgorithm._w_max, (in_w + np.random.rand(in_w.shape[0], in_w.shape[1]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))

        # hidden layers
        if nn.h_layers is not None:
            for i in range(0,len(nn.h_layers_w)):
                nn.h_layers_w[i] = np.maximum(GeneticAlgorithm._w_min , np.minimum(GeneticAlgorithm._w_max, (h_w[i] + np.random.rand(h_w[i].shape[0], h_w[i].shape[1]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))
                nn.h_layers_b[i] = np.maximum(GeneticAlgorithm._b_min , np.minimum(GeneticAlgorithm._b_max, (h_b[i] + np.random.rand(h_b[i].shape[0]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))

        # outputs
        nn.outputs_b = np.maximum(GeneticAlgorithm._b_min , np.minimum(GeneticAlgorithm._b_max, (o_b + np.random.rand(o_b.shape[0]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))

    def initialise_vars_mutation(self, in_w, h_w, h_b, o_b, nn):
        """
            Initialise the weights with mutations of the provdided weights bounded by -1 and 1
        """
        nn.init_type = 'MUT'
        # inputs
        nn.inputs_w = np.maximum(GeneticAlgorithm._w_min , np.minimum(GeneticAlgorithm._w_max, (in_w + np.random.rand(in_w.shape[0], in_w.shape[1]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))

        # hidden layers
        if nn.h_layers is not None:
            for i in range(0,len(nn.h_layers_w)):
                nn.h_layers_w[i] = np.maximum(GeneticAlgorithm._w_min , np.minimum(GeneticAlgorithm._w_max, (h_w[i] + np.random.rand(h_w[i].shape[0], h_w[i].shape[1]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))
                nn.h_layers_b[i] = np.maximum(GeneticAlgorithm._b_min , np.minimum(GeneticAlgorithm._b_max, (h_b[i] + np.random.rand(h_b[i].shape[0]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))

        # outputs
        nn.outputs_b = np.maximum(GeneticAlgorithm._b_min , np.minimum(GeneticAlgorithm._b_max, (o_b + np.random.rand(o_b.shape[0]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))


    def initialise_vars_crossover(self, in_ws, h_ws, h_b, o_b, nn):
        """
            Initialise the weights by cross breeding the list of parents
        """
        NParents = len(in_ws)
        nn.init_type = 'CROSS'

        # input weights - generate an array of random numbers between 0 and number of parants
        w = np.random.randint(NParents, size=(in_ws[0].shape[0], in_ws[0].shape[1]))
        for i in range(0,NParents):
            nn.inputs_w[w == i] = in_ws[i][w == i]

        # hidden layers
        if nn.h_layers is not None:
            for i in range(0,len(h_ws[0])):
                w = np.random.randint(NParents, size=(h_ws[0][i].shape[0], h_ws[0][i].shape[1]))
                for ii in range(0,NParents):
                    nn.h_layers_w[i][w == ii] = h_ws[ii][i][w == ii]
                w = np.random.randint(NParents, size=(h_b[0][i].shape[0]))
                for ii in range(0,NParents):
                    nn.h_layers_b[i][w == ii] = h_b[ii][i][w == ii]
        w = np.random.randint(NParents, size=(o_b[0].shape[0]))
        for i in range(0,NParents):
            nn.outputs_b[w == i] = o_b[i][w == i]
