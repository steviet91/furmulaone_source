from src.nn import NeuralNetwork



class GeneticAlgorithm(object):

    _SMALLMUT = 0.01
    _MUT = 0.1

    def __init__(self, max_gens=20, population_size=20, num_inputs=9, num_outputs=3, hidden_layer_lens=[], per_new_members=0):
        """
            Initialise the object
        """
        self.NGenMax = max_gens
        self.NPopSize = population_size
        self.NInputs = num_inputs
        self.NHiddenLayerLengths = hidden_layer_lens
        self.NOutputs = num_outputs
        self.rNewMembers = new_members  # percentage of brand new pop members in each generation
        self.NNewMembers = np.floor(self.PopSize * per_new_members)

    def create_population(self, is_first=False, parents=[]):
        """
            Initialise the population
        """
        self.fitness = [None] * self.NPopSize # set the fitness

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
            else:
                h_ws = None
            self.pop = []
            for i in range(0, self.NPopSize-self.NNewMembers):
                self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths))
                # randomly define 'the genes' of the member
                NParents = len(parents)
                NCreationChoice = np.random.randint(NParents * 2 + 1)
                if NCreationChoice == 0:
                    # Cross over - alwyas first
                    self.initialise_vars_crossover(in_ws, h_ws, self.pop[-1])
                else:
                    # we're going to mutate a single parent
                    p_idx = (NCreationChoice-1) // 2
                    is_small = ((NCreationChoice-1) % 2 == 0)
                    if is_small:
                        self.initialise_vars_small_mutation(parents[i].in_w, parents[i].h_w, self.pop[-1])
                    else:
                        self.initialise_vars_mutation(parents[i].in_ws, parents[i].h_w, self.pop[-1])
            # initialise any brand new members to the population
            if self.NNewMembers > 0:
                for i in range(self.NPopSize-self.NewMembers, self.NPopSize):
                    self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths))
                    self.initialise_vars_random(self.pop[-1])

    def initialise_vars_random(self, nn):
        """
            Initialise the weights randomly between -1 and +1
        """
        nn.inputs_w = np.random.rand(nn.inputs_w.shape[0], nn.inputs_w.shape[1]) * 2 - 1

        if nn.h_layers is not None:
            for i,h in enumerate(nn.h_layers_w):
                nn.h_layers_w[i] = np.random.rand(h.shape[0], h.shape[1]) * 2 - 1


    def initialise_vars_small_mutation(self, in_w, h_w, nn):
        """
            Initialise the weights with small mutations of the provdided weights bounded by -1 and 1
        """
        nn.inputs_w = max(-1.0 , min(1.0, (in_w + np.random.rand(in_w.shape[0], in_w.shape[1]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))
        if nn.h_layers is not None:
            for i,h in nn.h_layers_w:
                nn.h_layers_w[i] = max(-1.0 , min(1.0, (h_w + np.random.rand(h_w[i].shape[0], h_w[i].shape[1]) * (GeneticAlgorithm._SMALLMUT * 2) - GeneticAlgorithm._SMALLMUT)))

    def initialise_vars_mutation(self, in_w, h_w, nn):
        """
            Initialise the weights with mutations of the provdided weights bounded by -1 and 1
        """
        nn.inputs_w = max(-1.0 , min(1.0, (in_w + np.random.rand(in_w.shape[0], in_w.shape[1]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))
        if nn.h_layers is not None:
            for i,h in nn.h_layers_w:
                nn.h_layers_w[i] = max(-1.0 , min(1.0, (h_w + np.random.rand(h_w[i].shape[0], h_w[i].shape[1]) * (GeneticAlgorithm._MUT * 2) - GeneticAlgorithm._MUT)))

    def initialise_vars_crossover(self, in_ws, h_ws, nn):
        """
            Initialise the weights by cross breeding the list of parents
        """
        NParents = len(in_ws)
        # input weights - generate an array of random numbers between 0 and number of parants
        w = np.random.randint(NParents, size=(in_w[0].shape[0], in_w[0].shape[1]))
        for i in range(0,NParents):
            nn.inputs_w[w == i] = in_ws[i][w == i]

        # hidden layers
        if nn.h_layers is not None:
            for i in range(0,len(h_ws[0])):
                w = np.random.randint(NParents, size=(h_ws[0][i].shape[0], h_ws[0][i].shape[1]))
                for ii in range(0,NParents):
                    nn.h_layers_w[i][w == ii] = h_ws[ii][i][w == ii]
