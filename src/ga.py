from src.nn import NeuralNetwork
import numpy as np
import copy
from bitarray import bitarray


class GeneticAlgorithm(object):

    _w_min = -1.0
    _w_max = 1.0
    _b_min = -4
    _b_max = 4
    _float_dtype = np.float64
    _bin_dtype = np.int16
    _bin_bitlength = 16
    _w_scaling = 32767
    _b_scaling = 32767 / _b_max
    _bin_max = 32767
    _bin_min = -32767
    _member_retention = 0.5
    _elite_members = 0.01
    _male_rank_power = 1
    _prob_mutation = 0.5
    _prob_small_mutation = 0.5

    def __init__(self, max_gens=20, population_size=20, num_inputs=9, num_outputs=3, hidden_layer_lens=[]):
        """
            Initialise the object
        """
        self.NGenMax = max_gens
        self.NPopSize = population_size
        self.NInputs = num_inputs
        self.NHiddenLayerLengths = hidden_layer_lens
        self.NOutputs = num_outputs
        self.NMembersToRetain = int(np.ceil(self.NPopSize * GeneticAlgorithm._member_retention))
        self.NEliteMembers = int(np.ceil(self.NPopSize * GeneticAlgorithm._elite_members))

    # #######################
    # BINARY GA CONVERSIONS #
    # #######################
    def convert_to_float(self, x, scaling):
        """
            Convert the provided binary representation into float
        """
        return x.astype(GeneticAlgorithm._float_dtype) / scaling

    def convert_to_binary(self, x, scaling):
        """
            Convert the provided floating point representation to binary form
        """
        return (x * scaling).astype(GeneticAlgorithm._bin_dtype)

    def convert_all_to_binary(self):
        """
            Convert all the weights into binary form
        """
        for p in self.pop:
            p.inputs_w = self.convert_to_binary(p.inputs_w, GeneticAlgorithm._w_scaling)
            if p.h_layers is not None:
                for i,h in enumerate(p.h_layers_w):
                    p.h_layers_w[i] = self.convert_to_binary(h, GeneticAlgorithm._w_scaling)
                    p.h_layers_b[i] = self.convert_to_binary(p.h_layers_b[i], GeneticAlgorithm._b_scaling)
            p.outputs_b = self.convert_to_binary(p.outputs_b, GeneticAlgorithm._b_scaling)

    def convert_all_to_float(self):
        """
            Convert all the weights into float form
        """
        for p in self.pop:
            p.inputs_w = self.convert_to_float(p.inputs_w, GeneticAlgorithm._w_scaling)
            if p.h_layers is not None:
                for i,h in enumerate(p.h_layers_w):
                    p.h_layers_w[i] = self.convert_to_float(h, GeneticAlgorithm._w_scaling)
                    p.h_layers_b[i] = self.convert_to_float(p.h_layers_b[i], GeneticAlgorithm._b_scaling)
            p.outputs_b = self.convert_to_float(p.outputs_b, GeneticAlgorithm._b_scaling)


    def create_population(self, is_first=False):
        """
            Initialise the population
        """
        if is_first:
            # this is the first population
            self.fitness = np.zeros(self.NPopSize)
            self.pop = []
            for i in range(0, self.NPopSize):
                # intiialise in binary land and then convert to float
                self.pop.append(NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths, dtype=GeneticAlgorithm._bin_dtype))
                self.apply_random_vars(self.pop[-1])
            self.convert_all_to_float()
        else:
            # create a new generation of the population

            # find the top x fitness values, kill off the other members
            ret_idxs = np.flip(np.argpartition(self.fitness, -1 * self.NMembersToRetain)[-1 * self.NMembersToRetain:])
            old_pop = copy.deepcopy(self.pop)
            old_fit = np.copy(self.fitness)
            self.pop = [old_pop[i] for i in ret_idxs]
            self.fitness = np.array([old_fit[i] for i in ret_idxs])

            # set all the weightings (genes) to binary form
            self.convert_all_to_binary()

            # select which are male and which are female with 50/50 probabilty
            # rank the males in terms of elitness and give them a probabilty of

            # determine the sex of the members
            sex_type = np.random.randint(0, 2, size=(self.NMembersToRetain)) # 0 = male, 1 = female

            # just check that both sexes are represeted - if not then set the firt value to the missing sex
            if len(np.where(sex_type == 0)[0]) == 0:
                # missing a male
                sex_type[0] = 0
            elif len(np.where(sex_type == 1)[0]) == 0:
                # missing a female
                sex_type[0] = 1

            # divide the population
            m_pop = [self.pop[i] for i in sex_type if i == 0]
            m_f = np.array([self.fitness[i] for i in sex_type if i == 0])
            f_pop = [self.pop[i] for i in sex_type if i == 1]
            f_f = np.array([self.fitness[i] for i in sex_type if i == 1])

            # calculate the probablity of any one male breeding
            m_r_idxs = np.argsort(m_f)
            m_r = np.array([1 / (x**GeneticAlgorithm._male_rank_power) for x in range(1, len(m_pop) + 1)]) # rank values
            m_p = np.cumsum(m_r) / np.sum(m_r)  # these are the probabilty of a male member breeding

            # set the probability of any one female breeding
            f_p = np.cumsum(np.ones(len(f_pop)) / len(f_pop))

            # self.pop currently contains the top set from the previous generation
            # in order of decreasing fitness. Pad out the population to the pop size
            # with cross overs of the males and females

            # roll the dice for the males and female cross overs
            parent_pairs = np.random.rand(self.NPopSize - self.NMembersToRetain, 2)
            for p, m in parent_pairs:
                parents = []
                # find the male parent
                for i in range(0, m_p.shape[0]):
                    if m_p[i] >= p:
                        parents.append(m_pop[i])
                        break
                # find the female parent
                for i in range(0, f_p.shape[0]):
                    if f_p[i] >= m:
                        parents.append(f_pop[i])
                        break

                # breed the new member
                self.pop.append(self.apply_crossover(parents))

            # determine if mutation is required for each member
            # elite members are not mutated
            for i in range(self.NEliteMembers, self.NPopSize):
                if np.random.rand() <= GeneticAlgorithm._prob_mutation:
                    # this member will be mutated
                    if np.random.rand() <= GeneticAlgorithm._prob_small_mutation:
                        # only a small mutation will be applied
                        self.apply_small_mutation(self.pop[i])
                    else:
                        # a more major mutation will occur
                        self.apply_mutation(self.pop[i])

            # convert back to floats
            self.convert_all_to_float()

            # reinitialise the new fitness metric
            self.fitness = np.zeros(self.NPopSize)

    def apply_random_vars(self, nn):
        """
            Set the weights and bias values randomly
        """
        # inputs
        nn.inputs_w = np.random.randint(GeneticAlgorithm._bin_min, GeneticAlgorithm._bin_max, size=(nn.inputs_w.shape[0], nn.inputs_w.shape[1]), dtype=GeneticAlgorithm._bin_dtype)

        # hidden layers
        if nn.h_layers is not None:
            for i,h in enumerate(nn.h_layers_w):
                nn.h_layers_w[i] = np.random.randint(GeneticAlgorithm._bin_min, GeneticAlgorithm._bin_max, size=(h.shape[0], h.shape[1]), dtype=GeneticAlgorithm._bin_dtype)
                nn.h_layers_b[i] = np.random.randint(GeneticAlgorithm._bin_min, GeneticAlgorithm._bin_max, size=(nn.h_layers_b[i].shape[0]), dtype=GeneticAlgorithm._bin_dtype)

        # outputs
        nn.outputs_b = np.random.randint(GeneticAlgorithm._bin_min, GeneticAlgorithm._bin_max, size=(nn.outputs_b.shape[0]), dtype=GeneticAlgorithm._bin_dtype)


    def apply_small_mutation(self, nn):
        """
            Initialise the weights with small mutations of the provdided weights bounded by -1 and 1
        """
        nn.init_type = 'SMALL_MUT'
        # flatten the weights and bias and turn to binary
        arr = self.flatten_nn(nn)
        arr = self.get_bits(arr)

        # determine the byte to flip
        num_bits = len(arr)
        b = np.random.randint(0, num_bits)
        arr[b] = arr[b] ^ True

        # deflatten
        arr = self.get_ints(arr)
        self.deflatten(arr, nn)

    def apply_mutation(self, nn):
        """
            Apply a mutation by flipping a whole byte
        """
        nn.init_type = 'MUT'
        # flatten the weights and bias and turn to binary
        arr = self.flatten_nn(nn)
        arr = self.get_bits(arr)

        # determine the byte to flip
        num_bytes = len(arr) // 8
        b = np.random.randint(0, num_bytes - 1)
        for i in range(b * 8, (b * 8) + 8):
            arr[i] = arr[i] ^ True

        # deflatten
        arr = self.get_ints(arr)
        self.deflatten(arr, nn)

    def apply_crossover(self, parents):
        """
            Initialise the weights by cross breeding the list of parents
        """
        NewMember = NeuralNetwork(self.NInputs, self.NOutputs, self.NHiddenLayerLengths, dtype=GeneticAlgorithm._bin_dtype)
        NewMember.init_type = 'CROSS'

        # flatten the weights and bias values and turn to binary representations
        p = self.flatten_nn(parents[0])
        p_bin = self.get_bits(p)
        m = self.flatten_nn(parents[1])
        m_bin = self.get_bits(m)

        # select the crossover point (prevent singel parent)
        cross_idx = np.random.randint(1, len(p_bin)-1)

        # perform the crossover
        if np.random.rand() < 0.5:
            c_bin = p_bin[0: cross_idx] + m_bin[cross_idx:]
        else:
            c_bin = m_bin[0: cross_idx]+ p_bin[cross_idx:]

        # convert back to int form
        c = self.get_ints(c_bin)
        # turn back into the weights and bias
        self.deflatten(c, NewMember)

        return NewMember

    def flatten_nn(self, nn):
        """
            Flatten the weights and bias in the NN
        """
        flat =  nn.inputs_w.flatten()

        if nn.h_layers_w is not None:
            for i in range(0, len(nn.h_layers_w)):
                flat = np.hstack((flat, nn.h_layers_w[i].flatten()))
                flat = np.hstack((flat, nn.h_layers_b[i].flatten()))

        flat = np.hstack((flat, nn.outputs_b.flatten()))

        return flat

    def deflatten(self, arr, nn):
        """
            Extract the weights and bias from the flat array
        """
        # get the inputs
        c_len = nn.inputs_w.shape[1]
        seg_len = c_len * nn.inputs_w.shape[0]
        for i in range(0, nn.inputs_w.shape[0]):
            nn.inputs_w[i,:] = arr[i * c_len: (i + 1) * c_len]
        arr = arr[seg_len:]
        # get the hidden layers
        if nn.h_layers_w is not None:
            for i,h in enumerate(nn.h_layers_w):
                # take the weights first
                c_len = h.shape[1]
                seg_len = c_len * h.shape[0]
                for ii in range(0, h.shape[0]):
                    nn.h_layers_w[i][ii, :] = arr[ii * c_len: (ii + 1) * c_len]
                arr = arr[seg_len:]
                # take the bias values
                seg_len = nn.h_layers_b[i].shape[0]
                nn.h_layers_b[i][:] = arr[0: seg_len]
                arr = arr[seg_len:]

        # get the ouput bias
        nn.outputs_b[:] = arr

    def get_bits(self, arr):
        """
            Returns a flattened bit representation of the array
        """
        b = bitarray(endian='big')
        b.frombytes(arr.tobytes())
        return b

    def get_ints(self, b_arr):
        """
            Return a flat int array from binary string
        """
        return np.frombuffer(b_arr.tobytes(), dtype=GeneticAlgorithm._bin_dtype)

if __name__ == "__main__":
    import copy
    from src.ga import GeneticAlgorithm
    ga = GeneticAlgorithm(hidden_layer_lens=[4,6])
    ga.create_population(is_first=True)
    ga_orig = copy.deepcopy(ga)
    ga.convert_all_to_binary()
    for nn in ga.pop:
        ga.apply_small_mutation(nn)
    ga.convert_all_to_float()

    for i in range(0, ga.NPopSize):
        nn = ga.pop[i]
        nn_orig = ga_orig.pop[i]
        print(i)
        if not np.array_equal(nn.inputs_w, nn_orig.inputs_w):
            print('Inputs not equal')
            print(nn.inputs_w)
            print(nn_orig.inputs_w)
        for ii in range(0, len(nn.h_layers_w)):
            if not np.array_equal(nn.h_layers_w[ii], nn_orig.h_layers_w[ii]):
                print('H weights not equal')
                print(nn.h_layers_w[ii])
                print(nn_orig.h_layers_w[ii])
            if not np.array_equal(nn.h_layers_b[ii],nn_orig.h_layers_b[ii]):
                print('H bias not equal')
                print(nn.h_layers_b[ii])
                print(nn_orig.h_layers_b[ii])
        if not np.array_equal(nn.outputs_b, nn_orig.outputs_b):
            print('Ouput bias not equal')
            print(nn.outputs_b)
            print(nn_orig.outputs_b)


class IslandGA(GeneticAlgorithm):

    _prob_migration = 0.01
    _migration_pop_max = 0.1

    def __init__(self, max_gens=20, population_size=20, num_inputs=9, num_outputs=3, hidden_layer_lens=[], id=0):
        # run the parent class initialisation
        super(IslandGA, self).__init__( max_gens=max_gens, population_size=population_size, num_inputs=num_inputs, num_outputs=num_outputs, hidden_layer_lens=hidden_layer_lens)
        self.id = id
        self.location = tuple(np.random.rand(2))
        self.NMigrantsMax = int(np.ceil(self.NPopSize * IslandGA._migration_pop_max))

    def set_island_probabilities(self, locations, ids):
        """
            Set the probility of migrating to each island from this one (if we migrate)
        """
        from src.geom import calc_euclid_distance_2d_sq
        self.loc_ids = np.array(ids) # set the ids of the other islands, this needs to match the order of the probabilities

        # determine the probabilites
        self.loc_probs = np.zeros(len(locations))
        dsqs = np.zeros(len(locations))
        for i,l in enumerate(locations):
            dsqs[i] = calc_euclid_distance_2d_sq(self.location, l)
        tot_dsq = np.sum(dsqs)
        self.loc_probs = np.cumsum(dsqs / tot_dsq)

    def handle_immigration(self, pop, fit):
        """
            Handle the migrants
        """
        self.pop.extend(pop)
        self.fitness = np.hstack((self.fitness, fit))

    def handle_migration(self):
        """
            Determine which of the population are going to migrate
        """
        m_idxs = np.sort(np.unique(np.random.randint(0, self.NPopSize, size=(self.NMigrantsMax))))
        f_list = list(self.fitness)
        migrants = []
        f_list_migrants = []
        for i,m in enumerate(m_idxs):
            m -= i # account for the list getting shorter each time we pop a member (if idx 1 is poped then 2 will become one) - works because we sorted the list earlier
            migrants.append(self.pop.pop(m))
            f_list_migrants.append(f_list.pop(m)) # the member takes their fitness with them

        # convert fitness back to numpy
        migrant_fitness = np.array(f_list_migrants)
        self.fitness = np.array(f_list)

        # chose the island to migrate to
        i_rand = np.random.rand()
        for i in range(0, len(self.loc_ids)):
            if i_rand <= self.loc_probs[i]:
                id = self.loc_ids[i]
                break

        return id, migrants, migrant_fitness
