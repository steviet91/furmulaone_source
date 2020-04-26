from src.vehicle import Vehicle
from src.visualiser import Vis
from src.track import TrackHandler
from src.ga import GeneticAlgorithm
from src.ga import IslandGA
from src.lidar import Lidar
from sandbox.game_pad_inputs import GamePad
import numpy as np
from time import sleep
import time
import multiprocessing
from joblib import Parallel, delayed
import copy

# ####################
# NN OUTPUT HANDLING #
# ####################
def update_nn_outputs(i, outputs, nn):
    """
        Assumes outputs are [Thr/Brake, Steering]
    """

    thr_brake = max(0.0, min(1.0, nn.outputs[0])) * 2 - 1
    if thr_brake >= 0:
        outputs[0, i] = thr_brake
        outputs[1, i] = 0.0
    else:
        outputs[0, i] = 0.0
        outputs[1, i] = thr_brake
    outputs[2, i] = (max(0.0, min(1.0, nn.outputs[1])) * 2 - 1) * 360.0

# ###################
# NN INPUT HANDLING #
# ###################
def update_nn_inputs(i, inputs, veh):
    """
        Fixes the order of the inputs
        Lidar is scaled 0-1 - out of range lidar is set to max distance.
    """
    collArr = np.hstack((veh.lidar_front.collision_array, veh.lidar_left.collision_array, veh.lidar_right.collision_array))
    collArr[np.where(collArr < 0)[0]] = Lidar._xLidarRange
    collArr = collArr / Lidar._xLidarRange
    inputs[:, i] = collArr

# ##################
# ALIVE DEFINITION #
# ##################
def check_alive_state(i, is_alive, veh, t_stationary, tSim):
    """
        Update the alive state of a living car
    """
    # check how long a car has been 'stationary' for
    if veh.vVehicle > 2 and t_stationary[i] is not None:
        t_stationary[i] = None
    elif veh.vVehicle <= 2 and t_stationary[i] is None:
        t_stationary[i] = tSim
    if t_stationary[i] is None:
        bHasStopped = False
    else:
        if tSim - t_stationary[i] > 5:
            bHasStopped = True
        else:
            bHasStopped = False
    if veh.bHasCollided or veh.bMovingBackwards or veh.NLapsComplete > 0 or bHasStopped:
        is_alive[i] = False

# ####################
# FITNESS DEFINITION #
# ####################
def calc_fitness(veh, max_steer):
    """
        Set the fitness of the vehicle
    """
    if max_steer < 10:
        return 0.0
    else:
        return veh.rLapProgress + 1.0 * veh.NLapsComplete

def run_sim(nns, task_rate, track, num_inputs):
    """
        Run the simulation for this set of nns
    """
    vehs, is_alive, nn_inputs, nn_outputs, t_stationary = init_pop(len(nns), task_rate, track, num_inputs)
    tSim = 0.0
    max_steer = np.zeros(len(nns)) # log the maximum steering angle applied
    while any(is_alive):
        for i in range(0,len(is_alive)):
            if is_alive[i]:
                nns[i].update_network(nn_inputs[:, i])
                update_nn_outputs(i, nn_outputs, nns[i])
                max_steer[i] = max(max_steer[i], abs(nn_outputs[2, i]))
                vehs[i].update(nn_outputs[0, i], nn_outputs[1, i], nn_outputs[2, i])
                update_nn_inputs(i, nn_inputs, vehs[i])
                check_alive_state(i, is_alive, vehs[i], t_stationary, tSim)
        tSim += task_rate
        if tSim > 10 * 60.0:
            break
    # determine and return the fitness
    f_list = [calc_fitness(v, max_steer[i]) for i,v in enumerate(vehs)]
    return f_list

def init_pop(pop_size, task_rate, track, num_inputs):

    vehs = [Vehicle(i, track, 60, 60, 60, auto_reset=False, task_rate=task_rate) for i in range(0, pop_size)]
    is_alive = [True for i in range(0, pop_size)]
    nn_inputs = np.zeros((num_inputs, pop_size))
    nn_outputs = np.zeros((3, pop_size))
    t_stationary = [None] * pop_size
    return vehs, is_alive, nn_inputs, nn_outputs, t_stationary

# ######
# MAIN #
# ######
def main():
    # instantiate the objects
    track = TrackHandler('dodec_track')
    gp = GamePad()
    run_game = True
    pop_size = 1200
    num_inputs = Lidar._NRays * 3
    gen_number = 0
    max_gens = 500
    task_rate = 0.1
    num_car_render = 50
    num_cores = multiprocessing.cpu_count()
    use_islands = True
    use_parallel = True or use_islands # islands need the cpus


    if use_islands:
        isl_pop_size = int(np.ceil(pop_size / num_cores))
        fit_data = np.zeros((max_gens, num_cores))
        migr_data = np.zeros(4)
    if use_parallel:
        sims_per_core = pop_size / num_cores
        core_sim_idxs = []
        for i in range(0,num_cores):
            core_sim_idxs.append((int(np.ceil(sims_per_core * i)), int(np.ceil(sims_per_core * (i + 1)))))
        print('Parallel Mode [ENABLED]')
    else:
        print('Parallel Mode [DISABLED]')

    if use_islands:
        print('Islands [ENABLED]')
        gas = [IslandGA(id=i, max_gens=max_gens, population_size=isl_pop_size, num_inputs=num_inputs, num_outputs=2, hidden_layer_lens=[6, 4]) for i in range(0,num_cores)]
        for i,ga in enumerate(gas):
            locs = [gas[ii].location for ii in range(0, num_cores) if ii != i]
            ids = [gas[ii].id for ii in range(0, num_cores) if ii != i]
            ga.set_island_probabilities(locs, ids)
            ga.create_population(is_first=True)
    else:
        ga = GeneticAlgorithm(max_gens=max_gens, population_size=pop_size, num_inputs=num_inputs, num_outputs=2, hidden_layer_lens=[6, 4])
        ga.create_population(is_first=True)

    tTotal = time.time()

    while run_game:

        gen_number += 1
        tSim = 0
        t = time.time()
        if use_islands:
            f_list = Parallel(n_jobs=num_cores)(delayed(run_sim)(gas[i].pop, task_rate, track, num_inputs) for i in range(0,num_cores))
            for i,f in enumerate(f_list):
                gas[i].fitness = np.array(f)
        else:
            if use_parallel:
                f_list = Parallel(n_jobs=num_cores)(delayed(run_sim)(ga.pop[x:y], task_rate, track, num_inputs) for x,y in core_sim_idxs)
                fitness_list = []
                for f in f_list:
                    fitness_list.extend(f)

            else:
                vehs, is_alive, nn_inputs, nn_outputs, t_stationary = init_pop(pop_size, task_rate, track, num_inputs)
                while any(is_alive):
                    tSim += task_rate
                    # update the living populus
                    for i in range(0, pop_size):
                        if is_alive[i]:
                            ga.pop[i].update_network(nn_inputs[:, i])
                            update_nn_outputs(i,nn_outputs, ga.pop[i])
                            vehs[i].update(nn_outputs[0, i], nn_outputs[1, i], nn_outputs[2, i])
                            update_nn_inputs(i, nn_inputs, vehs[i])
                            check_alive_state(i, is_alive, vehs[i], t_stationary, tSim)
                # calculate the fitness
                fitness_list = [calc_fitness(v) for v in vehs]

            ga.fitness = np.array(fitness_list)

        if use_islands:
            print('Gen -',gen_number, 'after - {:.3f} s'.format(time.time()-t))
            f_string = ''
            for i in range(0, num_cores):
                if i > 0:
                    f_string += ' - '
                f_string += 'ID {} @ {:.5f}'.format(i, float(max(gas[i].fitness)))
                fit_data[gen_number,i] = float(max(gas[i].fitness))
            print(f_string)
        else:
            print('Gen -',gen_number,'Max Fitness {:.5f}'.format(float(max(ga.fitness))),'after - {:.3f} s'.format(time.time()-t))

        if gp.quit_requested:
            print("Quitting...")
            run_game = False
            gp.exit_thread()
            break

        if not run_game:
            break

        if gen_number >= max_gens:
            # just quit the game
            run_game = False
        else:
            if use_islands:
                # run any migration
                migrants = []
                ids = []
                fitness = []
                # first collect up the migrants - only migrant once they are all collected
                for i in range(0, num_cores):
                    p = np.random.rand()
                    if p < IslandGA._prob_migration:
                        id, m, f = gas[i].handle_migration()
                        print('Island {0} is migrating {1} members to island {2}'.format(i, len(m), id))
                        migr_data = np.vstack((migr_data, np.array([gen_number, i, id, len(m)])))
                        migrants.append(m)
                        fitness.append(f)
                        ids.append(id)

                # perform the immigration
                if len(ids) > 0:
                    for i,id in enumerate(ids):
                        gas[id].handle_immigration(migrants[i], fitness[i])

                # create the new populations
                for ga in gas:
                    ga.create_population()
            else:
                # prepare the next generation
                ga.create_population()

    # save the fitessed
    if use_islands:
        print('Programme completed in {:.3f} s'.format(time.time()-tTotal))
        for i in range(0, num_cores):
            gas[i].pop[int(np.argmax(gas[i].fitness))].pickle_nn(suffix='id_{}'.format(i))
            print('Island {} s with a max fitness of {:.5f}'.format(i, float(max(gas[i].fitness))))
        np.savetxt('fit_data.csv', fit_data, delimiter=',')
        np.savetxt('migr_data.csv', migr_data, delimiter=',')
    else:
        ga.pop[int(np.argmax(ga.fitness))].pickle_nn()
        print('Programme completed in {:.3f} s with a max fitness of {:.5f}'.format(time.time()-tTotal, float(max(ga.fitness))))

if __name__ == "__main__":
    main()
    sleep(5)
