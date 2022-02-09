import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import sys
import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiporcessing
import time
from tqdm import tqdm

sys.path.append('ready functions in py/')

from helper_fcts import *
from iterative_BOT_solver import iterative_bot_solver

def wrapper_fct_for_multithreading(bot_problem_dict, topo_list):

    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    cost_arr = np.zeros(len(topo_list))
    best_topo = None
    best_cost = np.inf
    for i,topo in enumerate(topo_list):
        cost, _ = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                             relative_improvement_threshold=1e-6, min_iterations=-1,
                                             max_iterations=2000,
                                             plot=False, title="", fov=None, save=False, save_name="img")

        cost_arr[i] = cost
        if cost < best_cost:
            best_cost = cost
            best_topo = topo.copy()

    return [best_topo, best_cost, cost_arr]

if __name__ == '__main__':
    N = 100
    num_threads = int(input("num of threads="))
    terminals_list = [4, 5, 6, 7, 8, 9]

    # store them in a pickle file:
    pkl_file_path = f"brute-force_results{N}.pkl"
    data_dict = {}
    output = open(pkl_file_path, 'wb')
    pickle.dump(data_dict, output)
    output.close()

    count = 0

    for k in range(N):
        print(f"started run {k}")

        for num_terminals in terminals_list:
            num_topos = np.prod(np.arange(1, 2 * num_terminals - 3)[::2])

            # generate random bot problem:
            num_sources = np.random.randint(1, num_terminals)
            num_sinks = num_terminals - num_sources
            bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2,
                                                           max_length=1.)

            # generate all topos:
            all_topos = get_random_topos_sparsely(bot_problem_dict, num_topos)
            all_costs = np.zeros(num_topos)
            all_topos_list = list(all_topos.values())

            topos_per_thread = int(num_topos/ num_threads)
            joint_topo_list = []
            for i in range(num_threads):
                if i == num_threads - 1:
                    joint_topo_list.append(all_topos_list[i * topos_per_thread: num_topos])
                else:
                    joint_topo_list.append(all_topos_list[i * topos_per_thread: (i + 1) * topos_per_thread])

            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(wrapper_fct_for_multithreading, itertools.repeat(bot_problem_dict),
                                       joint_topo_list)
                result_list = list(results)

            overall_best_topo = None
            overall_best_cost = np.inf
            overall_cost_arr = np.array([])
            for result in result_list:
                best_topo, best_cost, cost_arr = result

                overall_cost_arr = np.append(overall_cost_arr, cost_arr)
                if best_cost < overall_best_cost:
                    overall_best_cost = best_cost
                    overall_best_topo = best_topo.copy()

            # read pickle:
            pkl_file = open(pkl_file_path, 'rb')
            data_dict = pickle.load(pkl_file)
            pkl_file.close()

            # append:
            data_dict[count] = {"bot_problem_dict":bot_problem_dict,
                                "overall_best_topo":overall_best_topo,
                                "overall_best_cost":overall_best_cost,
                                "overall_cost_arr":overall_cost_arr
                                }
            print(f"completed {count}.")
            count += 1

            # store them in a pickle file:
            output = open(pkl_file_path, 'wb')
            pickle.dump(data_dict, output)
            output.close()
