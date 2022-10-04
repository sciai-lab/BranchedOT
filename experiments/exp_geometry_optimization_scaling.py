import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiporcessing
import pickle
import copy

sys.path.append('../../ready functions in py/')

from helper_fcts import generate_random_bot_problem, generate_random_binary_tree_topo
from fast_optimizer import fast_optimize

num_dims_arr = [2, 3, 4, 5, 10, 30, 50, 100]
#num_dims_arr = [10, 30, 50, 100]
threshold_arr = [1e-6, 1e-7, 1e-8]
alpha_arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
num_terminals_arr = np.array([10,20,30,50,70,100,150,200,300,400,500,600,700,800,900,1000])
#num_terminals_arr = np.array([10,20])
num_problems = 1000
num_threads = 10

def wrapper_fct_for_multithreading(dim, num_terminals, seed_list):
    T = 0  # zero temperature MC
    results = {}

    for i, seed in enumerate(seed_list):
        np.random.seed(seed)
        num_sources = np.random.randint(1, num_terminals)
        num_sinks = num_terminals - num_sources
        bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=dim,
                                                       max_length=1.)

        al = bot_problem_dict["al"]
        coords_sources = bot_problem_dict["coords_sources"]
        coords_sinks = bot_problem_dict["coords_sinks"]
        supply_arr = bot_problem_dict["supply_arr"]
        demand_arr = bot_problem_dict["demand_arr"]

        # use random tree as init:
        topo = generate_random_binary_tree_topo(len(supply_arr) + len(demand_arr))
        results[i] = {}
        for threshold in threshold_arr:
            costs_wrt_al = np.zeros(len(alpha_arr))
            iters_wrt_al = np.zeros(len(alpha_arr))
            for j, al in enumerate(alpha_arr):
                _, cost, coords, num_iters = fast_optimize(topo, supply_arr, demand_arr, coords_sources,
                                                                    coords_sinks, al,
                                                                    improv_threshold=threshold)
                costs_wrt_al[j] = cost
                iters_wrt_al[j] = num_iters

            results[i][threshold] = {
            #"topo":topo,
            #"coords":coords, "bot_problem_dict":bot_problem_dict,
            "cost":np.mean(costs_wrt_al),
            "num_iters":np.mean(iters_wrt_al)
            }
            print(f"dim={dim}: completed problem {i} with threshold={threshold} after {num_iters} iterations.")

    return results

if __name__ == '__main__':
    for dim in num_dims_arr:
        for num_terminals in num_terminals_arr:
            probs_per_thread = int(num_problems / num_threads)
            list_of_seedlists = []
            for k in range(num_threads):
                list_of_seedlists.append(list(np.random.randint(100000, size=probs_per_thread)))

            #print(list_of_seedlists)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(wrapper_fct_for_multithreading, itertools.repeat(dim),
                                       itertools.repeat(num_terminals),
                                            list_of_seedlists)
                result_list = list(results)

            #print(result_list)

            # store the results in a pickle file:
            pkl_file_path = f"tmp_output/Smith_alpha_dim{dim}_probs{num_problems}_size{num_terminals}_new.pkl"
            output = open(pkl_file_path, 'wb')
            pickle.dump(result_list, output)
            output.close()
