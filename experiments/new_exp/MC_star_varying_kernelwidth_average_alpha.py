import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiporcessing
import pickle
import copy

sys.path.append('../../ready functions in py/')

from mc_update import monte_carlo_step
from helper_fcts import generate_random_bot_problem, generate_random_binary_tree_topo
from iterative_BOT_solver import iterative_bot_solver

kernel_width_arr = np.array([0.1, 0.5, 1, 2, 3, 4, 5, 10])
#num_terminals_arr = np.array([10, 20, 30, 50, 70, 100, 150])
num_terminals_arr = np.array([30, 50, 70])
alpha_arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
num_problems = 50
num_threads = 10

def wrapper_fct_for_multithreading(num_terminals, seed_list):
    T = 0  # zero temperature MC
    results = {}

    for i, seed in enumerate(seed_list):
        np.random.seed(seed)
        num_sources = np.random.randint(1, num_terminals)
        num_sinks = num_terminals - num_sources
        bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2,
                                                       max_length=1.)

        all_costs_arr = np.zeros((len(alpha_arr), len(kernel_width_arr)))
        all_iters_arr = np.zeros((len(alpha_arr), len(kernel_width_arr)))
        for j,al in enumerate(alpha_arr):
            bot_problem_dict["al"] = al
            coords_sources = bot_problem_dict["coords_sources"]
            coords_sinks = bot_problem_dict["coords_sinks"]
            supply_arr = bot_problem_dict["supply_arr"]
            demand_arr = bot_problem_dict["demand_arr"]

            # init star graph:
            # topo = nx.Graph()
            # for node in range(len(supply_arr) + len(demand_arr)):
            #     topo.add_edge(-1, node)

            # use random tree instead as init:
            topo = generate_random_binary_tree_topo(len(supply_arr) + len(demand_arr))

            cost, coords_iter = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                     relative_improvement_threshold=1e-6, min_iterations=-1,
                                                     max_iterations=1000,
                                                     plot=False, title="", fov=None, save=False, save_name="img")

            topo_init = copy.deepcopy(topo)
            cost_init = cost
            coords_iter_init = coords_iter.copy()

            results[i] = {}
            for k,kernel_width in enumerate(kernel_width_arr):

                # restore initial setup:
                topo = copy.deepcopy(topo_init)
                cost = cost_init
                coords_iter = coords_iter_init.copy()

                # MC iterations:
                iterations_till_full = 0
                full = False
                cost_arr = np.array([])
                keep_going = True
                iteration = 0
                sample_edge_list = list(topo.edges())
                while keep_going and iteration < 60000:
                    iteration += 1
                    topo, sample_edge_list, cost, coords_iter, accepted = monte_carlo_step(topo, sample_edge_list, cost,
                                                    coords_iter, bot_problem_dict, temperature=T, kernel_width=kernel_width)
                    cost_arr = np.append(cost_arr, cost)

                    # check if tree is full:
                    if nx.number_of_nodes(topo) == 2 * (len(supply_arr) + len(demand_arr)) - 3 and not full:
                        iterations_till_full = iteration
                        full = True

                    # check if converged:
                    if len(sample_edge_list) == 0:
                        # this means that we have stuck with one topology for an entire run through.
                        keep_going = False

                    if iteration >= 60000:
                        print(f"Iteration {iteration} reached.")

                all_costs_arr[j,k] = cost
                all_iters_arr[j,k] = iteration - nx.number_of_edges(topo)

        results[i] = {
        #"final_topo":topo,
        "average_costs_wrt_width":np.mean(all_costs_arr, axis=0),  # average over alpha
        #"bot_problem_dict":bot_problem_dict,
        #"iter_till_full_tree":iterations_till_full,
        "average_iters_wrt_width":np.mean(all_iters_arr, axis=0)  # average over alpha
        }
        print(f"completed problem {i}.")

    return results

if __name__ == '__main__':
    for num_terminals in num_terminals_arr:
        probs_per_thread = int(num_problems / num_threads)
        list_of_seedlists = []
        for k in range(num_threads):
            list_of_seedlists.append(list(np.random.randint(100000, size=probs_per_thread)))

        #print(list_of_seedlists)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(wrapper_fct_for_multithreading, itertools.repeat(num_terminals),
                                        list_of_seedlists)
            result_list = list(results)

        #print(result_list)

        # store the results in a pickle file:
        pkl_file_path = f"tmp_output/MC_star_kernelwidth_alpha_probs{num_problems}_size{num_terminals}_new.pkl"
        output = open(pkl_file_path, 'wb')
        pickle.dump(result_list, output)
        output.close()
