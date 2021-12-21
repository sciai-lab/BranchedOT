import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiporcessing
import pickle

sys.path.append('../helper functions/')
sys.path.append('../numerical BP optimization/')


from mc_update import monte_carlo_step
from helper_fcts import generate_random_bot_problem
from iterative_BOT_solver import iterative_bot_solver

def wrapper_fct_for_multithreading(num_terminals, seed_list):
    T = 0  # zero temperature MC
    results = {}

    for i,seed in enumerate(seed_list):
        np.random.seed(seed)
        num_sources = np.random.randint(1, num_terminals)
        num_sinks = num_terminals - num_sources
        bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2,
                                                       max_length=1.)

        al = bot_problem_dict["al"]
        coords_sources = bot_problem_dict["coords_sources"]
        coords_sinks = bot_problem_dict["coords_sinks"]
        supply_arr = bot_problem_dict["supply_arr"]
        demand_arr = bot_problem_dict["demand_arr"]

        # init star graph:
        topo = nx.Graph()
        for node in range(len(supply_arr) + len(demand_arr)):
            topo.add_edge(-1, node)

        cost, coords_iter = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                 relative_improvement_threshold=1e-6, min_iterations=-1,
                                                 max_iterations=1000,
                                                 plot=False, title="", fov=None, save=False, save_name="img")

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
                                            coords_iter, bot_problem_dict, temperature=T)
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

        results[i] = {
        "final_topo":topo, "cost_arr":cost_arr, "bot_problem_dict":bot_problem_dict,
        "iter_till_full_tree":iterations_till_full,
        "iter_till_converged":iteration - nx.number_of_edges(topo)
        }
        print(f"completed problem {i} after {iteration} iterations.")

        # plt.plot(np.arange(len(cost_arr)), cost_arr)
        # plt.show()
        # _ = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
        #                      relative_improvement_threshold=1e-6, min_iterations=-1,
        #                      max_iterations=1000,
        #                      plot=True, title="", fov=None, save=False, save_name="img")

    return results

if __name__ == '__main__':
    num_terminals = int(input("num terminals="))
    num_problems = int(input("num of problems="))
    num_threads = int(input("num of threads="))

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
    pkl_file_path = f"MC_star_probs{num_problems}_size{num_terminals}_new.pkl"
    output = open(pkl_file_path, 'wb')
    pickle.dump(result_list, output)
    output.close()
