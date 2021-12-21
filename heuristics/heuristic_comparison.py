import numpy as np
import networkx as nx
import sys
import pickle
import concurrent.futures  # for multiprocessing

sys.path.append('../helper functions/')
sys.path.append('../numerical BP optimization/')


from angular_stress_heuristic import *
from helper_fcts import *
from incremental_growth import incremental_growth_heuristic
from interpolating_prior import interpolated_prior
from mc_update import monte_carlo_step
from iterative_BOT_solver import iterative_bot_solver

def tilde_beta(al):
    d = 0.38
    b = 4* d - 3
    c = 1
    a = (-1 - b)

    return a*al**2 + b*al + c

def wrapper_fct_for_multithreading(thread_dict):
    T = 0
    results_dict = {}
    for key in thread_dict:
        print("key:", key)
        bot_problem_dict = thread_dict[key]["bot_problem_dict"]
        al = bot_problem_dict["al"]
        coords_sources = bot_problem_dict["coords_sources"]
        coords_sinks = bot_problem_dict["coords_sinks"]
        supply_arr = bot_problem_dict["supply_arr"]
        demand_arr = bot_problem_dict["demand_arr"]

        MC_cost = thread_dict[key]["cost_arr"][-1]
        MC_topo = thread_dict[key]["final_topo"]
        MC_iterations = thread_dict[key]["iter_till_converged"]

        # now apply the two heuristics:
        # 1) incremental growth heuristic:
        m = 5
        growth_cost, growth_topo = incremental_growth_heuristic(bot_problem_dict, m, plot=False, final_plot=False)

        # 2) stress reduction with learnable prior:
        beta = tilde_beta(al)
        int_prior_topo = interpolated_prior(bot_problem_dict, beta)
        stress_cost, stress_topo = angular_stress_reduction(int_prior_topo, bot_problem_dict, plot_final=False,
                                                            plot=False)

        # 3) incremental growth with MC continued:
        # topo = growth_topo.copy()
        # cost, coords_iter = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
        #                                          relative_improvement_threshold=1e-6, min_iterations=-1,
        #                                          max_iterations=1000,
        #                                          plot=False, title="", fov=None, save=False, save_name="img")
        # keep_going = True
        # iteration = 0
        # sample_edge_list = list(topo.edges())
        # while keep_going and iteration < 60000:
        #     iteration += 1
        #     topo, sample_edge_list, cost, coords_iter, accepted = monte_carlo_step(topo, sample_edge_list, cost,
        #                                                                            coords_iter, bot_problem_dict,
        #                                                                            temperature=T)
        #     # check if converged:
        #     if len(sample_edge_list) == 0:
        #         # this means that we have stuck with one topology for an entire run through.
        #         keep_going = False
        #
        # growth_topo_MC = topo.copy()
        # growth_cost_MC = cost
        # growth_MC_iterations = iteration - nx.number_of_edges(topo)

        # 4) stress heuristic with MC continued:
        # topo = stress_topo.copy()
        # cost, coords_iter = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
        #                                          relative_improvement_threshold=1e-6, min_iterations=-1,
        #                                          max_iterations=1000,
        #                                          plot=False, title="", fov=None, save=False, save_name="img")
        # keep_going = True
        # iteration = 0
        # sample_edge_list = list(topo.edges())
        # while keep_going and iteration < 60000:
        #     iteration += 1
        #     topo, sample_edge_list, cost, coords_iter, accepted = monte_carlo_step(topo, sample_edge_list, cost,
        #                                                                            coords_iter, bot_problem_dict,
        #                                                                            temperature=T)
        #     # check if converged:
        #     if len(sample_edge_list) == 0:
        #         # this means that we have stuck with one topology for an entire run through.
        #         keep_going = False
        #
        # stress_topo_MC = topo.copy()
        # stress_cost_MC = cost
        # stress_MC_iterations = iteration - nx.number_of_edges(topo)

        results_dict[key] = {"bot_problem_dict": bot_problem_dict,
                        "MC_topo": MC_topo,
                        #"growth_topo": growth_topo,
                        "stress_topo": stress_topo,
                        #"growth_topo_MC":growth_topo_MC,
                        "stress_topo_MC":stress_topo_MC,
                        "MC_cost": MC_cost,
                        #"growth_cost": growth_cost,
                        "stress_cost": stress_cost,
                        #"growth_cost_MC":growth_cost_MC,
                        "stress_cost_MC": stress_cost_MC,
                        "MC_iterations":MC_iterations,
                        #"growth_MC_iterations":growth_MC_iterations,
                        "stress_MC_iterations":stress_MC_iterations
                        }
    return results_dict

if __name__ == '__main__':
    num_terminals = int(input("num terminals="))

    # load data:
    pkl_file_path = f"MC_star_probs150_size{num_terminals}_new.pkl"
    pkl_file = open(pkl_file_path, 'rb')
    list_thread_dict = pickle.load(pkl_file)
    pkl_file.close()

    print(f"{len(list_thread_dict)} threads required.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(wrapper_fct_for_multithreading, list_thread_dict)
        result_list = list(results)

    # print(result_list)

    # store the results in a pickle file:
    pkl_file_path = f"heuristic_comparison_size{num_terminals}_redone.pkl"
    output = open(pkl_file_path, 'wb')
    pickle.dump(result_list, output)
    output.close()

