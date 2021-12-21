import numpy as np
import sys
import pickle
import time

sys.path.append('../helper functions/')
sys.path.append('../numerical BP optimization/')


from angular_stress_heuristic import *
from helper_fcts import *
from interpolating_prior import interpolated_prior
from mc_update import monte_carlo_step
from iterative_BOT_solver import iterative_bot_solver

def tilde_beta(al):
    d = 0.38
    b = 4* d - 3
    c = 1
    a = (-1 - b)

    return a*al**2 + b*al + c

if __name__ == '__main__':
    T = 0
    num_terminals = 300
    fix_al = 0.5

    # generate a random problem:
    num_sources = np.random.randint(1, num_terminals)
    num_sinks = num_terminals - num_sources
    bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2,
                                                   max_length=1.)

    bot_problem_dict["al"] = fix_al
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    time0 = time.time()

    # stress reduction with learnable prior:
    beta = tilde_beta(al)
    int_prior_topo = interpolated_prior(bot_problem_dict, beta)
    stress_cost, stress_topo = angular_stress_reduction(int_prior_topo, bot_problem_dict, plot_final=False,
                                                        plot=False)

    print("stress completed.")


    # stress heuristic with MC continued:
    topo = stress_topo.copy()
    cost, coords_iter = iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                             relative_improvement_threshold=1e-6, min_iterations=-1,
                                             max_iterations=1000,
                                             plot=False, title="", fov=None, save=False, save_name="img")
    keep_going = True
    iteration = 0
    sample_edge_list = list(topo.edges())
    while keep_going and iteration < 60000:
        iteration += 1
        topo, sample_edge_list, cost, coords_iter, accepted = monte_carlo_step(topo, sample_edge_list, cost,
                                                                               coords_iter, bot_problem_dict,
                                                                               temperature=T)
        # check if converged:
        if len(sample_edge_list) == 0:
            # this means that we have stuck with one topology for an entire run through.
            keep_going = False

    print(f"final result, after {iteration} iterations and {np.round(time.time() - time0, 2)} seconds.")

    # store the results in a pickle file:
    result_dict = {"bot_problem_dict":bot_problem_dict,
                   "topo":topo,
                   "iteration":iteration
                   }
    pkl_file_path = f"combined_heuristic2_size{num_terminals}_al{np.round(al,2)}.pkl"
    output = open(pkl_file_path, 'wb')
    pickle.dump(result_dict, output)
    output.close()
