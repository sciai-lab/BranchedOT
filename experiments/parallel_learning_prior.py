import concurrent.futures  # for multiprocessing
import itertools  # to repeat constant arguments in the map of multiporcessing
import pickle
import sys

sys.path.append('../src/')
sys.path.append('../numerical BP optimization/')


from src.interpolating_prior import interpolated_prior
from src.angular_stress_heuristic import *
from helper_fcts import *
from src.iterative_geometry_solver import iterative_geometry_solver


def wrapper_fct_for_multithreading(num_terminals, num_al, num_beta, num_problems, seed, list_indices_for_al_arr):
    np.random.seed(seed * list_indices_for_al_arr[0])
    al_arr = np.linspace(0.0001, 0.9999, num_al)
    beta_arr = np.linspace(0.0001, 0.9999, num_beta)

    score_mat = np.zeros((len(al_arr), len(beta_arr)))
    score_stress_mat = np.zeros((len(al_arr), len(beta_arr)))

    for i,al in enumerate(al_arr):
        if i % 5 == 0:
            print(f"al {np.round(al,2)} was reached.")
        for k in range(num_problems):  # calculate each two problems

            # generate a problem:
            num_sources = np.random.randint(1, num_terminals)
            num_sinks = num_terminals - num_sources
            bot_problem_dict = generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2,
                                                           max_length=1.)
            bot_problem_dict["al"] = al

            al = bot_problem_dict["al"]
            coords_sources = bot_problem_dict["coords_sources"]
            coords_sinks = bot_problem_dict["coords_sinks"]
            supply_arr = bot_problem_dict["supply_arr"]
            demand_arr = bot_problem_dict["demand_arr"]

            for j, beta in enumerate(beta_arr):
                # generate interpolated prior:
                int_prior_topo = interpolated_prior(bot_problem_dict, beta)
                cost, _ = iterative_geometry_solver(int_prior_topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                    relative_improvement_threshold=1e-6, min_iterations=30,
                                                    max_iterations=1000,
                                                    plot=False, title="", fov=None, save=False, save_name="img")
                score_mat[i, j] += cost

                # now do the stress reduction and find the cost for this:
                cost_reduced, reduced_topo = angular_stress_reduction(int_prior_topo, bot_problem_dict,
                                                                      plot_final=False, plot=False)
                score_stress_mat[i, j] += cost_reduced

    return score_mat, score_stress_mat

if __name__ == '__main__':
    # parallelized systematic studies without the stress reduction: find the optimal beta as a function of alpha.
    seed = int(input("random seed="))
    num_al = int(input("num for alpha grid="))
    num_beta = int(input("num for beta grid="))
    num_terminals = int(input("num terminals="))
    num_problems = int(input("num of problems="))
    num_threads = int(input("num of threads="))

    al_per_thread = int(num_al / num_threads)
    joint_list_indices_for_al_arr = []
    for i in range(num_threads):
        if i == num_threads - 1:
            joint_list_indices_for_al_arr.append(list(np.arange(i * al_per_thread, num_al)))
        else:
            joint_list_indices_for_al_arr.append(list(np.arange(i * al_per_thread, (i + 1) * al_per_thread)))

    print(joint_list_indices_for_al_arr)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(wrapper_fct_for_multithreading, itertools.repeat(num_terminals),
                               itertools.repeat(num_al), itertools.repeat(num_beta),
                               itertools.repeat(num_problems), itertools.repeat(seed),
                               joint_list_indices_for_al_arr)
        result_list = list(results)

    score_mat = np.zeros((num_al, num_beta))
    score_stress_mat = np.zeros((num_al, num_beta))
    for i, _ in enumerate(result_list):
        A, B = result_list[i]
        score_mat += A
        score_stress_mat += B

    # store them in a pickle file:
    pkl_file_path = f"{num_al}x{num_beta}_a{num_problems}probs_size{num_terminals}.pkl"
    data_dict = {"score_mat": score_mat, "score_stress_mat": score_stress_mat}
    output = open(pkl_file_path, 'wb')
    pickle.dump(data_dict, output)
    output.close()
