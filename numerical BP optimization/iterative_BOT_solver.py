import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

sys.path.append('../helper functions/')

from general_preprocessing import preprocess_from_topo_to_flows
from helper_fcts import visualise_BOT_solution


def build_A_and_b(coords_arr, num_bps, dim, edges_arr_ext, edges_arr_int, edges_arr_tot, edge_weights_arr_ext,
                  edge_weights_arr_int, edge_weights_arr_tot, sum_separator_ext, sum_separator_tot):
    
    """
    using edgelists and cumulative sums the matrices A and b for the linear system of equations
    can be build in a vectorised numpy way for more efficiency.
    """
    
    # first calculate b in a vectorised fashion:
    # need external edges distances:
    d_ext = np.clip(np.sqrt(np.sum((coords_arr[edges_arr_ext[:, 0]]
                                    - coords_arr[edges_arr_ext[:, 1]])**2, axis=1)), 1e-9, None)
    b_components = (edge_weights_arr_ext / d_ext)[:, None] * coords_arr[edges_arr_ext[:,1]] # (n_ext,1) * (n_ext, dim)

    # now perform the cumulative sum:
    b_cum = np.cumsum(b_components, axis=0)
    b_cum = np.vstack((np.zeros(dim), b_cum)) #add zeros at the very beginning

    # take the appropiate differences to get b:
    b = b_cum[sum_separator_ext[:,1]] - b_cum[sum_separator_ext[:,0]]

    # next calculate A in a vectorised fashion:
    A = np.zeros((num_bps, num_bps))

    # first the diagonal elements:
    # need all edges distances:
    d_tot = np.clip(np.sqrt(np.sum((coords_arr[edges_arr_tot[:,0]]
                                    - coords_arr[edges_arr_tot[:,1]])**2, axis=1)), 1e-9, None)
    diag_components = edge_weights_arr_tot / d_tot #(n_tot,1)

    #cumulative sum and differences:
    diag_cum = np.cumsum(diag_components)
    diag_cum = np.append(np.array([0]), diag_cum) #add zeros at the very beginning
    diag = diag_cum[sum_separator_tot[:, 1]] - diag_cum[sum_separator_tot[:, 0]]
    A[np.arange(num_bps), np.arange(num_bps)] = diag

    # lastly the off-diagonal elements:
    # need all internal distances:
    d_int = np.clip(np.sqrt(np.sum((coords_arr[edges_arr_int[:,0]]
                                    - coords_arr[edges_arr_int[:,1]])**2, axis=1)), 1e-9, None)
    offdiag_components = edge_weights_arr_int / d_int #(n_int,1)
    A[edges_arr_int[:, 0], edges_arr_int[:, 1]] = - offdiag_components

    return A, b


# now build the iterative solver for this topo:
def iterative_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                relative_improvement_threshold=1e-5, min_iterations=30, max_iterations=300,
                plot=False, title="", fov=None, save=False, save_name="img"):
    
    """
    inputs:
    - topo: topology in form of an undirected nx.Graph or a children_dict, e.g. {0:[-1], -1:[3,-2], ...}
            with flow conservation at branching points. BPs must have at least degree 3. Terminals must have 
            degree 1.
    - supply_array, demand_array: containing the supplies of the sources and demands of the sinks in accordance 
    with the order used in the labelling
    - coords_sources, coords_sinks: position coordinates of the sources and of the sinks
    - al: alpha parameter

    - iteration parameters: 
        - relative_improvement_threshold 
        - min_iterations = 30
        - max_iterations = 300

    - visualisation parameters:
        - plot: final solution is visualised using the "visualise_BOT_solution" function
        - title: string to show as title of the plot
        - fov: field of view (2,2)-array, or None then fov is automatically chosen
        - save: boolean to determine if image should be saved with save_name as name

    outputs:
    - cost of minimal BP configuration
    - positions of the BPs
    - 
    """
    

    #use general preprocessing function to calculate the edge-flows:
    topo, coords_arr, list_bp_idx = preprocess_from_topo_to_flows(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al)

    # start iterative optimization of the branching point positions:
    dim = len(coords_arr[0])  # spatial dimensions
    num_terminals = len(supply_arr) + len(demand_arr)
    num_bps = len(list_bp_idx)

    # set up edge list external and internal and corresponding edge weights:
    edges_arr_ext = np.array([], dtype=int)
    edges_arr_int = np.array([], dtype=int)
    edges_arr_tot = np.array([], dtype=int)
    edges_arr_uni = np.array([], dtype=int)
    edge_weights_arr_ext = np.array([])
    edge_weights_arr_int = np.array([])
    edge_weights_arr_tot = np.array([])
    edge_weights_arr_uni = np.array([])
    sum_separator_ext = np.zeros((num_bps, 2), dtype=int)
    sum_separator_tot = np.zeros((num_bps, 2), dtype=int)
    current_separator_ext = 0
    current_separator_tot = 0

    # order the edges in the way they will be summed to parallelise summing with cummulative sum:
    # list_bp_idx is sorted like [-1,-2,-3,..], but
    # process them in the order in which they appear in all_coords -> reverse order of list:
    old_cost = np.inf
    for i, bp in enumerate(list_bp_idx[::-1]):
        sum_separator_ext[i, 0] = current_separator_ext
        sum_separator_tot[i, 0] = current_separator_tot
        for neighbour in nx.neighbors(topo, bp):
            weight = np.abs(topo[bp][neighbour]["weight"]) ** al
            edges_arr_tot = np.append(edges_arr_tot, [bp, neighbour])
            edge_weights_arr_tot = np.append(edge_weights_arr_tot, weight)
            current_separator_tot += 1

            #avoid double counting of internal edges:
            if neighbour > bp:
                edges_arr_uni = np.append(edges_arr_uni, [bp, neighbour])
                edge_weights_arr_uni = np.append(edge_weights_arr_uni, weight)

            if neighbour >= 0:
                # in this case it's an external edge:
                edges_arr_ext = np.append(edges_arr_ext, [bp, neighbour])
                edge_weights_arr_ext = np.append(edge_weights_arr_ext, weight)
                current_separator_ext += 1
            else:
                # in this case it's an internal edge:
                edges_arr_int = np.append(edges_arr_int, [bp, neighbour])
                edge_weights_arr_int = np.append(edge_weights_arr_int, weight)
        sum_separator_ext[i, 1] = current_separator_ext
        sum_separator_tot[i, 1] = current_separator_tot

    edges_arr_ext = edges_arr_ext.reshape((-1, 2))
    edges_arr_int = edges_arr_int.reshape((-1, 2))
    edges_arr_tot = edges_arr_tot.reshape((-1, 2))
    edges_arr_uni = edges_arr_uni.reshape((-1, 2))

    # here, the actual iterations start
    for iteration in range(max_iterations):
        if num_bps == 0:
            # special case where no BPs are left after deleting zero flow edges.
            cost = 0.
            for edge in topo.edges():
                cost += np.abs(topo[edge[0]][edge[1]]["weight"])**al * \
                        np.sqrt(np.sum(coords_arr[edge[0]] - coords_arr[edge[1]])**2)
            break
        if iteration == max_iterations - 1:
            print(f"maximum number of iterations {max_iterations} was reached.")

        # get matrices A and b for linear system of equations:
        A, b = build_A_and_b(coords_arr, num_bps, dim, edges_arr_ext, edges_arr_int, edges_arr_tot,
                             edge_weights_arr_ext, edge_weights_arr_int, edge_weights_arr_tot, sum_separator_ext,
                             sum_separator_tot)

        # solve this system of equations using the numpy solver:
        coords_arr[num_terminals:, :] = np.array(np.linalg.solve(A, b))

        # calculate the cost and relative improvement:
        if iteration > min_iterations:
            # parallelised way of calculating the cost:
            cost = np.sum(edge_weights_arr_uni *
                        np.sqrt(np.sum((coords_arr[edges_arr_uni[:, 0]] - coords_arr[edges_arr_uni[:, 1]])**2, axis=1)))

            # calculate relative improvement in cost:
            relative_improvement = (old_cost - cost) / cost
            old_cost = cost

            if relative_improvement < relative_improvement_threshold:
                #print(f"Finished after {iteration} iterations.")
                break

    if plot and dim == 2:
        # visualise the final solution:
        visualise_BOT_solution(topo, coords_arr, supply_arr, demand_arr, title=title, fov=fov, save=save, save_name=save_name)

    return cost, coords_arr
