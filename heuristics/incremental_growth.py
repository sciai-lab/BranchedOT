import sys

sys.path.append('../helper functions/')
sys.path.append('../numerical BP optimization/')

from iterative_BOT_solver import iterative_bot_solver
from helper_fcts import *


# implement incremental growth heuristic.
def incremental_growth_heuristic(bot_problem_dict, m,
                                 plot=False, final_plot=False, fov=None, save=False):
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]
    dim = len(coords_sources[0])

    # switch source and sink if sink has larger demand then largest source:
    if np.max(supply_arr) < np.max(demand_arr):
        coords_sources_o = np.copy(coords_sources)
        supply_arr_o = np.copy(supply_arr)
        coords_sources = np.copy(coords_sinks)
        supply_arr = np.copy(demand_arr)
        coords_sinks = coords_sources_o
        demand_arr = supply_arr_o

    list_source_idx = list(np.arange(len(supply_arr)))
    list_sink_idx = list(np.arange(len(supply_arr), len(supply_arr) + len(demand_arr)))

    # start from largest source:
    start_source = np.argmax(supply_arr)
    start_source_coords = coords_sources[start_source]

    # sort other terminals accroding to the distance:
    ext_coords = np.vstack((coords_sources, coords_sinks))

    dist_source_terminals = np.sqrt(np.sum((start_source_coords - ext_coords) ** 2, axis=1))
    index_order_arr = np.argsort(dist_source_terminals)

    # closest_sink:
    mask = (index_order_arr >= len(supply_arr))
    closest_sink = index_order_arr[mask][0]
    index_order = list(index_order_arr)[1:]
    index_order.remove(closest_sink)

    # set up initial problem with 3 nodes:
    coords_sources_inc = np.array([start_source_coords])
    coords_sinks_inc = np.array([ext_coords[closest_sink]])
    supply_arr_inc = np.array([supply_arr[start_source]])
    demand_arr_inc = np.array([demand_arr[closest_sink - len(supply_arr)]])

    topo = nx.Graph()
    topo.add_edges_from([(0, -1), (-1, 1), (-1, 2)])
    min_label = -1
    max_label = 2

    # now add peu a peu new nodes:
    for i, node in enumerate(index_order):
        node_coords = ext_coords[node]
        topo_candidates_dict = {}
        edges_checked = []
        if node in list_source_idx:
            coords_sources_inc = np.vstack((coords_sources_inc, coords_sources[node]))
            supply_arr_inc = np.append(supply_arr_inc, supply_arr[node])
        else:
            coords_sinks_inc = np.vstack((coords_sinks_inc, coords_sinks[node - len(supply_arr)]))
            demand_arr_inc = np.append(demand_arr_inc, demand_arr[node - len(supply_arr)])

        if i == 0:
            # normalizes the demand and supply:
            cost, coords_iter = iterative_bot_solver(topo, supply_arr_inc / np.sum(supply_arr_inc),
                                                     demand_arr_inc / np.sum(demand_arr_inc)
                                                     , coords_sources_inc, coords_sinks_inc, al,
                                                     relative_improvement_threshold=1e-6, min_iterations=30,
                                                     max_iterations=1000,
                                                     plot=plot, title="", fov=fov, save=save, save_name="img")
            bp_coords = np.array([coords_iter[-1]])


        # topology_modifications if more than 4 > nodes:
        elif i != 0:
            min_label -= 1
            max_label += 1
            # find the m closest branching points and connect to the three edges of it as modified topologies:
            dist_node_bps = np.sqrt(np.sum((node_coords - bp_coords) ** 2, axis=1))
            bp_order_arr = np.argsort(dist_node_bps) - len(dist_node_bps)

            count = 0
            for k in range(m):
                if k >= len(bp_order_arr):
                    continue
                bp = bp_order_arr[k]
                #print(f"{k} closest bp: {bp} at {bp_coords[bp]} away a distance {dist_node_bps[bp]}")
                for neighbour in nx.neighbors(topo, bp):
                    if (bp, neighbour) in edges_checked or (neighbour, bp) in edges_checked:
                        # print(f"{(bp, neighbour)} already checked.")
                        continue
                    edges_checked.append((bp, neighbour))
                    topo_mod = topo.copy()
                    topo_mod.remove_edge(bp, neighbour)
                    topo_mod.add_edges_from([(bp, min_label), (neighbour, min_label), (min_label, max_label)])

                    if node in list_source_idx:
                        # relabel the topology:
                        mapping_arr = np.zeros((len(demand_arr_inc), 2), dtype=int)
                        mapping_arr[:, 0] = np.arange(len(supply_arr_inc) - 1,
                                                      len(supply_arr_inc) + len(demand_arr_inc) - 1)
                        mapping_arr[:, 1] = np.arange(len(supply_arr_inc), len(supply_arr_inc) + len(demand_arr_inc))
                        mapping_dict = dict(mapping_arr)
                        mapping_dict[max_label] = len(supply_arr_inc) - 1

                        topo_mod = nx.relabel.relabel_nodes(topo_mod, mapping_dict, copy=True)

                    topo_candidates_dict[count] = topo_mod
                    count += 1

            # accept some topology and give some position to the branching point:
            cost_candidates = np.zeros(len(topo_candidates_dict))
            coords_candidates_dict = {}
            for key in topo_candidates_dict:
                topo_candidate = topo_candidates_dict[key]
                cost_candidates[key], coords_candidates_dict[key] = iterative_bot_solver(topo_candidate,
                                supply_arr_inc / np.sum(supply_arr_inc),
                                demand_arr_inc / np.sum(demand_arr_inc),
                                                     coords_sources_inc,
                                                     coords_sinks_inc, al,
                                                     relative_improvement_threshold=1e-6,
                                                     min_iterations=30,
                                                     max_iterations=1000,
                                                     plot=False, title="", fov=None,
                                                     save=False, save_name="img")

            argmin = np.argmin(cost_candidates)
            cost = cost_candidates[argmin]
            topo = topo_candidates_dict[argmin]
            bp_coords = coords_candidates_dict[argmin][len(supply_arr_inc) + len(demand_arr_inc):]

            # visualize final solution:
            if (i == len(index_order) - 1 and final_plot) or plot:
                _ = iterative_bot_solver(topo,
                                         supply_arr_inc / np.sum(supply_arr_inc),
                                         demand_arr_inc / np.sum(demand_arr_inc),
                                         coords_sources_inc, coords_sinks_inc, al,
                                         relative_improvement_threshold=1e-6, min_iterations=-1, max_iterations=1000,
                                         plot=True, title="", fov=fov, save=save, save_name=f"img{i}")

    return cost, topo
