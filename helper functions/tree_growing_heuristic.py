import numpy as np
import networkx as nx
import sys

sys.path.append('../ready functions in py/')
from helper_fcts import *
from iterative_BOT_solver import iterative_bot_solver

# helper functions:
def f(k, al):
    return np.arccos((k ** (2 * al) + 1 - (1 - k) ** (2 * al)) / (2 * k ** al))


def get_angles(k, al, case):
    if case == 1:
        return f(k, al), f(1 - k, al)
    elif case == 2:
        # the first angle is the large one assuming that m2 is larger than m1
        return np.pi - f(k, al) - f(1 - k, al), f(k, al)


def init_topo(source0_supply, max_edge_length, al):
    # init graph:
    topo = nx.Graph()
    topo.add_edges_from([(0, 1), (1, 2), (1, 3)])

    # flows:
    topo[0][1]["weight"] = source0_supply
    left_fraction = np.random.random()
    m1 = left_fraction * source0_supply
    m2 = (1 - left_fraction) * source0_supply
    topo[1][2]["weight"] = m1
    topo[1][3]["weight"] = m2

    # angles:
    topo[0][1]["angle"] = np.pi / 2
    k = m1 / source0_supply
    case = 1
    angle1, angle2 = get_angles(k, al, case)
    topo[1][2]["angle"] = np.pi / 2 + angle1
    topo[1][3]["angle"] = np.pi / 2 - angle2

    # positions:
    topo.nodes[0]["pos"] = np.array([0., 0.])
    l0, l1, l2 = np.random.random(3) * max_edge_length
    topo.nodes[1]["pos"] = np.array([0., l0])
    topo.nodes[2]["pos"] = np.array([0., l0]) + l1 * np.array(
        [np.cos(topo[1][2]["angle"]), np.sin(topo[1][2]["angle"])])
    topo.nodes[3]["pos"] = np.array([0., l0]) + l2 * np.array(
        [np.cos(topo[1][3]["angle"]), np.sin(topo[1][3]["angle"])])

    return topo, m1, m2, l0, l1, l2


def topo_relabelling(topo, list_bps_to_relabel, list_sources_to_relabel, list_sinks_to_relabel):
    # relabel topo, setup mapping dict:
    bp_mapping_arr = np.zeros((len(list_bps_to_relabel), 2), dtype=int)
    sources_mapping_arr = np.zeros((len(list_sources_to_relabel), 2), dtype=int)
    sinks_mapping_arr = np.zeros((len(list_sinks_to_relabel), 2), dtype=int)
    bp_mapping_arr[:, 0] = list_bps_to_relabel
    bp_mapping_arr[:, 1] = - np.arange(1, len(list_bps_to_relabel) + 1)
    sources_mapping_arr[:, 0] = list_sources_to_relabel
    sources_mapping_arr[:, 1] = np.arange(len(list_sources_to_relabel))
    sinks_mapping_arr[:, 0] = list_sinks_to_relabel
    sinks_mapping_arr[:, 1] = np.arange(len(list_sources_to_relabel),
                                        len(list_sources_to_relabel) + len(list_sinks_to_relabel))

    mapping_dict = dict(np.vstack((bp_mapping_arr, sources_mapping_arr, sinks_mapping_arr)))

    return nx.relabel.relabel_nodes(topo, mapping_dict, copy=True)


"""
This function moves nodes to edges.
input:
- pairs like this [(node, (edge[0], edge[1])), (..), ..]

note that the edges must be different otherwise the second update will be no longer work, since the edge has been altered. 

output:
- altered topology with the nodes moved to the respective edges 
"""
def move_single_nodes_to_edges(topo_to_mod, node_edge_pair_list):
    topo = topo_to_mod.copy()
    for k,node_edge_pair in enumerate(node_edge_pair_list):
        try:
            node, edge = node_edge_pair
            parent_of_node = list(nx.neighbors(topo, node))[0]
            topo.remove_edges_from([(node, parent_of_node), (edge[0], edge[1])])
            n1, n2 = list(nx.neighbors(topo, parent_of_node))
            topo.remove_node(parent_of_node)
            topo.add_edges_from([(n1,n2), (edge[0], parent_of_node), (edge[1], parent_of_node), (node, parent_of_node)])
        except Exception:
            # print(f"move nr.{k+1} of node to edge failed or was no longer necessary.")
            return 0
    return topo


"""
This function moves a pair of nodes to an edge.
input: 
- parent_node: the pair summarised by it's parent (common neighbor)
- edge: the target edge

output:
- altered topology with the pair of nodes moved to the given edge
"""
def move_pair_to_single_edge(topo_to_mod, parent, parent_of_parent, edge):
    topo = topo_to_mod.copy()
    ns = list(nx.neighbors(topo, parent_of_parent))
    ns.remove(parent)
    topo.remove_node(parent_of_parent)
    topo.remove_edge(edge[0], edge[1])
    topo.add_edges_from([(ns[0], ns[1]), (parent, parent_of_parent), (edge[0], parent_of_parent), (edge[1], parent_of_parent)])
    return topo


"""
function that incrementally grows relative optimal trees. It thereby creates a BOT problem 
and produces a strong topology as it goes.

input:
- num_sinks and sources (convention: num_sinks >= num_sources)
- al: alpha-parameter of BOT
- max_edge_length: grown edges are at most this long
- source0_supply: supply of source0
- max_source_supply_factor: later generated sources are at most 1.5 times bigger than source0
- num_tires: number of tries to grow at a chosen terminal
- N_e: number of close edges to the children 1 and 2 used for alternative topos
- N_n: number of other terminals close to the children edges used for alternative topos
- max_refill_counter: number of times when the sample lists are refilled after they are completely empty.

output:
- topology in standard labelling convention.
- the cost theoretically optimal cost given this topology  
- the correpsonding bot_problem_dict
"""
def tree_growing_heuristic(num_sinks, num_sources, al, max_edge_length=1., source0_supply=1.,
                           max_source_supply_factor=1.5, num_tries=3, max_refill_counter=3,
                            N_e = 3, N_n = 2,
                           plot_final=False, plot_accepted=False, plot_alternatives=False):
    if num_sinks < num_sources:
        print("switched number of sinks and sources")
        num_sinks_s = num_sinks
        num_sinks = num_sources
        num_sources = num_sinks_s

    topo, m1, m2, l0, l1, l2 = init_topo(source0_supply, max_edge_length, al)

    # lists to sample terminals from:
    sources_to_sample = []  # can't use 0 as source to sample cause angles are not correct then
    sinks_to_sample = [2, 3]
    list_sources_to_relabel = [0]
    supply_arr = np.array([source0_supply])
    coords_sources = np.array([
        [0., 0.]
    ])
    list_sinks_to_relabel = [2, 3]
    demand_arr = np.array([m1, m2])  # in same order as labels in "sinks_to_relabel", same with the coords_sinks:
    coords_sinks = np.array([
        topo.nodes[2]["pos"],
        topo.nodes[3]["pos"]
    ])
    list_bps_to_relabel = [1]
    max_idx = 3

    # coordinates of edges (a[0], a[1], b[0], b[1]), order determined by edges_arr
    edges_coords_arr = np.array([
        np.append(topo.nodes[0]["pos"], topo.nodes[1]["pos"]),
        np.append(topo.nodes[1]["pos"], topo.nodes[2]["pos"]),
        np.append(topo.nodes[1]["pos"], topo.nodes[3]["pos"])
    ])
    edges_arr = np.array([
        [0, 1],
        [1, 2],
        [1, 3]
    ])

    # init running_cost:
    running_cost = source0_supply ** al * l0 + m1 ** al * l1 + m2 ** al * l2

    # visualise starting config:
    if plot_accepted or 2 * (num_sources + num_sinks) - 3 == 3:
        topo_relabelled = topo_relabelling(topo, list_bps_to_relabel, list_sources_to_relabel, list_sinks_to_relabel)

        coords_arr = np.zeros((nx.number_of_nodes(topo), 2))
        for node in topo_relabelled.nodes():
            coords_arr[node] = topo_relabelled.nodes[node]["pos"]

        if plot_accepted or (2 * (num_sources + num_sinks) - 3 == 3 and plot_final):
            visualise_BOT_solution(topo_relabelled, coords_arr, supply_arr, demand_arr, title="", fov=None, save=False,
                                   save_name="img")

        if 2 * (num_sources + num_sinks) - 3 == 3:
            # already done:
            bot_problem_dict = {
                "al": al,
                "coords_sources": coords_sources,
                "coords_sinks": coords_sinks,
                "supply_arr": supply_arr,
                "demand_arr": demand_arr
            }

            return topo_relabelled, running_cost, coords_arr, bot_problem_dict

    refill_counter = 0
    # now iterate tree growing until number of sources and number of sinks is satisfied.
    while nx.number_of_nodes(topo) < 2 * (num_sources + num_sinks) - 3:

        # first sample any terminal according to how old the are:
        terminals_to_sample = sources_to_sample + sinks_to_sample
        probs = 1 / (np.array(terminals_to_sample) + 1) + 0.3
        chosen_terminal = np.random.choice(terminals_to_sample, p=probs / np.sum(probs))
        parent = list(nx.neighbors(topo, chosen_terminal))[0]
        mass_at_terminal = topo[chosen_terminal][parent]["weight"]
        incoming_angle = topo[chosen_terminal][parent]["angle"]
        position_terminal = topo.nodes[chosen_terminal]["pos"]

        # check if source of sink:
        terminal_type = "source"
        if chosen_terminal in sinks_to_sample:
            terminal_type = "sink"

        # now decide if a new sink or a new source sould be added based on what is needed more:
        # this determines if we have a case 1 or case 2 branching:
        new_sink = True
        diff_sources = num_sources - len(list_sources_to_relabel)
        diff_sinks = num_sinks - len(list_sinks_to_relabel)
        p_source = diff_sources / (diff_sources + diff_sinks)
        if np.random.random() < p_source:
            new_sink = False  # want a new source instead

        if new_sink:
            if terminal_type == "sink":
                case = 1
                type1 = "sink"  # type of the children
                type2 = "sink"
            elif terminal_type == "source":
                case = 2  # second children has joint flow
                type1 = "sink"
                type2 = "source"
        elif not new_sink:
            if terminal_type == "sink":
                case = 2
                type1 = "source"
                type2 = "sink"
            elif terminal_type == "source":
                case = 1
                type1 = "source"
                type2 = "source"

        l1_max = max_edge_length
        l2_max = max_edge_length

        # to symmetrize case 2 swtich roles of A1 and A2 with 50%:
        switch_case2 = False
        if case == 2 and np.random.random() < 0.5:
            switch_case2 = True
            type1_s = type1
            type1 = type2
            type2 = type1_s

        # now start the different tries:
        for run in range(num_tries):

            # get left flow fraction k:
            if case == 1:
                k = (1 - 1e-4) * np.random.random() + 1e-4  # to avoid zero flow edges
                m1 = k * mass_at_terminal
                m2 = (1 - k) * mass_at_terminal
            elif case == 2:
                m2 = np.random.uniform(low=mass_at_terminal + 1e-4, high=source0_supply * max_source_supply_factor)
                m1 = m2 - mass_at_terminal
                k = mass_at_terminal / m2

            # get angles:
            angle1, angle2 = get_angles(k, al, case)

            # to symmetrize case 2 randomly switch 1 and 2:
            if switch_case2:
                m1_s = m1
                angle1_s = angle1
                m1 = m2
                angle1 = angle2
                m2 = m1_s
                angle2 = angle1_s

            # get lengths and positions:
            l1 = np.random.random() * l1_max
            l2 = np.random.random() * l2_max
            angle1_absolute = incoming_angle + angle1
            angle2_absolute = incoming_angle - angle2
            pos1 = position_terminal + l1 * np.array([np.cos(angle1_absolute), np.sin(angle1_absolute)])
            pos2 = position_terminal + l2 * np.array([np.cos(angle2_absolute), np.sin(angle2_absolute)])

            cost_candidate = running_cost + m1 ** al * l1 + m2 ** al * l2

            # now test against all alternatives:
            rejected = False

            # insert all checks here!
            """
            build a dictionary with alternative topologies and then parallelize the prozess of trying them out and calculating the cost:
            use the following alternative topologies:
            A) move left child to its closest edges and keep right child (multiple, for now 2)
            B) move right child to its closest edges and keep left child (multiple, for now 2)
            C) move both children to the closest edges of the left child (multiple, for now 2)
            D) move both children to the closest edges of the right child (multiple, for now 2)
            E) move closest terminals individually to left edge (multiple, for now 2)
            F) move closest terminals individually to right edge (multiple, for now 2)
            G) move both children to their respective closest edges (single)

            not included:
            - take multiple terminals (e.g. siblings) and jointly move them to left or right edge
            """

            topo_to_mod = topo.copy()
            topo_to_mod.add_edges_from([(chosen_terminal, max_idx + 1), (chosen_terminal, max_idx + 2)])

            # get the N_e closest edges of the left and right child:

            left_closest_edges = []
            left_closest_edges_for_pair = []
            # all distances of left child to edges:
            dist_left_edges = dist_point_segments(pos1, edges_coords_arr[:, :2], edges_coords_arr[:, 2:])
            # find the closest edges and if they are not attached to chosen_terminal collect them
            while len(left_closest_edges_for_pair) < N_e:
                argmin = np.argmin(dist_left_edges)
                dist_left_edges[argmin] = np.inf
                next_closest_edge = edges_arr[argmin]
                if not chosen_terminal in next_closest_edge and len(left_closest_edges) < N_e:
                    left_closest_edges.append(next_closest_edge)
                if not chosen_terminal in next_closest_edge and not parent in next_closest_edge:
                    left_closest_edges_for_pair.append(next_closest_edge)
                if (dist_left_edges == np.inf).all():
                    break

            right_closest_edges = []
            right_closest_edges_for_pair = []
            # all distances of right child to edges:
            dist_right_edges = dist_point_segments(pos2, edges_coords_arr[:, :2], edges_coords_arr[:, 2:])
            # find the closest edges and if they are not attached to chosen_terminal collect them
            while len(right_closest_edges_for_pair) < N_e:
                argmin = np.argmin(dist_right_edges)
                dist_right_edges[argmin] = np.inf
                next_closest_edge = edges_arr[argmin]
                if not chosen_terminal in next_closest_edge and len(right_closest_edges) < N_e:
                    right_closest_edges.append(next_closest_edge)
                if not chosen_terminal in next_closest_edge and not parent in next_closest_edge:
                    right_closest_edges_for_pair.append(next_closest_edge)
                if (dist_right_edges == np.inf).all():
                    break

            # construct the modified topologies for case A,B,C,D and G:
            alternative_counter = 0
            alternative_topos = {}
            # A) move left child only
            for edge in left_closest_edges:
                alternative_topos[alternative_counter] = move_single_nodes_to_edges(topo_to_mod,
                                                                                    [(max_idx + 1, (edge[0], edge[1]))])
                alternative_counter += 1
            # B) move right child only
            for edge in right_closest_edges:
                alternative_topos[alternative_counter] = move_single_nodes_to_edges(topo_to_mod,
                                                                                    [(max_idx + 2, (edge[0], edge[1]))])
                alternative_counter += 1
            # C) and D) move both children to closest edges of left and right node:
            joint_edges_for_pair = left_closest_edges_for_pair + right_closest_edges_for_pair
            joint_edges_for_pair_unique = []
            for e in joint_edges_for_pair:
                joint_edges_for_pair_unique.append((e[0], e[1]))
            joint_edges_for_pair_unique = list(set(joint_edges_for_pair_unique))
            for edge in joint_edges_for_pair_unique:
                alternative_topos[alternative_counter] = move_pair_to_single_edge(topo_to_mod, chosen_terminal, parent,
                                                                                  edge)
                alternative_counter += 1
            # G) move both children to their closest edge simultaneously (and separately) if edges are different:
            if len(left_closest_edges) > 0 and len(right_closest_edges) > 0:
                if not np.allclose(left_closest_edges[0], right_closest_edges[0]):
                    edge_l = left_closest_edges[0]
                    edge_r = right_closest_edges[0]
                    # print("individual edges:", edge_l, edge_r)
                    topo_candidate = move_single_nodes_to_edges(topo_to_mod, [(max_idx + 1, (edge_l[0], edge_l[1])),
                                                                              (max_idx + 2, (edge_r[0], edge_r[1]))])
                    # check if modification was successful:
                    if topo_candidate != 0:
                        alternative_topos[alternative_counter] = topo_candidate
                        alternative_counter += 1

            # now find terminal points closest to the left and right wanna-be edge and construct alternative E, F:

            # closest points to the left edge:
            all_terminal_coords = np.vstack((coords_sources, coords_sinks))
            all_indeces = list_sources_to_relabel + list_sinks_to_relabel
            dist_edge_points_left = dist_segment_points(all_terminal_coords, position_terminal, pos1)
            list_left_close_terminals = []
            while len(list_left_close_terminals) < N_n:
                argmin = np.argmin(dist_edge_points_left)
                dist_edge_points_left[argmin] = np.inf
                node = all_indeces[argmin]
                if node != chosen_terminal:
                    list_left_close_terminals.append(node)
                if (dist_edge_points_left == np.inf).all():
                    break

            # the same for the right side:
            dist_edge_points_right = dist_segment_points(all_terminal_coords, position_terminal, pos2)
            list_right_close_terminals = []
            while len(list_right_close_terminals) < N_n:
                argmin = np.argmin(dist_edge_points_right)
                dist_edge_points_right[argmin] = np.inf
                node = all_indeces[argmin]
                if node != chosen_terminal:
                    list_right_close_terminals.append(node)
                if (dist_edge_points_right == np.inf).all():
                    break

            # print("left_closest_edges =", left_closest_edges)
            # print("left_closest_edges_pair=", left_closest_edges_for_pair)
            # print("r_closest_edges =", right_closest_edges)
            # print("r_closest_edges_pair=", right_closest_edges_for_pair)
            # print("left_nodes =", list_left_close_terminals)
            # print("right_nodes=", list_right_close_terminals)

            # now construct alternative topologies for cases E, F:
            for close_terminal in list_left_close_terminals:
                alternative_topos[alternative_counter] = move_single_nodes_to_edges(topo_to_mod, [
                    (close_terminal, (chosen_terminal, max_idx + 1))])
                alternative_counter += 1
            for close_terminal in list_right_close_terminals:
                alternative_topos[alternative_counter] = move_single_nodes_to_edges(topo_to_mod, [
                    (close_terminal, (chosen_terminal, max_idx + 2))])
                alternative_counter += 1

            # now loop through all alternative topologies and collect the alternative costs:
            alternative_costs = np.zeros(alternative_counter)
            # need to modify the coords_arr and supply and demand_arr:
            # store old things to restore them in case of rejection:
            list_sinks_to_relabel_s = list_sinks_to_relabel.copy()
            list_sources_to_relabel_s = list_sources_to_relabel.copy()
            list_bps_to_relabel_s = list_bps_to_relabel.copy()
            coords_sinks_s = np.copy(coords_sinks)
            coords_sources_s = np.copy(coords_sources)
            demand_arr_s = np.copy(demand_arr)
            supply_arr_s = np.copy(supply_arr)

            if terminal_type == "sink":
                idx_pos = np.where(np.array(list_sinks_to_relabel) == chosen_terminal)[0][0]
                list_sinks_to_relabel.remove(chosen_terminal)
                demand_arr = np.append(demand_arr[:idx_pos], demand_arr[idx_pos + 1:])
                coords_sinks = np.vstack((coords_sinks[:idx_pos, :], coords_sinks[idx_pos + 1:, :]))
            else:
                idx_pos = np.where(np.array(list_sources_to_relabel) == chosen_terminal)[0][0]
                list_sources_to_relabel.remove(chosen_terminal)
                supply_arr = np.append(supply_arr[:idx_pos], supply_arr[idx_pos + 1:])
                coords_sources = np.vstack((coords_sources[:idx_pos, :], coords_sources[idx_pos + 1:, :]))
            list_bps_to_relabel.append(chosen_terminal)

            if type1 == "sink":
                list_sinks_to_relabel.append(max_idx + 1)
                demand_arr = np.append(demand_arr, m1)
                coords_sinks = np.vstack((coords_sinks, pos1))
            elif type1 == "source":
                list_sources_to_relabel.append(max_idx + 1)
                supply_arr = np.append(supply_arr, m1)
                coords_sources = np.vstack((coords_sources, pos1))
            if type2 == "sink":
                list_sinks_to_relabel.append(max_idx + 2)
                demand_arr = np.append(demand_arr, m2)
                coords_sinks = np.vstack((coords_sinks, pos2))
            elif type2 == "source":
                list_sources_to_relabel.append(max_idx + 2)
                supply_arr = np.append(supply_arr, m2)
                coords_sources = np.vstack((coords_sources, pos2))

            # print("case=", case, "terminal_type=", terminal_type)
            # print("types:", type1, type2)
            # print("masses=", m1, m2, mass_at_terminal)
            # print(supply_arr, demand_arr, coords_sources, coords_sinks)
            # print("check the following alternatives:", alternative_counter)
            for i, key in enumerate(alternative_topos):
                # relabel topo and calculate cost:
                topo_altern = topo_relabelling(alternative_topos[key], list_bps_to_relabel, list_sources_to_relabel,
                                               list_sinks_to_relabel)
                alternative_costs[i], _ = iterative_bot_solver(topo_altern, supply_arr, demand_arr, coords_sources,
                                                               coords_sinks, al,
                                                               relative_improvement_threshold=1e-6, min_iterations=30,
                                                               max_iterations=600,
                                                               plot=plot_alternatives, title="altern.", fov=None,
                                                               save=False, save_name="img")
                # print("checked", i)
            # print("compare costs:", cost_candidate, alternative_costs)

            rejected = (alternative_costs < cost_candidate).any()
            if rejected:
                # adapt lengths
                l1_max = l1
                l2_max = l2

                # undo changes in supply and demand array etc.:
                list_sinks_to_relabel = list_sinks_to_relabel_s
                list_sources_to_relabel = list_sources_to_relabel_s
                list_bps_to_relabel = list_bps_to_relabel_s
                coords_sinks = coords_sinks_s
                coords_sources = coords_sources_s
                demand_arr = demand_arr_s
                supply_arr = supply_arr_s

                if run == num_tries - 1:
                    # take terminal from the list to sample from
                    # print(f"removed terminal {chosen_terminal} from sample list")
                    if terminal_type == "sink":
                        sinks_to_sample.remove(chosen_terminal)
                    elif terminal_type == "source":
                        sources_to_sample.remove(chosen_terminal)

                    # if list to sample from is empty print "restart" and call function again.
                    if len(sinks_to_sample + sources_to_sample) == 0:
                        refill_counter += 1
                        print("refill the sample list! counter:", refill_counter)
                        sources_to_sample = list_sources_to_relabel.copy()
                        sources_to_sample.remove(0)
                        sinks_to_sample = list_sinks_to_relabel.copy()
                        if refill_counter == max_refill_counter:
                            print("max refill counter: End growth procedure.")
                            bot_problem_dict = {
                                "al": al,
                                "coords_sources": coords_sources,
                                "coords_sinks": coords_sinks,
                                "supply_arr": supply_arr,
                                "demand_arr": demand_arr
                            }
                            topo_relabelled = topo_relabelling(topo, list_bps_to_relabel, list_sources_to_relabel,
                                                               list_sinks_to_relabel)

                            coords_arr = np.zeros((nx.number_of_nodes(topo), 2))
                            for node in topo_relabelled.nodes():
                                coords_arr[node] = topo_relabelled.nodes[node]["pos"]

                            return topo_relabelled, running_cost, coords_arr, bot_problem_dict

            elif not rejected:
                # accept the candidate:
                running_cost = cost_candidate
                topo.add_edges_from([(chosen_terminal, max_idx + 1), (chosen_terminal, max_idx + 2)])
                new_edges_coords_arr = np.array([
                    np.append(position_terminal, pos1),
                    np.append(position_terminal, pos2)
                ])
                new_edges_arr = np.array([
                    [chosen_terminal, max_idx + 1],
                    [chosen_terminal, max_idx + 2]
                ])
                edges_coords_arr = np.vstack((edges_coords_arr, new_edges_coords_arr))
                edges_arr = np.vstack((edges_arr, new_edges_arr))

                topo[chosen_terminal][max_idx + 1]["weight"] = m1
                topo[chosen_terminal][max_idx + 2]["weight"] = m2
                topo[chosen_terminal][max_idx + 1]["angle"] = angle1_absolute
                topo[chosen_terminal][max_idx + 2]["angle"] = angle2_absolute
                topo.nodes[max_idx + 1]["pos"] = pos1
                topo.nodes[max_idx + 2]["pos"] = pos2

                if terminal_type == "sink":
                    sinks_to_sample.remove(chosen_terminal)
                else:
                    sources_to_sample.remove(chosen_terminal)

                if type1 == "sink":
                    sinks_to_sample.append(max_idx + 1)
                else:
                    sources_to_sample.append(max_idx + 1)
                if type2 == "sink":
                    sinks_to_sample.append(max_idx + 2)
                else:
                    sources_to_sample.append(max_idx + 2)

                max_idx += 2

                if plot_accepted or nx.number_of_nodes(topo) == 2 * (num_sources + num_sinks) - 2:
                    topo_relabelled = topo_relabelling(topo, list_bps_to_relabel, list_sources_to_relabel,
                                                       list_sinks_to_relabel)
                    coords_arr = np.zeros((nx.number_of_nodes(topo), 2))
                    for node in topo_relabelled.nodes():
                        coords_arr[node] = topo_relabelled.nodes[node]["pos"]

                    if plot_accepted or (plot_final and nx.number_of_nodes(topo) == 2 * (num_sources + num_sinks) - 2):
                        visualise_BOT_solution(topo_relabelled, coords_arr, supply_arr, demand_arr, title="accepted",
                                               fov=None, save=False, save_name="img")

                    if nx.number_of_nodes(topo) == 2 * (num_sources + num_sinks) - 2:
                        bot_problem_dict = {
                            "al": al,
                            "coords_sources": coords_sources,
                            "coords_sinks": coords_sinks,
                            "supply_arr": supply_arr,
                            "demand_arr": demand_arr
                        }

                        return topo_relabelled, running_cost, coords_arr, bot_problem_dict

                # end for-loop over tries:
                break