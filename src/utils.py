import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualise_BOT_solution(topo, coords_arr, supply_arr, demand_arr, title="", fov=None, save=False, save_name="img", figsize = (8,8)):
    
    """
    a general function for visualising a solution.

    input:
    - topo: current solution in form of a graph with weights as edge features
    - coords_arr: coordinates of branching points and terminals
    - supply_arr, demand_arr
    - title: string to show as title of the plot
    - fov: field of view (2,2)-array, or None then fov is automatically chosen
    - save: boolean to determine if image should be saved with save_name as name

    output:
    - it produces the desired plot
    """

    plt.figure(figsize=figsize)
    if title != "":
        plt.title(title)

    linescale = 15 / sum(supply_arr)  # defines thickness of edges relative to total flow
    markerscale = 25 / sum(supply_arr)
    list_source_idx = np.arange(len(supply_arr))
    list_sink_idx = np.arange(len(supply_arr) + len(demand_arr))

    # plot sources and sinks:
    sources_labelled = False
    sinks_labelled = False

    #plot edges:
    for node in topo.nodes:
        x = coords_arr[node]
        for neighbour in nx.neighbors(topo, node):
            if neighbour > node:
                # don't plot edges twice
                continue

            flow = np.abs(topo[node][neighbour]["weight"])
            y = coords_arr[neighbour]
            plt.plot([x[0], y[0]], [x[1], y[1]], color="black", linewidth=linescale * flow + 1, alpha=0.85)

    # plot sources and sinks on top:
    for node in topo.nodes:
        x = coords_arr[node]
        if node in list_source_idx:
            if not sources_labelled:
                plt.plot(x[0], x[1], marker="o", color="r", linestyle="", markersize=markerscale *
                                                                                     supply_arr[node] + 3, alpha=0.8,
                         label="sources")
                sources_labelled = True
            else:
                plt.plot(x[0], x[1], marker="o", color="r", linestyle="", markersize=markerscale *
                                                                                     supply_arr[node] + 3, alpha=0.8)

        elif node in list_sink_idx:
            if not sinks_labelled:
                plt.plot(x[0], x[1], marker="o", color="b", linestyle="", markersize=markerscale *
                                                    demand_arr[node - len(supply_arr)] + 3,alpha=0.8, label="sinks")
                sinks_labelled = True
            else:
                plt.plot(x[0], x[1], marker="o", color="b", linestyle="", markersize=markerscale *
                                                    demand_arr[node - len(supply_arr)] + 3, alpha=0.8)


    if isinstance(fov, np.ndarray):
        if fov.shape != (2, 2):
            print("Error. invalid fov.")
        plt.xlim(fov[0, 0], fov[0, 1])
        plt.ylim(fov[1, 0], fov[1, 1])
    else:
        plt.axis('equal')
    legend = plt.legend(fontsize=14)
    # make all markers the same size eventhough they are not in the image:
    for legend_handle in legend.legendHandles:
        # print(dir(legend_handle))
        # legend_handle._legmarker.set_markersize(10)
        legend_handle.set_markersize(10)
    if save:
        #plt.xticks(np.linspace(0, 1, 3), fontsize=15)
        #plt.yticks(np.linspace(0, 1, 3), fontsize=15)
        #plt.xticks(fontsize=14)
        #plt.yticks(fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_name + ".pdf", bbox_inches="tight")
        # plt.savefig(save_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()
    return


def eucl_dist(x, Y):
    """
    inputs:
    - x: d-dimensional array
    - Y: Nxd-dimensional array (N must be at least 1, as in np.array([[3,3]]))

    output:
    N-dimensional array containing the distance of vector x to each of the vectors y
    """

    return np.sqrt(np.sum((x - Y) ** 2, axis=1))


# write function for distance between point and line segment:
def dist_point_segments(p, a, b):
    """
    p - point to project (dim)
    a - array of starting points of the segments (n, dim)
    b - array of end points of the segments (n, dim)
    """
    # project point onto straight line and check if inside line segment:
    # print(a.shape, b.shape, p.shape)
    # print(a.shape, b.shape, np.sum((b - a) * (p[None,:] - a), axis=1).shape, np.clip(eucl_dist(a, b) ** 2, 1e-8, None).shape)
    lam = np.sum((b - a) * (p - a), axis=1) / np.clip(eucl_dist(a, b) ** 2, 1e-8, None)
    p_proj = a + lam[:, None] * (b - a)

    p_proj[lam <= 0] = a[lam <= 0]
    p_proj[lam >= 1] = b[lam >= 1]

    dist_arr = np.sqrt(np.sum((p - p_proj) ** 2, axis=1))
    return dist_arr


def dist_segment_points(p, a, b):
    """
    p - array of n points to project (n, dim)
    a - starting point of the segment (dim)
    b - end point of the segment (dim)

    output: dist_arr (n)
    """
    # project all points onto the line:
    dist_ab = np.sqrt(np.sum((b - a) ** 2))
    lam = np.dot(p - a, b - a) / np.max([dist_ab ** 2, 1e-8])
    p_proj = a + np.outer(lam, b - a)

    p_proj[lam <= 0] = a
    p_proj[lam >= 1] = b

    dist_arr = np.sqrt(np.sum((p - p_proj) ** 2, axis=1))
    return dist_arr


def generate_random_binary_tree_topo(N):
    
    """
    generates a random binary tree with N terminals
    returns topo as nx.Graph()
    """
    
    #generate random binary tree topology with N terminals
    topo = nx.Graph()
    cluster_idx_list = list(np.arange(1,N))
    cluster_idx = -1

    for i in range(N-2):
        c1, c2 = np.random.choice(cluster_idx_list, size=2, replace=False)

        #unite those clusters
        topo.add_edges_from([(c1, cluster_idx), (c2, cluster_idx)])

        cluster_idx_list.remove(c1)
        cluster_idx_list.remove(c2)
        cluster_idx_list.append(cluster_idx)

        cluster_idx -= 1

    topo.add_edge(cluster_idx + 1, 0)
    return topo


def get_random_topos_sparsely(bot_problem_dict, num_of_topos_requested):
    
    """
    input: 
    - bot_problem_dict containing problem setup in the following form:
            al = bot_problem_dict["al"]
            coords_sources = bot_problem_dict["coords_sources"]
            coords_sinks = bot_problem_dict["coords_sinks"]
            supply_arr = bot_problem_dict["supply_arr"]
            demand_arr = bot_problem_dict["demand_arr"]
    - num_of_topos_requested: number of random topos which should be generated

    output: 
    - a dictionary with random topos as tuple (children_dict, nx.Graph)
    """
    
    # note: generating all topos and then sampling a certain number of them randomly is not feasible (it makes the RAM blow up for larger problems).
    # numbers of sinks and sources:
    num_sources = len(bot_problem_dict["supply_arr"])
    num_sinks = len(bot_problem_dict["demand_arr"])

    # total number of topos:
    num_of_topos = 1
    for i in range(2, num_sources + num_sinks):
        num_of_topos *= (2 * i - 3)

    if num_of_topos_requested >= num_of_topos:
        #print("too many topos requested, number was set to the most possible: ", num_of_topos)

        # now loop over all possible topologies with the number of nodes:
        # intitialisation for loop:
        children_dict_start_topo = {
            0: [-1],
            -1: [1, 2]
        }
        largest_label_ext = 2
        smallest_label_int = -1

        start_topo_3_nodes = nx.Graph()
        for key in children_dict_start_topo:
            for child in children_dict_start_topo[key]:
                start_topo_3_nodes.add_edge(key, child)

        # initialise:
        old_topos = {0: start_topo_3_nodes}  # topos with n external nodes

        for n in range(4, num_sources + num_sinks + 1):
            count = 0
            new_topos = {}
            largest_label_ext += 1
            smallest_label_int -= 1
            for key in old_topos:
                # modify all:
                old_edge_list = list(old_topos[key].edges())
                idx_edge_list = np.arange(len(old_edge_list))
                for i, idx in enumerate(idx_edge_list):
                    edge = old_edge_list[idx]
                    new_graph = old_topos[key].copy()
                    left_end, right_end = edge
                    new_graph.remove_edge(*edge)
                    new_graph.add_nodes_from([smallest_label_int, largest_label_ext])
                    new_graph.add_edges_from([(left_end, smallest_label_int),
                                              (smallest_label_int, right_end),
                                              (smallest_label_int, largest_label_ext)])

                    new_topos[count] = new_graph
                    count += 1

            # use all topos to construct the next generation
            old_topos = new_topos.copy()

    else:
        # i.e. if num_of_topos_requested < num_of_topos:
        old_topos = {}
        num_nodes = num_sources + num_sinks

        for i in range(num_of_topos_requested):
            # generate a random binary tree topo with the usual labelling convention:
            topo = generate_random_binary_tree_topo(num_nodes)
            # add this topo to old_topos:
            old_topos[i] = topo

    return old_topos


def generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2, max_length=1.):
    
    """
    create a random 2d problem inside the [0,1] x [0,1] square with non-integer demand and supplies.
    demand and supplies as well as the coordinates are uniformly distributed.
    input:
    - num_sources and sinks
    - normalised_to: number to which demand and supply are normalised
    - dim: spatial dimension of coordinates
    - max_length: size of the hyper-cube from which coordinates are sampled

    output:
    - return a bot_problem_dict in the usual form.
    """
    
    al = np.random.random()
    coords_sources = np.random.random((num_sources, dim)) * max_length
    coords_sinks = np.random.random((num_sinks, dim)) * max_length
    supply_arr = np.random.random(num_sources)
    supply_arr = normalised_to * supply_arr / np.sum(supply_arr)
    demand_arr = np.random.random(num_sinks)
    demand_arr = normalised_to * demand_arr / np.sum(demand_arr)

    bot_problem_dict = {
        "al": al,
        "coords_sources": coords_sources,
        "coords_sinks": coords_sinks,
        "supply_arr": supply_arr,
        "demand_arr": demand_arr
    }

    return bot_problem_dict

"""
convert topo to children_dict:
dict(nx.bfs_successors(topo, 0))


count = -1
mapping_dict = {}
for node in topo:
    if topo.degree(node) > 1:
        mapping_dict[node] = count
        count -= 1
        
topo = nx.relabel.relabel_nodes(topo, mapping_dict, copy=True)
"""