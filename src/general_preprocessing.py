import numpy as np
import networkx as nx


def preprocess_from_topo_to_flows(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al):
    
    """
    labelling convention for nodes:
    Enumerate sources with intergers starting from 0, then continue enumerating sinks and enumerate internal branching points with negative integers

    input: 
    - topo: topology in form of an undirected nx.Graph or a children_dict, e.g. {0:[-1], -1:[3,-2], ...}
            with flow conservation at branching points. BPs must have at least degree 3. Terminals must have degree 1.
    - supply_array, demand_array: containing the supplies of the sources and demands of the sinks in accordance with the order used in the labelling
    - coords_sources, coords_sinks: position coordinates of the sources and of the sinks
    - al: alpha parameter

    output:
    - topology in form of a graph with position as node attributes and edge weights equal to the edge-flow
      note: edge weights have a sign indicating the direction of flow based on the following convention:
            Each edge (i,j) has an implicit direction from lower index i to higher index j, given i < j.
            The edge weight is now positive if the flow follows this direction and negative if the flow is directed
            opposite to the edge direction. 
    - coords_arr: containing all coordinates of the nodes (with random coordinates for BPs)
    - list_bp_idx: a list containing the branching point indeces.
    """

    # convert children_dict to nx.Graph:
    if isinstance(topo, dict):
        children_dict = topo.copy()
        topo = nx.Graph()
        try:
            for parent in children_dict:
                for child in children_dict[parent]:
                    topo.add_edge(parent, child)
        except:
            print("conversion of topo in form of children_dict to graph has failed.")
            return

    # setup index lists:
    list_source_idx = np.arange(len(supply_arr))
    list_sink_idx = np.arange(len(supply_arr), len(supply_arr) + len(demand_arr))
    num_nodes = nx.number_of_nodes(topo)
    num_bps = num_nodes - len(supply_arr) - len(demand_arr)
    list_bp_idx = list(np.arange(1, 1 + num_bps) * (-1))

    #check that all terminals have degree 1:
    assert (np.asarray(topo.degree(list_source_idx))[:,1] == 1).all(), "Warning: source does not have degree 1."
    assert (np.asarray(topo.degree(list_sink_idx))[:, 1] == 1).all(), "Warning: sink does not have degree 1."

    # check if labelling convention is correct:
    all_idx_list = list(topo.nodes)
    all_idx_list.sort()

    assert all_idx_list == list(np.arange(- (num_bps), len(supply_arr) + len(demand_arr))) , \
        f"mistake in labelling convention! {all_idx_list, list(np.arange(-(num_bps), len(supply_arr) + len(demand_arr)))}"

    # now, solve the system of equations to determine the flows in all edges:
    # build the system of equations of the form Ax = b:
    num_edges = nx.number_of_edges(topo)
    A = np.zeros((num_edges + 1, num_edges))   # one equation is redundant, include it anyway for simplicity
    b = np.zeros(num_edges + 1)
    row = -1

    # build dictionary to map from edge to index:
    edge2idx_dict = {}
    edge_list = []
    for i, edge in enumerate(topo.edges):
        edge_list.append(edge)
        edge2idx_dict[edge] = i
        edge2idx_dict[edge[::-1]] = i

    for bp in list_bp_idx:
        row += 1
        current_row = row
        for neighbour in nx.neighbors(topo, bp):
            edge_idx = edge2idx_dict[(bp, neighbour)]
            if neighbour in list_source_idx:
                row += 1
                A[row, edge_idx] = 1
                b[row] = supply_arr[neighbour]
            elif neighbour in list_sink_idx:
                row += 1
                A[row, edge_idx] = 1
                b[row] = -demand_arr[neighbour - len(supply_arr)]
            if neighbour > bp:
                A[current_row, edge_idx] = 1
            else:
                A[current_row, edge_idx] = -1

    # now we get rid of the last line and solve the system of equations:
    try:
        edge_flows = - np.linalg.solve(A[:-1, :], b[:-1])
    except:
        print("The linear system to determine the edge flows could not be solved.")
        return

    # Add weights to the edges in topo and remove edges with zero flow:
    removed_edges_list = []
    for i, flow in enumerate(edge_flows):
        edge = edge_list[i]
        if np.abs(flow) < 1e-10:
            topo.remove_edge(edge[0], edge[1])  # remove the edge to get the different components in the next step
            removed_edges_list.append(edge)
        else:
            topo[edge[0]][edge[1]]["weight"] = flow

    # remove all nodes with degree 2 or 0:
    node_list_c = list(topo.nodes)
    for node in node_list_c:
        if topo.degree(node) == 0:
            topo.remove_node(node)
        elif topo.degree(node) == 2 and node in list_bp_idx:
            end1, end2 = topo.neighbors(node)
            topo.add_edge(end1, end2)
            if (end1 < node and end1 < end2) or (end1 > node and end1 > end2):
                topo[end1][end2]["weight"] = topo[end1][node]["weight"]
            if (end1 < node and end1 > end2) or (end1 > node and end1 < end2):
                topo[end1][end2]["weight"] = -topo[end1][node]["weight"]
            topo.remove_node(node)

    # add positions to the nodes:
    dim = len(coords_sources[0])
    coords_arr = np.vstack((coords_sources, coords_sinks, np.random.random((num_bps, dim))))

    if len(removed_edges_list) > 0:
        num_bps = nx.number_of_nodes(topo) - len(supply_arr) - len(demand_arr)
        coords_arr = np.vstack((coords_sources, coords_sinks, np.random.random((num_bps, dim))))
        print(f"preprocessing: removed zero flow edges {removed_edges_list}")
        list_bp_idx = list(np.arange(1, 1 + num_bps) * (-1))
        # use labelling convention:
        count = -1
        mapping_dict = {}
        for node in topo:
            if topo.degree(node) > 1:
                mapping_dict[node] = count
                count -= 1
        topo = nx.relabel.relabel_nodes(topo, mapping_dict, copy=True)

    return topo, coords_arr, list_bp_idx

