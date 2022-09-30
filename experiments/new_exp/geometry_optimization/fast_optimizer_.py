import ctypes
import numpy as np
from numpy import ctypeslib as npct

#load shared C-library for optimizer
opt_lib = npct.load_library("fast_optimizer","./BOT/geometry_optimization/lib")

#define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

opt_lib.iterations.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_2d_int, array_2d_double, array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double]
opt_lib.iterations.restype = ctypes.c_double


def fast_optimize(itopo, supply_arr, demand_arr, coords_sources, coords_sinks, al, improv_threshold=1e-7):

    #use general preprocessing function to calculate the edge-flows:
    topo, coords_arr, idx = preprocess_topo(itopo, supply_arr, demand_arr, coords_sources, coords_sinks, al, remove_zero_flows=False)
    dim = len(coords_sources[0])
    nsites = len(coords_sinks)+len(coords_sources)

    #construct adjacency and Edge weight arrays to pass to optimizer
    adj_ =  np.array([list(nbdict.keys())+[n] for n,nbdict in topo.adjacency() if n<0]).astype(np.intc)
    adj = np.ascontiguousarray(adj_[:,:-1])
    ns_bp = adj_[:,-1]
    sort_ind = np.argsort(ns_bp)[::-1]
    ns_bp=ns_bp[sort_ind]
    adj=adj[sort_ind]
    adj[adj<0] = -adj[adj<0] + nsites - 1#
    
    EW = np.array(np.zeros((nsites-2,3))).astype(np.double)
    demands = np.append(supply_arr,-demand_arr)
    coords_arr = coords_arr.flatten()

    #output variables
    iter = ctypes.c_int(0)

    #run optimization
    cost = opt_lib.iterations(ctypes.byref(iter),dim,nsites,adj,EW,demands,coords_arr,al,improv_threshold)

    coords_arr = coords_arr.reshape((-1,dim))
    #hot fix: BP-coords are in reversed order (?!)
    coords_arr[nsites:] = coords_arr[nsites:][::-1]

    #store flows into nodes
    adj[adj>=nsites] = -adj[adj>=nsites] + nsites - 1
    for j,(n,a) in enumerate(zip(ns_bp,adj)):
        for i in range(3):
            topo[n][a[i]]["weight"] = EW[j][i]

    return itopo, cost, coords_arr, iter.value


def preprocess_topo(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al, remove_zero_flows=True):

    # here starts the function:
    if al < 0. or al > 1:
        print("Chosen alpha parameter must lie between [0,1].")
        return

    assert abs(np.sum(supply_arr) - np.sum(demand_arr)) < 1e-9, "Given supply and demand arrays do not match."
    assert len(supply_arr) == coords_sources.shape[0], "mismatch in number of sources regarding supply_arr and coords_sources!"
    assert len(demand_arr) == coords_sinks.shape[0], "mismatch in number of sinks regarding demand_arr and coords_sinks!"

    # convert children_dict to nx.Graph:
    if isinstance(topo, dict):
        children_dict = topo.copy()
        topo = nx.Graph()
        try:
            for parent in children_dict:
                for child in children_dict[parent]:
                    topo.add_edge(parent, child)
        except:
            print("convertion of topo in form of children_dict to graph has failed.")
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

    # remove all nodes with degree 2 or 0:
    node_list_c = list(topo.nodes)
    for node in node_list_c:
        if topo.degree(node) == 0:
            topo.remove_node(node)
        elif topo.degree(node) == 2 and node in list_bp_idx:
            end1, end2 = topo.neighbors(node)
            topo.add_edge(end1, end2)
            topo.remove_node(node)

    # add positions to the nodes:
    dim = len(coords_sources[0])
    coords_arr = np.vstack((coords_sources, coords_sinks, np.random.random((num_bps, dim))))

    return topo, coords_arr, list_bp_idx





    


    
