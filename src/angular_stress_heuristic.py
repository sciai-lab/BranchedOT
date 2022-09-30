import numpy as np
import networkx as nx
import sys

sys.path.append('/')
sys.path.append('../numerical BP optimization/')

from iterative_geometry_solver import iterative_geometry_solver


def get_phi(x_arr):
    """
    input: Nx2 vector
    """
    assert x_arr.shape[1] == 2, "this is a 2d method"
    r = np.sqrt(np.sum(x_arr ** 2, axis=1))
    mask_r = (r > 1e-10)
    phi_arr = np.zeros(x_arr.shape[0])

    mask_y = (x_arr[:, 1] >= 0)
    phi_arr[mask_y * mask_r] = np.arccos(x_arr[mask_y * mask_r][:, 0] / r[mask_y * mask_r])
    phi_arr[np.invert(mask_y) * mask_r] = 2 * np.pi - np.arccos(
        x_arr[np.invert(mask_y) * mask_r][:, 0] / r[np.invert(mask_y) * mask_r])
    phi_arr[np.invert(mask_r)] = np.inf

    return phi_arr


def f(al, k):
    return np.arccos((k ** (2 * al) + 1 - (1 - k) ** (2 * al)) / (2 * k ** al))


# input ABSOLUTE VALUES of flows:
def get_stationary_angle(flow1, flow2, al, case):
    if case == 1:
        k = flow1 / (flow1 + flow2)
        return f(al, k) + f(al, 1 - k)

    if case == 2:
        m2 = np.max([flow1, flow2])
        m1 = np.min([flow1, flow2])
        k = (m2 - m1) / m2
        return np.pi - f(al, 1 - k)

def get_angular_stress(topo, coords_arr, al):
    # determine the center nodes:
    degree_arr = np.array([topo.degree()])[0]
    degree_mask = (degree_arr[:, 1] > 3)

    # list of BP centers where stress reduction is necessary:
    center_list = degree_arr[:, 0][degree_mask]

    all_closest_neighbours_list = []
    all_angular_stress_arr = np.array([])

    for center in center_list:
        for k in range(len(list(nx.neighbors(topo, center))) - 3):
            center_coords = coords_arr[center]

            # get all neigbours of center:
            neighbours = np.array(list(nx.neighbors(topo, center)))

            # get all pairs of angular neighbours, for now only as 2d method:
            phi_neighbours = get_phi(coords_arr[neighbours] - center_coords)
            angular_sorting = np.argsort(phi_neighbours)
            neighbours_sorted = neighbours[angular_sorting]
            phi_sorted = phi_neighbours[angular_sorting]

            for i, n1 in enumerate(neighbours_sorted):
                n2 = neighbours_sorted[i - 1]

                # calculate the angle between two neighbours:
                if np.abs(phi_sorted[i]) == np.inf or np.abs(phi_sorted[i-1]) == np.inf:
                    actual_angle = np.inf
                else:
                    actual_angle = np.abs(phi_sorted[i] - phi_sorted[i - 1])
                    if actual_angle > np.pi:
                        actual_angle = 2*np.pi - actual_angle

                # calculate the stationary angles:
                # first decide the case:
                flow1 = topo[center][n1]["weight"]
                flow2 = topo[center][n2]["weight"]

                case = 2
                if np.sign(flow1 * (center - n1) * flow2 * (center - n2)) == 1:
                    case = 1

                angular_discrepancy = get_stationary_angle(abs(flow1), abs(flow2), al, case) - actual_angle
                all_closest_neighbours_list.append([n1, n2, center])
                all_angular_stress_arr = np.append(all_angular_stress_arr, angular_discrepancy)

    return all_closest_neighbours_list, all_angular_stress_arr

# now stress reduction. For that we need the directions of flow to calculate the optimal angles based on case 1 or case 2.
# Therefore the general preprocessing function was altered such that the edge weigths have signs indicating the direction.

# input:topo which is not yet a full tree, coords_arr, topo, supply_arr, demand_arr.
def angular_stress_reduction(topo, bot_problem_dict, plot=False, plot_final=False):
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    # first BP location and flows for topo:
    cost, coords_arr = iterative_geometry_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                 relative_improvement_threshold=1e-6, min_iterations=30,
                                                 max_iterations=1000,
                                                 plot=plot, title="", fov=None, save=False, save_name="img")

    # determine how many reduction steps are necessary:
    degree_arr = np.array([topo.degree()])[0]
    degree_mask = (degree_arr[:, 1] > 3)
    degree_excess = np.sum(degree_arr[:,1][degree_mask] - 3)
    #print("degree_excess=", degree_excess)
    if degree_excess == 0:
        if plot_final:
            plot = True
        cost, coords_arr = iterative_geometry_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                     relative_improvement_threshold=1e-7, min_iterations=30,
                                                     max_iterations=2000,
                                                     plot=plot, title="", fov=None, save=False, save_name="img")


    #start iterations:
    label_min = np.min(topo.nodes())
    for iteration in range(degree_excess):
        # get all closest neighbour pairs and the correspinding angular stress:
        all_closest_neighbours_list, all_angular_stress_arr = get_angular_stress(topo, coords_arr, al)

        #print(f"{iteration}: number of pairs {len(all_angular_stress_arr)}.")

        # pick the neighbour couple with largest angular stress and insert a BP there:
        argmax = np.argmax(all_angular_stress_arr)
        n1, n2, center = all_closest_neighbours_list[argmax]

        topo.remove_edges_from([(n1, center), (n2, center)])
        label_min -= 1
        topo.add_edges_from([(center, label_min), (n1, label_min), (n2, label_min)])

        # optimize the BP positions and automatically calculate the flows
        # (that is done implicitly in the iterative solver):
        if iteration == degree_excess - 1 and plot_final:
            plot = True

        cost, coords_arr = iterative_geometry_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                                                     relative_improvement_threshold=1e-7, min_iterations=30,
                                                     max_iterations=2000,
                                                     plot=plot, title="", fov=None, save=False, save_name="img")

        # sanity check of the angle reduction: compare actual and theoretical angle now.
        flow1 = topo[label_min][n1]["weight"]
        flow2 = topo[label_min][n2]["weight"]

        case = 2
        if np.sign(flow1 * (label_min - n1) * flow2 * (label_min - n2)) == 1:
            case = 1

        stat_angle = get_stationary_angle(abs(flow1), abs(flow2), al, case)
        phi = get_phi(coords_arr[[n1, n2]] - coords_arr[label_min])
        actual_phi = np.abs(phi[1] - phi[0])
        if actual_phi > np.pi:
            actual_phi = 2 * np.pi - actual_phi

        neighbours = list(nx.neighbors(topo, label_min))
        dist = np.sqrt(np.sum((coords_arr[neighbours] - coords_arr[label_min])**2, axis=1))
        # calculate the deviation if neither L- nor V-branching:
        if (np.min(dist) > 1e-2) and (np.abs(stat_angle - actual_phi) > 1e-2):
            # print("case afterwards=", np.sign(flow1 * (label_min - n1) * flow2 * (label_min - n2)) == 1)
            # print("flows_afterwards=", flow1, flow2)
            # print("coords:", coords_arr[label_min], coords_arr[[n1, n2]])
            # print("actual_phi=", actual_phi)

            #print("angular discrepancy afterwards=", np.abs(stat_angle - actual_phi))
            pass

    return cost, topo

# if __name__ == '__main__':
#     x = np.ones(5)
#     x[[2,3]] = np.inf
#     for i,_ in enumerate(x):
#         y = np.abs(x[i] - x[i-1])
#         print(y)
