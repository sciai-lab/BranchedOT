import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

sys.path.append('/')

from general_preprocessing import *

def eucl_dist(x, Y):
    dim_x = len(x)
    if len(x.shape) > 1:
        dim_x = x.shape[1]
    if Y.shape == (dim_x,):
        return np.sqrt(np.sum((Y - x) ** 2, axis=0))
    elif Y.shape[1] == dim_x:
        return np.sqrt(np.sum((Y - x) ** 2, axis=1))
    else:
        print("dim error")


# rotate around (0,0):
def phi_from_x(x_arr):
    eps = 1e-7
    x, y = x_arr
    if x > eps:
        return np.arctan(y / x)
    elif x < -eps:
        if abs(y) > eps:
            return np.arctan(y / x) + np.pi * np.sign(y)
        else:
            return np.pi
    else:
        return np.pi / 2 * np.sign(y)



def calc_pivot_point(m1, m2, A1, A2, al, common_source, debug_plot):
    """
    Inputs:
    m1 - flow to the left, m2 - flow to the right
    A1 - coordinates of A1, A2 - coordinates of A2
    al - alpha value
    common_source = position of common source
    Output:
    Calculates the pivot point. 
    """
    if not ((al >= 0) & (al < 1)):
        print("invalid alpha. For alpha = 1 use usual OT.")
        return

    if m1 < 1e-6 or m2 < 1e-6:
        if debug_plot:
            print("WARNING! either m1 or m2 are zero.")

    if m1 < 1e-6 and m2 < 1e-6:
        m2 += 1e-6
        if debug_plot:
            print("WARNING! both m1 and m2 are zero.")

    # distance between A1 and A2:
    A1A2 = np.sqrt((A1[0] - A2[0]) ** 2 + (A1[1] - A2[1]) ** 2)

    # calculate th1 and th2:
    # print("m1=", m1,"m2=", m2)
    k1 = max(m1 / (m1 + m2), 1e-7);
    k2 = max(1 - k1, 1e-7)
    # print("k1=",k1,"k2=",k2)
    th1 = np.arccos((k1 ** (2 * al) + 1 - k2 ** (2 * al)) / (2 * k1 ** al))
    th2 = np.arccos((k2 ** (2 * al) + 1 - k1 ** (2 * al)) / (2 * k2 ** al))

    # FORWARD TRAFO of A1 and A2:

    # put A1 into the origin:
    A1_new = np.array([0, 0]);
    A2_new = A2 - A1;
    common_source_new = common_source - A1

    # calculate the angle to rotate A1 and A2 onto the x_axis:
    phi = phi_from_x(A2_new)
    # rotate A2 by -phi onto the x_axis:
    rot_matr1 = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])
    A2_new2 = rot_matr1 @ A2_new
    common_source_new2 = rot_matr1 @ common_source_new

    # put them at the right place:
    A1_x = np.array([-A1A2 / 2, 0]);
    A2_x = A2_new2 - np.array([A1A2 / 2, 0])
    common_source_x = common_source_new2 - np.array([A1A2 / 2, 0])

    # calculate r:
    r = A1A2 / (2 * np.sin(th1 + th2))
    # calculate centre of circle O_x and O_original to later be able to draw the circle:
    O_x = r * np.array([0, -np.cos(th1 + th2)])

    switch = False
    # get the pivot point in the upper half plane if common_source y-component is negative:
    if common_source_new2[1] < 0:
        A12 = r * np.array([-np.sin(th2 - th1), np.cos(th2 - th1) - np.cos(th2 + th1)])
        halfplane = 1
    else:
        # reflect everything at x-axis:
        A12 = np.array([1, -1]) * r * np.array([-np.sin(th2 - th1), np.cos(th2 - th1) - np.cos(th2 + th1)])
        O_x = np.array([1, -1]) * O_x
        switch = True
        halfplane = -1

    # BACKWARD TRAFO of A12 and O:
    rot_matr2 = np.array([
        [np.cos(-phi), np.sin(-phi)],
        [-np.sin(-phi), np.cos(-phi)]
    ])

    A12_original = rot_matr2 @ (A12 + np.array([A1A2 / 2, 0])) + A1
    O_original = rot_matr2 @ (O_x + np.array([A1A2 / 2, 0])) + A1

    return A12_original, np.array(
        [r, O_original[0], O_original[1], th1, th2, A1A2, phi, A12[0], A12[1], switch]), halfplane




def pivot_to_branching(A12, A1, A2, A3, param_arr):
    """
    Inputs:
    positions of A1 and A2 as well as the 2 pivot points and the position of the previous branching point A3 

    Returns: position of respective optimal branching point.
    """
    L_pivot_correction = False

    r, O1, O2, th1, th2, A1A2, phi, A12_x0, A12_x1, switch = param_arr  # all the parameters from calculating the pivot point

    A12_x = np.array([A12_x0, A12_x1])

    # rotate A2 by -phi onto the x_axis:
    rot_matr1 = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])

    # FORWARD TRAFO OF A3:
    A3_x = rot_matr1 @ (A3 - A1) - np.array([A1A2 / 2, 0])

    # SWITCH FACTOR:
    switch_factor = 1.
    if switch:
        # print("switched!")
        switch_factor = -1.

    # alternative calculation of lam1 and lam2:
    th = th1 + th2
    v1, v2 = A12_x - A3_x
    W = A3_x[1] + switch_factor * r * np.cos(th)
    P = (2 * v1 * A3_x[0] + 2 * v2 * W) / (v1 ** 2 + v2 ** 2 + 1e-9)
    Q = (A3_x[0] ** 2 + W ** 2 - r ** 2) / (v1 ** 2 + v2 ** 2 + 1e-9)
    # lam1_alt = - P/2 + np.sqrt(P**2/4 - Q)
    lam2_alt = - P / 2 - np.sqrt(np.max([P ** 2 / 4 - Q, 0.]))

    # branch candidate calculated  from lam2:
    branch_candidate = lam2_alt * (A12_x - A3_x) + A3_x

    # but only accept it if it is not in same half plane as the pivot point:
    candidate = False
    if np.sign(branch_candidate[1]) * np.sign(A12_x[1]) == -1:
        # candidate and pivot lie not in same half plane
        candidate = True

    eps = 1e-7
    branch_point_original = np.array([0, 0])
    V_branch_bool = False
    # V-shaped branching:
    if lam2_alt <= 0:
        # branching point is at A3:
        branch_point_original = A3 * 1
        V_branch_bool = True

    # Y-shaped branching:
    elif candidate & (lam2_alt > 0) & (lam2_alt < 1 - eps):
        branch_point_original = lam2_alt * (A12 - A3) + A3  # no need for backtrafo the A_i are already the right one

    # L-shaped branching:
    else:
        B = lam2_alt * (A12 - A3) + A3
        branch_point_original = np.array([None, B[0], B[1]])
        L_pivot_correction = True

    return branch_point_original, L_pivot_correction, V_branch_bool




def calc_pivot_point_case2(m1, m2, A1, A2, al, common_source, debug_plot):
    
    """
    Inputs:
    m1 - flow to the left, m2 - flow to the right
    A1 - coordinates of A1, A2 - coordinates of A2

    NOTE: A2 has the combined flow! Meaning that in the converging case it is the node with flow from the branching point.
    And in the diverging case it is the node with flow into the branching point. Hence need to also find out 
    whether we have a converging or diverging scenario. Since m2 = m1 + m3, we can much simpler find out 
    which one is A2 by comparing the two m_i find. 

    al - alpha value
    common_source = position of common source
    Output:
    Calculates the pivot point. 
    """
    
    if not ((al >= 0) & (al < 1)):
        print("invalid alpha. For alpha = 1 use usual OT.")
        return
        # find out which one needs to be the A2. Ie the one with the larger absolute flow value:
    mass_switch = False
    if m1 > m2:
        # switch A1 and A2 and their respective masses:
        mass_switch = True
        A1_old = A1 * 1
        m1_old = m1 * 1
        A1 = A2 * 1
        m1 = m2 * 1
        A2 = A1_old * 1
        m2 = m1_old * 1

    if m1 < 1e-6 or m2 < 1e-6:
        if debug_plot:
            print("WARNING! either m1 or m2 are zero.")
    if abs(m1 - m2) < 1e-6:
        if debug_plot:
            print("WARNING! case2 to with m1=m2.")
            # the flow towards the branching point is zero! in this case L-branching is necessary

    if m1 < 1e-6 and m2 < 1e-6:
        m2 += 1e-6
        if debug_plot:
            print("WARNING! both m1 and m2 are zero.")

    # distance between A1 and A2:
    A1A2 = np.sqrt((A1[0] - A2[0]) ** 2 + (A1[1] - A2[1]) ** 2)

    # calculate th1 and th2:
    # print("m1=", m1,"m2=",m2)
    k2 = max(m1 / m2, 1e-7);
    k1 = max(1 - k2, 1e-7)
    # print("k1=",k1,"k2=",k2)
    th1 = np.arccos((k1 ** (2 * al) + 1 - k2 ** (2 * al)) / (2 * k1 ** al))
    th2 = np.arccos((k2 ** (2 * al) + 1 - k1 ** (2 * al)) / (2 * k2 ** al))

    # print("th1=",th1,"th2=",th2)

    # from this calculate phi's:
    phi1 = np.pi - th1 - th2
    phi2 = th1
    th1 = phi1 * 1
    th2 = phi2 * 1

    # FORWARD TRAFO of A1 and A2:

    # put A1 into the origin:
    A1_new = np.array([0, 0]);
    A2_new = A2 - A1;
    common_source_new = common_source - A1

    # calculate the angle to rotate A1 and A2 onto the x_axis:
    phi = phi_from_x(A2_new)
    # rotate A2 by -phi onto the x_axis:
    rot_matr1 = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])
    A2_new2 = rot_matr1 @ A2_new
    common_source_new2 = rot_matr1 @ common_source_new

    # put them at the right place:
    A1_x = np.array([-A1A2 / 2, 0]);
    A2_x = A2_new2 - np.array([A1A2 / 2, 0])
    common_source_x = common_source_new2 - np.array([A1A2 / 2, 0])

    # calculate r:
    r = A1A2 / (2 * np.sin(th1 + th2))
    # calculate centre of circle O_x and O_original to later be able to draw the circle:
    O_x = r * np.array([0, -np.cos(th1 + th2)])

    switch = False
    # get the pivot point in the upper half plane if common_source y-component is negative:
    if common_source_new2[1] < 0:
        A12 = r * np.array([-np.sin(th2 - th1), np.cos(th2 - th1) - np.cos(th2 + th1)])
        halfplane = 1
    else:
        # reflect everything at x-axis:
        A12 = np.array([1, -1]) * r * np.array([-np.sin(th2 - th1), np.cos(th2 - th1) - np.cos(th2 + th1)])
        O_x = np.array([1, -1]) * O_x
        switch = True
        halfplane = -1

    # BACKWARD TRAFO of A12 and O:
    rot_matr2 = np.array([
        [np.cos(-phi), np.sin(-phi)],
        [-np.sin(-phi), np.cos(-phi)]
    ])

    A12_original = rot_matr2 @ (A12 + np.array([A1A2 / 2, 0])) + A1
    O_original = rot_matr2 @ (O_x + np.array([A1A2 / 2, 0])) + A1

    return A12_original, np.array(
        [r, O_original[0], O_original[1], th1, th2, A1A2, phi, A12[0], A12[1], switch, mass_switch]), halfplane




def pivot_to_branching_case2(A12, A1, A2, A3, param_arr):
    """
    Inputs:
    positions of A1 and A2 as well as the pivot point and the position of the previous branching point A3 

    Returns: position of respective optimal branching point.
    """
    L_pivot_correction = False

    r, O1, O2, th1, th2, A1A2, phi, A12_x0, A12_x1, switch, mass_switch = param_arr  # all the parameters from calculating the pivot point

    # only have mass_switch in case2:
    if mass_switch:
        # switch A1 and A2
        A1_old = A1 * 1
        A1 = A2 * 1
        A2 = A1_old * 1

    A12_x = np.array([A12_x0, A12_x1])

    # rotate A2 by -phi onto the x_axis:
    rot_matr1 = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])

    # FORWARD TRAFO OF A3:
    A3_x = rot_matr1 @ (A3 - A1) - np.array([A1A2 / 2, 0])

    # SWITCH FACTOR:
    switch_factor = 1.
    if switch:
        # print("switched!")
        switch_factor = -1.

    # alternative calculation of lam1 and lam2:
    th = th1 + th2
    v1, v2 = A12_x - A3_x
    W = A3_x[1] + switch_factor * r * np.cos(th)
    P = (2 * v1 * A3_x[0] + 2 * v2 * W) / (v1 ** 2 + v2 ** 2 + 1e-9)
    Q = (A3_x[0] ** 2 + W ** 2 - r ** 2) / (v1 ** 2 + v2 ** 2 + 1e-9)
    # lam1_alt = - P/2 + np.sqrt(P**2/4 - Q)
    lam2_alt = - P / 2 - np.sqrt(np.max([P ** 2 / 4 - Q, 0]))

    # print("lam1_alt = ", lam1_alt)
    # print("lam2_alt = ", lam2_alt)

    # branch candidate calculated  from lam2:
    branch_candidate = lam2_alt * (A12_x - A3_x) + A3_x

    # but only accept it if it is not in same half plane as the pivot point:
    candidate = False
    if np.sign(branch_candidate[1]) * np.sign(A12_x[1]) == -1:
        # candidate and pivot lie not in same half plane
        candidate = True

    eps = 1e-7
    branch_point_original = np.array([0, 0])
    # L1-shaped branching (former V-shaped branching):
    V_branch_bool = False
    if lam2_alt <= 0:
        # branching point is at A3:
        branch_point_original = A3 * 1
        V_branch_bool = True

    # Y-shaped branching (also former Y-shaped branching):
    elif candidate & (lam2_alt > 0) & (lam2_alt < 1 - eps):
        # print("Y")
        branch_point_original = lam2_alt * (A12 - A3) + A3  # no need for backtrafo the A_i are already the right one

    # L2 and V-shaped branching (former L_shape):
    else:
        B = lam2_alt * (A12 - A3) + A3
        branch_point_original = np.array([None, B[0], B[1]])
        L_pivot_correction = True

    return branch_point_original, L_pivot_correction, V_branch_bool


# define a branching point class with all the important properties:
class branching_point_class:

    # class variables:

    # runs every time new wertpapier is created
    def __init__(self, children_flow=None, parent_flow=None, coords=None, pivot_point=None, common_source=None,
                 halfplane=None, interm_results=None, case=None, predecessors=None, L_fixed=False):
        # parents_flow is a 2xn numpy array with the indices and edge flows to the parents

        self.children_flow = children_flow
        self.parent_flow = parent_flow
        self.coords = coords
        self.pivot_point = pivot_point
        self.common_source = common_source
        self.halfplane = halfplane
        self.interm_results = interm_results
        self.case = case
        self.predecessors = predecessors
        self.L_fixed = L_fixed


def calc_total_cost(al, branching_point_dict, ext_coords, list_ext_label):
    cost = 0
    for branching_point_label in branching_point_dict:
        branching_point = branching_point_dict[branching_point_label]
        # get flow and distance to parents and children:
        child1, m1, child2, m2 = branching_point.children_flow.reshape(-1)
        child1 = int(child1)
        child2 = int(child2)
        parent = int(branching_point.parent_flow[0])
        m3 = branching_point.parent_flow[1]
        if child1 in list_ext_label:
            cost += abs(m1) ** al * eucl_dist(branching_point.coords, ext_coords[child1])
        else:
            cost += abs(m1) ** al * eucl_dist(branching_point.coords, branching_point_dict[child1].coords)
        if child2 in list_ext_label:
            cost += abs(m2) ** al * eucl_dist(branching_point.coords, ext_coords[child2])
        else:
            cost += abs(m2) ** al * eucl_dist(branching_point.coords, branching_point_dict[child2].coords)
        if parent in list_ext_label:
            # only add cost if parent is external to avoid double counting
            cost += abs(m3) ** al * eucl_dist(branching_point.coords, ext_coords[parent])

    return cost


def visualise_current_sol(branching_point_dict, coords_sources, coords_sinks, ext_coords, list_ext_label, supply_arr,
                          demand_arr, supply_arr_full, demand_arr_full, fov, save, labelled, debug_plot,
                          iteration=None):
    # fov can be customized or can be None, then the whole thing is plotted.
    if iteration == None:
        fig = plt.figure(-1, figsize=(8, 8))
        #plt.title("final solution")
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.title("debug plot, iter=" + str(iteration))

    linescale = 15 / sum(supply_arr_full)  # defines thickness of edges relative to total flow
    markerscale = 25 / sum(supply_arr_full)

    # plot sources and sinks:
    sources_labelled = False
    sinks_labelled = False
    pivot_labelled = False
    for i, x_arr in enumerate(coords_sources):
        supply = supply_arr[i]
        if not sources_labelled and not labelled and len(branching_point_dict) != 0:
            # use 3 as a base value
            plt.plot(x_arr[0], x_arr[1], marker="o", color="r", linestyle="", markersize=markerscale * supply + 3,
                     alpha=0.7, label="sources", zorder=1)
            sources_labelled = True
        else:
            plt.plot(x_arr[0], x_arr[1], marker="o", color="r", linestyle="", markersize=markerscale * supply + 3,
                     alpha=0.7, zorder=1)
    for i, x_arr in enumerate(coords_sinks):
        demand = demand_arr[i]
        if not sinks_labelled and not labelled and len(branching_point_dict) != 0:
            plt.plot(x_arr[0], x_arr[1], marker="o", color="b", linestyle="", markersize=markerscale * demand + 3,
                     alpha=0.7, label="sinks", zorder=1)
            sinks_labelled = True
        else:
            plt.plot(x_arr[0], x_arr[1], marker="o", color="b", linestyle="", markersize=markerscale * demand + 3,
                     alpha=0.7, zorder=1)

    if len(branching_point_dict) == 0:
        source = coords_sources[0]
        sink = coords_sinks[0]
        plt.plot([source[0], sink[0]], [source[1], sink[1]], color="black", linewidth=linescale * supply_arr[0] + 1)
    else:
        for branching_point_label in branching_point_dict:
            branching_point = branching_point_dict[branching_point_label]
            # get flow and distance to parents and children:
            child1, m1, child2, m2 = branching_point.children_flow.reshape(-1)
            child1 = int(child1)
            child2 = int(child2)
            parent = int(branching_point.parent_flow[0])
            m3 = branching_point.parent_flow[1]
            B = branching_point.coords
            A1 = ext_coords[child1] if (child1 in list_ext_label) else branching_point_dict[child1].coords
            A2 = ext_coords[child2] if (child2 in list_ext_label) else branching_point_dict[child2].coords
            if iteration == None:
                plt.plot([A1[0], B[0]], [A1[1], B[1]], color="black", linewidth=linescale * m1 + 1, zorder=-1)
                plt.plot([A2[0], B[0]], [A2[1], B[1]], color="black", linewidth=linescale * m2 + 1, zorder=-1)
            else:
                plt.plot([A1[0], B[0]], [A1[1], B[1]], color="black", zorder=-1)
                plt.plot([A2[0], B[0]], [A2[1], B[1]], color="black", zorder=-1)

            if parent in list_ext_label:
                A3 = ext_coords[parent]
                if iteration == None:
                    plt.plot([A3[0], B[0]], [A3[1], B[1]], color="black", linewidth=linescale * m3 + 1, zorder=-1)
                else:
                    plt.plot([A3[0], B[0]], [A3[1], B[1]], color="black", zorder=-1)

            # plot pivot points
            pivot = branching_point.pivot_point
            if pivot[0] != None:
                if not pivot_labelled and not labelled:
                    plt.plot(pivot[0], pivot[1], "o", color="dodgerblue", markersize=5, label="pivot points",
                             zorder=1)
                    pivot_labelled = True
                else:
                    plt.plot(pivot[0], pivot[1], "o", color="dodgerblue", markersize=5, zorder=1)

                # draw the edge:
                plt.plot([B[0], pivot[0]], [B[1],pivot[1]], linestyle="dotted", color="gray", linewidth=2)

            # plot circles:
            r, O1, O2 = branching_point.interm_results[:3]
            # get current figure and axis and add artist to add circle:
            circle1 = plt.Circle((O1, O2), r, fill=False, color="gray")
            plt.gcf().gca().add_artist(circle1)

    if len(branching_point_dict) != 0:
        legend = plt.legend(fontsize=14)
        # make all markers the same size eventhough they are not in the image:
        for legend_handle in legend.legendHandles:
            legend_handle.set_markersize(10)

    plt.axis('equal')
    if isinstance(fov, np.ndarray):
        if fov.shape != (2, 2): print("Error. invalid fov.")
        plt.xlim(fov[0, 0], fov[0, 1])
        plt.ylim(fov[1, 0], fov[1, 1])
    if save and iteration is not None:
        a = coords_sources[0]
        b = branching_point_dict[-1].coords
        plt.plot([a[0],b[0]],[a[1],b[1]], color="black", linewidth=linescale * supply_arr[0] + 1, zorder=-1)
        plt.savefig("img.pdf", bbox_inches="tight")
        plt.show()
    if debug_plot:
        plt.show()
    return


# now build the geometric solver:
def geometric_bot_solver(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al,
                         left_label_dict, debug_plot=False,
                         plot=False, title="", fov=None, save=False, save_name="img"):
    # use general preprocessing function to calculate the edge-flows:
    topo, _, _ = preprocess_from_topo_to_flows(topo, supply_arr, demand_arr, coords_sources, coords_sinks, al)
    n = len(supply_arr) + len(demand_arr)
    if nx.number_of_nodes(topo) != 2*n - 2:
        print("This approach works only for full tree topologies.")
        return

    children_dict = dict(nx.bfs_successors(topo, 0))
    net_edgeflow_dict = {}
    for edge in topo.edges():
        if edge[0] < edge[1]:
            net_edgeflow_dict[(edge[0], edge[1])] = topo[edge[0]][edge[1]]["weight"]
        else:
            net_edgeflow_dict[(edge[1], edge[0])] = topo[edge[0]][edge[1]]["weight"]

    list_source_idx = list(np.arange(len(supply_arr)))
    list_sink_idx = list(np.arange(len(supply_arr), len(supply_arr) + len(demand_arr)))

    # here starts the whole pipeline:
    branching_point_dict = {}

    # combine the coordinates of sinks and sources:
    ext_coords = np.vstack((coords_sources, coords_sinks))
    list_ext_label = list_source_idx + list_sink_idx

    # need to treat the trivial special case, where one sink is filled by just one source:
    cost = 0
    if len(children_dict) == 1:
        # print("trivial case!!")
        if (0, 1) in net_edgeflow_dict:
            cost += net_edgeflow_dict[(0, 1)] ** al * eucl_dist(ext_coords[0], ext_coords[1])
        else:
            cost += net_edgeflow_dict[(1, 0)] ** al * eucl_dist(ext_coords[0], ext_coords[1])

        # plot the trivial solution:
        if debug_plot or plot:
            visualise_current_sol(branching_point_dict, coords_sources, coords_sinks, ext_coords,
                                  list_ext_label, supply_arr, demand_arr, supply_arr_full, demand_arr_full,
                                  fov=fov, save=save, labelled=False, debug_plot=debug_plot, iteration=None)

        return branching_point_dict, cost

    # set up new_topological order relative to arbitrarily chosen "boss source":
    boss_source = list_source_idx[0]
    new_topological_order = []
    for pair in list(nx.bfs_successors(topo, boss_source)):
        parent, children_list = pair
        new_topological_order.append(parent)
        if parent not in list_source_idx and parent not in list_sink_idx:
            if len(children_list) != 2:
                print("ERROR! Not exactly 2 children.")
            child1, child2 = children_list

            # get the flows to the children:
            flow1 = net_edgeflow_dict[(parent, child1)] if ((parent, child1) in net_edgeflow_dict) else - \
            net_edgeflow_dict[(child1, parent)]
            flow2 = net_edgeflow_dict[(parent, child2)] if ((parent, child2) in net_edgeflow_dict) else - \
            net_edgeflow_dict[(child2, parent)]

            # distiguish case 1 and case 2 according to diverging and converging flow:
            case = 1 if (np.sign(flow1) == np.sign(flow2)) else 2

            # initialise branching point:
            branching_point_dict[parent] = branching_point_class(
                children_flow=np.array([[child1, abs(flow1)], [child2, abs(flow2)]]),
                case=case,
                common_source=boss_source,  # in the first iteration the boss source acts as common source!
                predecessors=[])

    for pair in list(nx.bfs_predecessors(topo, 0)):
        child, parent = pair
        if child in branching_point_dict:
            flow = net_edgeflow_dict[(parent, child)] if ((parent, child) in net_edgeflow_dict) else net_edgeflow_dict[
                (child, parent)]
            branching_point_dict[child].parent_flow = np.array([parent, flow])

    # -------- START BACKWARD PASS FOR FINDING THE PIVOT POINTS: -----------
    for branching_point_label in new_topological_order[::-1]:
        if branching_point_label not in branching_point_dict:
            # the label is of an external node. So continue to next loop.
            continue

        # get branching_point instance:
        branching_point = branching_point_dict[branching_point_label]

        # get case at branching point, flows and coordinates of children A1 and A2:
        case = branching_point.case
        child1, m1, child2, m2 = branching_point.children_flow.reshape(-1)
        child1 = int(child1)
        child2 = int(child2)
        A1 = ext_coords[child1] if (child1 in list_ext_label) else branching_point_dict[child1].pivot_point
        A2 = ext_coords[child2] if (child2 in list_ext_label) else branching_point_dict[child2].pivot_point

        # manage position of the common source:
        # based on the position of the children and the left_label_dict:
        if left_label_dict[branching_point_label] == child1:
            common_source = A1 + np.array([[0, 1], [-1, 0]]) @ (A2 - A1)
        else:
            common_source = A1 + np.array([[0, -1], [1, 0]]) @ (A2 - A1)

        # have all ingredients to call calc_pivot_point function:
        # so find pivot point according to case 1 or case 2:
        if case == 1:
            pivot_point, interm_results, halfplane = calc_pivot_point(m1, m2, A1, A2, al, common_source, debug_plot)
        else:
            pivot_point, interm_results, halfplane = calc_pivot_point_case2(m1, m2, A1, A2, al, common_source,
                                                                            debug_plot)

        branching_point.pivot_point = pivot_point

        # but always update halfplane and interm_results:
        branching_point.halfplane = halfplane
        branching_point.interm_results = interm_results * 1

    # --------START FORWARD PASS TO FIND BRANCHING POINTS: --------
    # always go through full topo order for branching points and use the info about L-branching that we have
    for branching_point_label in new_topological_order:
        if branching_point_label not in branching_point_dict:
            # the label is of an external node. So continue to next loop.
            continue

        # get branching_point instance:
        branching_point = branching_point_dict[branching_point_label]

        # get pivot_point A12, children pivot_point coords A1, A2, parent coords A3, interm_results:
        A12 = branching_point.pivot_point
        child1, _, child2, _ = branching_point.children_flow.reshape(-1)
        child1 = int(child1)
        child2 = int(child2)
        A1 = ext_coords[child1] if (child1 in list_ext_label) else branching_point_dict[child1].pivot_point
        A2 = ext_coords[child2] if (child2 in list_ext_label) else branching_point_dict[child2].pivot_point
        parent = int(branching_point.parent_flow[0])
        A3 = ext_coords[parent] if (parent in list_ext_label) else branching_point_dict[parent].coords
        A12 = branching_point.pivot_point

        # now have all the ingredients to call:
        if branching_point.case == 1:
            branching_point_coords, L_pivot_correction, V_branch_bool = pivot_to_branching(A12, A1, A2, A3,
                                                                                           branching_point.interm_results)
        else:
            branching_point_coords, L_pivot_correction, V_branch_bool = pivot_to_branching_case2(A12, A1, A2, A3,
                                                                                                 branching_point.interm_results)

        if L_pivot_correction:
            print("An L branching did occur and the geometric construction is not optimal.")
            return
        branching_point.coords = branching_point_coords * 1

    # --------FORWARD PASS IS DONE TOO----------------

    # --------END OF WHILE LOOP!------
    # plot the resulting BOT solution:
    if plot:
        visualise_current_sol(branching_point_dict, coords_sources, coords_sinks, ext_coords, list_ext_label,
                              supply_arr, demand_arr, supply_arr, demand_arr, fov=fov, save=save, labelled=False,
                              debug_plot=debug_plot, iteration=None)

        # calculate cost:
    cost = calc_total_cost(al, branching_point_dict, ext_coords, list_ext_label)
    return branching_point_dict, cost
