import numpy as np
import matplotlib.pyplot as plt

#definitions of the functions h and f which appear in the inequality
def f(a, k):
    return np.arccos((k ** (2 * a) + 1 - (1 - k) ** (2 * a)) / (2 * k ** a))

def h(a, k):
    return f(a, k) + f(a, 1 - k)



def lower_bound_hhf(a_l, a_h, m1_l, m1_h, m2_l, m2_h):
    """
    lower bound for hhf inside the interval [a_l, a_h]x[m1_l, m1_h]x[m2_1,m2_h]
    using the monotonicities of h and f.
    """
    
    # identify mask for tiles which are outside the domain:
    outside_domain_mask = (m2_l / (1 - m1_l) > 1. - m1_l / (1 - m1_l))

    # clip s to the value of the domain edge:
    s_max = np.clip(m2_h / (1 - m1_h), None, 1. - m1_l / (1 - m1_l))
    r_max = m1_h / (m1_h + m2_l)

    mhhf = h(a_h, r_max) + f(a_h, s_max) - h(a_l, m1_l)
    mhhf[outside_domain_mask] = 1.
    return mhhf


# now implement oct-tree grid check:
def create_index(dim):
    index = np.zeros((2 ** dim, dim), dtype=int)
    for i in range(2 ** dim):
        string = np.binary_repr(i)
        while len(string) < dim:
            string = "0" + string
        for j, char in enumerate(string):
            index[i, j] = char

    return index



def divide_cuboid(low_arr, high_arr, index):
    """
    input:
    low_arr and high_arr define a cuboid by the edges: [low_arr[i], high_arr[i]].
    we may also have an array of low_arrs and high_arrs.

    output:
    a list of low_arr and high_arr pairs describing the edges of the new cuboids. The length of this list is:  
    2**dim. 
    """
    
    N, dim = low_arr.shape

    # get the stepsize and all lower corners:
    step_size_arr = (high_arr - low_arr) / 2  # dimension (N, dim)

    low_and_middle_corner = np.zeros((N, 2, dim))
    low_and_middle_corner[:, 0, :] = low_arr
    low_and_middle_corner[:, 1, :] = low_arr + step_size_arr

    low_corners_arr = low_and_middle_corner[:, index, np.arange(dim)]
    high_corners_arr = low_corners_arr + step_size_arr[:, None, :]

    low_corners_arr = low_corners_arr.reshape((-1, dim))  # dimension (2**(N*dim), dim)
    high_corners_arr = high_corners_arr.reshape((-1, dim))  # dimension (2**(N*dim), dim)

    return low_corners_arr, high_corners_arr

if __name__ == '__main__':
    a_up = np.float32(input("a_up = ? [e.g. 0.999]"))
    m1_low = np.float32(input("m1_low = [e.g. 0.001]"))
    threshold = np.float32(input("threshold = [e.g. 0.0001]"))

    # start test within the following bounds
    a_low = 0.5
    m1_up = 0.25
    m2_low = 0.25
    m2_up = 1 - 2 * m1_low

    all_low_corners_arr = np.array([
        [a_low, m1_low, m2_low]
    ])
    all_high_corners_arr = np.array([
        [a_up, m1_up, m2_up]
    ])

    _, dim = all_high_corners_arr.shape
    index = create_index(dim)

    success_percentage_tot = 0.
    while len(all_low_corners_arr) != 0:
        all_low_corners_arr, all_high_corners_arr = divide_cuboid(all_low_corners_arr, all_high_corners_arr, index)
        mhhf = lower_bound_hhf(all_low_corners_arr[:, 0], all_high_corners_arr[:, 0], all_low_corners_arr[:, 1],
                               all_high_corners_arr[:, 1], all_low_corners_arr[:, 2], all_high_corners_arr[:, 2])

        # create a mask where mhhf is larger than zero:
        mask = (mhhf < threshold)
        success_percentage = 1 - np.sum(mask) / len(mhhf)
        success_percentage_tot = success_percentage_tot + success_percentage * (1 - success_percentage_tot)
        all_low_corners_arr = all_low_corners_arr[
            mask]  # only split the cuboids further where inequality is not yet fulfilled
        all_high_corners_arr = all_high_corners_arr[mask]

        print(len(all_low_corners_arr))
        print("success_percentage_tot= ", success_percentage_tot)
