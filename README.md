# Branched Optimal Transport (BOT)

This repository contains the code supplementary to the NeurIPS 2022 submission *Theory and Approximate Solvers for Branched Optimal Transport with Multiple Sources*. <br>
Here, we provide code for the following algorithms and experiments:
1) the numerical scheme to systematically check inequality Γ<sub>2</sub>,
2) the geometric construction algorithm for solutions with optimal geometry,
3) the numerical BP optimization routine,
4) the Brute-force solver for topology optimization 
5) the heuristic for the topology optimization. 

For instructions on setting up an anaconda environment see [environment.txt](https://github.com/hci-unihd/BranchedOT/blob/main/environment.txt).

<br>

BOT is an NP-hard optimization problem. Given a set of sources (below in red) and sinks (below in blue), the objective is to optimize a transportation network with respect to the following cost function:

<img src="https://user-images.githubusercontent.com/73332106/146935786-2f133488-0daf-4349-9e55-79f13e1705b2.png" 
     width="400"  />

where α is a parameter between 0 and 1. It determines how much more efficient it is to transport mass with nearby destinations along the same route. The mathematical reason is the subadditivity of (m<sub>2</sub> + m<sub>2</sub>)<sup>α</sup> < m<sub>1</sub><sup>α</sup> + m<sub>2</sub><sup>α</sup>. The smaller α the more branching is observed in optimal solutions. The case of α=0 corresponds to the Euclidean Steiner Tree Problem. For α=1, BOT relaxes to the efficiently solvable problem of Optimal Transport (no more branching). The BOT optimization can be split into a convex geometric optimization of the branching point positions and a combinatorial optimization of the topology.     

<br>

<b>1) Inequality check </b>

<img src="https://user-images.githubusercontent.com/73332106/146936223-af44d771-86e9-4e65-9c34-4b70e6a7bc68.png" 
     width="300"  />

By proving the inequality Γ<sub>2</sub>, one can show that degree-4 branchings in a BOT solution are never globally optimal.  
The white region is the only region which could not be ruled out analytically and which is therefore dealt with numerically. <br>
The code can be found [here](https://github.com/hci-unihd/BranchedOT/blob/main/experiments/exp_numerical_check_hhf_inequality.py).

<br>

<b>2) Geometric construction of solutions with optimal geometry for a given topology  </b>

<img src="https://user-images.githubusercontent.com/73332106/146936548-ae11b15b-7a7c-4821-93d1-a26956177b4c.png" 
     width="300"  />
     
The presented geometric construction with so-called pivot points and pivot circles was generalized in the thesis to be applicable to BOT problems with multiple sources. The construction is very efficient, but works only if the optimal solution does not contain any edges contractions. The algorithm is only applicable to BOT problems in the Euclidean plane.  <br>
More examples can be found [here](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Experiment%20-%20Geometric%20construction%20solver.ipynb).

<br>

<b>3) Numerical optimization of the BP configuration for a given topology  </b>

<img src="https://user-images.githubusercontent.com/73332106/146946443-77519606-93e7-4b88-afcb-4285c1653b8e.gif" 
     width="400"  />
     
     
This numerical optimization routine is an effective algorithm to optimize the BP configuration for a given tree topology.
It is applicable in two- and higher-dimensional Euclidean space for all tree topologies. It is the basis of all developed heuristics which address the the combinatorial of the BOT topology (see below). <br>
The respective code can be found [here](https://github.com/hci-unihd/BranchedOT/blob/main/src/iterative_geometry_solver.py).
In `notebooks` please find the [runtime analysis](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Experiment%20-%20Geometry%20optimization%20runtime.ipynb),
and the [scaling analysis](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Evaluation%20-%20Geometry%20optimization%20scaling.ipynb).

<br>

<b>4) Brute-force topology optimization  </b>

<img src="https://user-images.githubusercontent.com/73332106/146962214-19121365-94a5-4c46-9b23-024c7ea46b63.png" 
     width="600"  />

The number of possible tree topologies for BOT grow super-exponentially with the problem size. For 9 terminals, there exist already 135135 distinct topologies. Trying them all out is computationally very costly. A ground truth example and the histogram of all costs of the different topologies is shown below. For larger problems all brute-force approahes become infeasible.    <br>
More ground truth examples from brute-force search can be found [here](https://github.com/hci-unihd/BranchedOT/tree/main/results/brute%20forced%20examples).

<br>

<b>5) Heuristics for topology optimization  </b>

<img src="https://user-images.githubusercontent.com/73332106/146940339-3d862bbd-ff07-416d-8b79-f6df33e2b69f.gif" 
     width="400"  />

In `notebooks` please find the [comparison of the greedy heuristic to brute-force solutions](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Evaluation%20-%20Brute-Force%20vs.%20Heuristic.ipynb), 
[number of iterations analysis](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Evaluation%20-%20Greedy%20topology%20optimization.ipynb), 
[influence of the kernel width](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Evaluation%20-%20Kernelwidth%20in%20topology%20optimization.ipynb) and 
a [topology optimization step by step](https://github.com/hci-unihd/BranchedOT/blob/main/notebooks/Experiment%20-%20Topology%20optimization%20step%20by%20step.ipynb).
