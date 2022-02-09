# BOT experiments

<b> Required Packages: </b>
*numpy
*matplotlib
*networkx
*pickle
*concurrent.futures  # for multiprocessing  
*tqdm
*POT (python Optimal transport, https://pythonot.github.io/)

<b> Python Files: </b>
1) Runtime of generalized Smith solver:<br>
--> Experiment - Smith solver performance.ipynb

2) Greedy heuristic from star graph to strong topology:<br>
--> Experiment - MC star step by step.ipynb

3) Brute force solver and comparison with greedy heuristic:<br>
--> brute-force_solver.py (to generate data)<br>
--> Analysis - Brute-Force.ipynb  (for analysis)

4) Analysis - Iterations until convergence in greedy heuristic, plus some example solutions:<br>
--> MC_star_baseline.py (to generate data) <br>
--> Analysis - MC star baseline.ipynb (for analysis)

5) Experiments shown in Fig. 1 of paper:<br>
--> Fancy experiments.ipynb

6) numerical inequality check of \Gamma_{2,1}:<br>
--> tesselate_hhf.py <br>
need to specify: <br>
a_up = alpha upper limit <br>
m1_low = m_1 lower limit <br>
and threshold = when do we accept the lower bound of Gamma2,1 as positive.

<br>

"results" contain computed data/results, so one does not have to recompute (which requires multithreading and some time in many cases).


