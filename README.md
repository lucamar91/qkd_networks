Modeling and simulation of QKD networks.

### Code description
- qkd_networks.py: is the main script. When executed it produces several outputs explained below;
- network_funcs.py: contains all the user-defined functions to generate, handle and visualize instances of the ${S}^2$ network model;
- q_opt_funcs.py: contains the functions needed to compute the key rate bounds for Continuous-Variable QKD and for the BB84 protocol (DVQKD);
- plots.ipynb: the jupyter notebook used for the generation of the pictures in the paper.

### Requirements
- python 3.x

These libraries must be installed:
- numpy
- scipy
- matplotlib
- os
- time
- networkX
- heapq

Two subdirectories should be created in the directory containing the code:
- outputs/ : the output files of the main code are stored here;
- plots/ : the plots produced in the plots.ipynb notebook are saved here.

### The outputs
Output files are given a name depending on a certain set of parameters (including the number of nodes $N$) and on the quantity computed:

- `*rhos_aspl*` and `*rhos_parK*`: respectively, the average shortest path length and the corresponding average key rate;
- `*rhos_conn*`: the connectivity of the system (average relative size of the giant component);
- `*rhos_geod*`: average geodesic distance between the nodes in the network;
- `*rhos_topd*`: average topological distance between the nodes in the network;
- `*rhos_clus*`: clustering coefficient of the network.

The files listed above are organised as arrays of three rows, containing (1) the set of densities of nodes in the network, (2) the quantity of interest corresponding to the given densities, (3) the error bars.

One last file, `*rhos_degr*`, is saved in a different format: the first row contains again the densities, and the column under each density of this row represents the histogram of the corresponding average degree distribution.
