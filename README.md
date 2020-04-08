## Sighnet

Sighnet is the Brian2 implementation of the sigh generation model described in Severs et al.

Find the Brian2 documentation at https://brian2.readthedocs.io/en/stable/

------------------

### Running sighnet

To run sighnet, first create a conda environment from the included `environment.yaml` file

`conda env create -n sighnet -f environment.yaml`

`cd` to the data directory and run:

`python ../sighnet.py edgelist_N300_k6.csv example_sighnet` 

Where the `edgelist_N300_k6.csv` file defines the source and target synaptic connections.

This runs sighnet with all the default parameters. You can modify the parameter Ko, as in figure 3 of Severs et al. with the `--ko` flag:

`python ../sighnet.py example_sighnet --ko 2`

Once run, sighnet saves a pickle file to `example_sighnet.dat`. This file contains the saved brian2 states:

- 'synapses'
  - Contains the synaptic state variables and associated parameters (e.g., list of source and target neurons)
- 'spikes'
  - Contains the timestamps and neuron identities of all spikes
- 'neuron_state'
  - Contains the voltage and intracellular calcium for every timestep of the simulation for a subset (15 neurons) of the simulated population
- 'glia'
  - Contains the temporal evolution of "C" and "l", the slow external calcium oscillation
- 'prebot_pop'
  - Contains parameters and final timestep states of all neurons simulated
- 'pop_rate'
  - Contains the summed number of spikes in each timestep across the population. Likely needs to be smoothed to be useable.
- 'eqs'
  - The equations used for the simulation
- 'params'
  - All parameters passed in the namespace for the simulation

------------

### Postprocessing

Once run, the sigh frequency and smoothed population rates can be extracted with `postprocess.py`.

`postprocess` is intended to be run on a list of files that vary over the Ko parameter. e.g.:

```
data/
	sighnet_ko1.dat
	sighnet_ko2.dat
	sighnet_ko3.dat
	...
```

`python postprocess.py "./sighnet*.dat" --savemat --savecsv`

The flags `--savemat` and `--savecsv` allow for saving of smoothed population rates (for all files, a subset of 100s of simulation time), shapes of sigh and eupnic bursts, and inter-burst intervals for sighs and eupneas. The `csv` file is very small as it contains only the interburst intervals and the ko values, while the `mat` file is larger in that it contains population rates.







