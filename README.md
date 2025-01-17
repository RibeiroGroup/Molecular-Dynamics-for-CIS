# Molecular Dynamic and Coulomb-Gauge Classical Electrodynamics simulation

By Tuan Nguyen

## DEPENDENCIES:
python >= 3.10 (lower version is probably okay)  
numpy  
matplotlib  

## USAGE
### Running simulation from command line
Monte Carlo simulation of Argon - Xenon collisions in the presence of multimodes electromagnetic field with `src/simulation_monte.py`
```
$cd src
$python simulation_monte.py -s 100110 -t 100 -f cavity -a boltzmann -c Lxy1e1 -r 1
```
For running simulation of Monte Carlo collision of argon-xenon in a cavity with reduced z dimension (e.g. 1 micrometer), cavity field with amplitude sampled from Boltzmann distribution. Both the field and atoms are sampled at 100K.
Arguments:
- s (int, required): random seed
- t (float, required): temperature for atomic gas and the field is the field amplitude is Boltzmann-distributed
- f (str, default: None): type of electromagnetic field, valid args: 'cavity', 'free' or left blank for no field.
- a (str, default: 'zero'): 'boltzmann' for initiating field's amplitude to Boltzmann distribution with temperature specified in argument -t, 'zero' for initiating all to be zeros
- c (str, default: None): coupling strength, dipole scalar, gamma. Must specified string that is map to certain value in dictionary `coupling_strength_dict` in config.py
- r (int default: 0): specified 1 to reduced z dimension, 0 (default) otherwise
- m (int, default: 0): minimum integer n that correspond to the minimum wavevector. If 0 is provided, value from config.py will be used (recommended)
- n (int, default: 0): maximum integer n that correspond to the maximum wavevector If 0 is provided, value from config.py will be used (recommended)

Additional modification to the simulation parameter can be done in `config.py`, for instance, number of argon-xenon pair can be changed in `N_atom_pairs` variable.

### Notebooks:
- `analyze_single.ipynb`: running simulations involving single pair collision presented in the the MS.
- `analyze_monte_carlo.ipynb`: analyzing output `.pkl` files from `simulation_monte.py` and plotting.
- `error.ipynb`: simulation of single pair collision and plotting/analyzing energy conservation
- `animation.ipynb`: simulation of single pair collision and plotting/animating the collision
- `distribution.ipynb`: examine the distribution of the field's amplitude





