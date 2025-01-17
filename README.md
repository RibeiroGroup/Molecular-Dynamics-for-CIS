# Molecular Dynamic and Coulomb-Gauge Classical Electrodynamics simulation

By Tuan Nguyen

## DEPENDENCIES:
python >= 3.10 (lower version is probably okay)  
numpy  
matplotlib  

## USAGE
### Running simulation from command line
Monte Carlo simulation of Argon - Xenon collisions in the presence of multimodes electromagnetic field
```
$python src/simulation_monte.py 
```
Arguments:
- s (int, required): random seed
- t (float, required): temperature for atomic gas and the field is the field amplitude is Boltzmann-distributed
- f (str, default: None): type of electromagnetic field, valid args: 'cavity', 'free' or blank
- a (str, default: 'zero'): 'boltzmann' for initiating field's amplitude to Boltzmann distribution with temperature specified in argument -t, 'zero' for initiating all to be zeros
- c (str, default: None): coupling strength, dipole scalar, gamma. Must specified string that is map to certain value in dictionary `coupling_strength_dict` in config.py
- r (int default: 0): specified 1 to reduced z dimension, 0 (default) otherwise
- m (int, default: 0): minimum integer n that correspond to the minimum wavevector. If 0 is provided, value from config.py will be used (recommended)
- n (int, default: 0): maximum integer n that correspond to the maximum wavevector If 0 is provided, value from config.py will be used (recommended)
