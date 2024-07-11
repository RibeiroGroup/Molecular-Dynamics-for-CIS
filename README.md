# Molecular Dynamic and Coulomb-Gauge Classical Electrodynamics simulation

By Tuan Nguyen

## DEPENDENCIES:
python >= 3.10 (lower version is probably okay)  
numpy  
matplotlib  

## USAGE
### Simulation of a charged harmonic oscillator in free/confined electromagnetic field
```
$python src/simulation_point_charge.py
```

### Simulation of Argon - Xenon mixture dynamics in the absence of electromagnetic field
```
$python src/simulation_nofield.py
```

### Simulation of Argon - Xenon mixture dynamics in the presence of multimodes electromagnetic field
```
$python src/simulation_full.py
```

### Monte Carlo simulation of Argon - Xenon collisions in the presence of multimodes electromagnetic field
```
$python src/simulation_monte.py
```

### Experiments so far:
- T = 1000K, 256 of total atoms per run @ 09:28, July 11st, 2024. mu0 multiplier: 1e3  
- T = 15000K, 512 of total atoms per run @ 14:21, July 10th, 2024. mu0 multiplier: 1e3  

Note: the figure can be found at folder
```
src/figure/result_[date]_[time]_[note]
```
The Pickle files are stored locally due to its size.
