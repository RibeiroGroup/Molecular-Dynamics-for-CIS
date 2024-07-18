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
Initial sampling space: Ar-Xe distance = 3 r.u., angle phi and theta = +/- np.pi/4
- T = 200K, 512 of total atoms per run for 20 runs @ 09:56, July 13rd, 2024, mu0 multiplier: 1e3
- T = 292K, 256 of total atoms per run for 10 runs @ 13:19, July 11st, 2024. mu0 multiplier: 1e3
- T = 292K, 512 of total atoms per run for 20 runs @ 09:39, July 12nd, 2024. mu0 multiplier: 1e3
- T = 1000K, 256 of total atoms per run for 10 runs @ 09:28, July 11st, 2024. mu0 multiplier: 1e3  
- T = 15000K, 512 of total atoms per run for 10 runs @ 14:21, July 10th, 2024. mu0 multiplier: 1e3  

Initial sampling space: Ar-Xe distance = 4 r.u., angle phi and theta = +/- np.pi/4
- T = 200K, 512 of total atoms per run for 20 runs @ 09:02, July 15th, 2024, mu0 multiplier: 1e3
- T = 292K, 256 of total atoms per run for 20 runs @ 09:17, July 18th, 2024, mu0 multiplier: 1, coupling strength: 1 in free space and 1e3 for cavity
- + cavity mode: 40-80 * 2 pi / L
- T = 292K, 256 of total atoms per run for 20 runs @ TBA
  + cavity mode: 20-40 * 2 pi / L
- T = 292K, 256 of total atoms per run for 20 runs @ TBA
  + cavity mode: 40-60 * 2 pi / L
- T = 292K, 256 of total atoms per run for 20 runs @ TBA
  + cavity mode: 60-80 * 2 pi / L
- T = 300K, 512 of total atoms per run for 20 runs @ 15:39, July 16th, 2024, mu0 multiplier: 1e3
- T = 325K, 512 of total atoms per run for 20 runs @ 10:33, July 16th, 2024. mu0 multiplier: 1e3
- T = 350K, 528 of total atoms per run for 20 runs @ 15:36, July 15th, 2024. mu0 multiplier: 1e3
- T = 375K, 512 of total atoms per run for 50 runs @ 09:22, July 17th, 2024. mu0 multiplier: 1e3

Experiments to probe the effect of cavity modes: at 292K,  
initial sampling space: Ar-Xe distance = 4 r.u., angle phi and theta = +/- np.pi/4

Note: the figure can be found at folder
```
src/figure/result_[date]_[time]_[note]
```
The Pickle files are stored locally due to its size.
