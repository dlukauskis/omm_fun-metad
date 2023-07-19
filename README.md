# About

Example OpenMM script for running funnel metadynamics on a host-guest system in vacuum. \DeltaF typically converges in 20 ns, which on my laptop GPU completes in 30 mim, at 980 ns/day. If you do not have CUDA installed, try using 'OpenCL' or 'CPU' as platforms for OpenMM simulations.

Included ```analysis.ipynb``` for how to visualise teh results and analyse the simulation.
