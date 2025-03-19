# Certifiably Optimal Anisotropic Rotation Averaging

Julia code for the paper [Certifiably Optimal Anisotropic Rotation Averaging](https://ylochman.github.io/anisotropic-ra) that (1) incorporates uncertainties of optimized two-view relative rotations into anisotropic costs in a certifiably optimal rotation averaging, and (2) proposes a *conv(SO(3))* relaxation which is stronger than the commonly used *O(3)*, demonstrating its superior performance.

## Requirements
The code is tested on Julia 1.11.3. The required packages can be installed using [`requirements.jl`](./requirements.jl). These are:
* LinearAlgebra
* Statistics
* JuMP
* SCS

## Running the code
The main code for the solver is in [`anisotropic_ra.jl`](./anisotropic_ra/anisotropic_ra.jl).

An example of running it is in [`demo.jl`](./demos/demo.jl).

## Solver details
The SDP program is implemented using the [conic splitting solver](https://arxiv.org/abs/1312.3039) with the [JuMP](https://jump.dev/JuMP.jl) wrapper in Julia. 

## Citation
If you find this work useful in your research, consider citing:
```bibtex
@article{olsson2025certifiably,
    author    = {Olsson, Carl and Lochman, Yaroslava and Malmport, Johan and Zach, Christopher},
    title     = {Certifiably Optimal Anisotropic Rotation Averaging},
    journal   = {arXiv preprint arXiv:2503.07353},
    year      = {2025},
}
```
