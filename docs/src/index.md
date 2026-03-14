# MultiPhaseVlasov.jl

Documentation for MultiPhaseVlasov.jl

## Sources

```
git clone git@gitlab.inria.fr:mingus/code-vp-multistream.git
cd code-vp-multistream
```

## Install and Run

```
julia
julia> import Pkg
julia> Pkg.update() # update your Julia setup
julia> Pkg.add("Plots") # Install the plot library
julia> Pkg.activate(pwd()) # Activate the local environment
julia> Pkg.instantiate() # Install package dependencies
julia> include("main.jl") # Run the example
```

## Example

```@example main
using MultiPhaseVlasov
using Plots

eps = 1.0
nx = 200
k = 0.5
xmin, xmax = 0.0, 2π / k
vmin, vmax = -6.0, 6.0
ng = 200
mesh = UniformMesh(xmin, xmax, nx, vmin, vmax, ng)

rho, u, rho_tot = compute_initial_condition(mesh, k)

poisson = NonLinearPoissonSolver(eps, nx)

phi = -log.(rho_tot)
plot(mesh.x, rho_tot)
```

## Functions

```@autodocs
Modules = [MultiPhaseVlasov]
Order   = [:type, :function]
```
