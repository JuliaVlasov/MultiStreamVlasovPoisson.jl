# Code VP Multistream

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Sources

```
git clone git@gitlab.inria.fr:mingus/code-vp-multistream.git
cd code-vp-multistream
```

## Compilation C++

```
g++ -O3 -I. *.cpp -o main
```

## Run C++

```
./main
```

## Julia code

```
julia
julia> import Pkg
julia> Pkg.update()
julia> Pkg.add("Plots")
julia> Pkg.activate(pwd())
julia> Pkg.instantiate()
julia> include("main.jl")
```

