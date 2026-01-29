using FastGaussQuadrature

abstract type AbstractMesh end

export GaussHermiteMesh

struct GaussHermiteMesh <: AbstractMesh

    nx::Int
    dx::Float64
    ng::Int
    x::Vector{Float64}
    w::Vector{Float64}

    function GaussHermiteMesh(nx, ng)

        dx = 1.0 / (nx + 1)
        x, w = gausshermite(ng)

        return new(nx, dx, ng, x, w)

    end

end

export UniformMesh

struct UniformMesh <: AbstractMesh

    eps::Float64
    nx::Int
    dx::Float64
    ng::Int
    vmin::Float64
    vmax::Float64
    x::Vector{Float64}
    v::Vector{Float64}

    function UniformMesh(eps, nx, vmin, vmax, ng)

        dx = 1.0 / (nx + 1)
        v = LinRange(vmin, vmax, nx + 1)

        return new(nx, dx, ng, vmin, vmax, x, v)

    end

end
