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

mutable struct UniformMesh <: AbstractMesh

    xmin::Float64
    xmax::Float64
    nx::Int
    dx::Float64
    ng::Int
    vmin::Float64
    vmax::Float64
    sf0::Float64
    x::Vector{Float64}
    kx::Vector{Float64}

    function UniformMesh(xmin, xmax, nx, vmin, vmax, ng)

        x = LinRange(xmin, xmax, nx+2)[1:end-1]
        dx = (xmax - xmin) / (nx + 1)
        sf0 = 0.0
        kx = collect( 2π / (xmax - xmin) * fftfreq(nx+1, nx+1))
        return new(xmin, xmax, nx, dx, ng, vmin, vmax, sf0, x, kx)

    end

end
