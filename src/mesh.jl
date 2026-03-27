abstract type AbstractMesh end

export UniformMesh

struct UniformMesh <: AbstractMesh

    xmin::Float64
    xmax::Float64
    nx::Int
    x::Vector{Float64}
    kx::Vector{Float64}
    dx::Float64

    function UniformMesh(xmin::Float64, xmax::Float64, nx::Int)
        dx = (xmax - xmin) / nx
        x = LinRange(xmin, xmax, nx+1)[1:end-1]
        kx = collect(2π / (xmax - xmin) * fftfreq(nx, nx))
        return new(xmin, xmax, nx, x, kx, dx)
    end

end
