abstract type AbstractMesh
end

export UniformMesh

struct UniformMesh <: AbstractMesh
    nx::Int
    x::Vector{Float64}
    kx::Vector{Float64}
    dx::Float64
    L::Float64
    function UniformMesh(xmin::Float64, xmax::Float64, nx::Int)
        dx = (xmax - xmin) / (nx + 1)
        x = range(start=xmin, step=dx, length=nx+1)
        L = (xmax - xmin)
        kx = collect(2π / (xmax - xmin) * fftfreq(nx + 1, nx + 1))
        return new(nx, x, kx, dx, L)
    end

end
