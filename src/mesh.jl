abstract type AbstractMesh
end

export UniformMesh

struct UniformMesh <:AbstractMesh
    nx::Int
    x::Vector{Float64}
    kx::Vector{Float64}
    dx::Float64
    function UniformMesh(xmin::Float64,xmax::Float64,nx::Int)
        x = LinRange(xmin,xmax,nx+2)[1:end-1]
        dx = (xmax-xmin)/(nx+1)
        kx = collect( 2π / (xmax - xmin) * fftfreq(nx+1, nx+1))
    new(nx,x,kx,dx)
    end
    
end