using Parameters

abstract type InitialCondition end

include("landau_damping.jl")
include("two_streams.jl")
include("bump_on_tail.jl")
include("mono_kinetic.jl")

function mean_f0(test_case::InitialCondition, mesh::UniformMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, xmin, xmax = mesh.nx, mesh.dx, mesh.xmin, mesh.xmax
    for i in 1:nx
        x = mesh.x[i]
        mf0 += f0(test_case, x, v) * dx / (xmax - xmin)
    end
    return mf0
end

