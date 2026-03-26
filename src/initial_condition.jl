using Parameters

abstract type InitialCondition end

include("landau_damping.jl")
include("two_streams.jl")
include("bump_on_tail.jl")
include("mono_kinetic.jl")

function mean_f0(test_case::InitialCondition, mesh_x::AbstractMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, L = mesh_x.nx, mesh_x.dx, mesh_x.L
    for i in 1:(nx + 1)
        x = mesh_x.x[i]
        mf0 += f0(test_case, x, v) * dx / L
    end
    return mf0
end
