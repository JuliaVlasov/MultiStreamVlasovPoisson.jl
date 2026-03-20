using Parameters

export mean_f0
export f0
export u_ini

abstract type InitialCondition end

include("landau_damping.jl")
include("two_streams.jl")
include("bump_on_tail.jl")
include("mono_kinetic.jl")

function mean_f0(v::Float64, T::Float64, mesh_x::AbstractMesh, test_case::InitialCondition)::Float64
    mf0 = 0.0
    nx, dx, L = mesh_x.nx, mesh_x.dx, mesh_x.L
    k = 2π / L
    for i in 1:(nx + 1)
        x = mesh_x.x[i]
        u0 = u_ini(test_case, x, k)
        mf0 += (f0(x, v, k, T, u0, test_case) * dx) / L
    end
    return mf0
end

