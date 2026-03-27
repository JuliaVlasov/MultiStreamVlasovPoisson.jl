export TwoStreams

"""
$(TYPEDEF)

## The Two-Stream Distribution Function

The standard choice is a **double-shifted Maxwellian**:

```math
f_0(x, v) = \\frac{n_0}{2\\sqrt{2\\pi}v_{th}} \\left[ \\exp\\!\\left(-\frac{(v - v_0)^2}{2v_{th}^2}\right) 
+ \\exp\\!\\left(-\\frac{(v + v_0)^2}{2v_{th}^2}\\right) \\right] \\cdot \\left(1 + \\epsilon \\cos(kx)\\right)
```

where:
- ``\\pm v_0`` are the **drift velocities** of the two beams (equal and opposite)
- ``v_{th}`` is the **thermal velocity** of each beam
- ``\\epsilon \\ll 1`` is the **perturbation amplitude** (typically ``\\epsilon \\sim 0.01–0.05``)
- ``k = 2\\pi/L`` is the **wavenumber** of the perturbation
- ``L`` is the domain length
"""
@with_kw struct TwoStreams 

    n0::Float64 = 1.0
    α::Float64 = 0.01
    k::Float64 = 0.2
    v0::Float64 = 2.4
    vth::Float64 = 1.0

end

function f0(x::Float64, v::Float64, a, k::Float64, T::Float64, u0::Float64)::Float64
    f0 =
        (1.0 / sqrt(2π*T)) *
        (0.5 * exp(-0.5 * (v-u0) * (v-u0)/T) + 0.5 * exp(-0.5 * (v+u0) * (v+u0)/T)) *
        (1 + a * cos(k * x))
    return f0
end

function mean_f0(test_case::TwoStreams, mesh_x::AbstractMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, xmin, xmax = mesh_x.nx, mesh_x.dx, mesh_x.xmin, mesh_x.xmax
    k = 2pi/(xmax - xmin)
    T = test_case.vth
    u0 = test_case.v0
    α = test_case.α
    for i = 1:nx
        x = mesh_x.x[i]
        mf0 += (f0(x, v, α, k, T, u0) * dx)/(xmax-xmin)
    end
    return mf0
end

function compute_initial_condition(
    mesh_x::AbstractMesh,
    grid_v::AbstractGrid,
    test_case::TwoStreams,
)

    k = test_case.k
    T = test_case.vth
    nx, nv = mesh_x.nx, grid_v.nv
    rho = zeros(nx, nv)
    u = zeros(nx, nv)
    α = test_case.α
    for j = 1:nv
        alpha = grid_v.v[j]
        for i = 1:nx
            x_i = mesh_x.x[i]
            rho[i, j] =
                f0(x_i, alpha, α, k, T, test_case.v0) /
                mean_f0(test_case, mesh_x, alpha)
            u[i, j] = alpha
        end
    end
    return rho, u
end


