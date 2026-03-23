export LandauDamping

@with_kw struct LandauDamping <: InitialCondition

    α::Float64 = 0.001
    k::Float64 = 0.5
    vth::Float64 = 1.0
    v0::Float64 = 0.0

end

function f0(test_case::LandauDamping, x::Float64, v::Float64)::Float64

    α = test_case.α
    k = test_case.k
    vth = test_case.vth
    v0 = test_case.v0
    return (1.0 / sqrt(2π * vth)) * exp(-0.5 * (v - v0) * (v - v0) / vth) * (1 + α * cos(k * x))

end
