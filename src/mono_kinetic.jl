export MonoKinetic

@with_kw struct MonoKinetic <: InitialCondition

    α::Float64 = 0.001
    k::Float64 = 0.5
    vth::Float64 = 1.0
    v0::Float64 = 0.0

end

u_initial(test_case::MonoKinetic, x) = test_case.α * cos(test_case.k * x)

function f0(test_case::MonoKinetic, x::Float64, v::Float64)::Float64

    vth = test_case.vth
    v0 = test_case.v0
    return (1.0 / sqrt(2π * vth)) * exp(-0.5 * (v - v0) * (v - v0) / vth)
end
