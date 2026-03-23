export BumpOnTail

@with_kw struct BumpOnTail <: InitialCondition

    α::Float64 = 0.001
    k::Float64 = 0.5
    vth::Float64 = 1.0
    v0::Float64 = 1.0
    n0::Float64 = 1.0

end

function f0(test_case::BumpOnTail, x::Float64, v::Float64)::Float64

    α = test_case.α
    k = test_case.k
    n0 = test_case.n0
    vth = test_case.vth # thermal velocity
    v0 = test_case.v0 # drift velocity

    return n0 / (2 * sqrt(2π) * vth) * (9 * exp(-0.5 * v * v) + 2 * exp(-2 * (v - v0) * (v - v0))) * (1 + α * cos(k * x))
end
