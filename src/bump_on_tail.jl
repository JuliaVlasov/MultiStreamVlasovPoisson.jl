@with_kw struct BumpOnTail <: InitialCondition

    α :: Float64 = 0.001
    k :: Float64 = 0.5

end

function f0(test_case::BumpOnTail, x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64)::Float64

    a = test_case.α
    k = test_case.a
    n0 = test_case.n0
    vth = test_case.vth # thermal velocity
    v0 = test_case.v0 # drift velocity
    
    return n0 / ( 2 * sqrt(2π) * vth) * (9 * exp(-0.5 * v * v) + 2 * exp(-2 * (v - u0) * (v - u0))) * (1 + a * cos(k * x))
end
