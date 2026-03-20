@with_kw struct MonoKinetic <: InitialCondition

    α :: Float64 = 0.001
    k :: Float64 = 0.5

end

u_initial(test_case::MonoKinetic, x, k) = test_case.a * cos(k * x )

function f0(test_case::MonoKinetic, x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64)::Float64
    return (1.0 / sqrt(2π * T)) * exp(-0.5 * (v - u0) * (v - u0) / T)
end
