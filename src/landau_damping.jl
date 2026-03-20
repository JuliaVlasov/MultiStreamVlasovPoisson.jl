@with_kw struct LandauDamping <: InitialCondition

    α :: Float64 = 0.001
    k :: Float64 = 0.5

end

function f0(test_case::LandauDamping, x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64)::Float64

   return (1.0 / sqrt(2π*T)) * exp(-0.5 * (v-u0) * (v-u0)/T) * (1 + a * cos(k * x))

end
