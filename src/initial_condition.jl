export mean_f0
export f0
"""
$(SIGNATURES)
Mean of the initial condition in x // The perturbation in x must be of zero mean
"""
function mean_f0(v::Float64, T::Float64)::Float64
    return (1.0 / sqrt(2π*T)) * exp(-0.5 * v * v/T)
end

"""
$(SIGNATURES)
Initial condition for the Vlasov-equation (Penrose-stable equilibra with a perturbation).
"""
function f0(x::Float64, v::Float64, k::Float64, T::Float64)::Float64
    a = 0.001
    return (1.0 / sqrt(2π*T)) *  exp(-0.5 * v * v/T) * (1 + a * cos(k * x))
end



