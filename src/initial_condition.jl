export mean_f0
export f0
export u_ini
"""
# Here is defined the initial condition:
1) Maxwellian case is when (u_0 = 0)
2) Two stream case is when (u_0 != 0)
$(SIGNATURES)
Mean of the initial condition in x // The perturbation in x must be of zero mean
"""
#function mean_f0(v::Float64, T::Float64,u0::Float64)::Float64
#    return (1.0 / sqrt(2π*T))  *  ( 0.5 * exp(-0.5 * (v-u0) * (v-u0)/T) + 0.5* exp(-0.5 * (v+u0) * (v+u0)/T) ) 
#end

function u_ini(x::Float64,k::Float64,test_case::String)::Float64
    u = 0.0
    a = 0.1
    if(test_case== "landau_damping")
        u = 0
    elseif(test_case == "two_streams")
        u = 2.4
    elseif(test_case =="mono_kinetic")
        u = a*cos(k*x)
    else
        u = 0.0
    end
    return u
end

#This is used when the initial condition is not variable separated
function mean_f0(v::Float64, T::Float64,mesh_x::AbstractMesh,test_case::String)::Float64
    mf0 = 0.0
    nx, dx, L  = mesh_x.nx, mesh_x.dx, mesh_x.L
    k = 2*pi/L
    for i in 1:(nx+1)
        x = mesh_x.x[i]
        u0 = u_ini(x,k,test_case)
        mf0+= (f0(x,v,k,T,u0,test_case) * dx)/L
    end
    return mf0
end

"""
$(SIGNATURES)
Initial condition for the Vlasov-equation 
    1) Maxwellian with perturbation : f_0(x,v) = M_T(v) * (1+a * cos(kx))
    2) Two stream with perturbation : f_0(x,v) = 0.5 *( M_T(v-u0) + M_T(v+u0)) * (1+ a * cos(kx)), u_0 = 2.4 k = 0.2
    3) Monokinetic  : f_0(x,v) = M_T(v-u0(x))
"""
function f0(x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64,test_case::String)::Float64
    a = 0.01
    f0 = 0.0
    if(test_case == "mono_kinetic")
        f0 = (1.0 / sqrt(2π*T)) * exp(-0.5 * (v-u0) * (v-u0)/T)
    else
        f0 = (1.0 / sqrt(2π*T)) *   ( 0.5 * exp(-0.5 * (v-u0) * (v-u0)/T) + 0.5* exp(-0.5 * (v+u0) * (v+u0)/T) ) * (1 + a * cos(k * x))
    end
    return f0
end



