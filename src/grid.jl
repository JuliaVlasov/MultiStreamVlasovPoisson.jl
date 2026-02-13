using Sobol
using FastGaussQuadrature

abstract type AbstractGrid end

export  GaussHermiteGrid
##Generate the grid in velocity to approach the measure dmu_0 by an empirical measure in three different ways :
# 1) Gauss-Hermite : Points in velocity are using Gauss-Hermite quadrature
struct GaussHermiteGrid <: AbstractGrid
    nv::Int
    v::Vector{Float64}
    w::Vector{Float64}
    T::Float64

    function GaussHermiteGrid(nv,T)
        v , w = gausshermite(nv) 
        for i in 1:nv
            v[i] = sqrt(2*T) * v[i]
            w[i] = w[i] * mean_f0(v[i],T) * exp(v[i]*v[i]/(2*T)) * sqrt(2*T)
        end
        new(nv,v,w,T)
    end

end

export UniformGrid
#2) Uniform grid in velocity with n points starting v_l = vmin + (l-1) (vmax-vmin)/(n-1)
struct UniformGrid <: AbstractGrid
    nv::Int
    v::Vector{Float64}
    dv::Float64
    w::Vector{Float64}
    T::Float64

    function UniformGrid(vmin, vmax, nv,T)
        
        sf0 = 0.0
        dv = Float64
        dv = (vmax-vmin)/(nv-1)
        v = LinRange(vmin,vmax,nv)
        w = zeros(nv)
        for i in 1:nv
            sf0+= mean_f0(v[i],T) * dv 
        end
        for i in 1:nv
            w[i] = mean_f0(v[i],T) * dv / sf0
        end
        new(nv, v, dv, w, T)
    end

end

#Function to construct the Monte Carlo grid based on the inverse of the cumulative distribution function
function newton(r)
        kx, alpha = 0.5, 0.001
        x0, x1 = 0.0, 1.0
        r *= 2π / kx
        while (abs(x1 - x0) > 1e-12)
            p = x0 + alpha * sin(kx * x0) / kx
            f = 1 + alpha * cos(kx * x0)
            x0, x1 = x1, x0 - (p - r) / f
        end
        return x1
 end




function landau( nbpart :: Int64)

   xp = Float64[]
   vp = Float64[]

   s = SobolSeq(2)

   for k=0:nbpart-1

      a = sqrt(-2 * log( (k+0.5)/nbpart))
      r1, r2 = next!(s)
      θ = r1 * 2π
      push!(xp,  newton(r2))
      push!(vp,  a * sin(θ))

   end
    return vp
end

export MonteCarloGrid
struct MonteCarloGrid <:AbstractGrid
    nv::Int
    v::Vector{Float64}
    w::Vector{Float64}
    function MonteCarloGrid(nv)
        v = landau(nv)
        w = zeros(nv)
        for i in 1:nv
            w[i] = 1.0/nv
        end
        new(nv,v,w)
    end
end

