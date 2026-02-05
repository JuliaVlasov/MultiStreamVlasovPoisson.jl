using Sobol

"""
$(SIGNATURES)

Mean of the initial condition in x.
"""
function mean_f0(v::Float64)::Float64
    return (1.0 / sqrt(2π)) * exp(-0.5 * v * v)
end

"""
$(SIGNATURES)
    
Initial condition for the Vlasov-equation (Penrose-stable equilibra with a perturbation).
"""
function f0(x::Float64, v::Float64, k)::Float64
    a = 0.001
    return (1.0 / sqrt(2π)) *  exp(-0.5 * v * v) * (1 + a * cos(k * x))
end

export compute_initial_condition

"""
$(SIGNATURES)
    
Compute the initial condition for rho, u, and total density.
"""
function compute_initial_condition(mesh::GaussHermiteMesh)

    nx, ng = mesh.nx, mesh.ng
    rho = zeros(nx + 1, ng)
    u = zeros(nx + 1, ng)
    rho_tot = zeros(nx + 1)

    for j in 1:ng
        alpha = mesh.x[j]
        for i in 1:(nx + 1)
            x_i = (i - 1) * mesh.dx
            rho[i, j] = f0(x_i, alpha) / mean_f0(alpha)
            u[i, j] = alpha
            rho_tot[i] += mesh.w[j] * rho[i, j] * mean_f0(alpha) * exp(alpha * alpha)
        end
    end

    return rho, u, rho_tot

end

function compute_initial_condition(mesh::UniformMesh, k)

    nx, ng = mesh.nx, mesh.ng
    vmin, vmax = mesh.vmin, mesh.vmax
    rho = zeros(nx + 1, ng)
    u = zeros(nx + 1, ng)
    rho_tot = zeros(nx + 1)
    xp,vp,vps = zeros(ng)
    
    xp,vp = landau(ng)
    vps=sort(vp)
    
    sf0 = 0.0
    for j in 1:ng
#        alpha = vmin + (j - 1) * (vmax - vmin) / (ng - 1)
        alpha = vps[j]
        sf0 += mean_f0(alpha)
    end

    @show sf0

    for j in 1:ng
#        alpha = vmin + (j - 1) * (vmax - vmin) / (ng - 1)
	alpha =	vps[j]
        for i in 1:(nx + 1)
            x_i = mesh.x[i]
            rho[i, j] = f0(x_i, alpha, k) / mean_f0(alpha)
            u[i, j] = alpha
#            rho_tot[i] += rho[i, j] * mean_f0(alpha) / sf0
            rho_tot[i] += rho[i, j] / ng 
        end
    end

    @show rho_tot

    mesh.sf0 = sf0

    return rho, u, rho_tot, vps

end


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

export landau


function landau( nbpart :: Int64)

   xp = Float64[]
   vp = Float64[]

   s = SobolSeq(2)

   for k=0:nbpart-1

      v = sqrt(-2 * log( (k+0.5)/nbpart))
      r1, r2 = next!(s)
      θ = r1 * 2π
      push!(xp,  newton(r2))
      push!(vp,  v * sin(θ))

   end

   xp, vp

end
