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

    sf0 = 0.0
    for j in 1:ng
        alpha = vmin + (j - 1) * (vmax - vmin) / (ng - 1)
        sf0 += mean_f0(alpha)
    end
    for j in 1:ng
        alpha = vmin + (j - 1) * (vmax - vmin) / (ng - 1)
        for i in 1:(nx + 1)
            x_i = mesh.x[i]
            rho[i, j] = f0(x_i, alpha, k) / mean_f0(alpha)
            u[i, j] = alpha
            rho_tot[i] += rho[i, j] * mean_f0(alpha) / sf0
        end
    end

    mesh.sf0 = sf0

    return rho, u, rho_tot

end
"""

"""
    Landau( α, kx)

Test structure to initialize a particles distribtion for
Landau damping test case in 1D1V and 1D2V

"""
struct LandauDamping
    alpha::Float64
    kx::Float64
end

"""
    sample!(d, pg)

Sampling from a probability distribution to initialize a Landau damping in
1D2V space.

```math
f_0(x,v,t) = \\frac{n_0}{2π v_{th}^2} ( 1 + \\alpha cos(k_x x))
 exp( - \\frac{v_x^2+v_y^2}{2 v_{th}^2})
```
The newton function solves the equation ``P(x)-r=0`` with Newton’s method
```math
x^{n+1} = x^n – (P(x)-(2\\pi r / k)/f(x) 
```
with 
```math
P(x) = \\int_0^x (1 + \\alpha cos(k_x y)) dy = x + \\frac{\\alpha}{k_x} sin(k_x x)
```
"""
function sample!(d::LandauDamping, pg::ParticleGroup{1,2})
    alpha, kx = d.alpha, d.kx

    function newton(r)
        x0, x1 = 0.0, 1.0
        r *= 2π / kx
        while (abs(x1 - x0) > 1e-12)
            p = x0 + alpha * sin(kx * x0) / kx
            f = 1 + alpha * cos(kx * x0)
            x0, x1 = x1, x0 - (p - r) / f
        end
        return x1
    end

    s = SobolSeq(2)
    nbpart = pg.n_particles

    for i in 1:nbpart
        v = sqrt(-2 * log((i - 0.5) / nbpart))
        r1, r2 = Sobol.next!(s)
        θ = r1 * 2π
        set_x!(pg, i, newton(r2))
        set_v!(pg, i, [v * cos(θ), v * sin(θ)])
        set_weights!(pg, i, 2 * pi / kx / nbpart)
    end
end

"""