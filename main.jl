using LinearAlgebra
using MultiStreamVlasovPoisson
using Plots
using .Threads


"""
    mymax(a::Float64, b::Float64)::Float64
    
Maximum of two values.
"""
function mymax(a::Float64, b::Float64)::Float64
    if a >= b
        return a
    else
        return b
    end
end

"""
    g(x::Float64, dx::Float64)::Float64
    
Regularization function of (x)^{+}.
"""
function g(x::Float64, dx::Float64)::Float64
    if x < -dx
        return 0.0
    elseif x >= -dx && x <= dx
        return (x + dx) * (x + dx) / (4dx)
    else
        return x
    end
end

"""
    G(s::Float64, t::Float64, u::Float64)::Float64
    
Function G: (s,t,u) --> s*g(u) - t*(g(u)-u).
"""
function G(s::Float64, t::Float64, u::Float64, dx::Float64)::Float64
    return s * g(u, dx) - t * (g(u, dx) - u)
end

"""
    dg(x::Float64)::Float64
    
Numerical derivative of g using finite differences.
"""
function dg(x::Float64, dx::Float64)::Float64
    h = 1.0e-16
    return (g(x + h, dx) - g(x - h, dx)) / (2h)
end

"""
    t_rho(s::Float64, t::Float64, u::Float64)::Float64
    
Helper function for density calculation.
"""
function t_rho(s::Float64, t::Float64, u::Float64, dx::Float64)::Float64
    if abs(u) < 1.0e-16
        return t - (t - s) * dg(0.0, dx)
    else
        return (G(s, t, u, dx) - G(s, t, 0.0, dx)) / u
    end
end


"""
Update one single fluid solution using fixed-point iteration.
"""
function update_single_fluid_sol!(
        mesh::GaussHermiteMesh, poisson::NonLinearPoissonSolver,
        rho::AbstractVector, u::AbstractVector, rho_tot::Vector, phi::Vector,
        dt::Float64, maxiter::Int = 1
    )

    nx, dx = mesh.nx, mesh.dx
    iter = 0

    rho_at_step_n = copy(rho)
    u_at_step_n = copy(u)
    new_rho = zeros(nx + 1)
    old_u = zeros(nx + 1)
    new_phi = zeros(nx + 1)
    q = zeros(nx + 1)

    m_upd_rho = zeros(nx + 1, nx + 1)
    m_upd_u = zeros(nx + 1, nx + 1)
    m_upd_rho2 = zeros(nx + 1, nx + 1)

    # For Newton part for v
    t = zeros(nx + 1)
    t[1] = 1.0
    jac_t = zeros(nx + 1, nx + 1)
    v = copy(u)
    delta = zeros(nx + 1)
    old_v = copy(u)
    h = 1.0e-10
    it_newt = 0

    while norm(u .- old_u, Inf) / norm(u, Inf) > 1.0e-7 && iter < maxiter
        # Reinitialize matrices
        fill!(m_upd_rho, 0.0)
        fill!(jac_t, 0.0)

        # Update rho with new computed velocity
        for i in 1:(nx + 1)
            ir = mod1(i + 1, nx + 1)
            il = mod1(i - 1, nx + 1)
            m_upd_rho[i, i] = 1 + (dt / dx) * (g(u[i], dx) + g(u[il], dx) - u[il])
            m_upd_rho[i, ir] = -(dt / dx) * (g(u[i], dx) - u[i])
            m_upd_rho[i, il] = -(dt / dx) * g(u[il], dx)
        end

        # Solve for rho
        rho .= m_upd_rho \ rho_at_step_n


        # Update u using Newton algorithm
        v .= u
        it_newt = 0
        fill!(t, 0.0)
        t[1] = 1.0

        while norm(t, Inf) > 1.0e-10 && it_newt < maxiter

            for i in 1:(nx + 1)

                irr = mod1(i + 2, nx + 1)
                ir = mod1(i + 1, nx + 1)
                il = mod1(i - 1, nx + 1)
                ill = mod1(i - 2, nx + 1)

                # Compute flux of momentum
                q_r = 0.5 * (G(rho[ir], rho[irr], u[ir], dx) + G(rho[i], rho[ir], u[i], dx))
                q_l = 0.5 * (G(rho[i], rho[ir], u[i], dx) + G(rho[il], rho[i], u[il], dx))

                # Compute Jacobian
                jac_t[i, i] = 0.5 * (rho[ir] + rho[i]) + (dt / dx) * (mymax(q_r, 0.0) + mymax(q_l, 0.0) - q_l) -
                    dt * ((phi[ir] - phi[i]) / dx) * (t_rho(rho[i], rho[ir], v[i] + h, dx) - t_rho(rho[i], rho[ir], v[i] - h, dx)) / (2h)
                jac_t[i, ir] = -(dt / dx) * (mymax(q_r, 0.0) - q_r)
                jac_t[i, il] = -(dt / dx) * mymax(q_l, 0.0)

                # Compute T(v)
                t[i] = 0.5 * (rho[ir] + rho[i]) * v[i] - 0.5 * (rho_at_step_n[ir] + rho_at_step_n[i]) * u_at_step_n[i] +
                    (dt / dx) * (mymax(q_r, 0.0) + mymax(q_l, 0.0) - q_l) * v[i] -
                    (dt / dx) * (mymax(q_r, 0.0) - q_r) * v[ir] -
                    (dt / dx) * mymax(q_l, 0.0) * v[il] -
                    dt * ((phi[ir] - phi[i]) / dx) * t_rho(rho[i], rho[ir], v[i], dx)
            end

            delta .= jac_t \ t
            v .-= delta

            it_newt += 1
        end

        old_u .= u
        u .= v

        iter += 1
    end
    return
end

function main()

    eps = 1.0
    nx = 100
    vmin, vmax = -4.0, 4.0
    ng = 100

    mesh = GaussHermiteMesh(nx, ng)

    rho, u, rho_tot = compute_initial_condition(mesh)

    poisson = NonLinearPoissonSolver(eps, nx)

    phi = -log.(rho_tot)

    dt = mesh.dx
    tfinal = 800 * dt  # Final time
    time = [0.0]

    elec_energy = [compute_elec_energy(phi, mesh, eps)]

    n = 0
    while n * dt <= tfinal

        # Update phi
        solve!(phi, poisson, rho_tot)
        @threads for j in 1:ng
            update_single_fluid_sol!(mesh, poisson, view(rho, :, j), view(u, :, j), rho_tot, phi, dt)
        end
        compute_rho_total!(rho_tot, mesh, rho)
        push!(elec_energy, compute_elec_energy(phi, mesh, eps))
        n += 1
        push!(time, n * dt)
        println("iteration: $n and time = $(n * dt)")

    end

    return time, elec_energy

end


@time time, elec_energy = main()

plot(time, elec_energy, yscale = :log10)
