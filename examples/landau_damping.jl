# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using MultiPhaseVlasov
using FastInterpolations
using Plots
using .Threads
using DispersionRelations
using TimerOutputs

const to = TimerOutput()

function main(; tfinal = 10)

    test_case = LandauDamping()
    solver = SemiLagrangian()

    k = 0.5
    α = 0.01
    nx, xmin, xmax = 128, 0.0, 2π / k
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)

    rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)

    phi = zeros(nx + 1)
    u_before = zeros(nx + 1, nv)
    u_remapped = zeros(nx + 1, nv)
    poisson!(phi, mesh_x, rho_tot)
    e = -1.0 .* compute_dx(phi, mesh_x)

    #Set the CFL number and the final time
    dt = 0.1 #0.5*mesh_x.dx
    time = [0.0]
    remap_time = 0.0

    #Array of physical quantities
    elec_energy = [compute_elec_energy(phi, mesh_x)]
    mass = [compute_total_mass(rho_tot, mesh_x)]
    momentum = [compute_momentum(rho, u, mesh_x, grid_v)]
    total_energy = [compute_elec_energy(phi, mesh_x) + compute_kinetic_energy(rho, u, mesh_x, grid_v)]

    n = 0

    e_pred = zeros(nx + 1)
    e_new = zeros(nx + 1)
    phi_pred = zeros(nx + 1)
    rho_pred = zeros(nx + 1)
    xq = zeros(nx + 1)
    rho_at_step_n = zeros(nx + 1)
    u_at_step_n = zeros( nx + 1)
    du_dx = zeros( nx + 1)
    du_dx_plus = zeros( nx + 1)

    while n * dt <= tfinal

        norm_dx_u = compute_norm_dx_u(mesh_x, grid_v, u)
        threshold = 0.5 / (n * dt - remap_time)

        if norm_dx_u > threshold

            @info "Remapping f at time = $(n * dt),  threshold = $threshold, dxu = $norm_dx_u"

            remap_time = n * dt
            u_before .= u
            remap_f!(mesh_x, grid_v, rho, u)
            u_remapped .= u

        end

        nx = mesh_x.nx
        dx = mesh_x.dx
        fill!(rho_tot, 0.0)
        xi = 1.0:(nx+1)
        bc = PeriodicBC(endpoint=:exclusive)

        fill!(rho_tot, 0.0)

        @timeit to "predictor" for j in 1:nv
            for i in eachindex(e)
                d = - (dt * u[j][i] - 0.5 * dt * dt * e[i]) / dx
                xq[i] = mod1(i + d, nx+1)
            end
            du_dx .= compute_dx(u[j], mesh_x)
            du_dx .= cubic_interp(xi, du_dx, xq, bc = bc)
            rho_pred .= cubic_interp(xi, rho[j], xq, bc = bc)
            rho_pred .*= exp.(-dt .* du_dx)
            rho_tot .+= rho_pred .* grid_v.w[j]
        end

        @timeit to "poisson" begin
            poisson!(phi_pred, mesh_x, rho_tot)
            e_pred .= -1.0 .* compute_dx(phi_pred, mesh_x)
        end

        @timeit to "corrector" for j in 1:nv

            u_at_step_n .= u[j]
            rho_at_step_n .= rho[j]

            for i in eachindex(e)
                b = -0.5 * dt * dt * e[i]
                d = - (dt * u[j][i] + b) / dx
                xq[i] = mod1(i + d, nx+1)
            end
            u[j] .= cubic_interp(xi, u_at_step_n, xq, bc = bc)
            e_new .= cubic_interp(xi, e, xq, bc = bc)
            e_new .+= e_pred
            u[j] .+= 0.5dt .* e_new

            du_dx = compute_dx(u_at_step_n, mesh_x)
            du_dx_plus = compute_dx(u[j], mesh_x)

            rho[j] .= cubic_interp(xi, rho_at_step_n, xq, bc = bc)
            du_dx .= cubic_interp(xi, du_dx, xq, bc = bc)
            du_dx .+= du_dx_plus
            rho[j] .*= exp.(-0.5 * dt .* du_dx)
        end

        @timeit to "poisson" begin
            compute_rho_total!(rho_tot, grid_v, rho)
            poisson!(phi, mesh_x, rho_tot)
            e .= -1.0 * compute_dx(phi, mesh_x)
        end


        push!(elec_energy, compute_elec_energy(phi, mesh_x))
        push!(mass, compute_total_mass(rho_tot, mesh_x))
        push!(momentum, compute_momentum(rho, u, mesh_x, grid_v))
        push!(total_energy, compute_elec_energy(phi, mesh_x) + compute_kinetic_energy(rho, u, mesh_x, grid_v))
        n += 1
        push!(time, n * dt)
        #println("iteration: $n , time = $(n * dt), elec energy = $(last(elec_energy)), mass = $(last(mass)),   ||dxU|| = $norm_dx_u")

    end

    return time, elec_energy

end

@time time, elec_energy = main( tfinal = 50.0 )

show(to)

plot(time, elec_energy, yscale = :ln)
line, ω = fit_complex_frequency(time, elec_energy)
plot!(time, line)
title!("$(imag(ω))")
