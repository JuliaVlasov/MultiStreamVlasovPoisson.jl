# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using MultiPhaseVlasov
using Plots
using .Threads
using DispersionRelations
using TimerOutputs

const to = TimerOutput()

function main(; tfinal = 100)

    test_case = LandauDamping()
    solver = SemiLagrangian()

    k = 0.5
    α = 0.01
    nx, xmin, xmax = 96, 0.0, 2π / k
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)

    rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)

    phi = zeros(nx + 1)
    u_before = zeros(nx + 1, nv)
    u_remapped = zeros(nx + 1, nv)
    poisson!(phi, mesh_x, rho_tot)
    E = -1.0 * compute_dx!(phi, mesh_x)
    plot(mesh_x.x, E)

    # +
    x_feet_mesh = zeros(nx + 1, nv)
    rho_pred = zeros(nx + 1, nv)
    phi_pred = zeros(nx + 1)

    #Initialize the streams
    u_at_step_n = zeros(nx + 1, nv)
    rho_at_step_n = zeros(nx + 1, nv)

    #Set the CFL number and the final time
    dt = 0.1 #0.5*mesh_x.dx
    time = [0.0]
    remap_time = 0.0

    #Array of physical quantities
    elec_energy = [compute_elec_energy(phi, mesh_x)]
    display(elec_energy)
    mass = [compute_total_mass(rho_tot, mesh_x)]
    momentum = [compute_momentum(rho, u, mesh_x, grid_v)]
    total_energy = [compute_elec_energy(phi, mesh_x) + compute_kinetic_energy(rho, u, mesh_x, grid_v)]

    n = 0

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

        copyto!(rho_at_step_n, rho)
        copyto!(u_at_step_n, u)
        for j in 1:nv
            @timeit to "feet" compute_x_feet_mesh!(dt, mesh_x, view(x_feet_mesh, :, j), view(u_at_step_n, :, j), E)
            @timeit to "predictor" update_rho_predictor!(
                solver,
                mesh_x, view(rho_pred, :, j), view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt,
                view(x_feet_mesh, :, j)
            )
        end
        #Assemble rho
        @timeit to "compute rho" compute_rho_total!(rho_tot, grid_v, rho_pred)
        #Solve Poisson
        @timeit to "compute phi" poisson!(phi_pred, mesh_x, rho_tot)
        E_pred = -1.0 .* compute_dx!(phi_pred, mesh_x)
        for j in 1:nv
            @timeit to "update u" update_u!(solver, mesh_x, E, E_pred, view(u, :, j), view(u_at_step_n, :, j), dt, view(x_feet_mesh, :, j))
            @timeit to "corrector" update_rho_corrector!(solver, mesh_x, view(rho, :, j), view(rho_at_step_n, :, j), view(u_at_step_n, :, j), view(u, :, j), dt, view(x_feet_mesh, :, j))
        end
        #Assemble rho
        @timeit to "compute rho" compute_rho_total!(rho_tot, grid_v, rho)
        #Solve Poisson
        @timeit to "compute phi" poisson!(phi, mesh_x, rho_tot)
        E = -1.0 * compute_dx!(phi, mesh_x)

        @timeit to "compute rho" compute_rho_total!(rho_tot, grid_v, rho)

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

@time time, elec_energy = main()

show(to)

plot(time, elec_energy, yscale = :ln)
line, ω = fit_complex_frequency(time, elec_energy)
plot!(time, line)
title!("$(imag(ω))")
