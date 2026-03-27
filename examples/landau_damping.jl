# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using DispersionRelations
using MultiPhaseVlasov
using Plots

function main(; tfinal = 40)

    k = 0.5
    α = 0.001
    test_case = LandauDamping(α = α, k = k)
    nx, xmin, xmax = 128, 0.0, 2π / k
    nv, vmin, vmax = 256, -6.0, 6.0
    
    mesh_x = UniformMesh(xmin, xmax, nx)
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)

    rho, u = compute_initial_condition(test_case, mesh_x, grid_v)

    e = zeros(nx)
    phi = zeros(nx)
    compute_electric_field!(e, phi, mesh_x, grid_v, rho)

    dt = 0.1 
    time = [0.0]

    elec_energy = [compute_elec_energy(phi, mesh_x)]

    e_pred = zeros(nx)
    rho_pred = zeros(nx, nv)

    v = zeros(nx)
    dv = zeros(nx, nv)
    dv_plus = zeros(nx, nv)
    v_hat = zeros(ComplexF64, nx, nv)

    t = 0.0
    @show nstep = floor(Int, tfinal / dt) + 1

    solver = SemiLagrangian(mesh_x, grid_v)

    for istep in 1:nstep

        dt = min(dt, tfinal-t)

        compute_x_feet_mesh!(solver, u, e, dt)

        compute_dx!(dv, mesh_x, u, v_hat)

        update_rho_predictor!(rho_pred, solver, mesh_x, rho, v, dv, dt)

        compute_electric_field!(e_pred, phi, mesh_x, grid_v, rho_pred)

        update_u!( u, solver, mesh_x, e, e_pred, dt)

        compute_dx!(dv_plus, mesh_x, u, v_hat)

        update_rho_corrector!( rho, solver, mesh_x, dv, dv_plus, dt)

        compute_electric_field!(e, phi, mesh_x, grid_v, rho)

        push!(elec_energy, compute_elec_energy(phi, mesh_x))
        t += dt
        push!(time, t)
        println("iteration: $istep , time = $(t), elec energy = $(last(elec_energy))")

    end  

    return time, elec_energy

end

@time time, elec_energy = main( tfinal = 100.0 )

plot(time, elec_energy, yscale = :ln)
line, ω = fit_complex_frequency(time, elec_energy)
plot!(time, line)
title!("$(imag(ω))")
