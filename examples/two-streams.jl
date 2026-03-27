using MultiPhaseVlasov
using Plots

function main( tfinal = 400 )

    test_case = TwoStreams()

    nx, xmin, xmax = 96, 0.0, 20π
    nv, vmin, vmax = 256, -9.0, 9.0

    mesh_x = UniformMesh(xmin, xmax, nx)
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)

    solver = SemiLagrangian(mesh_x, grid_v)

    rho, u = compute_initial_condition(test_case, mesh_x, grid_v)

    e = zeros(nx)
    phi = zeros(nx)
    compute_electric_field!(e, phi, mesh_x, grid_v, rho)

    dt = 0.1 
    time = [0.0]

    elec_energy = [compute_elec_energy(phi, mesh_x)]
    mass = [compute_total_mass(rho, mesh_x, grid_v)]
    momentum = [compute_momentum(rho, u, mesh_x, grid_v)]
    total_energy = [compute_elec_energy(phi, mesh_x) + compute_kinetic_energy(rho, u, mesh_x, grid_v)]

    e_pred = zeros(nx)
    rho_pred = zeros(nx, nv)

    v = zeros(nx)
    dv = zeros(nx, nv)
    dv_plus = zeros(nx, nv)
    v_hat = zeros(ComplexF64, nx, nv)

    t = 0.0
    @show nstep = floor(Int, tfinal / dt) + 1

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
        push!(mass, compute_total_mass(rho, mesh_x, grid_v))
        push!(momentum, compute_momentum(rho, u, mesh_x, grid_v))
        push!(total_energy, compute_elec_energy(phi, mesh_x) + compute_kinetic_energy(rho, u, mesh_x, grid_v))

        t += dt
        push!(time, t)

        norm_dx_u = compute_norm_dx_u(mesh_x, grid_v, u)

        println("iteration: $istep , time = $(t), elec energy = $(last(elec_energy)), mass = $(last(mass)),   ||dxU|| = $norm_dx_u")

    end

    f_on_grid = interpolate_f_on_grid(mesh_x, grid_v, rho, u)
    plot_f = plot(grid_v.v, mesh_x.x, f_on_grid, st = [:surface], camera = (0, 90), xlabel = "v", ylabel = "x")

    return time, elec_energy, mass, momentum, total_energy, plot_f

end

@time t, elec_energy, mass, momentum, total_energy, plot_f = main()
