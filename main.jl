using DispersionRelations
using LinearAlgebra
using MultiStreamVlasovPoisson
using Plots
using .Threads

function main(hermite_quad)

    eps = 1.0
    nx = 200
    k = 0.5
    xmin, xmax = 0.0, 2π / k
    vmin, vmax = -6.0, 6.0
    ng = 200

    if hermite_quad
        mesh = GaussHermiteMesh(nx, ng)
    else
        mesh = UniformMesh(xmin, xmax, nx, vmin, vmax, ng)
    end

    rho, u, rho_tot = compute_initial_condition(mesh, k)

    poisson = NonLinearPoissonSolver(eps, nx)

    phi = -log.(rho_tot)
    display(plot(mesh.x, rho_tot))

    dt = mesh.dx
    tfinal = 100 * dt  # Final time
    time = [0.0]

    @show elec_energy = [compute_elec_energy(phi, mesh, eps)]

    n = 0
    while n * dt <= tfinal

        # Update phi
        solve!(phi, poisson, rho_tot)
        

        @threads for j in 1:ng
            update!(mesh, poisson, view(rho, :, j), view(u, :, j), phi, dt)
        end

        compute_rho_total!(rho_tot, mesh, rho)
        push!(elec_energy, compute_elec_energy(phi, mesh, eps))
        n += 1
        push!(time, n * dt)
        println("iteration: $n , time = $(n * dt), elec energy = $(last(elec_energy))")

    end

    return time, elec_energy

end

@time time, elec_energy = main(false)
plot(time, elec_energy, yaxis = :log, label = "uniform")

# @time time, elec_energy = main(true)
# plot!(time, elec_energy, yaxis = :log, label = "hermite")

# line, ω, = fit_complex_frequency(time, elec_energy, use_peaks = 1:2)
# plot!(time, line; yaxis = :log)
# title!("ω = $(imag(ω))")
