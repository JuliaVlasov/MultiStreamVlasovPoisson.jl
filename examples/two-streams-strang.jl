using DispersionRelations
using FFTW
using Plots
using .Threads
using MultiPhaseVlasov
using FastInterpolations
using TimerOutputs

const to = TimerOutput()

function main(tfinal = 50)

    ϵ = 1.0
    test_case = TwoStreams()
    solver = "SL-Strang"
    T = 1.0                #Temperature of the Maxwellian
    k = 0.2               #Wave number
    L = 2π / k            #Size of the domain
    nx, xmin, xmax = 256, 0.0, L
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)
    rho, u = compute_initial_condition(mesh_x, grid_v, test_case)
    phi = zeros(nx)
    rho_tot = vec(sum(rho .* grid_v.w', dims = 2))
    poisson!(phi, mesh_x, rho_tot, ϵ)
    e = -1.0*compute_dx(phi, mesh_x)
    rho_pred = zeros(nx, nv)
    u_pred = zeros(nx, nv)
    phi_pred = zeros(nx)

    dt = 0.1 
    time = [0.0]
    remap_time = 0.0

    elec_energy = [compute_elec_energy(phi, mesh_x)]

    n = 0
    sum_norm_dx_u = 0.0
    remap_time = 0.0

    dx_u = zeros(nx, nv)
    u_hat = zeros(ComplexF64, nx, nv)

    bc = PeriodicBC(endpoint = :exclusive)
    xq = [zeros(nx) for j in 1:nv]
    xi = 1.0:nx
    dx = mesh_x.dx

    nstep = floor(Int, tfinal / dt) + 1

    anim = @gif for n in 1:nstep

        dt = min(dt, tfinal - n*dt)

        iter = 0
        err=1e-10
        maxiter=50
        sum_norm_dx_u += dt*compute_norm_dx_u(mesh_x, grid_v, u)

        threshold = 0.2 #Numerical remapping threshold : you may change it or not depending on the test case

        @timeit to "remap" if sum_norm_dx_u > threshold
            @info "Remapping f at time = $(n*dt), solver = $solver"
            remap_time = n * dt
            remap_f!(rho, u, mesh_x, grid_v)
            sum_norm_dx_u = 0.0
        end

        u_pred .= u .+ 0.5dt .* e

        @timeit to "fft" compute_dx!(dx_u, mesh_x, u, u_hat)

        @timeit to "feet" begin
            compute_xfeet!(xq, xi, u_pred, mesh_x, dt, bc)
        end

        @timeit to "advection" @threads for j = 1:nv
            jac = 1 .+ dt * cubic_interp(xi, view(dx_u, :, j), xq[j], bc = bc)
            rho[:, j] .= cubic_interp(xi, view(rho, :, j), xq[j], bc = bc) ./ jac
            u[:, j] .= cubic_interp(xi, view(u_pred, :, j), xq[j], bc = bc)
        end

        u .+= 0.5dt .* e

        @timeit to "poisson" begin
            rho_tot .= vec(sum(rho .* grid_v.w', dims = 2))
            poisson!(phi, mesh_x, rho_tot, ϵ)
            e .= -1.0*compute_dx(phi, mesh_x)
        end

        push!(elec_energy, compute_elec_energy(phi, mesh_x))
        n += 1
        push!(time, n * dt)


        plot(
            mesh_x.x,
            grid_v.v,
            interpolate_f_on_grid(mesh_x, grid_v, rho, u)',
            st = [:surface],
            camera = (0, 90),
            xlabel = "x",
            ylabel = "v",
        )
        title!("$n")

    end every 100

    return time, elec_energy, anim

end

@time time, elec_energy, anim = main()

show(to)

plot(time, log.(elec_energy))
