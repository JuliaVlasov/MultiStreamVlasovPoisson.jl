using DispersionRelations
using LinearAlgebra
using MultiPhaseVlasov
using Plots
using .Threads
using CSV
using JLD2
using DataFrames

" Vlasov-Poisson solver "

const k = 0.3                #Wave number
const test_case::String      #test_case =  {landau_damping,two_streams,mono_kinetic}
const T = 1.0                #Temperature
const L = 20π #2π / k            #Size of the domain
const eps = 1.0              #Debye length
const solver::String         #SOLVER = {FV,SL} First Order Implicit AP Finite Volume Schem or Cubic Implicit Semi-Lagragian scheme

function main(hermite_quad)
    test_case = "two_streams"
    solver = "SL"
    nx, xmin, xmax = 96, 0.0, L
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -9.0, 9.0
    grid_v = UniformGrid(vmin, vmax, nv, T, mesh_x, test_case)
    rho, u, rho_tot = compute_initial_condition(mesh_x, grid_v, k, T, test_case)
    phi = zeros(nx + 1)
    u_before = zeros(nx + 1, nv)
    u_remapped = zeros(nx + 1, nv)
    poisson!(phi, mesh_x, rho_tot, eps)
    E = -1.0 * compute_dx!(phi, mesh_x)
    if (solver == "SL")
        #For the semi-lagragian scheme
        x_feet_mesh = zeros(nx + 1, nv)
        rho_pred = zeros(nx + 1, nv)
        phi_pred = zeros(nx + 1)
    end

    #Initialize the streams
    u_at_step_n = zeros(nx + 1, nv)
    rho_at_step_n = zeros(nx + 1, nv)

    #Set the CFL number and the final time
    dt = 0.1 #0.5*mesh_x.dx
    tfinal = 400
    time = [0.0]
    remap_time = 0.0

    #Array of physical quantities
    elec_energy = [compute_elec_energy(phi, mesh_x, eps)]
    display(elec_energy)
    mass = [compute_total_mass(rho_tot, mesh_x)]
    momentum = [compute_momentum(rho, u, mesh_x, grid_v)]
    total_energy = [compute_elec_energy(phi, mesh_x, eps) + compute_kinetic_energy(rho, u, mesh_x, grid_v)]

    #Temporal loop
    n = 0
    anim = Animation()

    while n * dt <= tfinal
        iter = 0
        err = 1.0e-10
        maxiter = 50
        norm_dx_u = 0.0
        norm_dx_u = compute_norm_dx_u(mesh_x, grid_v, u)
        threshold = 0.5 / (n * dt - remap_time)
        if (norm_dx_u > threshold)
            println("Remapping f at time = $(n * dt),  threshold = $threshold, dxu = $norm_dx_u")
            remap_time = n * dt
            u_before = u
            rho, u = remap_f_on_uniform_grid(mesh_x, grid_v, rho, u)
            u_remapped = u
        end

        if (solver == "FV")
            copyto!(rho_at_step_n, rho)
            copyto!(u_at_step_n, u)
            old_u = zeros(nx + 1, nv)
            #Fixed point loop to solve the non linear MultiStream pressureless Euler-Poisson system
            while norm(u .- old_u, Inf) / norm(u, Inf) > err && iter < maxiter
                #Update rho : the streams are packed into groups that are solved on each threads
                @threads for j in 1:nv
                    update_rho!(mesh_x, view(rho, :, j), view(u, :, j), view(rho_at_step_n, :, j), dt)
                end
                #Assemble rho
                compute_rho_total!(rho_tot, grid_v, rho)
                #Solve Poisson
                poisson!(phi, mesh_x, rho_tot, eps)
                copyto!(old_u, u)
                #Update u : the streams are packed into groups that are solved on each threads
                @threads for j in 1:nv
                    update_u!(
                        mesh_x, view(rho, :, j), view(u, :, j), phi,
                        view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt, maxiter
                    )
                end
                iter += 1
            end
        elseif (solver == "SL")
            copyto!(rho_at_step_n, rho)
            copyto!(u_at_step_n, u)
            @threads for j in 1:nv
                compute_x_feet_mesh!(dt, mesh_x, view(x_feet_mesh, :, j), view(u_at_step_n, :, j), E)
                update_rho_predictor_SL!(
                    mesh_x, view(rho_pred, :, j), view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt,
                    view(x_feet_mesh, :, j)
                )
            end
            #Assemble rho
            compute_rho_total!(rho_tot, grid_v, rho_pred)
            #Solve Poisson
            poisson!(phi_pred, mesh_x, rho_tot, eps)
            #display(phi_pred)
            E_pred = -1.0 .* compute_dx!(phi_pred, mesh_x)
            #display(E_plus)
            @threads for j in 1:nv
                update_u_SL!(mesh_x, E, E_pred, view(u, :, j), view(u_at_step_n, :, j), dt, view(x_feet_mesh, :, j))
                update_rho_corrector_SL!(mesh_x, view(rho, :, j), view(rho_at_step_n, :, j), view(u_at_step_n, :, j), view(u, :, j), dt, view(x_feet_mesh, :, j))
            end
            #Assemble rho
            compute_rho_total!(rho_tot, grid_v, rho)
            #Solve Poisson
            poisson!(phi, mesh_x, rho_tot, eps)
            E = -1.0 * compute_dx!(phi, mesh_x)
        end
        compute_rho_total!(rho_tot, grid_v, rho)

        push!(elec_energy, compute_elec_energy(phi, mesh_x, eps))
        push!(mass, compute_total_mass(rho_tot, mesh_x))
        push!(momentum, compute_momentum(rho, u, mesh_x, grid_v))
        push!(total_energy, compute_elec_energy(phi, mesh_x, eps) + compute_kinetic_energy(rho, u, mesh_x, grid_v))
        n += 1
        push!(time, n * dt)
        println("iteration: $n , time = $(n * dt), elec energy = $(last(elec_energy)), mass = $(last(mass)),   ||dxU|| = $norm_dx_u")
        #Movie of the the solution
        per = 1000 #Plot every 100 iterations
        if (mod(n, per) == 1)
            f_on_grid = interpolate_f_on_grid(mesh_x, grid_v, rho, u)
            X = []
            Y = []
            Z = []
            #ZZ = []
            for i in 1:(nx + 1)
                for j in 1:nv
                    X = push!(X, mesh_x.x[i])
                    Y = push!(Y, grid_v.v[j])
                    Z = push!(Z, f_on_grid[i, j])
                    #ZZ = push!(ZZ,f_on_grid[i,j]-exp(-0.5*grid_v.v[j]*grid_v.v[j]/T)/sqrt(2*π*T))
                end
            end
            #           p = plot(X,Y,Z,st = [:surface],camera = (0,90),xlabel = "x", ylabel="v",)
            #           plot!(p; ylims=(-6.,6.))
            #           frame(anim)
        end

    end
    #    gif(anim, "fig/mono_kinetic_SL+REMP.gif", fps = 15)


    #Plot the final distribution function
    f_on_grid = interpolate_f_on_grid(mesh_x, grid_v, rho, u)
    X = []
    Y = []
    Z = []
    ZZ = []
    for i in 1:(nx + 1)
        for j in 1:nv
            X = push!(X, mesh_x.x[i])
            Y = push!(Y, grid_v.v[j])
            Z = push!(Z, f_on_grid[i, j])
            ZZ = push!(ZZ, f_on_grid[i, j] - exp(-0.5 * grid_v.v[j] * grid_v.v[j] / T) / sqrt(2 * π * T))
        end
    end
    plot_f = plot(X, Y, Z, st = [:surface], camera = (0, 90), xlabel = "x", ylabel = "v")
    plot_df = plot(X, Y, ZZ, st = [:surface], camera = (0, 90), xlabel = "x", ylabel = "v")

    return time, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, u_before, u_remapped, rho_tot, plot_f, plot_df, phi, E, X, Y, Z, ZZ
    #    return dt, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, rho_tot, plot_f, plot_df, phi, E

end
@time t, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, u_before, u_remapped, rho_tot, plot_f, plot_df, phi, E, X, Y, Z, ZZ = main(true)
#@time dt, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, rho_tot, plot_f, plot_df,phi,E = main(true)
#aa=size(elec_energy)
#tt=dt*[(i) for i=1:aa[1]]


CSV.write("frame_elec.csv", DataFrame(time = t, elec = elec_energy))
CSV.write("frame_f.csv", DataFrame(x = X, v = Y, f = Z, df = ZZ))

plot(t, log.(elec_energy))

#plot(mesh_x.x,u[:,grid_v.nv/2-10:grid_v.nv/2+10], legend = false)

#p_1 = plot(time, elec_energy, yaxis = :log, label = "UniformGrid")
#p_2 = plot(X,Y,Z,st = [:surface], label="f")
#plot!(p_2,camera = (0,90))
# @time t, elec_energy = main(true)
#plot!(t, elec_energy, yaxis = :log, label = "hermite")

# line, ω, = fit_complex_frequency(t, elec_energy, use_peaks = 1:2)
# plot!(time, line; yaxis = :log)
# title!("ω = $(imag(ω))")


#for k=0.4, we have from Eric's book (a=0.001)
#E_k(t)=0.002.*0.424666.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.−0.3357725)))
#plot(t,log.(elec_energy))
#plot!(t,log.(0.002.*0.42466.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.-0.33577))))
#plot!(t,log.(0.0075.*0.42466.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.-0.33577))))   for a=0.001

#for k=0.5, we have from Eric's book
#E_k(t)=4.*a.*0.3677.*exp.(-0.1533.*t).*sqrt.(2pi).*abs.(cos(1.4156.*t-0.536245))
#plot(t,log.(elec_energy))
#plot!(t,log.(0.01.*exp.(-0.1533.*t).*sqrt(2*pi).*abs.(cos.(1.4156.*t.-0.536245))))           for a=0.01
#plot!(t,log.(0.0025.*0.3677.*exp.(-0.1533.*t).*sqrt(2*pi).*abs.(cos.(1.4156.*t.-0.536245))))  for a=0.001
