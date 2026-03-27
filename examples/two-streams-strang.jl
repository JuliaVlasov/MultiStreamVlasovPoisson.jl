using DispersionRelations
using FFTW
using Plots
using .Threads
import MultiPhaseVlasov: AbstractMesh, UniformMesh, compute_elec_energy
import MultiPhaseVlasov: compute_dx!, poisson!
import MultiPhaseVlasov: compute_norm_dx_u, interpolate_f_on_grid
import MultiPhaseVlasov: TwoStreams, AbstractMesh, AbstractGrid
import MultiPhaseVlasov: Spline
import MultiPhaseVlasov: remap_f!, compute_dx, compute_initial_condition
using FastInterpolations
using TimerOutputs

u_ini(x::Float64, k::Float64, test_case::String)::Float64 = 2.4

function f0(x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64)::Float64
    a = 0.01
    f0 =
        (1.0 / sqrt(2π*T)) *
        (0.5 * exp(-0.5 * (v-u0) * (v-u0)/T) + 0.5 * exp(-0.5 * (v+u0) * (v+u0)/T)) *
        (1 + a * cos(k * x))
    return f0
end

function mean_f0(test_case::TwoStreams, mesh_x::AbstractMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, xmin, xmax = mesh_x.nx, mesh_x.dx, mesh_x.xmin, mesh_x.xmax
    k = 2pi/(xmax - xmin)
    T = test_case.vth
    u0 = test_case.v0
    for i = 1:nx
        x = mesh_x.x[i]
        mf0 += (f0(x, v, k, T, u0) * dx)/(xmax-xmin)
    end
    return mf0
end

struct UniformGrid <: AbstractGrid
    nv::Int
    v::Vector{Float64}
    dv::Float64
    w::Vector{Float64}
    T::Float64

    function UniformGrid(vmin, vmax, nv, mesh_x, test_case)

        T = test_case.vth
        sf0 = 0.0
        dv = Float64
        dv = (vmax-vmin)/(nv)
        v = LinRange(vmin, vmax, nv)
        w = zeros(nv)
        for i = 1:nv
            sf0 += mean_f0(test_case, mesh_x, v[i]) * dv
        end
        for i = 1:nv
            w[i] = mean_f0(test_case, mesh_x, v[i]) * dv / (sf0)
        end
        new(nv, v, dv, w, T)
    end

end


function compute_initial_condition(
    mesh_x::AbstractMesh,
    grid_v::AbstractGrid,
    k::Float64,
    T::Float64,
    test_case::String,
)

    nx, nv = mesh_x.nx, grid_v.nv
    rho = zeros(nx, nv)
    u = zeros(nx, nv)
    rho_tot = zeros(nx)
    for j = 1:nv
        alpha = grid_v.v[j]
        for i = 1:nx
            x_i = mesh_x.x[i]
            rho[i, j] =
                f0(x_i, alpha, k, T, u_ini(x_i, k, test_case)) /
                mean_f0(alpha, T, mesh_x, test_case)
            u[i, j] = alpha
            rho_tot[i] += grid_v.w[j] * rho[i, j]
        end
    end
    return rho, u, rho_tot
end

const to = TimerOutput()

function main(tfinal = 50)

    ϵ = 1.0
    test_case = "two_streams"
    solver = "SL-Strang"
    T = 1.0                #Temperature of the Maxwellian
    k = 0.2               #Wave number
    L = 2π / k            #Size of the domain
    nx, xmin, xmax = 256, 0.0, L
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, T, mesh_x, test_case)
    rho, u, rho_tot = compute_initial_condition(mesh_x, grid_v, k, T, test_case)
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

    while n * dt <= tfinal

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

        @timeit to "feet" @threads for j in 1:nv
            itp = cubic_interp(xi, view(u_pred, :, j), bc = bc)
            for i in 1:nx
                it = 0
                err = 1.0
                x_feet = i
                while err > 1e-5
                    x_old = x_feet
                    x_feet = mod1(i - dt * itp(x_feet) / dx, nx)
                    err = abs(x_old-x_feet)
                    if it > 50
                        @info "err = $err,  it = $it, x_feet = $x_feet, FIXED-POINT DOES NOT CONVERGE"
                        break
                    end
                    it += 1
                end
                xq[j][i] = x_feet
            end
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

    end

    f_on_grid = interpolate_f_on_grid(mesh_x, grid_v, rho, u)
    plot_f = plot(
        mesh_x.x,
        grid_v.v,
        f_on_grid',
        st = [:surface],
        camera = (0, 90),
        xlabel = "x",
        ylabel = "v",
    )
    return time, elec_energy, plot_f

end

@time time, elec_energy, plot_f = main()

show(to)

png(plot_f, "plot_df")
plot(time, log.(elec_energy))
