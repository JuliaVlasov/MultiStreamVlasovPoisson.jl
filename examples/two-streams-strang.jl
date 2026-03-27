using DispersionRelations
using FFTW
using Plots
using .Threads
import MultiPhaseVlasov: AbstractMesh, UniformMesh, compute_elec_energy
import MultiPhaseVlasov: compute_dx!
using FastInterpolations

abstract type AbstractGrid end

struct UniformGrid <: AbstractGrid
    nv::Int
    v::Vector{Float64}
    dv::Float64
    w::Vector{Float64}
    T::Float64

    function UniformGrid(vmin, vmax, nv, T, mesh_x, test_case)

        sf0 = 0.0
        dv = Float64
        dv = (vmax-vmin)/(nv)
        v = LinRange(vmin, vmax, nv)
        w = zeros(nv)
        for i = 1:nv
            sf0 += mean_f0(v[i], T, mesh_x, test_case) * dv
        end
        for i = 1:nv
            w[i] = mean_f0(v[i], T, mesh_x, test_case) * dv / (sf0)
        end
        new(nv, v, dv, w, T)
    end

end

function compute_norm_dx_u(mesh_x::AbstractMesh, grid_v::AbstractGrid, u::Matrix{Float64})
    nv = grid_v.nv
    nx = mesh_x.nx
    dx = mesh_x.dx
    dx_u = 0.0
    for l = 1:nv
        for i = 1:(nx-1)
            du = abs((u[i+1, l]-u[i, l])/dx)
            dx_u = max(dx_u, du)
        end
    end
    return dx_u
end


function Spline(x::Float64, h::Float64)
    if (h < abs(x) < 2 * h)
        S = (1.0/6.0) * ((2-abs(x/h)) * (2-abs(x/h)) * (2-abs(x/h)))
    elseif (abs(x) < h)
        S = (1.0/6.0) * (4.0 - 6.0*(abs(x/h)*abs(x/h)) + 3.0*(abs(x/h)*abs(x/h)*abs(x/h)))
    else
        S = 0
    end
    return S/h
end

function interpolate_f_on_grid(
    mesh_x::AbstractMesh,
    grid_v::UniformGrid,
    rho::Matrix{Float64},
    u::Matrix{Float64},
)
    nx = mesh_x.nx
    nv = grid_v.nv
    dv = grid_v.dv
    f_grid = zeros(nx, nv)
    for i = 1:nx
        for j = 1:nv
            for l = 1:nv
                v_j = grid_v.v[j]
                f_grid[i, j] += grid_v.w[l] * rho[i, l] * Spline(v_j-u[i, l], dv)
            end
        end
    end
    return f_grid
end

u_ini(x::Float64, k::Float64, test_case::String)::Float64 = 2.4

function f0(x::Float64, v::Float64, k::Float64, T::Float64, u0::Float64)::Float64
    a = 0.01
    f0 =
        (1.0 / sqrt(2π*T)) *
        (0.5 * exp(-0.5 * (v-u0) * (v-u0)/T) + 0.5 * exp(-0.5 * (v+u0) * (v+u0)/T)) *
        (1 + a * cos(k * x))
    return f0
end

function mean_f0(v::Float64, T::Float64, mesh_x::AbstractMesh, test_case::String)::Float64
    mf0 = 0.0
    nx, dx, xmin, xmax = mesh_x.nx, mesh_x.dx, mesh_x.xmin, mesh_x.xmax
    k = 2*pi/(xmax - xmin)
    for i = 1:nx
        x = mesh_x.x[i]
        u0 = u_ini(x, k, test_case)
        mf0 += (f0(x, v, k, T, u0) * dx)/(xmax-xmin)
    end
    return mf0
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

function poisson!(
    phi::Vector{Float64},
    mesh::UniformMesh,
    rho_tot::Vector{Float64},
    ϵ::Float64,
)

    rho_tot_f=fft(rho_tot .- 1)
    rho_tot_f[1]=0
    kkx=mesh.kx
    kkx[1]=1
    ff_P = (ϵ * ϵ)*kkx .* kkx
    phi.=+real(ifft((rho_tot_f ./ (ff_P))))

end

function compute_feet_char(y::Int, dt::Float64, mesh::AbstractMesh, v::AbstractVector)
    it = 0
    err = 1.0
    x_feet = y
    nx, dx = mesh.nx, mesh.dx

    xi = 1.0:nx
    itp = cubic_interp(xi, v, bc = PeriodicBC(endpoint = :exclusive))

    while err > 1e-5
        x_old = x_feet
        x_feet = mod1(y - dt * itp(x_feet) / dx, nx)
        err = abs(x_old-x_feet)
        if it > 50
            @info "err = $err,  it = $it, x_feet = $x_feet, FIXED-POINT DOES NOT CONVERGE"
            break
        end
        it += 1
    end
    return x_feet
end

function compute_dx(v::AbstractVector, mesh::AbstractMesh)
    nx, dx, kx = mesh.nx, mesh.dx, mesh.kx
    dx_v=real(ifft(complex(0, 1)*kx .* fft(v)));
    return dx_v
end


function remap_f!(rho, u, mesh_x::AbstractMesh, grid_v::UniformGrid)

    nv, dv = grid_v.nv, grid_v.dv
    nx, dx, xmin, xmax = mesh_x.nx, mesh_x.dx, mesh_x.xmin, mesh_x.xmax
    L = xmax - xmin

    new_SF = 0.0
    new_weights = zeros(nv)
    new_rho = zeros(nx, nv)
    new_u   = zeros(nx, nv)

    new_u .= grid_v.v'

    @threads for l in 1:nv
        v_j = grid_v.v[l]
        m_f = 0.0
        for i = 1:nx
            f = 0.0
            for j = 1:nv
                df = grid_v.w[j] * rho[i, j] * Spline(v_j-u[i, j], dv)
                m_f += df * dx / L
                f += df
            end
            new_rho[i,l] = f 
        end
        new_weights[l] = m_f * dv
        new_rho[:,l] ./= m_f
    end

    new_SF = sum(new_weights)

    grid_v.w .= new_weights ./ new_SF

    rho .= new_rho
    u .= new_u
end

function main()

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
    poisson!(phi, mesh_x, rho_tot, ϵ)
    e = -1.0*compute_dx(phi, mesh_x)
    rho_pred = zeros(nx, nv)
    u_pred = zeros(nx, nv)
    phi_pred = zeros(nx)

    dt = 0.1 #1.0*mesh_x.dx
    tfinal = 20
    time = [0.0]
    remap_time = 0.0

    elec_energy = [compute_elec_energy(phi, mesh_x)]

    n = 0
    sum_norm_dx_u = 0.0
    remap_time = 0.0

    jac = zeros(nx)
    dx_u = zeros(nx, nv)
    u_hat = zeros(ComplexF64, nx, nv)

    bc = PeriodicBC(endpoint = :exclusive)
    xq = [zeros(nx) for j in 1:nv]

    while n * dt <= tfinal
        iter = 0
        err=1e-10
        maxiter=50
        sum_norm_dx_u += dt*compute_norm_dx_u(mesh_x, grid_v, u)

        threshold = 0.2 #Numerical remapping threshold : you may change it or not depending on the test case

        if (sum_norm_dx_u > threshold)
            @info "Remapping f at time = $(n*dt), solver = $solver"
            remap_time = n * dt
            remap_f!(rho, u, mesh_x, grid_v)
            sum_norm_dx_u = 0.0
        end

        u_pred .= u .+ 0.5dt .* e

        compute_dx!(dx_u, mesh_x, u, u_hat)

        for j in 1:nv, i in 1:nx
            xq[j][i] = compute_feet_char(i, dt, mesh_x, view(u_pred, :, j))
        end

        xi = 1.0:nx
        for j = 1:nv
            jac .= 1 .+ dt * cubic_interp(xi, view(dx_u, :, j), xq[j], bc = bc)
            rho[:, j] .= cubic_interp(xi, view(rho, :, j), xq[j], bc = bc) ./ jac
            u[:, j] .= cubic_interp(xi, view(u_pred, :, j), xq[j], bc = bc)
        end

        u .+= 0.5dt .* e

        rho_tot .= vec(sum(rho .* grid_v.w', dims = 2))

        poisson!(phi, mesh_x, rho_tot, ϵ)
        e .= -1.0*compute_dx(phi, mesh_x)

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

png(plot_f, "plot_df")
plot(time, log.(elec_energy))
