# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using DispersionRelations
using FastInterpolations
using FFTW
using Parameters
using Plots
using TimerOutputs

const to = TimerOutput()

struct UniformMesh 
    nx::Int
    x::Vector{Float64}
    kx::Vector{Float64}
    dx::Float64
    L::Float64
    function UniformMesh(xmin::Float64, xmax::Float64, nx::Int)
        dx = (xmax - xmin) / (nx + 1)
        x = range(start=xmin, step=dx, length=nx+1)
        L = (xmax - xmin)
        kx = collect(2π / (xmax - xmin) * fftfreq(nx + 1, nx + 1))
        return new(nx, x, kx, dx, L)
    end

end

struct UniformGrid 
    nv::Int
    v::Vector{Float64}
    dv::Float64
    w::Vector{Float64}

    function UniformGrid(vmin, vmax, nv, mesh_x, test_case)

        sf0 = 0.0
        dv = Float64
        dv = (vmax - vmin) / (nv)
        v = LinRange(vmin, vmax, nv)
        w = zeros(nv)
        for i in 1:nv
            sf0 += mean_f0(test_case, mesh_x, v[i]) * dv
        end
        for i in 1:nv
            w[i] = mean_f0(test_case, mesh_x, v[i]) * dv / sf0
        end
        return new(nv, v, dv, w)
    end

end

function compute_dx(v::AbstractVector, mesh::UniformMesh)
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end


@with_kw struct LandauDamping 

    α::Float64 = 0.001
    k::Float64 = 0.5
    vth::Float64 = 1.0
    v0::Float64 = 0.0

end

function f0(test_case::LandauDamping, x::Float64, v::Float64)::Float64

    α = test_case.α
    k = test_case.k
    vth = test_case.vth
    v0 = test_case.v0
    return (1.0 / sqrt(2π * vth)) * exp(-0.5 * (v - v0) * (v - v0) / vth) * (1 + α * cos(k * x))

end

function mean_f0(test_case::LandauDamping, mesh_x::UniformMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, L = mesh_x.nx, mesh_x.dx, mesh_x.L
    for i in 1:(nx + 1)
        x = mesh_x.x[i]
        mf0 += f0(test_case, x, v) * dx / L
    end
    return mf0
end

function compute_initial_condition(test_case::LandauDamping, mesh_x::UniformMesh, grid_v::UniformGrid)

    k = test_case.k
    nx, nv = mesh_x.nx, grid_v.nv
    rho = [zeros(nx + 1) for i in 1:nv]
    u = [zeros(nx + 1) for i in 1:nv]
    rho_tot = zeros(nx + 1)
    for j in 1:nv
        alpha = grid_v.v[j]
        for i in 1:(nx + 1)
            x_i = mesh_x.x[i]
            rho[j][i] = f0(test_case, x_i, alpha) / mean_f0(test_case, mesh_x, alpha)
            u[j][i] = alpha
            rho_tot[i] += grid_v.w[j] * rho[j][i]
        end
    end
    return rho, u, rho_tot
end


function compute_rho_total!(rho_tot::Vector{Float64}, grid_v::UniformGrid, rho)
    fill!(rho_tot, 0.0)

    for j in eachindex(grid_v.w), i in eachindex(rho_tot)
        rho_tot[i] += grid_v.w[j] * rho[j][i]
    end
end

function compute_norm_dx_u(mesh_x::UniformMesh, grid_v::UniformGrid, u)

    nv = grid_v.nv
    nx = mesh_x.nx
    dx = mesh_x.dx

    dx_u = 0.0
    for l in 1:nv, i in 1:nx
        du = abs(u[l][i + 1] - u[l][i]) / dx
        dx_u = max(dx_u, du)
    end

    return dx_u
end


function poisson!(phi::Vector{Float64}, mesh::UniformMesh, rho_tot::Vector{Float64}; ϵ = 1.0)

    rho_tot_f = fft(rho_tot .- 1)
    rho_tot_f[1] = 0
    kkx = mesh.kx
    kkx[1] = 1
    return phi .= +real(ifft((rho_tot_f ./ (kkx .* kkx)))) / (ϵ * ϵ)

end


function compute_elec_energy(phi::Vector{Float64}, mesh::UniformMesh; ϵ = 1.0)::Float64
    e = 0.0
    nx, dx = mesh.nx, mesh.dx
    for i in eachindex(phi)
        ir = mod1(i + 1, nx + 1)
        e += 0.5 * ϵ * ϵ * dx * (phi[ir] - phi[i])^2 / (dx * dx)
    end
    return sqrt(e)
end

function main(; tfinal = 10)

    k = 0.5
    α = 0.001
    test_case = LandauDamping(α = α, k = k)

    nx, xmin, xmax = 128, 0.0, 2π / k
    mesh_x = UniformMesh(xmin, xmax, nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)

    rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)

    phi = zeros(nx + 1)
    poisson!(phi, mesh_x, rho_tot)
    e = -1.0 .* compute_dx(phi, mesh_x)

    dt = 0.1 
    time = [0.0]

    #Array of physical quantities
    elec_energy = [compute_elec_energy(phi, mesh_x)]

    n = 0

    e_pred = zeros(nx + 1)
    phi_pred = zeros(nx + 1)
    rho_pred = zeros(nx + 1)

    xq = zeros(nx + 1)
    rho_at_step_n = zeros(nx + 1)
    u_at_step_n = zeros( nx + 1)

    du_dx = zeros( nx + 1)
    du_dx_plus = zeros( nx + 1)
    e_new = zeros(nx + 1)

    while n * dt <= tfinal

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
            rho_tot .+= rho_pred .* grid_v.w[j] .* exp.(-dt .* du_dx)
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
            rho[j] .*= exp.(-0.5dt .* du_dx)
        end

        @timeit to "poisson" begin
            compute_rho_total!(rho_tot, grid_v, rho)
            poisson!(phi, mesh_x, rho_tot)
            e .= -1.0 * compute_dx(phi, mesh_x)
        end


        push!(elec_energy, compute_elec_energy(phi, mesh_x))
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
