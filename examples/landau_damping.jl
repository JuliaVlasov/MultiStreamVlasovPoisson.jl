# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using DispersionRelations
using FastInterpolations
using FFTW
using Parameters
using Plots
using .Threads
using TimerOutputs

const to = TimerOutput()

struct UniformMesh 

    xmin::Float64
    xmax::Float64
    nx::Int
    x::Vector{Float64}
    kx::Vector{Float64}
    dx::Float64

    function UniformMesh(xmin::Float64, xmax::Float64, nx::Int)
        dx = (xmax - xmin) / nx
        x = LinRange(xmin, xmax, nx+1)[1:end-1]
        kx = collect(2π / (xmax - xmin) * fftfreq(nx, nx))
        return new(xmin, xmax, nx, x, kx, dx)
    end

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

function mean_f0(test_case::LandauDamping, mesh::UniformMesh, v::Float64)::Float64
    mf0 = 0.0
    nx, dx, xmin, xmax = mesh.nx, mesh.dx, mesh.xmin, mesh.xmax
    for i in 1:nx
        x = mesh.x[i]
        mf0 += f0(test_case, x, v) * dx / (xmax - xmin)
    end
    return mf0
end

struct UniformGrid 

    nv::Int
    v::Vector{Float64}
    dv::Float64
    w::Vector{Float64}

    function UniformGrid(vmin, vmax, nv, mesh, test_case)

        sf0 = 0.0
        dv = Float64
        dv = (vmax - vmin) / (nv)
        v = LinRange(vmin, vmax, nv)
        w = zeros(nv)
        for i in 1:nv
            sf0 += mean_f0(test_case, mesh, v[i]) * dv
        end
        for i in 1:nv
            w[i] = mean_f0(test_case, mesh, v[i]) * dv / sf0
        end
        return new(nv, v, dv, w)
    end

end

function compute_dx(v::AbstractVector, mesh::UniformMesh)
    nx = mesh.nx
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end


function compute_initial_condition(test_case::LandauDamping, mesh::UniformMesh, grid::UniformGrid)

    k = test_case.k
    nx, nv = mesh.nx, grid.nv
    rho = [zeros(nx) for i in 1:nv]
    u = zeros(nx, nv)
    rho_tot = zeros(nx)
    for j in 1:nv
        v = grid.v[j]
        for i in 1:nx
            x = mesh.x[i]
            rho[j][i] = f0(test_case, x, v) / mean_f0(test_case, mesh, v)
            u[i,j] = v
            rho_tot[i] += grid.w[j] * rho[j][i]
        end
    end
    return rho, u, rho_tot
end


function compute_rho_total!(rho_tot::Vector{Float64}, grid::UniformGrid, rho)

    fill!(rho_tot, 0.0)
    for j in eachindex(grid.w), i in eachindex(rho_tot)
        rho_tot[i] += grid.w[j] * rho[j][i]
    end
    rho_tot[end] = rho_tot[1]
end

function poisson!(phi::Vector{Float64}, mesh::UniformMesh, rho_tot::Vector{Float64}; ϵ = 1.0)

    rho_tot_f = fft(rho_tot .- 1)
    rho_tot_f[1] = 0
    kkx = mesh.kx
    kkx[1] = 1
    phi .= real(ifft((rho_tot_f ./ (kkx .* kkx)))) / (ϵ * ϵ)

end

function compute_elec_energy(phi::Vector{Float64}, mesh::UniformMesh; ϵ = 1.0)::Float64
    e = 0.0
    nx, dx = mesh.nx, mesh.dx
    for i in eachindex(phi)
        ir = mod1(i + 1, nx)
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

    phi = zeros(nx)
    poisson!(phi, mesh_x, rho_tot)
    e = -1.0 .* compute_dx(phi, mesh_x)

    dt = 0.1 
    time = [0.0]

    #Array of physical quantities
    elec_energy = [compute_elec_energy(phi, mesh_x)]

    n = 0

    e_pred = zeros(nx)
    phi_pred = zeros(nx)
    rho_pred = zeros(nx)

    xi = 1.0:nx
    xq = zeros(nx)
    rho_at_step_n = zeros(nx)
    u_at_step_n = zeros(nx)

    du_dx_plus = zeros(nx)
    e_new = zeros(nx)

    bc = PeriodicBC(endpoint = :exclusive)

    u_hat = zeros(ComplexF64, nx, nv)
    du_dx = zeros(nx)

    while n * dt <= tfinal

        nx = mesh_x.nx
        dx = mesh_x.dx
        fill!(rho_tot, 0.0)
        xi = 1.0:nx

        fill!(rho_tot, 0.0)

        for j in 1:nv
            for i in eachindex(e)
                d = - (dt * u[i,j] - 0.5 * dt * dt * e[i]) / dx
                xq[i] = mod1(i + d, nx)
            end
            @timeit to "dx" du_dx .= compute_dx(u[:,j], mesh_x)
            du_dx .= cubic_interp(xi, du_dx, xq, bc = bc)
            rho_pred .= cubic_interp(xi, rho[j], xq, bc = bc)
            rho_tot .+= rho_pred .* grid_v.w[j] .* exp.(-dt .* du_dx)
        end

        @timeit to "poisson" begin
            poisson!(phi_pred, mesh_x, rho_tot)
            e_pred .= -1.0 .* compute_dx(phi_pred, mesh_x)
        end

        fill!(rho_tot, 0.0)

        for j in 1:nv

            u_at_step_n .= u[:, j]
            rho_at_step_n .= rho[j]

            for i in eachindex(e)
                d = - (dt * u[i,j] -0.5 * dt * dt * e[i]) / dx
                xq[i] = mod1(i + d, nx)
            end

            u[:, j] .= cubic_interp(xi, u_at_step_n, xq, bc = bc)
            e_new .= cubic_interp(xi, e, xq, bc = bc)
            e_new .+= e_pred
            u[:, j] .+= 0.5dt .* e_new

            @timeit to "dx" du_dx .= compute_dx(u_at_step_n, mesh_x)
            @timeit to "dx" du_dx_plus .= compute_dx(u[:, j], mesh_x)

            rho[j] .= cubic_interp(xi, rho_at_step_n, xq, bc = bc)
            du_dx .= cubic_interp(xi, du_dx, xq, bc = bc)
            du_dx .+= du_dx_plus
            rho[j] .*= exp.(-0.5dt .* du_dx)
            rho_tot .+= rho[j] .* grid_v.w[j] 
        end

        @timeit to "poisson" begin
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

@show nthreads()

@time time, elec_energy = main( tfinal = 50.0 )

show(to)

plot(time, elec_energy, yscale = :ln)
line, ω = fit_complex_frequency(time, elec_energy)
plot!(time, line)
title!("$(imag(ω))")
