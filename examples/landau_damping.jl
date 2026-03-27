# -*- coding: utf-8 -*-
# # Landau damping with Vlasov-Poisson solver

using DispersionRelations
using FastInterpolations
using FFTW
using LinearAlgebra
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

abstract type InitialCondition end

@with_kw struct LandauDamping <: InitialCondition

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


function mean_f0(test_case::InitialCondition, mesh::UniformMesh, v::Float64)::Float64
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


function compute_initial_condition(test_case::InitialCondition, mesh::UniformMesh, grid::UniformGrid)

    k = test_case.k
    nx, nv = mesh.nx, grid.nv
    rho = zeros(nx, nv)
    u = zeros(nx, nv)
    rho_tot = zeros(nx)
    for j in 1:nv
        v = grid.v[j]
        for i in 1:nx
            x = mesh.x[i]
            rho[i,j] = f0(test_case, x, v) / mean_f0(test_case, mesh, v)
            u[i,j] = v
            rho_tot[i] += grid.w[j] * rho[i,j]
        end
    end
    return rho, u, rho_tot
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


function main(; tfinal = 40)

    k = 0.5
    α = 0.001
    test_case = LandauDamping(α = α, k = k)
    nx, xmin, xmax = 128, 0.0, 2π / k
    nv, vmin, vmax = 256, -6.0, 6.0
    
    mesh_x = UniformMesh(xmin, xmax, nx)
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
    rho_pred = zeros(nx, nv)

    xi = 1.0:nx
    xq = [zeros(nx) for i in 1:nv]
    e_new = zeros(nx)
    du_dx = zeros(nx)

    bc = PeriodicBC(endpoint = :exclusive)

    v = zeros(nx)
    dv = zeros(nx, nv)
    dv_plus = zeros(nx, nv)
    v_hat = zeros(ComplexF64, nx, nv)

    dx = mesh_x.dx

    t = 0.0
    @show nstep = floor(Int, tfinal / dt) + 1

    for istep in 1:nstep

        dt = min(dt, tfinal-t)

        for j in 1:nv, i in eachindex(e)
            d = - (dt * u[i,j] -0.5 * dt * dt * e[i]) / dx
            xq[j][i] = mod1(i + d, nx)
        end


        @timeit to "fft" begin
            v_hat .= fft(u, 1)
            v_hat .*= 1im .* mesh_x.kx
            dv .= real(ifft(v_hat, 1))
        end

        fill!(rho_tot, 0.0)

        @timeit to "predictor" for j in 1:nv
            v .= cubic_interp(xi, view(dv, :, j), xq[j], bc = bc)
            rho_pred[:, j] .= cubic_interp(xi, view(rho,:,j), xq[j], bc = bc)
            rho_pred[:, j] .*= exp.(-dt .* v)
        end

        @timeit to "poisson" begin
            rho_tot .= vec(sum(rho_pred .* grid_v.w', dims=2))
            poisson!(phi_pred, mesh_x, rho_tot)
            e_pred .= -1.0 .* compute_dx(phi_pred, mesh_x)
        end


        @timeit to "fft" begin
            v_hat .= fft(u, 1)
            v_hat .*= 1im .* mesh_x.kx
            dv .= real(ifft(v_hat, 1))
        end

        @timeit to "corrector" for j in 1:nv

            u[:, j] .= cubic_interp(xi, view(u, :, j), xq[j], bc = bc)
            e_new .= cubic_interp(xi, e, xq[j], bc = bc)
            e_new .+= e_pred
            u[:, j] .+= 0.5dt .* e_new

        end

        @timeit to "fft" begin
            v_hat .= fft(u, 1)
            v_hat .*= 1im .* mesh_x.kx
            dv_plus .= real(ifft(v_hat, 1))
        end

        fill!(rho_tot, 0.0)

        @timeit to "corrector" for j in 1:nv

            rho[:,j] .= cubic_interp(xi, view(rho, :, j), xq[j], bc = bc)
            du_dx .= cubic_interp(xi, view(dv, :, j), xq[j], bc = bc)
            du_dx .+= view(dv_plus, :, j)
            rho[:,j] .*= exp.(-0.5dt .* du_dx)

        end

        @timeit to "poisson" begin
            rho_tot .= vec(sum(rho .* grid_v.w', dims=2))
            poisson!(phi, mesh_x, rho_tot)
            e .= -1.0 * compute_dx(phi, mesh_x)
        end

        push!(elec_energy, compute_elec_energy(phi, mesh_x))
        t += dt
        push!(time, t)
        println("iteration: $istep , time = $(t), elec energy = $(last(elec_energy))")

    end  

    return time, elec_energy

end

@time time, elec_energy = main( tfinal = 100.0 )

show(to)

plot(time, elec_energy, yscale = :ln)
line, ω = fit_complex_frequency(time, elec_energy)
plot!(time, line)
title!("$(imag(ω))")
