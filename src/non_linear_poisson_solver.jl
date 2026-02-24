using LinearAlgebra
using SparseArrays

export NonLinearPoissonSolver

struct NonLinearPoissonSolver

    eps::Float64
    nx::Int
    jacobian::SparseMatrixCSC{Float64, Int64}
    rhs::Vector{Float64}
    maxiter::Int

    function NonLinearPoissonSolver(eps::Float64, nx::Int)

        rows = Int[]
        cols = Int[]
        vals = Float64[]
        push!(rows, 1)
        push!(cols, 1)
        push!(vals, 0.0)
        push!(rows, 1)
        push!(cols, 2)
        push!(vals, 0.0)
        push!(rows, 1)
        push!(cols, nx + 1)
        push!(vals, 0.0)

        for i in 2:nx
            push!(rows, i)
            push!(cols, i - 1)
            push!(vals, 0.0)
            push!(rows, i)
            push!(cols, i)
            push!(vals, 0.0)
            push!(rows, i)
            push!(cols, i + 1)
            push!(vals, 0.0)
        end

        push!(rows, nx + 1)
        push!(cols, 1)
        push!(vals, 0.0)
        push!(rows, nx + 1)
        push!(cols, nx + 1)
        push!(vals, 0.0)
        push!(rows, nx + 1)
        push!(cols, nx)
        push!(vals, 0.0)

        jacobian = sparse(rows, cols, vals, nx + 1, nx + 1)
        rhs = zeros(nx + 1)

        return new(eps, nx, jacobian, rhs, 1000)

    end

end

export solve!
export poisson!

"""
$(SIGNATURES)
    
Solve the quasi-linear elliptic problem: ϵ^2 Δϕ + exp(-ϕ) = ρ on the Torus.

Uses Newton's method.
"""
function solve!(phi::Vector{Float64}, solver::NonLinearPoissonSolver, rho_tot::Vector{Float64})

    ϵ = solver.eps
    nx = solver.nx
    iter = 0
    delta = zeros(nx + 1)

    while iter < solver.maxiter

        solver.rhs[1] = ϵ * ϵ * (phi[2] - 2 * phi[1] + phi[nx + 1]) + exp(-phi[1]) - rho_tot[1]
        solver.jacobian[1, 1] = -2 * ϵ * ϵ - exp(-phi[1])
        solver.jacobian[1, 2] = ϵ * ϵ
        solver.jacobian[1, nx + 1] = ϵ * ϵ

        for i in 2:nx
            solver.rhs[i] = ϵ * ϵ * (phi[i + 1] - 2 * phi[i] + phi[i - 1]) + exp(-phi[i]) - rho_tot[i]
            solver.jacobian[i, i - 1] = ϵ * ϵ
            solver.jacobian[i, i] = -2 * ϵ * ϵ - exp(-phi[i])
            solver.jacobian[i, i + 1] = ϵ * ϵ
        end

        solver.rhs[nx + 1] = ϵ * ϵ * (phi[1] - 2 * phi[nx + 1] + phi[nx]) + exp(-phi[nx + 1]) - rho_tot[nx + 1]
        solver.jacobian[nx + 1, 1] = ϵ * ϵ
        solver.jacobian[nx + 1, nx + 1] = -2 * ϵ * ϵ - exp(-phi[nx + 1])
        solver.jacobian[nx + 1, nx] = ϵ * ϵ

        norm(solver.rhs, Inf) < 1.0e-10 && return

        delta .= solver.jacobian \ solver.rhs
        phi .-= delta

    end
    return
end


"""
$(SIGNATURES)

Solve the elliptic problem: ϵ^2 Δϕ = ρ-1  on the Torus.

Uses Fourier method.
"""
function poisson!(phi::Vector{Float64}, mesh::UniformMesh, rho_tot::Vector{Float64}, eps::Float64)

rho_tot_f=fft(rho_tot.-1)
rho_tot_f[1]=0
kkx=mesh.kx
kkx[1]=1
phi.=+real(ifft((rho_tot_f./(kkx.*kkx))))/(eps*eps)

end
