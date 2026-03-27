using FastInterpolations

export SemiLagrangian

abstract type AbstractSolver end

struct SemiLagrangian <: AbstractSolver 

    nx :: Int
    nv :: Int
    dx :: Float64
    xq :: Vector{Vector{Float64}}

    function SemiLagrangian( mesh, grid )

        xq = [zeros(mesh.nx) for i in 1:grid.nv]

        new( mesh.nx, grid.nv, mesh.dx, xq )

    end

end

export compute_dx

function compute_dx(v::AbstractVector, mesh::AbstractMesh)
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end

export compute_dx!

function compute_dx!(dv, mesh::AbstractMesh, u, v_hat)
    v_hat .= fft(u, 1)
    v_hat .*= 1im .* mesh.kx
    dv .= real(ifft(v_hat, 1))
end


export compute_x_feet_mesh!

"""
$(SIGNATURES)

Use Fixed-Point methodod to compute the feet of the characteristic : X(t-dt) = X(t) + d we search for d
"""
function compute_x_feet_mesh!(solver::SemiLagrangian, u::Matrix{Float64}, e::Vector{Float64}, dt::Float64)

    nx, dx, nv = solver.nx, solver.dx, solver.nv
    for j in 1:nv, i in eachindex(e)
        d = - (dt * u[i,j] -0.5 * dt * dt * e[i]) / dx
        solver.xq[j][i] = mod1(i + d, nx)
    end

end

export update_rho_predictor!

function update_rho_predictor!(
        rho_pred::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        rho::Matrix{Float64}, 
        v::Vector{Float64},
        dv::Matrix{Float64},
        dt::Float64
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)
        v .= cubic_interp(xi, view(dv, :, j), solver.xq[j], bc = bc)
        rho_pred[:, j] .= cubic_interp(xi, view(rho,:,j), solver.xq[j], bc = bc)
        rho_pred[:, j] .*= exp.(-dt .* v)
    end

end

export update_u!

function update_u!(
        u::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        e::AbstractVector, 
        e_pred::AbstractVector,
        e_new::AbstractVector, 
        dt::Float64
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)

        u[:, j] .= cubic_interp(xi, view(u, :, j), solver.xq[j], bc = bc)
        e_new .= cubic_interp(xi, e, solver.xq[j], bc = bc)
        e_new .+= e_pred
        u[:, j] .+= 0.5dt .* e_new

    end

end

export update_rho_corrector!

function update_rho_corrector!(
        rho::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        du_dx::AbstractVector, 
        dv::Matrix{Float64},
        dv_plus::Matrix{Float64},
        dt::Float64, 
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)

        rho[:,j] .= cubic_interp(xi, view(rho, :, j), solver.xq[j], bc = bc)
        du_dx .= cubic_interp(xi, view(dv, :, j), solver.xq[j], bc = bc)
        du_dx .+= view(dv_plus, :, j)
        rho[:,j] .*= exp.(-0.5dt .* du_dx)

    end

end
