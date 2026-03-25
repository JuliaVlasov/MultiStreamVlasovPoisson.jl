using FastInterpolations

export SemiLagrangian

abstract type AbstractSolver end

struct SemiLagrangian <: AbstractSolver end

export compute_dx

function compute_dx(v::AbstractVector, mesh::AbstractMesh)
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end

export compute_x_feet_mesh!

"""
$(SIGNATURES)

Use Fixed-Point methodod to compute the feet of the characteristic : X(t-dt) = X(t) + d we search for d
"""
function compute_x_feet_mesh!(dt::Float64, mesh::AbstractMesh, 
           x_feet_mesh::AbstractVector, u::AbstractVector, e::AbstractVector)

    nx = mesh.nx
    dx = mesh.dx
    for i in eachindex(e)
        b = -0.5 * dt * dt * e[i]
        d = - (dt * u[i] + b) / dx
        x_feet_mesh[i] = mod1(i + d, nx+1)
    end

end

export update_rho_predictor!

function update_rho_predictor!(
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        pred_rho::AbstractVector, 
        rho::AbstractVector, 
        u::AbstractVector,
        dt::Float64, 
        x::AbstractVector
    )

    nx = mesh.nx
    du_dx = compute_dx!(u, mesh)
    du_dx .= cubic_interp(1.0:nx+1, du_dx, x, bc = PeriodicBC(endpoint=:exclusive))
    pred_rho .= cubic_interp(1.0:nx+1, rho, x, bc = PeriodicBC(endpoint=:exclusive))
    pred_rho .*= exp.(-dt .* du_dx)

end

export update_u!

function update_u!(
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        e_at_step_n::AbstractVector, 
        e_at_step_n_plus::AbstractVector, 
        u::AbstractVector, 
        u_at_step_n::AbstractVector,
        dt::Float64, 
        x_feet_mesh::AbstractVector
    )

    nx = mesh.nx
    u .= cubic_interp(1.0:nx+1, u_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive)) 
    e = cubic_interp(1.0:nx+1, e_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive)) 
    e .+= e_at_step_n_plus
    u .+= 0.5 * dt .* e

end

export update_rho_corrector!

function update_rho_corrector!(
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        rho::AbstractVector, 
        rho_at_step_n::AbstractVector, 
        u_at_step_n::AbstractVector, 
        u_at_step_n_plus::AbstractVector,
        dt::Float64, 
        x_feet_mesh::AbstractVector
    )

    nx = mesh.nx
    du_dx = compute_dx!(u_at_step_n, mesh)
    du_dx_plus = compute_dx!(u_at_step_n_plus, mesh)

    cubic_interp!(rho, 1.0:nx+1, rho_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive) ) 
    du_dx .= cubic_interp(1.0:nx+1, du_dx, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive) ) 
    du_dx .+= du_dx_plus
    rho .*= exp.(-0.5 * dt .* du_dx)

end
