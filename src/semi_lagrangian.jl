using FastInterpolations

export SemiLagrangian

abstract type AbstractSolver end

struct SemiLagrangian <: AbstractSolver end

"""
$(SIGNATURES)

Compute the local lagrange polynomial at ``x_i = (i-1)*dx``, ``L_{i+k}(y)`` with ``k =-1,0,1,2``
"""
function lagrange_3(y::Float64, i::Int, k::Int, mesh::AbstractMesh)
    dx = mesh.dx
    x_i = (i - 1) * dx
    x_l = (i - 2) * dx
    x_r = i * dx
    x_rr = (i + 1) * dx
    if (k == -1)
        return (y - x_i) * (y - x_r) * (y - x_rr) / ((x_l - x_i) * (x_l - x_r) * (x_l - x_rr))
    elseif (k == 0)
        return (y - x_l) * (y - x_r) * (y - x_rr) / ((x_i - x_l) * (x_i - x_r) * (x_i - x_rr))
    elseif (k == 1)
        return (y - x_l) * (y - x_i) * (y - x_rr) / ((x_r - x_l) * (x_r - x_i) * (x_r - x_rr))
    elseif (k == 2)
        return (y - x_l) * (y - x_i) * (y - x_r) / ((x_rr - x_l) * (x_rr - x_i) * (x_rr - x_r))
    end
end

function interpolate_cubic_on_mesh(x::Float64, mesh::AbstractMesh, u::AbstractVector)
    nx, dx = mesh.nx, mesh.dx
    ix = mod1(Int(floor(x / dx) + 1), nx + 1)
    il_mod = mod1(ix - 1, nx + 1)
    ir_mod = mod1(ix + 1, nx + 1)
    irr_mod = mod1(ix + 2, nx + 1)
    pi_u = u[il_mod] * lagrange_3(x, ix, -1, mesh) + u[ix] * lagrange_3(x, ix, 0, mesh) + u[ir_mod] * lagrange_3(x, ix, 1, mesh) + u[irr_mod] * lagrange_3(x, ix, 2, mesh)
    return pi_u
end

export compute_dx!

function compute_dx!(v::AbstractVector, mesh::AbstractMesh)
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end

export update_rho_predictor!

function update_rho_predictor!(
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        pred_rho::AbstractVector, 
        rho_at_step_n::AbstractVector, 
        u_at_step_n::AbstractVector,
        dt::Float64, 
        x_feet_mesh::AbstractVector
    )

    dx_u_n = compute_dx!(u_at_step_n, mesh)

    dx_u_n_feet = cubic_interp(mesh.x, dx_u_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L))
    pred_rho .= cubic_interp(mesh.x, rho_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L))
    pred_rho .*= exp.(-dt .* dx_u_n_feet)

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

    u_n = cubic_interp(mesh.x, u_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L)) 
    e_n = cubic_interp(mesh.x, e_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L)) 
    e_n .+= e_at_step_n_plus
    u .= u_n .+ 0.5 * dt .* e_n

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

    dx_u_n = compute_dx!(u_at_step_n, mesh)
    dx_u_n_plus = compute_dx!(u_at_step_n_plus, mesh)

    cubic_interp!(rho, mesh.x, rho_at_step_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L) ) 
    du_dx = cubic_interp(mesh.x, dx_u_n, x_feet_mesh, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L) ) 
    du_dx .+= dx_u_n_plus
    rho .*= exp.(-0.5 * dt .* du_dx)

end


export compute_x_feet_mesh!

"""
$(SIGNATURES)

Use Fixed-Point methodod to compute the feet of the characteristic : X(t-dt) = X(t) + d we search for d
"""
function compute_x_feet_mesh!(dt::Float64, mesh::AbstractMesh, x_feet_mesh::AbstractVector, u::AbstractVector, e::AbstractVector)

    L = mesh.L
    h = 1e-10
    itp = cubic_interp(mesh.x, u, bc = PeriodicBC(endpoint=:exclusive, period = mesh.L))
    for i in eachindex(e)
        d = 0.0
        err = 1.0
        x = mesh.x[i]
        b = -0.5 * dt * dt * e[i]
        while abs(err) > h
            p = itp(x + d)
            d = -dt * p + b
            err = d + dt * p - b
        end
        x_feet_mesh[i] = mod(x + d, L)
    end
end
