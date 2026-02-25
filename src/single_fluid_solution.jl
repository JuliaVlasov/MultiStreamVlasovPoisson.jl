using LinearAlgebra
export SingleFluidSolution
export update_rho!
export update_u!
export update_u_SL!
export update_rho_corrector_SL!
export update_rho_predictor_SL!
export compute_x_feet_mesh!
export compute_dx!

struct SingleFluidSolution

    m_upd_rho::SparseMatrixCSC{Float64, Int}
    jac_t::SparseMatrixCSC{Float64, Int}

    function SingleFluidSolution(mesh::AbstractMesh)

        nx = mesh.nx

        rows = Int[]
        cols = Int[]
        vals = Float64[]
        for i in 1:(nx + 1)
            ir = mod1(i + 1, nx + 1)
            il = mod1(i - 1, nx + 1)

            push!(rows, i); push!(cols, i); push!(vals, 0.0)
            push!(rows, i); push!(cols, ir); push!(vals, 0.0)
            push!(rows, i); push!(cols, il); push!(vals, 0.0)

        end
        m_upd_rho = sparse(rows, cols, vals, nx + 1, nx + 1)
        jac_t = sparse(rows, cols, vals, nx + 1, nx + 1)

        return new(m_upd_rho, jac_t)
    end


end

"""
$(SIGNATURES)
    
Maximum of two values.
"""
function mymax(a::Float64, b::Float64)::Float64
    if a >= b
        return a
    else
        return b
    end
end

"""
$(SIGNATURES)
    
Regularization function of (x)^{+}.
"""
function g(x::Float64, dx::Float64)::Float64
    if x < -dx
        return 0.0
    elseif x >= -dx && x <= dx
        return (x + dx) * (x + dx) / (4dx)
    else
        return x
    end
end

"""
$(SIGNATURES)
    
Function G: (s,t,u) --> s*g(u) - t*(g(u)-u).
"""
function G(s::Float64, t::Float64, u::Float64, dx::Float64)::Float64
    return s * g(u, dx) - t * (g(u, dx) - u)
end

"""
$(SIGNATURES)
    
Numerical derivative of g using finite differences.
"""
function dg(x::Float64, dx::Float64)::Float64
    h = 1.0e-16
    return (g(x + h, dx) - g(x - h, dx)) / (2h)
end

"""
$(SIGNATURES)
    
Helper function for density calculation.
"""
function t_rho(s::Float64, t::Float64, u::Float64, dx::Float64)::Float64
    if abs(u) < 1.0e-16
        return t - (t - s) * dg(0.0, dx)
    else
        return (G(s, t, u, dx) - G(s, t, 0.0, dx)) / u
    end
end






"""
$(SIGNATURES)

Update one single fluid solution using fixed-point iteration.
"""
function update_rho!(
        mesh::AbstractMesh, rho::AbstractVector, u::AbstractVector,
	rho_at_step_n::AbstractVector, dt::Float64
    )

    sol = SingleFluidSolution(mesh)
    nx, dx = mesh.nx, mesh.dx
    iter = 0

    # Update rho with new computed velocity
    for i in 1:(nx + 1)
        ir = mod1(i + 1, nx + 1)
        il = mod1(i - 1, nx + 1)
        sol.m_upd_rho[i, i] = 1 + (dt / dx) * (g(u[i], dx) + g(u[il], dx) - u[il])
        sol.m_upd_rho[i, ir] = -(dt / dx) * (g(u[i], dx) - u[i])
        sol.m_upd_rho[i, il] = -(dt / dx) * g(u[il], dx)
     end

     # Solve for rho
     rho .= sol.m_upd_rho \ rho_at_step_n

    return
end


"""
$(SIGNATURES)

Update one single fluid solution using fixed-point iteration.
"""
function update_u!(
        mesh::AbstractMesh, rho::AbstractVector, u::AbstractVector,
	phi::Vector, rho_at_step_n::AbstractVector, u_at_step_n::AbstractVector,
	dt::Float64, maxiter::Int = 10
    )

    sol = SingleFluidSolution(mesh)
    nx, dx = mesh.nx, mesh.dx
    iter = 0

    # For Newton part for v
    t = zeros(nx + 1)
    t[1] = 1.0
    v = copy(u)
    delta = zeros(nx + 1)
    h = 1.0e-10
    it_newt = 0
    err_t=1e-10

    # Update u using Newton algorithm
    copyto!(v, u)
    it_newt = 0

    while norm(t, Inf) > err_t && it_newt < maxiter

        for i in 1:(nx + 1)

            irr = mod1(i + 2, nx + 1)
            ir = mod1(i + 1, nx + 1)
            il = mod1(i - 1, nx + 1)
            ill = mod1(i - 2, nx + 1)

            # Compute flux of momentum
            q_r = 0.5 * (G(rho[ir], rho[irr], u[ir], dx) + G(rho[i], rho[ir], u[i], dx))
            q_l = 0.5 * (G(rho[i], rho[ir], u[i], dx) + G(rho[il], rho[i], u[il], dx))

            # Compute Jacobian
            sol.jac_t[i, i] = 0.5 * (rho[ir] + rho[i]) + (dt / dx) * (mymax(q_r, 0.0) + mymax(q_l, 0.0) - q_l) -
                    dt * ((phi[ir] - phi[i]) / dx) * (t_rho(rho[i], rho[ir], v[i] + h, dx) - t_rho(rho[i], rho[ir], v[i] - h, dx)) / (2h)
            sol.jac_t[i, ir] = -(dt / dx) * (mymax(q_r, 0.0) - q_r)
            sol.jac_t[i, il] = -(dt / dx) * mymax(q_l, 0.0)

            # Compute T(v)
            t[i] = 0.5 * (rho[ir] + rho[i]) * v[i] - 0.5 * (rho_at_step_n[ir] + rho_at_step_n[i]) * u_at_step_n[i] +
                  (dt / dx) * (mymax(q_r, 0.0) + mymax(q_l, 0.0) - q_l) * v[i] -
                  (dt / dx) * (mymax(q_r, 0.0) - q_r) * v[ir] -
                  (dt / dx) * mymax(q_l, 0.0) * v[il] +
                  dt * ((phi[ir] - phi[i]) / dx) * t_rho(rho[i], rho[ir], v[i], dx)
         end

         delta .= sol.jac_t \ t
         v .-= delta

         it_newt += 1

     end

     copyto!(u, v)

    return

end

"""
$(SIGNATURES)
Update one single fluid solution with :
1) cubic interpolation 
2) Semi Laggrangian scheme with trapeze rules for the integral of the force
"""

#Compute the local lagrange polynomial at $x_i = (i-1)*dx$, L_{i+k}(y) with k =-1,0,1,2
function lagrange_3(y::Float64,i::Int,k::Int,mesh::AbstractMesh)
    dx   = mesh.dx
    x_i  = (i-1) * dx
    x_l  = (i-2) * dx
    x_r  = i     * dx
    x_rr = (i+1) * dx
    if( k== -1)
        return (y-x_i) * (y-x_r) * (y-x_rr)/(( x_l-x_i) * (x_l-x_r) * (x_l-x_rr) )
    elseif( k== 0)
        return (y-x_l) * (y-x_r) * (y-x_rr)/(( x_i-x_l) * (x_i-x_r) * (x_i-x_rr) )
    elseif( k == 1)
        return (y-x_l) * (y-x_i) * (y-x_rr)/((x_r-x_l) * (x_r-x_i) * (x_r-x_rr))
    elseif( k == 2)
        return (y-x_l) * (y-x_i) * (y-x_r)/((x_rr-x_l) * (x_rr-x_i) * (x_rr-x_r))
    end
end


function interpolate_cubic_on_mesh(x::Float64,mesh::AbstractMesh,u::AbstractVector)
    nx, dx = mesh.nx, mesh.dx
    ix  = mod1(Int(floor(x/dx) +1), nx+1)
    il_mod =  mod1(ix-1,  nx+1)
    ir_mod  = mod1(ix+1, nx+1)
    irr_mod = mod1(ix+2, nx+1)
    pi_u = u[il_mod] * lagrange_3(x,ix,-1,mesh) + u[ix] * lagrange_3(x,ix,0,mesh) + u[ir_mod] * lagrange_3(x,ix,1,mesh) + u[irr_mod] * lagrange_3(x,ix,2,mesh)
    return pi_u
end


function compute_char_foot(x::Float64,dt::Float64,mesh::AbstractMesh,u::AbstractVector,E::Float64)
    b = -0.5 * dt * dt * E
    d = 0.0
    L = mesh.L
    err = 1.0
    h = 1E-10
    #Use Fixed-Point methodod to compute the feet of the characteristic : X(t-dt) = X(t) + d we search for d
    while( abs(err) >1E-10 )
        p = interpolate_cubic_on_mesh(x+d,mesh,u)
        d = -dt *p + b
        err = d + dt * p  - b
    end
    x_feet = mod(x+d,L)
    return x_feet
end

function compute_x_feet_mesh!(dt::Float64,mesh::AbstractMesh,x_feet_mesh::AbstractVector,u::AbstractVector,E::AbstractVector)
    nx = mesh.nx
    for i in 1:(nx+1)
        x_feet_mesh[i] = compute_char_foot(mesh.x[i],dt,mesh,u,E[i])
    end
    return
end

function compute_dx!(v::AbstractVector,mesh::AbstractMesh)
    nx, dx ,kx = mesh.nx, mesh.dx, mesh.kx
    dx_v=real(ifft(complex(0,1)*kx.*fft(v)));
    return dx_v
end

function update_rho_predictor_SL!(
        mesh::AbstractMesh, pred_rho::AbstractVector, rho_at_step_n::AbstractVector, u_at_step_n::AbstractVector,
	dt::Float64, x_feet_mesh::AbstractVector)
    nx = mesh.nx
    dx_u_n   = compute_dx!(u_at_step_n,mesh)
    for i in 1:(nx+1)
        dx_u_n_feet = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,dx_u_n)
        pred_rho[i] = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,rho_at_step_n) * exp(-dt * dx_u_n_feet)
    end
    return
end

function update_u_SL!(mesh::AbstractMesh,E_at_step_n::AbstractVector, E_at_step_n_plus::AbstractVector,u::AbstractVector, u_at_step_n::AbstractVector,
	dt::Float64,x_feet_mesh::AbstractVector)
    nx = mesh.nx
    for i in 1:(nx+1)
        u[i] = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,u_at_step_n) +0.5*dt*(interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,E_at_step_n)+E_at_step_n_plus[i] )
    end
    return 
end

function update_rho_corrector_SL!(
        mesh::AbstractMesh, rho::AbstractVector, rho_at_step_n::AbstractVector, u_at_step_n::AbstractVector,u_at_step_n_plus::AbstractVector,
	dt::Float64, x_feet_mesh::AbstractVector)
    nx = mesh.nx
    dx_u_n      = compute_dx!(u_at_step_n,mesh)
    dx_u_n_plus = compute_dx!(u_at_step_n_plus,mesh)
    for i in 1:(nx+1)
        rho[i] = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,rho_at_step_n) * exp(-0.5 * dt *( interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,dx_u_n) +dx_u_n_plus[i] ) )
    end
    return 
end


